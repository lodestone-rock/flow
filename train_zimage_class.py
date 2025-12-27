"""
Single-GPU Z-Image Trainer (Class-based)

A simplified trainer for Z-Image model without distributed training dependencies.
Refactored from src/trainer/train_z_image.py for easier debugging and development.
"""

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity

from tqdm import tqdm
from safetensors.torch import safe_open
from transformers import AutoTokenizer, Qwen3ForCausalLM

from ramtorch import AdamW

from src.dataloaders.dataloader import TextImageDataset
from src.models.zimage.model import ZImage, z_image_params
from src.models.zimage.sampling import get_noise, get_schedule, denoise_cfg
from src.models.zimage.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    make_text_position_ids,
)
from src.models.zimage.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport

# Optional: Aim for experiment tracking
try:
    from aim import Run, Image as AimImage
    from PIL import Image as PILImage
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    master_seed: int = 42
    cache_minibatch: int = 4
    train_minibatch: int = 1
    gradient_accumulation_steps: int = 4
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_every: int = 500
    save_folder: str = "checkpoints"
    trained_layer_keywords: List[str] = field(default_factory=list)
    
    # Profiling
    do_profiling: bool = False
    profile_steps: int = 2
    
    # Experiment tracking
    use_aim: bool = False
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference/validation settings."""
    inference_every: int = 100
    inference_folder: str = "inference_outputs"
    steps: int = 28
    cfg: float = 4.0
    prompts: List[str] = field(default_factory=lambda: ["a beautiful landscape painting"])
    first_n_steps_wo_cfg: int = 0
    image_dim: Tuple[int, int] = (512, 512)
    max_sequence_length: int = 512


@dataclass
class DataloaderConfig:
    """Dataloader settings."""
    batch_size: int = 8
    jsonl_metadata_path: str = "metadata.jsonl"
    image_folder_path: str = "images"
    base_resolution: List[int] = field(default_factory=lambda: [1024])
    shuffle_tags: bool = True
    tag_drop_percentage: float = 0.1
    uncond_percentage: float = 0.1
    resolution_step: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    ratio_cutoff: float = 2.0
    offset: int = 0


@dataclass
class ModelConfig:
    """Model paths and settings."""
    z_image_path: str = ""
    vae_path: str = ""
    qwen_path: str = ""
    qwen_tokenizer_path: str = ""
    max_sequence_length: int = 512
    use_x0: bool = False


# =============================================================================
# Base Trainer Class
# =============================================================================

class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.global_step = 0
        
        # Load and parse config
        self.config_data = self._load_config(config_path)
        self._parse_configs()
        
        # Initialize components (to be set by subclasses)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.logger = None
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return json.load(f)
    
    def _save_config(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.config_data, f, indent=4)
    
    @abstractmethod
    def _parse_configs(self):
        """Parse configuration into dataclasses."""
        pass
    
    @abstractmethod
    def setup(self):
        """Setup models, optimizer, dataset, etc."""
        pass
    
    @abstractmethod
    def train_step(self, batch) -> float:
        """Execute a single training step."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def run_inference(self) -> torch.Tensor:
        """Run inference for validation."""
        pass
    
    def train(self):
        """Main training loop."""
        raise NotImplementedError


# =============================================================================
# Timestep Sampler
# =============================================================================

class TimestepSampler:
    """Handles timestep sampling with custom distribution."""
    
    def __init__(self, num_points: int = 1000, device: torch.device = None):
        self.num_points = num_points
        self.device = device
        self._x = None
        self._probabilities = None
        self._cdf = None
    
    def _build_distribution(self, device: torch.device):
        """Build the timestep distribution (lazy initialization)."""
        if self._x is None or self._x.device != device:
            self._x = torch.linspace(0, 1, self.num_points, device=device)
            # Custom distribution favoring middle timesteps
            self._probabilities = -7.7 * ((self._x - 0.5) ** 2) + 2
            self._probabilities = self._probabilities.clamp(min=0)
            self._probabilities /= self._probabilities.sum()
            self._cdf = torch.cumsum(self._probabilities, dim=0)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from the distribution."""
        self._build_distribution(device)
        uniform_samples = torch.rand(num_samples, device=device)
        indices = torch.searchsorted(self._cdf, uniform_samples, right=True)
        indices = indices.clamp(max=self.num_points - 1)
        return self._x[indices]


# =============================================================================
# Text Encoder Wrapper
# =============================================================================

class TextEncoder:
    """Wrapper for Qwen3 text encoding."""
    
    def __init__(self, tokenizer, encoder, max_length: int, device: torch.device):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_length = max_length
        self.device = device
    
    def _format_prompts(self, captions: List[str]) -> List[str]:
        """Format prompts with chat template."""
        formatted = []
        for caption in captions:
            messages = [{"role": "user", "content": caption}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted.append(text)
        return formatted
    
    @torch.no_grad()
    def encode(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode captions to embeddings.
        
        Returns:
            embeddings: [B, seq_len, hidden_dim]
            mask: [B, seq_len] boolean mask
        """
        formatted = self._format_prompts(captions)
        
        inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        
        embeddings = outputs.hidden_states[-2]
        mask = inputs.attention_mask.bool()
        
        return embeddings, mask


# =============================================================================
# Experiment Logger
# =============================================================================

class ExperimentLogger:
    """Handles experiment tracking with Aim."""
    
    def __init__(self, config: TrainingConfig, hparams: Dict[str, Any]):
        self.enabled = AIM_AVAILABLE and config.use_aim and config.aim_path
        self.run = None
        
        if self.enabled:
            self.run = Run(
                repo=config.aim_path,
                experiment=config.aim_experiment_name,
            )
            self.run["hparams"] = hparams
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value."""
        if self.run:
            self.run.track(value, name=name, step=step)
    
    def log_image(self, name: str, image_path: str, caption: str, step: int):
        """Log an image."""
        if self.run and AIM_AVAILABLE:
            pil_img = PILImage.open(image_path)
            self.run.track(
                AimImage(pil_img, caption=caption),
                name=name,
                step=step,
            )
    
    def close(self):
        """Close the logger."""
        if self.run:
            self.run.close()


# =============================================================================
# Z-Image Trainer
# =============================================================================

class ZImageTrainer(BaseTrainer):
    """Trainer for Z-Image model."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        super().__init__(config_path, device)
        
        # Additional components
        self.ae = None
        self.tokenizer = None
        self.encoder = None
        self.text_encoder = None
        self.timestep_sampler = TimestepSampler()
    
    def _parse_configs(self):
        """Parse configuration into dataclasses."""
        self.training_config = TrainingConfig(**self.config_data.get("training", {}))
        self.inference_config = InferenceConfig(**self.config_data.get("inference", {}))
        self.dataloader_config = DataloaderConfig(**self.config_data.get("dataloader", {}))
        self.model_config = ModelConfig(**self.config_data.get("model", {}))
    
    def setup(self):
        """Setup all components for training."""
        print(f"Setting up trainer on device: {self.device}")
        
        # Set seeds
        torch.manual_seed(self.training_config.master_seed)
        random.seed(self.training_config.master_seed)
        
        # Create directories
        os.makedirs(self.training_config.save_folder, exist_ok=True)
        os.makedirs(self.inference_config.inference_folder, exist_ok=True)
        
        # Save config copy
        self._save_config(f"{self.training_config.save_folder}/training_config.json")
        
        # Load models
        self._load_models()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup dataset
        self._setup_dataset()
        
        # Setup logger
        self.logger = ExperimentLogger(self.training_config, self.config_data)
        
        print("Setup complete!")
    
    def _load_models(self):
        """Load all required models."""
        print("Loading models...")
        
        # Load Z-Image
        self._load_zimage()
        
        # Load VAE
        self._load_vae()
        
        # Load Qwen3
        self._load_text_encoder()
        
        print("All models loaded!")
    
    def _load_zimage(self):
        """Load Z-Image model."""
        print("  Loading Z-Image...")
        
        z_image_params._use_compiled = False
        z_image_params.use_x0 = self.model_config.use_x0
        
        with torch.device("meta"):
            self.model = ZImage(z_image_params)
        
        # Load weights
        state_dict = self._load_state_dict(self.model_config.z_image_path)
        if self.model_config.use_x0:
            state_dict["__x0__"] = torch.tensor([])
        
        self.model.load_state_dict(state_dict, assign=True)
        self.model = self.model.to(self.device).to(torch.bfloat16)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Z-Image loaded ({param_count:,} params)")
    
    def _load_vae(self):
        """Load VAE model."""
        print("  Loading VAE...")
        
        with torch.device("meta"):
            self.ae = AutoEncoder(ae_params)
        
        state_dict = self._load_state_dict(self.model_config.vae_path)
        self.ae.load_state_dict(state_dict, assign=True)
        self.ae = self.ae.to(self.device).to(torch.bfloat16)
        self.ae.eval()
        
        print("  VAE loaded")
    
    def _load_text_encoder(self):
        """Load Qwen3 tokenizer and encoder."""
        print("  Loading Qwen3...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.qwen_tokenizer_path
        )
        self.encoder = Qwen3ForCausalLM.from_pretrained(
            self.model_config.qwen_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.encoder.eval()
        
        # Create wrapper
        self.text_encoder = TextEncoder(
            self.tokenizer,
            self.encoder,
            self.model_config.max_sequence_length,
            self.device,
        )
        
        print("  Qwen3 loaded")
    
    def _load_state_dict(self, path: str) -> Dict[str, torch.Tensor]:
        """Load state dict from file."""
        if path.endswith((".safetensors", ".sft")):
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            return torch.load(path, map_location="cpu")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Filter trainable parameters
        trained_params = []
        frozen_count = 0
        
        keywords = self.training_config.trained_layer_keywords
        
        for name, param in self.model.named_parameters():
            if not keywords or any(kw in name for kw in keywords):
                param.requires_grad = True
                trained_params.append((name, param))
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Training {len(trained_params)} param groups, {frozen_count} frozen")
        
        # Separate weight decay groups
        decay = [p for n, p in trained_params if "bias" not in n and "norm" not in n]
        no_decay = [p for n, p in trained_params if "bias" in n or "norm" in n]
        
        param_groups = [
            {"params": decay, "weight_decay": self.training_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.training_config.lr,
            betas=(0.9, 0.999),
        )
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=self.training_config.warmup_steps,
        )
    
    def _setup_dataset(self):
        """Setup training dataset."""
        self.dataset = TextImageDataset(
            batch_size=self.dataloader_config.batch_size,
            jsonl_path=self.dataloader_config.jsonl_metadata_path,
            image_folder_path=self.dataloader_config.image_folder_path,
            base_res=self.dataloader_config.base_resolution,
            shuffle_tags=self.dataloader_config.shuffle_tags,
            tag_drop_percentage=self.dataloader_config.tag_drop_percentage,
            uncond_percentage=self.dataloader_config.uncond_percentage,
            resolution_step=self.dataloader_config.resolution_step,
            seed=self.training_config.master_seed,
            rank=0,
            num_gpus=1,
            ratio_cutoff=self.dataloader_config.ratio_cutoff,
            offset=self.dataloader_config.offset,
        )
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latents using VAE."""
        batch_size = images.shape[0]
        cache_mb = self.training_config.cache_minibatch
        latents_list = []
        
        self.ae.eval()
        for i in range(0, batch_size, cache_mb):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = images[i:i + cache_mb].to(self.device)
                latent_chunk = self.ae.encode_for_train(chunk)
                latents_list.append(latent_chunk.cpu())
        
        return torch.cat(latents_list, dim=0).to(self.device)
    
    def _prepare_batch(
        self, 
        latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Prepare training batch with optimal transport pairings.
        
        Returns:
            noisy_latents: Noised latent images
            target: Training target
            timesteps: Sampled timesteps
            latent_shape: Original shape tuple
        """
        latents = latents.to(torch.float32)
        latents, latent_shape = vae_flatten(latents)
        n = latents.shape[0]
        
        # Sample timesteps
        timesteps = self.timestep_sampler.sample(n, latents.device)
        timesteps_expanded = timesteps[:, None, None]
        
        # Generate and pair noise with optimal transport
        noise = torch.randn_like(latents)
        _, indices = cosine_optimal_transport(
            latents.reshape(n, -1),
            noise.reshape(n, -1)
        )
        noise = noise[indices[1].view(-1)]
        
        # Interpolate
        noisy_latents = latents * (1 - timesteps_expanded) + noise * timesteps_expanded
        
        # Compute target
        if self.model_config.use_x0:
            eps = 5e-2
            target = (noisy_latents - latents) / (timesteps_expanded + eps)
        else:
            target = noise - latents
        
        return noisy_latents, target, timesteps, latent_shape
    
    def train_step(self, batch) -> float:
        """Execute a single training step."""
        images, captions, _, loss_weights = batch
        
        # Preprocess captions
        captions = [c if c else "" for c in captions]
        captions = [c.lower() if random.random() < 0.25 else c for c in captions]
        
        loss_weights = torch.tensor(loss_weights, device=self.device)
        batch_size = images.shape[0]
        
        # Encode images
        latents = self._encode_images(images)
        
        # Prepare batch
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noisy_latents, target, timesteps, latent_shape = self._prepare_batch(latents)
        
        noisy_latents.requires_grad_(True)
        n, c, h, w = latent_shape
        
        # Training over minibatches
        train_mb = self.training_config.train_minibatch
        num_minibatches = batch_size // train_mb
        total_loss = 0.0
        
        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            # Encode text
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_embeds, text_mask = self.text_encoder.encode(captions[start:end])
                
                # Prepare position IDs
                token_lengths = text_mask.sum(1)
                image_pos_ids = prepare_latent_image_ids(token_lengths, h, w).to(self.device)
                text_pos_ids = make_text_position_ids(
                    token_lengths, 
                    self.model_config.max_sequence_length
                )
                img_mask = torch.ones(
                    (train_mb, noisy_latents.shape[1]),
                    device=self.device, 
                    dtype=torch.bool
                )
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = self.model(
                    img=noisy_latents[start:end],
                    img_ids=image_pos_ids,
                    img_mask=img_mask,
                    txt=text_embeds,
                    txt_ids=text_pos_ids,
                    txt_mask=text_mask,
                    timesteps=timesteps[start:end],
                )
                
                # Compute loss
                loss = ((pred - target[start:end]) ** 2).mean(dim=(1, 2))
                
                # Apply weights
                mb_weights = loss_weights[start:end]
                mb_weights = mb_weights / mb_weights.sum()
                loss = (loss * mb_weights).sum() / num_minibatches
            
            loss.backward()
            total_loss += loss.item()
        
        return total_loss
    
    @torch.no_grad()
    def run_inference(self, extra_prompts: List[str] = None) -> torch.Tensor:
        """
        Run inference to generate sample images.
        
        Args:
            extra_prompts: Additional prompts to include (e.g., from training batch)
        """
        self.model.eval()
        
        config = self.inference_config
        width, height = config.image_dim
        prompts = list(config.prompts)  # Copy to avoid modifying original
        
        # Add extra prompts from training batch
        if extra_prompts:
            prompts.extend(extra_prompts)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate noise
            noise = get_noise(len(prompts), height, width, self.device, torch.bfloat16, 
                            self.training_config.master_seed)
            noise, shape = vae_flatten(noise)
            n, c, h, w = shape
            
            timesteps = get_schedule(config.steps, noise.shape[1])
            
            # Encode prompts
            pos_embeds, pos_mask = self.text_encoder.encode(prompts)
            neg_embeds, neg_mask = self.text_encoder.encode([""] * len(prompts))
            
            # Position IDs
            pos_lengths = pos_mask.sum(1)
            neg_lengths = neg_mask.sum(1)
            offset = torch.maximum(pos_lengths, neg_lengths)
            
            image_pos_ids = prepare_latent_image_ids(offset, h, w).to(self.device)
            pos_text_ids = make_text_position_ids(pos_lengths, config.max_sequence_length)
            neg_text_ids = make_text_position_ids(neg_lengths, config.max_sequence_length)
            
            # Denoise
            output = denoise_cfg(
                self.model, noise, image_pos_ids,
                pos_embeds, neg_embeds,
                pos_text_ids, neg_text_ids,
                pos_mask, neg_mask,
                timesteps, config.cfg,
                config.first_n_steps_wo_cfg,
            )
            
            # Decode
            images = self.ae.decode(vae_unflatten(output, shape))
        
        self.model.train()
        return images
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")
    
    def _create_dataloader(self) -> DataLoader:
        """Create a new dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.dataloader_config.num_workers,
            prefetch_factor=self.dataloader_config.prefetch_factor,
            pin_memory=True,
            collate_fn=self.dataset.dummy_collate_fn,
        )
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        self.model.train()
        
        # Setup profiler
        do_profiling = self.training_config.do_profiling
        profile_steps = self.training_config.profile_steps
        profiler_active = do_profiling
        
        profiler_ctx = (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
            if do_profiling
            else nullcontext()
        )
        
        with profiler_ctx as prof:
            while True:
                # Update seed each epoch
                self.training_config.master_seed += 1
                torch.manual_seed(self.training_config.master_seed)
                
                dataloader = self._create_dataloader()
                pbar = tqdm(enumerate(dataloader), total=len(self.dataset), desc="Training")
                
                for step, batch_data in pbar:
                    # Stop profiler after profile_steps
                    if profiler_active and step >= profile_steps:
                        prof.stop()
                        prof.export_chrome_trace("profiler_trace.json")
                        print(f"\nProfiler stopped. Trace saved to profiler_trace.json")
                        profiler_active = False
                    
                    batch_data = batch_data[0]  # Unwrap
                    
                    # Train step
                    loss = self.train_step(batch_data)
                    
                    # Gradient clipping
                    if self.training_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.max_grad_norm
                        )
                    
                    # Optimizer step with gradient accumulation
                    if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    # Update progress
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
                    
                    # Logging
                    self.logger.log_scalar("loss", loss, self.global_step)
                    self.logger.log_scalar("learning_rate", lr, self.global_step)
                    
                    # Checkpointing
                    if (step + 1) % self.training_config.save_every == 0:
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        ckpt_path = f"{self.training_config.save_folder}/{timestamp}.pth"
                        self.save_checkpoint(ckpt_path)
                        
                        # Update config
                        self.config_data["model"]["z_image_path"] = ckpt_path
                        self.config_data["dataloader"]["offset"] = self.dataloader_config.offset + step
                        self._save_config(f"{self.training_config.save_folder}/training_config.json")
                    
                    # Inference
                    if (step + 1) % self.inference_config.inference_every == 0:
                        print("\nRunning inference...")
                        # Get one prompt from current batch for inference
                        extra_prompts = [batch_data[1][0]] if batch_data[1] else []
                        images = self.run_inference(extra_prompts=extra_prompts)
                        
                        output_path = f"{self.inference_config.inference_folder}/{self.global_step}.png"
                        save_image(
                            images.clamp(-1, 1).add(1).div(2),
                            output_path,
                            nrow=min(4, len(images)),
                        )
                        print(f"Saved: {output_path}")
                        
                        caption = "\n".join(self.inference_config.prompts)
                        self.logger.log_image("samples", output_path, caption, self.global_step)
                        
                        self.model.train()
                    
                    self.global_step += 1
                    self.dataloader_config.offset += 1
                
                # End of epoch checkpoint
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                self.save_checkpoint(f"{self.training_config.save_folder}/{timestamp}_epoch_end.pth")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Z-Image model (single GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_example.json",
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    trainer = ZImageTrainer(config_path=args.config, device=args.device)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
