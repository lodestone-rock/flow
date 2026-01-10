"""Multi-GPU Z-Image Pixel Space Trainer (Class-based)

A trainer for Z-Image model in direct pixel space using ThreadPoolExecutor for multi-GPU parallelism.
Uses NCCL for gradient all-reduce across GPUs.

Based on train_zimage_dct_class_multi_gpu.py but operates directly in pixel space:
- No VAE encoding/decoding
- Uses patch_size=32 to convert RGB images to 3072-channel patches (32x32x3)
- Direct pixel-level flow matching
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import copy
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.cuda.nccl as nccl
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity, record_function

from tqdm import tqdm
from safetensors.torch import safe_open
from transformers import AutoTokenizer, Qwen3ForCausalLM

# from ramtorch import AdamW
from torch.optim import AdamW

from src.dataloaders.dataloader import TextImageDataset
from src.models.zimage.model_dct import ZImageDCT, ZImageDCTParams
from src.models.zimage.sampling import get_schedule, denoise_cfg
from src.models.zimage.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    make_text_position_ids,
)



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
    
    # Pixel space settings
    pixel_patch_size: int = 32  # 32x32 RGB patches
    
    # Profiling
    do_profiling: bool = False
    profile_steps: int = 2
    profile_json_dump: str = "profiler_dump.json"
    
    # Experiment tracking
    use_aim: bool = False
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None
    
    @property
    def pixel_in_channels(self) -> int:
        """Derived: patch_size * patch_size * 3 (RGB)."""
        return self.pixel_patch_size * self.pixel_patch_size * 3


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

# =============================================================================
# Weight Loading Utilities for Pixel Space Model
# =============================================================================

def load_weights_with_mismatch_handling(
    model: nn.Module,
    checkpoint_state_dict: dict,
    use_x0: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load weights from checkpoint into model, handling mismatches gracefully.
    
    - If checkpoint has a key with matching shape -> load it
    - If checkpoint has a key with mismatched shape -> keep random init, log warning
    - If model has a key not in checkpoint -> keep random init (new layer)
    - If checkpoint has extra keys not in model -> skip them
    
    Args:
        model: The model to load weights into
        checkpoint_state_dict: State dict from checkpoint file
        use_x0: Whether to register the __x0__ buffer
        
    Returns:
        Tuple of (loaded_keys, shape_mismatch_keys, new_keys)
    """
    model_state_dict = model.state_dict()
    
    loaded_keys = []
    shape_mismatch_keys = []
    new_keys = []
    extra_checkpoint_keys = []
    
    # Add __x0__ marker if needed
    if use_x0:
        checkpoint_state_dict["__x0__"] = torch.tensor([])
    
    # Check each key in the model
    for model_key, model_tensor in model_state_dict.items():
        if model_key in checkpoint_state_dict:
            ckpt_tensor = checkpoint_state_dict[model_key]
            if model_tensor.shape == ckpt_tensor.shape:
                # Shape matches - load the weight
                model_state_dict[model_key] = ckpt_tensor
                loaded_keys.append(model_key)
            else:
                # Shape mismatch - keep random init
                shape_mismatch_keys.append(
                    f"{model_key}: model={list(model_tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                )
        else:
            # Key not in checkpoint - this is a new layer, keep random init
            new_keys.append(model_key)
    
    # Check for extra keys in checkpoint that aren't in model
    for ckpt_key in checkpoint_state_dict.keys():
        if ckpt_key not in model_state_dict:
            extra_checkpoint_keys.append(ckpt_key)
    
    # Load the state dict (strict=False since we've already handled mismatches)
    model.load_state_dict(model_state_dict, assign=True)
    
    # Print summary
    print(f"    Loaded: {len(loaded_keys)} keys")
    if shape_mismatch_keys:
        print(f"    Shape mismatch (random init): {len(shape_mismatch_keys)} keys")
        for key in shape_mismatch_keys[:5]:  # Show first 5
            print(f"      - {key}")
        if len(shape_mismatch_keys) > 5:
            print(f"      ... and {len(shape_mismatch_keys) - 5} more")
    if new_keys:
        print(f"    New layers (random init): {len(new_keys)} keys")
        # Group by prefix for cleaner output
        prefixes = set(k.split('.')[0] for k in new_keys)
        for prefix in sorted(prefixes):
            count = sum(1 for k in new_keys if k.startswith(prefix))
            print(f"      - {prefix}.* ({count} params)")
    if extra_checkpoint_keys:
        print(f"    Ignored from checkpoint: {len(extra_checkpoint_keys)} keys")
    
    return loaded_keys, shape_mismatch_keys, new_keys


# =============================================================================
# Z-Image Pixel Space Trainer
# =============================================================================

class ZImagePixelSpaceTrainer(BaseTrainer):
    """Trainer for Z-Image model in pixel space with multi-GPU support (DDP-like)."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        super().__init__(config_path, device)
        
        # Multi-GPU setup - one of each component per GPU
        self.n_gpus = torch.cuda.device_count()
        self.models = []  # One model per GPU
        self.text_encoders = []  # One text encoder per GPU
        self.executor = None
        
        # Shared tokenizer (CPU-based, thread-safe)
        self.tokenizer = None
        self.timestep_samplers = []  # One per GPU
    
    def _parse_configs(self):
        """Parse configuration into dataclasses."""
        self.training_config = TrainingConfig(**self.config_data.get("training", {}))
        self.inference_config = InferenceConfig(**self.config_data.get("inference", {}))
        self.dataloader_config = DataloaderConfig(**self.config_data.get("dataloader", {}))
        self.model_config = ModelConfig(**self.config_data.get("model", {}))
    
    def setup(self):
        """Setup all components for training."""
        print(f"Setting up trainer with {self.n_gpus} GPUs")
        
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
        
        # Setup thread pool for multi-GPU execution
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)
        
        print("Setup complete!")
    
    def _load_models(self):
        """Load all required models and replicate to all GPUs."""
        print("Loading models...")
        
        # Load Z-Image (replicated) - configured for pixel space
        self._load_zimage()
        
        # Load Qwen3 text encoder (replicated)
        self._load_text_encoder()
        
        # Create timestep samplers (one per GPU)
        self._setup_timestep_samplers()
        
        print("All models loaded!")
    
    def _load_zimage(self):
        """Load Z-Image model for pixel space and replicate to all GPUs.
        
        Uses meta device to avoid allocating RAM for the base model,
        then materializes directly onto each GPU.
        """
        print("  Loading Z-Image (Pixel Space)...")
        
        patch_size = self.training_config.pixel_patch_size
        in_channels = self.training_config.pixel_in_channels
        
        # Create params for pixel space model
        # patch_size=32, in_channels=3072 (32x32x3 RGB patches)
        pixel_params = ZImageDCTParams(
            patch_size=1,
            in_channels=in_channels,
            use_x0=self.model_config.use_x0,
        )
        
        # Load checkpoint state dict first (if provided)
        checkpoint_state_dict = None
        if self.model_config.z_image_path:
            print(f"    Loading checkpoint: {self.model_config.z_image_path}")
            checkpoint_state_dict = self._load_state_dict(self.model_config.z_image_path)
            if self.model_config.use_x0:
                checkpoint_state_dict["__x0__"] = torch.tensor([])
        
        # Create models directly on each GPU using meta device to save RAM
        self.models = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            print(f"    Initializing model on {device}...")
            
            # Create model on meta device (no memory allocation)
            with torch.device('meta'):
                model = ZImageDCT(pixel_params)
            
            if checkpoint_state_dict is not None:
                # Materialize model with checkpoint weights directly on target device
                model_state_dict = model.state_dict()
                loaded_keys = []
                shape_mismatch_keys = []
                new_keys = []
                
                for model_key, model_tensor in model_state_dict.items():
                    if model_key in checkpoint_state_dict:
                        ckpt_tensor = checkpoint_state_dict[model_key]
                        if model_tensor.shape == ckpt_tensor.shape:
                            # Shape matches - use checkpoint weight
                            model_state_dict[model_key] = ckpt_tensor.to(device=device, dtype=torch.bfloat16)
                            loaded_keys.append(model_key)
                        else:
                            # Shape mismatch - random init on device
                            model_state_dict[model_key] = torch.empty(
                                model_tensor.shape, device=device, dtype=torch.bfloat16
                            )
                            nn.init.kaiming_uniform_(model_state_dict[model_key]) if model_state_dict[model_key].dim() > 1 else nn.init.zeros_(model_state_dict[model_key])
                            shape_mismatch_keys.append(
                                f"{model_key}: model={list(model_tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                            )
                    else:
                        # New layer - random init on device
                        model_state_dict[model_key] = torch.empty(
                            model_tensor.shape, device=device, dtype=torch.bfloat16
                        )
                        if model_state_dict[model_key].dim() > 1:
                            nn.init.kaiming_uniform_(model_state_dict[model_key])
                        else:
                            nn.init.zeros_(model_state_dict[model_key])
                        new_keys.append(model_key)
                
                # Load state dict with assign=True to replace meta tensors
                model.load_state_dict(model_state_dict, assign=True)
                
                if gpu_id == 0:
                    print(f"    Loaded: {len(loaded_keys)} keys")
                    if shape_mismatch_keys:
                        print(f"    Shape mismatch (random init): {len(shape_mismatch_keys)} keys")
                        for key in shape_mismatch_keys[:5]:
                            print(f"      - {key}")
                        if len(shape_mismatch_keys) > 5:
                            print(f"      ... and {len(shape_mismatch_keys) - 5} more")
                    if new_keys:
                        prefixes = set(k.split('.')[0] for k in new_keys)
                        print(f"    New layers (random init): {len(new_keys)} keys")
                        for prefix in sorted(prefixes):
                            count = sum(1 for k in new_keys if k.startswith(prefix))
                            print(f"      - {prefix}.* ({count} params)")
            else:
                # No checkpoint - random init directly on device
                print("    No checkpoint provided, using random initialization")
                model_state_dict = {}
                for name, param in model.named_parameters():
                    tensor = torch.empty(param.shape, device=device, dtype=torch.bfloat16)
                    if tensor.dim() > 1:
                        nn.init.kaiming_uniform_(tensor)
                    else:
                        nn.init.zeros_(tensor)
                    model_state_dict[name] = tensor
                for name, buf in model.named_buffers():
                    model_state_dict[name] = torch.zeros(buf.shape, device=device, dtype=torch.bfloat16)
                
                if self.model_config.use_x0:
                    model_state_dict["__x0__"] = torch.tensor([], device=device)
                
                model.load_state_dict(model_state_dict, assign=True)
            
            self.models.append(model)
        
        # Keep reference to first model for compatibility
        self.model = self.models[0]
        
        total_params = sum(p.numel() for p in self.model.parameters())
        dec_net_params = sum(p.numel() for n, p in self.model.named_parameters() if 'dec_net' in n)
        print(f"  Z-Image Pixel Space loaded on {self.n_gpus} GPUs ({total_params:,} params each)")
        print(f"  Patch size: {patch_size}x{patch_size}, in_channels: {in_channels}")
        print(f"  dec_net parameters: {dec_net_params:,}")
    
    
    def _load_text_encoder(self):
        """Load Qwen3 tokenizer and encoder, replicate to all GPUs."""
        print("  Loading Qwen3...")
        
        # Shared tokenizer (CPU-based)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.qwen_tokenizer_path
        )
        
        # Load base encoder
        base_encoder = Qwen3ForCausalLM.from_pretrained(
            self.model_config.qwen_path,
            torch_dtype=torch.bfloat16,
        )
        
        # Replicate to all GPUs
        self.text_encoders = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            encoder_copy = copy.deepcopy(base_encoder).to(device).to(torch.bfloat16)
            encoder_copy.eval()
            
            text_encoder = TextEncoder(
                self.tokenizer,
                encoder_copy,
                self.model_config.max_sequence_length,
                torch.device(device),
            )
            self.text_encoders.append(text_encoder)
        
        # Keep reference for compatibility
        self.text_encoder = self.text_encoders[0]
        self.encoder = self.text_encoders[0].encoder
        
        print(f"  Qwen3 loaded on {self.n_gpus} GPUs")
    
    def _setup_timestep_samplers(self):
        """Create one timestep sampler per GPU."""
        print("  Setting up timestep samplers...")
        self.timestep_samplers = [TimestepSampler() for _ in range(self.n_gpus)]
        print(f"  Timestep samplers created for {self.n_gpus} GPUs")
    
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
        """Setup optimizer and scheduler for each GPU."""
        keywords = self.training_config.trained_layer_keywords
        
        self.optimizers = []
        self.schedulers = []
        
        for gpu_id, model in enumerate(self.models):
            # Filter trainable parameters
            trained_params = []
            frozen_count = 0
            
            for name, param in model.named_parameters():
                if not keywords or any(kw in name for kw in keywords):
                    param.requires_grad = True
                    trained_params.append((name, param))
                else:
                    param.requires_grad = False
                    frozen_count += 1
                    print(f"param {name} is frozen!")
            
            if gpu_id == 0:
                print(f"Training {len(trained_params)} param groups, {frozen_count} frozen (per GPU)")
            
            # Separate weight decay groups
            decay = [p for n, p in trained_params if "bias" not in n and "norm" not in n]
            no_decay = [p for n, p in trained_params if "bias" in n or "norm" in n]
            
            param_groups = [
                {"params": decay, "weight_decay": self.training_config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            
            optimizer = AdamW(
                param_groups,
                lr=self.training_config.lr,
                betas=(0.9, 0.999),
            )
            
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.05,
                end_factor=1.0,
                total_iters=self.training_config.warmup_steps,
            )
            
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        
        # Keep reference for compatibility
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]
    
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
    
    def _all_reduce_gradients(self):
        """All-reduce gradients across all GPU models using NCCL."""
        # Get list of trainable parameters from each model
        param_lists = [
            [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            for model in self.models
        ]
        
        # All-reduce each parameter's gradient
        n_params = len(param_lists[0])
        for param_idx in range(n_params):
            grads = [param_lists[gpu_id][param_idx].grad for gpu_id in range(self.n_gpus)]
            nccl.all_reduce(grads, op=nccl.SUM)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.logger:
            self.logger.close()
    
    def _forward_backward_on_gpu(
        self,
        gpu_id: int,
        images_chunk: torch.Tensor,
        captions_chunk: List[str],
        loss_weights_chunk: List[float],
    ) -> float:
        """Run forward/backward pass on a single GPU with its data chunk (pixel space)."""
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        text_encoder = self.text_encoders[gpu_id]
        patch_size = self.training_config.pixel_patch_size
        
        batch_size = images_chunk.shape[0]
        
        # Move images to GPU and patchify directly (no VAE)
        # Images are [B, 3, H, W] in range [-1, 1]
        # vae_flatten with patch_size converts to [B, num_patches, patch_size*patch_size*3]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images_gpu = images_chunk.to(device)
            # Patchify: [B, 3, H, W] -> [B, num_patches, patch_size*patch_size*3]
            pixels, pixel_shape = vae_flatten(images_gpu, patch_size=patch_size)
            pixels = pixels.to(torch.float32)
            n = pixels.shape[0]
            
            # Sample timesteps using this GPU's sampler
            timesteps = self.timestep_samplers[gpu_id].sample(n, pixels.device)
            timesteps_expanded = timesteps[:, None, None]
            
            # Generate noise in patch space
            noise = torch.randn_like(pixels)
            
            # Interpolate (flow matching)
            noisy_pixels = pixels * (1 - timesteps_expanded) + noise * timesteps_expanded
            
            # Compute target
            if self.model_config.use_x0:
                eps = 5e-2
                target = (noisy_pixels - pixels) / (timesteps_expanded + eps)
            else:
                target = noise - pixels
        
        noisy_pixels.requires_grad_(True)
        n, c, h, w = pixel_shape
        # For pixel space: h and w are the spatial dims after patchifying
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        loss_weights = torch.tensor(loss_weights_chunk, device=device)
        
        # Training over minibatches within this GPU's chunk
        train_mb = self.training_config.train_minibatch
        num_minibatches = batch_size // train_mb
        total_loss = 0.0
        
        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            # Encode text using this GPU's text encoder
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_embeds, text_mask = text_encoder.encode(captions_chunk[start:end])
                
                # Prepare position IDs
                token_lengths = text_mask.sum(1)
                # Use patch dimensions for position IDs (not original image dims)
                image_pos_ids = prepare_latent_image_ids(
                    token_lengths, h_patches, w_patches, patch_size=1
                ).to(device)
                text_pos_ids = make_text_position_ids(
                    token_lengths, 
                    self.model_config.max_sequence_length
                ).to(device)
                img_mask = torch.ones(
                    (train_mb, noisy_pixels.shape[1]),
                    device=device,
                    dtype=torch.bool
                )
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
                    img=noisy_pixels[start:end],
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
    
    def _clip_grads_on_gpu(self, gpu_id: int):
        """Clip gradients on a specific GPU."""
        if self.training_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.models[gpu_id].parameters(),
                self.training_config.max_grad_norm
            )
    
    def _optimizer_step_on_gpu(self, gpu_id: int):
        """Run optimizer step on a specific GPU."""
        self.optimizers[gpu_id].step()
        self.schedulers[gpu_id].step()
        self.optimizers[gpu_id].zero_grad()
    
    def train_step(self, batch) -> float:
        """Execute forward/backward pass across all GPUs (without all-reduce)."""
        images, captions, _, loss_weights = batch
        
        # Preprocess captions
        captions = [c if c else "" for c in captions]
        captions = [c.lower() if random.random() < 0.25 else c for c in captions]
        
        batch_size = images.shape[0]
        samples_per_gpu = batch_size // self.n_gpus
        
        # Split batch across GPUs and run forward/backward in parallel
        def gpu_forward_backward(gpu_id):
            start = gpu_id * samples_per_gpu
            end = start + samples_per_gpu
            return self._forward_backward_on_gpu(
                gpu_id=gpu_id,
                images_chunk=images[start:end],
                captions_chunk=captions[start:end],
                loss_weights_chunk=loss_weights[start:end],
            )
        
        # Forward/backward on all GPUs in parallel
        losses = list(self.executor.map(gpu_forward_backward, range(self.n_gpus)))
        total_loss = sum(losses) / self.n_gpus  # Average loss across GPUs
        
        return total_loss
    
    def _inference_on_gpu(
        self,
        gpu_id: int,
        prompts: List[str],
        config: InferenceConfig,
    ) -> torch.Tensor:
        """Run inference on a single GPU in pixel space (no VAE)."""
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        text_encoder = self.text_encoders[gpu_id]
        patch_size = self.training_config.pixel_patch_size
        in_channels = self.training_config.pixel_in_channels
        
        model.eval()
        
        width, height = config.image_dim
        
        # Calculate patch grid dimensions
        h_patches = height // patch_size
        w_patches = width // patch_size
        num_patches = h_patches * w_patches
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate noise directly in patch space
            # Shape: [B, num_patches, patch_size*patch_size*3]
            generator = torch.Generator(device=device).manual_seed(
                self.training_config.master_seed + gpu_id
            )
            noise = torch.randn(
                len(prompts), num_patches, in_channels,
                device=device, dtype=torch.bfloat16, generator=generator
            )
            
            # Store shape info for unpatchifying later
            # We need to create a fake shape tuple that vae_unflatten expects
            # Original image shape: [B, 3, H, W]
            pixel_shape = (len(prompts), 3, height, width)
            
            timesteps = get_schedule(config.steps, noise.shape[1])
            
            # Encode prompts using this GPU's text encoder
            pos_embeds, pos_mask = text_encoder.encode(prompts)
            neg_embeds, neg_mask = text_encoder.encode([""] * len(prompts))
            
            # Position IDs (use patch grid dimensions)
            pos_lengths = pos_mask.sum(1)
            neg_lengths = neg_mask.sum(1)
            offset = torch.maximum(pos_lengths, neg_lengths)
            
            image_pos_ids = prepare_latent_image_ids(
                offset, h_patches, w_patches, patch_size=1
            ).to(device)
            pos_text_ids = make_text_position_ids(pos_lengths, config.max_sequence_length).to(device)
            neg_text_ids = make_text_position_ids(neg_lengths, config.max_sequence_length).to(device)
            
            # Denoise in patch space
            output = denoise_cfg(
                model, noise, image_pos_ids,
                pos_embeds, neg_embeds,
                pos_text_ids, neg_text_ids,
                pos_mask, neg_mask,
                timesteps, config.cfg,
                config.first_n_steps_wo_cfg,
            )
            
            # Unpatchify: [B, num_patches, in_channels] -> [B, 3, H, W]
            images = vae_unflatten(output, pixel_shape, patch_size=patch_size)
        
        model.train()
        return images
    
    def _set_models_train_mode(self, gpu_id: int):
        """Set model back to train mode on a specific GPU."""
        self.models[gpu_id].train()
    
    @torch.no_grad()
    def run_inference(self, extra_prompts_per_gpu: List[str] = None) -> torch.Tensor:
        """
        Run inference to generate sample images across multiple GPUs (pixel space).
        Each GPU generates: (inference prompts + 1 extra prompt from batch)
        No VAE decoding needed - outputs are directly in pixel space.
        
        Args:
            extra_prompts_per_gpu: One extra prompt per GPU (e.g., from training batch)
        """
        config = self.inference_config
        base_prompts = list(config.prompts)  # Copy to avoid modifying original
        
        def inference_on_gpu(gpu_id):
            # Each GPU gets the same base prompts + its own extra prompt
            prompts = list(base_prompts)
            if extra_prompts_per_gpu and gpu_id < len(extra_prompts_per_gpu):
                prompts.append(extra_prompts_per_gpu[gpu_id])
            return self._inference_on_gpu(gpu_id, prompts, config)
        
        # Run inference on all GPUs in parallel (each GPU generates full set)
        results = list(self.executor.map(inference_on_gpu, range(self.n_gpus)))
        
        # Set all models back to train mode
        list(self.executor.map(self._set_models_train_mode, range(self.n_gpus)))
        
        # Move all results to primary GPU for concatenation
        results = [r.to(self.device) for r in results]
        images = torch.cat(results, dim=0) if results else torch.empty(0)
        
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
                        prof.export_chrome_trace(self.training_config.profile_json_dump)
                        print(f"\nProfiler stopped. Trace saved to {self.training_config.profile_json_dump}")
                        profiler_active = False
                    
                    batch_data = batch_data[0]  # Unwrap
                    
                    # Train step (forward/backward on all GPUs)
                    loss = self.train_step(batch_data)

                    if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                        # All-reduce gradients across GPUs (synchronous on main thread)
                        self._all_reduce_gradients()
                        
                        # Gradient clipping on all GPUs in parallel
                        list(self.executor.map(self._clip_grads_on_gpu, range(self.n_gpus)))
                        
                        # Optimizer step with gradient accumulation (on all GPUs in parallel)
                        list(self.executor.map(self._optimizer_step_on_gpu, range(self.n_gpus)))
                    
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
                    
                    # Inference (distributed across GPUs)
                    if (step + 1) % self.inference_config.inference_every == 0:
                        print("\nRunning inference...")
                        # Get one prompt per GPU from current batch
                        captions = batch_data[1]
                        extra_prompts_per_gpu = []
                        if captions:
                            samples_per_gpu = len(captions) // self.n_gpus
                            for gpu_id in range(self.n_gpus):
                                idx = gpu_id * samples_per_gpu
                                if idx < len(captions) and captions[idx]:
                                    extra_prompts_per_gpu.append(captions[idx])
                        images = self.run_inference(extra_prompts_per_gpu=extra_prompts_per_gpu)
                        
                        output_path = f"{self.inference_config.inference_folder}/{self.global_step}.png"
                        save_image(
                            images.clamp(-1, 1).add(1).div(2),
                            output_path,
                            nrow=min(40, len(images)//self.n_gpus),
                        )
                        print(f"Saved: {output_path}")
                        
                        caption = "\n".join(self.inference_config.prompts)
                        self.logger.log_image("samples", output_path, caption, self.global_step)
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
    
    parser = argparse.ArgumentParser(description="Train Z-Image Pixel Space model (multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_example_pixelspace.json",
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    trainer = ZImagePixelSpaceTrainer(config_path=args.config, device=args.device)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
