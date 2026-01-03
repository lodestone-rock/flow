"""Multi-GPU Z-Image VAE-Only Trainer (Class-based)

A trainer for Z-Image model using ThreadPoolExecutor for multi-GPU parallelism.
Uses NCCL for gradient all-reduce across GPUs.

Based on train_zimage_class_multi_gpu.py but uses:
- ZImage model with 32-channel VAE (128 dim after patchify with patch_size=1)
- No NeRF/DCT head - direct VAE latent prediction
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
from src.models.zimage.model import ZImage, ZImageParams
from src.models.zimage.sampling import get_noise, get_schedule, denoise_cfg
from src.models.zimage.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    make_text_position_ids,
)
from src.models.zimage.autoencoder_c32 import AutoEncoder, AutoEncoderParams


# =============================================================================
# Z-Image Parameters for 32-channel VAE (128 dim)
# =============================================================================

# Parameters for 32-channel VAE model (patch_size=1, in_channels=128)
z_image_params_c32 = ZImageParams(
    all_patch_size=(1,),  # patch_size=1 for 32ch VAE
    all_f_patch_size=(1,),
    in_channels=128,  # 32 VAE channels * 4 (2x2 spatial) = 128 dim
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256,
    t_scale=1000.0,
    axes_dims=[32, 48, 48],
    axes_lens=[1536, 512, 512],
    adaln_embed_dim=256,
    use_x0=False,
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
    
    # Profiling
    do_profiling: bool = False
    profile_steps: int = 2
    profile_json_dump: str = "profiler_dump.json"
    
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

# =============================================================================
# Weight Loading Utilities for 32-channel VAE Model
# =============================================================================

def extend_weights_for_c32(original_state_dict: dict) -> dict:
    """
    Extend weights from 16-channel model (patch_size=2, 64 dim) to 128-channel model (patch_size=1, 128 dim).
    
    Key mappings:
    - all_x_embedder.2-1 -> all_x_embedder.1-1 (with dimension extension 64 -> 128)
    - all_final_layer.2-1 -> all_final_layer.1-1 (with dimension extension 64 -> 128)
    
    If weights are already in 128-dim format (1-1 keys), they are passed through unchanged.
    Pads with zeros to preserve original model behavior on first 64 dims.
    """
    new_state_dict = {}
    
    # Check if weights are already in 128-dim format (1-1 keys)
    has_c128_keys = any("all_x_embedder.1-1" in key for key in original_state_dict.keys())
    has_c64_keys = any("all_x_embedder.2-1" in key for key in original_state_dict.keys())
    
    if has_c128_keys and not has_c64_keys:
        # Already in 128-dim format with 1-1 keys, pass through unchanged
        print("  Weights already in 128-dim format (1-1 keys), skipping extension")
        return original_state_dict
    
    if has_c128_keys and has_c64_keys:
        print("  Warning: Found both 1-1 and 2-1 keys, using 2-1 keys for extension")
    
    for key, value in original_state_dict.items():
        # Map x_embedder from 2-1 to 1-1
        if "all_x_embedder.2-1.weight" in key:
            new_key = key.replace("all_x_embedder.2-1", "all_x_embedder.1-1")
            # Check if already correct size (128 dim)
            if value.shape[1] == 128:
                new_state_dict[new_key] = value
                print(f"  Already correct size, renamed {key} -> {new_key}: {value.shape}")
            else:
                # x_embedder.weight: [dim, 64] -> [dim, 128]
                old_in = value.shape[1]  # 64
                new_in = old_in * 2  # 128
                new_weight = torch.zeros(value.shape[0], new_in, dtype=value.dtype)
                new_weight[:, :old_in] = value
                new_state_dict[new_key] = new_weight
                print(f"  Extended {key} -> {new_key}: {value.shape} -> {new_weight.shape}")
        elif "all_x_embedder.2-1.bias" in key:
            new_key = key.replace("all_x_embedder.2-1", "all_x_embedder.1-1")
            new_state_dict[new_key] = value
            print(f"  Renamed {key} -> {new_key}")
        # Map final_layer from 2-1 to 1-1
        elif "all_final_layer.2-1" in key:
            new_key = key.replace("all_final_layer.2-1", "all_final_layer.1-1")
            if "linear.weight" in key:
                # Check if already correct size (128 dim)
                if value.shape[0] == 128:
                    new_state_dict[new_key] = value
                    print(f"  Already correct size, renamed {key} -> {new_key}: {value.shape}")
                else:
                    # final_layer linear.weight: [64, dim] -> [128, dim]
                    old_out = value.shape[0]  # 64
                    new_out = old_out * 2  # 128
                    new_weight = torch.zeros(new_out, value.shape[1], dtype=value.dtype)
                    new_weight[:old_out, :] = value
                    new_state_dict[new_key] = new_weight
                    print(f"  Extended {key} -> {new_key}: {value.shape} -> {new_weight.shape}")
            elif "linear.bias" in key:
                # Check if already correct size (128 dim)
                if value.shape[0] == 128:
                    new_state_dict[new_key] = value
                    print(f"  Already correct size, renamed {key} -> {new_key}: {value.shape}")
                else:
                    # final_layer linear.bias: [64] -> [128]
                    old_out = value.shape[0]  # 64
                    new_out = old_out * 2  # 128
                    new_bias = torch.zeros(new_out, dtype=value.dtype)
                    new_bias[:old_out] = value
                    new_state_dict[new_key] = new_bias
                    print(f"  Extended {key} -> {new_key}: {value.shape} -> {new_bias.shape}")
            else:
                # Other final_layer weights (adaLN_modulation, norm) - just rename
                new_state_dict[new_key] = value
                print(f"  Renamed {key} -> {new_key}")
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def load_zimage_weights_to_c32(c32_model: ZImage, original_state_dict: dict) -> ZImage:
    """
    Load weights from original ZImage model (16ch, patch_size=2) into 32-channel model (patch_size=1).
    Extends x_embedder and final_layer dimensions, maps key names.
    """
    # First extend weights for 32 channels
    extended_state_dict = extend_weights_for_c32(original_state_dict)
    
    c32_state_dict = c32_model.state_dict()
    loaded_keys = []
    skipped_keys = []
    new_keys = []
    
    for orig_key, value in extended_state_dict.items():
        if orig_key in c32_state_dict:
            if c32_state_dict[orig_key].shape == value.shape:
                c32_state_dict[orig_key] = value
                loaded_keys.append(orig_key)
            else:
                skipped_keys.append(f"{orig_key} (shape mismatch: {value.shape} vs {c32_state_dict[orig_key].shape})")
        else:
            skipped_keys.append(f"{orig_key} (not found in c32 model)")
    
    # Find keys that weren't loaded (new/randomly initialized)
    for key in c32_state_dict.keys():
        found = False
        for loaded_key in loaded_keys:
            if key == loaded_key:
                found = True
                break
        if not found:
            new_keys.append(key)
    
    c32_model.load_state_dict(c32_state_dict)
    
    print(f"  Loaded: {len(loaded_keys)} keys")
    print(f"  Skipped: {len(skipped_keys)} keys")
    if skipped_keys:
        for sk in skipped_keys[:10]:  # Show first 10
            print(f"    - {sk}")
        if len(skipped_keys) > 10:
            print(f"    ... and {len(skipped_keys) - 10} more")
    print(f"  New (randomly initialized): {len(new_keys)} keys")
    
    return c32_model


# =============================================================================
# Z-Image VAE-Only Trainer (32-channel)
# =============================================================================

class ZImageVAETrainer(BaseTrainer):
    """Trainer for Z-Image model with 32-channel VAE (multi-GPU support, DDP-like)."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        super().__init__(config_path, device)
        
        # Multi-GPU setup - one of each component per GPU
        self.n_gpus = torch.cuda.device_count()
        self.models = []  # One model per GPU
        self.vaes = []  # One VAE per GPU
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
        
        # Load Z-Image (replicated)
        self._load_zimage()
        
        # Load VAE (replicated)
        self._load_vae()
        
        # Load Qwen3 text encoder (replicated)
        self._load_text_encoder()
        
        # Create timestep samplers (one per GPU)
        self._setup_timestep_samplers()
        
        print("All models loaded!")
    
    def _load_zimage(self):
        """Load Z-Image model for 128-channel VAE and replicate to all GPUs."""
        print("  Loading Z-Image (128-channel from autoencoder_c32)...")
        
        # Use 128-channel params (patch_size=1, in_channels=128)
        # autoencoder_c32 has internal patchify, so output is already 128 channels
        c32_params = ZImageParams(
            all_patch_size=(1,),  # patch_size=1 since VAE already outputs 128ch
            all_f_patch_size=(1,),
            in_channels=128,  # 128 channels from autoencoder_c32 (has internal patchify)
            dim=3840,
            n_layers=30,
            n_refiner_layers=2,
            n_heads=30,
            n_kv_heads=30,
            norm_eps=1e-5,
            qk_norm=True,
            cap_feat_dim=2560,
            rope_theta=256,
            t_scale=1000.0,
            axes_dims=[32, 48, 48],
            axes_lens=[1536, 512, 512],
            adaln_embed_dim=256,
            use_x0=self.model_config.use_x0,
        )
        
        # Create base model with 32-channel config
        base_model = ZImage(c32_params)
        
        # Load original ZImage weights and extend for 32 channels
        original_state_dict = self._load_state_dict(self.model_config.z_image_path)
        if self.model_config.use_x0:
            original_state_dict["__x0__"] = torch.tensor([])
        
        # Use weight extension to map 16ch -> 32ch weights
        base_model = load_zimage_weights_to_c32(base_model, original_state_dict)
        
        # Replicate to all GPUs
        self.models = []
        for gpu_id in range(self.n_gpus):
            model_copy = copy.deepcopy(base_model).to(f'cuda:{gpu_id}').to(torch.bfloat16)
            self.models.append(model_copy)
        
        # Keep reference to first model for compatibility
        self.model = self.models[0]
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Z-Image (32ch VAE) loaded on {self.n_gpus} GPUs ({total_params:,} params each)")
    
    def _load_vae(self):
        """Load 32-channel VAE model and replicate to all GPUs."""
        print("  Loading 32-channel VAE...")
        
        # Load base VAE with 32-channel params
        ae_params = AutoEncoderParams()  # defaults to z_channels=32
        with torch.device("meta"):
            base_vae = AutoEncoder(ae_params)
        
        state_dict = self._load_state_dict(self.model_config.vae_path)
        base_vae.load_state_dict(state_dict, assign=True)
        
        # Replicate to all GPUs
        self.vaes = []
        for gpu_id in range(self.n_gpus):
            vae_copy = copy.deepcopy(base_vae).to(f'cuda:{gpu_id}').to(torch.bfloat16)
            vae_copy.eval()
            self.vaes.append(vae_copy)
        
        # Keep reference for compatibility
        self.ae = self.vaes[0]
        
        print(f"  32-channel VAE loaded on {self.n_gpus} GPUs")
    
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
        """Run forward/backward pass on a single GPU with its data chunk."""
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        vae = self.vaes[gpu_id]
        text_encoder = self.text_encoders[gpu_id]
        
        # Encode images on this GPU using its own VAE
        batch_size = images_chunk.shape[0]
        cache_mb = self.training_config.cache_minibatch
        latents_list = []
        
        for i in range(0, batch_size, cache_mb):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = images_chunk[i:i + cache_mb].to(device)
                latent_chunk = vae.encode(chunk)
                latents_list.append(latent_chunk)
        
        latents = torch.cat(latents_list, dim=0)
        
        # Prepare batch on this GPU
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = latents.to(torch.float32)
            latents, latent_shape = vae_flatten(latents, patch_size=1)
            n = latents.shape[0]
            
            # Sample timesteps using this GPU's sampler
            timesteps = self.timestep_samplers[gpu_id].sample(n, latents.device)
            timesteps_expanded = timesteps[:, None, None]
            
            # Generate and pair noise with optimal transport
            noise = torch.randn_like(latents)
            
            # Interpolate
            noisy_latents = latents * (1 - timesteps_expanded) + noise * timesteps_expanded
            
            # Compute target
            if self.model_config.use_x0:
                eps = 5e-2
                target = (noisy_latents - latents) / (timesteps_expanded + eps)
            else:
                target = noise - latents
        
        noisy_latents.requires_grad_(True)
        n, c, h, w = latent_shape
        
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
                image_pos_ids = prepare_latent_image_ids(token_lengths, h, w, patch_size=1).to(device)
                text_pos_ids = make_text_position_ids(
                    token_lengths, 
                    self.model_config.max_sequence_length
                ).to(device)
                img_mask = torch.ones(
                    (train_mb, noisy_latents.shape[1]),
                    device=device,
                    dtype=torch.bool
                )
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
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
        """Run inference on a single GPU using its own VAE and text encoder."""
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        vae = self.vaes[gpu_id]
        text_encoder = self.text_encoders[gpu_id]
        
        model.eval()
        
        width, height = config.image_dim
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate noise with different seed per GPU for variety
            # 32 channels = 128 latent depth after patchify
            noise = get_noise(
                len(prompts), height, width, device, torch.bfloat16,
                self.training_config.master_seed + gpu_id,
                latent_depth=128, spatial_compression=16
            )
            noise, shape = vae_flatten(noise, patch_size=1)
            n, c, h, w = shape
            
            timesteps = get_schedule(config.steps, noise.shape[1])
            
            # Encode prompts using this GPU's text encoder
            pos_embeds, pos_mask = text_encoder.encode(prompts)
            neg_embeds, neg_mask = text_encoder.encode([""] * len(prompts))
            
            # Position IDs
            pos_lengths = pos_mask.sum(1)
            neg_lengths = neg_mask.sum(1)
            offset = torch.maximum(pos_lengths, neg_lengths)
            
            image_pos_ids = prepare_latent_image_ids(offset, h, w, patch_size=1).to(device)
            pos_text_ids = make_text_position_ids(pos_lengths, config.max_sequence_length).to(device)
            neg_text_ids = make_text_position_ids(neg_lengths, config.max_sequence_length).to(device)
            
            # Denoise
            output = denoise_cfg(
                model, noise, image_pos_ids,
                pos_embeds, neg_embeds,
                pos_text_ids, neg_text_ids,
                pos_mask, neg_mask,
                timesteps, config.cfg,
                config.first_n_steps_wo_cfg,
            )
            
            # Decode using this GPU's VAE
            output_for_decode = vae_unflatten(output, shape, patch_size=1)
            images = vae.decode(output_for_decode)
        
        model.train()
        return images
    
    def _set_models_train_mode(self, gpu_id: int):
        """Set model back to train mode on a specific GPU."""
        self.models[gpu_id].train()
    
    @torch.no_grad()
    def run_inference(self, extra_prompts_per_gpu: List[str] = None) -> torch.Tensor:
        """
        Run inference to generate sample images across multiple GPUs.
        Each GPU generates: (inference prompts + 1 extra prompt from batch)
        
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
    
    parser = argparse.ArgumentParser(description="Train Z-Image model with 32-channel VAE (multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_example_dct.json",
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    trainer = ZImageVAETrainer(config_path=args.config, device=args.device)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
