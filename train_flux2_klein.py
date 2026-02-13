"""Multi-GPU Flux2 Klein Trainer (Class-based)

A trainer for Flux2 Klein 4B model using ThreadPoolExecutor for multi-GPU parallelism.
Uses NCCL for gradient all-reduce across GPUs.

Based on train_radiance.py structure but uses:
- Flux2 Klein model instead of Chroma
- Qwen3 text encoder with concatenated hidden states from layers [9, 18, 27]
- f16ch128 VAE (128 channel latents, 16x spatial compression)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import copy
import math
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nccl as nccl
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity, record_function
from einops import rearrange, repeat

from tqdm import tqdm
from safetensors.torch import safe_open

from transformers import AutoTokenizer, Qwen3ForCausalLM
from torch.optim import AdamW
# from ramtorch import AdamW

from src.dataloaders.dataloader import TextImageDataset
from src.models.flux2.model import Flux2, Klein4BParams, Klein9BParams
from src.models.flux2.sampling import get_schedule, denoise_cfg, denoise_cfg_with_reference
from src.models.flux2.autoencoder import AutoEncoder, AutoEncoderParams
from src.general_utils import load_file_multipart, load_safetensors
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
    first_n_steps_wo_cfg: int = 0
    prompts: List[str] = field(default_factory=lambda: ["a beautiful landscape painting"])
    image_dim: Tuple[int, int] = (512, 512)
    qwen_max_length: int = 512
    # Reference images for edit mode inference (list of lists, one per prompt)
    # Each inner list contains paths to reference images for that prompt
    # Example: [["ref1.png", "ref2.png"], ["ref3.png"]] for 2 prompts
    reference_image_paths: Optional[List[List[str]]] = None


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
    # Image editing settings
    num_reference_images: Optional[int] = None  # None = disabled, int = number of reference images
    reference_image_t_scale: int = 10  # T-coordinate scale for reference images (default: 10)


@dataclass
class ModelConfig:
    """Model paths and settings."""
    klein_path: str = ""
    vae_path: str = ""
    qwen_path: str = ""
    qwen_tokenizer_path: str = ""
    qwen_max_length: int = 512
    model_variant: str = "4b"  # "4b" or "9b"


# =============================================================================
# VAE Parameters for f16ch128
# =============================================================================

# f16ch128 VAE params - 16x spatial compression, 128 latent channels
flux2_ae_params = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=32,  # This becomes 128 after pixel shuffle (32 * 2 * 2 = 128)
)


# =============================================================================
# Qwen3 Output Layers Configuration
# =============================================================================

# Qwen3 layers to extract hidden states from for concatenation
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]


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
# Qwen Text Encoder Wrapper
# =============================================================================

class QwenTextEncoder:
    """Wrapper for Qwen3 text encoding with concatenated hidden states."""

    def __init__(
        self,
        tokenizer,
        qwen_model,
        max_length: int,
        device: torch.device,
        output_layers: List[int] = OUTPUT_LAYERS_QWEN3,
    ):
        self.tokenizer = tokenizer
        self.qwen_model = qwen_model
        self.max_length = max_length
        self.device = device
        self.output_layers = output_layers

    def _format_prompts(self, captions: List[str]) -> List[str]:
        """Format prompts with chat template."""
        formatted = []
        for caption in captions:
            messages = [{"role": "user", "content": caption}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            formatted.append(text)
        return formatted

    @torch.no_grad()
    def encode(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode captions through Qwen3 and concatenate hidden states from specified layers.

        Returns:
            embeddings: Concatenated hidden states [B, L, D*num_layers]
            attention_mask: Attention mask [B, L]
        """
        formatted = self._format_prompts(captions)

        inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Get Qwen hidden states from all layers
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.qwen_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # Extract and concatenate hidden states from specified layers
        # hidden_states is a tuple of (num_layers + 1) tensors, index 0 is embedding layer
        hidden_states = outputs.hidden_states
        selected_states = [hidden_states[layer_idx] for layer_idx in self.output_layers]

        # Concatenate along the feature dimension: [B, L, D] * 3 -> [B, L, 3*D]
        embeddings = torch.cat(selected_states, dim=-1)

        return embeddings, inputs.attention_mask

    @torch.no_grad()
    def encode_negative(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode empty/negative prompts."""
        empty_prompts = [""] * batch_size
        return self.encode(empty_prompts)


# =============================================================================
# Experiment Logger
# =============================================================================

class ExperimentLogger:
    """Handles experiment tracking with Aim or CSV fallback."""

    def __init__(self, config: TrainingConfig, hparams: Dict[str, Any], save_folder: str = "."):
        self.enabled = AIM_AVAILABLE and config.use_aim and config.aim_path
        self.run = None
        self.csv_path = None
        self.csv_file = None

        if self.enabled:
            self.run = Run(
                repo=config.aim_path,
                experiment=config.aim_experiment_name,
            )
            self.run["hparams"] = hparams
        else:
            # Fallback to CSV logging
            self.csv_path = os.path.join(save_folder, "training_log.csv")
            file_exists = os.path.exists(self.csv_path)
            self.csv_file = open(self.csv_path, "a", newline="")
            self.csv_writer = __import__("csv").writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow(["step", "name", "value"])
                self.csv_file.flush()

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value."""
        if self.run:
            self.run.track(value, name=name, step=step)
        elif self.csv_file:
            self.csv_writer.writerow([step, name, value])
            # Flush periodically to ensure data is written
            if step % 10 == 0:
                self.csv_file.flush()

    def log_image(self, name: str, image_path: str, caption: str, step: int):
        """Log an image."""
        if self.run and AIM_AVAILABLE:
            pil_img = PILImage.open(image_path)
            self.run.track(
                AimImage(pil_img, caption=caption),
                name=name,
                step=step,
            )
        # CSV doesn't support images, just skip

    def close(self):
        """Close the logger."""
        if self.run:
            self.run.close()
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None


# =============================================================================
# Flux2 Position ID Utilities
# =============================================================================

def prepare_img_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Prepare image position IDs for Flux2 (4D format).

    Flux2 uses 4D position IDs: [T, height, width, L]
    - T (dim0): Time/layer index (0 for output latent)
    - H (dim1): Height position
    - W (dim2): Width position
    - L (dim3): Always 0
    """
    # Latent spatial dimensions (16x compression)
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)

    # Create 4D position IDs: [T, h, w, L]
    # T=0 for output latent tokens
    img_ids = torch.zeros(h, w, 4, device=device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h, device=device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w, device=device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    return img_ids


def prepare_reference_img_ids(
    reference_latents: List[torch.Tensor],
    device: torch.device,
    scale: int = 10,
) -> torch.Tensor:
    """Prepare position IDs for reference images (4D format).

    Each reference image gets a unique T-coordinate offset to distinguish it
    from the output latent (T=0) and from other reference images.

    Args:
        reference_latents: List of latent tensors [(1, C, H, W), ...]
        device: Target device
        scale: T-coordinate scale (default 10). Reference image i gets T = scale + scale * i

    Returns:
        Combined position IDs tensor [1, total_seq_len, 4]
    """
    if not isinstance(reference_latents, list):
        raise ValueError(f"Expected reference_latents to be a list, got {type(reference_latents)}")

    # Create T-coordinate offsets for each reference image
    # Image 0 -> T=scale, Image 1 -> T=2*scale, etc.
    t_coords = [scale + scale * i for i in range(len(reference_latents))]

    all_ids = []
    for latent, t_val in zip(reference_latents, t_coords):
        # latent shape: (1, C, H, W)
        _, _, h, w = latent.shape

        # Create position IDs for this reference image
        ids = torch.zeros(h, w, 4, device=device)
        ids[..., 0] = t_val  # T-coordinate (unique per reference image)
        ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[:, None]  # H
        ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, :]  # W
        # ids[..., 3] = 0  # L (already zero)

        # Flatten spatial dims: (H, W, 4) -> (H*W, 4)
        ids = ids.view(-1, 4)
        all_ids.append(ids)

    # Concatenate all reference image IDs: (total_seq_len, 4)
    combined_ids = torch.cat(all_ids, dim=0)
    # Add batch dimension: (1, total_seq_len, 4)
    combined_ids = combined_ids.unsqueeze(0)

    return combined_ids


def prepare_txt_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Prepare text position IDs for Flux2 (4D format).

    Text tokens have all position dimensions set to 0.
    """
    txt_ids = torch.zeros(batch_size, seq_len, 4, device=device)
    return txt_ids


def pack_latents(latents: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pack latents from [B, C, H, W] to [B, H*W, C] for transformer.

    For Flux2 with 128 channels, we just flatten spatial dims.
    """
    b, c, h, w = latents.shape
    # Flatten: [B, C, H, W] -> [B, H*W, C]
    packed = rearrange(latents, "b c h w -> b (h w) c")
    return packed, (b, c, h, w)


def unpack_latents(latents: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    """Unpack latents from [B, H*W, C] back to [B, C, H, W]."""
    b, c, h, w = shape
    return rearrange(latents, "b (h w) c -> b c h w", h=h, w=w)


# =============================================================================
# Flux2 Klein Trainer
# =============================================================================

class Flux2KleinTrainer(BaseTrainer):
    """Trainer for Flux2 Klein model with multi-GPU support (DDP-like)."""

    def __init__(self, config_path: str, device: str = "cuda"):
        super().__init__(config_path, device)

        # Multi-GPU setup - one of each component per GPU
        self.n_gpus = torch.cuda.device_count()
        self.models = []  # One model per GPU
        self.text_encoders = []  # One text encoder per GPU
        self.vaes = []  # One VAE per GPU (for encoding images)
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
        self.logger = ExperimentLogger(
            self.training_config,
            self.config_data,
            save_folder=self.training_config.save_folder
        )

        # Setup thread pool for multi-GPU execution
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)

        print("Setup complete!")

    def _load_models(self):
        """Load all required models and replicate to all GPUs."""
        print("Loading models...")

        # Load VAE (replicated)
        self._load_vae()

        # Load Klein (replicated)
        self._load_klein()

        # Load Qwen text encoder (replicated)
        self._load_text_encoder()

        # Create timestep samplers (one per GPU)
        self._setup_timestep_samplers()

        print("All models loaded!")

    def _load_vae(self):
        """Load VAE and replicate to all GPUs."""
        print("  Loading VAE...")

        # Load base VAE
        with torch.device('meta'):
            base_ae = AutoEncoder(flux2_ae_params)

        if self.model_config.vae_path:
            print(f"    Loading VAE checkpoint: {self.model_config.vae_path}")
            state_dict = load_safetensors(self.model_config.vae_path)
            base_ae.load_state_dict(state_dict, assign=True)

        # Replicate to all GPUs
        self.vaes = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            ae_copy = copy.deepcopy(base_ae).to(device).to(torch.bfloat16)
            ae_copy.eval()
            for param in ae_copy.parameters():
                param.requires_grad = False
            self.vaes.append(ae_copy)

        # Keep reference for compatibility
        self.ae = self.vaes[0]

        print(f"  VAE loaded on {self.n_gpus} GPUs")

    def _load_klein(self):
        """Load Klein model and replicate to all GPUs."""
        print(f"  Loading Klein {self.model_config.model_variant}...")

        # Select params based on variant
        if self.model_config.model_variant == "4b":
            params = Klein4BParams()
        elif self.model_config.model_variant == "9b":
            params = Klein9BParams()
        else:
            raise ValueError(f"Unknown model variant: {self.model_config.model_variant}")

        params.use_gradient_checkpointing = True
        
        # Load checkpoint state dict first (if provided)
        checkpoint_state_dict = None
        if self.model_config.klein_path:
            print(f"    Loading checkpoint: {self.model_config.klein_path}")
            checkpoint_state_dict = load_safetensors(self.model_config.klein_path)

        # Create models directly on each GPU
        self.models = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            print(f"    Initializing model on {device}...")

            # Create model on meta device (no memory allocation)
            with torch.device('meta'):
                model = Flux2(params)

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
                            model_state_dict[model_key] = ckpt_tensor.to(device=device, dtype=torch.bfloat16)
                            loaded_keys.append(model_key)
                        else:
                            # Shape mismatch - random init
                            model_state_dict[model_key] = torch.empty(
                                model_tensor.shape, device=device, dtype=torch.bfloat16
                            )
                            if model_state_dict[model_key].dim() > 1:
                                nn.init.kaiming_uniform_(model_state_dict[model_key])
                            else:
                                nn.init.zeros_(model_state_dict[model_key])
                            shape_mismatch_keys.append(
                                f"{model_key}: model={list(model_tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                            )
                    else:
                        model_state_dict[model_key] = torch.empty(
                            model_tensor.shape, device=device, dtype=torch.bfloat16
                        )
                        if model_state_dict[model_key].dim() > 1:
                            nn.init.kaiming_uniform_(model_state_dict[model_key])
                        else:
                            nn.init.zeros_(model_state_dict[model_key])
                        new_keys.append(model_key)

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

                model.load_state_dict(model_state_dict, assign=True)

            self.models.append(model)

        # Keep reference to first model for compatibility
        self.model = self.models[0]

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Klein {self.model_config.model_variant} loaded on {self.n_gpus} GPUs ({total_params:,} params each)")

    def _load_text_encoder(self):
        """Load Qwen3 tokenizer and encoder, replicate to all GPUs."""
        print("  Loading Qwen3...")

        # Shared tokenizer (CPU-based)
        tokenizer_path = self.model_config.qwen_tokenizer_path or self.model_config.qwen_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load base Qwen model
        qwen_model = Qwen3ForCausalLM.from_pretrained(
            self.model_config.qwen_path,
            torch_dtype=torch.bfloat16,
        )
        qwen_model.eval()
        for param in qwen_model.parameters():
            param.requires_grad = False

        # Replicate to all GPUs
        self.text_encoders = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            encoder_copy = copy.deepcopy(qwen_model).to(device)
            encoder_copy.eval()

            text_encoder = QwenTextEncoder(
                self.tokenizer,
                encoder_copy,
                self.model_config.qwen_max_length,
                torch.device(device),
            )
            self.text_encoders.append(text_encoder)

        # Keep reference for compatibility
        self.text_encoder = self.text_encoders[0]

        print(f"  Qwen3 loaded on {self.n_gpus} GPUs")

    def _setup_timestep_samplers(self):
        """Create one timestep sampler per GPU."""
        print("  Setting up timestep samplers...")
        self.timestep_samplers = [TimestepSampler() for _ in range(self.n_gpus)]
        print(f"  Timestep samplers created for {self.n_gpus} GPUs")

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
                    if gpu_id == 0:
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
            num_reference_images=self.dataloader_config.num_reference_images,
        )

        # Log edit mode status
        if self.dataloader_config.num_reference_images is not None:
            print(f"Edit mode enabled: {self.dataloader_config.num_reference_images} reference images per sample")
            print(f"Reference image T-scale: {self.dataloader_config.reference_image_t_scale}")

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

    def _prepare_flow_matching_targets(
        self,
        latents: torch.Tensor,
        device: str,
        gpu_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Prepare flow matching targets for training.

        Args:
            latents: Encoded latents [B, 128, H/16, W/16]

        Returns:
            noisy_latents: Noised latents
            target: Target velocity (noise - x0)
            timesteps: Sampled timesteps
            img_ids: Image position IDs
            shape: Original latent shape
        """
        latents = latents.to(device).to(torch.float32)
        b, c, h, w = latents.shape

        # Prepare image position IDs (4D format for Flux2)
        img_ids = prepare_img_ids(b, h * 16, w * 16, device=torch.device(device))

        # Sample timesteps using this GPU's sampler
        timesteps = self.timestep_samplers[gpu_id].sample(b, latents.device)
        timesteps_expanded = timesteps[:, None, None, None]

        # Generate noise
        noise = torch.randn_like(latents)

        # Compute OT pairings for better training
        transport_cost, indices = cosine_optimal_transport(
            latents.reshape(b, -1), noise.reshape(b, -1)
        )
        noise = noise[indices[1].view(-1)]

        # Linear interpolation: x_t = (1-t) * x_0 + t * noise
        noisy_latents = latents * (1 - timesteps_expanded) + noise * timesteps_expanded

        # Flow matching target: v = noise - x_0
        target = noise - latents

        return noisy_latents, target, timesteps, img_ids, (b, c, h, w)

    def _prepare_reference_latents(
        self,
        reference_images: torch.Tensor,
        vae: nn.Module,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode reference images and prepare them for concatenation.

        Args:
            reference_images: Reference images [B, num_ref, 3, H, W]
            vae: VAE encoder
            device: Target device

        Returns:
            packed_ref_latents: Packed reference latents [B, total_ref_seq_len, C]
            ref_img_ids: Reference image position IDs [B, total_ref_seq_len, 4]
        """
        b, num_ref, c, h, w = reference_images.shape
        t_scale = self.dataloader_config.reference_image_t_scale

        all_packed_latents = []
        all_ref_ids = []

        for batch_idx in range(b):
            batch_ref_latents = []

            # Encode each reference image for this batch item
            for ref_idx in range(num_ref):
                ref_img = reference_images[batch_idx, ref_idx].unsqueeze(0).to(device)  # [1, 3, H, W]

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ref_latent = vae.encode(ref_img)  # [1, 128, H/16, W/16]

                batch_ref_latents.append(ref_latent)

            # Prepare position IDs for all reference images in this batch item
            ref_ids = prepare_reference_img_ids(batch_ref_latents, torch.device(device), scale=t_scale)
            # ref_ids: [1, total_ref_seq_len, 4]

            # Pack each reference latent and concatenate
            packed_refs = []
            for ref_latent in batch_ref_latents:
                packed, _ = pack_latents(ref_latent)  # [1, H*W, C]
                packed = packed.squeeze(0)  # [H*W, C]
                packed_refs.append(packed)

            # Concatenate all reference latents for this batch item
            packed_refs = torch.cat(packed_refs, dim=0)  # [total_ref_seq_len, C]

            all_packed_latents.append(packed_refs)
            all_ref_ids.append(ref_ids.squeeze(0))  # [total_ref_seq_len, 4]

        # Stack across batch dimension
        packed_ref_latents = torch.stack(all_packed_latents, dim=0)  # [B, total_ref_seq_len, C]
        ref_img_ids = torch.stack(all_ref_ids, dim=0)  # [B, total_ref_seq_len, 4]

        return packed_ref_latents, ref_img_ids

    def _forward_backward_on_gpu(
        self,
        gpu_id: int,
        images_chunk: torch.Tensor,
        captions_chunk: List[str],
        loss_weights_chunk: List[float],
        reference_images_chunk: Optional[torch.Tensor] = None,
    ) -> float:
        """Run forward/backward pass on a single GPU with its data chunk.

        Args:
            gpu_id: GPU index
            images_chunk: Target images [B, 3, H, W]
            captions_chunk: List of captions
            loss_weights_chunk: List of loss weights
            reference_images_chunk: Optional reference images [B, num_ref, 3, H, W]
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        vae = self.vaes[gpu_id]
        text_encoder = self.text_encoders[gpu_id]

        batch_size = images_chunk.shape[0]
        cache_mb = self.training_config.cache_minibatch

        # Encode target images to latents in chunks to save memory
        latents_list = []
        for i in range(0, batch_size, cache_mb):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = images_chunk[i:i + cache_mb].to(device)
                latent_chunk = vae.encode(chunk)
                latents_list.append(latent_chunk)

        latents = torch.cat(latents_list, dim=0)

        # Prepare flow matching targets
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noisy_latents, target, timesteps, img_ids, latent_shape = \
                self._prepare_flow_matching_targets(latents, device, gpu_id)

        # Pack latents for transformer: [B, C, H, W] -> [B, H*W, C]
        noisy_latents_packed, _ = pack_latents(noisy_latents)
        target_packed, _ = pack_latents(target)

        # Prepare reference image latents if provided (edit mode)
        ref_latents_packed = None
        ref_img_ids = None
        if reference_images_chunk is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_latents_packed, ref_img_ids = self._prepare_reference_latents(
                    reference_images_chunk, vae, device
                )

        noisy_latents_packed.requires_grad_(True)

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

            # Prepare text position IDs
            txt_ids = prepare_txt_ids(train_mb, text_embeds.shape[1], torch.device(device))

            # Prepare model input (concatenate with reference latents if in edit mode)
            mb_noisy_latents = noisy_latents_packed[start:end]
            mb_img_ids = img_ids[start:end]

            if ref_latents_packed is not None:
                # Concatenate reference latents along sequence dimension
                # [B, output_seq_len, C] + [B, ref_seq_len, C] -> [B, total_seq_len, C]
                mb_ref_latents = ref_latents_packed[start:end]
                mb_ref_ids = ref_img_ids[start:end]

                model_input = torch.cat([mb_noisy_latents, mb_ref_latents], dim=1)
                model_input_ids = torch.cat([mb_img_ids, mb_ref_ids], dim=1)
                output_seq_len = mb_noisy_latents.shape[1]
            else:
                model_input = mb_noisy_latents
                model_input_ids = mb_img_ids
                output_seq_len = None

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
                    x=model_input,
                    x_ids=model_input_ids,
                    timesteps=timesteps[start:end],
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,  # Klein models don't use guidance embedding
                )

                # Extract only output tokens (exclude reference tokens) if in edit mode
                if output_seq_len is not None:
                    pred = pred[:, :output_seq_len, :]

                # Compute loss (MSE between predicted velocity and target velocity)
                loss = ((pred - target_packed[start:end]) ** 2).mean(dim=(1, 2))

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
        """Execute forward/backward pass across all GPUs (without all-reduce).

        Supports both text-to-image and image editing modes.
        """
        # Handle both formats: with and without reference images
        if len(batch) == 5:
            images, captions, _, loss_weights, reference_images = batch
        else:
            images, captions, _, loss_weights = batch
            reference_images = None

        # Preprocess captions
        captions = [c if c else "" for c in captions]
        captions = [c.lower() if random.random() < 0.25 else c for c in captions]

        batch_size = images.shape[0]
        samples_per_gpu = batch_size // self.n_gpus

        # Split batch across GPUs and run forward/backward in parallel
        def gpu_forward_backward(gpu_id):
            start = gpu_id * samples_per_gpu
            end = start + samples_per_gpu

            ref_chunk = None
            if reference_images is not None:
                ref_chunk = reference_images[start:end]

            return self._forward_backward_on_gpu(
                gpu_id=gpu_id,
                images_chunk=images[start:end],
                captions_chunk=captions[start:end],
                loss_weights_chunk=loss_weights[start:end],
                reference_images_chunk=ref_chunk,
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
        reference_images: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run inference on a single GPU.

        Args:
            gpu_id: GPU index
            prompts: List of prompts
            config: Inference configuration
            reference_images: Optional list of reference image tensors [3, H, W] for edit mode
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        text_encoder = self.text_encoders[gpu_id]
        ae = self.vaes[gpu_id]

        model.eval()

        width, height = config.image_dim
        batch_size = len(prompts)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate initial noise [B, 128, H/16, W/16]
            generator = torch.Generator(device=device).manual_seed(
                self.training_config.master_seed + gpu_id
            )
            noise = torch.randn(
                batch_size,
                128,  # f16ch128 VAE has 128 latent channels
                math.ceil(height / 16),
                math.ceil(width / 16),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )

            # Pack latents for transformer: [B, C, H, W] -> [B, H*W, C]
            img, shape = pack_latents(noise)

            # Get denoising schedule
            image_seq_len = img.shape[1]
            timesteps = get_schedule(config.steps, image_seq_len)

            # Encode prompts
            qwen_embed, prompt_masks = text_encoder.encode(prompts)
            qwen_embed_neg, prompt_masks_neg = text_encoder.encode_negative(batch_size)

            # Prepare position IDs (4D format for Flux2)
            img_ids = prepare_img_ids(batch_size, height, width, device)
            txt_ids = prepare_txt_ids(batch_size, qwen_embed.shape[1], device)
            neg_txt_ids = prepare_txt_ids(batch_size, qwen_embed_neg.shape[1], device)

            # Prepare reference image latents if provided (edit mode inference)
            ref_latents = None
            ref_ids = None
            if reference_images is not None and len(reference_images) > 0:
                t_scale = self.dataloader_config.reference_image_t_scale
                ref_latent_list = []

                for ref_img in reference_images:
                    ref_img = ref_img.unsqueeze(0).to(device)  # [1, 3, H, W]
                    ref_latent = ae.encode(ref_img)  # [1, 128, H/16, W/16]
                    ref_latent_list.append(ref_latent)

                # Prepare reference position IDs
                ref_ids = prepare_reference_img_ids(ref_latent_list, torch.device(device), scale=t_scale)
                ref_ids = ref_ids.expand(batch_size, -1, -1)  # [B, ref_seq_len, 4]

                # Pack and concatenate reference latents
                packed_refs = []
                for ref_latent in ref_latent_list:
                    packed, _ = pack_latents(ref_latent)  # [1, H*W, C]
                    packed = packed.squeeze(0)  # [H*W, C]
                    packed_refs.append(packed)
                ref_latents = torch.cat(packed_refs, dim=0)  # [total_ref_seq_len, C]
                ref_latents = ref_latents.unsqueeze(0).expand(batch_size, -1, -1)  # [B, ref_seq_len, C]

            # Run denoising with CFG (with optional reference images)
            output_latent = denoise_cfg_with_reference(
                model,
                img,
                img_ids,
                qwen_embed,
                qwen_embed_neg,
                txt_ids,
                neg_txt_ids,
                timesteps,
                cfg=config.cfg,
                first_n_steps_without_cfg=config.first_n_steps_wo_cfg,
                ref_latents=ref_latents,
                ref_ids=ref_ids,
            )

            # Unpack and decode
            output_latent = unpack_latents(output_latent, shape)
            output_image = ae.decode(output_latent)

        model.train()
        return output_image

    def _set_models_train_mode(self, gpu_id: int):
        """Set model back to train mode on a specific GPU."""
        self.models[gpu_id].train()

    def _load_reference_images_for_inference(
        self,
        ref_paths: List[str],
        target_height: int,
        target_width: int,
    ) -> List[torch.Tensor]:
        """Load and preprocess reference images for inference.

        Args:
            ref_paths: List of paths to reference images
            target_height: Target height for resizing
            target_width: Target width for resizing

        Returns:
            List of preprocessed image tensors [3, H, W]
        """
        from PIL import Image as PILImageLoader
        import torchvision.transforms.v2 as v2

        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])

        ref_tensors = []
        for path in ref_paths:
            try:
                img = PILImageLoader.open(path).convert("RGB")
                # Resize to target dimensions
                img = img.resize((target_width, target_height), PILImageLoader.LANCZOS)
                tensor = transform(img)
                ref_tensors.append(tensor)
            except Exception as e:
                print(f"Warning: Failed to load reference image {path}: {e}")

        return ref_tensors

    @torch.no_grad()
    def run_inference(
        self,
        extra_prompts_per_gpu: List[str] = None,
        extra_reference_images_per_gpu: List[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Run inference to generate sample images across multiple GPUs.
        Each GPU generates: (inference prompts + 1 extra prompt from batch)

        Args:
            extra_prompts_per_gpu: One extra prompt per GPU (e.g., from training batch)
            extra_reference_images_per_gpu: Reference images for extra prompts (edit mode)
        """
        config = self.inference_config
        base_prompts = list(config.prompts)
        width, height = config.image_dim

        # Load reference images from config if specified (for edit mode)
        base_reference_images = []
        if config.reference_image_paths is not None:
            for prompt_idx, ref_paths in enumerate(config.reference_image_paths):
                if ref_paths:
                    ref_tensors = self._load_reference_images_for_inference(
                        ref_paths, height, width
                    )
                    base_reference_images.append(ref_tensors)
                else:
                    base_reference_images.append([])

        def inference_on_gpu(gpu_id):
            prompts = list(base_prompts)
            reference_images = list(base_reference_images) if base_reference_images else None

            # Add extra prompt and its reference images if provided
            if extra_prompts_per_gpu and gpu_id < len(extra_prompts_per_gpu):
                prompts.append(extra_prompts_per_gpu[gpu_id])
                # Add reference images for extra prompt if provided
                if extra_reference_images_per_gpu and gpu_id < len(extra_reference_images_per_gpu):
                    if reference_images is None:
                        reference_images = [[] for _ in range(len(base_prompts))]
                    reference_images.append(extra_reference_images_per_gpu[gpu_id])
                elif reference_images is not None:
                    reference_images.append([])  # No reference for extra prompt

            # For edit mode, we need to run inference one prompt at a time
            # since each prompt may have different reference images
            if reference_images is not None and any(len(refs) > 0 for refs in reference_images):
                # Edit mode: run each prompt separately with its reference images
                all_outputs = []
                for prompt_idx, prompt in enumerate(prompts):
                    refs = reference_images[prompt_idx] if prompt_idx < len(reference_images) else []
                    output = self._inference_on_gpu(
                        gpu_id,
                        [prompt],
                        config,
                        reference_images=refs if refs else None,
                    )
                    all_outputs.append(output)
                return torch.cat(all_outputs, dim=0)
            else:
                # Text-to-image mode: batch all prompts together
                return self._inference_on_gpu(gpu_id, prompts, config)

        # Run inference on all GPUs in parallel
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
                        self.config_data["model"]["klein_path"] = ckpt_path
                        self.config_data["dataloader"]["offset"] = self.dataloader_config.offset + step
                        self._save_config(f"{self.training_config.save_folder}/training_config.json")

                    # Inference (distributed across GPUs)
                    if (step + 1) % self.inference_config.inference_every == 0:
                        print("\nRunning inference...")
                        # Get one prompt per GPU from current batch
                        captions = batch_data[1]
                        extra_prompts_per_gpu = []
                        extra_reference_images_per_gpu = []

                        # Check if we have reference images in the batch (edit mode)
                        has_batch_refs = len(batch_data) == 5 and batch_data[4] is not None

                        if captions:
                            samples_per_gpu = len(captions) // self.n_gpus
                            for gpu_id in range(self.n_gpus):
                                idx = gpu_id * samples_per_gpu
                                if idx < len(captions) and captions[idx]:
                                    extra_prompts_per_gpu.append(captions[idx])
                                    # Get reference images for this sample if available
                                    if has_batch_refs:
                                        # batch_data[4] is [B, num_ref, 3, H, W]
                                        ref_imgs = batch_data[4][idx]  # [num_ref, 3, H, W]
                                        # Convert to list of tensors
                                        ref_list = [ref_imgs[i] for i in range(ref_imgs.shape[0])]
                                        extra_reference_images_per_gpu.append(ref_list)

                        images = self.run_inference(
                            extra_prompts_per_gpu=extra_prompts_per_gpu,
                            extra_reference_images_per_gpu=extra_reference_images_per_gpu if extra_reference_images_per_gpu else None,
                        )

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

    parser = argparse.ArgumentParser(description="Train Flux2 Klein model (multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="train_flux2_klein.json",
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)"
    )

    args = parser.parse_args()

    trainer = Flux2KleinTrainer(config_path=args.config, device=args.device)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
