"""Multi-GPU Flux2 Klein Trainer (Class-based)

A trainer for Flux2 Klein 4B model using ThreadPoolExecutor for multi-GPU parallelism.
Uses NCCL for gradient all-reduce across GPUs.

Based on train_radiance.py structure but uses:
- Flux2 Klein model instead of Chroma
- Qwen3 text encoder with concatenated hidden states from layers [9, 18, 27]
- f16ch128 VAE (128 channel latents, 16x spatial compression)
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
from safetensors.torch import safe_open, save_file as save_safetensors

from transformers import AutoTokenizer, Qwen3ForCausalLM
from torch.optim import AdamW, RMSprop
# from ramtorch import AdamW
from torch.autograd import grad as torch_grad

from src.dataloaders.dataloader import TextImageDataset
from src.models.flux2.model import Flux2, Klein4BParams, Klein9BParams
from src.models.flux2.sampling import get_schedule, denoise_cfg
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
class GANConfig:
    """GAN training configuration for adaptive discriminator."""
    enabled: bool = True
    discriminator_lr: float = 1e-5
    gan_loss_weight: float = 0.5  # Weight for GAN loss relative to flow matching loss
    
    # Noise mode:
    #   "adaptive" - learned noise level based on D loss (fixed noise at that level)
    #   "adaptive_uniform" - random uniform noise from 0 to adaptive level (gives D more leeway)
    #   "timestep" - use FM timestep as noise level
    #   "gan_only" - pure GAN training (no FM loss), one-step generation with adaptive noise
    #   "gan_only_uniform" - pure GAN training (no FM loss), one-step with uniform noise [0, adaptive]
    #   "gan_only_buffer" - pure GAN training, clean samples but adaptive replay buffer prob
    noise_mode: str = "adaptive"  # "adaptive", "adaptive_uniform", "timestep", "gan_only", "gan_only_uniform", or "gan_only_buffer"
    
    # Replay buffer settings
    replay_buffer_size: int = 50  # Number of samples to store
    replay_buffer_prob: float = 0.8  # Probability of using buffer vs new sample
    
    # Adaptive noise regularization (only used when noise_mode="adaptive")
    target_d_loss: float = 0.693147  # ln(2) - equilibrium point (noise_lerp=0 when D loss >= this)
    target_d_loss_floor: float = 0.0  # D loss at which noise_lerp hits 1.0 (max regularization)
                                       # Set > 0 to trigger max regularization before D loss hits 0
                                       # e.g., 0.1 means 100% noise/buffer when D loss <= 0.1
    noise_ema_decay: float = 0.95  # EMA decay for noise level
    initial_noise_level: float = 1.0  # Start with full noise (discriminator sees pure noise)
    
    # Gradient fusion settings
    grad_norm_eps: float = 1e-8  # Epsilon for gradient normalization
    
    # Discrete timestep sampling (for few-step distillation)
    # If non-empty, samples uniformly from these discrete timesteps instead of continuous
    # Example: [1.0, 0.75, 0.5, 0.25] for 4-step, [1.0, 0.5] for 2-step
    # Empty list [] means use continuous sampling (default behavior)
    discrete_timesteps: List[float] = field(default_factory=list)
    
    # Discrete noise bins for adaptive noise regularization
    # If > 0, rounds noise_lerp_val to nearest bin (e.g., 10 bins = [0.0, 0.1, 0.2, ..., 1.0])
    # If 0, uses continuous noise level (default behavior)
    noise_discrete_bins: int = 0
    
    # Discriminator checkpoint path (for resuming training)
    # If provided, loads discriminator weights from this safetensors file
    # If empty, initializes discriminator from generator weights
    discriminator_path: str = ""


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
# Discriminator Architecture (extends Flux2 Klein)
# =============================================================================

class Flux2Discriminator(nn.Module):
    """
    Discriminator based on Flux2 Klein architecture.
    
    Extends the generator by:
    1. Adding a learnable [CLS] token for classification
    2. Adding noise level conditioning (for adaptive noise regularization)
    3. Replacing the final layer with a classifier head
    
    Can be initialized from generator weights for better starting point.
    """
    def __init__(self, params):
        super().__init__()
        
        # Store params for reference
        self.params = params
        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        # Import the building blocks from flux2 model
        from src.models.flux2.model import (
            EmbedND, MLPEmbedder, DoubleStreamBlock, SingleStreamBlock, 
            Modulation, timestep_embedding
        )
        
        self.timestep_embedding = timestep_embedding
        
        # Core architecture (same as Flux2)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
        
        # Note: We reuse time_in for noise level conditioning since we don't use timesteps
        # (discriminator sees clean x0, not noisy x_t)
        
        # Learnable [CLS] token for classification output
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, 1, pe_dim//2, 2, 2) * 0.02)  # Position embedding for CLS
        
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
            )
            for _ in range(params.depth)
        ])
        
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
            )
            for _ in range(params.depth_single_blocks)
        ])
        
        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)
        
        # Classifier head instead of final layer (outputs single logit per image token)
        self.classifier_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.classifier_head = nn.Linear(self.hidden_size, 1, bias=True)
        
        # Initialize classifier head to output ~0 (balanced start)
        nn.init.zeros_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)
        
        self.use_gradient_checkpointing = params.use_gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,           # [B, seq_len, in_channels] - packed latent (x0 prediction)
        x_ids: torch.Tensor,       # [B, seq_len, 4] - position IDs
        noise_level: torch.Tensor, # [B] or scalar - noise regularization level
        ctx: torch.Tensor,         # [B, text_len, context_dim] - text embeddings
        ctx_ids: torch.Tensor,     # [B, text_len, 4] - text position IDs
    ) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Note: We only condition on noise_level (the regularization noise), not the
        flow matching timestep, since the discriminator sees "clean" x0 predictions.
        
        Returns:
            scores: [B, seq_len, 1] - real/fake scores per image token
        """
        batch_size = x.shape[0]
        num_txt_tokens = ctx.shape[1]
        
        # Noise level conditioning (reuses time_in MLP)
        if noise_level.dim() == 0:
            noise_level = noise_level.expand(batch_size)
        noise_emb = self.timestep_embedding(noise_level, 256)
        vec = self.time_in(noise_emb)
        
        # Compute modulations
        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)
        
        # Project inputs
        img = self.img_in(x)
        txt = self.txt_in(ctx)
        
        # Position embeddings
        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)
        
        # Double stream blocks (image and text separate)
        for block in self.double_blocks:
            if self.use_gradient_checkpointing and self.training:
                img, txt = torch.utils.checkpoint.checkpoint(
                    block,
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                    use_reentrant=False,
                )
            else:
                img, txt = block(
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                )
        
        # Concatenate text and image for single stream
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        img = torch.cat((cls_tokens, txt, img), dim=1)
        
        # Position embeddings: [CLS_pe, text_pe, img_pe]
        cls_pe = self.cls_pos_embed.expand(batch_size, -1, -1, -1, -1, -1)
        pe = torch.cat((cls_pe, pe_ctx, pe_x), dim=2)
        
        # Single stream blocks
        for block in self.single_blocks:
            if self.use_gradient_checkpointing and self.training:
                img = torch.utils.checkpoint.checkpoint(
                    block,
                    img,
                    pe,
                    single_block_mod,
                    use_reentrant=False,
                )
            else:
                img = block(
                    img,
                    pe,
                    single_block_mod,
                )
        
        # Extract image tokens only (skip CLS and text)
        img_start = 1 + num_txt_tokens
        img_tokens = img[:, img_start:, :]
        
        # Classifier head
        img_tokens = self.classifier_norm(img_tokens)
        scores = self.classifier_head(img_tokens)  # [B, img_seq, 1]
        
        return scores

    @classmethod
    def from_flux2(cls, flux2_model: nn.Module, params):
        """
        Create discriminator from a pretrained Flux2 generator.
        
        Copies all shared weights and initializes discriminator-specific layers.
        """
        discriminator = cls(params)
        
        # Copy shared weights from generator
        generator_state = flux2_model.state_dict()
        discriminator_state = discriminator.state_dict()
        
        # Keys to copy (shared between generator and discriminator)
        shared_prefixes = [
            'pe_embedder', 'img_in', 'time_in', 'txt_in',
            'double_blocks', 'single_blocks',
            'double_stream_modulation_img', 'double_stream_modulation_txt',
            'single_stream_modulation',
        ]
        
        copied_keys = []
        for key in generator_state:
            if any(key.startswith(prefix) for prefix in shared_prefixes):
                if key in discriminator_state and generator_state[key].shape == discriminator_state[key].shape:
                    discriminator_state[key] = generator_state[key].clone()
                    copied_keys.append(key)
        
        discriminator.load_state_dict(discriminator_state)
        print(f"  Copied {len(copied_keys)} keys from generator to discriminator")
        
        return discriminator


# =============================================================================
# Replay Buffer (CPU Storage, Per-Shape Buckets)
# =============================================================================

class ReplayBuffer:
    """
    Replay buffer for GAN training with CPU storage.
    
    Stores predicted x0 latents along with their text embeddings on CPU.
    Uses per-shape buckets to handle dynamic tensor sizes from bucketed training.
    Retrieves to GPU when needed - transfers are fast enough.
    """
    def __init__(
        self,
        max_size_per_bucket: int = 50,
        prob: float = 0.5,
    ):
        assert max_size_per_bucket > 0, "Replay buffer size must be positive"
        self.max_size_per_bucket = max_size_per_bucket
        self.prob = prob
        # Dict mapping (latent_seq_len, latent_dim) -> list of (x0_latent, text_embed) tuples
        self.buckets: Dict[Tuple[int, int], List] = {}

    def _get_shape_key(self, x0_latent: torch.Tensor) -> Tuple[int, int]:
        """Get bucket key from latent shape: (seq_len, dim)."""
        # x0_latent shape: [1, seq_len, dim]
        return (x0_latent.shape[1], x0_latent.shape[2])

    def push_and_pop(
        self,
        x0_latents: torch.Tensor,  # [B, seq, dim]
        text_embeds: torch.Tensor,  # [B, text_seq, text_dim]
        prob_override: float = None,  # Override default prob (for adaptive buffer)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Push new samples and pop samples for discriminator training.
        
        Returns a mix of new samples and buffered samples from the same shape bucket.
        All samples in a batch have the same shape, so we use one bucket per batch.
        
        Args:
            x0_latents: Latent samples [B, seq, dim]
            text_embeds: Text embeddings [B, text_seq, text_dim]
            prob_override: If provided, use this probability instead of self.prob
                          Higher value = more likely to use NEW samples
                          Lower value = more likely to use BUFFERED (stale) samples
        """
        device = x0_latents.device
        batch_size = x0_latents.shape[0]
        use_prob = prob_override if prob_override is not None else self.prob
        
        # Get bucket for this shape
        shape_key = (x0_latents.shape[1], x0_latents.shape[2])
        if shape_key not in self.buckets:
            self.buckets[shape_key] = []
        bucket = self.buckets[shape_key]
        
        to_return_latents = []
        to_return_texts = []
        
        for i in range(batch_size):
            latent = x0_latents[i:i+1]  # Keep batch dim
            text = text_embeds[i:i+1]
            
            if len(bucket) < self.max_size_per_bucket:
                # Buffer not full, add sample and return it
                bucket.append((latent.cpu().clone(), text.cpu().clone()))
                to_return_latents.append(latent)
                to_return_texts.append(text)
            else:
                # Buffer full, decide whether to use buffer or new sample
                if torch.rand(1).item() > use_prob:
                    # Use sample from buffer (same shape guaranteed)
                    idx = torch.randint(0, len(bucket), (1,)).item()
                    old_latent, old_text = bucket[idx]
                    to_return_latents.append(old_latent.to(device, non_blocking=True))
                    to_return_texts.append(old_text.to(device, non_blocking=True))
                    # Replace with new sample
                    bucket[idx] = (latent.cpu().clone(), text.cpu().clone())
                else:
                    # Use new sample directly
                    to_return_latents.append(latent)
                    to_return_texts.append(text)
        
        print (f"ReplayBuffer: shape {shape_key}, bucket size {len(bucket)}, prob {use_prob:.3f}, returned {len(to_return_latents)} samples")
        return torch.cat(to_return_latents, dim=0), torch.cat(to_return_texts, dim=0)

    def __len__(self):
        """Total samples across all buckets."""
        return sum(len(bucket) for bucket in self.buckets.values())
    
    def num_buckets(self) -> int:
        """Number of shape buckets."""
        return len(self.buckets)
    
    def bucket_sizes(self) -> Dict[Tuple[int, int], int]:
        """Get size of each bucket."""
        return {k: len(v) for k, v in self.buckets.items()}


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
    """Handles timestep sampling with custom distribution or discrete steps."""

    def __init__(
        self, 
        num_points: int = 1000, 
        device: torch.device = None,
        discrete_timesteps: List[float] = None,
    ):
        self.num_points = num_points
        self.device = device
        self._x = None
        self._probabilities = None
        self._cdf = None
        
        # Discrete timestep mode
        self.discrete_timesteps = discrete_timesteps
        self._discrete_tensor = None

    def _build_distribution(self, device: torch.device):
        """Build the timestep distribution (lazy initialization)."""
        if self._x is None or self._x.device != device:
            self._x = torch.linspace(0, 1, self.num_points, device=device)
            # Custom distribution favoring middle timesteps
            self._probabilities = -7.7 * ((self._x - 0.5) ** 2) + 2
            self._probabilities = self._probabilities.clamp(min=0)
            self._probabilities /= self._probabilities.sum()
            self._cdf = torch.cumsum(self._probabilities, dim=0)
    
    def _build_discrete(self, device: torch.device):
        """Build discrete timestep tensor (lazy initialization)."""
        if self._discrete_tensor is None or self._discrete_tensor.device != device:
            self._discrete_tensor = torch.tensor(self.discrete_timesteps, device=device)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from the distribution or discrete set."""
        if self.discrete_timesteps:
            # Discrete mode: sample uniformly from the discrete timesteps
            self._build_discrete(device)
            indices = torch.randint(0, len(self.discrete_timesteps), (num_samples,), device=device)
            return self._discrete_tensor[indices]
        else:
            # Continuous mode: sample from custom distribution
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

    Flux2 uses 4D position IDs: [dim0, height, width, dim3]
    Based on original Flux but extended to 4D.
    """
    # Latent spatial dimensions (16x compression)
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)

    # Create 4D position IDs: [dim0, h, w, dim3]
    # dim0 and dim3 are set to 0 for image tokens
    img_ids = torch.zeros(h, w, 4, device=device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h, device=device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w, device=device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    return img_ids


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
    """Trainer for Flux2 Klein model with multi-GPU support (DDP-like) and GAN loss."""

    def __init__(self, config_path: str, device: str = "cuda"):
        super().__init__(config_path, device)

        # Multi-GPU setup - one of each component per GPU
        self.n_gpus = torch.cuda.device_count()
        self.models = []  # One model per GPU (generator)
        self.discriminators = []  # One discriminator per GPU
        self.text_encoders = []  # One text encoder per GPU
        self.vaes = []  # One VAE per GPU (for encoding images)
        self.executor = None

        # Shared tokenizer (CPU-based, thread-safe)
        self.tokenizer = None
        self.timestep_samplers = []  # One per GPU
        
        # GAN training state (shared across GPUs)
        self.replay_buffers = []  # One per GPU
        self.noise_lerp_val = 1.0  # Adaptive noise level for discriminator
        self.prev_d_loss_metric = 0.693147  # Track D loss for adaptive noise

    def _parse_configs(self):
        """Parse configuration into dataclasses."""
        self.training_config = TrainingConfig(**self.config_data.get("training", {}))
        self.inference_config = InferenceConfig(**self.config_data.get("inference", {}))
        self.dataloader_config = DataloaderConfig(**self.config_data.get("dataloader", {}))
        self.model_config = ModelConfig(**self.config_data.get("model", {}))
        self.gan_config = GANConfig(**self.config_data.get("gan", {}))

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
        
        # Load discriminator if GAN is enabled
        if self.gan_config.enabled:
            self._load_discriminator()
            self._setup_replay_buffers()

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
        
        discrete_ts = self.gan_config.discrete_timesteps if self.gan_config.discrete_timesteps else None
        
        self.timestep_samplers = [
            TimestepSampler(discrete_timesteps=discrete_ts) 
            for _ in range(self.n_gpus)
        ]
        
        if discrete_ts:
            print(f"  Timestep samplers created for {self.n_gpus} GPUs (discrete: {discrete_ts})")
        else:
            print(f"  Timestep samplers created for {self.n_gpus} GPUs (continuous)")

    def _load_discriminator(self):
        """Load discriminator from checkpoint or initialize from generator weights."""
        print("  Loading Discriminator (from Klein architecture)...")
        
        # Select params based on variant (same as generator)
        if self.model_config.model_variant == "4b":
            params = Klein4BParams()
        elif self.model_config.model_variant == "9b":
            params = Klein9BParams()
        else:
            raise ValueError(f"Unknown model variant: {self.model_config.model_variant}")
        
        params.use_gradient_checkpointing = True
        
        # Load checkpoint state dict if provided
        checkpoint_state_dict = None
        if self.gan_config.discriminator_path:
            print(f"    Loading discriminator checkpoint: {self.gan_config.discriminator_path}")
            checkpoint_state_dict = load_safetensors(self.gan_config.discriminator_path)
        
        self.discriminators = []
        for gpu_id in range(self.n_gpus):
            device = f'cuda:{gpu_id}'
            
            # Create discriminator from generator weights (as base)
            discriminator = Flux2Discriminator.from_flux2(
                self.models[gpu_id], 
                params
            )
            
            if checkpoint_state_dict is not None:
                # Load from checkpoint
                d_state_dict = discriminator.state_dict()
                loaded_keys = []
                shape_mismatch_keys = []
                missing_keys = []
                
                for key, tensor in d_state_dict.items():
                    if key in checkpoint_state_dict:
                        ckpt_tensor = checkpoint_state_dict[key]
                        if tensor.shape == ckpt_tensor.shape:
                            d_state_dict[key] = ckpt_tensor.to(device=device, dtype=torch.bfloat16)
                            loaded_keys.append(key)
                        else:
                            # Shape mismatch - keep the from_flux2 initialized weights
                            d_state_dict[key] = tensor.to(device=device, dtype=torch.bfloat16)
                            shape_mismatch_keys.append(
                                f"{key}: model={list(tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                            )
                    else:
                        # Key not in checkpoint - keep the from_flux2 initialized weights
                        d_state_dict[key] = tensor.to(device=device, dtype=torch.bfloat16)
                        missing_keys.append(key)
                
                discriminator.load_state_dict(d_state_dict, assign=True)
                
                if gpu_id == 0:
                    print(f"    Loaded: {len(loaded_keys)} keys from checkpoint")
                    if shape_mismatch_keys:
                        print(f"    Shape mismatch (kept init): {len(shape_mismatch_keys)} keys")
                        for key in shape_mismatch_keys[:3]:
                            print(f"      - {key}")
                        if len(shape_mismatch_keys) > 3:
                            print(f"      ... and {len(shape_mismatch_keys) - 3} more")
                    if missing_keys:
                        print(f"    Missing in checkpoint (kept init): {len(missing_keys)} keys")
            else:
                # No checkpoint - just move to device
                discriminator = discriminator.to(device).to(torch.bfloat16)
            
            self.discriminators.append(discriminator)
        
        # Keep reference for compatibility
        self.discriminator = self.discriminators[0]
        
        total_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"  Discriminator loaded on {self.n_gpus} GPUs ({total_params:,} params each)")

    def _setup_replay_buffers(self):
        """Create one replay buffer per GPU."""
        print("  Setting up replay buffers...")
        
        gan_cfg = self.gan_config
        self.replay_buffers = [
            ReplayBuffer(
                max_size_per_bucket=gan_cfg.replay_buffer_size,
                prob=gan_cfg.replay_buffer_prob,
            )
            for _ in range(self.n_gpus)
        ]
        
        # Initialize adaptive noise state
        # Try to load from discriminator state file if resuming
        state_loaded = False
        if gan_cfg.discriminator_path:
            state_path = gan_cfg.discriminator_path.replace('.safetensors', '_state.json')
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        training_state = json.load(f)
                    self.noise_lerp_val = training_state.get('noise_lerp_val', gan_cfg.initial_noise_level)
                    self.prev_d_loss_metric = training_state.get('prev_d_loss_metric', gan_cfg.target_d_loss)
                    state_loaded = True
                    print(f"    Loaded training state: noise_lerp_val={self.noise_lerp_val:.4f}, prev_d_loss={self.prev_d_loss_metric:.4f}")
                except Exception as e:
                    print(f"    Warning: Could not load training state from {state_path}: {e}")
        
        if not state_loaded:
            self.noise_lerp_val = gan_cfg.initial_noise_level
            self.prev_d_loss_metric = gan_cfg.target_d_loss
        
        print(f"  Replay buffers created for {self.n_gpus} GPUs")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler for generator and discriminator on each GPU."""
        keywords = self.training_config.trained_layer_keywords

        # Generator optimizers
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
                print(f"Generator: Training {len(trained_params)} param groups, {frozen_count} frozen (per GPU)")

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
        
        # Discriminator optimizers (if GAN enabled)
        if self.gan_config.enabled and self.discriminators:
            self.d_optimizers = []
            
            for gpu_id, discriminator in enumerate(self.discriminators):
                # All discriminator params are trainable
                d_params = list(discriminator.parameters())
                
                if gpu_id == 0:
                    print(f"Discriminator: Training {len(d_params)} param groups (per GPU)")
                
                # Use RMSprop for discriminator (more stable for GAN training)
                d_optimizer = RMSprop(
                    d_params,
                    lr=self.gan_config.discriminator_lr,
                )
                
                self.d_optimizers.append(d_optimizer)
            
            # Keep reference
            self.d_optimizer = self.d_optimizers[0]

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
        """All-reduce gradients across all GPU generator models using NCCL."""
        # Get list of trainable parameters from each model
        param_lists = [
            [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            for model in self.models
        ]

        # All-reduce each parameter's gradient
        if param_lists and param_lists[0]:
            n_params = len(param_lists[0])
            for param_idx in range(n_params):
                grads = [param_lists[gpu_id][param_idx].grad for gpu_id in range(self.n_gpus)]
                nccl.all_reduce(grads, op=nccl.SUM)

    def _all_reduce_discriminator_gradients(self):
        """All-reduce gradients across all GPU discriminator models using NCCL."""
        if not self.gan_config.enabled or not self.discriminators:
            return
            
        param_lists = [
            [p for p in d.parameters() if p.requires_grad and p.grad is not None]
            for d in self.discriminators
        ]

        if param_lists and param_lists[0]:
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

    def _encode_batch_on_gpu(
        self,
        gpu_id: int,
        images_chunk: torch.Tensor,
        captions_chunk: List[str],
        loss_weights_chunk: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images and text on a single GPU. Shared preprocessing for D and G training.
        
        Returns dict with all tensors needed for training.
        """
        device = f'cuda:{gpu_id}'
        vae = self.vaes[gpu_id]
        text_encoder = self.text_encoders[gpu_id]

        batch_size = images_chunk.shape[0]
        cache_mb = self.training_config.cache_minibatch

        # Encode images to latents in chunks to save memory
        latents_list = []
        for i in range(0, batch_size, cache_mb):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = images_chunk[i:i + cache_mb].to(device)
                latent_chunk = vae.encode(chunk)
                latents_list.append(latent_chunk)

        latents = torch.cat(latents_list, dim=0)  # Real x0 latents

        # Prepare flow matching targets
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noisy_latents, target, timesteps, img_ids, latent_shape = \
                self._prepare_flow_matching_targets(latents, device, gpu_id)

        # Pack latents for transformer: [B, C, H, W] -> [B, H*W, C]
        noisy_latents_packed, _ = pack_latents(noisy_latents)
        target_packed, _ = pack_latents(target)
        real_x0_packed, _ = pack_latents(latents)

        loss_weights = torch.tensor(loss_weights_chunk, device=device)
        
        # Encode all text at once
        train_mb = self.training_config.train_minibatch
        num_minibatches = batch_size // train_mb
        
        text_embeds_list = []
        txt_ids_list = []
        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_embeds, _ = text_encoder.encode(captions_chunk[start:end])
            txt_ids = prepare_txt_ids(train_mb, text_embeds.shape[1], torch.device(device))
            text_embeds_list.append(text_embeds)
            txt_ids_list.append(txt_ids)

        return {
            'noisy_latents_packed': noisy_latents_packed,
            'target_packed': target_packed,
            'real_x0_packed': real_x0_packed,
            'timesteps': timesteps,
            'img_ids': img_ids,
            'loss_weights': loss_weights,
            'text_embeds_list': text_embeds_list,
            'txt_ids_list': txt_ids_list,
            'num_minibatches': num_minibatches,
        }

    def _train_discriminator_on_gpu(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator on a single GPU.
        
        Phase 1: G forward (no grad) → D forward/backward
        Following imagenet_gan.py pattern.
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]
        noise_level = torch.tensor(self.noise_lerp_val, device=device)

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]

            # G forward with no grad to get fake x0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                # Convert velocity to x0: x0 = x_t - t * v
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb - t_expanded * pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach()
                )
                
                # Add noise regularization
                real_noise = torch.randn_like(real_x0_mb)
                fake_noise = torch.randn_like(fake_from_buffer)
                
                real_noisy = torch.lerp(real_x0_mb, real_noise, noise_level)
                fake_noisy = torch.lerp(fake_from_buffer, fake_noise, noise_level)
                
                # D forward (only conditioned on noise_level, not timesteps)
                d_real_scores = discriminator(
                    x=real_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_level,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                d_fake_scores = discriminator(
                    x=fake_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_level,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss: D wants real > fake
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_discriminator_on_gpu_adaptive_uniform(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator on a single GPU with uniform random noise up to adaptive level.
        
        Similar to adaptive mode, but instead of fixed noise at noise_lerp_val,
        samples random noise uniformly from [0, noise_lerp_val] per sample.
        This gives the discriminator more leeway and variety in noise levels.
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]
        max_noise_level = self.noise_lerp_val  # Upper bound for uniform sampling

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]

            # G forward with no grad to get fake x0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                # Convert velocity to x0: x0 = x_t - t * v
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb - t_expanded * pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach()
                )
                
                # Sample uniform noise level per sample: [0, max_noise_level]
                # Shape: [B] for per-sample noise levels
                noise_levels = torch.rand(train_mb, device=device) * max_noise_level
                noise_levels_expanded = noise_levels[:, None, None]  # [B, 1, 1] for broadcasting
                
                # Add noise regularization with per-sample noise levels
                real_noise = torch.randn_like(real_x0_mb)
                fake_noise = torch.randn_like(fake_from_buffer)
                
                # Per-sample lerp with uniform random noise levels
                real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), noise_levels_expanded.float())
                fake_noisy = torch.lerp(fake_from_buffer.float(), fake_noise.float(), noise_levels_expanded.float())
                
                # D forward conditioned on per-sample noise_level
                d_real_scores = discriminator(
                    x=real_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_levels,  # Per-sample noise levels
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                d_fake_scores = discriminator(
                    x=fake_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_levels,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss: D wants real > fake
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_discriminator_on_gpu_timestep(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator on a single GPU using FM timestep as noise level.
        
        Instead of adaptive noise, uses the same timestep that G sees.
        D sees x0 with noise level = timestep (same distribution G is trained on).
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]

            # G forward with no grad to get fake x0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                # Convert velocity to x0: x0 = x_t - t * v
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb - t_expanded * pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach()
                )
                
                # Use timestep as noise level (same noise G sees)
                # Each sample in batch can have different timestep
                t_for_noise = timesteps_mb  # [B]
                
                # Add noise using per-sample timestep
                real_noise = torch.randn_like(real_x0_mb)
                fake_noise = torch.randn_like(fake_from_buffer)
                
                # Per-sample lerp: x_noisy = (1-t)*x0 + t*noise
                t_expanded_noise = t_for_noise[:, None, None]
                # somehow the lerp here is not happy with bfloat16, so we do it in float32 and then cast back
                real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), t_expanded_noise.float())
                fake_noisy = torch.lerp(fake_from_buffer.float(), fake_noise.float(), t_expanded_noise.float())
                
                # D forward conditioned on timestep (same as G's timestep)
                d_real_scores = discriminator(
                    x=real_noisy,
                    x_ids=img_ids_mb,
                    noise_level=timesteps_mb,  # Use timestep as noise_level
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                d_fake_scores = discriminator(
                    x=fake_noisy,
                    x_ids=img_ids_mb,
                    noise_level=timesteps_mb,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss: D wants real > fake
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_discriminator_on_gpu_gan_only(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator for one-step GAN (no FM loss).
        
        G receives pure noise (t=1.0) and outputs x0 directly.
        D sees real x0 vs G's one-step prediction with adaptive noise regularization.
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]
        noise_level = torch.tensor(self.noise_lerp_val, device=device)

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            # Generate fresh noise as input
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)  # t=1.0

            # G forward with no grad: noise -> x0 in one step
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # G predicts velocity from pure noise
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                # At t=1.0: x0 = x_t - t * v = noise - 1.0 * v = noise - v
                pred_x0 = pure_noise - pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach()
                )
                
                # Add adaptive noise regularization to prevent collapse
                real_noise = torch.randn_like(real_x0_mb)
                fake_noise = torch.randn_like(fake_from_buffer)
                
                real_noisy = torch.lerp(real_x0_mb, real_noise, noise_level)
                fake_noisy = torch.lerp(fake_from_buffer, fake_noise, noise_level)
                
                # D forward on real (with noise)
                d_real_scores = discriminator(
                    x=real_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_level,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                # D forward on fake (with noise)
                d_fake_scores = discriminator(
                    x=fake_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_level,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss: D wants real > fake
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_generator_on_gpu(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator on a single GPU.
        
        Phase 2: G forward/backward with FM loss + optional GAN loss
        Following imagenet_gan.py pattern where G only trains with GAN when D is strong.
        
        Returns dict with 'fm_loss' and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None
        noise_level = torch.tensor(self.noise_lerp_val, device=device)
        grad_eps = self.gan_config.grad_norm_eps

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_fm_loss = 0.0
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]
            target_mb = encoded_batch['target_packed'][start:end]
            loss_weights = encoded_batch['loss_weights']

            # G forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )

                # FM loss
                fm_loss_per_sample = ((pred_velocity - target_mb) ** 2).mean(dim=(1, 2))
                mb_weights = loss_weights[start:end]
                mb_weights = mb_weights / mb_weights.sum()
                fm_loss = (fm_loss_per_sample * mb_weights).sum() / num_minibatches

            total_fm_loss += fm_loss.item()

            # Compute FM gradient
            fm_grad = torch_grad(
                outputs=fm_loss,
                inputs=pred_velocity,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # GAN loss for G (only if D is strong enough)
            if train_with_gan and discriminator is not None:
                # Convert velocity to x0
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb.detach() - t_expanded * pred_velocity
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Add noise regularization
                    fake_noise = torch.randn_like(pred_x0)
                    fake_noisy = torch.lerp(pred_x0, fake_noise, noise_level)
                    
                    real_noise = torch.randn_like(real_x0_mb)
                    real_noisy = torch.lerp(real_x0_mb, real_noise, noise_level)
                    
                    # D scores for G training (only conditioned on noise_level, not timesteps)
                    d_fake_scores = discriminator(
                        x=fake_noisy,
                        x_ids=img_ids_mb,
                        noise_level=noise_level,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_noisy.detach(),
                        x_ids=img_ids_mb,
                        noise_level=noise_level,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                # Compute G gradient w.r.t. pred_x0
                g_grad_x0 = torch_grad(
                    outputs=g_loss,
                    inputs=pred_x0,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                
                # Chain rule: d(x0)/d(v) = -t
                g_grad_velocity = -t_expanded * g_grad_x0
                
                total_g_loss += g_loss.item()
                
                # Fuse FM and GAN gradients with normalization
                fm_grad_norm = fm_grad.norm() + grad_eps
                g_grad_norm = g_grad_velocity.norm() + grad_eps
                
                fm_grad_normalized = fm_grad / fm_grad_norm
                g_grad_normalized = g_grad_velocity / g_grad_norm
                
                # Proportional weighting: (1 - w) * FM + w * GAN
                w = self.gan_config.gan_loss_weight
                combined_grad = (1.0 - w) * fm_grad_normalized + w * g_grad_normalized
                combined_grad = combined_grad * fm_grad_norm
                
                pred_velocity.backward(combined_grad.detach())
            else:
                # No GAN, just FM gradient
                pred_velocity.backward(fm_grad.detach())

        return {
            'fm_loss': total_fm_loss,
            'g_loss': total_g_loss,
        }

    def _train_generator_on_gpu_adaptive_uniform(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator on a single GPU with uniform random noise up to adaptive level.
        
        Similar to adaptive mode, but samples random noise uniformly from [0, noise_lerp_val]
        per sample. This gives the discriminator more leeway and variety.
        
        Returns dict with 'fm_loss' and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None
        max_noise_level = self.noise_lerp_val  # Upper bound for uniform sampling
        grad_eps = self.gan_config.grad_norm_eps

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_fm_loss = 0.0
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]
            target_mb = encoded_batch['target_packed'][start:end]
            loss_weights = encoded_batch['loss_weights']

            # G forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )

                # FM loss
                fm_loss_per_sample = ((pred_velocity - target_mb) ** 2).mean(dim=(1, 2))
                mb_weights = loss_weights[start:end]
                mb_weights = mb_weights / mb_weights.sum()
                fm_loss = (fm_loss_per_sample * mb_weights).sum() / num_minibatches

            total_fm_loss += fm_loss.item()

            # Compute FM gradient
            fm_grad = torch_grad(
                outputs=fm_loss,
                inputs=pred_velocity,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # GAN loss for G (only if D is strong enough)
            if train_with_gan and discriminator is not None:
                # Convert velocity to x0
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb.detach() - t_expanded * pred_velocity
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Sample uniform noise level per sample: [0, max_noise_level]
                    noise_levels = torch.rand(train_mb, device=device) * max_noise_level
                    noise_levels_expanded = noise_levels[:, None, None]
                    
                    # Add noise regularization with per-sample noise levels
                    fake_noise = torch.randn_like(pred_x0)
                    fake_noisy = torch.lerp(pred_x0.float(), fake_noise.float(), noise_levels_expanded.float())
                    
                    real_noise = torch.randn_like(real_x0_mb)
                    real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), noise_levels_expanded.float())
                    
                    # D scores for G training with per-sample noise levels
                    d_fake_scores = discriminator(
                        x=fake_noisy,
                        x_ids=img_ids_mb,
                        noise_level=noise_levels,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_noisy.detach(),
                        x_ids=img_ids_mb,
                        noise_level=noise_levels,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                # Compute G gradient w.r.t. pred_x0
                g_grad_x0 = torch_grad(
                    outputs=g_loss,
                    inputs=pred_x0,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                
                # Chain rule: d(x0)/d(v) = -t
                g_grad_velocity = -t_expanded * g_grad_x0
                
                total_g_loss += g_loss.item()
                
                # Fuse FM and GAN gradients with normalization
                fm_grad_norm = fm_grad.norm() + grad_eps
                g_grad_norm = g_grad_velocity.norm() + grad_eps
                
                fm_grad_normalized = fm_grad / fm_grad_norm
                g_grad_normalized = g_grad_velocity / g_grad_norm
                
                # Proportional weighting: (1 - w) * FM + w * GAN
                w = self.gan_config.gan_loss_weight
                combined_grad = (1.0 - w) * fm_grad_normalized + w * g_grad_normalized
                combined_grad = combined_grad * fm_grad_norm
                
                pred_velocity.backward(combined_grad.detach())
            else:
                # No GAN, just FM gradient
                pred_velocity.backward(fm_grad.detach())

        return {
            'fm_loss': total_fm_loss,
            'g_loss': total_g_loss,
        }

    def _train_generator_on_gpu_timestep(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator on a single GPU using FM timestep as noise level for D.
        
        Instead of adaptive noise, uses the same timestep that G sees.
        
        Returns dict with 'fm_loss' and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None
        grad_eps = self.gan_config.grad_norm_eps

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_fm_loss = 0.0
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            timesteps_mb = encoded_batch['timesteps'][start:end]
            img_ids_mb = encoded_batch['img_ids'][start:end]
            noisy_latents_mb = encoded_batch['noisy_latents_packed'][start:end]
            target_mb = encoded_batch['target_packed'][start:end]
            loss_weights = encoded_batch['loss_weights']

            # G forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=noisy_latents_mb,
                    x_ids=img_ids_mb,
                    timesteps=timesteps_mb,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )

                # FM loss
                fm_loss_per_sample = ((pred_velocity - target_mb) ** 2).mean(dim=(1, 2))
                mb_weights = loss_weights[start:end]
                mb_weights = mb_weights / mb_weights.sum()
                fm_loss = (fm_loss_per_sample * mb_weights).sum() / num_minibatches

            total_fm_loss += fm_loss.item()

            # Compute FM gradient
            fm_grad = torch_grad(
                outputs=fm_loss,
                inputs=pred_velocity,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # GAN loss for G
            if train_with_gan and discriminator is not None:
                # Convert velocity to x0
                t_expanded = timesteps_mb[:, None, None]
                pred_x0 = noisy_latents_mb.detach() - t_expanded * pred_velocity
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Use timestep as noise level (same noise G sees)
                    t_for_noise = timesteps_mb
                    t_expanded_noise = t_for_noise[:, None, None]
                    
                    # Add noise using per-sample timestep
                    fake_noise = torch.randn_like(pred_x0)
                    fake_noisy = torch.lerp(pred_x0.float(), fake_noise.float(), t_expanded_noise.float())
                    
                    real_noise = torch.randn_like(real_x0_mb)
                    real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), t_expanded_noise.float())
                    
                    # D scores for G training (conditioned on timestep)
                    d_fake_scores = discriminator(
                        x=fake_noisy,
                        x_ids=img_ids_mb,
                        noise_level=timesteps_mb,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_noisy.detach(),
                        x_ids=img_ids_mb,
                        noise_level=timesteps_mb,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                # Compute G gradient w.r.t. pred_x0
                g_grad_x0 = torch_grad(
                    outputs=g_loss,
                    inputs=pred_x0,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                
                # Chain rule: d(x0)/d(v) = -t
                g_grad_velocity = -t_expanded * g_grad_x0
                
                total_g_loss += g_loss.item()
                
                # Fuse FM and GAN gradients with normalization
                fm_grad_norm = fm_grad.norm() + grad_eps
                g_grad_norm = g_grad_velocity.norm() + grad_eps
                
                fm_grad_normalized = fm_grad / fm_grad_norm
                g_grad_normalized = g_grad_velocity / g_grad_norm
                
                # Proportional weighting: (1 - w) * FM + w * GAN
                w = self.gan_config.gan_loss_weight
                combined_grad = (1.0 - w) * fm_grad_normalized + w * g_grad_normalized
                combined_grad = combined_grad * fm_grad_norm
                
                pred_velocity.backward(combined_grad.detach())
            else:
                # No GAN, just FM gradient
                pred_velocity.backward(fm_grad.detach())

        return {
            'fm_loss': total_fm_loss,
            'g_loss': total_g_loss,
        }

    def _train_generator_on_gpu_gan_only(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator for one-step GAN (no FM loss).
        
        G receives pure noise (t=1.0) and learns to output x0 directly.
        Only GAN loss is used - no flow matching loss.
        Uses adaptive noise regularization to prevent collapse.
        
        Returns dict with 'fm_loss' (always 0) and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None
        noise_level = torch.tensor(self.noise_lerp_val, device=device)

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)  # t=1.0

            # G forward: noise -> x0 in one step
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                # At t=1.0: x0 = x_t - t * v = noise - v
                pred_x0 = pure_noise.detach() - pred_velocity

            # GAN loss for G
            if train_with_gan and discriminator is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Add adaptive noise regularization to prevent collapse
                    fake_noise = torch.randn_like(pred_x0)
                    fake_noisy = torch.lerp(pred_x0, fake_noise, noise_level)
                    
                    real_noise = torch.randn_like(real_x0_mb)
                    real_noisy = torch.lerp(real_x0_mb, real_noise, noise_level)
                    
                    # D scores for G training (with noise)
                    d_fake_scores = discriminator(
                        x=fake_noisy,
                        x_ids=img_ids_mb,
                        noise_level=noise_level,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_noisy.detach(),
                        x_ids=img_ids_mb,
                        noise_level=noise_level,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                total_g_loss += g_loss.item()
                
                # Direct backward through G (no FM gradient fusion needed)
                g_loss.backward()
            else:
                # No GAN loss - nothing to do (no FM loss in this mode)
                pass

        return {
            'fm_loss': 0.0,  # No FM loss in gan_only mode
            'g_loss': total_g_loss,
        }

    def _train_discriminator_on_gpu_gan_only_uniform(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator for one-step GAN with uniform random noise.
        
        G receives pure noise (t=1.0) and outputs x0 directly.
        D sees samples with uniform random noise from [0, noise_lerp_val] per sample.
        This gives D more variety compared to fixed adaptive noise.
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]
        max_noise_level = self.noise_lerp_val  # Upper bound for uniform sampling

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)

            # G forward with no grad: noise -> x0 in one step
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                pred_x0 = pure_noise - pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach()
                )
                
                # Sample uniform noise level per sample: [0, max_noise_level]
                noise_levels = torch.rand(train_mb, device=device) * max_noise_level
                noise_levels_expanded = noise_levels[:, None, None]
                
                # Add uniform random noise regularization
                real_noise = torch.randn_like(real_x0_mb)
                fake_noise = torch.randn_like(fake_from_buffer)
                
                real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), noise_levels_expanded.float())
                fake_noisy = torch.lerp(fake_from_buffer.float(), fake_noise.float(), noise_levels_expanded.float())
                
                # D forward with per-sample noise levels
                d_real_scores = discriminator(
                    x=real_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_levels,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                d_fake_scores = discriminator(
                    x=fake_noisy,
                    x_ids=img_ids_mb,
                    noise_level=noise_levels,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_generator_on_gpu_gan_only_uniform(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator for one-step GAN with uniform random noise.
        
        G receives pure noise (t=1.0) and learns to output x0 directly.
        D sees samples with uniform random noise from [0, noise_lerp_val] per sample.
        
        Returns dict with 'fm_loss' (always 0) and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None
        max_noise_level = self.noise_lerp_val

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)

            # G forward: noise -> x0 in one step
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                pred_x0 = pure_noise.detach() - pred_velocity

            # GAN loss for G
            if train_with_gan and discriminator is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Sample uniform noise level per sample: [0, max_noise_level]
                    noise_levels = torch.rand(train_mb, device=device) * max_noise_level
                    noise_levels_expanded = noise_levels[:, None, None]
                    
                    # Add uniform random noise
                    fake_noise = torch.randn_like(pred_x0)
                    fake_noisy = torch.lerp(pred_x0.float(), fake_noise.float(), noise_levels_expanded.float())
                    
                    real_noise = torch.randn_like(real_x0_mb)
                    real_noisy = torch.lerp(real_x0_mb.float(), real_noise.float(), noise_levels_expanded.float())
                    
                    # D scores for G training with per-sample noise levels
                    d_fake_scores = discriminator(
                        x=fake_noisy,
                        x_ids=img_ids_mb,
                        noise_level=noise_levels,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_noisy.detach(),
                        x_ids=img_ids_mb,
                        noise_level=noise_levels,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                total_g_loss += g_loss.item()
                g_loss.backward()
            else:
                pass

        return {
            'fm_loss': 0.0,
            'g_loss': total_g_loss,
        }

    def _train_discriminator_on_gpu_gan_only_buffer(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Train discriminator for one-step GAN with adaptive replay buffer.
        
        G receives pure noise (t=1.0) and outputs x0 directly.
        D sees clean samples (no noise), but replay buffer probability is adaptive:
        - When D is too strong: use more stale/buffered samples (lower prob)
        - When D is too weak: use more fresh samples (higher prob)
        
        Returns D loss value.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id]
        replay_buffer = self.replay_buffers[gpu_id]
        
        # Adaptive buffer probability: noise_lerp_val controls staleness
        # High noise_lerp_val (D too strong) -> low prob -> more buffered samples
        # Low noise_lerp_val (D too weak) -> high prob -> more fresh samples
        buffer_prob = 1.0 - self.noise_lerp_val  # Invert: high lerp = low prob

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_d_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)

            # G forward with no grad: noise -> x0 in one step
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                pred_x0 = pure_noise - pred_velocity

            # D forward/backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Get fake samples from replay buffer with adaptive probability
                fake_from_buffer, text_from_buffer = replay_buffer.push_and_pop(
                    pred_x0.detach(), text_embeds.detach(),
                    prob_override=buffer_prob,
                )
                
                # D sees clean samples (no noise regularization)
                noise_level_zero = torch.zeros(train_mb, device=device)
                
                # D forward on real (clean)
                d_real_scores = discriminator(
                    x=real_x0_mb,
                    x_ids=img_ids_mb,
                    noise_level=noise_level_zero,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                )
                
                # D forward on fake (clean, possibly stale from buffer)
                d_fake_scores = discriminator(
                    x=fake_from_buffer,
                    x_ids=img_ids_mb,
                    noise_level=noise_level_zero,
                    ctx=text_from_buffer,
                    ctx_ids=txt_ids,
                )
                
                # Relativistic D loss
                d_relativistic = F.softplus(d_real_scores - d_fake_scores)
                d_loss = d_relativistic.mean() / num_minibatches
            
            d_loss.backward()
            total_d_loss += d_loss.item()

        return total_d_loss

    def _train_generator_on_gpu_gan_only_buffer(
        self,
        gpu_id: int,
        encoded_batch: Dict[str, torch.Tensor],
        train_with_gan: bool = True,
    ) -> Dict[str, float]:
        """
        Train generator for one-step GAN with adaptive replay buffer.
        
        G receives pure noise (t=1.0) and learns to output x0 directly.
        D sees clean samples (no noise regularization).
        
        Returns dict with 'fm_loss' (always 0) and 'g_loss'.
        """
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        discriminator = self.discriminators[gpu_id] if self.discriminators else None

        train_mb = self.training_config.train_minibatch
        num_minibatches = encoded_batch['num_minibatches']
        
        total_g_loss = 0.0

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end = start + train_mb
            
            text_embeds = encoded_batch['text_embeds_list'][mb_idx]
            txt_ids = encoded_batch['txt_ids_list'][mb_idx]
            real_x0_mb = encoded_batch['real_x0_packed'][start:end].detach()
            img_ids_mb = encoded_batch['img_ids'][start:end]
            
            # For one-step GAN: G receives pure noise (t=1.0)
            pure_noise = torch.randn_like(real_x0_mb)
            timestep_one = torch.ones(train_mb, device=device)

            # G forward: noise -> x0 in one step
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity = model(
                    x=pure_noise,
                    x_ids=img_ids_mb,
                    timesteps=timestep_one,
                    ctx=text_embeds,
                    ctx_ids=txt_ids,
                    guidance=None,
                )
                pred_x0 = pure_noise.detach() - pred_velocity

            # GAN loss for G
            if train_with_gan and discriminator is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # D sees clean samples (no noise)
                    noise_level_zero = torch.zeros(train_mb, device=device)
                    
                    # D scores for G training (G always uses fresh samples)
                    d_fake_scores = discriminator(
                        x=pred_x0,
                        x_ids=img_ids_mb,
                        noise_level=noise_level_zero,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    d_real_scores = discriminator(
                        x=real_x0_mb,
                        x_ids=img_ids_mb,
                        noise_level=noise_level_zero,
                        ctx=text_embeds,
                        ctx_ids=txt_ids,
                    )
                    
                    # G wants fake > real
                    g_relativistic = F.softplus(d_fake_scores - d_real_scores)
                    g_loss = g_relativistic.mean() / num_minibatches
                
                total_g_loss += g_loss.item()
                g_loss.backward()
            else:
                pass

        return {
            'fm_loss': 0.0,
            'g_loss': total_g_loss,
        }

    def _clip_grads_on_gpu(self, gpu_id: int):
        """Clip gradients on generator on a specific GPU."""
        if self.training_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.models[gpu_id].parameters(),
                self.training_config.max_grad_norm
            )

    def _clip_d_grads_on_gpu(self, gpu_id: int):
        """Clip gradients on discriminator on a specific GPU."""
        if self.gan_config.enabled and self.discriminators:
            if self.training_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminators[gpu_id].parameters(),
                    self.training_config.max_grad_norm
                )

    def _optimizer_step_on_gpu(self, gpu_id: int):
        """Run optimizer step for generator on a specific GPU."""
        self.optimizers[gpu_id].step()
        self.schedulers[gpu_id].step()
        self.optimizers[gpu_id].zero_grad()

    def _d_optimizer_step_on_gpu(self, gpu_id: int):
        """Run optimizer step for discriminator on a specific GPU."""
        if self.gan_config.enabled and self.d_optimizers:
            self.d_optimizers[gpu_id].step()
            self.d_optimizers[gpu_id].zero_grad()

    def _update_adaptive_noise(self, avg_d_loss: float):
        """Update adaptive noise level based on discriminator loss."""
        if not self.gan_config.enabled:
            return
            
        target = self.gan_config.target_d_loss
        floor = self.gan_config.target_d_loss_floor
        ema_decay = self.gan_config.noise_ema_decay
        
        # Compute raw scale with floor:
        # - D loss >= target: raw_scale = 0.0 (no regularization needed)
        # - D loss <= floor: raw_scale = 1.0 (max regularization)
        # - D loss in (floor, target): linear interpolation
        #
        # This prevents collapse by hitting max regularization before D loss = 0
        # e.g., floor=0.1 means 100% noise/buffer when D loss drops to 0.1
        if avg_d_loss >= target:
            raw_scale = 0.0
        elif avg_d_loss <= floor:
            raw_scale = 1.0
        else:
            # Linear interpolation between floor and target
            # (target - avg_d_loss) / (target - floor) maps [floor, target] -> [1, 0]
            raw_scale = (target - avg_d_loss) / (target - floor)
        
        raw_scale = max(0.0, min(1.0, raw_scale))
        
        # EMA update
        new_noise_lerp = (ema_decay * self.noise_lerp_val) + ((1 - ema_decay) * raw_scale)
        
        # Discretize to bins if configured
        num_bins = self.gan_config.noise_discrete_bins
        if num_bins > 0:
            # Round to nearest bin: e.g., 10 bins -> [0.0, 0.1, 0.2, ..., 1.0]
            # bin_size = 1.0 / num_bins
            # discretized = round(value / bin_size) * bin_size
            bin_size = 1.0 / num_bins
            new_noise_lerp = round(new_noise_lerp / bin_size) * bin_size
            new_noise_lerp = max(0.0, min(1.0, new_noise_lerp))  # Clamp to [0, 1]
        
        self.noise_lerp_val = new_noise_lerp
        self.prev_d_loss_metric = avg_d_loss

    def _preprocess_batch(self, batch) -> Tuple[List[Dict], List[str], int]:
        """
        Preprocess batch and encode on all GPUs.
        
        Returns:
            encoded_batches: List of encoded batch dicts per GPU
            captions: Preprocessed captions
            samples_per_gpu: Number of samples per GPU
        """
        images, captions, _, loss_weights = batch

        # Preprocess captions
        captions = [c if c else "" for c in captions]
        captions = [c.lower() if random.random() < 0.25 else c for c in captions]

        batch_size = images.shape[0]
        samples_per_gpu = batch_size // self.n_gpus

        # Encode batch on all GPUs (shared preprocessing)
        def encode_on_gpu(gpu_id):
            start = gpu_id * samples_per_gpu
            end = start + samples_per_gpu
            return self._encode_batch_on_gpu(
                gpu_id=gpu_id,
                images_chunk=images[start:end],
                captions_chunk=captions[start:end],
                loss_weights_chunk=loss_weights[start:end],
            )
        
        encoded_batches = list(self.executor.map(encode_on_gpu, range(self.n_gpus)))
        return encoded_batches, captions, samples_per_gpu

    def train_step_discriminator(self, encoded_batches: List[Dict]) -> float:
        """
        Train discriminator on one batch (accumulates gradients).
        
        Call this N times, then call step_discriminator() to update weights.
        Following imagenet_gan.py pattern: D, D, D, ... D update
        
        Dispatches to adaptive or timestep-based training based on noise_mode config.
        
        Args:
            encoded_batches: Pre-encoded batch from _preprocess_batch()
            
        Returns:
            D loss for this batch
        """
        if not self.gan_config.enabled or not self.discriminators:
            return 0.0
        
        # Select training method based on noise mode
        noise_mode = self.gan_config.noise_mode
        
        # D forward/backward on all GPUs (accumulates gradients)
        def train_d_on_gpu(gpu_id):
            if noise_mode == "gan_only":
                return self._train_discriminator_on_gpu_gan_only(gpu_id, encoded_batches[gpu_id])
            elif noise_mode == "gan_only_uniform":
                return self._train_discriminator_on_gpu_gan_only_uniform(gpu_id, encoded_batches[gpu_id])
            elif noise_mode == "gan_only_buffer":
                return self._train_discriminator_on_gpu_gan_only_buffer(gpu_id, encoded_batches[gpu_id])
            elif noise_mode == "timestep":
                return self._train_discriminator_on_gpu_timestep(gpu_id, encoded_batches[gpu_id])
            elif noise_mode == "adaptive_uniform":
                return self._train_discriminator_on_gpu_adaptive_uniform(gpu_id, encoded_batches[gpu_id])
            else:  # adaptive
                return self._train_discriminator_on_gpu(gpu_id, encoded_batches[gpu_id])
        
        d_losses = list(self.executor.map(train_d_on_gpu, range(self.n_gpus)))
        total_d_loss = sum(d_losses) / self.n_gpus
        
        return total_d_loss

    def step_discriminator(self, avg_d_loss: float):
        """
        Update discriminator weights after gradient accumulation.
        
        Call after accumulating D gradients over N batches.
        """
        if not self.gan_config.enabled or not self.discriminators:
            return
        
        # All-reduce D gradients across GPUs
        self._all_reduce_discriminator_gradients()
        
        # Clip and step D
        list(self.executor.map(self._clip_d_grads_on_gpu, range(self.n_gpus)))
        list(self.executor.map(self._d_optimizer_step_on_gpu, range(self.n_gpus)))
        
        # Update adaptive noise based on D loss (in adaptive, adaptive_uniform, gan_only, gan_only_uniform, and gan_only_buffer modes)
        if self.gan_config.noise_mode in ("adaptive", "adaptive_uniform", "gan_only", "gan_only_uniform", "gan_only_buffer"):
            self._update_adaptive_noise(avg_d_loss)

    def zero_discriminator_grads(self):
        """Zero discriminator gradients before accumulation."""
        if not self.gan_config.enabled or not self.discriminators:
            return
        list(self.executor.map(
            lambda gpu_id: self.d_optimizers[gpu_id].zero_grad(set_to_none=True),
            range(self.n_gpus)
        ))

    def train_step_generator(self, encoded_batches: List[Dict], train_with_gan: bool = True) -> Dict[str, float]:
        """
        Train generator on one batch (accumulates gradients).
        
        Call this N times, then call step_generator() to update weights.
        Following imagenet_gan.py pattern: G, G, G, ... G update
        
        Dispatches to adaptive or timestep-based training based on noise_mode config.
        
        Args:
            encoded_batches: Pre-encoded batch from _preprocess_batch()
            train_with_gan: Whether to include GAN loss (only when D is strong)
            
        Returns:
            Dict with 'fm_loss' and 'g_loss' for this batch
        """
        # Select training method based on noise mode
        noise_mode = self.gan_config.noise_mode
        
        def train_g_on_gpu(gpu_id):
            if noise_mode == "gan_only":
                return self._train_generator_on_gpu_gan_only(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
            elif noise_mode == "gan_only_uniform":
                return self._train_generator_on_gpu_gan_only_uniform(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
            elif noise_mode == "gan_only_buffer":
                return self._train_generator_on_gpu_gan_only_buffer(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
            elif noise_mode == "timestep":
                return self._train_generator_on_gpu_timestep(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
            elif noise_mode == "adaptive_uniform":
                return self._train_generator_on_gpu_adaptive_uniform(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
            else:  # adaptive
                return self._train_generator_on_gpu(
                    gpu_id, 
                    encoded_batches[gpu_id],
                    train_with_gan=train_with_gan,
                )
        
        g_losses = list(self.executor.map(train_g_on_gpu, range(self.n_gpus)))
        
        return {
            'fm_loss': sum(d['fm_loss'] for d in g_losses) / self.n_gpus,
            'g_loss': sum(d['g_loss'] for d in g_losses) / self.n_gpus,
        }

    def step_generator(self):
        """
        Update generator weights after gradient accumulation.
        
        Call after accumulating G gradients over N batches.
        """
        # All-reduce G gradients across GPUs
        self._all_reduce_gradients()

        # Gradient clipping on all GPUs in parallel
        list(self.executor.map(self._clip_grads_on_gpu, range(self.n_gpus)))

        # G optimizer step
        list(self.executor.map(self._optimizer_step_on_gpu, range(self.n_gpus)))

    def zero_generator_grads(self):
        """Zero generator gradients before accumulation."""
        list(self.executor.map(
            lambda gpu_id: self.optimizers[gpu_id].zero_grad(set_to_none=True),
            range(self.n_gpus)
        ))

    def train_step(self, batch) -> float:
        """
        Required by BaseTrainer abstract class.
        
        Note: This trainer uses train_step_discriminator() and train_step_generator()
        separately for proper GAN training. This method is not used directly.
        """
        # This is just to satisfy the abstract method requirement
        # The actual training uses train_step_discriminator and train_step_generator
        raise NotImplementedError(
            "Use train_step_discriminator() and train_step_generator() instead"
        )

    def _inference_on_gpu(
        self,
        gpu_id: int,
        prompts: List[str],
        config: InferenceConfig,
    ) -> torch.Tensor:
        """Run inference on a single GPU."""
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
            
            # Use discrete timesteps if configured, otherwise use default schedule
            if self.gan_config.discrete_timesteps:
                # Use the discrete timesteps from training config
                # These should be in descending order (e.g., [1.0, 0.75, 0.5, 0.25])
                timesteps = list(self.gan_config.discrete_timesteps)
                # Ensure they end at 0 for final denoising step
                if timesteps[-1] != 0.0:
                    timesteps.append(0.0)
            else:
                timesteps = get_schedule(config.steps, image_seq_len)

            # Encode prompts
            qwen_embed, prompt_masks = text_encoder.encode(prompts)
            qwen_embed_neg, prompt_masks_neg = text_encoder.encode_negative(batch_size)

            # Prepare position IDs (4D format for Flux2)
            img_ids = prepare_img_ids(batch_size, height, width, device)
            txt_ids = prepare_txt_ids(batch_size, qwen_embed.shape[1], device)
            neg_txt_ids = prepare_txt_ids(batch_size, qwen_embed_neg.shape[1], device)

            # Run denoising with CFG
            output_latent = denoise_cfg(
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
            )

            # Unpack and decode
            output_latent = unpack_latents(output_latent, shape)
            output_image = ae.decode(output_latent)

        model.train()
        return output_image

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
        base_prompts = list(config.prompts)

        def inference_on_gpu(gpu_id):
            prompts = list(base_prompts)
            if extra_prompts_per_gpu and gpu_id < len(extra_prompts_per_gpu):
                prompts.append(extra_prompts_per_gpu[gpu_id])
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
        """Save model checkpoint (generator and optionally discriminator)."""
        # Save generator
        torch.save(self.model.state_dict(), path)
        print(f"Saved generator checkpoint: {path}")
        
        # Save discriminator if GAN enabled
        if self.gan_config.enabled and self.discriminators:
            # Save discriminator weights as safetensors (for easy loading with load_safetensors)
            d_path = path.replace('.pth', '_discriminator.safetensors')
            d_state_dict = {k: v.contiguous() for k, v in self.discriminator.state_dict().items()}
            save_safetensors(d_state_dict, d_path)
            print(f"Saved discriminator checkpoint: {d_path}")
            
            # Save training state (noise_lerp_val, etc.) separately as JSON
            state_path = path.replace('.pth', '_discriminator_state.json')
            training_state = {
                'noise_lerp_val': self.noise_lerp_val,
                'prev_d_loss_metric': self.prev_d_loss_metric,
            }
            with open(state_path, 'w') as f:
                json.dump(training_state, f, indent=2)
            print(f"Saved discriminator training state: {state_path}")

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
        """Main training loop with GAN training."""
        print("Starting training...")
        self.model.train()
        
        # Set discriminators to train mode if enabled
        if self.gan_config.enabled and self.discriminators:
            for d in self.discriminators:
                d.train()
            if self.gan_config.noise_mode == "adaptive":
                print(f"GAN training enabled with ADAPTIVE noise (initial: {self.noise_lerp_val:.3f})")
            elif self.gan_config.noise_mode == "adaptive_uniform":
                print(f"GAN training enabled with ADAPTIVE_UNIFORM noise (uniform [0, {self.noise_lerp_val:.3f}])")
            elif self.gan_config.noise_mode == "timestep":
                print(f"GAN training enabled with TIMESTEP noise (D sees same noise level as G)")
            elif self.gan_config.noise_mode == "gan_only":
                print(f"GAN training enabled with GAN_ONLY mode (one-step generation, no FM loss, adaptive noise: {self.noise_lerp_val:.3f})")
            elif self.gan_config.noise_mode == "gan_only_uniform":
                print(f"GAN training enabled with GAN_ONLY_UNIFORM mode (one-step, no FM loss, uniform [0, {self.noise_lerp_val:.3f}])")
            else:  # gan_only_buffer
                print(f"GAN training enabled with GAN_ONLY_BUFFER mode (one-step, no FM loss, adaptive buffer prob)")

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
                
                grad_accum_steps = self.training_config.gradient_accumulation_steps
                
                # Accumulators for logging
                accum_d_losses = []
                accum_fm_losses = []
                accum_g_losses = []
                accum_encoded_batches = []
                
                # Last computed losses (carry over for logging on non-accum steps)
                last_losses = {'fm_loss': 0.0, 'd_loss': 0.0, 'g_loss': 0.0}

                for step, batch_data in pbar:
                    # Stop profiler after profile_steps
                    if profiler_active and step >= profile_steps:
                        prof.stop()
                        prof.export_chrome_trace(self.training_config.profile_json_dump)
                        print(f"\nProfiler stopped. Trace saved to {self.training_config.profile_json_dump}")
                        profiler_active = False

                    batch_data = batch_data[0]  # Unwrap
                    
                    # Preprocess and encode batch
                    encoded_batches, captions, _ = self._preprocess_batch(batch_data)
                    accum_encoded_batches.append(encoded_batches)
                    
                    accum_step = step % grad_accum_steps
                    is_last_accum_step = (step + 1) % grad_accum_steps == 0

                    # =============================================================
                    # Following imagenet_gan.py pattern:
                    # D, D, D, ... D update → G, G, G, ... G update
                    # =============================================================
                    
                    if is_last_accum_step:
                        # -----------------------------------------------------
                        # Phase 1: Train Discriminator (accumulate then step)
                        # D, D, D, ... D update
                        # -----------------------------------------------------
                        if self.gan_config.enabled:
                            self.zero_discriminator_grads()
                            
                            for enc_batch in accum_encoded_batches:
                                d_loss = self.train_step_discriminator(enc_batch)
                                accum_d_losses.append(d_loss)
                            
                            avg_d_loss = sum(accum_d_losses) / len(accum_d_losses) if accum_d_losses else 0.0
                            self.step_discriminator(avg_d_loss)
                        
                        # -----------------------------------------------------
                        # Phase 2: Train Generator (accumulate then step)
                        # G, G, G, ... G update
                        # In adaptive mode: only train G with GAN when D is strong
                        # In timestep/gan_only mode: always train with GAN
                        # -----------------------------------------------------
                        if self.gan_config.noise_mode in ("timestep", "gan_only", "gan_only_uniform", "gan_only_buffer"):
                            # Timestep/GAN-only modes: always train with GAN
                            train_g_with_gan = self.gan_config.enabled
                        else:
                            # Adaptive mode: only train G with GAN when D is strong enough
                            train_g_with_gan = (
                                self.gan_config.enabled and 
                                self.prev_d_loss_metric < self.gan_config.target_d_loss
                            )
                        
                        self.zero_generator_grads()
                        
                        for enc_batch in accum_encoded_batches:
                            g_losses = self.train_step_generator(enc_batch, train_with_gan=train_g_with_gan)
                            accum_fm_losses.append(g_losses['fm_loss'])
                            accum_g_losses.append(g_losses['g_loss'])
                        
                        self.step_generator()
                        
                        # Compute average losses for logging
                        last_losses = {
                            'fm_loss': sum(accum_fm_losses) / len(accum_fm_losses) if accum_fm_losses else 0.0,
                            'd_loss': sum(accum_d_losses) / len(accum_d_losses) if accum_d_losses else 0.0,
                            'g_loss': sum(accum_g_losses) / len(accum_g_losses) if accum_g_losses else 0.0,
                        }
                        
                        # Reset accumulators
                        accum_d_losses = []
                        accum_fm_losses = []
                        accum_g_losses = []
                        accum_encoded_batches = []
                    
                    # Use last computed losses for logging (carries over on non-accum steps)
                    losses = last_losses

                    # Update progress bar
                    lr = self.scheduler.get_last_lr()[0]
                    if self.gan_config.enabled:
                        pbar.set_postfix({
                            "fm": f"{losses['fm_loss']:.4f}",
                            "d": f"{losses['d_loss']:.4f}",
                            "g": f"{losses['g_loss']:.4f}",
                            "noise": f"{self.noise_lerp_val:.3f}",
                            "lr": f"{lr:.2e}",
                        })
                    else:
                        pbar.set_postfix({"loss": f"{losses['fm_loss']:.4f}", "lr": f"{lr:.2e}"})

                    # Logging
                    self.logger.log_scalar("fm_loss", losses['fm_loss'], self.global_step)
                    self.logger.log_scalar("learning_rate", lr, self.global_step)
                    
                    if self.gan_config.enabled:
                        self.logger.log_scalar("d_loss", losses['d_loss'], self.global_step)
                        self.logger.log_scalar("g_loss", losses['g_loss'], self.global_step)
                        self.logger.log_scalar("noise_level", self.noise_lerp_val, self.global_step)

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

    parser = argparse.ArgumentParser(description="Train Flux2 Klein model (multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="train_flux2_klein_experimental.json",
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
