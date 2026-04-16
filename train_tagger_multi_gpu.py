"""Multi-GPU DINOv3 ViT Tagger Trainer

Trains a multi-label image tagger (DINOv3 ViT backbone + linear projection head)
using the same ThreadPoolExecutor + NCCL all-reduce parallelism pattern as
train_zimage_pixelspace_class_multi_gpu.py.

Architecture recap
------------------
- Backbone: DINOv3 ViT (configurable from the supported model list).
- Head: linear projection from concat(CLS, register_tokens) → num_tags.
  No bottleneck — concat_dim = hidden_size × (1 + num_register_tokens).
- Loss: BCEWithLogitsLoss with per-tag pos_weight for class imbalance.
- Dataset: ParquetTaggerDataset — a thin subclass of ParquetTextImageDataset
  that (a) hard-filters to tag-only caption columns, (b) builds a tag
  vocabulary (tag → index LUT) from the loaded data, and (c) returns
  multi-hot label tensors instead of caption strings.

Resolution handling
-------------------
DINOv3 ViT accepts any resolution that is a multiple of 16 px (patch size).
The dataset reuses the existing bucketing logic from ParquetTextImageDataset —
just set base_resolution to whatever resolutions you want (e.g. [512, 768, 1024]).
Images are normalised with ImageNet mean/std as expected by DINOv3.

Config JSON layout
------------------
{
    "training": {
        "master_seed": 42,
        "train_minibatch": 4,
        "gradient_accumulation_steps": 4,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        "save_every": 500,
        "save_folder": "tagger_checkpoints",
        "pos_weight_cap": 100.0,
        "use_aim": false,
        "aim_path": null,
        "aim_experiment_name": null
    },
    "model": {
        "backbone_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "checkpoint_path": "",
        "dtype": "bfloat16",
        "freeze_backbone": false,
        "projection_bias": false
    },
    "eval": {
        "eval_every": 500,
        "eval_folder": "tagger_eval"
    },
    "parquet_dataloader": {
        "batch_size": 64,
        "parquet_sources": {
            "e621": {"path": "/data/parquet/source=e621", "n_samples": 1000000}
        },
        "tag_columns": {
            "tags": {"weight": 1.0}
        },
        "filename_column": "url",
        "width_column":    "image_width",
        "height_column":   "image_height",
        "loss_weight_column": null,
        "image_folder_path": "",
        "base_resolution": [512, 768, 1024],
        "ratio_cutoff": 2.0,
        "resolution_step": 16,
        "min_tag_count": 10,
        "num_workers": 4,
        "prefetch_factor": 2,
        "offset": 0,
        "vocab_path": "tagger_vocab.json"
    }
}
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.cuda.nccl as nccl
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import safe_open, save_file

from src.dataloaders.parquet_dataloader import ParquetTextImageDataset
from src.models.tagger.model import DINOv3Tagger, DINOv3TaggerParams

# Optional Aim tracking
try:
    from aim import Run as _AimRun  # type: ignore[import-untyped]
    AIM_AVAILABLE = True
except ImportError:
    _AimRun = None  # type: ignore[assignment,misc]
    AIM_AVAILABLE = False


# =============================================================================
# Tag vocabulary helpers
# =============================================================================

def _parse_tags(caption: str) -> list[str]:
    """Split a comma-separated tag string into a cleaned list."""
    return [t.strip() for t in caption.split(",") if t.strip()]


def _build_vocab(
    batches: list[list[dict]],
    min_count: int,
) -> tuple[dict[str, int], list[str], Counter]:
    """Count tags across all batches and return (tag2idx, idx2tag, counter).

    Tags are sorted by frequency (desc) then alphabetically for determinism.
    Only tags appearing >= min_count times are kept.
    The raw Counter is returned so callers can reuse it (e.g. for pos_weight)
    without a second pass over the data.
    """
    counter: Counter = Counter()
    for batch in batches:
        for sample in batch:
            counter.update(sample.get("_tags", []))

    sorted_tags = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    idx2tag = [tag for tag, cnt in sorted_tags if cnt >= min_count]
    tag2idx = {tag: i for i, tag in enumerate(idx2tag)}
    return tag2idx, idx2tag, counter


def _tags_to_multihot(tags: list[str], tag2idx: dict[str, int], num_tags: int) -> torch.Tensor:
    vec = torch.zeros(num_tags, dtype=torch.float32)
    for tag in tags:
        idx = tag2idx.get(tag)
        if idx is not None:
            vec[idx] = 1.0
    return vec


# =============================================================================
# ParquetTaggerDataset — thin subclass of ParquetTextImageDataset
# =============================================================================

# ImageNet normalisation expected by DINOv3 ViT
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class ParquetTaggerDataset(ParquetTextImageDataset):
    """Parquet dataset for multi-label tag classification.

    Extends ParquetTextImageDataset with:
    1. Hard filter: only caption columns marked ``is_tag_based: true`` are used.
       Any column without that flag raises an error at construction time.
    2. Tag vocabulary built from the loaded data (or loaded from a JSON file).
    3. Image normalisation switched to ImageNet mean/std (DINOv3 expects this).
    4. __getitem__ returns (images, multi_hot_labels, index, loss_weights)
       instead of (images, captions, index, loss_weights).

    Parameters
    ----------
    tag_columns : dict
        ``{col_name: {"weight": float}}`` — must all be tag-based columns.
        The ``is_tag_based`` flag is injected automatically (always True here).
    min_tag_count : int
        Tags appearing fewer than this many times are excluded from the vocab.
    vocab_path : str | None
        Path to a JSON vocab file.  If the file exists it is loaded; otherwise
        the vocab is built from the data and saved here.
    All other parameters are forwarded to ParquetTextImageDataset.
    """

    def __init__(
        self,
        *,
        tag_columns: dict[str, dict],
        min_tag_count: int = 10,
        vocab_path: str | None = "tagger_vocab.json",
        # forwarded to parent
        batch_size: int,
        parquet_sources: dict,
        filename_column: str = "url",
        width_column: str = "image_width",
        height_column: str = "image_height",
        loss_weight_column: str | None = None,
        image_folder_path: str = "",
        base_res: list[int] | None = None,
        ratio_cutoff: float = 2.0,
        resolution_step: int = 16,
        seed: int = 0,
        rank: int = 0,
        num_gpus: int = 1,
        offset: int = 0,
    ):
        # Inject is_tag_based=True for every column — this is a tagger dataset
        caption_columns = {
            col: {"weight": cfg.get("weight", 1.0), "is_tag_based": True}
            for col, cfg in tag_columns.items()
        }

        # Parent uses [-0.5, 0.5] normalisation by default; we override below
        super().__init__(
            batch_size=batch_size,
            parquet_sources=parquet_sources,
            caption_columns=caption_columns,
            filename_column=filename_column,
            width_column=width_column,
            height_column=height_column,
            loss_weight_column=loss_weight_column,
            image_folder_path=image_folder_path,
            base_res=base_res if base_res is not None else [512],
            ratio_cutoff=ratio_cutoff,
            resolution_step=resolution_step,
            shuffle_tags=False,       # we keep all tags for label construction
            tag_drop_percentage=0.0,  # no dropout — we need complete labels
            uncond_percentage=0.0,    # no unconditional drop
            seed=seed,
            rank=rank,
            num_gpus=num_gpus,
            offset=offset,
        )

        # Override image transforms: ImageNet normalisation for DINOv3
        self.image_transforms = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Stash parsed tags on every sample in every batch for vocab building
        for batch in self.batches:
            for sample in batch:
                sample["_tags"] = _parse_tags(sample.get("caption_or_tags", ""))

        # Build or load vocabulary
        self.min_tag_count = min_tag_count
        self.vocab_path = vocab_path

        if vocab_path and os.path.exists(vocab_path):
            print(f"[ParquetTaggerDataset] Loading vocab from {vocab_path}")
            with open(vocab_path) as f:
                data = json.load(f)
            self.idx2tag: list[str] = data["idx2tag"]
            self.tag2idx: dict[str, int] = {t: i for i, t in enumerate(self.idx2tag)}
            print(f"[ParquetTaggerDataset] Vocab loaded: {len(self.idx2tag):,} tags")
            # Vocab was loaded from file — still need counts for pos_weight.
            # One fast pass over the already-in-memory batch list (no I/O).
            self.tag_counts: Counter = Counter()
            for batch in self.batches:
                for sample in batch:
                    self.tag_counts.update(sample.get("_tags", []))
        else:
            print("[ParquetTaggerDataset] Building tag vocabulary...")
            self.tag2idx, self.idx2tag, self.tag_counts = _build_vocab(self.batches, min_tag_count)
            print(f"[ParquetTaggerDataset] Vocab: {len(self.idx2tag):,} tags (min_count={min_tag_count})")
            if vocab_path:
                with open(vocab_path, "w") as f:
                    json.dump({"idx2tag": self.idx2tag}, f)
                print(f"[ParquetTaggerDataset] Vocab saved to {vocab_path}")

        self.num_tags = len(self.idx2tag)

    # ------------------------------------------------------------------
    # Override __getitem__ to return multi-hot labels
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        # Call parent to get images + captions (caption_or_tags is the raw tag string)
        result = super().__getitem__(index)

        # Parent returns (images, captions, index, loss_weights[, ref_images])
        images = result[0]          # [B, 3, H, W]
        captions = result[1]        # list[str] — comma-separated tags
        idx = result[2]
        loss_weights = result[3]

        # Build multi-hot label matrix
        labels = torch.stack([
            _tags_to_multihot(_parse_tags(cap), self.tag2idx, self.num_tags)
            for cap in captions
        ])  # [B, num_tags]

        return images, labels, idx, loss_weights

    # ------------------------------------------------------------------
    # resample — rebuild batches and re-stash tags
    # ------------------------------------------------------------------

    def resample(self):
        super().resample()
        for batch in self.batches:
            for sample in batch:
                sample["_tags"] = _parse_tags(sample.get("caption_or_tags", ""))


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    master_seed: int = 42
    train_minibatch: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    save_every: int = 500
    save_folder: str = "tagger_checkpoints"
    pos_weight_cap: float = 100.0
    # Temperature for pos_weight scaling: pos_weight = (neg/pos)^(1/T).
    # T=1.0 → standard inverse-frequency weighting.
    # T>1.0 → softer weights, closer to uniform (less aggressive upweighting).
    # T<1.0 → harder weights, more aggressive upweighting of rare tags.
    pos_weight_temperature: float = 1.0
    use_aim: bool = False
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None


@dataclass
class ModelConfig:
    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    checkpoint_path: str = ""
    dtype: str = "bfloat16"
    freeze_backbone: bool = False
    projection_bias: bool = False


@dataclass
class EvalConfig:
    eval_every: int = 500
    eval_folder: str = "tagger_eval"


@dataclass
class ParquetDataloaderConfig:
    batch_size: int = 64
    parquet_sources: Dict[str, Any] = field(default_factory=dict)
    tag_columns: Dict[str, Any] = field(default_factory=dict)
    filename_column: str = "url"
    width_column: str = "image_width"
    height_column: str = "image_height"
    loss_weight_column: Optional[str] = None
    image_folder_path: str = ""
    base_resolution: List[int] = field(default_factory=lambda: [512, 768, 1024])
    ratio_cutoff: float = 2.0
    resolution_step: int = 16
    min_tag_count: int = 10
    num_workers: int = 4
    prefetch_factor: int = 2
    offset: int = 0
    vocab_path: str = "tagger_vocab.json"


# =============================================================================
# Experiment logger
# =============================================================================

class ExperimentLogger:
    def __init__(self, config: TrainingConfig, hparams: Dict[str, Any], save_folder: str = "."):
        self.run = None
        self.csv_file = None

        if AIM_AVAILABLE and _AimRun is not None and config.use_aim and config.aim_path:
            self.run = _AimRun(repo=config.aim_path, experiment=config.aim_experiment_name)
            self.run["hparams"] = hparams
        else:
            csv_path = os.path.join(save_folder, "training_log.csv")
            file_exists = os.path.exists(csv_path)
            self.csv_file = open(csv_path, "a", newline="")
            self.csv_writer = __import__("csv").writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow(["step", "name", "value"])
                self.csv_file.flush()

    def log_scalar(self, name: str, value: float, step: int):
        if self.run:
            self.run.track(value, name=name, step=step)
        elif self.csv_file:
            self.csv_writer.writerow([step, name, value])
            if step % 10 == 0:
                self.csv_file.flush()

    def close(self):
        if self.run:
            self.run.close()
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None


# =============================================================================
# Tagger Trainer
# =============================================================================

class TaggerTrainer:
    """Multi-GPU trainer for DINOv3Tagger.

    Reuses the same ThreadPoolExecutor + NCCL all-reduce pattern as the
    diffusion trainer: one model replica per GPU, forward/backward in parallel
    threads, gradients averaged via NCCL, optimizer stepped on main thread.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.global_step = 0

        with open(config_path) as f:
            self.config_data = json.load(f)

        self._parse_configs()

        self.n_gpus = torch.cuda.device_count()
        self.models: List[DINOv3Tagger] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.schedulers: List[torch.optim.lr_scheduler.LRScheduler] = []
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.dataset: ParquetTaggerDataset
        self.logger: ExperimentLogger
        self._pos_weights: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _parse_configs(self):
        self.training_config = TrainingConfig(**self.config_data.get("training", {}))
        self.model_config = ModelConfig(**self.config_data.get("model", {}))
        self.eval_config = EvalConfig(**self.config_data.get("eval", {}))
        self.dataloader_config = ParquetDataloaderConfig(**self.config_data["parquet_dataloader"])

    def _save_config(self, path: str):
        with open(path, "w") as f:
            json.dump(self.config_data, f, indent=4)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        print(f"[TaggerTrainer] Setting up with {self.n_gpus} GPUs")
        torch.manual_seed(self.training_config.master_seed)
        random.seed(self.training_config.master_seed)

        os.makedirs(self.training_config.save_folder, exist_ok=True)
        os.makedirs(self.eval_config.eval_folder, exist_ok=True)
        self._save_config(f"{self.training_config.save_folder}/training_config.json")

        self._setup_dataset()   # must come first — num_tags needed for model
        self._load_models()
        self._setup_optimizer()

        self.logger = ExperimentLogger(
            self.training_config, self.config_data,
            save_folder=self.training_config.save_folder,
        )

        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)

        print("[TaggerTrainer] Setup complete!")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _setup_dataset(self):
        cfg = self.dataloader_config
        self.dataset = ParquetTaggerDataset(
            batch_size=cfg.batch_size,
            parquet_sources=cfg.parquet_sources,
            tag_columns=cfg.tag_columns,
            filename_column=cfg.filename_column,
            width_column=cfg.width_column,
            height_column=cfg.height_column,
            loss_weight_column=cfg.loss_weight_column,
            image_folder_path=cfg.image_folder_path,
            base_res=cfg.base_resolution,
            ratio_cutoff=cfg.ratio_cutoff,
            resolution_step=cfg.resolution_step,
            min_tag_count=cfg.min_tag_count,
            seed=self.training_config.master_seed,
            rank=0,
            num_gpus=1,
            offset=cfg.offset,
            vocab_path=cfg.vocab_path,
        )
        self.num_tags = self.dataset.num_tags
        self.tag2idx  = self.dataset.tag2idx
        self.idx2tag  = self.dataset.idx2tag
        print(f"[TaggerTrainer] num_tags = {self.num_tags:,}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        print("[TaggerTrainer] Loading models...")
        params = DINOv3TaggerParams(
            backbone_name=self.model_config.backbone_name,
            num_tags=self.num_tags,
            freeze_backbone=self.model_config.freeze_backbone,
            projection_bias=self.model_config.projection_bias,
            dtype=self.model_config.dtype,
        )

        checkpoint_sd = None
        if self.model_config.checkpoint_path:
            print(f"  Loading checkpoint: {self.model_config.checkpoint_path}")
            checkpoint_sd = self._load_state_dict(self.model_config.checkpoint_path)

        self.models = []
        for gpu_id in range(self.n_gpus):
            device = f"cuda:{gpu_id}"
            print(f"  Initialising model on {device}...")
            model = DINOv3Tagger(params)

            if checkpoint_sd is not None:
                loaded, mismatched, new = self._apply_state_dict(model, checkpoint_sd, device)
                if gpu_id == 0:
                    print(f"    Loaded {len(loaded)} keys, {len(mismatched)} mismatched, {len(new)} new")

            model = model.to(device)
            model.projection = model.projection.to(torch.float32)
            model.train()
            self.models.append(model)

        self.model = self.models[0]
        total     = self.model.total_parameter_count()
        trainable = self.model.trainable_parameter_count()
        print(f"  Model on {self.n_gpus} GPUs | total={total:,} | trainable={trainable:,}")

        self._build_pos_weights()

    def _build_pos_weights(self):
        """Compute per-tag BCEWithLogitsLoss pos_weight from tag frequency.

        Reads directly from dataset.tag_counts (a Counter built during vocab
        construction) — no second pass over the data needed.
        """
        cap        = self.training_config.pos_weight_cap
        tag_counts = self.dataset.tag_counts  # Counter already in memory
        total_samples = sum(len(b) for b in self.dataset.batches)

        if total_samples == 0:
            self._pos_weights = [None] * self.n_gpus
            return

        # Build count tensor from the Counter — O(num_tags), no I/O
        counts = torch.tensor(
            [float(tag_counts.get(tag, 0)) for tag in self.idx2tag],
            dtype=torch.float32,
        )
        counts     = counts.clamp(min=1.0)
        neg_counts = (total_samples - counts).clamp(min=1.0)
        T = max(self.training_config.pos_weight_temperature, 1e-6)
        pos_weight = (neg_counts / counts).pow(1.0 / T).clamp(max=cap)

        print(f"[TaggerTrainer] pos_weight (T={T}): min={pos_weight.min():.2f}, "
              f"max={pos_weight.max():.2f}, mean={pos_weight.mean():.2f}")

        self._pos_weights = [pos_weight.to(f"cuda:{g}") for g in range(self.n_gpus)]

    @staticmethod
    def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
        if path.endswith((".safetensors", ".sft")):
            sd = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    sd[key] = f.get_tensor(key)
            return sd
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _remap_key(key: str, checkpoint_sd: Dict[str, torch.Tensor]) -> Optional[str]:
        """Try several key remappings to bridge checkpoint ↔ model naming differences.

        Known differences between a saved tagger checkpoint and the live model:
          - Checkpoint saved without the HuggingFace `.model.` wrapper sub-module:
              ckpt:  backbone.layer.N.*
              model: backbone.model.layer.N.*
          - Raw DINOv3 pretrained file (no backbone prefix):
              ckpt:  layer.N.*  /  model.layer.N.*
              model: backbone.model.layer.N.*  /  backbone.layer.N.*
        """
        candidates = [
            key,                                          # 1. exact
            key.replace("backbone.model.", "backbone."),  # 2. drop .model. wrapper
            key.replace("backbone.", "backbone.model."),  # 3. add .model. wrapper
            key[len("backbone."):] if key.startswith("backbone.") else None,  # 4. strip backbone.
            f"backbone.{key}",                            # 5. add backbone.
            key.replace("backbone.model.", ""),           # 6. strip backbone.model.
            f"backbone.model.{key}",                      # 7. add backbone.model.
        ]
        for c in candidates:
            if c and c in checkpoint_sd:
                return c
        return None

    @staticmethod
    def _apply_state_dict(
        model: nn.Module,
        checkpoint_sd: Dict[str, torch.Tensor],
        device: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        model_sd = model.state_dict()
        loaded, mismatched, new = [], [], []

        for key, model_tensor in model_sd.items():
            ckpt_key = TaggerTrainer._remap_key(key, checkpoint_sd)

            if ckpt_key is not None:
                ckpt_tensor = checkpoint_sd[ckpt_key]
                if model_tensor.shape == ckpt_tensor.shape:
                    model_sd[key] = ckpt_tensor.to(device=device, dtype=model_tensor.dtype)
                    loaded.append(key)
                else:
                    mismatched.append(
                        f"{key} (ckpt: {ckpt_key}): "
                        f"model={list(model_tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                    )
            else:
                new.append(key)

        model.load_state_dict(model_sd, strict=False)
        return loaded, mismatched, new

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        self.optimizers = []
        self.schedulers = []
        for gpu_id, model in enumerate(self.models):
            decay    = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "norm" not in n]
            no_decay = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)]
            optimizer = AdamW(
                [{"params": decay, "weight_decay": self.training_config.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.training_config.lr, betas=(0.9, 0.999),
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.05, end_factor=1.0,
                total_iters=self.training_config.warmup_steps,
            )
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _forward_backward_on_gpu(
        self,
        gpu_id: int,
        images_chunk: torch.Tensor,
        labels_chunk: torch.Tensor,
        loss_weights_chunk: torch.Tensor,
    ) -> float:
        device     = f"cuda:{gpu_id}"
        model      = self.models[gpu_id]
        pos_weight = self._pos_weights[gpu_id]

        images       = images_chunk.to(device)
        labels       = labels_chunk.to(device)
        loss_weights = loss_weights_chunk.to(device)

        batch_size      = images.shape[0]
        train_mb        = self.training_config.train_minibatch
        num_minibatches = max(1, batch_size // train_mb)
        total_loss      = 0.0

        criterion = (
            nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            if pos_weight is not None
            else nn.BCEWithLogitsLoss(reduction="none")
        )

        for mb_idx in range(num_minibatches):
            start = mb_idx * train_mb
            end   = min(start + train_mb, batch_size)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images[start:end])  # [mb, num_tags]

            # BCEWithLogitsLoss in fp32 for numerical stability
            per_sample_loss = criterion(logits.float(), labels[start:end].float()).mean(dim=-1)
            loss = (per_sample_loss * loss_weights[start:end]).sum()

            (loss / self.training_config.gradient_accumulation_steps).backward()
            total_loss += loss.item()

        return total_loss

    def _all_reduce_gradients(self):
        param_lists = [
            [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            for model in self.models
        ]
        n_params = len(param_lists[0])
        for i in range(n_params):
            grads = [param_lists[g][i].grad for g in range(self.n_gpus)]
            nccl.all_reduce(grads, op=nccl.SUM)

    def _clip_grads_on_gpu(self, gpu_id: int):
        if self.training_config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.models[gpu_id].parameters(), self.training_config.max_grad_norm)

    def _optimizer_step_on_gpu(self, gpu_id: int):
        self.optimizers[gpu_id].step()
        self.schedulers[gpu_id].step()
        self.optimizers[gpu_id].zero_grad()

    def train_step(self, batch) -> float:
        images, labels, _, loss_weights = batch

        batch_size      = images.shape[0]
        samples_per_gpu = batch_size // self.n_gpus

        lw = torch.tensor(loss_weights, dtype=torch.float32)
        lw = lw / lw.sum()

        def gpu_work(gpu_id):
            s, e = gpu_id * samples_per_gpu, (gpu_id + 1) * samples_per_gpu
            return self._forward_backward_on_gpu(
                gpu_id=gpu_id,
                images_chunk=images[s:e],
                labels_chunk=labels[s:e],
                loss_weights_chunk=lw[s:e],
            )

        return sum(self.executor.map(gpu_work, range(self.n_gpus)))

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        sd = self.model.state_dict()
        if path.endswith(".safetensors"):
            save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[TaggerTrainer] Saved: {path}")

    # ------------------------------------------------------------------
    # Evaluation — top-k precision / recall on a single batch (GPU 0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_batch(self, images: torch.Tensor, labels: torch.Tensor, k: int = 10) -> Dict[str, float]:
        device = "cuda:0"
        model  = self.models[0]
        model.eval()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images.to(device)).float()

        topk_idx = logits.topk(k, dim=-1).indices
        pred = torch.zeros_like(labels.to(device))
        pred.scatter_(1, topk_idx, 1.0)

        tp        = (pred * labels.to(device)).sum(dim=-1)
        precision = (tp / k).mean().item()
        recall    = (tp / labels.to(device).sum(dim=-1).clamp(min=1)).mean().item()

        model.train()
        return {"precision@k": precision, "recall@k": recall, "k": k}

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _create_dataloader(self) -> DataLoader:
        cfg = self.dataloader_config
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            pin_memory=True,
            collate_fn=self.dataset.dummy_collate_fn,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self.executor.shutdown(wait=True)
        if hasattr(self, "logger"):
            self.logger.close()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print("[TaggerTrainer] Starting training...")
        for model in self.models:
            model.train()

        while True:
            self.training_config.master_seed += 1
            torch.manual_seed(self.training_config.master_seed)

            dataloader = self._create_dataloader()
            pbar = tqdm(enumerate(dataloader), total=len(self.dataset), desc="Tagger Training")

            for step, batch_data in pbar:
                batch_data = batch_data[0]

                loss = self.train_step(batch_data)

                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.n_gpus > 1:
                        self._all_reduce_gradients()
                    list(self.executor.map(self._clip_grads_on_gpu, range(self.n_gpus)))
                    list(self.executor.map(self._optimizer_step_on_gpu, range(self.n_gpus)))

                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
                self.logger.log_scalar("loss/bce", loss, self.global_step)
                self.logger.log_scalar("learning_rate", lr, self.global_step)

                if (step + 1) % self.eval_config.eval_every == 0:
                    images, labels, _, _ = batch_data
                    metrics = self.evaluate_batch(images, labels, k=10)
                    print(
                        f"\n[eval step={self.global_step}] "
                        f"precision@10={metrics['precision@k']:.4f}  "
                        f"recall@10={metrics['recall@k']:.4f}"
                    )
                    self.logger.log_scalar("eval/precision@10", metrics["precision@k"], self.global_step)
                    self.logger.log_scalar("eval/recall@10",    metrics["recall@k"],    self.global_step)
                    eval_path = os.path.join(self.eval_config.eval_folder, f"eval_2_{self.global_step}.json")
                    with open(eval_path, "w") as f:
                        json.dump({"step": self.global_step, **metrics}, f, indent=2)

                if (step + 1) % self.training_config.save_every == 0:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    ckpt_path = os.path.join(self.training_config.save_folder, f"{timestamp}.safetensors")
                    self.save_checkpoint(ckpt_path)
                    self.config_data["model"]["checkpoint_path"] = ckpt_path
                    self.config_data["parquet_dataloader"]["offset"] = self.dataloader_config.offset + step
                    self._save_config(os.path.join(self.training_config.save_folder, "training_config.json"))

                self.global_step += 1
                self.dataloader_config.offset += 1

            # End of epoch
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_checkpoint(
                os.path.join(self.training_config.save_folder, f"{timestamp}_epoch_end.safetensors")
            )
            self.dataloader_config.offset = 0
            self.dataset.offset = 0
            self.dataset.resample()


# =============================================================================
# Entry point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT multi-label tagger (multi-GPU)")
    parser.add_argument("--config", type=str, default="training_config_tagger.json")
    args = parser.parse_args()

    trainer = TaggerTrainer(config_path=args.config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
