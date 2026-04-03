"""Binary AR-Diffusion LM Trainer — AR backbone + per-token diffusion head

Architecture:
  - Tokenizer-free: text encoded as ASCII binary vectors (64 chars = 512 bits/token)
  - Causal transformer backbone (teacher-forced, like GPT)
  - Per-token diffusion head conditioned on backbone hidden + timestep
  - CFG dropout during training; CFG at inference
  - Multi-GPU via ThreadPoolExecutor + NCCL all-reduce

Usage:
    python train_binary_lm_ar.py --config training_config_binary_lm_ar.json
"""

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.cuda.nccl as nccl
from torch.utils.data import DataLoader
from torch.optim import AdamW
from safetensors.torch import save_file, safe_open
from tqdm import tqdm

from src.dataloaders.binary_lm_dataloader import BinaryLMDataset, make_dataloader
from src.models.binary_lm.model_ar import (
    BinaryLMAR, BinaryLMARParams,
    text_to_binary_tokens_padded,
    binary_tokens_to_text,
)

try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False


# =============================================================================
# Config Dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    master_seed: int = 42
    gradient_accumulation_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    save_every: int = 500
    save_folder: str = "checkpoints_binary_lm_ar"
    trained_layer_keywords: List[str] = field(default_factory=list)
    do_profiling: bool = False
    profile_steps: int = 2
    profile_json_dump: str = "profiler_binary_lm_ar.json"
    use_aim: bool = False
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = "binary_lm_ar"


@dataclass
class BinaryLMDataloaderConfig:
    data_dir: str = "datasets/codelion/fineweb-edu-1B/data"
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    max_docs: Optional[int] = None   # None = use all documents
    text_column: str = "text"


# =============================================================================
# Shared utilities
# =============================================================================

class TimestepSampler:
    """Samples timesteps with a distribution biased toward the middle."""

    def __init__(self, num_points: int = 1000):
        self.num_points = num_points
        self._x = None
        self._cdf = None

    def _build(self, device: torch.device):
        if self._x is None or self._x.device != device:
            x = torch.linspace(0, 1, self.num_points, device=device)
            probs = (-7.7 * (x - 0.5) ** 2 + 2).clamp(min=0)
            probs /= probs.sum()
            self._x   = x
            self._cdf = torch.cumsum(probs, dim=0)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        self._build(device)
        u = torch.rand(n, device=device)
        idx = torch.searchsorted(self._cdf, u, right=True).clamp(max=self.num_points - 1)
        return self._x[idx]


class ExperimentLogger:
    def __init__(self, config: "TrainingConfig", hparams: Dict[str, Any], save_folder: str = "."):
        self.run = None
        self.csv_file = None
        enabled = AIM_AVAILABLE and config.use_aim and config.aim_path

        if enabled:
            self.run = Run(repo=config.aim_path, experiment=config.aim_experiment_name)
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


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith((".safetensors", ".sft")):
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd
    return torch.load(path, map_location="cpu")


def _rand_init(shape: torch.Size, device: str) -> torch.Tensor:
    t = torch.empty(shape, device=device, dtype=torch.float32)
    if t.dim() > 1:
        nn.init.kaiming_uniform_(t)
    else:
        nn.init.zeros_(t)
    return t


def _save_checkpoint(model: nn.Module, path: str):
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    save_file(sd, path)
    print(f"  Saved: {path}")


def _setup_optimizer(model: nn.Module, training_config: TrainingConfig) -> tuple:
    keywords = training_config.trained_layer_keywords
    trained, frozen = [], 0
    for name, param in model.named_parameters():
        if not keywords or any(kw in name for kw in keywords):
            param.requires_grad = True
            trained.append((name, param))
        else:
            param.requires_grad = False
            frozen += 1
    decay    = [p for n, p in trained if "bias" not in n and "norm" not in n]
    no_decay = [p for n, p in trained if "bias" in n or "norm" in n]
    opt = AdamW(
        [{"params": decay, "weight_decay": training_config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=training_config.lr, betas=(0.9, 0.999),
    )
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.05, end_factor=1.0,
        total_iters=training_config.warmup_steps,
    )
    return opt, sched, len(trained), frozen


def _recompute_buffer(model: nn.Module, key: str, device: str) -> torch.Tensor:
    """Recompute a named buffer by temporarily materialising the submodule on CPU.

    Used during meta-device init: buffers like RoPE freqs_cos/freqs_sin are
    deterministic from the model config and must not be random-initialised.
    We re-instantiate just the owning submodule on CPU to get the correct value,
    then move it to the target device.
    """
    parts  = key.split(".")
    # Walk to the owning module
    owner = model
    for part in parts[:-1]:
        owner = getattr(owner, part)
    attr_name = parts[-1]

    # The owner module is on meta; call _apply to move it to CPU so __init__
    # buffers are recomputed. We only need the one buffer value.
    # Simplest approach: re-create the owner module on CPU with same args.
    # For RotaryEmbedding we can just recompute directly from its stored attrs.
    if hasattr(owner, "freqs_cos") or hasattr(owner, "freqs_sin"):
        # RotaryEmbedding: recompute from inv_freq logic
        # We can reconstruct from the meta tensor shape
        buf_meta = getattr(owner, attr_name)   # meta tensor
        # freqs_cos/sin shape: [max_seq_len, head_dim//2]
        max_seq_len, half = buf_meta.shape
        # Recover head_dim and theta from the existing (meta) inv_freq if present,
        # otherwise recompute. Since we have the shape we can recompute generically.
        # inv_freq = 1 / (theta ** (arange(half) / half))  with theta=10000
        # We don't store theta, but we can read it from the buffer values if
        # materialised — instead just recompute with default theta=10000.
        # This is safe because theta is always 10000 in BinaryLMARParams.
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        value = freqs.cos() if attr_name == "freqs_cos" else freqs.sin()
        return value.to(device)

    # Fallback for any other buffer: zeros on the correct device
    buf_meta = getattr(owner, attr_name)
    return torch.zeros(buf_meta.shape, dtype=torch.float32, device=device)


def _all_reduce_gradients(models: list, n_gpus: int):
    param_lists = [
        [p for p in m.parameters() if p.requires_grad and p.grad is not None]
        for m in models
    ]
    for i in range(len(param_lists[0])):
        grads = [param_lists[g][i].grad for g in range(n_gpus)]
        nccl.all_reduce(grads, op=nccl.SUM)


# =============================================================================
# Config
# =============================================================================

@dataclass
class ModelConfig:
    checkpoint_path: str = ""
    max_seq_len: int = 256
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    ffn_mult: float = 4.0
    head_channels: int = 768
    head_num_res_blocks: int = 4
    adaln_embed_dim: int = 768
    cfg_dropout: float = 0.1
    use_x0: bool = True


@dataclass
class InferenceConfig:
    inference_every: int = 200
    inference_folder: str = "inference_binary_lm_ar"
    prompts: List[str] = field(default_factory=lambda: [
        "a beautiful sunset",
        "a cat sitting on",
    ])
    steps: int = 32
    cfg_scale: float = 3.0
    max_gen_tokens: int = 8


# =============================================================================
# Trainer
# =============================================================================

class BinaryLMARTrainer:
    """Multi-GPU trainer for BinaryLMAR (AR backbone + diffusion head)."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.global_step = 0

        with open(config_path) as f:
            self.config_data = json.load(f)

        self._parse_configs()

        self.n_gpus = torch.cuda.device_count()
        assert self.n_gpus >= 1, "No CUDA GPUs found"

        self.models: List[BinaryLMAR] = []
        self.optimizers = []
        self.schedulers = []
        self.timestep_samplers: List[TimestepSampler] = []
        self.executor: ThreadPoolExecutor = None   # type: ignore[assignment]
        self.dataset: BinaryLMDataset = None        # type: ignore[assignment]
        self.dataloader: DataLoader = None          # type: ignore[assignment]
        self.logger: ExperimentLogger = None        # type: ignore[assignment]

    def _parse_configs(self):
        self.training_config   = TrainingConfig(**self.config_data.get("training", {}))
        self.model_config      = ModelConfig(**self.config_data.get("model", {}))
        self.inference_config  = InferenceConfig(**self.config_data.get("inference", {}))
        self.dataloader_config = BinaryLMDataloaderConfig(**self.config_data.get("dataloader", {}))

    def _save_config(self, path: str):
        with open(path, "w") as f:
            json.dump(self.config_data, f, indent=4)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        print(f"[BinaryLMAR] Setting up on {self.n_gpus} GPU(s)")
        torch.manual_seed(self.training_config.master_seed)
        random.seed(self.training_config.master_seed)

        os.makedirs(self.training_config.save_folder, exist_ok=True)
        os.makedirs(self.inference_config.inference_folder, exist_ok=True)
        self._save_config(f"{self.training_config.save_folder}/training_config.json")

        self._load_models()
        self._setup_optimizer()
        self._setup_dataset()
        self.logger  = ExperimentLogger(
            self.training_config, self.config_data,
            save_folder=self.training_config.save_folder,
        )
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)
        print("[BinaryLMAR] Setup complete!")

    def _setup_dataset(self):
        cfg = self.dataloader_config
        mc  = self.model_config
        self.dataset, self.dataloader = make_dataloader(
            data_dir       = cfg.data_dir,
            seq_len        = mc.max_seq_len,
            batch_size     = cfg.batch_size,
            num_workers    = cfg.num_workers,
            prefetch_factor= cfg.prefetch_factor,
            seed           = self.training_config.master_seed,
            rank           = 0,
            num_gpus       = 1,
            max_docs       = cfg.max_docs,
            text_column    = cfg.text_column,
        )
        print(f"  Dataset: {len(self.dataset):,} sequences of {mc.max_seq_len} tokens")

    def _build_params(self) -> BinaryLMARParams:
        mc = self.model_config
        return BinaryLMARParams(
            max_seq_len=mc.max_seq_len, hidden_dim=mc.hidden_dim,
            n_layers=mc.n_layers, n_heads=mc.n_heads, n_kv_heads=mc.n_kv_heads,
            ffn_mult=mc.ffn_mult, head_channels=mc.head_channels,
            head_num_res_blocks=mc.head_num_res_blocks,
            adaln_embed_dim=mc.adaln_embed_dim, cfg_dropout=mc.cfg_dropout, use_x0=mc.use_x0,
        )

    def _load_models(self):
        print("  Loading BinaryLMAR...")
        params = self._build_params()

        checkpoint_sd = None
        if self.model_config.checkpoint_path:
            print(f"    Checkpoint: {self.model_config.checkpoint_path}")
            checkpoint_sd = _load_state_dict(self.model_config.checkpoint_path)

        self.models = []
        for gpu_id in range(self.n_gpus):
            device = f"cuda:{gpu_id}"
            with torch.device("meta"):
                model = BinaryLMAR(params)

            model_sd = model.state_dict()
            loaded, new, mismatch = [], [], []

            # Collect the set of parameter names (vs buffer names).
            # Buffers like RoPE freqs_cos/freqs_sin must be recomputed from the
            # live model, not random-initialized — they are deterministic from
            # the model config and would be silently corrupted by _rand_init.
            param_names = {n for n, _ in model.named_parameters()}

            def _materialize(key: str, meta_t: torch.Tensor) -> torch.Tensor:
                if key not in param_names:
                    # Buffer: get the correctly computed value from the live model
                    # by looking it up via the module hierarchy.
                    parts  = key.split(".")
                    obj    = model
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    buf = getattr(obj, parts[-1])
                    # buf is still on meta; recompute by re-running __init__ logic
                    # is complex — instead just copy the shape/dtype and let the
                    # model recompute on first forward. We materialise with zeros
                    # so it's on the right device; the RotaryEmbedding.forward()
                    # slices these, so we need the real values. Re-register them.
                    return _recompute_buffer(model, key, device)
                return _rand_init(meta_t.shape, device)

            if checkpoint_sd is not None:
                for key, meta_t in model_sd.items():
                    if key in checkpoint_sd:
                        ckpt_t = checkpoint_sd[key]
                        if meta_t.shape == ckpt_t.shape:
                            model_sd[key] = ckpt_t.to(device=device, dtype=torch.float32)
                            loaded.append(key)
                        else:
                            model_sd[key] = _materialize(key, meta_t)
                            mismatch.append(key)
                    else:
                        model_sd[key] = _materialize(key, meta_t)
                        new.append(key)
            else:
                for key, meta_t in model_sd.items():
                    model_sd[key] = _materialize(key, meta_t)

            model.load_state_dict(model_sd, assign=True)
            self.models.append(model)

            if gpu_id == 0:
                total      = sum(p.numel() for p in model.parameters())
                backbone_p = sum(p.numel() for n, p in model.named_parameters() if "backbone" in n or "x_embedder" in n or "bos" in n)
                head_p     = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
                print(f"    Total: {total:,}  Backbone: {backbone_p:,}  Head: {head_p:,}")
                if checkpoint_sd:
                    print(f"    Loaded {len(loaded)}, new {len(new)}, mismatch {len(mismatch)}")

        self.timestep_samplers = [TimestepSampler() for _ in range(self.n_gpus)]
        print(f"  BinaryLMAR ready on {self.n_gpus} GPU(s)")

    def _setup_optimizer(self):
        self.optimizers, self.schedulers = [], []
        for gpu_id, model in enumerate(self.models):
            opt, sched, n_trained, n_frozen = _setup_optimizer(model, self.training_config)
            self.optimizers.append(opt)
            self.schedulers.append(sched)
            if gpu_id == 0:
                print(f"  Training {n_trained} param groups, {n_frozen} frozen")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _forward_backward_on_gpu(
        self,
        gpu_id: int,
        x0: torch.Tensor,       # [B, T, 512]  already on CPU, {-1,+1}
        mask: torch.Tensor,     # [B, T]  bool
    ) -> float:
        device  = f"cuda:{gpu_id}"
        model   = self.models[gpu_id]
        sampler = self.timestep_samplers[gpu_id]
        mc      = self.model_config

        x0   = x0.to(device)
        mask = mask.to(device)
        B, T, C = x0.shape

        # Per-token timestep (all valid — packed dataset has no padding)
        t = sampler.sample(B * T, torch.device(device)).reshape(B, T)

        # CFG dropout mask
        cfg_mask = torch.rand(B, T, device=device) < mc.cfg_dropout

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_v = model.forward_train(x0, t, mask, cfg_mask)

            # Target velocity
            noise   = torch.randn_like(x0)
            t_exp   = t.unsqueeze(-1)
            noisy_x = x0 * (1.0 - t_exp) + noise * t_exp
            eps     = 5e-2
            target  = (noisy_x - x0) / (t_exp + eps)

            # Mean MSE over all tokens (no padding in packed dataset)
            loss = ((pred_v - target) ** 2).mean()

        (loss / self.training_config.gradient_accumulation_steps).backward()
        return loss.item()

    def _clip_grads(self, gpu_id: int):
        if self.training_config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.models[gpu_id].parameters(), self.training_config.max_grad_norm)

    def _optimizer_step(self, gpu_id: int):
        self.optimizers[gpu_id].step()
        self.schedulers[gpu_id].step()
        self.optimizers[gpu_id].zero_grad()

    def train_step(self, batch: dict) -> float:
        """batch = {"tokens": [B, T, 512], "mask": [B, T]}"""
        tokens = batch["tokens"]   # [B, T, 512]  {-1,+1}
        mask   = batch["mask"]     # [B, T]
        B = tokens.shape[0]
        samples_per_gpu = max(1, B // self.n_gpus)

        def _work(gpu_id):
            s = gpu_id * samples_per_gpu
            e = s + samples_per_gpu
            return self._forward_backward_on_gpu(gpu_id, tokens[s:e], mask[s:e])

        return sum(self.executor.map(_work, range(self.n_gpus)))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_inference(self, prefix_texts: Optional[List[str]] = None) -> List[str]:
        if prefix_texts is None:
            prefix_texts = self.inference_config.prompts

        device    = "cuda:0"
        model     = self.models[0]
        mc        = self.model_config
        ic        = self.inference_config

        model.eval()
        results = []

        for prefix in prefix_texts:
            # Pad prefix to 64-byte boundary so tokens are always aligned
            # (matches the random alignment offset injected during training)
            prefix_tokens, prefix_mask = text_to_binary_tokens_padded([prefix], mc.max_seq_len, torch.device(device))
            prefix_len = int(prefix_mask[0].sum().item())
            tokens = prefix_tokens[:, :prefix_len, :]   # [1, prefix_len, 512]

            for _ in range(ic.max_gen_tokens):
                if tokens.shape[1] >= mc.max_seq_len:
                    break

                cur_len = tokens.shape[1]
                pad     = torch.ones(1, cur_len, dtype=torch.bool, device=device)
                hidden  = model._run_backbone(tokens, pad)
                h_last  = hidden[:, -1, :]
                null_h  = torch.zeros_like(h_last)

                x = torch.randn(1, 512, device=device)
                t_schedule = torch.linspace(1.0, 0.0, ic.steps + 1, device=device)

                for i in range(ic.steps):
                    t_cur  = t_schedule[i].unsqueeze(0)
                    t_next = t_schedule[i + 1].unsqueeze(0)
                    dt     = t_next - t_cur

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        x0_cond = model.forward_head_only(x, h_last, t_cur)
                        x0_null = model.forward_head_only(x, null_h, t_cur)

                    x0_cfg = x0_null + ic.cfg_scale * (x0_cond.float() - x0_null.float())
                    t_exp  = t_cur.unsqueeze(-1).clamp(min=1e-6)
                    v      = (x - x0_cfg) / t_exp
                    x      = x + v * dt

                # Round to {-1,+1}: positive -> +1, non-positive -> -1
                new_token = torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
                tokens = torch.cat([tokens, new_token.unsqueeze(1)], dim=1)

            # stop_at_stop_byte=True: truncate at first 0xFF byte
            results.append(binary_tokens_to_text(tokens, stop_at_stop_byte=True)[0])

        model.train()
        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        _save_checkpoint(self.models[0], path)

    def cleanup(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.logger:
            self.logger.close()

    def train(self):
        print("[BinaryLMAR] Starting training...")
        for m in self.models:
            m.train()

        epoch = 0
        while True:
            self.training_config.master_seed += 1
            torch.manual_seed(self.training_config.master_seed)

            pbar = tqdm(enumerate(self.dataloader), total=len(self.dataset) // self.dataloader_config.batch_size, desc=f"BinaryLMAR e{epoch}")

            for step, batch_data in pbar:
                loss = self.train_step(batch_data)

                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.n_gpus > 1:
                        _all_reduce_gradients(self.models, self.n_gpus)
                    list(self.executor.map(self._clip_grads,     range(self.n_gpus)))
                    list(self.executor.map(self._optimizer_step, range(self.n_gpus)))

                lr = self.schedulers[0].get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
                self.logger.log_scalar("loss/total", loss, self.global_step)
                self.logger.log_scalar("learning_rate", lr, self.global_step)

                if (step + 1) % self.training_config.save_every == 0:
                    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    path = f"{self.training_config.save_folder}/{ts}.safetensors"
                    self.save_checkpoint(path)
                    self.config_data["model"]["checkpoint_path"] = path
                    self._save_config(f"{self.training_config.save_folder}/training_config.json")

                if (step + 1) % self.inference_config.inference_every == 0:
                    print("\n[Inference] Generating...")
                    decoded  = self.run_inference()
                    out_path = f"{self.inference_config.inference_folder}/{self.global_step}.txt"
                    with open(out_path, "w") as f:
                        for prefix, dec in zip(self.inference_config.prompts, decoded):
                            f.write(f"PREFIX : {prefix}\nDECODED: {dec}\n{'-'*60}\n")
                    print(f"  {out_path}")
                    if decoded:
                        print(f"  {decoded[0][:120]!r}")

                self.global_step += 1

            # End of epoch: resample with new seed for fresh shuffle
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_checkpoint(f"{self.training_config.save_folder}/{ts}_epoch_end.safetensors")
            self.dataset.resample()
            epoch += 1


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train BinaryLM AR (AR backbone + diffusion head)")
    parser.add_argument("--config", type=str, default="training_config_binary_lm_ar.json")
    args = parser.parse_args()

    trainer = BinaryLMARTrainer(config_path=args.config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
