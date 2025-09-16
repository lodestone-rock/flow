import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Iterable, List, Dict, Any, Tuple

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from torchastic import StochasticAccumulator
from torchastic.stochastic_optim import copy_stochastic_, Optimizer
import random

from transformers import T5Tokenizer
import wandb
import shutil
from torchvision import transforms


from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model_dct import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import prepare_latent_image_ids
from src.math_utils import cosine_optimal_transport
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant

from huggingface_hub import HfApi, upload_file
import time

from aim import Run, Image as AimImage
from datetime import datetime
from PIL import Image as PILImage

@dataclass
class TrainingConfig:
    master_seed: int
    cache_minibatch: int
    train_minibatch: int
    updates_per_large_batch: int
    offload_param_count: int
    lr: float
    weight_decay: float
    warmup_steps: int
    change_layer_every: int
    trained_single_blocks: int
    trained_double_blocks: int
    save_every: int
    save_folder: str
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None
    aim_hash: Optional[str] = None
    aim_steps: Optional[int] = 0
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class InferenceConfig:
    inference_every: int
    inference_folder: str
    steps: int
    guidance: int
    cfg: int
    prompts: list[str]
    first_n_steps_wo_cfg: int
    image_dim: tuple[int, int]
    t5_max_length: int


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    shuffle_tags: bool
    tag_drop_percentage: float
    uncond_percentage: float
    resolution_step: int
    num_workers: int
    prefetch_factor: int
    ratio_cutoff: float
    thread_per_worker: int
    offset: int


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int


def create_zero_param_groups(param_groups: List[Dict[str, Any]], rank: int, world_size: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Create parameter groups for ZeRO-1 optimizer sharding and generate a map
    of parameter owners for broadcasting.
    
    Args:
        param_groups: Original parameter groups (list of dicts with 'params' key)
        rank: Current process rank
        world_size: Total number of processes
    
    Returns:
        A tuple containing:
        - List of parameter groups containing only parameters owned by this rank.
        - A list where the index corresponds to a parameter's global index and
          the value is the rank of the process that owns it. This is the pre-computed
          index for the broadcast function.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    
    sharded_groups = []
    owner_ranks = []
    global_param_idx = 0
    
    for group in param_groups:
        # Copy all group settings except params
        sharded_group = {k: v for k, v in group.items() if k != 'params'}
        sharded_group['params'] = []
        
        # Add only parameters owned by this rank
        for param in group['params']:
            owner_rank = global_param_idx % world_size
            owner_ranks.append(owner_rank)
            
            if owner_rank == rank:
                sharded_group['params'].append(param)
            global_param_idx += 1
        
        # Only add group if it has parameters for this rank
        if sharded_group['params']:
            sharded_groups.append(sharded_group)
    
    return sharded_groups, owner_ranks


def broadcast_zero_params(all_params: List[torch.Tensor], owner_ranks: List[int]):
    """
    Broadcast updated parameters from their owning ranks to all other ranks
    using a pre-computed index of owner ranks.
    
    Args:
        all_params: List of all model parameters (in the same order across all ranks)
        owner_ranks: A list mapping each parameter's global index to its owner rank.
    """
    with torch.no_grad():
        for i, param in enumerate(all_params):
            owner_rank = owner_ranks[i]
            dist.broadcast(param.data, src=owner_rank)


class AdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        chunk_size (int):
            Number of parameters to process before synchronizing.
            A larger chunk size can improve performance but uses more
            temporary GPU memory. (default: 16)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        chunk_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(AdamW, self).__init__(params, defaults)
        
        self.chunk_size = chunk_size

        # Initialize state in pinned memory for faster async transfers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16, device='cpu').pin_memory()
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16, device='cpu').pin_memory()


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Enumerate to keep track of the parameter index for chunking
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]
                device = p.device

                # Lazy state initialization
                if not state:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16, device='cpu').pin_memory()
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16, device='cpu').pin_memory()

                # ========= Asynchronously queue all operations for this parameter =========
                
                # 1. Queue Host-to-Device copy
                ema_gpu = state["ema"].to(device, non_blocking=True)
                ema_squared_gpu = state["ema_squared"].to(device, non_blocking=True)

                # 2. Queue computations on the GPU
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema_fp32 = ema_gpu.to(torch.float32)
                ema_squared_fp32 = ema_squared_gpu.to(torch.float32)

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                ema_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                ema_squared_fp32.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (ema_squared_fp32.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                p_fp32.data.addcdiv_(ema_fp32, denom, value=-step_size)

                copy_stochastic_(p, p_fp32)
                copy_stochastic_(ema_gpu, ema_fp32)
                copy_stochastic_(ema_squared_gpu, ema_squared_fp32)
                
                # 3. Queue Device-to-Host copy
                state["ema"].copy_(ema_gpu, non_blocking=True)
                state["ema_squared"].copy_(ema_squared_gpu, non_blocking=True)
                
                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss
    
def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def create_distribution(num_points, device=None):
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


# Upload the model to Hugging Face Hub
def upload_to_hf(model_filename, path_in_repo, repo_id, token, max_retries=3):
    api = HfApi()

    for attempt in range(max_retries):
        try:
            upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"Model uploaded to {repo_id}/{path_in_repo}")
            return  # Exit function if successful

        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    print("Upload failed after multiple attempts.")


def sample_from_distribution(x, probabilities, num_samples, device=None):
    # Step 1: Compute the cumulative distribution function
    cdf = torch.cumsum(probabilities, dim=0)

    # Step 2: Generate uniform random samples
    uniform_samples = torch.rand(num_samples, device=device)

    # Step 3: Map uniform samples to the x values using the CDF
    indices = torch.searchsorted(cdf, uniform_samples, right=True)

    # Get the corresponding x values for the sampled indices
    sampled_values = x[indices]

    return sampled_values


def prepare_sot_pairings(images):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    images = images.to(torch.float32)
    n, c, h, w = images.shape
    image_pos_id = prepare_latent_image_ids(n, h, w, patch_size=16)

    # randomize ode timesteps
    # input_timestep = torch.round(
    #     F.sigmoid(torch.randn((n,), device=images.device)), decimals=3
    # )
    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=images.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=images.device
    )

    # biasing towards earlier more noisy steps where it's the most uncertain
    # input_timestep = time_shift(0.5, 1, input_timestep)

    timesteps = input_timestep[:, None, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(images)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        images.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_images = images * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    target = noise - images

    return noisy_images, target, input_timestep, image_pos_id, images.shape


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps, rank, world_size):
    # TODO: pack this into a function
    trained_params = []
    broadcast_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
            broadcast_params.append(param)
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)
    # init optimizer
    param_groups = [
        {
            "params": [
                param
                for name, param in trained_params
                if ("bias" not in name and "norm" not in name)
            ]
        },
        {
            "params": [
                param
                for name, param in trained_params
                if ("bias" in name or "norm" in name)
            ],
            "weight_decay": 0.0,
        },
    ]
    # optimizer only optimize the local shard the we broadcast it later
    local_shard_param_group, owner_ranks = create_zero_param_groups(param_groups, rank, world_size)
    optimizer = AdamW(
        params=local_shard_param_group,
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, hooks, trained_params, owner_ranks, broadcast_params


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def optimizer_state_to(optimizer, device):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            # Check if the item is a tensor
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def save_part(model, trained_layer_keywords, counter, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(
        filtered_state_dict, os.path.join(save_folder, f"trained_part_{counter}.pth")
    )


def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)


def save_config_to_json(filepath: str, **configs):
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


@torch.no_grad()
def inference_wrapper(
    model,
    t5_tokenizer,
    t5,
    seed: int,
    steps: int,
    guidance: int,
    cfg: int,
    prompts: list,
    rank: int,
    first_n_steps_wo_cfg: int,
    image_dim=(512, 512),
    t5_max_length=512,
):
    #############################################################################
    # test inference
    # aliasing
    SEED = seed
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    GUIDANCE = guidance
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg
    # DEVICE = model.device
    PROMPT = prompts

    T5_MAX_LENGTH = t5_max_length

    # store device state of each model
    # t5_device = t5.device
    # model_device = model.device
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # init random noise
            noise = torch.randn(
                [len(PROMPT), 3, WIDTH, HEIGHT], 
                device=rank, 
                dtype=torch.bfloat16, 
                generator=torch.Generator(device=rank).manual_seed(seed)
            )
            n, c, h, w = noise.shape
            image_pos_id = prepare_latent_image_ids(n, h, w, patch_size=16).to(rank)

            timesteps = get_schedule(STEPS, noise.shape[1])

            model.to("cpu")
            t5.to(rank)  # load t5 to gpu
            text_inputs = t5_tokenizer(
                PROMPT,
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)

            t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to(rank)

            text_inputs_neg = t5_tokenizer(
                [""]*len(PROMPT),
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)

            t5_embed_neg = t5(text_inputs_neg.input_ids, text_inputs_neg.attention_mask).to(
                rank
            )

            text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)
            neg_text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)

            # t5.to("cpu")
            model.to(rank)  # load model to gpu
            output_image = denoise_cfg(
                model,
                noise,
                image_pos_id,
                t5_embed,
                t5_embed_neg,
                text_ids,
                neg_text_ids,
                text_inputs.attention_mask,
                text_inputs_neg.attention_mask,
                timesteps,
                GUIDANCE,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
            )

            # restore back state
            model.to("cpu")
            # t5.to("cpu")
    del noise
    del image_pos_id
    del timesteps
    del text_inputs
    del t5_embed
    del text_inputs_neg
    del t5_embed_neg
    del text_ids
    del neg_text_ids

    return output_image


def train_chroma(rank, world_size, debug=False, json_config="training_config.json"):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json(json_config)

    training_config = TrainingConfig(**config_data["training"])
    inference_config = InferenceConfig(**config_data["inference"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    extra_inference_config = [
        InferenceConfig(**conf) for conf in config_data["extra_inference_config"]
    ]


    # Setup Aim run
    if training_config.aim_path is not None and rank == 0:
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = Run(repo=training_config.aim_path, run_hash=training_config.aim_hash, experiment=training_config.aim_experiment_name, force_resume=True)

        hparams = config_data.copy()
        hparams["training"]['aim_path'] = None
        run["hparams"] = hparams


    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )

    os.makedirs(inference_config.inference_folder, exist_ok=True)
    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        chroma_params._use_compiled = True
        with torch.device("meta"):
            model = Chroma(chroma_params)

        # Check file extension to determine loading method
        if model_config.chroma_path.endswith('.safetensors') or model_config.chroma_path.endswith('.sft'):
            model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)
        else:  # Assume PyTorch format (.pth)
            model.load_state_dict(torch.load(model_config.chroma_path, map_location="cpu"), assign=True)

        # model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)

        # randomly train inner layers at a time
        trained_double_blocks = list(range(len(model.double_blocks)))
        trained_single_blocks = list(range(len(model.single_blocks)))
        random.shuffle(trained_double_blocks)
        random.shuffle(trained_single_blocks)
        # lazy :P
        trained_double_blocks = trained_double_blocks * 1000000
        trained_single_blocks = trained_single_blocks * 1000000

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(
            replace_keys(load_file_multipart(model_config.t5_path)), assign=True
        )
        t5.eval()
        t5.to(torch.bfloat16)
        if model_config.t5_to_8bit:
            cast_linear(t5, torch.float8_e4m3fn)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        # don't use this tag implication pruning it's slow!
        # preprocess the jsonl tags before training!
        # tag_implication_path="tag_implications.csv",
        base_res=dataloader_config.base_resolution,
        shuffle_tags=dataloader_config.shuffle_tags,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
        offset=dataloader_config.offset
    )

    optimizer = None
    scheduler = None
    hooks = []
    optimizer_counter = 0

    global_step = training_config.aim_steps
    while True:
        training_config.master_seed += 1
        torch.manual_seed(training_config.master_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is handled in the dataset
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor,
            pin_memory=True,
            collate_fn=dataset.dummy_collate_fn,
        )
        for counter, data in tqdm(
            enumerate(dataloader),
            total=len(dataset),
            desc=f"training, Rank {rank}",
            position=rank,
        ):
            images, caption, index, loss_weighting = data[0]
            # just in case the dataloader is failing
            caption = [x if x is not None else "" for x in caption]
            caption = [x.lower() if torch.rand(1).item() < 0.25 else x for x in caption]
            loss_weighting = torch.tensor(loss_weighting, device=rank)
            if counter % training_config.change_layer_every == 0:
                # periodically remove the optimizer and swap it with new one

                # aliasing to make it cleaner
                o_c = optimizer_counter
                n_ls = training_config.trained_single_blocks
                n_ld = training_config.trained_double_blocks
                trained_layer_keywords = (
                    [
                        f"double_blocks.{x}."
                        for x in trained_double_blocks[o_c * n_ld : o_c * n_ld + n_ld]
                    ]
                    + [
                        f"single_blocks.{x}."
                        for x in trained_single_blocks[o_c * n_ls : o_c * n_ls + n_ls]
                    ]
                    # + ["txt_in", "img_in_patch", "nerf_final_layer", "nerf_blocks"]
                    # train nerf only for now
                    + ["img_in_patch", "nerf_final_layer", "nerf_blocks"]
                )

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                optimizer, scheduler, hooks, trained_params, owner_ranks, broadcast_params = init_optimizer(
                    model,
                    trained_layer_keywords,
                    training_config.lr,
                    training_config.weight_decay,
                    training_config.warmup_steps,
                    rank,
                    world_size
                )

                optimizer_counter += 1

            # MODIFICATION START: The caching loop has been removed.
            # We no longer pre-compute and store embeddings.

            # The full-size image tensor remains in CPU memory
            acc_images = images
            
            # process the full batch now!
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                # prepare flat image and the target lerp
                (
                    noisy_images,
                    target,
                    input_timestep,
                    image_pos_id,
                    latent_shape,
                ) = prepare_sot_pairings(acc_images.to(rank))
                noisy_images = noisy_images.to(torch.bfloat16)
                target = target.to(torch.bfloat16)
                input_timestep = input_timestep.to(torch.bfloat16)
                image_pos_id = image_pos_id.to(rank)

                # t5 text id for the model
                text_ids = torch.zeros((noisy_images.shape[0], 512, 3), device=rank)
                # NOTE:
                # using static guidance 1 for now
                # this should be disabled later on !
                static_guidance = torch.tensor(
                    [0.0] * acc_images.shape[0], device=rank
                )

            # set the input to requires grad to make autograd works
            noisy_images.requires_grad_(True)
            # acc_embeddings no longer exists, so this is removed.
            # acc_embeddings.requires_grad_(True)
            ot_bs = acc_images.shape[0]

            # aliasing
            mb = training_config.train_minibatch
            
            # MODIFICATION START
            # Calculate how many minibatches to process for each parameter update
            local_num_minibatches = dataloader_config.batch_size // mb // world_size
            updates_per_large_batch = max(1, training_config.updates_per_large_batch)
            minibatches_per_update = max(1, local_num_minibatches // updates_per_large_batch)

            # move model to device before the minibatching loop
            model.to(rank)
            torch.cuda.empty_cache()

            loss_log = []
            for tmb_i in tqdm(
                range(local_num_minibatches),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                # MODIFICATION START: T5 computation is now inside the minibatch loop
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 1. Move T5 to GPU for this minibatch
                    t5.to(rank)

                    # 2. Slice captions for the current minibatch
                    minibatch_captions = caption[tmb_i * mb : tmb_i * mb + mb]

                    # 3. Tokenize and compute embeddings on the fly
                    text_inputs = t5_tokenizer(
                        minibatch_captions,
                        padding="max_length",
                        max_length=model_config.t5_max_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).to(rank)
                    
                    minibatch_embeddings = t5(text_inputs.input_ids, text_inputs.attention_mask)
                    minibatch_mask = text_inputs.attention_mask

                    # 4. Move T5 back to CPU to free VRAM for the main model
                    # t5.to("cpu")
                    torch.cuda.empty_cache()
                # MODIFICATION END

                # do this inside for loops!
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        img=noisy_images[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt=minibatch_embeddings,
                        txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_mask=minibatch_mask,
                        timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                    )
                    # TODO: need to scale the loss with rank count and grad accum!

                    # Compute per-element squared error and mean over sequence and feature dims
                    loss = ((pred - target[tmb_i * mb : tmb_i * mb + mb]) ** 2).mean(dim=(1, 2, 3))  # Shape: [mb]

                    # Normalize per full batch
                    loss = loss / (dataloader_config.batch_size // mb)  # Shape: [mb]

                    # Apply per-sample weight
                    weights = loss_weighting[tmb_i * mb : tmb_i * mb + mb]  # Shape: [mb]

                    # Normalize weights to ensure the overall loss scale is consistent
                    weights = weights / weights.sum()

                    # Compute final weighted loss
                    loss = (loss * weights).sum()

                    # correct!
                    # loss = F.mse_loss(
                    #     pred,
                    #     target[tmb_i * mb : tmb_i * mb + mb],
                    # ) / (dataloader_config.batch_size // mb)
                torch.cuda.empty_cache()
                # loss.backward()
                # loss_log.append(
                #     loss.detach().clone() * (dataloader_config.batch_size // mb)
                # )
                loss.backward()
                loss_log.append(
                    loss.item() * (dataloader_config.batch_size // mb)
                )

                # MODIFICATION START
                # Check if it's time for a parameter update
                is_update_step = (tmb_i + 1) % minibatches_per_update == 0
                is_last_minibatch = (tmb_i + 1) == local_num_minibatches

                if is_update_step or is_last_minibatch:
                    print("param updated")
                    torch.cuda.empty_cache()
                    # offload_param_count = 0
                    # for name, param in model.named_parameters():
                    #     if not any(keyword in name for keyword in trained_layer_keywords):
                    #         if offload_param_count < training_config.offload_param_count:
                    #             offload_param_count += param.numel()
                    #             param.data = param.data.to("cpu", non_blocking=True)
                    # optimizer_state_to(optimizer, rank)
                    StochasticAccumulator.reassign_grad_buffer(model)

                    if not debug:
                        synchronize_gradients(model)
                        torch.cuda.empty_cache()
                    scheduler.step()
                    optimizer.step()
                    broadcast_zero_params(broadcast_params, owner_ranks)
                    model.zero_grad()
                    # optimizer.zero_grad()

                    # optimizer_state_to(optimizer, "cpu")
                    model.to(rank)
                    torch.cuda.empty_cache()
                # MODIFICATION END

            # loss_log = sum(loss_log) / len(loss_log)
            # offload some params to cpu just enough to make room for the caching process
            # and only offload non trainable params
            del noisy_images, target, input_timestep, image_pos_id, acc_images

            
            # MODIFICATION START
            # The original update block was here and has been moved inside the loop.
            # MODIFICATION END

            if rank == 0:
                with torch.no_grad():
                    # run.track(loss_log, name='loss', step=global_step)
                    # The aggregation now works with Python floats, no tensor involved
                    final_loss_value = sum(loss_log) / len(loss_log)
                    # if you need it as a tensor for tracking:
                    loss_log_tensor = torch.tensor(final_loss_value, device=rank)
                    run.track(loss_log_tensor, name='loss', step=global_step)
                    run.track(training_config.lr, name='learning_rate', step=global_step)

            dataloader_config.offset += 1
            
            if (counter + 1) % training_config.save_every == 0 and rank == 0:
                model.to("cpu")
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(
                    model.state_dict(),
                    model_filename,
                )
                config_data["dataloader"]["offset"] = dataloader_config.offset
                config_data["model"]["chroma_path"] = model_filename
                dump_dict_to_json(
                    config_data, f"{training_config.save_folder}/training_config.json"
                )

                if training_config.hf_token:
                    upload_to_hf(
                        model_filename,
                        model_filename,
                        training_config.hf_repo_id,
                        training_config.hf_token,
                    )
                torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            if (counter + 1) % inference_config.inference_every == 0:
                # Part 1: Each rank generates and saves its own images and prompts to temporary files.
                # A temporary subdirectory is used to avoid clutter and simplify cleanup.
                temp_inference_folder = os.path.join(inference_config.inference_folder, f"step_{counter}_temp")
                os.makedirs(temp_inference_folder, exist_ok=True)

                # Each rank gets its own unique prompt from its current batch data
                preview_prompts_this_rank = inference_config.prompts + caption[:1]
                
                # Combine the main inference config with any extra ones
                all_inference_configs = [inference_config] + extra_inference_config

                # MODIFICATION START: The for loop is now parallelized
                for config_idx, current_config in enumerate(all_inference_configs):
                    # Generate all images for the current config in a single batch
                    images_tensor = inference_wrapper(
                        model=model,
                        t5_tokenizer=t5_tokenizer,
                        t5=t5,
                        seed=training_config.master_seed + rank,  # Seed ensures different images per rank
                        steps=current_config.steps,
                        guidance=current_config.guidance,
                        cfg=current_config.cfg,
                        prompts=preview_prompts_this_rank,  # Pass the entire list of prompts
                        rank=rank,
                        first_n_steps_wo_cfg=current_config.first_n_steps_wo_cfg,
                        image_dim=current_config.image_dim,
                        t5_max_length=current_config.t5_max_length,
                    )

                    # Loop through the generated images and prompts to save them individually
                    for prompt_idx, prompt in enumerate(preview_prompts_this_rank):
                        # Define unique filenames for the image and its prompt
                        base_filename = f"config{config_idx}_prompt{prompt_idx}_rank{rank}"
                        img_path = os.path.join(temp_inference_folder, f"{base_filename}.jpg")
                        prompt_path = os.path.join(temp_inference_folder, f"{base_filename}.txt")

                        # Save the specific image from the tensor batch
                        save_image(images_tensor[prompt_idx:prompt_idx+1].clamp(-1, 1).add(1).div(2), img_path)
                        with open(prompt_path, 'w') as f:
                            f.write(prompt)
                # MODIFICATION END
                
                # Clean up the tensor to free VRAM on each rank
                del images_tensor
                torch.cuda.empty_cache()

                # Synchronize all processes. This is crucial to ensure all ranks have finished
                # writing their files before rank 0 attempts to read them.
                if not debug:
                    dist.barrier()

                # Part 2: Rank 0 is responsible for collating images and prompts.
                if rank == 0:

                    all_images_for_grid = []
                    all_prompts_for_caption = []

                    # Discover all generated files by looking for the prompt text files
                    # Sorting ensures a consistent order when building the grid
                    prompt_files = sorted([f for f in os.listdir(temp_inference_folder) if f.endswith('.txt')])
                    
                    for filename in prompt_files:
                        img_path = os.path.join(temp_inference_folder, filename.replace('.txt', '.jpg'))
                        prompt_path = os.path.join(temp_inference_folder, filename)

                        if os.path.exists(img_path):
                            # Load the image and convert it to a tensor
                            pil_img = PILImage.open(img_path)
                            to_tensor = transforms.ToTensor()
                            img_tensor = to_tensor(pil_img)
                            all_images_for_grid.append(img_tensor)

                            # Read the corresponding prompt
                            with open(prompt_path, 'r') as f:
                                all_prompts_for_caption.append(f.read())
                    
                    if all_images_for_grid:
                        # Create a single grid from all collected images
                        # Adjust nrow to control the layout of the grid
                        final_grid = make_grid(all_images_for_grid, nrow=world_size, normalize=False)

                        # Save the combined grid to its final destination
                        final_image_path = os.path.join(inference_config.inference_folder, f"{counter}.jpg")
                        save_image(final_grid, final_image_path)
                        print(f"Combined image grid saved to {final_image_path}")
                        
                        # Track the final collaged image and its combined caption with Aim
                        final_pil_image = PILImage.open(final_image_path)
                        combined_caption = "\n---\n".join(all_prompts_for_caption)
                        aim_img = AimImage(final_pil_image, caption=combined_caption)
                        run.track(aim_img, name='example_image', step=global_step)
                        
                        # Clean up memory
                        del final_grid
                        del final_pil_image
                        del aim_img
                        del all_images_for_grid
                    
                    # Clean up the temporary directory now that its contents are processed
                    shutil.rmtree(temp_inference_folder)

                # This barrier ensures the next training step doesn't start until rank 0 has
                # finished its file operations, preventing any potential conflicts.
                if not debug:
                    dist.barrier()

            # flush
            acc_embeddings = []
            global_step += 1
            

        # save final model
        if rank == 0:
            model.to("cpu")
            model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            torch.save(
                model.state_dict(),
                model_filename,
            )

            config_data["dataloader"]["offset"] = dataloader_config.offset
            config_data["model"]["chroma_path"] = model_filename
            dump_dict_to_json(
                config_data, f"{training_config.save_folder}/training_config.json"
            )

            if training_config.hf_token:
                upload_to_hf(
                    model_filename,
                    model_filename,
                    training_config.hf_repo_id,
                    training_config.hf_token,
                )
            torch.cuda.empty_cache()
    if not debug:
        dist.destroy_process_group()