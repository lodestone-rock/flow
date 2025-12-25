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
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

import random

from transformers import AutoTokenizer, Qwen3ForCausalLM
import wandb
import shutil
from torchvision import transforms


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
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
from ramtorch import AdamW
from ramtorch.helpers import replace_linear_with_ramtorch, reattach_is_ramtorch_flags
from ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params
from ramtorch.zero2 import setup_grad_sharding_hooks
from huggingface_hub import HfApi, upload_file
import time

from aim import Run, Image as AimImage
from datetime import datetime
from PIL import Image as PILImage

from torch.profiler import profile, record_function, ProfilerActivity


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
    do_profiling: bool
    trained_layer_keywords: list[str]
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
    cfg: int
    prompts: list[str]
    first_n_steps_wo_cfg: int
    image_dim: tuple[int, int]
    max_sequence_length: int


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

    z_image_path: str
    vae_path: str
    qwen_path: str
    qwen_tokenizer_path: str
    max_sequence_length: int
    use_x0: bool = False


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


def prepare_sot_pairings(latents, is_x0_v=False):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    image_pos_id = prepare_latent_image_ids(n, h, w)

    # randomize ode timesteps
    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=latents.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=latents.device
    )

    timesteps = input_timestep[:, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = latents * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    if is_x0_v:
        # this is equivalent to target = noise - images
        # but since there's singularity we just shift the target up a bit
        # and the model also doing shifted prediction
        target = (noisy_latents - latents) / (timesteps + 5e-2)
    else:
        target = noise - latents

    return noisy_latents, target, input_timestep, latent_shape


def init_optimizer(
    model, trained_layer_keywords, lr, wd, warmup_steps, rank, world_size
):
    # TODO: pack this into a function
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
            print(f"training {name}")
        else:
            param.requires_grad = False  # Optionally disable grad for others

    # Setup ZeRO-1: Shard optimizer states across workers
    # Each worker only maintains optimizer states for a subset of parameters
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
    shard_param_group = create_zero_param_groups(param_groups, world_size)
    optimizer = AdamW(
        params=shard_param_group[rank],
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
        dtype=torch.float32,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, shard_param_group


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
    ae,
    qwen_tokenizer,
    qwen_encoder,
    seed: int,
    steps: int,
    cfg: int,
    prompts: list,
    rank: int,
    first_n_steps_wo_cfg: int,
    image_dim=(512, 512),
    max_sequence_length=512,
):
    """Run inference with Z-Image model"""
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg
    PROMPT = prompts
    MAX_SEQUENCE_LENGTH = max_sequence_length

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = get_noise(len(PROMPT), HEIGHT, WIDTH, rank, torch.bfloat16, seed)
            noise, shape = vae_flatten(noise)
            n, c, h, w = shape

            timesteps = get_schedule(STEPS, noise.shape[1])

            # Encode text prompts with Qwen3 chat template
            formatted_prompts = []
            for p in PROMPT:
                messages = [{"role": "user", "content": p}]
                formatted_prompt = qwen_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                formatted_prompts.append(formatted_prompt)

            text_inputs = qwen_tokenizer(
                formatted_prompts,
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(qwen_encoder.device)
            prompt_masks = text_inputs.attention_mask.to(qwen_encoder.device).bool()

            # Get embeddings from second-to-last hidden state
            qwen_embed = (
                qwen_encoder(
                    input_ids=text_input_ids,
                    attention_mask=prompt_masks,
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .to(rank)
            )

            # Encode negative prompts (empty strings)
            formatted_prompts_neg = []
            for _ in range(len(PROMPT)):
                messages_neg = [{"role": "user", "content": ""}]
                formatted_prompt_neg = qwen_tokenizer.apply_chat_template(
                    messages_neg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                formatted_prompts_neg.append(formatted_prompt_neg)

            text_inputs_neg = qwen_tokenizer(
                formatted_prompts_neg,
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids_neg = text_inputs_neg.input_ids.to(qwen_encoder.device)
            prompt_masks_neg = text_inputs_neg.attention_mask.to(
                qwen_encoder.device
            ).bool()

            qwen_embed_neg = (
                qwen_encoder(
                    input_ids=text_input_ids_neg,
                    attention_mask=prompt_masks_neg,
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .to(rank)
            )

            # Prepare position IDs
            token_length = prompt_masks.sum(1)
            neg_token_length = prompt_masks_neg.sum(1)
            offset_for_img_id = torch.maximum(token_length, neg_token_length)

            image_pos_id = prepare_latent_image_ids(offset_for_img_id, h, w).to(rank)
            text_ids = make_text_position_ids(token_length, MAX_SEQUENCE_LENGTH)
            neg_text_ids = make_text_position_ids(neg_token_length, MAX_SEQUENCE_LENGTH)

            # Run denoising with CFG
            output_image = denoise_cfg(
                model,
                noise,
                image_pos_id,
                qwen_embed,
                qwen_embed_neg,
                text_ids,
                neg_text_ids,
                prompt_masks,
                prompt_masks_neg,
                timesteps,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
            )

            # Decode latents to images
            output_image = ae.decode(vae_unflatten(output_image, shape))

    model.train()
    return output_image


def train_zimage(rank, world_size, wrap_models, wrap_config, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    torch.cuda.set_device(rank)
    latch = True
    profile_steps = 3

    (
        config_data,
        training_config,
        inference_config,
        dataloader_config,
        model_config,
        extra_inference_config,
    ) = wrap_config

    (qwen_tokenizer, qwen_encoder, model, ae) = wrap_models

    # Replace Linear layers with RamTorch (shares CPU memory across workers)
    reattach_is_ramtorch_flags(model, rank)
    reattach_is_ramtorch_flags(qwen_encoder, rank)

    qwen_encoder.to(rank)
    model.to(rank)
    model.train()

    # Setup Aim run
    if training_config.aim_path is not None and rank == 0:
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = Run(
            repo=training_config.aim_path,
            run_hash=training_config.aim_hash,
            experiment=training_config.aim_experiment_name,
            force_resume=True,
        )

        hparams = config_data.copy()
        hparams["training"]["aim_path"] = None
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

    trained_layer_keywords = training_config.trained_layer_keywords

    # Setup ZeRO-1: Shard optimizer states across workers
    # Each worker only maintains optimizer states for a subset of parameters
    optimizer, scheduler, shard_param_group = init_optimizer(
        model,
        trained_layer_keywords,
        training_config.lr,
        training_config.weight_decay,
        training_config.warmup_steps,
        rank,
        world_size,
    )

    # Setup ZeRO-2: Shard gradients across workers
    # Gradients are partitioned and only linked on the worker responsible for them
    setup_grad_sharding_hooks(shard_param_group, rank)

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
        offset=dataloader_config.offset,
    )

    global_step = training_config.aim_steps

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        if not training_config.do_profiling:
            prof.stop()
        while True:
            training_config.master_seed += 1
            torch.manual_seed(training_config.master_seed)
            model.train()

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
                if counter == profile_steps and latch and training_config.do_profiling:
                    prof.stop()
                    prof.export_chrome_trace(f"trace_rank_{rank}.json")
                    print(f"Stopped profiling and saved trace at step {counter}")
                    latch = False

                images, caption, index, loss_weighting = data[0]
                # just in case the dataloader is failing
                caption = [x if x is not None else "" for x in caption]
                caption = [
                    x.lower() if torch.rand(1).item() < 0.25 else x for x in caption
                ]
                loss_weighting = torch.tensor(loss_weighting, device=rank)

                # Encode images to latents using VAE
                ae.to(rank)
                acc_images = []
                for mb_i in tqdm(
                    range(
                        dataloader_config.batch_size
                        // training_config.cache_minibatch
                        // world_size
                    ),
                    desc=f"preparing latents, Rank {rank}",
                    position=rank,
                ):
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        latents = ae.encode_for_train(
                            images[
                                mb_i
                                * training_config.cache_minibatch : mb_i
                                * training_config.cache_minibatch
                                + training_config.cache_minibatch
                            ].to(rank)
                        ).to("cpu", non_blocking=True)
                        acc_images.append(latents)
                        torch.cuda.empty_cache()

                acc_images = torch.cat(acc_images, dim=0)
                # TODO: double check the training logic
                # positional embedding must be calibrated !
                #
                #  Prepare SOT pairings
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    (
                        noisy_images,
                        target,
                        input_timestep,
                        latent_shape,
                    ) = prepare_sot_pairings(acc_images.to(rank), model_config.use_x0)

                noisy_images.requires_grad_(True)
                ot_bs = acc_images.shape[0]

                mb = training_config.train_minibatch
                local_num_minibatches = dataloader_config.batch_size // mb // world_size
                updates_per_large_batch = max(
                    1, training_config.updates_per_large_batch
                )
                minibatches_per_update = max(
                    1, local_num_minibatches // updates_per_large_batch
                )

                torch.cuda.empty_cache()

                loss_log = []
                for tmb_i in tqdm(
                    range(local_num_minibatches),
                    desc=f"minibatch training, Rank {rank}",
                    position=rank,
                ):
                    model.train()
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        # Slice captions for the current minibatch
                        minibatch_captions = caption[tmb_i * mb : tmb_i * mb + mb]

                        # Format prompts with Qwen chat template
                        formatted_prompts = []
                        for p in minibatch_captions:
                            messages = [{"role": "user", "content": p}]
                            formatted_prompt = qwen_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True,
                            )
                            formatted_prompts.append(formatted_prompt)

                        # Tokenize and compute embeddings
                        text_inputs = qwen_tokenizer(
                            formatted_prompts,
                            padding="max_length",
                            max_length=model_config.max_sequence_length,
                            truncation=True,
                            return_tensors="pt",
                        ).to(rank)

                        minibatch_embeddings = qwen_encoder(
                            input_ids=text_inputs.input_ids,
                            attention_mask=text_inputs.attention_mask,
                            output_hidden_states=True,
                        ).hidden_states[-2]

                        minibatch_mask = text_inputs.attention_mask.bool()

                        # Prepare position IDs
                        n, c, h, w = latent_shape
                        token_length = minibatch_mask.sum(1)
                        offset_for_img_id = token_length[tmb_i * mb : tmb_i * mb + mb]

                        image_pos_id = prepare_latent_image_ids(
                            offset_for_img_id, h, w
                        ).to(rank)

                        text_ids = make_text_position_ids(
                            token_length[tmb_i * mb : tmb_i * mb + mb],
                            model_config.max_sequence_length,
                        )

                        torch.cuda.empty_cache()

                    # Forward pass
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(
                            img=noisy_images[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            img_ids=image_pos_id,
                            txt=minibatch_embeddings[tmb_i * mb : tmb_i * mb + mb],
                            txt_ids=text_ids,
                            txt_mask=minibatch_mask[tmb_i * mb : tmb_i * mb + mb],
                            timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                        )

                        # Compute per-element squared error and mean over sequence and feature dims
                        loss = (
                            (pred - target[tmb_i * mb : tmb_i * mb + mb]) ** 2
                        ).mean(
                            dim=(1, 2)
                        )  # Shape: [mb]

                        # Normalize per full batch
                        loss = loss / (
                            dataloader_config.batch_size // mb
                        )  # Shape: [mb]

                        # Apply per-sample weight
                        weights = loss_weighting[
                            tmb_i * mb : tmb_i * mb + mb
                        ]  # Shape: [mb]

                        # Normalize weights to ensure the overall loss scale is consistent
                        weights = weights / weights.sum()

                        # Compute final weighted loss
                        loss = (loss * weights).sum()

                    torch.cuda.empty_cache()
                    # Synchronize before backward to ensure all workers are ready
                    torch.cuda.synchronize()
                    loss.backward()
                    torch.cuda.synchronize()
                    loss_log.append(loss.item() * (dataloader_config.batch_size // mb))

                    # Check if it's time for a parameter update
                    is_update_step = (tmb_i + 1) % minibatches_per_update == 0
                    is_last_minibatch = (tmb_i + 1) == local_num_minibatches

                    if is_update_step or is_last_minibatch:
                        print("param updated")
                        torch.cuda.empty_cache()

                        scheduler.step()
                        optimizer.step()

                        # Broadcast updated parameters to all workers
                        # RamTorch parameters are already shared via CPU memory,
                        # but standard PyTorch parameters need explicit broadcasting
                        broadcast_zero_params(shard_param_group)

                        # IMPORTANT: Use model.zero_grad(), not optimizer.zero_grad()
                        # Each worker handles partial gradients, so we need to zero
                        # gradients at the model level to properly clear all workers' buffers
                        model.zero_grad()

                        torch.cuda.empty_cache()

                del noisy_images, target, input_timestep, acc_images

                if rank == 0:
                    with torch.no_grad():
                        # The aggregation now works with Python floats, no tensor involved
                        final_loss_value = sum(loss_log) / len(loss_log)
                        # if you need it as a tensor for tracking:
                        loss_log_tensor = torch.tensor(final_loss_value, device=rank)
                        run.track(loss_log_tensor, name="loss", step=global_step)
                        run.track(
                            training_config.lr, name="learning_rate", step=global_step
                        )

                dataloader_config.offset += 1

                # Save checkpoint
                if (counter + 1) % training_config.save_every == 0 and rank == 0:
                    model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                    torch.save(model.state_dict(), model_filename)

                    config_data["dataloader"]["offset"] = dataloader_config.offset
                    config_data["model"]["z_image_path"] = model_filename
                    dump_dict_to_json(
                        config_data,
                        f"{training_config.save_folder}/training_config.json",
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
                    temp_inference_folder = os.path.join(
                        inference_config.inference_folder, f"step_{counter}_temp"
                    )
                    os.makedirs(temp_inference_folder, exist_ok=True)

                    # Each rank gets its own unique prompt from its current batch data
                    preview_prompts_this_rank = inference_config.prompts + caption[:1]

                    # Combine the main inference config with any extra ones
                    all_inference_configs = [inference_config] + extra_inference_config

                    for config_idx, current_config in enumerate(all_inference_configs):
                        images_tensor = inference_wrapper(
                            model=model,
                            ae=ae,
                            qwen_tokenizer=qwen_tokenizer,
                            qwen_encoder=qwen_encoder,
                            seed=training_config.master_seed + rank,
                            steps=current_config.steps,
                            cfg=current_config.cfg,
                            prompts=preview_prompts_this_rank,
                            rank=rank,
                            first_n_steps_wo_cfg=current_config.first_n_steps_wo_cfg,
                            image_dim=current_config.image_dim,
                            max_sequence_length=current_config.max_sequence_length,
                        )

                        # Loop through the generated images and prompts to save them individually
                        for prompt_idx, prompt in enumerate(preview_prompts_this_rank):
                            # Define unique filenames for the image and its prompt
                            base_filename = (
                                f"config{config_idx}_prompt{prompt_idx}_rank{rank}"
                            )
                            img_path = os.path.join(
                                temp_inference_folder, f"{base_filename}.jpg"
                            )
                            prompt_path = os.path.join(
                                temp_inference_folder, f"{base_filename}.txt"
                            )

                            # Save the specific image from the tensor batch
                            save_image(
                                images_tensor[prompt_idx : prompt_idx + 1]
                                .clamp(-1, 1)
                                .add(1)
                                .div(2),
                                img_path,
                            )
                            with open(prompt_path, "w") as f:
                                f.write(prompt)

                    # Clean up the tensor to free VRAM on each rank
                    del images_tensor
                    torch.cuda.empty_cache()

                    if not debug:
                        dist.barrier()

                    # Part 2: Rank 0 is responsible for collating images and prompts.
                    if rank == 0:

                        all_images_for_grid = []
                        all_prompts_for_caption = []

                        # Discover all generated files by looking for the prompt text files
                        # Sorting ensures a consistent order when building the grid
                        prompt_files = sorted(
                            [
                                f
                                for f in os.listdir(temp_inference_folder)
                                if f.endswith(".txt")
                            ]
                        )

                        for filename in prompt_files:
                            img_path = os.path.join(
                                temp_inference_folder, filename.replace(".txt", ".jpg")
                            )
                            prompt_path = os.path.join(temp_inference_folder, filename)

                            if os.path.exists(img_path):
                                # Load the image and convert it to a tensor
                                pil_img = PILImage.open(img_path)
                                to_tensor = transforms.ToTensor()
                                img_tensor = to_tensor(pil_img)
                                all_images_for_grid.append(img_tensor)

                                # Read the corresponding prompt
                                with open(prompt_path, "r") as f:
                                    all_prompts_for_caption.append(f.read())

                        if all_images_for_grid:
                            # Create a single grid from all collected images
                            # Adjust nrow to control the layout of the grid
                            final_grid = make_grid(
                                all_images_for_grid, nrow=world_size, normalize=False
                            )

                            # Save the combined grid to its final destination
                            final_image_path = os.path.join(
                                inference_config.inference_folder, f"{counter}.jpg"
                            )
                            save_image(final_grid, final_image_path)
                            print(f"Combined image grid saved to {final_image_path}")

                            # Track the final collaged image and its combined caption with Aim
                            final_pil_image = PILImage.open(final_image_path)
                            combined_caption = "\n---\n".join(all_prompts_for_caption)
                            aim_img = AimImage(
                                final_pil_image, caption=combined_caption
                            )
                            run.track(aim_img, name="example_image", step=global_step)

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

                global_step += 1

            # save final model
            if rank == 0:
                # model.to("cpu")
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(
                    model.state_dict(),
                    model_filename,
                )

                config_data["dataloader"]["offset"] = dataloader_config.offset
                config_data["model"]["z_image_path"] = model_filename
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


def main(config="training_config.json"):

    world_size = torch.cuda.device_count()

    # load all config
    config_data = load_config_from_json(config)
    training_config = TrainingConfig(**config_data["training"])
    inference_config = InferenceConfig(**config_data["inference"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    extra_inference_config = [
        InferenceConfig(**conf) for conf in config_data["extra_inference_config"]
    ]

    wrap_config = [
        config_data,
        training_config,
        inference_config,
        dataloader_config,
        model_config,
        extra_inference_config,
    ]

    # load model
    with torch.no_grad():
        # Load Z-Image model
        # TODO: set the x0 setting for z-image!
        z_image_params._use_compiled = True
        z_image_params.use_x0 = model_config.use_x0
        print(f"Using x0: {model_config.use_x0}")

        with torch.device("meta"):
            model = ZImage(z_image_params)

        # Check file extension to determine loading method
        if model_config.z_image_path.endswith(
            ".safetensors"
        ) or model_config.z_image_path.endswith(".sft"):
            state_dict = load_safetensors(model_config.z_image_path)

        else:  # Assume PyTorch format (.pth)
            state_dict = torch.load(model_config.z_image_path, map_location="cpu")

        if model_config.use_x0:
            print("using x0")

            state_dict["__x0__"] = torch.tensor([])

        model.load_state_dict(state_dict, assign=True)

        model = replace_linear_with_ramtorch(model)

        print("Z-Image loaded")

        # load VAE
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)
        print("VAE loaded")

        # Load Qwen3
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config.qwen_tokenizer_path)
        qwen_encoder = Qwen3ForCausalLM.from_pretrained(
            model_config.qwen_path,
            torch_dtype=torch.bfloat16,
        )
        qwen_encoder = replace_linear_with_ramtorch(qwen_encoder)
        qwen_encoder.eval()
        print("Qwen3 loaded")

        wrap_models = [qwen_tokenizer, qwen_encoder, model, ae]

    # Use spawn method for starting processes
    mp.spawn(
        train_zimage,
        args=(world_size, wrap_models, wrap_config, False),
        nprocs=world_size,
        join=True,
    )
