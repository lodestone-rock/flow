import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from copy import deepcopy
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

from torchastic import AdamW, StochasticAccumulator
import random

from transformers import T5Tokenizer
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.dataloaders.bucketing_logic import _bucket_generator 
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
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
    echo_times: int = 1 # New parameter for Reflow echoing
    teacher_ode_steps: int = 10 # New parameter for teacher ODE solver steps
    cfg_scale: float = 1.1
    teacher_uncond_percentage: float = 0.1

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
    teacher: bool


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


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    teacher_chroma_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int



def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

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

@torch.no_grad()
def prepare_reflow_pairings(
    model,
    ae,
    t5_tokenizer,
    t5,
    prompts,
    rank,
    n_steps,
    cfg_scale=1.2,
    image_dim=(512, 512), # TODO since we're not using images and just generating it, we should randomize this dim!
    t5_max_length=512,
):
    """
    the goal of this function is basically just doing an inference but we keep track the noise and image pairs
    since ODE path is deterministic the noise image pairs generated by this function are bijective!
    the bijective pairs generated by this function is approximately an optimal transport path!
    but keep in mind, straight flow path is only one part of the equation. if the model cannot generate proper 
    image from the learned data manifold, no matter what you do to straighten it it will just generate garbage!
    so make sure the data manifold sample generated by the teacher model is coherent before applying this rectification method
    you can gauge how much coherent the teacher manifold by generating stuff unconditionally, if it's spitting garbage
    then you have bad flow vector and you need to fix that first using constrastive loss or using online CFG for just a couple
    training steps first.
    """
    B = len(prompts)
    model.eval()
    
    # generate noise set this will return [n, c, h//8, w//8] atleast for the original VAE
    noise = get_noise(B, image_dim[1], image_dim[0], rank, torch.bfloat16, torch.randint(0, 2**32 - 1, []).item()) # check
    # now it's being flattened to [B, L, D] but the og shape preserved in the shape variable as a list
    noise, shape = vae_flatten(noise)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        n, c, h, w = shape
        image_pos_id = prepare_latent_image_ids(n, h, w).to(rank)

        # --- Pre-calculate embeddings and IDs to avoid redundancy in the loop ---
        model.to("cpu")
        t5.to(rank)

        # compute the text embedding for positive conditions
        text_inputs = t5_tokenizer(
            prompts,
            padding="max_length",
            max_length=t5_max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).to(t5.device, non_blocking=True)
        t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to(rank)
        
        # compute the text embedding for negative conditions
        text_inputs_neg = t5_tokenizer(
            [""]*n,
            padding="max_length",
            max_length=t5_max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).to(t5.device, non_blocking=True)
        t5_embed_neg = t5(text_inputs_neg.input_ids, text_inputs_neg.attention_mask).to(
            rank
        )

        # text ids on the 3d rope is just bunch of zeros for all 3 dim
        # bfl reserved the 3rd dim for vid but probably will never get used
        text_ids = torch.zeros((n, t5_max_length, 3), device=rank)
        neg_text_ids = torch.zeros((n, t5_max_length, 3), device=rank)
        
        # vestigial code from chroma distillation, keep it as zero!
        static_guidance = torch.zeros(n, device=rank)

        # switch back t5 and model to save space
        t5.to("cpu") 
        model.to(rank)
        
        # --- Helper function for calculating velocity with CFG and static guidance ---
        def get_velocity(current_x, current_t):

            if cfg_scale > 1:
                # CORRECTED: Added the 'guidance' parameter to the model calls
                v_cond = model(current_x, image_pos_id, t5_embed, text_ids, text_inputs.attention_mask, current_t, guidance=static_guidance)
                v_uncond = model(current_x, image_pos_id, t5_embed_neg, neg_text_ids, text_inputs_neg.attention_mask, current_t, guidance=static_guidance)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return model(current_x, image_pos_id, t5_embed, text_ids, text_inputs.attention_mask, current_t, guidance=static_guidance)

        # --- Main integration loop using Heun's method ---
        x = noise
        dt = 1.0 / n_steps
        for i in tqdm(range(n_steps, 0, -1), desc="generating pairs"):
            t_current = torch.full((n,), i / n_steps, device=rank)
            
            # 1. Predictor step (Euler): Estimate the velocity at the current point.
            k1 = get_velocity(x, t_current)
            x_pred = x - k1 * dt # Make a temporary step forward (backward in time).

            # 2. Corrector step: Estimate the velocity at the predicted next point.
            t_next = torch.full((n,), (i - 1) / n_steps, device=rank)
            k2 = get_velocity(x_pred, t_next)

            # 3. Final update: Average the two velocities for a more accurate step.
            x = x - (k1 + k2) / 2.0 * dt

        latents = x

    ae.to(rank)
    output_image = ae.decode(vae_unflatten(x, shape)) #for debug
    save_image(make_grid(output_image.clamp(-1, 1).add(1).div(2), nrow=8, normalize=True), f"{rank}_test.jpg")
    ae.to("cpu")
    model.to("cpu") # move teacher back to cpu


    # TODO! doublecheck the pairs, noise and image here must be deterministic and bijective 

    # only return noise and latent pairs because we already generated it here, we can do lerp in the training loop

    return noise, latents, shape, t5_embed, text_inputs.attention_mask, t5_embed_neg, text_inputs_neg.attention_mask

def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)
    # init optimizer
    optimizer = AdamW(
        [
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
        ],
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

    return optimizer, scheduler, hooks, trained_params


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


def inference_wrapper(
    model,
    ae,
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
    DEVICE = model.device
    PROMPT = prompts

    T5_MAX_LENGTH = t5_max_length

    # store device state of each model
    t5_device = t5.device
    ae_device = ae.device
    model_device = model.device
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # init random noise
            noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, SEED)
            noise, shape = vae_flatten(noise)
            noise = noise.to(rank)
            n, c, h, w = shape
            image_pos_id = prepare_latent_image_ids(n, h, w).to(rank)

            timesteps = get_schedule(STEPS, noise.shape[1])

            model.to("cpu")
            ae.to("cpu")
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
                [""],
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

            ae.to("cpu")
            t5.to("cpu")
            model.to(rank)  # load model to gpu
            latent_cfg = denoise_cfg(
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

            model.to("cpu")
            t5.to("cpu")
            ae.to(rank)  # load ae to gpu
            output_image = ae.decode(vae_unflatten(latent_cfg, shape))

            # restore back state
            model.to("cpu")
            t5.to("cpu")
            ae.to("cpu")

    return output_image


def train_chroma(rank, world_size, debug=False, json_config="training_config_reflowing.json"):
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
        run = Run(repo=training_config.aim_path, run_hash=training_config.aim_hash, experiment=training_config.aim_experiment_name)

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

        with torch.device("meta"):
            teacher_model = Chroma(chroma_params)

        # Check file extension to determine loading method
        if model_config.teacher_chroma_path.endswith('.safetensors') or model_config.teacher_chroma_path.endswith('.sft'):
            teacher_model.load_state_dict(load_safetensors(model_config.teacher_chroma_path), assign=True)
        else:  # Assume PyTorch format (.pth)
            teacher_model.load_state_dict(torch.load(model_config.teacher_chroma_path, map_location="cpu"), assign=True)

        # randomly train inner layers at a time
        trained_double_blocks = list(range(len(model.double_blocks)))
        trained_single_blocks = list(range(len(model.single_blocks)))
        random.shuffle(trained_double_blocks)
        random.shuffle(trained_single_blocks)
        # lazy :P
        trained_double_blocks = trained_double_blocks * 1000000
        trained_single_blocks = trained_single_blocks * 1000000

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

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
        uncond_percentage=0.0, # all generation must be conditional, just drop the teacher caption later on
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
        dummy_image=True # we don't use image during training so it's better to not load it in the dataloader
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
                    + ["txt_in", "img_in", "final_layer"]
                )

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                optimizer, scheduler, hooks, trained_params = init_optimizer(
                    model,
                    trained_layer_keywords,
                    training_config.lr,
                    training_config.weight_decay,
                    training_config.warmup_steps,
                )

                optimizer_counter += 1

            # we load and unload vae and t5 here to reduce vram usage
            # think of this as caching on the fly
            # load t5 and vae to GPU
            # ae.to(rank)
            t5.to(rank)

            acc_noise = []
            acc_latents = []
            acc_embeddings = []
            acc_mask = []
            acc_embeddings_neg = []
            acc_mask_neg = []

            # image dim is randomized here instead of inside the loop so it can be concatenated
            random_image_dim = random.choice(_bucket_generator(random.choice(dataloader_config.base_resolution)))
            for mb_i in tqdm(
                range(
                    dataloader_config.batch_size
                    // training_config.cache_minibatch
                    // world_size
                ),
                desc=f"preparing latents, Rank {rank}",
                position=rank,
            ):

                # --- Reflow Data Generation we generate this once and reuse it many times in echo ---
                _noises, _latents, latent_shape, _t5_embed, _t5_mask, _t5_embed_neg, _t5_mask_neg = prepare_reflow_pairings(
                    model=teacher_model,
                    ae=ae, # pass ae for debugging
                    t5_tokenizer=t5_tokenizer,
                    t5=t5,
                    prompts=caption[
                        mb_i
                        * training_config.cache_minibatch : mb_i
                        * training_config.cache_minibatch
                        + training_config.cache_minibatch
                    ],
                    rank=rank,
                    n_steps=training_config.teacher_ode_steps,
                    cfg_scale=training_config.cfg_scale,
                    image_dim=random_image_dim
                )
                acc_noise.append(_noises)
                acc_latents.append(_latents)
                acc_embeddings.append(_t5_embed)
                acc_mask.append(_t5_mask)
                acc_embeddings_neg.append(_t5_embed_neg)
                acc_mask_neg.append(_t5_mask_neg)

                # flush
                torch.cuda.empty_cache()

            # accumulate the latents and embedding in a variable
            # unload t5 and vae

            t5.to("cpu")
            ae.to("cpu")
            torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            # move model to device
            # teacher already moved to cpu in prepare_reflow_pairings
            model.to(rank)
            
            # latents is disabled!
            acc_noise = torch.cat(acc_noise, dim=0)
            acc_latents = torch.cat(acc_latents, dim=0)
            acc_embeddings = torch.cat(acc_embeddings, dim=0)
            acc_mask = torch.cat(acc_mask, dim=0)
            acc_embeddings_neg = torch.cat(acc_embeddings_neg, dim=0)
            acc_mask_neg = torch.cat(acc_mask_neg, dim=0)
            B = dataloader_config.batch_size // world_size

            # Pick random indices for replacement (10% of batch)
            replace_mask = torch.rand(B, device=acc_embeddings.device) < training_config.teacher_uncond_percentage

            # Do replacement
            acc_embeddings[replace_mask] = acc_embeddings_neg[replace_mask]
            acc_mask[replace_mask] = acc_mask_neg[replace_mask]


            # define the bijective velocity using the noise image pairs pregenerated by the ODE sampling
            direct_latent_pred = acc_latents # acc_noise - acc_latents

            # generate timestep distribution to sample from
            num_points = 1000  # Number of points in the range
            x, probabilities = create_distribution(num_points, device=rank)

            # "Echo" loop: Reuse the data pair for multiple training steps
            # echoing here is fine compared to the flow matching objective, because the pairs is already bijective
            # instead of stochastic random pairs to marginally induce the flow vector we already have the proper one
            # we want the model to regress to this exact vector !
            for _ in range(training_config.echo_times):
                # we need to resample the lerp points every echo
                input_timestep = sample_from_distribution(x, probabilities, B, device=rank)
                # NOTE forced x0 prediction
                # 1 is full noise
                input_timestep = torch.ones_like(input_timestep)
                timesteps = input_timestep[:, None, None]
                # get the random interpolation points 
                # NOTE input is pure noise
                noisy_latents = acc_noise # acc_latents * (1 - timesteps) + acc_noise * timesteps
                _, c, h, w = latent_shape # n channel is not used because of minibatching loop before
                image_pos_id = prepare_latent_image_ids(B, h, w).to(rank)
                text_ids = torch.zeros((B, 512, 3), device=rank)
                static_guidance = torch.tensor([0.0] * B, device=rank) # check


                # set the input to requires grad to make autograd works
                noisy_latents.requires_grad_(True) # check
                acc_embeddings.requires_grad_(True)

                # aliasing
                mb = training_config.train_minibatch
                loss_log = []
                for tmb_i in tqdm(
                    range(dataloader_config.batch_size // mb // world_size),
                    desc=f"minibatch training, Rank {rank}",
                    position=rank,
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(
                            img=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt_mask=acc_mask[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                        )
                        # TODO: need to scale the loss with rank count and grad accum!
                        pred = acc_noise[tmb_i * mb : tmb_i * mb + mb] - pred
                        # Compute per-element squared error and mean over sequence and feature dims
                        loss = ((pred - direct_latent_pred[tmb_i * mb : tmb_i * mb + mb]) ** 2).mean(dim=(1, 2))  # Shape: [mb]

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
                    loss.backward()
                    loss_log.append(
                        loss.detach().clone() * (dataloader_config.batch_size // mb)
                    )
                loss_log = sum(loss_log) / len(loss_log)
            # offload some params to cpu just enough to make room for the caching process
            # and only offload non trainable params
            del acc_embeddings, noisy_latents, acc_mask # check
            torch.cuda.empty_cache()
            offload_param_count = 0
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in trained_layer_keywords):
                    if offload_param_count < training_config.offload_param_count:
                        offload_param_count += param.numel()
                        param.data = param.data.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()
            optimizer_state_to(optimizer, rank)

            StochasticAccumulator.reassign_grad_buffer(model)

            if not debug:
                synchronize_gradients(model)

            scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                run.track(loss_log, name='loss', step=global_step)
                run.track(training_config.lr, name='learning_rate', step=global_step)

            optimizer_state_to(optimizer, "cpu")
            torch.cuda.empty_cache()

            if (counter + 1) % training_config.save_every == 0 and rank == 0:
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(
                    model.state_dict(),
                    model_filename,
                )
                if training_config.hf_token:
                    upload_to_hf(
                        model_filename,
                        model_filename,
                        training_config.hf_repo_id,
                        training_config.hf_token,
                    )
            if not debug:
                dist.barrier()

            if (counter + 1) % inference_config.inference_every == 0:
                all_grids = []

                preview_prompts = inference_config.prompts + caption[:1]

                # teacher model inference
                if inference_config.teacher:
                    model.to("cpu")
                    teacher_model.to(rank)
                # student model inference
                else:
                    model.to(rank)
                    teacher_model.to("cpu")

                for prompt in preview_prompts:
                    images_tensor = inference_wrapper(
                        model=teacher_model if inference_config.teacher else model,
                        ae=ae,
                        t5_tokenizer=t5_tokenizer,
                        t5=t5,
                        seed=training_config.master_seed + rank,
                        steps=inference_config.steps,
                        guidance=inference_config.guidance,
                        cfg=inference_config.cfg,
                        prompts=[prompt],  # Pass single prompt as a list
                        rank=rank,
                        first_n_steps_wo_cfg=inference_config.first_n_steps_wo_cfg,
                        image_dim=inference_config.image_dim,
                        t5_max_length=inference_config.t5_max_length,
                    )

                    # gather from all gpus
                    if not debug:
                        gather_list = (
                            [torch.empty_like(images_tensor) for _ in range(world_size)]
                            if rank == 0
                            else None
                        )
                        dist.gather(images_tensor, gather_list=gather_list, dst=0)

                    if rank == 0:
                        # Concatenate gathered tensors
                        if not debug:
                            gathered_images = torch.cat(
                                gather_list, dim=0
                            )  # (total_images, C, H, W)
                        else:
                            gathered_images = images_tensor

                        # Create a grid for this prompt
                        grid = make_grid(
                            gathered_images.clamp(-1, 1).add(1).div(2),
                            nrow=8,
                            normalize=True,
                        )  # Adjust nrow as needed
                        all_grids.append(grid)


                for extra_inference in extra_inference_config:
                    # teacher model inference
                    if extra_inference.teacher:
                        model.to("cpu")
                        teacher_model.to(rank)
                    # student model inference
                    else:
                        model.to(rank)
                        teacher_model.to("cpu")


                    for prompt in preview_prompts:
                        images_tensor = inference_wrapper(
                            model=teacher_model if extra_inference.teacher else model,
                            ae=ae,
                            t5_tokenizer=t5_tokenizer,
                            t5=t5,
                            seed=training_config.master_seed + rank,
                            steps=extra_inference.steps,
                            guidance=extra_inference.guidance,
                            cfg=extra_inference.cfg,
                            prompts=[prompt],  # Pass single prompt as a list
                            rank=rank,
                            first_n_steps_wo_cfg=extra_inference.first_n_steps_wo_cfg,
                            image_dim=extra_inference.image_dim,
                            t5_max_length=extra_inference.t5_max_length,
                        )

                        # gather from all gpus
                        if not debug:
                            gather_list = (
                                [
                                    torch.empty_like(images_tensor)
                                    for _ in range(world_size)
                                ]
                                if rank == 0
                                else None
                            )
                            dist.gather(images_tensor, gather_list=gather_list, dst=0)

                        if rank == 0:
                            # Concatenate gathered tensors
                            if not debug:
                                gathered_images = torch.cat(
                                    gather_list, dim=0
                                )  # (total_images, C, H, W)
                            else:
                                gathered_images = images_tensor

                            # Create a grid for this prompt
                            grid = make_grid(
                                gathered_images.clamp(-1, 1).add(1).div(2),
                                nrow=8,
                                normalize=True,
                            )  # Adjust nrow as needed
                            all_grids.append(grid)

                # send prompt to rank 0
                if rank != 0:
                    dist.send_object_list(caption[:1], dst=0)

                else:
                    all_prompt = []
                    # Rank 0 receives from all other ranks
                    for src_rank in range(1, world_size):
                        # Initialize empty list with the same size to receive strings
                        received_strings = [None]
                        # Receive the list of strings
                        dist.recv_object_list(received_strings, src=src_rank)
                        all_prompt.extend(received_strings)

                if rank == 0:
                    # Combine all grids vertically
                    final_grid = torch.cat(
                        all_grids, dim=1
                    )  # Concatenate along height dimension

                    # Save the combined grid
                    file_path = os.path.join(
                        inference_config.inference_folder, f"{counter}.jpg"
                    )
                    save_image(final_grid, file_path)
                    print(f"Combined image grid saved to {file_path}")

                    # upload preview to aim

                    # Load your file_path into a PIL image if it isn't already
                    img_pil = PILImage.open(file_path)

                    # Wrap it for Aim
                    aim_img = AimImage(img_pil, caption="\n".join(preview_prompts + all_prompt))

                    run.track(aim_img, name='example_image', step=global_step)

            # flush
            acc_latents = []
            acc_embeddings = []
            global_step += 1

        # save final model
        if rank == 0:
            model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            torch.save(
                model.state_dict(),
                model_filename,
            )
            if training_config.hf_token:
                upload_to_hf(
                    model_filename,
                    model_filename,
                    training_config.hf_repo_id,
                    training_config.hf_token,
                )

    if not debug:
        dist.destroy_process_group()
