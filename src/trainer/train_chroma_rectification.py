import platform
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

from torchastic import Compass, StochasticAccumulator
import random

from transformers import T5Tokenizer
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg, denoise
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
    teacher_steps: int
    distillation_steps: int


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    backend = "nccl"  # Default backend for distributed training
    if platform.system() == "Windows":
        # Windows does not support NCCL, use GLOO instead
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
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


def sigmoid_schedule(T, alpha=5.12):
    x = torch.linspace(1, 0, T + 1)
    ts = torch.sigmoid(x * alpha - alpha / 2)
    return (ts - ts.min()) / (ts.max() - ts.min())  # norm


def denoise(
    model,
    img,
    img_ids,
    txt,
    txt_ids,
    txt_mask,
    timesteps,
):
    guidance_vec = torch.tensor([0.0] * img.shape[0], device=img.device)
    # TODO modify the timestep so it's not hard coded and can accept batch
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def prepare_rectification_vector(
    latents,
    embeddings,
    mask,
    teacher_model,
    distillation_steps=4,
    teacher_steps=40,
    t5_max_length=512,
    device="cpu",
):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    steps_per_segment = teacher_steps // distillation_steps
    image_pos_id = prepare_latent_image_ids(n, h, w).to(device)

    # get the starting point noise level index
    distilled_timestep_indices = torch.randint(
        0, distillation_steps, [1], device=latents.device
    )

    # high step teacher ODE sample
    # t = sigmoid_schedule(teacher_steps, 5.12)
    t = torch.linspace(1, 0, teacher_steps + 1)

    # teacher will do x inference steps to provide target vector for the student to regressed on
    # timestep used for this batch for teacher model
    # example:
    # tensor([0.1702, 0.1457, 0.1230, 0.1021, 0.0830, 0.0655, 0.0496, 0.0352, 0.0222, 0.0105, 0.0000])
    teacher_timestep_segments = t[
        distilled_timestep_indices
        * steps_per_segment : distilled_timestep_indices
        * steps_per_segment
        + steps_per_segment
        + 1
    ].to(device)
    # tensor([0.1702]) and repeated to match batch size and image dim
    teacher_starting_input_timestep = teacher_timestep_segments[:1].repeat([n])[
        :, None, None
    ]

    # since the teacher vector is not spanning from noise to image but only a chunk of it we need to scale the magnitude
    target_vector_scaler = torch.abs(
        teacher_timestep_segments[0] - teacher_timestep_segments[-1]
    )

    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents, device=device)

    # random lerp points for both student and the teacher
    # student will try to regress on the teacher vector directly
    # while the teacher use this noisy latent as the starting point to generate target vector
    noisy_latents = (
        latents * (1 - teacher_starting_input_timestep)
        + noise * teacher_starting_input_timestep
    )

    # TODO: replace this with teacher output!
    # TODO: chunk this either in here or outside
    text_ids = torch.zeros((n, t5_max_length, 3), device=device)

    # teacher final point to generate target vector
    teacher_less_noisy_latents = denoise(
        model=teacher_model,
        img=noisy_latents,
        img_ids=image_pos_id,
        txt=embeddings,
        txt_ids=text_ids,
        txt_mask=mask,
        timesteps=teacher_timestep_segments,
    )

    # target vector that being regressed on
    # TODO: double check the vector direction here! im easily confused which direction the vector should point at
    target = (noisy_latents - teacher_less_noisy_latents) / target_vector_scaler

    # now we need to pick random point between noisy latents and less noisy latents for student input latents
    # within
    # tensor([0.1702, ..., 0.0000])
    student_input_timestep = torch.empty_like(teacher_starting_input_timestep).uniform_(
        teacher_timestep_segments.min(), teacher_timestep_segments.max()
    )

    # now we need to interpolate inside the teacher vector
    # we need to map the point from this
    # tensor([0.1702, ..., 0.0000])
    # to this
    # tensor([1.0000, ..., 0.0000])
    # Remap from [min, max] to [0, 1]
    t_min = teacher_timestep_segments.min()
    t_max = teacher_timestep_segments.max()
    student_lerp_factor = (student_input_timestep - t_min) / (t_max - t_min)

    student_noisy_latents = (
        teacher_less_noisy_latents * (1 - student_lerp_factor)
        + noisy_latents * student_lerp_factor
    )

    return (
        student_noisy_latents,
        target,
        student_input_timestep.squeeze(),
        image_pos_id,
        latent_shape,
    )


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
    optimizer = Compass(
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
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def load_config_from_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as json_file:
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

            # timesteps = sigmoid_schedule(STEPS, 5.12).to(rank)
            timesteps = torch.linspace(1, 0, STEPS + 1).to(rank)

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


def train_chroma(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config_reflowing.json")

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
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
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
            ae.to(rank)
            t5.to(rank)

            acc_latents = []
            acc_embeddings = []
            acc_mask = []
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
                    # init random noise
                    text_inputs = t5_tokenizer(
                        caption[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ],
                        padding="max_length",
                        max_length=model_config.t5_max_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).to(t5.device)

                    # offload to cpu
                    t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask)
                    acc_embeddings.append(t5_embed)
                    acc_mask.append(text_inputs.attention_mask)

                    # flush
                    torch.cuda.empty_cache()
                    latents = ae.encode_for_train(
                        images[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ].to(rank)
                    )
                    acc_latents.append(latents)

                    # flush
                    torch.cuda.empty_cache()

            # accumulate the latents and embedding in a variable
            # unload t5 and vae

            t5.to("cpu")
            ae.to("cpu")
            torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            ############################################################################

            # move model to cpu to make room for teacher inference
            # and move teacher to device
            model.to("cpu")
            teacher_model.to(rank)
            # process the cache buffer now!
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):

                noisy_latents = []
                target = []
                input_timestep = []
                image_pos_id = []
                latent_shape = []
                # TODO check this zip, i think it's correct but double check it
                for mb_latents, mb_embeddings, mb_mask in tqdm(
                    zip(
                        acc_latents,
                        acc_embeddings,
                        acc_mask,
                    ),
                    desc=f"preparing target vector, Rank {rank}",
                    position=rank,
                    total=len(acc_latents),
                ):
                    # prepare flat image and the target lerp
                    (
                        mb_noisy_latents,
                        mb_target,
                        mb_input_timestep,
                        mb_image_pos_id,
                        mb_latent_shape,
                    ) = prepare_rectification_vector(
                        latents=mb_latents,
                        embeddings=mb_embeddings,
                        mask=mb_mask,
                        teacher_model=teacher_model,
                        # TODO: put this to json config
                        distillation_steps=model_config.distillation_steps,
                        teacher_steps=model_config.teacher_steps,
                        t5_max_length=512,
                        device=rank,
                    )

                    noisy_latents.append(mb_noisy_latents.to(torch.bfloat16))
                    target.append(mb_target.to(torch.bfloat16))
                    input_timestep.append(mb_input_timestep.to(torch.bfloat16))
                    image_pos_id.append(mb_image_pos_id.to(rank))

                acc_embeddings = torch.cat(acc_embeddings, dim=0)
                acc_mask = torch.cat(acc_mask, dim=0)
                noisy_latents = torch.cat(noisy_latents, dim=0)
                target = torch.cat(target, dim=0)
                input_timestep = torch.cat(input_timestep, dim=0)
                image_pos_id = torch.cat(image_pos_id, dim=0)

                # t5 text id for the model
                text_ids = torch.zeros((acc_embeddings.shape[0], 512, 3), device=rank)
                # NOTE:
                # using static guidance 1 for now
                # this should be disabled later on !
                static_guidance = torch.tensor(
                    [0.0] * acc_embeddings.shape[0], device=rank
                )

            # move model back to device and unload teacher
            # to make room for training
            model.to(rank)
            teacher_model.to("cpu")

            # set the input to requires grad to make autograd works
            noisy_latents.requires_grad_(True)
            acc_embeddings.requires_grad_(True)

            ot_bs = acc_embeddings.shape[0]

            # aliasing
            mb = training_config.train_minibatch
            loss_log = []
            for tmb_i in tqdm(
                range(dataloader_config.batch_size // mb // world_size),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                # do this inside for loops!
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

                    # Compute per-element squared error and mean over sequence and feature dims
                    loss = ((pred - target[tmb_i * mb : tmb_i * mb + mb]) ** 2).mean(
                        dim=(1, 2)
                    )  # Shape: [mb]

                    # Normalize per full batch
                    loss = loss / (dataloader_config.batch_size // mb)  # Shape: [mb]

                    # Apply per-sample weight
                    weights = loss_weighting[
                        tmb_i * mb : tmb_i * mb + mb
                    ]  # Shape: [mb]

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
            del acc_embeddings, noisy_latents, acc_latents
            torch.cuda.empty_cache()
            offload_param_count = 0
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in trained_layer_keywords):
                    if offload_param_count < training_config.offload_param_count:
                        offload_param_count += param.numel()
                        param.data = param.data.to("cpu", non_blocking=True)
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
