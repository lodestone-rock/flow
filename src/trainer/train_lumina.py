import platform
import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

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

from transformers import Gemma2Model, Gemma2Config, GemmaTokenizerFast
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.models.lumina.model import Lumina_2b
from src.models.lumina.sampling import get_noise, get_schedule, denoise_cfg

from src.models.lumina.autoencoder import AutoEncoder, ae_params
from src.models.lumina.sampling import time_shift, get_lin_function
from src.math_utils import cosine_optimal_transport
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors


from huggingface_hub import HfApi, upload_file
import time


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
    trained_blocks: int
    save_every: int
    save_folder: str
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_entity: Optional[str] = None
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
    gemma_max_length: int


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

    lumina_path: str
    vae_path: str
    gemma_path: str
    gemma_to_8bit: bool
    gemma_max_length: int


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


def prepare_sot_pairings(latents):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    n, c, h, w = latents.shape

    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=latents.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=latents.device
    )

    # biasing towards earlier more noisy steps where it's the most non linear
    input_timestep = time_shift(get_lin_function()(int(h * w)), 1, input_timestep)

    timesteps = input_timestep[:, None, None, None]
    # 0 is full noise 1 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = noise * (1 - timesteps) + latents * timesteps

    # target vector that being regressed on
    target = latents - noise

    return noisy_latents, target, input_timestep


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function

    # wipe the flag just in case
    for name, param in model.named_parameters():
        param.requires_grad = False
    trained_params = []
    for name, param in model.named_parameters():
        # Exclude 'norm_final.weight' from being included
        if "norm_final.weight" in name:
            param.requires_grad = False

        elif any(keyword in name for keyword in trained_layer_keywords):
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
    gemma_tokenizer,
    gemma,
    seed: int,
    steps: int,
    cfg: int,
    prompts: list,
    rank: int,
    first_n_steps_wo_cfg: int,
    image_dim=(512, 512),
    gemma_max_length=512,
):
    #############################################################################
    # test inference
    # aliasing
    SEED = seed
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg
    DEVICE = model.device
    PROMPT = prompts

    GEMMA_MAX_LENGTH = gemma_max_length

    # store device state of each model
    gemma_device = gemma.device
    ae_device = ae.device
    model_device = model.device
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # init random noise
            noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, SEED)
            noise = noise.to(rank)

            timesteps = get_schedule(STEPS, WIDTH // 16 * HEIGHT // 16)

            model.to("cpu")
            ae.to("cpu")
            gemma.to(rank)  # load gemma to gpu
            text_inputs = gemma_tokenizer(
                PROMPT,
                padding="max_length",
                max_length=GEMMA_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(gemma.device)

            # should be batched but lazy :P
            gemma_embed = gemma(
                text_inputs.input_ids,
                text_inputs.attention_mask,
                output_hidden_states=True,
            ).hidden_states[-2]

            text_inputs_neg = gemma_tokenizer(
                [""] * len(PROMPT),
                padding="max_length",
                max_length=GEMMA_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(gemma.device)

            # <bos> will be the wildcard token and will get affected by CFG
            gemma_embed_neg = gemma(
                text_inputs_neg.input_ids,
                text_inputs_neg.attention_mask,
                output_hidden_states=True,
            ).hidden_states[-2]

            ae.to("cpu")
            gemma.to("cpu")
            model.to(rank)  # load model to gpu
            latent_cfg = denoise_cfg(
                model,
                noise,
                gemma_embed,
                text_inputs.attention_mask,
                gemma_embed_neg,
                text_inputs_neg.attention_mask,
                timesteps,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
            )

            model.to("cpu")
            gemma.to("cpu")
            ae.to(rank)  # load ae to gpu
            output_image = ae.decode(latent_cfg)

            # restore back state
            model.to("cpu")
            gemma.to("cpu")
            ae.to("cpu")

    return output_image


def train_lumina(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config_lumina.json")

    training_config = TrainingConfig(**config_data["training"])
    inference_config = InferenceConfig(**config_data["inference"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    extra_inference_config = [
        InferenceConfig(**conf) for conf in config_data["extra_inference_config"]
    ]

    # wandb logging
    if training_config.wandb_project is not None and rank == 0:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if training_config.wandb_key:
            wandb.login(key=training_config.wandb_key)
        wandb.init(
            project=training_config.wandb_project,
            name=f"{training_config.wandb_run}_{current_datetime}",
            entity=training_config.wandb_entity,
        )

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
        # load lumina and enable grad
        with torch.device("meta"):
            model = Lumina_2b()
        model.set_use_compiled()
        model.load_state_dict(load_safetensors(model_config.lumina_path), assign=True)
        model.to(torch.bfloat16)
        model.train()

        # randomly train inner layers at a time
        trained_blocks = list(range(len(model.layers)))
        random.shuffle(trained_blocks)
        # lazy :P
        trained_blocks = trained_blocks * 1000000

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

        # load gemma
        gemma_tokenizer = GemmaTokenizerFast.from_pretrained(model_config.gemma_path)
        gemma_tokenizer.padding_side = "right"
        gemma = Gemma2Model.from_pretrained(
            model_config.gemma_path, torch_dtype=torch.bfloat16
        )
        gemma.eval()
        gemma.to(torch.bfloat16)
        if model_config.gemma_to_8bit:
            cast_linear(gemma, torch.float8_e4m3fn)

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
            images, caption, index = data[0]
            # just in case the dataloader is failing
            caption = [x if x is not None else "" for x in caption]
            caption = [
                (
                    f"You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> {x}"
                    if x
                    else ""
                )
                for x in caption
            ]
            if counter % training_config.change_layer_every == 0:
                # periodically remove the optimizer and swap it with new one
                # aliasing to make it cleaner
                o_c = optimizer_counter
                n_ld = training_config.trained_blocks
                trained_layer_keywords = [
                    f"layers.{x}."
                    for x in trained_blocks[o_c * n_ld : o_c * n_ld + n_ld]
                ] + [
                    "context_refiner",
                    "noise_refiner",
                    "x_embedder",
                    "norm_final",
                    "final_layer",
                    "t_embedder",
                    "cap_embedder",
                ]

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

                # if counter % training_config.change_layer_every == 0 and counter !=0:
                #     # from torchviz import make_dot
                #     # make_dot(loss, params=dict(model.named_parameters())).render("model_graph", format="png")
                #     print()

                optimizer_counter += 1

            # we load and unload vae and gemma here to reduce vram usage
            # think of this as caching on the fly
            # load gemma and vae to GPU
            ae.to(rank)
            gemma.to(rank)

            acc_latents = []
            acc_embeddings = []
            acc_embed_mask = []
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
                    text_inputs = gemma_tokenizer(
                        caption[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ],
                        padding="max_length",
                        max_length=model_config.gemma_max_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).to(gemma.device)

                    # offload to cpu
                    gemma_embed = (
                        gemma(
                            text_inputs.input_ids,
                            text_inputs.attention_mask,
                            output_hidden_states=True,
                        )
                        .hidden_states[-2]
                        .to("cpu", non_blocking=True)
                    )
                    acc_embeddings.append(gemma_embed)
                    acc_embed_mask.append(
                        text_inputs.attention_mask.to("cpu", non_blocking=True)
                    )

                    # flush
                    torch.cuda.empty_cache()
                    latents = ae.encode_for_train(
                        images[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ].to(rank)
                    ).to("cpu", non_blocking=True)
                    acc_latents.append(latents)

                    # flush
                    torch.cuda.empty_cache()

            # accumulate the latents and embedding in a variable
            # unload gemma and vae

            gemma.to("cpu")
            ae.to("cpu")
            torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            # move model to device
            model.to(rank)

            acc_latents = torch.cat(acc_latents, dim=0)
            acc_embeddings = torch.cat(acc_embeddings, dim=0)
            acc_embed_mask = torch.cat(acc_embed_mask, dim=0)

            # process the cache buffer now!
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                # prepare flat image and the target lerp
                (
                    noisy_latents,
                    target,
                    input_timestep,
                ) = prepare_sot_pairings(acc_latents.to(rank, non_blocking=True))
                noisy_latents = noisy_latents.to(torch.bfloat16)
                target = target.to(torch.bfloat16)
                input_timestep = input_timestep.to(torch.bfloat16)

            # set the input to requires grad to make autograd works
            input_timestep.requires_grad_(True)
            noisy_latents.requires_grad_(True)
            acc_embeddings.requires_grad_(True)

            ot_bs = acc_latents.shape[0]

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
                        x=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        t=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        cap_feats=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        cap_mask=acc_embed_mask[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                    )
                    # TODO: need to scale the loss with rank count and grad accum!
                    loss = (
                        F.mse_loss(
                            pred,
                            target[tmb_i * mb : tmb_i * mb + mb],
                        )
                        / dataloader_config.batch_size
                    )

                torch.cuda.empty_cache()
                loss.backward()
                loss_log.append(loss.detach().clone() * dataloader_config.batch_size)
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

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if training_config.wandb_project is not None and rank == 0:
                wandb.log({"loss": loss_log, "lr": training_config.lr})

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

                for prompt in preview_prompts:
                    images_tensor = inference_wrapper(
                        model=model,
                        ae=ae,
                        gemma_tokenizer=gemma_tokenizer,
                        gemma=gemma,
                        seed=training_config.master_seed + rank,
                        steps=inference_config.steps,
                        cfg=inference_config.cfg,
                        prompts=[prompt],  # Pass single prompt as a list
                        rank=rank,
                        first_n_steps_wo_cfg=inference_config.first_n_steps_wo_cfg,
                        image_dim=inference_config.image_dim,
                        gemma_max_length=inference_config.gemma_max_length,
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
                    for prompt in preview_prompts:
                        images_tensor = inference_wrapper(
                            model=model,
                            ae=ae,
                            gemma_tokenizer=gemma_tokenizer,
                            gemma=gemma,
                            seed=training_config.master_seed + rank,
                            steps=extra_inference.steps,
                            cfg=extra_inference.cfg,
                            prompts=[prompt],  # Pass single prompt as a list
                            rank=rank,
                            first_n_steps_wo_cfg=extra_inference.first_n_steps_wo_cfg,
                            image_dim=extra_inference.image_dim,
                            gemma_max_length=extra_inference.gemma_max_length,
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

                    # upload preview to wandb
                    if training_config.wandb_project is not None:
                        wandb.log(
                            {
                                "example_image": wandb.Image(
                                    file_path,
                                    caption="\n".join(preview_prompts + all_prompt),
                                )
                            }
                        )

            # flush
            acc_latents = []
            acc_embeddings = []

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
