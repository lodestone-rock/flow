import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def vae_flatten(latents, patch_size=2):
    # nchw to nhwc then pixel shuffle 2 then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (dh dw c)
    # n h w c -> n (h w) c
    # n, c, h, w = latents.shape
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (dh dw c)",
            dh=patch_size,
            dw=patch_size,
        ),
        latents.shape,
    )


def vae_unflatten(latents, shape, patch_size=2):
    n, c, h, w = shape
    # The input latents have shape [N, SeqLen, Dim]
    # The Dim is packed as (dh dw c) because patchify put C at the end.
    return rearrange(
        latents,
        "n (h w) (dh dw c) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


def prepare_latent_image_ids(start_indices, height, width, patch_size=2, max_offset=0):
    """
    Generates positional embeddings for a latent image.

    Args:
        start_indices (list or torch.Tensor): The starting index for each image in the batch (e.g., [2, 10, 20]).
        height (int): The height of the image.
        width (int): The width of the image.
        patch_size (int, optional): The size of the patches. Defaults to 2.
        max_offset (int, optional): The maximum random offset to apply. Defaults to 0.

    Returns:
        torch.Tensor: A tensor containing the positional embeddings.
    """
    # Convert to tensor if it's a list
    if isinstance(start_indices, list):
        start_indices = torch.tensor(start_indices)

    batch_size = len(start_indices)

    # the random pos embedding helps generalize to larger res without training at large res
    # pos embedding for rope, 2d pos embedding, corner embedding and not center based
    latent_image_ids = torch.zeros(height // patch_size, width // patch_size, 3)

    # Add positional encodings
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // patch_size)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // patch_size)[None, :]
    )

    # Add random offset if specified
    if max_offset > 0:
        offset_y = torch.randint(0, max_offset + 1, (1,)).item()
        offset_x = torch.randint(0, max_offset + 1, (1,)).item()
        latent_image_ids[..., 1] += offset_y
        latent_image_ids[..., 2] += offset_x

    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    # Reshape for batch
    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)

    # Set the first dimension to the start_indices for each batch item
    for i, start_idx in enumerate(start_indices):
        latent_image_ids[i, :, :, 0] = start_idx

    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.int()


def make_text_position_ids(valid_len, max_sequence_length, extra_padding=0):
    # will return [this dim increment then repeat, 0, 0]
    device = valid_len.device
    valid_len += extra_padding
    B = valid_len.shape[0]
    seq = torch.arange(1, max_sequence_length + 1, device=device)  # [L]
    seq = seq.unsqueeze(0).expand(B, max_sequence_length)  # [B, L]

    increment_then_repeat = torch.minimum(seq, valid_len.unsqueeze(1))
    pos_ids = torch.zeros((B, max_sequence_length, 3), device=device)  # [B, L, D]
    pos_ids[:, :, 0] = increment_then_repeat
    return pos_ids.int()


print()
