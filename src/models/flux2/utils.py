import math
import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def vae_flatten(latents, patch_size=1):
    # nchw to nhwc then pixel shuffle 2 then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (c dh dw)
    # n h w c -> n (h w) c
    # n, c, h, w = latents.shape
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (c dh dw)",
            dh=patch_size,
            dw=patch_size,
        ),
        latents.shape,
    )


def vae_unflatten(latents, shape, patch_size=1):
    # reverse of that operator above
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


def prepare_latent_image_ids(start_indices, height, width, patch_size=2, max_offset=0, device=None):
    """
    Generates positional embeddings for a latent image.
    pos id format: [time, height, width, text]
    for img: [t, h, w, 0] where text dim is dummy (0)

    Args:
        start_indices (list or torch.Tensor): The starting index for each image in the batch (e.g., [2, 10, 20]).
        height (int): The height of the image.
        width (int): The width of the image.
        patch_size (int, optional): The size of the patches. Defaults to 2.
        max_offset (int, optional): The maximum random offset to apply. Defaults to 0.
        device: The device to place the tensor on.

    Returns:
        torch.Tensor: A tensor containing the positional embeddings with shape [B, H*W, 4].
    """
    # Convert to tensor if it's a list
    if isinstance(start_indices, list):
        start_indices = torch.tensor(start_indices, device=device)
    
    if device is None:
        device = start_indices.device

    batch_size = len(start_indices)
    h = height // patch_size
    w = width // patch_size

    # Add random offset if specified
    offset_y = 0
    offset_x = 0
    if max_offset > 0:
        offset_y = torch.randint(0, max_offset + 1, (1,)).item()
        offset_x = torch.randint(0, max_offset + 1, (1,)).item()

    # Build coordinate grids using cartesian_prod pattern
    # pos id [time, height, width, text] - for img: [t, h, w, 0]
    h_coords = torch.arange(h, device=device) + offset_y
    w_coords = torch.arange(w, device=device) + offset_x
    l_coords = torch.arange(1, device=device)  # dummy text dimension

    # Generate position ids for each batch item
    all_ids = []
    for t_coord in start_indices:
        t = torch.tensor([t_coord], device=device)
        ids = torch.cartesian_prod(t, h_coords, w_coords, l_coords)  # [H*W, 4]
        all_ids.append(ids)
    
    latent_image_ids = torch.stack(all_ids, dim=0)  # [B, H*W, 4]

    return latent_image_ids.int()


def make_text_position_ids(valid_len, max_sequence_length, extra_padding=0, t_coord=None):
    """
    Generates positional embeddings for text sequences.
    pos id format: [time, height, width, text]
    for text: [t, 0, 0, l] where height and width are dummy dims

    Args:
        valid_len: Tensor of valid lengths for each batch item.
        max_sequence_length: Maximum sequence length.
        extra_padding: Extra padding to add to valid lengths.
        t_coord: Optional time coordinate tensor. If None, uses 0.

    Returns:
        torch.Tensor: A tensor containing the positional embeddings with shape [B, L, 4].
    """
    device = valid_len.device
    valid_len = valid_len + extra_padding
    B = valid_len.shape[0]

    # pos id [time, height, width, text] - for text: [t, 0, 0, l]
    coords = {
        "t": torch.arange(1, device=device) if t_coord is None else t_coord,
        "h": torch.arange(1, device=device),  # dummy dimension
        "w": torch.arange(1, device=device),  # dummy dimension
        "l": torch.arange(max_sequence_length, device=device),
    }
    
    # Generate base position ids using cartesian_prod
    base_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])  # [L, 4]
    
    # Expand for batch
    pos_ids = base_ids.unsqueeze(0).expand(B, -1, -1).clone()  # [B, L, 4]
    
    # Clamp text positions based on valid_len (increment then repeat pattern)
    seq = torch.arange(1, max_sequence_length + 1, device=device)  # [L]
    seq = seq.unsqueeze(0).expand(B, max_sequence_length)  # [B, L]
    clamped_text_pos = torch.minimum(seq, valid_len.unsqueeze(1))  # [B, L]
    pos_ids[:, :, 3] = clamped_text_pos
    
    return pos_ids.int()


if __name__ == "__main__":
    # Test prepare_latent_image_ids
    print("=" * 50)
    print("Testing prepare_latent_image_ids")
    print("=" * 50)
    
    start_indices = [0, 1, 2]  # batch of 3
    height, width = 8, 8  # small image for testing
    patch_size = 2
    
    img_ids = prepare_latent_image_ids(start_indices, height, width, patch_size=patch_size)
    print(f"Input: start_indices={start_indices}, height={height}, width={width}, patch_size={patch_size}")
    print(f"Output shape: {img_ids.shape}")  # Expected: [3, 16, 4]
    print(f"Format: [time, height, width, text]")
    print(f"\nBatch 0 (t=0):")
    print(img_ids[0])
    print(f"\nBatch 1 (t=1):")
    print(img_ids[1])
    
    # Test with offset
    print("\n" + "=" * 50)
    print("Testing prepare_latent_image_ids with max_offset=5")
    print("=" * 50)
    img_ids_offset = prepare_latent_image_ids([0], height, width, patch_size=patch_size, max_offset=5)
    print(f"Output shape: {img_ids_offset.shape}")
    print(f"Sample positions (may have random offset):")
    print(img_ids_offset[0])

    # Test make_text_position_ids
    print("\n" + "=" * 50)
    print("Testing make_text_position_ids")
    print("=" * 50)
    
    valid_len = torch.tensor([3, 5, 2])  # batch of 3 with different valid lengths
    max_seq_len = 8
    
    text_ids = make_text_position_ids(valid_len, max_seq_len)
    print(f"Input: valid_len={valid_len.tolist()}, max_sequence_length={max_seq_len}")
    print(f"Output shape: {text_ids.shape}")  # Expected: [3, 8, 4]
    print(f"Format: [time, height, width, text]")
    print(f"\nBatch 0 (valid_len=3):")
    print(text_ids[0])
    print(f"\nBatch 1 (valid_len=5):")
    print(text_ids[1])
    print(f"\nBatch 2 (valid_len=2):")
    print(text_ids[2])
    
    # Test with extra_padding
    print("\n" + "=" * 50)
    print("Testing make_text_position_ids with extra_padding=2")
    print("=" * 50)
    text_ids_padded = make_text_position_ids(valid_len, max_seq_len, extra_padding=2)
    print(f"Batch 0 (valid_len=3+2=5):")
    print(text_ids_padded[0])
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

