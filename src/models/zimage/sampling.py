import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import ZImage


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    latent_depth: int = 16,
    spatial_compression: int = 8,
):
    return torch.randn(
        num_samples,
        latent_depth,
        # allow for packing
        math.ceil(height / spatial_compression),
        math.ceil(width / spatial_compression),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise_cfg(
    model: ZImage,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    neg_txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4,
):
    # this is ignored for schnell
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    step_count = 0
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
        # disable cfg for x steps before using cfg
        if step_count < first_n_steps_without_cfg or first_n_steps_without_cfg == -1:
            img = img.to(pred) + (t_prev - t_curr) * pred
        else:
            pred_neg = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                txt_mask=neg_txt_mask,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            pred_cfg = pred_neg + (pred - pred_neg) * cfg

            img = img + (t_prev - t_curr) * pred_cfg

        step_count += 1

    return img


def denoise_cfg_batched_timesteps(
    model: ZImage,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    neg_txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    # sampling parameters
    timesteps: Tensor,  # Shape: (B, N), where N is the number of time points
    guidance: float = 0.0,
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4,
):
    """
    Performs ODE solving using the Euler method with Classifier-Free Guidance (CFG)
    and potentially different timestep sequences for each sample in the batch.

    Args:
        model: The flow matching model.
        img: Input tensor (e.g., noise) shape (B, C, H, W).
        img_ids: Image IDs tensor, shape (B, ...).
        txt: Positive text conditioning tensor, shape (B, L, D).
        neg_txt: Negative text conditioning tensor, shape (B, L, D).
        txt_ids: Positive text IDs tensor, shape (B, L).
        neg_txt_ids: Negative text IDs tensor, shape (B, L).
        txt_mask: Positive text mask tensor, shape (B, L).
        neg_txt_mask: Negative text mask tensor, shape (B, L).
        timesteps: Tensor containing the time points for each batch sample.
                   Shape (B, N), where B is the batch size and N is the
                   number of time points (e.g., [t_start, ..., t_end]).
                   Time should generally decrease (e.g., [1.0, 0.8, ..., 0.0]).
        guidance: Guidance strength passed to the model (potentially ignored).
        cfg: Classifier-Free Guidance scale. A value of 1.0 disables CFG.
        first_n_steps_without_cfg: The number of initial integration steps
                                   (intervals) for which CFG will *not* be
                                   applied, even if cfg > 1.0. Set to 0 to
                                   apply CFG from the start, or -1 to always
                                   apply CFG (if cfg > 1.0).
    Returns:
        Denoised image tensor, shape (B, C, H, W).
    """
    batch_size = img.shape[0]
    num_time_points = timesteps.shape[1]
    num_steps = num_time_points - 1  # Number of integration steps

    # --- Input Validation ---
    if timesteps.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: img has {batch_size}, "
            f"but timesteps has {timesteps.shape[0]}"
        )
    if timesteps.ndim != 2:
        raise ValueError(
            f"timesteps tensor must be 2D (B, N), but got shape {timesteps.shape}"
        )
    # Check consistency of conditioning tensors
    for name, tensor in [
        ("txt", txt),
        ("neg_txt", neg_txt),
        ("txt_ids", txt_ids),
        ("neg_txt_ids", neg_txt_ids),
        ("txt_mask", txt_mask),
        ("neg_txt_mask", neg_txt_mask),
    ]:
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: img has {batch_size}, "
                f"but {name} has {tensor.shape[0]}"
            )
    # --- End Validation ---

    # Guidance vector (its effect depends on the model)
    guidance_vec = torch.full(
        (batch_size,), guidance, device=img.device, dtype=img.dtype
    )

    # Ensure timesteps tensor is on the same device and dtype as img
    timesteps = timesteps.to(device=img.device, dtype=img.dtype)

    # Iterate through the integration steps (intervals)
    for i in range(num_steps):
        # Get the current time for each batch element
        t_curr_batch = timesteps[:, i]  # Shape: (B,)
        # Get the next time for each batch element
        t_next_batch = timesteps[:, i + 1]  # Shape: (B,)

        # --- Positive Prediction ---
        pred_pos = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_curr_batch,  # Batched timesteps
            guidance=guidance_vec,
        )

        # --- CFG Logic ---
        # Determine if CFG should be applied in this step
        # Apply CFG if cfg > 1.0 AND (we are past the initial steps OR first_n_steps_without_cfg is -1)
        apply_cfg = cfg > 1.0 and (
            i >= first_n_steps_without_cfg or first_n_steps_without_cfg == -1
        )

        if apply_cfg:
            # --- Negative Prediction ---
            pred_neg = model(
                img=img,  # Use the *same* input image state as for positive pred
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                txt_mask=neg_txt_mask,
                timesteps=t_curr_batch,  # Use the same batched timesteps
                guidance=guidance_vec,  # Pass guidance here too
            )
            # Combine predictions using CFG formula
            # pred = uncond + cfg * (cond - uncond)
            pred_final = pred_neg + cfg * (pred_pos - pred_neg)
        else:
            # If not applying CFG, use the positive prediction directly
            pred_final = pred_pos
        # --- End CFG Logic ---

        # Calculate the step size (dt) for each batch element
        dt_batch = t_next_batch - t_curr_batch  # Shape: (B,)

        # Reshape dt for broadcasting: (B,) -> (B, 1, 1)
        dt_batch_reshaped = dt_batch.view(batch_size, 1, 1)

        # Euler step update: x_{t+1} = x_t + dt * v(x_t, t)
        # Ensure img is on the correct device/dtype if pred_final changes it (unlikely but safe)
        img = img.to(pred_final) + dt_batch_reshaped * pred_final

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
