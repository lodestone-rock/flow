"""Yanked and modified from Z-Image Transformer implementation."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn
from dataclasses import dataclass
from typing import Tuple
import torch.utils.checkpoint as ckpt
from einops import rearrange


@dataclass
class ZImageParams:
    all_patch_size: Tuple[int, ...]
    all_f_patch_size: Tuple[int, ...]
    in_channels: int
    dim: int
    n_layers: int
    n_refiner_layers: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    qk_norm: bool
    cap_feat_dim: int
    rope_theta: int
    t_scale: float
    axes_dims: list[int]
    axes_lens: list[int]
    adaln_embed_dim: int


z_image_params = ZImageParams(
    all_patch_size=(2,),
    all_f_patch_size=(1,),
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256,
    t_scale=1000.0,
    axes_dims=[32, 48, 48],
    axes_lens=[1536, 512, 512],
    adaln_embed_dim=256,
)


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype):
    if attn_mask is None:
        return None

    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]

    # Convert bool mask to float additive mask
    if attn_mask.dtype == torch.bool:
        # NOTE: We skip checking for all-True mask (torch.all) to avoid graph breaks in torch.compile
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask

    return attn_mask


def _native_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn_mask = _process_mask(attn_mask, query.dtype)

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    return out.transpose(1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList(
            [nn.Linear(n_heads * self.head_dim, dim, bias=False)]
        )

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        hidden_states = _native_attention_wrapper(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        return output


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        adaln_embed_dim=256,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            # lightweight modulation
            self.adaLN_modulation = nn.ModuleList(
                [nn.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True)]
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, adaln_embed_dim=256):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, adaln_embed_dim), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256,
        axes_dims: List[int] = [32, 48, 48],
        axes_lens: List[int] = [1536, 512, 512],
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(
                    torch.complex64
                )
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        # Modified: Allow 2D [Seq, Axes] or 3D [Batch, Seq, Axes]
        assert ids.ndim >= 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims, self.axes_lens, theta=self.theta
            )
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            # Modified: Use '...' to select the column for the specific axis
            # regardless of batch dimensions.
            # If ids is [B, S, 3], index becomes [B, S].
            # If ids is [S, 3], index becomes [S].
            index = ids[..., i]

            # PyTorch advanced indexing:
            # table: [MaxLen, Dim]
            # index: [B, S]
            # output: [B, S, Dim]
            result.append(self.freqs_cis[i][index])

        return torch.cat(result, dim=-1)


class ZImage(nn.Module):
    def __init__(self, params: ZImageParams):
        super().__init__()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.all_patch_size = params.all_patch_size
        self.all_f_patch_size = params.all_f_patch_size
        self.dim = params.dim
        self.n_heads = params.n_heads
        self.rope_theta = params.rope_theta
        self.t_scale = params.t_scale
        self.adaln_embed_dim = params.adaln_embed_dim

        assert len(params.all_patch_size) == len(params.all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        # seems for ablations here, gonna keep it
        for patch_size, f_patch_size in zip(
            params.all_patch_size, params.all_f_patch_size
        ):
            x_embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * params.in_channels,
                params.dim,
                bias=True,
            )
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(
                params.dim,
                patch_size * patch_size * f_patch_size * self.out_channels,
                adaln_embed_dim=params.adaln_embed_dim,
            )
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=True,
                    adaln_embed_dim=params.adaln_embed_dim,
                )
                for layer_id in range(params.n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=False,
                )
                for layer_id in range(params.n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(
            min(params.dim, params.adaln_embed_dim), mid_size=1024
        )
        self.cap_embedder = nn.Sequential(
            RMSNorm(params.cap_feat_dim, eps=params.norm_eps),
            nn.Linear(params.cap_feat_dim, params.dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, params.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, params.dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=True,
                    adaln_embed_dim=params.adaln_embed_dim,
                )
                for layer_id in range(params.n_layers)
            ]
        )

        head_dim = params.dim // params.n_heads
        assert head_dim == sum(params.axes_dims)
        self.axes_dims = params.axes_dims
        self.axes_lens = params.axes_lens

        self.rope_embedder = RopeEmbedder(
            theta=params.rope_theta,
            axes_dims=params.axes_dims,
            axes_lens=params.axes_lens,
        )

    def forward(
        self,
        img: Tensor,  # image tensor
        img_ids: Tensor,  # img id is 3d where the first dim is text, second and 3rd is the hw
        img_mask: Tensor,  # mask for image
        txt: Tensor,  # embedding from text encoder
        txt_ids: Tensor,  # text id starts at 1 for the backbone
        txt_mask: Tensor,  # mask the padding tensor
        timesteps: Tensor,  # 1-0 scale 1 is full noise 0 is full image
    ):
        # zimage use 0-1000 standard where 0 is full noise and 1000 is full image
        timesteps = (
            1 - timesteps
        )  # flips the timestep so i don't have to change the code
        timesteps *= self.t_scale  # scaling it from 0-1 to 0-1000
        timesteps_embedding = self.t_embedder(timesteps)
        # seems they have plan for different projection, gonna keep the name for compatibility
        img_hidden = self.all_x_embedder[f"2-1"](img)
        txt_hidden = self.cap_embedder(txt)

        img_pe = self.rope_embedder(img_ids)
        txt_pe = self.rope_embedder(txt_ids)

        # noise input precond transformer
        for layer in self.noise_refiner:
            if self.training:
                img_hidden = ckpt.checkpoint(
                    layer, img_hidden, img_mask, img_pe, timesteps_embedding
                )
            else:
                img_hidden = layer(img_hidden, img_mask, img_pe, timesteps_embedding)

        # text input precond transformer
        for layer in self.context_refiner:
            if self.training:
                txt_hidden = ckpt.checkpoint(layer, txt_hidden, txt_mask, txt_pe)
            else:
                txt_hidden = layer(txt_hidden, txt_mask, txt_pe)

        # fuse everything
        mixed_hidden = torch.cat((txt_hidden, img_hidden), 1)
        mixed_mask = torch.cat((txt_mask, img_mask), 1)
        mixed_pe = torch.cat((txt_pe, img_pe), 1)

        # pipe it into classical transformers
        for layer in self.layers:
            if self.training:
                mixed_hidden = ckpt.checkpoint(
                    layer, mixed_hidden, mixed_mask, mixed_pe, timesteps_embedding
                )
            else:
                mixed_hidden = layer(
                    mixed_hidden, mixed_mask, mixed_pe, timesteps_embedding
                )

        # BLD cut out the text and only return the image tensor
        img_hidden = mixed_hidden[:, txt.shape[1] :, ...]

        # final layer then project back to data space
        if self.training:
            output = ckpt.checkpoint(
                self.all_final_layer[f"2-1"],
                mixed_hidden[:, txt.shape[1] :, ...],
                timesteps_embedding,
            )
        else:
            output = self.all_final_layer[f"2-1"](
                mixed_hidden[:, txt.shape[1] :, ...], timesteps_embedding
            )

        # flip the output vectors
        # this took way too long to figure out 
        # why the vector direction is wrong
        return -output
