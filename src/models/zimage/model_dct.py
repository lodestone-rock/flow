"""Z-Image Transformer with SimpleMLPAdaLN decoder head (DCT variant)."""

import math
from typing import List, Optional
from functools import lru_cache
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as ckpt


@dataclass
class ZImageDCTParams:
    patch_size: int = 1
    f_patch_size: int = 1
    in_channels: int = 128
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: int = 256
    t_scale: float = 1000.0
    axes_dims: list[int] = field(default_factory=lambda: [32, 48, 48])
    axes_lens: list[int] = field(default_factory=lambda: [1536, 512, 512])
    adaln_embed_dim: int = 256
    use_x0: bool = True
    # DCT decoder params
    decoder_hidden_size: int = 3840
    decoder_num_res_blocks: int = 4
    decoder_max_freqs: int = 8


# Default params instance
z_image_dct_params = ZImageDCTParams()


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype):
    if attn_mask is None:
        return None

    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]

    if attn_mask.dtype == torch.bool:
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
            index = ids[..., i]
            result.append(self.freqs_cis[i][index])

        return torch.cat(result, dim=-1)


# ============================================================================
# Decoder Components
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    This module takes an input tensor of shape (B, P^2, C), where P is the
    patch size, and enriches it with positional information before projecting
    it to a new hidden size.
    """
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        """Generates and caches 2D DCT-like positional embeddings."""
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        
        return dct

    def forward(self, inputs):
        B, P2, C = inputs.shape
        original_dtype = inputs.dtype
        
        with torch.autocast("cuda", enabled=False):
            patch_size = int(P2 ** 0.5)
            inputs = inputs.float()
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            dct = dct.repeat(B, 1, 1)
            inputs = torch.cat([inputs, dct], dim=-1)
            inputs = self.embedder.float()(inputs)
        
        return inputs.to(original_dtype)


class ResBlock(nn.Module):
    """
    A residual block with AdaLN modulation.
    Initialized to identity (zero modulation output).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for MLP
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Zero init for modulation (identity at init)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class DCTFinalLayer(nn.Module):
    """The final layer adopted from DiT."""
    
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        
        # Zero init for output layer
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP decoder for Z-Image DCT variant.
    Uses NerfEmbedder for input projection and ResBlocks with AdaLN.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        max_freqs=8,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        # Condition embedding from transformer backbone
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)

        # NerfEmbedder for input projection with DCT positional encoding
        self.input_embedder = NerfEmbedder(
            in_channels=in_channels,
            hidden_size_input=model_channels,
            max_freqs=max_freqs
        )
        
        # Residual blocks with AdaLN (each block handles its own init)
        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])

        # Final layer (handles its own init)
        self.final_layer = DCTFinalLayer(model_channels, out_channels)
        
        # Init cond_embed with xavier
        nn.init.xavier_uniform_(self.cond_embed.weight)
        nn.init.constant_(self.cond_embed.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x P^2 x C] Tensor of inputs (pixel values per patch).
        :param c: conditioning from transformer backbone [N x z_channels].
        :return: an [N x P^2 x C] Tensor of outputs.
        """
        # Project input with DCT positional encoding
        x = self.input_embedder(x)
        
        # Embed condition and reshape for per-position modulation
        c = self.cond_embed(c)
        y = c.reshape(c.shape[0], self.patch_size**2, -1)

        # Pass through residual blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = ckpt.checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x)


# ============================================================================
# Main ZImageDCT Model
# ============================================================================

class ZImageDCT(nn.Module):
    def __init__(self, params: ZImageDCTParams):
        super().__init__()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.patch_size = params.patch_size
        self.f_patch_size = params.f_patch_size
        self.dim = params.dim
        self.n_heads = params.n_heads
        self.rope_theta = params.rope_theta
        self.t_scale = params.t_scale
        self.adaln_embed_dim = params.adaln_embed_dim

        # Input embedder for the backbone (same as original)
        self.x_embedder = nn.Linear(
            self.f_patch_size * self.patch_size * self.patch_size * params.in_channels,
            params.dim,
            bias=True,
        )

        # Noise refiner blocks
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

        # Context refiner blocks
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

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(
            min(params.dim, params.adaln_embed_dim), mid_size=1024
        )

        # Caption embedder
        self.cap_embedder = nn.Sequential(
            RMSNorm(params.cap_feat_dim, eps=params.norm_eps),
            nn.Linear(params.cap_feat_dim, params.dim, bias=True),
        )

        # Padding tokens (kept for compatibility)
        self.x_pad_token = nn.Parameter(torch.empty((1, params.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, params.dim)))

        # Main transformer layers
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

        # RoPE embedder
        head_dim = params.dim // params.n_heads
        assert head_dim == sum(params.axes_dims)
        self.axes_dims = params.axes_dims
        self.axes_lens = params.axes_lens

        self.rope_embedder = RopeEmbedder(
            theta=params.rope_theta,
            axes_dims=params.axes_dims,
            axes_lens=params.axes_lens,
        )

        # SimpleMLPAdaLN decoder head with NerfEmbedder
        # Uses DCT positional encoding for input projection
        self.dec_net = SimpleMLPAdaLN(
            in_channels=params.in_channels,
            model_channels=params.decoder_hidden_size,
            out_channels=params.in_channels,
            z_channels=params.dim,
            num_res_blocks=params.decoder_num_res_blocks,
            patch_size=self.patch_size,
            max_freqs=params.decoder_max_freqs,
            grad_checkpointing=False
        )

        if params.use_x0:
            print("the model is using x0 prediction")
            self.register_buffer("__x0__", torch.tensor([]))

    @property
    def device(self):
        return next(self.parameters()).device

    def _forward(
        self,
        img: Tensor,  # image tensor [B, N, C*P*P] where N is num patches
        img_ids: Tensor,  # img id is 3d where the first dim is text, second and 3rd is the hw
        img_mask: Tensor,  # mask for image
        txt: Tensor,  # embedding from text encoder
        txt_ids: Tensor,  # text id starts at 1 for the backbone
        txt_mask: Tensor,  # mask the padding tensor
        timesteps: Tensor,  # 1-0 scale 1 is full noise 0 is full image
    ):
        B = img.shape[0]
        num_patches = img.shape[1]
        
        # Store raw pixel values for the decoder
        # img shape: [B, N, C*P*P] -> reshape to [B*N, P*P, C]
        pixel_values = img.reshape(B * num_patches, self.patch_size ** 2, self.in_channels)
        
        # zimage use 0-1000 standard where 0 is full noise and 1000 is full image
        timesteps = (1 - timesteps)  # flips the timestep
        timesteps *= self.t_scale  # scaling from 0-1 to 0-1000
        timesteps_embedding = self.t_embedder(timesteps)

        # Embed input for backbone
        img_hidden = self.x_embedder(img)
        txt_hidden = self.cap_embedder(txt)

        img_pe = self.rope_embedder(img_ids)
        txt_pe = self.rope_embedder(txt_ids)

        # Noise input precond transformer
        for layer in self.noise_refiner:
            if self.training:
                img_hidden = ckpt.checkpoint(
                    layer, img_hidden, img_mask, img_pe, timesteps_embedding
                )
            else:
                img_hidden = layer(img_hidden, img_mask, img_pe, timesteps_embedding)

        # Text input precond transformer
        for layer in self.context_refiner:
            if self.training:
                txt_hidden = ckpt.checkpoint(layer, txt_hidden, txt_mask, txt_pe)
            else:
                txt_hidden = layer(txt_hidden, txt_mask, txt_pe)

        # Fuse everything
        mixed_hidden = torch.cat((txt_hidden, img_hidden), 1)
        mixed_mask = torch.cat((txt_mask, img_mask), 1)
        mixed_pe = torch.cat((txt_pe, img_pe), 1)

        # Main transformer layers
        for layer in self.layers:
            if self.training:
                mixed_hidden = ckpt.checkpoint(
                    layer, mixed_hidden, mixed_mask, mixed_pe, timesteps_embedding
                )
            else:
                mixed_hidden = layer(
                    mixed_hidden, mixed_mask, mixed_pe, timesteps_embedding
                )

        # Extract image hidden states (cut out text)
        img_hidden = mixed_hidden[:, txt.shape[1]:, ...]  # [B, N, dim]

        # Prepare for decoder
        # Reshape hidden states: [B, N, dim] -> [B*N, dim]
        decoder_condition = img_hidden.reshape(B * num_patches, self.dim)

        # Pass through SimpleMLPAdaLN decoder (NerfEmbedder handles input projection)
        if self.training:
            output = ckpt.checkpoint(
                self.dec_net,
                pixel_values,
                decoder_condition
            )
        else:
            output = self.dec_net(pixel_values, decoder_condition)

        # Reshape output back: [B*N, P*P, C] -> [B, N, C*P*P]
        output = output.reshape(B, num_patches, -1)

        # Flip the output vectors (same as original)
        return -output

    def _apply_x0_residual(self, predicted, noisy, timesteps):
        eps = 5e-2 if self.training else 0.0
        return (noisy - predicted) / (timesteps.view(-1, 1, 1) + eps)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        img_mask: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
    ):
        out = self._forward(
            img=img,
            img_ids=img_ids,
            img_mask=img_mask,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=timesteps,
        )

        # If x0 variant -> v-pred
        if hasattr(self, "__x0__"):
            return self._apply_x0_residual(out, img, timesteps)

        return out
