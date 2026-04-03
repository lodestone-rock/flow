"""Binary AR-Diffusion Language Model (BinaryLM AR variant)

Autoregressive backbone + per-token diffusion head.

Architecture overview
─────────────────────
  Backbone  (causal transformer, classic AR):
    - Input tokens are shifted by 1 (teacher-forcing, same as GPT)
    - Causal attention mask so position i only sees tokens 0..i-1
    - Outputs hidden[i]  [dim]  — the "context vector" for predicting token i

  Diffusion head  (per-token, conditioned on backbone hidden):
    head(noisy_token[i], timestep, hidden[i]) → x0_pred[i]

    Internally:
      1. noisy_token [512] → linear up-proj → [head_channels]
      2. condition = hidden[i] [dim] + t_emb [dim]   (summed after projection)
      3. N × ResBlock(x, condition)  with AdaLN modulation
      4. final linear → x0_pred [512]

    This is the same pattern as SimpleMLPAdaLN in model_dct.py but 1-D
    (no patch grid, no NerfEmbedder — just a linear input proj).

Training
────────
  For each position i the head receives:
    - noisy_token[i]  = lerp(noise, x0[i], t[i])   where t[i] ~ U(0,1)
    - timestep        = t[i]
    - hidden[i]       = backbone output at position i
                        (with CFG dropout: zeroed with prob cfg_dropout)

  Loss: MSE on velocity  v = (noisy - x0) / (t + eps)
        computed only on valid (non-padding) positions.

Inference
─────────
  Autoregressive generation token by token:
    for i in 0..T-1:
        hidden[i] = backbone(tokens[0..i])[-1]
        run diffusion ODE for token i+1:
            x = randn(512)
            for each ODE step:
                v_cond = head(x, t, hidden[i])
                v_null = head(x, t, zeros)
                v = v_null + cfg_scale * (v_cond - v_null)
                x = x + v * dt
        tokens[i+1] = round(x)

  Or in parallel (non-AR) for a full sequence given a prefix:
    run backbone on prefix → hidden
    diffuse all suffix positions simultaneously with per-token t
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as ckpt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 64          # chars per diffusion token
BITS_PER_CHAR   = 8           # ASCII = 1 byte = 8 bits
BITS_PER_TOKEN  = CHARS_PER_TOKEN * BITS_PER_CHAR   # 512


# ---------------------------------------------------------------------------
# Text <-> binary helpers
# ---------------------------------------------------------------------------

# Stop token byte value — 0xFF (255).  Upper 128 ASCII values are unused in
# normal text, so 0xFF is a safe sentinel.  During inference, generation stops
# at the first token that contains a 0xFF byte.
STOP_BYTE: int = 0xFF

# Padding byte — 0xFE (254).  Used to align a prefix to a 64-byte token
# boundary before encoding.  The model sees this during training via random
# alignment offsets, so it is never OOD at inference time.
PAD_BYTE: int = 0xFE

# Precomputed bit-unpack table: byte value -> 8 floats in {-1, +1} (MSB first)
# Bit 1 -> +1.0,  Bit 0 -> -1.0
# Shape: [256, 8]  — built once at import time, reused every call
_BIT_TABLE: Tensor = torch.zeros(256, 8, dtype=torch.float32)
for _v in range(256):
    for _b in range(8):
        _BIT_TABLE[_v, 7 - _b] = 1.0 if ((_v >> _b) & 1) else -1.0


def text_to_binary_tokens(
    texts: list[str],
    max_seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode a batch of strings into ASCII binary token tensors.

    Each string is chunked into windows of CHARS_PER_TOKEN (64) characters.
    Each character is encoded as a single byte (ASCII, 8 bits).
    Non-ASCII characters are replaced with '?' (0x3F).
    The bits are packed into a float32 tensor of shape [B, T, 512]
    where values are exactly 0.0 or 1.0.

    Encoding layout per token (512 bits):
      byte 0  -> bits [0:8]   (MSB first)
      byte 1  -> bits [8:16]
      ...
      byte 63 -> bits [504:512]

    Args:
        texts:       List of B strings.
        max_seq_len: Maximum number of tokens (chunks) per sequence.
        device:      Target device.

    Returns:
        tokens:  [B, max_seq_len, BITS_PER_TOKEN]  float32, values in {-1, +1}
        mask:    [B, max_seq_len]                   bool, True = valid token
    """
    B = len(texts)
    tokens = torch.zeros(B, max_seq_len, BITS_PER_TOKEN, dtype=torch.float32)
    mask   = torch.zeros(B, max_seq_len, dtype=torch.bool)

    bit_table = _BIT_TABLE  # [256, 8] on CPU

    for b, text in enumerate(texts):
        # Replace non-ASCII with '?', then chunk into CHARS_PER_TOKEN windows
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        chunks = [
            safe_text[i : i + CHARS_PER_TOKEN]
            for i in range(0, len(safe_text), CHARS_PER_TOKEN)
        ]
        n_chunks = min(len(chunks), max_seq_len)

        for t, chunk in enumerate(chunks[:n_chunks]):
            # Pad to exactly CHARS_PER_TOKEN bytes with null (0x00)
            raw_bytes = chunk.encode("ascii").ljust(CHARS_PER_TOKEN, b"\x00")
            # byte_vals: [CHARS_PER_TOKEN]  int64
            byte_vals = torch.frombuffer(raw_bytes, dtype=torch.uint8).long()  # [64]
            # Look up bits for each byte: [64, 8] -> flatten to [512]
            bits = bit_table[byte_vals].reshape(BITS_PER_TOKEN)  # [512]
            tokens[b, t] = bits
            mask[b, t]   = True

    return tokens.to(device), mask.to(device)


def text_to_binary_tokens_padded(
    texts: list[str],
    max_seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode strings for inference, aligning each to a 64-byte token boundary.

    Prepends PAD_BYTE (0xFE) bytes so that the first real character always
    starts at a clean token boundary.  This matches the random alignment
    offsets injected during training, preventing OOD inputs at inference.

    Alignment rule:
        pad_len = (CHARS_PER_TOKEN - len(safe_bytes) % CHARS_PER_TOKEN) % CHARS_PER_TOKEN
        If the text already fits exactly, no padding is added.

    Args:
        texts:       List of B prefix strings.
        max_seq_len: Maximum number of tokens.
        device:      Target device.

    Returns:
        tokens:  [B, T, BITS_PER_TOKEN]  float32, values in {-1, +1}
        mask:    [B, T]                   bool, True = valid token
    """
    padded_texts = []
    for text in texts:
        safe = text.encode("ascii", errors="replace")
        # How many PAD_BYTE bytes to prepend to reach a 64-byte boundary
        remainder = len(safe) % CHARS_PER_TOKEN
        pad_len   = (CHARS_PER_TOKEN - remainder) % CHARS_PER_TOKEN
        padded    = bytes([PAD_BYTE] * pad_len) + safe
        # Decode back to str for text_to_binary_tokens (which re-encodes)
        # We pass raw bytes directly via a workaround: encode as latin-1
        # so byte values are preserved 1:1
        padded_texts.append(padded.decode("latin-1"))

    # Use latin-1 encoding path: encode each char as its ordinal byte value
    B = len(padded_texts)
    tokens = torch.zeros(B, max_seq_len, BITS_PER_TOKEN, dtype=torch.float32)
    mask   = torch.zeros(B, max_seq_len, dtype=torch.bool)
    bit_table = _BIT_TABLE

    for b, text in enumerate(padded_texts):
        # latin-1 preserves byte values 0x00-0xFF exactly
        raw = text.encode("latin-1")
        chunks = [
            raw[i : i + CHARS_PER_TOKEN]
            for i in range(0, len(raw), CHARS_PER_TOKEN)
        ]
        n_chunks = min(len(chunks), max_seq_len)
        for t, chunk in enumerate(chunks[:n_chunks]):
            chunk_padded = chunk.ljust(CHARS_PER_TOKEN, b"\x00")
            byte_vals    = torch.frombuffer(chunk_padded, dtype=torch.uint8).long()
            tokens[b, t] = bit_table[byte_vals].reshape(BITS_PER_TOKEN)
            mask[b, t]   = True

    return tokens.to(device), mask.to(device)


# Precomputed bit-pack table: 8 bits (MSB first) -> byte value
# Used for fast decoding. Shape: [256] indexed by packed int.
# We reconstruct byte values via vectorised dot product instead.
_BIT_WEIGHTS: Tensor = torch.tensor(
    [128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.float32
)  # MSB-first positional weights


def binary_tokens_to_text(tokens: Tensor, stop_at_stop_byte: bool = True) -> list[str]:
    """Decode ASCII binary token tensor back to strings.

    Tokens are in {-1, +1} space (or continuous floats during diffusion).
    Values > 0 are decoded as bit 1, values <= 0 as bit 0.

    Stop token: byte value 0xFF (255) acts as a document separator.
    When stop_at_stop_byte=True (default), decoding stops at the first 0xFF
    byte and everything after is pruned.  The raw string including the stop
    byte is still returned for debugging via the 'raw' key if needed.

    Args:
        tokens:            [B, T, BITS_PER_TOKEN] float32
        stop_at_stop_byte: If True, truncate at first 0xFF byte.

    Returns:
        List of B decoded strings.
    """
    B, T, _ = tokens.shape
    # Convert {-1,+1} (or continuous) to {0,1}: positive -> 1, non-positive -> 0
    bits = (tokens > 0).float().cpu()             # [B, T, 512]

    # Reshape to [B, T, 64, 8] — one group of 8 bits per byte
    bits = bits.reshape(B, T, CHARS_PER_TOKEN, BITS_PER_CHAR)  # [B, T, 64, 8]

    # Dot product with positional weights to recover byte values: [B, T, 64]
    byte_vals = (bits * _BIT_WEIGHTS).sum(dim=-1).round().long()  # [B, T, 64]

    results = []
    for b in range(B):
        raw_flat = byte_vals[b].reshape(-1).tolist()  # [T*64]
        decoded = bytearray()
        for v in raw_flat:
            v = int(v) & 0xFF
            decoded.append(v)

        if stop_at_stop_byte:
            # Find first stop byte (0xFF) and truncate there
            stop_idx = decoded.find(STOP_BYTE)
            if stop_idx != -1:
                decoded = decoded[:stop_idx]
        else:
            # Strip trailing null padding only
            decoded = decoded.rstrip(b"\x00")

        # Replace non-printable bytes (except stop byte already removed)
        printable = bytearray(
            v if 0x20 <= v <= 0x7E else ord('?') for v in decoded
        )
        results.append(printable.decode("ascii", errors="replace"))

    return results

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BinaryLMARParams:
    """Configuration for BinaryLM AR variant.

    Defaults are GPT-2 124M scale backbone + lightweight diffusion head.
    """
    # Sequence
    max_seq_len: int = 256          # max tokens (each = 64 ASCII chars)

    # Input / output
    in_channels: int = BITS_PER_TOKEN   # 512 bits per token

    # Backbone transformer
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    ffn_mult: float = 4.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 10000.0

    # Diffusion head
    head_channels: int = 768        # internal width of the head ResBlocks
    head_num_res_blocks: int = 4    # number of ResBlocks in the head
    adaln_embed_dim: int = 256      # timestep embedding dim fed into AdaLN
    t_scale: float = 1000.0

    # CFG dropout probability during training
    cfg_dropout: float = 0.1

    # x0 prediction (always True for this model, kept for checkpoint compat)
    use_x0: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → embedding MLP.
    Accepts [B] or [B, T] input, returns matching [..., out_size] output.
    """

    def __init__(self, out_size: int, mid_size: int = 1024, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return emb

    def forward(self, t: Tensor) -> Tensor:
        shape = t.shape
        t_flat = t.reshape(-1)
        t_freq = self.timestep_embedding(t_flat, self.freq_dim)
        dtype = self.mlp[0].weight.dtype
        if dtype.is_floating_point:
            t_freq = t_freq.to(dtype)
        return self.mlp(t_freq).reshape(*shape, -1)


class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.zeros_(self.w2.weight)  # identity init: zero output proj

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _apply_rope_1d(x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tensor:
    B, T, H, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs_cos[:T].unsqueeze(0).unsqueeze(2)
    sin = freqs_sin[:T].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        # persistent=True so these appear in state_dict() and get materialized
        # correctly when using the meta-device init + load_state_dict(assign=True)
        # pattern in the trainer. persistent=False would leave them on meta device.
        self.register_buffer("freqs_cos", freqs.cos(), persistent=True)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=True)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Backbone blocks  (causal attention)
# ---------------------------------------------------------------------------

class CausalAttention(nn.Module):
    """Multi-head causal self-attention with optional GQA and QK-norm."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = dim // n_heads
        self.n_rep      = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        nn.init.zeros_(self.wo.weight)  # identity init: zero output proj

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,                          # [B, T, dim]
        freqs_cos: Tensor,
        freqs_sin: Tensor,
        pad_mask: Optional[Tensor] = None,  # [B, T] bool, True = valid
    ) -> Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = _apply_rope_1d(q, freqs_cos, freqs_sin)
        k = _apply_rope_1d(k, freqs_cos, freqs_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)   # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal mask: upper-triangular -inf  [1, 1, T, T]
        causal = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)

        # Combine with padding mask if provided
        if pad_mask is not None:
            # [B, 1, 1, T]  — mask out padding keys
            pad = torch.zeros(B, 1, 1, T, dtype=q.dtype, device=x.device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float("-inf"))
            attn_mask = causal + pad
        else:
            attn_mask = causal

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class BackboneBlock(nn.Module):
    """Causal transformer block — no AdaLN, plain pre-norm (like GPT)."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: float,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.attn  = CausalAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.ffn   = FeedForward(dim, int(dim * ffn_mult))

    def forward(
        self,
        x: Tensor,
        freqs_cos: Tensor,
        freqs_sin: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, pad_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Diffusion head  (per-token, conditioned on backbone hidden + timestep)
# ---------------------------------------------------------------------------

class HeadResBlock(nn.Module):
    """ResBlock with AdaLN modulation — same pattern as ZImage's ResBlock.

    condition = backbone_hidden + t_emb  (both projected to head_channels)
    AdaLN produces (shift, scale, gate) from condition.
    Initialized to identity (zero modulation output).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.mlp  = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )
        # Zero-init → identity at start of training
        nn.init.zeros_(self.adaLN[-1].weight)   # type: ignore[index]
        nn.init.zeros_(self.adaLN[-1].bias)     # type: ignore[index]
        # Kaiming init for MLP input proj; zero-init output proj for identity residual
        mlp_linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for m in mlp_linears[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(mlp_linears[-1].weight)  # identity init: zero output proj
        nn.init.zeros_(mlp_linears[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    [N, channels]   noisy token hidden state
            cond: [N, channels]   backbone_hidden + t_emb (already projected)
        """
        shift, scale, gate = self.adaLN(cond).chunk(3, dim=-1)
        h = (1 + scale) * self.norm(x) + shift
        h = self.mlp(h)
        return x + gate * h


class DiffusionHead(nn.Module):
    """Per-token diffusion head.

    Takes:
      noisy_token  [N, 512]       — noisy binary vector at timestep t
      backbone_h   [N, hidden_dim] — context from causal backbone
      t_emb        [N, adaln_dim]  — timestep embedding

    Returns:
      x0_pred      [N, 512]       — predicted clean binary vector

    Condition for AdaLN = project(backbone_h) + project(t_emb)
    Both projections go to head_channels so they can be summed.
    """

    def __init__(
        self,
        in_channels: int,       # 512
        hidden_dim: int,        # backbone hidden dim
        head_channels: int,     # internal head width
        adaln_embed_dim: int,   # timestep embedding dim
        num_res_blocks: int,
    ):
        super().__init__()

        # Input: noisy token → head_channels
        self.input_proj = nn.Linear(in_channels, head_channels, bias=True)

        # Learned token bias: a single vector added after input_proj.
        # The head processes one whole token (512 bits) as a flat vector so
        # there is only ever one "position" per call — no index needed.
        # Sequence-level position awareness comes from backbone_h, not here.
        self.token_bias = nn.Parameter(torch.zeros(head_channels))
        nn.init.normal_(self.token_bias, std=0.02)

        # Condition projections (both → head_channels, then summed)
        self.backbone_proj = nn.Linear(hidden_dim,      head_channels, bias=True)
        self.timestep_proj = nn.Linear(adaln_embed_dim, head_channels, bias=True)

        # ResBlocks
        self.res_blocks = nn.ModuleList([
            HeadResBlock(head_channels) for _ in range(num_res_blocks)
        ])

        # Final output layer — zero-init so model starts near identity
        self.final_norm   = nn.LayerNorm(head_channels, eps=1e-6)
        self.final_linear = nn.Linear(head_channels, in_channels, bias=True)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        # Xavier init for projection layers
        for proj in [self.input_proj, self.backbone_proj, self.timestep_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        noisy: Tensor,      # [N, 512]
        backbone_h: Tensor, # [N, hidden_dim]
        t_emb: Tensor,      # [N, adaln_embed_dim]
    ) -> Tensor:
        x    = self.input_proj(noisy) + self.token_bias  # [N, head_channels]
        cond = self.backbone_proj(backbone_h) + self.timestep_proj(t_emb)  # [N, head_channels]

        for block in self.res_blocks:
            x = block(x, cond)

        return self.final_linear(self.final_norm(x))             # [N, 512]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BinaryLMAR(nn.Module):
    """Autoregressive backbone + per-token diffusion head.

    Training (teacher-forced, all positions in parallel):
      1. Shift tokens right by 1  (prepend learned BOS token)
      2. Run causal backbone → hidden[0..T-1]
      3. For each position i, sample t[i] ~ U(0,1)
         With prob cfg_dropout, zero out hidden[i] (null condition)
      4. head(noisy_token[i], t[i], hidden[i]) → x0_pred[i]
      5. Loss: MSE on velocity, valid positions only

    Inference (autoregressive, token by token):
      See run_inference() in the trainer.
    """

    def __init__(self, params: BinaryLMARParams):
        super().__init__()
        self.params = params
        dim = params.hidden_dim

        # ── Backbone ──────────────────────────────────────────────────────
        # Token input embedder: 512 → hidden_dim
        self.x_embedder = nn.Linear(params.in_channels, dim, bias=True)

        # Learned BOS token (prepended as position 0 during training)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.bos_token, std=0.02)

        self.rope = RotaryEmbedding(
            head_dim=params.head_dim,
            max_seq_len=params.max_seq_len,
            theta=params.rope_theta,
        )

        self.backbone_layers = nn.ModuleList([
            BackboneBlock(
                dim=dim,
                n_heads=params.n_heads,
                n_kv_heads=params.n_kv_heads,
                ffn_mult=params.ffn_mult,
                qk_norm=params.qk_norm,
                norm_eps=params.norm_eps,
            )
            for _ in range(params.n_layers)
        ])
        self.backbone_norm = RMSNorm(dim, eps=params.norm_eps)

        # ── Diffusion head ────────────────────────────────────────────────
        self.t_embedder = TimestepEmbedder(
            out_size=params.adaln_embed_dim,
            mid_size=1024,
        )

        self.head = DiffusionHead(
            in_channels     = params.in_channels,
            hidden_dim      = dim,
            head_channels   = params.head_channels,
            adaln_embed_dim = params.adaln_embed_dim,
            num_res_blocks  = params.head_num_res_blocks,
        )

        if params.use_x0:
            self.register_buffer("__x0__", torch.tensor([]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ── Backbone ──────────────────────────────────────────────────────────

    def _run_backbone(
        self,
        tokens: Tensor,             # [B, T, 512]  clean tokens (teacher-forced)
        pad_mask: Optional[Tensor], # [B, T] bool
    ) -> Tensor:
        """Run causal backbone on shifted input.

        Prepends BOS, embeds tokens, runs causal transformer.

        Returns:
            hidden: [B, T, dim]  — hidden[i] is the context for predicting token i
                                   (it has seen tokens 0..i-1 via the shift)
        """
        B, T, _ = tokens.shape

        # Embed tokens: [B, T, 512] → [B, T, dim]
        tok_emb = self.x_embedder(tokens)

        # Shift right: prepend BOS, drop last token
        # BOS: [1, 1, dim] → [B, 1, dim]
        bos = self.bos_token.expand(B, 1, -1)
        # shifted: [B, T, dim]  — position i contains embedding of token i-1
        shifted = torch.cat([bos, tok_emb[:, :-1, :]], dim=1)

        freqs_cos, freqs_sin = self.rope(T)

        h = shifted
        for layer in self.backbone_layers:
            if self.training:
                h = ckpt.checkpoint(
                    layer, h, freqs_cos, freqs_sin, pad_mask,
                    use_reentrant=False,
                )
            else:
                h = layer(h, freqs_cos, freqs_sin, pad_mask)

        return self.backbone_norm(h)   # [B, T, dim]

    # ── Diffusion head helpers ────────────────────────────────────────────

    def _apply_x0_residual(
        self,
        predicted: Tensor,  # [N, 512]  x0 prediction
        noisy: Tensor,      # [N, 512]
        t: Tensor,          # [N]
    ) -> Tensor:
        """x0 → velocity:  v = (noisy - x0) / (t + eps)"""
        eps = 5e-2 if self.training else 0.0
        return (noisy - predicted) / (t.unsqueeze(-1) + eps)

    # ── Training forward ──────────────────────────────────────────────────

    def forward_train(
        self,
        x0: Tensor,                     # [B, T, 512]  clean binary tokens
        t: Tensor,                      # [B, T]        per-token timesteps
        pad_mask: Optional[Tensor],     # [B, T] bool
        cfg_dropout_mask: Tensor,       # [B, T] bool   True = zero out condition
    ) -> Tensor:
        """Full training forward pass.

        Returns:
            pred_v: [B, T, 512]  predicted velocity (v-space x0 residual)
        """
        B, T, C = x0.shape

        # 1. Backbone: get context for each position
        hidden = self._run_backbone(x0, pad_mask)   # [B, T, dim]

        # 2. CFG dropout: zero out condition where mask is True
        hidden = hidden.masked_fill(cfg_dropout_mask.unsqueeze(-1), 0.0)

        # 3. Build noisy tokens
        noise   = torch.randn_like(x0)
        t_exp   = t.unsqueeze(-1)                   # [B, T, 1]
        noisy_x = x0 * (1.0 - t_exp) + noise * t_exp

        # 4. Timestep embedding: [B, T] → [B, T, adaln_dim]
        t_scaled = (1.0 - t) * self.params.t_scale
        t_emb    = self.t_embedder(t_scaled)        # [B, T, adaln_dim]

        # 5. Run head on all positions in parallel
        # Flatten batch × time for the head: [B*T, ...]
        N = B * T
        noisy_flat  = noisy_x.reshape(N, C)
        hidden_flat = hidden.reshape(N, -1)
        t_emb_flat  = t_emb.reshape(N, -1)
        x0_pred_flat = self.head(noisy_flat, hidden_flat, t_emb_flat)  # [N, 512]
        x0_pred = x0_pred_flat.reshape(B, T, C)

        # 6. Convert x0 → velocity
        pred_v = self._apply_x0_residual(
            x0_pred.reshape(N, C),
            noisy_x.reshape(N, C),
            t.reshape(N),
        ).reshape(B, T, C)

        return pred_v

    def forward_head_only(
        self,
        noisy: Tensor,      # [N, 512]  noisy token
        backbone_h: Tensor, # [N, dim]  backbone hidden (or zeros for null)
        t: Tensor,          # [N]       timestep
    ) -> Tensor:
        """Run only the diffusion head (used during AR inference).

        Returns x0 prediction [N, 512].
        """
        t_scaled = (1.0 - t) * self.params.t_scale
        t_emb    = self.t_embedder(t_scaled)          # [N, adaln_dim]
        return self.head(noisy, backbone_h, t_emb)    # [N, 512]
