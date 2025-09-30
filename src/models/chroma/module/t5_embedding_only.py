# forcefully yanked from HF

import math

import torch
from torch import nn
import torch.utils.checkpoint as ckpt

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN


# reusing t5 config 
class T5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a T5 embedding model.
    It is used to instantiate a T5 embedding model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling the embedding model.
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
    """

    model_type = "t5"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`.
        # It is kept for backward compatibility
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class T5EmbeddingWithRoPE(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_emb = RotaryEmbedding(config.d_model)

    def forward(self, input_ids, attention_mask=None):
        # 1. Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. Apply Rotary Position Embedding
        batch_size, seq_length, _ = hidden_states.shape
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length)
        
        # The rotary embedding function expects a 4D tensor, so we unsqueeze
        # to add a dummy "num_heads" dimension.
        hidden_states = hidden_states.unsqueeze(1) # [batch, 1, seq_len, dim]
        
        positional_embeddings = apply_rotary_pos_emb(hidden_states, cos, sin, position_ids)
        
        # Remove the dummy dimension
        positional_embeddings = positional_embeddings.squeeze(1) # [batch, seq_len, dim]
        
        return positional_embeddings


class T5EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # just to match keys hierarchy
        self.encoder = T5EmbeddingWithRoPE(config)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

from safetensors.torch import save_file, load_file, safe_open

def load_safetensors(file_path: str, device: str = "cpu"):
    statedict = {}

    with safe_open(file_path, framework="pt", device=device) as f:
        # Iterate over all keys and select layers that match the criteria
        for layer_name in f.keys():
            statedict[layer_name] = f.get_tensor(layer_name)

    return statedict

# Example Usage:
if __name__ == '__main__':
    # 1. Initialize T5Config
    config = T5Config.from_json_file("models/flux/text_encoder_2/config.json")

    # 2. Create the embedding model
    embedding_model = T5EncoderModel(config)
    embedding_model.load_state_dict(load_safetensors("models/flux/text_encoder_2/model-00001-of-00002.safetensors"), strict=False)

    # 3. Create a dummy input
    input_ids = torch.randint(0, config.vocab_size, (4, 128)) # Batch size of 4, sequence length of 128

    # 4. Forward pass
    output_embeddings = embedding_model(input_ids)

    # 5. Print the output shape
    print("Shape of output embeddings:", output_embeddings.shape)
    # Expected output: torch.Size([4, 128, 768])