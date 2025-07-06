# forcefully yanked from HF

import math

import torch
from torch import nn
import torch.utils.checkpoint as ckpt

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from concurrent.futures import ThreadPoolExecutor


class T5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Model`] or a [`TFT5Model`]. It is used to
    instantiate a T5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        classifier_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # if (
        #     isinstance(self.wo.weight, torch.Tensor)
        #     and hidden_states.dtype != self.wo.weight.dtype
        #     and self.wo.weight.dtype != torch.int8
        # ):
        #     hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask

        # Use torch.scaled_dot_product_attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=position_bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1,  # ofc t5 is really unstable, they don't scale the q to be numerically stable, this model is old af
        )

        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        self.layer.append(T5LayerFF(config))

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
        )
        hidden_states, _ = self_attention_outputs[:2]
        # Keep self-attention outputs and relative position weights
        attention_outputs = self_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        return outputs + attention_outputs


class T5Stack(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dispatcher = None

        self.num_heads = config.num_heads

    def init_dispatcher(self, max_workers=100):
        self.dispatcher = ThreadPoolExecutor(max_workers=max_workers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        b, l, d = hidden_states.shape
        device = hidden_states.device

        if attention_mask == None:
            attention_mask = -torch.zeros([b, 1, 1, l], device=device)

        else:
            attention_mask = attention_mask.float().T @ attention_mask.float()
            attention_mask = attention_mask[None, None, ...].repeat(
                b, self.num_heads, 1, 1
            )
            attention_mask[attention_mask == 0] = torch.finfo(torch.bfloat16).min
            attention_mask[attention_mask == 1] = 0

        position_bias = None
        for i, layer_module in enumerate(self.block):
            device = layer_module.device
            if self.training:
                layer_outputs = ckpt.checkpoint(
                    layer_module,
                    hidden_states.to(device),
                    attention_mask.to(device),
                    (
                        position_bias.to(device)
                        if position_bias != None
                        else position_bias
                    ),
                    use_reentrant=False
                )
                pass
            else:
                layer_outputs = layer_module(
                    hidden_states.to(device),
                    attention_mask=attention_mask.to(device),
                    position_bias=(
                        position_bias.to(device)
                        if position_bias != None
                        else position_bias
                    ),
                )

            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, _ = layer_outputs[:2]
            position_bias = layer_outputs[2]
            # hidden_states = layer_outputs

        hidden_states = self.final_layer_norm(
            hidden_states.to(self.final_layer_norm.weight.device)
        )
        return self.dropout(hidden_states)

    def forward_first(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        b, l, d = hidden_states.shape
        device = hidden_states.device

        if attention_mask == None:
            attention_mask = -torch.zeros([b, 1, 1, l], device=device)

        return hidden_states, attention_mask

    def forward_last(
        self,
        hidden_states,
    ):
        hidden_states = self.final_layer_norm(
            hidden_states.to(self.final_layer_norm.weight.device)
        )
        return self.dropout(hidden_states)

    def forward_mid(
        self, hidden_states, attention_mask, position_bias=None, layer_slice=(0, -1)
    ):
        for i, layer_module in enumerate(self.block[layer_slice[0] : layer_slice[1]]):
            device = layer_module.device
            if self.training:
                layer_outputs = ckpt.checkpoint(
                    layer_module,
                    hidden_states.to(device),
                    attention_mask.to(device),
                    (
                        position_bias.to(device)
                        if position_bias != None
                        else position_bias
                    ),
                    use_reentrant=False
                )
                pass
            else:
                layer_outputs = layer_module(
                    hidden_states.to(device),
                    attention_mask=attention_mask.to(device),
                    position_bias=(
                        position_bias.to(device)
                        if position_bias != None
                        else position_bias
                    ),
                )

            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, _ = layer_outputs[:2]
            position_bias = layer_outputs[2]
        return hidden_states, attention_mask, position_bias

    def transfer_ops(self, device, *args):
        return [arg.to(device, non_blocking=True) for arg in args]

    # @torch.inference_mode()
    def forward_dual_gpu(
        self,
        input_ids=None,
        attention_mask=None,
        microbatch=1,
    ):
        # Helper function for no_grad wrapping
        def _run_with_no_grad(func, *args, **kwargs):
            with torch.no_grad():
                return func(*args, **kwargs)

        # we need to keep the activation and detached activation
        # separate so we could backprop it later in decoupled manner

        forward_buffer_first = {}  # gpu:0 # embedding
        forward_buffer_stacks_1st = {}  # gpu:0 # transformer resnet stack
        forward_buffer_stacks_2nd = {}  # gpu:1 # transformer resnet stack
        forward_buffer_last = {}

        output_buffer = []

        # first layer forward (embedding)
        for i in range(input_ids.shape[0] // microbatch):
            # embedding forward
            # dispatch process
            forward_buffer_first[i] = self.dispatcher.submit(
                lambda: _run_with_no_grad(
                    self.forward_first,
                    (input_ids[(i) * microbatch : (i) * microbatch + microbatch]),
                )
            )

            # first stack forward
            # materialize embedding activation
            forward_buffer_first[i] = forward_buffer_first[i].result()
            # dispatch process
            forward_buffer_stacks_1st[i] = self.dispatcher.submit(
                lambda: _run_with_no_grad(
                    self.forward_mid,
                    hidden_states=forward_buffer_first[i][0],
                    attention_mask=forward_buffer_first[i][1],
                    position_bias=None,
                    layer_slice=(0, 12),
                )
            )

            # second stack forward
            # materialize resnet activation
            forward_buffer_stacks_1st[i] = self.transfer_ops(
                next(self.block[12].parameters()).device,
                *forward_buffer_stacks_1st[i].result(),
            )
            # dispatch process
            forward_buffer_stacks_2nd[i] = self.dispatcher.submit(
                lambda: _run_with_no_grad(
                    self.forward_mid,
                    hidden_states=forward_buffer_stacks_1st[i][0],
                    attention_mask=forward_buffer_stacks_1st[i][1],
                    position_bias=forward_buffer_stacks_1st[i][2],
                    layer_slice=(12, 24),
                )
            )

            # final forward
            # materialize resnet activation
            forward_buffer_stacks_2nd[i] = forward_buffer_stacks_2nd[i].result()
            forward_buffer_last[i] = self.dispatcher.submit(
                lambda: _run_with_no_grad(
                    self.forward_last,
                    hidden_states=forward_buffer_stacks_2nd[i][0],
                )
            )

            # flush pipeline queue
            del forward_buffer_first[i]
            del forward_buffer_stacks_1st[i]
            del forward_buffer_stacks_2nd[i]
            # del forward_buffer_last[i]

        for i in range(input_ids.shape[0] // microbatch):
            # materialize final layer and then append to output buffer
            forward_buffer_last[i] = forward_buffer_last[i].result()
            output_buffer.append(forward_buffer_last[i])

        return torch.cat(output_buffer, dim=0)


class T5EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # just to match keys hierarchy
        self.encoder = T5Stack(config)

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


def replace_keys(state_dict):
    state_dict["encoder.embed_tokens.weight"] = state_dict.pop("shared.weight")
    return state_dict
