"""DINOv3 ViT Tagger Model

Multi-label image tagger built on a DINOv3 ViT backbone.

Architecture
------------
1. DINOv3 ViT backbone (configurable HuggingFace model ID).
   Fine-tuned end-to-end by default (freeze_backbone=False).
2. After the forward pass, extract:
   - CLS token       → last_hidden_state[:, 0, :]
   - Register tokens → last_hidden_state[:, 1 : 1+R, :]  (R = num_register_tokens)
3. Concatenate along the hidden-dim axis:
       features = [CLS, reg_0, ..., reg_{R-1}]  →  shape [B, (1+R)*D]
   This avoids a bottleneck when num_tags > D.
4. Single nn.Linear((1+R)*D → num_tags) projection head.
5. Raw logits returned — use BCEWithLogitsLoss during training.

DINOv3 ViT accepts any resolution that is a multiple of the patch size (16 px).
No fixed 224 crop is required; the model was pre-trained at up to 1024+ px.

Supported model IDs
-------------------
  facebook/dinov3-vits16-pretrain-lvd1689m
  facebook/dinov3-vitb16-pretrain-lvd1689m
  facebook/dinov3-vitl16-pretrain-lvd1689m
  facebook/dinov3-vith16plus-pretrain-lvd1689m
  facebook/dinov3-vit7b16-pretrain-lvd1689m
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


# ---------------------------------------------------------------------------
# Supported DINOv3 ViT model IDs (hard-coded allowlist)
# ---------------------------------------------------------------------------
DINOV3_VIT_MODELS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# All DINOv3 ViT variants use 16-px patches — images must be multiples of this
DINOV3_PATCH_SIZE = 16


@dataclass
class DINOv3TaggerParams:
    """Configuration for DINOv3Tagger.

    Attributes
    ----------
    backbone_name : str
        HuggingFace model ID.  Must be one of the DINOv3 ViT variants listed
        in DINOV3_VIT_MODELS.
    num_tags : int
        Number of output classes (unique tags in the vocabulary).
    freeze_backbone : bool
        If True, only the projection head is trained.
    projection_bias : bool
        Whether the linear projection head has a bias term.
    dtype : str
        Backbone weight dtype: "float32", "bfloat16", or "float16".
    """
    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    num_tags: int = 10_000
    freeze_backbone: bool = False
    projection_bias: bool = False
    dtype: str = "bfloat16"


class DINOv3Tagger(nn.Module):
    """Multi-label tagger with a DINOv3 ViT backbone and a linear projection head."""

    def __init__(self, params: DINOv3TaggerParams):
        super().__init__()
        self.params = params

        # if params.backbone_name not in DINOV3_VIT_MODELS:
        #     raise ValueError(
        #         f"backbone_name '{params.backbone_name}' is not a supported DINOv3 ViT model.\n"
        #         f"Supported models:\n" + "\n".join(f"  {m}" for m in sorted(DINOV3_VIT_MODELS))
        #     )

        # ------------------------------------------------------------------ #
        # Backbone
        # ------------------------------------------------------------------ #
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        backbone_dtype = dtype_map.get(params.dtype, torch.bfloat16)

        self.backbone = AutoModel.from_pretrained(params.backbone_name, torch_dtype=backbone_dtype)

        for p in self.backbone.parameters():
            p.requires_grad = not params.freeze_backbone

        # ------------------------------------------------------------------ #
        # Detect hidden size and register token count from model config
        # ------------------------------------------------------------------ #
        cfg = self.backbone.config
        hidden_size: int = cfg.hidden_size
        num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        self.num_register_tokens = num_register_tokens

        concat_dim = hidden_size * (1 + num_register_tokens)

        print(
            f"[DINOv3Tagger] {params.backbone_name} | "
            f"hidden={hidden_size} | registers={num_register_tokens} | "
            f"concat_dim={concat_dim} | num_tags={params.num_tags}"
        )

        # ------------------------------------------------------------------ #
        # Projection head: concat_dim → num_tags
        # ------------------------------------------------------------------ #
        self.projection = nn.Linear(concat_dim, params.num_tags, bias=params.projection_bias)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if params.projection_bias:
            nn.init.zeros_(self.projection.bias)

    # ---------------------------------------------------------------------- #
    # Alternative constructor — no pretrained weights loaded
    # ---------------------------------------------------------------------- #

    @classmethod
    def empty_init(cls, params: "DINOv3TaggerParams") -> "DINOv3Tagger":
        """Construct the model from config only, without loading pretrained weights.

        Use this when you are about to load a full checkpoint with
        ``load_state_dict(..., assign=True)`` and don't want to waste time
        (and RAM) downloading / initialising the HuggingFace pretrained weights
        first.
        """
        # Temporarily monkey-patch AutoModel so __init__ calls from_config
        # instead of from_pretrained, then restore it immediately.
        original = AutoModel.from_pretrained

        def _from_config_shim(name_or_path, **kwargs):
            kwargs.pop("torch_dtype", None)  # from_config doesn't accept this
            cfg = AutoConfig.from_pretrained(name_or_path)
            return AutoModel.from_config(cfg)

        AutoModel.from_pretrained = _from_config_shim  # type: ignore[method-assign]
        try:
            instance = cls(params)
        finally:
            AutoModel.from_pretrained = original  # type: ignore[method-assign]
        return instance

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for each tag.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Shape ``[B, 3, H, W]``.  H and W must be multiples of 16.
            Normalised with ImageNet stats (mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]).

        Returns
        -------
        logits : torch.Tensor
            Shape ``[B, num_tags]``.  Raw (un-sigmoided) logits.
        """
        outputs = self.backbone(pixel_values=pixel_values)

        # Token layout: [CLS, reg_0, ..., reg_{R-1}, patch_0, patch_1, ...]
        hidden = outputs.last_hidden_state  # [B, seq_len, D]

        cls_token = hidden[:, 0, :]  # [B, D]

        if self.num_register_tokens > 0:
            reg_tokens = hidden[:, 1: 1 + self.num_register_tokens, :]  # [B, R, D]
            reg_tokens = reg_tokens.reshape(reg_tokens.shape[0], -1)    # [B, R*D]
            features = torch.cat([cls_token, reg_tokens], dim=-1)       # [B, (1+R)*D]
        else:
            features = cls_token  # [B, D]

        # Project in fp32 for numerical stability
        return self.projection(features.float())  # [B, num_tags]

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @property
    def num_tags(self) -> int:
        return self.params.num_tags

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
