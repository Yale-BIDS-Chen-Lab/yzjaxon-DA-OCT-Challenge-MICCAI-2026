"""Small registered SAM encoder for CPU model and pipeline tests only."""
from __future__ import annotations

from timm.models import register_model
from timm.models.vision_transformer_sam import VisionTransformerSAM


@register_model
def samvit_tiny_fixture(pretrained: bool = False, **kwargs):
    if pretrained:
        raise ValueError("the synthetic SAM fixture has no pretrained weights")
    # timm supplies factory metadata alongside the actual model parameters.
    return VisionTransformerSAM(
        img_size=64, patch_size=16, embed_dim=32, depth=4, num_heads=4,
        neck_chans=0, window_size=4, global_attn_indexes=(1, 3),
        in_chans=int(kwargs.get("in_chans", 1)), num_classes=0,
    )
