"""Model registry for the released SAM-FPN and plain convolutional U-Net."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch.nn as nn

from octtta.models.ema import ModelEMA

__all__ = ["build_model", "ModelEMA", "MODEL_NAMES", "third_party_requirements"]


@dataclass(frozen=True)
class _Entry:
    module: str
    attr: str
    requires: frozenset[str]

    def build(self, **params: Any) -> nn.Module:
        return getattr(import_module(self.module), self.attr)(**params)


_REGISTRY: dict[str, _Entry] = {
    "nnunet2d": _Entry("octtta.models.nnunet_like", "PlainConvUNet",
                       frozenset({"numpy", "torch"})),
    "vit_fpn": _Entry("octtta.models.vit_fpn", "ViTFPN",
                      frozenset({"timm", "torch"})),
}
MODEL_NAMES = tuple(_REGISTRY)


def third_party_requirements(model_name: str) -> frozenset[str]:
    if model_name not in _REGISTRY:
        raise ValueError(f"unknown model name {model_name!r}; valid: {MODEL_NAMES}")
    return _REGISTRY[model_name].requires


def build_model(cfg: dict, *, pretrained: bool | None = None) -> nn.Module:
    """Build a released model; an explicit false override suppresses weight loading.

    Training reads its local encoder export from model.params.encoder_weights. Checkpoint
    reconstruction passes pretrained=False so all weights come from the checkpoint.
    """
    if "name" not in cfg and isinstance(cfg.get("model"), dict):
        cfg = cfg["model"]
    name = cfg.get("name")
    if name not in _REGISTRY:
        raise ValueError(f"unknown model name {name!r}; valid: {MODEL_NAMES}")
    params = dict(cfg.get("params") or {})
    if pretrained is False:
        if "pretrained" in params:
            params["pretrained"] = False
        if "encoder_weights" in params:
            params["encoder_weights"] = None
    try:
        return _REGISTRY[name].build(**params)
    except TypeError as exc:
        raise TypeError(f"building model {name!r} with params {sorted(params)}: {exc}") from exc
