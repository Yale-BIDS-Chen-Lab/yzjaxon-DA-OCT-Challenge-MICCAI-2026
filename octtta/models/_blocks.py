"""Shape-agnostic plumbing and the conv blocks every model in the zoo shares.

H and W vary per sample and per vendor, so each model pads its input up to a multiple of
its own total stride and crops the logits back to the caller's exact (H, W); resizing
would move the layer boundaries. Padding is bottom/right only, so the crop origin stays
(0, 0) at every deep-supervision scale, and it replicates the border rather than filling
with a constant. Models pad to at least two strides: ``InstanceNorm2d`` over a 1x1 map
outputs exactly zero and silently deletes the bottleneck."""

from __future__ import annotations

import functools
import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "pad_to_multiple",
    "crop_to",
    "crop_pyramid",
    "scaled_size",
    "accepts_any_size",
    "total_stride",
    "build_norm",
    "build_nonlin",
    "ConvNormAct",
    "ConvStack",
    "init_weights_he",
    "to_pair",
]

Size2 = tuple[int, int]


def to_pair(value: int | Sequence[int]) -> Size2:
    """Normalise ``3`` / ``[3, 3]`` / ``(3, 3)`` to a ``(h, w)`` tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"expected a length-2 sequence, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def total_stride(strides: Sequence[Sequence[int] | int]) -> Size2:
    """Product of per-stage strides, separately per axis."""
    h, w = 1, 1
    for s in strides:
        sh, sw = to_pair(s)
        h *= sh
        w *= sw
    return h, w


def _pad_bottom_right(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Replicate-pad the bottom/right, in rounds for very small inputs."""
    while pad_h > 0 or pad_w > 0:
        h, w = x.shape[-2:]
        step_h = min(pad_h, h)
        step_w = min(pad_w, w)
        if step_h == 0 and step_w == 0:
            raise ValueError(f"cannot pad a {tuple(x.shape[-2:])} tensor")
        x = F.pad(x, (0, step_w, 0, step_h), mode="replicate")
        pad_h -= step_h
        pad_w -= step_w
    return x


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int | Sequence[int],
    *,
    min_size: int | Sequence[int] | None = None,
) -> tuple[torch.Tensor, tuple[slice, slice]]:
    """Pad ``(..., H, W)`` up to a multiple of ``multiple``; return it and the undo crop
    ``(slice_h, slice_w)`` for ``y[..., sh, sw]``. ``min_size`` floors the padded extent,
    which is how a model keeps its bottleneck above 1x1."""
    if x.dim() < 2:
        raise ValueError(f"expected at least 2 spatial dims, got shape {tuple(x.shape)}")
    mh, mw = to_pair(multiple)
    h, w = int(x.shape[-2]), int(x.shape[-1])

    tgt_h = int(math.ceil(h / mh)) * mh
    tgt_w = int(math.ceil(w / mw)) * mw
    if min_size is not None:
        floor_h, floor_w = to_pair(min_size)
        tgt_h = max(tgt_h, int(math.ceil(floor_h / mh)) * mh)
        tgt_w = max(tgt_w, int(math.ceil(floor_w / mw)) * mw)

    crop = (slice(0, h), slice(0, w))
    if tgt_h == h and tgt_w == w:
        return x, crop
    return _pad_bottom_right(x, tgt_h - h, tgt_w - w), crop


def scaled_size(hw: Size2, level: int) -> Size2:
    """Size of deep-supervision entry ``level``: ``hw`` halved ``level`` times (ceil).
    Ceil-halving composes, so entry ``i`` is both half of entry ``i-1`` and ``ceil(H/2**i)``."""
    h, w = hw
    denom = 1 << level
    return int(math.ceil(h / denom)), int(math.ceil(w / denom))


def crop_to(x: torch.Tensor, hw: Size2) -> torch.Tensor:
    h, w = hw
    if x.shape[-2] < h or x.shape[-1] < w:
        raise ValueError(f"cannot crop {tuple(x.shape[-2:])} up to {(h, w)}")
    return x[..., :h, :w]


def crop_pyramid(outputs: Sequence[torch.Tensor], hw: Size2) -> list[torch.Tensor]:
    """Crop a fine-to-coarse deep-supervision list back to the caller's geometry."""
    return [crop_to(t, scaled_size(hw, i)) for i, t in enumerate(outputs)]




def _crop_seg(out, hw: Size2):
    if torch.is_tensor(out):
        return crop_to(out, hw)
    if isinstance(out, (list, tuple)):
        return type(out)(crop_pyramid(list(out), hw))
    raise TypeError(f"forward returned {type(out)!r}")


def accepts_any_size(forward: Callable) -> Callable:
    """Decorate ``Module.forward`` so the module accepts any ``(H, W)``: the wrapper pads
    to ``self.stride_multiple`` (floored at ``self.min_input_size``) and crops the result
    back to the caller's geometry."""

    @functools.wraps(forward)
    def wrapper(self, x: torch.Tensor, *args, **kwargs):
        hw = (int(x.shape[-2]), int(x.shape[-1]))
        padded, _ = pad_to_multiple(
            x,
            getattr(self, "stride_multiple", 1),
            min_size=getattr(self, "min_input_size", None),
        )
        out = forward(self, padded, *args, **kwargs)
        return _crop_seg(out, hw)

    return wrapper


def build_norm(name: str | None, channels: int, *, groups: int = 8,
               dims: int = 2) -> nn.Module:
    """The affine per-image normalisation used by both released decoders."""
    if name != "instance" or dims != 2 or groups != 8:
        raise ValueError("Final models use 2-D affine instance normalisation")
    return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)




def build_nonlin(name: str | None, *, inplace: bool = True) -> nn.Module:
    if name != "leakyrelu":
        raise ValueError("Final models use leakyrelu")
    return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)


class ConvNormAct(nn.Module):
    """conv -> norm -> nonlin, with same-padding for odd kernels."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | Sequence[int] = 3,
                 stride: int | Sequence[int] = 1,
                 norm: str = "instance", nonlin: str = "leakyrelu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        kh, kw = to_pair(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw),
                              stride=to_pair(stride), padding=(kh // 2, kw // 2), bias=True)
        self.norm = build_norm(norm, out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = build_nonlin(nonlin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(self.norm(self.conv(x))))


class ConvStack(nn.Module):
    """``n`` ConvNormAct in a row; only the first one carries the stride."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int,
                 kernel_size: int | Sequence[int] = 3,
                 stride: int | Sequence[int] = 1,
                 norm: str = "instance", nonlin: str = "leakyrelu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        if n_convs < 1:
            raise ValueError(f"n_convs must be >= 1, got {n_convs}")
        blocks = [ConvNormAct(in_channels, out_channels, kernel_size, stride,
                              norm, nonlin, dropout)]
        blocks += [ConvNormAct(out_channels, out_channels, kernel_size, 1,
                               norm, nonlin, dropout) for _ in range(n_convs - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def init_weights_he(module: nn.Module, negative_slope: float = 1e-2) -> None:
    """nnU-Net's initialisation: He-normal on convs, zero bias, unit affine norms."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, a=negative_slope, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm,
                            nn.InstanceNorm1d, nn.BatchNorm1d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
