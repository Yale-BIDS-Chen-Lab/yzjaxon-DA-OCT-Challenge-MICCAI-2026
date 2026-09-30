"""Plain convolutional U-Net in the nnU-Net mould: the main line of this entry.

``strides[0] == [1, 1]``, so stage 0 runs at full resolution: the score is dominated by a
boundary distance and a half-resolution decoder cannot place an interface to the pixel.
Normalisation is InstanceNorm with no running statistics, which would otherwise be fitted
on the one labelled vendor. Deep supervision returns a fine-to-coarse list, one entry per
decoder stage; cap it with ``ds_levels``."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from octtta.models._blocks import (
    ConvStack,
    accepts_any_size,
    init_weights_he,
    to_pair,
    total_stride,
)

__all__ = ["PlainConvUNet"]


class _UpBlock(nn.Module):
    """Transposed-conv upsample, concat the skip, then a conv stack."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 n_convs: int, kernel_size, stride, norm: str, nonlin: str,
                 dropout: float) -> None:
        super().__init__()
        sh, sw = to_pair(stride)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(sh, sw),
                                     stride=(sh, sw), bias=True)
        self.convs = ConvStack(out_channels + skip_channels, out_channels, n_convs,
                               kernel_size, 1, norm, nonlin, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.convs(torch.cat([self.up(x), skip], dim=1))


class PlainConvUNet(nn.Module):
    """nnU-Net-style plain conv U-Net; kwargs mirror the shipped CNN configuration.
    ``ds_levels`` caps how many deep-supervision maps come back, finest first, and
    ``in_channels`` is how many preprocessed views of the B-scan arrive stacked."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 10,
        n_stages: int = 7,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 512, 512, 512),
        kernel_sizes: Sequence[Sequence[int]] = ((3, 3),) * 7,
        strides: Sequence[Sequence[int]] = ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
        n_conv_per_stage: Sequence[int] = (2, 2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder: Sequence[int] = (2, 2, 2, 2, 2, 2),
        norm: str = "instance",
        nonlin: str = "leakyrelu",
        deep_supervision: bool = True,
        *,
        ds_levels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_stages = int(n_stages)
        features_per_stage = [int(f) for f in features_per_stage]
        kernel_sizes = [to_pair(k) for k in kernel_sizes]
        strides = [to_pair(s) for s in strides]
        n_conv_per_stage = [int(n) for n in n_conv_per_stage]
        n_conv_per_stage_decoder = [int(n) for n in n_conv_per_stage_decoder]

        for name, seq, want in (
            ("features_per_stage", features_per_stage, n_stages),
            ("kernel_sizes", kernel_sizes, n_stages),
            ("strides", strides, n_stages),
            ("n_conv_per_stage", n_conv_per_stage, n_stages),
            ("n_conv_per_stage_decoder", n_conv_per_stage_decoder, n_stages - 1),
        ):
            if len(seq) != want:
                raise ValueError(f"{name} must have {want} entries for n_stages={n_stages}, "
                                 f"got {len(seq)}")

        self.num_classes = int(out_channels)
        self.deep_supervision = bool(deep_supervision)
        if norm != "instance" or nonlin != "leakyrelu" or dropout != 0.0:
            raise ValueError("Final plain U-Net uses instance norm, leakyrelu and no dropout")
        self.stride_multiple = total_stride(strides)
        # Never let the deepest map reach 1x1: affine InstanceNorm would output zeros.
        self.min_input_size = (self.stride_multiple[0] * 2, self.stride_multiple[1] * 2)

        self.encoder = nn.ModuleList()
        prev = int(in_channels)
        for s in range(n_stages):
            self.encoder.append(
                ConvStack(prev, features_per_stage[s], n_conv_per_stage[s],
                          kernel_sizes[s], strides[s], norm, nonlin, dropout)
            )
            prev = features_per_stage[s]

        n_dec = n_stages - 1
        self.decoder = nn.ModuleList()
        for i in range(n_dec):
            src = n_stages - 1 - i          # encoder stage being upsampled from
            self.decoder.append(
                _UpBlock(features_per_stage[src], features_per_stage[src - 1],
                         features_per_stage[src - 1], n_conv_per_stage_decoder[i],
                         kernel_sizes[src - 1], strides[src], norm, nonlin, dropout)
            )

        if ds_levels is not None and not 1 <= int(ds_levels) <= n_dec:
            raise ValueError(f"ds_levels must be in [1, {n_dec}], got {ds_levels}")
        self.ds_levels = n_dec if ds_levels is None else int(ds_levels)

        # Heads below the requested depth are never built rather than built and ignored.
        self.heads = nn.ModuleList([
            nn.Conv2d(features_per_stage[n_stages - 2 - i], self.num_classes, 1)
            if (n_dec - 1 - i) < self.ds_levels else nn.Identity()
            for i in range(n_dec)
        ])

        init_weights_he(self)

    @property
    def num_ds_outputs(self) -> int:
        return self.ds_levels if self.deep_supervision else 1

    def warm_start_optional_keys(self) -> frozenset[str]:
        """state_dict keys a warm start may leave at their fresh initialisation.
        The model declares them, so ``load_state_dict`` stays strict about everything else."""
        return frozenset()

    @accepts_any_size
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Return full-resolution segmentation, plus deep supervision while training."""

        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)

        want_ds = self.deep_supervision and self.training
        out: list[torch.Tensor] = []
        h = skips[-1]
        n_dec = len(self.decoder)
        for i, block in enumerate(self.decoder):
            h = block(h, skips[-2 - i])
            level = n_dec - 1 - i
            if level == 0 or (want_ds and level < self.ds_levels):
                out.append(self.heads[i](h))

        out.reverse()                       # collected coarse->fine, contract is fine->coarse
        return out if want_ds else out[0]
