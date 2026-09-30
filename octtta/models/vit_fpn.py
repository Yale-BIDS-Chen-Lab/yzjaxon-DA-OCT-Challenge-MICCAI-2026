"""ViT encoder + lightweight FPN/DPT decoder.

Four block taps (shallow to deep) are reassembled to strides 4/8/16/32 and fused top-down,
so the decoder sees both the local texture that places an interface and the semantics that
names the layer. Upsampling is nearest + 3x3 conv rather than transposed conv, the decoder
walks all the way to stride 1, and normalisation is InstanceNorm. ``min_input_size`` is
``4 * patch_size``, not ``2 * stride_multiple``, because the coarsest pyramid level sits at
``2 * patch_size`` and InstanceNorm over a 1x1 map outputs zero."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from octtta.models._blocks import ConvNormAct, accepts_any_size, init_weights_he

__all__ = ["ViTFPN", "FinetuneGroup", "plan_finetune", "apply_finetune_plan",
           "build_timm_encoder"]


def build_timm_encoder(backbone: str, *, in_chans: int, pretrained: bool,
                       cache_dir: str | None = None, **extra: Any) -> nn.Module:
    """The shared SAM construction site for training and the pinned local exporter."""
    if not str(backbone).startswith("samvit_"):
        raise ValueError(f"unsupported encoder {backbone!r}; Final uses SAM-ViT")
    import timm

    return timm.create_model(backbone, pretrained=bool(pretrained), in_chans=int(in_chans),
                             num_classes=0, cache_dir=cache_dir, **extra)


def _adapt_encoder_in_place(encoder: nn.Module, *, window_size: int | None) -> dict[str, Any]:
    """Drop the unused SAM neck and set local-attention windows after loading weights."""
    changed: dict[str, Any] = {}
    neck = getattr(encoder, "neck", None)
    if neck is not None and not isinstance(neck, nn.Identity):
        changed["neck_params_dropped"] = sum(p.numel() for p in neck.parameters())
        encoder.neck = nn.Identity()
    if window_size is not None:
        want = int(window_size)
        if want <= 0:
            raise ValueError(f"encoder_window_size must be positive, got {window_size}")
        touched = []
        for i, block in enumerate(encoder.blocks):
            if int(getattr(block, "window_size", 0) or 0) > 0:
                block.window_size = want
                touched.append(i)
        if not touched:
            raise RuntimeError("SAM encoder has no positive local-attention window")
        changed["window_size"] = want
        changed["window_blocks"] = touched
    return changed


def _tap_channels(encoder: nn.Module, indices: Sequence[int]) -> int:
    """Read trunk width from feature_info, never from SAM's projected neck width."""
    info = getattr(encoder, "feature_info", None)
    if info is None:
        raise TypeError("SAM encoder exposes no feature_info")
    channels = [int(info[int(i)]["num_chs"]) for i in indices]
    if len(set(channels)) != 1:
        raise ValueError(f"pyramid tap widths disagree: {channels}")
    return channels[0]


#: Keys a checkpoint may hide the encoder state dict behind, most specific first.
_STATE_KEYS = ("teacher_backbone", "state_dict")

_STRIP_PREFIXES = ("module.", "backbone.", "encoder.")


def _extract_state_dict(payload: object) -> dict:
    """Accept either a full Stage-A payload or a bare state dict."""
    if not isinstance(payload, dict):
        raise TypeError(f"encoder_weights must hold a dict, got {type(payload).__name__}")
    for key in _STATE_KEYS:
        inner = payload.get(key)
        if isinstance(inner, dict):
            return inner
    return payload


def _strip_prefix(state: dict) -> dict:
    """Drop a uniform wrapper prefix, if and only if *every* key carries it."""
    for prefix in _STRIP_PREFIXES:
        if state and all(k.startswith(prefix) for k in state):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_encoder_weights(encoder: nn.Module, path: str | os.PathLike) -> tuple[int, int, int]:
    """Load encoder weights into ``encoder``; return ``(loaded, missing, unexpected)``.
    Coverage is strict: a partial load would leave the rest randomly initialised while the
    run still looked healthy, so anything short of full coverage raises."""
    resolved = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not resolved.is_file():
        raise FileNotFoundError(
            f"encoder_weights={str(path)!r} resolves to {resolved} which is not a file; "
            f"point it at the Stage-A export (a payload with a 'teacher_backbone' key) or "
            f"set it to null and let a checkpoint supply the weights"
        )
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    state = _strip_prefix(_extract_state_dict(payload))

    own = encoder.state_dict()
    missing = sorted(k for k in own if k not in state)
    mismatched = sorted(k for k in own if k in state and own[k].shape != state[k].shape)
    if missing or mismatched:
        detail = []
        if missing:
            detail.append(f"{len(missing)} missing (e.g. {missing[:4]})")
        if mismatched:
            detail.append(f"{len(mismatched)} shape-mismatched (e.g. {mismatched[:4]})")
        raise ValueError(
            f"{resolved} covers only {len(own) - len(missing) - len(mismatched)}/{len(own)} "
            f"encoder tensors: {'; '.join(detail)}. Refusing a partial load -- the uncovered "
            f"parameters would stay randomly initialised and the run would silently train "
            f"from (mostly) scratch. Wrong file, or wrong model.name for these weights? "
            f"File has {len(state)} entries; first few file keys: {sorted(state)[:4]}"
        )
    encoder.load_state_dict({k: state[k] for k in own}, strict=True)
    n_unexpected = len(state) - len(own)
    print(f"[vit_fpn] loaded {len(own)}/{len(own)} encoder tensors from {resolved} "
          f"(unexpected-and-ignored={n_unexpected})", flush=True)
    return len(own), 0, n_unexpected


class _Reassemble(nn.Module):
    """One DPT reassemble stage: token LayerNorm -> 1x1 projection -> resample.
    ``log2_scale`` is signed: ``+n`` upsamples by ``2**n``, ``-n`` downsamples. The LayerNorm
    standardises from step 0, which the FPN needs because raw block outputs differ in scale."""

    def __init__(self, in_dim: int, out_dim: int, *, log2_scale: int,
                 norm: str, nonlin: str) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        layers: list[nn.Module] = []
        for _ in range(max(log2_scale, 0)):
            layers += [nn.Upsample(scale_factor=2.0, mode="nearest"),
                       ConvNormAct(out_dim, out_dim, 3, norm=norm, nonlin=nonlin)]
        for _ in range(max(-log2_scale, 0)):
            # stride-2 3x3 with padding 1 outputs ceil(n / 2), the size the next level wants.
            layers.append(ConvNormAct(out_dim, out_dim, 3, stride=2, norm=norm, nonlin=nonlin))
        self.resample = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.resample(self.proj(x))


class _UpBlock(nn.Module):
    """Nearest x2 then a 3x3 ConvNormAct -- the decoder's step back towards stride 1."""

    def __init__(self, in_dim: int, out_dim: int, *, norm: str, nonlin: str) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_dim, out_dim, 3, norm=norm, nonlin=nonlin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ViTFPN(nn.Module):
    """ViT encoder + multi-scale FPN decoder + segmentation head.
    ``encoder_weights`` is the pinned local SAM export used during training."""

    def __init__(
        self,
        backbone: str = "samvit_large_patch16.sa1b",
        patch_size: int = 16,
        in_channels: int = 1,
        out_channels: int = 10,
        indices: Sequence[int] = (2, 5, 8, 11),
        dim: int = 256,
        encoder_weights: str | None = None,
        pretrained: bool = False,
        deep_supervision: bool = True,
        *,
        ds_levels: int | None = None,
        encoder_window_size: int | None = None,
    ) -> None:
        super().__init__()

        indices = [int(i) for i in indices]
        if len(indices) != 4:
            raise ValueError(
                f"vit_fpn builds a four-level pyramid (strides patch/4 .. patch*2), so it "
                f"needs exactly four block indices; got {indices}"
            )
        if sorted(indices) != indices:
            raise ValueError(f"indices must be shallow-to-deep, got {indices}")

        if pretrained:
            raise ValueError("Final SAM training uses a pinned local encoder export")
        if int(in_channels) != 1:
            raise ValueError("Final SAM model takes one input channel")

        self.backbone_name = str(backbone)
        self.encoder = build_timm_encoder(
            self.backbone_name,
            in_chans=int(in_channels),
            pretrained=False,
            cache_dir=None,
        )
        self.pretrained = False

        depth = len(self.encoder.blocks)
        if max(indices) >= depth:
            raise ValueError(
                f"backbone {self.backbone_name!r} has {depth} blocks; indices {indices} "
                f"index past the end"
            )
        # The deepest tap must be the last block: on a deeper backbone the same indices stay
        # legal while the blocks past the deepest tap feed no pyramid level and nothing errors.
        if max(indices) != depth - 1:
            want = [round(q * depth) - 1 for q in (0.25, 0.5, 0.75, 1.0)]
            raise ValueError(
                f"backbone {self.backbone_name!r} has {depth} blocks but the deepest tap "
                f"is block {max(indices)} (indices={indices}), so blocks "
                f"{max(indices) + 1}..{depth - 1} would reach no pyramid level and the "
                f"decoder would silently see only part of the encoder. The four depth "
                f"quantiles for {depth} blocks are {want}."
            )
        self._indices = indices

        # Assert the declared patch size against the model: stride_multiple rests on it.
        actual = getattr(getattr(self.encoder, "patch_embed", None), "patch_size", None)
        patch = int(patch_size)
        if actual is not None and tuple(int(p) for p in actual) != (patch, patch):
            raise ValueError(
                f"patch_size={patch} but backbone {self.backbone_name!r} has patch "
                f"{tuple(actual)}; stride_multiple would pad to the wrong multiple"
            )
        # From feature_info, never encoder.num_features: for SAM the latter is the neck's
        # width while forward_intermediates hands back the trunk's embed_dim.
        embed_dim = _tap_channels(self.encoder, indices)
        self._embed_dim = int(embed_dim)
        self._taps_checked = False

        if encoder_weights is not None:
            load_encoder_weights(self.encoder, encoder_weights)

        # After every weight is in place: a window override needs the published table shapes.
        self.encoder_adaptations = _adapt_encoder_in_place(
            self.encoder, window_size=encoder_window_size)
        self.encoder_window_size = encoder_window_size
        if self.encoder_adaptations:
            print(f"[vit_fpn] encoder adaptations: {self.encoder_adaptations}", flush=True)

        self.patch_size = patch
        self.num_classes = int(out_channels)
        self.deep_supervision = bool(deep_supervision)
        self.stride_multiple = (patch, patch)
        # 4 * patch, not 2 * stride: the coarsest pyramid level is 2 * patch.
        self.min_input_size = (4 * patch, 4 * patch)

        dim = int(dim)
        # Shallow tap -> finest level. log2 of (level stride / patch), signed.
        self.reassemble = nn.ModuleList([
            _Reassemble(embed_dim, dim, log2_scale=s, norm="instance", nonlin="leakyrelu")
            for s in (2, 1, 0, -1)
        ])
        self.smooth = nn.ModuleList([
            ConvNormAct(dim, dim, 3, norm="instance", nonlin="leakyrelu")
            for _ in self.reassemble
        ])
        # Pyramid level 0 sits at stride 4; two more x2 steps reach full resolution.
        self.up_blocks = nn.ModuleList([
            _UpBlock(dim, dim // 2, norm="instance", nonlin="leakyrelu"),
            _UpBlock(dim // 2, dim // 4, norm="instance", nonlin="leakyrelu"),
        ])

        level_channels = [dim // 4, dim // 2] + [dim] * len(self.reassemble)
        n_levels = len(level_channels)
        if ds_levels is not None and not 1 <= int(ds_levels) <= n_levels:
            raise ValueError(f"ds_levels must be in [1, {n_levels}], got {ds_levels}")
        self.ds_levels = n_levels if ds_levels is None else int(ds_levels)

        self.heads = nn.ModuleList([
            nn.Conv2d(c, self.num_classes, 1) if i < self.ds_levels else nn.Identity()
            for i, c in enumerate(level_channels)
        ])

        # Only the decoder is (re-)initialised; init_weights_he does not touch nn.LayerNorm.
        init_weights_he(self.reassemble)
        init_weights_he(self.smooth)
        init_weights_he(self.up_blocks)
        init_weights_he(self.heads)

    @property
    def num_ds_outputs(self) -> int:
        return self.ds_levels if self.deep_supervision else 1

    @accepts_any_size
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        feats = self.encoder.forward_intermediates(
            x, indices=self._indices, output_fmt="NCHW", intermediates_only=True,
        )
        if len(feats) != len(self.reassemble):
            raise RuntimeError(
                f"backbone returned {len(feats)} intermediates for indices "
                f"{self._indices}; the pyramid needs {len(self.reassemble)}"
            )
        if not self._taps_checked:
            # feature_info is metadata; this is the tensor that was actually delivered.
            got = [int(f.shape[1]) for f in feats]
            if any(c != self._embed_dim for c in got):
                raise RuntimeError(
                    f"backbone {self.backbone_name!r} declares tap width "
                    f"{self._embed_dim} via feature_info but forward_intermediates "
                    f"delivered {got} for indices {self._indices}"
                )
            self._taps_checked = True

        lateral = [r(f) for r, f in zip(self.reassemble, feats)]

        # Top-down fusion, coarse to fine. The upsample takes an explicit size, not a scale
        # factor: the coarsest level is ceil(n / 2) and doubling it back overshoots on odd n.
        p = lateral[-1]
        pyramid = [self.smooth[-1](p)]
        for i in range(len(lateral) - 2, -1, -1):
            p = F.interpolate(p, size=lateral[i].shape[-2:], mode="nearest") + lateral[i]
            pyramid.append(self.smooth[i](p))
        pyramid.reverse()                       # fine to coarse: strides 4, 8, 16, 32

        h = pyramid[0]
        fine: list[torch.Tensor] = []
        for block in self.up_blocks:
            h = block(h)
            fine.append(h)

        # Fine to coarse: strides 1, 2, 4, 8, 16, 32.
        maps = [fine[-1], fine[-2]] + pyramid

        want_ds = self.deep_supervision and self.training
        n_out = self.ds_levels if want_ds else 1
        out = [self.heads[i](maps[i]) for i in range(n_out)]
        return out if want_ds else out[0]


#: Prefix under which every encoder parameter lives inside :class:`ViTFPN`.
_ENC = "encoder."


@dataclass(frozen=True)
class FinetuneGroup:
    """One optimiser parameter group: name, learning rate, members.
    ``param_names`` are names on the unwrapped model, in the model's own iteration order."""

    name: str
    lr: float
    param_names: tuple[str, ...]

    @property
    def n_params(self) -> int:
        return len(self.param_names)


def _encoder_of(model: nn.Module) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    blocks = getattr(encoder, "blocks", None) if encoder is not None else None
    if encoder is None or blocks is None or len(blocks) == 0:
        raise TypeError(
            f"train.finetune (two-phase encoder freezing) needs a vit_fpn-shaped model: "
            f"a module with an ``.encoder`` that has ``.blocks``. Got "
            f"{type(model).__name__}"
            + (" (no .encoder)" if encoder is None else " (.encoder has no .blocks)")
            + ". Either set model.name=vit_fpn or drop the train.finetune block -- there "
            f"is no meaningful 'unfreeze the last 4 transformer blocks' for this "
            f"architecture, and silently freezing nothing would run to completion looking "
            f"healthy."
        )
    return encoder


def _layernorm_param_names(encoder: nn.Module) -> set[str]:
    """``encoder.*`` names of every parameter owned by an :class:`nn.LayerNorm`.
    Matched by ``isinstance``, not by name pattern: a norm missed here is silently frozen."""
    out: set[str] = set()
    for mod_name, module in encoder.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for p_name, _ in module.named_parameters(recurse=False):
            out.add(f"{_ENC}{mod_name}.{p_name}" if mod_name else f"{_ENC}{p_name}")
    return out


def plan_finetune(
    model: nn.Module,
    *,
    phase: int,
    enc_lr: float,
    dec_lr: float,
    layer_decay: float = 0.8,
    unfreeze_last_blocks: int = 4,
) -> list[FinetuneGroup]:
    """The parameter groups phase ``phase`` trains. Pure: nothing is mutated here. Phase 1
    is the decoder alone; phase 2 adds one group per unfrozen block (deepest first, at
    ``enc_lr * layer_decay ** k``), then the remaining encoder LayerNorms."""
    phase = int(phase)
    if phase not in (1, 2):
        raise ValueError(f"train.finetune.phase must be 1 or 2, got {phase!r}")
    enc_lr, dec_lr, layer_decay = float(enc_lr), float(dec_lr), float(layer_decay)
    unfreeze_last_blocks = int(unfreeze_last_blocks)
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError(f"train.finetune.layer_decay must be in (0, 1], got {layer_decay}")

    encoder = _encoder_of(model)
    depth = len(encoder.blocks)
    if not 0 <= unfreeze_last_blocks <= depth:
        raise ValueError(
            f"train.finetune.unfreeze_last_blocks={unfreeze_last_blocks} but the encoder "
            f"has {depth} blocks"
        )

    all_names = [n for n, _ in model.named_parameters()]
    decoder = tuple(n for n in all_names if not n.startswith(_ENC))
    groups = [FinetuneGroup("decoder", dec_lr, decoder)]
    if phase == 1:
        # Phase 1 freezes the encoder completely.
        return [g for g in groups if g.n_params]

    unfrozen_idx = list(range(depth - unfreeze_last_blocks, depth))
    for k, idx in enumerate(reversed(unfrozen_idx)):        # deepest block first
        prefix = f"{_ENC}blocks.{idx}."
        members = tuple(n for n in all_names
                        if n.startswith(prefix))
        groups.append(FinetuneGroup(f"encoder.blocks.{idx}", enc_lr * layer_decay ** k,
                                    members))

    in_unfrozen_block = {n for n in all_names
                         if any(n.startswith(f"{_ENC}blocks.{i}.") for i in unfrozen_idx)}
    rest = tuple(n for n in all_names
                 if n in _layernorm_param_names(encoder) and n not in in_unfrozen_block
                 )
    groups.append(FinetuneGroup("encoder.layernorm_rest",
                                enc_lr * layer_decay ** unfreeze_last_blocks, rest))
    return [g for g in groups if g.n_params]


def apply_finetune_plan(model: nn.Module, groups: Sequence[FinetuneGroup]) -> None:
    """Freeze everything, then unfreeze exactly the planned parameters, and assert it.
    The assertion is the point: training the wrong subset produces a healthy-looking run
    whose only symptom is a slightly worse final number."""
    named = dict(model.named_parameters())
    planned: list[str] = []
    for g in groups:
        for name in g.param_names:
            if name not in named:
                raise KeyError(
                    f"fine-tune group {g.name!r} names parameter {name!r}, which this "
                    f"model does not have"
                )
            planned.append(name)
    duplicated = sorted({n for n in planned if planned.count(n) > 1})
    if duplicated:
        raise ValueError(
            f"fine-tune groups overlap on {len(duplicated)} parameter(s) -- a parameter in "
            f"two groups would get two learning rates and two weight-decay decisions: "
            f"{duplicated[:4]}"
        )

    want = set(planned)
    for name, p in named.items():
        p.requires_grad_(name in want)

    got = {name for name, p in named.items() if p.requires_grad}
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        raise AssertionError(
            f"freezing did not take: {len(missing)} planned parameter(s) are not trainable "
            f"{missing[:4]}, {len(extra)} unplanned parameter(s) are "
            f"{extra[:4]}"
        )


def format_finetune_plan(model: nn.Module, groups: Sequence[FinetuneGroup]) -> str:
    """A printable table of group / learning rate / tensor count / parameter count."""
    named = dict(model.named_parameters())
    lines = [f"{'group':<26} {'lr':>10} {'tensors':>8} {'params':>13}"]
    total = 0
    for g in groups:
        n = sum(int(named[k].numel()) for k in g.param_names)
        total += n
        lines.append(f"{g.name:<26} {g.lr:>10.3e} {g.n_params:>8d} {n:>13,d}")
    frozen = sum(int(p.numel()) for p in named.values()) - total
    lines.append(f"{'-- trainable':<26} {'':>10} {'':>8} {total:>13,d}")
    lines.append(f"{'-- frozen':<26} {'':>10} {'':>8} {frozen:>13,d}")
    return "\n".join(lines)
