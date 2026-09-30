"""Final segmentation objective and signed-distance map helpers."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

NUM_CLASSES = 10
IGNORE_INDEX = -1

#: Height-normalised distance assigned to a class with no ground-truth pixels.
DEFAULT_ABSENT_FILL = 1.0

#: ``'scipy'`` is the reference kernel; ``'cv2'`` is the same L2 transform, ~3.4x faster.
BACKENDS = ("scipy", "cv2")


def _edt(mask: np.ndarray, backend: str) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest zero of ``mask``."""
    if backend == "cv2":
        import cv2

        return cv2.distanceTransform(
            np.ascontiguousarray(mask, dtype=np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
    if backend != "scipy":
        raise ValueError(f"unknown EDT backend {backend!r}; known: {BACKENDS}")
    return distance_transform_edt(mask)


def signed_distance_maps(
    mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
    *,
    normalize_by_height: bool = True,
    absent_fill: float = DEFAULT_ABSENT_FILL,
    clip: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    backend: str = "scipy",
) -> np.ndarray:
    """Per-class signed EDT of ``mask``: negative inside the class, positive outside.
    ``mask`` is ``(H, W)`` integer labels; returns ``(num_classes, H, W)`` float32, divided
    by ``H`` when ``normalize_by_height`` so the units match the evaluator's MASD."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected (H, W) labels, got {mask.shape}")
    labelled = (mask >= 0) & (mask < num_classes) & (mask != ignore_index)
    regions = [(mask == c) & labelled for c in range(num_classes)]
    return _signed_from_regions(
        regions, mask.shape, labelled.any(),
        normalize_by_height=normalize_by_height, absent_fill=absent_fill,
        clip=clip, backend=backend,
    )


def signed_distance_maps_from_intervals(
    lo: np.ndarray,
    hi: np.ndarray,
    num_classes: int = NUM_CLASSES,
    *,
    normalize_by_height: bool = True,
    absent_fill: float = DEFAULT_ABSENT_FILL,
    clip: float | None = None,
    backend: str = "scipy",
) -> np.ndarray:
    """The same maps where a pixel's label is an interval of classes ``[lo, hi]``. A class
    no interval admits is absent and gets ``absent_fill``; a class that is merely unlocated
    gets the distance to its band, so a partial label must never use the exact version."""
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    if lo.shape != hi.shape or lo.ndim != 2:
        raise ValueError(f"lo and hi must both be (H, W); got {lo.shape} and {hi.shape}")
    known = (lo >= 0) & (hi >= lo) & ~((lo == 0) & (hi == num_classes - 1))
    regions = [known & (lo <= c) & (c <= hi) for c in range(num_classes)]
    return _signed_from_regions(
        regions, lo.shape, known.any(),
        normalize_by_height=normalize_by_height, absent_fill=absent_fill,
        clip=clip, backend=backend,
    )


def _signed_from_regions(
    regions: list[np.ndarray],
    shape: tuple[int, int],
    any_labelled: bool,
    *,
    normalize_by_height: bool,
    absent_fill: float,
    clip: float | None,
    backend: str,
) -> np.ndarray:
    H, W = shape
    out = np.zeros((len(regions), H, W), dtype=np.float32)
    if not any_labelled:
        return out

    scale = 1.0 / float(H) if normalize_by_height else 1.0

    for c, pos in enumerate(regions):
        # ``absent_fill`` is already in output units, so it bypasses the height scaling.
        if not pos.any():
            out[c] = absent_fill
            continue
        neg = ~pos
        if not neg.any():
            # Class fills the frame: EDT has no zero to measure to, so mirror the absent case.
            out[c] = -absent_fill
            continue
        # Inside: distance to the nearest outside pixel, shifted so the outer ring sits at 0.
        signed = _edt(neg, backend) * neg - (_edt(pos, backend) - 1) * pos
        out[c] = signed * scale

    if clip is not None:
        np.clip(out, -clip, clip, out=out)
    return out.astype(np.float32, copy=False)


class SignedDistanceTransform:

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        normalize_by_height: bool = True,
        absent_fill: float = DEFAULT_ABSENT_FILL,
        clip: float | None = None,
        ignore_index: int = IGNORE_INDEX,
        backend: str = "scipy",
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown EDT backend {backend!r}; known: {BACKENDS}")
        self.num_classes = num_classes
        self.normalize_by_height = normalize_by_height
        self.absent_fill = absent_fill
        self.clip = clip
        self.ignore_index = ignore_index
        self.backend = backend

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return signed_distance_maps(
            mask,
            self.num_classes,
            normalize_by_height=self.normalize_by_height,
            absent_fill=self.absent_fill,
            clip=self.clip,
            ignore_index=self.ignore_index,
            backend=self.backend,
        )

    def __repr__(self) -> str:
        return (f"SignedDistanceTransform(num_classes={self.num_classes}, "
                f"normalize_by_height={self.normalize_by_height}, "
                f"absent_fill={self.absent_fill}, clip={self.clip}, "
                f"backend={self.backend!r})")


class BoundaryLoss(nn.Module):
    """Excess signed-distance loss over labelled pixels."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist: torch.Tensor | None = None,
                interval: torch.Tensor | None = None) -> torch.Tensor:
        if dist is None:
            raise ValueError(
                "BoundaryLoss needs signed distance maps. Build the dataset with "
                "need_distance_map=True and pass dist=(B,10,H,W) to the loss."
            )
        if dist.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                f"dist {tuple(dist.shape)} and logits {tuple(logits.shape)} disagree on "
                "spatial size; deep supervision must downsample both together"
            )

        prob = logits.softmax(dim=1)
        dist = dist.to(prob.dtype)

        if interval is None:
            keep = target != self.ignore_index
            floor = dist.gather(1, target.clamp_min(0).unsqueeze(1))
        else:
            # A merged band carries ``target == ignore`` yet is fully supervised here: every
            # class in its interval has this band as its region and the rest are pushed away.
            lo, hi = interval[:, 0], interval[:, 1]
            keep = (lo >= 0) & (hi >= lo) & ~((lo == 0) & (hi == self.num_classes - 1))
            idx = torch.arange(self.num_classes, device=dist.device).view(1, -1, 1, 1)
            in_set = (idx >= lo.unsqueeze(1)) & (idx <= hi.unsqueeze(1))
            # The shift is a per-pixel constant with no gradient, so any interval member
            # would do; the smallest keeps the reported value >= 0.
            floor = dist.masked_fill(~in_set, float("inf")).amin(dim=1, keepdim=True)
            floor = torch.where(torch.isfinite(floor), floor, torch.zeros_like(floor))

        valid = keep.unsqueeze(1).to(prob.dtype)
        n_valid = valid.sum()
        if n_valid.item() == 0:
            return logits.sum() * 0.0

        per_pixel = (prob * dist).sum(dim=1, keepdim=True)
        per_pixel = per_pixel - floor
        return (per_pixel * valid).sum() / n_valid


def verify_backends(mask: np.ndarray | None = None, tol: float = 1e-3) -> float:
    """Check the cv2 kernel still reproduces the scipy one; returns the max deviation.
    Guards against a build where ``DIST_MASK_PRECISE`` degrades to a chamfer transform."""
    if mask is None:
        mask = np.zeros((97, 61), dtype=np.uint8)
        row = 0
        for c in range(NUM_CLASSES):
            mask[row:row + 9] = c
            row += 9
        mask[40:60, 10:30] = 4
    a = signed_distance_maps(mask, backend="scipy")
    b = signed_distance_maps(mask, backend="cv2")
    dev = float(np.abs(a - b).max())
    if dev > tol:
        raise AssertionError(f"cv2 EDT deviates from scipy by {dev:.4g} > {tol}")
    return dev


def warmup_ramp(epoch: int, warmup_epochs: int) -> float:
    """Linear 0 -> 1 ramp over ``warmup_epochs``, exactly 0 at epoch 0."""
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch / float(warmup_epochs))))


_TERMS = frozenset({"cross_entropy", "dice", "boundary"})


def _downsample_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour resample of a ``(B,H,W)`` int64 label map, preserving ``-1``."""
    if tuple(target.shape[-2:]) == size:
        return target
    t = F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest")
    return t.squeeze(1).long()


def _downsample_interval(interval: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest resample of a ``(B,2,H,W)`` int64 interval, in step with the target.
    Both bounds ride the same grid, so ``lo`` and ``hi`` always come from one source pixel."""
    if tuple(interval.shape[-2:]) == size:
        return interval
    t = F.interpolate(interval.float(), size=size, mode="nearest")
    return t.long()


def _downsample_dist(dist: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resample ``(B,C,H,W)`` distance maps to match a deep-supervision head.
    Nearest, not bilinear: the maps must stay aligned with the nearest-resampled target."""
    if tuple(dist.shape[-2:]) == size:
        return dist
    return F.interpolate(dist, size=size, mode="nearest")


class CrossEntropyTerm(nn.Module):
    """Weighted cross-entropy, honouring ``ignore_index``."""

    def __init__(self, class_weights: torch.Tensor | None = None,
                 ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().float().clone())

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist: torch.Tensor | None = None,
                interval: torch.Tensor | None = None) -> torch.Tensor:
        if interval is None:
            if (target != self.ignore_index).sum().item() == 0:
                return logits.sum() * 0.0        # all-unlabelled batch: no signal, no NaN
            weight = self.class_weights
            if weight is not None:
                weight = weight.to(device=logits.device, dtype=logits.dtype)
            return F.cross_entropy(
                logits, target, weight=weight,
                ignore_index=self.ignore_index,
            )
        return self._interval_forward(logits, interval)

    def _interval_forward(self, logits: torch.Tensor,
                          interval: torch.Tensor) -> torch.Tensor:
        """``-log sum_{c in [lo,hi]} p_c``, the set-valued generalisation of the above.
        ``interval`` is ``(B,2,H,W)`` int64; unknown pixels and all-class intervals are
        dropped BEFORE the normaliser."""
        if logits.dtype in (torch.float16, torch.bfloat16):
            # Only where it buys something: the value is a difference of two nearly-equal
            # log-sum-exps, and in bf16 a well-fitted interval rounds to exactly zero.
            logits = logits.float()
        n_cls = logits.shape[1]
        lo = interval[:, 0]
        hi = interval[:, 1]
        # An interval spanning every class says nothing; so does (-1,-1).
        valid = (lo >= 0) & (hi >= lo) & ~((lo == 0) & (hi == n_cls - 1))
        if not bool(valid.any()):
            return logits.sum() * 0.0
        # Substituting the full range for invalid pixels avoids an all -inf row (0 * inf = NaN).
        lo = torch.where(valid, lo, torch.zeros_like(lo))
        hi = torch.where(valid, hi, torch.full_like(hi, n_cls - 1))

        idx = torch.arange(n_cls, device=logits.device).view(1, n_cls, 1, 1)
        in_set = (idx >= lo.unsqueeze(1)) & (idx <= hi.unsqueeze(1))

        lse_all = torch.logsumexp(logits, dim=1)
        lse_set = torch.logsumexp(logits.masked_fill(~in_set, float("-inf")), dim=1)
        nll = lse_all - lse_set

        weight = self.class_weights
        weight = (torch.ones(n_cls, device=logits.device, dtype=logits.dtype)
                  if weight is None else weight.to(logits.device, logits.dtype))
        # The interval's weight is its members' mean, i.e. exactly ``w_y`` for a singleton.
        w_q = (weight.view(1, n_cls, 1, 1) * in_set).sum(dim=1) / in_set.sum(dim=1)

        num = w_q * nll
        keep = valid.to(num.dtype)
        return (num * keep).sum() / (w_q * keep).sum().clamp_min(1e-12)


class DiceTerm(nn.Module):
    """Soft Dice over softmax probabilities, per image, ignoring unlabelled pixels.
    ``include_background`` defaults to True because class 0 is scored like any other."""

    def __init__(self, num_classes: int = NUM_CLASSES, include_background: bool = True,
                 ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        if not include_background:
            warnings.warn(
                "dice.include_background=False drops class 0, which the official metric "
                "scores like any other class", RuntimeWarning, stacklevel=2,
            )
        self.num_classes = num_classes
        self.include_background = include_background
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist: torch.Tensor | None = None,
                interval: torch.Tensor | None = None) -> torch.Tensor:
        # ``interval`` is accepted and ignored: Dice needs a hard region per class and a
        # merged band has none, so this term scores the exact pixels only.
        prob = logits.softmax(dim=1)
        valid = (target != self.ignore_index)
        if valid.sum().item() == 0:
            return logits.sum() * 0.0

        onehot = F.one_hot(target.clamp_min(0), self.num_classes)
        onehot = onehot.permute(0, 3, 1, 2).to(prob.dtype)
        m = valid.unsqueeze(1).to(prob.dtype)
        prob = prob * m
        onehot = onehot * m

        dims = (2, 3)
        inter = (prob * onehot).sum(dim=dims)
        denom = prob.sum(dim=dims) + onehot.sum(dim=dims)
        dice = (2.0 * inter + 1e-5) / (denom + 1e-5)

        start = 0 if self.include_background else 1
        dice = dice[:, start:]

        # Average only over images with labelled pixels, so unlabelled ones add no free zero.
        per_image = 1.0 - dice.mean(dim=1)
        keep = valid.flatten(1).any(dim=1).to(per_image.dtype)
        return (per_image * keep).sum() / keep.sum().clamp_min(1.0)


class _Term:
    """A configured term: the module, its static weight, and its warmup schedule."""

    def __init__(self, name: str, module: nn.Module, weight: float, warmup_epochs: int) -> None:
        self.name = name
        self.module = module
        self.weight = float(weight)
        self.warmup_epochs = int(warmup_epochs)

    def ramp(self, epoch: int) -> float:
        return warmup_ramp(epoch, self.warmup_epochs)


class CompoundLoss(nn.Module):
    """Sum of configured terms, applied across every deep-supervision head. ``forward``
    returns ``(loss, logs)``; ``logs`` holds each term's unweighted value and live ramp."""

    def __init__(self, terms: Sequence[_Term], ds_weights: Sequence[float] | None = None,
                 ds_enabled: bool = True) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompoundLoss needs at least one term")
        self._terms = list(terms)
        self.term_modules = nn.ModuleList([t.module for t in self._terms])
        self.ds_enabled = bool(ds_enabled)
        self.ds_weights = list(ds_weights) if ds_weights else [1.0]

    def level_weights(self, n_levels: int) -> list[float]:
        """Deep-supervision weights for ``n_levels`` heads, renormalised to sum to 1. The
        model, not the config, says how many heads exist; a short list is extended by halving."""
        w = list(self.ds_weights[:n_levels])
        while len(w) < n_levels:
            w.append(w[-1] * 0.5 if w else 1.0)
        total = float(sum(w))
        if total <= 0:
            raise ValueError(f"deep supervision weights must be positive, got {self.ds_weights}")
        return [x / total for x in w]

    def forward(self, pred: torch.Tensor | list[torch.Tensor], target: torch.Tensor,
                epoch: int = 0, dist: torch.Tensor | None = None,
                interval: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, dict[str, float]]:
        preds = list(pred) if isinstance(pred, (list, tuple)) else [pred]
        if not self.ds_enabled:
            preds = preds[:1]
        weights = self.level_weights(len(preds))

        if target.ndim != 3:
            raise ValueError(f"target must be (B,H,W) int64, got {tuple(target.shape)}")
        if preds[0].shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"head 0 must be full resolution: got {tuple(preds[0].shape[-2:])} "
                f"vs target {tuple(target.shape[-2:])}"
            )
        needs_dist = any(isinstance(t.module, BoundaryLoss) and t.weight != 0.0
                         for t in self._terms)
        if dist is None and needs_dist:
            # Checked up front rather than when the ramp turns positive, which would let the
            # run train through the whole warmup and then crash.
            raise ValueError(
                "the loss config includes a boundary term but no distance maps were "
                "passed; build the dataset with need_distance_map=True"
            )

        zero = preds[0].sum() * 0.0        # keeps the graph alive when every term is ramped off
        total = zero
        logs: dict[str, float] = {}
        for term in self._terms:
            ramp = term.ramp(epoch)
            acc = zero
            if ramp > 0.0:
                for p, lw in zip(preds, weights):
                    size = (int(p.shape[-2]), int(p.shape[-1]))
                    t = _downsample_target(target, size)
                    d = _downsample_dist(dist, size) if dist is not None else None
                    v = _downsample_interval(interval, size) if interval is not None else None
                    acc = acc + lw * term.module(p, t, d, v)
            total = total + term.weight * ramp * acc
            logs[term.name] = float(acc.detach())
            if term.warmup_epochs > 0:
                logs[f"{term.name}_ramp"] = ramp
        logs["total"] = float(total.detach())
        return total, logs

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{t.name}(w={t.weight}" + (f", warmup={t.warmup_epochs})" if t.warmup_epochs else ")")
            for t in self._terms
        )
        return f"CompoundLoss({parts}, ds={self.ds_enabled}:{self.ds_weights})"


_SCHEMES = ("auto_inverse_sqrt_freq",)


def _weights_signature(mask_paths: Sequence[Path], num_classes: int, scheme: str) -> str:
    h = hashlib.sha1()
    h.update(f"{scheme}|{num_classes}|{len(mask_paths)}|".encode())
    for p in sorted(str(p) for p in mask_paths):
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def count_class_pixels(samples: Iterable[Any], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Total pixels per class over the labelled members of ``samples``.
    Accepts anything with a ``.mask`` attribute or a bare path; masks are read as grayscale."""
    import cv2

    counts = np.zeros(num_classes, dtype=np.int64)
    for s in samples:
        # An interval label is a uint8 PNG too and every byte would be counted as a class
        # id, so most classes read as empty. Refuse rather than define a set-valued frequency.
        if getattr(s, "label_kind", "exact") != "exact":
            raise ValueError(
                "class weights must be derived from exact ten-class labels only; "
                "filter the pool before calling compute_class_weights()"
            )
        path = getattr(s, "mask", s)
        if path is None:
            continue
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            warnings.warn("unreadable mask skipped", RuntimeWarning, stacklevel=2)
            continue
        counts += np.bincount(m.ravel(), minlength=256)[:num_classes]
    return counts


def _weights_from_counts(counts: np.ndarray, scheme: str) -> np.ndarray:
    freq = counts.astype(np.float64)
    freq = freq / max(freq.sum(), 1.0)
    present = freq > 0
    if not present.any():
        raise ValueError("no labelled pixels found; cannot derive class weights")

    raw = np.zeros_like(freq)
    raw[present] = 1.0 / np.sqrt(freq[present])

    # A class with zero pixels has no inverse frequency; cap it at the largest observed one.
    if not present.all():
        missing = np.flatnonzero(~present).tolist()
        warnings.warn(
            f"classes {missing} have zero pixels in the labelled pool; their weights are "
            "capped at the maximum observed weight", RuntimeWarning, stacklevel=2,
        )
        raw[~present] = raw[present].max()

    return raw / raw.mean()


def compute_class_weights(samples: Iterable[Any], num_classes: int = NUM_CLASSES,
                          scheme: str = "auto_inverse_sqrt_freq",
                          cache_path: Path | str | None = None) -> torch.Tensor:
    """Per-class loss weights from pixel frequency, cached as JSON. ``auto_inverse_sqrt_freq``
    gives ``1/sqrt(freq_c)`` normalised to mean 1, a 4-6x spread rather than the ~50x of full
    inverse frequency. The cache is keyed by the mask paths, so another pool recomputes."""
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown class-weight scheme {scheme!r}; known: {_SCHEMES}")

    samples = list(samples)
    mask_paths = [p for p in (getattr(s, "mask", s) for s in samples) if p is not None]
    sig = _weights_signature(mask_paths, num_classes, scheme)

    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.is_file():
        try:
            blob = json.loads(cache.read_text())
        except json.JSONDecodeError:
            blob = {}
        if blob.get("signature") == sig:
            return torch.tensor(blob["weights"], dtype=torch.float32)
        warnings.warn(
            "class-weight cache was built from a different sample set; recomputing",
            RuntimeWarning, stacklevel=2,
        )

    counts = count_class_pixels(samples, num_classes)
    weights = _weights_from_counts(counts, scheme)
    if not np.isfinite(weights).all():
        raise AssertionError(f"non-finite class weights from counts {counts.tolist()}")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "signature": sig,
            "scheme": scheme,
            "num_classes": num_classes,
            "n_masks": len(mask_paths),
            "pixel_counts": counts.tolist(),
            "weights": weights.tolist(),
        }, indent=2))

    return torch.tensor(weights, dtype=torch.float32)


def _resolve_class_weights(spec: Any, provided: torch.Tensor | None,
                           num_classes: int) -> torch.Tensor | None:
    """Turn ``terms[].class_weights`` into a tensor, or refuse to guess."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, (list, tuple)):
        if len(spec) != num_classes:
            raise ValueError(f"class_weights list has {len(spec)} entries, need {num_classes}")
        return torch.tensor([float(x) for x in spec], dtype=torch.float32)
    if isinstance(spec, str):
        if spec not in _SCHEMES:
            raise ValueError(f"unknown class_weights {spec!r}; known: {_SCHEMES}")
        if provided is None:
            # Silently falling back to uniform would make a weighted arm a copy of the control.
            raise ValueError(
                f"loss config asks for class_weights={spec!r} but none were passed to "
                "build_loss(). Call compute_class_weights(labeled_samples(root), "
                f"scheme={spec!r}, cache_path=...) and pass the result, or set "
                "class_weights: null to train unweighted."
            )
        return provided
    raise TypeError(f"class_weights must be a scheme name, a list or null, got {spec!r}")


def build_compound_loss(cfg: dict, class_weights: torch.Tensor | None = None) -> CompoundLoss:
    """Instantiate the loss described by the resolved ``loss`` config block."""
    num_classes = int(cfg.get("num_classes", NUM_CLASSES))
    ignore_index = int(cfg.get("ignore_index", IGNORE_INDEX))

    if class_weights is not None and len(class_weights) != num_classes:
        raise ValueError(
            f"class_weights has {len(class_weights)} entries, model has {num_classes} classes"
        )

    terms: list[_Term] = []
    for spec in cfg.get("terms", []):
        spec = dict(spec)
        raw_name = str(spec.pop("name"))
        name = raw_name.lower()
        if name not in _TERMS:
            raise ValueError(
                f"unknown loss term {raw_name!r}; known: {sorted(_TERMS)}"
            )
        weight = float(spec.pop("weight", 1.0))
        warmup = int(spec.pop("warmup_epochs", 0))

        if name == "cross_entropy":
            module: nn.Module = CrossEntropyTerm(
                class_weights=_resolve_class_weights(
                    spec.pop("class_weights", None), class_weights, num_classes),
                ignore_index=ignore_index,
            )
        elif name == "dice":
            module = DiceTerm(
                num_classes=num_classes,
                include_background=bool(spec.pop("include_background", True)),
                ignore_index=ignore_index,
            )
        else:
            module = BoundaryLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        if spec:
            raise ValueError(f"unused keys in loss term {raw_name!r}: {sorted(spec)}")
        terms.append(_Term(name, module, weight, warmup))

    ds = cfg.get("deep_supervision") or {}
    return CompoundLoss(
        terms,
        ds_weights=ds.get("weights", [1.0]),
        ds_enabled=bool(ds.get("enabled", False)),
    )


def build_loss(cfg: dict, class_weights: torch.Tensor | None = None) -> CompoundLoss:
    return build_compound_loss(cfg, class_weights=class_weights)
