"""Dataset, cropping and normalisation for the OCT layer-segmentation task.

:class:`OCTSegDataset` turns a list of :class:`~octtta.data.release_dataset.Sample` into
tensors: read, augment on an oversized source crop, then take the final tile.
:func:`build_datasets` is the entry point. Masks are opaque labels
throughout -- nearest resampling only, so no augmentation can invent a class.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from octtta.data.partial_labels import (
    official_intervals_from_synthetic, resolve_synthetic_boundary_map,
)
from octtta.data.release_dataset import INFERENCE_DEVICE, NUM_CLASSES, Sample
from octtta.data import sampling as _sampling

if TYPE_CHECKING:  # avoid a hard import cycle with the augmenter module
    from octtta.data.transforms import Augmenter

# Each DataLoader worker is already a process; letting OpenCV fan out inside it turns
# num_workers=8 into 8*N threads fighting over the same cores.
cv2.setNumThreads(0)

IGNORE_INDEX = -1
IGNORE_UINT8 = 255

#: Seed prefix for the A-Band consistency view's RNG stream, distinct from every other stage's.
CONSISTENCY_RNG_TAG = 0xC0A51

DEFAULT_NORMALIZE: dict[str, Any] = {
    "mode": "per_image_zscore",
    "clip_percentiles": (0.5, 99.5),
}

#: Training-crop policy; overridden by the ``data.crop`` config block.
DEFAULT_CROP: dict[str, Any] = {
    "foreground_prob": 0.6,
    "foreground_classes": None,
    "augment_margin": "auto",
}

#: ``data.official_true_boundaries_only`` when the key is absent.
OFFICIAL_TRUE_BOUNDARIES_ONLY_DEFAULT = True


# --- IO and normalisation ------------------------------------------------------------

def read_image(path: str | Path) -> np.ndarray:
    """Read a B-scan as a single-channel float32 array at its native size."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def read_mask(path: str | Path, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Read a label map as uint8 in ``[0, num_classes)``."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    vals = np.unique(arr)
    if vals.max() >= num_classes:
        scaled = set(range(0, 10 * num_classes, 10))
        if set(vals.tolist()) <= scaled:
            arr = arr // 10
        else:
            raise ValueError(
                f"{path}: mask values {vals.tolist()} are not labels in [0,{num_classes})"
            )
    return arr.astype(np.uint8)


def read_interval_plane(path: str | Path, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Read a partial annotation as the packed ``uint8`` interval plane."""
    from octtta.data.partial_labels import interval_lut

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"could not read interval plane: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)
    lo, _ = interval_lut()
    bad = np.unique(arr[(lo[arr] < 0) & (arr != IGNORE_UINT8)])
    if bad.size:
        raise ValueError(f"{path}: bytes {bad.tolist()} are not interval codes")
    return arr


#: Only the published per-image z-score mode is live.
NORMALIZE_KEYS = frozenset({"mode", "clip_percentiles"})


def normalize_image(image: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Clip and z-score one B-scan as float32 without changing its shape."""
    cfg = {**DEFAULT_NORMALIZE, **(cfg or {})}
    unknown = sorted(set(cfg) - NORMALIZE_KEYS)
    if unknown:
        raise ValueError(f"data.normalize has unknown key(s): {unknown}")
    if cfg["mode"] != "per_image_zscore":
        raise ValueError("only per_image_zscore normalization is supported")
    img = image.astype(np.float32)
    pct = cfg.get("clip_percentiles")
    if pct is not None:
        lo, hi = np.percentile(img, [float(pct[0]), float(pct[1])])
        if hi > lo:
            img = np.clip(img, lo, hi)
    std = float(img.std())
    return (img - float(img.mean())) / std if std > 1e-6 else np.zeros_like(img)


def signed_distance_maps(
    mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_value: int = IGNORE_UINT8,
    height: int | None = None,
) -> np.ndarray:
    """Per-class signed EDT, negative inside the class (Kervadec surface-loss convention)."""
    from octtta.losses import DEFAULT_ABSENT_FILL
    from octtta.losses import signed_distance_maps as _sdm

    arr = np.asarray(mask).astype(np.int32)
    h = int(height) if height else 0
    if h <= 0:
        h = int(arr.shape[0])

    # The delegate can only normalise by its own input's height, so ask it for raw pixel
    # distances and scale here. ``absent_fill`` bypasses the delegate's scaling (it is
    out = _sdm(arr, num_classes,
               normalize_by_height=False,
               absent_fill=DEFAULT_ABSENT_FILL * float(h),
               ignore_index=int(ignore_value), backend="cv2")
    out /= np.float32(h)
    return out


def interval_distance_maps(
    lo: np.ndarray,
    hi: np.ndarray,
    num_classes: int = NUM_CLASSES,
    height: int | None = None,
) -> np.ndarray:
    """:func:`signed_distance_maps` for a label that names a set of classes per pixel."""
    from octtta.losses import DEFAULT_ABSENT_FILL
    from octtta.losses import signed_distance_maps_from_intervals as _sdmi

    h = int(height) if height else 0
    if h <= 0:
        h = int(np.asarray(lo).shape[0])
    out = _sdmi(lo, hi, num_classes,
                normalize_by_height=False,
                absent_fill=DEFAULT_ABSENT_FILL * float(h), backend="cv2")
    out /= np.float32(h)
    return out



class OCTSegDataset(Dataset):
    """B-scans and (where available) their layer masks, at native size unless cropped."""

    def __init__(
        self,
        samples: list[Sample],
        *,
        num_classes: int = NUM_CLASSES,
        augmenter: "Augmenter | None" = None,
        normalize: dict | None = None,
        crop_size: tuple[int, int] | None = None,
        crop: dict | None = None,
        need_distance_map: bool = False,
        seed: int = 0,
        failaug: dict | None = None,
        discaug: dict | None = None,
        consistency: dict | None = None,
        official_true_boundaries_only: bool = False,
        official_synthetic_boundary_map=None,
    ) -> None:
        from octtta.data.discaug import DiscAug  # local: same reason as FailAug below
        from octtta.data.failaug import FailAug  # local: keeps import cost off val-only users

        self.samples = list(samples)
        self.num_classes = int(num_classes)
        self.augmenter = augmenter
        self.normalize = {**DEFAULT_NORMALIZE, **(normalize or {})}
        self.crop_size = tuple(crop_size) if crop_size is not None else None  # type: ignore[assignment]
        self.crop_cfg = {**DEFAULT_CROP, **(crop or {})}
        self.need_distance_map = bool(need_distance_map)
        self.seed = int(seed)
        self._epoch = 0
        self._calls = 0
        # Full resolution, before the crop and outside ``augment.pipeline`` so it cannot
        # inflate ``augment_source_size``; its own RNG stream costs ``_rng`` no draw.
        self.failaug = FailAug.from_config(failaug)
        self._fcalls = 0
        # Synthetic optic disc, the same shape of stage for the same reasons:
        self.discaug = DiscAug.from_config(discaug, num_classes=self.num_classes)
        self._dcalls = 0
        # A-Band consistency. Unlike ``failaug`` this does not replace the sample: it
        # appends ``image_dirty``, a second view of the already cropped tile.
        self.consistency = FailAug.consistency_from_config(consistency)
        self._ccalls = 0
        #: Enter the release's planes as PARTIAL labels, supervising its five real boundaries.
        self.official_true_boundaries_only = bool(official_true_boundaries_only)
        #: Which OFFICIAL boundary each real synthetic line IS: ``beta`` (in place) or ``gamma``.
        self.synthetic_boundary_map = None
        if self.official_true_boundaries_only or official_synthetic_boundary_map is not None:
            self.synthetic_boundary_map = resolve_synthetic_boundary_map(
                official_synthetic_boundary_map)
        self.foreground_prob = float(self.crop_cfg.get("foreground_prob", 0.0))
        if not 0.0 <= self.foreground_prob <= 1.0:
            raise ValueError(
                f"crop.foreground_prob must be in [0, 1], got {self.foreground_prob}"
            )
        fg = self.crop_cfg.get("foreground_classes")
        if fg is None:
            # 0 (above the retina) and num_classes-1 (below it) are ~80% of all pixels and
            # are in almost every tile already; biasing towards them would be a no-op.
            fg = list(range(1, max(1, self.num_classes - 1)))
        self.foreground_classes = tuple(int(c) for c in fg)

        #: Oversized crop handed to the augmenter, before clamping to the source size.
        self.source_crop_size = (
            augment_source_size(self.crop_size, self.augmenter,
                                margin=self.crop_cfg.get("augment_margin", "auto"))
            if self.crop_size is not None else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def add_samples(self, samples) -> None:
        """Append additional labelled training samples in their supplied order."""
        self.samples.extend(samples)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        self._calls += 1
        base = torch.initial_seed() & 0xFFFF_FFFF
        return np.random.default_rng([self.seed, self._epoch, idx, self._calls, base])

    def _failaug_rng(self, idx: int) -> np.random.Generator:
        # Same derivation shape as `_rng` but a disjoint stream: own tag, own counter.
        from octtta.data.failaug import RNG_TAG

        self._fcalls += 1
        base = torch.initial_seed() & 0xFFFF_FFFF
        return np.random.default_rng(
            [RNG_TAG, self.seed, self._epoch, idx, self._fcalls, base])

    def _discaug_rng(self, idx: int) -> np.random.Generator:
        # A fourth disjoint stream: ``discaug.RNG_TAG`` differs from ``failaug.RNG_TAG``.
        from octtta.data.discaug import RNG_TAG as DISC_RNG_TAG

        self._dcalls += 1
        base = torch.initial_seed() & 0xFFFF_FFFF
        return np.random.default_rng(
            [DISC_RNG_TAG, self.seed, self._epoch, idx, self._dcalls, base])

    def _consistency_rng(self, idx: int) -> np.random.Generator:
        # A third disjoint stream. ``CONSISTENCY_RNG_TAG != RNG_TAG`` keeps the dirty view
        # from replaying the failaug draws when both stages are enabled.
        self._ccalls += 1
        base = torch.initial_seed() & 0xFFFF_FFFF
        return np.random.default_rng(
            [CONSISTENCY_RNG_TAG, self.seed, self._epoch, idx, self._ccalls, base])

    def load_numpy(self, idx: int) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Everything up to tensorisation: read, normalise, crop, augment."""
        s = self.samples[idx]
        image = normalize_image(read_image(s.image), self.normalize)
        if s.mask is None:
            mask = None
        elif s.partial:
            mask = read_interval_plane(s.mask, self.num_classes)
        else:
            mask = read_mask(s.mask, self.num_classes)
        meta = {
            "vendor": s.vendor,
            "status": s.status.lower(),
            "device": s.device,
            "stem": s.stem,
            "anatomy": "Macula",
            "orig_hw": (int(image.shape[0]), int(image.shape[1])),
            "label_kind": s.label_kind,
        }

        if self.failaug is not None:
            image, mask = self.failaug(image, mask, self._failaug_rng(idx))

        if self.discaug is not None:
            # ``partial`` and ``protocol`` come from the SAMPLE, never from the plane's
            # values: byte 5 is class 5 on an exact plane and the interval [0, 5] on a partial one.
            image, mask, fired = self.discaug(
                image, mask, self._discaug_rng(idx),
                partial=s.partial, protocol=s.protocol)
            # Emitted on every sample of a discaug-enabled dataset, 0 or 1, so the trainer
            # can report what actually fired instead of what was configured.
            meta["discaug_fired"] = int(fired)

        if self.crop_size is None and self.augmenter is None:
            return image, self._true5(mask, s, meta), meta

        rng = self._rng(idx)
        if self.crop_size is not None:
            centre = None
            # Not for interval labels: ``_foreground_centre`` reads the plane's values as
            # class ids, which an interval byte is not.
            if mask is not None and not s.partial and self.foreground_prob > 0.0 \
                    and rng.random() < self.foreground_prob:
                centre = _foreground_centre(mask, rng, self.foreground_classes)
            image, mask = _random_crop(
                image, mask, self._clamped_source_crop(image.shape), rng, centre=centre,
            )
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask, rng)
        if self.crop_size is not None and image.shape != self.crop_size:
            image, mask = _centre_crop(image, mask, self.crop_size)
        return image, self._true5(mask, s, meta), meta

    def _true5(self, mask, s: Sample, meta: dict):
        """Re-encode an exact plane to supervise the release's five REAL boundaries only."""
        if not self.official_true_boundaries_only or mask is None or s.partial:
            return mask
        meta["label_kind"] = "interval"
        return official_intervals_from_synthetic(mask, self.synthetic_boundary_map)

    def _clamped_source_crop(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Oversized crop, shrunk to what this image can actually supply."""
        assert self.crop_size is not None and self.source_crop_size is not None
        (ch, cw), (sh, sw) = self.crop_size, self.source_crop_size
        h, w = int(shape[0]), int(shape[1])
        return (max(int(ch), min(int(sh), h)), max(int(cw), min(int(sw), w)))

    def finalize(self, image: np.ndarray, mask: np.ndarray | None, meta: dict) -> dict:
        """Arrays -> the tensor dict the training loop consumes."""
        image = np.ascontiguousarray(image, dtype=np.float32)
        out: dict[str, Any] = {"image": torch.from_numpy(image[None])}

        partial = meta.get("label_kind") == "interval"
        if mask is None:
            labels = np.full(image.shape, IGNORE_UINT8, dtype=np.uint8)
        elif partial:
            labels = np.ascontiguousarray(mask, dtype=np.uint8)
        else:
            labels = np.ascontiguousarray(mask, dtype=np.uint8)
            bad = np.unique(labels)
            bad = bad[(bad >= self.num_classes) & (bad != IGNORE_UINT8)]
            if bad.size:
                raise ValueError(f"{meta['stem']}: illegal mask labels {bad.tolist()}")

        # Native height, not the crop's: the metric's MASD/H uses the height of the whole
        # B-scan, and the surface loss is only interpretable if it agrees.
        native_h = int(meta.get("orig_hw", labels.shape)[0])

        if partial:
            from octtta.data.partial_labels import exact_from_code, unpack_interval

            lo, hi = unpack_interval(labels)
            exact = exact_from_code(labels)
            as_i64 = exact.astype(np.int64)
            as_i64[exact == IGNORE_UINT8] = IGNORE_INDEX
            out["mask"] = torch.from_numpy(as_i64)
            # Emitted only for interval samples; an ordinary ten-class batch carries no
            # ``interval`` key and the loss sees ``interval=None``.
            out["interval"] = torch.from_numpy(
                np.stack([lo, hi]).astype(np.int64))
            if self.need_distance_map:
                out["dist"] = torch.from_numpy(
                    interval_distance_maps(lo, hi, self.num_classes, height=native_h))
        else:
            as_i64 = labels.astype(np.int64)
            as_i64[labels == IGNORE_UINT8] = IGNORE_INDEX
            out["mask"] = torch.from_numpy(as_i64)
            if self.need_distance_map:
                dist = signed_distance_maps(labels, self.num_classes, height=native_h)
                out["dist"] = torch.from_numpy(dist)

        out.update(meta)
        return out

    def __getitem__(self, idx: int) -> dict:
        image, mask, meta = self.load_numpy(idx)
        out = self.finalize(image, mask, meta)
        if self.consistency is not None:
            # After load_numpy, i.e. after the crop, the augmenter and the centre crop:
            # the dirty view must be the same tile the clean view is.
            dirty, _ = self.consistency(image, None, self._consistency_rng(idx))
            # The dirty view has the same geometry and one channel as the clean tile.
            out["image_dirty"] = torch.from_numpy(
                np.ascontiguousarray(dirty, dtype=np.float32)[None])
        return out


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray | None,
    crop: tuple[int, int],
    rng: np.random.Generator,
    *,
    centre: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(h, w)`` crop, replicate-padding the image and ignore-padding the mask first."""
    ch, cw = int(crop[0]), int(crop[1])
    h, w = image.shape
    pad_h, pad_w = max(0, ch - h), max(0, cw - w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
        if mask is not None:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_UINT8)
        h, w = image.shape

    if centre is None:
        y = int(rng.integers(0, h - ch + 1))
        x = int(rng.integers(0, w - cw + 1))
    else:
        y = min(max(int(centre[0]) - ch // 2, 0), h - ch)
        x = min(max(int(centre[1]) - cw // 2, 0), w - cw)
    cropped_mask = mask[y:y + ch, x:x + cw] if mask is not None else None
    return image[y:y + ch, x:x + cw], cropped_mask


def _centre_crop(
    image: np.ndarray,
    mask: np.ndarray | None,
    crop: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Centre window of ``crop`` size; a no-op along any axis already small enough."""
    ch, cw = int(crop[0]), int(crop[1])
    h, w = image.shape
    y, x = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
    cropped_mask = mask[y:y + ch, x:x + cw] if mask is not None else None
    return image[y:y + ch, x:x + cw], cropped_mask


def _foreground_centre(
    mask: np.ndarray,
    rng: np.random.Generator,
    classes: Sequence[int],
) -> tuple[int, int] | None:
    """A random pixel of a random class present in ``mask``; ``None`` if none are."""
    counts = np.bincount(np.asarray(mask).ravel(), minlength=256)
    available = [c for c in classes if 0 <= c < counts.size and counts[c] > 0]
    if not available:
        return None
    target = int(available[int(rng.integers(len(available)))])
    flat = np.flatnonzero(np.asarray(mask).ravel() == target)
    pixel = int(flat[int(rng.integers(flat.size))])
    return divmod(pixel, int(mask.shape[1]))


def augment_source_size(
    crop: tuple[int, int] | None,
    augmenter: "Augmenter | None",
    margin: Any = "auto",
) -> tuple[int, int] | None:
    """How large a source window the augmenter needs to fill ``crop`` with real tissue."""
    if crop is None:
        return None
    ch, cw = float(crop[0]), float(crop[1])
    if margin is None or margin is False:
        return int(ch), int(cw)
    if not isinstance(margin, str):
        m = float(margin)
        if m < 1.0:
            raise ValueError(f"crop.augment_margin must be >= 1.0, got {m}")
        return int(np.ceil(ch * m)), int(np.ceil(cw * m))
    if margin != "auto":
        raise ValueError(f"crop.augment_margin must be 'auto' or a number, got {margin!r}")
    if augmenter is None:
        return int(ch), int(cw)

    # Reuse the augmenter's own range parsing: a scalar means [-v, v] for translation but
    # a constant for a scale range, and re-deriving that here would drift from it silently.
    from octtta.data.transforms import _pair, _symmetric

    ops = {op.name: op.params for op in getattr(augmenter, "ops", ())}

    if "elastic" in ops:
        alpha = float(ops["elastic"].get("alpha", 20.0))
        ch, cw = ch + 2.0 * alpha, cw + 2.0 * alpha

    if "random_rotate" in ops:
        lo, hi = _symmetric(ops["random_rotate"].get("degrees", 5.0), name="degrees")
        theta = np.deg2rad(max(abs(lo), abs(hi)))
        cos_t, sin_t = abs(float(np.cos(theta))), abs(float(np.sin(theta)))
        ch, cw = ch * cos_t + cw * sin_t, cw * cos_t + ch * sin_t

    if "random_translate" in ops:
        params = ops["random_translate"]
        default = params.get("frac", 0.05)
        fy = max(abs(v) for v in _symmetric(params.get("axial_frac", default)))
        fx = max(abs(v) for v in _symmetric(params.get("lateral_frac", default)))
        ch /= max(1e-3, 1.0 - 2.0 * min(fy, 0.49))
        cw /= max(1e-3, 1.0 - 2.0 * min(fx, 0.49))

    sy = sx = 1.0
    if "random_scale" in ops:
        params = ops["random_scale"]
        smallest = min(_pair(params.get("range", (0.9, 1.1)), name="range"))
        axis = str(params.get("axis", "both"))
        if axis in ("both", "axial"):
            sy *= smallest
        if axis in ("both", "lateral"):
            sx *= smallest
    if "random_anisotropic_scale" in ops:
        params = ops["random_anisotropic_scale"]
        sy *= min(_pair(params.get("axial", (0.85, 1.2)), name="axial"))
        sx *= min(_pair(params.get("lateral", (0.85, 1.2)), name="lateral"))
    ch /= max(1e-3, min(1.0, sy))
    cw /= max(1e-3, min(1.0, sx))

    return int(np.ceil(ch)), int(np.ceil(cw))


# --- Collation -----------------------------------------------------------------------

_TENSOR_KEYS = ("image", "mask", "interval", "dist", "image_dirty")


def _meta_keys(batch: list[dict]) -> list[str]:
    return [k for k in batch[0] if k not in _TENSOR_KEYS]


def collate_pad(batch: list[dict]) -> dict:
    """Pad a batch of differently-sized items to the batch maximum."""
    if not batch:
        raise ValueError("empty batch")
    max_h = max(int(b["image"].shape[-2]) for b in batch)
    max_w = max(int(b["image"].shape[-1]) for b in batch)

    images, masks, dists, pad_hw = [], [], [], []
    dirty: list[torch.Tensor] = []
    intervals: list[torch.Tensor] = []
    any_interval = any("interval" in b for b in batch)
    for b in batch:
        h, w = int(b["image"].shape[-2]), int(b["image"].shape[-1])
        pad = (0, max_w - w, 0, max_h - h)  # F.pad order: last dim first
        images.append(torch.nn.functional.pad(b["image"], pad, value=0.0))
        masks.append(torch.nn.functional.pad(b["mask"], pad, value=IGNORE_INDEX))
        if any_interval:
            iv = b.get("interval")
            if iv is None:
                iv = torch.stack([b["mask"], b["mask"]])
            intervals.append(torch.nn.functional.pad(iv, pad, value=IGNORE_INDEX))
        if "image_dirty" in b:
            # Same pad value and same geometry as ``image``: the consistency KL is taken
            # between two passes that must differ only where the corruption acted.
            dirty.append(torch.nn.functional.pad(b["image_dirty"], pad, value=0.0))
        if "dist" in b:
            dists.append(torch.nn.functional.pad(b["dist"], pad, value=0.0))
        pad_hw.append((h, w))

    out: dict[str, Any] = {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "pad_hw": pad_hw,
    }
    if dirty:
        if len(dirty) != len(batch):
            raise AssertionError(
                f"{len(dirty)}/{len(batch)} items carry 'image_dirty'. A batch that is "
                "half clean-only cannot be split into a (clean, dirty) pair, and the "
                "consistency term would silently score mismatched rows.")
        out["image_dirty"] = torch.stack(dirty)
    if intervals:
        out["interval"] = torch.stack(intervals)
    if dists:
        out["dist"] = torch.stack(dists)
    for k in _meta_keys(batch):
        out[k] = [b[k] for b in batch]
    return out


def native_collate(batch: list[dict]) -> dict:
    """Batch-of-one collation for the native-resolution path: no padding at all."""
    if len(batch) != 1:
        raise AssertionError(
            f"native_collate requires batch_size=1 (got {len(batch)}); native resolution "
            "means every sample has its own shape"
        )
    b = batch[0]
    h, w = int(b["image"].shape[-2]), int(b["image"].shape[-1])
    out: dict[str, Any] = {
        "image": b["image"].unsqueeze(0),
        "mask": b["mask"].unsqueeze(0),
        "pad_hw": [(h, w)],
    }
    if "interval" in b:
        out["interval"] = b["interval"].unsqueeze(0)
    if "dist" in b:
        out["dist"] = b["dist"].unsqueeze(0)
    for k in _meta_keys(batch):
        out[k] = [b[k]]
    return out


def assert_no_inference_pool(samples: Sequence[Sample], where: str) -> None:
    """Refuse the transductive pool anywhere a *labelled* set is being built."""
    bad = [s.stem for s in samples if s.device == INFERENCE_DEVICE]
    if bad:
        raise AssertionError(
            f"{where} was handed {len(bad)} sample(s) tagged {INFERENCE_DEVICE!r} "
            f"(first: {bad[:3]}). Those are the evaluation container's own TEST images: "
            "they carry no ground truth, so any set built from them is either a training "
            "set with an all-IGNORE target or a ruler that measures nothing. They belong "
            "in the unlabeled adaptation pool only (octtta.train's selftrain/coteach "
            "stream), never in a labelled or held-out set.")


def summarize_samples(samples: Sequence[Sample]) -> dict:
    """Cohort counts, for run logs and the pool audit."""
    by_device = Counter(s.device for s in samples)
    by_status = Counter(s.status.lower() for s in samples)
    return {
        "n": len(samples),
        "n_labeled": sum(s.labeled for s in samples),
        "by_device": dict(sorted(by_device.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_vendor": dict(sorted(Counter(s.vendor for s in samples).items())),
    }


def build_datasets(
    cfg: dict,
    samples: list[Sample],
    augmenter: "Augmenter | None" = None,
) -> tuple[OCTSegDataset, OCTSegDataset]:
    """Train/val datasets from the resolved ``data`` block."""
    assert_no_inference_pool(samples, "build_datasets")
    data = cfg.get("data", cfg)
    split = data.get("split", {})
    train_s, val_s = _sampling.split_samples(
        samples,
        val_fraction=float(split.get("val_fraction", 0.2)),
        seed=int(split.get("seed", 0)),
        scheme=str(split.get("scheme", "hash_by_stem")),
    )
    crop = data.get("train_size")
    terms = cfg.get("loss", {}).get("terms", []) or []
    need_dist = any(t.get("name") == "boundary" for t in terms)
    common = dict(
        num_classes=int(data.get("num_classes", NUM_CLASSES)),
        normalize=data.get("normalize"),
        seed=int(split.get("seed", 0)),
    )
    train_ds = OCTSegDataset(
        train_s, augmenter=augmenter, crop_size=tuple(crop) if crop else None,
        crop=data.get("crop"), need_distance_map=need_dist,
        # Top-level block, train side only: the val dataset must keep seeing the
        # images the leaderboard sees (same reason it gets no augmenter).
        failaug=cfg.get("failaug"),
        # Top-level block, train side only, and NOT passed to ``build_unlabeled_dataset``
        # below: the unlabelled branch takes only an ``augmenter``.
        discaug=cfg.get("discaug"),
        # Train side only, third time for the same reason: dropping the release's four
        # invented boundaries from the VAL labels would redefine the local score.
        official_true_boundaries_only=bool(
            data.get("official_true_boundaries_only", OFFICIAL_TRUE_BOUNDARIES_ONLY_DEFAULT)),
        # No default: the two readings supervise different anatomy from the same pixels,
        # so a run has to name the one it means.
        official_synthetic_boundary_map=data.get("official_synthetic_boundary_map"),
        # Read straight out of ``train.consistency`` rather than taken as an argument:
        consistency=(cfg.get("train") or {}).get("consistency"),
        **common,
    )
    val_ds = OCTSegDataset(val_s, augmenter=None, crop_size=None,
                           need_distance_map=False, **common)
    return train_ds, val_ds


def split_val_dataset(cfg: dict, samples: list[Sample]) -> OCTSegDataset:
    """The held-out half of :func:`build_datasets`, for the paths that score but never train."""
    data = dict(cfg.get("data", cfg))
    data["official_true_boundaries_only"] = False
    scoped = dict(cfg) if "data" in cfg else {}
    scoped["data"] = data
    return build_datasets(scoped, samples, None)[1]


def build_unlabeled_dataset(cfg: dict, samples: list[Sample],
                            augmenter: "Augmenter | None") -> OCTSegDataset:
    """The self-training twin of ``build_datasets``'s train side, for mask-less samples."""
    for s in samples:
        if s.mask is not None:
            raise ValueError(
                f"build_unlabeled_dataset got a LABELLED sample ({s.stem}); feeding "
                "labelled data through the pseudo-label path would silently replace its "
                "ground truth with the teacher's guess")
    data = cfg.get("data", cfg)
    split = data.get("split", {})
    crop = data.get("train_size")
    return OCTSegDataset(
        samples, augmenter=augmenter, crop_size=tuple(crop) if crop else None,
        crop=data.get("crop"),
        num_classes=int(data.get("num_classes", NUM_CLASSES)),
        normalize=data.get("normalize"),
        seed=int(split.get("seed", 0)) + 7919,
        consistency=(cfg.get("train") or {}).get("consistency"),
    )
