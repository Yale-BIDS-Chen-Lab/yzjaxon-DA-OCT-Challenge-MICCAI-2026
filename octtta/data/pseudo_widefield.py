"""Pseudo wide-field stress test: a validation construct, never a training set.

:func:`to_pseudo_widefield` stretches a macula B-scan laterally, degrades the periphery
and makes thin layers genuinely vanish, so code assuming a macula aspect ratio or all
10 classes per column fails here rather than on the hidden set. WARNING: never train on
it; the dropped layers stay visible, so fitting them teaches hallucinated absences.
"""

from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import Dataset

from octtta.data.dataset import OCTSegDataset
from octtta.data.release_dataset import NUM_CLASSES

cv2.setNumThreads(0)

#: Fraction of the half-width that stays untouched; layers only start vanishing outside it.
DROP_ONSET = 0.45

#: A layer thinner than this fraction of the image height may vanish at full peripherality.
MAX_THIN_FRAC = 0.14


def _peripherality(width: int) -> np.ndarray:
    """0 at the centre column, 1 at the outermost columns."""
    if width <= 1:
        return np.zeros(max(width, 0), dtype=np.float32)
    centre = (width - 1) / 2.0
    return (np.abs(np.arange(width, dtype=np.float32) - centre) / centre).astype(np.float32)


def _nearest_fill(column: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Replace non-kept rows by the nearest kept row's label, ties going upward."""
    h = column.shape[0]
    rows = np.arange(h)
    big = h * 4

    above = np.where(keep, rows, -big)
    np.maximum.accumulate(above, out=above)
    below = np.where(keep, rows, big)
    below = np.minimum.accumulate(below[::-1])[::-1]

    d_above = np.where(above < 0, big, rows - above)
    d_below = np.where(below >= h, big, below - rows)
    src = np.where(d_above <= d_below, np.maximum(above, 0), np.minimum(below, h - 1))
    return column[src]


def drop_peripheral_layers_(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    num_classes: int = NUM_CLASSES,
    onset: float = DROP_ONSET,
    max_thin_frac: float = MAX_THIN_FRAC,
) -> np.ndarray:
    """Make thin layers genuinely disappear toward the edges. Returns a new mask.
    Classes 0 and ``num_classes - 1`` are outside the retina and are never dropped.
    """
    from scipy.ndimage import gaussian_filter1d

    h, w = mask.shape
    pressure = np.clip((_peripherality(w) - onset) / max(1e-6, 1.0 - onset), 0.0, 1.0)

    jitter = gaussian_filter1d(rng.normal(size=w), sigma=max(2.0, w / 60.0))
    jitter = jitter / max(float(jitter.std()), 1e-6)
    thresh = pressure * max_thin_frac * h * np.clip(1.0 + 0.35 * jitter, 0.2, 2.0)

    interior = np.arange(1, max(num_classes - 1, 1))
    out = mask.copy()
    for x in np.flatnonzero(pressure > 0):
        col = mask[:, x]
        counts = np.bincount(col, minlength=256)
        doomed = interior[(counts[interior] > 0) & (counts[interior] <= thresh[x])]
        if doomed.size == 0:
            continue
        keep = ~np.isin(col, doomed)
        if not keep.any():
            continue
        out[:, x] = _nearest_fill(col, keep)
    return out


def _degrade_periphery(
    image: np.ndarray,
    vignette: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Contrast falloff + lateral blur + additive noise, all growing toward the edges."""
    t = _peripherality(image.shape[1])[None, :]
    scale = float(image.std()) or 1.0
    mean = float(image.mean())

    blend = (t ** 2) * vignette
    blurred = cv2.GaussianBlur(np.ascontiguousarray(image), (0, 0), sigmaX=1.6, sigmaY=0.7)
    img = image * (1.0 - blend) + blurred * blend
    img = mean + (img - mean) * (1.0 - vignette * t ** 1.5)
    noise = rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    return (img + noise * (vignette * scale) * (t ** 2)).astype(np.float32)


def to_pseudo_widefield(
    image: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    width_scale: float = 2.5,
    vignette: float = 0.5,
    drop_peripheral_layers: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Turn a macula B-scan into a wide-field-shaped stress case.
    Only the lateral axis is stretched: output is ``(H, round(W * width_scale))``.
    """
    if image.ndim != 2:
        raise ValueError(f"expected (H, W) image, got {image.shape}")
    if width_scale <= 0:
        raise ValueError(f"width_scale must be positive, got {width_scale}")
    if mask is not None and mask.shape != image.shape:
        raise ValueError(f"image {image.shape} and mask {mask.shape} disagree")

    h, w = image.shape
    new_w = max(1, int(round(w * width_scale)))
    if new_w != w:
        image = cv2.resize(np.ascontiguousarray(image, dtype=np.float32),
                           (new_w, h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(np.ascontiguousarray(mask, dtype=np.uint8),
                              (new_w, h), interpolation=cv2.INTER_NEAREST)

    if mask is not None and drop_peripheral_layers:
        mask = drop_peripheral_layers_(mask, rng)

    image = _degrade_periphery(image.astype(np.float32), float(vignette), rng)
    return image, (mask.astype(np.uint8) if mask is not None else None)


class PseudoWideFieldDataset(Dataset):
    """Wraps an :class:`OCTSegDataset` and reports ``anatomy='WideField'``.
    Seeded from ``(seed, index)`` only, so the set is byte-identical every epoch.
    """

    def __init__(
        self,
        base: OCTSegDataset,
        *,
        width_scale: float = 2.5,
        vignette: float = 0.5,
        drop_peripheral_layers: bool = True,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.width_scale = float(width_scale)
        self.vignette = float(vignette)
        self.drop_peripheral_layers = bool(drop_peripheral_layers)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        """Accepted and ignored: this set must not vary across epochs."""

    def __getitem__(self, idx: int) -> dict:
        image, mask, meta = self.base.load_numpy(idx)
        rng = np.random.default_rng([self.seed, idx])
        image, mask = to_pseudo_widefield(
            image, mask, rng,
            width_scale=self.width_scale,
            vignette=self.vignette,
            drop_peripheral_layers=self.drop_peripheral_layers,
        )
        meta = {
            **meta,
            "anatomy": "WideField",
            "stem": f"{meta['stem']}-pwf",
            "orig_hw": (int(image.shape[0]), int(image.shape[1])),
        }
        return self.base.finalize(image, mask, meta)
