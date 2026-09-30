"""One owner of "a column is 9 boundaries"; numpy-only.

``b_k = #{ y : L(y) < k }`` for ``k = 1..C-1`` (cumulative, not "first row of class k"),
so the stack is ordered even where the column is not. Rows use input-image pixels."""

from __future__ import annotations

import numpy as np

__all__ = [
    "NUM_CLASSES",
    "num_boundaries",
    "boundaries_from_labels",
    "labels_from_boundaries",
    "expected_boundaries",
    "boundary_rows",
]

NUM_CLASSES = 10

IGNORE_UINT8 = 255
IGNORE_INDEX = -1


def num_boundaries(num_classes: int = NUM_CLASSES) -> int:
    return int(num_classes) - 1






def boundaries_from_labels(
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
    *,
    ignore_values: tuple[int, ...] = (IGNORE_UINT8, IGNORE_INDEX),
    require_interior: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(H, W)`` label map -> ``((C-1, W) float32 rows in pixels, (C-1, W) bool valid)``.
    Validity is per boundary, not per column; invalid entries still carry finite numbers."""
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"labels must be (H, W), got {lab.shape}")
    # int64 first: one of the sentinels (255, -1) is unrepresentable in whichever dtype came in.
    lab = lab.astype(np.int64, copy=False)
    c = int(num_classes)
    h, w = lab.shape

    counts = np.stack([(lab == k).sum(axis=0) for k in range(c)]).astype(np.float32)
    cum = np.cumsum(counts, axis=0)
    bounds = cum[: c - 1].copy()

    unlabelled = np.zeros_like(lab, dtype=bool)
    for sentinel in ignore_values:
        unlabelled |= lab == sentinel
    n_labelled = float(h) - unlabelled.sum(axis=0)
    accounted = np.isclose(cum[c - 1], n_labelled)
    suffix = ~(unlabelled[:-1] & ~unlabelled[1:]).any(axis=0) if h > 1 else np.ones(w, bool)

    valid = np.broadcast_to(accounted & suffix, bounds.shape).copy()
    if require_interior:
        valid &= (bounds > 0) & (bounds < n_labelled[None, :])

    return bounds.astype(np.float32), valid


def labels_from_boundaries(
    bounds: np.ndarray,
    height: int,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """``(C-1, W)`` rows (pixels) -> ``(H, W)`` uint8 labels: ``label(y,x) = #{k : y >= b_k(x)}``."""
    b = np.asarray(bounds, dtype=np.float32)
    if b.ndim != 2 or b.shape[0] != num_classes - 1:
        raise ValueError(f"bounds must be ({num_classes - 1}, W), got {b.shape}")
    rows = np.arange(int(height), dtype=np.float32)[:, None, None]
    out = (rows >= b[None, :, :]).sum(axis=1)
    return out.astype(np.uint8)




def expected_boundaries(prob: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """``(C,H,W)`` probabilities -> ``(C-1,W)`` boundary rows, ordered by construction."""
    p = np.asarray(prob, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != num_classes:
        raise ValueError(f"prob must be ({num_classes}, H, W), got {p.shape}")
    if (p < 0).any():
        raise ValueError("prob has negative entries; it is not a probability map")
    if not np.isfinite(p).all():
        raise ValueError("prob has non-finite entries (inf or nan)")

    total = p.sum(axis=0, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("prob has a pixel whose class mass is zero; cannot normalise")
    p = p / total

    cum = np.cumsum(p, axis=0)
    return cum[: num_classes - 1].sum(axis=1).astype(np.float32)


def boundary_rows(labels: np.ndarray, *, n_boundaries: int = NUM_CLASSES - 1,
                  ignore: int = IGNORE_UINT8,
                  columns: np.ndarray | None = None) -> np.ndarray:
    """``(9, W)`` first row at or below each interface, ``NaN`` where the column has none.
    The rule is ``lab >= i``, not ``lab == i``: on a column where a class is absent ``==``
    loses the boundary. ``ignore`` pixels are demoted below every class first."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"expected an (H, W) label map, got {arr.shape}")
    work = arr.astype(np.int16, copy=True)
    work[arr == ignore] = -1

    h, w = work.shape
    out = np.full((n_boundaries, w), np.nan, dtype=float)
    for i in range(1, n_boundaries + 1):
        hit = work >= i
        any_ = hit.any(axis=0)
        first = np.argmax(hit, axis=0).astype(float)
        first[~any_] = np.nan
        out[i - 1] = first
    if columns is not None:
        out[:, ~np.asarray(columns, dtype=bool)] = np.nan
    return out
