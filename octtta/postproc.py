"""Inference postprocessing: presence gate, monotonic columns, and class guard."""

from __future__ import annotations


from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from octtta.surface import NUM_CLASSES


def postproc_cfg_or_none(cfg: dict) -> dict | None:
    """Return the enabled post-processing block, or disable it end to end."""
    block = dict(cfg.get("postproc") or {})
    if not block.pop("enabled", True):
        return None
    return block

DEFAULT_MIN_PIXEL_FRAC = 5e-4
DEFAULT_MIN_PIXELS_FLOOR = 32
DEFAULT_MIN_MEAN_CONF = 0.5
DEFAULT_PROTECTED_CLASSES: tuple[int, ...] = (0, 9)
_EPS = 1e-12


def resolve_size_threshold(area: int, *, frac: float | None = None,
                           floor: int | None = None,
                           default_frac: float = DEFAULT_MIN_PIXEL_FRAC,
                           default_floor: int = DEFAULT_MIN_PIXELS_FLOOR) -> int:
    """Resolve a fraction of image area with an absolute lower bound."""
    if area < 0:
        raise ValueError(f"area must be non-negative, got {area}")
    if frac is None:
        frac = default_frac
    frac = float(frac)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"size threshold fraction must be in [0, 1], got {frac}")
    return max(int(default_floor if floor is None else floor), int(round(frac * area)))


def _as_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob)
    if prob.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {prob.shape}")
    if prob.size and not float(prob.min()) >= 0.0:
        raise ValueError("expected softmax probabilities, got negative or NaN values")
    return prob


def _gate_decisions(prob: np.ndarray, argmax: np.ndarray, min_pixels: int,
                    min_mean_conf: float, protected: Sequence[int]) -> list[int]:
    protected_set = {int(c) for c in protected}
    counts = np.bincount(argmax.ravel(), minlength=prob.shape[0])
    observed = {c for c in range(prob.shape[0]) if counts[c] > 0}
    dropped = []
    for c in range(prob.shape[0]):
        if c in protected_set:
            continue
        n = int(counts[c])
        if 0 < n < min_pixels and float(prob[c][argmax == c].mean()) < min_mean_conf:
            dropped.append(c)
    return [] if dropped and observed.issubset(dropped) else dropped


def _suppress(prob: np.ndarray, classes: Sequence[int],
              allowed: np.ndarray | None = None) -> np.ndarray:
    out = prob.astype(np.float32, copy=True)
    if allowed is None:
        out[list(classes)] = 0.0
    else:
        keep = np.zeros(prob.shape[0], dtype=bool)
        keep[np.asarray(allowed, dtype=np.int64)] = True
        keep[list(classes)] = False
        if not keep.any():
            return prob
        out[~keep] = 0.0
    np.divide(out, np.maximum(out.sum(axis=0, keepdims=True), _EPS), out=out)
    return out


def presence_gate(prob: np.ndarray, min_mean_conf: float = DEFAULT_MIN_MEAN_CONF,
                  protected_classes: Sequence[int] = DEFAULT_PROTECTED_CLASSES, *,
                  min_pixel_frac: float | None = None,
                  min_pixels_floor: int | None = None,
                  never_invent: bool = True) -> np.ndarray:
    """Suppress classes with both tiny footprint and low confidence."""
    prob = _as_prob(prob)
    argmax = np.argmax(prob, axis=0)
    thr = resolve_size_threshold(int(argmax.size), frac=min_pixel_frac,
                                 floor=min_pixels_floor)
    dropped = _gate_decisions(prob, argmax, thr, min_mean_conf, protected_classes)
    if not dropped:
        return prob
    return _suppress(prob, dropped, np.unique(argmax) if never_invent else None)


def enforce_monotonic_columns(prob: np.ndarray) -> np.ndarray:
    """Viterbi decode each column so labels never decrease downward."""
    if prob.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {prob.shape}")
    C, H, W = prob.shape
    logp = np.log(np.clip(prob, 1e-12, None))
    out = np.empty((H, W), dtype=np.uint8)
    for x in range(W):
        col = logp[:, :, x]
        dp = np.full((H, C), -np.inf)
        back = np.zeros((H, C), dtype=np.int16)
        dp[0, :] = col[:, 0]
        for y in range(1, H):
            prev = dp[y - 1]
            best_idx = np.maximum.accumulate(
                np.where(prev == np.maximum.accumulate(prev), np.arange(C), -1))
            dp[y] = np.maximum.accumulate(prev) + col[:, y]
            back[y] = best_idx
        c = int(np.argmax(dp[H - 1]))
        for y in range(H - 1, -1, -1):
            out[y, x] = c
            c = int(back[y, c])
            if c < 0:
                c = 0
    return out


def _feasible_columns(reduced: np.ndarray) -> np.ndarray:
    if reduced.shape[0] < 2:
        ok = np.ones(reduced.shape[1], dtype=bool)
    else:
        d = np.diff(reduced.astype(np.int16), axis=0)
        ok = (d >= 0).all(axis=0)
    return ok


def _monotonic_step(prob: np.ndarray, argmax: np.ndarray,
                    states: np.ndarray, info: dict) -> np.ndarray:
    inverse = np.full(prob.shape[0], -1, dtype=np.int16)
    inverse[states] = np.arange(states.size, dtype=np.int16)
    reduced = inverse[argmax]
    if (reduced < 0).any():
        raise RuntimeError("argmax contains a class outside the decoder state space")
    bad = np.flatnonzero(~_feasible_columns(reduced))
    info["viterbi_columns"] = int(bad.size)
    info["n_columns"] = int(prob.shape[2])
    if bad.size:
        sub = prob[:, :, bad][states]
        reduced = reduced.copy()
        reduced[:, bad] = enforce_monotonic_columns(sub)
    return states[reduced].astype(np.uint8)


def column_flags(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unordered and fragmented flags for each column."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a (H, W) label map, got {labels.shape}")
    H, W = labels.shape
    if H == 0 or W == 0:
        return np.zeros(W, dtype=bool), np.zeros(W, dtype=bool)
    lab = labels.astype(np.int16, copy=False)
    diff = np.diff(lab, axis=0)
    unordered = (diff < 0).any(axis=0)
    n_runs = 1 + (diff != 0).sum(axis=0)
    n_distinct = np.zeros(W, dtype=np.int32)
    for c in range(int(lab.min()), int(lab.max()) + 1):
        n_distinct += (lab == c).any(axis=0)
    return unordered, n_runs != n_distinct


def column_topology_violations(labels: np.ndarray) -> dict:
    """Count columns that violate the layer stacking assumption."""
    labels = np.asarray(labels)
    _, W = labels.shape
    n_unordered = n_fragmented = 0
    for x in range(W):
        col = labels[:, x]
        runs = np.split(col, np.flatnonzero(np.diff(col)) + 1)
        vals = [int(r[0]) for r in runs]
        n_unordered += vals != sorted(vals)
        n_fragmented += len(set(vals)) != len(vals)
    return {"n_columns": W, "unordered_columns": n_unordered,
            "fragmented_columns": n_fragmented,
            "unordered_frac": n_unordered / max(W, 1),
            "fragmented_frac": n_fragmented / max(W, 1)}


def _enforce_allowed(labels: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, list[int]]:
    bad_mask = ~np.isin(labels, allowed)
    if not bad_mask.any():
        return labels, []
    invented = sorted(np.unique(labels[bad_mask]).tolist())
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(bad_mask, return_indices=True)
    out = labels.copy()
    out[bad_mask] = labels[tuple(i[bad_mask] for i in idx)]
    return out, invented


def _as_block(value: Any, key: str) -> tuple[bool, dict]:
    if value is None:
        return False, {}
    if isinstance(value, bool):
        return value, {}
    if isinstance(value, Mapping):
        block = dict(value)
        enabled = block.get("enabled", True)
        if enabled is None:
            return False, block
        if not isinstance(enabled, bool):
            raise TypeError(f"postproc.{key}.enabled must be a bool or null")
        return enabled, block
    raise TypeError(f"postproc.{key} must be a bool, null, or a mapping")


def _gate_params(cfg: dict) -> tuple[bool, dict]:
    enabled, block = _as_block(cfg.get("presence_gate", True), "presence_gate")
    return enabled, {
        "min_pixel_frac": block.get("min_pixel_frac"),
        "min_pixels_floor": block.get("min_pixels_floor"),
        "min_mean_conf": float(block.get("min_mean_conf", DEFAULT_MIN_MEAN_CONF)),
        "protected_classes": tuple(int(c) for c in
                                   block.get("protected_classes", DEFAULT_PROTECTED_CLASSES)),
    }


def _changed(before: np.ndarray, after: np.ndarray) -> tuple[int, list[int]]:
    diff = before != after
    n = int(diff.sum())
    if not n:
        return 0, []
    gone = sorted(set(np.unique(before).tolist()) - set(np.unique(after).tolist()))
    return n, [int(c) for c in gone]


def postprocess(prob: np.ndarray, cfg: dict, *, info: dict | None = None) -> np.ndarray:
    """Apply the shipped gate, Viterbi repair, and never-invent guard."""
    cfg = dict(cfg or {})
    info = info if info is not None else {}
    prob = _as_prob(prob)
    argmax = np.argmax(prob, axis=0)
    allowed_in = np.unique(argmax)
    area = int(argmax.size)
    info.update(area=area, n_columns=int(prob.shape[2]), gated_classes=[],
                gate_pixels_changed=0, gate_classes_removed=[], viterbi_columns=0,
                mono_pixels_changed=0, mono_classes_removed=[], invented_repaired=[])
    if not _as_block(cfg.get("enabled", True), "enabled")[0]:
        out = np.clip(argmax, 0, NUM_CLASSES - 1).astype(np.uint8)
        info["components_enabled"] = {"presence_gate": False,
                                      "enforce_monotonic_columns": False}
        info["classes_out"] = np.unique(out).tolist()
        return out
    never_invent = bool(cfg.get("never_invent_classes", True))
    if cfg.get("allow_class_skips", True) is not True:
        raise ValueError("postproc.allow_class_skips must be true")
    gate_on, gp = _gate_params(cfg)
    mono_on = _as_block(cfg.get("enforce_monotonic_columns", True),
                        "enforce_monotonic_columns")[0]
    info["components_enabled"] = {"presence_gate": gate_on,
                                  "enforce_monotonic_columns": mono_on}
    min_pixels = resolve_size_threshold(area, frac=gp["min_pixel_frac"],
                                        floor=gp["min_pixels_floor"])
    info["min_pixels_used"] = min_pixels
    if gate_on:
        before = argmax
        dropped = _gate_decisions(prob, argmax, min_pixels, gp["min_mean_conf"],
                                  gp["protected_classes"])
        if dropped:
            prob = _suppress(prob, dropped, allowed_in if never_invent else None)
            argmax = np.argmax(prob, axis=0)
        info["gated_classes"] = dropped
        info["gate_pixels_changed"], info["gate_classes_removed"] = _changed(before, argmax)
    if mono_on:
        before = argmax
        states = np.unique(argmax) if never_invent else np.arange(prob.shape[0])
        labels = _monotonic_step(prob, argmax, states.astype(np.int64), info)
        info["mono_pixels_changed"], info["mono_classes_removed"] = _changed(before, labels)
    else:
        labels = argmax.astype(np.uint8)
    if never_invent:
        labels, info["invented_repaired"] = _enforce_allowed(labels, allowed_in)
    out = np.clip(labels, 0, NUM_CLASSES - 1).astype(np.uint8)
    info["classes_out"] = np.unique(out).tolist()
    return out
