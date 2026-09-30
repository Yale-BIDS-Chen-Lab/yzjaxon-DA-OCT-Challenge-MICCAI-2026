"""Failure-mode augmentation: local geometric warps and band corruptions.

:class:`FailAug` runs on the full-resolution image before cropping, not as an
``augment.pipeline`` op, and draws from its own RNG stream (:data:`RNG_TAG`) so a
matched-seed control arm stays matched. Vertical warps are strictly monotone per column
and labels follow the same backward map with nearest sampling, so no operator can cross
a ground-truth boundary or invent a label value.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

#: Mixed into every seed list here, so this stream cannot collide with the dataset's own.
RNG_TAG = 0xFA11A


# REPLICATE, not REFLECT: a large downward shift under a reflecting border would
# mirror the retina above itself, i.e. invert the layer order.
_BORDER = cv2.BORDER_REPLICATE


DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "p": 0.30,
    "extreme_frac": 1.0 / 6.0,
    # Geometry and band corruption are configured as separate branches.
    "mode": "gwarp",
    "gwarp": {
        "n_windows": [1, 2],
        "window_frac": [0.10, 0.30],       # lateral FWHM as fraction of W
        "p_bulge": 0.6,
        "shift_frac": [0.04, 0.15],        # bulge amplitude, fraction of H
        "shift_frac_extreme": [0.15, 0.30],
        "squeeze_frac": [0.03, 0.10],      # squeeze amplitude, fraction of H
        "squeeze_frac_extreme": [0.10, 0.20],
        "depth_sigma_frac": [0.05, 0.12],  # vertical extent of the squeeze
        "centre_jitter_frac": 0.15,
    },
    "aband": {
        "n_ops": [1, 2],
        "window_frac": [0.10, 0.30],
        "ops": ["blur", "noise", "contrast", "attenuate", "patch"],
        "blur_sigma": [2.0, 6.0],
        "blur_sigma_extreme": [6.0, 12.0],
        "noise_sigma": [0.5, 1.5],         # units of the image's own std
        "noise_sigma_extreme": [1.5, 3.0],
        "contrast_loss": [0.5, 0.9],       # 1.0 = fully collapsed to local mean
        "contrast_loss_extreme": [0.9, 1.0],
        "attenuation": [0.3, 0.7],         # multiplier below the onset row
        "attenuation_extreme": [0.10, 0.30],
        "patch_amp": [0.8, 2.0],           # units of the image's own std
        "patch_amp_extreme": [2.0, 3.5],
        "patch_sigma_frac": [0.04, 0.12],  # vertical extent of the blob
    },
}

_MODES = ("gwarp", "aband")

#: Keys accepted in ``train.consistency``; ``weight_*`` belong to :mod:`octtta.train`.
CONSISTENCY_KEYS = ("enabled", "weight_dirty", "weight_kl", "p", "extreme_frac", "aband")


def _merge(defaults: dict, override: dict, path: str) -> dict:
    out = dict(defaults)
    for k, v in override.items():
        if k not in defaults:
            raise ValueError(f"failaug: unknown key {path}{k!r}")
        if isinstance(defaults[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"failaug: {path}{k} must be a mapping")
            out[k] = _merge(defaults[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def _pair(v, name: str) -> tuple[float, float]:
    lo, hi = float(v[0]), float(v[1])
    if not lo <= hi:
        raise ValueError(f"failaug: {name} range {v!r} is not ordered")
    return lo, hi


def band_centre_row(image: np.ndarray, rng: np.random.Generator,
                    jitter_frac: float) -> float:
    """Vertical centre of the bright band, from the image alone (never the GT)."""
    h = image.shape[0]
    thr = float(image.mean()) + 0.5 * float(image.std())
    prof = (image > thr).sum(axis=1).astype(np.float64)
    total = prof.sum()
    centre = float((prof * np.arange(h)).sum() / total) if total > 0 else h / 2.0
    return centre + float(rng.uniform(-jitter_frac, jitter_frac)) * h


def _lateral_window(w: int, rng: np.random.Generator,
                    window_frac: tuple[float, float]) -> np.ndarray:
    cx = float(rng.uniform(0.05, 0.95)) * w
    fwhm = float(rng.uniform(*window_frac)) * w
    sigma = max(fwhm / 2.355, 1.0)
    x = np.arange(w, dtype=np.float32)
    return np.exp(-0.5 * ((x - cx) / sigma) ** 2).astype(np.float32)


def _as_label(mask: np.ndarray | None) -> tuple[np.ndarray | None, Any]:
    if mask is None:
        return None, None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"failaug: expected an (H, W) mask, got {arr.shape}")
    if arr.dtype in (np.uint8, np.int16):
        return np.array(arr, copy=True), arr.dtype
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"failaug: mask must be integer labels, got {arr.dtype}")
    return arr.astype(np.int16), arr.dtype


class FailAug:
    """One resolved ``failaug:`` block, callable on a full-resolution sample. Stateless: the
    caller owns the RNG derivation, so one instance serves every worker.
    """

    def __init__(self, cfg: dict | None) -> None:
        cfg = _merge(DEFAULTS, dict(cfg or {}), "")
        if cfg["mode"] not in _MODES:
            raise ValueError(f"failaug: mode must be one of {_MODES}, got {cfg['mode']!r}")
        if not 0.0 <= float(cfg["p"]) <= 1.0:
            raise ValueError(f"failaug: p must be in [0, 1], got {cfg['p']}")
        if not 0.0 <= float(cfg["extreme_frac"]) <= 1.0:
            raise ValueError(
                f"failaug: extreme_frac must be in [0, 1], got {cfg['extreme_frac']}")
        self.cfg = cfg
        self.p = float(cfg["p"])
        self.extreme_frac = float(cfg["extreme_frac"])
        self.mode = str(cfg["mode"])

    @staticmethod
    def from_config(cfg: dict | None) -> "FailAug | None":
        """None when the block is absent or disabled -- the byte-identical path."""
        if not cfg or not cfg.get("enabled", True):
            return None
        return FailAug(cfg)

    def __repr__(self) -> str:
        return f"FailAug(mode={self.mode}, p={self.p:g})"

    @staticmethod
    def consistency_from_config(cfg: dict | None) -> "FailAug | None":
        """The A-Band corrupter behind ``train.consistency``: a second, corrupted VIEW of the
        clean sample. ``mode`` is pinned to ``aband`` because both views share one mask.
        """
        if not cfg or not cfg.get("enabled", False):
            return None
        unknown = sorted(set(cfg) - set(CONSISTENCY_KEYS))
        if unknown:
            raise ValueError(
                f"train.consistency: unknown key(s) {unknown}; known: {list(CONSISTENCY_KEYS)}"
            )
        for banned in ("mode", "gwarp"):
            if banned in cfg:
                raise ValueError(
                    f"train.consistency.{banned} is not configurable: the dirty view must "
                    f"be geometry-preserving so the clean view's mask supervises it")
        block: dict[str, Any] = {
            "enabled": True,
            "mode": "aband",
            "p": float(cfg.get("p", 1.0)),
            "extreme_frac": float(cfg.get("extreme_frac", DEFAULTS["extreme_frac"])),
        }
        if cfg.get("aband"):
            block["aband"] = dict(cfg["aband"])
        return FailAug(block)


    def apply_gwarp(self, image: np.ndarray, mask: np.ndarray | None,
                    rng: np.random.Generator, *, extreme: bool,
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        if image.ndim != 2:
            raise ValueError(f"failaug: expected an (H, W) image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"failaug: image {image.shape} vs mask {mask.shape}")
        return self._gwarp(image, mask, rng, bool(extreme))

    def apply_aband(self, image: np.ndarray, rng: np.random.Generator, *,
                    extreme: bool) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"failaug: expected an (H, W) image, got {image.shape}")
        return self._aband(image, rng, bool(extreme))

    def __call__(self, image: np.ndarray, mask: np.ndarray | None,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        if image.ndim != 2:
            raise ValueError(f"failaug: expected an (H, W) image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"failaug: image {image.shape} vs mask {mask.shape}")
        if rng.random() >= self.p:
            return image, mask
        extreme = bool(rng.random() < self.extreme_frac)
        if self.mode == "gwarp":
            return self._gwarp(image, mask, rng, extreme)
        img = self._aband(image, rng, extreme)
        return img, mask


    def _gwarp(self, image: np.ndarray, mask: np.ndarray | None,
               rng: np.random.Generator, extreme: bool,
               ) -> tuple[np.ndarray, np.ndarray | None]:
        p = self.cfg["gwarp"]
        h, w = image.shape
        n_lo, n_hi = int(p["n_windows"][0]), int(p["n_windows"][1])
        n_win = int(rng.integers(n_lo, n_hi + 1))
        window_frac = _pair(p["window_frac"], "gwarp.window_frac")

        disp = np.zeros((h, w), dtype=np.float32)
        yy = np.arange(h, dtype=np.float32)[:, None]
        for _ in range(n_win):
            wx = _lateral_window(w, rng, window_frac)[None, :]
            sign = -1.0 if rng.random() < 0.5 else 1.0
            if rng.random() < float(p["p_bulge"]):
                rng_key = "shift_frac_extreme" if extreme else "shift_frac"
                amp = sign * float(rng.uniform(*_pair(p[rng_key], rng_key))) * h
                disp += amp * wx  # constant in y: monotone for any amplitude
            else:
                rng_key = "squeeze_frac_extreme" if extreme else "squeeze_frac"
                amp = sign * float(rng.uniform(*_pair(p[rng_key], rng_key))) * h
                cy = band_centre_row(image, rng, float(p["centre_jitter_frac"]))
                sy = float(rng.uniform(*_pair(p["depth_sigma_frac"],
                                              "depth_sigma_frac"))) * h
                gy = np.exp(-0.5 * ((yy - cy) / max(sy, 1.0)) ** 2)
                disp += amp * wx * gy

        # Monotonicity is the safety property everything else rests on: rescale the
        # whole field rather than clipping it (clipping would kink the map).
        slope = np.diff(disp, axis=0)
        worst = float(slope.min(initial=0.0))
        if worst <= -0.95:
            disp *= 0.95 / -worst
        map_y = yy + disp
        if not (np.diff(map_y, axis=0) > 0.0).all():
            raise AssertionError("failaug: gwarp map lost monotonicity")
        map_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
        map_x = np.ascontiguousarray(map_x)

        img = cv2.remap(np.ascontiguousarray(image, dtype=np.float32),
                        map_x, map_y, cv2.INTER_LINEAR, borderMode=_BORDER)
        if mask is None:
            return img, None
        lab, lab_dtype = _as_label(mask)
        out = cv2.remap(lab, map_x, map_y, cv2.INTER_NEAREST, borderMode=_BORDER)
        invented = np.setdiff1d(np.unique(out), np.unique(lab))
        if invented.size:
            raise AssertionError(  # nearest sampling makes this unreachable
                f"failaug: gwarp invented label values {invented.tolist()}")
        return img, np.ascontiguousarray(out.astype(lab_dtype, copy=False))


    def _aband(self, image: np.ndarray, rng: np.random.Generator,
               extreme: bool) -> np.ndarray:
        p = self.cfg["aband"]
        h, w = image.shape
        out = np.array(image, dtype=np.float32, copy=True)
        std = float(image.std()) + 1e-6
        wx = _lateral_window(w, rng, _pair(p["window_frac"], "aband.window_frac"))[None, :]
        n_lo, n_hi = int(p["n_ops"][0]), int(p["n_ops"][1])
        n_ops = min(int(rng.integers(n_lo, n_hi + 1)), len(p["ops"]))
        chosen = list(rng.choice(np.asarray(p["ops"], dtype=object), size=n_ops,
                                 replace=False))

        def rr(key: str) -> float:
            k = f"{key}_extreme" if extreme else key
            return float(rng.uniform(*_pair(p[k], k)))

        for op in chosen:
            if op == "blur":
                blurred = cv2.GaussianBlur(out, (0, 0), rr("blur_sigma"))
                out = out * (1.0 - wx) + blurred * wx
            elif op == "noise":
                amp = rr("noise_sigma") * std
                out = out + wx * amp * rng.standard_normal((h, w)).astype(np.float32)
            elif op == "contrast":
                local_mean = cv2.GaussianBlur(out, (0, 0), max(h / 8.0, 3.0))
                c = wx * rr("contrast_loss")
                out = out * (1.0 - c) + local_mean * c
            elif op == "attenuate":
                cy = band_centre_row(image, rng, 0.0) + float(
                    rng.uniform(-0.10, 0.25)) * h
                ramp = 1.0 / (1.0 + np.exp(-(np.arange(h, dtype=np.float32)
                                             - cy) / max(0.02 * h, 1.0)))
                floor = float(np.quantile(image, 0.05))
                alpha = rr("attenuation")
                w2 = wx * ramp[:, None]
                out = out * (1.0 - w2) + (floor + alpha * (out - floor)) * w2
            elif op == "patch":
                cy = band_centre_row(image, rng,
                                     float(self.cfg["gwarp"]["centre_jitter_frac"]))
                sy = float(rng.uniform(*_pair(p["patch_sigma_frac"],
                                              "patch_sigma_frac"))) * h
                blob = wx * np.exp(-0.5 * ((np.arange(h, dtype=np.float32)[:, None]
                                            - cy) / max(sy, 1.0)) ** 2)
                sign = -1.0 if rng.random() < 0.5 else 1.0
                out = out + sign * rr("patch_amp") * std * blob
            else:  # pragma: no cover - _merge refuses unknown keys upstream
                raise ValueError(f"failaug: unknown aband op {op!r}")
        return np.ascontiguousarray(out, dtype=np.float32)
