"""OCT-specific augmentation.

Top-to-bottom layer order is the invariant everything rests on: vertical flip is refused
outright and every geometric op extends with ``BORDER_REPLICATE``, never a constant.
Consecutive affine ops are composed into one matrix so a boundary is resampled once.
Intensity ops receive an already normalised image and map through its own [0, 1] range.
Masks are nearest-resampled, so no augmentation can invent a class or lose the sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

#: Ops that are a single affine warp. Consecutive ones are composed and resampled once.
AFFINE_OPS = frozenset({
    "random_scale",
    "random_anisotropic_scale",
    "random_rotate",
    "random_translate",
})

#: Ops that move the mask with the image. Everything else leaves the mask untouched.
GEOMETRIC_OPS = AFFINE_OPS | {"elastic"}

PHOTOMETRIC_OPS = frozenset({
    "gamma",
    "brightness_contrast",
    "gaussian_noise",
    "speckle_noise",
})

#: The vendor axis that is neither intensity nor geometry: how much detail the device resolves.
RESOLUTION_OPS = frozenset({"gaussian_blur", "sharpen", "low_res_simulation"})

KNOWN_OPS = GEOMETRIC_OPS | PHOTOMETRIC_OPS | RESOLUTION_OPS

#: Never a constant border: it would paste vitreous below the choroid.
_BORDER = cv2.BORDER_REPLICATE
_EPS = 1e-6

#: cv2 warps accept these label dtypes directly; anything else round-trips via int16.
_CV_LABEL_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.int16))


def _pair(value: Any, *, name: str = "range") -> tuple[float, float]:
    """A scalar means a constant; a two-element sequence means a closed interval."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float(value)
    seq = tuple(float(v) for v in value)
    if len(seq) != 2:
        raise ValueError(f"{name} must be a scalar or two values, got {value!r}")
    lo, hi = seq
    return (lo, hi) if lo <= hi else (hi, lo)


def _symmetric(value: Any, *, name: str = "amount") -> tuple[float, float]:
    """A scalar ``v`` means ``[-v, v]``; a pair is taken literally."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = abs(float(value))
        return -v, v
    return _pair(value, name=name)


def _axes(value: Any, *, name: str = "value") -> tuple[float, float]:
    """``(axial, lateral)`` -- like :func:`_pair` but order is meaning, not an interval."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float(value)
    seq = tuple(float(v) for v in value)
    if len(seq) != 2:
        raise ValueError(f"{name} must be a scalar or [axial, lateral], got {value!r}")
    return seq


def _int_pair(value: Any) -> tuple[int, int]:
    lo, hi = _pair(value)
    return int(round(lo)), int(round(hi))


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Sample a multiplicative factor so that ``k`` and ``1/k`` are equally likely."""
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(f"multiplicative range must be positive, got [{lo}, {hi}]")
    if lo == hi:
        return float(lo)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _intensity_scale(img: np.ndarray) -> float:
    """Amplitude unit for additive ops: the image's own std, so config sigmas read as
    fractions of a standard deviation under any upstream normalisation.
    """
    s = float(img.std())
    return s if s > _EPS else 1.0


def _to_unit(img: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Map to an intensity-like [0, 1] view; ``None`` if the image is constant."""
    base = float(img.min())
    span = float(img.max()) - base
    if span < _EPS:
        return None
    return (img - base) / span, base, span


def _op_gamma(img: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    unit = _to_unit(img)
    if unit is None:
        return img
    u, base, span = unit
    g = _log_uniform(rng, *_pair(params.get("range", (0.7, 1.5)), name="gamma.range"))
    return (u ** np.float32(g)) * span + base


def _op_brightness_contrast(img: np.ndarray, params: dict,
                            rng: np.random.Generator) -> np.ndarray:
    """Gain and offset, i.e. the two knobs that most obviously differ between vendors."""
    out = img
    contrast = float(params.get("contrast", 0.0))
    if contrast > 0.0:
        # Log-uniform in [1/(1+c), 1+c] so widening and narrowing are equally likely.
        factor = _log_uniform(rng, 1.0 / (1.0 + contrast), 1.0 + contrast)
        mean = float(out.mean())
        out = (out - mean) * np.float32(factor) + mean
    brightness = params.get("brightness", 0.0)
    lo, hi = _symmetric(brightness, name="brightness")
    if hi > lo:
        out = out + np.float32(rng.uniform(lo, hi) * _intensity_scale(img))
    return out


def _op_gaussian_noise(img: np.ndarray, params: dict,
                       rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(*_pair(params.get("sigma", (0.0, 0.1)), name="sigma"))
    if sigma <= 0.0:
        return img
    noise = rng.standard_normal(img.shape, dtype=np.float32)
    return img + np.float32(sigma * _intensity_scale(img)) * noise


def _op_speckle_noise(img: np.ndarray, params: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """Multiplicative coherent speckle: it scales WITH the signal, where additive Gaussian
    noise would do the opposite and teach the network that dark regions are the noisy ones.
    """
    sigma = rng.uniform(*_pair(params.get("sigma", (0.0, 0.15)), name="sigma"))
    if sigma <= 0.0:
        return img
    unit = _to_unit(img)
    if unit is None:
        return img
    u, base, span = unit

    noise = rng.standard_normal(img.shape, dtype=np.float32)
    gy, gx = _axes(params.get("grain", 0.8), name="grain")
    if max(gy, gx) > 0.0:
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(gx, 1e-2), sigmaY=max(gy, 1e-2))
        s = float(noise.std())
        if s > _EPS:
            noise /= s
    return (u * (1.0 + np.float32(sigma) * noise)) * span + base


def _op_gaussian_blur(img: np.ndarray, params: dict,
                      rng: np.random.Generator) -> np.ndarray:
    lo, hi = _pair(params.get("sigma", (0.0, 1.0)), name="sigma")
    if params.get("anisotropic", False):
        sy, sx = rng.uniform(lo, hi), rng.uniform(lo, hi)
    else:
        sy = sx = rng.uniform(lo, hi)
    if max(sy, sx) < 1e-3:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sx, 1e-2), sigmaY=max(sy, 1e-2))


def _op_sharpen(img: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Unsharp mask: some vendors ship edge-enhanced B-scans straight off the device."""
    amount = rng.uniform(*_pair(params.get("amount", (0.3, 1.0)), name="amount"))
    radius = rng.uniform(*_pair(params.get("radius", (0.5, 1.5)), name="radius"))
    if amount <= 0.0 or radius <= 0.0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    return img + np.float32(amount) * (img - blurred)


def _op_low_res_simulation(img: np.ndarray, params: dict,
                           rng: np.random.Generator) -> np.ndarray:
    """Downsample then upsample -- a lower-resolution device, not a blur. Nearest down and
    cubic up keep the staircase and ringing a real coarse-sampled scan has.
    """
    H, W = img.shape
    lo, hi = _pair(params.get("scale", (0.5, 1.0)), name="scale")
    axis = str(params.get("axis", "random"))
    if axis == "random":
        axis = ("axial", "lateral", "both")[int(rng.integers(3))]
    independent = bool(params.get("independent", False))

    fy = _log_uniform(rng, lo, hi) if axis in ("axial", "both") else 1.0
    if axis == "lateral":
        fx = _log_uniform(rng, lo, hi)
    elif axis == "both":
        fx = _log_uniform(rng, lo, hi) if independent else fy
    else:
        fx = 1.0

    new_h, new_w = max(1, int(round(H * fy))), max(1, int(round(W * fx)))
    if (new_h, new_w) == (H, W):
        return img
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    up = cv2.INTER_CUBIC if min(new_h, new_w) >= 4 else cv2.INTER_LINEAR
    return cv2.resize(small, (W, H), interpolation=up)


_IMAGE_OPS: dict[str, Callable[[np.ndarray, dict, np.random.Generator], np.ndarray]] = {
    "gamma": _op_gamma,
    "brightness_contrast": _op_brightness_contrast,
    "gaussian_noise": _op_gaussian_noise,
    "speckle_noise": _op_speckle_noise,
    "gaussian_blur": _op_gaussian_blur,
    "sharpen": _op_sharpen,
    "low_res_simulation": _op_low_res_simulation,
}


def _scale_matrix(sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 0], m[0, 2] = sx, cx * (1.0 - sx)
    m[1, 1], m[1, 2] = sy, cy * (1.0 - sy)
    return m


def _affine_matrix(op: "_Op", shape: tuple[int, int],
                   rng: np.random.Generator) -> np.ndarray:
    """Forward 3x3 map for one affine op, about the image centre."""
    H, W = shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    p = op.params

    if op.name == "random_scale":
        s = _log_uniform(rng, *_pair(p.get("range", (0.9, 1.1)), name="range"))
        axis = str(p.get("axis", "both"))
        sy = s if axis in ("both", "axial") else 1.0
        sx = s if axis in ("both", "lateral") else 1.0
        return _scale_matrix(sx, sy, cx, cy)

    if op.name == "random_anisotropic_scale":
        # A-scan density and axial pitch differ per device, so the retina's aspect
        # ratio in pixels is a device property, not an anatomical one.
        sy = _log_uniform(rng, *_pair(p.get("axial", (0.85, 1.2)), name="axial"))
        sx = _log_uniform(rng, *_pair(p.get("lateral", (0.85, 1.2)), name="lateral"))
        return _scale_matrix(sx, sy, cx, cy)

    if op.name == "random_rotate":
        # Rotation is in pixel space, so it is really a shear in physical space --
        # which is what imperfect eye alignment produces anyway.
        angle = rng.uniform(*_symmetric(p.get("degrees", 5.0), name="degrees"))
        m = np.eye(3, dtype=np.float64)
        m[:2] = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        return m

    if op.name == "random_translate":
        default = p.get("frac", 0.05)
        m = np.eye(3, dtype=np.float64)
        m[0, 2] = rng.uniform(*_symmetric(p.get("lateral_frac", default))) * W
        m[1, 2] = rng.uniform(*_symmetric(p.get("axial_frac", default))) * H
        return m

    raise ValueError(f"{op.name!r} is not an affine op")


def _flush_affine(img: np.ndarray, lab: np.ndarray | None,
                  matrix: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if matrix is None or np.allclose(matrix, np.eye(3), atol=1e-6):
        return img, lab
    H, W = img.shape
    a = matrix[:2]
    img = cv2.warpAffine(img, a, (W, H), flags=cv2.INTER_LINEAR, borderMode=_BORDER)
    if lab is not None:
        lab = cv2.warpAffine(lab, a, (W, H), flags=cv2.INTER_NEAREST, borderMode=_BORDER)
    return img, lab


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int],
                  sigma: float) -> np.ndarray:
    """Unit-peak smooth random field, so ``alpha`` reads as pixels of displacement.
    Generated coarse and upsampled: the field is band-limited at ``sigma`` px anyway.
    """
    H, W = shape
    step = max(1, int(sigma / 1.5))
    if step > 1:
        f = rng.standard_normal((max(2, H // step), max(2, W // step)), dtype=np.float32)
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma / step, sigmaY=sigma / step)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        f = rng.standard_normal((H, W), dtype=np.float32)
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    peak = float(np.abs(f).max())
    return f / peak if peak > _EPS else f


def _op_elastic(img: np.ndarray, lab: np.ndarray | None, params: dict,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """One displacement field, applied to image and mask alike."""
    alpha = float(params.get("alpha", 20.0))
    sigma = float(params.get("sigma", 6.0))
    if alpha <= 0.0 or sigma <= 0.0:
        return img, lab
    H, W = img.shape
    dx = _smooth_field(rng, (H, W), sigma) * np.float32(alpha)
    dy = _smooth_field(rng, (H, W), sigma) * np.float32(alpha)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x, map_y = xs + dx, ys + dy
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=_BORDER)
    if lab is not None:
        lab = cv2.remap(lab, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                        borderMode=_BORDER)
    return out, lab


@dataclass(frozen=True)
class _Op:
    name: str
    p: float
    params: dict[str, Any] = field(default_factory=dict)


def _parse_op(spec: Any) -> _Op:
    if isinstance(spec, _Op):
        return spec
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"augmentation op must be a dict or a name, got {spec!r}")

    params = dict(spec)
    name = params.pop("name", None)
    if name is None:
        raise ValueError(f"augmentation op is missing 'name': {spec!r}")
    p = float(params.pop("p", 1.0))

    if name in ("horizontal_flip", "vertical_flip"):
        raise ValueError(
            f"{name!r} is a constructor flag on Augmenter, not a pipeline entry; "
            "flips are handled before the pipeline so they can be inverted by a "
            "consistency branch."
        )
    if name not in KNOWN_OPS:
        # Fatal on purpose: a typo in YAML would silently disable an op and the
        # ablation ladder would then compare two identical runs.
        raise ValueError(f"unknown augmentation op {name!r}; known: {sorted(KNOWN_OPS)}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name}: p must be in [0, 1], got {p}")
    return _Op(name=str(name), p=p, params=params)


def _as_label_array(mask: np.ndarray | None) -> tuple[np.ndarray | None, Any]:
    """Copy the mask into a dtype cv2 can warp, remembering the original dtype."""
    if mask is None:
        return None, None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"expected an (H, W) mask, got {arr.shape}")
    if arr.dtype in _CV_LABEL_DTYPES:
        return np.array(arr, copy=True), arr.dtype
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"mask must be an integer label map, got {arr.dtype}")
    # int32/int64 are not warpable by cv2; class ids (and -1 ignore) fit int16 easily.
    return arr.astype(np.int16), arr.dtype


class Augmenter:
    """Applies an op pipeline to an (image, mask) pair. ``horizontal_flip`` runs before the
    pipeline and ``vertical_flip`` is accepted only to be refused.
    """

    def __init__(self, ops: list[dict], horizontal_flip: bool,
                 vertical_flip: bool = False) -> None:
        if vertical_flip:
            raise ValueError(
                "vertical_flip is never valid for retinal OCT: it inverts the "
                "top-to-bottom layer order, which is the invariant the class indexing, "
                "the monotonic-column decoding and the topology repair all rest on."
            )
        self.ops: tuple[_Op, ...] = tuple(_parse_op(o) for o in (ops or []))
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = False

    @property
    def op_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.ops)

    def __repr__(self) -> str:  # goes into run logs next to the config snapshot
        listed = ", ".join(f"{op.name}@{op.p:g}" for op in self.ops)
        return f"Augmenter(hflip={self.horizontal_flip}, ops=[{listed}])"

    def __call__(self, image: np.ndarray, mask: np.ndarray | None,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        """Augment one sample. Shapes and dtypes come back exactly as they went in."""
        if image.ndim != 2:
            raise ValueError(f"expected a single-channel (H, W) image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"image {image.shape} and mask {mask.shape} disagree")

        img = np.array(image, dtype=np.float32, copy=True)
        lab, lab_dtype = _as_label_array(mask)

        if self.horizontal_flip and rng.random() < 0.5:
            img = cv2.flip(img, 1)
            if lab is not None:
                lab = cv2.flip(lab, 1)

        shape = img.shape
        pending: np.ndarray | None = None
        for op in self.ops:
            if rng.random() >= op.p:
                continue
            if op.name in AFFINE_OPS:
                m = _affine_matrix(op, shape, rng)
                pending = m if pending is None else m @ pending
                continue
            # Anything else must see the geometry that the config placed before it.
            img, lab = _flush_affine(img, lab, pending)
            pending = None
            if op.name == "elastic":
                img, lab = _op_elastic(img, lab, op.params, rng)
            else:
                img = _IMAGE_OPS[op.name](img, op.params, rng)
        img, lab = _flush_affine(img, lab, pending)

        img = np.ascontiguousarray(img, dtype=np.float32)
        if lab is None:
            return img, None
        return img, np.ascontiguousarray(lab.astype(lab_dtype, copy=False))


def build_augmenter(cfg: dict) -> Augmenter:
    """Build the training augmenter from the resolved ``augment`` config block."""
    cfg = cfg or {}
    return Augmenter(
        ops=list(cfg.get("pipeline", [])),
        horizontal_flip=bool(cfg.get("horizontal_flip", False)),
        vertical_flip=bool(cfg.get("vertical_flip", False)),
    )
