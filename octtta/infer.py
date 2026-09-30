"""Per-frame inference numerics and the stable deployment API."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from octtta.data.release_dataset import NUM_CLASSES
from octtta.fusion import alpha_for_image, fuse_probs, fusion_spec

cv2.setNumThreads(0)
MAX_LABEL = NUM_CLASSES - 1
DEFAULT_PATCH = (256, 256)
DEFAULT_AUTO_FACTOR = 1.5
BLANK_COLUMNS_DEFAULTS: dict[str, Any] = {
    "enabled": False, "threshold": 0.0, "min_band": 16, "fill": "extend",
}
BLANK_COLUMNS_FILLS = ("extend", "background")
MIN_SIGNAL_COLUMNS = 32
BLANK_REFUSED_TOO_NARROW = "signal_width_below_min_signal_columns"
BLANK_THRESHOLD_MAX = 0.5

@dataclass(frozen=True)
class Plan:
    """Inference geometry and postprocessing, replaced as a whole by budget rungs."""
    mode: str = "auto"
    patch: tuple[int, int] = DEFAULT_PATCH
    overlap: float = 0.5
    auto_factor: float = DEFAULT_AUTO_FACTOR
    patch_batch: int = 4
    postproc: dict | None = None
    max_height: int | None = None
    blank_columns: dict[str, Any] | None = None
    overlap_cols: float | None = None
    dual_max_pixels: int | None = None
    large_frame_pixels: int | None = None

    def col_overlap(self) -> float:
        return self.overlap if self.overlap_cols is None else float(self.overlap_cols)

    def dual_for_shape(self, hw: tuple[int, int]) -> bool:
        return (self.dual_max_pixels is None or
                int(hw[0]) * int(hw[1]) <= int(self.dual_max_pixels))

    def describe(self) -> str:
        pp = "off" if self.postproc is None else "+".join(
            [k for k, on in (("gate", _gate_on(self.postproc)),
                             ("viterbi", self.postproc.get("enforce_monotonic_columns", True)),
                             ("smooth", bool(self.postproc.get("boundary_smooth", False))),
                             ("islands", self.postproc.get("remove_small_islands", True)))
             if on] or ["clip-only"])
        rs = "" if self.max_height is None else f" max_h={self.max_height}"
        co = "" if self.overlap_cols is None else f" cols_ov={self.col_overlap():g}"
        du = "" if self.dual_max_pixels is None else f" dual<={int(self.dual_max_pixels)}px"
        bc = "" if not self.blank_columns else (
            f" blank_cols={self.blank_columns['fill']}"
            f"(thr={self.blank_columns['threshold']:g},"
            f"min={self.blank_columns['min_band']})")
        return (f"mode={self.mode} patch={self.patch} overlap={self.overlap} "
                f"tta=off postproc={pp}{rs}{co}{du}{bc}")



def _gate_on(cfg: dict) -> bool:
    block = cfg.get("presence_gate", True)
    return bool(block) if isinstance(block, bool) else bool(dict(block or {}).get("enabled", True))


def _forward_probs(model: torch.nn.Module, x: torch.Tensor, *, amp: bool) -> torch.Tensor:
    """Return full-resolution softmax probabilities for a batch."""
    device = x.device
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=amp and device.type == "cuda"):
        prob = _softmax(model, x)
    if prob.shape[-2:] != x.shape[-2:]:
        raise RuntimeError(
            f"model returned {tuple(prob.shape[-2:])} for an input of {tuple(x.shape[-2:])}; "
            "a probability map that is not anchored to the input grid shifts every layer "
            "boundary and the scorer will not notice")
    return prob



def _softmax(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]                       # deep supervision: entry 0 is full resolution
    return torch.softmax(out.float(), dim=1)


def predict_probs(model: torch.nn.Module, image: np.ndarray, *, device: torch.device,
                  amp: bool = True) -> np.ndarray:
    """Whole-image softmax probabilities at the image's own resolution."""
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device)
    return _forward_probs(model, x, amp=amp)[0].float().cpu().numpy()



def patch_starts(extent: int, patch: int, overlap: float) -> list[int]:
    """Evenly spaced patch origins covering [0, extent), nnU-Net style.
    The last origin is pinned to extent - patch, so no patch is a thin edge sliver whose
    Gaussian weight barely overlaps its neighbour."""
    if patch >= extent:
        return [0]
    step = max(1, int(round(patch * (1.0 - float(overlap)))))
    n = int(np.ceil((extent - patch) / step)) + 1
    return [int(round(i * (extent - patch) / (n - 1))) for i in range(n)]


def _gaussian_patch_weight(h: int, w: int, sigma_scale: float = 0.125) -> np.ndarray:
    """Separable Gaussian, peak 1 at the patch centre, floored at 1/1000 of the peak.
    The floor matters: a pixel covered by exactly one patch, at its corner, would otherwise
    have no prediction at all."""
    def axis(n: int) -> np.ndarray:
        centre = (n - 1) / 2.0
        sigma = max(n * sigma_scale, 1e-6)
        g = np.exp(-((np.arange(n) - centre) ** 2) / (2.0 * sigma ** 2))
        return g
    g = np.outer(axis(h), axis(w)).astype(np.float32)
    g /= g.max()
    return np.maximum(g, 1e-3, out=g)


def sliding_window_probs(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    device: torch.device,
    patch: tuple[int, int] = DEFAULT_PATCH,
    overlap: float = 0.5,
    overlap_cols: float | None = None,
    amp: bool = True,
    batch_size: int = 4,
) -> np.ndarray:
    """nnU-Net-style tiled inference: (C, H, W) softmax over Gaussian-weighted patches.
    Patches are cut from the already-normalised image and the views are built per patch,
    exactly as training does; doing either per frame instead is a third distribution."""
    H, W = int(image.shape[0]), int(image.shape[1])
    ph, pw = min(int(patch[0]), H), min(int(patch[1]), W)
    ys = patch_starts(H, ph, overlap)
    xs = patch_starts(W, pw, overlap if overlap_cols is None else float(overlap_cols))

    arr = np.ascontiguousarray(image, dtype=np.float32)
    weight = torch.from_numpy(_gaussian_patch_weight(ph, pw)).to(device)
    coords = [(y, x) for y in ys for x in xs]

    acc: torch.Tensor | None = None
    norm = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, len(coords), max(1, int(batch_size))):
            chunk = coords[i: i + max(1, int(batch_size))]
            batch = torch.from_numpy(
                np.stack([arr[y: y + ph, x: x + pw][None] for y, x in chunk])
            ).to(device)
            prob = _forward_probs(model, batch, amp=amp).float()
            if acc is None:
                acc = torch.zeros((prob.shape[1], H, W), dtype=torch.float32, device=device)
            for j, (y, x) in enumerate(chunk):
                acc[:, y: y + ph, x: x + pw] += prob[j] * weight
                norm[:, y: y + ph, x: x + pw] += weight

    assert acc is not None                        # coords is never empty: ys, xs >= [0]
    acc /= norm.clamp_min(1e-8)
    return acc.cpu().numpy()


def choose_mode(plan: Plan, hw: tuple[int, int]) -> str:
    """Resolve mode='auto' for one image's shape."""
    if plan.mode in ("whole_image", "sliding_window"):
        return plan.mode
    H, W = hw
    ph, pw = plan.patch
    return ("sliding_window"
            if H > ph * plan.auto_factor or W > pw * plan.auto_factor
            else "whole_image")


def geometry_key(plan: Plan, hw: tuple[int, int]) -> str:
    """A short, comparable label for how one image was cut up: "sw/3x2", "whole".
    Recorded per image because max_height selects the geometry as well as the scale, at two
    thresholds that coincide at 384 rows under the shipped defaults."""
    mode = choose_mode(plan, hw)
    if mode != "sliding_window":
        return "whole"
    H, W = int(hw[0]), int(hw[1])
    ph, pw = min(int(plan.patch[0]), H), min(int(plan.patch[1]), W)
    return (f"sw/{len(patch_starts(H, ph, plan.overlap))}"
            f"x{len(patch_starts(W, pw, plan.col_overlap()))}")


def plan_tiles(plan: Plan, hw: tuple[int, int], *, signal_width: int | None = None) -> float:
    """Estimate model-A forward work in tile units."""
    h0, w0 = int(hw[0]), int(hw[1])
    h1 = min(h0, int(plan.max_height)) if plan.max_height else h0
    w1 = w0 if signal_width is None else max(1, min(w0, int(signal_width)))
    ph, pw = min(int(plan.patch[0]), h1), min(int(plan.patch[1]), w1)
    if choose_mode(plan, (h1, w1)) == "sliding_window":
        return float(len(patch_starts(h1, ph, plan.overlap))
                     * len(patch_starts(w1, pw, plan.col_overlap())))
    return float(h1 * w1) / float(max(1, ph * pw))



def resize_probs_rows(prob: np.ndarray, height: int) -> np.ndarray:
    """(C, H', W) -> (C, height, W) by linear interpolation along rows.
    The probabilities are resampled, not the labels: nearest on an argmax map quantises
    every interface to the resampling grid."""
    if int(prob.shape[1]) == int(height):
        return prob
    moved = np.ascontiguousarray(np.moveaxis(prob, 0, -1))          # (H', W, C)
    out = cv2.resize(moved, (moved.shape[1], int(height)), interpolation=cv2.INTER_LINEAR)
    if out.ndim == 2:                                # cv2 drops a trailing axis of size 1
        out = out[:, :, None]
    return np.ascontiguousarray(np.moveaxis(out, -1, 0))


def resize_probs_cols(prob: np.ndarray, width: int) -> np.ndarray:
    """(C, H, W') -> (C, H, width) by NEAREST sampling along columns.
    Nearest, unlike the row resize: a column only says where you looked, and linear would
    shift the argmax on a sloped interface by a slope-dependent fraction of a pixel."""
    if int(prob.shape[2]) == int(width):
        return prob
    moved = np.ascontiguousarray(np.moveaxis(prob, 0, -1))          # (H, W', C)
    out = cv2.resize(moved, (int(width), moved.shape[0]), interpolation=cv2.INTER_NEAREST)
    if out.ndim == 2:                                # cv2 drops a trailing axis of size 1
        out = out[:, :, None]
    return np.ascontiguousarray(np.moveaxis(out, -1, 0))


def resize_labels_cols(labels: np.ndarray, width: int) -> np.ndarray:
    """(H, W') uint8 label map -> (H, width) by nearest sampling along columns."""
    if int(labels.shape[1]) == int(width):
        return labels
    out = cv2.resize(np.ascontiguousarray(labels, dtype=np.uint8),
                     (int(width), int(labels.shape[0])), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(out)


def work_width(hw: tuple[int, int], *plans: Plan | None) -> int:
    """Return the frame's native column count for the fused map."""
    return int(hw[1])



def protocol_shape(image: np.ndarray, shape_hw: tuple[int, int] | None) -> tuple[int, int]:
    """Validate the original protocol shape supplied with a transformed canvas."""
    h0, w0 = int(image.shape[-2]), int(image.shape[-1])
    if shape_hw is None:
        return h0, w0
    if isinstance(shape_hw, (str, bytes)) or not isinstance(shape_hw, (tuple, list)) or len(shape_hw) != 2:
        raise ValueError(f"shape_hw must be the original (H, W), got {shape_hw!r}")
    ph, pw = int(shape_hw[0]), int(shape_hw[1])
    if ph <= 0 or pw <= 0 or pw != w0:
        raise ValueError(f"shape_hw={(ph, pw)} is incompatible with image {(h0, w0)}")
    return ph, pw



def resolve_blank_columns(raw: Any) -> dict[str, Any] | None:
    """Validate inference.blank_columns into a fully stated block, or None when off."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            "inference.blank_columns must be null or a mapping, got the bare boolean "
            f"{raw!r}. Write the block out: {{enabled: true, threshold: 0.0, min_band: 16, "
            "fill: extend}} -- a bare `true` names no threshold, no band floor and no fill.")
    if not isinstance(raw, dict):
        raise ValueError(
            f"inference.blank_columns must be null or a mapping, got "
            f"{type(raw).__name__} ({raw!r}). A LIST is the shape of the neighbouring "
            f"inference.blank_columns block and is refused rather than read: it names no keys of "
            f"this block, so every setting would be one nobody typed.")
    unknown = sorted(set(raw) - set(BLANK_COLUMNS_DEFAULTS))
    if unknown:
        raise ValueError(
            f"inference.blank_columns has unknown key(s) {unknown}; accepted keys are "
            f"{sorted(BLANK_COLUMNS_DEFAULTS)}. A dropped key is a setting nobody typed.")

    block = {**BLANK_COLUMNS_DEFAULTS, **raw}
    if not isinstance(block["enabled"], bool):
        raise ValueError(f"inference.blank_columns.enabled must be a bool, got "
                         f"{block['enabled']!r}.")
    if not block["enabled"]:
        return None

    thr = block["threshold"]
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not np.isfinite(thr):
        raise ValueError(f"inference.blank_columns.threshold must be a real number, got "
                         f"{thr!r}.")
    if not (0.0 <= float(thr) <= BLANK_THRESHOLD_MAX):
        raise ValueError(
            f"inference.blank_columns.threshold must be in [0, {BLANK_THRESHOLD_MAX}], got "
            f"{thr!r}. It is a FRACTION of the frame's own intensity range above the frame's "
            f"floor, not a grey level -- see the section comment above: this code runs after "
            f"per-image normalisation, where a raw 0 is not a 0.")

    mb = block["min_band"]
    if isinstance(mb, bool) or not isinstance(mb, int) or mb < 1:
        raise ValueError(
            f"inference.blank_columns.min_band must be an integer >= 1, got {mb!r}. Zero "
            f"would let a single dark column at the frame edge trigger a crop.")

    fill = block["fill"]
    if fill not in BLANK_COLUMNS_FILLS:
        raise ValueError(f"inference.blank_columns.fill must be one of "
                         f"{list(BLANK_COLUMNS_FILLS)}, got {fill!r}.")

    return {"enabled": True, "threshold": float(thr), "min_band": int(mb), "fill": str(fill)}


def blank_column_band(canvas: np.ndarray, *, threshold: float = 0.0,
                      min_band: int = 16) -> tuple[int, int]:
    """Widths of the blank bands anchored to the left and right edges of one canvas.
    threshold is a fraction of the canvas's own intensity range above its floor, not a grey
    level: this runs after per-image normalisation, where a raw 0 is not a 0."""
    arr = np.asarray(canvas)
    if arr.ndim != 2:
        raise ValueError(f"blank_column_band expects one (H, W) canvas, got shape "
                         f"{tuple(arr.shape)}")
    floor = float(arr.min())
    span = float(arr.max()) - floor
    if span <= 0.0:                      # a constant frame: everything is "blank", so nothing is
        return 0, 0
    blank = arr.max(axis=0) <= floor + float(threshold) * span
    if blank.all():                      # cannot happen once span > 0, but never crop to nothing
        return 0, 0
    left = int(np.argmax(~blank))
    right = int(np.argmax(~blank[::-1]))
    mb = int(min_band)
    return (left if left >= mb else 0), (right if right >= mb else 0)


def _fill_blank_band(prob: np.ndarray, left: int, right: int, *, fill: str) -> np.ndarray:
    """Put the cropped band back onto the probability map under the named fill rule."""
    if left == 0 and right == 0:
        return prob
    c, h, core = int(prob.shape[0]), int(prob.shape[1]), int(prob.shape[2])
    out = np.empty((c, h, left + core + right), dtype=prob.dtype)
    out[:, :, left: left + core] = prob
    if fill == "extend":
        if left:
            out[:, :, :left] = prob[:, :, :1]
        if right:
            out[:, :, left + core:] = prob[:, :, -1:]
    elif fill == "background":
        if left:
            out[:, :, :left] = 0.0
            out[0, :, :left] = 1.0
        if right:
            out[:, :, left + core:] = 0.0
            out[0, :, left + core:] = 1.0
    else:                                        # unreachable: resolve_blank_columns guards
        raise ValueError(f"unknown blank_columns fill {fill!r}")
    return out


def _canvas_probs(model: torch.nn.Module, canvas: np.ndarray, plan: Plan, *,
                  device: torch.device, amp: bool) -> tuple[np.ndarray, str, str]:
    """Run one model on one canvas and report its tiling geometry."""
    hw = (int(canvas.shape[0]), int(canvas.shape[1]))
    mode = choose_mode(plan, hw)
    if mode == "sliding_window":
        prob = sliding_window_probs(model, canvas, device=device, patch=plan.patch,
                                    overlap=plan.overlap, overlap_cols=plan.col_overlap(),
                                    amp=amp, batch_size=plan.patch_batch)
    else:
        prob = predict_probs(model, canvas, device=device, amp=amp)
    return prob, mode, geometry_key(plan, hw)



def probs_for_image(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                    device: torch.device, amp: bool, info: dict | None = None,
                    out_width: int | None = None,
                    shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Submission probabilities for one frame at its native row and column count."""
    h0, w0 = int(image.shape[0]), int(image.shape[1])
    protocol_shape(image, shape_hw)
    work = image
    if plan.max_height and h0 > int(plan.max_height):
        work = cv2.resize(np.ascontiguousarray(work, dtype=np.float32),
                          (w0, int(plan.max_height)), interpolation=cv2.INTER_LINEAR)
        if info is not None:
            info["resized_from"] = h0
    band_left = band_right = 0
    if plan.blank_columns:
        w_canvas = int(work.shape[1])
        left, right = blank_column_band(work, threshold=plan.blank_columns["threshold"],
                                        min_band=plan.blank_columns["min_band"])
        core = w_canvas - left - right
        record = {**plan.blank_columns, "canvas_width": w_canvas, "left": int(left),
                  "right": int(right), "signal_width": int(core), "applied": False}
        if (left or right) and core >= MIN_SIGNAL_COLUMNS:
            band_left, band_right = int(left), int(right)
            work = np.ascontiguousarray(work[:, band_left: band_left + core])
            record["applied"] = True
        elif left or right:
            record["refused"] = BLANK_REFUSED_TOO_NARROW
        if info is not None:
            info["blank_columns"] = record
    prob, mode, geometry = _canvas_probs(model, work, plan, device=device, amp=amp)
    if band_left or band_right:
        prob = _fill_blank_band(prob, band_left, band_right,
                                fill=plan.blank_columns["fill"])
    if info is not None:
        info["out_width"] = w0 if out_width is None else int(out_width)
        info["mode"] = mode
        info["geometry"] = geometry
    return resize_probs_cols(resize_probs_rows(prob, h0),
                             w0 if out_width is None else int(out_width))



def sanitize_labels(labels: np.ndarray) -> np.ndarray:
    """Clip to [0, MAX_LABEL] and cast to uint8; NaN becomes 0.
    A clip alone lets NaN through into an implementation-defined uint8 cast, and one
    out-of-range pixel aborts the scoring of the whole submission."""
    arr = np.asarray(labels)
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=float(MAX_LABEL), neginf=0.0)
    elif arr.dtype.kind not in "iub":
        raise TypeError(f"label map has non-numeric dtype {arr.dtype}")
    return np.clip(arr, 0, MAX_LABEL).astype(np.uint8)


def labels_from_probs(prob: np.ndarray, postproc_cfg: dict | None,
                      info: dict | None = None) -> np.ndarray:
    """Postprocess a probability map and return scorer-safe uint8 labels."""
    if postproc_cfg is None:
        labels = np.argmax(prob, axis=0)
    else:
        from octtta.postproc import postprocess
        labels = postprocess(prob, postproc_cfg, info=info)
    return sanitize_labels(labels)



def _postproc_block(raw: Any) -> tuple[dict, bool]:
    """Split a postproc block into its keys and its enabled flag."""
    block = dict(raw or {})
    return block, bool(block.pop("enabled", True))


def plan_from_config(cfg: dict) -> Plan:
    """Resolve the checkpoint's inference and postprocessing blocks."""
    inf = dict(cfg.get("inference") or {})
    data = dict(cfg.get("data") or {})
    patch = inf.get("patch_size") or data.get("train_size") or DEFAULT_PATCH
    patch_hw = (int(patch[0]), int(patch[1])) if isinstance(patch, (list, tuple)) else (int(patch), int(patch))
    postproc, enabled = _postproc_block(cfg.get("postproc"))
    blank_columns = resolve_blank_columns(inf.get("blank_columns"))
    if blank_columns:
        print(f"[plan] blank_columns: threshold={blank_columns['threshold']:g} of the "
              f"frame's intensity range above its floor, min_band="
              f"{blank_columns['min_band']} columns, fill={blank_columns['fill']}.", flush=True)
    max_height = inf.get("max_height")
    return Plan(mode=str(inf.get("mode", "auto")), patch=patch_hw,
                overlap=float(inf.get("overlap", 0.5)),
                auto_factor=float(inf.get("auto_factor", DEFAULT_AUTO_FACTOR)),
                patch_batch=int(inf.get("patch_batch", 4)),
                postproc=postproc if enabled else None,
                max_height=int(max_height) if max_height else None,
                blank_columns=blank_columns)



def load_fusion_partner(checkpoint_b: Path | str, cfg: dict, plan: Plan, *,
                        device: torch.device | str, prefer_ema: bool = True):
    """Load the baked partner, its plan and the primary's fusion table."""
    from octtta.engine import load_inference_model
    fusion = fusion_spec(cfg)
    if fusion is None:
        raise ValueError("a second checkpoint was given but the primary checkpoint's config "
                         "has no enabled fusion block")
    model_b, cfg_b = load_inference_model(checkpoint_b, device=torch.device(device),
                                          prefer_ema=prefer_ema)
    normalize = (cfg.get("data") or {}).get("normalize")
    norm_b = (cfg_b.get("data") or {}).get("normalize")
    if norm_b != normalize:
        raise ValueError(f"the two checkpoints normalise their input differently "
                         f"({normalize} vs {norm_b}); one image cannot feed both")
    plan_b = plan_from_config(cfg_b)
    if plan_b.blank_columns != plan.blank_columns:
        raise AssertionError(f"the two checkpoints disagree about inference.blank_columns -- "
                             f"A {plan.blank_columns!r}, B {plan_b.blank_columns!r}")
    if plan_b.max_height is None and plan.max_height is not None:
        plan_b = replace(plan_b, max_height=plan.max_height)
    return model_b, plan_b, fusion



def fused_probs_for_image(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                          model_b: torch.nn.Module | None = None, plan_b: Plan | None = None,
                          fusion=None, device: torch.device, amp: bool,
                          info: dict | None = None) -> np.ndarray:
    """Run the available models on a frame and mix their probability maps."""
    info = {} if info is None else info
    hw = (int(image.shape[-2]), int(image.shape[-1]))
    dual_here = model_b is not None and plan.dual_for_shape(hw)
    width = work_width(hw, plan, plan_b if dual_here else None)
    prob = probs_for_image(model, image, plan, device=device, amp=amp, info=info,
                           out_width=width)
    if model_b is None:
        return prob
    if not dual_here:
        info["fusion_alpha"] = {"skipped": "plan.dual_max_pixels", "hw": [hw[0], hw[1]],
                                "pixels": hw[0] * hw[1],
                                "dual_max_pixels": int(plan.dual_max_pixels or 0)}
        return prob
    if plan_b is None or fusion is None:
        raise ValueError("model_b needs plan_b and a FusionSpec (see load_fusion_partner)")
    info_b: dict = {}
    prob_b = probs_for_image(model_b, image, plan_b, device=device, amp=amp, info=info_b,
                             out_width=width)
    if "blank_columns" in info_b:
        info["blank_columns_b"] = info_b["blank_columns"]
    alpha, decision = alpha_for_image(hw, prob, prob_b, fusion, map_hw=(hw[0], width))
    info["fusion_alpha"] = decision
    return fuse_probs(prob, prob_b, alpha)



def probs_at_native_width(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                          device: torch.device, amp: bool,
                          info: dict | None = None) -> np.ndarray:
    """Probabilities at the frame's own width, for callers that post-process elsewhere."""
    return probs_for_image(model, image, plan, device=device, amp=amp, info=info,
                           out_width=int(image.shape[-1]))


def probs_and_work_width(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                         device: torch.device, amp: bool,
                         info: dict | None = None) -> tuple[np.ndarray, int]:
    width = work_width((int(image.shape[-2]), int(image.shape[-1])), plan)
    prob = probs_for_image(model, image, plan, device=device, amp=amp, info=info,
                           out_width=width)
    return prob, width


def labels_for_image(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                     postproc: dict | None, model_b: torch.nn.Module | None = None,
                     plan_b: Plan | None = None, fusion=None, device: torch.device,
                     amp: bool, info: dict | None = None) -> np.ndarray:
    """The complete deployed per-frame probability and label chain."""
    prob = fused_probs_for_image(model, image, plan, model_b=model_b, plan_b=plan_b,
                                 fusion=fusion, device=device, amp=amp, info=info)
    labels = labels_from_probs(prob, postproc, info=info)
    return resize_labels_cols(labels, int(image.shape[-1]))



def main(argv: Sequence[str] | None = None) -> int:
    """Keep python -m octtta.infer as the deployment command."""
    from octtta.infer_run import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
