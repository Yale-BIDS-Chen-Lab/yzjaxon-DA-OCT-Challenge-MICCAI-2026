"""Synthetic optic-disc dive augmentation.

A top-level ``discaug:`` block, applied at full resolution before cropping and only on the
supervised train side. ``map_y(y, x) = y - a(x) * s(y, x)``: columns never move, the dive is
a raised cosine, and the breakpoints of ``s`` are the WARPED ILM and BM rows, which makes
"ILM dives by a, BM by a*r" exact. The map is checked strictly monotone per draw and labels
follow it with nearest sampling, so the op cannot cross a boundary or invent a label value.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from octtta.data.release_dataset import NUM_CLASSES

#: Mixed into every seed list here, and disjoint from every other stage's tag.
RNG_TAG = 0xD15CD

#: REPLICATE, not REFLECT: reflecting a downward shift inverts the layer stack.
_BORDER = cv2.BORDER_REPLICATE

#: ``(far, near)``: the fill samples rows ``[ILM - far, ILM - near]`` of each drawn column.
VITREOUS_REF_ROWS = (30, 6)
#: Below this many reference pixels the per-column window is not a sample; fall back.
_VITREOUS_REF_MIN_PX = 256


DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "p": 0.25,
    # One dip per frame; two overlapping dips have no single realised thickness.
    "n_dips": 1,
    # Lateral mouth width, as a fraction of W.
    "width_frac": [0.10, 0.22],
    # half_R / half_L: the asymmetric "stalk" of a real canal.
    "asymmetry": [0.6, 1.6],
    # Surviving band thickness at the dip centre, as a fraction of that column's retina.
    "residual_frac": [0.15, 0.45],
    # ...with an absolute floor in px, so a thin peripheral column cannot be crushed.
    "residual_px_min": 12,
    # r = BM dive / ILM dive. r = 0 (flat BM across the canal) is refused below.
    "bm_ratio": [0.15, 0.95],
    # Depth over which the displacement decays to zero below the warped BM, fraction of H.
    "tail_frac": 0.06,
    # Darkening strength: a lerp toward the vitreous level, never a multiplicative scale
    # (images are per-image z-scored, so a 0.5x factor BRIGHTENS every negative pixel).
    "darken": [0.3, 0.7],
    # Real disc frames are skipped: a synthetic canal on a real one is a shape nobody drew.
    "skip_protocols": ["Optic Disc, 6 x 6"],
    # Lateral band the dip centre is drawn from; the final tile is the horizontal centre.
    "centre_frac": [0.25, 0.75],
    # Fovea avoidance: the dip's whole SUPPORT (not just its centre) must miss this
    # fraction of W either side of the deepest drawn ILM column.
    "fovea_guard_frac": 0.08,
    "fovea_guard_tries": 8,
    # Hard cap on the dive as a fraction of H; a high r would otherwise demand more than H.
    "max_dive_frac": 0.30,
    # Rows a deep dive pulls in from above row 0 get vitreous-like noise instead of
    # BORDER_REPLICATE's copy of row 0, which would be a constant column -- a perfect
    # cue for "the boundaries dive here" that no test-time image carries.
    "vitreous_fill": True,
    # A dip column whose retina is thinner than this is skipped, not warped.
    "min_thickness_px": 16,
}


def _merge(defaults: dict, override: dict, path: str) -> dict:
    """Recursive merge that refuses unknown keys: a key nobody reads is not a setting."""
    out = dict(defaults)
    for k, v in override.items():
        if k not in defaults:
            raise ValueError(f"discaug: unknown key {path}{k!r}")
        if isinstance(defaults[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"discaug: {path}{k} must be a mapping")
            out[k] = _merge(defaults[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def _pair(v, name: str, *, lo_min: float | None = None, hi_max: float | None = None,
          lo_strict: bool = False, hi_strict: bool = False) -> tuple[float, float]:
    """An ordered ``[lo, hi]`` range, bounds-checked. Out of range is fatal, not clipped."""
    try:
        lo, hi = float(v[0]), float(v[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"discaug: {name} must be a [lo, hi] pair, got {v!r}") from exc
    if not lo <= hi:
        raise ValueError(f"discaug: {name} range {v!r} is not ordered")
    if lo_min is not None and (lo <= lo_min if lo_strict else lo < lo_min):
        raise ValueError(
            f"discaug: {name} lower bound {lo} must be "
            f"{'>' if lo_strict else '>='} {lo_min}")
    if hi_max is not None and (hi >= hi_max if hi_strict else hi > hi_max):
        raise ValueError(
            f"discaug: {name} upper bound {hi} must be "
            f"{'<' if hi_strict else '<='} {hi_max}")
    return lo, hi


def decode_interval_planes(mask: np.ndarray, *, partial: bool,
                           num_classes: int = NUM_CLASSES,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` int16 planes from a label plane, ``-1`` where nothing is known. The one
    place the encoding is read, and it branches on ``partial``, never on the values.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"discaug: expected an (H, W) mask, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"discaug: mask must be integer labels, got {arr.dtype}")
    if partial:
        from octtta.data.partial_labels import interval_lut

        if arr.min() < 0 or arr.max() > 255:
            raise ValueError(
                f"discaug: an interval plane is one byte per pixel; got values "
                f"[{int(arr.min())}, {int(arr.max())}]")
        lo_lut, hi_lut = interval_lut()
        codes = arr.astype(np.uint8)
        return lo_lut[codes].astype(np.int16), hi_lut[codes].astype(np.int16)
    valid = (arr >= 0) & (arr < int(num_classes))
    out = np.where(valid, arr, -1).astype(np.int16)
    return out, out.copy()


def boundary_rows(lo: np.ndarray, hi: np.ndarray, k: int) -> np.ndarray:
    """``(W,)`` rows of official boundary ``k``, NaN if not drawn. ``hi[r-1] == k-1`` strictly,
    so two coincident surfaces report "not drawn" rather than fabricating an interface.
    """
    hit = lo >= int(k)
    any_ = hit.any(axis=0)
    first = np.argmax(hit, axis=0)
    out = np.full(lo.shape[1], np.nan, dtype=np.float64)
    cols = np.where(any_ & (first > 0))[0]
    if cols.size:
        rows = first[cols]
        ok = hi[rows - 1, cols] == int(k) - 1
        out[cols[ok]] = rows[ok].astype(np.float64)
    return out


def raised_cosine(w: int, centre: float, half_l: float, half_r: float) -> np.ndarray:
    """``(W,)`` asymmetric raised-cosine dip: 1 at ``centre``, 0 at and beyond the mouths."""
    x = np.arange(w, dtype=np.float64)
    u = np.where(x < centre,
                 (x - centre) / max(half_l, 1e-6),
                 (x - centre) / max(half_r, 1e-6))
    u = np.clip(u, -1.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * u))


class DiscAug:
    """One resolved ``discaug:`` block, callable on a full-resolution sample; returns
    ``(image, mask, fired)`` and is stateless, so one instance serves every worker.
    """

    def __init__(self, cfg: dict | None, *, num_classes: int = NUM_CLASSES) -> None:
        cfg = _merge(DEFAULTS, dict(cfg or {}), "")
        self.num_classes = int(num_classes)
        if self.num_classes < 3:
            raise ValueError(f"discaug: num_classes must be >= 3, got {num_classes}")

        p = float(cfg["p"])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"discaug: p must be in [0, 1], got {cfg['p']}")
        n_dips = int(cfg["n_dips"])
        if n_dips != 1:
            raise ValueError(
                f"discaug: n_dips must be 1, got {n_dips}. Overlapping dips have no single "
                "amplitude and no single realised band thickness, so the self-check's "
                "contracts would stop meaning what they say.")

        self.p = p
        self.n_dips = n_dips
        self.width_frac = _pair(cfg["width_frac"], "width_frac",
                                lo_min=0.0, lo_strict=True, hi_max=1.0)
        self.asymmetry = _pair(cfg["asymmetry"], "asymmetry", lo_min=0.0, lo_strict=True)
        self.residual_frac = _pair(cfg["residual_frac"], "residual_frac",
                                   lo_min=0.0, lo_strict=True, hi_max=1.0, hi_strict=True)
        # r = 1 (BM follows the ILM) and r = 0 (flat BM) are refused, not clipped.
        self.bm_ratio = _pair(cfg["bm_ratio"], "bm_ratio",
                              lo_min=0.0, lo_strict=True, hi_max=1.0, hi_strict=True)
        self.darken = _pair(cfg["darken"], "darken", lo_min=0.0, hi_max=1.0)
        self.centre_frac = _pair(cfg["centre_frac"], "centre_frac", lo_min=0.0, hi_max=1.0)

        self.residual_px_min = int(cfg["residual_px_min"])
        if self.residual_px_min < 1:
            raise ValueError(
                f"discaug: residual_px_min must be >= 1 px, got {self.residual_px_min}")
        self.tail_frac = float(cfg["tail_frac"])
        if not 0.0 < self.tail_frac <= 0.5:
            raise ValueError(
                f"discaug: tail_frac must be in (0, 0.5], got {self.tail_frac}")
        self.max_dive_frac = float(cfg["max_dive_frac"])
        if not 0.0 < self.max_dive_frac <= 2.0:
            raise ValueError(
                f"discaug: max_dive_frac must be in (0, 2], got {self.max_dive_frac}")
        self.fovea_guard_frac = float(cfg["fovea_guard_frac"])
        if not 0.0 <= self.fovea_guard_frac < 0.5:
            raise ValueError(
                f"discaug: fovea_guard_frac must be in [0, 0.5), got "
                f"{self.fovea_guard_frac}")
        self.fovea_guard_tries = int(cfg["fovea_guard_tries"])
        if self.fovea_guard_tries < 1:
            raise ValueError(
                f"discaug: fovea_guard_tries must be >= 1, got {self.fovea_guard_tries}")
        self.vitreous_fill = bool(cfg["vitreous_fill"])
        self.min_thickness_px = int(cfg["min_thickness_px"])
        if self.min_thickness_px < 1:
            raise ValueError(
                f"discaug: min_thickness_px must be >= 1 px, got {self.min_thickness_px}")

        protos = cfg["skip_protocols"]
        if isinstance(protos, str) or not isinstance(protos, Sequence):
            raise ValueError(
                f"discaug: skip_protocols must be a list of protocol strings, got "
                f"{protos!r}")
        self.skip_protocols = tuple(str(x) for x in protos)
        self.cfg = cfg


    @staticmethod
    def from_config(cfg: dict | None, *, num_classes: int = NUM_CLASSES) -> "DiscAug | None":
        """None when the block is absent or disabled -- the bit-identical path."""
        if not cfg or not cfg.get("enabled", True):
            return None
        return DiscAug(cfg, num_classes=num_classes)

    def __repr__(self) -> str:
        return f"DiscAug(p={self.p:g}, bm_ratio={list(self.bm_ratio)})"

    def summary(self) -> str:
        """The one startup line. Names every knob that changes what the model is taught."""
        return (f"p={self.p:g} width_frac={list(self.width_frac)} "
                f"residual_frac={list(self.residual_frac)} "
                f"bm_ratio={list(self.bm_ratio)} "
                f"skip={list(self.skip_protocols)}")


    def __call__(self, image: np.ndarray, mask: np.ndarray | None,
                 rng: np.random.Generator, *, partial: bool,
                 protocol: str | None = None,
                 ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """``(image, mask, fired)``; the inputs come back untouched when ``fired`` is False."""
        if image.ndim != 2:
            raise ValueError(f"discaug: expected an (H, W) image, got {image.shape}")
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"discaug: image {image.shape} vs mask {mask.shape}")

        # Structural skips first, before any draw, so a skipped frame consumes nothing.
        if protocol is not None and str(protocol) in self.skip_protocols:
            return image, mask, False
        if mask is None:
            return image, mask, False
        if rng.random() >= self.p:
            return image, mask, False

        plan = self._plan(image.shape, mask, rng, partial=partial)
        if plan is None:
            return image, mask, False
        return (*self._apply(image, plan, rng), True)


    def _plan(self, shape: tuple[int, int], mask: np.ndarray,
              rng: np.random.Generator, *, partial: bool) -> dict | None:
        """Every decision, taken before a pixel moves. ``None`` = skip this frame.
        The last decision is taken on the WARPED label, and that warp is kept in the plan.
        """
        h, w = int(shape[0]), int(shape[1])
        lo, hi = decode_interval_planes(mask, partial=partial,
                                        num_classes=self.num_classes)
        y_top = boundary_rows(lo, hi, 1)                       # b1 = ILM
        y_bot = boundary_rows(lo, hi, self.num_classes - 1)    # b9 = BM
        drawn = np.isfinite(y_top) & np.isfinite(y_bot)
        if not drawn.any():
            return None

        # The deepest drawn ILM column is the fovea on a macula frame, where the vendors'
        # labels state the opposite convention to this op's.
        deepest = int(np.nanargmax(np.where(drawn, y_top, -np.inf)))
        guard = self.fovea_guard_frac * w

        # Width and asymmetry are drawn BEFORE the centre, so the guard covers the support.
        width = float(rng.uniform(*self.width_frac)) * w
        asym = float(rng.uniform(*self.asymmetry))
        half_l = width / (1.0 + asym)
        half_r = width - half_l
        c_lo, c_hi = self.centre_frac[0] * w, self.centre_frac[1] * w

        centre = None
        for _ in range(self.fovea_guard_tries):
            cand = float(rng.uniform(c_lo, c_hi))
            if cand + half_r < deepest - guard or cand - half_l > deepest + guard:
                centre = cand
                break
        if centre is None:
            return None

        g = raised_cosine(w, centre, half_l, half_r)
        window = g > 0.0
        if not window.any():
            return None
        # Every column the dip touches needs a drawn b1 and b9, no unknown pixel and a
        # real retina; anything else is a skip, never a repair.
        if not drawn[window].all():
            return None
        if (lo[:, window] < 0).any():
            return None
        thickness = y_bot - y_top
        if float(np.nanmin(thickness[window])) < self.min_thickness_px:
            return None

        r = float(rng.uniform(*self.bm_ratio))
        frac = float(rng.uniform(*self.residual_frac))
        c_idx = int(np.argmax(g))
        t_centre = float(thickness[c_idx])
        # Requested residual: a fraction of THIS column's retina, floored in absolute px.
        out_req = min(max(frac * t_centre, float(self.residual_px_min)), t_centre)
        a_req = (t_centre - out_req) / (1.0 - r)
        a_centre = min(a_req, self.max_dive_frac * h)
        amplitude = a_centre / max(float(g[c_idx]), 1e-9)

        floor = np.minimum(float(self.residual_px_min), thickness)
        with np.errstate(invalid="ignore"):
            cap = (thickness - floor) / (1.0 - r)
        a_col = np.where(window, np.minimum(amplitude * g, np.nan_to_num(cap, nan=0.0)), 0.0)
        a_col = np.maximum(a_col, 0.0)

        plan = {
            "h": h, "w": w, "lo": lo, "hi": hi,
            # Fallback ceiling for the fill when a frame has no vitreous above its ILM.
            "vitreous_ceiling": float(np.min(y_top[drawn])),
            "drawn": drawn,
            "y_top": np.where(drawn, y_top, 0.0),
            "y_bot": np.where(drawn, y_bot, float(h)),
            "g": g, "window": window, "a_col": a_col,
            "r": r, "residual_frac": frac,
            "centre": centre, "centre_col": c_idx,
            "half_l": half_l, "half_r": half_r,
            "t_centre": t_centre,
            "dive_px": float(a_col[c_idx]),
            "requested_out_px": out_req,
            "target_out_px": t_centre - float(a_col[c_idx]) * (1.0 - r),
            "dive_clamped": bool(a_req > self.max_dive_frac * h),
            "tail_px": max(self.tail_frac * h, 1.0),
        }

        # The acceptance test is stated on the OUTPUT label: nothing above implies b1 and
        # b9 are still READABLE after the resample, since nearest sampling can drop a band.
        map_y = self.vertical_map(plan)
        self._assert_monotone(map_y, plan)
        label_out = self._warp_label(mask, map_y)
        if not self._output_is_readable(label_out, window, partial=partial):
            return None
        plan["map_y"] = map_y
        plan["label_out"] = label_out
        return plan

    def _output_is_readable(self, label_out: np.ndarray, window: np.ndarray, *,
                            partial: bool) -> bool:
        """Every column the dip touches still yields a b1 AND a b9 out of the OUTPUT label."""
        out_lo, out_hi = decode_interval_planes(label_out, partial=partial,
                                                num_classes=self.num_classes)
        readable = (np.isfinite(boundary_rows(out_lo, out_hi, 1))
                    & np.isfinite(boundary_rows(out_lo, out_hi, self.num_classes - 1)))
        return bool(readable[window].all())

    def vertical_map(self, plan: dict) -> np.ndarray:
        """``(H, W)`` float32 ``map_y``, strictly increasing down every column."""
        h, w = plan["h"], plan["w"]
        r, tail = plan["r"], plan["tail_px"]
        a = plan["a_col"][None, :].astype(np.float64)
        y_i = (plan["y_top"] + plan["a_col"])[None, :]
        y_b = (plan["y_bot"] + plan["a_col"] * r)[None, :]
        band = np.maximum(y_b - y_i, 1e-6)
        yy = np.arange(h, dtype=np.float64)[:, None]

        s = np.zeros((h, w), dtype=np.float64)
        np.copyto(s, 1.0, where=yy <= y_i)
        mid = (yy > y_i) & (yy < y_b)
        np.copyto(s, 1.0 - (1.0 - r) * (yy - y_i) / band, where=mid)
        tl = (yy >= y_b) & (yy < y_b + tail)
        np.copyto(s, r * (1.0 - (yy - y_b) / tail), where=tl)
        return (yy - a * s).astype(np.float32)

    @staticmethod
    def _assert_monotone(map_y: np.ndarray, plan: dict) -> None:
        """Checked over EVERY segment, tail included, and never rescaled to fit: a map that
        folds has lost the layer order, and clipping it would kink the warp.
        """
        slope = np.diff(map_y.astype(np.float64), axis=0)
        worst = float(slope.min()) if slope.size else 1.0
        if not worst > 0.0:
            raise AssertionError(
                f"discaug: map lost monotonicity (min d(map_y)/dy = {worst:.4f}); "
                f"r={plan['r']:.3f} dive={plan['dive_px']:.1f}px T={plan['t_centre']:.1f}px")

    @staticmethod
    def _map_x(h: int, w: int) -> np.ndarray:
        return np.ascontiguousarray(
            np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)))

    def _warp_label(self, mask: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        """The label through the same backward map, nearest + replicate, so the warp cannot
        invent a class or destroy the 255 sentinel -- asserted anyway.
        """
        lab_in = np.asarray(mask)
        cv_lab = (lab_in if lab_in.dtype in (np.uint8, np.int16)
                  else lab_in.astype(np.int16))
        out = cv2.remap(np.ascontiguousarray(cv_lab),
                        self._map_x(*map_y.shape), map_y,
                        cv2.INTER_NEAREST, borderMode=_BORDER)
        invented = np.setdiff1d(np.unique(out), np.unique(cv_lab))
        if invented.size:
            raise AssertionError(  # nearest sampling makes this unreachable
                f"discaug: warp invented label values {invented.tolist()}")
        return np.ascontiguousarray(out.astype(lab_in.dtype, copy=False))

    def _apply(self, image: np.ndarray, plan: dict,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = plan["h"], plan["w"]
        map_y = plan["map_y"]
        self._assert_monotone(map_y, plan)
        img = cv2.remap(np.ascontiguousarray(image, dtype=np.float32),
                        self._map_x(h, w), map_y, cv2.INTER_LINEAR, borderMode=_BORDER)
        img = self._fill_above_frame(img, image, map_y, plan, rng)
        img = self._darken(img, plan, rng)
        # Already warped and accepted in ``_plan``; ``_apply`` reuses that warp.
        return img, plan["label_out"]

    def _fill_above_frame(self, img: np.ndarray, source: np.ndarray, map_y: np.ndarray,
                          plan: dict, rng: np.random.Generator) -> np.ndarray:
        """Replace the rows a deep dive pulled in from above row 0 with vitreous noise.
        The label is deliberately NOT touched: replicate copies row 0's class, which is right.
        """
        if not self.vitreous_fill:
            return img
        outside = map_y < 0.0
        n = int(outside.sum())
        if n == 0:
            return img
        mu, sd = self._vitreous_stats(source, plan)
        out = np.array(img, copy=True)
        out[outside] = (mu + sd * rng.standard_normal(n)).astype(np.float32)
        return np.ascontiguousarray(out, dtype=np.float32)

    def _vitreous_stats(self, source: np.ndarray, plan: dict) -> tuple[float, float]:
        """``(mu, sd)`` of this frame's real vitreous: per column, ``[ILM - 30, ILM - 6]``. NOT
        every row above the shallowest ILM, which catches black bands and bright artefacts.
        """
        far, near = VITREOUS_REF_ROWS
        h = plan["h"]
        yy = np.arange(h, dtype=np.float64)[:, None]
        ilm = np.floor(plan["y_top"])[None, :]
        band = plan["drawn"][None, :] & (yy >= ilm - far) & (yy <= ilm - near)
        ref = source[band]
        if ref.size < _VITREOUS_REF_MIN_PX:
            # No frame-own vitreous (an ILM within a few rows of the top edge).
            ceiling = int(np.floor(plan["vitreous_ceiling"]))
            ref = (source[:ceiling] if ceiling >= 8
                   else source[:max(8, source.shape[0] // 10)])
        return float(ref.mean()), float(ref.std())

    def _darken(self, img: np.ndarray, plan: dict, rng: np.random.Generator) -> np.ndarray:
        """Lerp the warped ILM-BM band toward the vitreous level, in OUTPUT coordinates.
        A lerp, never a multiplicative scale: the images are per-image z-scored.
        """
        h, w = plan["h"], plan["w"]
        amount = float(rng.uniform(*self.darken))
        if amount <= 0.0:
            return np.ascontiguousarray(img, dtype=np.float32)
        y_i = (plan["y_top"] + plan["a_col"])[None, :]
        y_b = (plan["y_bot"] + plan["a_col"] * plan["r"])[None, :]
        yy = np.arange(h, dtype=np.float64)[:, None]
        band = ((yy >= y_i) & (yy < y_b)).astype(np.float64)
        weight = amount * plan["g"][None, :] * band
        bg = float(np.percentile(img, 10.0))
        out = img.astype(np.float64) * (1.0 - weight) + bg * weight
        return np.ascontiguousarray(out, dtype=np.float32)
