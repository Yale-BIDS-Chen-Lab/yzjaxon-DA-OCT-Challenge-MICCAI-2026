"""Inference run control, budget degradation, masks, status and CLI."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

from octtta.data.dataset import normalize_image, read_image
from octtta.data.release_dataset import index_flat_inference_dir, output_name_for
from octtta.infer import (Plan, plan_from_config, plan_tiles, load_fusion_partner,
                          labels_for_image, sanitize_labels, MIN_SIGNAL_COLUMNS, MAX_LABEL)

LADDER_MIN_SAMPLE_IMAGES = 20
LADDER_TOLERANCE = 0.05
RATE_MIN_SAMPLES_PER_KEY = 5
SINGLE_OVER_DUAL_TILE_RATIO = 0.5636
LADDER_OVERLAP_COLS = 0.5
LADDER_OVERLAP_ROWS = 0.5
LARGE_FRAME_PERCENTILE = 0.75
LADDER_PRICED_MIN_SAVING = 0.01
LADDER_TAIL_FIXED_SECONDS = 5.0
CENSUS_MAX_SECONDS = 10.0
CENSUS_MIN_SECONDS = 1.0
DEGRADE_LOG_SCHEMA = 3
OOM_ABORT_EXIT_CODE = 17
_OOM_TYPES: tuple[type[BaseException], ...] = tuple({
    getattr(torch.cuda, "OutOfMemoryError", RuntimeError),
    getattr(torch, "OutOfMemoryError", RuntimeError),
} - {RuntimeError}) or (RuntimeError,)

def process_start_seconds() -> float | None:
    """How long this process has been alive, or None when the OS will not say.
    Read it at the same instant as the caller's own t_start; a later reading double-counts.
    """
    try:
        with open("/proc/uptime", "r", encoding="ascii") as fh:
            uptime = float(fh.read().split()[0])
        with open("/proc/self/stat", "r", encoding="ascii") as fh:
            stat = fh.read()
        fields = stat[stat.rindex(")") + 2:].split()
        ticks = float(fields[19])                      # field 22, 1-based, after comm/state
        hz = float(os.sysconf("SC_CLK_TCK"))
        age = uptime - ticks / hz
    except Exception:                                                     # noqa: BLE001
        return None
    return age if 0.0 <= age else None


class InferenceOutOfMemory(RuntimeError):
    """CUDA OOM survived the patch_batch=1 retry: the run aborts, no mask is invented."""

    def __init__(self, message: str, *, image: str | None = None, n_written: int = 0,
                 n_images: int = 0, model_b_status: str = "absent") -> None:
        super().__init__(message)
        self.image = image
        self.n_written = int(n_written)
        self.n_images = int(n_images)
        self.model_b_status = model_b_status


def is_cuda_oom(exc: BaseException) -> bool:
    """Whether exc is the allocator saying it has no memory left.
    Several kernels report an exhausted allocator as a plain RuntimeError, so the message
    is tested as well as the class."""
    if isinstance(exc, _OOM_TYPES):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "alloc_failed" in msg
    return False


def release_cuda_memory() -> None:
    """Give the cached blocks back before retrying. A no-op without CUDA.
    Call it only once the failed attempt's traceback is out of scope: while it is alive it
    pins the activations and this frees nothing."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:                                              # noqa: BLE001
        print(f"!! [infer] torch.cuda.empty_cache() failed: {exc!r}", flush=True)


def _without(cfg: dict, **off: Any) -> dict:
    return {**cfg, **off}


LadderStep = Callable[[Plan], "Plan | None"]

def _reduce_col_overlap(plan: Plan) -> Plan | None:
    """Fewer column tiles per frame; the row geometry is untouched."""
    if plan.col_overlap() <= LADDER_OVERLAP_COLS:
        return None
    return replace(plan, overlap_cols=LADDER_OVERLAP_COLS)


def _drop_dual_on_large_frames(plan: Plan) -> Plan | None:
    """The second model stops on this run's largest frames (Plan.large_frame_pixels).
    None when there is no line or no frame is above it, so next_rung skips the rung."""
    line = plan.large_frame_pixels
    if line is None:
        return None
    if plan.dual_max_pixels is not None and int(plan.dual_max_pixels) <= int(line):
        return None
    return replace(plan, dual_max_pixels=int(line))


def _drop_dual(plan: Plan) -> Plan | None:
    if plan.dual_max_pixels == 0:
        return None
    return replace(plan, dual_max_pixels=0)


def _reduce_row_overlap(plan: Plan) -> Plan | None:
    if plan.overlap <= LADDER_OVERLAP_ROWS:
        return None
    return replace(plan, overlap=LADDER_OVERLAP_ROWS)


def _drop_sliding_window(plan: Plan) -> Plan | None:
    """One forward instead of N. Costs the train/test geometry match, so it is late."""
    if plan.mode == "whole_image":
        return None
    return replace(plan, mode="whole_image")


def _drop_viterbi(plan: Plan) -> Plan | None:
    """Also forces islands and boundary smoothing off.
    Islands is the slow stage on raw argmax, and smoothing assumes the column-monotone map
    the Viterbi produces."""
    if plan.postproc is None or not plan.postproc.get("enforce_monotonic_columns", True):
        return None
    return replace(plan, postproc=_without(plan.postproc, enforce_monotonic_columns=False,
                                           remove_small_islands=False,
                                           boundary_smooth=False))


def _bare_argmax(plan: Plan) -> Plan | None:
    """The floor: whole-image inference with plain argmax and minimal overlap."""
    at_floor = (plan.postproc is None and plan.mode == "whole_image"
                and plan.overlap <= LADDER_OVERLAP_ROWS
                and plan.col_overlap() <= LADDER_OVERLAP_COLS
                and plan.dual_max_pixels == 0)
    if at_floor:
        return None
    return replace(plan, postproc=None, mode="whole_image",
                   overlap=min(plan.overlap, LADDER_OVERLAP_ROWS),
                   overlap_cols=min(plan.col_overlap(), LADDER_OVERLAP_COLS),
                   dual_max_pixels=0)



DEGRADE_LADDER: tuple[tuple[str, LadderStep], ...] = (
    ("reduce sliding-window column overlap", _reduce_col_overlap),
    ("drop second model on large frames", _drop_dual_on_large_frames),
    ("drop second model on every frame", _drop_dual),
    ("reduce sliding-window row overlap", _reduce_row_overlap),
    ("sliding window -> whole image", _drop_sliding_window),
    ("drop monotonic column repair (and islands)", _drop_viterbi),
    ("plain argmax + clip", _bare_argmax),
)

def _rung_index(step: LadderStep) -> int:
    """Where a rung sits in DEGRADE_LADDER, found by identity so reordering stays safe."""
    return next(i for i, (_n, f) in enumerate(DEGRADE_LADDER) if f is step)


COL_OVERLAP_RUNG_INDEX = _rung_index(_reduce_col_overlap)
ROW_OVERLAP_RUNG_INDEX = _rung_index(_reduce_row_overlap)
DUAL_LARGE_RUNG_INDEX = _rung_index(_drop_dual_on_large_frames)
DUAL_RUNG_INDEX = _rung_index(_drop_dual)
POSTPROC_RUNG_INDEXES = frozenset(
    i for i, (_name, fn) in enumerate(DEGRADE_LADDER)
    if fn in (_drop_viterbi, _bare_argmax))
FLOORED_PARTNER_RUNGS: tuple[tuple[int, str, float], ...] = (
    (COL_OVERLAP_RUNG_INDEX, "overlap_cols", LADDER_OVERLAP_COLS),
    (ROW_OVERLAP_RUNG_INDEX, "overlap", LADDER_OVERLAP_ROWS),
)

def partner_plan(rung: int, plan_b: Plan | None) -> Plan | None:
    """Apply reached overlap rungs to model B's own tiling plan."""
    if plan_b is None:
        return None
    changes: dict[str, Any] = {}
    for index, field, ceiling in FLOORED_PARTNER_RUNGS:
        if int(rung) <= index:
            continue
        current = plan_b.col_overlap() if field == "overlap_cols" else getattr(plan_b, field)
        if float(current) > float(ceiling):
            changes[field] = float(ceiling)
    return replace(plan_b, **changes) if changes else None



def next_rung(plan: Plan, start: int = 0) -> tuple[int, str, Plan] | None:
    """The next ladder step that actually changes something, or None at the floor."""
    for i in range(start, len(DEGRADE_LADDER)):
        name, step = DEGRADE_LADDER[i]
        new = step(plan)
        if new is not None:
            return i + 1, name, new
    return None


def write_mask(path: Path, labels: np.ndarray) -> None:
    """Single-channel uint8 PNG, the only format the scorer reads."""
    labels = sanitize_labels(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected an (H, W) label map, got {labels.shape}")
    if not cv2.imwrite(str(path), labels):
        raise IOError(f"failed to write {path}")


def write_fallback_mask(path: Path, hw: tuple[int, int]) -> None:
    """An all-class-0 mask at the input's shape: one bad image, not a bad submission."""
    write_mask(path, np.zeros((int(hw[0]), int(hw[1])), dtype=np.uint8))


def _png_header_shape(path: Path) -> tuple[int, int] | None:
    """(H, W) from a PNG's IHDR chunk, or None when the file is not a readable PNG."""
    try:
        head = path.open("rb").read(33)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    return (h, w) if 0 < h < 1 << 16 and 0 < w < 1 << 16 else None


def _image_shape(path: Path) -> tuple[int, int] | None:
    """Best-effort (H, W) for an image we could not load normally, else None."""
    for flags in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_ANYDEPTH):
        try:
            arr = cv2.imread(str(path), flags)
        except Exception:                                                 # noqa: BLE001
            continue
        if arr is not None and arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    return _png_header_shape(path)


def verify_written(pairs: Sequence[tuple[Path, tuple[int, int]]], *,
                   strict: bool = True) -> dict[str, Any]:
    """Read every written mask back and check what the official scorer asserts on.
    Catches a truncated write as well as our own bugs; strict=False reports instead of
    raising, and returns the pairs that failed under _bad_paths."""
    bad: list[str] = []
    bad_paths: list[tuple[Path, tuple[int, int]]] = []
    labels_seen: set[int] = set()
    for path, hw in pairs:
        problems: list[str] = []
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            problems.append("unreadable")
        elif arr.ndim != 2:
            problems.append(f"{arr.ndim} channels, expected 1")
        else:
            if arr.dtype != np.uint8:
                problems.append(f"dtype {arr.dtype}, expected uint8")
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 0 or hi > MAX_LABEL:
                problems.append(f"labels [{lo}, {hi}] outside [0, {MAX_LABEL}]")
            if tuple(arr.shape) != tuple(hw):
                problems.append(f"shape {arr.shape}, input was {hw}")
            labels_seen.update(np.unique(arr).tolist())
        if problems:
            bad.append(f"{path.name}: {'; '.join(problems)}")
            bad_paths.append((path, hw))
    if bad and strict:
        raise AssertionError(
            f"{len(bad)} written mask(s) would crash the official scorer:\n  "
            + "\n  ".join(bad[:20]))
    return {"n_files": len(pairs) - len(bad_paths),
            "labels_present": sorted(labels_seen),
            "bad_files": bad,
            "_bad_paths": bad_paths}


@dataclass
class _Stats:
    per_image: list[float] = field(default_factory=list)
    viterbi_cols: int = 0
    gated: dict[int, int] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    #: Per model tag ("a"/"b"), what the blank-column rule did, frames as well as hits.
    blank: dict[str, dict[str, Any]] = field(default_factory=dict)

    def note_blank(self, tag: str, record: dict | None) -> None:
        """Fold one frame's blank-column record into the per-model counters."""
        if not record:
            return
        row = self.blank.setdefault(tag, {"frames": 0, "detected": 0, "applied": 0,
                                          "refused": 0, "reasons": {},
                                          "max_left": 0, "max_right": 0,
                                          "_lefts": [], "_rights": []})
        row["frames"] += 1
        left, right = int(record.get("left", 0)), int(record.get("right", 0))
        if left or right:
            row["detected"] += 1
            row["_lefts"].append(left)
            row["_rights"].append(right)
        if record.get("applied"):
            row["applied"] += 1
        elif left or right:
            row["refused"] += 1
            why = str(record.get("refused") or "refused")
            row["reasons"][why] = row["reasons"].get(why, 0) + 1
        row["max_left"] = max(int(row["max_left"]), left)
        row["max_right"] = max(int(row["max_right"]), right)

    def blank_block(self) -> dict[str, dict[str, Any]]:
        """The blank-column counters as they go into the summary, private keys dropped."""
        def median(xs: list[int]) -> int:
            if not xs:
                return 0
            v = sorted(xs)
            return int(v[len(v) // 2]) if len(v) % 2 else int(
                round(0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])))

        out: dict[str, dict[str, Any]] = {}
        for tag, row in self.blank.items():
            out[tag] = {k: v for k, v in row.items() if not k.startswith("_")}
            out[tag]["median_left"] = median(row["_lefts"])
            out[tag]["median_right"] = median(row["_rights"])
        return out



def census_shapes(images: Sequence[Path], *, deadline_seconds: float
                  ) -> tuple[list[tuple[int, int] | None], dict[str, Any]]:
    """Every frame's (H, W) from its PNG header, plus a record, inside deadline_seconds.
    Frames it could not read come back as None, one entry per frame either way, so the
    caller's indices stay aligned with the run's own order."""
    shapes: list[tuple[int, int] | None] = []
    t0 = time.perf_counter()
    deadline = max(0.0, float(deadline_seconds))
    stopped_early = False
    for path in images:
        if time.perf_counter() - t0 > deadline:
            stopped_early = True
            break
        shapes.append(_png_header_shape(Path(path)))
    shapes.extend([None] * (len(images) - len(shapes)))
    known = [hw for hw in shapes if hw is not None]
    by_hw: dict[str, int] = {}
    for h, w in known:
        by_hw[f"{h}x{w}"] = by_hw.get(f"{h}x{w}", 0) + 1
    record = {
        "n_read": len(known),
        "n_estimated": len(shapes) - len(known),
        "seconds": round(time.perf_counter() - t0, 3),
        "stopped_early": stopped_early,
        "megapixels_total": round(sum(h * w for h, w in known) / 1e6, 3),
        "by_hw": dict(sorted(by_hw.items())),
    }
    return shapes, record


def budget_line_seconds(budget_seconds: float | None,
                        hard_deadline_seconds: float | None = None, *,
                        tolerance: float = LADDER_TOLERANCE) -> float | None:
    """The line the ladder judges by: the soft budget plus tolerance, capped by the hard
    deadline less the closing work."""
    if not budget_seconds:
        return None
    soft_line = float(budget_seconds) * (1.0 + float(tolerance))
    if hard_deadline_seconds is None:
        return soft_line
    return min(soft_line, float(hard_deadline_seconds) - LADDER_TAIL_FIXED_SECONDS)


def large_frame_line(shapes: Sequence[tuple[int, int] | None]) -> int | None:
    """The 75th-percentile pixel count of this run, or None when no frame is above it.
    None as well when any frame is unmeasured: a line drawn through a partial census would
    move with whichever files the filesystem happened to answer for."""
    if not shapes or any(hw is None for hw in shapes):
        return None
    pixels = sorted(int(h) * int(w) for h, w in shapes)          # type: ignore[misc]
    line = pixels[int(LARGE_FRAME_PERCENTILE * len(pixels))]
    return line if pixels[-1] > line else None


@dataclass
class BudgetProjection:
    """Projects the run's finishing time from measured seconds per tile, and says when the
    degrade ladder may fire."""

    #: Per frame, in the run's own order; None where the census could not read the header.
    shapes: list[tuple[int, int] | None]
    #: The SOFT budget the caller was given. Never the hard deadline.
    budget_seconds: float
    hard_deadline_seconds: float | None = None
    dual_available: bool = True
    tolerance: float = LADDER_TOLERANCE
    min_sample: int = LADDER_MIN_SAMPLE_IMAGES
    min_per_key: int = RATE_MIN_SAMPLES_PER_KEY
    #: Seconds per tile, one sample per timed frame, split by whether both models ran.
    dual_samples: list[float] = field(default_factory=list)
    single_samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shapes = list(self.shapes)
        self._fill_cache: dict[tuple, float] = {}
        #: How many rungs have fired; stamped on the read-out so samples can be told apart.
        self.generation: int = 0
        #: The samples measured under the plan the run is on now; cleared by note_rung.
        self.recent: dict[str, list[float]] = {"dual": [], "single": []}
        #: Frames that must be timed before another rung may fire.
        self._fire_gate: int = 0
        self._suffix_cache: dict[tuple, tuple[list[float], list[float]]] = {}
        known = [int(h) * int(w) for hw in self.shapes if hw is not None
                 for h, w in (hw,)]
        #: The pixel count an unreadable frame is priced at.
        self.median_pixels: int | None = (sorted(known)[len(known) // 2] if known else None)
        self.tail_seconds: float = LADDER_TAIL_FIXED_SECONDS
        #: The two halves of the line: the budget band, and the wall clock less the tail.
        self.soft_line: float = float(self.budget_seconds) * (1.0 + float(self.tolerance))
        self.hard_line: float | None = (
            None if self.hard_deadline_seconds is None
            else float(self.hard_deadline_seconds) - self.tail_seconds)
        line = budget_line_seconds(self.budget_seconds, self.hard_deadline_seconds,
                                   tolerance=float(self.tolerance))
        #: The earlier of the two halves, which is what a log line reports.
        self.fire_line: float = 0.0 if line is None else float(line)

    def observe(self, index: int, tiles: float, dual: bool, seconds: float) -> None:
        """Record one frame's seconds per tile, under the plan the run is on now."""
        if tiles <= 0 or seconds <= 0:
            return
        rate = float(seconds) / float(tiles)
        (self.dual_samples if dual else self.single_samples).append(rate)
        self.recent["dual" if dual else "single"].append(rate)

    @property
    def n_measured(self) -> int:
        """Frames whose seconds-per-tile this run has actually measured."""
        return len(self.dual_samples) + len(self.single_samples)

    def rates(self) -> dict[str, Any]:
        """Seconds per tile for dual and single frames, and whether each kind was measured.
        A kind with too few samples of its own is derived from the other through
        SINGLE_OVER_DUAL_TILE_RATIO and reported as not measured for as long as that lasts."""
        def median(xs: list[float]) -> float | None:
            if not xs:
                return None
            s = sorted(xs)
            return float(s[len(s) // 2]) if len(s) % 2 else float(
                0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2]))

        def bag(key: str, whole: list[float]) -> list[float]:
            fresh = self.recent[key]
            return fresh if len(fresh) >= self.min_per_key else whole

        dual_bag, single_bag = bag("dual", self.dual_samples), bag("single", self.single_samples)
        med_d, med_s = median(dual_bag), median(single_bag)
        have_d = len(dual_bag) >= self.min_per_key
        have_s = len(single_bag) >= self.min_per_key
        if have_d:
            rate_d = med_d
        elif med_s is not None:
            rate_d = med_s / SINGLE_OVER_DUAL_TILE_RATIO
        else:
            rate_d = med_d
        if have_s:
            rate_s = med_s
        elif med_d is not None:
            rate_s = med_d * SINGLE_OVER_DUAL_TILE_RATIO
        else:
            rate_s = med_s
        return {"dual": float(rate_d or 0.0), "single": float(rate_s or 0.0),
                "dual_measured": bool(have_d), "single_measured": bool(have_s),
                "dual_this_plan": len(self.recent["dual"]) >= self.min_per_key,
                "single_this_plan": len(self.recent["single"]) >= self.min_per_key}

    def note_rung(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """Tell the projection a rung fired; says whether it could price the saving.
        A rung whose saving the tile model cannot see is paid for in measurement instead: the
        next rung waits for min_sample more timed frames."""
        self.generation += 1
        self.recent = {"dual": [], "single": []}
        remaining = float(before.get("remaining_seconds", 0.0))
        saved = remaining - float(after.get("remaining_seconds", 0.0))
        priced = bool(remaining > 0.0 and saved > remaining * LADDER_PRICED_MIN_SAVING)
        if not priced:
            self._fire_gate = self.n_measured + self.min_sample
        return {"priced": priced,
                "saved_seconds": round(saved, 3),
                "generation": self.generation,
                "measured_before_next_rung": (0 if priced
                                              else self._fire_gate - self.n_measured)}

    def tiles_for(self, index: int, plan: Plan) -> float:
        """Tiles for one frame, or the run's median when its shape could not be read."""
        hw = self.shapes[index]
        if hw is not None:
            return plan_tiles(plan, hw)
        return self._fill_for(plan)

    @staticmethod
    def _plan_key(plan: Plan) -> tuple:
        """Cache key over every plan field a tile count depends on."""
        return (plan.mode, plan.patch, plan.overlap, plan.col_overlap(), plan.auto_factor,
                plan.max_height, plan.dual_max_pixels)

    def _fill_for(self, plan: Plan) -> float:
        key = self._plan_key(plan)
        if key not in self._fill_cache:
            known = sorted(plan_tiles(plan, hw) for hw in self.shapes if hw is not None)
            self._fill_cache[key] = float(known[len(known) // 2]) if known else 1.0
        return self._fill_cache[key]

    def _suffixes(self, plan: Plan) -> tuple[list[float], list[float]]:
        """Per plan, the dual and single tiles still to come after each frame index."""
        key = self._plan_key(plan)
        cached = self._suffix_cache.get(key)
        if cached is None:
            n = len(self.shapes)
            dual_after = [0.0] * (n + 1)
            single_after = [0.0] * (n + 1)
            for j in range(n - 1, -1, -1):
                tiles = self.tiles_for(j, plan)
                if self.dual_for(j, plan):
                    dual_after[j] = dual_after[j + 1] + tiles
                    single_after[j] = single_after[j + 1]
                else:
                    dual_after[j] = dual_after[j + 1]
                    single_after[j] = single_after[j + 1] + tiles
            cached = (dual_after, single_after)
            self._suffix_cache[key] = cached
        return cached

    def dual_for(self, index: int, plan: Plan) -> bool:
        """Whether the frame at index runs through both models under plan."""
        if not self.dual_available:
            return False
        hw = self.shapes[index]
        if hw is None:
            if self.median_pixels is None:
                return True
            hw = (1, int(self.median_pixels))
        return plan.dual_for_shape(hw)

    def project(self, index: int, elapsed: float, plan: Plan, *,
                wall_elapsed: float | None = None) -> dict[str, Any]:
        """Where the run is heading, and whether that fires the ladder.
        "fires" also needs enough timed frames and a cleared gate, so a rung never fires on a
        projection the run has no evidence for."""
        r = self.rates()
        dual_after, single_after = self._suffixes(plan)
        at = min(max(int(index) + 1, 0), len(self.shapes))
        remaining = dual_after[at] * r["dual"] + single_after[at] * r["single"]
        projected = float(elapsed) + remaining
        wall = float(elapsed if wall_elapsed is None else wall_elapsed)
        projected_wall = wall + remaining
        over_line = (projected > self.soft_line
                     or (self.hard_line is not None and projected_wall > self.hard_line))
        n = self.n_measured
        return {
            "projected_seconds": projected,
            "projected_wall_seconds": projected_wall,
            "remaining_seconds": remaining,
            "elapsed_seconds": float(elapsed),
            "wall_elapsed_seconds": wall,
            "soft_line": float(self.soft_line),
            "hard_line": self.hard_line,
            "n_remaining": max(0, len(self.shapes) - int(index) - 1),
            "rate_dual": r["dual"],
            "rate_single": r["single"],
            "rate_dual_measured": r["dual_measured"],
            "rate_single_measured": r["single_measured"],
            "n_measured": n,
            "n_measured_this_plan": len(self.recent["dual"]) + len(self.recent["single"]),
            "plan_generation": self.generation,
            "rate_dual_this_plan": r["dual_this_plan"],
            "rate_single_this_plan": r["single_this_plan"],
            "budget_seconds": float(self.budget_seconds),
            "fire_line": float(self.fire_line),
            "tail_seconds": (0.0 if self.hard_deadline_seconds is None
                             else float(self.tail_seconds)),
            "hard_deadline_seconds": (None if self.hard_deadline_seconds is None
                                      else float(self.hard_deadline_seconds)),
            # Two of these four are about EVIDENCE, not time: a rung waits for min_sample timed
            # frames and for the gate an unpriced rung set.
            "fires": bool(self.budget_seconds > 0 and n >= self.min_sample
                          and n >= self._fire_gate and over_line),
            "gated_until_measured": int(self._fire_gate),
        }


def oom_retry_plan(plan: Plan | None) -> Plan | None:
    """Retry with one patch per forward while keeping frame geometry unchanged."""
    return None if plan is None else replace(plan, patch_batch=1)



def labels_with_oom_retry(model: torch.nn.Module, image: np.ndarray, plan: Plan, *,
                          postproc: dict | None, model_b: torch.nn.Module | None = None,
                          plan_b: Plan | None = None, fusion=None, device: torch.device,
                          amp: bool, info: dict, name: str = "") -> tuple[np.ndarray, bool]:
    """Retry a CUDA OOM once at patch_batch=1, then abort the run on another OOM."""
    try:
        return labels_for_image(model, image, plan, postproc=postproc, model_b=model_b,
                                plan_b=plan_b, fusion=fusion, device=device, amp=amp,
                                info=info), False
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        first = f"{type(exc).__name__}: {exc}"
    release_cuda_memory()
    retry, retry_b = oom_retry_plan(plan), oom_retry_plan(plan_b)
    print(f"!! [infer] CUDA OOM on {name}: {first}", flush=True)
    print(f"!! [infer] cache emptied; retrying this image ONCE at patch_batch=1 "
          f"-> {retry.describe()}", flush=True)
    info.clear()
    try:
        labels = labels_for_image(model, image, retry, postproc=postproc, model_b=model_b,
                                  plan_b=retry_b, fusion=fusion, device=device, amp=amp,
                                  info=info)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        raise InferenceOutOfMemory(
            f"CUDA out of memory on {name} and again after emptying the cache and "
            f"retrying at patch_batch=1 (first: {first}; retry: "
            f"{type(exc).__name__}: {exc})", image=name) from exc
    return labels, True



def run_inference(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    *,
    device: str = "cuda",
    budget_seconds: float | None = None,
    hard_deadline_seconds: float | None = None,
    degrade_on_budget: bool = True,
    start_rung: int = 0,
    checkpoint_b: Path | None = None,
    fusion_block: dict | None = None,
) -> dict[str, Any]:
    """Index the input directory, predict, write and verify every mask; returns the summary.
    Per-image failures are contained and get all-zero masks. CUDA OOM is not: it is a
    property of the process, so it aborts instead of blanking every remaining frame."""
    t_start = time.perf_counter()
    # Read HERE, at the same instant as t_start: a later reading would already contain
    # the checkpoint loads that t_start also counts, and charge them twice.
    process_age_at_start = process_start_seconds()
    from octtta.engine import load_inference_model

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] cuda requested but unavailable; using cpu")
        device = "cpu"
    dev = torch.device(device)

    model, cfg = load_inference_model(checkpoint, device=dev)
    if fusion_block is not None:
        from octtta.fusion import explicit_block_problems, fusion_spec as _resolve_fusion

        def _refuse_block(why: str) -> None:
            print("##################################################################", flush=True)
            print(f"!! [infer] the selected fusion block was refused: {why}", flush=True)
            print("!! [infer] keeping the BAKED fusion table and continuing.", flush=True)
            print("##################################################################", flush=True)

        block = dict(fusion_block)
        chosen_spec = None
        if checkpoint_b is None:
            _refuse_block("a block was selected but no second checkpoint was given; "
                          "there is nothing to mix and the wiring is wrong")
        else:
            chosen_spec = _resolve_fusion({"fusion": block})
            if chosen_spec is None:
                _refuse_block(f"it resolves to no mixture: {block!r}")
            else:
                leaks = explicit_block_problems(block, chosen_spec)
                if leaks:
                    _refuse_block("it is not the canonical block; each line below is a "
                                  "number nobody typed:\n  " + "\n  ".join(leaks))
                    chosen_spec = None
        if chosen_spec is not None:
            cfg = {**cfg, "fusion": block}
            print(f"[infer] fusion table from the on-server selection: "
                  f"{chosen_spec.describe()}")
    runtime = cfg.get("runtime") or {}
    amp = bool(runtime.get("amp", True))
    normalize = (cfg.get("data") or {}).get("normalize")
    plan = plan_from_config(cfg)

    model_b = None
    plan_b: Plan | None = None
    fusion = None
    # One of octtta.runenv_status.MODEL_B_STATES.
    second_model_status = "absent" if checkpoint_b is None else "loaded"
    if checkpoint_b is not None:
        try:
            model_b, plan_b, fusion = load_fusion_partner(
                checkpoint_b, cfg, plan, device=dev)
            print(f"[infer] fusion: second model {Path(checkpoint_b).name} "
                  f"{fusion.describe()}  plan_b={plan_b.describe()}", flush=True)
        except Exception as exc:                                            # noqa: BLE001
            model_b = plan_b = fusion = None
            second_model_status = "dropped_oom" if is_cuda_oom(exc) else "dropped_error"
            print("##################################################################", flush=True)
            print(f"!! [infer] the second model could not be loaded from {checkpoint_b}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            print("!! [infer] falling back to a SINGLE-MODEL run: the plan now says "
                  "single-model inference (dual<=0px). Masks are still written for every "
                  "image.", flush=True)
            print(f"!! [infer] second model status: {second_model_status}  "
                  "(recorded in the summary and in $STATE_DIR/run_status.json; the final "
                  "SUBMISSION MODE reads it)", flush=True)
            print("##################################################################", flush=True)
            traceback.print_exc()

    if model_b is None:
        # Said in the plan, once, so every later reader sees a single-model run.
        plan = replace(plan, dual_max_pixels=0)

    images = index_flat_inference_dir(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] {len(images)} images from {input_dir}")
    if not images:
        raise FileNotFoundError(f"no input images found under {input_dir}")

    # Who set the soft budget: the caller, the config's reserve, or nobody.
    budget_source = "caller"
    if budget_seconds is None:
        budget = (cfg.get("server_budget") or runtime.get("server_budget") or {})
        budget_seconds = float(budget.get("reserve_for_inference", 0.0)) or None
        if budget_seconds:
            budget_source = "config:server_budget.reserve_for_inference"
            print(f"[infer] no budget was passed; falling back to the config's inference "
                  f"reserve ({budget_seconds:.0f}s). That number is what the STAGE was "
                  "reserved, not what this container has left -- and no hard deadline came "
                  "with it, so the ladder's fire line is that plus "
                  f"{LADDER_TOLERANCE:.0%}.", flush=True)
        else:
            budget_source = "none"

    shapes, shape_census = census_shapes(
        images, deadline_seconds=(
            max(CENSUS_MIN_SECONDS, min(CENSUS_MAX_SECONDS, 0.02 * budget_seconds))
            if budget_seconds else CENSUS_MAX_SECONDS))
    line = large_frame_line(shapes)
    plan = replace(plan, large_frame_pixels=line)
    shape_census["large_frame_pixels"] = line
    shape_census["n_large_frames"] = (
        0 if line is None else sum(1 for hw in shapes if hw is not None
                                   and hw[0] * hw[1] > line))
    # Priced at the census's own widths, which is an upper bound: a frame with a blank
    # band really runs on fewer columns, and a PNG header cannot see one.
    shape_census["tiles_total"] = round(
        sum(plan_tiles(plan, hw) for hw in shapes if hw is not None), 1)
    print(f"[infer] shape census: read {shape_census['n_read']}/{len(images)} headers in "
          f"{shape_census['seconds']:.2f}s, {shape_census['megapixels_total']} Mpx, "
          f"{shape_census['tiles_total']} tiles, shapes {shape_census['by_hw']}")
    if line is None:
        print(f"!! [infer] no large-frame line: {shape_census['n_estimated']} frame(s) "
              "unmeasured or every frame the same size. The ladder's 'drop second model on "
              "large frames' rung is a no-op for this run and will be skipped; the rung "
              "below it drops the second model on every frame.", flush=True)
    else:
        print(f"[infer] large-frame line: >{line} px -> "
              f"{shape_census['n_large_frames']}/{len(images)} frames")

    rung = 0
    # The rung the run STARTS on, in degrade_log's 1-based numbering; None at the top.
    start_rung_applied: int | None = None
    for _ in range(max(0, int(start_rung))):
        step = next_rung(plan, rung)
        if step is None:
            break
        rung, name, plan = step
        start_rung_applied = rung
        followed = partner_plan(rung, plan_b)
        if followed is not None:
            plan_b = followed
            print("[infer] pre-degraded: model B's own plan follows the rung it belongs to "
                  f"-> plan_b={plan_b.describe()}")
        print(f"[infer] pre-degraded: {name}")

    # Captured here, after the pre-degrade and before the first frame, so the summary says
    # what the run was armed with rather than what the config asked for.
    blank_armed = {"a": plan.blank_columns,
                   "b": (plan_b.blank_columns
                         if plan_b is not None and model_b is not None else None)}

    print(f"[infer] weights={cfg.get('_weights')}  device={dev}  {plan.describe()}  "
          f"budget={f'{budget_seconds:.0f}s' if budget_seconds else 'none'}"
          + ("" if hard_deadline_seconds is None
             else f"  hard deadline={float(hard_deadline_seconds):.0f}s"))

    startup_seconds = 0.0
    if hard_deadline_seconds is not None:
        age = process_age_at_start
        if age is None:
            print("[infer] this OS does not report a process start time; the projection "
                  "measures from the moment inference began, so the wall clock is ahead of "
                  "it by however long the interpreter took", flush=True)
        elif age > float(hard_deadline_seconds):
            print(f"!! [infer] this process has been alive {age:.0f}s, longer than the "
                  f"{float(hard_deadline_seconds):.0f}s deadline it was handed -- so that "
                  "age is not this run's start-up, and nothing is charged to it. The hard "
                  "deadline is being read as if it started here.", flush=True)
        else:
            startup_seconds = float(age)
            print(f"[infer] start-up before inference began: {startup_seconds:.2f}s, "
                  "charged to the projection (the wall clock already counted it)",
                  flush=True)

    projection = BudgetProjection(
        shapes=shapes, budget_seconds=float(budget_seconds or 0.0),
        hard_deadline_seconds=hard_deadline_seconds,
        dual_available=model_b is not None)
    if budget_seconds:
        print(f"[infer] ladder fire line {projection.fire_line:.0f}s "
              f"(budget {budget_seconds:.0f}s + {LADDER_TOLERANCE:.0%}"
              + ("" if hard_deadline_seconds is None else
                 f", capped by the {float(hard_deadline_seconds):.0f}s wall clock less "
                 f"{projection.tail_seconds:.0f}s of closing work")
              + ")", flush=True)
        if projection.fire_line < float(budget_seconds):
            print(f"!! [infer] the closing work ({projection.tail_seconds:.0f}s) does not "
                  f"fit inside the wall clock's grace, so the ladder's line "
                  f"({projection.fire_line:.0f}s) is EARLIER than the budget "
                  f"({budget_seconds:.0f}s): this run will degrade before it has spent "
                  "what it was given.", flush=True)

    written: list[tuple[Path, tuple[int, int]]] = []
    labels_present: set[int] = set()
    pending_verify: list[tuple[Path, tuple[int, int]]] = []
    stats = _Stats()
    degrade_log: list[dict[str, Any]] = []
    n_dual = 0
    n_single = 0
    oom_retried: list[str] = []

    for i, img_path in enumerate(images):
        t0 = time.perf_counter()
        out_path = output_dir / output_name_for(img_path)
        hw: tuple[int, int] | None = None
        info: dict[str, Any] = {}
        tiles_run = 0.0
        dual_here = False
        try:
            raw = read_image(img_path)
            hw = (int(raw.shape[0]), int(raw.shape[1]))
            dual_here = model_b is not None and plan.dual_for_shape(hw)
            tiles_run = plan_tiles(plan, hw)
            image = normalize_image(raw, normalize)
            labels, retried = labels_with_oom_retry(
                model, image, plan, postproc=plan.postproc,
                model_b=model_b, plan_b=plan_b,
                fusion=fusion, device=dev, amp=amp, info=info, name=img_path.name)
            # Re-priced on the columns the network really saw, so the rate stays per tile.
            band = info.get("blank_columns")
            if band and band.get("applied"):
                tiles_run = plan_tiles(plan, hw,
                                       signal_width=int(band["signal_width"]))
            if retried:
                oom_retried.append(img_path.name)
            if labels.shape != hw:
                raise RuntimeError(f"label map {labels.shape} != input {hw}")
            write_mask(out_path, labels)
            written.append((out_path, hw))
            # Verified inside this frame's own timed cost, so the read-back is priced by the
            # evaluation machine's filesystem through the projection.
            try:
                one = verify_written([(out_path, hw)], strict=False)
            except Exception as exc:                                      # noqa: BLE001
                print(f"!! [infer] could not verify {out_path.name} now ({exc!r}); "
                      "re-checked after the loop", flush=True)
                pending_verify.append((out_path, hw))
            else:
                if one["_bad_paths"]:
                    pending_verify.extend(one["_bad_paths"])
                else:
                    labels_present.update(one["labels_present"])
            if dual_here:
                n_dual += 1
            else:
                n_single += 1
        except InferenceOutOfMemory as oom:
            # Must precede the generic clause below: it is a RuntimeError, and the generic one
            # would turn a process-wide memory failure into one all-zero mask per image.
            oom.n_written = len(written)
            oom.n_images = len(images)
            oom.model_b_status = second_model_status
            print("##################################################################", flush=True)
            print(f"!! [infer] {oom}", flush=True)
            print(f"!! [infer] ABORTING after {len(written)}/{len(images)} masks. No "
                  "all-zero mask is written for this image: a zero mask here would spend "
                  "the rest of the budget turning a memory problem into a permanent 0.09.",
                  flush=True)
            print("##################################################################", flush=True)
            raise
        except Exception as exc:                                          # noqa: BLE001
            # One unreadable file must not cost the other images their masks.
            stats.failures.append(f"{img_path.name}: {type(exc).__name__}: {exc}")
            print(f"!! [infer] FAILED on {img_path.name}: {type(exc).__name__}: {exc}",
                  flush=True)
            if len(stats.failures) <= 3:
                traceback.print_exc()
            if hw is None:
                hw = _image_shape(img_path)
            if hw is None:
                print(f"!! [infer] shape of {img_path.name} unknown; no mask written "
                      f"(the scorer will score this image 0)", flush=True)
            else:
                try:
                    write_fallback_mask(out_path, hw)
                    written.append((out_path, hw))
                    pending_verify.append((out_path, hw))
                    print(f"!! [infer] wrote all-zero fallback mask {hw} for "
                          f"{img_path.name}", flush=True)
                except Exception as exc2:                                 # noqa: BLE001
                    print(f"!! [infer] fallback write also failed: {exc2!r}", flush=True)

        stats.note_blank("a", info.get("blank_columns"))
        stats.note_blank("b", info.get("blank_columns_b"))
        stats.viterbi_cols += int(info.get("viterbi_columns", 0))
        stats.modes[info.get("mode", "n/a")] = stats.modes.get(info.get("mode", "n/a"), 0) + 1
        for c in info.get("gated_classes", []):
            stats.gated[int(c)] = stats.gated.get(int(c), 0) + 1
        image_seconds = time.perf_counter() - t0
        stats.per_image.append(max(image_seconds, 0.0))

        now = time.perf_counter() - t_start + startup_seconds
        elapsed = now
        projection.observe(i, tiles_run, dual_here, stats.per_image[-1])
        read = projection.project(i, elapsed, plan, wall_elapsed=now)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"[infer] {i + 1}/{len(images)}  {stats.per_image[-1]:.2f}s  "
                  f"mean {np.mean(stats.per_image):.2f}s  "
                  + (f"projected {read['projected_seconds']:.0f}s / line "
                     f"{read['fire_line']:.0f}s  " if budget_seconds else
                     f"projected {read['projected_seconds']:.0f}s (no budget)  ")
                  + f"measured {read['n_measured']}  "
                  f"rate dual {read['rate_dual']:.4f}s/tile"
                  f"{'' if read['rate_dual_measured'] else ' (derived)'}  single "
                  f"{read['rate_single']:.4f}s/tile"
                  f"{'' if read['rate_single_measured'] else ' (derived)'}", flush=True)

        if budget_seconds and degrade_on_budget and read["fires"]:
            step = next_rung(plan, rung)
            if step is not None:
                print(f"!! [infer] projected {read['projected_seconds']:.0f}s > line "
                      f"{read['fire_line']:.0f}s (budget {budget_seconds:.0f}s + "
                      f"{LADDER_TOLERANCE:.0%}"
                      + ("" if hard_deadline_seconds is None
                         else f", capped by the {float(hard_deadline_seconds):.0f}s hard "
                              "deadline")
                      + f") after {i + 1} images, {read['n_measured']} of them timed",
                      flush=True)
                rung, name, plan = step
                followed = partner_plan(rung, plan_b)
                if followed is not None:
                    plan_b = followed
                after = projection.project(i, elapsed, plan)
                noted = projection.note_rung(read, after)
                degrade_log.append({
                    "schema": DEGRADE_LOG_SCHEMA,
                    "after_image": i + 1,
                    "rung": rung, "step": name,
                    "projected_seconds": round(read["projected_seconds"], 1),
                    "fire_line_seconds": round(read["fire_line"], 1),
                    "budget_seconds": round(float(budget_seconds), 1),
                    "n_measured": read["n_measured"],
                    "rate_dual": round(read["rate_dual"], 6),
                    "rate_single": round(read["rate_single"], 6),
                    "rate_dual_measured": read["rate_dual_measured"],
                    "rate_single_measured": read["rate_single_measured"],
                    "predicted_seconds_after": round(after["projected_seconds"], 1),
                    "projection_priced_it": noted["priced"],
                    "measured_before_next_rung": noted["measured_before_next_rung"],
                    "model_b_followed": followed is not None,
                })
                print(f"!! [infer] DEGRADING: {name}  ->  {plan.describe()}  "
                      f"(projection {read['projected_seconds']:.0f}s -> "
                      f"{after['projected_seconds']:.0f}s"
                      + ("" if noted["priced"] else
                         f"; the tile model cannot price this rung, so the next one waits "
                         f"for {noted['measured_before_next_rung']} more timed frames")
                      + ")"
                      + ("  (model B's plan follows: "
                         f"{plan_b.describe() if plan_b else ''})"
                         if followed is not None else ""), flush=True)

    rewritten: list[str] = []
    bad_files: list[str] = []
    still_bad: list[tuple[Path, tuple[int, int]]] = []
    for pair in pending_verify:
        one = verify_written([pair], strict=False)
        if not one["_bad_paths"]:
            labels_present.update(one["labels_present"])
            continue
        rewritten.extend(one["bad_files"])
        path, hw = pair
        try:
            write_fallback_mask(path, hw)
            stats.failures.append(f"{path.name}: rewritten as fallback after verify")
        except Exception as exc:                                          # noqa: BLE001
            print(f"!! [infer] could not rewrite {path}: {exc!r}", flush=True)
        again = verify_written([pair], strict=False)
        if again["_bad_paths"]:
            still_bad.append(pair)
            bad_files.extend(again["bad_files"])
        else:
            labels_present.update(again["labels_present"])
    if rewritten:
        print(f"!! [infer] {len(rewritten)} mask(s) failed verification and were rewritten "
              f"as fallbacks ({len(still_bad)} still unreadable):\n  "
              + "\n  ".join(rewritten[:20]), flush=True)
    n_written = len(written) - len(still_bad)

    total = time.perf_counter() - t_start
    wall_clock = time.perf_counter() - t_start + startup_seconds
    ladder_clock = wall_clock
    final_read = projection.project(len(images) - 1, ladder_clock, plan,
                                    wall_elapsed=wall_clock)
    summary = {
        "n_images": len(images),
        "n_written": n_written,
        "n_failed": len(stats.failures),
        "failures": stats.failures[:50],
        "seconds_total": round(total, 2),
        "seconds_per_image_mean": round(float(np.mean(stats.per_image)), 3),
        "seconds_per_image_max": round(float(np.max(stats.per_image)), 3),
        "budget_seconds": budget_seconds,
        "budget_source": budget_source,
        "over_budget": bool(budget_seconds and (
            ladder_clock > projection.soft_line
            or (projection.hard_line is not None and wall_clock > projection.hard_line))),
        "over_soft_budget": bool(budget_seconds and wall_clock > float(budget_seconds)),
        "budget_line_seconds": (None if not budget_seconds
                                else round(float(projection.fire_line), 1)),
        "soft_line_seconds": (None if not budget_seconds
                              else round(float(projection.soft_line), 1)),
        "hard_line_seconds": (None if not budget_seconds or projection.hard_line is None
                              else round(float(projection.hard_line), 1)),
        "seconds_wall_clock": round(wall_clock, 2),
        "seconds_ladder_clock": round(ladder_clock, 2),
        "startup_seconds": round(startup_seconds, 2),
        "viterbi_columns_total": stats.viterbi_cols,
        "gated_class_counts": stats.gated,
        "inference_modes": stats.modes,
        "blank_columns": {"armed_at_start": blank_armed,
                          "counts": stats.blank_block(),
                          "min_signal_columns": MIN_SIGNAL_COLUMNS},
        "final_plan": plan.describe(),
        "shape_census": shape_census,
        "projection": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in final_read.items()},
        "fusion_frames": {
            "n_dual": n_dual, "n_single": n_single,
            "n_failed": len(stats.failures),
            "n_images": len(images),
            "n_large_frames": shape_census.get("n_large_frames"),
            "large_frame_pixels": shape_census.get("large_frame_pixels"),
        },
        "degrade_log": degrade_log,
        "start_rung": start_rung_applied,
        "degrade_log_schema": DEGRADE_LOG_SCHEMA,
        "second_model": (str(checkpoint_b) if checkpoint_b is not None else None),
        "second_model_status": second_model_status,
        "inference_status": "oom_retry_patch1" if oom_retried else "ok",
        "oom": {"n_retried": len(oom_retried), "images": oom_retried[:50],
                "retry_plan": "patch_batch=1"},
        "fusion": (None if fusion is None
                   else {**fusion.as_block(),
                         "used_to_the_end": bool(n_dual > 0 and n_single == 0
                                                 and n_dual == len(images))}),
        "fusion_block_source": ("selected" if fusion_block is not None else "baked"),
        "postproc_degraded_after_image": next(
            (d["after_image"] for d in degrade_log
             if int(d.get("rung", 0)) - 1 in POSTPROC_RUNG_INDEXES), None),
        "weights": cfg.get("_weights"),
        "n_files": n_written,
        "labels_present": sorted(labels_present),
        "bad_files": bad_files,
    }
    print(f"[infer] wrote {n_written}/{len(images)} masks in {total:.1f}s "
          f"({summary['seconds_per_image_mean']}s/image); labels present "
          f"{sorted(labels_present)}")
    blank_counts = stats.blank_block()
    for tag in ("a", "b"):
        armed, row = blank_armed[tag], blank_counts.get(tag)
        if not armed and not row:
            continue
        if not row:
            print(f"[infer] blank columns (model {tag.upper()}): rule on, 0 of "
                  f"{len(images)} images reached the detector", flush=True)
            continue
        why = ", ".join(f"{k} {v}" for k, v in sorted(row["reasons"].items())) or "none"
        print(f"[infer] blank columns (model {tag.upper()}): {row['detected']} of "
              f"{row['frames']} frames carry a band (min_band "
              f"{(armed or {}).get('min_band', '?')}), {row['applied']} cropped and filled "
              f"with {(armed or {}).get('fill', '?')}, {row['refused']} refused "
              f"(reasons: {why}); band widths left median {row['median_left']} / max "
              f"{row['max_left']}, right median {row['median_right']} / max "
              f"{row['max_right']} columns", flush=True)
        if row["frames"] and not row["detected"]:
            print(f"!! [infer] blank columns (model {tag.upper()}): the rule is ARMED and "
                  f"not one of {row['frames']} frames carried a band -- these masks are "
                  "byte-identical to a run with it off.", flush=True)
    if stats.failures:
        print(f"!! [infer] {len(stats.failures)} image(s) failed and got fallback masks",
              flush=True)
    if oom_retried:
        print(f"!! [infer] {len(oom_retried)} image(s) hit CUDA OOM and were re-run at "
              f"patch_batch=1: {', '.join(oom_retried[:10])}"
              + (" ..." if len(oom_retried) > 10 else ""), flush=True)
        print("!! [infer] their masks are real predictions, but not the pipeline this "
              "package was measured with -- the final SUBMISSION MODE says so.", flush=True)
    if shape_census["n_estimated"]:
        print(f"!! [infer] the shape census could not read {shape_census['n_estimated']} "
              f"of {len(images)} frame headers"
              + (" (it ran out of its deadline)" if shape_census["stopped_early"] else "")
              + ". Those frames were priced at the median of the ones it did read, and the "
              "large-frame rung was switched off for this run.", flush=True)
    if model_b is not None:
        print(f"[infer] second model ran on {n_dual}/{n_dual + n_single} frames "
              f"({n_single} single-model)"
              + ("" if plan.dual_max_pixels is None
                 else f"; the plan ended at dual<={plan.dual_max_pixels}px"), flush=True)
    if summary["over_budget"]:
        print(f"!! [infer] OVER BUDGET: {ladder_clock:.0f}s without the read-out against "
              f"the {projection.soft_line:.0f}s band (budget {budget_seconds:.0f}s + "
              f"{LADDER_TOLERANCE:.0%})"
              + ("" if projection.hard_line is None else
                 f", {wall_clock:.0f}s on the wall clock against the {projection.hard_line:.0f}s "
                 f"cap ({float(hard_deadline_seconds):.0f}s less {projection.tail_seconds:.0f}s "
                 "of closing work)")
              + " -- the lines the ladder judged by", flush=True)
    elif summary["over_soft_budget"]:
        print(f"[infer] {total:.0f}s against a {budget_seconds:.0f}s budget -- inside the "
              f"{LADDER_TOLERANCE:.0%} band the ladder judges by, so not a degradation",
              flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m octtta.infer",
        description="Segment a flat directory of OCT B-scans into 10-class label PNGs.")
    ap.add_argument("--input", "--input_dir", dest="input", required=True,
                    help="flat directory of images (no metadata is read from it)")
    ap.add_argument("--output", "--output_dir", dest="output", required=True)
    ap.add_argument("--checkpoint", "--model_path", dest="checkpoint", required=True)
    ap.add_argument("--checkpoint-b", dest="checkpoint_b", default=None,
                    help="second model mixed with the baked fusion rule")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="soft wall-clock budget; defaults to runtime.server_budget")
    ap.add_argument("--hard-deadline-seconds", type=float, default=None,
                    help="wall-clock deadline including process startup")
    ap.add_argument("--no-degrade", "--keep-postproc-on-overrun", dest="degrade",
                    action="store_false", help="never step down the degrade ladder")
    ap.add_argument("--start-rung", type=int, default=0,
                    help="pre-apply the first N ladder steps; model B drops at rungs 2 and 3")
    ap.add_argument("--fusion-block", default=None,
                    help="JSON file holding one canonical selected fusion block")
    ap.add_argument("--summary-json", default=None, help="write the run summary here")
    return ap.parse_args(argv)



def record_run_status(**fields: Any) -> None:
    """Write the run's status line for the entry point to read back."""
    from octtta.runenv_status import SET_PREFIX, line, record

    written = record(**fields)
    print(line(fields if written is None else written, prefix=SET_PREFIX), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = _run_from_args(args)
    except InferenceOutOfMemory as oom:
        record_run_status(inference="oom_abort_baked_retry", model_b=oom.model_b_status,
                          n_images=oom.n_images, n_written=oom.n_written)
        print(f"!! [infer] exiting {OOM_ABORT_EXIT_CODE} (CUDA OOM survived the "
              "patch_batch=1 retry); infer.sh will retry from the baked weights",
              flush=True)
        return OOM_ABORT_EXIT_CODE
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    record_run_status(inference=summary["inference_status"],
                      model_b=summary["second_model_status"],
                      n_images=summary["n_images"], n_written=summary["n_written"],
                      n_failed=summary["n_failed"],
                      n_oom_retried=summary["oom"]["n_retried"],
                      inference_seconds=summary["seconds_total"])
    # A partial submission scores; a nonzero exit scores zero.
    return 0 if summary["n_written"] > 0 else 1


def _run_from_args(args: argparse.Namespace) -> dict:
    """Apply the CLI's run controls to the checkpoint's baked configuration."""
    return run_inference(
        Path(args.input), Path(args.output), Path(args.checkpoint),
        device=args.device, budget_seconds=args.budget_seconds,
        hard_deadline_seconds=args.hard_deadline_seconds,
        degrade_on_budget=args.degrade, start_rung=args.start_rung,
        checkpoint_b=Path(args.checkpoint_b) if args.checkpoint_b else None,
        fusion_block=(json.loads(Path(args.fusion_block).read_text())
                      if args.fusion_block else None),
    )



if __name__ == "__main__":
    raise SystemExit(main())
