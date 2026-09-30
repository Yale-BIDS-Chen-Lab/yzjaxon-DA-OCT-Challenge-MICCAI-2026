"""The acceptance metrics for a run, produced in one pass.

:func:`evaluate_model` runs the submission's inference path image by image and
:func:`accumulate` folds ``(pred, gt, meta)`` triples into an :class:`EvalReport`: the
official challenge score, per-class Dice / MASD / layer score, a topology violation rate,
false-invention and missed-class rates, plus the bookkeeping needed to read them (empty
denominators, cells the eval set has no images for)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from octtta.data.dataset import OCTSegDataset
from octtta.data.pseudo_widefield import PseudoWideFieldDataset
from octtta.engine import WorkerPool

from octtta.eval.challenge_metric import (
    NUM_CLASSES,
    ALPHA,
    LAMBDA_PENALTY,
    SEEN_VENDORS,
    ChallengeScore,
    ScoreRecord,
    aggregate,
    compute_image_breakdown,
    normalize_anatomy,
)
__all__ = ["EvalReport", "accumulate", "evaluate_model", "column_topology_violations_fast",
           "cohort_keys", "stratified_indices", "limit_stratified", "evaluate_parallel"]

_STATUSES = ("healthy", "diseased")
_ANATOMIES = ("Macula", "WideField")

#: Cost of a false invention: that class drops from 0.5 to ~0.0 in a mean over 10 classes.

@dataclass
class EvalReport:
    """The seven acceptance metrics plus the bookkeeping needed to read them honestly."""

    challenge_score: float
    per_class_dice: np.ndarray            # (10,) mean over GT-present images; NaN if none
    per_class_masd: np.ndarray            # (10,) height-normalised, GT-present AND predicted
    per_class_score: np.ndarray           # (10,) 0.5*dice + 0.5*masd_score, GT-present
    topology_violation_rate: float        # mean unordered_frac over predictions
    false_invention_rate: dict[int, float]
    missed_class_rate: dict[int, float]
    by_status: dict[str, float]
    by_anatomy: dict[str, float]
    by_vendor: dict[str, float]
    n_images: int
    challenge: ChallengeScore

    #: Column subsampling used for the topology diagnostic. 1 = every column (exact).
    topology_stride: int = 1

    #: Images whose GT contains the class. Denominator of dice/score/miss.
    class_support: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    #: Images whose GT contains the class *and* whose prediction does. MASD denominator.
    masd_support: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    #: Images whose GT lacks the class. Denominator of false_invention_rate.
    class_absence: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, int))
    #: Plain mean of per-image scores -- the number people mistake for the challenge score.
    mean_image_score: float = float("nan")
    #: (anatomy, vendor, status) cells the official aggregation will silently score 0.0.
    missing_cells: list[str] = field(default_factory=list)
    #: Gaps that do not trigger a 0.0 substitution but do make the number incomparable.
    coverage_notes: list[str] = field(default_factory=list)
    counts_by_status: dict[str, int] = field(default_factory=dict)
    counts_by_anatomy: dict[str, int] = field(default_factory=dict)
    counts_by_vendor: dict[str, int] = field(default_factory=dict)
    #: How many images were cut up which way, keyed by :func:`octtta.infer.geometry_key`.
    #: Two reports with different histograms were not measured under the same geometry.
    geometry: dict[str, int] = field(default_factory=dict)
    #: Per-image records in evaluation order. Aggregates alone cannot answer "is X better
    #: than Y": a paired test needs the per-image values.
    per_image: list[dict] = field(default_factory=list)

    def format(self) -> str:
        """Compact fixed-width table -- printed every val epoch, so kept under ~30 lines."""
        lines: list[str] = []

        if self.missing_cells:
            lines.append("!! INCOMPLETE EVAL SET -- official aggregation scores missing cells 0.0:")
            for note in self.missing_cells[:4]:
                lines.append(f"!!   {note}")
            if len(self.missing_cells) > 4:
                lines.append(f"!!   (+{len(self.missing_cells) - 4} more)")
            lines.append("!! FINAL below is NOT comparable to a leaderboard score.")
        if self.coverage_notes:
            # Loud too: these do not invalidate the number, but they change the question.
            lines.append("!! COVERAGE GAPS -- FINAL is computed over a different vendor set "
                         "than the leaderboard's:")
            for note in self.coverage_notes:
                lines.append(f"!!   {note}")
        # Always stated: "the penalty printed 0.0000" must not be read as "we generalise".
        if "Triton" in self.counts_by_vendor:
            lines.append(f"!! Triton present locally (n={self.counts_by_vendor['Triton']}) -- "
                         f"unseen-vendor penalty is LIVE below (weight {LAMBDA_PENALTY:g}).")
        else:
            lines.append(f"!! NO TRITON DATA -- none is ever released, so the unseen-vendor "
                         f"penalty below is structurally 0.0000 here while it is live on the "
                         f"leaderboard (weight {LAMBDA_PENALTY:g}). Local anatomy scores are "
                         f"optimistic by exactly that term.")

        topo = f"topology_violation={self.topology_violation_rate * 100:.2f}% of columns"
        if self.topology_stride != 1:
            topo += f" (sampled every {self.topology_stride} columns)"
        lines.append(f"n_images={self.n_images}  mean_image_score={self.mean_image_score:.4f}"
                     f"  {topo}")
        lines.append("cls    dice   masd/H    score    sup   nMASD    f-inv   missed")
        for c in range(NUM_CLASSES):
            lines.append(
                f"{c:>3}  {_num(self.per_class_dice[c]):>6}  {_num(self.per_class_masd[c], 5):>7}"
                f"  {_num(self.per_class_score[c]):>6}  {self.class_support[c]:>5}"
                f"  {self.masd_support[c]:>5}   {_num(self.false_invention_rate.get(c)):>6}"
                f"   {_num(self.missed_class_rate.get(c)):>6}"
            )

        lines.append(_split_line("status ", _STATUSES, self.by_status, self.counts_by_status))
        lines.append(_split_line("anatomy", _ANATOMIES, self.by_anatomy, self.counts_by_anatomy))
        lines.append(_split_line("vendor ", sorted(self.by_vendor), self.by_vendor,
                                 self.counts_by_vendor))
        lines.append("geometry " + (
            "  ".join(f"{k}={v}" for k, v in self.geometry.items()) or "not recorded"))

        anat = "  ".join(
            f"{a} {self.challenge.anatomy_scores.get(a, 0.0):.4f}"
            f"(pen {self.challenge.penalties.get(a, 0.0):.4f})" for a in _ANATOMIES
        )
        lines.append(f"official: {anat}   FINAL challenge_score = {self.challenge_score:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable. Non-finite floats become ``None`` so strict parsers survive."""
        return _jsonable({
            "challenge_score": self.challenge_score,
            "mean_image_score": self.mean_image_score,
            "n_images": self.n_images,
            "per_class_dice": self.per_class_dice,
            "per_class_masd": self.per_class_masd,
            "per_class_score": self.per_class_score,
            "class_support": self.class_support,
            "masd_support": self.masd_support,
            "class_absence": self.class_absence,
            "topology_violation_rate": self.topology_violation_rate,
            "topology_stride": self.topology_stride,
            "false_invention_rate": self.false_invention_rate,
            "missed_class_rate": self.missed_class_rate,
            "by_status": self.by_status,
            "by_anatomy": self.by_anatomy,
            "by_vendor": self.by_vendor,
            "counts_by_status": self.counts_by_status,
            "counts_by_anatomy": self.counts_by_anatomy,
            "counts_by_vendor": self.counts_by_vendor,
            "missing_cells": self.missing_cells,
            "coverage_notes": self.coverage_notes,
            "geometry": self.geometry,
            "per_image": self.per_image,
    #: False on every local run: no Triton data is released, so the penalty is 0.0 here.
            "unseen_vendor_penalty_live": "Triton" in self.counts_by_vendor,
            "challenge": {
                "final_score": self.challenge.final_score,
                "anatomy_scores": self.challenge.anatomy_scores,
                "penalties": self.challenge.penalties,
                "vendor_scores": {str(k): v for k, v in self.challenge.vendor_scores.items()},
                "cohort_scores": {str(k): v for k, v in self.challenge.cohort_scores.items()},
            },
        })


def _num(value: float | None, decimals: int = 4) -> str:
    """Empty denominators print as ``--`` rather than a misleading 0.0000."""
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def _split_line(label: str, keys: Sequence[str], scores: Mapping[str, float],
                counts: Mapping[str, int]) -> str:
    parts = [f"{k} {scores[k]:.4f}(n={counts.get(k, 0)})" for k in keys if k in scores]
    return f"{label}: " + ("  ".join(parts) if parts else "none")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


def _vendor_of(meta: Mapping[str, Any]) -> str:
    """Vendor string the official CSV uses; falls back to mapping the device directory."""
    vendor = meta.get("vendor")
    if vendor:
        return str(vendor)
    device = str(meta.get("device", ""))
    from octtta.data.release_dataset import DEVICE_TO_VENDOR

    return DEVICE_TO_VENDOR.get(device, device or "unknown")


def _check_labels(arr: np.ndarray, what: str) -> np.ndarray:
    """A single out-of-range pixel crashes the official evaluator -- catch it here, loudly."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{what} must be (H, W), got {arr.shape}")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi >= NUM_CLASSES:
        raise ValueError(f"{what} has labels outside [0, {NUM_CLASSES - 1}]: min={lo} max={hi}")
    return arr


def column_topology_violations_fast(labels: np.ndarray, stride: int = 1) -> dict:
    """Vectorised twin of :func:`octtta.postproc.column_topology_violations`.
    Identical definitions with no Python loop; ``stride`` subsamples columns and says so."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a (H, W) label map, got {labels.shape}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > 1:
        labels = labels[:, ::stride]

    H, W = labels.shape
    if W == 0 or H == 0:
        return {"n_columns": 0, "unordered_columns": 0, "fragmented_columns": 0,
                "unordered_frac": 0.0, "fragmented_frac": 0.0, "stride": stride}

    from octtta.postproc import column_flags

    unordered, fragmented = column_flags(labels)

    n_unordered = int(unordered.sum())
    n_fragmented = int(fragmented.sum())
    return {
        "n_columns": W,
        "unordered_columns": n_unordered,
        "fragmented_columns": n_fragmented,
        "unordered_frac": n_unordered / W,
        "fragmented_frac": n_fragmented / W,
        "stride": stride,
    }


def _missing_cell_notes(records: Sequence[ScoreRecord]) -> tuple[list[str], list[str]]:
    """Split eval-set gaps into the fatal ones and the merely incomparable ones.
    Fatal = the official aggregation substitutes 0.0 for a missing anatomy or cohort."""
    present: set[tuple[str, str, str]] = {(r.anatomy, r.device, r.status) for r in records}
    fatal: list[str] = []
    coverage: list[str] = []
    for anatomy in _ANATOMIES:
        vendors = sorted({v for a, v, _ in present if a == anatomy})
        if not vendors:
            fatal.append(f"no {anatomy} images at all -> 0.5 of FINAL is 0.0 by construction")
            continue
        for vendor in vendors:
            for status in _STATUSES:
                if (anatomy, vendor, status) not in present:
                    weight = ALPHA if status == "healthy" else 1.0 - ALPHA
                    fatal.append(f"{anatomy}/{vendor}: no {status} -> {weight:g} of that "
                                 "vendor's weight scores 0.0")
        absent_seen = [v for v in SEEN_VENDORS if v not in vendors]
        if absent_seen:
            coverage.append(f"{anatomy} has no {'/'.join(absent_seen)}")
    # The missing unseen vendor is not listed here: ``EvalReport.format`` states it always.
    return fatal, coverage


def accumulate(preds_gts_meta: Iterable[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]],
               *, topology_stride: int = 1) -> EvalReport:
    """Score and fold image triples through the same path as parallel validation."""
    return fold_breakdowns(
        (_score_one((None, pred, gt, meta, None), topology_stride=topology_stride)
         for pred, gt, meta in preds_gts_meta),
        topology_stride=topology_stride,
    )


def _as_hw_pairs(value: Any, batch_size: int) -> list[tuple[int, int]] | None:
    """Accept the several shapes a collate function can turn ``(H, W)`` metadata into.
    The default collate turns a list of tuples into a tuple of tensors."""
    if value is None:
        return None
    try:
        seq = list(value)
    except TypeError:
        return None

    def _sized(v: Any, n: int) -> bool:
        try:
            return len(v) == n
        except TypeError:
            return False

    if len(seq) == batch_size and all(_sized(v, 2) for v in seq):
        return [(int(v[0]), int(v[1])) for v in seq]
    # Ambiguous when batch_size == 2; the documented list-of-pairs form wins above.
    if len(seq) == 2 and all(_sized(v, batch_size) for v in seq):
        return [(int(seq[0][i]), int(seq[1][i])) for i in range(batch_size)]
    return None


def _valid_hw(mask: np.ndarray, hint: tuple[int, int] | None) -> tuple[int, int]:
    """Region of a padded sample that is real data.
    Padding is assumed anchored top-left, which is what ``collate_pad`` does."""
    H, W = mask.shape
    if hint is not None:
        h, w = hint
        if 0 < h <= H and 0 < w <= W and not (mask[:h, :w] < 0).any():
            return h, w
    valid = mask >= 0
    if not valid.any():
        raise ValueError("sample is entirely padding / unlabelled -- cannot be evaluated")
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    return int(rows[-1]) + 1, int(cols[-1]) + 1


def deployment_plan(plan=None):
    """Normalise a caller's plan into the one this module is allowed to evaluate under.
    ``postproc`` is cleared because every function here applies the block itself.
    ``plan=None`` resolves to the shipped defaults, not whole-image."""
    from dataclasses import replace

    from octtta.infer import Plan

    return replace(plan if plan is not None else Plan(), postproc=None)


def evaluate_model(model, loader, *, postproc_cfg: dict | None, device,
                   amp: bool = True,
                   topology_stride: int = 1, plan=None,
                   model_b=None, plan_b=None, fusion=None) -> EvalReport:
    """Run ``model`` over ``loader``, post-process, and accumulate the metrics.
    This runs the submission's inference path, per image at its native unpadded size, so
    padding never reaches the model and a score cannot depend on a sample's batch-mates."""
    geometry: Counter[str] = Counter()
    it = iter_predictions(model, loader, postproc_cfg=postproc_cfg, device=device,
                          amp=amp, plan=plan, geometry=geometry,
                          model_b=model_b, plan_b=plan_b, fusion=fusion)
    try:
        report = accumulate(it, topology_stride=topology_stride)
        report.geometry = dict(sorted(geometry.items()))
        return report
    finally:
        it.close()


def iter_predictions(model, loader, *, postproc_cfg: dict | None, device,
                     amp: bool = True, plan=None,
                     geometry: Counter | None = None,
                     model_b=None, plan_b=None, fusion=None):
    """Yield ``(pred, gt, meta)`` per image through the submission's inference path.
    The single owner of the deploy-path evaluation loop: a second copy of it is how the
    whole-image shortcut happened. ``geometry``, when given, counts the geometries used."""
    import torch

    from octtta.infer import labels_for_image

    if (model_b is None) != (plan_b is None and fusion is None):
        raise ValueError("model_b, plan_b and fusion travel together "
                         "(octtta.infer.load_fusion_partner resolves all three)")
    # No local "postproc or bare argmax" branch: a second implementation of the label step
    # is what let the deploy path and a probe disagree about WHERE post-processing runs.
    dev = torch.device(device)
    plan = deployment_plan(plan)
    if plan_b is not None:
        plan_b = deployment_plan(plan_b)

    was_training = model.training
    was_training_b = model_b.training if model_b is not None else False
    model.eval()
    if model_b is not None:
        model_b.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                for image, gt, meta in _iter_native_samples(batch):
                    info: dict = {}
                    pred = labels_for_image(model, image, plan, postproc=postproc_cfg,
                                            model_b=model_b, plan_b=plan_b, fusion=fusion,
                                            device=dev, amp=bool(amp), info=info)
                    if geometry is not None:
                        geometry[str(info.get("geometry", "?"))] += 1
                    # A model that pads to a stride multiple and forgets to crop back returns
                    # a larger map, and slicing [:h, :w] would score a mis-anchored prediction.
                    if pred.shape[-2:] != image.shape[-2:]:
                        raise ValueError(
                            f"label map spatial size {tuple(pred.shape[-2:])} != input "
                            f"spatial size {tuple(image.shape[-2:])}. The model must crop "
                            "back to the input size; cropping here would score a "
                            "mis-anchored map.")
                    yield np.asarray(pred), gt, meta
    finally:
        if was_training:
            model.train()
        if was_training_b:
            model_b.train()


def _iter_native_samples(batch: Mapping[str, Any]):
    """One collated batch -> ``(image (H, W) float32, gt (H, W) uint8, meta)`` per sample.
    The image is already normalised, so slicing the pad off leaves the array the submission
    path would build. One channel only: the multi-view stack is built downstream, not here."""
    images = batch["image"]
    masks = batch["mask"].cpu().numpy()
    if images.shape[1] != 1:
        raise ValueError(
            f"expected single-channel images, got {images.shape[1]} channels; the "
            "submission path feeds one FRAME per image to octtta.infer.probs_for_image, "
            "which builds data.input_channels' view stack itself (D22 + s35). A val "
            "dataset built with input_channels= would have the views applied twice; only "
            "the train side takes that argument (octtta.data.dataset.build_datasets)")
    if masks.shape[-2:] != tuple(images.shape[-2:]):
        raise ValueError(
            f"mask spatial size {tuple(masks.shape[-2:])} != image spatial size "
            f"{tuple(images.shape[-2:])}")

    b = int(images.shape[0])
    hints = _as_hw_pairs(batch.get("orig_hw"), b) or _as_hw_pairs(batch.get("pad_hw"), b)
    for i in range(b):
        h, w = _valid_hw(masks[i], hints[i] if hints else None)
        image = np.ascontiguousarray(
            images[i, 0, :h, :w].detach().to("cpu", copy=False).float().numpy())
        yield image, masks[i][:h, :w].astype(np.uint8), _meta_at(batch, i)


def _meta_at(batch: Mapping[str, Any], i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("vendor", "status", "anatomy", "device", "stem"):
        value = batch.get(key)
        if value is None:
            continue
        out[key] = value[i] if isinstance(value, (list, tuple)) else value
    return out


# ---- server validation helpers -------------------------------------------------

class _Subset(torch.utils.data.Dataset):
    """A fixed index selection, preserving the dict interface. See :func:`_limit`."""

    def __init__(self, base: torch.utils.data.Dataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        return self.base[self.indices[i]]


def cohort_keys(dataset: torch.utils.data.Dataset) -> list[tuple[str, str, str]]:
    """``(vendor, status, anatomy)`` per index, read from metadata without loading pixels.

    A missing cell scores 0.0 officially, so emptying one by truncation changes the score."""
    if isinstance(dataset, _Subset):
        base = cohort_keys(dataset.base)
        return [base[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        out: list[tuple[str, str, str]] = []
        for ds in dataset.datasets:
            out.extend(cohort_keys(ds))
        return out
    if isinstance(dataset, PseudoWideFieldDataset):
        return [(s.vendor, s.status.lower(), "WideField") for s in dataset.base.samples]
    if isinstance(dataset, OCTSegDataset):
        return [(s.vendor, s.status.lower(), "Macula") for s in dataset.samples]
    return [("unknown", "unknown", "unknown")] * len(dataset)            # type: ignore[arg-type]


def stratified_indices(keys: Sequence[tuple], max_images: int | None) -> list[int]:
    """Round-robin over cohorts, so truncation thins every cell instead of dropping one."""
    n = len(keys)
    if not max_images or n <= int(max_images):
        return list(range(n))
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    order = [groups[k] for k in sorted(groups, key=lambda k: tuple(map(str, k)))]

    picked: list[int] = []
    depth = 0
    while len(picked) < int(max_images):
        progressed = False
        for bucket in order:
            if depth < len(bucket):
                picked.append(bucket[depth])
                progressed = True
                if len(picked) >= int(max_images):
                    break
        if not progressed:
            break
        depth += 1
    return sorted(picked)


def limit_stratified(dataset: torch.utils.data.Dataset,
                     max_images: int | None) -> torch.utils.data.Dataset:
    """:func:`stratified_indices` applied to a dataset."""
    idx = stratified_indices(cohort_keys(dataset), max_images)
    if len(idx) == len(dataset):                                         # type: ignore[arg-type]
        return dataset
    return _Subset(dataset, idx)


def _score_one(task: tuple, *, topology_stride: int = 1) -> dict:
    """One image's contribution to the report; pure CPU and poolable."""
    prob, pred, gt, meta, postproc_cfg = task
    if pred is None:
        from octtta.postproc import postprocess

        pred = np.asarray(postprocess(prob, postproc_cfg))
    pred = _check_labels(pred, "prediction")                     # noqa: SLF001
    gt = _check_labels(gt, "ground truth")                       # noqa: SLF001
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    bd = compute_image_breakdown(pred, gt)
    return {
        "dice": bd["dice"],
        "masd_norm": bd["masd_norm"],
        "layer_score": bd["layer_score"],
        "image_score": float(bd["image_score"]),
        "in_gt": np.bincount(gt.ravel(), minlength=NUM_CLASSES) > 0,
        "in_pred": np.bincount(pred.ravel(), minlength=NUM_CLASSES) > 0,
        "topology": float(column_topology_violations_fast(pred, topology_stride)["unordered_frac"]),
        "meta": meta,
    }


def fold_breakdowns(items: Iterable[dict], *, topology_stride: int = 1) -> EvalReport:
    """Fold :func:`_score_one` results into an :class:`EvalReport`, the same arithmetic as
    :func:`octtta.eval.report.accumulate` (a parity test pins the two together)."""
    dice_sum = np.zeros(NUM_CLASSES)
    masd_sum = np.zeros(NUM_CLASSES)
    score_sum = np.zeros(NUM_CLASSES)
    support = np.zeros(NUM_CLASSES, dtype=int)
    masd_n = np.zeros(NUM_CLASSES, dtype=int)
    absence = np.zeros(NUM_CLASSES, dtype=int)
    invented = np.zeros(NUM_CLASSES, dtype=int)
    missed = np.zeros(NUM_CLASSES, dtype=int)

    records: list[ScoreRecord] = []
    per_image: list[dict] = []
    topo_sum = 0.0
    group_sums: dict[str, dict[str, list[float]]] = {"status": {}, "anatomy": {}, "vendor": {}}
    image_scores: list[float] = []

    for r in items:
        in_gt, in_pred = r["in_gt"], r["in_pred"]
        support += in_gt
        absence += ~in_gt
        invented += (~in_gt) & in_pred
        missed += in_gt & (~in_pred)

        dice_sum[in_gt] += r["dice"][in_gt]
        score_sum[in_gt] += r["layer_score"][in_gt]
        finite = in_gt & np.isfinite(r["masd_norm"])
        masd_sum[finite] += r["masd_norm"][finite]
        masd_n += finite

        topo_sum += r["topology"]

        meta = r["meta"]
        status = str(meta.get("status", "")).lower()
        anatomy = normalize_anatomy(meta.get("anatomy", "Macula"))
        vendor = _vendor_of(meta)                                # noqa: SLF001
        score = float(r["image_score"])
        image_scores.append(score)
        records.append(ScoreRecord(image_score=score, device=vendor, status=status,
                                   anatomy=anatomy))
        per_image.append({"id": f"{meta.get('stem', f'idx{len(records) - 1}')}|{anatomy}",
                          "score": score, "vendor": vendor, "status": status,
                          "anatomy": anatomy})
        for key, value in (("status", status), ("anatomy", anatomy), ("vendor", vendor)):
            group_sums[key].setdefault(value, []).append(score)

    n = len(records)
    if n == 0:
        raise ValueError("no images scored -- an empty val set scores 0.0 silently")

    challenge = aggregate(records)
    fatal_gaps, coverage_gaps = _missing_cell_notes(records)     # noqa: SLF001
    with np.errstate(invalid="ignore", divide="ignore"):
        per_dice = np.where(support > 0, dice_sum / np.maximum(support, 1), np.nan)
        per_masd = np.where(masd_n > 0, masd_sum / np.maximum(masd_n, 1), np.nan)
        per_score = np.where(support > 0, score_sum / np.maximum(support, 1), np.nan)

    return EvalReport(
        challenge_score=float(challenge.final_score),
        per_class_dice=per_dice,
        per_class_masd=per_masd,
        per_class_score=per_score,
        topology_violation_rate=topo_sum / n,
        false_invention_rate={c: (float(invented[c] / absence[c]) if absence[c] else float("nan"))
                              for c in range(NUM_CLASSES)},
        missed_class_rate={c: (float(missed[c] / support[c]) if support[c] else float("nan"))
                           for c in range(NUM_CLASSES)},
        by_status={k: float(np.mean(v)) for k, v in group_sums["status"].items()},
        by_anatomy={k: float(np.mean(v)) for k, v in group_sums["anatomy"].items()},
        by_vendor={k: float(np.mean(v)) for k, v in group_sums["vendor"].items()},
        n_images=n,
        challenge=challenge,
        topology_stride=int(topology_stride),
        class_support=support,
        masd_support=masd_n,
        class_absence=absence,
        mean_image_score=float(np.mean(image_scores)),
        missing_cells=fatal_gaps,
        coverage_notes=coverage_gaps,
        counts_by_status={k: len(v) for k, v in group_sums["status"].items()},
        counts_by_anatomy={k: len(v) for k, v in group_sums["anatomy"].items()},
        counts_by_vendor={k: len(v) for k, v in group_sums["vendor"].items()},
        per_image=per_image,
    )


def _iter_scoring_tasks(model, loader, *, postproc_cfg: dict | None, device,
                        amp: bool, plan=None) -> Iterator[tuple]:
    """GPU forward pass -> one poolable task per image.

    Widened frames are post-processed here instead: the gate does not commute with a resample."""
    from octtta.infer import labels_for_image, probs_for_image, work_width

    dev = torch.device(device)
    plan = deployment_plan(plan)
    with torch.no_grad():
        for batch in loader:
            for image, gt, meta in _iter_native_samples(batch):   # noqa: SLF001
                hw = (int(image.shape[-2]), int(image.shape[-1]))
                if postproc_cfg is not None and work_width(hw, plan) == hw[1]:
                    crop = np.ascontiguousarray(
                        probs_for_image(model, image, plan, device=dev, amp=bool(amp)))
                    yield (crop, None, gt, meta, postproc_cfg)
                else:
                    pred = labels_for_image(model, image, plan, postproc=postproc_cfg,
                                            device=dev, amp=bool(amp))
                    yield (None, pred, gt, meta, None)


def evaluate_parallel(model, loader, *, postproc_cfg: dict | None, device,
                      amp: bool = True,
                      pool: WorkerPool | None = None, plan=None,
                      ) -> EvalReport:
    """Same contract as :func:`octtta.eval.report.evaluate_model`, fanned out over CPUs."""
    if pool is None or pool.workers <= 1:
        return evaluate_model(model, loader, postproc_cfg=postproc_cfg,
                                      device=device, amp=amp, plan=plan,
                                      )

    was_training = model.training
    model.eval()
    try:
        tasks = _iter_scoring_tasks(model, loader, postproc_cfg=postproc_cfg,
                                    device=device, amp=amp, plan=plan,
                                    )
        try:
            return fold_breakdowns(pool.imap(_score_one, tasks))
        except Exception as exc:                                         # noqa: BLE001
            # A dead pool costs throughput, not the validation; a real bug fails again serially.
            print(f"[eval] worker pool failed ({exc!r}); falling back to serial scoring",
                  flush=True)
            pool.close()
            pool.workers = 1
            return evaluate_model(model, loader, postproc_cfg=postproc_cfg,
                                          device=device, amp=amp, plan=plan,
                                          )
    finally:
        if was_training:
            model.train()
