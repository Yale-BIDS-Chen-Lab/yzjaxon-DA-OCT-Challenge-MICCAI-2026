"""Dual-teacher self-training: the pure, testable half.

Two students teach each other on unlabelled frames. A pseudo-label is weighted rather than
accepted or rejected outright, and the weight comes from one of two gates: ``column`` ranks
whole columns by how much the two teachers agree and how confident the chosen teacher is,
and ``dmt`` decides per (boundary, column) with an asymmetric rule that lets a confident
teacher correct a disagreeing student. Everything here is pure: the training loop in
:mod:`octtta.train` owns the models, the optimiser and the EMA shadows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


__all__ = ["CoTeachSpec", "coteach_spec", "COTEACH_KEYS", "COTEACH_TEACHERS",
           "soft_column_weights", "weighted_pixel_ce", "column_structure",
           "column_agreement",
           "COTEACH_GATES", "DMT_KEYS", "DmtSpec", "DmtUnits", "dmt_spec",
           "boundary_rows_torch",
           "dmt_units", "dmt_boundary_weights", "boundary_weights_to_pixels",
           "ramp_factor",
           "pseudo_pool_flag", "FULL_PASS_REMAINDER_KEYS", "FullPassRemainder",
           "full_pass_remainder_spec", "COTEACH_KEYS_OPTIONAL", "COTEACH_KEYS_REQUIRED"]

COTEACH_KEYS = frozenset({"enabled", "weight", "every", "batch_size", "quantile", "weight_floor",
                          "structure_discount", "target", "include_maestro2_unlabeled",
                          "include_release_unlabeled", "assert_full_pass",
                          "full_pass_remainder", "lr_b",
                          "teacher", "teacher_decay", "agree_min", "hard_shapes",
                          "scope_b", "freeze_norm_affine_b",
                          "gate", "dmt", "weight_ramp_steps",
                          "teacher_temp_a", "teacher_temp_b"})

#: Keys of :data:`COTEACH_KEYS` that may be missing from an older package, resolving to a stated default.
COTEACH_KEYS_OPTIONAL = frozenset({"full_pass_remainder"})

#: What a shipped ``train.coteach`` block must STATE; derived from ``COTEACH_KEYS``, never re-typed.
COTEACH_KEYS_REQUIRED = COTEACH_KEYS - COTEACH_KEYS_OPTIONAL

#: ``column`` is the symmetric per-column quality ramp, ``dmt`` the asymmetric per-unit gate.
COTEACH_GATES: tuple[str, ...] = ("column", "dmt")

#: Every key of ``train.coteach.dmt``; all are required, since none has a defensible default.
DMT_KEYS = frozenset({"delta_px", "tau_high", "tau_mid", "delta_q", "gamma", "band_rows",
                      "disqualify_crossing", "thickness"})

#: Every key of ``train.*.full_pass_remainder``; both required when the block is present.
FULL_PASS_REMAINDER_KEYS = frozenset({"allow", "max_frames"})

COTEACH_TEACHERS: tuple[str, ...] = ("ema",)


@dataclass(frozen=True)
class DmtSpec:
    """``train.coteach.dmt`` resolved. Every field is stated by the config, none defaults.
    ``thickness`` is deliberately absent: :func:`dmt_spec` refuses any non-null value, so a
    config asking for it gets an error instead of a silently ignored knob."""
    delta_px: float             #: how many ROWS apart the two teachers may be and still "agree"
    tau_high: float             #: reliability the teacher needs when the two agree
    tau_mid: float              #: reliability the teacher needs when they disagree
    delta_q: float              #: reliability MARGIN over the other model, when they disagree
    gamma: float                #: exponent on the student-agreement factor
    band_rows: int              #: half-height of the row band q and s are averaged over
    disqualify_crossing: bool   #: a non-monotone teacher column teaches nothing at all


@dataclass(frozen=True)
class FullPassRemainder:
    """``train.*.full_pass_remainder`` resolved; ``None`` is the strict behaviour, refusing
    a pool that is not a whole number of unlabelled batches. ``max_frames`` bounds the
    remainder a recipe accepts, so the frames dropped every epoch are stated and printed."""
    allow: bool
    max_frames: int      #: the largest remainder this recipe accepts; 0 when allow is False


@dataclass(frozen=True)
class CoTeachSpec:
    weight: float
    every: int                    #: teach every N-th supervised step
    batch_size: int
    quantile: float               #: fraction of each image's columns that teach at full weight
    weight_floor: float           #: no column teaches below this weight
    structure_discount: float     #: multiplier for a column whose target stack is not legal
    target: str                   #: "cross" (the other teacher)
    include_maestro2_unlabeled: bool
    lr_b: float | None = None
    teacher: str = "ema"
    gate: str = "column"
    dmt: DmtSpec | None = None
    weight_ramp_steps: int = 0
    #: Whether the release's own unlabelled vendor directories join the pool.
    include_release_unlabeled: bool = True
    #: The run claims one full pass over the unlabelled pool per epoch; turns the check on.
    assert_full_pass: bool = False
    #: How large a remainder the epoch may be rounded down by; ``None`` is the strict default.
    full_pass_remainder: FullPassRemainder | None = None

    def __repr__(self) -> str:
        extra = ""
        if self.gate != "column":
            extra += f", gate={self.gate}, dmt={self.dmt}"
        if self.weight_ramp_steps:
            extra += f", ramp={self.weight_ramp_steps}"
        if self.full_pass_remainder is not None and self.full_pass_remainder.allow:
            extra += f", remainder<={self.full_pass_remainder.max_frames}"
        return (f"CoTeachSpec(w={self.weight:g}, every={self.every}, bs={self.batch_size}, "
                f"q={self.quantile:g}, floor={self.weight_floor:g}, "
                f"struct={self.structure_discount:g}, target={self.target}{extra})")






def pseudo_pool_flag(block: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """One boolean switch of the pseudo-label pool, or a refusal naming the key."""
    raw = block.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"train.*.{key} must be a boolean (true/false), got {raw!r} "
                         f"({type(raw).__name__}); a quoted 'false' is truthy in Python "
                         f"and would choose the opposite pool without any line disagreeing")
    return raw


def full_pass_remainder_spec(block: Mapping[str, Any],
                             *, where: str = "train.coteach") -> FullPassRemainder | None:
    """``train.*.full_pass_remainder`` resolved, or ``None`` for the strict default."""
    raw = block.get("full_pass_remainder", None)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, Mapping):
        raise ValueError(
            f"{where}.full_pass_remainder must be a mapping with keys "
            f"{sorted(FULL_PASS_REMAINDER_KEYS)} (or null for the strict default), got "
            f"{raw!r} ({type(raw).__name__})")
    unknown = set(raw) - FULL_PASS_REMAINDER_KEYS
    if unknown:
        raise ValueError(f"{where}.full_pass_remainder: unknown key(s) {sorted(unknown)}; "
                         f"known: {sorted(FULL_PASS_REMAINDER_KEYS)}")
    missing = sorted(FULL_PASS_REMAINDER_KEYS - set(raw))
    if missing:
        raise ValueError(
            f"{where}.full_pass_remainder is stated but does not name {missing}. Both keys "
            f"are required: an absent 'allow' would read as a code default deciding whether "
            f"frames are dropped, and an absent 'max_frames' would make the bound the "
            f"dataclass's signature rather than the recipe's choice")
    allow = raw["allow"]
    if not isinstance(allow, bool):
        raise ValueError(
            f"{where}.full_pass_remainder.allow must be a boolean (true/false), got "
            f"{allow!r} ({type(allow).__name__}); a quoted 'false' is truthy in Python and "
            f"would drop frames from every epoch while the config's own text said it would "
            f"not")
    max_frames = raw["max_frames"]
    if isinstance(max_frames, bool) or not isinstance(max_frames, int):
        raise ValueError(
            f"{where}.full_pass_remainder.max_frames must be a non-negative integer, got "
            f"{max_frames!r} ({type(max_frames).__name__})")
    if max_frames < 0:
        raise ValueError(
            f"{where}.full_pass_remainder.max_frames must be a non-negative integer, got "
            f"{max_frames}")
    if not allow and max_frames:
        raise ValueError(
            f"{where}.full_pass_remainder states allow=false with max_frames={max_frames}: "
            f"a bound that decides nothing, wearing the name of a decision. Write "
            f"max_frames: 0 (or null for the whole block) to mean 'refuse any remainder'")
    return FullPassRemainder(allow=allow, max_frames=int(max_frames))




def _unit(raw: Mapping[str, Any], key: str) -> float:
    """A ``dmt`` threshold that must lie in ``[0, 1]``, being compared to a softmax maximum."""
    val = float(raw[key])
    if not 0.0 <= val <= 1.0:
        raise ValueError(f"train.coteach.dmt.{key} must be in [0, 1], got {val}")
    return val


def dmt_spec(block: Mapping[str, Any]) -> DmtSpec | None:
    """``train.coteach.dmt`` -> :class:`DmtSpec`, or ``None`` when the key says ``null``."""
    raw = block.get("dmt")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"train.coteach.dmt must be a mapping or null, got "
                         f"{type(raw).__name__}")
    unknown = sorted(set(raw) - DMT_KEYS)
    if unknown:
        raise ValueError(f"train.coteach.dmt: unknown key(s) {unknown}; "
                         f"known: {sorted(DMT_KEYS)}")
    missing = sorted(DMT_KEYS - set(raw))
    if missing:
        raise ValueError(f"train.coteach.dmt does not state {missing}; every threshold of "
                         f"this gate has to be a number somebody chose, not a code default "
                         f"nobody can read back out of the run's config snapshot")
    if raw["thickness"] is not None:
        raise ValueError(
            "train.coteach.dmt.thickness is reserved and not implemented in v1: a "
            "per-layer thickness predicate needs a thickness prior this gate does not "
            "have (D39 forbids the surface/thickness branch for the E07 lineage). The key "
            "is known so that asking for it is an error rather than a silently ignored "
            f"knob; state it as null. Got {raw['thickness']!r}")
    delta_px = float(raw["delta_px"])
    if not delta_px >= 0.0:
        raise ValueError(f"train.coteach.dmt.delta_px is a row distance and must be >= 0, "
                         f"got {delta_px}")
    tau_high, tau_mid, delta_q = (_unit(raw, "tau_high"), _unit(raw, "tau_mid"),
                                  _unit(raw, "delta_q"))
    if tau_mid > tau_high:
    # Rule (ii) already carries an extra margin, so tau_mid above tau_high would make the correcting rule stricter on both axes.
        raise ValueError(f"train.coteach.dmt.tau_mid ({tau_mid}) is above tau_high "
                         f"({tau_high}); rule (ii) already demands delta_q on top of it, so "
                         f"this makes the only rule that can correct a student the hardest "
                         f"one to satisfy")
    gamma = float(raw["gamma"])
    if not gamma >= 0.0:
        raise ValueError(f"train.coteach.dmt.gamma must be >= 0, got {gamma}")
    band = raw["band_rows"]
    if isinstance(band, bool) or not isinstance(band, int) or band < 0:
        raise ValueError(f"train.coteach.dmt.band_rows must be a non-negative integer "
                         f"(0 = the boundary row alone), got {band!r}")
    cross = raw["disqualify_crossing"]
    if not isinstance(cross, bool):
    # ``"false"`` is truthy, and a YAML quoting accident would switch the rule ON everywhere.
        raise ValueError(f"train.coteach.dmt.disqualify_crossing must be a boolean, got "
                         f"{cross!r} ({type(cross).__name__})")
    return DmtSpec(delta_px=delta_px, tau_high=tau_high, tau_mid=tau_mid, delta_q=delta_q,
                   gamma=gamma, band_rows=int(band), disqualify_crossing=bool(cross))


def coteach_spec(cfg: Mapping[str, Any] | None) -> CoTeachSpec | None:
    """``train.coteach`` resolved, or ``None`` when absent / disabled. Unknown keys are fatal."""
    block = dict(((cfg or {}).get("train") or {}).get("coteach") or {})
    if not block or not bool(block.get("enabled", False)):
        return None
    unknown = set(block) - COTEACH_KEYS
    if unknown:
        raise ValueError(f"train.coteach: unknown key(s) {sorted(unknown)}; known: {sorted(COTEACH_KEYS)}")
    q = float(block.get("quantile", 0.5))
    floor = float(block.get("weight_floor", 0.3))
    disc = float(block.get("structure_discount", 0.3))
    target = str(block.get("target", "cross"))
    if not 0.0 < q <= 1.0:
        raise ValueError(f"train.coteach.quantile must be in (0, 1], got {q}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"train.coteach.weight_floor must be in [0, 1], got {floor}")
    if not 0.0 <= disc <= 1.0:
        raise ValueError(f"train.coteach.structure_discount must be in [0, 1], got {disc}")
    if target != "cross":
        raise ValueError(f"train.coteach.target must be 'cross', got {target!r}")
    teacher = str(block.get("teacher", "ema"))
    if teacher not in COTEACH_TEACHERS:
        raise ValueError(f"train.coteach.teacher must be one of {list(COTEACH_TEACHERS)}, "
                         f"got {teacher!r}")
    for key in ("teacher_decay", "agree_min", "hard_shapes"):
        if block.get(key) is not None:
            raise ValueError(f"train.coteach.{key} must be null in Final")
    scope_b = str(block.get("scope_b", "full"))
    if scope_b != "full":
        raise ValueError(f"train.coteach.scope_b must be 'full', got {scope_b!r}")
    if block.get("freeze_norm_affine_b", False) is not False:
        raise ValueError("train.coteach.freeze_norm_affine_b must be false")
    gate = str(block.get("gate", "column"))
    if gate not in COTEACH_GATES:
        raise ValueError(f"train.coteach.gate must be one of {list(COTEACH_GATES)}, "
                         f"got {gate!r}")
    dmt = dmt_spec(block)
    if gate == "dmt":
        if dmt is None:
            raise ValueError("train.coteach.gate='dmt' needs a train.coteach.dmt block; "
                             "its thresholds are the gate")
    elif dmt is not None:
        raise ValueError("train.coteach.dmt is stated but gate='column' never reads it: a "
                         "block of thresholds that decides nothing, wearing the name of a "
                         "decision. State gate='dmt' or dmt: null")
    ramp_steps = block.get("weight_ramp_steps", 0)
    if isinstance(ramp_steps, bool) or not isinstance(ramp_steps, int) or ramp_steps < 0:
        raise ValueError(f"train.coteach.weight_ramp_steps must be a non-negative integer "
                         f"(0 = no ramp), got {ramp_steps!r}")
    for key in ("teacher_temp_a", "teacher_temp_b"):
        if type(block.get(key, 1.0)) not in (int, float) or float(block.get(key, 1.0)) != 1.0:
            raise ValueError(f"train.coteach.{key} must be 1.0 in Final")
    include_release = pseudo_pool_flag(block, "include_release_unlabeled", default=True)
    full_pass = pseudo_pool_flag(block, "assert_full_pass", default=False)
    remainder = full_pass_remainder_spec(block, where="train.coteach")
    return CoTeachSpec(
        weight=float(block.get("weight", 0.5)),
        every=max(1, int(block.get("every", 2))),
        batch_size=max(1, int(block.get("batch_size", 4))),
        quantile=q, weight_floor=floor, structure_discount=disc, target=target,
        include_maestro2_unlabeled=bool(block.get("include_maestro2_unlabeled", True)),
        lr_b=(None if block.get("lr_b") is None else float(block["lr_b"])),
        teacher=teacher,
        gate=gate, dmt=dmt, weight_ramp_steps=int(ramp_steps),
        include_release_unlabeled=include_release,
        assert_full_pass=full_pass,
        full_pass_remainder=remainder,
    )


def ramp_factor(global_step: int, steps: int) -> float:
    """Linear warm-up of the co-teaching term: ``0`` at step 0, ``1`` from ``steps`` on."""
    if int(steps) <= 0:
        return 1.0
    return float(min(1.0, max(0.0, int(global_step) / float(int(steps)))))




def column_structure(pseudo: torch.Tensor, num_classes: int, *, check_span: bool) -> torch.Tensor:
    """``(B, H, W)`` argmax -> ``(B, W)`` bool: the column is a legal top-to-bottom stack."""
    if pseudo.ndim != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(pseudo.shape)}")
    legal = (pseudo[:, 1:, :] >= pseudo[:, :-1, :]).all(dim=1)
    if check_span:
        legal = legal & (pseudo[:, 0, :] == 0) & (pseudo[:, -1, :] == int(num_classes) - 1)
    return legal


def column_agreement(pseudo_a: torch.Tensor, pseudo_b: torch.Tensor,
                     valid: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, H, W)`` argmax x2 -> ``(B, W)``: the fraction of rows the two teachers agree on."""
    if pseudo_a.shape != pseudo_b.shape or pseudo_a.ndim != 3:
        raise ValueError(f"teacher argmax maps disagree: {tuple(pseudo_a.shape)} vs "
                         f"{tuple(pseudo_b.shape)}")
    agree = (pseudo_a == pseudo_b).float()
    if valid is None:
        return agree.mean(dim=1)
    v = valid.to(agree.dtype).squeeze(1)
    return (agree * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)


def soft_column_weights(pseudo_a: torch.Tensor, pseudo_b: torch.Tensor, target_prob: torch.Tensor,
                        *, quantile: float, floor: float, structure_discount: float,
                        valid: torch.Tensor | None = None,
                        check_span: bool) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-column teaching weights ``(B, W)`` in ``[floor, 1]``, plus a few logs.
    Columns are ranked WITHIN the image by agreement times confidence, so the ramp adapts
    to a frame the teachers find hard instead of muting it wholesale."""
    if pseudo_a.shape != pseudo_b.shape or pseudo_a.ndim != 3:
        raise ValueError(f"teacher argmax maps disagree: {tuple(pseudo_a.shape)} vs {tuple(pseudo_b.shape)}")
    if target_prob.ndim != 4 or target_prob.shape[0] != pseudo_a.shape[0] or target_prob.shape[-2:] != pseudo_a.shape[-2:]:
        raise ValueError(f"target_prob {tuple(target_prob.shape)} does not match {tuple(pseudo_a.shape)}")
    conf, target = target_prob.max(dim=1)
    agree_col = column_agreement(pseudo_a, pseudo_b, valid)
    if valid is not None:
        # float32, the dtype the agreement above masks with, so both halves weight alike.
        v = valid.to(torch.float32).squeeze(1)
        conf_col = (conf * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
    else:
        conf_col = conf.mean(dim=1)
    q = agree_col * conf_col
    # Rank within the image: the (1 - quantile) quantile is the ramp's floor, the best column its top.
    lo = torch.quantile(q.float(), 1.0 - float(quantile), dim=1, keepdim=True)
    hi = q.max(dim=1, keepdim=True).values
    span = hi - lo
    ramp = torch.where(span > 1e-6, ((q - lo) / span.clamp_min(1e-6)).clamp(0.0, 1.0),
                       torch.ones_like(q))
    scaled = torch.where(q >= lo - 1e-6, ramp, torch.zeros_like(q))
    w = float(floor) + (1.0 - float(floor)) * scaled
    logs = {"ct_agree": float(agree_col.mean()), "ct_conf": float(conf_col.mean())}
    legal = column_structure(target, int(target_prob.shape[1]), check_span=check_span)
    w = torch.where(legal, w, w * float(structure_discount))
    logs["ct_w"] = float(w.mean())
    logs["ct_legal"] = float(legal.float().mean())
    return w, logs


def weighted_pixel_ce(logits: torch.Tensor, target: torch.Tensor, col_w: torch.Tensor,
                      valid: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted cross-entropy against a hard target; the mean is over the weights, so an
    all-floor batch returns the floor-weighted mean rather than 0."""
    if logits.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(f"logits {tuple(logits.shape)} vs target {tuple(target.shape)}")
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    if col_w.ndim == 2:
        w = col_w[:, None, :].expand_as(ce).to(ce.dtype)
    elif col_w.ndim == 3:
        if tuple(col_w.shape) != tuple(ce.shape):
            raise ValueError(f"per-pixel weights {tuple(col_w.shape)} do not match the "
                             f"loss map {tuple(ce.shape)}")
        w = col_w.to(ce.dtype)
    else:
        raise ValueError(f"weights must be (B, W) or (B, H, W), got {tuple(col_w.shape)}")
    if valid is not None:
        w = w * valid.to(ce.dtype).squeeze(1)
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


def boundary_rows_torch(labels: torch.Tensor, num_classes: int = 10, *,
                        valid: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, H, W)`` argmax -> ``(rows, ok)``, both ``(B, K, W)`` with ``K = num_classes-1``.
    ``rows[b, k-1, x]`` is the first row at or below interface ``k``, by the ``>= k`` rule;
    ``ok`` is False where the column has no such interface and ``rows`` is then NaN."""
    if labels.ndim != 3:
        raise ValueError(f"expected (B, H, W) argmax, got {tuple(labels.shape)}")
    n_b = int(num_classes) - 1
    if n_b < 1:
        raise ValueError(f"num_classes must be >= 2 to have an interface, got {num_classes}")
    b, h, w = labels.shape
    dev = labels.device
    ks = torch.arange(1, n_b + 1, device=dev, dtype=labels.dtype).view(1, n_b, 1, 1)
    hit = labels[:, None, :, :] >= ks
    if valid is not None:
        hit = hit & valid.to(torch.bool).squeeze(1)[:, None, :, :]
    ok = hit.any(dim=2)
    first = hit & (hit.to(torch.int32).cumsum(dim=2) == 1)
    idx = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    rows = (first.to(torch.float32) * idx).sum(dim=2)
    return torch.where(ok, rows, torch.full_like(rows, float("nan"))), ok


def _band_mean(field: torch.Tensor, rows: torch.Tensor, band_rows: int,
               valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of ``field`` ``(B, H, W)`` over ``rows +- band_rows``, per ``(B, K, W)`` unit."""
    b, h, w = field.shape
    n_b = rows.shape[1]
    band = int(band_rows)
    off = torch.arange(-band, band + 1, device=field.device, dtype=rows.dtype)
    idx = rows[:, :, None, :] + off.view(1, 1, -1, 1)
    inb = torch.isfinite(idx) & (idx >= 0) & (idx <= h - 1)
    take = torch.nan_to_num(idx, nan=0.0).clamp(0, h - 1).round().long()
    got = field[:, None].expand(b, n_b, h, w).gather(2, take)
    if valid is not None:
        vb = valid.to(torch.bool).squeeze(1)[:, None].expand(b, n_b, h, w).gather(2, take)
        inb = inb & vb
    n = inb.sum(dim=2)
    out = (got * inb.to(got.dtype)).sum(dim=2) / n.clamp_min(1).to(got.dtype)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


@dataclass(frozen=True)
class DmtUnits:
    """One direction's per-``(k, x)`` verdict, before the student agreement scales it."""
    rows_t: torch.Tensor        #: (B, K, W) teacher's boundary rows, NaN where absent
    rows_s: torch.Tensor        #: (B, K, W) comparator's boundary rows
    ok: torch.Tensor            #: (B, K, W) bool: BOTH models place this interface here
    dist: torch.Tensor          #: (B, K, W) |rows_t - rows_s|, NaN where not ok
    q_t: torch.Tensor           #: (B, K, W) teacher reliability (uncalibrated max-prob)
    q_s: torch.Tensor           #: (B, K, W) comparator reliability
    lab_t: torch.Tensor         #: (B, H, W) teacher argmax -- the labels being taught
    crossing: torch.Tensor      #: (B, W) bool: the teacher's column is not monotone
    rule_i: torch.Tensor
    rule_ii: torch.Tensor
    rule_iii: torch.Tensor
    rule_iv: torch.Tensor


def dmt_units(teacher_prob: torch.Tensor, comparator_prob: torch.Tensor, *, spec: DmtSpec,
              num_classes: int | None = None,
              valid: torch.Tensor | None = None) -> DmtUnits:
    """The four-way verdict for the direction "``teacher_prob`` teaches, ``comparator_prob``
    is the student's own side"."""
    if teacher_prob.ndim != 4 or teacher_prob.shape != comparator_prob.shape:
        raise ValueError(f"teacher {tuple(teacher_prob.shape)} and comparator "
                         f"{tuple(comparator_prob.shape)} must be the same (B, C, H, W)")
    n_cls = int(teacher_prob.shape[1]) if num_classes is None else int(num_classes)
    conf_t, lab_t = teacher_prob.max(dim=1)
    conf_s, lab_s = comparator_prob.max(dim=1)
    rows_t, ok_t = boundary_rows_torch(lab_t, n_cls, valid=valid)
    rows_s, ok_s = boundary_rows_torch(lab_s, n_cls, valid=valid)
    ok = ok_t & ok_s
    dist = (rows_t - rows_s).abs()
    q_t = _band_mean(conf_t, rows_t, spec.band_rows, valid)
    q_s = _band_mean(conf_s, rows_s, spec.band_rows, valid)
    # ``check_span=False``: a column cut by the tile edge cannot span 0..9, so discounting it would judge the crop.
    crossing = ~column_structure(lab_t, n_cls, check_span=False)
    bad = (crossing[:, None, :].expand_as(ok) & ok if spec.disqualify_crossing
           else torch.zeros_like(ok))
    near = ok & (dist <= float(spec.delta_px))
    far = ok & (dist > float(spec.delta_px))
    rule_i = near & (q_t >= float(spec.tau_high)) & ~bad
    rule_ii = (far & ((q_t - q_s) >= float(spec.delta_q))
               & (q_t >= float(spec.tau_mid)) & ~bad)
    rule_iii = ok & ~bad & ~rule_i & ~rule_ii
    return DmtUnits(rows_t=rows_t, rows_s=rows_s, ok=ok, dist=dist, q_t=q_t, q_s=q_s,
                    lab_t=lab_t, crossing=crossing, rule_i=rule_i, rule_ii=rule_ii,
                    rule_iii=rule_iii, rule_iv=bad)


def dmt_boundary_weights(teacher_prob: torch.Tensor, comparator_prob: torch.Tensor,
                         student_prob: torch.Tensor, *, spec: DmtSpec,
                         num_classes: int | None = None,
                         valid: torch.Tensor | None = None,
                         info: dict | None = None
                         ) -> tuple[torch.Tensor, dict[str, float]]:
    """One direction's ``(B, K, W)`` teaching weights, plus its bucket read-out.
    ``student_prob`` is the taught student's probabilities on the DIRTY view, so the
    agreement factor measures what that student would have said under augmentation."""
    units = dmt_units(teacher_prob, comparator_prob, spec=spec, num_classes=num_classes,
                      valid=valid)
    # p_student(the teacher's own label) at every pixel, then averaged over the band.
    p_sel = student_prob.detach().gather(1, units.lab_t[:, None]).squeeze(1)
    agree = _band_mean(p_sel, units.rows_t, spec.band_rows, valid)
    agree = torch.nan_to_num(agree, nan=0.0).clamp(0.0, 1.0)
    accept = units.rule_i | units.rule_ii
    w = torch.where(accept, agree.pow(float(spec.gamma)), torch.zeros_like(agree))
    n = units.ok.sum()
    den = n.clamp_min(1).to(w.dtype)
    logs = {"rule_i": float(units.rule_i.sum() / den),
            "rule_ii": float(units.rule_ii.sum() / den),
            "rule_iii": float(units.rule_iii.sum() / den),
            "rule_iv": float(units.rule_iv.sum() / den),
            "n": float(n), "w": float(w.sum() / den)}
    if info is not None:
        info["units"] = units
        info["accept"] = accept
    return w, logs


def boundary_weights_to_pixels(w_kx: torch.Tensor, target: torch.Tensor,
                               num_classes: int | None = None) -> torch.Tensor:
    """``(B, K, W)`` per-interface weights -> ``(B, H, W)`` per-pixel weights: a pixel of
    class ``c`` takes ``min(w_c, w_{c+1})``, the two interfaces that bound it."""
    if w_kx.ndim != 3:
        raise ValueError(f"expected (B, K, W) boundary weights, got {tuple(w_kx.shape)}")
    b, n_b, w = w_kx.shape
    n_cls = n_b + 1 if num_classes is None else int(num_classes)
    if n_cls != n_b + 1:
        raise ValueError(f"{n_b} interfaces cannot bound {n_cls} classes")
    if target.ndim != 3 or target.shape[0] != b or target.shape[-1] != w:
        raise ValueError(f"target {tuple(target.shape)} does not match weights "
                         f"{tuple(w_kx.shape)}")
    h = int(target.shape[1])
    inf = torch.full((b, 1, w), float("inf"), dtype=w_kx.dtype, device=w_kx.device)
    # below[c] = w_{c+1} (class C-1 has none); above[c] = w_c (class 0 has none).
    below = torch.cat([w_kx, inf], dim=1)
    above = torch.cat([inf, w_kx], dim=1)
    per_class = torch.minimum(below, above)
    return per_class[:, :, None, :].expand(b, n_cls, h, w).gather(
        1, target[:, None]).squeeze(1)
