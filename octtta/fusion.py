"""Dual-model probability fusion: the one owner of "which model to trust, where".

Two networks ship in the package and their probability maps are mixed per interface, by a
table of nine weights per protocol shape (:func:`shape_class` decides the shape from pixel
dimensions alone). The mixture runs on the cumulative distribution ``F_k = P(label < k)``
rather than on the class probabilities, so a convex combination of two ordered stacks is
itself ordered and a running max repairs the float dust. An optional QC gate can override the table per image."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["FusionSpec", "QCSpec", "ColumnSwitchSpec", "fusion_spec", "shape_class",
           "alpha_for_shape", "alpha_for_image",
           "qc_features", "qc_would_trip", "fuse_probs",
           "FUSION_KEYS", "WIDEFIELD_RULE_KEYS", "QC_KEYS", "COLUMN_SWITCH_KEYS",
           "SHAPE_CLASSES", "RULE_ORDER", "DEFAULT_SHAPE_RULES",
           "NUM_CLASSES", "NUM_BOUNDARIES",
           "explicit_block_problems",
           "MENU_BAKED", "MENU_NAMES", "MENU_SCHEMA", "MENU_FILE_KEYS",
           "MENU_FREE_ROW", "MENU_FROZEN_ROWS", "DEFAULT_SELECT_MARGIN",
           "load_menu", "menu_problems", "menu_block", "pick_block", "two_axis_pick"]

#: Ten classes, nine interfaces. Stated here so this module stays numpy-only on import.
NUM_CLASSES = 10
NUM_BOUNDARIES = NUM_CLASSES - 1

FUSION_KEYS = frozenset({"enabled", "alpha_table", "shape_rules", "qc", "column_switch"})
WIDEFIELD_RULE_KEYS = frozenset({"height_min", "height_max", "width_min"})
QC_KEYS = frozenset({"enabled", "unordered_frac_max", "fragmented_frac_max", "spike_max",
                     "thickness_frac_max", "thickness_ratio", "thickness_floor_px",
                     "median_window", "edge_frac", "alpha_when_a_trips",
                     "alpha_when_b_trips"})
COLUMN_SWITCH_KEYS = frozenset({"enabled", "tau", "alpha_win", "smooth_cols"})

#: Every shape class, in the order the boxes are tried; ``macula`` is the fallback.
SHAPE_CLASSES = ("macula", "maestro2_wide", "spectralis_wide", "triton_12x12")
RULE_ORDER = ("maestro2_wide", "spectralis_wide", "triton_12x12")

#: Measured geometry in pixels, with a tolerance: Maestro2 885 rows, Spectralis 496,
#: Triton 992, Cirrus 1024, and the column counts each protocol produces.
DEFAULT_SHAPE_RULES: dict[str, dict[str, int]] = {
    "maestro2_wide": {"height_min": 860, "height_max": 910, "width_min": 480},
    "spectralis_wide": {"height_min": 480, "height_max": 520, "width_min": 700},
    "triton_12x12": {"height_min": 980, "height_max": 1000, "width_min": 480},
}

#: PLACEHOLDER thresholds: "a healthy frame trips at most 5%" has not been measured.
DEFAULT_QC: dict[str, Any] = {
    "enabled": False,
    "unordered_frac_max": 0.05,
    "fragmented_frac_max": 0.05,
    "spike_max": 0.15,
    "thickness_frac_max": 0.25,
    "thickness_ratio": 3.0,
    "thickness_floor_px": 1.0,
    "median_window": 15,
    "edge_frac": 0.08,
    "alpha_when_a_trips": 0.1,
    "alpha_when_b_trips": 0.9,
}

#: PLACEHOLDER as well; ``tau`` is in units of image height.
DEFAULT_COLUMN_SWITCH: dict[str, Any] = {
    "enabled": False,
    "tau": 0.04,
    "alpha_win": 0.9,
    "smooth_cols": 9,
}


@dataclass(frozen=True)
class QCSpec:
    """Per-image collapse detector. ``enabled: false`` computes nothing at all."""

    enabled: bool
    unordered_frac_max: float
    fragmented_frac_max: float
    spike_max: float
    thickness_frac_max: float
    thickness_ratio: float
    thickness_floor_px: float
    median_window: int
    edge_frac: float
    alpha_when_a_trips: float
    alpha_when_b_trips: float

    def as_block(self) -> dict:
        return {"enabled": bool(self.enabled),
                "unordered_frac_max": float(self.unordered_frac_max),
                "fragmented_frac_max": float(self.fragmented_frac_max),
                "spike_max": float(self.spike_max),
                "thickness_frac_max": float(self.thickness_frac_max),
                "thickness_ratio": float(self.thickness_ratio),
                "thickness_floor_px": float(self.thickness_floor_px),
                "median_window": int(self.median_window),
                "edge_frac": float(self.edge_frac),
                "alpha_when_a_trips": float(self.alpha_when_a_trips),
                "alpha_when_b_trips": float(self.alpha_when_b_trips)}


@dataclass(frozen=True)
class ColumnSwitchSpec:
    """Per-column model switch where the two disagree by more than ``tau * H`` rows."""

    enabled: bool
    tau: float
    alpha_win: float
    smooth_cols: int

    def as_block(self) -> dict:
        return {"enabled": bool(self.enabled), "tau": float(self.tau),
                "alpha_win": float(self.alpha_win), "smooth_cols": int(self.smooth_cols)}


@dataclass(frozen=True)
class FusionSpec:
    """``fusion`` block resolved, ALWAYS to the full table.
    ``alpha_table[shape_class]`` is nine weights of model A, one per interface."""

    enabled: bool
    alpha_table: dict[str, tuple[float, ...]]
    shape_rules: dict[str, dict[str, int]]
    qc: QCSpec
    column_switch: ColumnSwitchSpec

    def shape_class(self, hw: tuple[int, int]) -> str:
        return shape_class(hw, self.shape_rules)

    def alpha(self, hw: tuple[int, int]) -> np.ndarray:
        return alpha_for_shape(hw, self)

    def as_block(self) -> dict:
        """The canonical, fully explicit block: what the export bakes and verify reads.
        Every shipped number is written out; nothing may fall back to a code default."""
        return {
            "enabled": bool(self.enabled),
            "alpha_table": {k: [float(v) for v in self.alpha_table[k]]
                            for k in SHAPE_CLASSES},
            "shape_rules": {k: {kk: int(vv) for kk, vv in sorted(self.shape_rules[k].items())}
                            for k in RULE_ORDER},
            "qc": self.qc.as_block(),
            "column_switch": self.column_switch.as_block(),
        }

    def describe(self) -> str:
        """One line for a log: the table, plus which of the two gates are armed.
        A row whose nine weights agree prints as one number."""
        rows = []
        for k in SHAPE_CLASSES:
            row = self.alpha_table[k]
            rows.append(f"{k}=" + (f"{row[0]:g}" if len(set(row)) == 1
                                   else "[" + ",".join(f"{v:g}" for v in row) + "]"))
        return (f"alpha_table {' '.join(rows)} · qc={'on' if self.qc.enabled else 'off'}"
                f" · column_switch={'on' if self.column_switch.enabled else 'off'}")


def _alpha(value: Any, name: str) -> float:
    a = float(value)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"fusion.{name} must be in [0, 1], got {value!r}")
    return a


def _alpha_row(value: Any, name: str) -> tuple[float, ...]:
    """One table row: nine weights, or a scalar meaning "the same nine"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return tuple([_alpha(value, f"alpha_table.{name}")] * NUM_BOUNDARIES)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"fusion.alpha_table.{name} must be a number or {NUM_BOUNDARIES} "
                         f"numbers, got {value!r}")
    row = list(value)
    if len(row) != NUM_BOUNDARIES:
        raise ValueError(f"fusion.alpha_table.{name} must have {NUM_BOUNDARIES} entries "
                         f"(b1..b{NUM_BOUNDARIES}), got {len(row)}")
    return tuple(_alpha(v, f"alpha_table.{name}[{i}]") for i, v in enumerate(row))




def _box(rule: Mapping[str, Any], name: str) -> dict[str, int]:
    bad = set(rule) - WIDEFIELD_RULE_KEYS
    if bad:
        raise ValueError(f"fusion.shape_rules.{name}: unknown key(s) {sorted(bad)}")
    box = {k: int(v) for k, v in rule.items()}
    if not (0 < box["height_min"] <= box["height_max"]) or box["width_min"] <= 0:
        raise ValueError(f"fusion.shape_rules.{name} is not a valid box: {box}")
    return box


def _sub_block(block: Mapping[str, Any], name: str, defaults: Mapping[str, Any],
               keys: frozenset) -> dict:
    given = dict(block.get(name) or {})
    bad = set(given) - keys
    if bad:
        raise ValueError(f"fusion.{name}: unknown key(s) {sorted(bad)}; known: {sorted(keys)}")
    return {**defaults, **given}


def fusion_spec(cfg: Mapping[str, Any] | None) -> FusionSpec | None:
    """``cfg["fusion"]`` -> :class:`FusionSpec`, or ``None`` when absent or disabled.
    Two accepted forms, always resolved to the same full table; unknown keys are fatal."""
    block = dict((cfg or {}).get("fusion") or {})
    if not block or not bool(block.get("enabled", False)):
        return None
    unknown = set(block) - FUSION_KEYS
    if unknown:
        raise ValueError(f"fusion: unknown key(s) {sorted(unknown)}; known: {sorted(FUSION_KEYS)}")

    rules = {k: dict(v) for k, v in DEFAULT_SHAPE_RULES.items()}
    given_rules = dict(block.get("shape_rules") or {})
    bad = set(given_rules) - set(RULE_ORDER)
    if bad:
        raise ValueError(f"fusion.shape_rules: unknown shape class(es) {sorted(bad)}; "
                         f"known: {list(RULE_ORDER)}")
    for name, rule in given_rules.items():
        rules[name] = _box({**rules[name], **dict(rule or {})}, name)
    for name in RULE_ORDER:
        rules[name] = _box(rules[name], name)

    if "alpha_table" not in block:
        raise ValueError("fusion.alpha_table must state every shape class")
    given = dict(block["alpha_table"] or {})
    bad = set(given) - set(SHAPE_CLASSES)
    if bad:
        raise ValueError(f"fusion.alpha_table: unknown shape class(es) {sorted(bad)}")
    missing = [k for k in SHAPE_CLASSES if k not in given]
    if missing:
        raise ValueError(f"fusion.alpha_table names no weight for {missing}")
    table = {k: _alpha_row(given[k], k) for k in SHAPE_CLASSES}

    qc_raw = _sub_block(block, "qc", DEFAULT_QC, QC_KEYS)
    qc = QCSpec(enabled=bool(qc_raw["enabled"]),
                unordered_frac_max=float(qc_raw["unordered_frac_max"]),
                fragmented_frac_max=float(qc_raw["fragmented_frac_max"]),
                spike_max=float(qc_raw["spike_max"]),
                thickness_frac_max=float(qc_raw["thickness_frac_max"]),
                thickness_ratio=float(qc_raw["thickness_ratio"]),
                thickness_floor_px=float(qc_raw["thickness_floor_px"]),
                median_window=int(qc_raw["median_window"]),
                edge_frac=float(qc_raw["edge_frac"]),
                alpha_when_a_trips=_alpha(qc_raw["alpha_when_a_trips"], "qc.alpha_when_a_trips"),
                alpha_when_b_trips=_alpha(qc_raw["alpha_when_b_trips"], "qc.alpha_when_b_trips"))
    if qc.thickness_ratio <= 1.0:
        raise ValueError(f"fusion.qc.thickness_ratio must be > 1, got {qc.thickness_ratio}")
    if qc.median_window < 1 or qc.median_window % 2 == 0:
        raise ValueError(f"fusion.qc.median_window must be a positive odd number of columns, "
                         f"got {qc.median_window}")
    if not 0.0 < qc.edge_frac < 0.5:
        raise ValueError(f"fusion.qc.edge_frac must be in (0, 0.5), got {qc.edge_frac}")

    cs_raw = _sub_block(block, "column_switch", DEFAULT_COLUMN_SWITCH, COLUMN_SWITCH_KEYS)
    if bool(cs_raw["enabled"]):
        raise ValueError("fusion.column_switch.enabled must be false")
    cs = ColumnSwitchSpec(enabled=bool(cs_raw["enabled"]), tau=float(cs_raw["tau"]),
                          alpha_win=_alpha(cs_raw["alpha_win"], "column_switch.alpha_win"),
                          smooth_cols=int(cs_raw["smooth_cols"]))
    if cs.tau <= 0.0:
        raise ValueError(f"fusion.column_switch.tau must be > 0 (units of H), got {cs.tau}")
    if cs.smooth_cols < 1 or cs.smooth_cols % 2 == 0:
        raise ValueError(f"fusion.column_switch.smooth_cols must be a positive odd number "
                         f"of columns, got {cs.smooth_cols}")

    return FusionSpec(enabled=True, alpha_table=table, shape_rules=rules, qc=qc,
                      column_switch=cs)


def _same_leaf(canon: Any, raw: Any) -> bool:
    """Leaf equality that ignores int/float/bool spelling but nothing else."""
    if isinstance(canon, (int, float)) and isinstance(raw, (int, float)):
        return float(canon) == float(raw)
    return bool(canon == raw)


def _walk(canon: Any, raw: Any, path: str, out: list[str]) -> None:
    if isinstance(canon, dict):
        if not isinstance(raw, Mapping):
            out.append(f"{path} is not written out as a block (got {raw!r})")
            return
        for key in canon:
            if key not in raw:
                out.append(f"{path}.{key} is not stated")
            else:
                _walk(canon[key], raw[key], f"{path}.{key}", out)
        for key in sorted(set(raw) - set(canon)):
            out.append(f"{path}.{key} is not part of the canonical block")
        return
    if isinstance(canon, list):
        # A bare number is the accepted shorthand for "the same nine", so it is explicit.
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if not all(_same_leaf(v, raw) for v in canon):
                out.append(f"{path} = {raw!r} but resolves to {canon}")
            return
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            out.append(f"{path} must be a number or {len(canon)} numbers, got {raw!r}")
            return
        row = list(raw)
        if len(row) != len(canon):
            out.append(f"{path} has {len(row)} entries, not {len(canon)}")
            return
        for i, (c, r) in enumerate(zip(canon, row)):
            if not _same_leaf(c, r):
                out.append(f"{path}[{i}] = {r!r} but resolves to {c!r}")
        return
    if not _same_leaf(canon, raw):
        out.append(f"{path} = {raw!r} but resolves to {canon!r}")


def explicit_block_problems(block: Mapping[str, Any] | None, spec: FusionSpec,
                            *, path: str = "fusion") -> list[str]:
    """Why ``block`` is not the canonical block ``spec`` resolves to; empty list = it is.
    The one owner of "a shipped fusion block is complete"."""
    problems: list[str] = []
    _walk(spec.as_block(), dict(block or {}), path, problems)
    return problems


#: The entry that is not a literal block: "whatever the primary checkpoint carries".
MENU_BAKED = "baked"

#: The menu, by name and in measurement order: the one owner of which tables a package may
#: choose between, shared by the packer, the server selector and the verifier.
MENU_NAMES: tuple[str, ...] = ("baked", "even", "cnn_heavy")

#: The ONE ``alpha_table`` row a menu entry may move, and the ones it may not: the container
#: measures its wide column on PSEUDO wide-field, so a stretched macula must not pick it.
MENU_FREE_ROW = "macula"
MENU_FROZEN_ROWS: tuple[str, ...] = tuple(c for c in SHAPE_CLASSES if c != MENU_FREE_ROW)

#: The menu file format this code reads; an unknown value is refused, never best-guessed.
MENU_SCHEMA = 1

#: Top-level keys the menu file may carry; an unknown key is fatal, as in :func:`fusion_spec`.
MENU_FILE_KEYS = frozenset({"schema", "_doc", "blocks"})

#: Default margin a challenger must beat ``baked`` by; the shipped path reads it from config.
DEFAULT_SELECT_MARGIN = 0.002

#: Slack on the margin comparison, so "beats it by exactly the margin" is a tie rather than
#: a coin flip decided by float representation. Nine orders below the margin itself.
_MARGIN_EPS = 1e-9

#: Version of the two-axis adopt rule, recorded by every selector that decides with it.
def two_axis_pick(scores: Mapping[str, float], *, margin: float, baseline: str,
                  margin_name: str = "margin") -> tuple[str, str]:
    """Pick a challenger only when it beats the incumbent by the stated margin."""
    margin = float(margin)
    if not margin >= 0.0:
        raise ValueError(f"{margin_name} must be >= 0, got {margin!r}")
    if not scores:
        return baseline, "nothing was measured"
    if baseline not in scores:
        return baseline, (f"{baseline!r} was not among the measured blocks {sorted(scores)}; "
                          "with no baseline score there is nothing to beat")
    ranked = sorted(scores, key=lambda n: (-float(scores[n]), n))
    best = ranked[0]
    base = float(scores[baseline])
    if best == baseline:
        return baseline, (f"{baseline} is the best of {len(scores)} measured block(s) "
                          f"(final {base:.4f})")
    gain = float(scores[best]) - base
    if gain - margin > _MARGIN_EPS:
        return best, (f"{best} final {float(scores[best]):.4f} beats {baseline} {base:.4f} by "
                      f"+{gain:.4f} > margin {margin:.4f}")
    return baseline, (f"best challenger {best} beats {baseline} by only {gain:+.4f} <= margin "
                      f"{margin:.4f}; keeping {baseline} (a selection that flips on noise is "
                      "worse than none)")



def load_menu(path: Path | str) -> dict[str, dict | None]:
    """Read the shipped menu file -> ``{name: block or None}``, in file order.
    An unknown schema, a missing entry or an unknown top-level key is refused."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: the menu file must be a JSON object, got {type(raw).__name__}")
    unknown = set(raw) - MENU_FILE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}; "
                         f"known: {sorted(MENU_FILE_KEYS)}")
    schema = raw.get("schema")
    if schema != MENU_SCHEMA:
        raise ValueError(f"{path}: schema is {schema!r}, not {MENU_SCHEMA}; this reader only "
                         "understands the format it was written against, and half-understanding "
                         "a menu is how a weight nobody typed gets shipped")
    blocks = raw.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError(f"{path}: 'blocks' must be a non-empty JSON object")
    if tuple(blocks) != MENU_NAMES:
        missing = [n for n in MENU_NAMES if n not in blocks]
        extra = [n for n in blocks if n not in MENU_NAMES]
        raise ValueError(
            f"{path}: the menu must name exactly {list(MENU_NAMES)}, in that order; got "
            f"{list(blocks)}"
            + (f" (missing {missing})" if missing else "")
            + (f" (not on the menu: {extra})" if extra else "")
            + ". An extra entry is a table nobody reviewed, a missing one narrows the "
              "choice while the log still says 'the menu', and a reordering changes which "
              "blocks a tight budget drops.")
    out: dict[str, dict | None] = {}
    for name, entry in blocks.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: {name!r} is not a usable block name")
        if entry is None:
            if name != MENU_BAKED:
                raise ValueError(f"{path}: only {MENU_BAKED!r} may be null; {name!r} has to "
                                 "write its table out")
            out[name] = None
        elif isinstance(entry, Mapping):
            out[name] = dict(entry)
        else:
            raise ValueError(f"{path}: blocks.{name} must be an object, got "
                             f"{type(entry).__name__}")
    return out


def menu_problems(menu: Mapping[str, Mapping[str, Any] | None], baked: FusionSpec,
                  *, path: str = "menu") -> list[str]:
    """Why this menu is not shippable against ``baked``; empty list = it is.
    The one owner of "a shipped menu is complete and comparable"."""
    problems: list[str] = []
    if MENU_BAKED not in menu:
        problems.append(f"{path} does not list {MENU_BAKED!r}")
    baked_block = baked.as_block()
    for name, entry in menu.items():
        if entry is None:
            if name != MENU_BAKED:
                problems.append(f"{path}.{name} is null; only {MENU_BAKED!r} may be")
            continue
        try:
            spec = fusion_spec({"fusion": dict(entry)})
        except Exception as exc:                                          # noqa: BLE001
            problems.append(f"{path}.{name} is not a valid fusion block: {exc}")
            continue
        if spec is None:
            problems.append(f"{path}.{name} resolves to no mixture (enabled is false or "
                            "missing); a menu entry is a table, not an off switch")
            continue
        problems.extend(explicit_block_problems(entry, spec, path=f"{path}.{name}"))
        merged = {**dict(entry), "alpha_table": baked_block["alpha_table"]}
        for line in explicit_block_problems(merged, baked, path=f"{path}.{name}"):
            problems.append(line + "   (a menu entry may differ from the baked block in "
                                   "alpha_table ONLY)")
        # ...and inside alpha_table, only the free row, read off the RESOLVED spec.
        table = spec.as_block()["alpha_table"]
        for cls in MENU_FROZEN_ROWS:
            if table[cls] != baked_block["alpha_table"][cls]:
                problems.append(
                    f"{path}.{name}.alpha_table.{cls} is {table[cls]}, but the baked block "
                    f"says {baked_block['alpha_table'][cls]}; a menu entry may move the "
                    f"{MENU_FREE_ROW!r} row ONLY. The container's wide-field column is "
                    "pseudo-wide-field (macula stretched 2.5x, which lands in this same "
                    "shape class), so a menu free to move a wide row lets a fake anatomy "
                    "choose the weight the hidden set's real wide-field half ships with")
    return problems


def menu_block(name: str, menu: Mapping[str, Mapping[str, Any] | None],
               baked: FusionSpec) -> dict:
    """The CANONICAL block named ``name``; :data:`MENU_BAKED` resolves to ``baked``.
    Always ``FusionSpec.as_block()``, never the raw JSON."""
    if name not in menu:
        raise ValueError(f"{name!r} is not on the menu ({sorted(menu)}); the server may only "
                         "choose between blocks the package ships")
    entry = menu[name]
    if entry is None:
        if name != MENU_BAKED:
            raise ValueError(f"menu entry {name!r} is null but is not {MENU_BAKED!r}")
        return baked.as_block()
    spec = fusion_spec({"fusion": dict(entry)})
    if spec is None:
        raise ValueError(f"menu entry {name!r} resolves to no mixture")
    return spec.as_block()


def pick_block(scores: Mapping[str, float], *, margin: float,
               baked: str = MENU_BAKED) -> tuple[str, str]:
    """Select a shipped fusion block by the final-score margin."""
    return two_axis_pick(scores, margin=margin, baseline=baked,
                         margin_name="fusion_selection.margin")



def shape_class(hw: tuple[int, int],
                rules: Mapping[str, Mapping[str, int]] | None = None) -> str:
    """Which protocol-shaped box ``(H, W)`` falls in; first hit in :data:`RULE_ORDER` wins."""
    r = DEFAULT_SHAPE_RULES if rules is None else rules
    h, w = int(hw[0]), int(hw[1])
    for name in RULE_ORDER:
        box = r.get(name)
        if box is None:
            continue
        if int(box["height_min"]) <= h <= int(box["height_max"]) and w >= int(box["width_min"]):
            return name
    return "macula"




def alpha_for_shape(hw: tuple[int, int], spec: FusionSpec) -> np.ndarray:
    """The nine weights of model A for a frame of shape ``hw``: ``(9,) float64``, b1..b9."""
    return np.asarray(spec.alpha_table[shape_class(hw, spec.shape_rules)], dtype=np.float64)




def _checked_alpha(alpha, nb: int, width: int, ndim: int) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim and a.shape not in ((nb,), (nb, width)):
        raise ValueError(f"alpha must be a scalar, ({nb},) or ({nb}, {width}), got shape "
                         f"{a.shape}")
    if a.ndim and ndim != 3:
        raise ValueError(f"a per-boundary alpha needs a (C, H, W) probability map, got "
                         f"{ndim} dimensions")
    if a.size == 0 or not np.isfinite(a).all() or float(a.min()) < 0.0 or float(a.max()) > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return a


def _cumulative_fuse(prob_a: np.ndarray, prob_b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The per-boundary mixture; ``a`` is ``(C-1,)`` or ``(C-1, W)``."""
    C = prob_a.shape[0]
    pa = np.asarray(prob_a, dtype=np.float32)
    pb = np.asarray(prob_b, dtype=np.float32)
    sa, sb = pa.sum(axis=0, keepdims=True), pb.sum(axis=0, keepdims=True)
    if not (np.all(sa > 0) and np.all(sb > 0)):
        raise ValueError("a pixel has zero class mass; that is not a probability map")
    fa = np.cumsum(pa / sa, axis=0)[: C - 1]
    fb = np.cumsum(pb / sb, axis=0)[: C - 1]
    w = a.astype(np.float32)
    w = w[:, None, None] if w.ndim == 1 else w[:, None, :]
    f = w * fa + (1.0 - w) * fb
    np.maximum.accumulate(f, axis=0, out=f)
    np.clip(f, 0.0, 1.0, out=f)
    p = np.concatenate([f[:1], f[1:] - f[:-1], 1.0 - f[-1:]], axis=0)
    np.clip(p, 0.0, None, out=p)
    p /= p.sum(axis=0, keepdims=True)
    return np.ascontiguousarray(p, dtype=np.float32)


def fuse_probs(prob_a: np.ndarray, prob_b: np.ndarray, alpha) -> np.ndarray:
    """Mix two class-probability maps of identical shape by ``alpha``, the weight of A.
    ``alpha`` is a scalar, a ``(C-1,)`` vector, or a ``(C-1, W)`` per-column table."""
    if prob_a.shape != prob_b.shape:
        raise ValueError(f"cannot fuse probability maps of shapes {prob_a.shape} and "
                         f"{prob_b.shape}")
    width = int(prob_a.shape[-1])
    # Derived from the map, not from NUM_CLASSES, so a different C raises instead of broadcasting.
    nb = int(prob_a.shape[0]) - 1 if prob_a.ndim == 3 else NUM_BOUNDARIES
    a = _checked_alpha(alpha, nb, width, prob_a.ndim)
    if a.ndim == 0 or float(a.max()) == float(a.min()):
        scalar = float(a.reshape(-1)[0]) if a.ndim else float(a)
        if scalar == 1.0:
            return np.ascontiguousarray(prob_a, dtype=np.float32)
        if scalar == 0.0:
            return np.ascontiguousarray(prob_b, dtype=np.float32)
        out = (scalar * prob_a.astype(np.float32, copy=False)
               + (1.0 - scalar) * prob_b.astype(np.float32, copy=False))
        return np.ascontiguousarray(out, dtype=np.float32)
    return _cumulative_fuse(prob_a, prob_b, a)




def _spike(bounds: np.ndarray, height: int, window: int,
           edge_frac: float) -> tuple[float, float]:
    """``(local, edge)`` spike scores for one model's ``(9, W)`` boundary rows, in units of H.
    ``local`` is the p99 deviation from a running median; ``edge`` looks at the frame border."""
    from scipy.ndimage import median_filter

    w = int(bounds.shape[1])
    if w == 0 or height <= 0:
        return 0.0, 0.0
    win = min(int(window), w if w % 2 else max(w - 1, 1))
    smooth = median_filter(bounds, size=(1, max(win, 1)), mode="nearest")
    local = float(np.percentile(np.abs(bounds - smooth), 99)) / height
    n_edge = max(1, int(round(edge_frac * w)))
    core = bounds[:, w // 4: w - w // 4] if w >= 4 else bounds
    core_med = np.median(core, axis=1)
    left = np.median(bounds[:, :n_edge], axis=1)
    right = np.median(bounds[:, -n_edge:], axis=1)
    edge = float(np.max(np.maximum(np.abs(left - core_med),
                                   np.abs(right - core_med)))) / height
    return local, edge


def _thickness_frac(bounds: np.ndarray, ratio: float, floor_px: float) -> float:
    """Fraction of columns where an inner class is thicker than ``ratio`` x, or thinner than
    1/``ratio`` x, its own image-median thickness."""
    thick = np.diff(bounds, axis=0)
    if thick.size == 0:
        return 0.0
    med = np.median(thick, axis=1)
    live = med >= float(floor_px)
    if not live.any():
        return 0.0
    t, m = thick[live], med[live][:, None]
    bad = (t > float(ratio) * m) | (t < m / float(ratio))
    return float(bad.any(axis=0).mean())


def _per_model_features(prob: np.ndarray, bounds: np.ndarray, qc: QCSpec) -> dict:
    from octtta.postproc import column_flags

    height = int(prob.shape[1])
    unordered, fragmented = column_flags(np.argmax(prob, axis=0))
    local, edge = _spike(bounds, height, qc.median_window, qc.edge_frac)
    return {
        "unordered_frac": float(unordered.mean()) if unordered.size else 0.0,
        "fragmented_frac": float(fragmented.mean()) if fragmented.size else 0.0,
        "spike_local": local,
        "spike_edge": edge,
        "spike": max(local, edge),
        "thickness_frac": _thickness_frac(bounds, qc.thickness_ratio, qc.thickness_floor_px),
        "_unordered": unordered,
    }


def qc_features(prob_a: np.ndarray, prob_b: np.ndarray, spec: FusionSpec) -> dict:
    """Every per-image QC number for the two maps, with no threshold applied.
    Separated from :func:`alpha_for_image` so an offline sweep can re-threshold them."""
    from octtta.surface import expected_boundaries

    if prob_a.shape != prob_b.shape:
        raise ValueError(f"cannot compare probability maps of shapes {prob_a.shape} and "
                         f"{prob_b.shape}")
    if prob_a.ndim != 3 or prob_a.shape[0] != NUM_CLASSES:
        raise ValueError(f"qc features need a ({NUM_CLASSES}, H, W) map, got {prob_a.shape}")
    height, width = int(prob_a.shape[1]), int(prob_a.shape[2])
    ba = expected_boundaries(prob_a).astype(np.float64)
    bb = expected_boundaries(prob_b).astype(np.float64)
    fa = _per_model_features(prob_a, ba, spec.qc)
    fb = _per_model_features(prob_b, bb, spec.qc)
    disagree = (np.median(np.abs(ba - bb), axis=1) / max(height, 1) if width
                else np.zeros(NUM_BOUNDARIES))
    return {
        "height": height, "width": width,
        "a": {k: v for k, v in fa.items() if not k.startswith("_")},
        "b": {k: v for k, v in fb.items() if not k.startswith("_")},
        "disagree": [float(v) for v in disagree],
        "arrays": {
            "bounds_a": ba, "bounds_b": bb,
            "conf_a": prob_a.max(axis=0).mean(axis=0).astype(np.float64),
            "conf_b": prob_b.max(axis=0).mean(axis=0).astype(np.float64),
            "unordered_a": fa["_unordered"], "unordered_b": fb["_unordered"],
        },
    }


def _trips(feat: Mapping[str, float], qc: QCSpec) -> list[str]:
    """Which thresholds this model's features cross, by name (empty = clean)."""
    out = []
    for name, limit in (("unordered_frac", qc.unordered_frac_max),
                        ("fragmented_frac", qc.fragmented_frac_max),
                        ("spike", qc.spike_max),
                        ("thickness_frac", qc.thickness_frac_max)):
        if float(feat[name]) > float(limit):
            out.append(name)
    return out


def qc_would_trip(features: Mapping[str, Any], spec: FusionSpec) -> dict[str, list[str]]:
    """``{"a": [...], "b": [...]}``: which QC thresholds each model's features cross.
    The same call :func:`alpha_for_image` makes when the gate is armed."""
    return {"a": _trips(features["a"], spec.qc), "b": _trips(features["b"], spec.qc)}




def alpha_for_image(hw: tuple[int, int], prob_a: np.ndarray, prob_b: np.ndarray,
                    spec: FusionSpec, *, features: Mapping[str, Any] | None = None,
                    map_hw: tuple[int, int] | None = None
                    ) -> tuple[np.ndarray, dict]:
    """The weights to fuse THIS image with: the shape table, then the two optional gates.
    Returns ``(alpha, info)``, where ``alpha`` is ``(9,)`` or ``(9, W)`` once the column
    switch has fired, and ``info`` records which gate decided."""
    want = (int(hw[0]), int(hw[1])) if map_hw is None else (int(map_hw[0]), int(map_hw[1]))
    if getattr(prob_a, "ndim", 0) == 3 and tuple(prob_a.shape[-2:]) != want:
        raise ValueError(f"alpha_for_image was given map_hw={want} but probability maps of "
                         f"{tuple(prob_a.shape)}; the shape rule and the features would "
                         "describe different frames")
    base = alpha_for_shape(hw, spec)
    info: dict[str, Any] = {
        "shape_class": shape_class(hw, spec.shape_rules),
        "alpha_table_row": [float(v) for v in base],
        "qc_enabled": bool(spec.qc.enabled),
        "column_switch_enabled": bool(spec.column_switch.enabled),
        "decision": "table",
        "per_column": False,
        "n_switched_columns": 0,
    }
    if not spec.qc.enabled:
        info["alpha_mean"] = [float(v) for v in base]
        return base, info

    try:
        feats = qc_features(prob_a, prob_b, spec) if features is None else features
    except Exception as exc:                                             # noqa: BLE001
        info["decision"] = "qc_error"
        info["qc_error"] = f"{type(exc).__name__}: {exc}"
        info["alpha_mean"] = [float(v) for v in base]
        return base, info

    info["features"] = {"a": dict(feats["a"]), "b": dict(feats["b"])}
    info["disagree"] = list(feats["disagree"])
    alpha = base
    if spec.qc.enabled:
        trip_a, trip_b = _trips(feats["a"], spec.qc), _trips(feats["b"], spec.qc)
        info["tripped"] = {"a": trip_a, "b": trip_b}
        if trip_a and not trip_b:
            alpha = np.full(NUM_BOUNDARIES, spec.qc.alpha_when_a_trips, dtype=np.float64)
            info["decision"] = "a_tripped"
        elif trip_b and not trip_a:
            alpha = np.full(NUM_BOUNDARIES, spec.qc.alpha_when_b_trips, dtype=np.float64)
            info["decision"] = "b_tripped"
        elif trip_a and trip_b:
            info["decision"] = "both_tripped"
    info["alpha_mean"] = [float(v) for v in (alpha.mean(axis=1) if alpha.ndim == 2 else alpha)]
    return alpha, info
