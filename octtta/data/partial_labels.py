"""Partial boundary annotations -> a label plane the loss can consume.

Public OCT sets annotate some of the nine official boundaries, never all nine, and often
not on every column. What survives at a pixel is a contiguous interval of classes packed
into one byte: ``code = 10 * lo + hi`` (an exact class ``c`` is ``11 * c``), with ``255``
for "nothing known here". :data:`SURFACE_TO_BOUNDARY` is the only machine-readable copy of
which surface feeds which boundary, applied at BUILD time, so editing it rebuilds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from octtta.surface import IGNORE_UINT8, NUM_CLASSES

__all__ = [
    'NUM_BOUNDARIES',
    'SURFACE_TO_BOUNDARY',
    'DatasetMapping',
    'MAPPING_VERSION',
    'pack_interval',
    'unpack_interval',
    'interval_lut',
    'rasterize_partial',
    'exact_from_code',
    'render_mapping_table',
    'SYNTHETIC_REAL_BOUNDARIES',
    'SYNTHETIC_FILL_BOUNDARIES',
    'SYNTHETIC_BOUNDARY_MAP_PRESETS',
    'SyntheticBoundaryMap',
    'SyntheticBoundaryMapError',
    'resolve_synthetic_boundary_map',
    'class_groups_from_cuts',
    'official_intervals_from_synthetic',
    'merge_synthetic_labels',
    'merge_official_labels',
    'BoundaryOffsetViolation',
    'OFFSET_COLLAPSE_MAX_FRACTION',
    'apply_boundary_offsets',
    'offsets_to_json',
    'offsets_from_json',
    'surfaces_to_json',
    'surfaces_from_json',
]

NUM_BOUNDARIES = NUM_CLASSES - 1

#: The stamp that makes a built pool provable; a pool carrying a different one is refused.
MAPPING_VERSION = "gamma-d88"


class BoundaryOffsetViolation(RuntimeError):
    """An offset moved a supervised surface somewhere it is not allowed to go."""


#: Maximum permitted collapse after correcting a supervised boundary.
OFFSET_COLLAPSE_MAX_FRACTION = 0.001


@dataclass(frozen=True)
class DatasetMapping:
    """One dataset's surfaces, in its own storage order, mapped to official boundaries."""

    key: str
    order: str                    # "storage" | "depth" -- how surface i is identified
    boundaries: tuple[int | None, ...]
    note: str = ""
    #: ``(boundary index, signed row shift)`` applied BEFORE rasterisation; POSITIVE is DEEPER.
    interpolatable: tuple[int, ...] = field(default_factory=tuple)
    offsets_px: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        got = [b for b in self.boundaries if b is not None]
        if len(set(got)) != len(got):
            raise ValueError(f"{self.key}: two surfaces map to the same boundary {got}")
        if any(not 0 <= b < NUM_BOUNDARIES for b in got):
            raise ValueError(f"{self.key}: boundary index out of range in {got}")
        if list(got) != sorted(got):
            raise ValueError(
                f"{self.key}: surfaces must be listed in depth order, got {got}")
        seen: list[int] = []
        for pair in self.offsets_px:
            if len(tuple(pair)) != 2:
                raise ValueError(
                    f"{self.key}: offsets_px entries are (boundary, shift) pairs, got "
                    f"{pair!r}")
            b, dv = int(pair[0]), int(pair[1])
            if b not in got:
                # An offset on a boundary this dataset does not supervise moves nothing,
                # and reads in every log exactly like one that does.
                raise ValueError(
                    f"{self.key}: offsets_px shifts boundary {b}, which this dataset does "
                    f"not supervise (it draws {tuple(got)}). The offset would move nothing "
                    f"while every record of it says a correction was applied.")
            if dv == 0:
                # "Declared and does nothing" is invisible in the labels but not in the audit.
                raise ValueError(
                    f"{self.key}: offsets_px declares a zero shift on boundary {b}. Omit "
                    f"the boundary instead; a zero offset changes no pixel while making "
                    f"index.json claim a correction that was never measured.")
            if b in seen:
                raise ValueError(
                    f"{self.key}: offsets_px names boundary {b} twice; only one of the two "
                    f"shifts could be applied and nothing says which.")
            seen.append(b)
        if seen != sorted(seen):
            raise ValueError(
                f"{self.key}: offsets_px must be listed in boundary order, got {seen}")

    @property
    def available(self) -> tuple[int, ...]:
        """Official boundary indices this dataset can supervise."""
        return tuple(b for b in self.boundaries if b is not None)

    @property
    def offsets(self) -> dict[int, int]:
        """:attr:`offsets_px` as ``{official boundary index: signed row shift}``."""
        return {int(b): int(dv) for b, dv in self.offsets_px}


#: Owner of the surface -> official boundary mapping; each row's ``note`` names its source.
SURFACE_TO_BOUNDARY: dict[str, DatasetMapping] = {
    "duke_dme_2015": DatasetMapping(
        key="duke_dme_2015", order="storage",
        boundaries=(0, 1, 3, 4, None, 6, 7, 8),
        interpolatable=(2, 5),
        note="D84/γ. 8 surfaces in the .mat, depth-ordered: ILM->b1, NFL/GCL->b2, "
             "IPL/INL->b4, INL/OPL->b5, OPL/ONL->NOT FED (not an official interface), "
             "IS/OS (ISM/ISE)->b7, OS/RPE->b8, BM->b9. b3 (GCL/IPL) and b6 (ELM) are "
             "simply not drawn by Duke and stay unsupervised. Pre-D84 this row fed "
             "IPL/INL->b3, INL/OPL->b4 and OPL/ONL->b5, i.e. one anatomical layer too "
             "deep on all three -- half of the b3 +8 / b4 +9 / b5 +9 the D76 probe "
             "measured. Neither gap is interpolated: an interpolated line is not an "
             "annotation (D58's usage ruling)"),
    "jhu_hcms": DatasetMapping(
        key="jhu_hcms", order="storage",
        boundaries=(0, 1, 3, 4, None, 5, 6, 7, 8),
        interpolatable=(2,),
        note="D84/γ. 9 surfaces: ILM->b1, RNFL/GCL->b2, IPL/INL->b4, INL/OPL->b5, "
             "OPL/ONL->NOT FED, ELM->b6, IS/OS->b7, OS/RPE->b8, BM->b9. Pre-D84 this "
             "was read as 'the only public set that covers all nine' and fed straight "
             "through 0..8 -- but JHU never draws GCL/IPL, so what it actually did was "
             "feed IPL/INL as b3, INL/OPL as b4 and OPL/ONL as b5. Under γ b3 is "
             "unsupervised and no public set supplies it. Sparse control points, PCHIP "
             "interpolated to dense surfaces (that interpolation is along a single "
             "annotated surface, not across missing ones)"),
    "oct5k": DatasetMapping(
        key="oct5k", order="storage",
        boundaries=(0, 4, 6, 7, 8),
        note="D84/γ (was D59). 5 surfaces: ILM->b1, the 'OPL' line->b5, IS-OS->b7, "
             "IBRPE->b8, OBRPE->b9. The one change γ makes here: OCT5k's 'OPL' line is "
             "the INL/OPL interface (LABELS §3 -- its relative depth 0.456 is the "
             "official b5's, and the official synthetic release draws its own b4 from "
             "this same line, which is why the synthetic set cannot be used to read the "
             "official convention). Pre-D84 it fed b4. D59's geometry still holds: OBRPE "
             "is the outer boundary, then two official lines are skipped up to the OPL "
             "line, then two more up to ILM. Measured agreement <= 0.022 of retinal "
             "thickness on all five"),
    "aireadi::Heidelberg_Spectralis": DatasetMapping(
        key="aireadi::Heidelberg_Spectralis", order="depth",
        boundaries=(0, 1, 2, 3, 4, None, 5, 6, 7, None, 8),
        # No ``offsets_px`` on b1 or b9: that zero is MEASURED, not assumed.
        note="Registered data source; selection and exclusion checks run during pool preparation."),
    "aireadi::Topcon_Maestro2": DatasetMapping(
        key="aireadi::Topcon_Maestro2", order="depth",
        boundaries=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        note="D84/γ. Selected by depth index, never by name: F8 established the shipped "
             "names are a permutation. The 172-eye cross-device comparison against "
             "Spectralis's named surfaces (the recorded cross-device comparison, 170/172 same-sign) identifies idx0..8 as ILM, "
             "RNFL/GCL, GCL/IPL, IPL/INL, INL/OPL, ELM, IS/OS, OS/RPE, BM -- i.e. every "
             "depth index feeds b(idx+1) and this vendor supplies all nine. Pre-D84 idx3 "
             "was dropped as having 'no official counterpart' and idx4 fed b4, which the "
             "E11 §⑩ model bridge caught from the other side: against the container's "
             "REAL Maestro2 ground truth seven of the eight fed boundaries were within "
             "±1 px and only the one fed from idx4 was 12 px deep -- 12 px is one INL "
             "thickness. This is the ONLY row with an official-GT anchor"),
    "aireadi::Topcon_Triton": DatasetMapping(
        key="aireadi::Topcon_Triton", order="depth",
        boundaries=(0, 1, 2, 3, 4, 5, 6, 7, 8, None),
        note="D84/γ. Same ten-surface set as Maestro2 for idx0..8 (same cross-device "
             "identification), so every depth index feeds b(idx+1). The tenth surface, "
             "labelled CSI, is the choroid/sclera interface -- not an official boundary "
             "under any reading, and degenerate with the ninth anyway (75.7% of columns "
             "bit-equal, 97.5% within 2 px). Dropped"),
    "aireadi::Zeiss_Cirrus": DatasetMapping(
        key="aireadi::Zeiss_Cirrus", order="depth",
        boundaries=(0, 8),
        # +9 = DEEPER. The stored surface is 9 rows ABOVE where 8|9 belongs.
        offsets_px=((8, 9),),
        note="unchanged by D84 -- it draws none of the interfaces γ moved. Ships only "
             "ILM (->b1) and a surface whose CodeMeaning is 'Surface of the center of "
             "the RPE'; measured 9 px (17.6 um) shallower than BM on the 144-image tune "
             "pool, a documented systematic offset, so after ``offsets_px`` it stands in "
             "for b9 (BM)"),
}


def pack_interval(lo: np.ndarray | int, hi: np.ndarray | int) -> np.ndarray:
    """``[lo, hi]`` -> the one-byte code. Vectorised; does not validate."""
    return (np.asarray(lo) * 10 + np.asarray(hi)).astype(np.uint8)


def interval_lut() -> tuple[np.ndarray, np.ndarray]:
    """256-entry ``(lo, hi)`` lookup tables, int8, with ``-1`` for every invalid code."""
    lo = np.full(256, -1, dtype=np.int8)
    hi = np.full(256, -1, dtype=np.int8)
    for a in range(NUM_CLASSES):
        for b in range(a, NUM_CLASSES):
            lo[10 * a + b] = a
            hi[10 * a + b] = b
    return lo, hi


def unpack_interval(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(H, W)`` uint8 codes -> two int8 planes, ``-1`` where nothing is known."""
    lo_lut, hi_lut = interval_lut()
    code = np.asarray(code, dtype=np.uint8)
    return lo_lut[code], hi_lut[code]


SYNTHETIC_REAL_BOUNDARIES = (1, 4, 7, 8, 9)

#: The other four synthetic lines, which are interpolations rather than real interfaces.
SYNTHETIC_FILL_BOUNDARIES = (2, 3, 5, 6)

#: Which OFFICIAL boundary each real synthetic line is, as a choice a config makes.
SYNTHETIC_BOUNDARY_MAP_PRESETS: dict[str, dict[int, int | None]] = {
    #: The evidence-based reading, from the cross-device comparison and the model bridge.
    "gamma": {1: 1, 4: 5, 7: 7, 8: 8, 9: 9},
}


class SyntheticBoundaryMapError(ValueError):
    """The synthetic->official boundary table is missing or is not a depth-ordered map."""


@dataclass(frozen=True)
class SyntheticBoundaryMap:
    """One reading of "which official boundary is each real synthetic line".
    Holds the chosen reading and its derived class partitions.
    """

    name: str
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise SyntheticBoundaryMapError(
                f"{self.name}: the table supervises no boundary at all")
        syn = [s for s, _ in self.pairs]
        off = [o for _, o in self.pairs]
        for name_, values in (("synthetic", syn), ("official", off)):
            if not all(1 <= v <= NUM_BOUNDARIES for v in values):
                raise SyntheticBoundaryMapError(
                    f"{self.name}: {name_} boundary numbers must be 1..{NUM_BOUNDARIES}, "
                    f"got {values}")
            if any(b <= a for a, b in zip(values, values[1:])):
                raise SyntheticBoundaryMapError(
                    f"{self.name}: {name_} boundary numbers must be strictly increasing "
                    f"(boundaries are depth-ordered), got {values}")
        stray = sorted(set(syn) - set(SYNTHETIC_REAL_BOUNDARIES))
        if stray:
            raise SyntheticBoundaryMapError(
                f"{self.name}: synthetic boundaries {stray} are equal-thirds fills with no "
                f"anatomy behind them (LABELS.md §2); only {list(SYNTHETIC_REAL_BOUNDARIES)} "
                "may be mapped onto an official boundary")

    @property
    def synthetic_groups(self) -> tuple[tuple[int, ...], ...]:
        """Synthetic classes, grouped by the lines this table keeps."""
        return class_groups_from_cuts([s for s, _ in self.pairs])

    @property
    def official_groups(self) -> tuple[tuple[int, ...], ...]:
        """Official classes, grouped by the boundaries this table supervises."""
        return class_groups_from_cuts([o for _, o in self.pairs])

    @property
    def num_groups(self) -> int:
        return len(self.pairs) + 1

    @property
    def official_intervals(self) -> tuple[tuple[int, int], ...]:
        """``(lo, hi)`` official classes per synthetic group, in depth order."""
        return tuple((g[0], g[-1]) for g in self.official_groups)

    @property
    def supervised_official(self) -> tuple[int, ...]:
        return tuple(o for _, o in self.pairs)

    @property
    def unsupervised_official(self) -> tuple[int, ...]:
        got = set(self.supervised_official)
        return tuple(b for b in range(1, NUM_BOUNDARIES + 1) if b not in got)

    def describe(self) -> str:
        """One line for the run log and the resolved-config snapshot."""
        arrows = " ".join(f"b{s}->b{o}" for s, o in self.pairs)
        return (f"{self.name}: {arrows} | official boundaries unsupervised: "
                + ",".join(f"b{b}" for b in self.unsupervised_official))


def class_groups_from_cuts(cuts: Sequence[int],
                           num_classes: int = NUM_CLASSES) -> tuple[tuple[int, ...], ...]:
    """Classes ``0..num_classes-1`` cut at boundary numbers ``cuts`` (``b_n`` splits n-1|n)."""
    cuts = sorted(int(c) for c in cuts)
    if any(not 1 <= c <= num_classes - 1 for c in cuts):
        raise SyntheticBoundaryMapError(
            f"boundary numbers must be 1..{num_classes - 1}, got {cuts}")
    edges = [0, *cuts, num_classes]
    return tuple(tuple(range(a, b)) for a, b in zip(edges[:-1], edges[1:]))


def resolve_synthetic_boundary_map(spec) -> SyntheticBoundaryMap:
    """``data.official_synthetic_boundary_map`` -> a :class:`SyntheticBoundaryMap`.
    An unknown name raises: a silently defaulted map relabels every layer.
    """
    if isinstance(spec, SyntheticBoundaryMap):
        return spec
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        raise SyntheticBoundaryMapError(
            "data.official_synthetic_boundary_map is unset. It says which OFFICIAL boundary "
            "each real line of the synthetic release actually is, and the answer is not "
            f"set: choose {sorted(SYNTHETIC_BOUNDARY_MAP_PRESETS)} or an explicit mapping.")
    if isinstance(spec, str):
        key = spec.strip()
        if key not in SYNTHETIC_BOUNDARY_MAP_PRESETS:
            raise SyntheticBoundaryMapError(
                f"unknown synthetic boundary map preset {key!r}; known: "
                f"{sorted(SYNTHETIC_BOUNDARY_MAP_PRESETS)}")
        table = SYNTHETIC_BOUNDARY_MAP_PRESETS[key]
        name = key
    elif isinstance(spec, Mapping):
        table = spec
        name = "custom"
    else:
        raise SyntheticBoundaryMapError(
            f"data.official_synthetic_boundary_map must be a preset name or a mapping, "
            f"got {type(spec).__name__}")
    pairs = tuple(sorted((int(s), int(o)) for s, o in table.items() if o is not None))
    return SyntheticBoundaryMap(name=name, pairs=pairs)


def _lut_from_groups(groups: Sequence[Sequence[int]], values: Sequence) -> np.ndarray:
    lut = np.full(256, IGNORE_UINT8, dtype=np.uint8)
    for group, value in zip(groups, values):
        for c in group:
            lut[c] = value
    return lut


def official_intervals_from_synthetic(labels: np.ndarray,
                                      bmap: SyntheticBoundaryMap) -> np.ndarray:
    """A synthetic ten-class plane -> the OFFICIAL-class interval plane it actually supports.
    The one place a synthetic label is reinterpreted; every caller goes through it.
    """
    lut = _lut_from_groups(bmap.synthetic_groups,
                           [int(pack_interval(lo, hi)) for lo, hi in bmap.official_intervals])
    return lut[np.asarray(labels, dtype=np.uint8)]


def merge_synthetic_labels(labels: np.ndarray, bmap: SyntheticBoundaryMap) -> np.ndarray:
    """A synthetic ten-class plane -> the scoring label space (one id per retained group)."""
    return _lut_from_groups(bmap.synthetic_groups, range(bmap.num_groups))[
        np.asarray(labels, dtype=np.uint8)]


def merge_official_labels(labels: np.ndarray, bmap: SyntheticBoundaryMap) -> np.ndarray:
    """An OFFICIAL ten-class plane (a prediction) -> the same scoring label space."""
    return _lut_from_groups(bmap.official_groups, range(bmap.num_groups))[
        np.asarray(labels, dtype=np.uint8)]


def exact_from_code(code: np.ndarray) -> np.ndarray:
    """``(H, W)`` uint8 codes -> the classic label plane: exact class, or 255."""
    lo, hi = unpack_interval(code)
    out = np.full(lo.shape, IGNORE_UINT8, dtype=np.uint8)
    exact = (lo >= 0) & (lo == hi)
    out[exact] = lo[exact].astype(np.uint8)
    return out


def offsets_to_json(offsets_px: tuple[tuple[int, int], ...]) -> list[list[int]]:
    """:attr:`DatasetMapping.offsets_px` in the form ``index.json`` stores."""
    return [[int(b), int(dv)] for b, dv in offsets_px]


def offsets_from_json(blob: object) -> tuple[tuple[int, int], ...]:
    """The inverse of :func:`offsets_to_json`; a missing key means "no offsets"."""
    if blob is None:
        return ()
    return tuple((int(b), int(dv)) for b, dv in blob)


def surfaces_to_json(boundaries: tuple[int | None, ...]) -> list[int | None]:
    """:attr:`DatasetMapping.boundaries` in the form ``index.json`` stores."""
    return [None if b is None else int(b) for b in boundaries]


def surfaces_from_json(blob: object) -> tuple[int | None, ...] | None:
    """The inverse of :func:`surfaces_to_json`; ``None`` means the key was absent."""
    if blob is None:
        return None
    return tuple(None if b is None else int(b) for b in blob)


def apply_boundary_offsets(
    rows: np.ndarray,
    avail: np.ndarray,
    height: int,
    offsets: dict[int, int],
    *,
    key: str = "",
    max_collapse_fraction: float = OFFSET_COLLAPSE_MAX_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shift annotated boundary rows by signed per-boundary pixel offsets, with guards.
    Raises rather than clipping: a clipped line rasterises cleanly and teaches the wrong one.
    """
    rows = np.asarray(rows, dtype=np.float64)
    avail = np.asarray(avail, dtype=bool)
    n_b = rows.shape[0]
    width = rows.shape[1]
    who = f"{key}: " if key else ""
    audit = {
        "offsets_px": offsets_to_json(tuple(sorted((int(b), int(v))
                                                   for b, v in offsets.items()))),
        "n_measured": 0, "n_collapsed": 0, "n_inverted": 0,
        "frac_collapsed": None,
        "max_collapse_fraction": float(max_collapse_fraction),
    }
    if not offsets:
        return rows, avail, audit

    shifted = rows.copy()
    touched = np.zeros(n_b, dtype=bool)
    for b, dv in offsets.items():
        b = int(b)
        if not 0 <= b < n_b:
            raise ValueError(f"{who}offset boundary {b} is not in 0..{n_b - 1}")
        shifted[b] = rows[b] + float(dv)
        touched[b] = True

    # ---- guard 1: the corrected line must still be inside the image ------------------
    drawn = avail & np.isfinite(rows)
    out_of_frame = drawn & touched[:, None] & ~(
        (shifted >= 0.0) & (shifted < float(height)))
    n_out = int(out_of_frame.sum())
    if n_out:
        b_bad, c_bad = (int(x[0]) for x in np.where(out_of_frame))
        raise BoundaryOffsetViolation(
            f"{who}offsets {offsets_to_json(tuple(sorted(offsets.items())))} push {n_out} "
            f"annotated row(s) outside [0, {height}) -- e.g. boundary {b_bad} column "
            f"{c_bad} moves {rows[b_bad, c_bad]:.1f} -> {shifted[b_bad, c_bad]:.1f}. A "
            f"boundary drawn off the image labels one whole side of it, and the plane "
            f"rasterises without complaint.")

    # ---- guards 2 & 3: walk depth order, each surface against the one above it -------
    prev_before = np.full(width, np.nan)
    prev_after = np.full(width, np.nan)
    prev_touched = np.zeros(width, dtype=bool)
    n_measured = n_collapsed = n_inverted = 0
    first_inversion: tuple[int, int] | None = None
    for k in range(n_b):
        here = drawn[k]
        # Only pairs the offset could have changed are measured.
        pair = here & np.isfinite(prev_before) & (prev_touched | touched[k])
        if pair.any():
            t_before = rows[k] - prev_before
            t_after = shifted[k] - prev_after
            new_inv = pair & (t_after < 0.0) & (t_before >= 0.0)
            new_zero = pair & (t_after <= 0.0) & (t_before > 0.0)
            n_measured += int(pair.sum())
            n_inverted += int(new_inv.sum())
            n_collapsed += int(new_zero.sum())
            if first_inversion is None and new_inv.any():
                first_inversion = (k, int(np.flatnonzero(new_inv)[0]))
        prev_before = np.where(here, rows[k], prev_before)
        prev_after = np.where(here, shifted[k], prev_after)
        prev_touched = np.where(here, touched[k], prev_touched)

    audit["n_measured"] = n_measured
    audit["n_inverted"] = n_inverted
    audit["n_collapsed"] = n_collapsed
    audit["frac_collapsed"] = (n_collapsed / n_measured) if n_measured else None

    if n_inverted:
        k_bad, c_bad = first_inversion                                # type: ignore[misc]
        raise BoundaryOffsetViolation(
            f"{who}offsets {offsets_to_json(tuple(sorted(offsets.items())))} invert the "
            f"depth order on {n_inverted} of {n_measured} measured column-pair(s) -- e.g. "
            f"boundary {k_bad} column {c_bad}. rasterize_partial DROPS crossing columns, "
            f"so this would show up as supervision quietly going missing and be blamed on "
            f"the source.")
    if n_measured and n_collapsed / n_measured > float(max_collapse_fraction):
        raise BoundaryOffsetViolation(
            f"{who}offsets {offsets_to_json(tuple(sorted(offsets.items())))} squeeze a "
            f"supervised interval to zero thickness on {n_collapsed} of {n_measured} "
            f"measured column-pair(s) ({n_collapsed / n_measured:.4f} > "
            f"{float(max_collapse_fraction)}). That is D76's failure -- a constant pixel "
            f"shift larger than the layer it has to fit through -- and the labels it "
            f"produces teach b(k) == b(k+1) while rasterising and training cleanly. "
            f"Measure the offset on this geometry before applying it here.")
    return shifted, avail, audit


def rasterize_partial(
    rows: np.ndarray,
    avail: np.ndarray,
    height: int,
    *,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, dict]:
    """``(9, W)`` boundary rows + availability -> the ``(H, W)`` uint8 code plane."""
    n_b = num_classes - 1
    rows = np.asarray(rows, dtype=np.float64)
    avail = np.asarray(avail, dtype=bool)
    if rows.shape != avail.shape or rows.ndim != 2 or rows.shape[0] != n_b:
        raise ValueError(f"rows and avail must both be ({n_b}, W); got {rows.shape} "
                         f"and {avail.shape}")
    height = int(height)
    width = rows.shape[1]

    avail = avail & np.isfinite(rows)
    b = np.where(avail, rows, np.nan)

    # A crossing is "some annotated boundary sits above an annotated shallower one".
    with np.errstate(invalid="ignore"):
        ordered = np.ones(width, dtype=bool)
        last = np.full(width, -np.inf)
        for k in range(n_b):
            here = avail[k]
            ordered &= ~(here & (b[k] < last))
            last = np.where(here, b[k], last)
    dropped = int((~ordered).sum())
    avail = avail & ordered[None, :]

    y = np.arange(height, dtype=np.float64)[:, None, None]                 # (H,1,1)
    bb = np.where(avail, b, 0.0)[None]                                     # (1,9,W)
    av = avail[None]                                                       # (1,9,W)
    below = av & (y >= bb)                                                 # (H,9,W)
    above = av & (y < bb)

    idx = np.arange(n_b)[None, :, None]
    j_last = np.where(below, idx, -1).max(axis=1)                          # (H,W)
    j_first = np.where(above, idx, n_b).min(axis=1)                        # (H,W)

    lo = (j_last + 1).astype(np.int16)
    hi = j_first.astype(np.int16)
    code = pack_interval(lo, hi)
    code[(lo == 0) & (hi == n_b)] = IGNORE_UINT8

    n_pix = float(height * width) or 1.0
    stats = {
        "width": width,
        "columns_dropped_crossing": dropped,
        "pixels_no_information": int((code == IGNORE_UINT8).sum()),
        "pixels_exact": int((lo == hi).sum()),
        "fraction_exact": round(float((lo == hi).sum()) / n_pix, 4),
    }
    return code, stats


def render_mapping_table() -> str:
    """The mapping as one deterministic text block, for pasting into the decisions log."""
    names = [f"{k}|{k + 1}" for k in range(NUM_BOUNDARIES)]
    lines = ["dataset                          | official boundaries it supervises",
             "---------------------------------|----------------------------------"]
    for key in sorted(SURFACE_TO_BOUNDARY):
        m = SURFACE_TO_BOUNDARY[key]
        have = " ".join(names[b] for b in m.available)
        lines.append(f"{key:<32s} | {have}")
    return "\n".join(lines)
