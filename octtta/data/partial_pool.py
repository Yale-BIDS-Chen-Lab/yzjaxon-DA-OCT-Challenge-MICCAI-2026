"""Read the Final partial-label pool; identity checks run during reproduction checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from octtta import paths
from octtta.data import partial_labels
from octtta.data.partial_labels import SURFACE_TO_BOUNDARY
from octtta.data.release_dataset import Sample

OFFSETS_MUST_MATCH_MAPPING = "must_match_mapping"

_DONOR_VERIFIED_PREFIX = "aireadi"


def is_aireadi_dataset(name: str) -> bool:
    """Is *name* a partial-pool directory whose frames come out of the AI-READI release?"""
    return name.split("::", 1)[0].startswith(_DONOR_VERIFIED_PREFIX)


def _root(root: Path | str | None = None) -> Path:
    """Which built pool to read. ``None`` means :data:`octtta.paths.PARTIAL_POOL_D88_DIR`."""
    return Path(root) if root is not None else Path(paths.PARTIAL_POOL_D88_DIR)


def dataset_of(sample) -> str:
    """Which built directory a partial sample came from."""
    if sample.label_kind == "exact":
        return "OFFICIAL"
    return sample.stem.split("__", 1)[0]


def cell_key(sample) -> str:
    """The ``device|protocol`` cell a sample belongs to -- the unit ``cell_balance`` moves."""
    if sample.label_kind == "exact":
        return "OFFICIAL"
    return f"{sample.device}|{sample.protocol or dataset_of(sample)}"


#: ``__v<12 hex volume digest>__f<frame>`` at the END of a built stem.
_VOLUME_FRAME_RE = re.compile(r"__v(?P<vol>[0-9a-f]{12})__f(?P<frame>\d+)(?:__g\d+)?$")


def _volume_and_frame(sample, cell: str) -> tuple[str, int]:
    """``(volume digest, frame number)`` for one built partial-label sample."""
    m = _VOLUME_FRAME_RE.search(str(sample.stem))
    if m is None:
        raise ValueError(
            f"data.partial_pool.frame_cap names the cell {cell!r}, but the sample "
            f"record in it carries no '__v<12 hex>__f<frame>' volume/frame tag, so "
            f"there is no volume to count frames per. Only directories built by "
            f"scripts/repro.py build on or after 2026-08-30 carry the digest, "
            f"and the non-AI-READI sources (oct5k, jhu_hcms, duke_dme_2015) have no volume "
            f"identity to carry. Cap a cell whose frames belong to volumes, or drop the "
            f"cell with data.partial_pool.exclude_cells.")
    return m.group("vol"), int(m.group("frame"))


def no_frame_cap_record() -> dict:
    """The audit record of a run that declared no ``frame_cap`` at all."""
    return {"frame_cap": None, "capped": {"n": 0, "by_cell": {}}}


def validate_frame_cap(spec: object) -> dict[str, int] | None:
    """``data.partial_pool.frame_cap`` as ``{cell key: frames per volume}``, or ``None``."""
    if spec is None:
        return None
    if isinstance(spec, str):
        raise ValueError(
            f"data.partial_pool.frame_cap must be a MAPPING of 'device|protocol' cell keys "
            f"to a per-volume frame count; a bare string ({spec!r}) carries no count at all, "
            f"so nothing would be capped and the config would read as though something was.")
    if isinstance(spec, (list, tuple)):
        raise ValueError(
            f"data.partial_pool.frame_cap must be a MAPPING of 'device|protocol' cell keys "
            f"to a per-volume frame count ({{}} for none, null for 'not configured'); got a "
            f"{type(spec).__name__}, which is the shape of exclude_cells. A list says WHICH "
            f"cells and never HOW MANY frames -- the two knobs are not interchangeable, and "
            f"one written as the other is the mistake this refusal exists for.")
    if not isinstance(spec, dict):
        raise ValueError(
            f"data.partial_pool.frame_cap must be a mapping of 'device|protocol' cell keys "
            f"to a per-volume frame count ({{}} for none, null for 'not configured'); got "
            f"{type(spec).__name__}.")
    out: dict[str, int] = {}
    for cell, cap in spec.items():
        if not isinstance(cell, str):
            raise ValueError(
                f"data.partial_pool.frame_cap holds the key {cell!r} "
                f"({type(cell).__name__}); every key is a 'device|protocol' cell key, i.e. "
                f"a string.")
        if "|" not in cell:
            raise ValueError(
                f"data.partial_pool.frame_cap entry {cell!r} is not in the canonical "
                f"'device|protocol' form that octtta/data/partial_pool.py::cell_key writes "
                f"(e.g. 'Zeiss_Cirrus|Optic Disc, 6 x 6'). A device name on its own is not a "
                f"cell and would cap nothing.")
        # ``bool`` IS an ``int`` in Python, so a YAML ``true`` would otherwise sail through.
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise ValueError(
                f"data.partial_pool.frame_cap[{cell!r}] is {cap!r} ({type(cap).__name__}); a "
                f"per-volume frame cap is a positive integer. A float would truncate to a "
                f"depth nobody declared and a bool reads as 1.")
        if cap < 1:
            raise ValueError(
                f"data.partial_pool.frame_cap[{cell!r}] is {cap}; a cap of {cap} keeps no "
                f"frame of any volume, which empties the cell. 'Do not train this cell' is "
                f"data.partial_pool.exclude_cells, and it says so in its own audit record -- "
                f"a zero cap would report the cell as capped rather than excluded.")
        out[cell] = cap
    return out


def _evenly_spaced(n_total: int, keep: int) -> list[int]:
    """Indices of ``keep`` items spread evenly over ``n_total``, ascending."""
    return [(2 * i + 1) * n_total // (2 * keep) for i in range(keep)]


def apply_frame_cap(samples: list[Sample], frame_cap: object) -> tuple[list[Sample], dict]:
    """Thin each named cell to at most N frames per VOLUME, and report what went.
    Deterministic: the kept frames are evenly spaced, never sampled.
    """
    declared = validate_frame_cap(frame_cap)
    if declared is None:
        return list(samples), no_frame_cap_record()

    present: dict[str, int] = {}
    for s in samples:
        k = cell_key(s)
        present[k] = present.get(k, 0) + 1

    unknown = [c for c in declared if c not in present]
    if unknown:
        raise ValueError(
            f"data.partial_pool.frame_cap names {unknown}, which match no cell of the pool "
            f"this run loads. Cells present: {sorted(present)}. A key that matches nothing "
            f"caps nothing: the run would train the cell at its full per-volume depth while "
            f"its own config says it was thinned, and every count in pools.audit.json would "
            f"look exactly as intended. (A cell data.partial_pool.exclude_cells already "
            f"removed is gone before this runs, so naming one cell in both lands here too --"
            f" on purpose, rather than quietly resolving to 'excluded'.)")

    # Two passes, so the result does not depend on input order.
    per_volume: dict[str, dict[str, set[int]]] = {}
    for s in samples:
        cell = cell_key(s)
        if cell in declared:
            vol, frame = _volume_and_frame(s, cell)
            per_volume.setdefault(cell, {}).setdefault(vol, set()).add(frame)

    keep: dict[tuple[str, str], set[int]] = {}
    for cell, volumes in per_volume.items():
        cap = declared[cell]
        for vol, frames in volumes.items():
            ordered = sorted(frames)
            keep[(cell, vol)] = (
                set(ordered) if len(ordered) <= cap
                else {ordered[i] for i in _evenly_spaced(len(ordered), cap)})

    kept: list[Sample] = []
    for s in samples:
        cell = cell_key(s)
        if cell not in declared:
            kept.append(s)
            continue
        vol, frame = _volume_and_frame(s, cell)
        if frame in keep[(cell, vol)]:
            kept.append(s)

    surviving: dict[str, int] = {}
    for s in kept:
        c = cell_key(s)
        if c in declared:
            surviving[c] = surviving.get(c, 0) + 1

    by_cell: dict[str, dict] = {}
    for cell, cap in declared.items():
        volumes = per_volume.get(cell, {})
        n_kept = surviving.get(cell, 0)
        dropped = present[cell] - n_kept
        if dropped == 0:
            deepest = max((len(f) for f in volumes.values()), default=0)
            raise ValueError(
                f"data.partial_pool.frame_cap[{cell!r}]={cap} drops no frame at all: the "
                f"deepest of that cell's {len(volumes)} volumes holds {deepest} frames, so "
                f"the cap never binds. The run would train the cell whole while its config, "
                f"its [data] line and pools.audit.json all report a cap that was applied. "
                f"Write a cap below {deepest}, or drop the key.")
        by_cell[cell] = {"cap": cap, "volumes": len(volumes),
                         "kept": n_kept, "dropped": dropped}

    return kept, {"frame_cap": dict(declared),
                  "capped": {"n": sum(v["dropped"] for v in by_cell.values()),
                             "by_cell": by_cell}}


def format_frame_cap_line(record: dict) -> str | None:
    """The ``[data]`` line for a frame-cap record; ``None`` when none was configured."""
    declared = record.get("frame_cap")
    if declared is None:
        return None
    if not declared:
        return "partial pool frame_cap: {} declared, nothing capped"
    by_cell = record["capped"]["by_cell"]
    per = "; ".join(
        f"{c!r} <={by_cell[c]['cap']}/volume over {by_cell[c]['volumes']} volumes "
        f"({by_cell[c]['kept']} kept, -{by_cell[c]['dropped']})" for c in declared)
    return (f"partial pool frame_cap: {per} "
            f"(total -{record['capped']['n']} frames, dropped at load)")


def no_offsets_record() -> dict:
    """The offsets record of a summary built without one -- "nobody looked"."""
    return {"policy": None, "pool": None, "as_built": {}, "mapping_declares": {},
            "uncorrected": []}


def offsets_record(policy: str, pool_name: str | None,
                   as_built: dict[str, tuple[tuple[int, int], ...]]) -> dict:
    """What ``pools.audit.json`` stores about this pool's per-boundary corrections."""
    built = {k: partial_labels.offsets_to_json(v) for k, v in as_built.items() if v}
    declares: dict[str, list[list[int]]] = {}
    uncorrected: list[str] = []
    for name in sorted(as_built):
        m = SURFACE_TO_BOUNDARY.get(name)
        if m is None or not m.offsets_px:
            continue
        declares[name] = partial_labels.offsets_to_json(m.offsets_px)
        if as_built[name] != m.offsets_px:
            uncorrected.append(name)
    return {"policy": policy, "pool": pool_name, "as_built": built,
            "mapping_declares": declares, "uncorrected": uncorrected}


def _spell(pairs: list[list[int]]) -> str:
    return " ".join(f"{b}|{b + 1}{dv:+d}px" for b, dv in pairs)


def format_offsets_line(record: dict) -> str | None:
    """The ``[data]`` line for a run's per-boundary corrections, or ``None``."""
    if record.get("policy") is None:
        return None
    built, uncorrected = record["as_built"], record["uncorrected"]
    if not built and not uncorrected:
        return None
    parts = []
    if built:
        parts.append("applied " + "; ".join(f"{n} {_spell(p)}"
                                            for n, p in sorted(built.items())))
    if uncorrected:
        parts.append(
            "NOT applied (frozen pool " + str(record["pool"]) + ", loaded as built) "
            + "; ".join(f"{n} {_spell(record['mapping_declares'][n])}"
                        for n in uncorrected)
            + " -- these labels teach the UNCORRECTED line")
    return ("partial pool boundary offsets (+ = deeper, baked at build time): "
            + " | ".join(parts))


def available_datasets(root: Path | str | None = None) -> list[str]:
    """Dataset directories that have actually been built under ``root``."""
    base = _root(root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "index.json").is_file())


def load_partial_samples(
    root: Path | str | None = None,
    datasets: list[str] | None = None,
    *,
    graders: tuple[int, ...] = (1,),
    frame_cap: object = None,
) -> list[Sample]:
    """Every built partial-label B-scan, as ``Sample(label_kind="interval")``."""
    return load_partial_pool(root, datasets, graders=graders,
                             frame_cap=frame_cap)[0]


def load_partial_pool(
    root: Path | str | None = None,
    datasets: list[str] | None = None,
    *,
    graders: tuple[int, ...] = (1,),
    frame_cap: object = None,
) -> tuple[list[Sample], dict, dict]:
    """Load samples, frame-cap counts, and the boundary-offset audit."""
    base = _root(root)
    names = list(datasets) if datasets is not None else available_datasets(base)
    policy = OFFSETS_MUST_MATCH_MAPPING
    pool_name = "public_partial_labels_all16_d88"
    out: list[Sample] = []
    as_built: dict[str, tuple[tuple[int, int], ...]] = {}
    for name in names:
        samples, as_built[name] = _read_index(base, name, graders)
        out.extend(samples)
    kept, cap = apply_frame_cap(out, frame_cap)
    return kept, cap, offsets_record(policy, pool_name, as_built)


def _read_index(base: Path, name: str, graders: tuple[int, ...]) -> tuple[list[Sample], tuple[tuple[int, int], ...]]:
    """Read samples in their recorded order; historical restamping metadata is ignored."""
    index = base / name / "index.json"
    if not index.is_file():
        raise FileNotFoundError("Partial pool index is missing; run scripts/repro.py build.")
    blob = json.loads(index.read_text())
    mapping = SURFACE_TO_BOUNDARY[name]
    if tuple(blob.get("boundaries", ())) != mapping.available:
        raise ValueError("Partial pool boundary mapping differs; rebuild the pool.")
    built_off = partial_labels.offsets_from_json(blob.get("offsets_px"))
    if built_off != mapping.offsets_px:
        raise ValueError("Partial pool boundary offsets differ; rebuild the pool.")
    out: list[Sample] = []
    for entry in blob["entries"]:
        labels = entry["labels"]
        for grader in graders:
            filename = labels.get(str(grader))
            if filename is None:
                continue
            out.append(Sample(
                image=base / name / entry["image"], mask=base / name / filename,
                device=entry.get("device", "Heidelberg_Spectralis"),
                status=entry.get("status", "unknown"),
                stem=f"{name}__{entry['stem']}" + (f"__g{grader}" if len(graders) > 1 else ""),
                label_kind="interval", protocol=str(entry.get("protocol", "")),
            ))
    return out, built_off


def pool_summary(samples: list[Sample], *, cap: dict | None = None,
                 offsets: dict | None = None) -> dict:
    """Counts by source dataset, device and ``device|protocol`` cell, for the audit."""
    by_dataset: dict[str, int] = {}
    by_device: dict[str, int] = {}
    by_cell: dict[str, int] = {}
    for s in samples:
        name = dataset_of(s)
        by_dataset[name] = by_dataset.get(name, 0) + 1
        by_device[s.device] = by_device.get(s.device, 0) + 1
        cell = cell_key(s)
        by_cell[cell] = by_cell.get(cell, 0) + 1
    rec = no_frame_cap_record() if cap is None else cap
    return {"n": len(samples), "by_dataset": by_dataset, "by_device": by_device,
            "by_cell": dict(sorted(by_cell.items())),
            # Beside ``by_cell`` and not inside it: a capped cell is still present.
            "frame_cap": rec["frame_cap"],
            "capped": rec["capped"],
            # Which boundaries this run's labels were corrected on -- AS BUILT.
            "boundary_offsets": (no_offsets_record() if offsets is None else offsets)}
