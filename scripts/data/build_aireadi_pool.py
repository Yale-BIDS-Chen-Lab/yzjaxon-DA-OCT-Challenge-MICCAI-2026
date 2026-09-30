"""Build the licensed local AI-READI labelled pool."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from octtta import paths
from octtta.data.partial_labels import (
    MAPPING_VERSION, SURFACE_TO_BOUNDARY, offsets_to_json,
    rasterize_partial, surfaces_to_json,
)
from scripts.data import aireadi_common as aireadi
from scripts.data import aireadi_common as pretrain_pool
from scripts.data.build_public_pool import (
    _surface_rows, _boundaries_lost_to_sentinels, _accumulate_offset_audit,
    _write_pair, write_pool_index, SURFACE_MIN_VALID_COLUMN_FRAC,
)

AIREADI_VENDORS = ("Heidelberg_Spectralis", "Topcon_Maestro2",
                   "Topcon_Triton", "Zeiss_Cirrus")
VOLUME_TAG_HEX = aireadi.VOLUME_TAG_HEX
volume_tag = aireadi.volume_tag
MAX_ESCAPED_DROP_FRAC = 0.05
MAX_OFFSET_DROP_FRAC = 0.005

def aireadi_dataset_dirs(vendors: Sequence[str]) -> list[str]:
    """The DIRECTORY names an ``--datasets aireadi`` run writes, for the vendors it asks for;
    expanding to all four would refuse every parallel per-vendor job after the first."""
    return [f"aireadi::{v}" for v in vendors]


def stem_for(rec, frame: int) -> str:
    """The one place an AI-READI stem is spelled. The digest sits BEFORE the frame field,
    because readers take the LAST field as the frame."""
    protocol = rec.protocol.replace(", ", "_").replace(" ", "")
    return (f"{rec.vendor}__{protocol}__{rec.status_proxy}__{rec.person_id}"
            f"__{rec.laterality or 'NA'}__{volume_tag(rec.structural_path)}"
            f"__f{int(frame):04d}")


class _VolumeUnusable(Exception):
    """A volume whose own data contradicts the mapping's premise: dropped, not repaired."""


def _aireadi_stats() -> dict:
    # Each numerator ships with its denominator: "0 masked" and "nothing looked at" differ.
    return {"volumes": 0, "volumes_dropped": 0, "volumes_dropped_escaped": 0,
            "volumes_dropped_offset": 0, "persons": 0,
            "images": 0, "labels": 0,
            "frames_failed_qc": 0, "frames_without_annotation": 0,
            # Per-criterion split of ``frames_failed_qc``; the total is asserted to be the sum.
            "frames_failed_qc_by_reason": {r: 0 for r in pretrain_pool.QC_REASONS},
            "frames_failed_qc_no_retina": 0,
            # IMAGE fine but the height map lost a whole declared surface to sentinels.
            "frames_without_surface_ilm": 0,
            "frames_without_surface_by_boundary": {},
            "columns_dropped_crossing": 0, "heights_total": 0, "heights_sentinel": 0,
            "offset_columns_measured": 0, "offset_columns_collapsed": 0}


def _build_aireadi_volume(rec, root: Path, out_dir: Path, key: str, *,
                          frames_per_volume: int, stats: dict,
                          seen_stems: set[str]) -> list[dict]:
    """One AI-READI volume -> its selected B-scans, written as image/interval pairs."""
    mapping = SURFACE_TO_BOUNDARY[key]
    heights, valid, labels = aireadi.read_heightmap(root / rec.seg_path, rec.vendor)

    # One boundary slot per stored surface. WARNING: a different surface count means the
    # mapping row describes a different volume, and every layer below shifts by one.
    if heights.shape[0] != len(mapping.boundaries):
        raise ValueError(
            f"{key}: {rec.seg_path} has {heights.shape[0]} height-map surfaces but "
            f"SURFACE_TO_BOUNDARY[{key!r}] describes {len(mapping.boundaries)}. The "
            "mapping is stale; measure before building against it")
    order = aireadi.depth_ordered_surfaces(rec.vendor, labels)

    n_struct, h_struct, w_struct = aireadi.structural_geometry(root / rec.structural_path)
    _, n_rows, n_cols = heights.shape
    if (n_rows, n_cols) != (n_struct, w_struct):
        raise ValueError(
            f"{key}: grid mismatch -- the height map is {n_rows}x{n_cols} but the paired "
            f"structural volume has {n_struct} frames of width {w_struct} "
            f"({rec.seg_path}); the pairing is meaningless")

    # Sentinels are masked per vendor; the masking is asserted by its converse below.
    in_range = (heights >= 0.0) & (heights < float(h_struct))
    escaped = int((valid & ~in_range).sum())
    if escaped:
        # One bad annotation is dropped, but a sentinel-encoding change must not hide in that.
        stats["volumes_dropped_escaped"] = int(stats.get("volumes_dropped_escaped", 0)) + 1
        raise _VolumeUnusable(
            f"{key}: {escaped} height(s) survived the {rec.vendor} sentinel mask "
            f"{aireadi.SENTINEL_BOUNDS[rec.vendor]} but fall outside [0, {h_struct}) "
            f"({rec.seg_path})")
    stats["heights_total"] += int(valid.size)
    stats["heights_sentinel"] += int((~valid).sum())
    depth = np.where(valid, heights, np.nan).astype(np.float64)

    # Asked of the WHOLE volume first: per frame it would raise after writing orphan PNGs.
    for surf_i, target in zip(order, mapping.boundaries):
        if target is None or target not in mapping.offsets:
            continue
        moved = depth[surf_i] + float(mapping.offsets[target])
        bad = int((np.isfinite(moved)
                   & ~((moved >= 0.0) & (moved < float(h_struct)))).sum())
        if bad:
            stats["volumes_dropped_offset"] = int(
                stats.get("volumes_dropped_offset", 0)) + 1
            raise _VolumeUnusable(
                f"{key}: the +{mapping.offsets[target]} px correction on boundary "
                f"{target} pushes {bad} height(s) outside [0, {h_struct}) "
                f"({rec.seg_path}); this volume's deep surface sits against the edge of "
                f"its own image, so the correction cannot be drawn there")

    # Surface i must be the i-th deepest. A volume that contradicts it is dropped, never
    # reordered, because a reordering we invent is not an annotation.
    med: list[float] = []
    for surf_i, target in zip(order, mapping.boundaries):
        if target is None:
            continue
        col = depth[surf_i][np.isfinite(depth[surf_i])]
        med.append(float(np.median(col)) if col.size else float("nan"))
    known = [m for m in med if m == m]
    if any(b < a for a, b in zip(known, known[1:])):
        raise _VolumeUnusable(
            f"median depths of the mapped surfaces are not increasing "
            f"({[round(m, 1) for m in med]}); the depth order the mapping assumes does "
            f"not hold here")

    take = pretrain_pool.choose_pretrain_frames(n_struct, max_take=frames_per_volume)
    entries: list[dict] = []
    for f in take:
        image = aireadi.read_structural_frame(root / rec.structural_path, f)
        if image.shape != (h_struct, w_struct):
            raise ValueError(
                f"{key}: frame {f} of {rec.structural_path} decoded to {image.shape} but "
                f"the header says {(h_struct, w_struct)}")
        reason = pretrain_pool.frame_qc_reason(image)
        if reason is not None:
            # One counter per criterion so the total stays decomposable.
            stats["frames_failed_qc"] += 1
            stats["frames_failed_qc_by_reason"][reason] = (
                stats["frames_failed_qc_by_reason"].get(reason, 0) + 1)
            if reason in pretrain_pool.QC_NO_RETINA_REASONS:
                stats["frames_failed_qc_no_retina"] += 1
            continue
        rows_, avail, off = _surface_rows(key, [depth[i, f] for i in order], w_struct,
                                          h_struct)
        _accumulate_offset_audit(stats, off)
        if not avail.any():
            # Every column is sentinel: a plane that is 255 everywhere teaches nothing.
            stats["frames_without_annotation"] += 1
            continue
        gone = _boundaries_lost_to_sentinels(key, avail)
        if gone:
            # WARNING: the image passed every pixel gate and the plane still supervises FEWER
            # boundaries than the mapping row promises (measured: 88 Cirrus frames).
            stats["frames_without_surface_ilm"] += 1
            for b in gone:
                key_b = str(int(b))
                stats["frames_without_surface_by_boundary"][key_b] = (
                    stats["frames_without_surface_by_boundary"].get(key_b, 0) + 1)
            continue
        code, st = rasterize_partial(rows_, avail, h_struct)
        stats["columns_dropped_crossing"] += st["columns_dropped_crossing"]

        stem = stem_for(rec, f)
        if stem in seen_stems:
            # Reachable only through a genuine 48-bit digest collision.
            raise ValueError(f"{key}: stem {stem!r} is already taken; _write_pair would "
                             "overwrite an earlier volume's pair in silence")
        seen_stems.add(stem)
        _write_pair(out_dir, stem, image, code)
        stats["images"] += 1
        stats["labels"] += 1
        entries.append({
            "stem": stem, "image": f"{stem}-image.png",
            "labels": {"1": f"{stem}-label.png"},
            "device": rec.vendor, "status": rec.status_proxy,
            # Grouped by PERSON, not volume: a volume-level split would leak across eyes.
            "group": rec.person_id, "volume": volume_tag(rec.structural_path),
            "volume_tag": volume_tag(rec.structural_path),
            "person_id": rec.person_id, "protocol": rec.protocol,
            "anatomy": rec.anatomy, "laterality": rec.laterality, "frame": int(f),
            "seg_path": rec.seg_path, "structural_path": rec.structural_path,
        })
    return entries


def _assert_stems_unique(chosen) -> None:
    """Refuse duplicate volume stems before writing any image."""
    prefixes=[stem_for(rec,0)[:-len("__f0000")] for rec in chosen]
    if len(prefixes)!=len(set(prefixes)):
        raise ValueError("AI-READI selected volumes contain duplicate stem prefixes")



def build_aireadi(out_root: Path, data_root: Path, limit: int | None = None, *,
                  per_cell: int = 100000, frames_per_volume: int = 16,
                  seed: int = 20260823,
                  max_per_person_per_cell: int = 100000,
                  include_optic_disc: bool = True,
                  vendors: Sequence[str] = AIREADI_VENDORS) -> list[dict]:
    """Build the fixed donor selection, preserving local stems and frame order."""
    root=Path(data_root)/"public"/"ai_readi"
    ex=aireadi.Exclusions.load(root)
    records=aireadi.index_octa(root,include_optic_disc=include_optic_disc)
    donors=[r for r in records if not ex.is_never_train(r.person_id,r.structural_path)]
    if not donors:
        raise ValueError("AI-READI donor selection is empty")
    chosen=aireadi.balanced_sample(donors,per_cell=per_cell,seed=seed,
                                   max_per_person_per_cell=max_per_person_per_cell)
    # Apply the quality list after the whole four-vendor sample and before any vendor slice.
    quality_by_vendor={v:0 for v in AIREADI_VENDORS}
    kept=[]
    for rec in chosen:
        if ex.excludes_labelled_quality(rec.structural_path,rec.vendor):
            quality_by_vendor[rec.vendor]+=1
        else:
            kept.append(rec)
    chosen=kept[:limit] if limit else kept
    want=tuple(vendors)
    if not want or len(set(want))!=len(want) or any(v not in AIREADI_VENDORS for v in want):
        raise ValueError("vendors must be a nonempty distinct subset of the four models")
    if set(want)!=set(AIREADI_VENDORS):
        if limit:
            raise ValueError("limit cannot be combined with a vendor subset")
        chosen=[r for r in chosen if r.vendor in want]
    _assert_stems_unique(chosen)
    parts={v:{"entries":[],"stats":_aireadi_stats(),"records":[],"failures":0}
           for v in want}
    seen_stems=set()
    for rec in chosen:
        part=parts[rec.vendor]
        part["stats"]["volumes"]+=1
        try:
            built=_build_aireadi_volume(
                rec,root,Path(out_root)/f"aireadi::{rec.vendor}",
                f"aireadi::{rec.vendor}",frames_per_volume=frames_per_volume,
                stats=part["stats"],seen_stems=seen_stems)
        except _VolumeUnusable:
            part["stats"]["volumes_dropped"]+=1
            part["failures"]+=1
            continue
        except Exception as exc:
            raise RuntimeError(f"AI-READI volume build failed ({type(exc).__name__}); "
                               "inspect the local licensed input") from None
        part["entries"].extend(built)
        part["records"].append(rec)
    for vendor,part in parts.items():
        stats=part["stats"]
        if stats["volumes"] and stats["volumes_dropped_escaped"]/stats["volumes"]>MAX_ESCAPED_DROP_FRAC:
            raise ValueError(f"{vendor}: escaped height-map rate exceeds threshold")
        if stats["volumes"] and stats["volumes_dropped_offset"]/stats["volumes"]>MAX_OFFSET_DROP_FRAC:
            raise ValueError(f"{vendor}: offset height-map rate exceeds threshold")
    written=[e for part in parts.values() for e in part["entries"]]
    n_never=sum(ex.is_never_train(e["person_id"],e["structural_path"]) for e in written)
    if n_never:
        raise AssertionError(f"{n_never} never-train AI-READI frames were written")
    n_off_split=sum(r.split!="tune" for part in parts.values() for r in part["records"])
    if n_off_split:
        raise AssertionError(f"{n_off_split} AI-READI volumes have unexpected split")
    reports=[]
    for vendor in want:
        part=parts[vendor]
        key=f"aireadi::{vendor}"
        part["stats"]["persons"]=len({e["person_id"] for e in part["entries"]})
        if SURFACE_TO_BOUNDARY[key].offsets_px and part["entries"] and not part["stats"]["offset_columns_measured"]:
            raise AssertionError(f"{key}: offset audit measured no columns")
        reports.append({
            "dataset":key,"dir":str(Path(out_root)/key),
            "entries":part["entries"],"stats":part["stats"],
            "index_extra":{
                "selection":{
                    "per_cell":per_cell,"seed":seed,
                    "max_per_person_per_cell":max_per_person_per_cell,
                    "include_optic_disc":bool(include_optic_disc),
                    "frames_per_volume":frames_per_volume,
                    "frame_rule":"evenly spaced, first and last included",
                    "qc_surface_min_valid_column_frac":SURFACE_MIN_VALID_COLUMN_FRAC,
                    "surface_depth_order":list(aireadi.DEPTH_ORDER[vendor]),
                    "surface_basis":aireadi.SELECTION_BASIS[vendor],
                    "sentinel_bounds":list(aireadi.SENTINEL_BOUNDS[vendor]),
                },
                "excluded_persons":{
                    "exclusions_sha256":ex.config_sha256,
                    "test_split_persons":len(ex.test_split),
                    "never_train_volumes":len(ex.never_train),
                },
                "labelled_quality":{"excluded_here":quality_by_vendor[vendor]},
                "cohorts":aireadi.cohort_counts(part["records"]),
                "volumes_dropped":part["failures"],
            },
        })
    return reports



def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=paths.PARTIAL_POOL_D88_DIR)
    p.add_argument("--data-root",type=Path,default=paths.DATA_DIR)
    p.add_argument("--vendors",default=",".join(AIREADI_VENDORS))
    p.add_argument("--limit",type=int,default=None)
    args=p.parse_args(argv)
    want=tuple(x.strip() for x in args.vendors.split(",") if x.strip())
    for vendor in want:
        dest=args.out/f"aireadi::{vendor}"
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"AI-READI output directory already contains data: {vendor}")
    reports=build_aireadi(args.out,args.data_root,args.limit,vendors=want)
    for report in reports:
        write_pool_index(report)
        print(f"{report['dataset']}: {report['stats']['images']} images")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
