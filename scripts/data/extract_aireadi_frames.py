"""Build the pinned local AI-READI manifest and extract its planned OCT frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from octtta import paths  # noqa: E402
from scripts.data import aireadi_common as common  # noqa: E402

MANIFEST_COLUMNS = (
    "dataset", "license_class", "person_id", "vendor", "model",
    "anatomic_region", "is_widefield", "laterality", "height", "width",
    "n_frames", "take_frames", "src_path", "npz_relpath",
)
SHARDS = 32
QC_SIDECAR_SCHEMA = 1


def expected_sha256() -> str:
    doc = json.loads((REPO_ROOT / "configs" / "expected.json").read_text())
    pin = ((doc.get("aireadi") or {}).get("pretrain_manifest") or {}).get("sha256")
    if not isinstance(pin, str) or len(pin) != 64:
        raise ValueError("expected.json lacks aireadi.pretrain_manifest.sha256")
    return pin


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(data_root: Path, out: Path, *, exclusions=None,
                   expected: str | None = None) -> dict[str, int | str]:
    """Write the original deterministic local TSV after verifying its frozen hash."""
    root = Path(data_root) / "public" / "ai_readi"
    ex = exclusions if exclusions is not None else common.Exclusions.load(root)
    volumes = common.aireadi_pretrain_volumes(root, exclusions=ex,
                                              max_take=common.DEFAULT_MAX_TAKE)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(MANIFEST_COLUMNS)
            for vol in volumes:
                if ex.is_never_train(vol.person_id, vol.src_path):
                    raise AssertionError("never-train volume reached pretraining manifest")
                writer.writerow([
                    "ai_readi", "aireadi-custom", vol.person_id, vol.vendor,
                    vol.model, vol.anatomic_region, int(vol.is_widefield),
                    vol.laterality, vol.height, vol.width, vol.n_frames,
                    ",".join(map(str, vol.take_frames)), vol.src_path,
                    common.npz_relpath(vol),
                ])
        digest = sha256_file(tmp)
        pin = expected if expected is not None else expected_sha256()
        if digest != pin:
            raise ValueError("AI-READI manifest differs from the frozen selection hash")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return {"sha256": digest, "volumes": len(volumes),
            "persons": len({v.person_id for v in volumes}),
            "planned_frames": sum(len(v.take_frames) for v in volumes)}


def extract_volume(src: Path, take: list[int], out: Path) -> dict:
    """Decode the frozen frame plan with the original PNG/NPZ pixel conversion."""
    from pydicom.pixels import pixel_array

    images, kept, dropped = [], [], []
    by_reason = {reason: 0 for reason in common.QC_REASONS}
    for index in take:
        frame = common.structural_frame_to_uint8(pixel_array(str(src), index=index))
        reason = common.frame_qc_reason(frame)
        if reason is None:
            images.append(frame)
            kept.append(index)
        else:
            dropped.append(index)
            by_reason[reason] += 1
    if images:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".npz.tmp.npz")
        try:
            np.savez_compressed(tmp, images=np.stack(images),
                                frames=np.asarray(kept, dtype=np.int32),
                                qc_dropped=np.asarray(dropped, dtype=np.int32))
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        # A resumed extraction can replace a previously published NPZ with an empty
        # result.  The complete sidecar below is authoritative only if none remains.
        out.unlink(missing_ok=True)
    return {"kept": len(kept), "dropped": len(dropped),
            "qc_dropped_by_reason": by_reason}


def qc_sidecar_path(npz: Path) -> Path:
    return Path(npz).with_suffix(".qc.json")


def _write_json_atomic(dest: Path, payload: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _sidecar_base(pin: str, row_index: int, planned: int) -> dict:
    return {"schema": QC_SIDECAR_SCHEMA, "manifest_sha256": pin,
            "row_index": row_index, "planned": planned}


def _complete_sidecar(base: dict, result: dict, npz: Path) -> dict:
    reasons = result["qc_dropped_by_reason"]
    if result["kept"] + sum(reasons.values()) != base["planned"]:
        raise AssertionError("extractor QC accounting does not cover every planned frame")
    if bool(result["kept"]) != npz.is_file():
        raise AssertionError("extractor NPZ presence disagrees with kept frame count")
    return {**base, "status": "complete", "kept": result["kept"],
            "qc_dropped_by_reason": reasons,
            "npz_sha256": sha256_file(npz) if result["kept"] else None}


def _valid_complete_sidecar(path: Path, base: dict, npz: Path) -> bool:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in base.items()):
        return False
    reasons = record.get("qc_dropped_by_reason")
    if (record.get("status") != "complete" or not isinstance(reasons, dict)
            or set(reasons) != set(common.QC_REASONS)
            or any(type(v) is not int or v < 0 for v in reasons.values())
            or type(record.get("kept")) is not int or record["kept"] < 0
            or record["kept"] + sum(reasons.values()) != base["planned"]):
        return False
    if record["kept"] == 0:
        return record.get("npz_sha256") is None and not npz.exists()
    return npz.is_file() and record.get("npz_sha256") == sha256_file(npz)


def extract_shard(data_root: Path, manifest: Path, out: Path, *, shard: int,
                  num_shards: int = SHARDS, expected: str | None = None) -> dict[str, int]:
    """Extract one stable row-index shard; fail if any selected volume cannot decode."""
    if num_shards <= 0 or not 0 <= shard < num_shards:
        raise ValueError("shard must be in [0, num_shards)")
    pin = expected if expected is not None else expected_sha256()
    if sha256_file(manifest) != pin:
        raise ValueError("AI-READI manifest differs from the frozen selection hash")
    with Path(manifest).open(newline="", encoding="utf-8") as fh:
        rows = [(i, row) for i, row in enumerate(csv.DictReader(fh, delimiter="\t"))
                if i % num_shards == shard]
    stats = {"rows": len(rows), "extracted": 0, "skipped_existing": 0,
             "frames_kept": 0, "frames_qc_dropped": 0,
             "all_qc_rejected": 0, "failed": 0}
    root = Path(data_root) / "public" / "ai_readi"
    for row_index, row in rows:
        dest = Path(out) / row["npz_relpath"]
        sidecar = qc_sidecar_path(dest)
        take = [int(x) for x in row["take_frames"].split(",")]
        base = _sidecar_base(pin, row_index, len(take))
        if sidecar.exists() and _valid_complete_sidecar(sidecar, base, dest):
            stats["skipped_existing"] += 1
            continue
        if dest.exists() and not sidecar.exists():
            # A pre-B6 local NPZ has no sidecar.  Stage evaluates those stored frames itself.
            stats["skipped_existing"] += 1
            continue
        # An incomplete sidecar is an intent marker.  It is written before extraction,
        # so a crash between NPZ publication and QC publication re-extracts this volume.
        _write_json_atomic(sidecar, {**base, "status": "intent"})
        src = root / row["src_path"]
        try:
            result = extract_volume(src, take, dest)
            _write_json_atomic(sidecar, _complete_sidecar(base, result, dest))
        except Exception:
            stats["failed"] += 1
            continue
        stats["extracted"] += 1
        stats["frames_kept"] += result["kept"]
        stats["frames_qc_dropped"] += result["dropped"]
        stats["all_qc_rejected"] += int(result["kept"] == 0)
    if stats["failed"]:
        raise RuntimeError(f"{stats['failed']} AI-READI volumes failed extraction")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--pool-root", type=Path,
                        default=paths.DATA_DIR / "derived" / "pretrain_pool")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    extraction = sub.add_parser("extract")
    extraction.add_argument("--shard", type=int, default=None)
    extraction.add_argument("--num-shards", type=int, default=SHARDS)
    args = parser.parse_args(argv)
    manifest = args.pool_root / "manifest_aireadi.tsv"
    if args.command == "manifest":
        result = write_manifest(args.data_root, manifest)
    else:
        shard = args.shard
        if shard is None:
            shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        result = extract_shard(args.data_root, manifest, args.pool_root,
                               shard=shard, num_shards=args.num_shards)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
