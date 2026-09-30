"""Materialise a LOCAL unlabeled adaptation pool out of the pretraining pool.

    python scripts/data/stage_unlabeled_pool.py --dry-run
    python scripts/repro.py build
Reshapes a slice of ``derived/pretrain_pool`` into the release layout
``<out>/<Vendor>_<Model>_unlabeled_<slug>/<person>__<lat>__<vol8>__f<frame>-image.png``
plus index.json, MANIFEST.tsv and build_report.json. Images only, stems globally unique.
``--out`` defaults to the canonical local root; an interrupted build may resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from octtta import paths  # noqa: E402
from scripts.data import aireadi_common as pretrain_pool  # noqa: E402
from scripts.data.extract_aireadi_frames import QC_SIDECAR_SCHEMA, qc_sidecar_path  # noqa: E402

cv2.setNumThreads(1)  # login node has one core; a job gets its parallelism from shards

# ---- the selection table: the one owner of "which frames get staged, and how many" ----

#: Optic-disc protocol names as AI-READI spells them.
ONH_REGIONS = ("Optic Disc", "Optic Disc, 6 x 6")


@dataclass(frozen=True)
class GroupRule:
    """One row of the selection table."""

    key: str                     #: group id; the key used in build_report.json
    vendor_dir: str              #: output directory name, release-layout style
    vendor: str                  #: manifest ``vendor`` column, exact
    model: str                   #: manifest ``model`` column, exact
    regions: tuple[str, ...]     #: manifest ``anatomic_region`` values, case-insensitive
    take: int                    #: frames per volume, evenly spaced over the stored ones
    min_width: int = 0           #: manifest ``width`` floor, inclusive
    why: str = ""

    @property
    def _regions_cf(self) -> tuple[str, ...]:
        return tuple(r.casefold() for r in self.regions)

    def matches(self, vendor: str, model: str, region: str, width: int) -> bool:
        return (vendor == self.vendor and model == self.model
                and region.casefold() in self._regions_cf
                and width >= self.min_width)


#: Printed at the start of every run and copied verbatim into ``build_report.json``, so a
#: report can never disagree with the code that produced it. 16 is what the pretraining
#: pool stores, so ``take`` is not a second truncation on top of the extractor's.
TAKE_PER_VOLUME = 16

SELECTION: tuple[GroupRule, ...] = (
    GroupRule(
        key="maestro2_widefield",
        vendor_dir="Topcon_Maestro2_unlabeled_widefield",
        vendor="Topcon", model="Maestro2", regions=("Wide Field",),
        take=TAKE_PER_VOLUME,
        why="the real 12 mm wide-field scan (885x512); the protocol the official test set "
            "punishes hardest and the one we own zero local labels for",
    ),
    GroupRule(
        key="maestro2_macula",
        vendor_dir="Topcon_Maestro2_unlabeled_macula",
        vendor="Topcon", model="Maestro2", regions=("Macula",),
        take=TAKE_PER_VOLUME,
        why="Maestro2 'Macula' (885x512), 3,342 volumes, a cell no other pool holds: the "
            "partial pool takes Maestro2 'Macula, 6 x 6' (885x360). NOTE: 885x512 is the "
            "same pixel shape as Maestro2 Wide Field, so octtta/fusion.py::shape_class "
            "calls it maestro2_wide; harmless while target=cross and hard_shapes is null, "
            "but it is why this row is named by protocol and not by shape",
    ),
    GroupRule(
        key="maestro2_onh",
        vendor_dir="Topcon_Maestro2_unlabeled_onh",
        vendor="Topcon", model="Maestro2", regions=ONH_REGIONS,
        take=TAKE_PER_VOLUME,
        why="optic disc on the labelled device: 0 volumes in the current manifest, kept as "
            "a row so a future AI-READI release does not silently skip it",
    ),
    GroupRule(
        key="triton_onh",
        vendor_dir="Topcon_Triton_unlabeled_onh",
        vendor="Topcon", model="Triton", regions=ONH_REGIONS,
        take=TAKE_PER_VOLUME,
        why="optic disc on the unseen vendor (992x512)",
    ),
    GroupRule(
        key="cirrus_onh",
        vendor_dir="Zeiss_Cirrus_unlabeled_onh",
        vendor="Zeiss", model="Cirrus", regions=("Optic Disc",),
        take=TAKE_PER_VOLUME,
        why="Cirrus optic disc 1024x200, the narrowest frames in the whole pool. WARNING: "
            "spelling this row ONH_REGIONS would swallow 'Optic Disc, 6 x 6' as well "
            "(3,629 + 3,291 = 6,920 volumes). That second cell is the one AI-READI ships "
            "HEIGHT MAPS for and it is supervised in the labelled pool, so staging it here "
            "too would put the same B-scans in one run twice, once with a label and once as "
            "a pseudo-label target",
    ),
    GroupRule(
        key="cirrus_macula",
        vendor_dir="Zeiss_Cirrus_unlabeled_macula",
        vendor="Zeiss", model="Cirrus", regions=("Macula",),
        take=TAKE_PER_VOLUME,
        why="Cirrus 'Macula' (1024x512), 3,441 volumes, the other cell no other pool holds "
            "(the partial pool takes Cirrus 'Macula, 6 x 6', 1024x350)",
    ),
    GroupRule(
        key="spectralis_onh",
        vendor_dir="Heidelberg_Spectralis_unlabeled_onh",
        vendor="Heidelberg", model="Spectralis", regions=ONH_REGIONS,
        take=TAKE_PER_VOLUME,
        why="optic disc, 496x768",
    ),
    GroupRule(
        key="spectralis_wide",
        vendor_dir="Heidelberg_Spectralis_unlabeled_wide",
        vendor="Heidelberg", model="Spectralis", regions=("Macula",),
        take=TAKE_PER_VOLUME, min_width=700,
        why="the wide Spectralis macula rasters (768 and 1536 columns). NOTE: 'wide' here "
            "is COLUMN COUNT, not field width: Spectralis 'Macula, 20 x 20' has a 5.93 mm "
            "fast axis and 512 columns and is NOT in this group",
    ),
)


@dataclass(frozen=True)
class ExcludedCell:
    """One ``(vendor, model, anatomic_region)`` cell that is deliberately not staged."""

    claimed_by: str              #: who holds these B-scans instead; "" for nobody
    why: str


#: Deliberately NOT staged, with the reason. Also the table :func:`_assert_no_orphan_onh`
#: consults, so a cell cannot be excluded by design in the printout and an unexplained
#: orphan to the assertion.
EXCLUDED_BY_DESIGN: dict[tuple[str, str, str], ExcludedCell] = {
    ("Heidelberg", "Spectralis", "Macula, 20 x 20"): ExcludedCell(
        claimed_by="aireadi::Heidelberg_Spectralis @ partial_all16",
        why="supervised through the partial pool; 496x512, and despite the name its fast "
            "axis is 5.93 mm, i.e. an ordinary macula scan"),
    ("Topcon", "Maestro2", "Macula, 6 x 6"): ExcludedCell(
        claimed_by="aireadi::Topcon_Maestro2 @ partial_all16",
        why="supervised through the partial pool (885x360)"),
    ("Topcon", "Triton", "Macula, 12 x 12"): ExcludedCell(
        claimed_by="aireadi::Topcon_Triton @ partial_all16",
        why="supervised through the partial pool (992x512); staging it again as unlabeled "
            "would double-count the same B-scans with a weaker signal"),
    ("Topcon", "Triton", "Macula, 6 x 6"): ExcludedCell(
        claimed_by="aireadi::Topcon_Triton @ partial_all16",
        why="supervised through the partial pool (992x320)"),
    ("Zeiss", "Cirrus", "Macula, 6 x 6"): ExcludedCell(
        claimed_by="aireadi::Zeiss_Cirrus @ partial_all16",
        why="supervised through the partial pool (1024x350); Cirrus draws only ILM and 8|9"),
    ("Zeiss", "Cirrus", "Optic Disc, 6 x 6"): ExcludedCell(
        claimed_by="aireadi::Zeiss_Cirrus @ partial_all16",
        why="the ONE optic-disc cell in the release that ships height maps (4,427 volumes, "
            "1024x350). It is supervised in the labelled pool, so this stager must NOT "
            "claim it, and _assert_no_orphan_onh reads this row rather than raising: an "
            "unclaimed ONH row otherwise means the table lost a device"),
}

#: 885x512 -> ~0.3 MB, measured; scaled by pixel count for the other protocols.
PNG_BYTES_PER_PIXEL = 0.3e6 / (885 * 512)

IMAGE_SUFFIX = "-image.png"

MANIFEST_COLUMNS = (
    "vendor_dir", "stem", "relpath", "group", "vendor", "model", "anatomic_region",
    "height", "width", "person_id", "laterality", "src_npz", "frame", "sha256",
)


def default_pool_root() -> Path:
    return paths.DERIVED_DIR / "pretrain_pool"


def default_out() -> Path:
    """Where a plain run writes. One owner: :data:`octtta.paths.UNLABELED_ALL_D88_DIR`, so
    the writer here and the reader in :mod:`octtta.data.unlabeled_local` cannot differ."""
    return Path(paths.UNLABELED_ALL_D88_DIR)


def render_selection_table() -> str:
    """The table, as printed at the start of a run and stored in the report."""
    head = (f"{'group':<20s} {'vendor_dir':<40s} {'protocol':<28s} "
            f"{'w>=':>5s} {'take':>5s}")
    rows = [head, "-" * len(head)]
    for r in SELECTION:
        rows.append(f"{r.key:<20s} {r.vendor_dir:<40s} "
                    f"{r.vendor + ' ' + r.model + ' / ' + '|'.join(r.regions):<28.28s} "
                    f"{r.min_width:>5d} {r.take:>5d}")
    rows.append("")
    for (vendor, model, region), cell in EXCLUDED_BY_DESIGN.items():
        claimed = f" [claimed by {cell.claimed_by}]" if cell.claimed_by else ""
        rows.append(f"NOT STAGED: {vendor} {model} {region!r}{claimed} -- {cell.why}")
    return "\n".join(rows)


def selection_table_as_json() -> dict:
    return {
        "groups": [
            {"key": r.key, "vendor_dir": r.vendor_dir, "vendor": r.vendor,
             "model": r.model, "regions": list(r.regions), "take": r.take,
             "min_width": r.min_width, "why": r.why}
            for r in SELECTION
        ],
        "not_staged": [
            {"vendor": v, "model": m, "anatomic_region": a,
             "claimed_by": cell.claimed_by, "why": cell.why}
            for (v, m, a), cell in EXCLUDED_BY_DESIGN.items()
        ],
        "frame_rule": ("scripts.data.aireadi_common.choose_pretrain_frames applied to the "
                       "STORED frames: evenly spaced, first and last always included"),
    }


# ---- refusing an output directory that would overwrite data ----

def forbidden_out_reason(out: Path) -> str | None:
    """Reject the source root as an output root; staging may resume its own root."""
    if Path(out).resolve()==default_pool_root().resolve():
        return "output root equals source pretrain pool"
    return None



# ---- manifest reading and planning ----

def volume_tag(npz_relpath: str) -> str:
    """``vol8``: the 8-hex identity of one staged volume, from its npz relative path. 32 bits
    is not unique across 23,720 volumes, which is why the exclusion format also matches
    ``person_id`` and ``laterality`` and why the staged stem carries both."""
    return hashlib.sha256(str(npz_relpath).encode("utf-8")).hexdigest()[:8]




@dataclass(frozen=True)
class PlannedVolume:
    rule: GroupRule
    person_id: str
    vendor: str
    model: str
    anatomic_region: str
    laterality: str
    height: int
    width: int
    npz_relpath: str
    src_path: str
    n_stored: int              #: frames actually in the npz (or the plan, in --dry-run)
    positions: tuple[int, ...]  #: indices into the stored array
    manifest_index: int = -1
    src_take_frames: str = ""

    @property
    def vol8(self) -> str:
        return volume_tag(self.npz_relpath)


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for index, row in enumerate(rows):
        row["_manifest_index"] = index
    if not rows:
        raise SystemExit(f"!! {path} has no rows")
    missing = {"person_id", "vendor", "model", "anatomic_region", "laterality",
               "height", "width", "take_frames", "npz_relpath", "src_path"} - set(rows[0])
    if missing:
        raise SystemExit(f"!! {path} is missing column(s) {sorted(missing)}")
    return rows


def rule_for_row(vendor: str, model: str, region: str, width: int) -> GroupRule | None:
    """The one rule that claims this row, or ``None``. Raises if two rules claim it:
    overlapping rows would be staged twice under two directory names."""
    hits = [r for r in SELECTION if r.matches(vendor, model, region, width)]
    if len(hits) > 1:
        raise AssertionError(
            f"selection table is ambiguous for {vendor} {model} {region!r} w={width}: "
            f"matched {[h.key for h in hits]}")
    return hits[0] if hits else None


def rules_blocked_by_min_width(vendor: str, model: str, region: str,
                               width: int) -> list[GroupRule]:
    """Rules this row matches on vendor/model/region but is too narrow for. ``min_width`` is
    the one filter in this table that drops rows SILENTLY, so the report carries both the
    number considered and the number dropped."""
    return [r for r in SELECTION
            if r.vendor == vendor and r.model == model
            and region.casefold() in r._regions_cf and width < r.min_width]




def _assert_no_orphan_onh(vendor: str, model: str, region: str, width: int) -> None:
    """An optic-disc row nobody claimed means the table lost a device. The exemption is read
    out of :data:`EXCLUDED_BY_DESIGN` rather than a second list."""
    if region.casefold() not in {r.casefold() for r in ONH_REGIONS}:
        return
    claimed = EXCLUDED_BY_DESIGN.get((vendor, model, region))
    if claimed is not None and claimed.claimed_by:
        return
    raise AssertionError(
        f"manifest has an optic-disc volume the selection table does not name: "
        f"{vendor} {model} {region!r} w={width}. The table claims to cover every "
        "vendor's ONH scans: add a GroupRule, or an EXCLUDED_BY_DESIGN row saying which "
        "pool holds it instead, or the pool silently skips a device.")


def _stored_counts(pool_root: Path) -> dict[str, int]:
    """``npz_relpath -> stored frame count``, from the extractor's own index if present."""
    idx = Path(pool_root) / "extracted_index.json"
    if not idx.is_file():
        return {}
    data = json.loads(idx.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def plan(rows: Iterable[dict], *, limit: int | None = None,
         stored: dict[str, int] | None = None,
         pool_root: Path | None = None,
         read_npz: bool = False,
         width_audit: dict[str, dict[str, int]] | None = None) -> Iterator[PlannedVolume]:
    """Walk the manifest in file order and yield the volumes to stage. ``limit`` is PER
    GROUP, not global, so a smoke run still touches every group; ``width_audit``, when given,
    is filled in place with per-rule counts."""
    stored = stored or {}
    if width_audit is not None:
        for r in SELECTION:
            if r.min_width:
                width_audit.setdefault(
                    r.key, {"min_width": r.min_width, "considered": 0,
                            "dropped_by_min_width": 0})
    seen_per_group: dict[str, int] = {}
    for row in rows:
        vendor = (row["vendor"] or "").strip()
        model = (row["model"] or "").strip()
        region = (row["anatomic_region"] or "").strip()
        width = int(row["width"] or 0)
        height = int(row["height"] or 0)
        rule = rule_for_row(vendor, model, region, width)
        if width_audit is not None:
            if rule is not None and rule.min_width:
                width_audit[rule.key]["considered"] += 1
            for blocked in rules_blocked_by_min_width(vendor, model, region, width):
                width_audit[blocked.key]["considered"] += 1
                width_audit[blocked.key]["dropped_by_min_width"] += 1
        if rule is None:
            _assert_no_orphan_onh(vendor, model, region, width)
            continue
        if limit is not None and seen_per_group.get(rule.key, 0) >= limit:
            continue
        relpath = (row["npz_relpath"] or "").strip()
        take_plan = [t for t in (row["take_frames"] or "").split(",") if t]
        if read_npz:
            npz = Path(pool_root) / relpath
            if npz.is_file():
                with np.load(npz) as z:
                    n_stored = int(np.asarray(z["images"]).shape[0])
            elif qc_sidecar_path(npz).is_file():
                n_stored = 0  # Validated as a complete empty volume before staging.
            else:
                raise FileNotFoundError("selected AI-READI source NPZ is missing")
        else:
            n_stored = int(stored.get(relpath, len(take_plan)))
        if n_stored <= 0 and not (read_npz and qc_sidecar_path(Path(pool_root) / relpath).is_file()):
            continue
        positions = (pretrain_pool.choose_pretrain_frames(n_stored, max_take=rule.take)
                     if n_stored else ())
        seen_per_group[rule.key] = seen_per_group.get(rule.key, 0) + 1
        yield PlannedVolume(
            rule=rule, person_id=str(row["person_id"]).strip(), vendor=vendor,
            model=model, anatomic_region=region,
            laterality=(row["laterality"] or "").strip() or "U",
            height=height, width=width, npz_relpath=relpath,
            src_path=str(row["src_path"]).strip(),
            n_stored=n_stored, positions=positions,
            manifest_index=int(row.get("_manifest_index", -1)),
            src_take_frames=str(row["take_frames"]),
        )


# ---- PNG IO ----

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """``(height, width)`` straight out of the IHDR chunk, no decode. Raises ``ValueError``
    on anything that is not a PNG, so a truncated write is never taken for a match."""
    if len(data) < 24 or not data.startswith(_PNG_MAGIC) or data[12:16] != b"IHDR":
        raise ValueError("not a PNG (or truncated before IHDR)")
    width, height = struct.unpack(">II", data[16:24])
    return int(height), int(width)


def decode_png(data: bytes) -> np.ndarray:
    """A stored PNG back as a 2-D uint8 array. Raises ``ValueError`` for a 16-bit PNG, whose
    values would compare equal to a uint8 array of the same numbers."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("cv2 could not decode these bytes as an image")
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"not a single-channel uint8 PNG (ndim={arr.ndim} "
                         f"dtype={arr.dtype})")
    return arr


def png_matches(data: bytes, frame: np.ndarray) -> bool:
    """Does this stored PNG hold exactly ``frame``? H x W alone is not identity: a stale or
    wrong frame of the same shape would be kept and re-indexed with its own sha256. Pixels
    are compared, not bytes."""
    try:
        return np.array_equal(decode_png(data), np.asarray(frame))
    except ValueError:
        return False


def encode_png(frame: np.ndarray) -> bytes:
    """Single-channel uint8 PNG bytes."""
    arr = np.asarray(frame)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise AssertionError(
            f"expected a 2-D uint8 B-scan, got ndim={arr.ndim} dtype={arr.dtype}")
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed on a uint8 B-scan")
    return bytes(buf.tobytes())


def write_atomic(dest: Path, data: bytes) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


# ---- the build ----

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validated_qc_sidecar(vol: PlannedVolume, pool_root: Path,
                          manifest_sha: str) -> dict | None:
    """Read extractor QC accounting, rejecting incomplete or mismatched sources."""
    npz = pool_root / vol.npz_relpath
    path = qc_sidecar_path(npz)
    if not path.is_file():
        if not npz.is_file():
            raise FileNotFoundError("selected AI-READI source has no NPZ or QC sidecar")
        return None  # Legacy local NPZ: stage performs QC on its stored frames.
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("AI-READI QC sidecar is unreadable") from exc
    reasons = record.get("qc_dropped_by_reason") if isinstance(record, dict) else None
    take = [int(x) for x in vol.src_take_frames.split(",") if x]
    planned = len(take)
    if (not isinstance(record, dict) or record.get("schema") != QC_SIDECAR_SCHEMA
            or record.get("status") != "complete"
            or record.get("manifest_sha256") != manifest_sha
            or record.get("row_index") != vol.manifest_index
            or type(record.get("planned")) is not int or record["planned"] != planned
            or type(record.get("kept")) is not int or record["kept"] < 0
            or not isinstance(reasons, dict)
            or set(reasons) != set(pretrain_pool.QC_REASONS)
            or any(type(n) is not int or n < 0 for n in reasons.values())
            or record["kept"] + sum(reasons.values()) != planned):
        raise ValueError("AI-READI QC sidecar accounting or manifest pin differs")
    if record["kept"]:
        if (not npz.is_file() or record.get("npz_sha256") != _sha256_file(npz)
                or vol.n_stored != record["kept"]):
            raise ValueError("AI-READI QC sidecar NPZ integrity differs")
        with np.load(npz) as z:
            kept = np.asarray(z["frames"]).astype(int).tolist()
            dropped = np.asarray(z["qc_dropped"]).astype(int).tolist()
            if (len(kept) != record["kept"]
                    or len(dropped) != sum(reasons.values())
                    or sorted([*kept, *dropped]) != sorted(take)):
                raise ValueError("AI-READI QC sidecar NPZ frame counts differ")
    elif npz.exists() or record.get("npz_sha256") is not None or vol.n_stored != 0:
        raise ValueError("empty AI-READI QC sidecar has an NPZ")
    return record




def frozen_manifest_sha256() -> str:
    """Read the published pretraining manifest digest from expected.json."""
    expected=json.loads((REPO_ROOT/"configs"/"expected.json").read_text())
    pin=((expected.get("aireadi") or {}).get("pretrain_manifest") or {}).get("sha256")
    if not isinstance(pin,str) or len(pin)!=64:
        raise ValueError("expected.json lacks aireadi.pretrain_manifest.sha256")
    return pin



def _group_report_skeleton() -> dict:
    # Every reason code starts at zero rather than appearing when first seen: "rejected
    # nothing" and "never ran" are the same missing key otherwise.
    return {"volumes": 0, "frames_written": 0, "frames_reused": 0, "frames_qc_rejected": 0,
            "frames_qc_rejected_by_reason": {r: 0 for r in pretrain_pool.QC_REASONS},
            "bytes": 0, "persons": set()}


def stage_unlabeled_pool(
    *,
    out: Path,
    manifest: Path,
    pool_root: Path,
    limit: int | None = None,
    dry_run: bool = False,
    exclusions=None,
    expect_manifest_sha256: str | None = None,
    check_out: bool = True,
) -> dict:
    """Build (or plan) the pool; return aggregate counts and local output metadata."""
    out, manifest, pool_root = Path(out), Path(manifest), Path(pool_root)
    if check_out:
        reason = forbidden_out_reason(out)
        if reason:
            raise SystemExit(f"!! {reason}")

    ex=exclusions if exclusions is not None else pretrain_pool.Exclusions.load(
        paths.DATA_DIR/"public"/"ai_readi")

    manifest_sha = _sha256_file(manifest)
    if expect_manifest_sha256 and manifest_sha != expect_manifest_sha256:
        raise SystemExit("AI-READI pretraining manifest differs from expected.json")

    rows = read_manifest(manifest)
    # The duplicate list is applied to the full frozen manifest before group selection.
    excluded_rows=[]
    kept_rows=[]
    for row in rows:
        (excluded_rows if ex.excludes_unlabelled_duplicate(
            row["src_path"], row["vendor"]) else kept_rows).append(row)
    rows=kept_rows
    duplicate_by_vendor={}
    for row in excluded_rows:
        vendor=row["vendor"]
        duplicate_by_vendor[vendor]=duplicate_by_vendor.get(vendor,0)+1

    stored = _stored_counts(pool_root)

    report: dict = {
        "created_by":"scripts/data/stage_unlabeled_pool.py",
        "dry_run":bool(dry_run),"limit_per_group":limit,
        "manifest_sha256":manifest_sha,
        "manifest_sha256_frozen":expect_manifest_sha256,
        "extracted_index_used":bool(stored),
        "selection_table":selection_table_as_json(),
        "exclusions":{
            "file_sha256":ex.config_sha256,
            "test_split_persons":len(ex.test_split),
            "never_train_volumes":len(ex.never_train),
            "unlabelled_duplicates":len(excluded_rows),
            "unlabelled_duplicates_by_vendor":duplicate_by_vendor,
        },
        "groups":{},"totals":{},
    }

    groups: dict[str, dict] = {r.key: _group_report_skeleton() for r in SELECTION}
    records: list[dict] = []
    stems: dict[str, str] = {}
    qc_rejected: list[str] = []

    width_audit: dict[str, dict[str, int]] = {}
    for vol in plan(rows, limit=limit, stored=stored, pool_root=pool_root,
                    read_npz=not dry_run, width_audit=width_audit):
        if ex.is_never_train(vol.person_id,vol.src_path):
            raise AssertionError("never-train volume reached the staged selection")
        g = groups[vol.rule.key]
        g["volumes"] += 1
        g["persons"].add(vol.person_id)

        if dry_run:
            g["frames_written"] += len(vol.positions)
            g["bytes"] += int(round(len(vol.positions) * vol.height * vol.width
                                    * PNG_BYTES_PER_PIXEL))
            continue

        sidecar = _validated_qc_sidecar(vol, pool_root, manifest_sha)
        if sidecar is not None:
            for reason in pretrain_pool.QC_NO_RETINA_REASONS:
                count = sidecar["qc_dropped_by_reason"][reason]
                g["frames_qc_rejected"] += count
                g["frames_qc_rejected_by_reason"][reason] += count
                qc_rejected.extend([reason] * count)
        if vol.n_stored == 0:
            continue

        vdir = out / vol.rule.vendor_dir
        vdir.mkdir(parents=True, exist_ok=True)
        with np.load(pool_root / vol.npz_relpath) as z:
            images = np.asarray(z["images"])
            frames = np.asarray(z["frames"]).astype(int)
        if images.shape[1:] != (vol.height, vol.width):
            raise AssertionError(
                f"stored frames are {images.shape[1]}x{images.shape[2]} "
                f"but the manifest row says {vol.height}x{vol.width}. The manifest and the "
                "npz tree have drifted apart; refusing to stage either version.")
        if frames.shape[0] != images.shape[0]:
            raise AssertionError(
                f"{images.shape[0]} images but {frames.shape[0]} frame "
                "indices; the npz is not self-consistent.")

        for pos in vol.positions:
            frame_idx = int(frames[pos])
            img = images[pos]
            reason = pretrain_pool.frame_qc_reason(img)
            if reason is not None:
                if sidecar is not None:
                    raise AssertionError("fresh extracted frame fails QC despite complete sidecar")
                qc_rejected.append(reason)
                g["frames_qc_rejected"] += 1
                g["frames_qc_rejected_by_reason"][reason] += 1
                # WARNING: the two halves of this gate have OPPOSITE meanings.
                # ``column_range`` was applied by the extractor, so failing it here means the
                # threshold moved and that must raise; the two no-retina criteria post-date
                # the npz tree, so failing them is the gate working: drop and count.
                if reason not in pretrain_pool.QC_NO_RETINA_REASONS:
                    raise AssertionError(
                        f"staged volume frame fails extractor QC criterion {reason!r}")
                continue

            stem = f"{vol.person_id}__{vol.laterality}__{vol.vol8}__f{frame_idx}"
            dest = vdir / f"{stem}{IMAGE_SUFFIX}"
            prev = stems.get(stem)
            if prev is not None:
                raise AssertionError(
                    "a release stem is claimed by two sources; release stems must be "
                    "globally unique or a flat prediction directory collides.")
            stems[stem] = vol.npz_relpath

            data: bytes | None = None
            reused = False
            if dest.is_file():
                # Reuse only a file whose PIXELS are the ones we are about to write.
                blob = dest.read_bytes()
                try:
                    same_shape = png_size(blob) == (vol.height, vol.width)
                except ValueError:
                    same_shape = False        # truncated or foreign: rewrite it
                if same_shape and png_matches(blob, img):
                    data, reused = blob, True
            if data is None:
                data = encode_png(img)
                write_atomic(dest, data)
            g["frames_reused" if reused else "frames_written"] += 1
            g["bytes"] += len(data)

            records.append({
                "vendor_dir": vol.rule.vendor_dir,
                "stem": stem,
                "relpath": f"{vol.rule.vendor_dir}/{dest.name}",
                "group": vol.rule.key,
                "vendor": vol.vendor,
                "model": vol.model,
                "anatomic_region": vol.anatomic_region,
                "height": vol.height,
                "width": vol.width,
                "person_id": vol.person_id,
                "laterality": vol.laterality,
                "src_npz": vol.npz_relpath,
                "frame": frame_idx,
                "sha256": hashlib.sha256(data).hexdigest(),
            })

    n_frames = sum(g["frames_written"] + g["frames_reused"] for g in groups.values())
    if not dry_run and n_frames != len(records):
        raise AssertionError(
            f"counted {n_frames} staged frames but wrote {len(records)} index records; "
            "the report and the index disagree.")

    all_persons: set[str] = set()
    for key, g in groups.items():
        all_persons |= g["persons"]
        report["groups"][key] = {
            "vendor_dir": next(r.vendor_dir for r in SELECTION if r.key == key),
            "take_per_volume": next(r.take for r in SELECTION if r.key == key),
            "volumes": g["volumes"],
            "persons": len(g["persons"]),
            "frames": g["frames_written"] + g["frames_reused"],
            "frames_written": g["frames_written"],
            "frames_reused": g["frames_reused"],
            "frames_qc_rejected": g["frames_qc_rejected"],
            "frames_qc_rejected_by_reason": dict(g["frames_qc_rejected_by_reason"]),
            "bytes": g["bytes"],
            "gb": round(g["bytes"] / 1e9, 3),
        }
    report["totals"] = {
        "volumes": sum(g["volumes"] for g in report["groups"].values()),
        "frames": sum(g["frames"] for g in report["groups"].values()),
        "persons": len(all_persons),
        "bytes": sum(g["bytes"] for g in report["groups"].values()),
        "gb": round(sum(g["bytes"] for g in report["groups"].values()) / 1e9, 3),
        "manifest_rows": len(rows),
    }
    by_reason = {r: 0 for r in pretrain_pool.QC_REASONS}
    for g in groups.values():
        for r, n in g["frames_qc_rejected_by_reason"].items():
            by_reason[r] = by_reason.get(r, 0) + int(n)
    if sum(by_reason.values()) != len(qc_rejected):
        raise AssertionError(
            f"QC rejected {len(qc_rejected)} frame(s) but the per-criterion counters add "
            f"up to {sum(by_reason.values())} ({by_reason}). A total and its parts kept "
            f"separately is how one of them silently stops being maintained.")
    report["qc_rejected"] = {
        "n": len(qc_rejected),
        "owner": "scripts.data.aireadi_common.frame_qc_reason",
        # Read off the constants at run time: a report naming the criteria but not the
        # numbers cannot tell "the frames are fine" from "somebody loosened a threshold".
        "thresholds": {
            "column_range": pretrain_pool.QC_MIN_COLUMN_RANGE,
            "column_correlation": pretrain_pool.QC_MIN_COLUMN_CORRELATION,
            "tissue_band_snr": pretrain_pool.QC_MIN_TISSUE_BAND_SNR,
        },
        # Every criterion, always, including the ones that caught nothing.
        "by_reason": by_reason,
        "no_retina_reasons": list(pretrain_pool.QC_NO_RETINA_REASONS),
    }
    # Reported even when all zeros: "0 dropped" and "never reached" are the same line in a
    # report that omits the row.
    report["min_width_filter"] = {
        "note": ("rows a rule matched on vendor/model/region but rejected for width. "
                 "'considered' counts the rows the floor was applied to, so a zero here is "
                 "a MEASURED zero and not a filter nothing reached"),
        "by_group": width_audit,
    }
    if dry_run:
        report["bytes_note"] = (
            f"projection only: {PNG_BYTES_PER_PIXEL:.4f} bytes/pixel, anchored on "
            "~0.3 MB for one 885x512 frame")

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        _write_index(out, records)
        _write_manifest_tsv(out, records)
        write_atomic(out / "build_report.json",
                     (json.dumps(report, indent=2, sort_keys=False) + "\n").encode())
    return report


def _write_index(out: Path, records: list[dict]) -> None:
    blob = {
        "schema": 1,
        "created_by": "scripts/data/stage_unlabeled_pool.py",
        "image_suffix": IMAGE_SUFFIX,
        "note": ("release-layout mirror with no status subdirectory: point an "
                 "index_release-style loader at <root>/<vendor_dir>/, e.g. "
                 "release_dataset.index_split(root, vendor_dir, '')"),
        "n_images": len(records),
        "images": records,
    }
    write_atomic(out / "index.json",
                 (json.dumps(blob, indent=1, sort_keys=False) + "\n").encode())


def _write_manifest_tsv(out: Path, records: list[dict]) -> None:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for rec in records:
        lines.append("\t".join(str(rec[c]) for c in MANIFEST_COLUMNS))
    write_atomic(out / "MANIFEST.tsv", ("\n".join(lines) + "\n").encode())


# ---- CLI ----

def print_report(report: dict) -> None:
    """Only aggregate counts leave the local licensed data root."""
    for key,group in report["groups"].items():
        print(f"{key}: {group['volumes']} volumes, {group['frames']} frames")
    total=report["totals"]
    print(f"TOTAL: {total['volumes']} volumes, {total['frames']} frames")
    print(f"QC rejected: {report['qc_rejected']['n']}")
    print(f"duplicate volumes excluded: {report['exclusions']['unlabelled_duplicates']}")



def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out",type=Path,default=None)
    p.add_argument("--manifest",type=Path,default=None)
    p.add_argument("--pool-root",type=Path,default=None)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--dry-run",action="store_true")
    return p



def main(argv: list[str] | None = None) -> int:
    args=build_parser().parse_args(argv)
    pool_root=args.pool_root or default_pool_root()
    manifest=args.manifest or pool_root/"manifest_aireadi.tsv"
    out=args.out or default_out()
    report=stage_unlabeled_pool(
        out=out,manifest=manifest,pool_root=pool_root,limit=args.limit,
        dry_run=args.dry_run,expect_manifest_sha256=frozen_manifest_sha256())
    print_report(report)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
