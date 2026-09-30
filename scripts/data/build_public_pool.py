"""Build the labelled pool from public OCT datasets."""
from __future__ import annotations
import argparse
import csv
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from octtta import paths
from octtta.data.partial_labels import (
    MAPPING_VERSION, NUM_BOUNDARIES, SURFACE_TO_BOUNDARY,
    apply_boundary_offsets, offsets_to_json, rasterize_partial, surfaces_to_json,
)
cv2.setNumThreads(0)
PUBLIC_DATASETS = ("oct5k", "jhu_hcms", "duke_dme_2015")
JHU_EMPTY_COLUMNS = [2, 10]
SURFACE_MIN_VALID_COLUMN_FRAC = 0.05
MATLAB_ROW_OFFSET = 1.0
VOL_MAIN_HEADER_BYTES = 2048
VOL_MAGIC = b"HSF-OCT"
VOL_BSCAN_MAGIC = b"HSF-BS"
VOL_INVALID_THRESHOLD = 1e30

@dataclass(frozen=True)
class VolHeader:
    version: str
    size_x: int          # A-scans per B-scan == image width
    n_bscans: int
    size_z: int          # samples per A-scan == image height
    scale_x: float       # mm per A-scan
    distance: float      # mm between B-scans
    scale_z: float       # mm per depth sample
    size_x_slo: int
    size_y_slo: int
    bscan_hdr_size: int

    @property
    def bscan_stride(self) -> int:
        """Bytes from one B-scan record to the next (header + float32 pixels)."""
        return self.bscan_hdr_size + self.size_x * self.size_z * 4

    @property
    def first_bscan_offset(self) -> int:
        """Byte offset of B-scan 0: main header, then the uint8 SLO fundus image."""
        return VOL_MAIN_HEADER_BYTES + self.size_x_slo * self.size_y_slo

    @property
    def min_file_bytes(self) -> int:
        return self.first_bscan_offset + self.n_bscans * self.bscan_stride


def parse_vol_header(buf: bytes) -> VolHeader:
    """Parse the .vol main header; raises ``ValueError`` rather than addressing garbage."""
    if len(buf) < VOL_MAIN_HEADER_BYTES:
        raise ValueError(
            f"vol main header truncated: {len(buf)} < {VOL_MAIN_HEADER_BYTES} bytes")
    version = buf[:12].split(b"\x00", 1)[0].decode("ascii", "replace")
    if not buf.startswith(VOL_MAGIC):
        raise ValueError(f"not a Heidelberg .vol: magic is {version!r}")
    size_x, n_bscans, size_z = struct.unpack("<III", buf[12:24])
    scale_x, distance, scale_z = struct.unpack("<ddd", buf[24:48])
    size_x_slo, size_y_slo = struct.unpack("<II", buf[48:56])
    (bscan_hdr_size,) = struct.unpack("<I", buf[100:104])
    hdr = VolHeader(
        version=version, size_x=size_x, n_bscans=n_bscans, size_z=size_z,
        scale_x=scale_x, distance=distance, scale_z=scale_z,
        size_x_slo=size_x_slo, size_y_slo=size_y_slo, bscan_hdr_size=bscan_hdr_size,
    )
    for field in ("size_x", "n_bscans", "size_z", "bscan_hdr_size"):
        if getattr(hdr, field) <= 0:
            raise ValueError(f"vol header has non-positive {field}={getattr(hdr, field)}")
    return hdr


def vol_to_uint8(raw: np.ndarray) -> np.ndarray:
    """Heidelberg display transform: void -> 0, clip to [0,1], **0.25 (the viewer's
    contrast curve), x255."""
    arr = np.asarray(raw, dtype=np.float32)
    valid = np.isfinite(arr) & (arr < VOL_INVALID_THRESHOLD)
    x = np.clip(np.where(valid, arr, 0.0), 0.0, 1.0) ** 0.25
    return np.rint(x * 255.0).astype(np.uint8)


def _read_vol_bscan(f: BinaryIO, hdr: VolHeader, k: int) -> np.ndarray:
    offset = hdr.first_bscan_offset + k * hdr.bscan_stride
    f.seek(offset)
    sub = f.read(hdr.bscan_hdr_size)
    if len(sub) < 16 or not sub.startswith(VOL_BSCAN_MAGIC):
            # WARNING: the stride is derived, not stored per record, so a wrong stride
            # reads plausible garbage. The per-B-scan magic makes that fail here.
        raise ValueError(
            f"B-scan {k} at byte {offset} does not start with {VOL_BSCAN_MAGIC!r} "
            f"(got {sub[:12]!r}): the derived .vol layout is wrong")
    n = hdr.size_x * hdr.size_z
    payload = f.read(n * 4)
    if len(payload) != n * 4:
        raise ValueError(f"B-scan {k} pixel data truncated: {len(payload)} != {n * 4} bytes")
    raw = np.frombuffer(payload, dtype="<f4").reshape(hdr.size_z, hdr.size_x)
    return vol_to_uint8(raw)


def iter_vol_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, uint8 (size_z, size_x))`` for every B-scan."""
    with Path(path).open("rb") as f:
        hdr = parse_vol_header(f.read(VOL_MAIN_HEADER_BYTES))
        size = Path(path).stat().st_size
        if size < hdr.min_file_bytes:
            raise ValueError(
                f"{path.name}: {size} bytes but the header describes "
                f"{hdr.min_file_bytes}: truncated or misparsed")
        for k in range(hdr.n_bscans):
            yield k, _read_vol_bscan(f, hdr, k)


def _surface_rows(dataset: str, per_surface: list[np.ndarray | None],
                  width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scatter a dataset's own surfaces onto the nine official boundary slots, corrected.
    ``per_surface[i]`` is a length-``width`` row array, ``NaN`` where unknown and ``None``
    for a dropped surface. The one place a declared per-boundary offset is applied."""
    mapping = SURFACE_TO_BOUNDARY[dataset]
    if len(per_surface) != len(mapping.boundaries):
        raise ValueError(
            f"{dataset}: got {len(per_surface)} surfaces, the mapping describes "
            f"{len(mapping.boundaries)}")
    rows = np.full((NUM_BOUNDARIES, width), np.nan, dtype=np.float64)
    avail = np.zeros((NUM_BOUNDARIES, width), dtype=bool)
    for surf, target in zip(per_surface, mapping.boundaries):
        if target is None or surf is None:
            continue
        s = np.asarray(surf, dtype=np.float64)
        if s.shape != (width,):
            raise ValueError(f"{dataset}: surface has shape {s.shape}, expected {width}")
        rows[target] = s
        avail[target] = np.isfinite(s)
    return apply_boundary_offsets(rows, avail, height, mapping.offsets, key=dataset)


def _boundaries_lost_to_sentinels(dataset: str, avail: np.ndarray) -> list[int]:
    """Declared boundaries of ``dataset`` that this frame does not actually carry. Asks the
    mapping, not the frame: "which boundaries are present" cannot tell a Cirrus frame
    (two by design) from one that lost its ILM."""
    promised = SURFACE_TO_BOUNDARY[dataset].available
    return [int(b) for b in promised
            if float(avail[b].mean()) < SURFACE_MIN_VALID_COLUMN_FRAC]


def _accumulate_offset_audit(stats: dict, audit: dict) -> None:
    """Fold one image's offset audit into a build's running counters; both counts, because
    zero collapses over zero measured columns shows only that nothing looked."""
    stats["offset_columns_measured"] = (
        int(stats.get("offset_columns_measured", 0)) + int(audit["n_measured"]))
    stats["offset_columns_collapsed"] = (
        int(stats.get("offset_columns_collapsed", 0)) + int(audit["n_collapsed"]))


def _write_png_atomic(path: Path, array: np.ndarray) -> None:
    """Encode to a same-directory temp file, then ``os.replace`` it onto *path*.
    WARNING: ``cv2.imwrite`` writes THROUGH an existing file, and pool directories here are
    routinely hardlink clones, so a rebuild would rewrite a frozen pool's bytes."""
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        if not cv2.imwrite(str(tmp), array):
            raise OSError(f"could not write {tmp}")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _write_json_atomic(path: Path, obj) -> None:
    """Write ``obj`` as JSON through a same-directory temp file and ``os.replace``, so a
    root whose files are hardlink clones of a frozen pool is never written through."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _write_pair(out_dir: Path, stem: str, image: np.ndarray,
                code: np.ndarray, label_suffix: str = "-label.png") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}-image.png"
    lab_path = out_dir / f"{stem}{label_suffix}"
    if image.shape != code.shape:
        raise ValueError(f"{stem}: image {image.shape} vs label {code.shape}")
    _write_png_atomic(img_path, image)
    _write_png_atomic(lab_path, code)
    return img_path, lab_path


def build_oct5k(out_root: Path, data_root: Path, graders: tuple[int, ...],
                limit: int | None) -> dict:
    """OCT5k's manual CSVs on the Isfahan B-scans they were drawn on. Only the column axis
    is stretched to 512x512, so the IMAGE is resampled, never the labels."""
    ann = data_root / "public/extracted/oct5k_annotations/OCT5k"
    isfahan = data_root / "public/extracted/isfahan"
    out_dir = out_root / "oct5k"

    entries: list[dict] = []
    stats = {"columns_dropped_crossing": 0, "images": 0, "labels": 0}
    with open(ann / "Scripts/paths/manual_paths.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    for n, (to_rel, from_rel) in enumerate(rows[: limit or None]):
        key = to_rel.split("Images_Manual/", 1)[1][: -len(".png")]
        src = isfahan / from_rel.lstrip("./")
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(src)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shown = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        stem = key.replace("/", "__").replace(" ", "")
        written_image = False
        labels: dict[str, str] = {}
        for g in graders:
            csv_path = ann / f"Boundaries/Boundaries_Manual/Grading_{g}/{key}.csv"
            if not csv_path.is_file():
                continue
            surf = _read_oct5k_csv(csv_path)
            rows_, avail, off = _surface_rows("oct5k", list(surf), shown.shape[1],
                                              shown.shape[0])
            code, st = rasterize_partial(rows_, avail, shown.shape[0])
            stats["columns_dropped_crossing"] += st["columns_dropped_crossing"]
            _accumulate_offset_audit(stats, off)
            img_path, lab_path = _write_pair(out_dir, stem, shown, code,
                                             label_suffix=f"-g{g}-label.png")
            written_image = True
            labels[str(g)] = lab_path.name
            stats["labels"] += 1
        if written_image:
            stats["images"] += 1
            entries.append({
                "stem": stem, "source_key": key, "image": f"{stem}-image.png",
                "labels": labels, "device": "Heidelberg_Spectralis",
                "status": "unknown", "group": key.split("/", 1)[0],
                "volume": key.rsplit("/", 1)[0],
            })
    return {"dataset": "oct5k", "dir": str(out_dir), "entries": entries, "stats": stats}


def _read_oct5k_csv(path: Path) -> list[np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    expected = ["x", "ILM", "OPL", "IS-OS", "IBRPE", "OBRPE"]
    if header != expected:
        raise ValueError(f"{path}: header {header} != {expected}")
    width = len(rows) - 1
    out = [np.full(width, np.nan) for _ in range(5)]
    for i, row in enumerate(rows[1:]):
        if int(row[0]) != i:
            raise ValueError(f"{path}: column index out of order at line {i + 2}")
        for j, cell in enumerate(row[1:]):
            cell = cell.strip()
            if cell:
                out[j][i] = float(cell)
    return out


def build_jhu(out_root: Path, data_root: Path, limit: int | None) -> dict:
    """Sparse control points -> dense surfaces, on the B-scans from the paired ``.vol``. The
    two empty columns are dropped BY POSITION; interpolation is PCHIP along one surface."""
    import scipy.io as sio
    from scipy.interpolate import PchipInterpolator

    root = data_root / "public/extracted/jhu_hcms/OCT_Manual_Delineations-2018_June_29"
    out_dir = out_root / "jhu_hcms"
    entries: list[dict] = []
    stats = {"columns_dropped_crossing": 0, "images": 0, "volumes": 0,
             "frames_without_annotation": 0}

    mats = sorted((root / "delineation").glob("*.mat"))[: limit or None]
    for mat_path in mats:
        subject = mat_path.name.split("_", 1)[0]
        vol_path = root / "vol" / (mat_path.stem + ".vol")
        if not vol_path.is_file():
            raise FileNotFoundError(vol_path)
        blob = sio.loadmat(str(mat_path))
        # 32 of the 35 subjects ship sparse ``control_pts``, 3 the dense ``bd_pts``. Both are
        # handled directly; converting one to the other would front-run their interpolation.
        cp = blob.get("control_pts")
        bd = blob.get("bd_pts")
        if cp is None and bd is None:
            raise ValueError(f"{mat_path.name}: neither control_pts nor bd_pts")
        n_frames = int(cp.shape[0]) if cp is not None else int(bd.shape[1])

        frames = dict(iter_vol_frames(vol_path))
        if len(frames) != n_frames:
            raise ValueError(f"{mat_path.name}: {n_frames} annotated B-scans but the "
                             f".vol has {len(frames)}")
        stats["volumes"] += 1
        stats["volumes_dense" if cp is None else "volumes_control_points"] = stats.get(
            "volumes_dense" if cp is None else "volumes_control_points", 0) + 1

        for f_idx in range(n_frames):
            image = frames[f_idx]
            height, width = image.shape
            surfaces: list[np.ndarray | None] = []
            n_drawn = 0
            if cp is None:
                if bd.shape[0] != width:
                    raise ValueError(f"{mat_path.name}: bd_pts has {bd.shape[0]} columns, "
                                     f"the B-scan has {width}")
                for k in range(bd.shape[2]):
                    col = np.asarray(bd[:, f_idx, k], dtype=np.float64) - 1.0
                    surfaces.append(col)
                    n_drawn += 1
            else:
                for col_i in range(cp.shape[1]):
                    pts = np.asarray(cp[f_idx, col_i], dtype=np.float64)
                    if pts.size == 0:
                        surfaces.append(None)
                        continue
                    # One-based (x, y) from Matlab; PCHIP needs a strictly increasing x.
                    x = pts[:, 0] - 1.0
                    y = pts[:, 1] - 1.0
                    order = np.argsort(x, kind="stable")
                    x, y = x[order], y[order]
                    keep = np.concatenate([[True], np.diff(x) > 0])
                    x, y = x[keep], y[keep]
                    if x.size < 2:
                        surfaces.append(None)
                        continue
                    grid = np.arange(width, dtype=np.float64)
                    dense = np.full(width, np.nan)
                    inside = (grid >= x[0]) & (grid <= x[-1])
                    dense[inside] = PchipInterpolator(x, y)(grid[inside])
                    surfaces.append(dense)
                    n_drawn += 1
            if n_drawn == 0:
                stats["frames_without_annotation"] += 1
                continue

            # Dropping the empty columns BY POSITION lines the eleven up with the nine.
            drawn = [s for s in surfaces if s is not None]
            empty = [i for i, s in enumerate(surfaces) if s is None]
            if cp is not None and empty != JHU_EMPTY_COLUMNS:
                # Nine out of eleven with a DIFFERENT pair missing passes a length check.
                raise ValueError(
                    f"{mat_path.name} frame {f_idx}: empty control-point columns are "
                    f"{empty}, the mapping is written for {JHU_EMPTY_COLUMNS}")
            if len(drawn) != len(SURFACE_TO_BOUNDARY["jhu_hcms"].boundaries):
                raise ValueError(
                    f"{mat_path.name} frame {f_idx}: {len(drawn)} drawn surfaces, the "
                    f"mapping describes {len(SURFACE_TO_BOUNDARY['jhu_hcms'].boundaries)}")
            rows_, avail, off = _surface_rows("jhu_hcms", drawn, width, height)
            code, st = rasterize_partial(rows_, avail, height)
            stats["columns_dropped_crossing"] += st["columns_dropped_crossing"]
            _accumulate_offset_audit(stats, off)

            stem = f"{mat_path.stem}__f{f_idx:03d}"
            _write_pair(out_dir, stem, image, code)
            stats["images"] += 1
            entries.append({
                "stem": stem, "image": f"{stem}-image.png",
                "labels": {"1": f"{stem}-label.png"},
                "device": "Heidelberg_Spectralis",
                "status": "diseased" if subject.startswith("ms") else "healthy",
                "group": subject, "volume": mat_path.stem,
            })
    return {"dataset": "jhu_hcms", "dir": str(out_dir), "entries": entries, "stats": stats}


def build_duke_dme(out_root: Path, data_root: Path, graders: tuple[int, ...],
                   limit: int | None) -> dict:
    """``manualLayers{1,2}`` on the Spectralis frames they annotate: eleven of 61 B-scans,
    NaN on many columns, two graders each written as its own label plane."""
    import scipy.io as sio

    root = data_root / "public/extracted/duke_dme_2015/2015_BOE_Chiu"
    out_dir = out_root / "duke_dme_2015"
    entries: list[dict] = []
    stats = {"columns_dropped_crossing": 0, "images": 0, "labels": 0, "subjects": 0,
             "frames_without_annotation": 0}

    for mat_path in sorted(root.glob("Subject_*.mat"))[: limit or None]:
        blob = sio.loadmat(str(mat_path))
        images = blob["images"]                                   # (H, W, F) uint8
        layers = {g: blob[f"manualLayers{g}"] for g in graders if f"manualLayers{g}" in blob}
        if not layers:
            raise ValueError(f"{mat_path.name}: no manualLayers for graders {graders}")
        stats["subjects"] += 1
        height, width, n_frames = images.shape

        for f_idx in range(n_frames):
            per_grader = {g: np.asarray(v[:, :, f_idx], dtype=np.float64)
                          for g, v in layers.items()}
            per_grader = {g: v for g, v in per_grader.items() if np.isfinite(v).any()}
            if not per_grader:
                stats["frames_without_annotation"] += 1
                continue

            image = np.ascontiguousarray(images[:, :, f_idx])
            stem = f"{mat_path.stem}__f{f_idx:02d}"
            labels_out: dict[str, str] = {}
            for g, surf in per_grader.items():
                if surf.shape != (8, width):
                    raise ValueError(f"{mat_path.name}: manualLayers{g} is {surf.shape}, "
                                     f"expected (8, {width})")
                # Matlab rows are one-based: the SAME ``MATLAB_ROW_OFFSET`` the other readers
                rows_, avail, off = _surface_rows(
                    "duke_dme_2015",
                    [surf[i] - MATLAB_ROW_OFFSET for i in range(8)], width, height)
                code, st = rasterize_partial(rows_, avail, height)
                stats["columns_dropped_crossing"] += st["columns_dropped_crossing"]
                _accumulate_offset_audit(stats, off)
                _, lab_path = _write_pair(out_dir, stem, image, code,
                                          label_suffix=f"-g{g}-label.png")
                labels_out[str(g)] = lab_path.name
                stats["labels"] += 1
            stats["images"] += 1
            entries.append({
                "stem": stem, "image": f"{stem}-image.png", "labels": labels_out,
                "device": "Heidelberg_Spectralis", "status": "diseased",
                "group": mat_path.stem, "volume": mat_path.stem,
            })
    return {"dataset": "duke_dme_2015", "dir": str(out_dir), "entries": entries,
            "stats": stats}


def write_pool_index(report: dict) -> None:
    """Write the deterministic index record consumed by the release loader."""
    key = report["dataset"]
    mapping = SURFACE_TO_BOUNDARY[key]
    blob = {"dataset": key, "mapping": mapping.note,
            "mapping_version": MAPPING_VERSION,
            "boundaries": list(mapping.available),
            "surfaces": surfaces_to_json(mapping.boundaries),
            "offsets_px": offsets_to_json(mapping.offsets_px)}
    blob.update(report.get("index_extra") or {})
    blob["stats"] = report["stats"]
    blob["entries"] = report["entries"]
    dest = Path(report["dir"])
    dest.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(dest / "index.json", blob)


def build_public_pool(out: Path, data_root: Path, *,
                      datasets: Sequence[str] = PUBLIC_DATASETS,
                      graders: tuple[int, ...] = (1, 2, 3),
                      limit: int | None = None) -> list[dict]:
    """Build selected public directories and return aggregate reports."""
    wanted = tuple(datasets)
    if not wanted or any(d not in PUBLIC_DATASETS for d in wanted):
        raise ValueError("datasets must be a nonempty subset of public builders")
    out = Path(out)
    for name in wanted:
        dest = out / name
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"public output directory already contains data: {name}")
    dispatch = {
        "oct5k": lambda: build_oct5k(out, data_root, graders, limit),
        "jhu_hcms": lambda: build_jhu(out, data_root, limit),
        "duke_dme_2015": lambda: build_duke_dme(out, data_root, graders, limit),
    }
    reports = []
    for name in wanted:
        report = dispatch[name]()
        write_pool_index(report)
        reports.append({"dataset": name, "stats": report["stats"]})
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=paths.PARTIAL_POOL_D88_DIR)
    p.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    p.add_argument("--datasets", default=",".join(PUBLIC_DATASETS))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    wanted = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    reports = build_public_pool(args.out, args.data_root, datasets=wanted, limit=args.limit)
    for report in reports:
        print(f"{report['dataset']}: {report['stats'].get('images', 0)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
