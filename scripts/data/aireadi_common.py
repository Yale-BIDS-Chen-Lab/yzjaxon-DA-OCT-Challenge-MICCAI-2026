"""AI-READI metadata selection, frame QC, and keyed volume exclusions."""

from __future__ import annotations

import csv

import hashlib

import hmac

import json

import re

from collections import defaultdict

from dataclasses import dataclass

from pathlib import Path

from typing import Iterable, Sequence

import numpy as np

VENDOR_BY_DIR: dict[str, str] = {'topcon_maestro2': 'Topcon_Maestro2', 'topcon_triton': 'Topcon_Triton', 'heidelberg_spectralis': 'Heidelberg_Spectralis', 'zeiss_cirrus': 'Zeiss_Cirrus'}

SENTINEL_BOUNDS: dict[str, tuple[float, float]] = {'Topcon_Maestro2': (-0.5, 1000000000.0), 'Topcon_Triton': (-0.5, 1000000000.0), 'Zeiss_Cirrus': (0.5, 1000000000.0), 'Heidelberg_Spectralis': (-0.5, 1000000000.0)}

WIDEFIELD_MM = 12.0

@dataclass(frozen=True)
class OctaRecord:
    person_id: str
    vendor: str
    protocol: str
    anatomy: str
    field_mm: float
    laterality: str
    seg_path: str
    structural_path: str
    n_frames: int
    width: int
    study_group: str = ''
    status_proxy: str = 'unknown'
    split: str = 'unknown'

    @property
    def cell(self) -> tuple[str, str, str]:
        """The stratification cell: vendor x protocol x status proxy."""
        return (self.vendor, self.protocol, self.status_proxy)

def load_participants(root: Path | str | None=None) -> dict[str, dict]:
    if root is None:
        raise ValueError("AI-READI root is required")
    root = Path(root)
    path = root / 'participants.tsv'
    if not path.is_file():
        raise FileNotFoundError('AI-READI participant metadata is missing')
    out: dict[str, dict] = {}
    with open(path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if not reader.fieldnames or not {'person_id', 'recommended_split'} <= set(reader.fieldnames):
            raise ValueError('AI-READI participant metadata lacks required columns')
        for row in reader:
            pid = (row.get('person_id') or '').strip()
            split = (row.get('recommended_split') or '').strip().lower()
            if None in row or not re.fullmatch(r'[0-9]{4}', pid):
                raise ValueError('AI-READI participant metadata has an invalid record')
            if pid in out:
                raise ValueError('AI-READI participant metadata has a duplicate record')
            if split not in {'train', 'val', 'test'}:
                raise ValueError('AI-READI participant metadata has an invalid split')
            out[pid] = {k: (v or '').strip() for (k, v) in row.items()}
    if not out:
        raise ValueError('AI-READI participant metadata is empty')
    structural = root / 'retinal_oct' / 'manifest.tsv'
    if not structural.is_file():
        raise FileNotFoundError('AI-READI structural metadata is missing')
    for manifest, structural_only in ((structural, True), (octa_manifest_path(root), False)):
        with manifest.open(newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            required = {'person_id', 'imaging', 'filepath'} if structural_only else {'person_id'}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError('AI-READI manifest lacks required columns')
            checked = 0
            for row in reader:
                if structural_only and (row.get('imaging') or '').strip().upper() != 'OCT':
                    continue
                checked += 1
                if (row.get('person_id') or '').strip() not in out:
                    raise ValueError('AI-READI manifest references an unknown participant')
            if not checked:
                raise ValueError('AI-READI manifest has no records to validate')
    return out

def status_proxy_of(study_group: str) -> str:
    g = (study_group or '').strip().lower()
    if not g:
        return 'unknown'
    return 'healthy' if g == 'healthy' else 'diseased'

def split_of(recommended_split: str) -> str:
    s = (recommended_split or '').strip().lower()
    if s == 'test':
        return 'lockbox'
    if s in ('train', 'val'):
        return 'tune'
    raise ValueError('AI-READI participant metadata has an invalid split')

def _rel(p: str) -> str:
    p = (p or '').strip().lstrip('/')
    return p[len('dataset/'):] if p.startswith('dataset/') else p

def parse_field_mm(protocol: str) -> float:
    text = (protocol or '').lower().replace('×', 'x')
    for chunk in text.split(','):
        parts = [c.strip() for c in chunk.split('x')]
        if len(parts) == 2:
            try:
                return float(parts[0])
            except ValueError:
                continue
    return float('nan')

def anatomy_of(protocol: str) -> str:
    text = (protocol or '').lower()
    if 'optic disc' in text or 'optic_disc' in text:
        return 'OpticDisc'
    mm = parse_field_mm(protocol)
    if mm == mm and mm >= WIDEFIELD_MM:
        return 'WideField'
    return 'Macula'

_SOP_UID_RE = re.compile('^\\d+(?:\\.\\d+)+$')

def octa_manifest_path(root: Path) -> Path:
    for candidate in (root / 'retinal_octa' / 'manifest.tsv', root / '.transfer' / 'probe' / 'octa_manifest.tsv'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('AI-READI OCTA metadata is missing')

def index_octa(root: Path | str | None=None, *, require_local: bool=True, include_optic_disc: bool=False) -> list[OctaRecord]:
    if root is None:
        raise ValueError("AI-READI root is required")
    root = Path(root)
    people = load_participants(root)
    manifest = octa_manifest_path(root)
    out: list[OctaRecord] = []
    with open(manifest, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            pid = (row.get('person_id') or '').strip()
            if pid not in people:
                raise ValueError('AI-READI OCTA metadata references an unknown participant')
            seg = _rel(row.get('associated_segmentation_file_path', ''))
            struct = _rel(row.get('associated_structural_oct_file_path', ''))
            if not seg or not struct:
                continue
            parts = seg.split('/')
            vendor_dir = parts[2] if len(parts) > 2 else ''
            vendor = VENDOR_BY_DIR.get(vendor_dir)
            if vendor is None:
                continue
            protocol = (row.get('anatomic_region') or '').strip()
            anatomy = anatomy_of(protocol)
            if anatomy == 'OpticDisc' and (not include_optic_disc):
                continue
            if require_local and (not ((root / seg).is_file() and (root / struct).is_file())):
                continue
            info = people[pid]
            out.append(OctaRecord(person_id=pid, vendor=vendor, protocol=protocol, anatomy=anatomy, field_mm=parse_field_mm(protocol), laterality=(row.get('laterality') or '').strip(), seg_path=seg, structural_path=struct, n_frames=_int(row.get('flow_cube_number_of_frames')), width=_int(row.get('flow_cube_width')), study_group=info.get('study_group', ''), status_proxy=status_proxy_of(info.get('study_group', '')), split=split_of(info.get('recommended_split', ''))))
    return out

def _int(text: str | None) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return 0

VOLUME_TAG_HEX = 12

def volume_tag(structural_path: str) -> str:
    digest = hashlib.sha256(str(structural_path).encode('utf-8')).hexdigest()
    return f'v{digest[:VOLUME_TAG_HEX]}'

def exclusion_selectors(record) -> dict:
    return {'person_id': record.person_id, 'vendor': record.vendor, 'laterality': record.laterality, 'volume_tag': volume_tag(record.structural_path)}

def _cell_seed(seed: int, cell: tuple[str, ...]) -> int:
    key = '|'.join((str(seed), *cell)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')

def balanced_sample(records: Sequence[OctaRecord], *, per_cell: int, seed: int, split: str | None=None, max_per_person_per_cell: int=1) -> list[OctaRecord]:
    if split is not None:
        records = [r for r in records if r.split == split]
    by_cell: dict[tuple[str, str, str], dict[str, list[OctaRecord]]] = defaultdict(lambda : defaultdict(list))
    for rec in records:
        by_cell[rec.cell][rec.person_id].append(rec)
    picked: list[OctaRecord] = []
    for cell in sorted(by_cell):
        people = by_cell[cell]
        rng = np.random.default_rng(_cell_seed(seed, cell))
        order = sorted(people)
        rng.shuffle(order)
        for pid in order:
            people[pid] = sorted(people[pid], key=lambda r: r.seg_path)
        chosen: list[OctaRecord] = []
        depth = 0
        while len(chosen) < per_cell and depth < max_per_person_per_cell:
            progressed = False
            for pid in order:
                if depth < len(people[pid]):
                    chosen.append(people[pid][depth])
                    progressed = True
                    if len(chosen) >= per_cell:
                        break
            if not progressed:
                break
            depth += 1
        picked.extend(chosen)
    return picked

def cohort_counts(records: Iterable[OctaRecord]) -> dict[str, dict[str, int]]:
    vols: dict[str, int] = defaultdict(int)
    people: dict[str, set] = defaultdict(set)
    for r in records:
        key = f'{r.vendor}|{r.protocol}|{r.status_proxy}'
        vols[key] += 1
        people[key].add(r.person_id)
    return {k: {'volumes': vols[k], 'persons': len(people[k])} for k in sorted(vols)}

@dataclass(frozen=True)
class SurfacePick:
    ilm: int
    outer: int
    basis: str
    detail: str = ''

def pick_surfaces(vendor: str, labels: Sequence[str]) -> SurfacePick:
    names = [str(x).strip() for x in labels]
    upper = [n.upper() for n in names]
    n = len(names)
    if vendor in ('Topcon_Maestro2', 'Topcon_Triton'):
        if n < 9:
            raise ValueError(f'{vendor}: expected >=9 height-map frames, got {n} ({names})')
        return SurfacePick(ilm=0, outer=8, basis='depth', detail=f'frames are depth-ordered; label list {names} is a permutation' + (" ; frame 9 'CSI' discarded as degenerate" if n > 9 else ''))
    if vendor == 'Heidelberg_Spectralis':
        try:
            return SurfacePick(ilm=upper.index('ILM'), outer=upper.index('BM'), basis='name', detail='Spectralis labels are correct; BM is the deepest surface')
        except ValueError:
            raise ValueError(f'{vendor}: need ILM and BM in {names}') from None
    if vendor == 'Zeiss_Cirrus':
        try:
            return SurfacePick(ilm=upper.index('ILM'), outer=upper.index('RPE'), basis='name', detail='Cirrus ships only ILM and RPE; RPE is the outer boundary available (not BM -- a systematic offset of one RPE thickness, ~25 um, applies)')
        except ValueError:
            raise ValueError(f'{vendor}: need ILM and RPE in {names}') from None
    raise ValueError(f'unknown vendor {vendor!r}')

STORAGE_LABELS: dict[str, tuple[str, ...]] = {'Topcon_Maestro2': ('ILM', 'RNFL/GCL', 'GCL', 'IPL/INL', 'IS/OS', 'OS/RPE', 'BM', 'OPL', 'ELM'), 'Topcon_Triton': ('ILM', 'RNFL/GCL', 'GCL', 'IPL/INL', 'IS/OS', 'OS/RPE', 'BM', 'OPL', 'ELM', 'CSI'), 'Heidelberg_Spectralis': ('ILM', 'BM', 'RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ELM', 'PR1', 'PR2', 'RPE'), 'Zeiss_Cirrus': ('ILM', 'RPE')}

DEPTH_ORDER: dict[str, tuple[str, ...]] = {'Topcon_Maestro2': STORAGE_LABELS['Topcon_Maestro2'], 'Topcon_Triton': STORAGE_LABELS['Topcon_Triton'], 'Heidelberg_Spectralis': ('ILM', 'RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ELM', 'PR1', 'PR2', 'RPE', 'BM'), 'Zeiss_Cirrus': ('ILM', 'RPE')}

SELECTION_BASIS: dict[str, str] = {'Topcon_Maestro2': 'position', 'Topcon_Triton': 'position', 'Heidelberg_Spectralis': 'name', 'Zeiss_Cirrus': 'name'}

def depth_ordered_surfaces(vendor: str, labels: Sequence[str]) -> tuple[int, ...]:
    got = tuple((str(x).strip() for x in labels))
    expect = STORAGE_LABELS.get(vendor)
    if expect is None:
        raise ValueError(f'unknown vendor {vendor!r}')
    if got != expect:
        raise ValueError(f'{vendor}: this height map ships surfaces {list(got)}, the release ships {list(expect)}. The surface->boundary mapping is written for the latter; measure before mapping anything from this volume')
    order = DEPTH_ORDER[vendor]
    if SELECTION_BASIS[vendor] == 'position':
        return tuple(range(len(order)))
    return tuple((got.index(name) for name in order))

def valid_mask(heights: np.ndarray, vendor: str) -> np.ndarray:
    (lo, hi) = SENTINEL_BOUNDS[vendor]
    return np.isfinite(heights) & (heights > lo) & (heights < hi)

def read_heightmap(path: Path | str, vendor: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    labels = [(getattr(seg, 'SegmentLabel', '') or '').strip() for seg in getattr(ds, 'SegmentSequence', []) or []]
    return (arr, valid_mask(arr, vendor), labels)

def structural_geometry(path: Path | str) -> tuple[int, int, int]:
    import pydicom
    meta = pydicom.dcmread(str(path), stop_before_pixels=True)
    return (int(getattr(meta, 'NumberOfFrames', 1)), int(meta.Rows), int(meta.Columns))

def structural_frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        span = float(frame.max() - frame.min()) or 1.0
        frame = (frame - frame.min()) / span * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)

def read_structural_frame(path: Path | str, index: int) -> np.ndarray:
    from pydicom.pixels import pixel_array
    return structural_frame_to_uint8(np.asarray(pixel_array(str(path), index=int(index))))

def boundaries_for_bscan(heights: np.ndarray, valid: np.ndarray, pick: SurfacePick, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ilm = heights[pick.ilm, frame].astype(np.float64)
    outer = heights[pick.outer, frame].astype(np.float64)
    ok = valid[pick.ilm, frame] & valid[pick.outer, frame] & (outer > ilm)
    return (ilm, outer, ok)

QC_MIN_COLUMN_RANGE = 60.0

DEFAULT_MAX_TAKE = 16

QC_MIN_COLUMN_CORRELATION = 0.15

QC_MIN_TISSUE_BAND_SNR = 6.0

QC_SNR_BLOCK_PX = (16, 10)

QC_SNR_MIN_DEPTH_BLOCKS = 8

QC_REASON_COLUMN_RANGE = 'column_range'

QC_REASON_COLUMN_CORRELATION = 'column_correlation'

QC_REASON_TISSUE_BAND_SNR = 'tissue_band_snr'

QC_NO_RETINA_REASONS: tuple[str, ...] = (QC_REASON_COLUMN_CORRELATION, QC_REASON_TISSUE_BAND_SNR)

QC_REASONS: tuple[str, ...] = (QC_REASON_COLUMN_RANGE,) + QC_NO_RETINA_REASONS

_WIDEFIELD_REGIONS = {'wide field', 'macula, 12 x 12'}

def is_widefield_region(anatomic_region: str) -> bool:
    return (anatomic_region or '').strip().lower() in _WIDEFIELD_REGIONS

@dataclass(frozen=True)
class PretrainVolume:
    person_id: str
    vendor: str
    model: str
    anatomic_region: str
    laterality: str
    height: int
    width: int
    n_frames: int
    src_path: str
    take_frames: tuple[int, ...]

    @property
    def is_widefield(self) -> bool:
        return is_widefield_region(self.anatomic_region)

def choose_pretrain_frames(n_frames: int, max_take: int=DEFAULT_MAX_TAKE) -> tuple[int, ...]:
    if n_frames <= 0:
        return ()
    if n_frames <= max_take:
        return tuple(range(n_frames))
    idx = np.linspace(0, n_frames - 1, max_take)
    return tuple((int(i) for i in np.unique(np.round(idx).astype(int))))

def column_correlation(frame: np.ndarray) -> float:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return 0.0
    (x, y) = (arr[:, :-1], arr[:, 1:])
    xm = x - x.mean(axis=0, dtype=np.float64)
    ym = y - y.mean(axis=0, dtype=np.float64)
    num = (xm * ym).sum(axis=0, dtype=np.float64)
    den = np.sqrt((xm * xm).sum(axis=0, dtype=np.float64) * (ym * ym).sum(axis=0, dtype=np.float64))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)
    return float(r.mean())

def _block_mean(arr: np.ndarray, block_rows: int, block_cols: int) -> tuple[np.ndarray, float]:
    (h, w) = arr.shape
    rb = np.arange(0, h, max(1, min(block_rows, h)))
    cb = np.arange(0, w, max(1, min(block_cols, w)))
    s = np.add.reduceat(arr, rb, axis=0)
    s = np.add.reduceat(s, cb, axis=1)
    r_len = np.diff(np.append(rb, h))
    c_len = np.diff(np.append(cb, w))
    counts = np.outer(r_len, c_len).astype(np.float64)
    return (s / counts, float(counts.mean()))

def tissue_band_snr(frame: np.ndarray) -> float:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    block_rows = max(1, min(QC_SNR_BLOCK_PX[0], arr.shape[0] // QC_SNR_MIN_DEPTH_BLOCKS))
    (blk, n_per_block) = _block_mean(arr, block_rows, QC_SNR_BLOCK_PX[1])
    if blk.shape[0] < 2:
        return 0.0
    noise = float(arr.std()) / np.sqrt(max(n_per_block, 1.0)) + 1e-06
    peak = blk.max(axis=0) - np.median(blk, axis=0)
    return float(np.median(peak) / noise)

def frame_qc_reason(frame: np.ndarray) -> str | None:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return QC_REASON_COLUMN_RANGE
    col_range = arr.max(axis=0) - arr.min(axis=0)
    if float(np.median(col_range)) < QC_MIN_COLUMN_RANGE:
        return QC_REASON_COLUMN_RANGE
    if column_correlation(arr) < QC_MIN_COLUMN_CORRELATION:
        return QC_REASON_COLUMN_CORRELATION
    if tissue_band_snr(arr) < QC_MIN_TISSUE_BAND_SNR:
        return QC_REASON_TISSUE_BAND_SNR
    return None

def frame_passes_qc(frame: np.ndarray) -> bool:
    return frame_qc_reason(frame) is None

def npz_relpath(vol: PretrainVolume) -> str:
    stem = Path(vol.src_path).stem
    return f'aireadi/{vol.person_id}/{stem}.npz'


FP_KEY_DOMAIN = b"octtta/" b"volume-fingerprint/v1\n"
FP_HEX = 16


def sop_uid_from_path(path: str | Path) -> str:
    """Read the SOP UID from a licensed structural path without echoing it on failure."""
    token = Path(str(path)).stem.rsplit("_", 1)[-1]
    if not _SOP_UID_RE.match(token):
        raise ValueError("structural path has no numeric SOP UID suffix")
    return token


def _oct_paths(manifest: Path) -> list[str]:
    paths = []
    with Path(manifest).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("imaging") or "").strip().upper() == "OCT":
                rel = _rel(row.get("filepath", ""))
                if not rel:
                    raise ValueError("an OCT manifest row has no structural path")
                paths.append(rel)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("OCT manifest paths are empty or repeated")
    return paths


def fingerprint_key(manifest: Path | str) -> bytes:
    """Derive the private key from all sorted structural paths; never persist it."""
    paths = _oct_paths(Path(manifest))
    return hashlib.sha256(FP_KEY_DOMAIN + b"\n".join(
        p.encode("utf-8") for p in sorted(paths)) + b"\n").digest()


def fingerprint(path: str | Path, key: bytes) -> str:
    """Return a 16-hex keyed fingerprint of a normalized structural path."""
    if len(key) != hashlib.sha256().digest_size:
        raise ValueError("invalid volume fingerprint key")
    return hmac.new(key, _rel(str(path)).encode("utf-8"), hashlib.sha256).hexdigest()[:FP_HEX]


def test_split_persons(root: Path | str) -> set[str]:
    """Read the release's person-level test split from licensed metadata."""
    return {pid for pid, row in load_participants(root).items()
            if (row.get("recommended_split") or "").strip().lower() == "test"}


@dataclass(frozen=True)
class Exclusions:
    """Keyed exclusions validated against the local structural manifest."""
    root: Path
    key: bytes
    test_split: frozenset[str]
    participants: frozenset[str]
    never_train: frozenset[str]
    labelled_quality: frozenset[tuple[str, str]]
    unlabelled_duplicates: frozenset[tuple[str, str]]
    config_sha256: str

    @classmethod
    def load(cls, root: Path | str, config_path: Path | str | None = None) -> Exclusions:
        root = Path(root)
        use_default_config = config_path is None
        config_path = Path(config_path) if config_path is not None else (
            Path(__file__).resolve().parents[2] / "configs" / "aireadi_exclusions.json")
        raw = config_path.read_bytes()
        blob = json.loads(raw)
        if blob.get("schema") != 1 or blob.get("fingerprint") != "hmac-sha256-16-v1":
            raise ValueError("unsupported AI-READI exclusion schema")
        manifest = root / "retinal_oct" / "manifest.tsv"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != blob.get("manifest_sha256"):
            raise ValueError("AI-READI exclusion manifest digest mismatch")
        paths = _oct_paths(manifest)
        if blob.get("manifest_rows") != len(paths):
            raise ValueError("AI-READI exclusion manifest count mismatch")
        key = fingerprint_key(manifest)
        people = load_participants(root)
        rows = {}
        with manifest.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if (row.get("imaging") or "").strip().upper() != "OCT":
                    continue
                if (row.get("person_id") or "").strip() not in people:
                    raise ValueError("AI-READI structural metadata references an unknown participant")
                rows[_rel(row["filepath"])] = row
        fp_to_info = {fingerprint(path, key): (path, rows[path]) for path in paths}
        if len(fp_to_info) != len(paths):
            raise ValueError("AI-READI volume fingerprint collision")
        lists = blob.get("lists")
        if not isinstance(lists, dict):
            raise ValueError("AI-READI exclusion lists are missing")
        parsed = {}
        for name in ("never_train", "labelled_quality", "unlabelled_duplicates"):
            records = lists.get(name)
            if not isinstance(records, list):
                raise ValueError("AI-READI exclusion list is malformed")
            pairs = []
            for rec in records:
                if not isinstance(rec, dict) or set(rec) != {"fp", "vendor"}:
                    raise ValueError("AI-READI exclusion record is malformed")
                fp, vendor = rec["fp"], rec["vendor"]
                if (not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{16}", fp)
                        or not isinstance(vendor, str) or not vendor):
                    raise ValueError("AI-READI exclusion record has invalid fields")
                if fp not in fp_to_info:
                    raise ValueError("AI-READI exclusion fingerprint is absent from manifest")
                path, row = fp_to_info[fp]
                expected_vendor = (
                    (row.get("manufacturers_model_name") or "").strip()
                    if name == "never_train" else
                    VENDOR_BY_DIR.get(path.split("/")[2]) if name == "labelled_quality" else
                    (row.get("manufacturer") or "").strip()
                )
                if vendor != expected_vendor:
                    raise ValueError("AI-READI exclusion vendor differs from manifest")
                pairs.append((fp, vendor))
            if pairs != sorted(set(pairs)):
                raise ValueError("AI-READI exclusion list must be sorted and unique")
            parsed[name] = pairs
        for name in ("never_train", "labelled_quality", "unlabelled_duplicates"):
            fps = [fp for fp, _ in parsed[name]]
            if len(fps) != len(set(fps)):
                raise ValueError("AI-READI exclusion fingerprint occurs twice")
        digest = hashlib.sha256(raw).hexdigest()
        if use_default_config:
            expected = json.loads((Path(__file__).resolve().parents[2] / "configs" / "expected.json").read_text())
            pin = ((expected.get("aireadi") or {}).get("exclusions") or {}).get("file_sha256")
            if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin):
                raise ValueError("AI-READI exclusion file hash pin is missing")
            if digest != pin:
                raise ValueError("AI-READI exclusion file hash differs from expected.json")
        return cls(root, key, frozenset(pid for pid, row in people.items()
                   if row["recommended_split"].strip().lower() == "test"),
                   frozenset(people),
                   frozenset(fp for fp, _ in parsed["never_train"]),
                   frozenset(parsed["labelled_quality"]),
                   frozenset(parsed["unlabelled_duplicates"]), digest)

    def fingerprint(self, path: str | Path) -> str:
        return fingerprint(path, self.key)

    def is_never_train(self, person_id: str, structural_path: str) -> bool:
        if person_id not in self.participants:
            raise ValueError("AI-READI selection references an unknown participant")
        return person_id in self.test_split or self.fingerprint(structural_path) in self.never_train

    def excludes_labelled_quality(self, structural_path: str, vendor: str) -> bool:
        return (self.fingerprint(structural_path), vendor) in self.labelled_quality

    def excludes_unlabelled_duplicate(self, structural_path: str, vendor: str) -> bool:
        return (self.fingerprint(structural_path), vendor) in self.unlabelled_duplicates


def aireadi_pretrain_volumes(root: Path | str, *, exclusions: Exclusions | None = None,
                             max_take: int = DEFAULT_MAX_TAKE) -> list[PretrainVolume]:
    """Select structural OCT volumes with the established frame plan."""
    root = Path(root)
    ex = exclusions if exclusions is not None else Exclusions.load(root)
    volumes = []
    with (root / "retinal_oct" / "manifest.tsv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("imaging") or "").strip().upper() != "OCT":
                continue
            person = str(row.get("person_id") or "").strip()
            rel = _rel(row.get("filepath", ""))
            if ex.is_never_train(person, rel):
                continue
            count = int(row.get("number_of_frames") or 0)
            take = choose_pretrain_frames(count, max_take=max_take)
            if not take:
                continue
            volumes.append(PretrainVolume(person_id=person,
                vendor=(row.get("manufacturer") or "").strip(),
                model=(row.get("manufacturers_model_name") or "").strip(),
                anatomic_region=(row.get("anatomic_region") or "").strip(),
                laterality=(row.get("laterality") or "").strip(),
                height=int(row.get("height") or 0), width=int(row.get("width") or 0),
                n_frames=count, src_path=rel, take_frames=take))
    if not volumes:
        raise ValueError("no eligible structural OCT volumes")
    return volumes
