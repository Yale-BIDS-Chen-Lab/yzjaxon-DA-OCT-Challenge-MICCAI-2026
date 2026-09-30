"""Read staged Final B-scans, checking image/index agreement and absence of labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from octtta import paths
from octtta.data.release_dataset import (
    DEVICE_TO_VENDOR,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    Sample,
    index_split,
)

_INDEX_REQUIRED_FIELDS = ("vendor_dir", "stem")


def default_root() -> Path:
    """Return the prepared Final unlabelled pool root."""
    return Path(paths.UNLABELED_ALL_D88_DIR)


def _root(root: Path | str | None = None) -> Path:
    return default_root() if root in (None, "") else Path(str(root)).expanduser()


def available_groups(root: Path | str | None = None) -> list[str]:
    """Vendor directories that actually hold images under ``root``, discovered from disk."""
    base = _root(root)
    if not base.is_dir():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if next(d.glob(f"*{IMAGE_SUFFIX}"), None) is not None:
            out.append(d.name)
    return out


def _read_index(base: Path) -> list[dict]:
    idx = base / "index.json"
    if not idx.is_file():
        raise FileNotFoundError("Unlabelled pool index is missing; finish scripts/repro.py build.")
    records = json.loads(idx.read_text()).get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("Unlabelled pool index has no image records.")
    if any(not isinstance(row, dict) or any(key not in row for key in _INDEX_REQUIRED_FIELDS)
           for row in records):
        raise ValueError("Unlabelled pool index lacks vendor_dir or stem.")
    return records


def index_frame_count(root: Path | str | None = None) -> int:
    base = _root(root)
    blob = json.loads((base / "index.json").read_text())
    records = blob.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("Unlabelled index carries no image records.")
    claimed = blob.get("n_images")
    if claimed is not None and int(claimed) != len(records):
        raise ValueError(
            f"Unlabelled index says n_images={claimed} but holds {len(records)} records. "
            f"A header and a body that disagree cannot both be the pool size, and a recipe "
            f"deriving its epoch length from the wrong one still divides evenly.")
    return len(records)


def _scan_root(base: Path) -> list[tuple[str, str]]:
    """Index all image files and refuse labels in the unlabelled pool."""
    out: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*.png")):
        if path.name.endswith(MASK_SUFFIX):
            raise ValueError("A mask was found in the unlabelled pool.")
        if path.name.endswith(IMAGE_SUFFIX):
            out.append((path.relative_to(base).parent.as_posix(), path.name[:-len(IMAGE_SUFFIX)]))
    return out


def local_unlabeled_samples(
    root: Path | str | None = None,
    *,
    groups: Sequence[str] | None = None,
) -> list[Sample]:
    """Load selected staged groups in the historical group/frame order."""
    base = _root(root)
    if not base.is_dir():
        raise FileNotFoundError("Unlabelled pool is missing; run scripts/repro.py build.")
    available = available_groups(base)
    names = list(available) if groups is None else [str(group).strip() for group in groups]
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError("Unlabelled groups must be nonempty and unique.")
    if set(names) - set(available):
        raise FileNotFoundError("Requested unlabelled groups have no images.")
    if set(names) - set(DEVICE_TO_VENDOR):
        raise KeyError("Unlabelled group has no registered vendor.")
    on_disk = _scan_root(base)
    records = _read_index(base)
    by_key = {(str(row["vendor_dir"]), str(row["stem"])): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Unlabelled pool index contains duplicate image records.")
    if set(on_disk) != set(by_key):
        raise ValueError("Unlabelled pool images and index disagree; finish the build.")
    out: list[Sample] = []
    for name in names:
        for sample in index_split(base, name, ""):
            if sample.mask is not None:
                raise ValueError("A mask was found in the unlabelled pool.")
            out.append(Sample(
                image=sample.image, mask=None, device=name,
                status=str(by_key[(name, sample.stem)].get("status") or "unknown"),
                stem=sample.stem, label_kind="exact",
            ))
    return out
