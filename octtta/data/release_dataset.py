"""Indexing for the challenge's ``release_dataset`` layout.

``<device>/{Diseased,Healthy}/NNN-image.png`` plus ``NNN-mask.png`` where labels exist;
stems are globally unique and ``Topcon_Maestro2`` is the only labelled domain.
WARNING: at inference the model is handed a flat directory with no metadata, so code
that reads the device off a path works locally and fails on the server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LABELED_DEVICE = "Topcon_Maestro2"

DEVICES = (
    "Topcon_Maestro2",
    "Topcon_Maestro2_unlabeled",
    "Heidelberg_Spectralis",
    "Zeiss_Cirrus",
)

STATUSES = ("Diseased", "Healthy")

#: Device tag for the images to PREDICT; deliberately not one of :data:`DEVICES`.
INFERENCE_DEVICE = "Inference_unlabeled"

#: Status of an inference-directory image: the flat directory carries no metadata.
INFERENCE_STATUS = "unknown"

#: Directory name -> vendor string; a HARD lookup, so a missing row raises mid-run.
DEVICE_TO_VENDOR = {
    "Topcon_Maestro2": "Maestro2",
    "Topcon_Maestro2_unlabeled": "Maestro2",
    "Heidelberg_Spectralis": "Spectralis",
    "Zeiss_Cirrus": "Cirrus",
    "Topcon_Maestro2_unlabeled_widefield": "Maestro2",
    "Topcon_Maestro2_unlabeled_macula": "Maestro2",
    "Topcon_Maestro2_unlabeled_onh": "Maestro2",
    "Topcon_Triton_unlabeled_onh": "Triton",
    "Zeiss_Cirrus_unlabeled_onh": "Cirrus",
    "Zeiss_Cirrus_unlabeled_macula": "Cirrus",
    "Heidelberg_Spectralis_unlabeled_onh": "Spectralis",
    "Heidelberg_Spectralis_unlabeled_wide": "Spectralis",
    INFERENCE_DEVICE: "Unknown",
}

IMAGE_SUFFIX = "-image.png"
MASK_SUFFIX = "-mask.png"

NUM_CLASSES = 10


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    device: str
    status: str
    stem: str
#: What ``mask`` holds; both encodings are uint8 PNGs, so the reader uses this field only.
    label_kind: str = "exact"
#: Acquisition protocol, e.g. ``"Macula, 12 x 12"``; empty for sets that ship only one.
    protocol: str = ""

    def __post_init__(self) -> None:
        if self.label_kind not in ("exact", "interval"):
            raise ValueError(f"unknown label_kind {self.label_kind!r}")

    @property
    def vendor(self) -> str:
        # The release's own four devices stay strict, so a typo raises; public sets fall back.
        if self.label_kind == "exact":
            return DEVICE_TO_VENDOR[self.device]
        return DEVICE_TO_VENDOR.get(self.device, self.device)

    @property
    def labeled(self) -> bool:
        return self.mask is not None

    @property
    def partial(self) -> bool:
        """True when the label names a set of classes rather than one."""
        return self.label_kind == "interval"

    @property
    def mask_name(self) -> str:
        """Filename the evaluator expects a prediction under."""
        return f"{self.stem}{MASK_SUFFIX}"


def index_split(root: Path | str, device: str, status: str) -> list[Sample]:
    folder = Path(root) / device / status
    if not folder.is_dir():
        return []
    out = []
    for img in sorted(folder.glob(f"*{IMAGE_SUFFIX}")):
        stem = img.name[: -len(IMAGE_SUFFIX)]
        mask = folder / f"{stem}{MASK_SUFFIX}"
        out.append(
            Sample(image=img, mask=mask if mask.exists() else None,
                   device=device, status=status, stem=stem)
        )
    return out


def index_release(
    root: Path | str,
    devices: Sequence[str] = DEVICES,
    statuses: Sequence[str] = STATUSES,
) -> list[Sample]:
    return [s for d in devices for st in statuses for s in index_split(root, d, st)]


def labeled_samples(root: Path | str) -> list[Sample]:
    return [s for s in index_release(root, devices=[LABELED_DEVICE]) if s.labeled]


def unlabeled_samples(
    root: Path | str,
    include_maestro2_unlabeled: bool = True,
) -> list[Sample]:
    devices = [d for d in DEVICES if d != LABELED_DEVICE]
    if not include_maestro2_unlabeled:
        devices = [d for d in devices if d != "Topcon_Maestro2_unlabeled"]
    return index_release(root, devices=devices)


def index_flat_inference_dir(input_dir: Path | str) -> list[Path]:
    """Flat directory, images only. Globs ``*-image.png``, then falls back to any PNG."""
    d = Path(input_dir)
    imgs = sorted(d.glob(f"*{IMAGE_SUFFIX}"))
    if imgs:
        return imgs
    return sorted(p for p in d.glob("*.png") if not p.name.endswith(MASK_SUFFIX))


#: Probe order for the inference directory; three implementations must agree (a test runs all).
INFERENCE_DIR_PROBE_ORDER = (("val", "images"), ("testing_data",))


def resolve_inference_dir(input_data: Path | str) -> Path:
    """``input_data`` -> the flat directory of images to predict, first that exists."""
    root = Path(input_data)
    for parts in INFERENCE_DIR_PROBE_ORDER:
        cand = root.joinpath(*parts)
        if cand.is_dir():
            return cand
    return root


def output_name_for(image_path: Path | str) -> str:
    name = Path(image_path).name
    if name.endswith(IMAGE_SUFFIX):
        return name[: -len(IMAGE_SUFFIX)] + MASK_SUFFIX
    return Path(name).stem + MASK_SUFFIX
