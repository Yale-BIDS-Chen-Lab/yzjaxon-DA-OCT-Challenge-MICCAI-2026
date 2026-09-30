"""Fetch and verify the pinned public data, official data, and SAM-L encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from octtta import paths  # noqa: E402


def expected() -> dict:
    return json.loads((REPO_ROOT / "configs" / "expected.json").read_text())


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, spec: dict) -> None:
    """Require every stated size and digest; a missing pin is an error."""
    if not isinstance(spec.get("bytes"), int) or spec["bytes"] <= 0:
        raise ValueError("download spec lacks a positive byte-count pin")
    algo = "sha256" if "sha256" in spec else "md5" if "md5" in spec else None
    if algo is None or not isinstance(spec[algo], str):
        raise ValueError("download spec lacks a digest pin")
    if Path(path).stat().st_size != spec["bytes"]:
        raise ValueError("downloaded byte count differs from expected.json")
    if digest(path, algo) != spec[algo]:
        raise ValueError("downloaded digest differs from expected.json")


def fetch(url: str, dest: Path, spec: dict) -> Path:
    """Download once to an adjacent temporary file and verify before publishing."""
    if not url.startswith("https://"):
        raise ValueError("download URL must use HTTPS")
    dest = Path(dest)
    if dest.is_file():
        verify(dest, spec)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "octtta-release/1"})
        with urllib.request.urlopen(request, timeout=120) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=1 << 22)
        verify(tmp, spec)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _safe_member_path(root: Path, name: str) -> Path:
    if name.startswith("/"):
        raise ValueError("archive contains an absolute path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("archive contains a path outside extraction root")
    return target


def _archive_inventory(archive: Path) -> set[str]:
    """Read a verified archive's exact regular-file inventory without extracting it."""
    files: set[str] = set()
    root = Path("/archive-inventory")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = _safe_member_path(root, member.filename)
                mode = member.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError("archive contains a symbolic link")
                if member.is_dir():
                    continue
                rel = target.relative_to(root).as_posix()
                if rel in files:
                    raise ValueError("archive contains a duplicate member")
                files.add(rel)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = _safe_member_path(root, member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("archive contains a non-file member")
                rel = target.relative_to(root).as_posix()
                if rel in files:
                    raise ValueError("archive contains a duplicate member")
                files.add(rel)
    else:
        raise ValueError("archive format is not ZIP or TAR")
    if not files:
        raise ValueError("archive contains no regular files")
    return files


def _disk_inventory(root: Path, expected_files: set[str] | None = None) -> set[str]:
    """Keep archive members while ignoring extra generated Python bytecode."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and not (p.suffix == ".pyc"
                                    and "__pycache__" in p.relative_to(root).parts
                                    and expected_files is not None
                                    and p.relative_to(root).as_posix() not in expected_files)}


def extract(archive: Path, dest: Path, *, files: int | None = None) -> int:
    """Extract atomically, or validate every file of an existing extraction."""
    dest = Path(dest)
    inventory = _archive_inventory(archive)
    if files is not None and len(inventory) != files:
        raise ValueError("archive member count differs from expected.json")
    if dest.is_symlink():
        raise ValueError("existing extraction root is a symbolic link")
    if dest.is_dir():
        present = _disk_inventory(dest, inventory)
        if any(p.is_symlink() for p in dest.rglob("*")) or present != inventory:
            raise ValueError("existing extraction inventory differs from verified archive")
        return len(present)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError("temporary extraction directory already exists")
    tmp.mkdir()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(tmp)
        present = _disk_inventory(tmp, inventory)
        if present != inventory:
            raise ValueError("extracted inventory differs from verified archive")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return len(inventory)


def download_public(data_root: Path, pins: dict | None = None) -> dict[str, int]:
    pins = pins or (expected().get("inputs") or {}).get("public_archives")
    required = ("jhu_hcms.zip", "duke_dme_2015.zip", "oct5k_annotations.zip")
    if not isinstance(pins, dict) or any(k not in pins for k in required):
        raise ValueError("expected.json lacks the three public archive pins")
    result = {}
    # The URLs use different ZIP layouts.  The pin names the *destination* root;
    # each archive must contain the dataset directory consumed by its builder.
    expected_child = {"jhu_hcms.zip": "OCT_Manual_Delineations-2018_June_29",
                      "duke_dme_2015.zip": "2015_BOE_Chiu",
                      "oct5k_annotations.zip": "OCT5k"}
    for name in required:
        spec = pins[name]
        root_name = spec.get("extract_root")
        if root_name != Path(name).stem:
            raise ValueError("public archive extraction root differs from expected.json")
        archive = fetch(spec["url"], Path(data_root) / "public/_archives" / name, spec)
        extracted = Path(data_root) / "public/extracted" / root_name
        result[name] = extract(archive, extracted)
        child = expected_child[name]
        if not (extracted / child).is_dir():
            raise ValueError("public archive layout differs from the builder contract")
    return result


def download_official(data_root: Path, pins: dict | None = None) -> dict[str, int]:
    pins = pins or (expected().get("inputs") or {}).get("official")
    if not isinstance(pins, dict) or set(pins) < {"starting_kit", "synthetic"}:
        raise ValueError("expected.json lacks official data pins")
    result = {}
    names = {"starting_kit": "starting_kit.zip", "synthetic": "data_synthetic_v1.0.tar"}
    for key, filename in names.items():
        spec = pins[key]
        archive = fetch(spec["url"], Path(data_root) / "official_archives" / filename,
                        spec)
        result[key] = extract(archive, Path(data_root) / (
            "starting_kit" if key == "starting_kit" else "synthetic_v1.0"),
            files=spec["files"])
    return result


def download_sam(data_root: Path, ckpt_root: Path, spec: dict | None = None) -> dict:
    """Fetch pinned safetensors and export the original one-channel encoder state."""
    spec = spec or (expected().get("inputs") or {}).get("sam_l")
    if not isinstance(spec, dict) or not all(k in spec for k in (
            "repo", "filename", "sha256", "bytes", "export_tensors", "export_params")):
        raise ValueError("expected.json lacks pinned SAM-L source and export dimensions")
    if spec["repo"] != "timm/samvit_large_patch16.sa1b":
        raise ValueError("SAM-L repository differs from the published recipe")
    from huggingface_hub import hf_hub_download
    from octtta.models.vit_fpn import build_timm_encoder
    import torch

    local = Path(hf_hub_download(repo_id=spec["repo"], filename=spec["filename"],
                                 local_dir=Path(data_root) / "pretrained" / "sam_l" /
                                 "samvit_large_patch16.sa1b"))
    verify(local, spec)
    encoder = build_timm_encoder(
        "samvit_large_patch16.sa1b", in_chans=1, pretrained=True,
        pretrained_cfg_overlay={"file": str(local)})
    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    params = sum(int(v.numel()) for v in state.values())
    if len(state) != spec["export_tensors"] or params != spec["export_params"]:
        raise ValueError("SAM-L export tensor structure differs from expected.json")
    dest = Path(ckpt_root) / "e08" / "sam_l_offtheshelf.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": state,
               "provenance": {"kind": "published_encoder_weights",
                              "family": "sam",
                              "backbone": "samvit_large_patch16.sa1b", "in_chans": 1,
                              "source_sha256": spec["sha256"], "tensors": len(state),
                              "params": params, "oct_pretraining": "none"}}
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"source_sha256": spec["sha256"], "tensors": len(state), "params": params}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("public", "official", "sam", "all"))
    parser.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--ckpt-root", type=Path, default=paths.CKPT_DIR)
    args = parser.parse_args(argv)
    if args.group in ("public", "all"):
        print("public:", json.dumps(download_public(args.data_root), sort_keys=True))
    if args.group in ("official", "all"):
        print("official:", json.dumps(download_official(args.data_root), sort_keys=True))
    if args.group in ("sam", "all"):
        print("sam:", json.dumps(download_sam(args.data_root, args.ckpt_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
