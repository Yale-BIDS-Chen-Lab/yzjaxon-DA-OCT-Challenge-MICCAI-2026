"""Canonical local roots for data, checkpoints, and runs.

Root precedence is environment variable > ``configs/local.yaml`` > built-in default.
The local file is private (git-ignored) and contains only machine-specific roots here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_FILE = REPO_ROOT / "configs" / "local.yaml"
_ROOT_KEYS = {"scratch", "data", "runs", "ckpt"}


def _load_local_roots() -> dict[str, Any]:
    if not _LOCAL_FILE.exists():
        return {}
    if not _LOCAL_FILE.is_file():
        raise ValueError(f"local settings path is not a file: {_LOCAL_FILE}")
    try:
        raw = yaml.safe_load(_LOCAL_FILE.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read local settings {_LOCAL_FILE}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"local settings {_LOCAL_FILE} must be a YAML mapping")
    if "scratch" not in raw or not isinstance(raw["scratch"], str) or not raw["scratch"].strip():
        raise ValueError(f"local settings {_LOCAL_FILE} must provide a non-empty 'scratch' root")
    return {key: raw[key] for key in _ROOT_KEYS if key in raw}


_LOCAL_ROOTS = _load_local_roots()


def _root(env_name: str, local_name: str | None, default: str | Path) -> Path:
    """Resolve one root, rejecting configured empty or non-path values."""
    if env_name in os.environ:
        value: Any = os.environ[env_name]
        source = env_name
    elif (local_name is not None and local_name in _LOCAL_ROOTS
          and _LOCAL_ROOTS[local_name] is not None):
        value = _LOCAL_ROOTS[local_name]
        source = f"{_LOCAL_FILE}:{local_name}"
    else:
        value = default
        source = "built-in default"
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{source} must be a non-empty filesystem path")
    return Path(value).expanduser()


_BUILTIN_SCRATCH = Path.home() / "oct_tta_scratch"
SCRATCH = _root("OCTTTA_SCRATCH", "scratch", _BUILTIN_SCRATCH)
DATA_DIR = _root("OCTTTA_DATA", "data", SCRATCH / "oct_tta_data")
SYNTHETIC_ROOT = DATA_DIR / "synthetic_v1.0" / "release_dataset"
DERIVED_DIR = DATA_DIR / "derived"
PARTIAL_POOL_D88_DIR = _root("OCTTTA_PARTIAL_D88", None,
                            DERIVED_DIR / "partial_all16_d88")
UNLABELED_ALL_D88_DIR = _root("OCTTTA_UNLABELED_ALL_D88", None,
                              DERIVED_DIR / "unlabeled_all_d88")
AIREADI_ROOT = DATA_DIR / "public" / "ai_readi"
CKPT_DIR = _root("OCTTTA_CKPT", "ckpt", SCRATCH / "oct_tta_ckpts")
PRETRAINED_DIR = CKPT_DIR / "pretrained"
RUNS_DIR = _root("OCTTTA_RUNS", "runs", SCRATCH / "oct_tta_runs")
HF_CACHE = Path(os.environ.get("HF_HOME", str(SCRATCH / "hf_cache"))).expanduser()

#: The only trainable pool roots; the registry has no evaluation-only entries.
POOL_ROOTS = {
    "challenge_release_synthetic": SYNTHETIC_ROOT,
    "challenge_release": DATA_DIR / "release_dataset",
    "public_partial_labels_all16_d88": PARTIAL_POOL_D88_DIR,
    "unlabeled_local_all_d88": UNLABELED_ALL_D88_DIR,
}
