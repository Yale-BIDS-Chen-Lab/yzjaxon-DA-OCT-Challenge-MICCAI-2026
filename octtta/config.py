"""Composable YAML configs with one optional base, dotted overrides, and ``$VAR`` expansion.

The overlay recursively merges mappings, replaces lists, and lets explicit ``null`` values
replace a base value.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_path(ref: str, relative_to: Path) -> Path:
    """Resolve the single config base relative to its overlay, then to configs/."""
    for candidate in ((relative_to.parent / ref), (CONFIG_ROOT / ref)):
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
        if candidate.suffix == "":
            for suffix in (".yaml", ".yml"):
                with_suffix = candidate.with_suffix(suffix)
                if with_suffix.is_file():
                    return with_suffix
    raise FileNotFoundError(f"config base {ref!r} referenced from {relative_to} not found")


def load_config(path: str | Path) -> dict:
    """Load a flattened config or a one-level overlay."""
    path = Path(path)
    if not path.exists():
        alt = CONFIG_ROOT / path
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"config not found: {path}")
    path = path.resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    base_ref = raw.pop("_base_", None)
    if isinstance(base_ref, list):
        if len(base_ref) != 1:
            raise ValueError(f"config {path} must name at most one _base_")
        base_ref = base_ref[0]
    if base_ref is not None:
        if not isinstance(base_ref, str) or not base_ref.strip():
            raise ValueError(f"config {path} has an invalid _base_")
        base_path = _resolve_path(base_ref, path)
        base = yaml.safe_load(base_path.read_text()) or {}
        if not isinstance(base, dict) or "_base_" in base:
            raise ValueError(f"config base {base_path} must be flattened (no nested _base_)")
        merged = _deep_merge(base, raw)
    else:
        merged = raw
    merged.setdefault("_source", str(path))
    return merged


def _coerce(value: str) -> Any:
    """Turn a CLI string into the obvious Python type."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` strings in order."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {key!r} traverses non-dict at {p!r}")
        node[parts[-1]] = _coerce(value)
    return cfg


def _canonical_env() -> dict[str, str]:
    """Defaults for canonical path variables; real environment variables still win."""
    from octtta import paths

    return {
        "OCTTTA_SCRATCH": str(paths.SCRATCH),
        "OCTTTA_DATA": str(paths.DATA_DIR),
        "OCTTTA_CKPT": str(paths.CKPT_DIR),
        "OCTTTA_RUNS": str(paths.RUNS_DIR),
    }


def expand_env(cfg: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ``$VAR`` / ``${VAR}`` and ``~`` inside string values."""
    if env is None:
        env = {**_canonical_env(), **os.environ}
    if isinstance(cfg, dict):
        return {k: expand_env(v, env) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [expand_env(v, env) for v in cfg]
    if isinstance(cfg, str):
        out = os.path.expanduser(os.path.expandvars(
            _TEMPLATE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), cfg)
        ))
        return out
    return cfg


#: ``${NAME}`` or ``$NAME``, matched here so the canonical defaults can substitute too.
_TEMPLATE = __import__("re").compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def snapshot(cfg: dict, out_dir: str | Path, name: str = "config.resolved.yaml") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return dest


def get_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """The normal entry point: load, override, expand."""
    return expand_env(apply_overrides(load_config(path), overrides or []))



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Resolve and print a config.")
    ap.add_argument("config", help="path or name under configs/")
    ap.add_argument("overrides", nargs="*", help="key.path=value")
    args = ap.parse_args()
    print(yaml.safe_dump(get_config(args.config, args.overrides),
                         sort_keys=False, allow_unicode=True))
