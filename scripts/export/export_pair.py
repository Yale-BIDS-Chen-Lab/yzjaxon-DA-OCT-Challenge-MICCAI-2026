#!/usr/bin/env python
"""Export inference weights from one or two training checkpoints.

    python scripts/export/export_pair.py --a CHECKPOINT [--b CHECKPOINT] --out DIR

The checkpoint config travels with its weights. ``--final-recipe`` explicitly applies the
historical Final inference and fusion settings for reproducing that result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from octtta.paths import RUNS_DIR  # noqa: E402

OUT_NAMES = {"A": "model.pt", "B": "model_b.pt"}
FUSION_TABLE = REPO / "configs" / "reproduction" / "fusion.json"
FINAL_INFERENCE = {
    "max_height": None,
    "tta_shapes": None,
    "width_scale_shapes": None,
    "flatten_shapes": None,
    "blank_columns": {"enabled": True, "threshold": 0.0, "min_band": 16,
                      "fill": "background"},
}
PROVENANCE_KEYS = ("quarantine", "quarantine_reasons", "run_kind", "epoch_done",
                   "finetune_from", "epoch", "best_score", "monitor", "last_topology",
                   "config_fingerprint", "weights_only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights(payload: dict, half: str) -> tuple[dict, str]:
    """Select EMA when available, including both student-B checkpoint conventions."""
    keys = ("ema_b", "model_b") if half == "B" and any(
        payload.get(k) is not None for k in ("ema_b", "model_b")) else ("ema", "model")
    for key in keys:
        state = payload.get(key)
        if state is not None:
            if not isinstance(state, dict) or not state:
                raise ValueError(f"{key} must be a nonempty state dict")
            return state, key
    raise ValueError(f"checkpoint has no weights for model {half}")


def export_half(src: Path, dst: Path, half: str, *, final_recipe: bool = False) -> dict:
    """Write an inference-only checkpoint and return its provenance record."""
    import torch

    if half not in OUT_NAMES:
        raise ValueError(f"unknown model slot {half!r}")
    payload = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{src}: expected a checkpoint mapping")
    if half == "B" and any(payload.get(k) is not None for k in ("ema_b", "model_b")):
        if "config_b" not in payload:
            raise ValueError(f"{src}: combined model-B weights need config_b")
    cfg_key = "config_b" if half == "B" and "config_b" in payload else "config"
    cfg = payload.get(cfg_key)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
        raise ValueError(f"{src}: checkpoint needs a model config")
    cfg = copy.deepcopy(cfg)
    state, key = _weights(payload, half)
    if final_recipe:
        cfg["inference"] = {**(cfg.get("inference") or {}), **FINAL_INFERENCE}
        cfg["postproc"] = {**(cfg.get("postproc") or {}), "enabled": True}
        if half == "A":
            cfg["fusion"] = {**json.loads(FUSION_TABLE.read_text()), "enabled": True}
        else:
            cfg.pop("fusion", None)
    record = {"source": str(src), "selected_weights": key, "final_recipe": final_recipe}
    out = {k: payload[k] for k in PROVENANCE_KEYS if k in payload}
    out.update(config=cfg, model=state, ema=state, export=record)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, metavar="CKPT")
    ap.add_argument("--b", type=Path, metavar="CKPT",
                    help="optional second model checkpoint")
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "export_pair", metavar="DIR")
    ap.add_argument("--final-recipe", action="store_true",
                    help="apply the historical Final inference and fusion settings")
    args = ap.parse_args(argv)
    sources = {"A": args.a}
    if args.b is not None:
        sources["B"] = args.b
    for half, src in sources.items():
        if not src.is_file():
            ap.error(f"model {half} checkpoint does not exist: {src}")
    targets = {half: args.out / OUT_NAMES[half] for half in sources}
    info_path = args.out / "EXPORT_INFO.json"
    existing = [p for p in (*targets.values(), info_path) if p.exists()]
    if existing:
        ap.error(f"refusing to overwrite: {existing}; choose another --out")
    info = {"schema": "octtta/export_info/2", "sources": {}, "weights": {}}
    for half, src in sources.items():
        dst = targets[half]
        record = export_half(src, dst, half, final_recipe=args.final_recipe)
        info["sources"][dst.name] = str(src)
        info["weights"][dst.name] = {"sha256": _sha256(dst),
                                     "selected": record["selected_weights"]}
        print(f"[export] {half}: {record['selected_weights']} -> {dst}")
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
