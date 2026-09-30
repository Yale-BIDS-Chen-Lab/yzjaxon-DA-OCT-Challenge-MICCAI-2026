"""The shipped pair as ``configs/published_pair.json`` records it.

``configs/published_pair.json`` is the one reference of the published weights: the non-tensor
payload of ``model.pt`` (A, SAM ViT-L) and ``model_b.pt`` (B, CNN) and their tensor manifests.
Without the weights, the baked configs must rebuild exactly the recorded architectures, plans
and fusion block. With them (``needs("shipped_weights")``), the files must be the published
pins, their payloads must be what the reference records, and both must load for inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest
from conftest import needs, needs_root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLISHED_PAIR = ROOT / "configs" / "published_pair.json"
EXPECTED = ROOT / "configs" / "expected.json"
FUSION_TABLE = ROOT / "configs" / "reproduction" / "fusion.json"

PP = json.loads(PUBLISHED_PAIR.read_text())
EXP = json.loads(EXPECTED.read_text())
MODELS = ("A", "B")
FILE_OF = {"A": "model.pt", "B": "model_b.pt"}
TOKEN = "${OCTTTA_SCRATCH}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg(model: str) -> dict:
    return PP[model]["config"]


def manifest(state: dict) -> list:
    return [[k, list(t.shape), str(t.dtype).replace("torch.", "")] for k, t in state.items()]


def _normalise(value):
    """Paths spelled from ``${OCTTTA_SCRATCH}``, non-finite floats as strings."""
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, str):
        return re.sub(r"^/\S*?(?=/oct_tta_(?:data|runs|ckpts)(?:/|$))", TOKEN, value, count=1)
    return value


def payload_view(path: Path, model: str) -> dict:
    """The real payload the way published_pair.json records it (read through mmap; the
    strings the release does not carry replaced by their digest, as in the reference)."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload["ema"]
    view = {"payload_keys": list(payload), "model_is_ema": payload.get("model") is state,
            "meta": {k: payload[k] for k in payload
                     if k not in ("model", "ema", "config", "export")},
            "config": payload["config"], "export": payload["export"],
            "manifest": manifest(state)}
    del payload, state
    view = json.loads(json.dumps(_normalise(view)))
    for entry in PP["scrubbed_paths"][model]:
        section, _, rest = entry["path"].partition(".")
        if section not in ("config", "export"):
            section, rest = "meta", entry["path"]
        node, parts = view[section], rest.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = "sha256:" + hashlib.sha256(node[parts[-1]].encode()).hexdigest()
    return view


# ------------------------------------------------------------------ without the weights

def test_the_reference_is_pinned_by_expected_json() -> None:
    assert sha256_file(PUBLISHED_PAIR) == EXP["published_pair"]["sha256"]
    assert EXP["published_pair"]["path"] == "configs/published_pair.json"
    for model in MODELS:
        name = FILE_OF[model]
        assert PP["files"][model]["name"] == name
        assert PP["files"][model]["sha256"] == EXP["published_weights"][name]["sha256"]
        assert PP["files"][model]["bytes"] == EXP["published_weights"][name]["bytes"]
        assert EXP["needs"]["shipped_weights"]["files"][name]["bytes"] == PP["files"][model]["bytes"]


@pytest.mark.parametrize("model", MODELS)
def test_the_plan_each_half_bakes_is_the_recorded_plan(model: str) -> None:
    from octtta.infer import plan_from_config

    assert plan_from_config(cfg(model)).describe() == PP[model]["export"]["plan_after"]


def test_the_fusion_block_is_the_qc045_table_on_model_a_only() -> None:
    from octtta.fusion import fusion_spec

    table = json.loads(FUSION_TABLE.read_text())
    canonical = fusion_spec({"fusion": {**table, "enabled": True}}).as_block()
    assert fusion_spec(cfg("A")).as_block() == canonical == PP["A"]["export"]["fusion_block"]
    assert fusion_spec(cfg("B")) is None and PP["B"]["export"]["fusion_block"] is None


@pytest.mark.parametrize("model", MODELS)
def test_the_baked_model_config_builds_exactly_the_shipped_tensors(model: str) -> None:
    import torch

    from octtta.models import build_model

    with torch.device("meta"):
        net = build_model(cfg(model)["model"])
    got = manifest(net.state_dict())
    assert got == PP[model]["tensors"]["manifest"]
    assert sum(math.prod(s) for _, s, _ in got) == PP[model]["tensors"]["numel"]


# ------------------------------------------------------------------ with the weights

@needs("shipped_weights")
def test_the_files_are_the_published_pins() -> None:
    base = needs_root("shipped_weights")
    for model in MODELS:
        name = FILE_OF[model]
        assert sha256_file(base / name) == EXP["published_weights"][name]["sha256"]


@needs("shipped_weights")
@pytest.mark.parametrize("model", MODELS)
def test_published_pair_json_is_the_real_payload(model: str) -> None:
    view = payload_view(needs_root("shipped_weights") / FILE_OF[model], model)
    half = PP[model]
    assert view["payload_keys"] == half["payload_keys"]
    assert view["model_is_ema"] == half["model_is_ema"]
    assert view["manifest"] == half["tensors"]["manifest"]
    for section in ("meta", "config", "export"):
        assert view[section] == half[section], section


@needs("shipped_weights")
@pytest.mark.parametrize("model", MODELS)
def test_the_published_pair_loads_for_inference(model: str) -> None:
    import torch

    from octtta.engine import load_inference_model

    net, loaded = load_inference_model(needs_root("shipped_weights") / FILE_OF[model],
                                       device="cpu")
    assert loaded["_weights"] == "ema" and not net.training
    assert manifest(net.state_dict()) == PP[model]["tensors"]["manifest"]
    with torch.no_grad():
        out = net(torch.randn(1, 1, 64, 96, generator=torch.Generator().manual_seed(0)))
    assert tuple(out.shape) == (1, 10, 64, 96) and bool(torch.isfinite(out).all())
