"""Coarse CPU checks for the two released model families and checkpoint rebuilds."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import sam_fixture  # noqa: F401 -- registers the tiny SAM encoder with timm
from octtta.models import MODEL_NAMES, build_model, third_party_requirements

ROOT = Path(__file__).resolve().parents[1]
PAIR = json.loads((ROOT / "configs" / "published_pair.json").read_text())


def _small_model(letter: str) -> dict:
    cfg = copy.deepcopy(PAIR[letter]["config"]["model"])
    if letter == "A":
        cfg["params"].update(backbone="samvit_tiny_fixture", indices=[0, 1, 2, 3],
                             dim=32, encoder_window_size=4,
                             encoder_weights="/synthetic/missing_encoder.pt")
    else:
        cfg["params"]["features_per_stage"] = [4, 8, 8, 8, 8, 8, 8]
    return cfg


def test_only_final_model_families_are_registered():
    assert set(MODEL_NAMES) == {"vit_fpn", "nnunet2d"}
    assert third_party_requirements("vit_fpn") == frozenset({"timm", "torch"})
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "unet_timm", "params": {}})


@pytest.mark.parametrize("letter", ["A", "B"])
def test_strict_checkpoint_rebuild_preserves_outputs(letter):
    cfg = _small_model(letter)
    torch.manual_seed(1982)
    model = build_model(cfg, pretrained=False).eval()
    if letter == "A":
        assert isinstance(model.encoder.neck, torch.nn.Identity)
        assert [b.window_size for b in model.encoder.blocks] == [4, 0, 4, 0]
    torch.manual_seed(9001)
    x = torch.randn(1, 1, 64, 96) if letter == "A" else torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        before = model(x)
    rebuilt = build_model(cfg, pretrained=False).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        after = rebuilt(x)
    heads_before = before if isinstance(before, list) else [before]
    heads_after = after if isinstance(after, list) else [after]
    assert len(heads_before) == len(heads_after) == 1
    assert all(torch.equal(a, b) for a, b in zip(heads_before, heads_after))
    assert heads_before[0].shape == (1, 10, *x.shape[-2:])


def test_training_requires_explicit_encoder_export():
    with pytest.raises(FileNotFoundError, match="encoder_weights"):
        build_model(_small_model("A"))
