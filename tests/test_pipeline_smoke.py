"""Short CPU training through the model and engine, followed by export and inference."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from octtta.engine import build_optimizer, build_scheduler, load_inference_model
from octtta.models import build_model
from octtta.models.ema import ModelEMA
from octtta.train import Trainer
from scripts.export import export_pair

ROOT = Path(__file__).resolve().parents[1]
PAIR = json.loads((ROOT / "configs" / "published_pair.json").read_text())


def small_config() -> dict:
    cfg = copy.deepcopy(PAIR["B"]["config"])
    cfg["model"]["params"]["features_per_stage"] = [4, 8, 8, 8, 8, 8, 8]
    cfg["inference"] = {"mode": "sliding_window", "patch_size": [128, 128],
                        "overlap": 0.25, "max_height": 160}
    cfg["postproc"] = {"enabled": False}
    cfg.pop("fusion", None)
    cfg["train"]["optimizer"] = {"name": "adamw", "lr": 0.001}
    cfg["train"]["scheduler"] = {"name": "poly", "warmup_epochs": 0}
    return cfg


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_short_train_export_infer_single_model(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = small_config()
    model = build_model(cfg["model"], pretrained=False).train()
    before = next(model.parameters()).detach().clone()
    optimizer = build_optimizer(cfg["train"]["optimizer"], model)
    scheduler = build_scheduler(cfg["train"]["scheduler"], optimizer,
                                total_steps=2, steps_per_epoch=2)
    image = torch.rand(1, 1, 128, 128)
    target = torch.randint(0, 10, (1, 128, 128))
    class Criterion:
        def __call__(self, logits, labels, **_kwargs):
            loss = F.cross_entropy(logits[0] if isinstance(logits, list) else logits, labels)
            return loss, {"total": float(loss.detach())}

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.model_b = None
    trainer.ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    trainer.sampler = SimpleNamespace(set_epoch=lambda _: None)
    trainer.train_ds = SimpleNamespace()
    trainer.train_loader = [{"image": image, "mask": target}] * 2
    trainer.device = torch.device("cpu")
    trainer.epoch = 0
    trainer.global_step = 0
    trainer.steps_per_epoch = 2
    trainer.max_steps = 2
    trainer.scheduler_b = None
    trainer.consistency = None
    trainer.interval_share = None
    trainer.coteach = None
    trainer.criterion = Criterion()
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.grad_clip = 0.0
    trainer.log_every = 10
    trainer.amp = False
    trainer.amp_dtype = torch.bfloat16
    trainer._logged_start_lr = False
    trainer.configured_lrs = [group["lr"] for group in optimizer.param_groups]
    trainer.run_kind = "train"
    stats = trainer.train_one_epoch()
    assert stats["steps"] == 2 and np.isfinite(stats["total"])
    assert not torch.equal(before, next(model.parameters()))

    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = {k: v.detach().clone() for k, v in trainer.ema.module.state_dict().items()}
    src = tmp_path / "last.pt"
    torch.save({"config": cfg, "model": raw, "ema": ema,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, src)
    out = tmp_path / "export"
    assert export_pair.main(["--a", str(src), "--out", str(out)]) == 0
    exported = load(out / "model.pt")
    assert exported["config"] == cfg
    assert exported["export"]["selected_weights"] == "ema"
    assert not {"optimizer", "scheduler"} & set(exported)
    assert all(torch.equal(exported["model"][k], ema[k]) for k in ema)
    restored, restored_cfg = load_inference_model(out / "model.pt", device="cpu")
    assert restored_cfg["inference"] == cfg["inference"]
    with torch.no_grad():
        expected = model.eval()(image)
        got = restored(image)
    expected = expected[0] if isinstance(expected, list) else expected
    got = got[0] if isinstance(got, list) else got
    assert got.shape == expected.shape == (1, 10, 128, 128)
    assert not torch.equal(got, expected), "the inference loader should use the selected EMA"

    inp, masks = tmp_path / "images", tmp_path / "masks"
    inp.mkdir()
    assert cv2.imwrite(str(inp / "frame-image.png"), (image[0, 0].numpy() * 255).astype("uint8"))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "octtta.infer", "--input", str(inp),
                             "--output", str(masks), "--checkpoint", str(out / "model.pt"),
                             "--device", "cpu", "--no-degrade"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    mask = cv2.imread(str(masks / "frame-mask.png"), cv2.IMREAD_UNCHANGED)
    assert mask is not None and mask.shape == (128, 128)
    assert mask.dtype == np.uint8 and int(mask.max()) < 10


@pytest.mark.parametrize("style", ["separate", "combined"])
def test_second_model_selects_its_own_ema(tmp_path: Path, style: str) -> None:
    cfg = small_config()
    src = tmp_path / "last_b.pt"
    state = {"w": torch.tensor([1.0])}
    ema = {"w": torch.tensor([2.0])}
    payload = ({"config": cfg, "model": state, "ema": ema} if style == "separate" else
               {"config": {"model": {"name": "vit_fpn"}}, "config_b": cfg,
                "model": {"w": torch.tensor([9.0])}, "ema": {"w": torch.tensor([8.0])},
                "model_b": state, "ema_b": ema})
    torch.save(payload, src)
    dst = tmp_path / "model_b.pt"
    rec = export_pair.export_half(src, dst, "B")
    assert rec["selected_weights"] == ("ema" if style == "separate" else "ema_b")
    assert load(dst)["model"]["w"].item() == 2.0
    assert load(dst)["config"] == cfg


def test_final_recipe_is_explicit_and_inference_only(tmp_path: Path) -> None:
    cfg = small_config()
    src = tmp_path / "last.pt"
    torch.save({"config": cfg, "model": {"w": torch.ones(1)}}, src)
    dst = tmp_path / "model.pt"
    export_pair.export_half(src, dst, "A", final_recipe=True)
    got = load(dst)["config"]
    assert got["inference"]["max_height"] is None
    assert got["inference"]["blank_columns"]["enabled"] is True
    assert got["postproc"]["enabled"] is True
    assert got["fusion"]["enabled"] is True
    assert got["train"] == cfg["train"]
    assert cfg["inference"]["max_height"] == 160


def test_export_keeps_safety_and_lineage_fields(tmp_path: Path) -> None:
    src = tmp_path / "last.pt"
    source = {"config": small_config(), "model": {"w": torch.ones(1)},
              "quarantine": True, "quarantine_reasons": ["synthetic audit flag"],
              "run_kind": "train", "epoch": 3, "epoch_done": True,
              "config_fingerprint": "training-fingerprint", "monitor": "challenge_score",
              "best_score": 0.25, "global_step": 99, "optimizer": {}}
    torch.save(source, src)
    dst = tmp_path / "model.pt"
    export_pair.export_half(src, dst, "A")
    got = load(dst)
    for key in ("quarantine", "quarantine_reasons", "run_kind", "epoch", "epoch_done",
                "config_fingerprint", "monitor", "best_score"):
        assert got[key] == source[key]
    assert "optimizer" not in got and "global_step" not in got


def test_config_train_export_infer_cli(tmp_path: Path) -> None:
    """Run the public CLI through full trainer initialization on generated release-layout data."""
    data = tmp_path / "release_dataset" / "Topcon_Maestro2" / "Healthy"
    data.mkdir(parents=True)
    rows = np.arange(128)[:, None]
    for index in range(2):
        labels = np.broadcast_to(np.minimum((rows + index * 2) // 13, 9), (128, 128))
        image = np.broadcast_to((rows * 2 + index * 7) % 256, (128, 128)).astype(np.uint8)
        assert cv2.imwrite(str(data / f"frame{index}-image.png"), image)
        assert cv2.imwrite(str(data / f"frame{index}-mask.png"), labels.astype(np.uint8))

    cfg = small_config()
    cfg["experiment"] = {"id": "cli_smoke"}
    cfg["data"] = {
        "root": str(data.parents[1]), "num_classes": 10,
        "train_size": [128, 128], "crop": {"foreground_prob": 0},
        "partial_pool": {"enabled": False, "sampling": "multinomial"},
        "pool_entries": ["challenge_release"],
        "split": {"scheme": "hash_by_stem", "val_fraction": 0.5, "seed": 5},
        "normalize": {"mode": "per_image_zscore"},
        "loader": {"batch_size": 1, "num_workers": 0},
        "official_true_boundaries_only": False,
    }
    cfg["runtime"] = {"device": "cpu", "amp": False, "out_dir": str(tmp_path / "runs"),
                      "seed": 7, "val_every_epochs": 1}
    cfg["train"] = {"epochs": 1, "max_steps": 2,
                    "optimizer": {"name": "adamw", "lr": 0.001},
                    "scheduler": {"name": "poly", "warmup_epochs": 0},
                    "ema": {"enabled": True, "decay": 0.9}}
    cfg["loss"] = {"terms": [{"name": "cross_entropy", "weight": 1.0}],
                   "deep_supervision": {"enabled": False}}
    cfg["augment"] = {}
    cfg["discaug"] = {"enabled": False}
    cfg["failaug"] = {"enabled": False}
    cfg["eval"] = {"tune_select_metric": "none", "tta": False}
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2"}

    def run(*args: str) -> None:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]

    run("-m", "octtta.train", "--config", str(config_path))
    run_dir = tmp_path / "runs" / "cli_smoke"
    ckpt = run_dir / "checkpoints" / "last.pt"
    assert ckpt.is_file() and (run_dir / "config.resolved.yaml").is_file()
    assert load(ckpt)["global_step"] > 0
    out = tmp_path / "cli_export"
    run(str(ROOT / "scripts" / "export" / "export_pair.py"),
        "--a", str(ckpt), "--out", str(out))
    assert load(out / "model.pt")["config"]["inference"] == cfg["inference"]
    image_dir = tmp_path / "input"
    image_dir.mkdir()
    (image_dir / "frame-image.png").write_bytes((data / "frame0-image.png").read_bytes())
    output = tmp_path / "output"
    run("-m", "octtta.infer", "--input", str(image_dir), "--output", str(output),
        "--checkpoint", str(out / "model.pt"), "--device", "cpu", "--no-degrade")
    mask = cv2.imread(str(output / "frame-mask.png"), cv2.IMREAD_UNCHANGED)
    assert mask is not None and mask.shape == (128, 128) and int(mask.max()) < 10
