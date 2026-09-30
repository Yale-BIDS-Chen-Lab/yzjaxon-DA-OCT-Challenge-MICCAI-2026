"""Read-only reproduction checks on small, synthetic metadata trees."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import torch
import yaml

from octtta.config import expand_env, load_config
from scripts import repro_checks as checks


def test_framework_check_validates_inputs_and_keeps_historical_pins_advisory(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "custom.yaml"
    cfg.write_text(f"experiment:\n  id: custom\ndata:\n  root: {data}\n"
                   f"runtime:\n  out_dir: {tmp_path / 'runs'}\n")
    assert checks.check_framework(cfg)["status"] == "PASS"
    monkeypatch.setattr(checks, "check_env", lambda: {"status": "FAIL"})
    monkeypatch.setattr(checks, "check_data", lambda *a: {"status": "BLOCKED"})
    monkeypatch.setattr(checks, "check_pools", lambda *a: {"status": "FAIL"})
    report = checks.check_framework(cfg, reproduce=True)
    assert report["status"] == "WARN"
    data.rmdir()
    assert checks.check_framework(cfg, reproduce=True)["status"] == "FAIL"


def test_check_all_keeps_specific_advice_and_zero_exit(monkeypatch, capsys):
    monkeypatch.setattr(checks, "check_env", lambda: {"status": "FAIL", "rows": [
        {"name": "package.torch", "status": "FAIL", "reason": "required version differs"}]})
    monkeypatch.setattr(checks, "check_data", lambda *a, **k: {"status": "BLOCKED", "rows": [
        {"name": "official.archive.starting_kit", "status": "BLOCKED",
         "reason": "input is missing", "source": "https://example.org/kit"}]}
                        if k.get("only") != "published" else {"status": "BLOCKED", "rows": [
        {"name": "published.model.pt", "status": "BLOCKED", "reason": "input is missing"}]})
    monkeypatch.setattr(checks, "check_pools", lambda *a: {"status": "FAIL"})
    report = checks.check_all()
    assert report["status"] == "WARN"
    assert all(row["status"] != "FAIL" for row in report["rows"])
    assert any("download official" in row.get("next", "") for row in report["rows"])
    assert any("published_weights" in row.get("next", "") for row in report["rows"])
    assert checks.main(["data"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "WARN"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_run(tmp_path: Path, monkeypatch, *, name: str = "s33_cnn") -> tuple[Path, dict]:
    monkeypatch.setattr(checks, "_run_source_overlap", lambda: {
        "name": "runs.source_overlap", "status": "PASS", "n_shared": 0})
    runs = tmp_path / "runs"
    run = runs / name
    (run / "checkpoints").mkdir(parents=True)
    pin = checks.expected()["runs"][name]
    env = {**os.environ, "OCTTTA_RUNS": str(runs),
           "OCTTTA_DATA": str(checks.paths.DATA_DIR),
           "OCTTTA_CKPT": str(checks.paths.CKPT_DIR),
           "OCTTTA_SCRATCH": str(checks.paths.SCRATCH)}
    config = expand_env(load_config(checks._run_recipe(name)), env=env)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    audit = {"loaded": {"n_train": pin["n_train"], "n_val": pin["n_val"],
                        "partial_pool": copy.deepcopy(pin["partial_pool"]),
                        "draws_per_epoch": pin["draws_per_epoch"],
                        "draws_per_epoch_effective": pin["draws_per_epoch_effective"],
                        "sampling": copy.deepcopy(pin["sampling"]),
                        "full_pass": pin["full_pass"]}}
    (run / "pools.audit.json").write_text(json.dumps(audit))
    (run / "run_env.json").write_text(json.dumps({"launches": [
        {"torch": "fake", "cuda": "fake", "gpu_names": ["fake"]}]}))
    payload = {"epoch": pin["last_epoch"], "global_step": pin["global_step"],
               "finetune_from": (str(runs / pin["finetune_from"])
                                 if pin["finetune_from"] else None),
               "model": {"w": torch.zeros(1)}, "config": {"experiment": {"id": name}}}
    torch.save(payload, run / "checkpoints" / "last.pt")
    return runs, payload


def test_check_runs_pass_and_wrong_epoch(tmp_path, monkeypatch):
    runs, payload = _fake_run(tmp_path, monkeypatch)
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "PASS"
    payload["epoch"] += 1
    torch.save(payload, runs / "s33_cnn" / "checkpoints" / "last.pt")
    report = checks.check_runs(runs, only="s33_cnn")
    assert report["status"] == "FAIL"
    assert next(r for r in report["rows"] if r["name"].endswith(".last"))["status"] == "FAIL"


def test_check_runs_wrong_warm_start(tmp_path, monkeypatch):
    runs, payload = _fake_run(tmp_path, monkeypatch, name="s33_sam_phase2")
    assert checks.check_runs(runs, only="s33_sam_phase2")["status"] == "PASS"
    payload["finetune_from"] = str(runs / "s34_sam_phase1/checkpoints/last.pt")
    torch.save(payload, runs / "s33_sam_phase2" / "checkpoints" / "last.pt")
    assert checks.check_runs(runs, only="s33_sam_phase2")["status"] == "FAIL"


def test_check_runs_wrong_loaded_count_and_shared_source(tmp_path, monkeypatch):
    runs, _ = _fake_run(tmp_path, monkeypatch)
    audit_path = runs / "s33_cnn" / "pools.audit.json"
    audit = json.loads(audit_path.read_text())
    audit["loaded"]["n_train"] += 1
    audit_path.write_text(json.dumps(audit))
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"
    audit["loaded"]["n_train"] -= 1
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(checks, "_run_source_overlap", lambda: {
        "name": "runs.source_overlap", "status": "FAIL", "n_shared": 1})
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"


def test_check_runs_missing_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "_run_source_overlap", lambda: {
        "name": "runs.source_overlap", "status": "PASS", "n_shared": 0})
    result = checks.check_runs(tmp_path, only="s33_cnn")
    assert result["status"] == "BLOCKED"
    assert all(row["status"] == "BLOCKED" for row in result["rows"]
               if row["name"].startswith("run."))


def test_check_runs_rejects_cell_redistribution_and_sampler_drift(tmp_path, monkeypatch):
    runs, _ = _fake_run(tmp_path, monkeypatch)
    path = runs / "s33_cnn" / "pools.audit.json"
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    cells = changed["loaded"]["partial_pool"]["by_cell"]
    a, b = list(cells)[:2]
    cells[a] += 1
    cells[b] -= 1
    path.write_text(json.dumps(changed))
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"
    for field in ("draws_per_epoch", "draws_per_epoch_effective"):
        changed = copy.deepcopy(original)
        changed["loaded"][field] += 8
        path.write_text(json.dumps(changed))
        assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"
    changed = copy.deepcopy(original)
    changed["loaded"]["sampling"]["mode"] = "multinomial"
    path.write_text(json.dumps(changed))
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"
    path.write_text(json.dumps(original))
    doc = copy.deepcopy(checks.expected())
    doc["runs"]["s33_cnn"]["steps_per_epoch"] += 1
    monkeypatch.setattr(checks, "expected", lambda: doc)
    assert checks.check_runs(runs, only="s33_cnn")["status"] == "FAIL"


def test_unlabelled_check_requires_labelled_overlap_sources(tmp_path, monkeypatch):
    from scripts.data import aireadi_common
    monkeypatch.setattr(aireadi_common.Exclusions, "load", lambda root: object())
    monkeypatch.setattr(checks, "_unlabelled", lambda *args: ({}, {"unlabelled-fingerprint"}))
    with pytest.raises(ValueError, match="labelled pool root is missing"):
        checks.check_pools(tmp_path, only="unlabelled")
    monkeypatch.setattr(checks, "_labelled", lambda *args: ({}, {"labelled-fingerprint"}))
    assert checks.check_pools(tmp_path, only="unlabelled")["overlap"] == {
        "n_measured": 2, "n_hit": 0}
    monkeypatch.setattr(checks, "_labelled", lambda *args: ({}, {"unlabelled-fingerprint"}))
    with pytest.raises(ValueError, match="source associations failed"):
        checks.check_pools(tmp_path, only="unlabelled")


def test_report_directory_is_not_a_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "PUBLIC", ())
    monkeypatch.setattr(checks, "AIREADI", ())
    root = tmp_path / "pool"
    reports = root / "build_report"
    reports.mkdir(parents=True)
    (reports / "summary.json").write_text("{}")
    assert checks._labelled(tmp_path, {"root": "pool"}, ()) == ({}, set())
    (reports / "index.json").write_text("{}")
    with pytest.raises(ValueError, match="non-report inputs"):
        checks._labelled(tmp_path, {"root": "pool"}, ())
    (reports / "index.json").unlink()
    (root / "unexpected_pool").mkdir()
    with pytest.raises(ValueError, match="unexpected directory"):
        checks._labelled(tmp_path, {"root": "pool"}, ())


def test_isfahan_checks_selected_files_not_distribution_size(tmp_path):
    root = tmp_path / "distribution"
    root.mkdir()
    (root / "9001.tif").write_bytes(b"image")
    (root / "unrelated.tif").write_bytes(b"another image")
    selection = tmp_path / "paths.csv"
    with selection.open("w", newline="") as fh:
        csv.writer(fh).writerow(["image/9001.png", "./9001.tif"])
    pin = {"extract_root": "distribution", "selection_csv": "paths.csv",
           "n_files": 1, "inventory_sha256": checks._lines_digest(["9001.tif"])}
    assert checks._isfahan_selection(tmp_path, pin)["status"] == "PASS"
    (root / "9001.tif").unlink()
    assert checks._isfahan_selection(tmp_path, pin)["status"] == "FAIL"


def test_check_data_archive_pin_and_missing(tmp_path, monkeypatch):
    archive = tmp_path / "public" / "_archives" / "fake.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("part", b"one")
    extracted = tmp_path / "public" / "extracted" / "fake"
    extracted.mkdir(parents=True)
    (extracted / "part").write_bytes(b"one")
    source = tmp_path / "selected"
    source.mkdir()
    (source / "9001.tif").write_bytes(b"fake")
    selection = tmp_path / "paths.csv"
    selection.write_text("image/9001.png,9001.tif\n")
    doc = {"inputs": {"public_archives": {"fake.zip": {
        "bytes": archive.stat().st_size, "sha256": _sha(archive),
        "extract_root": "fake", "url": "https://example.org/fake.zip"}},
        "isfahan": {"extract_root": "selected", "selection_csv": "paths.csv",
                    "n_files": 1, "inventory_sha256": checks._lines_digest(["9001.tif"])}}}
    monkeypatch.setattr(checks, "expected", lambda: doc)
    assert checks.check_data(tmp_path, only="public")["status"] == "PASS"
    cache = extracted / "__pycache__"
    cache.mkdir()
    (cache / "scorer.cpython-311.pyc").write_bytes(b"generated bytecode")
    assert checks.check_data(tmp_path, only="public")["status"] == "PASS"
    from scripts.data.download import extract
    assert extract(archive, extracted, files=1) == 1
    (extracted / "unexpected.py").write_bytes(b"unexpected source")
    assert checks.check_data(tmp_path, only="public")["status"] == "FAIL"
    with pytest.raises(ValueError, match="inventory differs"):
        extract(archive, extracted, files=1)
    (extracted / "unexpected.py").unlink()
    archive.write_bytes(b"changed archive")
    assert checks.check_data(tmp_path, only="public")["status"] == "FAIL"
    archive.unlink()
    assert checks.check_data(tmp_path, only="public")["status"] == "BLOCKED"


def test_exclusions_static_and_sums():
    doc = checks.expected()
    assert checks._exclusions_static(doc)["never_train"] == 5868
    assert checks._static_sums(doc)["full_pass_steps"] == 47205


def test_ai_metadata_absent_preserves_static_checks(tmp_path):
    report = checks.check_data(tmp_path, only="aireadi")
    assert report["status"] == "BLOCKED"
    assert report["rows"][0]["status"] == "PASS"
    assert report["rows"][1]["status"] == "BLOCKED"
    assert "public-only build" in report["possible_without_aireadi"]
