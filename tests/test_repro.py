"""B6 wizard commands and measured pool checks on synthetic metadata."""
from __future__ import annotations

import hashlib
import json
import csv
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from octtta.data.partial_labels import SURFACE_TO_BOUNDARY, offsets_to_json
from scripts import repro, repro_checks as checks, repro_score
from scripts.data import aireadi_common as common


def test_wizard_published_weights_and_check_use_same_overrides(tmp_path, monkeypatch):
    local = {"published_weights": str(tmp_path / "local"),
             "aireadi_participants": str(tmp_path / "synthetic-participants.tsv")}
    monkeypatch.setattr(repro, "_local_settings", lambda: local)
    monkeypatch.setattr(checks.paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(repro.paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.delenv("OCTTTA_PUBLISHED_WEIGHTS", raising=False)
    assert repro._weights("published")[0] == tmp_path / "local/model.pt"
    monkeypatch.setenv("OCTTTA_PUBLISHED_WEIGHTS", str(tmp_path / "environment"))
    assert repro._weights("published")[0] == tmp_path / "environment/model.pt"
    monkeypatch.delenv("OCTTTA_PUBLISHED_WEIGHTS")
    calls = []
    monkeypatch.setattr(checks, "main", lambda argv, *, local=None:
                        calls.append((argv, local)) or 0)
    assert repro.main(["check", "data", "--only", "published,aireadi"]) == 0
    monkeypatch.setattr(repro, "checked_commit", lambda **kwargs: "a" * 40)
    monkeypatch.setattr(repro.download, "main", lambda argv: 0)
    assert repro.main(["download", "official"]) == 0
    assert [c[1] for c in calls] == [local, local]


def test_build_graph_and_sbatch_command(monkeypatch):
    jobs = repro.build_jobs()
    assert [j.name for j in jobs] == [
        "public", "labelled_aireadi", "manifest", "extract", "unlabelled"]
    assert jobs[3].array == "0-31"
    assert jobs[4].depends_on == ("extract",)
    assert [j.name for j in repro.build_jobs(public_only=True)] == ["public"]
    argv = repro.sbatch_argv(jobs[3], "a" * 40, ("42",))
    assert "--array=0-31" in argv
    assert "--dependency=afterok:42" in argv
    assert "OCTTTA_EXPECT_COMMIT" not in argv[-1]
    assert "scripts/data/extract_aireadi_frames.py" in argv[-1]
    monkeypatch.setattr(repro, "_git", lambda *args: "a" * 40)
    assert len(repro.build(dry=True, public_only=True)) == 1


def test_local_build_allows_source_change_between_stages(tmp_path, monkeypatch):
    jobs = (repro.BuildJob("first", ("synthetic-first",), ("synthetic-check",)),
            repro.BuildJob("second", ("synthetic-second",), depends_on=("first",)))
    monkeypatch.setattr(repro, "build_jobs", lambda **kw: jobs)
    calls = []
    dirty = {"value": False}
    def fake_git(*args):
        return "a" * 40 if args == ("rev-parse", "HEAD") else (
            " M synthetic.py" if dirty["value"] else "")
    def fake_run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "synthetic-check":
            dirty["value"] = True
    monkeypatch.setattr(repro, "_git", fake_git)
    monkeypatch.setattr(repro.subprocess, "run", fake_run)
    repro.build(local=True, run_root=tmp_path)
    assert calls == ["synthetic-first", "synthetic-check", "synthetic-second"]


def test_stems_digest_is_sorted_newline_canonical():
    wanted = hashlib.sha256(b"a\nb\n").hexdigest()
    assert checks.stems_sha256(["b", "a"]) == wanted
    assert checks.stems_sha256(["a", "b"]) == wanted


def _fake_labelled(tmp_path: Path, monkeypatch):
    key = "aireadi::Topcon_Maestro2"
    monkeypatch.setattr(checks, "AIREADI", (key,))
    root = tmp_path / "derived" / "labelled" / key
    root.mkdir(parents=True)
    entry = {"stem": "synthetic-frame", "image": "synthetic-frame-image.png",
             "labels": {"1": "synthetic-frame-label.png"}, "volume_tag": "synthetic-tag",
             "person_id": "9001", "structural_path": "synthetic/structural",
             "seg_path": "synthetic/seg", "protocol": "Macula, 6 x 6",
             "device": "Topcon_Maestro2"}
    for f in (entry["image"], *entry["labels"].values()):
        (root / f).write_bytes(b"synthetic")
    mapping = SURFACE_TO_BOUNDARY[key]
    index = {"entries": [entry], "offsets_px": offsets_to_json(mapping.offsets_px),
             "boundaries": list(mapping.available),
             "stats": {"volumes": 1},
             "labelled_quality": {"excluded_here": 0},
             "selection": {"per_cell": 100000, "seed": 20260823,
                           "frames_per_volume": 16,
                           "max_per_person_per_cell": 100000,
                           "include_optic_disc": True}}
    (root / "index.json").write_text(json.dumps(index))
    pin = {"labelled": {"root": "derived/labelled", "dirs": {key: {
        "frames": 1, "volumes": 1, "persons": 1,
        "stems_sha256": checks.stems_sha256([entry["stem"]])}}}}
    expected = {"aireadi": {"exclusions": {"labelled_quality": {
        "volumes_by_vendor": {"Topcon_Maestro2": 0}}},
        "labelled_selection": {"Topcon_Maestro2": {"volumes": 1}}}}
    monkeypatch.setattr(checks, "expected", lambda: expected)
    monkeypatch.setattr(common, "index_octa", lambda *a, **k: [
        SimpleNamespace(structural_path="synthetic/structural", seg_path="synthetic/seg",
                        person_id="9001", vendor="Topcon_Maestro2", split="tune")])
    class Ex:
        def fingerprint(self, path):
            return "synthetic-fingerprint"
        def is_never_train(self, pid, path):
            return False
    monkeypatch.setattr(common.Exclusions, "load", lambda root: Ex())
    return root, pin, entry


def test_labelled_check_measures_donor_and_never_train(tmp_path, monkeypatch):
    root, pin, entry = _fake_labelled(tmp_path, monkeypatch)
    report = checks.check_pools(tmp_path, only="aireadi", pins=pin)
    assert report["labelled"]["donor_slice"] == {"n_measured": 1, "n_hit": 0}
    assert report["labelled"]["never_train"] == {"n_measured": 1, "n_hit": 0}
    index = json.loads((root / "index.json").read_text())
    index["entries"][0]["seg_path"] = "synthetic/wrong"
    (root / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="donor slice"):
        checks.check_pools(tmp_path, only="aireadi", pins=pin)


def test_identity_check_rejects_zero_measurement_and_hits():
    with pytest.raises(ValueError, match="no source associations"):
        checks._identity_check("synthetic", 0, 0)
    with pytest.raises(ValueError, match="failed"):
        checks._identity_check("synthetic", 2, 1)


def test_train_graph_pins_warm_starts_and_checks(monkeypatch):
    monkeypatch.setattr(repro, "_slurm", lambda: {"gpu": "h100"})
    jobs = repro.train_jobs(lineage="both")
    assert len(jobs) == 9
    by_name = {j.name: j for j in jobs}
    assert by_name["s33_sam_phase2"].depends_on == ("s33_sam_phase1",)
    assert by_name["s34_coteach"].depends_on == ("s34_sam_phase2", "s34_cnn")
    assert by_name["export"].depends_on == ("s33_coteach", "s34_coteach")
    assert by_name["s33_coteach"].command[-4::2] == (
        "--finetune-from", "--finetune-from-b")
    assert by_name["s33_coteach"].command[-3].endswith(
        "s33_sam_phase2/checkpoints/last.pt")
    assert by_name["s33_coteach"].command[-1].endswith(
        "s33_cnn/checkpoints/last.pt")
    assert by_name["export"].command[3].endswith("s33_coteach/checkpoints/last.pt")
    assert by_name["export"].command[5].endswith("s34_coteach/checkpoints/last_b.pt")
    assert all(not j.check for j in jobs[:-1])
    assert len(repro.train_jobs(lineage="s33")) == 4


def test_single_run_uses_edited_config_without_warm_start(tmp_path, monkeypatch):
    config = tmp_path / "custom.yaml"
    config.write_text("experiment:\n  id: my_run\ndata:\n  root: /tmp/data\n")
    monkeypatch.setattr(repro, "_slurm", lambda: {"gpu": "h100"})
    job = repro.run_job(config)
    assert job.name == "my_run"
    assert job.command == ("bash", "scripts/train/train.sbatch", str(config), "--gpus", "1")
    first = repro._config_digest(str(config))
    config.write_text("experiment:\n  id: my_run\ndata:\n  root: /tmp/changed\n")
    assert repro._config_digest(str(config)) != first


def test_run_preview_works_before_data_is_installed(tmp_path, monkeypatch):
    config = tmp_path / "custom.yaml"
    config.write_text(f"experiment:\n  id: preview\ndata:\n  root: {tmp_path / 'missing'}\n")
    monkeypatch.setattr(repro, "_run_jobs", lambda jobs, **kwargs: [jobs[0].name])
    assert repro.run(config, dry=True) == ["preview"]
    with pytest.raises(ValueError, match="training_data"):
        repro.run(config, local=True)


def test_reproduction_uses_resolved_run_ids_and_warm_starts(tmp_path, monkeypatch):
    def config(path):
        family = "alpha" if "model_b" not in str(path) else "beta"
        return {"experiment": {"id": f"{family}_{Path(path).stem}"},
                "runtime": {"out_dir": str(tmp_path)}}
    monkeypatch.setattr(repro, "get_config", config)
    jobs = repro.train_jobs()
    by_name = {j.name: j for j in jobs}
    assert by_name["alpha_sam_phase2"].depends_on == ("alpha_sam_phase1",)
    assert by_name["beta_coteach"].depends_on == ("beta_sam_phase2", "beta_cnn")
    assert by_name["beta_coteach"].command[-1] == str(tmp_path / "beta_cnn/checkpoints/last.pt")
    assert by_name["export"].depends_on == ("alpha_coteach", "beta_coteach")


def test_reproduce_checks_then_builds_before_training(monkeypatch, capsys):
    from scripts import repro_checks
    monkeypatch.setattr(repro_checks, "check_all", lambda local: {"rows": [
        {"status": "WARN", "name": "data", "reason": "missing", "next": "download"}]})
    monkeypatch.setattr(repro, "build_jobs", lambda: (repro.BuildJob("unlabelled", ("build",)),))
    monkeypatch.setattr(repro, "train_jobs", lambda **kwargs: (
        repro.BuildJob("cnn", ("train",), direct_script=True),))
    observed = []
    monkeypatch.setattr(repro, "_run_jobs", lambda jobs, **kwargs:
                        observed.extend(jobs) or [])
    repro.reproduce(dry=True)
    assert observed[1].depends_on == ("unlabelled",)
    assert "[check] data" in capsys.readouterr().err
    observed.clear()
    repro.reproduce(dry=True, skip_build=True)
    assert [job.name for job in observed] == ["cnn"]


def test_git_metadata_is_optional(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(repro.subprocess, "check_output", missing)
    assert repro.checked_commit(dry=False) == "unknown"


def test_export_cli_accepts_single_model_and_custom_output(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(repro, "_single", lambda name, command, **kw: seen.append(command))
    assert repro.main(["export", "--a", "custom.pt", "--out", str(tmp_path), "--dry"]) == 0
    assert seen[0][2:] == ("--a", "custom.pt", "--out", str(tmp_path))
    assert repro.main(["export", "--a", "a.pt", "--b", "b.pt", "--out", str(tmp_path),
                       "--final-recipe", "--dry"]) == 0
    assert seen[1][-3:] == ("--b", "b.pt", "--final-recipe")


def test_train_sbatch_flags_and_config_advice(monkeypatch):
    monkeypatch.setattr(repro, "_slurm", lambda: {
        "gpu": "h100", "account": "project", "partition": None,
        "qos": "normal", "wall_hours": None})
    monkeypatch.setattr(repro, "_activate", lambda: None)
    job = repro.train_jobs(lineage="s33")[0]
    argv = repro.sbatch_argv(job, "a" * 40)
    assert "--gpus=h100:1" in argv
    assert "--cpus-per-task=16" in argv
    assert "--mem=96G" in argv
    assert "--account=project" in argv and "--qos=normal" in argv
    assert not any(x.startswith("--partition") for x in argv)
    assert "--signal=B:USR1@60" in argv and "--requeue" in argv
    assert any(x.startswith("--output=") and "/slurm/" in x for x in argv)
    assert any(x.startswith("--error=") and "/slurm/" in x for x in argv)
    assert "--wrap" not in argv
    assert any(x.endswith("scripts/train/train.sbatch") for x in argv)
    exports = next(x for x in argv if x.startswith("--export="))
    assert "OCTTTA_EXPECT_COMMIT=" not in exports
    assert "OCTTTA_EXPECT_CONFIG_SHA256=" in exports
    launcher = (repro.REPO_ROOT / "scripts/train/train.sbatch").read_text()
    assert "config changed since submission" in launcher
    assert "OCTTTA_EXPECT_COMMIT" not in launcher


def test_cpu_export_dry_uses_cpu_site_and_uncommitted_source(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/local.yaml").write_text(
        "slurm:\n  account: lab\n  partition: gpu\n  qos: qos_bids\n"
        "  cpu:\n    partition: day\n    qos: normal\n")
    monkeypatch.setattr(repro, "REPO_ROOT", tmp_path)  # no .git: an unpacked source archive
    assert repro.main(["export", "--a", "model.pt", "--out", "export", "--dry"]) == 0
    job = json.loads(capsys.readouterr().out)[0]
    argv = job["sbatch"]
    assert "--partition=day" in argv and "--qos=normal" in argv
    assert "--account=lab" in argv
    assert "--qos=qos_bids" not in argv and "--partition=gpu" not in argv
    assert not any(arg.startswith("--gpus=") for arg in argv)
    assert "--wrap" in argv and f"OCTTTA_REPO={tmp_path}" in argv[-1]
    assert "OCTTTA_EXPECT_COMMIT" not in argv[-1]


def test_cpu_explicit_null_clears_inherited_site_flags(monkeypatch):
    monkeypatch.setattr(repro, "_slurm", lambda: {
        "account": "lab", "partition": "gpu", "qos": "qos_bids",
        "cpu": {"partition": None, "qos": None}})
    argv = repro.sbatch_argv(repro.BuildJob("export", ("true",)), "unknown")
    assert "--account=lab" in argv
    assert not any(arg.startswith("--partition=") or arg.startswith("--qos=")
                   for arg in argv)


def test_cpu_override_leaves_gpu_flags_unchanged(monkeypatch):
    monkeypatch.setattr(repro, "_slurm", lambda: {
        "account": "lab", "partition": "gpu", "qos": "qos_bids",
        "cpu": {"partition": "day", "qos": "normal"}})
    job = repro.BuildJob("train", ("true",), gpu="h100")
    argv = repro.sbatch_argv(job, "unknown")
    assert "--account=lab" in argv
    assert "--partition=gpu" in argv and "--qos=qos_bids" in argv
    assert "--gpus=h100:1" in argv
    assert "--partition=day" not in argv and "--qos=normal" not in argv


@pytest.mark.parametrize("cpu", [None, [], "day", {"qos": 7},
                                       {"partition": ""}, {"gpu": "h100"}])
def test_cpu_site_rejects_bad_mapping(tmp_path, monkeypatch, cpu):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/local.yaml").write_text(
        json.dumps({"slurm": {"cpu": cpu}}))
    monkeypatch.setattr(repro, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="slurm.cpu"):
        repro._local_settings()


def test_local_train_allows_mutation_before_next_job(tmp_path, monkeypatch):
    jobs = (repro.BuildJob("a", ("synthetic-a",), ("synthetic-check",)),
            repro.BuildJob("b", ("synthetic-b",), depends_on=("a",)))
    monkeypatch.setattr(repro, "train_jobs", lambda **kw: jobs)
    monkeypatch.setattr(repro, "_activate", lambda: None)
    dirty = {"value": False}
    monkeypatch.setattr(repro, "_git", lambda *args: (
        "a" * 40 if args == ("rev-parse", "HEAD") else
        " M changed.py" if dirty["value"] else ""))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "synthetic-check":
            dirty["value"] = True
    monkeypatch.setattr(repro.subprocess, "run", fake_run)
    repro.train(local=True, run_root=tmp_path)
    assert calls == ["synthetic-a", "synthetic-check", "synthetic-b"]


def test_infer_contract_checks_count_shape_dtype_and_range(tmp_path):
    import cv2
    import numpy as np
    images = tmp_path / "images"
    out = tmp_path / "out"
    images.mkdir()
    out.mkdir()
    cv2.imwrite(str(images / "synthetic-image.png"), np.zeros((3, 5), np.uint8))
    mask = out / "synthetic-mask.png"
    cv2.imwrite(str(mask), np.full((3, 5), 9, np.uint8))
    assert repro_score.verify_infer(images, out)["n_masks"] == 1
    cv2.imwrite(str(mask), np.full((3, 5), 10, np.uint8))
    with pytest.raises(ValueError, match="range"):
        repro_score.verify_infer(images, out)
    cv2.imwrite(str(mask), np.zeros((2, 5), np.uint8))
    with pytest.raises(ValueError, match="shape"):
        repro_score.verify_infer(images, out)


def test_synthetic_scorer_layout_preserves_frozen_lexicographic_order(tmp_path):
    source = tmp_path / "source"
    for cohort, prefix, n in (("Healthy", "h", 173), ("Diseased", "d", 57)):
        directory = source / cohort
        directory.mkdir(parents=True)
        for i in range(1, n + 1):
            (directory / f"{prefix}{i}-image.png").write_bytes(b"fake")
            (directory / f"{prefix}{i}-mask.png").write_bytes(b"fake")
    images, predicted, mapping = repro_score._layout(tmp_path / "work", source, 230)
    assert len(mapping) == 230
    assert (images / "syn_Topcon_Maestro2_Healthy_h1-image.png").is_symlink()
    assert predicted.is_dir()
    with (tmp_path / "work/input/ref/val/val_release.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["release_image_name"] for r in rows[:3]] == [
        "h1-image.png", "h10-image.png", "h100-image.png"]
    assert rows[0]["release_mask_name"] == "h1-mask.png"
    assert rows[0]["device"] == "Maestro2"
    assert rows[0]["group"] == "Macula, 6 x 6"
    assert rows[-1]["status"] == "diseased"
