"""Coarse checks for the licensed-data boundary using synthetic 9xxx identities."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.data import aireadi_common as common
from scripts import scan_ids


def _tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _fake_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "licensed"
    _tsv(root / "participants.tsv", ["person_id", "recommended_split", "study_group"], [
        {"person_id": "9001", "recommended_split": "train", "study_group": "healthy"},
        {"person_id": "9002", "recommended_split": "test", "study_group": "diabetes"},
        {"person_id": "9003", "recommended_split": "val", "study_group": "healthy"},
    ])
    rows = []
    for person, model in (("9001", "Triton"), ("9002", "Cirrus"), ("9003", "Triton")):
        rows.append({"person_id": person, "imaging": "OCT",
                     "filepath": f"/dataset/retinal_oct/structural/{'topcon_triton' if model == 'Triton' else 'zeiss_cirrus'}/{person}/scan.dcm",
                     "manufacturer": "Topcon" if model == "Triton" else "Zeiss",
                     "manufacturers_model_name": model, "number_of_frames": "18",
                     "anatomic_region": "Macula, 12 x 12", "laterality": "L",
                     "height": "32", "width": "24"})
    _tsv(root / "retinal_oct/manifest.tsv", list(rows[0]), rows)
    _tsv(root / ".transfer/probe/octa_manifest.tsv", ["person_id"],
         [{"person_id": "9001"}])
    key = common.fingerprint_key(root / "retinal_oct/manifest.tsv")
    records = {"never_train": [{"fp": common.fingerprint(rows[2]["filepath"], key),
                                  "vendor": "Triton"}],
               "labelled_quality": [{"fp": common.fingerprint(rows[0]["filepath"], key),
                                      "vendor": "Topcon_Triton"}],
               "unlabelled_duplicates": []}
    config = tmp_path / "exclusions.json"
    config.write_text(json.dumps({"schema": 1, "fingerprint": "hmac-sha256-16-v1",
                                  "manifest_rows": len(rows),
                                  "manifest_sha256": hashlib.sha256(
                                      (root / "retinal_oct/manifest.tsv").read_bytes()).hexdigest(),
                                  "lists": records}))
    return root, config


def test_keyed_exclusion_and_octa_probe_fallback(tmp_path: Path) -> None:
    root, config = _fake_root(tmp_path)
    ex = common.Exclusions.load(root, config)
    with (root / "retinal_oct/manifest.tsv").open(newline="") as fh:
        paths = sorted(common._rel(row["filepath"]) for row in csv.DictReader(fh, delimiter="\t"))
    manual_key = hashlib.sha256(common.FP_KEY_DOMAIN +
                                b"\n".join(path.encode() for path in paths) + b"\n").digest()
    assert ex.key == manual_key
    assert ex.fingerprint(paths[0]) == hmac.new(
        manual_key, paths[0].encode(), hashlib.sha256).hexdigest()[:16]
    assert common.octa_manifest_path(root).parent.name == "probe"
    assert common.test_split_persons(root) == {"9002"}
    assert ex.is_never_train("9002", "/dataset/retinal_oct/structural/zeiss_cirrus/9002/scan.dcm")
    assert ex.is_never_train("9003", "/dataset/retinal_oct/structural/topcon_triton/9003/scan.dcm")
    assert not ex.is_never_train("9001", "/dataset/retinal_oct/structural/topcon_triton/9001/scan.dcm")
    assert ex.excludes_labelled_quality("/dataset/retinal_oct/structural/topcon_triton/9001/scan.dcm",
                                        "Topcon_Triton")
    assert len(ex.fingerprint("/dataset/retinal_oct/structural/topcon_triton/9001/scan.dcm")) == 16
    assert len(common.aireadi_pretrain_volumes(root, exclusions=ex)) == 1
    assert ex.config_sha256 == hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unknown participant"):
        ex.is_never_train("9004", "/dataset/retinal_oct/structural/topcon_triton/9004/scan.dcm")


@pytest.mark.parametrize("change", ["missing", "blank_id", "duplicate", "blank_split", "invalid_split"])
def test_participant_metadata_fails_closed(tmp_path: Path, change: str) -> None:
    root, config = _fake_root(tmp_path)
    table = root / "participants.tsv"
    with table.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if change == "missing":
        rows.pop()
    elif change == "blank_id":
        rows[0]["person_id"] = ""
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "blank_split":
        rows[0]["recommended_split"] = ""
    else:
        rows[0]["recommended_split"] = "holdout"
    _tsv(table, list(rows[0]), rows)
    for read in (lambda: common.load_participants(root),
                 lambda: common.test_split_persons(root),
                 lambda: common.Exclusions.load(root, config)):
        with pytest.raises(ValueError) as exc:
            read()
        assert "900" not in str(exc.value)


def test_manifests_reject_unlisted_person_even_with_valid_config(tmp_path: Path) -> None:
    root, config = _fake_root(tmp_path)
    manifest = root / "retinal_oct/manifest.tsv"
    with manifest.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    rows[0]["person_id"] = "9004"
    _tsv(manifest, list(rows[0]), rows)
    blob = json.loads(config.read_text())
    blob["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="unknown participant"):
        common.Exclusions.load(root, config)
    _tsv(root / ".transfer/probe/octa_manifest.tsv", ["person_id"],
         [{"person_id": "9004"}])
    with pytest.raises(ValueError, match="unknown participant"):
        common.index_octa(root)


def test_exclusion_rejects_manifest_or_fingerprint_drift(tmp_path: Path) -> None:
    root, config = _fake_root(tmp_path)
    blob = json.loads(config.read_text())
    blob["lists"]["never_train"][0]["fp"] = "0" * 16
    config.write_text(json.dumps(blob))
    try:
        common.Exclusions.load(root, config)
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("unknown fingerprint was accepted")
    root, config = _fake_root(tmp_path)
    with (root / "retinal_oct/manifest.tsv").open("a") as fh:
        fh.write("\n")
    try:
        common.Exclusions.load(root, config)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("manifest drift was accepted")


def test_default_exclusions_require_expected_pin_and_local_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = _fake_root(tmp_path)
    release = tmp_path / "release"
    (release / "scripts/data").mkdir(parents=True)
    (release / "configs").mkdir()
    installed_config = release / "configs/aireadi_exclusions.json"
    installed_config.write_bytes(config.read_bytes())
    expected_path = release / "configs/expected.json"
    digest = hashlib.sha256(installed_config.read_bytes()).hexdigest()
    expected_path.write_text(json.dumps({"aireadi": {"exclusions": {"file_sha256": digest}}}))
    monkeypatch.setattr(common, "__file__", str(release / "scripts/data/aireadi_common.py"))
    assert common.Exclusions.load(root).config_sha256 == digest
    expected_path.write_text(json.dumps({"aireadi": {"exclusions": {"file_sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="hash differs"):
        common.Exclusions.load(root)
    expected_path.write_text(json.dumps({"aireadi": {"exclusions": {"file_sha256": digest}}}))
    (root / "retinal_oct/manifest.tsv").unlink()
    with pytest.raises(FileNotFoundError):
        common.Exclusions.load(root)


def test_published_exclusion_shape() -> None:
    blob = json.loads((Path(__file__).resolve().parents[1] /
                       "configs/aireadi_exclusions.json").read_text())
    assert blob["schema"] == 1 and blob["fingerprint"] == "hmac-sha256-16-v1"
    assert blob["manifest_rows"] == 56477
    assert {name: len(records) for name, records in blob["lists"].items()} == {
        "never_train": 5868, "labelled_quality": 65, "unlabelled_duplicates": 15}
    for records in blob["lists"].values():
        assert records == sorted(records, key=lambda r: (r["fp"], r["vendor"]))
        assert all(set(r) == {"fp", "vendor"} and re.fullmatch(r"[0-9a-f]{16}", r["fp"])
                   for r in records)


def test_scanner_error_review_ack_and_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    participants = tmp_path / "participants.tsv"
    _tsv(participants, ["person_id"], [{"person_id": "9001"}])
    (repo / "README.md").write_text("A plain reference to 9001.\n")
    (repo / "fixtures").mkdir(parents=True)
    (repo / "fixtures/sample.csv").write_text("participant,9001\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=repo, check=True)
    first = scan_ids.scan(repo, "HEAD", participants)
    assert first["errors"] == 1 and first["reviews"] == 1
    ack = tmp_path / "ack.tsv"
    ack.write_text("README.md\t" + scan_ids._line_hash("A plain reference to 9001.")
                   + "\tsynthetic reference reviewed\n")
    second = scan_ids.scan(repo, "HEAD", participants, ack)
    assert second["errors"] == 1 and second["reviews"] == 0
    (repo / "README.md").write_text("participant 9001\n")
    (repo / "fixtures/new.csv").write_text("participant,9001\n")
    worktree = scan_ids.scan_worktree(repo, participants)
    assert worktree["errors"] == 3 and worktree["reviews"] == 0
    assert scan_ids.scan(repo, "HEAD", participants)["reviews"] == 1
    from PIL import Image, PngImagePlugin
    import io
    image = Image.new("L", (1, 1), color=0)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("comment", "participant 9001")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=meta)
    counts = {"files": 0, "matched_lines": 0, "errors": 0, "reviews": 0,
              "acknowledged": 0, "unknown_binary": 0, "weights": 0}
    scan_ids.scan_blob("fixtures/metadata.png", buffer.getvalue(), {"9001"}, set(), counts)
    assert counts["errors"] == 1


def test_scanner_reads_gzip_and_weight_metadata(tmp_path: Path) -> None:
    import gzip
    import torch

    counts = {"files": 0, "matched_lines": 0, "errors": 0, "reviews": 0,
              "acknowledged": 0, "unknown_binary": 0, "weights": 0}
    scan_ids.scan_blob("sample.csv.gz", gzip.compress(b"person_id\n9001\n"),
                       {"9001"}, set(), counts)
    assert counts["errors"] == 1
    weights = tmp_path / "model.pt"
    torch.save({"model": {"layer": torch.zeros(1)}, "metadata": {"participant": "9001"}},
               weights)
    text = scan_ids._weight_text(weights)
    scan_ids._classify_text("model.pt", text, {"9001"}, set(), counts)
    assert counts["errors"] == 2
    scan_ids.scan_blob("unknown.bin", b"\x00\x01", {"9001"}, set(), counts)
    assert counts["unknown_binary"] == 1 and counts["errors"] == 3


def test_structured_numeric_metadata_keeps_identity_fields_visible() -> None:
    counts = {"files": 0, "matched_lines": 0, "errors": 0, "reviews": 0,
              "acknowledged": 0, "unknown_binary": 0, "weights": 0}
    doc = {"config": {"train": {"samples_per_epoch": 9001},
                      "data": {"person_id": 9001, "filepath": "scan/9001.dcm"}},
           "run": "synthetic", "source_run": "synthetic"}
    scan_ids.scan_blob("fixtures/config.json", json.dumps(doc).encode(), {"9001"}, set(), counts)
    assert counts["errors"] == 2 and counts["reviews"] == 0
    assert counts["matched_lines"] == 2
    scan_ids.scan_blob("fixtures/unknown.json", b'{"members": [9001]}',
                       {"9001"}, set(), counts)
    assert counts["reviews"] == 1
    import yaml
    scan_ids.scan_blob("fixtures/config.yaml", yaml.safe_dump(doc).encode(),
                       {"9001"}, set(), counts)
    assert counts["errors"] == 4 and counts["reviews"] == 1
