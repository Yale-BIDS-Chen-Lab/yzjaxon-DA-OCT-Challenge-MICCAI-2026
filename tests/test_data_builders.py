"""Small, synthetic B6 builder and downloader contract tests."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import build_aireadi_pool as ai
from scripts.data import build_public_pool as public
from scripts.data import download, extract_aireadi_frames as frames
from scripts.data import stage_unlabeled_pool as stage


def test_vol_reader_checks_layout_and_pixels(tmp_path):
    header = bytearray(public.VOL_MAIN_HEADER_BYTES)
    header[:12] = b"HSF-OCT-100\0"
    struct.pack_into("<III", header, 12, 3, 1, 2)
    struct.pack_into("<ddd", header, 24, 1.0, 1.0, 1.0)
    struct.pack_into("<II", header, 48, 1, 1)
    struct.pack_into("<I", header, 100, 32)
    sub = bytearray(32)
    sub[:len(public.VOL_BSCAN_MAGIC)] = public.VOL_BSCAN_MAGIC
    pixels = np.array([[0.0, 1.0, 0.25], [np.nan, 1e31, 0.5]], dtype="<f4")
    vol = tmp_path / "synthetic.vol"
    vol.write_bytes(header + b"\0" + sub + pixels.tobytes())
    got = list(public.iter_vol_frames(vol))
    assert len(got) == 1 and got[0][0] == 0
    assert got[0][1].tolist() == [[0, 255, 180], [0, 0, 214]]
    vol.write_bytes(vol.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncated"):
        list(public.iter_vol_frames(vol))


def test_public_builder_dispatch_writes_stamped_index(tmp_path, monkeypatch):
    fake = {"dataset": "oct5k", "dir": str(tmp_path / "out" / "oct5k"),
            "entries": [{"stem": "synthetic", "image": "synthetic-image.png",
                         "labels": {"1": "synthetic-g1-label.png"}, "group": "synthetic",
                         "volume": "synthetic"}],
            "stats": {"images": 1, "labels": 1}}
    monkeypatch.setattr(public, "build_oct5k", lambda *args: fake)
    public.build_public_pool(tmp_path / "out", tmp_path,
                             datasets=("oct5k",), graders=(1,))
    import json
    index = json.loads((tmp_path / "out" / "oct5k" / "index.json").read_text())
    assert index["entries"] == fake["entries"]
    assert index["mapping_version"] == public.MAPPING_VERSION
    with pytest.raises(FileExistsError):
        public.build_public_pool(tmp_path / "out", tmp_path, datasets=("oct5k",))


def test_aireadi_selection_excludes_before_write(tmp_path, monkeypatch):
    def rec(pid, path):
        return SimpleNamespace(person_id=pid, structural_path=path,
                               vendor="Topcon_Maestro2", protocol="Macula, 6 x 6",
                               status_proxy="healthy", laterality="L", split="tune")
    donors = [rec("9001", "synthetic/a"), rec("9002", "synthetic/b"),
              rec("9003", "synthetic/c")]
    class Ex:
        config_sha256 = "0" * 64
        test_split = set()
        never_train = set()
        def is_never_train(self, pid, path):
            return pid == "9001"
        def excludes_labelled_quality(self, path, vendor):
            return path.endswith("b")
    monkeypatch.setattr(ai.aireadi.Exclusions, "load", lambda root: Ex())
    monkeypatch.setattr(ai.aireadi, "index_octa", lambda *a, **k: donors)
    monkeypatch.setattr(ai.aireadi, "balanced_sample", lambda rows, **k: list(rows))
    monkeypatch.setattr(ai.aireadi, "cohort_counts", lambda rows: {})
    called = []
    def fake_build(r, root, out, key, **kwargs):
        called.append(r.structural_path)
        return [{"person_id": r.person_id, "structural_path": r.structural_path,
                 "stem": ai.stem_for(r, 0), "protocol": r.protocol}]
    monkeypatch.setattr(ai, "_build_aireadi_volume", fake_build)
    reports = ai.build_aireadi(tmp_path, tmp_path, vendors=("Topcon_Maestro2",))
    assert called == ["synthetic/c"]
    assert reports[0]["index_extra"]["labelled_quality"]["excluded_here"] == 1


def test_manifest_bytes_and_extract_qc(tmp_path, monkeypatch):
    vol = SimpleNamespace(person_id="9001", vendor="Topcon", model="Maestro2",
                          anatomic_region="Macula", is_widefield=False,
                          laterality="L", height=3, width=4, n_frames=2,
                          take_frames=(0, 1), src_path="synthetic/volume")
    class Ex:
        def is_never_train(self, pid, path):
            return False
    monkeypatch.setattr(frames.common, "aireadi_pretrain_volumes",
                        lambda *a, **k: [vol])
    monkeypatch.setattr(frames.common, "npz_relpath", lambda v: "synthetic/volume.npz")
    rows = [frames.MANIFEST_COLUMNS,
            ("ai_readi", "aireadi-custom", "9001", "Topcon", "Maestro2", "Macula",
             0, "L", 3, 4, 2, "0,1", "synthetic/volume", "synthetic/volume.npz")]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerows(rows)
    pin = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    out = tmp_path / "manifest.tsv"
    stats = frames.write_manifest(tmp_path, out, exclusions=Ex(), expected=pin)
    assert stats == {"sha256": pin, "volumes": 1, "persons": 1, "planned_frames": 2}
    with pytest.raises(ValueError, match="frozen selection"):
        frames.write_manifest(tmp_path, out, exclusions=Ex(), expected="0" * 64)


def test_download_requires_pins_and_rejects_unsafe_zip(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="outside extraction root"):
        download.extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()
    with pytest.raises(ValueError, match="positive byte-count"):
        download.verify(archive, {"sha256": "0" * 64})


def test_public_download_uses_pinned_archive_roots(tmp_path):
    children = {"jhu_hcms.zip": "OCT_Manual_Delineations-2018_June_29",
                "duke_dme_2015.zip": "2015_BOE_Chiu",
                "oct5k_annotations.zip": "OCT5k"}
    pins = {}
    archives = tmp_path / "public" / "_archives"
    archives.mkdir(parents=True)
    for name, child in children.items():
        archive = archives / name
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(f"{child}/synthetic.txt", "synthetic")
        pins[name] = {"url": "https://example.test/synthetic.zip",
                      "bytes": archive.stat().st_size,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "extract_root": Path(name).stem}
    result = download.download_public(tmp_path, pins)
    assert result == {name: 1 for name in children}
    for name, child in children.items():
        assert (tmp_path / "public" / "extracted" / Path(name).stem /
                child / "synthetic.txt").is_file()
    damaged = tmp_path / "public" / "extracted" / "jhu_hcms" / children["jhu_hcms.zip"] / "synthetic.txt"
    damaged.unlink()
    with pytest.raises(ValueError, match="inventory"):
        download.download_public(tmp_path, pins)


def _qc_fixture(tmp_path, monkeypatch, *, all_rejected=False):
    import pydicom.pixels
    src = tmp_path / "manifest.tsv"
    row = dict(zip(frames.MANIFEST_COLUMNS, (
        "ai_readi", "aireadi-custom", "9001", "Topcon", "Maestro2", "Wide Field",
        "1", "L", "3", "4", "3", "0,1,2", "synthetic/volume",
        "synthetic/volume.npz")))
    with src.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=frames.MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    pin = frames.sha256_file(src)
    monkeypatch.setattr(pydicom.pixels, "pixel_array",
                        lambda path, index: np.full((3, 4), index + 1, np.uint8))
    monkeypatch.setattr(frames.common, "structural_frame_to_uint8", lambda img: img)
    def qc(img):
        marker = int(img[0, 0])
        if marker == 2 or (all_rejected and marker == 1):
            return "column_correlation"
        if marker == 3:
            return "tissue_band_snr"
        return None
    monkeypatch.setattr(frames.common, "frame_qc_reason", qc)
    class Ex:
        config_sha256 = "0" * 64
        test_split = set()
        never_train = set()
        def excludes_unlabelled_duplicate(self, path, vendor):
            return False
        def is_never_train(self, pid, path):
            return False
    return src, row, pin, Ex()


@pytest.mark.parametrize("all_rejected", [False, True])
def test_fresh_qc_sidecar_stage_matches_legacy_accounting(tmp_path, monkeypatch,
                                                           all_rejected):
    manifest, row, pin, ex = _qc_fixture(tmp_path, monkeypatch,
                                          all_rejected=all_rejected)
    fresh = tmp_path / "fresh"
    stats = frames.extract_shard(tmp_path, manifest, fresh, shard=0,
                                 num_shards=1, expected=pin)
    assert stats["all_qc_rejected"] == int(all_rejected)
    npz = fresh / row["npz_relpath"]
    sidecar = json.loads(frames.qc_sidecar_path(npz).read_text())
    assert sidecar["planned"] == 3
    assert sidecar["kept"] == (0 if all_rejected else 1)
    assert npz.is_file() != all_rejected
    fresh_report = stage.stage_unlabeled_pool(out=tmp_path / "staged_fresh",
        manifest=manifest, pool_root=fresh, exclusions=ex,
        expect_manifest_sha256=pin)
    legacy = tmp_path / "legacy" / row["npz_relpath"]
    legacy.parent.mkdir(parents=True)
    np.savez_compressed(legacy,
                        images=np.stack([np.full((3, 4), i + 1, np.uint8)
                                         for i in range(3)]),
                        frames=np.arange(3, dtype=np.int32),
                        qc_dropped=np.empty(0, dtype=np.int32))
    legacy_report = stage.stage_unlabeled_pool(out=tmp_path / "staged_legacy",
        manifest=manifest, pool_root=tmp_path / "legacy", exclusions=ex,
        expect_manifest_sha256=pin)
    key = "maestro2_widefield"
    assert fresh_report["groups"][key] == legacy_report["groups"][key]
    assert fresh_report["qc_rejected"] == legacy_report["qc_rejected"]
    assert fresh_report["totals"] == legacy_report["totals"]
    bad = {**sidecar, "manifest_sha256": "0" * 64}
    frames.qc_sidecar_path(npz).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest pin"):
        stage.stage_unlabeled_pool(out=tmp_path / "bad_pin", manifest=manifest,
            pool_root=fresh, exclusions=ex, expect_manifest_sha256=pin)
    frames.qc_sidecar_path(npz).write_text(json.dumps(sidecar))
    if not all_rejected:
        npz.write_bytes(npz.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="NPZ integrity"):
            stage.stage_unlabeled_pool(out=tmp_path / "bad_npz", manifest=manifest,
                pool_root=fresh, exclusions=ex, expect_manifest_sha256=pin)
        return
    if all_rejected:
        frames.qc_sidecar_path(npz).unlink()
        with pytest.raises(FileNotFoundError, match="source NPZ"):
            stage.stage_unlabeled_pool(out=tmp_path / "missing", manifest=manifest,
                pool_root=fresh, exclusions=ex, expect_manifest_sha256=pin)


def test_extractor_resume_rebuilds_incomplete_sidecar(tmp_path, monkeypatch):
    manifest, row, pin, _ = _qc_fixture(tmp_path, monkeypatch)
    pool = tmp_path / "pool"
    original = frames._write_json_atomic
    interrupted = {"done": False}
    def fail_complete(dest, payload):
        if payload["status"] == "complete" and not interrupted["done"]:
            interrupted["done"] = True
            raise OSError("synthetic interruption")
        return original(dest, payload)
    monkeypatch.setattr(frames, "_write_json_atomic", fail_complete)
    with pytest.raises(RuntimeError, match="failed extraction"):
        frames.extract_shard(tmp_path, manifest, pool, shard=0, num_shards=1,
                             expected=pin)
    npz = pool / row["npz_relpath"]
    assert npz.is_file()
    assert json.loads(frames.qc_sidecar_path(npz).read_text())["status"] == "intent"
    monkeypatch.setattr(frames, "_write_json_atomic", original)
    resumed = frames.extract_shard(tmp_path, manifest, pool, shard=0,
                                   num_shards=1, expected=pin)
    assert resumed["extracted"] == 1 and resumed["skipped_existing"] == 0
    assert json.loads(frames.qc_sidecar_path(npz).read_text())["status"] == "complete"


def test_stage_plan_selects_one_protocol_and_keeps_source_path():
    row = {"vendor": "Topcon", "model": "Maestro2", "anatomic_region": "Wide Field",
           "width": "885", "height": "512", "person_id": "9001", "laterality": "L",
           "npz_relpath": "synthetic/volume.npz", "src_path": "synthetic/volume",
           "take_frames": ",".join(map(str, range(16)))}
    selected = list(stage.plan([row], read_npz=False))
    assert len(selected) == 1
    assert selected[0].rule.key == "maestro2_widefield"
    assert selected[0].src_path == row["src_path"]
    assert len(selected[0].positions) == 16
