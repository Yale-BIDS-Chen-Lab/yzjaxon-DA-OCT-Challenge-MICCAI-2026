"""Local release-data checks.  Licensed identities are read only in memory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from octtta import paths  # noqa: E402

_DATA_GROUPS = frozenset({"public", "official", "sam", "published", "aireadi"})
_POSSIBLE_WITHOUT_AIREADI = (
    "download", "public checks", "public-only build", "inference with published weights",
    "score with published weights",
)


def _row(name: str, status: str, *, reason: str = "", **counts) -> dict:
    """A report row with no local filename, source path, or participant identifier."""
    out = {"name": name, "status": status}
    if reason:
        out["reason"] = reason
    out.update(counts)
    return out


def _status(rows: list[dict]) -> str:
    states = {r["status"] for r in rows}
    return ("FAIL" if "FAIL" in states else "BLOCKED" if "BLOCKED" in states
            else "WARN" if "WARN" in states else "PASS")


def _safe_failure(name: str, exc: Exception) -> dict:
    return _row(name, "FAIL", reason=f"{type(exc).__name__}; inspect local metadata")


def check_framework(config: Path, *, reproduce: bool = False) -> dict:
    """Check usable inputs; historical consistency is advice, never a run gate."""
    from octtta.config import get_config
    from octtta.train_data import resolve_run_dir

    rows = []
    try:
        cfg = get_config(config)
        root = cfg.get("data", {}).get("root")
        if not isinstance(root, str) or not root.strip() or "$" in root:
            raise ValueError("data.root must resolve to a directory")
        run_dir = resolve_run_dir(cfg)
        if run_dir == Path(root):
            raise ValueError("training output must differ from the data root")
        rows.append(_row("config", "PASS", run_name=run_dir.name))
        required = {"training_data": Path(root)}
        partial = cfg.get("data", {}).get("partial_pool") or {}
        if partial.get("enabled"):
            if not isinstance(partial.get("root"), str) or not partial["root"].strip():
                raise ValueError("enabled partial pool needs data.partial_pool.root")
            required["partial_pool"] = Path(partial["root"])
        for name, path in required.items():
            rows.append(_row(name, "PASS" if path.is_dir() else "FAIL",
                             reason="required input directory is missing" if not path.is_dir() else ""))
    except Exception as exc:
        rows.append(_safe_failure("config", exc))
    if reproduce:
        # These pins describe one historical result, not the acceptance criteria
        # for an edited config or a new training run.
        for label, probe in (("environment", lambda: check_env()),
                             ("data_recipe", lambda: check_data(paths.DATA_DIR)),
                             ("pool_recipe", lambda: check_pools(paths.DATA_DIR))):
            try:
                result = probe()
                rows.append(_row(label, "PASS" if result["status"] == "PASS" else "WARN",
                                 reason="historical reproduction differs" if result["status"] != "PASS" else ""))
            except Exception as exc:
                rows.append(_row(label, "WARN", reason=f"{type(exc).__name__}; historical check unavailable"))
    return {"status": _status(rows), "rows": rows}


def check_all(local: dict | None = None) -> dict:
    """Show concrete readiness findings; mismatches with historical pins are advice."""
    groups = (
        ("environment", lambda: check_env(), "Install requirements.txt in your environment."),
        ("data", lambda: check_data(paths.DATA_DIR, only="public,official,sam,aireadi",
                                    local=local),
         "Download public, official and SAM inputs, then configure licensed data if used."),
        ("pools", lambda: check_pools(paths.DATA_DIR),
         "Run 'python scripts/repro.py build --local' or use 'reproduce --local' to build pools."),
        ("published_weights", lambda: check_data(paths.DATA_DIR, only="published", local=local),
         "Set published_weights in configs/local.yaml to the directory with model.pt and model_b.pt."),
    )
    rows = []
    for name, probe, action in groups:
        try:
            result = probe()
            findings = result.get("rows")
            if findings is None:
                state = result.get("status", "FAIL")
                rows.append(_row(name, "PASS" if state == "PASS" else "WARN",
                                 reason="historical pool check differs" if state != "PASS" else "",
                                 next=action if state != "PASS" else ""))
                continue
            for item in findings:
                state = item["status"]
                label = item["name"]
                next_step = action
                if label == "tool.archive":
                    next_step = "Install unrar or 7z to extract the public archives."
                if name == "data":
                    if label.startswith("public."):
                        next_step = "Run 'python scripts/repro.py download public'."
                    elif label.startswith("official."):
                        next_step = "Run 'python scripts/repro.py download official'."
                    elif label.startswith("sam."):
                        next_step = "Run 'python scripts/repro.py download sam'."
                    elif label.startswith("aireadi."):
                        next_step = "Configure licensed AI-READI inputs if using those pools."
                    elif label.startswith("isfahan."):
                        next_step = "Download public inputs and verify the Isfahan selection."
                source = item.get("source", "")
                if not (source.startswith("https://") or source.startswith("http://") or
                        source.startswith("FAIRhub")):
                    source = ""
                rows.append(_row(label, "PASS" if state == "PASS" else "WARN",
                                 reason=item.get("reason", "historical check differs") if state != "PASS" else "",
                                 next=next_step if state != "PASS" else "",
                                 source=source))
        except Exception as exc:
            rows.append(_row(name, "WARN", reason=f"{type(exc).__name__}; check unavailable",
                             next=action))
    return {"status": _status(rows), "rows": rows, "advisory": True,
            "next": "Fix WARN rows for closer reproduction. To use your own data and config, run 'python scripts/repro.py run --config YOUR.yaml --local'."}


def _file_pin(name: str, path: Path, pin: dict, *, source: str = "") -> dict:
    if not path.is_file():
        return _row(name, "BLOCKED", reason="input is missing", source=source)
    size = path.stat().st_size
    if size != pin.get("bytes"):
        return _row(name, "FAIL", reason="byte count differs", bytes=size,
                    expected_bytes=pin.get("bytes"))
    algorithm = "sha256" if "sha256" in pin else "md5" if "md5" in pin else None
    if algorithm is None:
        return _row(name, "FAIL", reason="expected.json has no digest pin")
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != pin[algorithm]:
        return _row(name, "FAIL", reason="content digest differs", bytes=size)
    return _row(name, "PASS", bytes=size, digest=digest)


def _directory_count(name: str, path: Path, n_files: int, *, source: str = "") -> dict:
    if not path.is_dir():
        return _row(name, "BLOCKED", reason="extracted input is missing", source=source)
    count = sum(p.is_file() for p in path.rglob("*"))
    if count != n_files:
        return _row(name, "FAIL", reason="file count differs", files=count,
                    expected_files=n_files)
    return _row(name, "PASS", files=count)


def _extracted_inventory(name: str, archive: Path, extracted: Path,
                         archive_row: dict, *, n_files: int | None = None,
                         source: str = "") -> dict:
    if not extracted.is_dir():
        return _row(name, "BLOCKED", reason="extracted input is missing", source=source)
    from scripts.data.download import _archive_inventory, _disk_inventory
    inventory = None
    if archive_row["status"] == "PASS":
        try:
            inventory = _archive_inventory(archive)
        except Exception as exc:
            return _safe_failure(name, exc)
    present = _disk_inventory(extracted, inventory)
    if not present or (n_files is not None and len(present) != n_files):
        return _row(name, "FAIL", reason="extraction file count differs", files=len(present))
    if archive_row["status"] == "PASS":
        try:
            if present != inventory:
                return _row(name, "FAIL", reason="extraction differs from pinned archive",
                            files=len(present))
        except Exception as exc:
            return _safe_failure(name, exc)
    return _row(name, "PASS", files=len(present),
                inventory_sha256=_lines_digest(present))


def _required_versions() -> dict[str, str]:
    out = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            package, version = line.split("==", 1)
            out[package] = version
    return out


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _gpu_probe() -> list[dict]:
    """Only --gpu constructs SAM-L; the default environment check never touches CUDA."""
    rows = []
    try:
        import torch
        if not torch.cuda.is_available():
            return [_row("gpu.cuda", "FAIL", reason="CUDA device is unavailable")]
        if torch.version.cuda != "12.8":
            rows.append(_row("gpu.cuda_version", "FAIL", reason="CUDA runtime must be 12.8"))
        else:
            rows.append(_row("gpu.cuda_version", "PASS", version="12.8"))
        driver = subprocess.run(("nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"), capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip().splitlines()[0]
        major = int(driver.split(".", 1)[0])
        rows.append(_row("gpu.driver", "PASS" if major >= 570 else "FAIL",
                         reason="driver must be at least 570" if major < 570 else "",
                         major=major))
        rows.append(_row("gpu.bf16", "PASS" if torch.cuda.is_bf16_supported() else "FAIL"))
        mem = torch.cuda.get_device_properties(0).total_memory
        rows.append(_row("gpu.memory", "PASS" if mem >= 80 * 10**9 else "WARN",
                         reason="80 GB recommended" if mem < 80 * 10**9 else "",
                         bytes=mem))
        cudnn = torch.backends.cudnn.version()
        rows.append(_row("gpu.cudnn", "PASS" if cudnn else "FAIL", version=cudnn))
        from octtta.models.vit_fpn import _adapt_encoder_in_place, build_timm_encoder
        encoder = build_timm_encoder("samvit_large_patch16.sa1b", in_chans=1,
                                     pretrained=False, cache_dir=None).cuda().train()
        _adapt_encoder_in_place(encoder, window_size=16)
        x = torch.randn(1, 1, 256, 256, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = encoder.forward_intermediates(
                x, indices=(5, 11, 17, 23), output_fmt="NCHW", intermediates_only=True)
            loss = features[-1].float().square().mean()
        loss.backward()
        rows.append(_row("gpu.sam_bf16_forward_backward", "PASS", tensors=len(features)))
        del encoder, x, features, loss
        torch.cuda.empty_cache()
    except Exception as exc:  # no exception text may disclose a local cache path
        rows.append(_safe_failure("gpu.sam_bf16_forward_backward", exc))
    return rows


def check_env(gpu: bool = False) -> dict:
    """Read-only checks of hard environment pins and available execution resources."""
    rows: list[dict] = []
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    rows.append(_row("python", "PASS" if py == "3.11" else "FAIL", version=py))
    for package, pin in _required_versions().items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        rows.append(_row(f"package.{package}", "PASS" if actual == pin else "FAIL",
                         reason="required version differs" if actual != pin else "",
                         version=actual, expected=pin))
    rows.append(_row("cpu", "PASS", cores=os.cpu_count(), ram_bytes=_memory_bytes()))
    for name, root in (("scratch", paths.SCRATCH), ("data", paths.DATA_DIR),
                       ("runs", paths.RUNS_DIR)):
        try:
            anchor = next((p for p in (root, *root.parents) if p.exists()), None)
            if anchor is None:
                raise FileNotFoundError()
            disk = shutil.disk_usage(anchor)
            stat = os.statvfs(anchor)
            rows.append(_row(f"disk.{name}", "PASS", free_bytes=disk.free,
                             free_inodes=stat.f_favail))
        except OSError as exc:
            rows.append(_safe_failure(f"disk.{name}", exc))
    for name in ("git", "curl", "sbatch", "sacct"):
        rows.append(_row(f"tool.{name}", "PASS" if shutil.which(name) else "WARN"))
    rows.append(_row("tool.archive", "PASS" if shutil.which("unrar") or shutil.which("7z")
                     else "WARN"))
    archive_specs = [*expected().get("inputs", {}).get("public_archives", {}).values(),
                     *expected().get("inputs", {}).get("official", {}).values()]
    hosts = sorted({host for spec in archive_specs if isinstance(spec, dict)
                    for host in [urlparse(spec.get("url", "")).hostname] if host})
    hosts.append("huggingface.co")
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            rows.append(_row(f"network.{host}", "PASS"))
        except OSError:
            rows.append(_row(f"network.{host}", "WARN", reason="DNS unavailable"))
    try:
        from octtta.losses import verify_backends
        deviation = verify_backends()
        rows.append(_row("edt_backends", "PASS", max_deviation=deviation))
    except Exception as exc:
        rows.append(_safe_failure("edt_backends", exc))
    if gpu:
        rows.extend(_gpu_probe())
    return {"status": _status(rows), "rows": rows}


def _selected_groups(only: str | None) -> set[str]:
    groups = set(_DATA_GROUPS) if only is None else {x.strip() for x in only.split(",")}
    if not groups or not groups <= _DATA_GROUPS:
        raise ValueError("unknown data-check selection")
    return groups


def _lines_digest(values) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _exclusions_static(doc: dict) -> dict:
    """Validate the shipped keyed lists even when licensed metadata is unavailable."""
    path = REPO_ROOT / "configs" / "aireadi_exclusions.json"
    pin = doc["aireadi"]["exclusions"]
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == pin["file_sha256"],
             "exclusion config digest differs")
    blob = json.loads(raw)
    _require(blob.get("schema") == 1 and blob.get("fingerprint") == "hmac-sha256-16-v1",
             "exclusion config schema differs")
    _require(blob.get("manifest_rows") == doc["aireadi"]["release"]["oct_manifest_rows"],
             "exclusion config manifest count differs")
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(blob.get("manifest_sha256", "")))),
             "exclusion config manifest digest is malformed")
    lists = blob.get("lists")
    _require(isinstance(lists, dict), "exclusion config lists are missing")
    counts = {}
    for name in ("never_train", "labelled_quality", "unlabelled_duplicates"):
        records = lists.get(name)
        _require(isinstance(records, list), "exclusion config list is missing")
        pairs = []
        for record in records:
            _require(isinstance(record, dict) and set(record) == {"fp", "vendor"},
                     "exclusion config record is malformed")
            fp, vendor = record["fp"], record["vendor"]
            _require(isinstance(fp, str) and re.fullmatch(r"[0-9a-f]{16}", fp)
                     and isinstance(vendor, str) and bool(vendor),
                     "exclusion config fingerprint or vendor is malformed")
            pairs.append((fp, vendor))
        _require(pairs == sorted(set(pairs)), "exclusion config list is not sorted unique")
        _require(len({fp for fp, _ in pairs}) == len(pairs),
                 "exclusion config fingerprint repeats")
        if name == "never_train":
            _require(len(pairs) == pin[name]["total"], "never-train count differs")
            by_vendor: dict[str, int] = defaultdict(int)
            for _, vendor in pairs:
                by_vendor[vendor] += 1
            _require(dict(by_vendor) == pin[name]["by_model"],
                     "never-train vendor counts differ")
            digest = _lines_digest(fp for fp, _ in pairs)
            want_digest = pin[name]["sorted_fingerprints_sha256"]
        else:
            digest = _lines_digest(f"{fp}\t{vendor}" for fp, vendor in pairs)
            field = ("sorted_fingerprint_vendor_sha256")
            want_digest = pin[name][field]
            if name == "labelled_quality":
                by_vendor = defaultdict(int)
                for _, vendor in pairs:
                    by_vendor[vendor] += 1
                _require({v: by_vendor[v] for v in pin[name]["volumes_by_vendor"]}
                         == pin[name]["volumes_by_vendor"],
                         "labelled-quality vendor counts differ")
            else:
                _require(len(pairs) == pin[name]["volumes"],
                         "duplicate-exclusion count differs")
        _require(digest == want_digest, "exclusion config list digest differs")
        counts[name] = len(pairs)
    return counts


def _static_sums(doc: dict) -> dict:
    """Cross-check public pins without a licensed mount."""
    ai = doc["aireadi"]
    pools = doc["pools"]
    labelled = pools["labelled"]
    unlabelled = pools["unlabelled"]
    ai_frames = sum(labelled["dirs"][name]["frames"] for name in AIREADI)
    _require(ai_frames == labelled["total_aireadi_frames"] == 287036,
             "AI-READI labelled frame sum differs")
    expected_pretrain = (ai["release"]["oct_manifest_rows"]
                         - ai["exclusions"]["test_split_oct_volumes"]
                         - ai["exclusions"]["never_train"]["total"])
    _require(expected_pretrain == ai["pretrain_manifest"]["volumes"] == 41938,
             "pretraining volume arithmetic differs")
    frames = sum(row["frames"] for row in unlabelled["dirs"].values())
    _require(frames == unlabelled["frames"] == 377646,
             "unlabelled frame sum differs")
    full_pass = unlabelled["coteach_full_pass"]
    _require(frames == full_pass["frames"] and frames // full_pass["batch"] == full_pass["steps"]
             and frames % full_pass["batch"] == full_pass["remainder"] == 6,
             "co-teach full-pass arithmetic differs")
    return {"labelled_frames": ai_frames, "pretrain_volumes": expected_pretrain,
            "unlabelled_frames": frames, "full_pass_steps": full_pass["steps"]}


def _aireadi_metadata(data_root: Path, doc: dict) -> dict:
    """Read only TSV metadata; keep all participant IDs and structural paths in memory."""
    from scripts.data import aireadi_common as common
    root = data_root / "public" / "ai_readi"
    people = common.load_participants(root)
    ex = common.Exclusions.load(root)
    pin = doc["aireadi"]
    splits: dict[str, int] = defaultdict(int)
    for row in people.values():
        splits[row["recommended_split"].strip().lower()] += 1
    _require(len(people) == pin["release"]["participants"]
             and dict(splits) == pin["release"]["split"],
             "AI-READI participant split counts differ")
    _require(len(ex.test_split) == pin["exclusions"]["test_split_persons"],
             "AI-READI test split count differs")
    oct_manifest = root / "retinal_oct" / "manifest.tsv"
    oct_paths = common._oct_paths(oct_manifest)
    _require(len(oct_paths) == pin["release"]["oct_manifest_rows"],
             "AI-READI structural row count differs")
    test_volumes = 0
    with oct_manifest.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("imaging") or "").strip().upper() == "OCT":
                test_volumes += int((row.get("person_id") or "").strip() in ex.test_split)
    _require(test_volumes == pin["exclusions"]["test_split_oct_volumes"],
             "AI-READI test volume count differs")
    with common.octa_manifest_path(root).open(newline="", encoding="utf-8-sig") as fh:
        octa_rows = sum(1 for _ in csv.DictReader(fh, delimiter="\t"))
    _require(octa_rows == pin["release"]["octa_manifest_rows"],
             "AI-READI OCTA row count differs")
    records = common.index_octa(root, require_local=False, include_optic_disc=True)
    donors = [r for r in records if not ex.is_never_train(r.person_id, r.structural_path)]
    _require(len(donors) == pin["octa_donor_volumes"],
             "AI-READI OCTA donor count differs")
    chosen = common.balanced_sample(donors, per_cell=100000, seed=20260823,
                                    max_per_person_per_cell=100000)
    chosen = [r for r in chosen if not ex.excludes_labelled_quality(
        r.structural_path, r.vendor)]
    for vendor, want in pin["labelled_selection"].items():
        fps = [ex.fingerprint(r.structural_path) for r in chosen if r.vendor == vendor]
        _require(len(fps) == want["volumes"]
                 and _lines_digest(fps) == want["selection_sha256"],
                 "AI-READI labelled selection differs")
    volumes = common.aireadi_pretrain_volumes(root, exclusions=ex,
                                               max_take=common.DEFAULT_MAX_TAKE)
    from scripts.data.extract_aireadi_frames import MANIFEST_COLUMNS
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter="\t")
    writer.writerow(MANIFEST_COLUMNS)
    for vol in volumes:
        writer.writerow(["ai_readi", "aireadi-custom", vol.person_id, vol.vendor,
                         vol.model, vol.anatomic_region, int(vol.is_widefield),
                         vol.laterality, vol.height, vol.width, vol.n_frames,
                         ",".join(map(str, vol.take_frames)), vol.src_path,
                         common.npz_relpath(vol)])
    digest = hashlib.sha256(text.getvalue().encode("utf-8")).hexdigest()
    pretrain = pin["pretrain_manifest"]
    _require(digest == pretrain["sha256"] and len(volumes) == pretrain["volumes"]
             and len({v.person_id for v in volumes}) == pretrain["persons"]
             and sum(len(v.take_frames) for v in volumes) == pretrain["planned_frames"],
             "AI-READI in-memory pretrain manifest differs")
    models: dict[str, int] = defaultdict(int)
    for vol in volumes:
        models[vol.model] += 1
    _require(dict(models) == pretrain["by_model"],
             "AI-READI pretrain model counts differ")
    return {"participants": len(people), "split": dict(splits),
            "oct_volumes": len(oct_paths), "octa_rows": octa_rows,
            "donors": len(donors), "pretrain_volumes": len(volumes),
            "pretrain_sha256": digest}


def _isfahan_selection(data_root: Path, pin: dict) -> dict:
    """Count the OCT5k-selected B-scans, not the wider Isfahan distribution."""
    selection = data_root / pin["selection_csv"]
    root = data_root / pin["extract_root"]
    if not selection.is_file() or not root.is_dir():
        return _row("isfahan.selected", "BLOCKED", reason="selection or images missing",
                    source="OCT5k annotations and the Isfahan distribution")
    with selection.open(newline="") as fh:
        names = [row[1].lstrip("./") for row in csv.reader(fh) if len(row) == 2]
    if len(names) != pin["n_files"] or len(set(names)) != len(names):
        return _row("isfahan.selected", "FAIL", reason="selection count differs",
                    files=len(names))
    # Resolve each row to refuse traversal or a symlink to data outside this root.
    base = root.resolve()
    present = sum((root / name).resolve().is_relative_to(base) and
                  (root / name).is_file() for name in names)
    digest = _lines_digest(names)
    if present != len(names) or digest != pin["inventory_sha256"]:
        return _row("isfahan.selected", "FAIL", reason="selected inventory differs",
                    files=present)
    return _row("isfahan.selected", "PASS", files=present, inventory_sha256=digest)


def _sam_export(path: Path, pin: dict) -> dict:
    if not path.is_file():
        return _row("sam.export", "BLOCKED", reason="encoder export is missing",
                    source=pin["repo"])
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = payload["state_dict"]
        provenance = payload["provenance"]
        good = (len(state) == pin["export_tensors"] and
                sum(v.numel() for v in state.values()) == pin["export_params"] and
                provenance["source_sha256"] == pin["sha256"])
        return _row("sam.export", "PASS" if good else "FAIL",
                    reason="export tensor structure or provenance differs" if not good else "",
                    tensors=len(state))
    except Exception as exc:
        return _safe_failure("sam.export", exc)


def published_weights_root(local: dict | None = None) -> Path:
    """Resolve the same published pair for checks, scoring, inference, and packaging."""
    return Path(os.environ.get("OCTTTA_PUBLISHED_WEIGHTS") or
                (local or {}).get("published_weights") or
                paths.RUNS_DIR / "export_dual_rc20").expanduser()


def check_data(data_root: Path, only: str | None = None, local: dict | None = None) -> dict:
    """Check pinned source inputs; licensed records never enter returned rows."""
    data_root = Path(data_root)
    doc = expected()
    groups = _selected_groups(only)
    inputs = doc["inputs"]
    rows: list[dict] = []
    if "public" in groups:
        for name, pin in inputs["public_archives"].items():
            archive = data_root / "public" / "_archives" / name
            archive_row = _file_pin(f"public.archive.{Path(name).stem}", archive,
                                    pin, source=pin["url"])
            rows.append(archive_row)
            extracted = data_root / "public" / "extracted" / pin["extract_root"]
            rows.append(_extracted_inventory(f"public.extracted.{pin['extract_root']}",
                                             archive, extracted, archive_row,
                                             source=pin["url"]))
        rows.append(_isfahan_selection(data_root, inputs["isfahan"]))
    if "official" in groups:
        for key, filename, directory in (("starting_kit", "starting_kit.zip", "starting_kit"),
                                         ("synthetic", "data_synthetic_v1.0.tar",
                                          "synthetic_v1.0")):
            pin = inputs["official"][key]
            archive = data_root / "official_archives" / filename
            archive_row = _file_pin(f"official.archive.{key}", archive, pin,
                                    source=pin["url"])
            rows.append(archive_row)
            rows.append(_extracted_inventory(f"official.extracted.{key}", archive,
                                             data_root / directory, archive_row,
                                             n_files=pin["files"], source=pin["url"]))
        for filename, pin in doc["needs"]["kit"]["files"].items():
            rows.append(_file_pin(f"official.kit.{Path(filename).name}",
                                  data_root / filename, pin))
    if "sam" in groups:
        pin = inputs["sam_l"]
        rows.append(_file_pin("sam.source", data_root / "pretrained" / "sam_l" /
                              "samvit_large_patch16.sa1b" / pin["filename"], pin,
                              source=pin["repo"]))
        rows.append(_sam_export(paths.CKPT_DIR / "e08" / "sam_l_offtheshelf.pt", pin))
    if "published" in groups:
        weight_root = published_weights_root(local)
        for filename, pin in doc["published_weights"].items():
            rows.append(_file_pin(f"published.{filename}", weight_root / filename, pin))
    if "aireadi" in groups:
        try:
            counts = _exclusions_static(doc)
            counts.update(_static_sums(doc))
            rows.append(_row("aireadi.static", "PASS", **counts))
        except Exception as exc:
            rows.append(_safe_failure("aireadi.static", exc))
        ai_root = data_root / "public" / "ai_readi"
        metadata = (ai_root / "participants.tsv", ai_root / "retinal_oct" / "manifest.tsv")
        if not all(p.is_file() for p in metadata):
            rows.append(_row("aireadi.metadata", "BLOCKED",
                             reason="AI-READI v3.0.0 requires FAIRhub registered access",
                             source="FAIRhub DOI 10.60775/fairhub.3"))
        else:
            try:
                # A private override may point at a different licensed release.  It must
                # agree with the root used by the exclusion key derivation.
                override = (local or {}).get("aireadi_participants")
                if override and Path(override).resolve() != metadata[0].resolve():
                    raise ValueError("licensed participant override differs from data root")
                rows.append(_row("aireadi.metadata", "PASS",
                                 **_aireadi_metadata(data_root, doc)))
            except Exception as exc:
                rows.append(_safe_failure("aireadi.metadata", exc))
    report = {"status": _status(rows), "rows": rows}
    if any(r["name"] == "aireadi.metadata" and r["status"] == "BLOCKED" for r in rows):
        report["possible_without_aireadi"] = list(_POSSIBLE_WITHOUT_AIREADI)
    return report

PUBLIC = ("oct5k", "jhu_hcms", "duke_dme_2015")
AIREADI = tuple(f"aireadi::{v}" for v in (
    "Heidelberg_Spectralis", "Topcon_Maestro2", "Topcon_Triton", "Zeiss_Cirrus"))
IGNORE_ON_DISK = {f"aireadi_inner::{v}" for v in (
    "Heidelberg_Spectralis", "Topcon_Maestro2", "Topcon_Triton", "Zeiss_Cirrus")}


def expected() -> dict:
    return json.loads((REPO_ROOT / "configs" / "expected.json").read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stems_sha256(stems) -> str:
    return hashlib.sha256(("\n".join(sorted(stems)) + "\n").encode()).hexdigest()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _read_index(path: Path, key: str) -> dict:
    _require(path.is_file(), f"{key}: pool index is missing")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{key}: pool index is unreadable ({type(exc).__name__})") from None


def _identity_check(name: str, measured: int, hits: int) -> dict:
    _require(measured > 0, f"{name}: no source associations were measured")
    _require(hits == 0, f"{name}: {hits} of {measured} source associations failed")
    return {"n_measured": measured, "n_hit": hits}


def _labelled(data_root: Path, pin: dict, names: tuple[str, ...],
              ex=None) -> tuple[dict, set[str]]:
    from octtta.data.partial_labels import SURFACE_TO_BOUNDARY, offsets_to_json
    from scripts.data import aireadi_common as common
    root = data_root / pin["root"]
    _require(root.is_dir(), "labelled pool root is missing")
    if set(names) == set((*PUBLIC, *AIREADI)):
        present = {p.name for p in root.iterdir() if p.is_dir()}
        if "build_report" in present:
            reports = root / "build_report"
            _require(all(p.is_file() and p.suffix == ".json" and p.name != "index.json"
                         for p in reports.iterdir()),
                     "build report directory contains non-report inputs")
            present.remove("build_report")
        _require(present <= set(names) | IGNORE_ON_DISK,
                 "labelled pool has an unexpected directory")
    report = {}
    ai_fingerprints: set[str] = set()
    donor_measure = donor_hits = never_measure = never_hits = 0
    octa_rows = None
    if any(k in AIREADI for k in names):
        _require(ex is not None, "AI-READI exclusion configuration is missing")
        octa_rows = common.index_octa(data_root / "public" / "ai_readi",
                                      require_local=False, include_optic_disc=True)
        donor_sources = {(r.structural_path, r.seg_path, r.person_id, r.vendor, r.split)
                         for r in octa_rows}
        _require(bool(donor_sources), "AI-READI OCTA metadata has no donor associations")
    for name in names:
        index = _read_index(root / name / "index.json", name)
        entries = index.get("entries")
        _require(isinstance(entries, list) and entries, f"{name}: no indexed frames")
        want = pin["dirs"][name]
        ai = name in AIREADI
        stems = [e["stem"] for e in entries]
        volumes = {e["volume_tag"] if ai else e["volume"] for e in entries}
        persons = {e["person_id"] if ai else e["group"] for e in entries}
        observed = {"frames": len(entries), "volumes": len(volumes),
                    "persons": len(persons), "stems_sha256": stems_sha256(stems)}
        for field, actual in observed.items():
            _require(actual == want[field], f"{name}: {field} differs from expected.json")
        _require(len(stems) == len(set(stems)), f"{name}: duplicate indexed stems")
        mapping = SURFACE_TO_BOUNDARY[name]
        _require(index.get("offsets_px") == offsets_to_json(mapping.offsets_px),
                 f"{name}: boundary offsets differ from the mapping")
        _require(index.get("boundaries") == list(mapping.available),
                 f"{name}: boundaries differ from the mapping")
        refs = {e["image"] for e in entries}
        refs.update(label for e in entries for label in e.get("labels", {}).values())
        _require(refs <= set(os.listdir(root / name)),
                 f"{name}: indexed PNG is missing")
        if "frames_by_cell" in want:
            cell_count: dict[str, int] = defaultdict(int)
            for e in entries:
                cell_count[f"{e['device']}|{e['protocol']}"] += 1
            _require(dict(cell_count) == want["frames_by_cell"],
                     f"{name}: per-cell frame counts differ")
        if ai:
            vendor = name.split("::", 1)[1]
            quality = expected()["aireadi"]["exclusions"]["labelled_quality"]["volumes_by_vendor"][vendor]
            quality_count = (index.get("labelled_quality") or {}).get("excluded_here")
            if quality_count is None:  # existing licensed pool predates the B6 metadata name
                quality_count = (index.get("exclude_volumes") or {}).get(
                    "n_volumes_excluded_here")
            _require(quality_count == quality,
                     f"{name}: quality exclusion count differs")
            selection = index.get("selection") or {}
            fixed_selection = {"per_cell": 100000, "seed": 20260823,
                               "frames_per_volume": 16,
                               "max_per_person_per_cell": 100000,
                               "include_optic_disc": True}
            _require(all(selection.get(k) == v for k, v in fixed_selection.items()),
                     f"{name}: selection metadata differs from the frozen recipe")
            selected_volumes = expected()["aireadi"]["labelled_selection"][vendor]["volumes"]
            _require(index.get("stats", {}).get("volumes") == selected_volumes,
                     f"{name}: selected volume count differs from expected.json")
            for e in entries:
                donor_measure += 1
                never_measure += 1
                assoc = (e["structural_path"], e["seg_path"], e["person_id"], vendor, "tune")
                donor_hits += int(assoc not in donor_sources)
                never_hits += int(ex.is_never_train(e["person_id"], e["structural_path"]))
                ai_fingerprints.add(ex.fingerprint(e["structural_path"]))
        report[name] = observed
    if donor_measure:
        report["donor_slice"] = _identity_check("labelled donor slice", donor_measure, donor_hits)
        report["never_train"] = _identity_check("labelled never-train", never_measure, never_hits)
    return report, ai_fingerprints


def _unlabelled(data_root: Path, pin: dict, ex) -> tuple[dict, set[str]]:
    from scripts.data import stage_unlabeled_pool as stage
    root = data_root / pin["root"]
    index = _read_index(root / "index.json", "unlabelled")
    entries = index.get("images")
    _require(isinstance(entries, list) and entries, "unlabelled pool has no indexed frames")
    _require(len(entries) == pin["frames"], "unlabelled frame count differs")
    _require(len({e["src_npz"] for e in entries}) == pin["volumes_with_frames"],
             "unlabelled volume count differs")
    _require(len({e["person_id"] for e in entries}) == pin["persons"],
             "unlabelled person count differs")
    _require(sha256_file(root / "MANIFEST.tsv") == pin["manifest_tsv_sha256"],
             "unlabelled MANIFEST.tsv digest differs")
    with (root / "MANIFEST.tsv").open(newline="") as fh:
        manifest_rows = list(csv.DictReader(fh, delimiter="\t"))
    _require(len(manifest_rows) == pin["manifest_tsv_rows"] == len(entries),
             "unlabelled MANIFEST.tsv row count differs")
    _require(all(all(str(e[c]) == row[c] for c in stage.MANIFEST_COLUMNS)
                 for e, row in zip(entries, manifest_rows)),
             "unlabelled index and MANIFEST.tsv associations differ")
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e["vendor_dir"]].append(e)
    _require(set(groups) == set(pin["dirs"]), "unlabelled vendor directories differ")
    report = {}
    for name, want in pin["dirs"].items():
        rows = groups[name]
        stems = [r["stem"] for r in rows]
        observed = {"frames": len(rows),
                    "volumes": len({r["src_npz"] for r in rows}),
                    "persons": len({r["person_id"] for r in rows}),
                    "stems_sha256": stems_sha256(stems)}
        for field, actual in observed.items():
            _require(actual == want[field], f"{name}: {field} differs from expected.json")
        _require(len(stems) == len(set(stems)), f"{name}: duplicate indexed stems")
        _require({Path(r["relpath"]).name for r in rows} <=
                 set(os.listdir(root / name)),
                 f"{name}: indexed PNG is missing")
        report[name] = observed
    build = _read_index(root / "build_report.json", "unlabelled build report")
    _require(build.get("totals", {}).get("volumes") == pin["volumes_planned"],
             "unlabelled planned volume count differs")
    _require(build.get("qc_rejected", {}).get("n") == pin["qc_rejected"],
             "unlabelled QC count differs")
    _require(build.get("qc_rejected", {}).get("by_reason") == pin["qc_rejected_by_reason"],
             "unlabelled QC reasons differ")
    duplicate_count = (build.get("exclusions") or {}).get("unlabelled_duplicates")
    if duplicate_count is None:  # frozen local pool carries the pre-B6 metadata name
        duplicate_count = (build.get("exclude_volumes") or {}).get("n_volumes_excluded")
    _require(duplicate_count == pin["duplicates"]["volumes"],
             "unlabelled duplicate exclusion count differs")
    full = pin["coteach_full_pass"]
    _require(full["frames"] == len(entries) and
             divmod(len(entries), full["batch"]) == (full["steps"], full["remainder"])
             and 0 < full["remainder"] < full["batch"],
             "co-teaching full-pass arithmetic differs")
    pre = data_root / "derived" / "pretrain_pool" / "manifest_aireadi.tsv"
    _require(pre.is_file(), "AI-READI pretraining manifest is missing")
    _require(sha256_file(pre) == expected()["aireadi"]["pretrain_manifest"]["sha256"],
             "AI-READI pretraining manifest digest differs")
    with pre.open(newline="") as fh:
        source_rows = list(csv.DictReader(fh, delimiter="\t"))
    source_by_npz = {r["npz_relpath"]: r for r in source_rows}
    _require(len(source_by_npz) == len(source_rows), "pretraining source keys are duplicated")
    fingerprints: set[str] = set()
    measured = hit = never_measured = never_hit = 0
    for e in entries:
        measured += 1
        source = source_by_npz.get(e["src_npz"])
        if source is None:
            hit += 1
            continue
        if (source["person_id"] != e["person_id"] or
            source["vendor"] != e["vendor"] or source["model"] != e["model"] or
            source["laterality"] != e["laterality"]):
            hit += 1
        never_measured += 1
        never_hit += int(ex.is_never_train(source["person_id"], source["src_path"]))
        fingerprints.add(ex.fingerprint(source["src_path"]))
    report["source_associations"] = _identity_check("unlabelled source", measured, hit)
    report["never_train"] = _identity_check("unlabelled never-train", never_measured, never_hit)
    return report, fingerprints


def check_pools(data_root: Path, *, only: str | None = None,
                pins: dict | None = None) -> dict:
    """Check frozen local pools and measured licensed-source guards; return aggregates."""
    from scripts.data import aireadi_common as common
    if only not in (None, "public", "aireadi", "labelled", "unlabelled"):
        raise ValueError("unknown pool selection")
    pins = pins or expected()["pools"]
    do_public = only in (None, "public", "labelled")
    do_ai = only in (None, "aireadi", "labelled")
    do_unlab = only in (None, "unlabelled")
    ex = (common.Exclusions.load(Path(data_root) / "public" / "ai_readi")
          if do_ai or do_unlab else None)
    report = {"status": "PASS", "labelled": {}, "unlabelled": {}}
    labelled_fps: set[str] = set()
    if do_public or do_ai:
        names = tuple(PUBLIC if do_public else ()) + tuple(AIREADI if do_ai else ())
        report["labelled"], labelled_fps = _labelled(Path(data_root), pins["labelled"], names, ex)
    if do_unlab:
        report["unlabelled"], unlabelled_fps = _unlabelled(Path(data_root), pins["unlabelled"], ex)
        if do_ai:
            report["overlap"] = _identity_check(
                "labelled/unlabelled overlap", len(labelled_fps) + len(unlabelled_fps),
                len(labelled_fps & unlabelled_fps))
        else:
            _, labelled_fps = _labelled(Path(data_root), pins["labelled"], AIREADI, ex)
            report["overlap"] = _identity_check(
                "labelled/unlabelled overlap", len(labelled_fps) + len(unlabelled_fps),
                len(labelled_fps & unlabelled_fps))
    return report


def _run_recipe(name: str) -> Path:
    lineage, stage_name = name.split("_", 1)
    _require(lineage in ("s33", "s34") and stage_name in
             ("sam_phase1", "sam_phase2", "cnn", "coteach"),
             "run pin has an unknown recipe")
    base = REPO_ROOT / "configs" / "reproduction"
    return base / ("model_b" if lineage == "s34" else "model_a") / f"{stage_name}.yaml"


def _relative_source(value: object, runs_root: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return path.as_posix().removeprefix("./")
    try:
        return path.resolve(strict=False).relative_to(runs_root.resolve()).as_posix()
    except ValueError:
        return "outside-runs-root"


def _config_row(name: str, run_dir: Path, runs_root: Path, pin: dict) -> dict:
    path = run_dir / "config.resolved.yaml"
    if not path.is_file():
        return _row(f"run.{name}.config", "BLOCKED", reason="resolved config missing")
    try:
        import yaml
        from octtta.config import expand_env, load_config
        from octtta.engine import flatten_config
        actual = yaml.safe_load(path.read_text())
        env = {**os.environ, "OCTTTA_RUNS": str(runs_root),
               "OCTTTA_DATA": str(paths.DATA_DIR), "OCTTTA_CKPT": str(paths.CKPT_DIR),
               "OCTTTA_SCRATCH": str(paths.SCRATCH)}
        recipe = expand_env(load_config(_run_recipe(name)), env=env)
        _require(isinstance(actual, dict), "resolved config is not a mapping")
        got = flatten_config(actual)
        want = flatten_config(recipe)
        got.pop("_source", None)
        want.pop("_source", None)
        for key in ("train.finetune_from", "train.finetune_from_b"):
            if key in got or key in want:
                got[key] = _relative_source(got.get(key), runs_root)
                want[key] = _relative_source(want.get(key), runs_root)
        differences = sum(got.get(key) != want.get(key) for key in set(got) | set(want))
        # Co-teaching warm starts are CLI flags, recorded in the checkpoint;
        # the resolved recipe intentionally leaves train.finetune_from null.
        recipe_source = None if name.endswith("coteach") else pin["finetune_from"]
        pin_ok = (actual.get("train", {}).get("epochs") == pin["epochs"] and
                  actual.get("train", {}).get("stop_epoch") == pin["stop_epoch"] and
                  _relative_source(actual.get("train", {}).get("finetune_from"),
                                   runs_root) == recipe_source)
        return _row(f"run.{name}.config", "PASS" if not differences and pin_ok else "FAIL",
                    reason="recipe or run pin differs" if differences or not pin_ok else "",
                    differing_fields=differences)
    except Exception as exc:
        return _safe_failure(f"run.{name}.config", exc)


def _audit_row(name: str, run_dir: Path, pin: dict) -> dict:
    path = run_dir / "pools.audit.json"
    if not path.is_file():
        return _row(f"run.{name}.audit", "BLOCKED", reason="pool audit missing")
    try:
        audit = json.loads(path.read_text())
        loaded = audit["loaded"]
        partial = loaded["partial_pool"]
        cells = partial["by_cell"]
        capped = partial["capped"]
        valid = (loaded["n_train"] == pin["n_train"] and
                 loaded["n_val"] == pin["n_val"] and
                 {k: partial[k] for k in ("n", "by_cell", "capped")} == pin["partial_pool"] and
                 loaded["draws_per_epoch"] == pin["draws_per_epoch"] and
                 loaded["draws_per_epoch_effective"] == pin["draws_per_epoch_effective"] and
                 loaded["sampling"] == pin["sampling"] and
                 loaded["draws_per_epoch_effective"] // pin["batch_size"] == pin["steps_per_epoch"] and
                 loaded.get("full_pass") == pin["full_pass"])
        # Historical audits include this field.  New B8 audits move overlap to
        # check pools, so absence here is not evidence of zero overlap.
        overlap = loaded.get("volume_overlap")
        if overlap is not None:
            valid = valid and overlap.get("n_shared") == 0
        return _row(f"run.{name}.audit", "PASS" if valid else "FAIL",
                    reason="loaded pool counts or full-pass record differ" if not valid else "",
                    n_train=loaded["n_train"], n_val=loaded["n_val"],
                    capped=capped["n"], cells=len(cells),
                    n_shared=overlap.get("n_shared") if overlap else None)
    except Exception as exc:
        return _safe_failure(f"run.{name}.audit", exc)


def _checkpoint_row(name: str, run_dir: Path, runs_root: Path, pin: dict,
                    *, student_b: bool = False) -> dict:
    label = f"run.{name}.last_b" if student_b else f"run.{name}.last"
    path = run_dir / "checkpoints" / ("last_b.pt" if student_b else "last.pt")
    if not path.is_file():
        return _row(label, "BLOCKED", reason="pinned last checkpoint missing")
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        want_source = pin["finetune_from_b" if student_b else "finetune_from"]
        got_source = _relative_source(payload.get("finetune_from"), runs_root)
        valid = (payload.get("epoch") == pin["last_epoch"] and
                 payload.get("global_step") == pin["global_step"] and
                 got_source == want_source and isinstance(payload.get("model"), dict))
        if not student_b:
            config = payload.get("config") or {}
            valid = valid and config.get("experiment", {}).get("id") == name
        result = _row(label, "PASS" if valid else "FAIL",
                      reason="epoch, step, warm start, or model differs" if not valid else "",
                      epoch=payload.get("epoch"), global_step=payload.get("global_step"),
                      warm_start=got_source)
        del payload
        return result
    except Exception as exc:
        return _safe_failure(label, exc)


def _run_source_overlap() -> dict:
    pool_pins = expected()["pools"]
    try:
        from scripts.data import aireadi_common as common
        ex = common.Exclusions.load(paths.DATA_DIR / "public" / "ai_readi")
        _, labelled = _labelled(paths.DATA_DIR, pool_pins["labelled"], AIREADI, ex)
        _, unlabelled = _unlabelled(paths.DATA_DIR, pool_pins["unlabelled"], ex)
        overlap = _identity_check("labelled/unlabelled overlap",
                                  len(labelled) + len(unlabelled),
                                  len(labelled & unlabelled))
        return _row("runs.source_overlap", "PASS",
                    n_measured=overlap["n_measured"], n_shared=overlap["n_hit"])
    except Exception as exc:
        required = (paths.DATA_DIR / pool_pins["labelled"]["root"],
                    paths.DATA_DIR / pool_pins["unlabelled"]["root"],
                    paths.DATA_DIR / "public" / "ai_readi" / "participants.tsv")
        if all(path.exists() for path in required):
            return _safe_failure("runs.source_overlap", exc)
        return _row("runs.source_overlap", "BLOCKED",
                    reason="source pools or licensed metadata missing")


def check_runs(runs_root: Path, only: str | None = None) -> dict:
    """Check recipe lineage and pinned last checkpoints without evaluating scores."""
    runs_root = Path(runs_root)
    pins = expected()["runs"]
    names = (only,) if only is not None else tuple(pins)
    if any(name not in pins for name in names):
        raise ValueError("unknown run selection")
    rows: list[dict] = [_run_source_overlap()]
    for name in names:
        pin = pins[name]
        run_dir = runs_root / name
        rows.append(_config_row(name, run_dir, runs_root, pin))
        rows.append(_audit_row(name, run_dir, pin))
        runenv = run_dir / "run_env.json"
        try:
            if not runenv.is_file():
                rows.append(_row(f"run.{name}.environment", "BLOCKED",
                                 reason="run environment record missing"))
            else:
                record = json.loads(runenv.read_text())
                launches = record.get("launches")
                ok = isinstance(launches, list) and bool(launches) and all(
                    e.get("torch") and e.get("cuda") and e.get("gpu_names")
                    for e in launches)
                rows.append(_row(f"run.{name}.environment", "PASS" if ok else "FAIL",
                                 launches=len(launches) if isinstance(launches, list) else 0))
        except Exception as exc:
            rows.append(_safe_failure(f"run.{name}.environment", exc))
        rows.append(_checkpoint_row(name, run_dir, runs_root, pin))
        if name.endswith("coteach"):
            rows.append(_checkpoint_row(name, run_dir, runs_root, pin, student_b=True))
    return {"status": _status(rows), "rows": rows}


def main(argv: list[str] | None = None, *, local: dict | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env")
    env.add_argument("--gpu", action="store_true")
    data = sub.add_parser("data")
    data.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    data.add_argument("--only", help="comma-separated: public,official,sam,published,aireadi")
    runs = sub.add_parser("runs")
    runs.add_argument("--runs-root", type=Path, default=paths.RUNS_DIR)
    runs.add_argument("--only", choices=tuple(expected().get("runs", {})))
    pool = sub.add_parser("pools")
    pool.add_argument("--only", choices=("public", "aireadi", "labelled", "unlabelled"))
    pool.add_argument("--print-digests", action="store_true")
    pool.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    args = p.parse_args(argv)
    if args.command == "data":
        try:
            _selected_groups(args.only)
        except ValueError:
            p.error("--only must be a comma-separated subset of public,official,sam,published,aireadi")
    try:
        if args.command == "env":
            report = check_env(gpu=args.gpu)
        elif args.command == "data":
            report = check_data(args.data_root, only=args.only, local=local)
        elif args.command == "runs":
            report = check_runs(args.runs_root, only=args.only)
        else:
            report = check_pools(args.data_root, only=args.only)
    except Exception as exc:
        print(json.dumps({"status": "WARN", "reason": f"{type(exc).__name__}; inspect local inputs",
                          "next": "Review the selected inputs and retry the check.",
                          "advisory": True}, sort_keys=True))
        return 0
    if args.command == "pools" and not args.print_digests:
        for category in ("labelled", "unlabelled"):
            for entry in report[category].values():
                if isinstance(entry, dict):
                    entry.pop("stems_sha256", None)
    if "rows" in report:
        report["rows"] = [{**row, "status": "WARN" if row["status"] in {"FAIL", "BLOCKED"}
                           else row["status"]} for row in report["rows"]]
    if report["status"] in {"FAIL", "BLOCKED"}:
        report["status"] = "WARN"
    report["advisory"] = True
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
