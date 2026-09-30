"""Private synthetic scoring layout and checked inference for the reproduction wizard."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from octtta import paths  # noqa: E402
from octtta.data.release_dataset import index_flat_inference_dir, output_name_for  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected() -> dict:
    return json.loads((REPO / "configs/expected.json").read_text())


def _infer(a: Path, b: Path, images: Path, out: Path) -> None:
    cmd = (sys.executable, "-m", "octtta.infer", "--input", str(images),
           "--output", str(out), "--checkpoint", str(a), "--checkpoint-b", str(b),
           "--no-degrade")
    summary = out.parent / (out.name + "-summary.json")
    cmd = (*cmd, "--summary-json", str(summary))
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("inference failed; inspect the private run output")
    if not summary.is_file():
        raise RuntimeError("inference produced no completion summary")
    report = json.loads(summary.read_text())
    if report.get("n_failed") != 0 or report.get("failures") or report.get("inference_status") != "ok":
        raise RuntimeError("inference reported failures; inspect the private run output")
    masks = verify_infer(images, out)
    if report.get("n_images") != masks["n_images"] or report.get("n_written") != masks["n_masks"]:
        raise RuntimeError("inference completion counts differ from masks")


def verify_infer(images: Path, out: Path) -> dict:
    """Require one 8-bit class-0..9 mask at the exact source shape per image."""
    import cv2
    sources = index_flat_inference_dir(images)
    if not sources:
        raise ValueError("inference input contains no images")
    expected_names = {output_name_for(p) for p in sources}
    actual_names = {p.name for p in out.glob("*.png")}
    if expected_names != actual_names:
        raise ValueError("inference mask count or names differ from the input")
    for src in sources:
        source = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(out / output_name_for(src)), cv2.IMREAD_UNCHANGED)
        if source is None or mask is None or mask.dtype.name != "uint8":
            raise ValueError("inference image or mask cannot be read as uint8")
        if mask.ndim != 2 or mask.shape != source.shape[:2] or mask.max() > 9:
            raise ValueError("inference mask shape or label range differs from contract")
    return {"n_images": len(sources), "n_masks": len(actual_names), "status": "PASS"}


def _official_scorer(kit: Path, expected: dict) -> Path:
    pins = expected["needs"]["kit"]["files"]
    for rel in ("starting_kit/app_scoring/program/scoring.py",
                "starting_kit/app_scoring/program/metrics.py"):
        file = kit.parent / rel
        if not file.is_file():
            raise ValueError("official scorer is missing; download the official kit")
        if _sha(file) != pins[rel]["sha256"]:
            print("[WARN] official scorer differs from the recorded SHA-256; scores may not match the reference", file=sys.stderr)
    return kit / "app_scoring/program/scoring.py"


def _layout(root: Path, source: Path, n_expected: int) -> tuple[Path, Path, list[tuple[str, str]]]:
    images = root / "images"
    gt = root / "input/ref/val/masks"
    predicted = root / "predicted"
    res = root / "input/res"
    for d in (images, gt, predicted, res):
        d.mkdir(parents=True)
    rows: list[dict] = []
    output_names: list[tuple[str, str]] = []
    for cohort, want in (("Healthy", 173), ("Diseased", 57)):
        directory = source / cohort
        files = sorted(directory.glob("*-image.png"))
        if len(files) != want:
            raise ValueError("synthetic cohort image count differs from the pin")
        for image in files:
            mask = image.with_name(image.name.replace("-image.png", "-mask.png"))
            if not mask.is_file():
                raise ValueError("synthetic image has no reference mask")
            image_id = image.name[:-len("-image.png")]
            staged = f"syn_Topcon_Maestro2_{cohort}_{image.name}"
            mask_name = f"{image_id}-mask.png"
            (images / staged).symlink_to(image.resolve())
            (gt / mask_name).symlink_to(mask.resolve())
            output_names.append((staged.replace("-image.png", "-mask.png"), mask_name))
            rows.append({"release_image_name": image.name, "release_mask_name": mask_name,
                         "device": "Maestro2", "status": cohort.lower(),
                         "group": "Macula, 6 x 6",
                         "slice_path": f"{image_id}_L_Maestro2_Macula,6x6_ref001_dev001.png"})
    if len(rows) != n_expected:
        raise ValueError("synthetic image total differs from expected.json")
    with (root / "input/ref/val/val_release.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len({mask for _, mask in output_names}) != len(output_names):
        raise ValueError("synthetic mask names are not unique across cohorts")
    return images, predicted, output_names


def _environment() -> dict:
    import cv2
    import numpy
    import torch
    driver = None
    try:
        result = subprocess.run(("nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"), capture_output=True, text=True,
                                check=True, timeout=10)
        driver = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None,
            "cv2": cv2.__version__, "driver_versions": [driver] if driver else []}


def score(a: Path, b: Path, origin: str) -> dict:
    expected = _expected()
    reference = expected["synthetic_reference"]
    kit = paths.DATA_DIR / "starting_kit"
    scorer = _official_scorer(kit, expected)
    if not a.is_file() or not b.is_file():
        raise ValueError("the selected weight pair is missing")
    hashes = {"model.pt": _sha(a), "model_b.pt": _sha(b)}
    pinned_pair = all(hashes[name] == expected["published_weights"][name]["sha256"]
                      for name in hashes)
    if origin == "published" and not pinned_pair:
        print("[WARN] selected published weights differ from the recorded hashes; reporting their measured score", file=sys.stderr)
    paths.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repro-score-", dir=paths.RUNS_DIR) as temp:
        root = Path(temp)
        images, predicted, outputs = _layout(root, paths.SYNTHETIC_ROOT / "Topcon_Maestro2",
                                               reference["labelled_images"])
        _infer(a, b, images, predicted)
        for staged_name, mask_name in outputs:
            (root / "input/res" / mask_name).symlink_to(
                (predicted / staged_name).resolve())
        score_dir = root / "official_score"
        result = subprocess.run((sys.executable, str(scorer), str(root / "input"),
                                 str(score_dir)), cwd=scorer.parent,
                                capture_output=True, text=True)
        if result.returncode or not (score_dir / "scores.json").is_file():
            raise RuntimeError("official scorer failed on the synthetic masks")
        scores = json.loads((score_dir / "scores.json").read_text())
    env = _environment()
    scorer_hashes = {name: _sha(scorer.parent / name) for name in ("scoring.py", "metrics.py")}
    scorer_matches = all(scorer_hashes[name] == expected["needs"]["kit"]["files"][
        "starting_kit/app_scoring/program/" + name]["sha256"] for name in scorer_hashes)
    exact = pinned_pair and scorer_matches and env == reference["environment"]
    if env != reference["environment"]:
        print("[WARN] execution environment differs from the recorded reference; see check env", file=sys.stderr)
    delta = {k: scores[k] - reference[k] for k in
             ("final_score", "macula_score", "widefield_score")}
    if not all(math.isfinite(float(scores[k])) for k in delta):
        raise ValueError("official scorer returned a non-finite score")
    report = {"status": "MEASURED",
              "reference_matches_exactly": pinned_pair and all(v == 0 for v in delta.values()),
              "weight_sha256": hashes, "environment": env,
              "scorer_sha256": scorer_hashes, "scorer_matches_pin": scorer_matches,
              "n_images": len(outputs), "origin": "published" if pinned_pair else "retrained",
              "exact_comparison": exact, "scores": scores, "delta_from_published": delta,
              "environment_matches_pin": env == reference["environment"],
              "note": "These synthetic images were included in training."}
    (paths.RUNS_DIR / "repro_score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--a", required=True, type=Path)
    s.add_argument("--b", required=True, type=Path)
    s.add_argument("--origin", choices=("published", "mine"), required=True)
    i = sub.add_parser("infer")
    i.add_argument("--a", required=True, type=Path)
    i.add_argument("--b", required=True, type=Path)
    i.add_argument("--images", required=True, type=Path)
    i.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "score":
        score(args.a, args.b, args.origin)
        return 0
    _infer(args.a, args.b, args.images, args.out)
    print(json.dumps(verify_infer(args.images, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
