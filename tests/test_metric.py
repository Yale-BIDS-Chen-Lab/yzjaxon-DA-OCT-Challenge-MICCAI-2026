"""The metric port matches official outputs on fake cohorts and the pinned kit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs, needs_root

from octtta.eval import challenge_metric as cm

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "metric_golden.json"


def _fake_masks(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic masks; no source image or participant metadata is used."""
    height, width = 24, 32
    y = np.arange(height)[:, None]
    gt = np.broadcast_to(
        np.select([y < 4, y < 8, y < 12, y < 16, y < 20],
                  [0, 1, 2, 3, 4], default=9),
        (height, width),
    ).astype(np.uint8).copy()
    if case == "exact":
        pred = gt.copy()
    elif case == "shifted":
        pred = np.roll(gt, 2, axis=0)
    elif case == "local_error":
        pred = gt.copy()
        pred[8:15, 6:17] = 6
    elif case == "wide_error":
        pred = np.roll(gt, 4, axis=1)
        pred[4:19, 0:8] = 8
    else:
        raise ValueError(f"unknown fake-mask case: {case}")
    return gt, pred


def _fake_records() -> list[cm.ScoreRecord]:
    cohorts = (
        ("Maestro2", "healthy", "Macula", "exact"),
        ("Maestro2", "diseased", "Macula", "local_error"),
        ("Triton", "healthy", "Macula", "shifted"),
        ("Triton", "diseased", "Macula", "wide_error"),
        ("Cirrus", "healthy", "WideField", "exact"),
        ("Triton", "diseased", "WideField", "local_error"),
    )
    records = []
    for device, status, anatomy, case in cohorts:
        gt, pred = _fake_masks(case)
        records.append(cm.ScoreRecord(cm.compute_image_score(pred, gt), device, status, anatomy))
    return records


def _assert_official_pins() -> Path:
    drift = cm.verify_against_official()
    assert not drift, "official scorer pin or semantic check failed:\n" + "\n".join(drift)
    from octtta import paths

    return Path(paths.DATA_DIR) / "starting_kit" / "app_scoring" / "program"


def test_fake_cohort_matches_official_metric_golden():
    expected = json.loads(GOLDEN.read_text())
    got = cm.aggregate(_fake_records())
    assert got.final_score == pytest.approx(expected["final_score"], abs=1e-12)
    assert got.anatomy_scores["Macula"] == pytest.approx(expected["macula_score"], abs=1e-12)
    assert got.anatomy_scores["WideField"] == pytest.approx(expected["widefield_score"], abs=1e-12)


def _official_scores(program: Path, input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(program / "scoring.py"), str(input_dir), str(output_dir)],
        check=True, cwd=program, env=env, capture_output=True, text=True,
    )
    del result  # scorer output may contain kit filenames; keep test output identifier-free.
    return json.loads((output_dir / "scores.json").read_text())


def _port_scores(input_dir: Path) -> cm.ChallengeScore:
    import cv2
    import pandas as pd

    ref = input_dir / "ref" / "val"
    pred_dir = input_dir / "res"
    df = pd.read_csv(ref / "val_release.csv")
    df = df[df["release_mask_name"].notna()]
    records = []
    for _, row in df.iterrows():
        name = row["release_mask_name"]
        gt_path, pred_path = ref / "masks" / name, pred_dir / name
        if not gt_path.exists() or not pred_path.exists():
            continue  # the official scorer skips missing masks/predictions too
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred.ndim == 3:
            pred = pred[..., 0]
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        records.append(cm.ScoreRecord(
            image_score=cm.compute_image_score(pred.astype(np.uint8), gt),
            device=row["device"],
            status=str(row["status"]).lower(),
            anatomy=cm.normalize_anatomy(row["group"]),
        ))
    return cm.aggregate(records)


@needs("kit")
def test_live_pinned_official_scorer_parity(tmp_path: Path):
    program = _assert_official_pins()
    input_dir = needs_root("kit") / "starting_kit" / "app_scoring" / "input"
    official = _official_scores(program, input_dir, tmp_path / "official")
    ours = _port_scores(input_dir)
    assert ours.final_score == pytest.approx(official["final_score"], abs=1e-9)
    assert ours.anatomy_scores.get("Macula", 0.0) == pytest.approx(
        official["macula_score"], abs=1e-9)
    assert ours.anatomy_scores.get("WideField", 0.0) == pytest.approx(
        official["widefield_score"], abs=1e-9)
