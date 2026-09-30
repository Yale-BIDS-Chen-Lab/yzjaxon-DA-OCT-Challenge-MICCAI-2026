"""Faithful port of the official DA-OCT evaluator, quirks included.

The authority is the starting kit under ``$OCTTTA_DATA/starting_kit``
(``LAMBDA_PENALTY = 0.5``), not the public sample repo, and aggregation follows that
kit's ``scoring.py``.
``verify_against_official()`` fails loudly on upstream drift. Do not "fix" these: Dice uses
``(2*inter + eps) / (union + eps)``, so a class absent from both sides scores 1.0, while an
empty boundary makes the surface distance NaN which maps to 0.0, giving that class 0.5.
MASD is divided by image height before ``exp(-d / TAU)``, so boundary precision dominates."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


NUM_CLASSES = 10

ALPHA = 0.3            # weight on HEALTHY; diseased therefore carries 1 - ALPHA = 0.7

#: Multiplier on the unseen-vendor generalisation penalty. 0.5 per the starting kit, which
#: is what the platform runs; the public sample repo says 1.5 and is not what runs.
LAMBDA_PENALTY = 0.5

BETA_MACULA = 0.5
BETA_WIDEFIELD = 0.5

TAU = 0.02             # MASD is height-normalised before exp(-masd / TAU)

SEEN_VENDORS = ["Maestro2", "Spectralis", "Cirrus"]
UNSEEN_VENDORS = ["Triton"]

EPS = 1e-6


def dice_score(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES,
               eps: float = EPS) -> np.ndarray:
    dices = []
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dices.append((2.0 * intersection + eps) / (union + eps))
    return np.array(dices)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    return mask ^ binary_erosion(mask)


def surface_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return np.nan

    gt_dist = distance_transform_edt(~gt_boundary)
    pred_dist = distance_transform_edt(~pred_boundary)

    pred_to_gt = gt_dist[pred_boundary]
    gt_to_pred = pred_dist[gt_boundary]

    return (pred_to_gt.mean() + gt_to_pred.mean()) / 2.0


def masd_per_class(pred: np.ndarray, gt: np.ndarray,
                   num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Mean absolute surface distance per class, normalised by image height."""
    height = pred.shape[0]
    out = []
    for c in range(num_classes):
        d = surface_distance(pred == c, gt == c)
        out.append(np.nan if np.isnan(d) else d / height)
    return np.array(out)


def masd_to_score(masd: np.ndarray) -> np.ndarray:
    return np.exp(-masd / TAU)


def compute_image_score(pred: np.ndarray, gt: np.ndarray,
                        num_classes: int = NUM_CLASSES) -> float:
    dice = dice_score(pred, gt, num_classes)
    masd_score = np.nan_to_num(masd_to_score(masd_per_class(pred, gt, num_classes)), nan=0.0)
    return float((0.5 * (dice + masd_score)).mean())


def compute_image_breakdown(pred: np.ndarray, gt: np.ndarray,
                            num_classes: int = NUM_CLASSES) -> dict:
    """Same as :func:`compute_image_score` but keeps the per-class terms for diagnosis."""
    dice = dice_score(pred, gt, num_classes)
    masd = masd_per_class(pred, gt, num_classes)
    masd_score = np.nan_to_num(masd_to_score(masd), nan=0.0)
    layer = 0.5 * (dice + masd_score)
    return {
        "dice": dice,
        "masd_norm": masd,
        "masd_score": masd_score,
        "layer_score": layer,
        "image_score": float(layer.mean()),
    }


def normalize_anatomy(anatomy: str) -> str:
    a = str(anatomy).lower()
    if "wide" in a:
        return "WideField"
    if "macula" in a:
        return "Macula"
    raise ValueError(f"Unknown anatomy: {anatomy}")


@dataclass
class ScoreRecord:
    """One evaluated image."""
    image_score: float
    device: str      # "Maestro2" | "Spectralis" | "Cirrus" | "Triton"
    status: str      # "healthy" | "diseased" (lowercase, as the official CSV has them)
    anatomy: str     # "Macula" | "WideField"


@dataclass
class ChallengeScore:
    final_score: float
    anatomy_scores: dict = field(default_factory=dict)
    vendor_scores: dict = field(default_factory=dict)
    cohort_scores: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)

    def format(self) -> str:
        lines = ["cohort (anatomy, vendor, status) -> mean image score:"]
        for k in sorted(self.cohort_scores, key=str):
            lines.append(f"    {k}: {self.cohort_scores[k]:.4f}")
        # Every weight below is interpolated, never spelled out, so a log cannot disagree.
        lines.append(f"vendor (anatomy, vendor) -> {ALPHA:g}*healthy "
                     f"+ {1.0 - ALPHA:g}*diseased:")
        for k in sorted(self.vendor_scores, key=str):
            lines.append(f"    {k}: {self.vendor_scores[k]:.4f}")
        lines.append("anatomy -> max(0, mean_over_vendors - "
                     f"{LAMBDA_PENALTY:g} * unseen_penalty):")
        for k in sorted(self.anatomy_scores):
            pen = self.penalties.get(k, 0.0)
            lines.append(f"    {k}: {self.anatomy_scores[k]:.4f}   (penalty {pen:.4f})")
        lines.append(f"FINAL: {self.final_score:.4f}")
        return "\n".join(lines)


def aggregate(records: Iterable[ScoreRecord]) -> ChallengeScore:
    """Reproduce the official cohort -> vendor -> anatomy -> final aggregation."""
    buckets: Mapping[tuple, list] = defaultdict(list)
    for r in records:
        buckets[(r.anatomy, r.device, r.status)].append(r.image_score)
    cohort_scores = {k: float(np.mean(v)) for k, v in buckets.items()}

    by_vendor: Mapping[tuple, dict] = defaultdict(dict)
    for (anatomy, vendor, status), score in cohort_scores.items():
        by_vendor[(anatomy, vendor)][status] = score

    vendor_scores = {
        key: ALPHA * vals.get("healthy", 0.0) + (1.0 - ALPHA) * vals.get("diseased", 0.0)
        for key, vals in by_vendor.items()
    }

    anatomy_scores, penalties = {}, {}
    for anatomy in ("Macula", "WideField"):
        per_vendor = {v: s for (a, v), s in vendor_scores.items() if a == anatomy}
        if not per_vendor:
            anatomy_scores[anatomy] = 0.0
            penalties[anatomy] = 0.0
            continue

        overall = float(np.mean(list(per_vendor.values())))
        seen_vals = [per_vendor[v] for v in SEEN_VENDORS if v in per_vendor]
        # Without the guard an anatomy holding only unseen vendors gives ``np.mean([])``,
        # i.e. NaN, and the NaN propagates into final_score.
        seen_mean = float(np.mean(seen_vals)) if seen_vals else 0.0

        penalty = 0.0
        for v in UNSEEN_VENDORS:
            if v in per_vendor:
                penalty += max(0.0, seen_mean - per_vendor[v])
        penalty /= max(len(UNSEEN_VENDORS), 1)

        penalties[anatomy] = penalty
        anatomy_scores[anatomy] = max(0.0, overall - LAMBDA_PENALTY * penalty)

    final = (BETA_MACULA * anatomy_scores.get("Macula", 0.0)
             + BETA_WIDEFIELD * anatomy_scores.get("WideField", 0.0))

    return ChallengeScore(
        final_score=float(final),
        anatomy_scores=anatomy_scores,
        vendor_scores=vendor_scores,
        cohort_scores=cohort_scores,
        penalties=penalties,
    )


#: The scorer the platform actually runs, relative to the pinned starting kit.
_OFFICIAL = "metrics.py"

#: Where the aggregation lives; ``metrics.compute_anatomy_score`` is dead code.
_OFFICIAL_SCORING = "scoring.py"

_EXPECTED_OFFICIAL_CONSTANTS = {
    "NUM_CLASSES": "10",
    "ALPHA": "0.3",
    "LAMBDA_PENALTY": "0.5",
    "BETA_MACULA": "0.5",
    "BETA_WIDEFIELD": "0.5",
    "TAU": "0.02",
}

#: Load-bearing expressions of the official aggregation, keyed by the decision each encodes.
#: Matching runs on AST-normalised source, so reformatting upstream does not fire; only a
#: change to the expression itself does. Each value must be parsable Python on its own.
_EXPECTED_AGGREGATION = {
    "cohort = mean image score per (anatomy, device, status)":
        'cohort_scores[(row["anatomy"], row["device"], row["status"])]'
        '.append(row["image_score"])',
    "vendor = ALPHA*healthy + (1-ALPHA)*diseased, a missing status counting as 0.0":
        'ALPHA * vals.get("healthy", 0.0) + (1.0 - ALPHA) * vals.get("diseased", 0.0)',
    "an anatomy with no images scores 0.0 instead of being excluded":
        'if not anatomy_vendor_scores:\n    anatomy_scores[anatomy] = 0.0',
    "an empty seen-vendor set falls back to 0.0, not NaN":
        'seen_mean = np.mean(seen_vals) if seen_vals else 0.0',
    "the unseen penalty is one-sided and averaged over len(UNSEEN_VENDORS), floored at 1":
        'penalty = sum(max(0.0, seen_mean - anatomy_vendor_scores[v])\n'
        '              for v in UNSEEN_VENDORS if v in anatomy_vendor_scores'
        ') / max(len(UNSEEN_VENDORS), 1)',
    "the anatomy score is clamped at 0 after the penalty":
        'anatomy_scores[anatomy] = max(0.0, overall - LAMBDA_PENALTY * penalty)',
    "final = BETA_MACULA*Macula + BETA_WIDEFIELD*WideField, a missing one counting as 0.0":
        'BETA_MACULA * anatomy_scores.get("Macula", 0.0) '
        '+ BETA_WIDEFIELD * anatomy_scores.get("WideField", 0.0)',
}


def _normalise_source(text: str) -> str:
    """Canonicalise Python source so only semantic differences survive.
    Raises ``SyntaxError`` when ``text`` does not parse, which the caller should hear about."""
    import ast

    return " ".join(ast.unparse(ast.parse(text)).split())


def verify_against_official(data_root=None) -> list[str]:
    """Check the expected size/SHA pins and semantic parity of the official scorer.

    Both official source files are verified against ``configs/expected.json:needs.kit``
    before either file is read, so an unpinned checkout cannot be treated as authority.
    """
    from pathlib import Path
    from octtta import paths

    data_root = Path(data_root) if data_root is not None else Path(paths.DATA_DIR)
    scoring_root = data_root / "starting_kit" / "app_scoring" / "program"
    repo_root = Path(__file__).resolve().parents[2]
    expected = json.loads((repo_root / "configs" / "expected.json").read_text())
    kit_files = expected.get("needs", {}).get("kit", {}).get("files", {})
    problems = []
    sources = {}
    for name in (_OFFICIAL, _OFFICIAL_SCORING):
        rel = f"starting_kit/app_scoring/program/{name}"
        pin = kit_files.get(rel)
        path = scoring_root / name
        if not pin or "sha256" not in pin or "bytes" not in pin:
            problems.append(f"missing size/SHA pin for official {name} in expected.json needs.kit")
            continue
        if not path.is_file():
            problems.append(f"official {name} not found under the pinned starting kit")
            continue
        if path.stat().st_size != pin["bytes"]:
            problems.append(f"official {name} size differs from its expected.json pin")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pin["sha256"]:
            problems.append(f"official {name} SHA256 differs from its expected.json pin")
            continue
        sources[name] = path
    if problems:
        return problems

    official = sources[_OFFICIAL]
    text = official.read_text()
    for name, expected in _EXPECTED_OFFICIAL_CONSTANTS.items():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name} ") and "=" in stripped:
                actual = stripped.split("=", 1)[1].strip()
                if actual != expected:
                    problems.append(f"{name}: upstream={actual!r} our port={expected!r}")
                break
        else:
            problems.append(f"{name}: not found in upstream metrics.py")

    for vendor_list, expected_names in (("SEEN_VENDORS", SEEN_VENDORS),
                                        ("UNSEEN_VENDORS", UNSEEN_VENDORS)):
        for name in expected_names:
            if f'"{name}"' not in text:
                problems.append(f"{vendor_list}: {name!r} no longer in upstream metrics.py")

    scoring = sources[_OFFICIAL_SCORING]
    try:
        normalised = _normalise_source(scoring.read_text())
    except SyntaxError as exc:
        problems.append(f"official scoring.py does not parse ({exc}) -- cannot verify "
                        "the aggregation our aggregate() copies")
    else:
        for decision, fragment in _EXPECTED_AGGREGATION.items():
            if _normalise_source(fragment) not in normalised:
                problems.append(
                    f"aggregation drift -- upstream scoring.py no longer contains the "
                    f"expression for {decision!r}; re-read it against aggregate()")

    return problems


if __name__ == "__main__":
    drift = verify_against_official()
    print("\n".join(drift) if drift else "port matches upstream metrics.py")
