"""The recipes are the eight Final training runs.

Every key of every resolved recipe (``configs/reproduction/model_*/**.yaml``) equals the config its run
recorded (``tests/data/run_configs/<run>.json``), except the differences listed with their
reasons in ``tests/data/run_configs/REGISTER.yaml``. The selection state of every recipe
(metric, stop epoch, warm-start files) is pinned, the reproduction job graph passes the warm
starts the runs recorded, and the register is the same list as
``configs/published_pair.json:retrained_register`` (one list, two readers).

Normalisation (both sides): the scratch root is spelled ``${OCTTTA_SCRATCH}``; the subtrees in
:data:`DROPPED` are removed; ``train.finetune_from`` is kept, with the historical run names
mapped to the release run directories; text the release does not carry is stored as
``sha256:<hex>`` and listed in the snapshot's ``scrubbed``.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported here, with the real environment: root_env() below swaps the OCTTTA_* roots for
# placeholders, and a first import of octtta.paths inside it would freeze them for the session.
import octtta.paths  # noqa: E402,F401
from octtta.config import get_config  # noqa: E402

SNAPSHOTS = ROOT / "tests" / "data" / "run_configs"
REGISTER = SNAPSHOTS / "REGISTER.yaml"
PUBLISHED_PAIR = ROOT / "configs" / "published_pair.json"

STAGES = ("sam_phase1", "sam_phase2", "cnn", "coteach")
RUNS = tuple(f"{lineage}_{stage}" for lineage in ("s33", "s34") for stage in STAGES)
#: The recipes whose configs the shipped pair carries: A = s33 co-teaching, B = s34 CNN.
MODEL_OF_RUN = {"s33_coteach": "A", "s34_cnn": "B"}

DROPPED = ("experiment", "runtime.run_dir", "runtime.out_dir", "runtime.ckpt_dir", "_source",
           "notes", "eval.select_ruler_resolved")
RENAMED = ("train.finetune_from",)

TOKEN = "${OCTTTA_SCRATCH}"
#: The roots the recipes interpolate, spelled the way octtta/paths.py derives them from the
#: scratch root, so a resolved recipe comes out already normalised on any machine.
ROOT_ENV = {"OCTTTA_SCRATCH": TOKEN,
            "OCTTTA_DATA": f"{TOKEN}/oct_tta_data",
            "OCTTTA_RUNS": f"{TOKEN}/oct_tta_runs",
            "OCTTTA_CKPT": f"{TOKEN}/oct_tta_ckpts"}

KINDS = {"CHANGED", "ADDED", "REMOVED"}
TAGS = {"no_reader", "guard_only", "argv", "live", "off_value", "server_overridden", "decision"}
SCANNED_TAGS = {"no_reader", "guard_only"}
ABSENT = "<absent>"

#: Local reproduction uses fixed epochs and writes last.pt (owner decisions 6-7).
SELECT_METRIC = "none"
#: ``train.stop_epoch`` per run; a run not listed has none (runs its last epoch).
STOP_EPOCH: dict[str, int] = {"s33_cnn": 25}
#: Warm starts, relative to the runs root: run -> (model A file, model B file).
WARM_STARTS = {
    "s33_sam_phase2": ("s33_sam_phase1/checkpoints/last.pt", None),
    "s34_sam_phase2": ("s34_sam_phase1/checkpoints/last.pt", None),
    "s33_coteach": ("s33_sam_phase2/checkpoints/last.pt", "s33_cnn/checkpoints/last.pt"),
    "s34_coteach": ("s34_sam_phase2/checkpoints/last.pt", "s34_cnn/checkpoints/last.pt"),
}
# -------------------------------------------------------------------------------- helpers

def flatten(tree, prefix: tuple = ()) -> dict:
    """Nested dicts -> ``{"a.b.c": leaf}``; lists and empty dicts are leaves."""
    out: dict = {}
    if isinstance(tree, dict) and (tree or not prefix):
        for k, v in tree.items():
            out.update(flatten(v, prefix + (str(k),)))
    else:
        out[".".join(prefix)] = tree
    return out


def covered(key: str, path: str) -> bool:
    return key == path or key.startswith(path + ".")


def json_safe(value):
    """What a JSON snapshot can hold: non-finite floats as strings, tuples as lists."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def same(a, b) -> bool:
    """Equality that does not let ``True == 1`` through (bools compare only with bools)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def drop(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for dotted in DROPPED:
        node, parts = out, dotted.split(".")
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return out


@contextmanager
def root_env():
    saved = {k: os.environ.get(k) for k in ROOT_ENV}
    os.environ.update(ROOT_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def normalised_recipe(rel: str) -> dict:
    """``get_config`` of a recipe, roots spelled from ``${OCTTTA_SCRATCH}``, flattened."""
    with root_env():
        cfg = get_config(str(ROOT / rel))
    return flatten(json_safe(drop(cfg)))


def load_snapshot(run: str) -> dict:
    return json.loads((SNAPSHOTS / f"{run}.json").read_text())


def load_register() -> dict:
    return yaml.safe_load(REGISTER.read_text()) or {}


def applies(entry: dict, run: str) -> bool:
    runs = entry.get("runs", "all")
    return runs == "all" or run in runs


def with_lineage(value, lineage: str):
    if isinstance(value, str):
        return value.replace("<L>", lineage)
    if isinstance(value, list):
        return [with_lineage(v, lineage) for v in value]
    if isinstance(value, dict):
        return {k: with_lineage(v, lineage) for k, v in value.items()}
    return value


def subtree(flat: dict, path: str):
    """The value at ``path`` rebuilt from a flat dict, or ABSENT."""
    if path in flat:
        return flat[path]
    below = {k[len(path) + 1:]: v for k, v in flat.items() if k.startswith(path + ".")}
    if not below:
        return ABSENT
    out: dict = {}
    for k, v in below.items():
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def at(path: str, value) -> dict:
    """``{path: value}`` flattened below ``path`` when the value is a non-empty dict."""
    if isinstance(value, dict) and value:
        return {f"{path}.{k}": v for k, v in flatten(value).items()}
    return {path: value}


def expected_recipe(snapshot: dict, run: str) -> tuple[dict, list[str]]:
    """The run's config with the register applied (what the recipe must resolve to), and
    every register entry that does not match the run it names."""
    lineage = snapshot["lineage"]
    snap = flatten(snapshot["config"])
    expected, problems = dict(snap), []
    for e in load_register().get("entries") or []:
        if not applies(e, run):
            continue
        path, kind = e["path"], e["kind"]
        here = subtree(snap, path)
        if kind in ("CHANGED", "REMOVED"):
            want = with_lineage(e.get("snapshot"), lineage)
            if here is ABSENT:
                problems.append(f"register {kind} {path}: the run has no such key")
            elif not same(here, want):
                problems.append(f"register {kind} {path}: the run recorded {here!r}, the "
                                f"register pins {want!r}")
            for k in [k for k in expected if covered(k, path)]:
                del expected[k]
        if kind in ("CHANGED", "ADDED"):
            if kind == "ADDED" and here is not ABSENT:
                problems.append(f"register ADDED {path}: the run has this key ({here!r})")
            expected.update(at(path, with_lineage(e.get("recipe"), lineage)))
    return expected, problems


def recipe_problems(run: str, snapshot: dict | None = None) -> list[str]:
    """Every difference between the run's recipe and its config that the register does not
    explain (empty = the recipe is the run). ``snapshot`` defaults to the committed one; the
    research-side gate passes a fresh one from the live run directory."""
    snapshot = snapshot if snapshot is not None else load_snapshot(run)
    recipe = normalised_recipe(snapshot["recipe"])
    problems = [f"{k}: unexpanded variable in {v!r}" for k, v in recipe.items()
                if isinstance(v, str) and "$" in v.replace(TOKEN, "")]
    expected, reg_problems = expected_recipe(snapshot, run)
    problems += reg_problems
    for k in sorted(set(expected) | set(recipe)):
        want, got = expected.get(k, ABSENT), recipe.get(k, ABSENT)
        if not same(want, got):
            problems.append(f"{k}: the recipe resolves to {got!r}, the run (with the register) "
                            f"to {want!r}")
    return problems


def register_agreement_problems(published_pair: dict) -> list[str]:
    """The retrained register's ``config.*`` paths of A and B == the normalisation plus every
    non-live register entry of the recipe that config comes from."""
    entries = load_register().get("entries") or []
    problems = []
    for run, model in MODEL_OF_RUN.items():
        want = {f"config.{p}" for p in DROPPED + RENAMED}
        want |= {f"config.{e['path']}" for e in entries
                 if applies(e, run) and e["tag"] != "live"}
        live = {f"config.{e['path']}" for e in entries if applies(e, run) and e["tag"] == "live"}
        got = {r["path"] for r in published_pair["retrained_register"][model]
               if r["path"].startswith("config.")}
        if want != got:
            problems.append(f"{model} ({run}): only in the recipe register "
                            f"{sorted(want - got)}; only in retrained_register "
                            f"{sorted(got - want)}")
        if live & got:
            problems.append(f"{model} ({run}): live differences in retrained_register: "
                            f"{sorted(live & got)}")
    return problems


def dry_run_warm_starts(lineage: str, tmp_path: Path) -> tuple[str, str]:
    """The (A, B) starts in the repro job graph, spelled from the runs root."""
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if k != "OCTTTA_EXPECT_COMMIT"}
    env.update({"OCTTTA_SCRATCH": str(tmp_path), "OCTTTA_RUNS": str(runs)})
    p = subprocess.run([sys.executable, "scripts/repro.py", "reproduce", "--lineage", lineage,
                        "--skip-build", "--dry"], capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert p.returncode == 0, p.stdout + p.stderr
    jobs = json.loads(p.stdout)
    job = next(j for j in jobs if j["name"] == f"{lineage}_coteach")
    assert job["depends_on"] == [f"{lineage}_sam_phase2", f"{lineage}_cnn"]
    argv = job["sbatch"]
    assert argv.count("--finetune-from") == argv.count("--finetune-from-b") == 1
    a = argv[argv.index("--finetune-from") + 1]
    b = argv[argv.index("--finetune-from-b") + 1]
    spell = lambda s: s.replace(str(runs), f"{TOKEN}/oct_tta_runs")  # noqa: E731
    return spell(a), spell(b)


# ---------------------------------------------------------------------------------- tests

def test_the_snapshots_are_the_eight_final_runs() -> None:
    assert sorted(p.stem for p in SNAPSHOTS.glob("*.json")) == sorted(RUNS)
    for run in RUNS:
        doc = load_snapshot(run)
        lineage, stage = run.split("_", 1)
        assert (doc["run"], doc["lineage"], doc["stage"]) == (run, lineage, stage)
        assert (ROOT / doc["recipe"]).is_file(), doc["recipe"]
        assert set(doc["dropped"]) <= set(DROPPED), doc["dropped"]
        assert re.fullmatch(r"[0-9a-f]{64}", doc["source_sha256"])
        flat = flatten(doc["config"])
        assert not [k for k in flat if any(covered(k, p) for p in DROPPED)]
        paths = [v for v in flat.values() if isinstance(v, str) and v.startswith("/")]
        assert not paths, f"{run}: absolute paths survived normalisation: {paths}"
        for path in doc["scrubbed"]:
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", flat[path[len("config."):]]), (run, path)


@pytest.mark.parametrize("run", RUNS)
def test_the_recipe_is_its_run_up_to_the_register(run: str) -> None:
    problems = recipe_problems(run)
    assert not problems, f"{run}:\n  " + "\n  ".join(problems)


def test_the_register_is_well_formed() -> None:
    reg = load_register()
    assert reg.get("schema") == "octtta/recipe_register/1"
    scans = reg.get("scans") or {}
    seen = set()
    for e in reg.get("entries") or []:
        where = f"{e.get('path')} ({e.get('kind')})"
        assert e["kind"] in KINDS and e["tag"] in TAGS, where
        assert e.get("why") and e.get("batch"), where
        runs = e.get("runs", "all")
        assert runs == "all" or (runs and set(runs) <= set(RUNS)), where
        for run in (RUNS if runs == "all" else runs):
            assert (e["path"], run) not in seen, f"{where}: registered twice for {run}"
            seen.add((e["path"], run))
        if e["kind"] == "REMOVED":
            assert e["tag"] in {"no_reader", "off_value", "server_overridden", "decision"}, where
            assert "snapshot" in e, where
        if e["kind"] == "ADDED":
            assert "recipe" in e and "snapshot" not in e, where
        if e["kind"] == "CHANGED":
            assert "recipe" in e and "snapshot" in e, where
        if e["tag"] in SCANNED_TAGS:
            assert e.get("scan") in scans, f"{where}: tag {e['tag']} needs a scan"
        if e["tag"] == "live":
            assert e.get("fix"), f"{where}: a live difference must name the batch that fixes it"
    used = {e.get("scan") for e in reg.get("entries") or []}
    assert set(scans) <= used, f"unused scans: {sorted(set(scans) - used)}"


def test_the_register_is_the_retrained_register() -> None:
    problems = register_agreement_problems(json.loads(PUBLISHED_PAIR.read_text()))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("run", RUNS)
def test_the_selection_pins(run: str) -> None:
    cfg = normalised_recipe(load_snapshot(run)["recipe"])
    assert cfg.get("eval.tune_select_metric") == SELECT_METRIC, run
    assert cfg.get("train.stop_epoch") == STOP_EPOCH.get(run), run
    a_file, _ = WARM_STARTS.get(run, (None, None))
    if run.endswith("sam_phase2"):
        assert cfg["train.finetune_from"] == f"{TOKEN}/oct_tta_runs/{a_file}", run
    elif not run.endswith("coteach"):
        assert cfg.get("train.finetune_from") is None, run


@pytest.mark.parametrize("lineage", ("s33", "s34"))
def test_the_repro_graph_passes_the_warm_starts_the_runs_recorded(
        lineage: str, tmp_path: Path) -> None:
    a, b = dry_run_warm_starts(lineage, tmp_path)
    want_a, want_b = WARM_STARTS[f"{lineage}_coteach"]
    assert (a, b) == (f"{TOKEN}/oct_tta_runs/{want_a}", f"{TOKEN}/oct_tta_runs/{want_b}")
    snap = load_snapshot(f"{lineage}_coteach")
    historical = (snap["config"]["train"]["finetune_from"], snap["warm_start_b"])
    assert (a, b) == tuple(p.replace("/best.pt", "/last.pt") for p in historical)
