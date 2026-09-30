"""Check inputs, run a configurable model, or reproduce the historical recipe.

``--dry`` prints the exact jobs without submitting or writing anything. Source
and configs may be edited in a checkout or an unpacked source archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from octtta import paths  # noqa: E402
from scripts.data import download  # noqa: E402
from octtta.config import get_config, load_config  # noqa: E402


@dataclass(frozen=True)
class BuildJob:
    name: str
    command: tuple[str, ...]
    check: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    cpus: int = 8
    memory: str = "32G"
    hours: int = 6
    array: str | None = None
    gpu: str | None = None
    direct_script: bool = False


def _python(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def _local_settings() -> dict:
    file = REPO_ROOT / "configs" / "local.yaml"
    if not file.is_file():
        return {}
    raw = yaml.safe_load(file.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configs/local.yaml must be a mapping")
    slurm = raw.get("slurm") or {}
    if not isinstance(slurm, dict):
        raise ValueError("configs/local.yaml slurm must be a mapping")
    _cpu_overrides(slurm)
    return raw


def _slurm() -> dict:
    return _local_settings().get("slurm") or {}


def _cpu_overrides(site: dict) -> dict:
    """Validated CPU-only Slurm fields; absent keys inherit the generic fields."""
    if "cpu" not in site:
        return {}
    cpu = site["cpu"]
    if not isinstance(cpu, dict):
        raise ValueError("slurm.cpu must be a mapping")
    unknown = sorted(set(cpu) - {"account", "partition", "qos"})
    if unknown:
        raise ValueError(f"slurm.cpu has unknown keys: {unknown}")
    for key, value in cpu.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"slurm.cpu.{key} must be a nonempty string or null")
    return cpu


def _activate() -> str | None:
    value = _local_settings().get("python_activate")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("python_activate must be a shell command or script path")
    return value


def build_jobs(*, public_only: bool = False) -> tuple[BuildJob, ...]:
    """The five real data stages; public-only stops before licensed data."""
    public = BuildJob(
        "public", _python("scripts/data/build_public_pool.py"),
        (),
        cpus=8, memory="64G", hours=18)
    if public_only:
        return (public,)
    return (public,
            BuildJob("labelled_aireadi", _python("scripts/data/build_aireadi_pool.py"),
                     (),
                     depends_on=("public",), cpus=8, memory="64G", hours=18),
            BuildJob("manifest", _python("scripts/data/extract_aireadi_frames.py", "manifest"),
                     depends_on=("labelled_aireadi",), cpus=4, memory="16G", hours=2),
            BuildJob("extract", _python("scripts/data/extract_aireadi_frames.py", "extract"),
                     depends_on=("manifest",), cpus=8, memory="32G", hours=18,
                     array="0-31"),
            BuildJob("unlabelled", _python("scripts/data/stage_unlabeled_pool.py"),
                     (),
                     depends_on=("extract",), cpus=8, memory="32G", hours=6))


def train_jobs(*, lineage: str = "both") -> tuple[BuildJob, ...]:
    """Eight-stage recipe with run names and warm starts from resolved configs."""
    if lineage not in {"s33", "s34", "both"}:
        raise ValueError("lineage must be s33, s34, or both")
    selected = ("s33", "s34") if lineage == "both" else (lineage,)
    gpu = str(_slurm().get("gpu") or "h100")
    jobs: list[BuildJob] = []
    specs: dict[tuple[str, str], tuple[str, Path, Path]] = {}
    from octtta.train_data import resolve_run_dir
    for family in selected:
        prefix = (Path("configs/reproduction/model_a") if family == "s33"
                  else Path("configs/reproduction/model_b"))
        for stage in ("sam_phase1", "sam_phase2", "cnn", "coteach"):
            path = prefix / f"{stage}.yaml"
            cfg = get_config(path)
            run_dir = resolve_run_dir(cfg)
            name = run_dir.name
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("experiment.id must contain only letters, digits, '_' or '-'")
            specs[(family, stage)] = (name, path, run_dir)
    if len({item[0] for item in specs.values()}) != len(specs):
        raise ValueError("recipe experiment.id values must be unique")
    for family in selected:
        for stage in ("sam_phase1", "sam_phase2", "cnn", "coteach"):
            name, config, _ = specs[(family, stage)]
            deps = ((specs[(family, "sam_phase1")][0],) if stage == "sam_phase2" else
                    (specs[(family, "sam_phase2")][0], specs[(family, "cnn")][0])
                    if stage == "coteach" else ())
            args = ["bash", "scripts/train/train.sbatch", str(config), "--gpus", "1"]
            if stage == "sam_phase2":
                args += ["--finetune-from",
                         str(specs[(family, "sam_phase1")][2] / "checkpoints/last.pt")]
            if stage == "coteach":
                args += ["--finetune-from",
                         str(specs[(family, "sam_phase2")][2] / "checkpoints/last.pt"),
                         "--finetune-from-b",
                         str(specs[(family, "cnn")][2] / "checkpoints/last.pt")]
            jobs.append(BuildJob(name, tuple(args),
                                 depends_on=deps, cpus=16, memory="96G", hours=5, gpu=gpu,
                                 direct_script=True))
    if lineage == "both":
        a = specs[("s33", "coteach")]
        b = specs[("s34", "coteach")]
        jobs.append(BuildJob(
            "export", _python("scripts/export/export_pair.py", "--a",
                              str(a[2] / "checkpoints/last.pt"),
                              "--b", str(b[2] / "checkpoints/last_b.pt"),
                              "--out", str(paths.RUNS_DIR / "export_pair"), "--final-recipe"),
            depends_on=(a[0], b[0]),
            cpus=4, memory="16G", hours=2))
    return tuple(jobs)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(("git", "-C", str(REPO_ROOT), *args),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_digest(path: str) -> str:
    resolved = load_config(path)
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def checked_commit(*, dry: bool) -> str:
    return _git("rev-parse", "HEAD")


def _job_shell(job: BuildJob, commit: str) -> str:
    q = shlex.quote
    activation = _activate()
    cmds = ["set -euo pipefail", f"cd {q(str(REPO_ROOT))}",
            f"export OCTTTA_REPO={q(str(REPO_ROOT))}"]
    if activation:
        script = Path(activation)
        script = script if script.is_absolute() else REPO_ROOT / script
        cmds.append(f"source {q(str(script))}" if script.is_file() else activation)
    cmds.append(shlex.join(job.command))
    if job.check:
        cmds.append(shlex.join(job.check))
    return "; ".join(cmds)


def sbatch_argv(job: BuildJob, commit: str, dependency_ids: tuple[str, ...] = (),
                log_dir: Path | None = None) -> list[str]:
    site = _slurm()
    cpu = _cpu_overrides(site) if not job.gpu else {}
    hours = site.get("wall_hours") or job.hours
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("slurm.wall_hours must be a positive integer")
    argv = ["sbatch", "--parsable", f"--job-name=octtta-{job.name}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={job.cpus}",
            f"--mem={job.memory}", f"--time={hours:02d}:00:00"]
    log_dir = Path(log_dir or paths.RUNS_DIR / "slurm")
    argv.extend((f"--output={log_dir}/%x-%j.out", f"--error={log_dir}/%x-%j.err"))
    exports = ["ALL"]
    if job.direct_script:
        exports += [f"OCTTTA_REPO={REPO_ROOT}",
                    f"OCTTTA_EXPECT_CONFIG_SHA256={_config_digest(job.command[2])}"]
        activation = _activate()
        if activation:
            script = Path(activation)
            script = script if script.is_absolute() else REPO_ROOT / script
            key = "OCTTTA_ENV_ACTIVATE" if script.is_file() else "OCTTTA_REPRO_ACTIVATE_CMD"
            exports.append(f"{key}={script if script.is_file() else activation}")
    argv.append("--export=" + ",".join(exports))
    for key in ("account", "partition", "qos"):
        value = cpu[key] if key in cpu else site.get(key)
        if value:
            argv.append(f"--{key}={value}")
    if job.gpu:
        argv.extend((f"--gpus={job.gpu}:1", "--signal=B:USR1@60", "--requeue"))
    if job.array:
        argv.append(f"--array={job.array}")
    if dependency_ids:
        argv.append("--dependency=afterok:" + ":".join(dependency_ids))
    if job.direct_script:
        if job.command[:2] != ("bash", "scripts/train/train.sbatch"):
            raise ValueError("direct Slurm jobs must run the pinned train.sbatch")
        argv.extend((str(REPO_ROOT / "scripts/train/train.sbatch"), *job.command[2:]))
    else:
        argv.extend(("--wrap", "bash -lc " + shlex.quote(_job_shell(job, commit))))
    return argv


def _record_jobs(records: list[dict], run_root: Path) -> None:
    dest = Path(run_root) / "repro_jobs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(dest.read_text()) if dest.is_file() else []
    if not isinstance(old, list):
        raise ValueError("repro_jobs.json must contain a list")
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps([*old, *records], indent=2) + "\n")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _run_jobs(jobs: tuple[BuildJob, ...], *, dry: bool, local: bool,
              run_root: Path | None = None) -> list[dict]:
    if dry and local:
        raise ValueError("--dry and --local are mutually exclusive")
    commit = checked_commit(dry=dry)
    if dry:
        preview = [{"name": j.name, "depends_on": list(j.depends_on),
                    "sbatch": sbatch_argv(j, commit, tuple(f"<{d}>" for d in j.depends_on))}
                   for j in jobs]
        print(json.dumps(preview, indent=2))
        return preview
    run_root = Path(run_root or paths.RUNS_DIR)
    if not local:
        (run_root / "slurm").mkdir(parents=True, exist_ok=True)
    ids: dict[str, str] = {}
    records = []
    for job in jobs:
        if local:
            loops = range(32) if job.array else (None,)
            for shard in loops:
                env = {**os.environ, "OCTTTA_REPO": str(REPO_ROOT)}
                if shard is not None:
                    env["SLURM_ARRAY_TASK_ID"] = str(shard)
                if _activate():
                    subprocess.run(("bash", "-lc", _job_shell(BuildJob(job.name, job.command),
                                                               commit)), cwd=REPO_ROOT,
                                   env=env, check=True)
                else:
                    subprocess.run(job.command, cwd=REPO_ROOT, env=env, check=True)
            if job.check:
                if _activate():
                    subprocess.run(("bash", "-lc", _job_shell(BuildJob(job.name, job.check),
                                                               commit)), cwd=REPO_ROOT,
                                   env=env, check=True)
                else:
                    subprocess.run(job.check, cwd=REPO_ROOT, env=env, check=True)
            job_id = f"local-{job.name}"
        else:
            deps = tuple(ids[d] for d in job.depends_on)
            output = subprocess.check_output(sbatch_argv(job, commit, deps,
                                                         run_root / "slurm"), text=True).strip()
            job_id = output.split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", job_id):
                raise RuntimeError("sbatch returned no numeric job ID")
        ids[job.name] = job_id
        record = {"name": job.name, "job_id": job_id, "commit": commit,
                  "depends_on": list(job.depends_on), "array": job.array}
        records.append(record)
        _record_jobs([record], run_root)
        print(f"{job.name}: {job_id}")
    return records


def build(*, public_only: bool = False, dry: bool = False,
          local: bool = False, run_root: Path | None = None) -> list[dict]:
    return _run_jobs(build_jobs(public_only=public_only), dry=dry, local=local,
                     run_root=run_root)


def train(*, lineage: str = "both", dry: bool = False, local: bool = False,
          run_root: Path | None = None) -> list[dict]:
    return _run_jobs(train_jobs(lineage=lineage), dry=dry, local=local,
                     run_root=run_root)


def run_job(config: Path) -> BuildJob:
    """One training run from any valid YAML, with no historical warm start."""
    config = Path(config)
    cfg = get_config(config)
    name = str(cfg.get("experiment", {}).get("id") or config.stem)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("experiment.id must contain only letters, digits, '_' or '-'")
    return BuildJob(name, ("bash", "scripts/train/train.sbatch", str(config), "--gpus", "1"),
                    cpus=16, memory="96G", hours=5,
                    gpu=str(_slurm().get("gpu") or "h100"), direct_script=True)


def run(config: Path, *, dry: bool = False, local: bool = False) -> list[dict]:
    if not dry:
        from scripts.repro_checks import check_framework
        report = check_framework(config)
        if report["status"] == "FAIL":
            failed = [row["name"] for row in report["rows"] if row["status"] == "FAIL"]
            raise ValueError(f"run inputs invalid or missing: {', '.join(failed)}; run 'check framework --config' for details")
    return _run_jobs((run_job(config),), dry=dry, local=local)


def reproduce(*, lineage: str = "both", skip_build: bool = False,
              dry: bool = False, local: bool = False) -> list[dict]:
    from scripts.repro_checks import check_all
    advice = check_all(_local_settings())
    for row in advice["rows"]:
        if row["status"] != "PASS":
            print(f"[check] {row['name']}: {row.get('reason', '')} {row.get('next', '')}",
                  file=sys.stderr)
    jobs = train_jobs(lineage=lineage)
    if not skip_build:
        build_stages = build_jobs()
        jobs = build_stages + tuple(replace(job, depends_on=(*job.depends_on, "unlabelled"))
                                    if job.direct_script and not job.depends_on else job
                                    for job in jobs)
    return _run_jobs(jobs, dry=dry, local=local)


def _weights(origin: str) -> tuple[Path, Path]:
    if origin == "mine":
        root = paths.RUNS_DIR / "export_pair"
    else:
        from scripts import repro_checks
        root = repro_checks.published_weights_root(_local_settings())
    return root / "model.pt", root / "model_b.pt"


def _single(name: str, command: tuple[str, ...], *, dry: bool, local: bool,
            gpu: bool = False, hours: int = 2) -> list[dict]:
    gpu_type = str(_slurm().get("gpu") or "h100") if gpu else None
    return _run_jobs((BuildJob(name, command, cpus=8 if gpu else 4,
                               memory="96G" if gpu else "16G", hours=hours,
                               gpu=gpu_type),), dry=dry, local=local)


def display_check(report: dict, *, as_json: bool = False) -> None:
    """Print readiness advice with the command that resolves each finding."""
    if as_json:
        print(json.dumps(report, sort_keys=True))
        return
    print(f"Reproduction check: {report['status']}")
    for row in report.get("rows", []):
        detail = row.get("reason", "")
        print(f"[{row['status']}] {row['name']}" + (f": {detail}" if detail else ""))
        if row.get("next"):
            print(f"  Next: {row['next']}")
    if report.get("advisory"):
        print("Reproduction differences are advice; you can continue with your own configuration.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--public-only", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("target", nargs="?", choices=("framework", "all", "env", "data", "pools", "runs"),
                   default="all")
    c.add_argument("--config", type=Path, default=Path("configs/framework/cnn.yaml"))
    c.add_argument("--json", action="store_true", help="print the structured check report")
    c.add_argument("--reproduce", action="store_true",
                   help="show historical recipe checks as advice")
    c.add_argument("--gpu", action="store_true")
    c.add_argument("--only")
    c.add_argument("--print-digests", action="store_true")
    c.add_argument("--data-root", type=Path, default=paths.DATA_DIR)
    c.add_argument("--runs-root", type=Path, default=paths.RUNS_DIR)
    d = sub.add_parser("download")
    d.add_argument("group", nargs="?", choices=("public", "official", "sam", "all"),
                   default="all")
    t = sub.add_parser("reproduce")
    t.add_argument("--lineage", choices=("s33", "s34", "both"), default="both")
    t.add_argument("--skip-build", action="store_true", help="use existing data pools")
    single = sub.add_parser("run")
    single.add_argument("--config", required=True, type=Path)
    export = sub.add_parser("export")
    export.add_argument("--a", required=True, type=Path)
    export.add_argument("--b", type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--final-recipe", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--weights", choices=("published", "mine"), default="published")
    infer = sub.add_parser("infer")
    infer.add_argument("--images", required=True, type=Path)
    infer.add_argument("--out", required=True, type=Path)
    infer.add_argument("--weights", choices=("published", "mine"), default="published")
    for parser in (b, d, t, single, export, score, infer):
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--dry", action="store_true")
        mode.add_argument("--local", action="store_true")
    args = p.parse_args(argv)
    if args.command == "build":
        build(public_only=args.public_only, dry=args.dry, local=args.local)
    elif args.command == "reproduce":
        reproduce(lineage=args.lineage, skip_build=args.skip_build,
                  dry=args.dry, local=args.local)
    elif args.command == "run":
        run(args.config, dry=args.dry, local=args.local)
    elif args.command == "check":
        from scripts import repro_checks
        if args.target == "all":
            display_check(repro_checks.check_all(_local_settings()), as_json=args.json)
            return 0
        if args.target == "framework":
            report = repro_checks.check_framework(args.config, reproduce=args.reproduce)
            display_check(report, as_json=args.json)
            return 1 if report["status"] == "FAIL" else 0
        check_args = [args.target]
        if args.target == "env" and args.gpu:
            check_args.append("--gpu")
        if args.target in {"data", "pools"}:
            check_args += ["--data-root", str(args.data_root)]
        if args.target == "runs":
            check_args += ["--runs-root", str(args.runs_root)]
        if args.only:
            check_args += ["--only", args.only]
        if args.target == "pools" and args.print_digests:
            check_args += ["--print-digests"]
        return repro_checks.main(check_args, local=_local_settings())
    elif args.command == "download":
        if args.dry:
            print(json.dumps({"group": args.group, "command": "scripts/data/download.py",
                              "pinned": True}))
        else:
            checked_commit(dry=False)
            result = download.main([args.group])
            if result:
                return result
            from scripts import repro_checks
            only = ("public,official,sam" if args.group == "all" else args.group)
            return repro_checks.main(["data", "--only", only],
                                     local=_local_settings())
    elif args.command == "export":
        command = ["scripts/export/export_pair.py", "--a", str(args.a), "--out", str(args.out)]
        if args.b:
            command += ["--b", str(args.b)]
        if args.final_recipe:
            command.append("--final-recipe")
        _single("export", _python(*command), dry=args.dry, local=args.local)
    elif args.command == "score":
        a, b = _weights(args.weights)
        _single("score", _python("scripts/repro_score.py", "score", "--a", str(a),
                                 "--b", str(b), "--origin", args.weights),
                dry=args.dry, local=args.local, gpu=True, hours=6)
    elif args.command == "infer":
        a, b = _weights(args.weights)
        _single("infer", _python("scripts/repro_score.py", "infer", "--a", str(a),
                                 "--b", str(b), "--images", str(args.images),
                                 "--out", str(args.out)),
                dry=args.dry, local=args.local, gpu=True, hours=6)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        if str(exc).startswith("slurm.cpu"):
            raise SystemExit(f"invalid local Slurm CPU settings: {exc}") from None
        raise
