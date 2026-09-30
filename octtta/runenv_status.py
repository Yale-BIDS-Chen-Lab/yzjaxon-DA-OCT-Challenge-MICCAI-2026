"""What the two hours on the evaluation server did to this run, as one machine-readable file.

``$STATE_DIR/run_status.json`` is written by the entry-point shells and by this module's
CLI (``python -m octtta.runenv_status set k=v``), under a lock, one key per stage:
``finetune``, ``refit``, ``gate``, ``select``, ``inference``, ``ladder`` and the rest of
:data:`SCHEMA`. Keys are closed and validated on write, because a status file is read by
exactly one consumer and a key it does not know is a silent no-op. :func:`mode` turns what
was written into the single ``SUBMISSION MODE:`` line and :func:`nonnominal` its reasons."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import time
import sys
from pathlib import Path
from typing import Any, Sequence

#: The verdict file, inside ``$STATE_DIR``; ``infer_summary.json`` beside it is the detail.
FILENAME = "run_status.json"

#: Held across the read-modify-write, on its own inode so it survives a file replacement.
LOCKNAME = "run_status.lock"

#: ``inference``: what the mask-writing stage actually did, from ``not_run`` to ``timeout``.
INFERENCE_STATES = ("not_run", "ok", "oom_retry_patch1", "oom_abort_baked_retry",
                    "timeout", "crashed")

INFERENCE_ATTEMPTS = ("published", "baked_retry", "in_process")

MODEL_B_STATES = ("absent", "loaded", "dropped_oom", "dropped_error")


#: ``ladder``: where the budget ladder LANDED, i.e. which plan wrote the masks on disk.
#: ``top`` is the packaged plan; every other value is a different computation from it.
LADDER_STATES = (
    "top",
    "reduce_sliding_window_column_overlap",
    "drop_second_model_on_large_frames",
    "drop_second_model_on_every_frame",
    "reduce_sliding_window_row_overlap",
    "sliding_window_whole_image",
    "drop_monotonic_column_repair_and_islands",
    "plain_argmax_clip",
    "over_budget",
)

DEGRADE_LOG_SCHEMA = 3


def ladder_from_summary(summary: dict) -> dict:
    """Read the highest degradation rung and affected image count from a schema-3 summary."""
    entries = [e for e in (summary.get("degrade_log") or []) if isinstance(e, dict)]
    for entry in entries:
        if int(entry.get("schema", 0)) != DEGRADE_LOG_SCHEMA:
            raise ValueError("infer_summary degradation schema is not 3")
    if summary.get("start_rung") is not None and not entries:
        if int(summary.get("degrade_log_schema", 0)) != DEGRADE_LOG_SCHEMA:
            raise ValueError("infer_summary degradation schema is not 3")

    rungs: list[int] = []
    for entry in entries:
        try:
            rungs.append(int(entry["rung"]))
        except (KeyError, TypeError, ValueError):
            continue
    start = summary.get("start_rung")
    started_below = False
    if start is not None:
        try:
            rungs.append(int(start))
            started_below = True
        except (TypeError, ValueError):
            pass
    try:
        n_images = max(0, int(summary.get("n_images") or 0))
    except (TypeError, ValueError):
        n_images = 0
    if not rungs:
        return {"ladder": "over_budget" if summary.get("over_budget") else "top",
                "n_images_below_top": 0}
    index = min(max(max(rungs), 1), len(LADDER_STATES) - 2)
    firsts = []
    for entry in entries:
        try:
            firsts.append(int(entry["after_image"]))
        except (KeyError, TypeError, ValueError):
            continue
    first = 0 if started_below or not firsts else min(firsts)
    return {"ladder": LADDER_STATES[index],
            "n_images_below_top": max(0, n_images - first)}


#: What may be written and what to: a tuple is a closed set, a type is any value of that type.
SCHEMA: dict[str, Any] = {
    "inference": INFERENCE_STATES,
    "inference_attempt": INFERENCE_ATTEMPTS,
    "model_b": MODEL_B_STATES,
    #: Where the budget ladder landed, derived from ``infer_summary.json``.
    "ladder": LADDER_STATES,
    "n_images_below_top": int,
    "n_images": int,
    "n_written": int,
    "n_failed": int,
    "n_backfilled": int,
    "n_oom_retried": int,
    "inference_seconds": float,
    "inference_deadline_seconds": float,
    "finetune": str,
    "refit": str,
    "disk_gate": str,
    "ram": str,
    "select": str,
    "oom_retries": int,
    "published_kind": ("none", "finetune", "refit"),
    "published_generation": int,
    "published_bytes": int,
}


#: Placeholder for a key nobody wrote. A REASON, not a pass: "said ok" and "never spoke" differ.
UNSET = "<unset>"

#: Nominal values per key. ``oom_retry_batch_half`` is NOT nominal: half the rehearsed batch.
NOMINAL_FINETUNE = frozenset({"ok"})

#: ``refit``: the same, plus the two ways the full refit is legitimately not run.
NOMINAL_REFIT = frozenset({"ok", "not_needed",
                           "not_run:gate_kept_baked"})

NOMINAL_INFERENCE = frozenset({"ok"})

NOMINAL_ATTEMPT = frozenset({"published"})

NOMINAL_LADDER = frozenset({"top"})

def nonnominal(status: dict, *, finetune_requested: bool, model_b_packaged: bool,
               engine_ran: bool = True) -> list[str]:
    """Every status this run wrote that is not the outcome the package was measured with.
    Returned as ``["finetune=skipped_disk", "inference=timeout", ...]`` in a fixed order."""
    out: list[str] = []

    def value(key: str) -> str:
        got = status.get(key, UNSET)
        return UNSET if got is None else str(got)

    def judge(key: str, is_nominal, *, expected_present: bool) -> None:
        got = value(key)
        if got == UNSET:
            if expected_present:
                out.append(f"{key}={UNSET}")
            return
        if not is_nominal(got):
            out.append(f"{key}={got}")

        # When the fine-tune was not ordered the whole training half says ``not_run:<why>``.
    def not_run(got: str) -> bool:
        return got.split(":")[0] == "not_run"

    judge("finetune", NOMINAL_FINETUNE.__contains__ if finetune_requested else not_run,
          expected_present=finetune_requested)
    judge("refit", NOMINAL_REFIT.__contains__ if finetune_requested else not_run,
          expected_present=finetune_requested)
    judge("published_kind",
          ("finetune", "refit").__contains__ if finetune_requested else ("none",).__contains__,
          expected_present=finetune_requested)
    judge("inference", NOMINAL_INFERENCE.__contains__, expected_present=engine_ran)
        # Asked for only when the stage claims a clean ``ok``: an aborted run writes no summary.
    judge("ladder", NOMINAL_LADDER.__contains__,
          expected_present=value("inference") == "ok")
    judge("inference_attempt", NOMINAL_ATTEMPT.__contains__,
          expected_present=value("inference") == "ok")
    judge("model_b", ("loaded",).__contains__ if model_b_packaged else ("absent",).__contains__,
          expected_present=value("inference") == "ok")
    if value("inference") == "ok":
        got = value("n_failed")
        if got == UNSET:
            out.append(f"n_failed={UNSET}")
        elif int(float(got)) != 0:
            out.append(f"n_failed={int(float(got))}")
    return out


def submission_mode(status: dict, *, engine_ran: bool, finetune_requested: bool,
                    model_b_packaged: bool, n_images: int, n_backfilled: int,
                    backfill_known: bool = True) -> str:
    """The whole ``SUBMISSION MODE:`` value, built from what was WRITTEN.
    The three shapes are preserved to the character: the graders' log is grepped for them."""
    reasons: list[str] = []
    mask_reason = False
    if not engine_ran:
        reasons.append("constant-mask")
        mask_reason = True
    elif n_images and n_backfilled >= n_images:
        # Every mask is a constant: the engine produced nothing, so this is not "full".
        reasons.append(f"all-{n_backfilled}-masks-backfilled")
        mask_reason = True
    reasons += nonnominal(status, finetune_requested=finetune_requested,
                          model_b_packaged=model_b_packaged, engine_ran=engine_ran)
    if not reasons:
        if n_backfilled:
            return f"full+backfill({n_backfilled})"
        if not backfill_known:
            return "full+backfill(unknown)"
        return "full"
    if not mask_reason:
        if n_backfilled:
            reasons.append(f"backfill={n_backfilled}")
        elif not backfill_known:
            reasons.append("backfill=unknown")
    return "degraded(" + ",".join(reasons) + ")"


def mode_reasons(mode: str) -> list[str] | None:
    """The reason list inside ``degraded(...)``; ``[]`` for ``full*`` and ``None`` otherwise."""
    mode = mode.strip()
    if mode.startswith("full"):
        return []
    if mode.startswith("degraded(") and mode.endswith(")"):
        body = mode[len("degraded("):-1]
        return [r for r in body.split(",") if r]
    return None

#: The stdout line the final mode is built next to, so a log can be grepped for it exactly.
STDOUT_PREFIX = "RESOURCE_STATUS"

#: What ONE stage prints; a different prefix from ``STDOUT_PREFIX``, which is printed once.
SET_PREFIX = "STATUS_SET"


def default_state_dir() -> Path:
    """``$OCTTTA_STATE_DIR``, else the shells' own default: one derivation, not three."""
    env = os.environ.get("OCTTTA_STATE_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "octtta_state"


def status_path(state_dir: str | Path | None = None) -> Path:
    return (Path(state_dir) if state_dir is not None else default_state_dir()) / FILENAME


def coerce(key: str, value: Any) -> Any:
    """Validate one field against :data:`SCHEMA`, raising on anything that is not legal."""
    if key not in SCHEMA:
        raise ValueError(f"unknown status key {key!r}; the legal ones are "
                         f"{sorted(SCHEMA)} (add it to octtta.runenv_status.SCHEMA "
                         f"together with the consumer that reads it)")
    spec = SCHEMA[key]
    if isinstance(spec, tuple):
        if value not in spec:
            raise ValueError(f"status key {key!r} may only be one of {list(spec)}, "
                             f"got {value!r}")
        return value
    if spec is int:
        return int(value)
    if spec is float:
        return float(value)
    return str(value)


def read(state_dir: str | Path | None = None) -> dict:
    """Everything written so far; ``{}`` when nothing was, or when the file is unreadable."""
    path = status_path(state_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except Exception as exc:                                              # noqa: BLE001
        print(f"!! [status] {path} is unreadable ({type(exc).__name__}: {exc}); "
              "treating it as empty", flush=True)
        return {}
    return dict(data) if isinstance(data, dict) else {}


def update(state_dir: str | Path | None = None, **fields: Any) -> dict:
    """Merge ``fields`` into the status file, under a lock, and return the whole of it."""
    clean = {k: coerce(k, v) for k, v in fields.items()}
    path = status_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / LOCKNAME
    lfd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lfd, fcntl.LOCK_EX)
        current = read(path.parent)
        current.update(clean)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w") as fh:
            json.dump(current, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        fcntl.flock(lfd, fcntl.LOCK_UN)
        os.close(lfd)
    return current


def record(state_dir: str | Path | None = None, **fields: Any) -> dict | None:
    """:func:`update`, but a failure to write costs the status line and nothing else."""
    try:
        return update(state_dir, **fields)
    except Exception as exc:                                              # noqa: BLE001
        print(f"!! [status] could not record {fields!r}: {type(exc).__name__}: {exc}",
              flush=True)
        return None


def line(payload: dict | None = None, *, prefix: str = STDOUT_PREFIX,
         state_dir: str | Path | None = None) -> str:
    """The one-line stdout form: ``RESOURCE_STATUS {"inference": "ok", ...}``."""
    data = read(state_dir) if payload is None else payload
    return f"{prefix} {json.dumps(data, sort_keys=True)}"


def parse_line(text: str, *, prefix: str = STDOUT_PREFIX) -> dict | None:
    """The inverse of :func:`line` for one log line, or ``None`` when it is not one."""
    head = f"{prefix} "
    if not text.startswith(head):
        return None
    try:
        data = json.loads(text[len(head):])
    except Exception:                                                     # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _parse_assignments(items: Sequence[str]) -> dict:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = coerce(key.strip(), value.strip())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m octtta.runenv_status set k=v ... | show | path``, the shells' door."""
    ap = argparse.ArgumentParser(prog="python -m octtta.runenv_status",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("set", "show", "path"))
    ap.add_argument("assignments", nargs="*", help="key=value (action 'set')")
    ap.add_argument("--state-dir", default=None,
                    help="default: $OCTTTA_STATE_DIR, else ${TMPDIR:-/tmp}/octtta_state")
    ap.add_argument("--prefix", choices=(STDOUT_PREFIX, SET_PREFIX), default=SET_PREFIX,
                    help=f"stdout prefix for 'show' (default {SET_PREFIX})")
    # ``parse_known_args``: with a ``nargs="*"`` positional, argparse drops assignments
    # that follow an option.
    args, extra = ap.parse_known_args(argv)
    stray = [item for item in extra if "=" not in item or item.startswith("-")]
    if stray:
        ap.error(f"unrecognized arguments: {' '.join(stray)}")
    args.assignments = list(args.assignments) + [item for item in extra if item not in stray]

    if args.action == "path":
        print(status_path(args.state_dir))
        return 0
    if args.action == "show":
        print(line(state_dir=args.state_dir, prefix=args.prefix))
        return 0
    if not args.assignments:
        ap.error("set needs at least one key=value")
    try:
        fields = _parse_assignments(args.assignments)
    except ValueError as exc:
        print(f"!! [status] {exc}", flush=True)
        return 2
    print(line(update(args.state_dir, **fields), prefix=SET_PREFIX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


GIB = 1 << 30

ENVELOPE_FILENAME = "resource_envelope.jsonl"

ENVELOPE_ENV = "OCTTTA_RESOURCE_ENVELOPE"

PAYLOAD_FACTOR_FULL = 4

PAYLOAD_FACTOR_WEIGHTS_ONLY = 1

DISK_MARGIN_BYTES = 2 * GIB

def finetune_disk_requirement(baked_a_bytes: int, baked_b_bytes: int = 0, *,
                              payload_factor: int = PAYLOAD_FACTOR_FULL,
                              margin_bytes: int = DISK_MARGIN_BYTES) -> dict:
    """Free bytes ``$STATE_DIR`` must have before the fine-tune may start.
    ``4*A + 3*B`` is the documented transient peak, plus slack for everything in the state
    directory that is not a checkpoint."""
    a = int(baked_a_bytes) * int(payload_factor)
    b = int(baked_b_bytes) * int(payload_factor)
    return {
        "baked_a_bytes": int(baked_a_bytes),
        "baked_b_bytes": int(baked_b_bytes),
        "payload_factor": int(payload_factor),
        "checkpoint_a_bytes": a,
        "checkpoint_b_bytes": b,
        "margin_bytes": int(margin_bytes),
        "formula": "4*A + 3*B + margin",
        "required_bytes": 4 * a + 3 * b + int(margin_bytes),
    }


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def meminfo() -> dict:
    """``MemTotal``/``MemAvailable`` in bytes, ``None`` where the kernel does not say."""
    out: dict = {"MemTotal": None, "MemAvailable": None, "SwapTotal": None}
    raw = _read("/proc/meminfo")
    if raw is None:
        return out
    for line in raw.splitlines():
        key, _, rest = line.partition(":")
        if key in out:
            parts = rest.split()
            if parts and parts[0].isdigit():
                # /proc/meminfo is in kB (1024 bytes), always, on every arch Linux ships.
                out[key] = int(parts[0]) * 1024
    return out


def cgroup_memory() -> dict:
    """The memory wall the PROCESS hits, v2 first then v1; all ``None`` outside a cgroup.
    ``peak`` is the one that matters after the fact; ``current`` only ever says "not yet"."""
    out = {"version": None, "limit_bytes": None, "current_bytes": None,
           "peak_bytes": None, "oom_kill": None, "failcnt": None}
    if Path("/sys/fs/cgroup/memory.current").exists():
        out["version"] = "v2"
        out["limit_bytes"] = _read_int("/sys/fs/cgroup/memory.max")
        out["current_bytes"] = _read_int("/sys/fs/cgroup/memory.current")
        out["peak_bytes"] = _read_int("/sys/fs/cgroup/memory.peak")
        events = _read("/sys/fs/cgroup/memory.events") or ""
        for line in events.splitlines():
            name, _, value = line.partition(" ")
            if name == "oom_kill" and value.strip().isdigit():
                out["oom_kill"] = int(value)
    elif Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").exists():
        out["version"] = "v1"
        out["limit_bytes"] = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        out["current_bytes"] = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        out["peak_bytes"] = _read_int("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        out["failcnt"] = _read_int("/sys/fs/cgroup/memory/memory.failcnt")
    return out


def rss() -> dict:
    """Peak RSS of this process and of everything it has waited on, in bytes.
    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; this converts for Linux."""
    return {
        "self_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "children_max_bytes":
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }


def append_envelope(path: str | os.PathLike | None, record: dict, *,
                    echo: bool = True) -> None:
    """Append one JSON line and print it; only the file survives to be parsed."""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
    if echo:
        print(f"RESOURCE_ENVELOPE {line}", flush=True)
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"!! [resources] could not append to {path}: {exc}", file=sys.stderr,
              flush=True)


def record_event(event: str, **fields) -> dict:
    """A one-line envelope entry from library code: no census, no subprocess.
    Used to put the RSS on either side of a checkpoint load."""
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": f"event:{event}",
              "pid": os.getpid(), "rss": rss(), **fields}
    append_envelope(os.environ.get(ENVELOPE_ENV), record)
    return record
