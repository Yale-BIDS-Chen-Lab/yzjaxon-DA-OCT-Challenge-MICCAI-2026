"""Plumbing shared by the training and inference entry points.

Preemption: :func:`install_preemption_handler` writes one checkpoint and calls
``os._exit``, because a grace period of tens of seconds is not enough to drain a
dataloader; the write is atomic. Checkpoints carry their config."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "build_optimizer",
    "build_scheduler",
    "set_scheduler_step",
    "CheckpointManager",
    "install_preemption_handler",
    "preempted",
    "load_inference_model",
    "count_parameters",
    "cpu_budget",
    "WorkerPool",
    "flatten_config",
    "config_fingerprint",
    "config_changes",
]

#: Exit status after a preemption checkpoint. Zero, so the job requeues instead of failing.
EXIT_PREEMPTED = 0

#: Exit status when the run died of ``torch.cuda.OutOfMemoryError``, so the caller can retry.
EXIT_CUDA_OOM = 17

#: Dropped from a checkpoint written under ``checkpoint.weights_only``: only a resume reads them.
WEIGHTS_ONLY_DROP_KEYS = ("optimizer", "scheduler", "ema_state")


def set_seed(seed: int, deterministic: bool = True,
             cudnn_benchmark: bool = False) -> None:
    """Seed every RNG for the single training process."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Build from the ``train.optimizer`` block.
    Norm and bias parameters are excluded from weight decay."""
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or pname.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    raise ValueError(f"unknown optimizer {name!r}; valid: adamw")


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, *, total_steps: int,
                    steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step poly decay with a linear warmup, from the ``train.scheduler`` block.
    Stepped per optimiser step, not per epoch, so the schedule does not follow pool size."""
    name = str(cfg.get("name", "poly")).lower()
    power = float(cfg.get("power", 0.9))
    if name not in ("poly", "cosine"):
        raise ValueError(f"unknown scheduler {name!r}; valid: poly, cosine")
    warmup_steps = int(cfg.get("warmup_epochs", 0)) * steps_per_epoch
    warmup_steps = max(0, min(warmup_steps, max(total_steps - 1, 0)))
    decay_steps = max(1, total_steps - warmup_steps)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if name == "poly":
            return max(0.0, (1.0 - progress) ** power)
        if name == "cosine":
            return 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        raise AssertionError(f"unexpected scheduler {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def set_scheduler_step(scheduler: torch.optim.lr_scheduler.LambdaLR, step: int) -> None:
    """Move a :class:`LambdaLR` to an absolute step, recomputing every group's LR.
    Used on resume: ``load_state_dict`` restores the count of steps, not the position."""
    step = max(0, int(step))
    scheduler.last_epoch = step
    scheduler._step_count = step + 1                                     # noqa: SLF001
    for group, base_lr, fn in zip(scheduler.optimizer.param_groups,
                                  scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base_lr * fn(step)
    scheduler._last_lr = [g["lr"] for g in scheduler.optimizer.param_groups]  # noqa: SLF001


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a.b.c": value}`` over every leaf. Lists are leaves -- order is meaningful."""
    if isinstance(cfg, dict):
        out: dict[str, Any] = {}
        for k, v in cfg.items():
            out.update(flatten_config(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: cfg}


def _drop_excluded(flat: dict[str, Any], exclude: Sequence[str]) -> dict[str, Any]:
    excl = tuple(exclude)
    return {k: v for k, v in flat.items()
            if not any(k == e or k.startswith(e + ".") for e in excl)}


def config_fingerprint(cfg: dict, exclude: Sequence[str] = ()) -> str:
    """Stable short hash of a resolved config, ignoring ``exclude``d key subtrees.
    Hashes the FLATTENED config, so a key that moves between nesting levels still matches."""
    flat = _drop_excluded(flatten_config(cfg), exclude)
    blob = json.dumps({k: repr(v) for k, v in sorted(flat.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def config_changes(old: dict, new: dict, exclude: Sequence[str] = ()) -> list[str]:
    """Human-readable ``key: old -> new`` lines for every difference the fingerprint notices."""
    a = _drop_excluded(flatten_config(old or {}), exclude)
    b = _drop_excluded(flatten_config(new or {}), exclude)
    lines = []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            lines.append(f"{key}: <absent> -> {b[key]!r}")
        elif key not in b:
            lines.append(f"{key}: {a[key]!r} -> <absent>")
        elif repr(a[key]) != repr(b[key]):
            lines.append(f"{key}: {a[key]!r} -> {b[key]!r}")
    return lines


def cpu_budget() -> int:
    """CPUs this process may actually run on. ``os.cpu_count()`` reports the whole node
    while cgroups and the scheduler constrain ``sched_getaffinity``, which is what this reads."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                   # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _worker_init() -> None:
    """Keep each worker single-threaded: N processes x N threads thrashes the cores."""
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:                                                    # noqa: BLE001
        pass
    torch.set_num_threads(1)


class WorkerPool:
    """Ordered, bounded-in-flight ``map`` over a process pool, or serial if unavailable.
    Results come back in input order and at most ``max_in_flight`` tasks are outstanding."""

    DEFAULT_PRELOAD = ("octtta.engine",)

    def __init__(self, workers: int, *, context: str = "forkserver",
                 preload: Sequence[str] | None = None) -> None:
        self.workers = max(1, int(workers))
        self._context = context
        self._preload = tuple(preload) if preload else self.DEFAULT_PRELOAD
        self._pool: Any = None

    def __repr__(self) -> str:
        return f"WorkerPool(workers={self.workers}, started={self._pool is not None})"

    def _ensure(self) -> Any:
        if self._pool is None and self.workers > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            try:
                ctx = mp.get_context(self._context)
            except ValueError:                                   # pragma: no cover
                ctx = mp.get_context("spawn")
            if hasattr(ctx, "set_forkserver_preload"):
                ctx.set_forkserver_preload(list(self._preload))
            t0 = time.perf_counter()
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx,
                                             initializer=_worker_init)
        # Force the workers up now, so the cost shows here and not inside the first validation.
            list(self._pool.map(int, range(self.workers)))
            print(f"[pool] {self.workers} workers up in "
                  f"{time.perf_counter() - t0:.1f}s ({self._context}, "
                  f"preload={list(self._preload) or 'none'})", flush=True)
        return self._pool

    def imap(self, fn: Callable[[Any], Any], tasks: Iterable[Any], *,
             in_flight: int | None = None) -> Iterator[Any]:
        if self.workers <= 1:
            for task in tasks:
                yield fn(task)
            return

        from collections import deque

        pool = self._ensure()
        limit = int(in_flight or 2 * self.workers)
        pending: deque = deque()
        it = iter(tasks)
        try:
            for task in it:
                pending.append(pool.submit(fn, task))
                if len(pending) >= limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        except BaseException:
            for fut in pending:
                fut.cancel()
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


def fsync_dir(directory: Path) -> None:
    """Flush a directory entry, so a rename survives a power cut. Best effort.
    Not every filesystem allows a directory fsync; where it refuses there is nothing to do."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:                                                  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:                                                  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _atomic_save(payload: dict, path: Path, *, fsync: bool = False) -> None:
    """Write via a temp file in the same directory, then rename.
    A preemption during ``torch.save`` would otherwise leave a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    if fsync:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
    else:
        torch.save(payload, tmp)
    os.replace(tmp, path)
    if fsync:
        fsync_dir(path.parent)


def weights_only_payload(payload: dict) -> dict:
    """The same checkpoint with everything only a RESUME would read taken out.
    Nothing resumes on the evaluation server, and the optimiser moments are most of the file."""
    kept = {k: v for k, v in payload.items() if k not in WEIGHTS_ONLY_DROP_KEYS}
    weights = payload.get("ema")
    which = "ema"
    if weights is None:
        weights = payload.get("model")
        which = "model"
    if weights is None:
        raise ValueError("checkpoint payload has neither 'ema' nor 'model' weights; "
                         "there is nothing for a weights-only file to carry")
    kept["model"] = weights
    kept["ema"] = weights
        # Provenance, so a resume that finds this stamp refuses instead of raising deeper.
    kept["weights_only"] = {"kept": which, "dropped": list(WEIGHTS_ONLY_DROP_KEYS)}
    return kept


@dataclass
class CheckpointManager:
    """Writes ``last.pt`` and keeps the best ``save_top_k`` by the monitored score.
    ``monitor`` is ``challenge_score``: Dice and loss both rank checkpoints differently."""

    directory: Path
    monitor: str = "challenge_score"
    mode: str = "max"
    save_top_k: int = 3
    #: ``(score, path)`` for every kept checkpoint, best first. Persisted in ``last.pt``.
    top: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def save_last_as(self, payload: dict, path: Path, *, fsync: bool = False) -> Path:
        """``save_last``'s payload, written wherever the caller says.
        Exists so a paired commit can write one model's "last" payload to another file."""
        _atomic_save({**payload, "top_k": self.top}, path, fsync=fsync)
        return path

    def save_last(self, payload: dict) -> Path:
        return self.save_last_as(payload, self.last_path)

    def maybe_save_best(self, payload: dict, score: float, epoch: int) -> Path | None:
        """Record a validated checkpoint; returns its path if it made the top-k."""
        if not np.isfinite(score):
            return None
        tag = self.directory / f"epoch{epoch:04d}_{self.monitor}{score:.4f}.pt"
        entries = sorted(self.top + [(float(score), str(tag))],
                         key=lambda kv: kv[0], reverse=self.mode == "max")
        keep, drop = entries[: self.save_top_k], entries[self.save_top_k :]
        if str(tag) not in [p for _, p in keep]:
            return None

        _atomic_save({**payload, "top_k": keep}, tag)
        for _, path in drop:
            Path(path).unlink(missing_ok=True)
        self.top = keep
        # best.pt is a copy, not a symlink: the submission image copies files out of here.
        if keep[0][1] == str(tag):
            _atomic_save({**payload, "top_k": keep}, self.best_path)
        return tag

    def load_latest(self, map_location: str = "cpu") -> dict | None:
        """Resume payload, or ``None`` if this run directory is fresh."""
        if not self.last_path.is_file():
            return None
        payload = torch.load(self.last_path, map_location=map_location, weights_only=False)
        self.top = [(float(s), str(p)) for s, p in payload.get("top_k", [])]
        return payload


_interrupted = False
_emergency: Callable[[], None] | None = None


def preempted() -> bool:
    """True once a preemption signal has been seen (checked between steps)."""
    return _interrupted


def install_preemption_handler(save_fn: Callable[[], None] | None,
                               signals: Sequence[int] = (signal.SIGUSR1, signal.SIGTERM),
                               ) -> None:
    """Save one checkpoint and leave, as fast as possible.
    ``save_fn`` must be ``None`` on non-zero ranks: only rank 0 owns the run directory."""

    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        global _interrupted
        if _interrupted:
            os._exit(EXIT_PREEMPTED)
        _interrupted = True
        name = signal.Signals(signum).name
        print(f"\n[preempt] {name} received at {time.strftime('%H:%M:%S')}; checkpointing",
              flush=True)
        if save_fn is not None:
            try:
                save_fn()
                print("[preempt] checkpoint written", flush=True)
            except Exception as exc:                                     # noqa: BLE001
                print(f"[preempt] checkpoint FAILED: {exc!r}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_PREEMPTED)

    for sig in signals:
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass

        # Handshake with the batch script: touching this file says the handlers are installed,
        # so a preemption signal can be forwarded immediately.
    ready = os.environ.get("OCTTTA_SIGNAL_READY_FILE")
    if ready and save_fn is not None:
        try:
            Path(ready).touch()
        except OSError:
            pass


def load_inference_model(
    checkpoint: Path | str,
    *,
    device: torch.device | str = "cpu",
    prefer_ema: bool = True,
) -> tuple[nn.Module, dict]:
    """Rebuild the trained model from a checkpoint alone; returns ``(model, config)``.
    The EMA weights are preferred when present: they are what local validation selected on."""
    from octtta.models import build_model
    from octtta import runenv_status as resources

    rss_before = resources.rss()["self_max_bytes"]
    # ``mmap=True``, and the payload is released BEFORE the model is built: a multi-GiB
    # payload read into anonymous memory and held across construction doubles peak RSS.
    mmapped = True
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False,
                             mmap=True)
    except (RuntimeError, ValueError):
        mmapped = False
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = payload.get("config")
    if not cfg:
        raise ValueError(
            f"{checkpoint} carries no config; the architecture "
            "cannot be reconstructed from a bare state dict"
        )
    cfg = dict(cfg)

    state = payload.get("ema") if prefer_ema else None
    which = "ema"
    if state is None:
        state = payload.get("model")
        which = "model"
    if state is None:
        raise ValueError(f"{checkpoint} has neither 'ema' nor 'model' weights")
    del payload

    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(state)
    del state
    model.to(device).eval()
    cfg["_weights"] = which
    if os.environ.get(resources.ENVELOPE_ENV):
        try:
            size = Path(checkpoint).stat().st_size
        except OSError:
            size = None
        resources.record_event("load_inference_model", checkpoint=str(checkpoint),
                               bytes=size, mmap=mmapped, weights=which,
                               rss_before_bytes=rss_before)
    return model, cfg


def append_jsonl(path: Path | str, record: dict) -> None:
    """One JSON object per line; opened and closed per write so a kill loses nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
