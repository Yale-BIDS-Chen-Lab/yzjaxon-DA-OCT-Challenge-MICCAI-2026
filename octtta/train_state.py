"""Training run metadata and checkpoint state helpers.

Record what hardware a run executed on, in ``run_env.json`` in the run directory.

The launch history retains every GPU type used by a resumed run."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from octtta.engine import (WEIGHTS_ONLY_DROP_KEYS, config_changes,
                           set_scheduler_step, weights_only_payload)
from octtta.models import ModelEMA

FILENAME = "run_env.json"

_GRES_BY_MARKER = (
    ("L40S", "l40s"),
    ("H200", "h200"),
    ("H100", "h100"),
    ("A100", "a100"),
    ("A40", "a40"),
)

def slurm_gres_for(device_name: str) -> str:
    """``'NVIDIA H100 80GB HBM3'`` -> ``'h100'``; unrecognised names -> ``'unknown'``."""
    upper = device_name.upper()
    for marker, gres in _GRES_BY_MARKER:
        if marker in upper:
            return gres
    return "unknown"


def describe() -> dict:
    import torch

    devices = []
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    gres = sorted({slurm_gres_for(name) for name in devices})
    return {
        "gpu_names": devices,
        "gpu_count": len(devices),
        # A list, not a scalar: a run that lands on two GPU models must not be able to
        # report one of them and look consistent.
        "slurm_gres": gres,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    }


def record(out_dir: str | Path) -> Path:
    """Append this launch to the run's hardware history, under an exclusive flock:
    it is a read-modify-write, so a racing or overwriting write silently drops a GPU
    model that shaped the weights."""
    dest = Path(out_dir) / FILENAME
    entry = describe()
    fd = os.open(dest, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            raw = fh.read()
            launches: list[dict] = []
            if raw.strip():
                prev = json.loads(raw)
                launches = list(prev.get("launches") or [prev])
            launches.append(entry)
            union = sorted({g for e in launches for g in (e.get("slurm_gres") or [])})
            fh.seek(0)
            fh.truncate()
            json.dump({"slurm_gres": union, "launches": launches}, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return dest


_SEEDED_ID = re.compile(r"_s(\d{6,})")
CHECKPOINT_KEYS = frozenset({"weights_only"})

def resolve_finetune_source(cfg: dict, args: argparse.Namespace | None = None) -> Path | None:
    """The checkpoint to seed a fine-tune from, or ``None``; the flag wins over the config."""
    src = getattr(args, "finetune_from", None) if args is not None else None
    if not src:
        src = (cfg.get("train") or {}).get("finetune_from")
    if not src:
        return None
    path = Path(str(src)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"train.finetune_from / --finetune-from: {path} does not exist")
    _assert_seeded_chain(cfg, path)
    return path


def _assert_seeded_chain(cfg: dict, src: Path) -> None:
    """A seeded run may only continue a checkpoint carrying the SAME seed.

    Silent when this run's ``experiment.id`` carries no ``_s<seed>``."""
    mine = _SEEDED_ID.findall(str((cfg.get("experiment") or {}).get("id") or ""))
    if not mine:
        return
    want = mine[-1]
    # mmap: one string out of a multi-GB payload. Pre-zipfile checkpoints fall back instead.
    try:
        payload = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        payload = torch.load(src, map_location="cpu", weights_only=False)
    src_cfg = payload.get("config") or {}
    src_id = str((src_cfg.get("experiment") or {}).get("id") or "")
    del payload
    got = _SEEDED_ID.findall(src_id)
    if not got:
        raise ValueError(
            f"train.finetune_from {src} was written by run {src_id!r}, whose id carries no "
            f"seed, but this run's id declares seed {want}. A paired family's phase 2 must "
            f"continue its OWN phase 1 -- an unseeded source means every seed would share "
            f"one checkpoint and the paired differences would all be zero-variance "
            f"artefacts. Pass train.finetune_from=<...p1_s{want}>/checkpoints/last.pt.")
    if got[-1] != want:
        raise ValueError(
            f"train.finetune_from {src} was written at seed {got[-1]} (run {src_id!r}) but "
            f"this run declares seed {want}. Phase 2 must continue the phase 1 of its own "
            f"seed; crossing them makes the paired difference meaningless while every "
            f"job still succeeds.")


def load_warm_start_state(module: nn.Module, state: dict, *, what: str) -> list[str]:
    """``load_state_dict`` that allows exactly the keys the model declares as new.

    Plain ``strict=False`` would also accept a renamed encoder and train from scratch."""
    declared = getattr(module, "warm_start_optional_keys", None)
    optional = set(declared()) if callable(declared) else set()
    result = module.load_state_dict(state, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    undeclared = sorted(set(missing) - optional)
    if undeclared or unexpected:
        raise RuntimeError(
            f"warm start of {what} does not match this model.\n"
            f"  missing and NOT declared as new by the model ({len(undeclared)}): "
            f"{undeclared[:8]}{' ...' if len(undeclared) > 8 else ''}\n"
            f"  in the checkpoint but not in the model ({len(unexpected)}): "
            f"{unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            f"Only keys a model returns from warm_start_optional_keys() may start fresh; "
            f"everything else means the checkpoint and the config describe different "
            f"networks, and loading it anyway would train something nobody asked for.")
    return missing


def load_finetune_weights(payload: dict, model: nn.Module, ema: ModelEMA | None = None) -> str:
    """Copy **weights only** out of a checkpoint payload; returns what was loaded.

    The EMA step counter comes along, or the first update would overwrite the shadow."""
    state = payload.get("model") or payload.get("ema")
    if state is None:
        raise ValueError("checkpoint has neither 'model' nor 'ema' weights to fine-tune from")
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    fresh = load_warm_start_state(model, state, what="the student")
    loaded = "model" if payload.get("model") is not None else "ema->model"
    if fresh:
            # Named: a fresh block and a fresh network look identical in a loss curve.
        loaded += (f" [+{len(fresh)} key(s) newly initialised: "
                   f"{', '.join(fresh[:6])}{' ...' if len(fresh) > 6 else ''}]")

    if ema is not None:
        ema_state = payload.get("ema") or payload.get("model")
        ema_state = {k[len("module."):] if k.startswith("module.") else k: v
                     for k, v in ema_state.items()}
        load_warm_start_state(ema.module, ema_state, what="the EMA shadow")
        stored = payload.get("ema_state") or {}
        if "step" in stored:
            ema.step = int(stored["step"])
        else:
            # The export drops ``ema_state``, and ``step = 0`` would erase the loaded shadow.
            ema.step = max(ema.step, 10 * max(1, ema.warmup_steps))
            loaded += " [no ema_state: shadow started warm]"
        loaded += f" + ema(step={ema.step}, first_decay={ema.current_decay():.5f})"
    return loaded


def checkpoint_weights_only(cfg: dict) -> bool:
    """``checkpoint.weights_only``, default **False** so checkpoints stay resumable."""
    block = cfg.get("checkpoint") or {}
    if not isinstance(block, dict):
        raise ValueError(f"checkpoint: expected a mapping, got {type(block).__name__}")
    unknown = set(block) - CHECKPOINT_KEYS
    if unknown:
        raise ValueError(f"checkpoint: unknown key(s) {sorted(unknown)}; known: "
                         f"{sorted(CHECKPOINT_KEYS)}")
    return bool(block.get("weights_only", False))



CONFIG_VOLATILE_KEYS: tuple[str, ...] = (
    "_source",
    "slurm",
    "runtime.device",
    "runtime.out_dir",
    "runtime.run_dir",
    "runtime.ckpt_dir",
    "runtime.log_every",
    "runtime.headless",
    "data.loader.num_workers",
    "data.loader.persistent_workers",
    "data.loader.pin_memory",
    "data.loader.prefetch_factor",
    "eval.metric_workers",
)


class TrainStateMixin:
    """Atomic checkpoint pairs, resume, and preemption state for Trainer."""
    PAIR_KEEP_GENERATIONS = 1

    def _b_path(self, kind: str) -> Path:
        return self.ckpt.directory / f"{kind}_b.pt"


    def _payload_b(self, epoch_done: bool = True) -> dict:
        """Student B's checkpoint payload. One owner, two writers, so the twin of ``last.pt``
        and the twin of ``best.pt`` cannot carry different fields."""
        payload = {
            "run_kind": "coteach_b",
            "epoch_done": bool(epoch_done),
            "finetune_from": str(self.finetune_from_b),
            "model": self.model_b.state_dict(),
            "ema": self.ema_b.module.state_dict(),
            "ema_state": self.ema_b.state_dict(),
            "optimizer": self.optimizer_b.state_dict(),
            "scheduler": self.scheduler_b.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "config": self.cfg_b,
        }
        # The same switch as A's, off the same attribute: a half-trimmed pair is nobody's choice.
        return weights_only_payload(payload) if self.weights_only else payload


    def _save_b(self, kind: str, epoch_done: bool = True) -> None:
        """``best_b.pt`` beside A's ``best.pt``. ``kind="last"`` goes through
        :meth:`_save_last_pair` instead: that pair is committed, not merely written."""
        if self.model_b is None:
            return
        from octtta.engine import _atomic_save

        _atomic_save(self._payload_b(epoch_done), self._b_path(kind))


    @property
    def _pair_manifest(self) -> Path:
        return self.ckpt.directory / "last_pair.json"


    def _pair_paths(self, gen: int) -> tuple[Path, Path]:
        return (self.ckpt.directory / f"last.g{gen}.pt",
                self.ckpt.directory / f"last_b.g{gen}.pt")


    def _read_pair_manifest(self) -> dict | None:
        """The committed generation, or ``None`` for a run that never wrote one."""
        if not self._pair_manifest.is_file():
            return None
        try:
            man = json.loads(self._pair_manifest.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{self._pair_manifest} is unreadable ({exc}). It is the only record of "
                f"which A/B pair is consistent, and guessing from the directory listing is "
                f"exactly the mistake it exists to prevent.") from None
        for key in ("generation", "a", "b"):
            if key not in man:
                raise RuntimeError(f"{self._pair_manifest} has no {key!r}: not a pair manifest")
        return man


    def _link_alias(self, target: Path, alias: Path) -> None:
        """Point ``alias`` at ``target``'s bytes atomically: a hard link through a temp name,
        falling back to a copy. Never a symlink -- a dangling one ships as a silent zero."""
        tmp = alias.with_suffix(alias.suffix + f".link{os.getpid()}")
        tmp.unlink(missing_ok=True)
        try:
            os.link(target, tmp)
        except OSError:                                              # pragma: no cover
            shutil.copy2(target, tmp)
        os.replace(tmp, alias)


    def _save_last_pair(self, payload: dict, epoch_done: bool = True) -> None:
        """Commit A's and B's "last" as ONE object. Single-model runs take the first branch."""
        if self.model_b is None:
            self.ckpt.save_last(payload)
            return
        from octtta.engine import _atomic_save, fsync_dir

        man = self._read_pair_manifest()
        gen = int(man["generation"]) + 1 if man else 0
        a_path, b_path = self._pair_paths(gen)
        self.ckpt.save_last_as(payload, a_path, fsync=True)
        _atomic_save(self._payload_b(epoch_done), b_path, fsync=True)
        tmp = self._pair_manifest.with_suffix(f".json.tmp{os.getpid()}")
        with open(tmp, "w") as fh:
            json.dump({"generation": gen, "a": a_path.name, "b": b_path.name,
                       "epoch": int(self.epoch), "global_step": int(self.global_step),
                       "epoch_done": bool(epoch_done),
                       "written": datetime.now(timezone.utc).isoformat()},
                      fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._pair_manifest)
        fsync_dir(self._pair_manifest.parent)
        self._link_alias(a_path, self.ckpt.last_path)
        self._link_alias(b_path, self._b_path("last"))
        self._prune_pair_generations(gen)


    def _prune_pair_generations(self, current: int) -> None:
        """Delete every generation older than the one before ``current``."""
        for old in (sorted(self.ckpt.directory.glob("last.g*.pt"))
                    + sorted(self.ckpt.directory.glob("last_b.g*.pt"))):
            try:
                n = int(old.name.rsplit(".g", 1)[1].split(".")[0])
            except (IndexError, ValueError):                         # pragma: no cover
                continue
            if n <= current - self.PAIR_KEEP_GENERATIONS:
                old.unlink(missing_ok=True)


    def _repair_pair_from_manifest(self) -> None:
        """Make ``last.pt`` / ``last_b.pt`` agree with the committed generation, so that the
        manifest rather than the directory listing decides which pair is consistent."""
        man = self._read_pair_manifest()
        if man is None:
            return                        # legacy or single-model run: nothing to repair
        a_path = self.ckpt.directory / str(man["a"])
        b_path = self.ckpt.directory / str(man["b"])
        missing = [str(q) for q in (a_path, b_path) if not q.is_file()]
        if missing:
            raise RuntimeError(
                f"{self._pair_manifest} commits generation {man['generation']} but "
                f"{missing} is gone. The manifest is only replaced after both files are "
                f"fsynced, so this is a hand-edited or partly deleted run directory rather "
                f"than a torn write -- refusing, instead of falling back to whatever else "
                f"is lying around.")
        for target, alias in ((a_path, self.ckpt.last_path), (b_path, self._b_path("last"))):
            if alias.is_file() and alias.samefile(target):
                continue
            self._link_alias(target, alias)
        print(f"[pair] last = generation {man['generation']} "
              f"({man['a']} + {man['b']}, epoch {man.get('epoch')}, "
              f"step {man.get('global_step')})", flush=True)


    def _resume_b(self) -> None:
        """B resumes exactly when A did, from the ``last_b.pt`` written by the same save:
        anything that is not the twin of A's ``last.pt`` is refused rather than patched."""
        if self.model_b is None:
            return
        if not self._resumed_from_last:
            return
        last_b = self._b_path("last")
        if not last_b.is_file():
            raise RuntimeError(f"A resumed from {self.ckpt.last_path} but {last_b} does not "
                               "exist; the two students cannot be at different points")
        payload = torch.load(str(last_b), map_location="cpu", weights_only=False)
        # Epoch POSITION only: A's resume rewinds ``global_step`` to the start of the epoch.
        pos = int(payload.get("epoch", -1)) + int(payload.get("epoch_done", True))
        step_b = int(payload.get("global_step", -1))
        if pos != int(self.epoch) or step_b != self._resumed_global_step:
            raise RuntimeError(f"{last_b} (epoch {payload.get('epoch')}, done={payload.get('epoch_done')}, "
                               f"step {step_b}) is not the twin of the last.pt A resumed from "
                               f"(epoch {self.epoch}, saved step {self._resumed_global_step})")
        self.model_b.load_state_dict(payload["model"])
        self.optimizer_b.load_state_dict(payload["optimizer"])
        self.scheduler_b.load_state_dict(payload["scheduler"])
        self.ema_b.load_state_dict(payload["ema_state"])
        set_scheduler_step(self.scheduler_b, self.global_step)
        print(f"[resume B] {last_b} @ step {self.global_step}", flush=True)


    def _resume(self) -> None:
        """Continue this run, or seed a fine-tune, told apart by ``--finetune-from`` rather
        than by what is in the run directory: only a resume restores optimiser and schedule."""
        payload = self.ckpt.load_latest(map_location="cpu")
        if payload is not None and self.finetune_from is not None \
                and str(payload.get("run_kind", "train")) != "finetune":
            # A baked checkpoint dropped in as ``last.pt`` is a seed, not a resume.
            print(f"[finetune] {self.ckpt.last_path} was written by a '"
                  f"{payload.get('run_kind', 'train')}' run, not by this fine-tune; "
                  "treating it as weights only and discarding its optimiser, "
                  "schedule, epoch and top-k state.", flush=True)
            self.ckpt.top = []
            payload = None

        if payload is None:
            if self.finetune_from is not None:
                self._start_finetune()
            return
        # Refused, not worked around: weights-only drops exactly the state a resume needs.
        if payload.get("weights_only") or "optimizer" not in payload:
            raise RuntimeError(
                f"{self.ckpt.last_path} was written under checkpoint.weights_only "
                f"(dropped: {list(WEIGHTS_ONLY_DROP_KEYS)}) and cannot be resumed: the "
                "optimiser moments and the schedule position are not in it. Set "
                "checkpoint.weights_only=false for a run that has to survive preemption, "
                "or start a new experiment.id seeded with --finetune-from.")
        self._check_config_drift(payload)
        self._resumed_from_last = True
        self._resumed_global_step = int(payload.get("global_step", -1))

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self._enforce_configured_lr()
        if self.ema is not None and payload.get("ema_state"):
            self.ema.load_state_dict(payload["ema_state"])
        # A preemption checkpoint is written mid-epoch, so that epoch is redone, not skipped.
        self.epoch = int(payload.get("epoch", 0)) + int(payload.get("epoch_done", True))
        self.best_score = float(payload.get("best_score", self.best_score))
        self.last_topology = float(payload.get("last_topology", float("inf")))

        # The saved step count is ahead of the replayed epoch, so both are re-derived from it.
        taken = int(payload.get("global_step", 0))
        self.global_step = self.epoch * self.steps_per_epoch
        set_scheduler_step(self.scheduler, self.global_step)
        drift = taken - self.global_step
        print(f"[resume] {self.ckpt.last_path} -> epoch {self.epoch}, "
              f"step {self.global_step}, best {self.best_score:.4f}, "
              f"top_k={[round(s, 4) for s, _ in self.ckpt.top]}")
        if drift:
            print(f"[resume] rewound the LR schedule by {drift} replayed step(s) "
                  f"(lr={self.scheduler.get_last_lr()[0]:.3e})")


    def _start_finetune(self) -> None:
        """Load the baked weights, keep nothing else, start at the config's LR.

        No config-drift check: a fine-tune config differs from the pretraining one by design."""
        payload = torch.load(str(self.finetune_from), map_location="cpu", weights_only=False)
        loaded = load_finetune_weights(payload, self.model, self.ema)


        self.epoch = 0
        self.global_step = 0
        self.best_score = -math.inf if self.ckpt.mode == "max" else math.inf
        # The topology prior travels with the weights; a payload without it keeps ``inf``.
        self.last_topology = float(payload.get("last_topology", float("inf")))
        self.ckpt.top = []
        self._enforce_configured_lr()
        set_scheduler_step(self.scheduler, 0)

        print(f"[finetune] weights from {self.finetune_from} ({loaded}); "
              f"topology prior {self.last_topology:.4f} from the checkpoint")
        print(f"[finetune] optimiser and schedule are FRESH: lr="
              f"{self.configured_lrs[0]:.3e}, "
              f"{self.cfg.get('train', {}).get('scheduler', {}).get('name', 'poly')} "
              f"over {self.epochs} epoch(s) x {self.steps_per_epoch} step(s); "
              f"epoch/step reset to 0", flush=True)


    def _enforce_configured_lr(self) -> None:
        """The config's LR wins over any LR restored from a checkpoint: ``load_state_dict``
        brings back ``lr``, ``initial_lr`` and ``base_lrs``, and the only trace is the score."""
        restored = [float(x) for x in self.scheduler.base_lrs]
        # ``zip`` truncates, and a different freezing plan has a different group count.
        if len(restored) != len(self.configured_lrs):
            raise RuntimeError(
                f"the checkpoint's schedule has {len(restored)} parameter group(s), this "
                f"run's optimiser has {len(self.configured_lrs)}. That is a different "
                f"freezing plan (train.finetune.phase?), not a resumable run -- give it its "
                f"own experiment.id, or seed it with --finetune-from, which takes the "
                f"weights and leaves the optimiser behind.")
        if all(math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
               for a, b in zip(restored, self.configured_lrs)):
            return
        print(f"[lr] checkpoint carried base_lrs={[f'{x:.3e}' for x in restored]}; "
              f"this run is configured for "
              f"{[f'{x:.3e}' for x in self.configured_lrs]} -- using the CONFIG.",
              flush=True)
        self.scheduler.base_lrs = list(self.configured_lrs)
        for group, lr in zip(self.optimizer.param_groups, self.configured_lrs):
            group["initial_lr"] = lr
            group["lr"] = lr


    def _check_config_drift(self, payload: dict) -> None:
        """Refuse to continue a run whose config is no longer the one it started with: a run
        directory is keyed on ``experiment.id`` alone, so the old run would be continued."""
        stored = payload.get("config_fingerprint")
        if stored is None:
            print("[resume] checkpoint predates config fingerprinting -- cannot verify "
                  "the config matches; continuing.")
            return
        if stored == self.fingerprint:
            return
        changes = config_changes(payload.get("config") or {}, self.cfg, CONFIG_VOLATILE_KEYS)
        detail = "\n  ".join(changes[:20]) or "(differs in keys not shown)"
        more = f"\n  (+{len(changes) - 20} more)" if len(changes) > 20 else ""
        raise RuntimeError(
            f"refusing to resume {self.ckpt.last_path}: its config fingerprint "
            f"{stored} != {self.fingerprint} for this run.\n"
            f"Changed keys:\n  {detail}{more}\n"
            "Give this variant its own experiment.id; a changed config cannot resume."
        )


    def _payload(self, epoch_done: bool = True) -> dict:
        payload = {
            "epoch_done": bool(epoch_done),
            # "my own run, continued" vs "somebody else's product, to be used as weights".
            "run_kind": self.run_kind,
            "finetune_from": str(self.finetune_from) if self.finetune_from else None,
            # The stamp travels with the weights: what is copied into a submission is the .pt.
            "model": self.model.state_dict(),
            "ema": self.ema.module.state_dict() if self.ema is not None else None,
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "last_topology": self.last_topology,
            "monitor": self.ckpt.monitor,
            "config": self.cfg,
            "config_fingerprint": self.fingerprint,
            # Where in the cover walk these weights are. ``None`` under multinomial: no walk.
            "cover": (dict(self.sampling_audit)
                      if getattr(self, "sampling_mode", "multinomial") == "cover" else None),
        }
        # One place, so every writer of a payload agrees on what a checkpoint of this run holds.
        return weights_only_payload(payload) if self.weights_only else payload


    def _emergency_save(self) -> None:
        """Called from the signal handler. Must stay small and must not validate.

        ONE call: with a second student the pair is committed together or not at all."""
        self._save_last_pair(self._payload(epoch_done=False), epoch_done=False)
