"""Resolve the Final training pools, epoch lengths, and data-loader geometry."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from octtta import paths
from octtta.data import partial_pool, unlabeled_local
from octtta.data.dataset import (OCTSegDataset, assert_no_inference_pool,
                                 build_datasets, build_unlabeled_dataset, collate_pad,
                                 native_collate, summarize_samples)
from octtta.data.pseudo_widefield import PseudoWideFieldDataset
from octtta.data.release_dataset import labeled_samples, unlabeled_samples
from octtta.data.sampling import (ShardedCoverSampler, ShardedWeightedSampler,
                                  cover_repeats, make_cover_index, make_train_sampler,
                                  resolve_cell_balance, resolve_sampling_mode)
from octtta.data.transforms import build_augmenter
from octtta.eval.report import cohort_keys, limit_stratified

_UNEXPANDED_ID = re.compile(r"\{[^}]*\}|\$\{?[A-Za-z_]")
UNLABELED_PASS = "unlabeled_pass"
_MIN_SHM_BYTES_FOR_WORKERS = 512 * 1024 * 1024

PARTIAL_POOL_KEYS = frozenset({"enabled", "root", "datasets", "ignore_on_disk",
                               "graders", "samples_per_epoch", "sampling",
                               "cover_epoch_offset", "frame_cap", "cell_balance",
                               "official_share", "exclude_cells"})
UNLABELED_LOCAL_KEYS = frozenset({"unlabeled_roots", "unlabeled_groups"})


def _same_path(a: Path, b: Path) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


@dataclass
class PoolResolution:
    names: list[str]
    data_root: str
    entries: list[dict]
    unregistered: list[dict]

    def audit(self) -> dict:
        return {
            "entries": self.entries + self.unregistered,
            "data_root": self.data_root,
            "unregistered_roots": [e["root"] for e in self.unregistered],
            "all_roots_registered": not self.unregistered,
            "pools_touched": {"train": len(self.entries),
                              "unknown": len(self.unregistered)},
        }


def _validate_partial_pool_cfg(partial_cfg: dict) -> None:
    unknown = sorted(set(partial_cfg) - PARTIAL_POOL_KEYS)
    if unknown:
        raise ValueError(f"data.partial_pool has unknown key(s) {unknown}")
    resolve_cell_balance(partial_cfg.get("cell_balance"))
    capped = partial_pool.validate_frame_cap(partial_cfg.get("frame_cap"))
    if capped and not partial_cfg.get("enabled", False):
        raise ValueError("data.partial_pool.frame_cap requires an enabled partial pool")


def _validate_unlabeled_local_cfg(data: dict) -> None:
    neighbours = UNLABELED_LOCAL_KEYS | {"unlabeled_devices"}
    unknown = sorted(k for k in data if "unlabel" in str(k).lower().replace("_", "")
                     and str(k) not in neighbours)
    if unknown:
        raise ValueError(f"data has unknown unlabeled key(s) {unknown}")
    roots = data.get("unlabeled_roots") or []
    groups = data.get("unlabeled_groups")
    if not isinstance(roots, (list, tuple)):
        raise ValueError("data.unlabeled_roots must be a list")
    if groups is not None and (not isinstance(groups, (list, tuple)) or not roots):
        raise ValueError("data.unlabeled_groups needs a staged root and a list")


def _unlabeled_local_roots(data: dict) -> list[Path]:
    _validate_unlabeled_local_cfg(data)
    expected = paths.POOL_ROOTS["unlabeled_local_all_d88"]
    result: list[Path] = []
    for item in data.get("unlabeled_roots") or []:
        root = Path(str(item)).expanduser().resolve()
        if not _same_path(root, expected):
            raise ValueError("data.unlabeled_roots must name the Final D88 staged pool")
        if root in result:
            raise ValueError("data.unlabeled_roots repeats the staged pool")
        result.append(root)
    return result


def _partial_pool_root(partial_cfg: dict) -> Path:
    root = Path(str(partial_cfg.get("root") or paths.PARTIAL_POOL_D88_DIR)).expanduser()
    if not _same_path(root, paths.POOL_ROOTS["public_partial_labels_all16_d88"]):
        raise ValueError("data.partial_pool.root must name the Final D88 partial pool")
    return root


def _declared_ignored_partial_datasets(partial_cfg: dict) -> list[str]:
    raw = partial_cfg.get("ignore_on_disk") or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("data.partial_pool.ignore_on_disk must be a list")
    names = [str(item) for item in raw]
    if len(names) != len(set(names)):
        raise ValueError("data.partial_pool.ignore_on_disk repeats a name")
    return names


def _effective_ignored_partial_datasets(partial_cfg: dict) -> list[str]:
    return sorted(set(_declared_ignored_partial_datasets(partial_cfg))
                  - {str(item) for item in partial_cfg.get("datasets") or []})


def _declared_partial_datasets(partial_cfg: dict) -> list[str]:
    _validate_partial_pool_cfg(partial_cfg)
    root = _partial_pool_root(partial_cfg)
    raw = partial_cfg.get("datasets")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("enabled data.partial_pool.datasets must name its datasets")
    names = [str(item) for item in raw]
    if len(names) != len(set(names)):
        raise ValueError("data.partial_pool.datasets repeats a name")
    on_disk = partial_pool.available_datasets(root)
    missing = sorted(set(names) - set(on_disk))
    extra = sorted(set(on_disk) - set(names) - set(
        _effective_ignored_partial_datasets(partial_cfg)))
    if missing or extra:
        raise ValueError(f"partial pool declaration differs from disk: missing={missing}, "
                         f"unaccounted={extra}")
    return names


def resolve_pools(cfg: dict) -> PoolResolution:
    """Validate the four Final pool names against the roots the run really opens."""
    data = cfg.get("data") or {}
    root = Path(str(data.get("root", ""))).expanduser()
    partial_cfg = dict(data.get("partial_pool") or {})
    _validate_partial_pool_cfg(partial_cfg)
    unlabeled = _unlabeled_local_roots(data)
    declared = [str(name) for name in data.get("pool_entries") or []]
    if len(declared) != len(set(declared)) or any(n not in paths.POOL_ROOTS for n in declared):
        raise ValueError("data.pool_entries contains an unknown or repeated Final pool name")
    used = []
    for name in ("challenge_release_synthetic", "challenge_release"):
        if _same_path(root, paths.POOL_ROOTS[name]):
            used.append(name)
            break
    # The evaluation server mounts its official training set at a runtime path. Its
    # assembler declares exactly this one pool and disables both local additions.
    if (not used and declared == ["challenge_release"]
            and not partial_cfg.get("enabled", False) and not unlabeled):
        used.append("challenge_release")
    if partial_cfg.get("enabled", False):
        _declared_partial_datasets(partial_cfg)
        used.append("public_partial_labels_all16_d88")
    if unlabeled:
        if not (cfg.get("train", {}).get("coteach") or {}).get("enabled", False):
            raise ValueError("staged unlabelled pool requires train.coteach")
        used.append("unlabeled_local_all_d88")
    if sorted(declared) != sorted(used):
        raise ValueError(f"data.pool_entries {declared} differs from pools actually used {used}")
    unknown = []
    if not any(name in used for name in ("challenge_release_synthetic", "challenge_release")):
        if not data.get("allow_unregistered_root", True):
            raise ValueError("data.root is not one of the four Final registered roots")
        unknown.append({"name": "unregistered", "root": str(root), "pool": "unknown",
                        "provenance": "runtime", "exists": root.exists(),
                        "n_samples": None})
    entries = [{"name": name,
                "root": str(root if name == "challenge_release" else paths.POOL_ROOTS[name]),
                "pool": "train", "provenance": "registered",
                "exists": (root if name == "challenge_release" else paths.POOL_ROOTS[name]).exists()}
               for name in used]
    return PoolResolution(used, str(root), entries, unknown)


def build_pseudo_unlabeled_pool(data_cfg: dict, pseudo) -> tuple[list, dict]:
    """Append the staged Final adaptation pool after the release's own frames."""
    pool = (list(unlabeled_samples(data_cfg["root"],
                                   include_maestro2_unlabeled=pseudo.include_maestro2_unlabeled))
            if pseudo.include_release_unlabeled else [])
    n_release = len(pool)
    roots = _unlabeled_local_roots(data_cfg)
    groups = data_cfg.get("unlabeled_groups")
    for root in roots:
        pool.extend(unlabeled_local.local_unlabeled_samples(
            root, groups=None if groups is None else [str(g) for g in groups]))
    return pool, {"release_unlabeled": ("included" if pseudo.include_release_unlabeled
                                         else "excluded"),
                  "n_release": n_release, "n_local": len(pool) - n_release,
                  "local_roots": [str(root) for root in roots], "groups": groups}

def assert_expanded_run_id(exp_id: str) -> str:
    """Refuse an ``experiment.id`` that is still a template. Returns it unchanged.

    A dropped override is silent: every seed would train into one directory."""
    if _UNEXPANDED_ID.search(exp_id):
        raise ValueError(
            f"experiment.id={exp_id!r} still contains a placeholder. A config may ship its "
            f"id as a template, but something has to fill it in -- pass "
            f"experiment.id=<the real one> on the command line (that is what "
            f"the launchers under scripts/train/ do with the seed). Left as it is, "
            f"this is a legal directory name and every seed would train into it.")
    return exp_id


def resolve_run_dir(cfg: dict) -> Path:
    """``runtime.out_dir / experiment.id``, or ``runtime.run_dir``; stable across requeues."""
    runtime = cfg.get("runtime", {})
    if runtime.get("run_dir"):
        return Path(assert_expanded_run_id(str(runtime["run_dir"])))
    exp_id = str(cfg.get("experiment", {}).get("id") or Path(cfg.get("_source", "run")).stem)
    return Path(runtime.get("out_dir", paths.RUNS_DIR)) / assert_expanded_run_id(exp_id)


def unlabeled_pass_drop(n_unlabeled: int, *, pseudo_batch: int,
                        remainder: "FullPassRemainder | None" = None) -> int:
    """How many frames one epoch of ``samples_per_epoch: unlabeled_pass`` leaves out.

    ``drop_last=True`` silently drops ``n % batch``, so an undeclared remainder refuses."""
    if pseudo_batch <= 0:
        raise ValueError(f"unlabeled_pass_drop needs a positive batch, got {pseudo_batch}")
    r = n_unlabeled % pseudo_batch
    if r == 0:
        return 0
    if remainder is None or not remainder.allow:
        raise ValueError(
            f"samples_per_epoch={UNLABELED_PASS!r}: the unlabelled pool holds "
            f"{n_unlabeled} frames, which is not a whole number of batches of "
            f"{pseudo_batch} ({n_unlabeled} % {pseudo_batch} = {r}). The unlabelled loader "
            f"runs drop_last=True, so those {r} frame(s) would be dropped every epoch "
            f"without a word and the recipe's 'one full pass' would be false by exactly "
            f"that much. Change train.*.batch_size, declare the drop "
            f"(train.*.full_pass_remainder: {{allow: true, max_frames: N}} with N >= {r}), "
            f"or stop claiming a full pass.")
    if r > remainder.max_frames:
        raise ValueError(
            f"samples_per_epoch={UNLABELED_PASS!r}: the unlabelled pool holds "
            f"{n_unlabeled} frames, so batch {pseudo_batch} leaves a remainder of {r} "
            f"frame(s) -- more than the {remainder.max_frames} this recipe declared in "
            f"train.*.full_pass_remainder.max_frames. The bound is what makes the drop a "
            f"decision instead of a rounding habit: either raise max_frames on purpose "
            f"(and say why in the recipe), or change the batch size.")
    return r


def format_unlabeled_pass_line(n_unlabeled: int, *, pseudo_batch: int, dropped: int,
                               epochs: int) -> str:
    """The one start-up line that says what "one full pass" cost, in frames.

    Epoch-aware: only at ``epochs: 1`` are the dropped frames never seen at all."""
    steps = (n_unlabeled - dropped) // pseudo_batch
    head = (f"unlabeled pass: {n_unlabeled:,d} frames, batch {pseudo_batch} ⇒ "
            f"{steps:,d} steps")
    if dropped == 0:
        return f"{head}, nothing left out ({n_unlabeled:,d} is a whole number of batches)"
    if int(epochs) > 1:
        return (f"{head}, {dropped} frames per epoch left out (reshuffled each epoch, so "
                f"every frame is seen across epochs)")
    return (f"{head}, {dropped} frames per epoch left out (epochs=1, so these {dropped} "
            f"frames are NOT seen at all in this run; the loader reshuffles per epoch, so "
            f"which {dropped} they are is decided by runtime.seed)")


def unlabeled_pass_samples_per_epoch(n_unlabeled: int, *, pseudo_batch: int, every: int,
                                     sup_batch: int,
                                     remainder: "FullPassRemainder | None" = None) -> int:
    """The ``samples_per_epoch`` that makes one epoch teach on every unlabelled frame once:
    ``((n_unlabeled - dropped) // pseudo_batch) * every * sup_batch``."""
    if pseudo_batch <= 0 or every <= 0 or sup_batch <= 0:
        raise ValueError(
            f"samples_per_epoch={UNLABELED_PASS!r} needs positive batch sizes, got "
            f"pseudo_batch={pseudo_batch}, every={every}, sup_batch={sup_batch}")
    if n_unlabeled <= 0:
        raise ValueError(
            f"samples_per_epoch={UNLABELED_PASS!r} derives the epoch from the unlabelled "
            f"pool, and that pool reports {n_unlabeled} frames. An epoch of 0 draws trains "
            f"on nothing and raises nowhere else.")
    dropped = unlabeled_pass_drop(n_unlabeled, pseudo_batch=pseudo_batch,
                                  remainder=remainder)
    return ((n_unlabeled - dropped) // pseudo_batch) * every * sup_batch


def _samples_per_epoch(partial_cfg: dict, n_total: int, n_official: int,
                       pass_length: int | None = None,
                       unlabeled_pass_length: int | None = None) -> int:
    """How many indices one epoch draws: all, a numeric dose, or an unlabelled pass."""
    spec = partial_cfg.get("samples_per_epoch", "all") if partial_cfg.get("enabled") \
        else "all"
    if spec == "all":
        return int(n_total)
    if spec == UNLABELED_PASS:
        if unlabeled_pass_length is None:
            raise ValueError(
                f"data.partial_pool.samples_per_epoch={UNLABELED_PASS!r} means one full "
                f"pass of the UNLABELLED pool, and this run has no pseudo-label term "
                f"(train.coteach) -- there is no unlabelled pool to walk. "
                f"Give a number, or turn the term on.")
        return int(unlabeled_pass_length)
    n = int(spec)
    if n <= 0:
        raise ValueError(f"data.partial_pool.samples_per_epoch must be positive, got {n}")
    return n


def _cover_epoch_offset(partial_cfg: dict, mode: str, *, finetune_from,
                        pass_length: int | None, draws_per_epoch: int) -> int | None:
    """Where in the cover walk this run starts: an integer, or ``"finetune_from"``.

    Required in cover mode: two phases sharing a seed would otherwise replay one walk."""
    if mode != "cover":
        if partial_cfg.get("cover_epoch_offset") is not None:
            raise ValueError(
                f"data.partial_pool.cover_epoch_offset="
                f"{partial_cfg['cover_epoch_offset']!r} is set while "
                f"data.partial_pool.sampling={mode!r}, which draws with replacement and "
                "has no walk to offset. A knob that is set but ignored reads like a run "
                "that continued something.")
        return None
    spec = partial_cfg.get("cover_epoch_offset", "<unset>")
    if spec == "<unset>" or spec is None:
        raise ValueError(
            "data.partial_pool.cover_epoch_offset is not set while "
            "data.partial_pool.sampling='cover'. Write 0 for a run that starts a fresh "
            "walk, or 'finetune_from' for a phase that continues the walk of the "
            "checkpoint in train.finetune_from -- omitting it would let a Phase 2 silently "
            "replay Phase 1's windows (measured on e11a_final_p1/_p2, 2026-08-30).")
    if isinstance(spec, bool):
        raise ValueError(
            f"data.partial_pool.cover_epoch_offset={spec!r} is a boolean; it is an epoch "
            "count or the string 'finetune_from'")
    if isinstance(spec, str):
        if spec != "finetune_from":
            raise ValueError(
                f"data.partial_pool.cover_epoch_offset={spec!r}; the only string it "
                "accepts is 'finetune_from' (an integer says it outright)")
        if not finetune_from:
            raise ValueError(
                "data.partial_pool.cover_epoch_offset='finetune_from' but this run has no "
                "train.finetune_from / --finetune-from: there is no earlier walk to "
                "continue")
        return _cover_offset_from_checkpoint(finetune_from, pass_length=pass_length,
                                             draws_per_epoch=draws_per_epoch)
    n = int(spec)
    if n < 0:
        raise ValueError(
            f"data.partial_pool.cover_epoch_offset must be >= 0, got {n}")
    return n


def sampling_audit_record(mode: str, *, draws_per_epoch: int, pass_length: int | None,
                          epoch_offset: int | None, oversample_diseased: float) -> dict:
    """What the sampler is, in one shape, for the log line AND ``pools.audit.json``."""
    return {
        "mode": str(mode),
        "draws_per_epoch": int(draws_per_epoch),
        "pass_length": None if pass_length is None else int(pass_length),
        "epochs_per_pass": (None if pass_length is None
                            else round(pass_length / draws_per_epoch, 3)),
        "epoch_offset": None if epoch_offset is None else int(epoch_offset),
        "diseased_repeats": (cover_repeats(oversample_diseased) if mode == "cover"
                             else None),
        "oversample_diseased": float(oversample_diseased),
    }


def format_sampling_line(audit: dict) -> str:
    """The ``[data] sampling: ...`` line, off the same record the audit file gets."""
    if audit["mode"] == "cover":
        return (f"sampling: cover -- one pass = {audit['pass_length']} draws "
                f"(every image once, diseased x{audit['diseased_repeats']}); "
                f"{audit['draws_per_epoch']} draws per epoch = "
                f"{audit['epochs_per_pass']} epoch(s) per pass; "
                f"walk starts at epoch offset {audit['epoch_offset']}")
    return (f"sampling: multinomial (with replacement, no coverage guarantee); "
            f"{audit['draws_per_epoch']} draws per epoch")


def _cover_offset_from_checkpoint(src, *, pass_length: int, draws_per_epoch: int,
                                  ) -> int:
    """Continue the walk of the checkpoint at ``src``: its offset plus what it completed.

    The pass length, epoch size and repeat table must match, or it is not a continuation."""
    payload = torch.load(str(src), map_location="cpu", weights_only=False)
    cover = payload.get("cover")
    if not cover:
        raise ValueError(
            f"train.finetune_from={src} was not written by a cover-mode run (its payload "
            "carries no walk position), so there is no walk to continue. Retrain the "
            "previous phase under data.partial_pool.sampling='cover', or set "
            "data.partial_pool.cover_epoch_offset to an explicit number.")
    for key, want in (("pass_length", pass_length), ("draws_per_epoch", draws_per_epoch)):
        got = int(cover.get(key, -1))
        if got != int(want):
            raise ValueError(
                f"train.finetune_from={src} walked a pass with {key}={got}, this run has "
                f"{want}. Continuing an offset measured on a different walk lands at an "
                f"arbitrary position and every epoch after it looks normal. The two "
                f"phases must share the data recipe they continue.")
    # A mid-epoch preemption checkpoint has not finished the epoch it names; an end-of-epoch does
    done = int(payload.get("epoch", 0)) + int(payload.get("epoch_done", True))
    return int(cover.get("epoch_offset", 0)) + done


def _shm_supported_workers(workers: int) -> int:
    """Clamp ``num_workers`` to 0 when ``/dev/shm`` cannot carry worker->main transfers.

    Too small a mount hangs the loader (dead feeder threads, unfed queue) instead of raising."""
    if workers <= 0:
        return workers
    try:
        import shutil

        shm_bytes = shutil.disk_usage("/dev/shm").total
    except OSError:
        # No /dev/shm at all: the sharing strategy cannot work, same clamp applies.
        print(f"[data] /dev/shm is unreadable -> num_workers {workers} -> 0 "
              "(worker loaders move batches through shared memory)")
        return 0
    if shm_bytes < _MIN_SHM_BYTES_FOR_WORKERS:
        print(f"[data] /dev/shm holds {shm_bytes / 2**20:.0f} MiB "
              f"< {_MIN_SHM_BYTES_FOR_WORKERS / 2**20:.0f} MiB -> num_workers "
              f"{workers} -> 0 (8 workers on the evaluation platform's 64 MiB "
              "/dev/shm hung a whole fine-tune budget; submission #3285)")
        return 0
    return workers


def build_val_dataset(cfg: dict, val_ds: OCTSegDataset) -> torch.utils.data.Dataset:
    """Native-resolution macula val set, plus the pseudo wide-field stress copy.

    The inference-pool guard is repeated here: callers may hand in a hand-built dataset."""
    assert_no_inference_pool(val_ds.samples, "build_val_dataset")
    spec = dict((cfg.get("eval") or {}).get("pseudo_widefield") or {})
    if not spec.pop("enabled", True) or len(val_ds) == 0:
        return val_ds
    wide = PseudoWideFieldDataset(
        val_ds,
        width_scale=float(spec.get("width_scale", 2.5)),
        vignette=float(spec.get("vignette", 0.5)),
        drop_peripheral_layers=bool(spec.get("drop_peripheral_layers", True)),
        seed=int(spec.get("seed", 0)),
    )
    return ConcatDataset([val_ds, wide])



class TrainDataMixin:
    """Construct the exact Final train and held-out datasets and samplers."""

    def _build_data(self) -> None:
        cfg = self.cfg
        data_cfg = cfg.get("data", {})
        root = Path(str(data_cfg.get("root")))
        samples = labeled_samples(root)
        if not samples:
            raise FileNotFoundError(f"no labelled samples under {root}")

        augmenter = build_augmenter(cfg.get("augment") or {})
        train_ds, val_ds = build_datasets(cfg, samples, augmenter)
        # Order matters: the exact samples are split first, so an interval label cannot reach val.
        partial_cfg = dict(data_cfg.get("partial_pool") or {})
        _validate_partial_pool_cfg(partial_cfg)
        _validate_unlabeled_local_cfg(data_cfg)
        partial: list = []
        cap = partial_pool.no_frame_cap_record()
        offsets = partial_pool.no_offsets_record()
        partial_ignored: list[str] = []
        if partial_cfg.get("enabled", False):
            names = _declared_partial_datasets(partial_cfg)
            partial_ignored = _effective_ignored_partial_datasets(partial_cfg)
            partial, cap, offsets = partial_pool.load_partial_pool(
                root=_partial_pool_root(partial_cfg), datasets=names,
                graders=tuple(partial_cfg.get("graders") or (1,)),
                frame_cap=partial_cfg.get("frame_cap"))
            if not partial:
                raise FileNotFoundError("enabled partial pool loaded no training samples")
            train_ds.add_samples(partial)
        self.partial_ignored_on_disk = partial_ignored
        self.partial_summary = (partial_pool.pool_summary(partial, cap=cap, offsets=offsets)
                                if partial else None)
        self.train_ds, self.val_native = train_ds, val_ds
        if self.consistency is not None:
            if getattr(train_ds, "consistency", None) is None:
                raise RuntimeError(
                    "train.consistency is enabled but the training dataset was built "
                    "without it, so no dirty view will ever be produced and the term "
                    "would be silently absent from the loss.")
            # The dangerous direction: a val set carrying a corrupted copy would not crash.
            if getattr(val_ds, "consistency", None) is not None:
                raise AssertionError(
                    "the validation dataset was built with the consistency corrupter; "
                    "evaluation must only ever see clean images (D22)")
        n_official_train = len(train_ds) - len(partial)
        print(f"[data] official {summarize_samples(samples)}")
        if partial:
            print(f"[data] public partial {self.partial_summary}")
            line = partial_pool.format_frame_cap_line(cap)
            if line is not None:
                print(f"[data] {line}")
            line = partial_pool.format_offsets_line(
                self.partial_summary["boundary_offsets"])
            if line is not None:
                print(f"[data] {line}")
        print(f"[data] train={len(train_ds)} "
              f"(official {n_official_train} + interval {len(partial)}) "
              f"val={len(val_ds)}  augment={augmenter!r}")
        # The configured knobs here; what actually fired is counted over epoch 0 and printed.
        if train_ds.discaug is not None:
            print(f"[discaug] {train_ds.discaug.summary()}")

        loader_cfg = dict(data_cfg.get("loader") or {})
        workers = _shm_supported_workers(int(loader_cfg.get("num_workers", 0)))
        batch_size = int(loader_cfg.get("batch_size", 2))
        # Read here, not below: ``unlabeled_pass`` derives the epoch length from this geometry.
        pseudo = self.coteach
        unlabeled_pass_length = self._unlabeled_pass_length(
            data_cfg, partial_cfg, pseudo, sup_batch=batch_size)
        seed = int(cfg.get("runtime", {}).get("seed", 0))
        oversample = float(data_cfg.get("oversample_diseased", 1.0))
        self.sampling_mode = resolve_sampling_mode(partial_cfg)
        resolve_cell_balance(partial_cfg.get("cell_balance") if partial else None)
        if self.sampling_mode == "cover":
            cover_index = make_cover_index(train_ds.samples,
                                           oversample_diseased=oversample)
            pass_length = int(cover_index.size)
            per_epoch = _samples_per_epoch(partial_cfg, len(train_ds), n_official_train,
                                           pass_length=pass_length,
                                           unlabeled_pass_length=unlabeled_pass_length)
            epoch_offset = _cover_epoch_offset(
                partial_cfg, self.sampling_mode, finetune_from=self.finetune_from,
                pass_length=pass_length, draws_per_epoch=per_epoch)
            self.sampler = ShardedCoverSampler(
                cover_index, num_samples=per_epoch, seed=seed,
                epoch_offset=epoch_offset)
        else:
            base_sampler = make_train_sampler(train_ds.samples,
                                              oversample_diseased=oversample, seed=seed)
            _cover_epoch_offset(partial_cfg, self.sampling_mode,
                                finetune_from=self.finetune_from,
                                pass_length=None, draws_per_epoch=0)
            per_epoch = _samples_per_epoch(partial_cfg, len(train_ds), n_official_train,
                                           unlabeled_pass_length=unlabeled_pass_length)
            pass_length = epoch_offset = None
            self.sampler = ShardedWeightedSampler(
                len(train_ds), getattr(base_sampler, "weights", None),
                num_samples=per_epoch, seed=seed)
        self.sampling_audit = sampling_audit_record(
            self.sampling_mode, draws_per_epoch=per_epoch, pass_length=pass_length,
            epoch_offset=epoch_offset, oversample_diseased=oversample)
        print(f"[data] {format_sampling_line(self.sampling_audit)}")
        if partial:
            print(f"[data] mixture: natural source proportions; "
                  f"{per_epoch} draws per epoch")
        # The loader gets its own generator: otherwise PyTorch draws the worker seeds from the
        # global RNG, so every augmentation moves when the model consumes one more random number.
        loader_gen = torch.Generator()
        loader_gen.manual_seed(int(cfg.get("runtime", {}).get("seed", 0)))
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=self.sampler,
            num_workers=workers, collate_fn=collate_pad, generator=loader_gen,
            pin_memory=bool(loader_cfg.get("pin_memory", False)) and self.device.type == "cuda",
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and workers > 0,
            prefetch_factor=int(loader_cfg.get("prefetch_factor", 2)) if workers else None,
            drop_last=False,
        )

        full_set = limit_stratified(build_val_dataset(cfg, val_ds),
                                    self.eval_cfg.get("max_images"))
        fast_set = limit_stratified(full_set, self.fast_max_images) if self.fast_every else None
        # batch_size=1 is not a tuning knob: at native resolution every sample has its own shape.
        self.val_loader = self._val_loader(full_set, min(workers, 2))
        self.val_loader_fast = (self._val_loader(fast_set, min(workers, 2))
                                if fast_set is not None else None)
        cells = len(set(cohort_keys(full_set)))
        print(f"[eval] full={len(full_set)} images over {cells} (vendor,status,anatomy) "
              f"cells every {self.full_every} epoch(s); "
              f"fast={len(fast_set) if fast_set is not None else 0} images every "
              f"{self.fast_every or '-'}; metric workers="
              f"{self.pool.workers if self.pool is not None else 0}")

        # ------------------------------------------------ pseudo-label self-training --
        self.release_unlabeled_audit: str | None = None
        self.full_pass_audit: dict | None = None
        if pseudo is not None:
            if self.consistency is None:
                raise RuntimeError("train.coteach requires train.consistency")
            pool_u, pool_audit = build_pseudo_unlabeled_pool(data_cfg, pseudo)
            self.release_unlabeled_audit = pool_audit["release_unlabeled"]
            if pool_audit["local_roots"]:
                print(f"[data] local unlabeled pool: {pool_audit['n_local']} frames from "
                      f"{pool_audit['local_roots']} (groups={pool_audit['groups']})")
            if not pool_u:
                raise RuntimeError("train.coteach has no unlabeled images")
            u_ds = build_unlabeled_dataset(cfg, pool_u, augmenter)
            self._selftrain_ds = u_ds
            self._selftrain_seed = int(cfg.get("runtime", {}).get("seed", 0)) + 7919
            u_gen = torch.Generator()
            u_gen.manual_seed(self._selftrain_seed)
            self._selftrain_loader = DataLoader(
                u_ds, batch_size=pseudo.batch_size, shuffle=True,
                num_workers=workers, collate_fn=collate_pad, generator=u_gen,
                pin_memory=False, drop_last=True)
            print(f"[data] coteach pool: {summarize_samples(pool_u)} "
                  f"| release_unlabeled: {self.release_unlabeled_audit}")
            self._assert_full_unlabeled_pass(pseudo, len(pool_u), per_epoch, batch_size)
