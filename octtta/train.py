"""Supervised training entry point.

    python -m octtta.train CONFIG [key.path=value ...]
    python -m octtta.train --config CONFIG [key.path=value ...]

Single-process on CPU or one GPU. ``all10`` selects on the held-out challenge score;
``none`` trains to ``last.pt`` without validation passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from octtta import paths
from octtta.config import get_config, snapshot
from octtta.data import partial_pool
from octtta.data import unlabeled_local
from octtta.data.dataset import (
    OCTSegDataset,
    assert_no_inference_pool,
    build_datasets,
    build_unlabeled_dataset,
    collate_pad,
    native_collate,
    summarize_samples,
)
from octtta.data.sampling import (
    ShardedCoverSampler,
    ShardedWeightedSampler,
    cover_repeats,
    make_cover_index,
    make_train_sampler,
    resolve_cell_balance,
    resolve_sampling_mode,
)
from octtta.data.release_dataset import (
    index_release,
    labeled_samples,
    unlabeled_samples,
)
from octtta.data.transforms import build_augmenter
from octtta.engine import (
    EXIT_CUDA_OOM,
    WEIGHTS_ONLY_DROP_KEYS,
    CheckpointManager,
    WorkerPool,
    append_jsonl,
    build_optimizer,
    build_scheduler,
    config_changes,
    config_fingerprint,
    count_parameters,
    cpu_budget,
    install_preemption_handler,
    preempted,
    set_scheduler_step,
    set_seed,
    weights_only_payload,
)
from octtta.eval.challenge_metric import NUM_CLASSES
from octtta.eval.report import (EvalReport, cohort_keys, evaluate_parallel,
                                limit_stratified)
from octtta.infer import plan_from_config as infer_plan_from_config
from octtta.losses import build_loss, compute_class_weights
from octtta.train_state import (CONFIG_VOLATILE_KEYS, TrainStateMixin,
                                record as record_runenv, checkpoint_weights_only,
                                resolve_finetune_source, load_finetune_weights,
                                load_warm_start_state)
from octtta.models import ModelEMA, build_model
from octtta.postproc import column_topology_violations, postproc_cfg_or_none

#: Loss-term spellings that require :func:`compute_class_weights` to have been run.
_WEIGHT_SCHEMES = ("auto_inverse_sqrt_freq", "inverse_sqrt_freq", "inverse_freq")

#: Config subtrees excluded from the resume fingerprint; every other key forbids a resume.

#: Dotted-key override such as ``model.name=unet``; any other positional is the config path.
_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*=")


# ---- config -> concrete objects ------------------------------------------------------


from octtta.train_data import (
    PoolResolution, TrainDataMixin, UNLABELED_PASS, _cover_epoch_offset,
    _declared_partial_datasets, _effective_ignored_partial_datasets,
    _partial_pool_root, _samples_per_epoch, _shm_supported_workers,
    _unlabeled_local_roots, _validate_partial_pool_cfg,
    _validate_unlabeled_local_cfg, assert_expanded_run_id,
    build_pseudo_unlabeled_pool, build_val_dataset,
    format_sampling_line, format_unlabeled_pass_line, resolve_pools,
    resolve_run_dir, sampling_audit_record, unlabeled_pass_drop,
    unlabeled_pass_samples_per_epoch,
)








#: ``<family>_<arm>_p<phase>_s<seed>`` -- the naming every paired multi-seed family uses.








# ---- two-phase fine-tuning: frozen encoder, layer-wise decayed LRs -------------------
# ``train.finetune_from`` is where the weights come from; ``train.finetune`` says which move.

#: Keys accepted in ``train.finetune``; an unknown key raises rather than defaulting.
_FINETUNE_KEYS = ("enabled", "phase", "enc_lr", "dec_lr", "layer_decay",
                  "unfreeze_last_blocks")


def finetune_plan_cfg(cfg: dict) -> dict | None:
    """Resolve the live two-phase SAM fine-tuning plan."""
    block = dict((cfg.get("train") or {}).get("finetune") or {})
    if not block or not block.pop("enabled", True):
        return None
    # Baked/server configs retain these keys at their only supported values.
    if block.pop("scope", "full") != "full":
        raise ValueError("train.finetune.scope must be full")
    for key in ("freeze_norm_affine", "unfreeze_patch_embed"):
        if block.pop(key, False) is not False:
            raise ValueError(f"train.finetune.{key} must be false")
    unknown = sorted(set(block) - set(_FINETUNE_KEYS))
    if unknown:
        raise ValueError(f"train.finetune: unknown key(s) {unknown}")
    if "phase" not in block:
        raise ValueError("train.finetune needs an explicit phase")
    phase = int(block["phase"])
    if phase not in (1, 2):
        raise ValueError("train.finetune.phase must be 1 or 2")
    return {"phase": phase,
            "enc_lr": float(block.get("enc_lr", 1.0e-5)),
            "dec_lr": float(block.get("dec_lr", 1.0e-4)),
            "layer_decay": float(block.get("layer_decay", 0.8)),
            "unfreeze_last_blocks": int(block.get("unfreeze_last_blocks", 4))}


def build_grouped_optimizer(opt_cfg: dict, module: nn.Module,
                            groups: Sequence["FinetuneGroup"]) -> torch.optim.Optimizer:
    """One optimiser over per-layer groups; norm/bias parameters get ``weight_decay=0``."""
    name = str(opt_cfg.get("name", "adamw")).lower()
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))

    named = dict(module.named_parameters())
    param_groups: list[dict] = []
    group_names: list[str] = []
    for g in groups:
        decay, no_decay = [], []
        for pname in g.param_names:
            p = named[pname]
            (no_decay if p.ndim <= 1 or pname.endswith(".bias") else decay).append(p)
        for suffix, params, this_wd in (("", decay, wd), ("/nodecay", no_decay, 0.0)):
            if not params:
                continue
            param_groups.append({"params": params, "lr": g.lr, "weight_decay": this_wd})
            group_names.append(g.name + suffix)
    if not param_groups:
        raise ValueError(
            "the fine-tune plan leaves no trainable parameters at all; there is nothing "
            "for the optimiser to hold")

    if name == "adamw":
        opt: torch.optim.Optimizer = torch.optim.AdamW(param_groups, betas=betas)
    elif name == "adam":
        opt = torch.optim.Adam(param_groups, betas=betas)
    elif name == "sgd":
        opt = torch.optim.SGD(param_groups, momentum=float(opt_cfg.get("momentum", 0.99)),
                              nesterov=bool(opt_cfg.get("nesterov", True)))
    else:
        raise ValueError(f"unknown optimizer {name!r}; valid: adamw, adam, sgd")
    opt.octtta_group_names = group_names                                 # type: ignore[attr-defined]
    return opt


def _assert_no_batchnorm(model: nn.Module, why: str) -> None:
    """No module in ``model`` keeps cross-sample running statistics: both callers feed one
    network two batches from two different distributions inside a single step."""
    bn = [n for n, m in model.named_modules()
          if isinstance(m, nn.modules.batchnorm._BatchNorm)]          # noqa: SLF001
    if bn:
        raise ValueError(
            f"{why}, and this model has {len(bn)} BatchNorm module(s) ({bn[:3]}). Batch "
            f"statistics would be computed across both distributions and the deployed "
            f"(clean) prediction would depend on data it never sees at inference. Use an "
            f"InstanceNorm/LayerNorm variant, or turn that term off.")


def assert_optimizer_covers_trainable(optimizer: torch.optim.Optimizer,
                                      model: nn.Module) -> None:
    """The optimiser holds **exactly** the parameters that require grad.

    Otherwise a trainable parameter is never updated, or a frozen one is still decayed."""
    want = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    by_id = {id(p): n for n, p in model.named_parameters()}
    have: dict[int, int] = {}
    seen_twice: list[str] = []
    for gi, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            if id(p) in have:
                # Two groups holding one tensor = two LRs per step, counted before dict folds it.
                seen_twice.append(by_id.get(id(p), f"<unnamed:{id(p)}>"))
            have[id(p)] = gi
    if seen_twice:
        raise AssertionError(
            f"{len(seen_twice)} parameter(s) appear in more than one optimiser group "
            f"{sorted(seen_twice)[:4]}: each would be stepped once per group it sits in.")

    missing = sorted(want[i] for i in want.keys() - have.keys())
    extra = sorted(by_id.get(i, f"<unnamed:{i}>") for i in have.keys() - want.keys())
    if missing or extra:
        raise AssertionError(
            "optimiser coverage != trainable set. "
            f"{len(missing)} trainable parameter(s) the optimiser does not hold "
            f"{missing[:4]}; {len(extra)} frozen parameter(s) it does hold {extra[:4]}. "
            "The first would never be updated; the second would still be weight-decayed.")


# ---- A-Band consistency --------------------------------------------------------------


@dataclass
class ConsistencySpec:
    """``L = L_clean + w_dirty * L_dirty + w_kl * KL(p_dirty || sg(p_clean))``."""

    weight_dirty: float
    weight_kl: float

    def __repr__(self) -> str:                                           # pragma: no cover
        return (f"ConsistencySpec(dirty={self.weight_dirty:g}, kl={self.weight_kl:g})")


def consistency_spec(cfg: dict) -> ConsistencySpec | None:
    """``train.consistency``, or ``None`` when the term is off."""
    from octtta.data.failaug import CONSISTENCY_KEYS

    block = dict((cfg.get("train") or {}).get("consistency") or {})
    if not block:
        return None
    unknown = sorted(set(block) - set(CONSISTENCY_KEYS))
    if unknown:
        raise ValueError(
            f"train.consistency: unknown key(s) {unknown}; known: {list(CONSISTENCY_KEYS)}")
    if not block.get("enabled", False):
        return None
    return ConsistencySpec(weight_dirty=float(block.get("weight_dirty", 0.5)),
                           weight_kl=float(block.get("weight_kl", 0.5)))


def main_head(pred: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """The full-resolution logits, whether or not deep supervision returned a list."""
    return pred[0] if isinstance(pred, (list, tuple)) else pred


def split_pair(out, n: int):
    """Undo the ``cat([clean, dirty])`` batching, at every deep-supervision level."""
    if isinstance(out, (list, tuple)):
        return [o[:n] for o in out], [o[n:] for o in out]
    return out[:n], out[n:]


def padding_valid_mask(pad_hw: Sequence[tuple[int, int]] | None,
                       shape: tuple[int, int],
                       device: torch.device) -> torch.Tensor | None:
    """``(B,1,H,W)`` bool marking real pixels, or ``None`` when nothing was padded."""
    if not pad_hw:
        return None
    max_h, max_w = int(shape[0]), int(shape[1])
    if all(int(h) == max_h and int(w) == max_w for h, w in pad_hw):
        return None
    mask = torch.zeros(len(pad_hw), 1, max_h, max_w, dtype=torch.bool, device=device)
    for i, (h, w) in enumerate(pad_hw):
        mask[i, :, :int(h), :int(w)] = True
    return mask


def consistency_kl(dirty_logits: torch.Tensor, clean_logits: torch.Tensor,
                   valid: torch.Tensor | None = None) -> torch.Tensor:
    """``KL(p_dirty || stopgrad(p_clean))`` over the full-resolution head, T = 1.

    In fp32 whatever the autocast dtype: the signal is a difference of near-equal values."""
    logp_d = F.log_softmax(dirty_logits.float(), dim=1)
    logp_c = F.log_softmax(clean_logits.detach().float(), dim=1)
    per_pixel = (logp_d.exp() * (logp_d - logp_c)).sum(dim=1, keepdim=True)
    if valid is None:
        return per_pixel.mean()
    valid = valid.to(per_pixel.dtype)
    return (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)


# ---- anti-collapse regulariser for interval supervision ------------------------------

INTERVAL_SHARE_KEYS = {"weight", "max_share"}


@dataclass
class IntervalSharePenaltySpec:
    """``train.interval_share_penalty`` resolved: ``loss += weight * P(max_share)``."""

    weight: float
    max_share: float

    def __repr__(self) -> str:                                    # the [loss] banner
        return (f"IntervalSharePenaltySpec(w={self.weight:g}, "
                f"max_share={self.max_share:g})")


def interval_share_penalty_spec(cfg: dict) -> IntervalSharePenaltySpec | None:
    """``train.interval_share_penalty``, or ``None`` when off (``weight: 0.0`` included, so
    "off" means the term never runs); the values are validated before that early return."""
    block = dict((cfg.get("train") or {}).get("interval_share_penalty") or {})
    if not block:
        return None
    unknown = sorted(set(block) - INTERVAL_SHARE_KEYS)
    if unknown:
        raise ValueError(
            f"train.interval_share_penalty: unknown key(s) {unknown}; known: "
            f"{sorted(INTERVAL_SHARE_KEYS)}")
    weight = float(block.get("weight", 0.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"train.interval_share_penalty.weight must be a finite number >= 0 "
            f"(0 = off), got {block.get('weight')!r}")
    max_share = float(block.get("max_share", 0.6))
    if not math.isfinite(max_share) or not 0.0 < max_share <= 1.0:
        # 0 would penalise a correct class in every band, and > 1 is unreachable by a share.
        raise ValueError(
            f"train.interval_share_penalty.max_share must be in (0, 1], got "
            f"{block.get('max_share')!r}")
    if weight == 0.0:
        return None
    return IntervalSharePenaltySpec(weight=weight, max_share=max_share)


def interval_penalty_band(interval: torch.Tensor, n_cls: int = NUM_CLASSES) -> torch.Tensor:
    """``(B,H,W)`` bool: the pixels the share penalty may shape -- a closed interval of at
    least four classes touching neither class 0 nor ``n_cls - 1`` (open bands have no
    anatomical height; narrow ones would make the share a thickness prior)."""
    lo = interval[:, 0]
    hi = interval[:, 1]
    return (lo >= 1) & (hi >= lo + 3) & (hi <= int(n_cls) - 2)


def interval_share_penalty(logits: torch.Tensor, interval: torch.Tensor,
                           max_share: float = 0.6) -> tuple[torch.Tensor, float]:
    """Penalise one class eating a whole interval-supervised band, column by column.

    Returns ``(penalty, measured_column_fraction)``: zero over zero means no such band."""
    if logits.ndim != 4:
        raise ValueError(f"logits must be (B,C,H,W), got {tuple(logits.shape)}")
    if interval.ndim != 4 or interval.shape[1] != 2:
        raise ValueError(f"interval must be (B,2,H,W) int64, got {tuple(interval.shape)}")
    if interval.shape[0] != logits.shape[0] or interval.shape[-2:] != logits.shape[-2:]:
        raise ValueError(
            f"interval {tuple(interval.shape)} does not describe logits "
            f"{tuple(logits.shape)}; this term reads the full-resolution head only")
    max_share = float(max_share)
    if not 0.0 < max_share <= 1.0:
        raise ValueError(f"max_share must be in (0, 1], got {max_share}")

    n_cls = int(logits.shape[1])
    band = interval_penalty_band(interval, n_cls)                      # (B,H,W)
    cols_b = band.any(dim=1)                                           # (B,W)
    measured = float(cols_b.float().mean())
    if not bool(band.any()):
            # ``* 0.0`` rather than a bare zero: a constant would detach the graph here.
        return logits.sum() * 0.0, measured

    lo, hi = interval[:, 0], interval[:, 1]
    # A degenerate [0,0] outside the band keeps the comparison well-defined; ``& band`` empties it.
    lo_b = torch.where(band, lo, torch.zeros_like(lo))
    hi_b = torch.where(band, hi, torch.zeros_like(hi))
    idx = torch.arange(n_cls, device=logits.device).view(1, n_cls, 1, 1)
    in_set = (idx >= lo_b.unsqueeze(1)) & (idx <= hi_b.unsqueeze(1)) & band.unsqueeze(1)

    # Per-PIXEL conditional; non-band pixels are zeroed first, since an all -inf softmax is NaN.
    band_c = band.unsqueeze(1)                                         # (B,1,H,W)
    safe = torch.where(band_c.expand_as(logits), logits.float(),
                       torch.zeros((), dtype=torch.float32, device=logits.device))
    q = safe.masked_fill(band_c & ~in_set, float("-inf")).softmax(dim=1)
    q = q * in_set.to(q.dtype)                                         # (B,C,H,W)
    n_rows = band.sum(dim=1).to(q.dtype)                               # (B,W) rows per column
    share = q.sum(dim=2) / n_rows.clamp_min(1.0).unsqueeze(1)          # (B,C,W)
    over = (share - max_share).clamp_min(0.0).square().sum(dim=1)      # (B,W)

    cols = cols_b.to(over.dtype)                                       # (B,W)
    n_cols = cols.sum(dim=1)                                           # (B,)
    per_image = (over * cols).sum(dim=1) / n_cols.clamp_min(1.0)
    keep = (n_cols > 0).to(per_image.dtype)
    return (per_image * keep).sum() / keep.sum().clamp_min(1.0), measured


# ---- gated pseudo-label self-training ------------------------------------------------

from octtta.coteach import (boundary_weights_to_pixels,  # noqa: E402
                            column_agreement, coteach_spec, dmt_boundary_weights,
                            full_pass_remainder_spec, pseudo_pool_flag,
                            ramp_factor, soft_column_weights, weighted_pixel_ce)

def needs_class_weights(loss_cfg: dict) -> str | None:
    """The weighting scheme a term asks for by name, if any."""
    for term in loss_cfg.get("terms", []) or []:
        spec = term.get("class_weights")
        if isinstance(spec, str) and spec in _WEIGHT_SCHEMES:
            return spec
    return None


def has_boundary_term(loss_cfg: dict) -> bool:
    return any(str(t.get("name", "")).lower() in ("boundary", "surface")
               and float(t.get("weight", 1.0)) != 0.0
               for t in (loss_cfg.get("terms") or []))








# ---- cohort-stratified truncation of the val set --------------------------------------








# ---- parallel validation scoring ------------------------------------------------------










# ---- monitor validation ---------------------------------------------------------------


def scalar_report_fields() -> set[str]:
    """Names on :class:`EvalReport` that are a single number, i.e. selectable."""
    from dataclasses import fields as dc_fields

    return {f.name for f in dc_fields(EvalReport) if str(f.type) in ("float", "int")}


TUNE_SELECT_METRICS = frozenset({"none", "all10"})


def resolve_monitor(cfg: dict) -> str:
    """Select an EvalReport scalar only for the all10 server validation path."""
    eval_cfg = cfg.get("eval") or {}
    select = str(eval_cfg.get("tune_select_metric", "none"))
    if select not in TUNE_SELECT_METRICS:
        raise ValueError("eval.tune_select_metric must be 'none' or 'all10'")
    for key in ("tune_root", "tune_labels_sha256", "tune_max_images"):
        if eval_cfg.get(key) is not None:
            raise ValueError(f"eval.{key} must be null in the Final workflow")
    return str((cfg.get("runtime") or {}).get("monitor", "challenge_score"))


def validate_monitor(monitor: str, mode: str) -> None:
    """Fail at startup on an unselectable ``runtime.monitor``."""
    known = scalar_report_fields()
    if monitor not in known:
        raise ValueError(
            f"runtime.monitor={monitor!r} is not a scalar field of EvalReport; "
            f"valid: {sorted(known)}"
        )
    if mode not in ("max", "min"):
        raise ValueError(f"runtime.monitor_mode={mode!r} must be 'max' or 'min'")










# ---- trainer --------------------------------------------------------------------------


class Trainer(TrainStateMixin, TrainDataMixin):
    """Owns everything a run needs, so the preemption handler can be a one-liner."""

    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        runtime = cfg.get("runtime", {})
        train_cfg = cfg.get("train", {})
        self.eval_cfg = dict(cfg.get("eval") or {})
        self.select_mode = str(self.eval_cfg.get("tune_select_metric", "none"))
        want = str(runtime.get("device", "cuda"))
        if want.startswith("cuda") and not torch.cuda.is_available():
            print("[train] cuda requested but unavailable; falling back to cpu")
            want = "cpu"
        self.device = torch.device(want)
        set_seed(int(runtime.get("seed", 0)),
                 deterministic=bool(runtime.get("deterministic", True)),
                 cudnn_benchmark=bool(runtime.get("cudnn_benchmark", False)))
        self.monitor_field = resolve_monitor(cfg)
        monitor_mode = str(runtime.get("monitor_mode", "max"))
        validate_monitor(self.monitor_field, monitor_mode)
        self.finetune_from = resolve_finetune_source(cfg, args)
        self.run_kind = "finetune" if self.finetune_from else "train"
        self.run_dir = resolve_run_dir(cfg)
        self.fingerprint = config_fingerprint(cfg, CONFIG_VOLATILE_KEYS)
        self.ckpt = CheckpointManager(
            self.run_dir / "checkpoints", monitor=self.monitor_field,
            mode=monitor_mode, save_top_k=int(runtime.get("save_top_k", 3)))
        self.weights_only = checkpoint_weights_only(cfg)
        if self.weights_only:
            print("[ckpt] checkpoint.weights_only=true: cannot resume from these payloads "
                  f"(dropped: {list(WEIGHTS_ONLY_DROP_KEYS)})", flush=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.epochs = int(train_cfg.get("epochs", 1))
        self.stop_epoch = train_cfg.get("stop_epoch")
        self.stop_epoch = int(self.stop_epoch) if self.stop_epoch else None
        if self.stop_epoch is not None and self.stop_epoch < 1:
            raise ValueError("train.stop_epoch must be >= 1")
        self.max_steps = train_cfg.get("max_steps")
        self.max_steps = int(self.max_steps) if self.max_steps else None
        self.grad_clip = float(train_cfg.get("grad_clip", 0.0))
        self.log_every = max(1, int(runtime.get("log_every", 50)))
        self.amp = bool(runtime.get("amp", True)) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16
        self.postproc = postproc_cfg_or_none(cfg)
        if self.eval_cfg.get("tta", False) is not False:
            raise ValueError("eval.tta must be false in the release workflow")
        self.eval_plan = infer_plan_from_config(cfg)
        self.consistency = consistency_spec(cfg)
        self.interval_share = interval_share_penalty_spec(cfg)
        self.coteach = coteach_spec(cfg)
        self.finetune_from_b = getattr(args, "finetune_from_b", None)
        if self.coteach is not None and not self.finetune_from_b:
            raise RuntimeError("train.coteach needs --finetune-from-b")
        if self.coteach is None and self.finetune_from_b:
            raise RuntimeError("--finetune-from-b was given but train.coteach is off")
        self.model_b = self.ema_b = self.optimizer_b = self.scheduler_b = None
        self.criterion_b = None
        self.cfg_b: dict | None = None
        self._selftrain_loader = None
        self._selftrain_iter = None
        self._configure_validation(runtime)
        self._build_data()
        self._build_model()
        self.epoch = 0
        self.global_step = 0
        self.n_full_selectable = 0
        self.n_full_rejected = 0
        self.best_score = -math.inf if self.ckpt.mode == "max" else math.inf
        self.last_topology = float("inf")
        self._logged_start_lr = False
        self._resumed_from_last = False
        self._resumed_global_step: int | None = None
        self._repair_pair_from_manifest()
        self._resume()
        self._resume_b()
        install_preemption_handler(self._emergency_save)

    def _configure_validation(self, runtime: dict) -> None:
        """Resolve the two-tier validation schedule: ``eval.full_every`` scores the whole
        cohort and is the only pass that may select; ``eval.fast_every`` is a progress trace."""
        cfg = self.eval_cfg
        self.val_every = max(1, int(runtime.get("val_every_epochs", 1)))
        self.full_every = max(1, int(cfg.get("full_every", self.val_every)))
        fast = cfg.get("fast_every", 0)
        self.fast_every = max(0, int(fast or 0))
        self.fast_max_images = int(cfg.get("fast_max_images", 16))
        self.select_on_fast = bool(cfg.get("select_on_fast", False))
        self.postproc_warmup = int(cfg.get("postproc_warmup_epochs", 1))
        self.postproc_gate_topology = float(cfg.get("postproc_gate_topology", 0.20))

        workers = cfg.get("metric_workers")
        if workers is None:
            # Leave one core for GPU feeding; past 16 the pickling of probabilities dominates.
            workers = min(16, max(1, cpu_budget() - 1))
        # Preloading this module into the forkserver is what keeps the pool faster than serial.
        self.pool = (WorkerPool(int(workers), preload=("octtta.eval.report",))
                     if self.select_mode == "all10" else None)


    def _unlabeled_pass_length(self, data_cfg: dict, partial_cfg: dict, pseudo, *,
                               sup_batch: int) -> int | None:
        """``samples_per_epoch: unlabeled_pass`` resolved, or ``None`` when unused: a
        prediction read off each staged root's ``index.json``, which
        :meth:`_assert_full_unlabeled_pass` then checks against the pool that was loaded."""
        if not bool(partial_cfg.get("enabled", False)):
            return None
        if str(partial_cfg.get("samples_per_epoch", "")) != UNLABELED_PASS:
            return None
        if pseudo is None:
            return None                        # _samples_per_epoch raises with the reason
        roots = _unlabeled_local_roots(data_cfg)
        if not roots:
            raise ValueError(
                f"data.partial_pool.samples_per_epoch={UNLABELED_PASS!r} derives the epoch "
                f"from the staged unlabelled pool, and data.unlabeled_roots is empty.")
        if getattr(pseudo, "include_release_unlabeled", False):
            raise ValueError(
                f"data.partial_pool.samples_per_epoch={UNLABELED_PASS!r} cannot be derived "
                f"with train.*.include_release_unlabeled: true. The pseudo-label pool is "
                f"then the staged roots PLUS the release's own unlabelled directories, "
                f"which are not in any index this reads -- the derived epoch would be short "
                f"by exactly the frames that made this recipe's hand-written 379520 wrong. "
                f"Set it false, or give samples_per_epoch a number.")
        n_unlabeled = sum(unlabeled_local.index_frame_count(r) for r in roots)
        remainder = getattr(pseudo, "full_pass_remainder", None)
        dropped = unlabeled_pass_drop(n_unlabeled, pseudo_batch=int(pseudo.batch_size),
                                      remainder=remainder)
        value = unlabeled_pass_samples_per_epoch(
            n_unlabeled, pseudo_batch=int(pseudo.batch_size), every=int(pseudo.every),
            sup_batch=int(sup_batch), remainder=remainder)
        print("[data] " + format_unlabeled_pass_line(
            n_unlabeled, pseudo_batch=int(pseudo.batch_size), dropped=dropped,
            epochs=int(self.epochs)))
        print(f"[data] samples_per_epoch derived from the unlabelled pool: "
              f"{n_unlabeled} frames - {dropped} declared remainder / batch "
              f"{pseudo.batch_size} x every {pseudo.every} x supervised batch "
              f"{sup_batch} = {value} draws ({[str(r) for r in roots]})")
        return value

    def _assert_full_unlabeled_pass(self, pseudo, n_pool: int, samples_per_epoch: int,
                                    sup_batch: int) -> None:
        """``assert_full_pass``: make "one full pass over the unlabelled pool" checkable
        against the steps the LOOP takes, refusing an undeclared ``drop_last`` remainder."""
        if not getattr(pseudo, "assert_full_pass", False):
            return
        if int(self.epochs) != 1:
            raise RuntimeError(
                f"assert_full_pass is on with train.epochs={self.epochs}. The unlabelled "
                "loader is shuffle=True and RE-SEEDED every epoch, so N epochs are N "
                "independent random passes over the same pool, not N passes that cover it "
                "-- 'one full pass' is only a statement that can be true at epochs=1. Set "
                "epochs: 1, or drop the claim.")
        if samples_per_epoch % sup_batch:
            raise RuntimeError(
                f"assert_full_pass: samples_per_epoch={samples_per_epoch} is not a whole "
                f"number of supervised batches of {sup_batch} "
                f"({samples_per_epoch / sup_batch:.4f}). The last batch is short, so the "
                "step count is not the arithmetic the recipe's comment states.")
        sup_steps = samples_per_epoch // sup_batch
        # The steps the LOOP takes, not the ones the config asks for: ``max_steps`` breaks out.
        max_steps = getattr(self, "max_steps", None)
        effective_sup_steps = (sup_steps if max_steps is None
                               else min(sup_steps, int(max_steps)))
        teach_steps = -(-effective_sup_steps // int(pseudo.every))
        # ``drop_last=True``, so a teaching step always consumes a WHOLE batch.
        frames_taught = teach_steps * int(pseudo.batch_size)
        u_batches = -(-n_pool // int(pseudo.batch_size))      # ceil, i.e. what a full pass costs
        remainder = getattr(pseudo, "full_pass_remainder", None)
        dropped = n_pool % int(pseudo.batch_size)
        problems = []
        # Against the pool THAT WAS LOADED: the derivation read index.json, this reads len(pool_u).
        declared = 0
        if dropped:
            if remainder is None or not remainder.allow:
                problems.append(
                    f"the unlabelled loader runs drop_last=True and {n_pool} % "
                    f"{pseudo.batch_size} = {dropped}, so {dropped} frame(s) are silently "
                    f"dropped every epoch. Declare them "
                    f"(train.*.full_pass_remainder: {{allow: true, max_frames: N}} with "
                    f"N >= {dropped}) or change the batch size")
            elif dropped > int(remainder.max_frames):
                problems.append(
                    f"the pool leaves a remainder of {dropped} frame(s) at batch "
                    f"{pseudo.batch_size}, more than the {remainder.max_frames} this "
                    f"recipe declared in train.*.full_pass_remainder.max_frames -- the "
                    f"bound is what keeps the drop a decision")
            else:
                declared = dropped
        if max_steps is not None and int(max_steps) < sup_steps:
            problems.append(
                f"train.max_steps={int(max_steps)} truncates the epoch to "
                f"{effective_sup_steps} supervised steps out of {sup_steps}, so the pass "
                f"covers {frames_taught} of {n_pool} frames. (--dry-run injects "
                f"train.max_steps=3: smoke-test this recipe with "
                f"train.coteach.assert_full_pass=false on the command line, which wins "
                f"over the injected overrides)")
        if frames_taught != n_pool - declared:
            problems.append(
                f"the epoch teaches on {frames_taught} frames -- {teach_steps} pseudo-"
                f"label batches of {pseudo.batch_size} (samples_per_epoch "
                f"{samples_per_epoch} / supervised batch {sup_batch} = {sup_steps} steps, "
                f"effective {effective_sup_steps}, every={pseudo.every}) -- but the pool "
                f"is {n_pool} frames = {u_batches} batches"
                + (f" and the recipe declares a remainder of {declared} frame(s), so the "
                   f"epoch must teach on {n_pool - declared}" if declared else ""))
        self.full_pass_audit = {
            "claimed": True,
            "n_unlabeled_pool": int(n_pool),
            "samples_per_epoch": int(samples_per_epoch),
            "supervised_batch": int(sup_batch),
            "supervised_steps": int(sup_steps),
            "max_steps": None if max_steps is None else int(max_steps),
            "effective_supervised_steps": int(effective_sup_steps),
            "every": int(pseudo.every),
            "teach_steps": int(teach_steps),
            "pseudo_batch": int(pseudo.batch_size),
            "frames_taught": int(frames_taught),
            "unlabeled_batches": int(u_batches),
            "remainder": int(dropped),
            "remainder_declared_max": (None if remainder is None or not remainder.allow
                                       else int(remainder.max_frames)),
            "release_unlabeled": getattr(self, "release_unlabeled_audit", None),
        }
        if problems:
            raise RuntimeError(
                "assert_full_pass is on -- this run claims one full pass over the "
                "unlabelled pool in one epoch -- and the arithmetic does not hold: "
                + "; ".join(problems)
                + ". Fix samples_per_epoch (or the pool, or max_steps) so the two numbers "
                  "are equal; refusing to start rather than training a pass that is short.")
        tail = (f"= the whole pool ({n_pool}); max_steps={max_steps}, drop_last drops 0"
                if not declared else
                f"= the pool ({n_pool}) minus the {declared} frame(s) this recipe "
                f"declares in train.*.full_pass_remainder (max_frames="
                f"{remainder.max_frames}); max_steps={max_steps}")
        print(f"[data] full unlabelled pass asserted: {teach_steps} teaching steps x "
              f"batch {pseudo.batch_size} = {frames_taught} frames {tail}")
        print("[data] " + format_unlabeled_pass_line(
            n_pool, pseudo_batch=int(pseudo.batch_size), dropped=declared,
            epochs=int(self.epochs)))

    @staticmethod
    def _val_loader(dataset: torch.utils.data.Dataset, workers: int) -> DataLoader:
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                          collate_fn=native_collate, pin_memory=False)

    def _build_model(self) -> None:
        cfg = self.cfg
        model = build_model(cfg["model"]).to(self.device)
        self.model = model

        # Apply the encoder plan before building the optimizer.
        self.finetune_plan: list | None = None
        plan_cfg = finetune_plan_cfg(cfg)
        if plan_cfg is not None:
            from octtta.models.vit_fpn import (
                apply_finetune_plan,
                format_finetune_plan,
                plan_finetune,
            )

            self.finetune_plan = plan_finetune(model, **plan_cfg)
            apply_finetune_plan(model, self.finetune_plan)
            print(f"[finetune] phase {plan_cfg['phase']} freezing plan "
                  f"("
                  f"enc_lr={plan_cfg['enc_lr']:.3e}, dec_lr={plan_cfg['dec_lr']:.3e}, "
                  f"layer_decay={plan_cfg['layer_decay']}, "
                  f"unfreeze_last_blocks={plan_cfg['unfreeze_last_blocks']}):")
            print(format_finetune_plan(model, self.finetune_plan), flush=True)

        print(f"[model] {cfg['model'].get('name')}  {count_parameters(model):,} params "
              f"on {self.device}")

        loss_cfg = cfg.get("loss", {})
        scheme = needs_class_weights(loss_cfg)
        weights = None
        if scheme:
            weights = self._class_weights(scheme)
        if has_boundary_term(loss_cfg):
            from octtta.losses import verify_backends

            dev = verify_backends()
            print(f"[loss] EDT backends agree to {dev:.2e}")
        self.criterion = build_loss(loss_cfg, class_weights=weights).to(self.device)
        print(f"[loss] {self.criterion!r}")

        if self.consistency is not None:
            self._check_consistency_compatible(model)
            print(f"[loss] A-Band consistency ON: L_clean + "
                  f"{self.consistency.weight_dirty:g}*L_dirty + "
                  f"{self.consistency.weight_kl:g}*KL(p_dirty || sg(p_clean)) "
                  f"[full-resolution head, T=1]", flush=True)
        if self.interval_share is not None and True:
            print(f"[loss] interval share penalty ON (2026-08-31): "
                  f"{self.interval_share!r} -- on interval-supervised pixels only, "
                  f"per column, full-resolution head, student A", flush=True)

        train_cfg = cfg.get("train", {})
        opt_cfg = train_cfg.get("optimizer") or {}
        if self.finetune_plan is None:
            self.optimizer = build_optimizer(opt_cfg, self.model)
            self.lr_group_names = ["decay", "no_decay"][:len(self.optimizer.param_groups)]
        else:
            # engine.build_optimizer is one LR for the whole model by construction.
            self.optimizer = build_grouped_optimizer(opt_cfg, model, self.finetune_plan)
            self.lr_group_names = list(
                getattr(self.optimizer, "octtta_group_names", []))
            asked = opt_cfg.get("lr")
            if asked is not None and True:
                agrees = math.isclose(float(asked), plan_cfg["dec_lr"], rel_tol=1e-9)
                print(f"[finetune] train.optimizer.lr={float(asked):.3e} takes no part: "
                      f"with train.finetune set, every group's LR comes from "
                      f"enc_lr/dec_lr/layer_decay"
                      + ("." if agrees else ""), flush=True)
                if not agrees:
                    # Group-specific rates take precedence over the scalar optimizer rate.
                    print("!" * 86)
                    print(f"!! [lr] train.optimizer.lr={float(asked):.3e} DISAGREES with "
                          f"train.finetune.dec_lr={plan_cfg['dec_lr']:.3e}.")
                    print(f"!! [lr] The run uses the LADDER: decoder "
                          f"{plan_cfg['dec_lr']:.3e}, encoder {plan_cfg['enc_lr']:.3e} x "
                          f"{plan_cfg['layer_decay']}^depth. The requested "
                          f"{float(asked):.3e} is NOT applied anywhere.")
                    print("!! [lr] Set train.finetune group rates to change the learning rate.")
                    print("!" * 86, flush=True)
        assert_optimizer_covers_trainable(self.optimizer, model)
        #: The LR the CONFIG asked for, per group, captured before a state dict can load over it.
        self.configured_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
        steps_per_epoch = max(1, len(self.train_loader))
        if self.max_steps:
            steps_per_epoch = min(steps_per_epoch, self.max_steps)
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = max(1, steps_per_epoch * self.epochs)
        self.scheduler = build_scheduler(
            train_cfg.get("scheduler") or {}, self.optimizer,
            total_steps=self.total_steps, steps_per_epoch=steps_per_epoch)

        ema_cfg = train_cfg.get("ema") or {}
        self.ema = (ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))
                    if ema_cfg.get("enabled", False) else None)
        if self.coteach is not None:
            if self.ema is None:
                raise RuntimeError(
                    "train.coteach needs train.ema.enabled: the EMA shadow is the weight "
                    "this run selects on and ships (validate() scores self.ema.module and "
                    "_payload publishes it), whatever train.coteach.teacher says the "
                    "teacher is. Without it the run would train two students and publish "
                    "raw weights nothing was ever scored on.")
            self._build_model_b(weights, train_cfg, steps_per_epoch)

    def _build_model_b(self, class_weights, train_cfg: dict, steps_per_epoch: int) -> None:
        """The second student, built from ITS own checkpoint's config: same data, tiles and
        normalisation (asserted), its own loss, optimiser, schedule and EMA."""
        from octtta.engine import build_optimizer, build_scheduler
        from octtta.losses import build_loss
        from octtta.models import build_model

        src = Path(str(self.finetune_from_b))
        payload = torch.load(str(src), map_location="cpu", weights_only=False)
        cfg_b = payload.get("config")
        if not cfg_b:
            raise ValueError(f"{src} carries no config; the second model cannot be rebuilt")
        norm_a = (self.cfg.get("data") or {}).get("normalize")
        norm_b = (cfg_b.get("data") or {}).get("normalize")
        if norm_b != norm_a:
            raise RuntimeError("train.coteach: the two models normalise their input differently "
                               f"({norm_a} vs {norm_b}); one batch cannot feed both")
        if has_boundary_term(cfg_b.get("loss", {})) and not has_boundary_term(self.cfg.get("loss", {})):
            raise RuntimeError("train.coteach: model B's loss has a boundary term but this run's "
                               "dataset computes no distance maps (this run's loss has none)")
        model_b = build_model(cfg_b["model"]).to(self.device)
        self.model_b = model_b
        self.cfg_b = dict(cfg_b)
        # B is fed from two domains per teaching step, so a BatchNorm would leak the unlabeled one.
        _assert_no_batchnorm(model_b, "train.coteach feeds student B a labelled clean batch "
                                      "and an unlabeled corrupted one every teaching step")
        self.criterion_b = build_loss(cfg_b.get("loss", {}), class_weights=class_weights).to(self.device)

        opt_cfg = dict(train_cfg.get("optimizer") or {})
        if self.coteach.lr_b is not None:
            opt_cfg["lr"] = float(self.coteach.lr_b)
        self.optimizer_b = build_optimizer(opt_cfg, model_b)
        assert_optimizer_covers_trainable(self.optimizer_b, model_b)
        self.scheduler_b = build_scheduler(
            train_cfg.get("scheduler") or {}, self.optimizer_b,
            total_steps=self.total_steps, steps_per_epoch=steps_per_epoch)
        ema_cfg = train_cfg.get("ema") or {}
        self.ema_b = ModelEMA(model_b, decay=float(ema_cfg.get("decay", 0.999)))
        loaded = load_finetune_weights(payload, model_b, self.ema_b)
        n_train = sum(int(p.numel()) for p in model_b.parameters() if p.requires_grad)
        n_all = count_parameters(model_b)
        print(f"[model B] {cfg_b['model'].get('name')}  {n_all:,} params "
              f"seeded from {src.name} ({loaded}); lr={opt_cfg.get('lr')}  "
              f"coteach={self.coteach!r}", flush=True)



    # --------------------------------------------------------------- the "last" pair --
    # A new generation is written and the manifest replaced last, so no complete pair is lost.

    #: How many generations survive a commit; pruning runs after the manifest is replaced.









    def _coteach_terms(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Cross-teach on the next clean/dirty unlabelled batch."""
        ct = self.coteach
        ub = self._next_selftrain_batch()
        u_clean = ub["image"].to(self.device, non_blocking=True)
        u_dirty = ub["image_dirty"].to(self.device, non_blocking=True)
        if "orig_hw" not in ub:
            raise RuntimeError("unlabelled batch has no original frame geometry")
        teach_a, teach_b = self.ema, self.ema_b
        teach_a.eval()
        teach_b.eval()
        with torch.no_grad(), self._autocast():
            t_a = torch.softmax(main_head(teach_a(u_clean)).float(), dim=1)
            t_b = torch.softmax(main_head(teach_b(u_clean)).float(), dim=1)
        pseudo_a, pseudo_b = t_a.argmax(dim=1), t_b.argmax(dim=1)
        target_a, target_b = t_b, t_a
        valid = padding_valid_mask(ub.get("pad_hw"), tuple(u_clean.shape[-2:]),
                                   u_clean.device)
        if ct.gate == "column":
            w_a, logs = soft_column_weights(
                pseudo_a, pseudo_b, target_a, quantile=ct.quantile,
                floor=ct.weight_floor, structure_discount=ct.structure_discount,
                valid=valid, check_span=False)
            w_b = soft_column_weights(
                pseudo_a, pseudo_b, target_b, quantile=ct.quantile,
                floor=ct.weight_floor, structure_discount=ct.structure_discount,
                valid=valid, check_span=False)[0]
        else:
            logs = {"ct_agree": float(column_agreement(pseudo_a, pseudo_b, valid).mean())}
        hard_a, hard_b = target_a.argmax(dim=1), target_b.argmax(dim=1)
        with self._autocast():
            s_a = main_head(self.model(u_dirty))
            s_b = main_head(self.model_b(u_dirty))
        if ct.gate == "dmt":
            n_cls = int(t_a.shape[1])
            p_sa = torch.softmax(s_a.detach().float(), dim=1)
            p_sb = torch.softmax(s_b.detach().float(), dim=1)
            dmt_info_a: dict = {}
            dmt_info_b: dict = {}
            wk_a, log_a = dmt_boundary_weights(t_b, t_a, p_sa, spec=ct.dmt,
                                               num_classes=n_cls, valid=valid,
                                               info=dmt_info_a)
            wk_b, log_b = dmt_boundary_weights(t_a, t_b, p_sb, spec=ct.dmt,
                                               num_classes=n_cls, valid=valid,
                                               info=dmt_info_b)
            w_a = boundary_weights_to_pixels(wk_a, hard_a, n_cls)
            w_b = boundary_weights_to_pixels(wk_b, hard_b, n_cls)
            logs.update({f"ct_dmt_a_{k}": v for k, v in log_a.items()})
            logs.update({f"ct_dmt_b_{k}": v for k, v in log_b.items()})
        loss_a = weighted_pixel_ce(s_a, hard_a, w_a, valid)
        loss_b = weighted_pixel_ce(s_b, hard_b, w_b, valid)
        logs["ct_a"] = float(loss_a.detach())
        logs["ct_b"] = float(loss_b.detach())
        return loss_a, loss_b, logs

    def _check_consistency_compatible(self, model: nn.Module) -> None:
        """Consistency batches must not couple independent samples through BatchNorm."""
        _assert_no_batchnorm(model, "train.consistency runs the clean and the corrupted "
                                    "view through ONE forward pass")

    def _class_weights(self, scheme: str) -> torch.Tensor:
        """Frequency weights over the *training* split only; the val images would leak."""
        weights = compute_class_weights(
            [s for s in self.train_ds.samples if s.label_kind == "exact"],
            num_classes=int(self.cfg["data"].get("num_classes", 10)),
            scheme=scheme, cache_path=self.run_dir / "class_weights.json")
        print(f"[loss] class weights ({scheme}): "
              f"{np.round(weights.numpy(), 3).tolist()}")
        return weights

    # ------------------------------------------------------------------------ resume --






    # ------------------------------------------------------------------ checkpointing --


        # The only place the read-outs print on a run that uses its whole budget.




    # ----------------------------------------------------------------------- training --

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                              enabled=self.amp)

    def _log_starting_lr(self) -> None:
        """Print the LR the first optimiser step of this process will actually use, read from
        ``param_groups`` rather than from the config: the failure is the two disagreeing."""
        if self._logged_start_lr or not True:
            self._logged_start_lr = True
            return
        self._logged_start_lr = True
        lr = float(self.optimizer.param_groups[0]["lr"])
        want = self.configured_lrs[0]
        factor = lr / want if want else float("nan")
        print(f"[lr] {self.run_kind}: first optimiser step at epoch {self.epoch}, step "
              f"{self.global_step} runs at lr={lr:.3e} "
              f"(configured {want:.3e} x schedule factor {factor:.3f})", flush=True)
        if len(self.optimizer.param_groups) > 2:
            # The one-group print above is a lie once there is a layer-wise ladder.
            names = self.lr_group_names or [f"g{i}" for i in
                                            range(len(self.optimizer.param_groups))]
            for gname, group, base in zip(names, self.optimizer.param_groups,
                                          self.configured_lrs):
                n = sum(int(p.numel()) for p in group["params"])
                print(f"[lr]   {gname:<26} lr={float(group['lr']):.3e} "
                      f"(configured {base:.3e})  wd={float(group['weight_decay']):g}  "
                      f"{n:,d} params", flush=True)
        if lr <= 0.0:
            print("!! [lr] the effective learning rate is ZERO -- this run cannot change "
                  "the weights. Check train.scheduler / train.epochs, and whether a "
                  "fine-tune was spelled as a resume (--finetune-from).", flush=True)

    def _next_selftrain_batch(self) -> dict:
        """Cycle the unlabeled loader forever; epochs are the supervised pool's unit."""
        if self._selftrain_iter is None:
            self._selftrain_iter = iter(self._selftrain_loader)
        try:
            return next(self._selftrain_iter)
        except StopIteration:
            self._selftrain_iter = iter(self._selftrain_loader)
            return next(self._selftrain_iter)

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        if self.model_b is not None:
            # Said out loud rather than inherited from ``nn.Module``'s default.
            self.model_b.train()
        self.sampler.set_epoch(self.epoch)
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(self.epoch)
        if getattr(self, "_selftrain_loader", None) is not None:
            # The unlabeled stream is epoch-addressed too, or a replayed epoch would restart it.
            self._selftrain_ds.set_epoch(self.epoch)
            self._selftrain_loader.generator.manual_seed(
                self._selftrain_seed + self.epoch)
            self._selftrain_iter = None
        # The epoch is replayed from its first step after a preemption, so the schedule is pinned.
        set_scheduler_step(self.scheduler, self.epoch * self.steps_per_epoch)
        if self.scheduler_b is not None:
            set_scheduler_step(self.scheduler_b, self.epoch * self.steps_per_epoch)
        self.global_step = self.epoch * self.steps_per_epoch

        sums: dict[str, float] = {}
        n_steps = 0
        # Summed from a scalar in every sample dict: workers hold their own copies of the dataset.
        disc_fired = disc_seen = 0
        t0 = time.perf_counter()
        for step, batch in enumerate(self.train_loader):
            if self.max_steps is not None and step >= self.max_steps:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            target = batch["mask"].to(self.device, non_blocking=True)
            fired = batch.get("discaug_fired")
            if fired is not None:
                disc_fired += int(sum(int(v) for v in fired))
                disc_seen += len(fired)
            # Present only when some item carries a partial (interval) label.
            interval = batch.get("interval")
            if interval is not None:
                interval = interval.to(self.device, non_blocking=True)
            dist_maps = batch.get("dist")
            if dist_maps is not None:
                dist_maps = dist_maps.to(self.device, non_blocking=True)

            dirty = None
            if self.consistency is not None:
                if "image_dirty" not in batch:
                    raise RuntimeError(
                        "train.consistency is enabled but the batch has no 'image_dirty'. "
                        "The dataset builds it from the same block (octtta/data/dataset.py "
                        "build_datasets); this means the trainer and the loader disagree "
                        "about whether consistency is on.")
                dirty = batch["image_dirty"].to(self.device, non_blocking=True)

            with self._autocast():
                if dirty is None:
                    pred = self.model(images)
                else:
                    # One forward preserves the paired clean/dirty tensor ordering.
                    both = self.model(torch.cat([images, dirty], dim=0))
                    pred, pred_dirty = split_pair(both, images.shape[0])
                loss, logs = self.criterion(
                    pred, target, epoch=self.epoch, dist=dist_maps,
                    interval=interval,
                )
                if dirty is not None:
                    spec = self.consistency
                    loss_dirty, _ = self.criterion(
                        pred_dirty, target, epoch=self.epoch, dist=dist_maps,
                        interval=interval)
                    valid = padding_valid_mask(batch.get("pad_hw"),
                                               tuple(images.shape[-2:]), images.device)
                    kl = consistency_kl(main_head(pred_dirty), main_head(pred), valid)
                    loss = loss + spec.weight_dirty * loss_dirty + spec.weight_kl * kl
                    logs["clean"] = logs["total"]
                    logs["dirty"] = float(loss_dirty.detach())
                    logs["kl"] = float(kl.detach())
                    logs["total"] = float(loss.detach())

            # ---------------------------- interval anti-collapse penalty --
            # Guarded by ``is not None``, so "off" adds no graph node; fp32, student A only.
            if self.interval_share is not None and interval is not None:
                isp, isp_cols = interval_share_penalty(
                    main_head(pred), interval,
                    max_share=self.interval_share.max_share)
                loss = loss + self.interval_share.weight * isp
                logs["isp"] = float(isp.detach())
                # Beside the penalty, never instead of it: ``isp_cols=0`` means nothing measured.
                logs["isp_cols"] = isp_cols
                logs["total"] = float(loss.detach())

            # ------------------------------------------ co-teaching (two students) --
            loss_b = None
            if self.coteach is not None:
                with self._autocast():
                    pred_b = self.model_b(images)
                    loss_b, logs_b = self.criterion_b(
                        pred_b, target, epoch=self.epoch, dist=dist_maps, interval=interval)
                logs["b_sup"] = float(loss_b.detach())
                if step % self.coteach.every == 0:
                    ct_a, ct_b, ct_logs = self._coteach_terms()
                    # ``weight_ramp_steps=0`` makes this exactly ``self.coteach.weight``.
                    ramp = ramp_factor(self.global_step, self.coteach.weight_ramp_steps)
                    scale = self.coteach.weight * ramp
                    loss = loss + scale * ct_a
                    loss_b = loss_b + scale * ct_b
                    logs.update(ct_logs)
                    if self.coteach.weight_ramp_steps:
                        logs["ct_ramp"] = float(ramp)
                    logs["total"] = float(loss.detach())

            self._log_starting_lr()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            if loss_b is not None:
                self.optimizer_b.zero_grad(set_to_none=True)
                loss_b.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_b.parameters(), self.grad_clip)
                self.optimizer_b.step()
                self.scheduler_b.step()
                self.ema_b.update(self.model_b)

            self.global_step += 1
            n_steps += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            if (step % self.log_every == 0):
                terms = "  ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"[e{self.epoch:03d} s{step:04d}] lr={self.scheduler.get_last_lr()[0]:.2e}"
                      f"  {terms}", flush=True)
            if preempted():                     # belt and braces; the handler already exits
                break

        if self.epoch == 0 and disc_seen:
            print(f"[discaug] epoch0 fired {disc_fired}/{disc_seen} samples "
                  f"({100.0 * disc_fired / disc_seen:.1f}%)", flush=True)
        out = {k: v / max(1, n_steps) for k, v in sums.items()}
        out["steps"] = float(n_steps)
        out["seconds"] = time.perf_counter() - t0
        out["lr"] = float(self.scheduler.get_last_lr()[0])
        return out

    # --------------------------------------------------------------------- validation --

    def _postproc_for_pass(self, topology: float,
                           source: str = "this pass") -> tuple[dict | None, str]:
        """Whether to run the post-processing gate on **this** pass, and why. ``topology``
        must be the caller's own raw sweep of the images that select ``best.pt``: a reading
        one pass behind, or of another cohort, measures a different model."""
        if self.postproc is None:
            return None, "off (config)"
        if self.epoch < self.postproc_warmup:
            return None, f"off (warmup < epoch {self.postproc_warmup})"
        if not np.isfinite(topology):
            # ``nan > x`` is False, so an unmeasured pass would sail through the comparison.
            return None, (f"off (topology {topology} -- {source} measured nothing to "
                          "gate on)")
        if topology > self.postproc_gate_topology:
            return None, (f"off (topology {topology:.3f} > "
                          f"{self.postproc_gate_topology:.3f} measured by THIS pass on "
                          f"{source}: model not layer-like yet)")
        return self.postproc, (f"on (topology {topology:.3f} <= "
                               f"{self.postproc_gate_topology:.3f} measured by THIS pass "
                               f"on {source})")

    def _pass_may_select(self, *, label: str | None, do_full: bool,
                         deploy_consistent: bool, score: float) -> bool:
        """May this validation pass rank ``best.pt``? Only a pass that measured the deployed
        configuration may: not a fast one, not one with post-processing withheld, not a NaN."""
        return bool(label is not None
                    and (do_full or self.select_on_fast)
                    and deploy_consistent
                    and np.isfinite(score))

    def _deploy_consistent(self, pp: dict | None) -> bool:
        """Would a pass post-processed with ``pp`` rank the model the way the submission runs
        it? Only such a pass may select; the others stay useful as progress traces."""
        return pp is self.postproc or pp == self.postproc

    def validate(self, full: bool = True) -> tuple[float, dict | None, str, bool]:
        """Evaluate the held-out split under the deployed post-processing gate."""
        if self.select_mode != "all10":
            raise RuntimeError("validation is disabled by eval.tune_select_metric=none")
        score = -math.inf if self.ckpt.mode == "max" else math.inf
        label = "full" if full else "fast"
        loader = self.val_loader if full else self.val_loader_fast
        if loader is None or len(loader) == 0:
            return score, None, label, False
        target = self.ema.module if self.ema is not None else self.model
        t0 = time.perf_counter()
        raw = evaluate_parallel(target, loader, postproc_cfg=None, device=self.device,
                                amp=self.amp, pool=self.pool, plan=self.eval_plan)
        probe_seconds = time.perf_counter() - t0
        topology = float(raw.topology_violation_rate)
        if full:
            self.last_topology = topology
            pp, why = self._postproc_for_pass(topology, "the tune split")
            print(f"\n[gate] epoch {self.epoch}: raw sweep of the tune split "
                  f"({probe_seconds:.1f}s / {raw.n_images} images) measured topology "
                  f"{topology:.4f}, threshold {self.postproc_gate_topology:.3f} "
                  f"-> post-processing {why}", flush=True)
        else:
            pp, why = None, "off (fast pass)"
        deploy_consistent = self._deploy_consistent(pp)
        if full and not deploy_consistent:
            why += "  != submission -> this pass CANNOT select best.pt (D22)"
        if pp is None:
            report, elapsed = raw, probe_seconds
        else:
            t0 = time.perf_counter()
            report = evaluate_parallel(target, loader, postproc_cfg=pp,
                                       device=self.device, amp=self.amp,
                                       pool=self.pool, plan=self.eval_plan)
            elapsed = time.perf_counter() - t0
        print(f"\n--- {label} validation @ epoch {self.epoch} "
              f"({elapsed:.1f}s for {report.n_images} images, "
              f"{elapsed / max(1, report.n_images):.2f}s/img, "
              f"weights={'ema' if self.ema is not None else 'raw'}, "
              f"postproc={why}, workers={self.pool.workers}) ---")
        print(f"    inference plan: {self.eval_plan.describe()}")
        print(report.format(), flush=True)
        record = report.to_dict()
        record["_pass"] = label
        record["_postproc"] = why
        record["_deploy_consistent"] = bool(deploy_consistent)
        record["_seconds"] = elapsed
        score = float(getattr(report, self.monitor_field))
        if full:
            record["_gate_source"] = "the tune split"
            record["_gate_topology"] = topology
            record["_gate_threshold"] = float(self.postproc_gate_topology)
            record["_gate_probe_seconds"] = probe_seconds
            record["_gate_raw_score"] = float(getattr(raw, self.monitor_field))
            record["_tune_raw_topology"] = topology
            record["_tune_raw_seconds"] = probe_seconds
            record["_tune_raw_score"] = float(getattr(raw, self.monitor_field))
        return score, record, label, deploy_consistent

    def _plan_epoch(self) -> tuple[bool, bool]:
        """``(run full pass, run fast pass)`` for the current epoch."""
        if self.select_mode == "none":
            return False, False
        nth = self.epoch + 1
        do_full = (nth % self.full_every == 0) or (nth == self.epochs) or (nth == self.stop_epoch)
        do_fast = bool(self.fast_every) and not do_full and (nth % self.fast_every == 0)
        return do_full, do_fast

    def fit(self) -> float:
        try:
            return self._fit()
        finally:
            if self.pool is not None:
                self.pool.close()

    def _fit(self) -> float:
        while self.epoch < self.epochs and (self.stop_epoch is None or self.epoch < self.stop_epoch):
            stats = self.train_one_epoch()
            record: dict[str, Any] = {
                "epoch": self.epoch, "global_step": self.global_step, "train": stats,
            }

            do_full, do_fast = self._plan_epoch()
            score, label, deploy_consistent = -math.inf, None, False
            if do_full or do_fast:
                score, report_dict, label, deploy_consistent = self.validate(full=do_full)
                key = self.ckpt.monitor if do_full else f"{self.monitor_field}_fast"
                record[key] = score
                record["val"] = report_dict
            selectable = (self.select_mode == "all10" and self._pass_may_select(
                label=label, do_full=do_full, deploy_consistent=deploy_consistent,
                score=score))
            if do_full and not deploy_consistent:
                self.n_full_rejected += 1
            elif do_full:
                self.n_full_selectable += 1
            saved = None
            if selectable:
                better = (score > self.best_score if self.ckpt.mode == "max"
                          else score < self.best_score)
                if better:
                    self.best_score = score
                payload = self._payload()
                saved = self.ckpt.maybe_save_best(payload, score, self.epoch)
                if saved and better:
                    self._save_b("best")
                record["saved"] = str(saved) if saved else None
                print(f"[epoch {self.epoch}] {self.ckpt.monitor}={score:.4f} "
                      f"(best {self.best_score:.4f})"
                      f"{'  -> saved' if saved else ''}", flush=True)
            else:
                payload = self._payload()
                if label is not None and np.isfinite(score):
                    why_not = ("post-processing differs from submission"
                               if do_full and not deploy_consistent else "fast pass")
                    shown = self.ckpt.monitor if do_full else f"{self.monitor_field}_fast"
                    print(f"[epoch {self.epoch}] {shown}={score:.4f} "
                          f"(progress trace only, selects nothing: {why_not})", flush=True)
            self._save_last_pair(payload)
            append_jsonl(self.metrics_path, record)

            self.epoch += 1

        if self.select_mode == "all10":
            if self.n_full_selectable == 0 and self.n_full_rejected:
                print(f"\n!! [done] NO checkpoint was selected: all "
                      f"{self.n_full_rejected} full validation pass(es) withheld "
                      "post-processing by the warmup/topology gate.", flush=True)
            print(f"\n[done] best {self.ckpt.monitor} = {self.best_score:.4f}")
            print(f"[done] {self.n_full_selectable} full pass(es) could select, "
                  f"{self.n_full_rejected} rejected as != submission")
        else:
            print("[done] validation disabled; last.pt is the product", flush=True)
        print(f"[done] checkpoints in {self.ckpt.directory}")
        return self.best_score


# ---- entry point ----------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Accept both ``train.py CONFIG k=v`` and ``train.py --config CONFIG k=v``: positionals
    are classified by shape, not by position (``dotted.key=value`` is an override)."""
    ap = argparse.ArgumentParser(
        prog="python -m octtta.train",
        usage="%(prog)s [--config CONFIG] [CONFIG] [key.path=value ...] [options]",
        description="Train an OCT layer-segmentation model on the challenge release.")
    ap.add_argument("--config", dest="config_flag",
                    help="path to a config under configs/ (same as the positional form)")
    ap.add_argument("--dry-run", action="store_true",
                    help="3 steps, 1 epoch, 4 val images -- a smoke test, not a run")
    ap.add_argument("--finetune-from", dest="finetune_from", default=None, metavar="CKPT",
                    help="seed from CKPT's model/EMA weights ONLY and start a new run: "
                         "fresh optimiser, fresh schedule at the configured LR, epoch 0. "
                         "This is D2's server path -- it is NOT a resume, and a resume of "
                         "someone else's checkpoint would run at that checkpoint's LR")
    ap.add_argument("--finetune-from-b", dest="finetune_from_b", default=None, metavar="CKPT",
                    help="co-teaching: seed the SECOND student (the CNN) from CKPT's "
                         "weights; requires train.coteach.enabled")
    args, rest = ap.parse_known_args(argv)

    positional = [a for a in rest if not a.startswith("-")]
    unknown_flags = [a for a in rest if a.startswith("-")]
    if unknown_flags:
        ap.error(f"unrecognised argument(s): {' '.join(unknown_flags)}")

    overrides = [a for a in positional if _OVERRIDE_RE.match(a)]
    configs = [a for a in positional if not _OVERRIDE_RE.match(a)]
    bad = [a for a in configs if "=" in a]
    if bad:
        ap.error(f"argument(s) {bad} contain '=' but are not valid dotted overrides "
                 "(expected key.path=value)")
    if args.config_flag:
        configs = [args.config_flag] + configs
    if not configs:
        ap.error("a config is required (positional or --config)")
    if len(configs) > 1:
        ap.error(f"expected one config, got {configs}")

    args.config = configs[0]
    args.overrides = overrides
    return args


def build_config(args: argparse.Namespace) -> dict:
    overrides = list(args.overrides)
    if args.dry_run:
        # Prepended, so an explicit override on the command line still wins.
        overrides = ["train.epochs=1", "train.max_steps=3", "eval.max_images=4",
                     "eval.fast_every=0", "eval.metric_workers=1",
                     "data.loader.num_workers=0", "data.loader.persistent_workers=false",
                     *overrides]
    cfg = get_config(args.config, overrides)
    if args.dry_run:
        exp = cfg.setdefault("experiment", {})
        exp["id"] = f"{exp.get('id') or Path(args.config).stem}-dryrun"
    return cfg


def loaded_audit_record(trainer) -> dict:
    """The ``loaded`` half of ``pools.audit.json``: what the loaders actually produced."""
    return {
        "n_train": len(trainer.train_ds),
        "n_val": len(trainer.val_native),
        "draws_per_epoch": int(trainer.sampler.num_samples),
        "draws_per_epoch_effective": int(len(trainer.sampler)),
        # Whether an epoch COVERS or SAMPLES the pool leaves no other trace, so it goes in here.
        "sampling": trainer.sampling_audit,
        "partial_pool": trainer.partial_summary,
        # The summary counts what was TRAINED on, so a skipped directory is invisible in it.
        "partial_pool_ignored_on_disk": getattr(trainer, "partial_ignored_on_disk", None),
        # "included" / "excluded", never a count: a 0 would read as "the dirs were empty".
        "release_unlabeled": getattr(trainer, "release_unlabeled_audit", None),
        "full_pass": getattr(trainer, "full_pass_audit", None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    run_dir = resolve_run_dir(cfg)
    resolution = resolve_pools(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot(cfg, run_dir)
    record_runenv(run_dir)
    (run_dir / "pools.audit.json").write_text(
        json.dumps(resolution.audit(), indent=2, ensure_ascii=False))
    print(f"[run] {run_dir}")
    print(f"[run] pools: {resolution.names}")
    print(f"[run] config fingerprint: {config_fingerprint(cfg, CONFIG_VOLATILE_KEYS)}")
    trainer = Trainer(cfg, args)
    audit = json.loads((run_dir / "pools.audit.json").read_text())
    audit["loaded"] = loaded_audit_record(trainer)
    (run_dir / "pools.audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))
    try:
        trainer.fit()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"\n!! [oom] torch.cuda.OutOfMemoryError: {exc}", file=sys.stderr,
              flush=True)
        print(f"!! [oom] exiting {EXIT_CUDA_OOM} (EXIT_CUDA_OOM): retryable at a "
              "smaller batch; checkpointed work is untouched.", file=sys.stderr,
              flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return EXIT_CUDA_OOM
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
