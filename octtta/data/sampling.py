"""Deterministic dataset splits and training samplers.

The samplers preserve the Final training streams and cover-pass ordering.
"""
from __future__ import annotations

import hashlib
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from octtta.data.release_dataset import Sample

#: Splitting is stratified over these fields. See :func:`split_samples`.
STRATIFY_FIELDS = ("device", "status")

# --- Splitting -----------------------------------------------------------------------

def stem_hash_u01(stem: str, seed: int) -> float:
    """Deterministic ``[0, 1)`` from a stem."""
    digest = hashlib.blake2b(f"{seed}:{stem}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def patient_key(sample: Sample) -> str | None:
    """Patient/eye identifier, if the layout exposes one."""
    return None


#: ``content_key`` memo, keyed by resolved path; a process cache, not a persistent one.
_CONTENT_KEY_CACHE: dict[str, str] = {}


def content_key(sample: Sample) -> str:
    """Hash of the sample's image **file bytes** -- the one grouping signal this data has."""
    key = str(sample.image.resolve())
    cached = _CONTENT_KEY_CACHE.get(key)
    if cached is None:
        try:
            cached = hashlib.blake2b(sample.image.read_bytes(), digest_size=16).hexdigest()
        except OSError as exc:
            raise FileNotFoundError(
                f"scheme='group_by_content' must read {sample.image} to group duplicates; "
                f"it could not ({exc})") from exc
        _CONTENT_KEY_CACHE[key] = cached
    return cached


def split_samples(
    samples: list[Sample],
    val_fraction: float,
    seed: int,
    scheme: str = "hash_by_stem",
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/val split, stratified over device and status."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if scheme == "hash_by_patient":
        if any(patient_key(s) is None for s in samples):
            raise NotImplementedError(
                "scheme='hash_by_patient' needs subject IDs, which this layout does not "
                "expose; implement patient_key() first (see its docstring)"
            )

        def group_key(s: Sample) -> str:
            return str(patient_key(s))
    elif scheme == "group_by_content":
        group_key = content_key
    elif scheme in ("hash_by_stem", "random"):
        def group_key(s: Sample) -> str:
            return s.stem
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")

    strata: dict[tuple, list[int]] = {}
    groups: dict[str, set[tuple]] = {}
    for i, s in enumerate(samples):
        cell = tuple(getattr(s, f) for f in STRATIFY_FIELDS)
        strata.setdefault(cell, []).append(i)
        groups.setdefault(group_key(s), set()).add(cell)
    # Ranking happens INSIDE a stratum, so a group whose members sit in two strata could be
    # ranked onto opposite sides -- the exact leak this scheme exists to close.
    straddling = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if straddling:
        raise ValueError(
            f"scheme={scheme!r}: {len(straddling)} group(s) straddle the "
            f"{'/'.join(STRATIFY_FIELDS)} strata, so grouping cannot be honoured: "
            f"{dict(list(straddling.items())[:3])}")

    val_idx: set[int] = set()
    for _, idxs in sorted(strata.items()):
        ranked = sorted(idxs, key=lambda i: (stem_hash_u01(group_key(samples[i]), seed),
                                             samples[i].stem))
        n_val = int(round(val_fraction * len(ranked)))
        # With at least two members, keep the stratum on both sides: an empty val cohort
        # scores 0 in the official aggregation rather than being skipped.
        if val_fraction > 0 and len(ranked) >= 2:
            n_val = min(max(n_val, 1), len(ranked) - 1)
        members: dict[str, list[int]] = {}
        order: list[str] = []
        for i in ranked:
            key = group_key(samples[i])
            if key not in members:
                members[key] = []
                order.append(key)
            members[key].append(i)
        taken: list[str] = []
        count = 0
        for key in order:
            if count + len(members[key]) > n_val:
                break                      # stop, never skip: val stays a prefix of the rank
            taken.append(key)
            count += len(members[key])
        if val_fraction > 0 and len(order) >= 2 and not taken:
            taken = [order[0]]             # one group is bigger than the whole val quota
        for key in taken:
            val_idx.update(members[key])

    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    # The post-condition, asserted rather than trusted.
    crossed = {group_key(s) for s in train} & {group_key(s) for s in val}
    if crossed:
        raise AssertionError(
            f"scheme={scheme!r} put {len(crossed)} group(s) on BOTH sides of the split; "
            f"every member of a group must land together. First few: {sorted(crossed)[:3]}")
    return train, val


# --- Sampling ------------------------------------------------------------------------

def resolve_cell_balance(cfg: dict | None) -> None:
    """Accept the fixed identity declaration carried by the Final recipes."""
    if cfg is None or cfg == {"mode": "identity"}:
        return None
    raise ValueError("data.partial_pool.cell_balance must be null or mode: identity")


def make_train_sampler(
    samples: list[Sample],
    oversample_diseased: float = 1.0,
    *,
    seed: int = 0,
) -> Sampler | None:
    """Weighted diseased sampling, or ``None`` for the ordinary permutation."""
    if not samples or float(oversample_diseased) == 1.0:
        return None
    w = np.ones(len(samples), dtype=np.float64)
    diseased = np.array([s.status.lower() == "diseased" for s in samples])
    w[diseased] *= float(oversample_diseased)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(samples), replacement=True, generator=gen,
    )


# --- Sampling mode: draw with replacement, or walk a permutation that covers the pool -

#: The modes ``data.partial_pool.sampling`` accepts.
SAMPLING_MODES = ("multinomial", "cover")


def resolve_sampling_mode(partial_cfg: dict | None) -> str:
    """Which sampler ``data.partial_pool`` asks for. Refuses a block that does not say."""
    cfg = partial_cfg or {}
    if not cfg:
        return "multinomial"
    if "sampling" not in cfg:
        raise ValueError(
            "data.partial_pool.sampling is not set. It decides whether an epoch samples "
            f"the pool with replacement or walks it; allowed: {list(SAMPLING_MODES)}. "
            "Spell it -- octtta/config.py validates no schema, and the difference is "
            "invisible in every log line a run produces (recomputed 2026-08-30: 17.5% of the public "
            "pool never drawn once under 'multinomial' at 800 draws/epoch).")
    mode = str(cfg["sampling"])
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"data.partial_pool.sampling={cfg['sampling']!r} is not one of "
            f"{list(SAMPLING_MODES)}")
    return mode


def make_cover_index(
    samples: list[Sample], *, oversample_diseased: float = 1.0,
) -> np.ndarray:
    """The expanded index list one cover pass permutes."""
    if not samples:
        raise ValueError("cannot build a cover index over an empty pool")
    repeats = int(round(float(oversample_diseased)))
    if repeats < 1:
        raise ValueError("data.oversample_diseased rounds below one repeat")
    idx: list[int] = []
    for i, sample in enumerate(samples):
        n = repeats if sample.status.lower() == "diseased" else 1
        idx.extend([i] * n)
    out = np.asarray(idx, dtype=np.int64)
    if int(np.bincount(out, minlength=len(samples)).min()) < 1:
        raise AssertionError("cover pass omitted a training sample")
    return out


def cover_repeats(oversample_diseased: float) -> int:
    """How many times a diseased sample enters one pass. Reported alongside the pass."""
    return int(round(float(oversample_diseased)))


class ShardedWeightedSampler(Sampler[int]):
    """Single-process epoch sampler; the class name preserves the live caller API."""

    def __init__(
        self,
        dataset_size: int,
        weights: torch.Tensor | Sequence[float] | None = None,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("cannot sample from an empty dataset")
        self.dataset_size = int(dataset_size)
        self.num_samples = int(self.dataset_size if num_samples is None else num_samples)
        if self.num_samples <= 0:
            raise ValueError("an epoch that draws nothing is not an epoch")
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.double)
        if self.weights is not None and self.weights.numel() != self.dataset_size:
            raise ValueError("sampler weights do not cover the dataset")
        if self.weights is None and self.num_samples != self.dataset_size:
            raise ValueError("an unweighted epoch must permute the full dataset")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        if self.weights is not None:
            order = torch.multinomial(self.weights, self.num_samples, replacement=True,
                                      generator=g)
        elif self.shuffle:
            order = torch.randperm(self.dataset_size, generator=g)
        else:
            order = torch.arange(self.dataset_size)
        return iter(order.tolist())


class ShardedCoverSampler(Sampler[int]):
    """A contiguous window of the deterministic whole-pool permutation walk."""

    _STREAM = 0x436F_7665

    def __init__(
        self,
        expanded: torch.Tensor | Sequence[int] | np.ndarray,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        epoch_offset: int = 0,
    ) -> None:
        idx = torch.as_tensor(np.asarray(expanded), dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            raise ValueError("cannot cover an empty pool")
        self.expanded = idx
        self.pass_length = int(idx.numel())
        self.num_samples = int(self.pass_length if num_samples is None else num_samples)
        if self.num_samples <= 0:
            raise ValueError("an epoch that draws nothing is not an epoch")
        self.seed = int(seed)
        self.epoch_offset = int(epoch_offset)
        if self.epoch_offset < 0:
            raise ValueError(f"epoch_offset must be >= 0, got {self.epoch_offset}")
        self.epoch = 0

    @property
    def epochs_per_pass(self) -> float:
        return self.pass_length / self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _permutation(self, cycle: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self._STREAM + int(cycle))
        return torch.randperm(self.pass_length, generator=g)

    def window(self, start: int, count: int) -> torch.Tensor:
        """Dataset indices from the walk starting at global position ``start``."""
        parts: list[torch.Tensor] = []
        pos, left = int(start), int(count)
        while left > 0:
            cycle, offset = divmod(pos, self.pass_length)
            take = min(left, self.pass_length - offset)
            parts.append(self.expanded[self._permutation(cycle)[offset:offset + take]])
            pos += take
            left -= take
        return torch.cat(parts)

    def __iter__(self) -> Iterator[int]:
        start = (self.epoch + self.epoch_offset) * self.num_samples
        order = self.window(start, self.num_samples)
        return iter(order.tolist())
