"""Exponential moving average of weights, used for stability and as the UDA teacher.

The shadow is a deepcopy in eval mode with gradients off; float buffers are averaged and
integer ones copied. Decay ramps from 0 as ``decay * (1 - exp(-t / warmup_steps))``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterator

import torch
import torch.nn as nn

__all__ = ["ModelEMA"]

_WRAPPERS = (nn.parallel.DistributedDataParallel, nn.DataParallel)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, _WRAPPERS) else model


class ModelEMA:
    """Shadow copy of ``model`` updated as ``ema = d * ema + (1 - d) * model``.
    ``warmup_steps`` is the time constant of the decay ramp, not a hard cutoff."""

    def __init__(self, model: nn.Module, decay: float = 0.999, *,
                 warmup_steps: int = 1000, device: torch.device | str | None = None) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.step = 0

        self.module = copy.deepcopy(_unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    def decay_at(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-step / self.warmup_steps))

    def current_decay(self) -> float:
        """Decay the *next* :meth:`update` will use -- 0 before any update has run."""
        return self.decay_at(self.step)

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """Fold one optimiser step of ``model`` into the shadow; returns the decay used."""
        d = self.decay_at(self.step)
        self.step += 1

        src = _unwrap(model).state_dict()
        for name, shadow in self.module.state_dict().items():
            new = src[name]
            if shadow.dtype.is_floating_point:
                shadow.mul_(d).add_(new.detach().to(shadow.device, shadow.dtype), alpha=1.0 - d)
            else:
                shadow.copy_(new.detach().to(shadow.device))
        return d

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write the shadow weights into ``model`` (for finishing training on the EMA)."""
        _unwrap(model).load_state_dict(self.module.state_dict())

    def __call__(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.module.parameters()

    def to(self, *args: Any, **kwargs: Any) -> "ModelEMA":
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "ModelEMA":
        self.module.train(mode)
        return self

    def eval(self) -> "ModelEMA":
        self.module.eval()
        return self

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "step": self.step,
                "decay": self.decay, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.step = int(state.get("step", 0))
        self.decay = float(state.get("decay", self.decay))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
