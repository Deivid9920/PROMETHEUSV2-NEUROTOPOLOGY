"""Learning-rate schedule: linear warmup plus cosine decay to a floor."""

from __future__ import annotations

import math


def lr_at_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float = 0.02,
    min_lr_frac: float = 0.10,
) -> float:
    """Learning rate at ``step`` for a run of ``total_steps`` steps.

    Linear warmup from 0 to ``peak_lr`` during the first
    ``warmup_frac`` fraction of steps, then cosine decay from
    ``peak_lr`` down to ``peak_lr * min_lr_frac`` at the end. A
    ``total_steps`` of zero returns 0.0.
    """
    if total_steps <= 0:
        return 0.0
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    floor = peak_lr * min_lr_frac
    return floor + 0.5 * (peak_lr - floor) * (1.0 + math.cos(math.pi * progress))


class WarmupCosine:
    """Stateful callable around :func:`lr_at_step` for optimizer loops."""

    def __init__(self, total_steps: int, peak_lr: float, warmup_frac: float = 0.02, min_lr_frac: float = 0.10) -> None:
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.warmup_frac = warmup_frac
        self.min_lr_frac = min_lr_frac

    def __call__(self, step: int) -> float:
        return lr_at_step(step, self.total_steps, self.peak_lr, self.warmup_frac, self.min_lr_frac)
