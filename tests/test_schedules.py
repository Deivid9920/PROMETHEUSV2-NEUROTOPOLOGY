"""Tests for the warmup + cosine decay schedule."""

from __future__ import annotations

import math

from prometheus_ns.train.schedules import WarmupCosine, lr_at_step


def test_warmup_increases_to_peak() -> None:
    total, peak = 1000, 3.0e-4
    warmup_steps = max(1, int(round(total * 0.02)))
    first = lr_at_step(0, total, peak)
    assert 0 < first <= peak
    # Warmup climbs monotonically and peaks on its last step.
    assert lr_at_step(warmup_steps - 2, total, peak) < lr_at_step(warmup_steps - 1, total, peak)
    assert math.isclose(lr_at_step(warmup_steps - 1, total, peak), peak, rel_tol=1e-9)
    # After warmup the cosine decay never returns to the peak.
    assert lr_at_step(warmup_steps, total, peak) <= peak


def test_cosine_decays_to_floor() -> None:
    total, peak, floor_frac = 1000, 3.0e-4, 0.10
    final = lr_at_step(total, total, peak, min_lr_frac=floor_frac)
    assert math.isclose(final, peak * floor_frac, rel_tol=1e-6)
    mid = lr_at_step(total // 2, total, peak, min_lr_frac=floor_frac)
    assert peak * floor_frac < mid < peak


def test_callable_wrapper_matches_function() -> None:
    sched = WarmupCosine(500, 1.0e-3, warmup_frac=0.1, min_lr_frac=0.05)
    for step in (0, 25, 100, 300, 500):
        assert sched(step) == lr_at_step(step, 500, 1.0e-3, 0.1, 0.05)


def test_zero_total_steps_is_safe() -> None:
    assert lr_at_step(0, 0, 1.0e-3) == 0.0
