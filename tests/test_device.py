"""Tests for the compute policy module (device.py)."""

from __future__ import annotations

import torch

from prometheus_ns.device import autocast_ctx, device_summary, get_device, setup_compute


def test_get_device_is_cpu_here() -> None:
    device = get_device()
    assert device.type in ("cpu", "cuda")


def test_setup_compute_returns_summary() -> None:
    summary = setup_compute(None)
    assert set(summary) == {"device", "intraop_threads", "interop_threads", "float32_precision"}
    assert summary["intraop_threads"] >= 1
    assert summary["intraop_threads"] <= 8  # CPU-first thread budget ceiling
    assert summary["float32_precision"] == "high"


def test_autocast_ctx_is_null_on_cpu() -> None:
    device = torch.device("cpu")
    with autocast_ctx(device, enabled=True):
        x = torch.ones(4) * 2
        assert x.sum().item() == 8.0  # plain fp32 arithmetic, no autocast


def test_device_summary_fields() -> None:
    summary = device_summary()
    assert summary["device"] == str(get_device())
    assert isinstance(summary["cuda_available"], bool)
    if summary["cuda_available"]:
        assert isinstance(summary["cuda_name"], str)
    else:
        assert summary["cuda_name"] is None
        assert summary["bf16"] is False
