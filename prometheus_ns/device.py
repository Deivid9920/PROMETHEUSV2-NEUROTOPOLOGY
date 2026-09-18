"""Single source of truth for compute-target selection.

Every device decision in the repository funnels through this module:
CUDA vs CPU selection, thread counts, and autocast policy. No other
module may call torch.cuda.is_available() or change thread settings;
that boundary is what makes the GPU migration (docs/gpu_migration.md)
a configuration-only change.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch

_MAX_INTRAOP_THREADS = 8
_INTEROP_THREADS = 2
_MATMUL_PRECISION = "high"


def get_device() -> torch.device:
    """Return the cuda device when available, cpu otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_compute(cfg: dict | None = None) -> dict:
    """Apply the thread and precision policy for the active device.

    Args:
        cfg: Optional mapping reserved for future overrides.
            No keys are required today; future thread or precision
            overrides would be read from here.

    Returns:
        A summary dict with the applied settings, suitable for writing
        as the first line of logs/metrics.jsonl.
    """
    cpu_count = os.cpu_count() or 1
    intraop = max(1, min(_MAX_INTRAOP_THREADS, cpu_count))
    torch.set_num_threads(intraop)

    try:
        torch.set_num_interop_threads(_INTEROP_THREADS)
        interop = _INTEROP_THREADS
    except RuntimeError:
        # The inter-op pool cannot be resized after the first parallel
        # region; report the active value instead of failing.
        interop = torch.get_num_interop_threads()

    torch.set_float32_matmul_precision(_MATMUL_PRECISION)

    return {
        "device": str(get_device()),
        "intraop_threads": intraop,
        "interop_threads": interop,
        "float32_precision": _MATMUL_PRECISION,
    }


@contextlib.contextmanager
def autocast_ctx(device: torch.device, enabled: bool = True) -> Iterator[None]:
    """Autocast policy for the given device.

    - CUDA: bfloat16 autocast while ``enabled`` is True.
    - CPU: always a null context; the project trains in fp32 on CPU by
      design, and no caller may override that from the outside.

    Usage inside a training step::

        with autocast_ctx(device, enabled=use_bf16):
            logits = model(batch)
    """
    if device.type == "cuda" and enabled:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        with contextlib.nullcontext():
            yield


def device_summary() -> dict:
    """Static description of the compute environment for run logs."""
    cuda = torch.cuda.is_available()
    return {
        "device": str(get_device()),
        "cuda_available": cuda,
        "cuda_name": torch.cuda.get_device_name(0) if cuda else None,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "bf16": torch.cuda.is_bf16_supported() if cuda else False,
    }
