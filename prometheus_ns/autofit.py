"""Hardware auto-fit: automatic model-profile selection and sizing.

Setting ``model.profile: auto`` in config.yaml (or passing
``--profile auto`` / exporting ``PROMETHEUS_NS_PROFILE=auto``) makes
the pipeline probe the machine it runs on and pick the profile and
batch geometry that use the available compute best:

- a CUDA accelerator with enough VRAM unlocks the ``small`` or
  ``large`` profiles;
- a CPU-only box resolves to ``nano``, whose fp32 throughput is the
  practical ceiling without an accelerator;
- ``fit_memory`` shrinks the micro-batch (doubling gradient
  accumulation) whenever the configured step would not fit the RAM/VRAM
  budget, so a small laptop can train any profile without an
  out-of-memory crash.

Precedence is explicit over automatic: a concrete profile name on the
CLI wins over the environment variable, which wins over config.yaml.
An explicit name never triggers hardware probing, so existing
workflows behave exactly as before. Thresholds live in the optional
``autofit`` section of config.yaml (see ``_DEFAULTS`` for the keys).

Apple Silicon (MPS) is reported by the probe but not acted on: the
device policy (prometheus_ns/device.py) routes it to CPU until the GPU
migration (docs/gpu_migration.md) extends the accelerator matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import psutil
import torch

from prometheus_ns import cfg_get

_GIB = 1024**3
_ENV_VAR = "PROMETHEUS_NS_PROFILE"
_DEFAULTS = {
    "large_min_vram_gb": 40.0,   # CUDA VRAM before auto may select large
    "small_min_vram_gb": 16.0,   # CUDA VRAM before auto may select small
    "cpu_max_profile": "nano",   # highest profile allowed without CUDA
    "ram_headroom_cpu": 0.5,     # fraction of system RAM usable for tensors
    "ram_headroom_gpu": 0.6,     # fraction of VRAM usable for tensors
}

# fp32 bytes per token per layer per d_model unit: attention q/k/v/out
# working memory plus the SwiGLU sandwich, conservatively over-counted
# so the guard errs towards smaller batches, never towards OOM.
_ACTIVATION_BYTES_PER_ELEMENT = 34
# AdamW training footprint in fp32: 4 (params) + 4 (grads) + 8
# (exp_avg + exp_avg_sq) bytes per parameter.
_TRAINING_BYTES_PER_PARAM = 16


def _thresholds(cfg: dict | None) -> dict:
    """Merge the optional ``autofit`` config section over the defaults."""
    merged = dict(_DEFAULTS)
    section = cfg_get(cfg or {}, "autofit", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("config section 'autofit' must be a mapping")
    for key, value in section.items():
        if key in merged:
            merged[key] = value
    if merged["cpu_max_profile"] not in ("nano", "small"):
        raise ValueError(
            "autofit.cpu_max_profile must be 'nano' or 'small': the large "
            "profile is a CUDA-only target (see docs/gpu_migration.md)"
        )
    return merged


def detect_hardware() -> dict:
    """Probe the machine: effective CPUs, RAM, accelerator and disk.

    The probe is read-only and cheap; ``cpus`` honours the process
    affinity (cgroups/containers) rather than the raw core count, and
    VRAM is the device total reported by the CUDA driver.
    """
    try:
        cpus = len(os.sched_getaffinity(0))  # POSIX: cores this process may use
    except AttributeError:
        cpus = os.cpu_count() or 1
    ram_gb = 0.0
    try:
        ram_gb = psutil.virtual_memory().total / _GIB
    except Exception:  # pragma: no cover - psutil is a pinned dependency
        try:
            ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / _GIB
        except (ValueError, OSError, AttributeError):
            ram_gb = 0.0
    cuda = torch.cuda.is_available()
    try:
        disk_free_gb = shutil.disk_usage(Path.cwd()).free / _GIB
    except OSError:  # pragma: no cover - cwd always exists in practice
        disk_free_gb = 0.0
    return {
        "accelerator": "cuda" if cuda else "cpu",
        "cuda_name": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / _GIB, 2) if cuda else 0.0,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpus": cpus,
        "ram_gb": round(ram_gb, 2),
        "disk_free_gb": round(disk_free_gb, 2),
    }


def select_profile(hw: dict, cfg: dict | None = None) -> tuple[str, str]:
    """Map a :func:`detect_hardware` probe to a profile name and reason.

    CUDA devices are graded by VRAM (>= ``large_min_vram_gb`` selects
    ``large``, >= ``small_min_vram_gb`` selects ``small``); everything
    else resolves to ``cpu_max_profile`` because fp32 CPU throughput
    makes bigger profiles impractical rather than impossible. The
    returned reason string is human-readable and lands in run logs so
    every automatic choice is auditable after the fact.
    """
    t = _thresholds(cfg)
    vram = float(hw.get("vram_gb") or 0.0)
    ram = float(hw.get("ram_gb") or 0.0)
    if hw.get("accelerator") == "cuda":
        if vram >= float(t["large_min_vram_gb"]):
            return "large", f"CUDA with {vram:.0f} GB VRAM >= {float(t['large_min_vram_gb']):.0f} GB large-profile threshold"
        if vram >= float(t["small_min_vram_gb"]):
            return "small", f"CUDA with {vram:.0f} GB VRAM >= {float(t['small_min_vram_gb']):.0f} GB small-profile threshold"
        return "nano", f"CUDA with {vram:.0f} GB VRAM below the {float(t['small_min_vram_gb']):.0f} GB small-profile threshold"
    ceiling = t["cpu_max_profile"]
    return ceiling, (
        f"no CUDA accelerator (cpu-only, {ram:.1f} GB RAM, {hw.get('cpus', '?')} cores): "
        f"fp32 CPU throughput keeps '{ceiling}' as the efficient ceiling; "
        "small/large need a CUDA device (docs/gpu_migration.md)"
    )


def resolve_profile(cfg: dict | None, explicit: str | None = None) -> dict:
    """Resolve the effective profile for this machine.

    Args:
        cfg: Parsed project configuration (may be ``{}``).
        explicit: CLI profile override. Any concrete name wins over
            everything and is passed through untouched (custom profiles
            defined in config.yaml stay usable); the literal ``"auto"``
            forces detection; ``None`` defers to the environment
            variable, then to ``model.profile`` in config.yaml.

    Returns:
        Dictionary with ``profile`` (concrete name), ``source`` (one of
        ``cli``, ``env``, ``config``, ``auto``), the human-readable
        ``reason`` (auto only) and the ``hardware`` probe (auto only).
    """
    cfg = cfg or {}
    if explicit is not None and explicit != "auto":
        return {"profile": explicit, "source": "cli", "reason": None, "hardware": None}

    env = os.environ.get(_ENV_VAR, "").strip()
    if env and env != "auto":
        return {"profile": env, "source": "env", "reason": None, "hardware": None}

    configured = str(cfg_get(cfg, "model.profile", "nano"))
    # An explicit "auto" (CLI or environment) forces detection even when
    # the config names a concrete profile; otherwise auto only applies
    # when config.yaml itself defers.
    if explicit != "auto" and env != "auto" and configured != "auto":
        return {"profile": configured, "source": "config", "reason": None, "hardware": None}

    hw = detect_hardware()
    name, reason = select_profile(hw, cfg)
    print(f"[autofit] profile '{name}' selected for this machine: {reason}")
    return {"profile": name, "source": "auto", "reason": reason, "hardware": hw}


def resolved_profile(cfg: dict | None, explicit: str | None = None) -> str:
    """Name-only convenience wrapper over :func:`resolve_profile`."""
    return resolve_profile(cfg, explicit)["profile"]


def fit_memory(
    hw: dict,
    mc,
    device_type: str,
    micro_batch: int,
    grad_accum: int,
    cfg: dict | None = None,
) -> tuple[int, int, bool, float]:
    """Fit the training step into the memory budget, preserving the batch.

    The static footprint is estimated as ``16`` bytes per parameter
    (fp32 weights, gradients and the two AdamW moments) and the
    activation footprint as ``_ACTIVATION_BYTES_PER_ELEMENT`` bytes per
    (token, layer, d_model) unit. While the estimate exceeds the usable
    budget (``ram_headroom_cpu`` of RAM on CPU, ``ram_headroom_gpu`` of
    VRAM on CUDA) the micro-batch is halved and gradient accumulation
    doubled, keeping ``micro_batch * grad_accum`` — and therefore the
    optimization trajectory length — unchanged.

    Returns:
        ``(micro_batch, grad_accum, adjusted, estimated_gb)``. The
        function only shrinks; a machine that already fits gets its
        configured values back untouched.
    """
    from prometheus_ns.model.config import analytic_param_count

    t = _thresholds(cfg)
    if device_type == "cuda":
        usable = float(hw.get("vram_gb") or 0.0) * float(t["ram_headroom_gpu"])
    else:
        usable = float(hw.get("ram_gb") or 0.0) * float(t["ram_headroom_cpu"])
    budget = int(usable * _GIB)

    params = analytic_param_count(mc)
    static = _TRAINING_BYTES_PER_PARAM * params
    act_per_micro = int(mc.max_seq) * int(mc.d_model) * int(mc.n_layer) * _ACTIVATION_BYTES_PER_ELEMENT

    micro, accum = int(micro_batch), int(grad_accum)
    while micro > 1 and static + micro * act_per_micro > budget:
        micro //= 2
        accum *= 2
    estimated_gb = (static + micro * act_per_micro) / _GIB
    return micro, accum, (micro, accum) != (int(micro_batch), int(grad_accum)), round(estimated_gb, 2)


def recommend(cfg: dict) -> dict:
    """Full auto-fit preview: hardware, resolved profile and step sizing.

    Used by the ``python -m prometheus_ns.autofit`` CLI and by tests to
    answer "what would this machine do?" without touching any state.
    """
    fit = resolve_profile(cfg)
    hw = fit["hardware"] if fit["source"] == "auto" else detect_hardware()
    device_type = "cuda" if hw["accelerator"] == "cuda" else "cpu"

    from prometheus_ns.model.config import model_config_from_yaml

    mc = model_config_from_yaml(cfg, fit["profile"])
    micro = int(cfg_get(cfg, f"train.micro_batch.{device_type}", 16))
    accum = int(cfg_get(cfg, f"train.grad_accum.{device_type}", 4))
    micro, accum, adjusted, estimated_gb = fit_memory(hw, mc, device_type, micro, accum, cfg)
    return {
        "profile": fit["profile"],
        "source": fit["source"],
        "reason": fit["reason"],
        "hardware": hw,
        "device_type": device_type,
        "model_config": dict(vars(mc)),
        "micro_batch": micro,
        "grad_accum": accum,
        "effective_batch": micro * accum,
        "memory_fit_adjusted": adjusted,
        "estimated_step_gb": estimated_gb,
    }


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.autofit [--config PATH]``.

    Prints the hardware probe, the resolved profile and the tuned step
    geometry as JSON, preceded by a one-line human-readable verdict.
    """
    parser = argparse.ArgumentParser(description="Prometheus-NS hardware auto-fit preview")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: repository config)")
    parser.add_argument("--profile", default=None, help="profile override (auto|nano|small|large)")
    args = parser.parse_args()

    from prometheus_ns import load_config

    cfg = load_config(args.config) if args.config else load_config()
    report = recommend(cfg) if args.profile in (None, "auto") else resolve_profile(cfg, args.profile)
    hw = report.get("hardware") or {}
    print(
        f"[autofit] hardware: {hw.get('accelerator', '?')} | {hw.get('cpus', '?')} cores | "
        f"{hw.get('ram_gb', '?')} GB RAM | VRAM {hw.get('vram_gb', 0)} GB | disk free {hw.get('disk_free_gb', '?')} GB"
    )
    if report.get("reason"):
        print(f"[autofit] decision: {report['reason']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
