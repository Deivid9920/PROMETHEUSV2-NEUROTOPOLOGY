"""Tests for the hardware auto-fit module (profile selection and sizing)."""

from __future__ import annotations

import pytest

import prometheus_ns.autofit as autofit
from prometheus_ns import cfg_get
from prometheus_ns.autofit import (
    detect_hardware,
    fit_memory,
    resolve_profile,
    resolved_profile,
    select_profile,
)
from prometheus_ns.model.config import ModelConfig, analytic_param_count, model_config_from_yaml
from prometheus_ns.model.count_params import RANGES, count_profile


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep the override variable out of the resolution precedence tests."""
    monkeypatch.delenv(autofit._ENV_VAR, raising=False)


@pytest.fixture
def tiny_mc() -> ModelConfig:
    """A small ModelConfig for memory-fit arithmetic checks."""
    return ModelConfig(
        d_model=64, n_layer=2, n_head=4, n_kv_head=2, d_ff=128, max_seq=64, vocab=512, tie_embeddings=True
    )


def _cpu_box(ram_gb: float = 16.0, cpus: int = 4) -> dict:
    return {
        "accelerator": "cpu",
        "cuda_name": None,
        "vram_gb": 0.0,
        "mps_available": False,
        "cpus": cpus,
        "ram_gb": ram_gb,
        "disk_free_gb": 10.0,
    }


def _cuda_box(vram_gb: float) -> dict:
    return {**_cpu_box(), "accelerator": "cuda", "cuda_name": "Test GPU", "vram_gb": vram_gb}


# ---------------------------------------------------------------- selection


@pytest.mark.parametrize(
    ("vram", "expected"),
    [(80.0, "large"), (40.0, "large"), (39.9, "small"), (24.0, "small"), (16.0, "small"), (15.9, "nano"), (8.0, "nano")],
)
def test_cuda_vram_hierarchy(vram: float, expected: str) -> None:
    name, reason = select_profile(_cuda_box(vram))
    assert name == expected
    assert "CUDA" in reason


def test_cpu_ceiling_defaults_to_nano() -> None:
    name, reason = select_profile(_cpu_box(ram_gb=512.0, cpus=128))
    assert name == "nano"
    assert "cpu-only" in reason


def test_cpu_max_profile_is_configurable() -> None:
    name, _ = select_profile(_cpu_box(), {"autofit": {"cpu_max_profile": "small"}})
    assert name == "small"


def test_cpu_max_profile_rejects_large() -> None:
    with pytest.raises(ValueError, match="CUDA-only"):
        select_profile(_cpu_box(), {"autofit": {"cpu_max_profile": "large"}})


def test_thresholds_read_from_config() -> None:
    cfg = {"autofit": {"small_min_vram_gb": 24}}
    assert select_profile(_cuda_box(20.0), cfg)[0] == "nano"
    assert select_profile(_cuda_box(25.0), cfg)[0] == "small"


# ---------------------------------------------------------------- resolution


def test_explicit_profile_beats_everything() -> None:
    result = resolve_profile({"model": {"profile": "auto"}}, "small")
    assert result == {"profile": "small", "source": "cli", "reason": None, "hardware": None}


def test_env_beats_config(monkeypatch) -> None:
    monkeypatch.setenv(autofit._ENV_VAR, "small")
    result = resolve_profile({"model": {"profile": "nano"}})
    assert result["profile"] == "small" and result["source"] == "env"


def test_env_auto_forces_detection(monkeypatch) -> None:
    monkeypatch.setenv(autofit._ENV_VAR, "auto")
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box())
    result = resolve_profile({"model": {"profile": "nano"}})
    assert result["profile"] == "nano" and result["source"] == "auto"


def test_config_concrete_profile_is_respected() -> None:
    result = resolve_profile({"model": {"profile": "nano"}})
    assert result["profile"] == "nano" and result["source"] == "config"
    assert result["hardware"] is None


def test_config_auto_triggers_detection(monkeypatch, capsys) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box(ram_gb=2.9, cpus=2))
    result = resolve_profile({"model": {"profile": "auto"}})
    assert result["profile"] == "nano"
    assert result["source"] == "auto"
    assert result["reason"] and result["hardware"]["ram_gb"] == 2.9
    assert "[autofit]" in capsys.readouterr().out


def test_resolved_profile_helper_returns_name_only(monkeypatch) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box())
    assert resolved_profile({"model": {"profile": "auto"}}) == "nano"
    assert resolved_profile({"model": {"profile": "small"}}, "large") == "large"


def test_custom_profiles_pass_through() -> None:
    # The tiny profile of the test fixture repos must keep working.
    assert resolve_profile({"model": {"profile": "tiny"}}, "tiny")["profile"] == "tiny"


# ---------------------------------------------------------------- memory fit


def test_fit_memory_shrinks_and_preserves_effective_batch(tiny_mc) -> None:
    hw = _cpu_box(ram_gb=1.0)
    micro, accum, adjusted, _ = fit_memory(
        hw, tiny_mc, "cpu", 16, 1, {"autofit": {"ram_headroom_cpu": 0.005}}
    )
    assert adjusted is True
    assert micro < 16
    assert micro * accum == 16  # effective batch never changes
    assert micro >= 1


def test_fit_memory_floors_at_one_micro_batch(tiny_mc) -> None:
    hw = _cpu_box(ram_gb=0.5)
    micro, accum, adjusted, _ = fit_memory(
        hw, tiny_mc, "cpu", 16, 1, {"autofit": {"ram_headroom_cpu": 0.001}}
    )
    assert (micro, accum, adjusted) == (1, 16, True)


def test_fit_memory_untouched_when_roomy(tiny_mc) -> None:
    hw = _cpu_box(ram_gb=64.0)
    result = fit_memory(hw, tiny_mc, "cpu", 16, 4, None)
    assert result[:3] == (16, 4, False)


def test_fit_memory_uses_vram_budget_on_cuda(tiny_mc) -> None:
    hw = _cuda_box(0.001)  # ~1 MB VRAM: far below even the static footprint
    micro, accum, adjusted, _ = fit_memory(
        hw, tiny_mc, "cuda", 16, 1, {"autofit": {"ram_headroom_gpu": 0.6}}
    )
    assert adjusted is True
    assert micro * accum == 16


# ---------------------------------------------------------------- detection


def test_detect_hardware_probe_shape() -> None:
    hw = detect_hardware()
    assert {"accelerator", "cuda_name", "vram_gb", "mps_available", "cpus", "ram_gb", "disk_free_gb"} <= set(hw)
    assert hw["accelerator"] in ("cpu", "cuda")
    assert hw["cpus"] >= 1


# ------------------------------------------- integration with model config


def test_repository_config_defaults_to_auto(config) -> None:
    assert cfg_get(config, "model.profile") == "auto"


def test_model_config_from_yaml_auto(config, monkeypatch) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box())
    assert model_config_from_yaml(config) == model_config_from_yaml(config, "nano")


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        (_cpu_box(), "nano"),
        (_cuda_box(24.0), "small"),
        (_cuda_box(80.0), "large"),
    ],
)
def test_auto_resolved_profiles_respect_param_contract(config, monkeypatch, hw, expected) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: hw)
    mc = model_config_from_yaml(config, "auto")
    low, high = RANGES[expected]
    assert low <= analytic_param_count(mc) <= high


def test_count_profile_auto(config, monkeypatch) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box())
    report = count_profile(config, "auto")
    assert report["profile"] == "nano"
    assert report["in_range"] is True


def test_count_profile_materialized_auto_matches_explicit(config, monkeypatch) -> None:
    monkeypatch.setattr(autofit, "detect_hardware", lambda: _cpu_box())
    auto = count_profile(config, "auto")
    explicit = count_profile(config, "nano")
    assert auto["total"] == explicit["total"]


# ---------------------------------------------------- trainer integration


def test_trainer_resolves_profile_from_config(tiny_repo, tmp_path) -> None:
    from prometheus_ns.train.trainer import train

    result = train(
        tiny_repo["cfg"],
        None,  # config profile is the concrete "tiny": resolved without probing
        tiny_repo["tokenizer"],
        max_tokens=2 * 128,
        checkpoints_dir=tmp_path / "ck",
        metrics_path=tmp_path / "m.jsonl",
    )
    assert result["profile"] == "tiny"
    assert result["autofit"]["source"] == "config"
    assert result["autofit"]["memory_fit_adjusted"] is False


def test_trainer_env_override(tiny_repo, tmp_path, monkeypatch) -> None:
    from prometheus_ns.train.trainer import train

    monkeypatch.setenv(autofit._ENV_VAR, "tiny")
    result = train(
        tiny_repo["cfg"],
        None,
        tiny_repo["tokenizer"],
        max_tokens=2 * 128,
        checkpoints_dir=tmp_path / "ck",
        metrics_path=tmp_path / "m.jsonl",
    )
    assert result["profile"] == "tiny"
    assert result["autofit"]["source"] == "env"
