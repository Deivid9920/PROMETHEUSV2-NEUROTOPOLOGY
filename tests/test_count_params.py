"""Tests for the parameter count CLI contract."""

from __future__ import annotations

import json

from prometheus_ns.model.count_params import count_profile


def test_nano_profile_count_real_sum(config) -> None:
    report = count_profile(config, "nano", materialize=True)
    assert report["in_range"] is True
    assert 12_000_000 <= report["total"] <= 18_000_000
    assert "tok_emb (tied with lm_head)" in report["components"]


def test_small_profile_count(config) -> None:
    report = count_profile(config, "small")
    assert report["in_range"] is True


def test_large_profile_count(config) -> None:
    report = count_profile(config, "large")
    assert report["in_range"] is True


def test_component_table_sums_to_total(config) -> None:
    report = count_profile(config, "nano")
    assert sum(report["components"].values()) == report["total"]


def test_report_is_json_serializable(config) -> None:
    report = count_profile(config, "nano")
    payload = json.dumps(report)
    assert "total" in payload
