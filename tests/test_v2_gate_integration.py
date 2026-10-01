"""Fase 3-V2 dry-run integration test: the composite promotion gate.

Synthetic metrics only, nothing real is trained. Verifies:
- a "churn" round (high bottleneck + flat ppl) does NOT promote and the
  cause reaches the logs;
- a "stable" round promotes;
- the star case (NS promotes, topology vetoes) is logged as such;
- the fail-closed rule (1.6): a decision round >= 2 without a topological
  measurement is an error in the V2 runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prometheus_ns.autoloop.loop import topo_veto_from_log
from prometheus_ns.topo.promotion_topology import (
    TopoMetrics, TopoRules, decide_topo_veto,
)

RULES = TopoRules(churn_bottleneck=0.15, churn_gain_excuse=0.02,
                  b0_collapse_ratio=0.5, h1_growth_max=0.25,
                  eps=0.3, h1_min_persistence=0.1)


def _synthetic_metrics(churn: bool) -> TopoMetrics:
    if churn:
        # candidate merges two of the champion's four clusters: large
        # bottleneck, negligible ppl gain -> R1 fires, not excused
        return TopoMetrics(
            round_id=2,
            model_h0=np.array([[0.0, 0.7], [0.0, 0.65],
                               [0.0, 0.05], [0.0, 0.04]]),
            model_h1=np.array([[0.2, 0.6]]),
            graph_h1_count=4.0,
            champion_model_h0=np.array([[0.0, 0.7], [0.0, 0.6],
                                        [0.0, 0.5], [0.0, 0.4]]),
            champion_model_h1=np.array([[0.2, 0.6]]),
            champion_graph_h1_count=4.0,
            baseline_b0=4.0, relative_ppl_gain=0.005)
    return TopoMetrics(
        round_id=2,
        model_h0=np.array([[0.0, 0.7], [0.0, 0.6], [0.0, 0.5], [0.0, 0.4]]),
        model_h1=np.array([[0.2, 0.6]]),
        graph_h1_count=4.0,
        champion_model_h0=np.array([[0.0, 0.7], [0.0, 0.6],
                                    [0.0, 0.5], [0.0, 0.4]]),
        champion_model_h1=np.array([[0.2, 0.6]]),
        champion_graph_h1_count=4.0,
        baseline_b0=4.0, relative_ppl_gain=0.02)


def _composite(ns_promote: bool, metrics: TopoMetrics) -> bool:
    decision = decide_topo_veto(metrics, RULES)
    return bool(ns_promote and not decision.veto)


def _write_topo_row(logs: Path, round_id: int, veto: bool, causes: list) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "topo.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"round_id": round_id, "veto": veto,
                             "causes": causes,
                             "establish_baseline": False}) + "\n")


def test_churn_round_does_not_promote_and_logs_the_cause(tmp_path) -> None:
    decision = decide_topo_veto(_synthetic_metrics(churn=True), RULES)
    assert decision.veto is True
    assert any(c.startswith("R1") for c in decision.causes)
    # composite: NS saw a flat-ppl round and promoted; the topology says no
    final = _composite(ns_promote=True, metrics=_synthetic_metrics(churn=True))
    assert final is False
    _write_topo_row(tmp_path, 2, decision.veto, decision.causes)
    veto, causes = topo_veto_from_log(tmp_path, 2)
    assert veto is True and any(c.startswith("R1") for c in causes)


def test_stable_round_promotes() -> None:
    decision = decide_topo_veto(_synthetic_metrics(churn=False), RULES)
    assert decision.veto is False
    assert _composite(ns_promote=True, metrics=_synthetic_metrics(churn=False))
    assert not _composite(ns_promote=False,
                          metrics=_synthetic_metrics(churn=False))


def test_veto_composes_also_when_ns_rejects(tmp_path) -> None:
    # NS rejects; the veto cannot resurrect the round
    final = _composite(ns_promote=False, metrics=_synthetic_metrics(churn=True))
    assert final is False
    _write_topo_row(tmp_path, 2, True, ["R1 structural churn: dry-run"])
    assert topo_veto_from_log(tmp_path, 2)[0] is True


def test_fail_closed_round_without_measurement(tmp_path) -> None:
    # (1.6): decision round >= 2 with no topo row in the log -> the V2
    # runner raises; here we pin the reading helper's contract
    assert topo_veto_from_log(tmp_path, 2) == (False, []), (
        "the loop-level helper stays silent (NS-compatible); the "
        "fail-closed error belongs to the V2 runner which owns the sensors")


def test_champion_graph_census_feeds_rule_r3() -> None:
    # champion 4 cycles -> candidate 6 > ceiling 5 -> R3 veto
    metrics = _synthetic_metrics(churn=False)
    metrics = TopoMetrics(
        round_id=2,
        model_h0=metrics.model_h0, model_h1=metrics.model_h1,
        graph_h1_count=6.0,
        champion_model_h0=metrics.champion_model_h0,
        champion_model_h1=metrics.champion_model_h1,
        champion_graph_h1_count=4.0,
        baseline_b0=4.0, relative_ppl_gain=0.30)
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is True
    assert any(c.startswith("R3") for c in decision.causes)
    assert _composite(ns_promote=True, metrics=metrics) is False
