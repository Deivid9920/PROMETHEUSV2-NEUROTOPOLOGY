"""Numerical anchors for the topological veto. Immutable anchor (V2.1).

Every expected value below is derived by hand from the definitions in
promotion_topology.py, independent of ripser or any TDA library: the
diagrams are synthetic arrays. The 1-vs-2 fixture is the killer test for
the classic implementation bug: a sum-minimizing assignment
(linear_sum_assignment) returns 0.1 here, while the true bottleneck
minimax value is 0.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from prometheus_ns.topo.promotion_topology import (
    TopoMetrics, TopoRules, betti0_at_eps, bottleneck_distance,
    decide_topo_veto,
)

RULES = TopoRules(churn_bottleneck=0.15, churn_gain_excuse=0.02,
                  b0_collapse_ratio=0.5, h1_growth_max=0.25,
                  eps=0.3, h1_min_persistence=0.1)


def test_bottleneck_identical_diagrams_is_zero() -> None:
    d = np.array([[0.0, 0.8], [0.1, 0.5]])
    assert bottleneck_distance(d, d.copy()) == pytest.approx(0.0)


def test_bottleneck_single_point_vs_diagonal_hand_value() -> None:
    # A = {(0, 1)}: matching to B={(0, 0.6)} costs 0.4; both falling to
    # their diagonals costs max(0.5, 0.3) = 0.5. Optimal: 0.4.
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 0.6]])
    assert bottleneck_distance(a, b) == pytest.approx(0.4)


def test_bottleneck_one_vs_two_kills_sum_minimizing_impl() -> None:
    # A = {(0, 1)}, B = {(0, 0.6), (0, 0.9)}. Match A->(0, 0.9) at 0.1 and
    # let (0, 0.6) fall to its diagonal at (0.6-0)/2 = 0.3: bottleneck
    # 0.3. A sum-minimizing assignment reports 0.1: wrong metric.
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 0.6], [0.0, 0.9]])
    assert bottleneck_distance(a, b) == pytest.approx(0.3)


def test_bottleneck_three_vs_one_diagonal_dominates() -> None:
    # Best: (0,5)->B at 0.5; (0,3) diagonal 1.5; (0,1) diagonal 0.5 -> 1.5.
    a = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    b = np.array([[0.0, 4.5]])
    assert bottleneck_distance(a, b) == pytest.approx(1.5)


def test_bottleneck_symmetry() -> None:
    a = np.array([[0.0, 0.9], [0.2, 0.7]])
    b = np.array([[0.0, 0.5]])
    assert bottleneck_distance(a, b) == pytest.approx(
        bottleneck_distance(b, a))


def test_bottleneck_empty_diagrams() -> None:
    empty = np.empty((0, 2))
    d = np.array([[0.0, 1.0]])
    assert bottleneck_distance(empty, empty) == pytest.approx(0.0)
    assert bottleneck_distance(d, empty) == pytest.approx(0.5)


def test_betti0_at_eps_counts_survivors() -> None:
    d = np.array([[0.0, 0.8], [0.0, 0.25], [0.0, 0.5], [0.0, 2.0]])
    assert betti0_at_eps(d, 0.3) == 3
    assert betti0_at_eps(d, 0.0) == 4   # all points distinct at scale zero
    assert betti0_at_eps(d, 10.0) == 0  # every class merged below scale 10


def _metrics(**overrides) -> TopoMetrics:
    base = dict(
        round_id=2,
        model_h0=np.array([[0.0, 0.7], [0.0, 0.6], [0.0, 0.5], [0.0, 0.4]]),
        model_h1=np.array([[0.2, 0.6]]),
        graph_h1_count=4.0,
        champion_model_h0=np.array([[0.0, 0.7], [0.0, 0.6], [0.0, 0.5],
                                    [0.0, 0.4]]),
        champion_model_h1=np.array([[0.2, 0.6]]),
        champion_graph_h1_count=4.0,
        baseline_b0=4.0,
        relative_ppl_gain=0.02,
    )
    base.update(overrides)
    return TopoMetrics(**base)


def test_stable_round_no_veto() -> None:
    decision = decide_topo_veto(_metrics(), RULES)
    assert decision.veto is False
    assert decision.establish_baseline is False
    assert decision.causes == []


def test_r1_churn_vetoes_unexplained_shift() -> None:
    # Champion H0 has 4 clusters well separated; candidate merges two
    # (one class stretched), producing a large bottleneck.
    metrics = _metrics(
        model_h0=np.array([[0.0, 0.7], [0.0, 0.65], [0.0, 0.05], [0.0, 0.04]]),
        relative_ppl_gain=0.005,   # small gain: churn not excused
    )
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is True
    assert any(c.startswith("R1") for c in decision.causes)


def test_r1_churn_excused_by_large_gain() -> None:
    metrics = _metrics(
        model_h0=np.array([[0.0, 0.7], [0.0, 0.65], [0.0, 0.05], [0.0, 0.04]]),
        relative_ppl_gain=0.03,    # >= excuse: same shift is allowed
    )
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is False


def test_r2_collapse_vetoes_cluster_loss() -> None:
    # Baseline betti0 = 4; candidate keeps only 1 alive cluster at eps.
    metrics = _metrics(model_h0=np.array([[0.0, 0.9], [0.0, 0.2]]))
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is True
    assert any(c.startswith("R2") for c in decision.causes)


def test_r3_graph_growth_vetoes() -> None:
    metrics = _metrics(graph_h1_count=6.0)   # ceiling = 1.25*4 = 5
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is True
    assert any(c.startswith("R3") for c in decision.causes)


def test_r3_within_growth_is_allowed() -> None:
    metrics = _metrics(graph_h1_count=5.0)   # exactly at ceiling: not >
    assert decide_topo_veto(metrics, RULES).veto is False


def test_round_one_establishes_baseline_and_cannot_veto() -> None:
    metrics = TopoMetrics(
        round_id=1,
        model_h0=np.array([[0.0, 0.8]]),
        model_h1=np.empty((0, 2)),
        graph_h1_count=2.0,
        champion_model_h0=None, champion_model_h1=None,
        champion_graph_h1_count=None, baseline_b0=None,
        relative_ppl_gain=0.0,
    )
    decision = decide_topo_veto(metrics, RULES)
    assert decision.veto is False
    assert decision.establish_baseline is True
