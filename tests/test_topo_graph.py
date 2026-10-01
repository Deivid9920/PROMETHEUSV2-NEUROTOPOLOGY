"""Fixtures for the symbolic-graph sensors. Immutable anchor (V2.1).

R3a tests are pure networkx and always run. The R3b test needs the
persistence wrapper (Fase 1-V2) and skips with an explicit message until
it exists; its expected value is hand-derived: a 4-cycle with edge
distance 0.5 (conf 0.5) is born in the Rips complex at r = 0.5 and dies
when the chords complete K4 at r = 1.0, so the H1 persistence is 0.5.
"""

from __future__ import annotations

import pytest

from prometheus_ns.topo.graph_topology import (
    adjacency_distance_matrix, build_isa_graph, census_directed_cycles,
    load_triplets, rips_h1_persistence,
)


def _t(sub, obj, conf, tid=None):
    return {"sub": sub, "rel": "is_a", "obj": obj, "conf": conf,
            "triplet_id": tid or f"{sub}->{obj}"}


def test_directed_triangle_is_one_cycle() -> None:
    g = build_isa_graph(
        [_t("a", "b", 0.9), _t("b", "c", 0.9), _t("c", "a", 0.9)],
        topk=None, conf_min=0.6)
    cycles, truncated = census_directed_cycles(g)
    assert not truncated
    assert len(cycles) == 1
    assert cycles[0]["min_conf"] == pytest.approx(0.9)
    assert set(cycles[0]["path"]) == {"a", "b", "c"}


def test_acyclic_chain_is_zero() -> None:
    g = build_isa_graph(
        [_t("a", "b", 0.9), _t("b", "c", 0.9), _t("c", "d", 0.9)],
        topk=None, conf_min=0.6)
    cycles, truncated = census_directed_cycles(g)
    assert cycles == [] and not truncated


def test_two_node_cycle_counts() -> None:
    g = build_isa_graph([_t("a", "b", 0.8), _t("b", "a", 0.8)],
                        topk=None, conf_min=0.6)
    cycles, _ = census_directed_cycles(g)
    assert len(cycles) == 1
    assert len(cycles[0]["path"]) == 2


def test_confidence_floor_removes_the_cycle() -> None:
    g = build_isa_graph(
        [_t("a", "b", 0.9), _t("b", "c", 0.9), _t("c", "a", 0.5)],
        topk=None, conf_min=0.6)
    cycles, _ = census_directed_cycles(g)
    assert cycles == []


def test_dedupe_keeps_max_confidence() -> None:
    triplets = [_t("a", "b", 0.7, "t1"), _t("a", "b", 0.95, "t2")]
    g = build_isa_graph(triplets, topk=None, conf_min=0.6)
    assert g["a"]["b"]["conf"] == pytest.approx(0.95)
    assert g["a"]["b"]["triplet_id"] == "t2"


def test_cycle_cap_reports_truncation() -> None:
    g = build_isa_graph(
        [_t("a", "b", 0.9), _t("b", "c", 0.9), _t("c", "a", 0.9)],
        topk=None, conf_min=0.6)
    cycles, truncated = census_directed_cycles(g, max_cycles=1)
    assert len(cycles) == 1 and truncated is True


def test_adjacency_symmetrization_uses_stronger_relation() -> None:
    g = build_isa_graph([_t("a", "b", 0.9), _t("b", "a", 0.7)],
                        topk=None, conf_min=0.6)
    _, matrix = adjacency_distance_matrix(g)
    assert matrix[0, 1] == pytest.approx(1.0 - 0.9)
    assert matrix[1, 0] == pytest.approx(1.0 - 0.9)


def test_rips_h1_of_four_cycle_hand_value() -> None:
    g = build_isa_graph(
        [_t("a", "b", 0.5), _t("b", "c", 0.5),
         _t("c", "d", 0.5), _t("d", "a", 0.5)],
        topk=None, conf_min=0.4)
    try:
        h1, _ = rips_h1_persistence(g)
    except NotImplementedError:
        pytest.skip("persistence wrapper pending (Fase 1-V2)")
    persistence = h1[:, 1] - h1[:, 0]
    persistent = persistence[persistence > 0.1]
    assert len(persistent) == 1
    assert persistent[0] == pytest.approx(0.5, abs=1e-9)


def test_load_triplets_fail_loud(tmp_path) -> None:
    bad = tmp_path / "triplets.jsonl"
    bad.write_text('{"sub": "a", "rel": "is_a"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        load_triplets(bad)
