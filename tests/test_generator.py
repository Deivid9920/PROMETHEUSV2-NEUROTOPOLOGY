"""Tests for output validation against the consistency graph."""

from __future__ import annotations

import pytest
import spacy

from prometheus_ns.symbolic.consistency import ConsistencyGraph
from prometheus_ns.symbolic.extractor import Triplet
from prometheus_ns.symbolic.generator import check_sentences

pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:  # pragma: nocover
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)

DISJOINT = ["bird", "mammal", "fish", "insect", "vehicle", "food", "plant", "tool"]

GRAPH = [
    Triplet("robin", "is_a", "bird", 0.9, "d1"),
    Triplet("sparrow", "is_a", "bird", 0.9, "d2"),
]


def test_contradicting_sentence_flagged(tiny_repo) -> None:
    records = check_sentences(["A robin is a vehicle."], tiny_repo["cfg"], GRAPH)
    assert records[0]["status"] == "contradicts"


def test_matching_sentence_ok(tiny_repo) -> None:
    records = check_sentences(["A robin is a bird."], tiny_repo["cfg"], GRAPH)
    assert records[0]["status"] == "ok"


def test_unknown_claim_marked_unverified(tiny_repo) -> None:
    records = check_sentences(["A wombat is a mammal."], tiny_repo["cfg"], GRAPH)
    assert records[0]["status"] == "unverified"


def test_free_narrative_without_claims_is_ok(tiny_repo) -> None:
    records = check_sentences(["The rain fell gently over the hills and the night grew quiet."], tiny_repo["cfg"], GRAPH)
    assert records[0]["status"] == "ok"


def test_graph_caps_entity_growth() -> None:
    graph = ConsistencyGraph(max_entities=3)
    for i in range(10):
        graph.add(Triplet(f"entity_{i}", "is_a", f"class_{i}", 0.9, f"d{i}"))
    assert graph.capped is True
    assert graph.graph.number_of_nodes() <= 3
