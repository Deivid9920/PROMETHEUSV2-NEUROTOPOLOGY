"""Tests for spaCy triplet extraction, including passive voice handling."""

from __future__ import annotations

import pytest
import spacy

from prometheus_ns.symbolic.extractor import extract_triplets_from_doc, normalize_entity

pytest.importorskip("spacy")
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:  # pragma: nocover
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)


def _triplets(text: str, min_confidence: float = 0.6):
    doc = _NLP(text)
    return extract_triplets_from_doc(doc, min_confidence, "hash")


def test_active_verb_triplet() -> None:
    triplets = _triplets("The cat chased the mouse through the garden.")
    assert any(t.sub == "cat" and t.rel == "chase" and t.obj == "mouse" for t in triplets)


def test_copula_is_a_triplet() -> None:
    triplets = _triplets("A robin is a bird that sings in the morning.")
    assert any(t.sub == "robin" and t.rel == "is_a" and t.obj == "bird" for t in triplets)


def test_passive_voice_is_not_inverted() -> None:
    # "The mouse was chased by the cat" must yield (cat, chase, mouse),
    # never (mouse, chase, cat).
    triplets = _triplets("The mouse was chased by the cat.")
    assert any(t.sub == "cat" and t.rel == "chase" and t.obj == "mouse" for t in triplets)
    assert not any(t.sub == "mouse" and t.rel == "chase" and t.obj == "cat" for t in triplets)


def test_apposition_yields_is_a() -> None:
    triplets = _triplets("Socrates, a philosopher, walked through the agora of Athens.")
    assert any(t.rel == "is_a" and "socrat" in t.sub and "philosopher" in t.obj for t in triplets)


def test_low_confidence_and_noise_filtered() -> None:
    triplets = _triplets("It is a thing that people say sometimes.")
    # Pronouns/empty entities must not become triplet endpoints.
    assert all(t.sub not in {"it", "he", "she", "they"} and t.obj not in {"it", "he", "she", "they"} for t in triplets)


def test_min_confidence_gate() -> None:
    text = "The dog found the ball in the garden."
    strict = _triplets(text, min_confidence=0.95)
    lenient = _triplets(text, min_confidence=0.6)
    assert len(strict) <= len(lenient)


def test_normalize_entity_strips_determiners() -> None:
    doc = _NLP("A robin is a bird.")
    token = [t for t in doc if t.dep_ == "nsubj"][0]
    assert normalize_entity(token.text, head=token) in {"robin"}
