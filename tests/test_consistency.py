"""End-to-end symbolic contradiction test on a synthetic corpus.

Ten contradictory sentence pairs are planted alongside ten clean
controls. The contract: recall >= 0.8 and precision >= 0.8 at the
document level against the planted ground truth.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import spacy

from prometheus_ns.symbolic.consistency import find_contradictions
from prometheus_ns.symbolic.extractor import extract_triplets_from_doc

pytest.importorskip("spacy")
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:  # pragma: nocover
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)

DISJOINT = ["bird", "mammal", "fish", "insect", "vehicle", "food", "plant", "tool"]
EXCLUSIVE = [["alive", "dead"], ["male", "female"]]

CONTRADICTION_PAIRS = [
    ("A robin is a bird.", "A robin is a vehicle."),
    ("A dog is a mammal.", "A dog is an insect."),
    ("A shark is a fish.", "A shark is a mammal."),
    ("A bee is an insect.", "A bee is a bird."),
    ("A truck is a vehicle.", "A truck is a food."),
    ("An apple is a food.", "An apple is a tool."),
    ("An eagle is a bird.", "An eagle is a fish."),
    ("A rose is a plant.", "A rose is a vehicle."),
    ("A tuna is a fish.", "A tuna is a bird."),
    ("An oak is a plant.", "An oak is a tool."),
]

CLEAN_SENTENCES = [
    "A sparrow is a bird.",
    "A whale is a mammal.",
    "A salmon is a fish.",
    "An ant is an insect.",
    "A car is a vehicle.",
    "Rice is a food.",
    "A fern is a plant.",
    "A hammer is a tool.",
    "A crow is a bird.",
    "A dolphin is a mammal.",
]


def _doc_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _corpus_triplets() -> list:
    triplets = []
    for sentence in [s for pair in CONTRADICTION_PAIRS for s in pair] + CLEAN_SENTENCES:
        doc = _NLP(sentence)
        triplets.extend(extract_triplets_from_doc(doc, 0.6, _doc_hash(sentence)))
    return triplets


def test_extraction_covers_all_fixture_statements() -> None:
    triplets = _corpus_triplets()
    subjects = {t.sub for t in triplets}
    for subject in ("robin", "dog", "shark", "bee", "truck", "apple", "eagle", "rose", "tuna", "oak"):
        assert subject in subjects, f"extractor missed statements about '{subject}'"


def test_contradiction_recall_and_precision() -> None:
    triplets = _corpus_triplets()
    contradictions, quarantined = find_contradictions(triplets, DISJOINT, EXCLUSIVE, 0.6)

    truth = {_doc_hash(s) for pair in CONTRADICTION_PAIRS for s in pair}
    predicted = quarantined

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(
        "\nclassification report (document level)\n"
        f"  true positives:  {tp}\n"
        f"  false positives: {fp}\n"
        f"  false negatives: {fn}\n"
        f"  precision: {precision:.3f}\n"
        f"  recall:    {recall:.3f}"
    )
    assert len(contradictions) >= 10, "at least the ten planted conflicts must be found"
    assert recall >= 0.8, f"recall {recall:.3f} below contract"
    assert precision >= 0.8, f"precision {precision:.3f} below contract"
    assert not (predicted & {_doc_hash(s) for s in CLEAN_SENTENCES}), "clean controls must never be quarantined"


def test_low_confidence_side_does_not_quarantine() -> None:
    triplets = _corpus_triplets()
    # Drop the second statement of the first pair below the threshold:
    # a single low-confidence side must not quarantine the document.
    filtered = [t for t in triplets if not (t.sub == "robin" and t.obj == "vehicle")]
    _contradictions, quarantined = find_contradictions(filtered, DISJOINT, EXCLUSIVE, 0.6)
    assert _doc_hash("A robin is a bird.") not in quarantined


def test_exclusive_pairs_detected() -> None:
    triplets = _corpus_triplets()
    triplets = triplets + [
        type(triplets[0])("bell", "sounds", "alive", 0.9, "h1"),
        type(triplets[0])("bell", "sounds", "dead", 0.9, "h2"),
    ]
    _contradictions, quarantined = find_contradictions(triplets, DISJOINT, EXCLUSIVE, 0.6)
    assert {"h1", "h2"} <= quarantined
