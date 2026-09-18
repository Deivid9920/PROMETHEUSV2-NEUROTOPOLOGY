"""Tests for the distinct-bigram diversity guard."""

from __future__ import annotations

import json

from prometheus_ns.autoloop.diversity import distinct_bigram_ratio, measure_and_record

REPETITIVE = ["the cat sat the cat sat the cat sat"] * 50
VARIED = [
    f"document {i} explains how winds shift across the {i} valleys "
    "while rivers carve new paths through limestone caves and old forests"
    for i in range(50)
]


def test_repetitive_corpus_scores_low() -> None:
    ratio = distinct_bigram_ratio(REPETITIVE)
    assert 0.0 <= ratio < 0.2


def test_varied_corpus_scores_higher() -> None:
    varied_ratio = distinct_bigram_ratio(VARIED)
    repetitive_ratio = distinct_bigram_ratio(REPETITIVE)
    assert varied_ratio > repetitive_ratio


def test_empty_and_short_inputs_are_safe() -> None:
    assert distinct_bigram_ratio([]) == 0.0
    assert distinct_bigram_ratio(["", "one"]) == 0.0
    # A single two-word doc has one bigram: unique/total == 1.0.
    assert distinct_bigram_ratio(["two words"]) == 1.0


def test_measure_and_record_appends_row(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for i, text in enumerate(VARIED):
        (docs_dir / f"doc_{i:03d}.txt").write_text(text, encoding="utf-8")
    logs_dir = tmp_path / "logs"
    ratio = measure_and_record(docs_dir, round_id=1, logs_dir=logs_dir)
    assert 0.0 < ratio <= 1.0
    rows = [json.loads(line) for line in (logs_dir / "diversity.jsonl").read_text().splitlines()]
    assert rows[-1]["round_id"] == 1
    assert rows[-1]["ratio"] == ratio
