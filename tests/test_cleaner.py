"""Tests for the streaming cleaner on the synthetic fixture corpus.

The fixtures contain exactly: 2 identical docs, 2 near-duplicates,
1 doc with ~40% symbol characters and 1 doc whose paragraphs are all
shorter than the shingle size (the empty-shingle edge case).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus_ns.data.cleaner import (
    _NearDupIndex,
    doc_hash,
    minhash_from_shingles,
    normalize_text,
    paragraph_shingles,
)
from prometheus_ns.data.quality import gopher_checks

FIXTURES = Path(__file__).parent / "fixtures" / "cleaner"
PERMS = 64
SHINGLE_WORDS = 5


def _read(name: str) -> str:
    return normalize_text((FIXTURES / name).read_text(encoding="utf-8"))


def test_normalize_collapses_whitespace_but_keeps_lines() -> None:
    raw = "hello   world  \t mixed \n\n spacing \n  here"
    out = normalize_text(raw)
    assert out == "hello world mixed\nspacing\nhere"


def test_identical_docs_share_hash() -> None:
    assert doc_hash(_read("dup_original.txt")) == doc_hash(_read("dup_copy.txt"))


def test_near_dups_differ_in_hash_but_match_in_lsh() -> None:
    original, variant = _read("near_dup_original.txt"), _read("near_dup_variant.txt")
    assert doc_hash(original) != doc_hash(variant)

    index = _NearDupIndex(num_perm=PERMS, threshold=0.8, flush_every=200000)
    mh_original = minhash_from_shingles(paragraph_shingles(original, SHINGLE_WORDS), PERMS)
    mh_variant = minhash_from_shingles(paragraph_shingles(variant, SHINGLE_WORDS), PERMS)
    assert mh_original is not None and mh_variant is not None

    assert index.is_near_dup(mh_original, "original") is None  # inserted
    dup_of = index.is_near_dup(mh_variant, "variant")
    assert dup_of == "original"


def test_symbol_heavy_doc_fails_gopher() -> None:
    text = _read("symbols_heavy.txt")
    reasons = gopher_checks(text)
    assert "symbol_ratio" in reasons


def test_short_paragraphs_produce_no_shingles_and_no_crash() -> None:
    text = _read("short_paragraphs.txt")
    shingles = paragraph_shingles(text, SHINGLE_WORDS)
    assert shingles == set()
    mh = minhash_from_shingles(shingles, PERMS)
    assert mh is None  # caller must skip the near-dup check for this doc


def test_short_doc_is_not_flagged_as_near_dup_of_anything() -> None:
    index = _NearDupIndex(num_perm=PERMS, threshold=0.8, flush_every=200000)
    harbor = _read("dup_original.txt")
    mh_harbor = minhash_from_shingles(paragraph_shingles(harbor, SHINGLE_WORDS), PERMS)
    index.is_near_dup(mh_harbor, "harbor")
    short_mh = minhash_from_shingles(paragraph_shingles(_read("short_paragraphs.txt"), SHINGLE_WORDS), PERMS)
    assert short_mh is None  # skipped entirely, never a false near-dup


def test_cleaner_pipeline_end_to_end(tmp_path, monkeypatch) -> None:
    """Feed the fixtures through clean_corpus via raw .raw files."""
    from prometheus_ns import load_config

    root = tmp_path / "repo"
    (root / "data_raw" / "gutenberg.org").mkdir(parents=True)
    (root / "data_clean" / "holdout").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)

    for i, name in enumerate(sorted(FIXTURES.glob("*.txt"))):
        source_dir = root / "data_raw" / "gutenberg.org"
        # Wrap as minimal HTML so the extraction stage finds the text.
        html = f"<html><body><pre>{name.read_text(encoding='utf-8')}</pre></body></html>"
        (source_dir / f"{i:012x}.raw").write_text(html, encoding="utf-8")

    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "data: {sources: [gutenberg], max_pages_per_source: 10, out_dir: data_raw,",
                "       user_agent: t, rps: 1.0, timeout_s: 5, retries: 1, holdout_frac: 0.02}",
                "clean: {min_doc_chars: 100, max_doc_chars: 100000, minhash_perms: 64,",
                "        minhash_jaccard: 0.8, keep_quantile: 1.0, shingle_words: 5,",
                "        flush_every_docs: 200000}",
                "loop: {holdout_sha256: null}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)

    from prometheus_ns.data.cleaner import clean_corpus

    stats = clean_corpus(cfg)

    # dup_copy must be discarded as an exact duplicate of dup_original;
    # near_dup_variant must be discarded as a near duplicate;
    # symbols_heavy must be discarded by the symbol_ratio rule;
    # short_paragraphs fails min_doc_chars (length gate);
    # dup_original and near_dup_original survive (keep_quantile=1.0).
    assert stats["discarded"]["dup_exact"] == 1
    assert stats["discarded"]["dup_near"] == 1
    assert stats["discarded"]["symbol_ratio"] == 1
    assert stats["discarded"]["length"] == 1
    assert stats["kept"] == 2

    # Survivors are stored under their source directory (the crawl
    # domain, "gutenberg.org" here).
    kept = sorted(p.name for p in (root / "data_clean" / "gutenberg.org").glob("*.txt"))
    assert len(kept) == 2

    index_lines = (root / "data_clean" / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 2
    assert all(json.loads(line)["source"] == "gutenberg.org" for line in index_lines)


def test_lsh_flush_rebuilds_index() -> None:
    index = _NearDupIndex(num_perm=128, threshold=0.8, flush_every=2)
    text_a = "the fisherman repairs his net by the harbor before dawn every single morning of the week"
    text_b = "the fisherman repairs his net by the harbor before dawn every single morning of the week again"
    text_c = "distant galaxies emit measurable radiation across the entire electromagnetic spectrum today"
    mh_a = minhash_from_shingles(paragraph_shingles(normalize_text(text_a), 5), 128)
    mh_b = minhash_from_shingles(paragraph_shingles(normalize_text(text_b), 5), 128)
    mh_c = minhash_from_shingles(paragraph_shingles(normalize_text(text_c), 5), 128)
    assert index.is_near_dup(mh_a, "a") is None          # inserted, count=1
    assert index.is_near_dup(mh_b, "b") == "a"           # near duplicate detected
    assert index.is_near_dup(mh_c, "c") is None          # inserted, count=2 -> flush
    # After the flush the index is empty: the same signature no longer
    # matches anything until it is re-inserted.
    assert index.is_near_dup(mh_b, "b2") is None
