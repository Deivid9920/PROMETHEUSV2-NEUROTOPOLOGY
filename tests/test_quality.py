"""Tests for Gopher heuristics and the weighted quality score."""

from __future__ import annotations

from prometheus_ns.data.quality import (
    gopher_checks,
    keep_by_quantile,
    mean_word_len,
    punct_line_frac,
    quality_score,
    stopword_ratio,
    symbol_ratio,
)

CLEAN_TEXT = (
    "The old man walked along the quiet river and watched the birds fly over "
    "the water. He carried a small bag with bread and fruit for the trip. "
    "The sun was warm, and the wind was soft against his face as he walked. "
    "Every step felt calm and easy in the light of the late afternoon."
)

SYMBOL_TEXT = "%%% ### $$$ !!! @@@ ^^^ &&& *** ((( ))) ### $$$ %%% @@@ ^^^ &&& *** ((( )))"

STOPWORD_POOR_TEXT = "\n".join(
    f"quixotic perseverance transcends mundane obfuscation paradigms {i} vehemently" for i in range(12)
)


def test_mean_word_len_bounds() -> None:
    assert 3.0 <= mean_word_len(CLEAN_TEXT) <= 10.0
    assert mean_word_len("") == 0.0


def test_clean_text_passes_all_checks() -> None:
    assert gopher_checks(CLEAN_TEXT) == []


def test_symbol_heavy_text_fails_symbol_ratio() -> None:
    assert symbol_ratio(SYMBOL_TEXT) > 0.1
    assert "symbol_ratio" in gopher_checks(SYMBOL_TEXT)


def test_relaxed_symbol_ratio_for_technical_sources() -> None:
    # arXiv-style text: formulas push the symbol ratio above the strict
    # Gopher ceiling; the per-source relaxation must let it through.
    formula_text = "The energy satisfies E = mc^2 and momentum p = mv in flat spacetime." * 6
    ratio = symbol_ratio(formula_text)
    assert ratio > 0
    strict = gopher_checks(formula_text, symbol_ratio_max=ratio / 2)
    relaxed = gopher_checks(formula_text, symbol_ratio_max=ratio * 2)
    assert "symbol_ratio" in strict
    assert "symbol_ratio" not in relaxed


def test_stopword_poor_text_fails() -> None:
    assert stopword_ratio(STOPWORD_POOR_TEXT) < 0.3
    assert "stopword_ratio" in gopher_checks(STOPWORD_POOR_TEXT)


def test_punct_line_frac() -> None:
    assert punct_line_frac("First line.\nSecond line!\nThird line?") == 1.0
    assert punct_line_frac("no punctuation here\nstill none") == 0.0


def test_quality_score_range_and_ordering() -> None:
    clean_score = quality_score(CLEAN_TEXT)
    symbol_score = quality_score(SYMBOL_TEXT)
    assert 0.0 <= clean_score <= 1.0
    assert 0.0 <= symbol_score <= 1.0
    assert clean_score > symbol_score


def test_keep_by_quantile_keeps_top_fraction() -> None:
    items = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)]
    keep = keep_by_quantile(items, 0.8)
    assert keep == {"a", "b", "c", "d"}  # top 4 of 5
    assert "e" not in keep


def test_keep_by_quantile_empty_input() -> None:
    assert keep_by_quantile([], 0.8) == set()
