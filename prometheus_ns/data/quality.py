"""Gopher-style quality heuristics and a weighted document quality score.

The heuristics intentionally mirror the classic Gopher corpus filters
(mean word length, symbol ratio, punctuation line coverage, stopword
ratio). The symbol-ratio ceiling is relaxed per source so arXiv text
with formulas and citations is not discarded wholesale.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SYMBOL_RE = re.compile(r"[^a-zA-Z0-9 ]")
_WORD_RE = re.compile(r"[A-Za-z']+")

STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how i if in into is isn't it its itself let's me
    more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your
    yours yourself yourselves""".split()
)

_PUNCT_END = tuple(".!?'\"”’):;]")

# Weights of the weighted quality score; each component is normalized to 0..1.
_WEIGHTS = {"stopword": 0.40, "symbol": 0.20, "punct": 0.25, "word_len": 0.15}


def mean_word_len(text: str) -> float:
    """Average character length of alphabetic words (0.0 when empty)."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def symbol_ratio(text: str) -> float:
    """Fraction of characters outside ``[a-zA-Z0-9 ]`` (0.0 when empty)."""
    if not text:
        return 0.0
    return len(_SYMBOL_RE.findall(text)) / len(text)


def punct_line_frac(text: str) -> float:
    """Fraction of non-empty lines ending in punctuation (0.0 when empty)."""
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln.endswith(_PUNCT_END)) / len(lines)


def stopword_ratio(text: str) -> float:
    """Fraction of alphabetic tokens that are stopwords (0.0 when empty)."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in STOPWORDS) / len(words)


def gopher_checks(
    text: str,
    symbol_ratio_max: float = 0.1,
    mean_word_len_min: float = 3.0,
    mean_word_len_max: float = 10.0,
    punct_line_frac_min: float = 0.8,
    stopword_ratio_min: float = 0.3,
) -> list[str]:
    """Return the list of Gopher rules the document violates.

    An empty list means the document passes every heuristic.
    """
    reasons: list[str] = []
    mean_len = mean_word_len(text)
    if not mean_word_len_min <= mean_len <= mean_word_len_max:
        reasons.append("mean_word_len")
    if symbol_ratio(text) > symbol_ratio_max:
        reasons.append("symbol_ratio")
    if punct_line_frac(text) < punct_line_frac_min:
        reasons.append("punct_lines")
    if stopword_ratio(text) < stopword_ratio_min:
        reasons.append("stopword_ratio")
    return reasons


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def quality_score(text: str, symbol_ratio_max: float = 0.1) -> float:
    """Weighted 0..1 quality score built from the heuristic components.

    Higher is better. The score is only used to rank documents and keep
    the ``clean.keep_quantile`` fraction; documents are hard-filtered by
    ``gopher_checks`` before scoring.
    """
    sw = stopword_ratio(text)
    sym = symbol_ratio(text)
    punct = punct_line_frac(text)
    mean_len = mean_word_len(text)
    components = {
        "stopword": _clip01(sw / 0.45),
        "symbol": _clip01(1.0 - sym / max(symbol_ratio_max, 1e-9) / 1.5),
        "punct": _clip01(punct / 0.95),
        "word_len": _clip01(1.0 - abs(mean_len - 6.5) / 3.5),
    }
    return sum(_WEIGHTS[k] * v for k, v in components.items())


def keep_by_quantile(
    scores: Iterable[tuple[str, float]],
    keep_quantile: float,
) -> set[str]:
    """Return the hash keys whose score is at or above the quantile cut.

    Args:
        scores: Iterable of ``(doc_hash, score)`` pairs.
        keep_quantile: Fraction of documents to keep, e.g. 0.8 keeps the
            top 80%.
    """
    items = sorted(scores, key=lambda kv: kv[1], reverse=True)
    if not items:
        return set()
    keep_n = max(1, int(round(keep_quantile * len(items))))
    return {h for h, _ in items[:keep_n]}
