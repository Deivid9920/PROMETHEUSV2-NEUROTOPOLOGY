"""Corpus diversity guard: distinct-bigram ratio over a doc sample.

The ratio is computed over up to 10k documents per round and appended
to ``logs/diversity.jsonl`` so the promotion gate can compare rounds.
Lexical ratio naturally varies by domain; the guard exists to catch
model-collapse style repetition, not to judge domains.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

_WORD_RE = re.compile(r"[a-z']+")


def distinct_bigram_ratio(texts: list[str], max_docs: int = 10000) -> float:
    """Distinct-bigram ratio over up to ``max_docs`` documents.

    The ratio is ``unique bigrams / total bigrams``; documents with
    fewer than two words contribute nothing to either side.
    """
    unique: set[tuple[str, str]] = set()
    total = 0
    for text in texts[:max_docs]:
        words = _WORD_RE.findall(text.lower())
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            unique.add(bigram)
            total += 1
    return round(len(unique) / total, 6) if total else 0.0


def measure_and_record(docs_dir: Path, round_id: int, logs_dir: Path, max_docs: int = 10000) -> float:
    """Measure the round's corpus diversity and append the baseline row.

    Returns:
        The measured distinct-bigram ratio.
    """
    docs = sorted(docs_dir.glob("**/*.txt"))
    texts = []
    for path in docs[:max_docs]:
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    ratio = distinct_bigram_ratio(texts, max_docs=max_docs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "diversity.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"round_id": round_id, "ratio": ratio, "n_docs": len(texts), "ts": time.time()}) + "\n")
    return ratio
