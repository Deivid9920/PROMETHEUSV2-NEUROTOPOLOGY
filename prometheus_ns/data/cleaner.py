"""Streaming cleaner: normalize, deduplicate and filter extracted text.

The cleaner is the memory-sensitive heart of the data pipeline:

- text normalization runs through ``ftfy`` plus NFC and whitespace
  collapsing, line by line, so documents keep paragraph structure;
- exact deduplication uses SHA-256 over the normalized text, contrasted
  against the current batch, the historical index in
  ``data_clean/index.jsonl`` AND the frozen holdout (test-leak guard);
- near-duplicate detection uses MinHash with 128 permutations over
  5-word paragraph shingles and an LSH index at threshold 0.8. The
  index is flushed every ``clean.flush_every_docs`` documents to bound
  RAM. Documents without enough words for a single shingle skip the
  near-dup check instead of raising on empty shingle sets;
- Gopher heuristics filter by source-aware symbol ratio;
- a final quality-quantile pass keeps the top ``clean.keep_quantile``
  fraction of surviving candidates.
"""

from __future__ import annotations

import ftfy
import json
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from prometheus_ns import cfg_get, ensure_dir, load_config, repo_path
from prometheus_ns.data import extractor as _extractor
from prometheus_ns.data import quality as _quality


def normalize_text(text: str) -> str:
    """Repair unicode, apply NFC and collapse whitespace per line."""
    fixed = ftfy.fix_text(text)
    fixed = unicodedata.normalize("NFC", fixed)
    lines = []
    for raw_line in fixed.splitlines():
        collapsed = " ".join(raw_line.split())
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def unwrap_paragraphs(text: str) -> str:
    """Rebuild paragraphs from hard-wrapped prose.

    Plain-text sources (Project Gutenberg) break sentences across
    ~80-column lines, so a line-level punctuation heuristic would fail
    on almost every book. Lines that do not end a sentence are joined
    with the next one; blank lines flush the current paragraph.
    """
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(stripped)
        if stripped.endswith((".", "!", "?", '"', "'", ":", ";")):
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n".join(paragraphs)


def doc_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized text (exact-dedup key)."""
    return sha256(text.encode("utf-8")).hexdigest()


def paragraph_shingles(text: str, shingle_words: int) -> set[bytes]:
    """Build the union of word shingles over every paragraph.

    Paragraphs shorter than ``shingle_words`` contribute nothing, which
    keeps very short paragraphs from producing empty or degenerate
    shingle sets; a document whose every paragraph is short therefore
    has no shingles and skips near-dup checking.
    """
    shingles: set[bytes] = set()
    for paragraph in text.split("\n"):
        words = paragraph.lower().split()
        if len(words) < shingle_words:
            continue
        for i in range(len(words) - shingle_words + 1):
            gram = " ".join(words[i:i + shingle_words])
            shingles.add(gram.encode("utf-8"))
    return shingles


def minhash_from_shingles(shingles: set[bytes], num_perm: int) -> MinHash | None:
    """Build a MinHash signature; None when the shingle set is empty."""
    if not shingles:
        return None
    mh = MinHash(num_perm=num_perm)
    for gram in shingles:
        mh.update(gram)
    return mh


@dataclass
class _Stats:
    input_files: int = 0
    extracted: int = 0
    extraction_failed: int = 0
    challenge_page: int = 0
    length: int = 0
    dup_exact: int = 0
    dup_near: int = 0
    holdout_contamination: int = 0
    symbol_ratio: int = 0
    mean_word_len: int = 0
    punct_lines: int = 0
    stopword_ratio: int = 0
    dropped_quantile: int = 0
    kept: int = 0
    dup_pairs: list = field(default_factory=list)

    def as_dict(self) -> dict:
        discarded = {
            "length": self.length,
            "dup_exact": self.dup_exact,
            "dup_near": self.dup_near,
            "holdout_contamination": self.holdout_contamination,
            "symbol_ratio": self.symbol_ratio,
            "mean_word_len": self.mean_word_len,
            "punct_lines": self.punct_lines,
            "stopword_ratio": self.stopword_ratio,
            "quantile": self.dropped_quantile,
            "extraction_failed": self.extraction_failed,
            "challenge_page": self.challenge_page,
        }
        return {
            "input_files": self.input_files,
            "extracted": self.extracted,
            "discarded": discarded,
            "kept": self.kept,
            "dup_pairs_sample": self.dup_pairs[:100],
        }


def _load_hashes(index_path: Path) -> set[str]:
    """Load the historical exact-dedup hashes from the index."""
    hashes: set[str] = set()
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    hashes.add(json.loads(line)["hash"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return hashes


def _holdout_hashes(holdout_dir: Path) -> set[str]:
    return {p.stem for p in holdout_dir.glob("*.txt")}


class _NearDupIndex:
    """MinHash LSH wrapper with periodic flush and pair bookkeeping."""

    def __init__(self, num_perm: int, threshold: float, flush_every: int) -> None:
        self._num_perm = num_perm
        self._threshold = threshold
        self._flush_every = max(1, flush_every)
        self._count = 0
        self._build()

    def _build(self) -> None:
        self._lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        self._count = 0

    def is_near_dup(self, mh: MinHash, key: str) -> str | None:
        """Return the key of a near duplicate, inserting otherwise."""
        hits = self._lsh.query(mh)
        if hits:
            return str(hits[0])
        self._lsh.insert(key, mh)
        self._count += 1
        if self._count >= self._flush_every:
            self._build()  # RAM guard: drop the index window
        return None


def iter_raw_files(raw_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(source, path)`` for every raw crawl artifact, sorted."""
    for path in sorted(raw_dir.glob("*/*.raw")):
        yield path.parent.name, path


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split an over-long document at paragraph boundaries.

    Book-length sources (Project Gutenberg) exceed any reasonable
    per-document cap; dropping them wholesale would surrender the
    highest-quality prose in the corpus. Chunks stay within the
    ``max_doc_chars`` contract and each one keeps its own hash so the
    deduplication and quality stages operate on contract-sized docs.
    """
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n"):
        if size + len(paragraph) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def clean_corpus(cfg: dict) -> dict:
    """Run the full cleaning pass over ``data_raw`` and rebuild ``data_clean``.

    Returns:
        A stats dictionary also persisted to ``logs/clean_stats.json``.
    """
    raw_dir = repo_path(cfg, cfg_get(cfg, "data.out_dir", "data_raw"))
    clean_dir = repo_path(cfg, "data_clean")
    holdout_dir = clean_dir / "holdout"
    candidates_dir = ensure_dir(clean_dir / "_candidates")
    index_path = clean_dir / "index.jsonl"

    min_doc = int(cfg_get(cfg, "clean.min_doc_chars", 500))
    max_doc = int(cfg_get(cfg, "clean.max_doc_chars", 100000))
    num_perm = int(cfg_get(cfg, "clean.minhash_perms", 128))
    jaccard = float(cfg_get(cfg, "clean.minhash_jaccard", 0.8))
    shingle_words = int(cfg_get(cfg, "clean.shingle_words", 5))
    flush_every = int(cfg_get(cfg, "clean.flush_every_docs", 200000))
    keep_quantile = float(cfg_get(cfg, "clean.keep_quantile", 0.8))
    symbol_max = float(cfg_get(cfg, "clean.symbol_ratio_max", 0.1))
    arxiv_symbol_max = float(cfg_get(cfg, "clean.arxiv_symbol_ratio_max", 0.2))
    punct_min = float(cfg_get(cfg, "clean.punct_line_frac_min", 0.8))
    arxiv_punct_min = float(cfg_get(cfg, "clean.arxiv_punct_line_frac_min", 0.5))
    gopher = {
        "mean_word_len_min": float(cfg_get(cfg, "clean.mean_word_len_min", 3.0)),
        "mean_word_len_max": float(cfg_get(cfg, "clean.mean_word_len_max", 10.0)),
        "punct_line_frac_min": punct_min,
        "stopword_ratio_min": float(cfg_get(cfg, "clean.stopword_ratio_min", 0.3)),
    }

    stats = _Stats()
    seen_exact = _load_hashes(index_path)
    seen_exact |= _holdout_hashes(holdout_dir)
    near_dup = _NearDupIndex(num_perm, jaccard, flush_every)
    holdout_near = _NearDupIndex(num_perm, jaccard, flush_every)
    for holdout_doc in sorted(holdout_dir.glob("*.txt")):
        mh = minhash_from_shingles(paragraph_shingles(holdout_doc.read_text(encoding="utf-8"), shingle_words), num_perm)
        if mh is not None:
            holdout_near.is_near_dup(mh, holdout_doc.stem)

    # Phase 1: extract, normalize, dedup, filter; write surviving candidates.
    scored: list[tuple[str, str, float]] = []  # (hash, source, score)

    def process_chunk(chunk: str, source: str) -> bool:
        """Dedup, filter and stage one contract-sized chunk.

        Returns True when the chunk was written as a candidate.
        """
        if len(chunk) < min_doc:
            stats.length += 1
            return False
        dhash = doc_hash(chunk)
        if dhash in seen_exact:
            stats.dup_exact += 1
            return False
        seen_exact.add(dhash)
        shingles = paragraph_shingles(chunk, shingle_words)
        mh = minhash_from_shingles(shingles, num_perm)
        if mh is not None:
            dup_of = near_dup.is_near_dup(mh, dhash)
            if dup_of is not None:
                stats.dup_near += 1
                stats.dup_pairs.append([dhash[:12], str(dup_of)[:12]])
                return False
            if _query_holdout(holdout_near, mh):
                stats.holdout_contamination += 1
                return False
        effective_symbol_max = arxiv_symbol_max if source == "arxiv" else symbol_max
        effective_punct_min = arxiv_punct_min if source == "arxiv" else punct_min
        reasons = _quality.gopher_checks(
            chunk,
            symbol_ratio_max=effective_symbol_max,
            punct_line_frac_min=effective_punct_min,
            **{k: v for k, v in gopher.items() if k != "punct_line_frac_min"},
        )
        if reasons:
            for reason in reasons:
                if reason == "symbol_ratio":
                    stats.symbol_ratio += 1
                elif reason == "mean_word_len":
                    stats.mean_word_len += 1
                elif reason == "punct_lines":
                    stats.punct_lines += 1
                elif reason == "stopword_ratio":
                    stats.stopword_ratio += 1
            return False
        score = _quality.quality_score(chunk, symbol_ratio_max=effective_symbol_max)
        (candidates_dir / f"{dhash[:12]}.txt").write_text(chunk, encoding="utf-8")
        scored.append((dhash, source, score))
        return True

    for source, path in iter_raw_files(raw_dir):
        stats.input_files += 1
        html = path.read_bytes().decode("utf-8", errors="replace")
        text = _extractor.extract_text(html, source=source)
        if text is None:
            if _extractor.looks_like_challenge(html):
                stats.challenge_page += 1
            else:
                stats.extraction_failed += 1
            continue
        text = normalize_text(text)
        text = unwrap_paragraphs(text)
        stats.extracted += 1
        for chunk in _split_into_chunks(text, max_doc):
            process_chunk(chunk, source)

    # Phase 2: quality quantile cut over all surviving candidates.
    keep = _quality.keep_by_quantile(((h, s) for h, _, s in scored), keep_quantile)
    index_lines: list[str] = []
    if index_path.exists():
        index_lines = index_path.read_text(encoding="utf-8").splitlines()
    with index_path.open("a", encoding="utf-8") as index_fh:
        for dhash, source, _score in scored:
            candidate = candidates_dir / f"{dhash[:12]}.txt"
            if dhash in keep:
                final_dir = ensure_dir(clean_dir / source)
                final = final_dir / f"{dhash[:12]}.txt"
                candidate.replace(final)
                index_fh.write(json.dumps({"hash": dhash, "source": source, "path": str(final.relative_to(clean_dir)), "chars": len(final.read_text(encoding='utf-8'))}, ensure_ascii=False) + "\n")
                stats.kept += 1
            else:
                stats.dropped_quantile += 1
                candidate.unlink(missing_ok=True)
    for leftover in candidates_dir.glob("*.txt"):
        leftover.unlink(missing_ok=True)
    candidates_dir.rmdir()

    out = stats.as_dict()
    log_path = repo_path(cfg, "logs", "clean_stats.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _query_holdout(holdout_lsh: _NearDupIndex, mh: MinHash) -> bool:
    """Check a signature against the holdout index without inserting.

    ``_NearDupIndex.is_near_dup`` inserts on miss; the holdout index must
    stay read-only for candidate queries, so misses are tolerated.
    """
    hits = holdout_lsh._lsh.query(mh)  # noqa: SLF001 - deliberate internal use
    return bool(hits)


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.data.cleaner --config PATH``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS corpus cleaner")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    stats = clean_corpus(cfg)
    print(f"clean finished: kept {stats['kept']} docs, discarded {sum(stats['discarded'].values())}")


if __name__ == "__main__":
    main()
