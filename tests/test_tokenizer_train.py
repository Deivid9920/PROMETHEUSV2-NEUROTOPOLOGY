"""Tests for the BPE tokenizer: specials integrity and fertility."""

from __future__ import annotations

import random

from prometheus_ns.model.tokenizer_train import (
    SPECIAL_TOKENS,
    fertility,
    sample_corpus_files,
    train_tokenizer_from_files,
)

# Natural-language docs (used for the fertility measurement).
NATURAL_DOCS = [
    "The old lighthouse keeper lit the lamp every evening before the ships crossed the bay. "
    "The beam swept the water and the gulls scattered over the rocks. " * 4,
    "In the mountain village the innkeeper fed every traveler who crossed the snowy pass. "
    "She trusted the weather more than any printed almanac. " * 4,
    "The miller woke before dawn and opened the wooden doors while mist covered the river. "
    "Farmers brought sacks of wheat from the surrounding valleys. " * 4,
]

# Rich synthetic corpus: a large pool of distinct common English words
# so the BPE trainer has enough merge candidates to fill a 512-entry
# vocabulary and natural text stays fertile (< 2 tokens/word).
_ENGLISH_WORDS = """the of and to in a is that it for as with was his on be at by i this had not are but from or
have an they which one you were her all she there would their we him been has when who will more no if out so
said what up its about into than them can only other new some could time these two may then do first any my
now such like our over man me even most made after also did many before must through back years where much
your way well down should because each just those people mr how too little state good very make world still
own see men work long get here between both life being under never day same another know while last might us
great old year off come since against go came right used take three""".split()

_rng = random.Random(11)
RICH_DOCS = [
    " ".join(_rng.choice(_ENGLISH_WORDS) for _ in range(200))
    for _ in range(10)
]

# Training mix: the natural docs repeated for Zipfian frequency (so
# common words fuse into single tokens) plus the random-soup docs for
# extra merge diversity, filling the 512-entry vocabulary.
TRAIN_DOCS = NATURAL_DOCS * 30 + RICH_DOCS


def test_special_tokens_survive_intact() -> None:
    tokenizer = train_tokenizer_from_files([], 512)  # specials still registered
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None, f"{token} missing from vocabulary"
        # The literal string must encode to exactly one id: the BPE model
        # must never split special tokens into sub-tokens.
        assert tokenizer.encode(token).ids == [token_id]


def test_vocab_size_is_exact() -> None:
    tokenizer = train_tokenizer_from_files(TRAIN_DOCS, 512)
    assert tokenizer.get_vocab_size() == 512


def test_encode_decode_roundtrip() -> None:
    tokenizer = train_tokenizer_from_files(TRAIN_DOCS, 512)
    text = NATURAL_DOCS[0]
    ids = tokenizer.encode(text).ids
    assert ids
    # ByteLevel is byte-lossless; add_prefix_space may prepend one space.
    assert tokenizer.decode(ids).strip() == text.strip()


def test_fertility_below_two_on_clean_english() -> None:
    tokenizer = train_tokenizer_from_files(TRAIN_DOCS, 512)
    ftl = fertility(tokenizer, NATURAL_DOCS)
    assert ftl < 2.0, f"fertility {ftl:.2f} exceeds the nano contract of 2.0"


def test_sample_corpus_files_respects_budget_and_exclusion(tmp_path) -> None:
    for i in range(6):
        (tmp_path / f"doc_{i}.txt").write_text("word " * 300, encoding="utf-8")
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "frozen_doc.txt").write_text("holdout doc", encoding="utf-8")

    selected = sample_corpus_files(tmp_path, sample_mb=1.0, seed=42, exclude=holdout)
    names = [p.name for p in selected]
    assert "frozen_doc.txt" not in names  # holdout excluded
    assert len(selected) == 6
    assert sample_corpus_files(tmp_path, 1.0, 42, holdout) == sample_corpus_files(tmp_path, 1.0, 42, holdout)


def test_sample_corpus_files_respects_budget(tmp_path) -> None:
    for i in range(4):
        (tmp_path / f"big_{i}.txt").write_text("word " * 40000, encoding="utf-8")  # ~200 KB each
    selected = sample_corpus_files(tmp_path, sample_mb=0.5, seed=42)
    total = sum(p.stat().st_size for p in selected)
    assert total <= 0.5 * 1024 * 1024
    assert len(selected) == 2  # the 0.5 MB budget fits exactly two 200 KB docs
