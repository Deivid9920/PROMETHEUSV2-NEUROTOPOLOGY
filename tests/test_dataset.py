"""Tests for the packed streaming dataset: windowing, flags, no leaks."""

from __future__ import annotations

import numpy as np

from prometheus_ns.train.dataset import PackedDataset, pack_split, split_docs_for_training


def _build(tiny_repo):
    cfg = tiny_repo["cfg"]
    tokenizer = tiny_repo["tokenizer"]
    docs = sorted(p for p in (tiny_repo["root"] / "data_clean" / "gutenberg").glob("*.txt"))
    from prometheus_ns import repo_path

    packed_dir = repo_path(cfg, "data_clean", "packed")
    n_windows = pack_split(tokenizer, docs, 64, packed_dir / "train")
    return cfg, tokenizer, docs, packed_dir, n_windows


def test_packing_covers_all_tokens_without_overlap(tiny_repo) -> None:
    _cfg, tokenizer, docs, packed_dir, n_windows = _build(tiny_repo)
    stream: list[int] = []
    eos = tokenizer.token_to_id("<|eos|>")
    for doc in docs:
        stream.extend(tokenizer.encode(doc.read_text(encoding="utf-8")).ids)
        stream.append(eos)
    total = len(stream)
    assert n_windows * 64 >= total  # last window zero-padded
    assert (n_windows - 1) * 64 < total  # no fully-empty trailing window

    dataset = PackedDataset(packed_dir / "train")
    assert len(dataset) == n_windows
    flat = np.concatenate([np.asarray(dataset[i][0]) for i in range(n_windows)])
    assert list(flat[:total]) == stream  # exact order, no skips, no overlap
    assert all(v == 0 for v in flat[total:])  # padding only in the tail


def test_doc_flags_mark_every_document_start(tiny_repo) -> None:
    _cfg, _tokenizer, docs, packed_dir, n_windows = _build(tiny_repo)
    dataset = PackedDataset(packed_dir / "train")
    flags = np.concatenate([np.asarray(dataset[i][1]) for i in range(n_windows)])
    assert int(flags[0]) == 1  # first token of the stream starts a doc
    assert int(flags.sum()) == len(docs)


def test_window_slice_matches_dataset(tiny_repo) -> None:
    _cfg, _tokenizer, _docs, packed_dir, _n = _build(tiny_repo)
    dataset = PackedDataset(packed_dir / "train")
    tokens, flags = dataset.tokens[2], dataset.flags[2]
    sliced_tokens, sliced_flags = None, None
    from prometheus_ns.train.dataset import window_slice

    sliced_tokens, sliced_flags = window_slice(packed_dir / "train", 2, 3)
    assert np.asarray(sliced_tokens)[0].tolist() == np.asarray(tokens).tolist()
    assert np.asarray(sliced_flags)[0].tolist() == np.asarray(flags).tolist()


def test_split_docs_excludes_holdout_and_is_deterministic(tiny_repo) -> None:
    cfg, tokenizer = tiny_repo["cfg"], tiny_repo["tokenizer"]
    holdout = tiny_repo["root"] / "data_clean" / "holdout"
    (holdout / "frozen.txt").write_text("frozen holdout document", encoding="utf-8")

    train_docs, val_docs = split_docs_for_training(cfg, tokenizer)
    train_docs2, val_docs2 = split_docs_for_training(cfg, tokenizer)
    assert train_docs == train_docs2 and val_docs == val_docs2
    all_names = {p.name for p in train_docs} | {p.name for p in val_docs}
    assert "frozen.txt" not in all_names
    assert val_docs, "val split must be non-empty for the 2% rule"
