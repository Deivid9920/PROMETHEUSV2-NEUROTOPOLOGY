"""Streaming packed dataset backed by numpy memmaps.

Documents are tokenized on the fly during packing, joined with the
``<|eos|>`` separator, and cut into non-overlapping windows of
``max_seq``. Two parallel memmaps are produced per split:

- ``{split}.bin``  uint16 token ids (vocab <= 65535 fits);
- ``{split}.doc``  uint8 flags, 1 marking the first token of a document.

The doc flags drive block-diagonal attention so a packed window does
not mix unrelated documents. Window ``i`` is exactly
``tokens[i*max_seq:(i+1)*max_seq]``: consecutive windows never overlap
and no token is skipped, which is what the resume contract relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from prometheus_ns import cfg_get, repo_path


def _split_doc(text: str, tokenizer, max_seq: int) -> tuple[list[int], list[int]]:
    """Tokenize one document; return ids and same-length doc-start flags."""
    ids = tokenizer.encode(text).ids
    ids.append(int(tokenizer.token_to_id("<|eos|>")))
    flags = [1] + [0] * (len(ids) - 1)
    return ids, flags


def pack_split(
    tokenizer,
    docs: list[Path],
    max_seq: int,
    out_prefix: Path,
) -> int:
    """Pack ``docs`` into memmap windows at ``out_prefix`` (.bin and .doc).

    Returns:
        Number of windows written. The final window is zero-padded.
    """
    token_buf: list[int] = []
    flag_buf: list[int] = []
    windows_tokens: list[np.ndarray] = []
    windows_flags: list[np.ndarray] = []

    for path in docs:
        ids, flags = _split_doc(path.read_text(encoding="utf-8"), tokenizer, max_seq)
        token_buf.extend(ids)
        flag_buf.extend(flags)
        while len(token_buf) >= max_seq:
            windows_tokens.append(np.asarray(token_buf[:max_seq], dtype=np.uint16))
            windows_flags.append(np.asarray(flag_buf[:max_seq], dtype=np.uint8))
            token_buf = token_buf[max_seq:]
            flag_buf = flag_buf[max_seq:]

    if token_buf:
        pad = max_seq - len(token_buf)
        windows_tokens.append(np.asarray(token_buf + [0] * pad, dtype=np.uint16))
        windows_flags.append(np.asarray(flag_buf + [0] * pad, dtype=np.uint8))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens_arr = np.stack(windows_tokens) if windows_tokens else np.zeros((0, max_seq), dtype=np.uint16)
    flags_arr = np.stack(windows_flags) if windows_flags else np.zeros((0, max_seq), dtype=np.uint8)
    np.save(str(out_prefix) + ".bin.npy", tokens_arr)
    np.save(str(out_prefix) + ".doc.npy", flags_arr)
    return int(tokens_arr.shape[0])


def _load_windows(prefix: Path) -> tuple[np.memmap, np.memmap]:
    tokens = np.load(str(prefix) + ".bin.npy", mmap_mode="r")
    flags = np.load(str(prefix) + ".doc.npy", mmap_mode="r")
    return tokens, flags


def split_docs_for_training(cfg: dict, tokenizer) -> tuple[list[Path], list[Path]]:
    """Deterministically split cleaned docs into train and val sets.

    Documents are ordered by their file name (the first 12 chars of
    their content hash) and the last ``holdout_frac``-derived share of
    the non-holdout corpus becomes the validation slice. The frozen
    holdout directory never participates.
    """
    clean_dir = repo_path(cfg, "data_clean")
    holdout = clean_dir / "holdout"
    excluded = {p.resolve() for p in holdout.glob("**/*.txt")}
    docs = sorted(p for p in clean_dir.glob("**/*.txt") if p.resolve() not in excluded)
    val_frac = float(cfg_get(cfg, "data.holdout_frac", 0.02))
    val_n = max(1, int(round(val_frac * len(docs)))) if docs else 0
    return docs[:-val_n] if val_n else docs, docs[-val_n:] if val_n else []


def build_packed_dataset(
    cfg: dict, tokenizer, train_docs: list[Path] | None = None
) -> tuple[Path, Path, int, int]:
    """Pack train and val splits; return (train_prefix, val_prefix, n_train, n_val).

    ``train_docs`` lets the caller pass a preselected training set
    (quarantine filter and replay buffer already applied); when it is
    ``None`` the full cleaned corpus is used.
    """
    profile = cfg_get(cfg, "model.profile", "nano")
    max_seq = int(cfg_get(cfg, f"model.{profile}.max_seq", 256))
    packed_dir = repo_path(cfg, "data_clean", "packed")
    if train_docs is None:
        train_docs, val_docs = split_docs_for_training(cfg, tokenizer)
    else:
        _, val_docs = split_docs_for_training(cfg, tokenizer)
    train_prefix = packed_dir / "train"
    val_prefix = packed_dir / "val"
    n_train = pack_split(tokenizer, train_docs, max_seq, train_prefix)
    n_val = pack_split(tokenizer, val_docs, max_seq, val_prefix)
    return train_prefix, val_prefix, n_train, n_val


class PackedDataset(Dataset):
    """Random-access view over one packed split."""

    def __init__(self, prefix: Path) -> None:
        self.tokens, self.flags = _load_windows(prefix)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def max_seq(self) -> int:
        return int(self.tokens.shape[1])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.tokens[index], self.flags[index]


def window_slice(prefix: Path, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Read windows ``[start, stop)`` from a packed split (memmap-backed)."""
    tokens, flags = _load_windows(prefix)
    return tokens[start:stop], flags[start:stop]
