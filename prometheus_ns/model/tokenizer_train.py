"""BPE tokenizer training over the cleaned corpus.

Special tokens are registered through the trainer's ``special_tokens``
parameter so they enter the vocabulary as ``AddedToken`` entries with
``special=True``; appending them as plain corpus text would make the
BPE model split them into sub-tokens. The trained vocabulary size is
the profile's ``model.vocab`` and includes the four specials.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from prometheus_ns import cfg_get, load_config, repo_path

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")


def build_trainer(vocab_size: int) -> BpeTrainer:
    """BPE trainer with the four specials reserved inside ``vocab_size``."""
    specials = [AddedToken(tok, special=True, normalized=False) for tok in SPECIAL_TOKENS]
    return BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )


def train_tokenizer_from_files(files: list, vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer over ``files``.

    Args:
        files: Cleaned corpus items, processed in the given order. Each
            item is either a ``Path`` (read from disk) or a ``str``
            (used as the document text directly).
        vocab_size: Final vocabulary size, specials included.

    Returns:
        The trained :class:`tokenizers.Tokenizer`.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    def corpus_iter():
        for item in files:
            if isinstance(item, str):
                text = item
            else:
                with Path(item).open("r", encoding="utf-8") as fh:
                    text = fh.read()
            if text:
                yield text

    trainer = build_trainer(vocab_size)
    tokenizer.train_from_iterator(corpus_iter(), trainer=trainer)
    return tokenizer


def sample_corpus_files(clean_dir: Path, sample_mb: float, seed: int, exclude: Path | None = None) -> list[Path]:
    """Deterministically sample cleaned files up to ``sample_mb`` megabytes."""
    excluded = {p.resolve() for p in exclude.glob("**/*.txt")} if exclude and exclude.exists() else set()
    candidates = sorted(p for p in clean_dir.glob("**/*.txt") if p.resolve() not in excluded)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    budget = sample_mb * 1024 * 1024
    selected: list[Path] = []
    used = 0
    for path in candidates:
        size = path.stat().st_size
        if used + size > budget:
            continue
        selected.append(path)
        used += size
    return selected


def _read_doc(item: str | Path) -> str:
    """Read a corpus item that is either a file path or literal text."""
    if isinstance(item, Path):
        return item.read_text(encoding="utf-8")
    looks_like_path = len(item) < 240 and "\n" not in item and Path(item).exists()
    return Path(item).read_text(encoding="utf-8") if looks_like_path else item


def fertility(tokenizer: Tokenizer, files: list, max_files: int = 200) -> float:
    """Average tokens per whitespace word over a corpus sample."""
    total_tokens = 0
    total_words = 0
    for item in files[:max_files]:
        text = _read_doc(item)
        if not text.strip():
            continue
        n_words = len(text.split())
        if n_words == 0:
            continue
        total_words += n_words
        total_tokens += len(tokenizer.encode(text).ids)
    if total_words == 0:
        return float("inf")
    return total_tokens / total_words


def train_tokenizer(cfg: dict) -> dict:
    """Train and persist the tokenizer for the active model profile.

    Returns:
        Report dictionary also persisted to
        ``artifacts/tokenizer/report.json``.
    """
    profile = cfg_get(cfg, "model.profile", "nano")
    if profile == "nano":
        vocab = int(cfg_get(cfg, "tokenizer.vocab_nano", 8000))
    else:
        vocab = int(cfg_get(cfg, "tokenizer.vocab_small", 16000))
    sample_mb = float(cfg_get(cfg, "tokenizer.sample_mb", 200))
    seed = int(cfg_get(cfg, "seed", 42))

    clean_dir = repo_path(cfg, "data_clean")
    holdout = clean_dir / "holdout"
    files = sample_corpus_files(clean_dir, sample_mb, seed, exclude=holdout)
    if not files:
        raise RuntimeError("no cleaned corpus found: run the data pipeline first (make data)")

    started = time.time()
    tokenizer = train_tokenizer_from_files(files, vocab)
    elapsed = time.time() - started

    out_dir = repo_path(cfg, "artifacts", "tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    ftl = fertility(tokenizer, files)
    chars = sum(p.stat().st_size for p in files)
    report = {
        "profile": profile,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokens_per_word": round(ftl, 4),
        "train_seconds": round(elapsed, 2),
        "chars_per_second": round(chars / max(elapsed, 1e-9), 1),
        "files_used": len(files),
        "MB_used": round(chars / (1024 * 1024), 2),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_tokenizer(cfg: dict) -> Tokenizer:
    """Load the persisted tokenizer from ``artifacts/tokenizer``."""
    path = repo_path(cfg, "artifacts", "tokenizer", "tokenizer.json")
    if not path.exists():
        raise RuntimeError(f"tokenizer artifact missing: {path} (run make tokenize)")
    return Tokenizer.from_file(str(path))


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.model.tokenizer_train --config PATH``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS BPE tokenizer training")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = train_tokenizer(cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
