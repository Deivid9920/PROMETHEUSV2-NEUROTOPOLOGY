"""Shared fixtures: the real repo config and an isolated tiny repo.

The tiny repo fixture materializes a complete working environment
(config.yaml, cleaned docs, tokenizer, frozen holdout) inside
``tmp_path`` so trainer, loop and eval tests run in seconds without
touching the repository state.
"""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from prometheus_ns import load_config, repo_root
from prometheus_ns.model.tokenizer_train import train_tokenizer_from_files

REPO = repo_root()

_SUBJECTS = [
    ("the quick brown fox", "jumps over", "a lazy dog near the riverbank"),
    ("the old lighthouse keeper", "lights", "the lamp before sunset"),
    ("a young prince", "opened", "the ancient door of the castle"),
    ("the village baker", "bakes", "fresh bread every morning"),
    ("the tiny robot", "watches", "the stars through the window"),
    ("a quiet scholar", "reads", "old manuscripts in the library"),
    ("the fisherman", "repairs", "his net by the harbor"),
    ("the mountain guide", "climbs", "the eastern ridge at dawn"),
    ("a small orchestra", "plays", "in the town square"),
    ("the gardener", "waters", "the roses behind the greenhouse"),
]

_FILLER = (
    "The story continues with quiet details about the day, the weather "
    "and the small habits of the people who live there. Everyone knows "
    "everyone, and the rhythm of the seasons shapes the work and the "
    "conversation. "
)


def _make_doc(rng: random.Random, index: int) -> str:
    """One synthetic English document of roughly 1.2k characters."""
    parts: list[str] = [f"Document number {index} describes a small village scene."]
    for _ in range(6):
        subject, verb, obj = rng.choice(_SUBJECTS)
        parts.append(f"{subject.capitalize()} {verb} {obj}. {_FILLER}")
    return "".join(parts)


@pytest.fixture(scope="session")
def config() -> dict:
    """The repository's real config.yaml."""
    return load_config(REPO / "config.yaml")


@pytest.fixture
def tiny_repo(tmp_path: Path) -> dict:
    """A complete isolated micro-repo for fast end-to-end tests.

    Returns a dict with the loaded ``cfg``, the trained tiny
    ``tokenizer`` and the relevant paths.
    """
    root = tmp_path / "repo"
    for sub in ("data_clean/gutenberg", "data_clean/holdout", "data_raw/gutenberg.org", "logs", "docs", "triplets", "artifacts/checkpoints", "eval"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(7)
    docs: list[Path] = []
    for i in range(24):
        text = _make_doc(rng, i)
        dhash = sha256(text.encode("utf-8")).hexdigest()
        path = root / "data_clean" / "gutenberg" / f"{dhash[:12]}.txt"
        path.write_text(text, encoding="utf-8")
        docs.append(path)
        with (root / "data_clean" / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"hash": dhash, "source": "gutenberg", "path": f"gutenberg/{dhash[:12]}.txt", "chars": len(text)}) + "\n")

    cfg_dict = {
        "seed": 42,
        "data": {
            "sources": ["gutenberg"],
            "max_pages_per_source": 2,
            "out_dir": "data_raw",
            "user_agent": "PrometheusNS-Research/0.1",
            "rps": 50.0,
            "timeout_s": 5,
            "retries": 1,
            "holdout_frac": 0.02,
        },
        "clean": {
            "min_doc_chars": 100,
            "max_doc_chars": 100000,
            "minhash_perms": 32,
            "minhash_jaccard": 0.8,
            "keep_quantile": 0.8,
            "shingle_words": 5,
            "flush_every_docs": 200000,
        },
        "tokenizer": {"vocab_nano": 512, "vocab_small": 512, "sample_mb": 10},
        "model": {
            "profile": "tiny",
            "tiny": {
                "d_model": 64,
                "n_layer": 2,
                "n_head": 4,
                "n_kv_head": 2,
                "d_ff": 128,
                "max_seq": 64,
                "vocab": 512,
                "tie_embeddings": True,
            },
        },
        "train": {
            "lr_nano": 3.0e-4,
            "lr_small": 2.5e-4,
            "warmup_frac": 0.2,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "micro_batch": {"cpu": 2, "gpu": 2},
            "grad_accum": {"cpu": 1, "gpu": 1},
            "tokens_per_round_cpu": 5.0e7,
            "tokens_per_round_gpu": 5.0e8,
            "eval_every_steps": 2,
            "use_bf16_gpu": False,
            "grad_checkpoint_gpu": False,
            "checkpoint_every_steps": 1000,
            "continue_lr_frac": 0.5,
            "min_lr_frac": 0.1,
        },
        "inference": {
            "max_new_tokens": 12,
            "temperature": 0.8,
            "top_k": 20,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
        },
        "symbolic": {
            "spacy_model": "en_core_web_sm",
            "min_confidence": 0.6,
            "disjoint_classes": ["bird", "mammal", "fish", "insect", "vehicle", "food", "plant", "tool"],
            "exclusive_pairs": [["alive", "dead"]],
            "max_graph_entities": 200000,
        },
        "loop": {
            "max_rounds": 6,
            "human_gate_every": 3,
            "min_ppl_gain": 0.01,
            "max_diversity_drop": 0.20,
            "holdout_sha256": None,
            "round_timeout_s": 10800,
            "new_docs_per_round": 50000,
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)

    train_docs = sorted(p for p in (root / "data_clean" / "gutenberg").glob("*.txt"))
    tokenizer = train_tokenizer_from_files(train_docs, 512)

    return {"cfg": cfg, "tokenizer": tokenizer, "root": root, "docs": docs}
