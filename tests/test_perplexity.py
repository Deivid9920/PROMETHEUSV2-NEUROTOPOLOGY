"""Tests for held-out perplexity evaluation with holdout integrity."""

from __future__ import annotations

import pytest
import torch
from dataclasses import asdict

from prometheus_ns.autoloop.promotion import compute_holdout_digest
from prometheus_ns.eval.perplexity import evaluate_checkpoint_on_holdout, perplexity_of_docs
from prometheus_ns.model.config import ModelConfig
from prometheus_ns.model.model import TransformerLM


@pytest.fixture
def frozen_repo(tiny_repo):
    """Tiny repo with the holdout frozen exactly once."""
    from scripts.freeze_holdout import freeze_holdout

    result = freeze_holdout(tiny_repo["cfg"])
    tiny_repo["freeze"] = result
    return tiny_repo


@pytest.fixture
def trained_checkpoint(frozen_repo):
    """A checkpoint payload for the tiny profile (untrained weights are fine)."""
    mc = ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True)
    torch.manual_seed(3)
    model = TransformerLM(mc)
    path = frozen_repo["root"] / "artifacts" / "checkpoints" / "latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "profile": asdict(mc), "step": 0, "tokens_seen": 0}, path)
    return path


def test_perplexity_of_docs_is_finite_and_above_one(frozen_repo, trained_checkpoint) -> None:
    from prometheus_ns.device import get_device

    cfg, tokenizer = frozen_repo["cfg"], frozen_repo["tokenizer"]
    holdout = sorted((frozen_repo["root"] / "data_clean" / "holdout").glob("*.txt"))
    assert holdout
    mc = ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True)
    model = TransformerLM(mc)
    ppl, tokens = perplexity_of_docs(model, tokenizer, holdout, 64, get_device())
    assert 1.0 < ppl < 1e6
    assert tokens > 0


def test_evaluate_checkpoint_verifies_holdout_digest(frozen_repo, trained_checkpoint) -> None:
    cfg, tokenizer = frozen_repo["cfg"], frozen_repo["tokenizer"]
    result = evaluate_checkpoint_on_holdout(cfg, tokenizer, trained_checkpoint)
    expected = compute_holdout_digest(frozen_repo["root"] / "data_clean" / "holdout")
    assert result["holdout_digest"] == expected
    assert result["ppl"] > 1.0
    assert result["docs"] == frozen_repo["freeze"]["docs"]


def test_evaluate_aborts_on_holdout_tampering(frozen_repo, trained_checkpoint) -> None:
    cfg, tokenizer = frozen_repo["cfg"], frozen_repo["tokenizer"]
    holdout_dir = frozen_repo["root"] / "data_clean" / "holdout"
    next(holdout_dir.glob("*.txt")).write_text("tampered contents that change the digest", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        evaluate_checkpoint_on_holdout(cfg, tokenizer, trained_checkpoint)


def test_evaluate_requires_frozen_holdout(tiny_repo, trained_checkpoint) -> None:
    import copy

    # A config whose anchor was never written must abort evaluation,
    # even when holdout files exist on disk.
    cfg = copy.deepcopy(tiny_repo["cfg"])
    cfg["loop"]["holdout_sha256"] = None
    with pytest.raises(RuntimeError, match="freeze_holdout"):
        evaluate_checkpoint_on_holdout(cfg, tiny_repo["tokenizer"], trained_checkpoint)


def test_int8_quantized_evaluation_runs(frozen_repo, trained_checkpoint) -> None:
    from torch.ao.quantization import quantize_dynamic
    from torch import nn

    cfg, tokenizer = frozen_repo["cfg"], frozen_repo["tokenizer"]
    result = evaluate_checkpoint_on_holdout(cfg, tokenizer, trained_checkpoint, int8=True)
    assert result["int8"] is True
    assert result["ppl"] > 1.0
    # The quantized model must remain a working nn.Module.
    mc = ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True)
    assert isinstance(quantize_dynamic(TransformerLM(mc), {nn.Linear}), torch.nn.Module)
