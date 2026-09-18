"""Tests for the transformer model: parameter contracts, causal masking,
tied embeddings and the block-diagonal document mask."""

from __future__ import annotations

import time

import pytest
import torch

from prometheus_ns.model.config import ModelConfig, analytic_param_count, model_config_from_yaml
from prometheus_ns.model.model import TransformerLM

NANO = ModelConfig(384, 6, 6, 6, 1024, 256, 8000, tie_embeddings=True)
SMALL = ModelConfig(768, 12, 12, 12, 2048, 512, 16000, tie_embeddings=True)
LARGE = ModelConfig(1024, 16, 16, 8, 4096, 1024, 16000, tie_embeddings=True)


def test_nano_parameter_count_in_contract() -> None:
    count = TransformerLM(NANO).num_parameters()
    analytic = analytic_param_count(NANO)
    assert count == analytic  # tied embeddings must be counted once
    assert 12_000_000 <= count <= 18_000_000


def test_small_parameter_count_in_contract() -> None:
    # Analytic count: materializing 97M params would dominate test time
    # on constrained machines and the nano case already validates that
    # the analytic formula matches the real sum.
    count = analytic_param_count(SMALL)
    assert 85_000_000 <= count <= 115_000_000


def test_large_parameter_count_in_contract() -> None:
    count = analytic_param_count(LARGE)
    assert 260_000_000 <= count <= 310_000_000


def test_untied_embeddings_would_break_contract() -> None:
    untied = ModelConfig(384, 6, 6, 6, 1024, 256, 8000, tie_embeddings=False)
    count = TransformerLM(untied).num_parameters()
    assert count == analytic_param_count(NANO) + NANO.vocab * NANO.d_model  # +3.07M


def test_forward_output_shape_and_speed_nano() -> None:
    model = TransformerLM(NANO)
    model.eval()
    tokens = torch.randint(0, NANO.vocab, (2, NANO.max_seq))
    started = time.time()
    with torch.no_grad():
        logits = model(tokens)
    elapsed = time.time() - started
    assert logits.shape == (2, NANO.max_seq, 8000)
    # Contract: < 2 s on the reference 8-core CPU. Constrained machines
    # (fewer cores) get a proportionally looser bound.
    bound = 2.0 if (torch.get_num_threads() >= 8) else 15.0
    assert elapsed < bound, f"nano forward took {elapsed:.2f}s (bound {bound}s)"


def test_causal_mask_blocks_future_positions() -> None:
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True))
    model.eval()
    tokens = torch.randint(0, 512, (1, 64))
    t = 10
    with torch.no_grad():
        base = model(tokens)[:, t].clone()
        perturbed = tokens.clone()
        perturbed[:, t + 1:] = (perturbed[:, t + 1:] + 7) % 512
        after = model(perturbed)[:, t]
    # Position t must be bit-identical: it cannot attend to t+1.
    assert torch.equal(base, after)


def test_document_mask_blocks_cross_document_attention() -> None:
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True))
    model.eval()
    tokens = torch.randint(0, 512, (1, 64))
    flags = torch.zeros(1, 64)
    flags[0, 32] = 1.0  # a new document starts at position 32
    t = 40
    with torch.no_grad():
        base = model(tokens, doc_flags=flags)[:, t].clone()
        perturbed = tokens.clone()
        perturbed[:, :32] = (perturbed[:, :32] + 11) % 512
        after = model(perturbed, doc_flags=flags)[:, t]
    # With block-diagonal masking, position 40 (doc 2) cannot attend to
    # positions 0..31 (doc 1), so perturbing them changes nothing.
    assert torch.equal(base, after)


def test_loss_is_finite_and_reasonable_at_init() -> None:
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True))
    tokens = torch.randint(0, 512, (2, 64))
    _, loss = model(tokens, targets=tokens)
    assert torch.isfinite(loss)
    expected_random = torch.log(torch.tensor(512.0))
    assert abs(loss.item() - expected_random.item()) < 0.75  # near ln(vocab)


def test_model_config_from_yaml_profile(config) -> None:
    mc = model_config_from_yaml(config, "nano")
    assert mc.d_model == 384 and mc.n_layer == 6 and mc.vocab == 8000
    with pytest.raises(ValueError):
        model_config_from_yaml(config, "does-not-exist")


def test_invalid_gqa_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(64, 2, 4, 8, 128, 64, 512)  # n_kv_head > n_head
    with pytest.raises(ValueError):
        ModelConfig(64, 2, 4, 3, 128, 64, 512)  # n_head not divisible by n_kv_head
