"""Tests for the promotion gate: the four contracted scenarios plus
the audit-trail helpers (promotion log, rollback, metrics reload)."""

from __future__ import annotations

import json

import pytest

from prometheus_ns.autoloop.promotion import (
    PromotionDecision,
    RoundMetrics,
    compute_holdout_digest,
    decide_promotion,
    load_round_metrics,
    rollback_to_champion,
    write_promotion_log,
)


def _candidate(ppl: float, diversity: float) -> RoundMetrics:
    return RoundMetrics(round_id=2, val_ppl=ppl, diversity_ratio=diversity)


def _champion(ppl: float, diversity: float) -> RoundMetrics:
    return RoundMetrics(round_id=1, val_ppl=ppl, diversity_ratio=diversity)


DIGEST = "a" * 64


def test_scenario_valid_improvement_promotes() -> None:
    # 2% relative perplexity gain, diversity stable: all gates pass.
    decision = decide_promotion(
        _candidate(9.8, 0.80), _champion(10.0, 0.80),
        min_ppl_gain=0.01, max_diversity_drop=0.20,
        holdout_digest=DIGEST, expected_digest=DIGEST,
    )
    assert decision.promote and not decision.abort


def test_scenario_flat_ppl_keeps_champion() -> None:
    # 0.2% gain is below the 1% threshold.
    decision = decide_promotion(
        _candidate(9.98, 0.80), _champion(10.0, 0.80),
        min_ppl_gain=0.01, max_diversity_drop=0.20,
        holdout_digest=DIGEST, expected_digest=DIGEST,
    )
    assert not decision.promote and not decision.abort
    assert "below threshold" in decision.cause


def test_scenario_diversity_collapse_keeps_champion() -> None:
    # PPL improves but diversity drops 30% (> 20% floor).
    decision = decide_promotion(
        _candidate(9.0, 0.56), _champion(10.0, 0.80),
        min_ppl_gain=0.01, max_diversity_drop=0.20,
        holdout_digest=DIGEST, expected_digest=DIGEST,
    )
    assert not decision.promote and not decision.abort
    assert "diversity" in decision.cause


def test_scenario_hash_mismatch_aborts_before_evaluating() -> None:
    decision = decide_promotion(
        _candidate(9.0, 0.80), _champion(10.0, 0.80),
        min_ppl_gain=0.01, max_diversity_drop=0.20,
        holdout_digest="b" * 64, expected_digest=DIGEST,
    )
    assert decision.abort and not decision.promote
    assert "mismatch" in decision.cause


def test_first_measured_round_promotes() -> None:
    decision = decide_promotion(
        _candidate(10.5, 0.75), None,
        min_ppl_gain=0.01, max_diversity_drop=0.20,
        holdout_digest=DIGEST, expected_digest=DIGEST,
    )
    assert decision.promote and not decision.abort


def test_holdout_digest_is_order_independent(tmp_path) -> None:
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "aaa.txt").write_text("alpha", encoding="utf-8")
    (holdout / "bbb.txt").write_text("beta", encoding="utf-8")
    first = compute_holdout_digest(holdout)
    # Rewrite in reverse creation order: content identical, digest identical.
    (holdout / "aaa.txt").unlink()
    (holdout / "aaa.txt").write_text("alpha", encoding="utf-8")
    assert compute_holdout_digest(holdout) == first


def test_holdout_digest_detects_byte_change(tmp_path) -> None:
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "doc.txt").write_bytes(b"line one\n")
    original = compute_holdout_digest(holdout)
    (holdout / "doc.txt").write_bytes(b"line one\r\n")  # autocrlf-style rewrite
    assert compute_holdout_digest(holdout) != original


def test_holdout_digest_empty_dir_raises(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        compute_holdout_digest(empty)


def test_write_promotion_log_refuses_unmeasured_metrics(tmp_path) -> None:
    with pytest.raises(ValueError, match="refusing"):
        write_promotion_log(tmp_path, PromotionDecision(True, False, "x"), RoundMetrics(1, None, None))


def test_write_promotion_log_content(tmp_path) -> None:
    metrics = RoundMetrics(3, 9.1234, 0.7890, champion_val_ppl=10.0, champion_diversity=0.8012)
    path = write_promotion_log(tmp_path, PromotionDecision(True, False, "all gates passed"), metrics)
    text = path.read_text(encoding="utf-8")
    assert "## Round 3" in text
    assert "- decision: promoted" in text
    assert "- candidate val_ppl: 9.1234" in text
    assert "- champion val_ppl: 10.0000" in text
    assert "- champion diversity ratio: 0.8012" in text


def test_rollback_restores_champion_atomically(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "champion.pt").write_bytes(b"champion-bytes")
    (checkpoints / "latest.pt").write_bytes(b"rejected-candidate")
    restored = rollback_to_champion(checkpoints)
    assert restored == checkpoints / "latest.pt"
    assert restored.read_bytes() == b"champion-bytes"


def test_rollback_without_champion_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        rollback_to_champion(tmp_path)


def test_load_round_metrics_roundtrip(tmp_path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    row = {"round_id": 4, "val_ppl": 8.5, "diversity_ratio": 0.77, "champion_val_ppl": 9.0, "champion_diversity": 0.80}
    (logs / "round_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    metrics = load_round_metrics(logs, 4)
    assert metrics.val_ppl == 8.5
    assert metrics.diversity_ratio == 0.77
    assert metrics.champion_val_ppl == 9.0
    with pytest.raises(KeyError):
        load_round_metrics(logs, 99)
