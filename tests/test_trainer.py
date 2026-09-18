"""Tests for the trainer: exact resume contract and checkpoint lifecycle.

The resume contract is the hardest guarantee in the project: an
interrupted run continued from latest.pt must reproduce the
uninterrupted run's loss to 1e-6.
"""

from __future__ import annotations

import torch

from prometheus_ns.train.trainer import promote_checkpoint, train


def _run(cfg, tokenizer, checkpoints, metrics, max_tokens: float, resume=None, interrupt_at=None):
    return train(
        cfg,
        "tiny",
        tokenizer,
        max_tokens=max_tokens,
        resume_path=resume,
        checkpoints_dir=checkpoints,
        metrics_path=metrics,
        val_windows=2,
        interrupt_at_step=interrupt_at,
    )


def test_resume_reproduces_continuous_run_to_1e6(tiny_repo, tmp_path) -> None:
    cfg = tiny_repo["cfg"]
    tokenizer = tiny_repo["tokenizer"]
    checkpoints = tmp_path / "continuous"
    metrics = tmp_path / "metrics_continuous.jsonl"

    continuous = _run(cfg, tokenizer, checkpoints, metrics, max_tokens=6 * 128)

    split_checkpoints = tmp_path / "split"
    split_metrics = tmp_path / "metrics_split.jsonl"
    # Segment 1: same planned budget, but a simulated crash at step 2.
    first = _run(cfg, tokenizer, split_checkpoints, split_metrics, max_tokens=6 * 128, interrupt_at=2)
    assert first["step"] == 2 and first["interrupted"] is True
    # Segment 2: resume from latest.pt; the checkpoint restores the run
    # parameters (total steps, base lr, RNG, cursor) exactly.
    resumed = _run(
        cfg, tokenizer, split_checkpoints, split_metrics,
        max_tokens=6 * 128, resume=split_checkpoints / "latest.pt",
    )
    assert resumed["step"] == 6
    assert resumed["interrupted"] is False

    assert continuous["history"] == resumed["history"], (
        "resumed trajectory diverged from the continuous run: "
        f"{continuous['history']} != {resumed['history']}"
    )
    for (step_a, loss_a), (step_b, loss_b) in zip(continuous["history"], resumed["history"]):
        assert step_a == step_b
        assert abs(loss_a - loss_b) <= 1e-6


def test_val_loss_recorded_and_finite(tiny_repo, tmp_path) -> None:
    cfg = tiny_repo["cfg"]
    result = _run(cfg, tiny_repo["tokenizer"], tmp_path / "ck", tmp_path / "m.jsonl", max_tokens=4 * 128)
    assert result["val_history"], "at least one evaluation must be logged"
    for _step, val_loss, ppl in result["val_history"]:
        assert val_loss > 0 and 1.0 < ppl < float("inf")
    assert result["final_val_ppl"] >= 1.0


def test_checkpoint_is_atomic_and_resumable_payload(tiny_repo, tmp_path) -> None:
    cfg = tiny_repo["cfg"]
    checkpoints = tmp_path / "ck"
    _run(cfg, tiny_repo["tokenizer"], checkpoints, tmp_path / "m.jsonl", max_tokens=2 * 128)
    latest = checkpoints / "latest.pt"
    assert latest.exists()
    assert not latest.with_suffix(".pt.tmp").exists()  # temp file replaced
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    for key in ("model_state", "optim_state", "step", "tokens_seen", "rng", "total_steps", "base_lr"):
        assert key in payload, f"checkpoint missing {key}"
    assert "torch" in payload["rng"] and "numpy" in payload["rng"] and "python" in payload["rng"]


def test_metrics_log_schema(tiny_repo, tmp_path) -> None:
    import json

    cfg = tiny_repo["cfg"]
    metrics_path = tmp_path / "m.jsonl"
    _run(cfg, tiny_repo["tokenizer"], tmp_path / "ck", metrics_path, max_tokens=2 * 128)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert rows[0]["phase"] == "start"
    assert rows[-1]["phase"] == "end"
    eval_rows = [r for r in rows if r["phase"] == "eval"]
    assert eval_rows
    for row in eval_rows:
        assert {"step", "loss", "val_loss", "ppl", "rss_mb"} <= set(row)


def test_promote_checkpoint_copies_latest(tiny_repo, tmp_path) -> None:
    cfg = tiny_repo["cfg"]
    checkpoints = tmp_path / "ck"
    _run(cfg, tiny_repo["tokenizer"], checkpoints, tmp_path / "m.jsonl", max_tokens=2 * 128)
    champion = promote_checkpoint(checkpoints)
    assert champion == checkpoints / "champion.pt"
    assert champion.read_bytes() == (checkpoints / "latest.pt").read_bytes()
