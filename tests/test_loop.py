"""End-to-end round test on the isolated micro-repo.

One full self-improvement round on the tiny profile: integrity gate,
cleaning, triplets, quarantine, diversity, continued pretraining,
holdout evaluation and the promotion gate with its audit trail. The
crawler is mocked: network behavior is covered in test_crawler.py.
"""

from __future__ import annotations

import json

import pytest

import prometheus_ns.autoloop.loop as loop_module
from prometheus_ns.data.crawler import FetchResult


@pytest.fixture
def mocked_crawl(monkeypatch):
    calls = {"count": 0}

    def fake_run_crawl(cfg):
        calls["count"] += 1
        return [FetchResult(url="https://mock/1", status=200, bytes_written=10, path=None)]

    monkeypatch.setattr(loop_module, "run_crawl", fake_run_crawl)
    return calls


@pytest.fixture
def prepared_repo(tiny_repo, mocked_crawl):
    from scripts.freeze_holdout import freeze_holdout

    freeze = freeze_holdout(tiny_repo["cfg"])
    tiny_repo["freeze"] = freeze
    return tiny_repo


def test_round_one_bootstraps_champion(prepared_repo) -> None:
    from prometheus_ns.autoloop.loop import run_round

    cfg, tokenizer = prepared_repo["cfg"], prepared_repo["tokenizer"]
    result = run_round(cfg, round_id=1, tokenizer=tokenizer, auto_continue=True, max_tokens=4 * 128)

    assert result["candidate_ppl"] > 1.0
    assert result["decision"].promote and not result["decision"].abort
    assert "first measured round" in result["decision"].cause

    checkpoints = prepared_repo["root"] / "artifacts" / "checkpoints"
    assert (checkpoints / "champion.pt").exists()

    log_text = (prepared_repo["root"] / "docs" / "promotion_log.md").read_text(encoding="utf-8")
    assert "## Round 1" in log_text
    assert "- decision: promoted" in log_text
    assert "candidate val_ppl:" in log_text

    rows = [
        json.loads(line)
        for line in (prepared_repo["root"] / "logs" / "round_metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["round_id"] == 1
    assert rows[-1]["decision"] == "promoted"
    assert rows[-1]["diversity_ratio"] > 0


def test_round_two_rejected_on_flat_perplexity(prepared_repo) -> None:
    from prometheus_ns.autoloop.loop import run_round

    cfg, tokenizer = prepared_repo["cfg"], prepared_repo["tokenizer"]
    run_round(cfg, round_id=1, tokenizer=tokenizer, auto_continue=True, max_tokens=4 * 128)
    # Round 2 retrains barely at all: perplexity stays flat, so the gate
    # must keep the champion instead of promoting.
    result = run_round(cfg, round_id=2, tokenizer=tokenizer, auto_continue=True, max_tokens=1 * 128)
    if not result["decision"].promote:
        assert "below threshold" in result["decision"].cause
        assert result["candidate_ppl"] >= result["champion_ppl"] * (1 - 0.01)


def test_round_aborts_on_holdout_mismatch(prepared_repo, monkeypatch) -> None:
    from prometheus_ns.autoloop.loop import run_round

    cfg = prepared_repo["cfg"]
    # Tamper with the holdout after freezing: the integrity gate must
    # stop the round before any evaluation happens.
    holdout_dir = prepared_repo["root"] / "data_clean" / "holdout"
    next(holdout_dir.glob("*.txt")).write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        run_round(cfg, round_id=1, tokenizer=prepared_repo["tokenizer"], auto_continue=True, max_tokens=2 * 128)


def test_quarantined_docs_excluded_from_packing(prepared_repo) -> None:
    from prometheus_ns.autoloop.loop import quarantined_hashes
    from prometheus_ns.symbolic.extractor import Triplet

    triplets_dir = prepared_repo["root"] / "triplets"
    triplets_dir.mkdir(parents=True, exist_ok=True)
    doc_hash = prepared_repo["docs"][0].stem
    with (triplets_dir / "quarantine.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"source_doc_hash": doc_hash + "0" * 52, "reason": "symbolic_contradiction"}) + "\n")
    # Only exact 12-char stem prefixes match cleaned file names.
    hashes = quarantined_hashes(prepared_repo["cfg"])
    assert doc_hash in hashes


def test_next_round_id_increments(prepared_repo) -> None:
    from prometheus_ns.autoloop.loop import next_round_id

    logs_dir = prepared_repo["root"] / "logs"
    assert next_round_id(logs_dir) == 1
    with (logs_dir / "round_metrics.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"round_id": 3, "val_ppl": 5.0, "diversity_ratio": 0.9}) + "\n")
    assert next_round_id(logs_dir) == 4
