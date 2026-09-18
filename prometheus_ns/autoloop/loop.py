"""Self-improvement round orchestration.

Per round: crawl new documents, clean/dedup (against history AND the
frozen holdout), extract triplets, quarantine contradictions, rebuild
the packed dataset without quarantined docs, run continued pretraining
from the champion, evaluate on the frozen holdout and apply the
promotion gate. The round timeout checkpoints partial state instead of
exceeding the CPU budget.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.autoloop.diversity import measure_and_record
from prometheus_ns.autoloop.promotion import (
    RoundMetrics,
    compute_holdout_digest,
    decide_promotion,
    load_round_metrics,
    rollback_to_champion,
    write_promotion_log,
)
from prometheus_ns.data.cleaner import clean_corpus
from prometheus_ns.data.crawler import run_crawl
from prometheus_ns.symbolic.consistency import run_consistency
from prometheus_ns.symbolic.extractor import extract_corpus
from prometheus_ns.train.dataset import build_packed_dataset
from prometheus_ns.train.trainer import promote_checkpoint, train


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_round_id(logs_dir: Path) -> int:
    """One past the highest recorded round id (1 for a fresh repo)."""
    metrics_path = logs_dir / "round_metrics.jsonl"
    last = 0
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    last = max(last, int(json.loads(line).get("round_id", 0)))
                except (json.JSONDecodeError, ValueError):
                    continue
    return last + 1


def quarantined_hashes(cfg: dict) -> set[str]:
    """Prefixes (12-char) of docs excluded from pretraining this round."""
    path = repo_path(cfg, "triplets", "quarantine.jsonl")
    stems: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    stems.add(json.loads(line)["source_doc_hash"][:12])
                except (json.JSONDecodeError, KeyError):
                    continue
    return stems


def verify_holdout(cfg: dict) -> str:
    """Recompute the holdout digest and compare with the config anchor.

    Raises:
        RuntimeError: On any mismatch — the round must not evaluate.
    """
    holdout_dir = repo_path(cfg, "data_clean", "holdout")
    expected = cfg_get(cfg, "loop.holdout_sha256")
    if expected is None:
        raise RuntimeError("loop.holdout_sha256 is not set: run scripts/freeze_holdout.py exactly once before looping")
    actual = compute_holdout_digest(holdout_dir)
    if actual != expected:
        raise RuntimeError(
            f"holdout hash mismatch: config expects {expected}, holdout dir computes {actual}; "
            "aborting without evaluating (integrity gate)"
        )
    return actual


def run_round(
    cfg: dict,
    round_id: int | None = None,
    tokenizer=None,
    auto_continue: bool = False,
    max_tokens: float | None = None,
) -> dict:
    """Execute one full self-improvement round.

    Returns:
        Dictionary with the round decision, metrics and timings.
    """
    from prometheus_ns.eval.perplexity import evaluate_checkpoint_on_holdout

    logs_dir = repo_path(cfg, "logs")
    docs_dir = repo_path(cfg, "docs")
    checkpoints_dir = repo_path(cfg, "artifacts", "checkpoints")
    round_id = round_id or next_round_id(logs_dir)
    started = time.time()

    # 1. Integrity gate BEFORE any work.
    digest = verify_holdout(cfg)
    _append_jsonl(logs_dir / "loop.jsonl", {"round_id": round_id, "phase": "start", "holdout_digest": digest, "ts": time.time()})

    # 2. Incremental crawl (network failures degrade, never abort the round).
    crawl_cfg = copy.deepcopy(cfg)
    new_cap = int(cfg_get(cfg, "loop.new_docs_per_round", 50000))
    crawl_cfg["data"]["max_pages_per_source"] = min(
        int(cfg_get(cfg, "data.max_pages_per_source", 200)), new_cap
    )
    try:
        crawl_results = run_crawl(crawl_cfg)
        crawl_ok = sum(1 for r in crawl_results if r.status == 200)
    except Exception as exc:  # noqa: BLE001 - a dead source must not kill the round
        crawl_ok = 0
        _append_jsonl(logs_dir / "loop.jsonl", {"round_id": round_id, "phase": "crawl_error", "error": f"{type(exc).__name__}: {exc}"})
    _append_jsonl(logs_dir / "loop.jsonl", {"round_id": round_id, "phase": "crawled", "fetched": crawl_ok})

    # 3. Clean, dedup, quality (incremental over raw files; history-aware).
    clean_stats = clean_corpus(cfg)

    # 4. Symbolic layer: triplets + contradiction quarantine.
    triplet_stats = extract_corpus(cfg)
    consistency_stats = run_consistency(cfg)

    # 5. Diversity baseline of the round's training corpus.
    from prometheus_ns.train.dataset import split_docs_for_training

    if tokenizer is None:
        from prometheus_ns.model.tokenizer_train import load_tokenizer

        tokenizer = load_tokenizer(cfg)
    train_docs, _val = split_docs_for_training(cfg, tokenizer)
    quarantine = quarantined_hashes(cfg)
    train_docs = [p for p in train_docs if p.stem not in quarantine]
    sample_dir = docs_dir / "_diversity_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("*.txt"):
        old.unlink()
    for doc in train_docs[:10000]:
        try:
            (sample_dir / doc.name).write_text(doc.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            continue
    diversity_ratio = measure_and_record(sample_dir, round_id, logs_dir)

    # 6. Champion baseline on the frozen holdout (before any training).
    champion_path = checkpoints_dir / "champion.pt"
    champion_ppl = None
    champion_diversity = None
    if champion_path.exists():
        champion_eval = evaluate_checkpoint_on_holdout(cfg, tokenizer, champion_path)
        champion_ppl = champion_eval["ppl"]
        try:
            prev = load_round_metrics(logs_dir, round_id - 1) if round_id > 1 else None
            champion_diversity = prev.diversity_ratio if prev else None
        except (FileNotFoundError, KeyError):
            champion_diversity = None

    # 7. Rebuild the packed dataset (holdout and quarantined docs excluded)
    #    and run continued pretraining from the champion when present.
    build_packed_dataset(cfg, tokenizer)
    timeout_s = float(cfg_get(cfg, "loop.round_timeout_s", 10800))
    remaining_budget = timeout_s - (time.time() - started)
    train_result = train(
        cfg,
        cfg_get(cfg, "model.profile", "nano"),
        tokenizer,
        max_tokens=max_tokens,
        init_from=champion_path if champion_path.exists() else None,
        continue_lr_frac=float(cfg_get(cfg, "train.continue_lr_frac", 0.3)) if champion_path.exists() else None,
        timeout_s=max(remaining_budget, 60.0),
        checkpoints_dir=checkpoints_dir,
        metrics_path=logs_dir / "metrics.jsonl",
    )

    # 8. Candidate evaluation on the frozen holdout.
    candidate_eval = evaluate_checkpoint_on_holdout(cfg, tokenizer, checkpoints_dir / "latest.pt")
    candidate_ppl = candidate_eval["ppl"]

    candidate = RoundMetrics(
        round_id=round_id,
        val_ppl=candidate_ppl,
        diversity_ratio=diversity_ratio,
        champion_val_ppl=champion_ppl,
        champion_diversity=champion_diversity,
    )
    champion_metrics = RoundMetrics(
        round_id=round_id - 1,
        val_ppl=champion_ppl,
        diversity_ratio=champion_diversity,
    ) if champion_ppl is not None else None

    decision = decide_promotion(
        candidate,
        champion_metrics,
        float(cfg_get(cfg, "loop.min_ppl_gain", 0.01)),
        float(cfg_get(cfg, "loop.max_diversity_drop", 0.20)),
        digest,
        cfg_get(cfg, "loop.holdout_sha256"),
    )

    # 9. Checkpoint lifecycle + audit trail.
    if decision.promote and not decision.abort:
        promote_checkpoint(checkpoints_dir)
    elif not decision.promote and champion_path.exists():
        rollback_to_champion(checkpoints_dir)

    write_promotion_log(docs_dir, decision, candidate)
    _append_jsonl(
        logs_dir / "round_metrics.jsonl",
        {
            "round_id": round_id,
            "val_ppl": candidate_ppl,
            "diversity_ratio": diversity_ratio,
            "champion_val_ppl": champion_ppl,
            "champion_diversity": champion_diversity,
            "decision": "aborted" if decision.abort else ("promoted" if decision.promote else "kept champion"),
            "cause": decision.cause,
            "train": {
                "step": train_result["step"],
                "tokens_seen": train_result["tokens_seen"],
                "tokens_s": train_result["tokens_s"],
                "interrupted": train_result["interrupted"],
            },
            "duration_s": round(time.time() - started, 1),
            "ts": time.time(),
        },
    )

    # 10. Human gate every `human_gate_every` rounds: pause for explicit
    #     confirmation unless --auto-continue was passed explicitly.
    gate_every = int(cfg_get(cfg, "loop.human_gate_every", 3))
    if gate_every > 0 and round_id % gate_every == 0 and not auto_continue:
        print(
            f"\nhuman gate: round {round_id} finished ({decision.cause}). "
            "continue with the next round? [y/N]"
        )
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("stopping at the human gate; pass --auto-continue to skip it explicitly")
            results_note = "stopped at human gate"
        else:
            results_note = "gate confirmed"
        _append_jsonl(logs_dir / "loop.jsonl", {"round_id": round_id, "phase": "human_gate", "outcome": results_note})

    print(
        f"round {round_id}: {decision.cause} | candidate ppl {candidate_ppl:.4f} "
        f"(champion {champion_ppl if champion_ppl is not None else 'n/a'}) | diversity {diversity_ratio:.4f}"
    )
    return {
        "round_id": round_id,
        "decision": decision,
        "candidate_ppl": candidate_ppl,
        "champion_ppl": champion_ppl,
        "diversity_ratio": diversity_ratio,
        "clean_stats": clean_stats,
        "triplet_stats": triplet_stats,
        "consistency_stats": consistency_stats,
        "train": train_result,
        "duration_s": round(time.time() - started, 1),
    }


def run_loop(cfg: dict, rounds: int, auto_continue: bool = False, max_tokens: float | None = None) -> list[dict]:
    """Run ``rounds`` self-improvement rounds sequentially."""
    tokenizer = None
    results = []
    for _ in range(rounds):
        round_id = next_round_id(repo_path(cfg, "logs"))
        results.append(run_round(cfg, round_id=round_id, tokenizer=tokenizer, auto_continue=auto_continue, max_tokens=max_tokens))
        tokenizer = None  # reload per round: artifacts may be retrained
    return results


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.autoloop.loop --config PATH --rounds N [--auto-continue]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS self-improvement loop")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--rounds", type=int, required=True, help="number of rounds to run")
    parser.add_argument("--auto-continue", action="store_true", help="skip the human gate explicitly")
    parser.add_argument("--max-tokens", default=None, help="token budget override per round")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = run_loop(cfg, args.rounds, auto_continue=args.auto_continue, max_tokens=float(args.max_tokens) if args.max_tokens else None)
    promoted = sum(1 for r in results if r["decision"].promote and not r["decision"].abort)
    print(f"loop finished: {len(results)} rounds, {promoted} promotions")


if __name__ == "__main__":
    main()
