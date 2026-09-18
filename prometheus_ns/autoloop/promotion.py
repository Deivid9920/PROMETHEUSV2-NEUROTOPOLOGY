"""Promotion gate for the self-improvement loop.

The gate is deliberately immutable: every round must satisfy all three
rules before a candidate checkpoint replaces the champion:

(1) validation perplexity on the FROZEN holdout improves by at least
    ``min_ppl_gain`` relative to the champion;
(2) the corpus diversity ratio does not drop by more than
    ``max_diversity_drop`` versus the previous round;
(3) the holdout SHA-256 matches the digest recorded in config.yaml —
    a mismatch aborts the round before any comparison is produced.

Every number written to the promotion log comes from measured
arguments; entries are refused when a required metric is None so the
log can never contain fabricated values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the promotion gate for one round."""

    promote: bool
    abort: bool
    cause: str


@dataclass(frozen=True)
class RoundMetrics:
    """Measured metrics of one self-improvement round."""

    round_id: int
    val_ppl: float | None
    diversity_ratio: float | None
    champion_val_ppl: float | None = None
    champion_diversity: float | None = None


def compute_holdout_digest(holdout_dir: Path) -> str:
    """SHA-256 accumulator over the holdout directory contents.

    File iteration order is forced to be deterministic (sorted by
    relative posix path); the digest covers each relative path, a NUL
    separator and the raw bytes, so any byte-level rewrite (e.g. a git
    autocrlf conversion) invalidates the anchor instead of passing.
    """
    if not holdout_dir.exists():
        raise ValueError(f"holdout directory does not exist: {holdout_dir}")
    files = sorted(p for p in holdout_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"holdout directory is empty: {holdout_dir}")
    digest = sha256()
    for path in files:
        rel = path.relative_to(holdout_dir).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def decide_promotion(
    candidate: RoundMetrics,
    champion: RoundMetrics | None,
    min_ppl_gain: float,
    max_diversity_drop: float,
    holdout_digest: str | None,
    expected_digest: str | None,
) -> PromotionDecision:
    """Apply the three promotion rules in order.

    The integrity gate runs first: on a holdout hash mismatch the round
    aborts and no comparison is produced. The first measured round (no
    champion baseline) promotes by definition.
    """
    if expected_digest is not None and holdout_digest != expected_digest:
        return PromotionDecision(False, True, "holdout hash mismatch: integrity gate failed")
    if candidate.val_ppl is None:
        return PromotionDecision(False, False, "candidate perplexity not measured")
    if champion is None or champion.val_ppl is None:
        return PromotionDecision(True, False, "first measured round: candidate becomes champion")
    relative_gain = (champion.val_ppl - candidate.val_ppl) / champion.val_ppl
    if relative_gain < min_ppl_gain:
        return PromotionDecision(False, False, f"relative ppl gain {relative_gain:.4f} below threshold {min_ppl_gain}")
    if (
        champion.diversity_ratio is not None
        and candidate.diversity_ratio is not None
        and champion.diversity_ratio > 0
    ):
        drop = (champion.diversity_ratio - candidate.diversity_ratio) / champion.diversity_ratio
        if drop > max_diversity_drop:
            return PromotionDecision(False, False, f"diversity drop {drop:.4f} exceeds threshold {max_diversity_drop}")
    return PromotionDecision(True, False, "all promotion gates passed")


def write_promotion_log(
    docs_dir: Path,
    decision: PromotionDecision,
    metrics: RoundMetrics,
) -> Path:
    """Append one markdown entry to docs/promotion_log.md.

    Every number written comes from the arguments; entries are refused
    when any required metric is None so the log can never contain
    fabricated values.

    Returns:
        Path to the promotion log.
    """
    if metrics.val_ppl is None or metrics.diversity_ratio is None:
        raise ValueError("refusing to log a promotion without measured metrics")
    docs_dir.mkdir(parents=True, exist_ok=True)
    log_path = docs_dir / "promotion_log.md"
    outcome = "aborted" if decision.abort else (
        "promoted" if decision.promote else "kept champion"
    )
    lines = [
        f"## Round {metrics.round_id}",
        "",
        f"- decision: {outcome}",
        f"- cause: {decision.cause}",
        f"- candidate val_ppl: {metrics.val_ppl:.4f}",
        f"- candidate diversity ratio: {metrics.diversity_ratio:.4f}",
    ]
    if metrics.champion_val_ppl is not None:
        lines.append(f"- champion val_ppl: {metrics.champion_val_ppl:.4f}")
    if metrics.champion_diversity is not None:
        lines.append(f"- champion diversity ratio: {metrics.champion_diversity:.4f}")
    lines.append("")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return log_path


def rollback_to_champion(checkpoints_dir: Path) -> Path:
    """Restore champion.pt as the active checkpoint after a rejection.

    Returns the path of the restored champion. The checkpoint layout
    is defined by the training layer; the restore must be atomic
    (write to a temp file, then os.replace).
    """
    import os

    champion = checkpoints_dir / "champion.pt"
    if not champion.exists():
        raise FileNotFoundError(f"no champion checkpoint to restore: {champion}")
    latest = checkpoints_dir / "latest.pt"
    tmp = latest.with_suffix(".pt.tmp")
    tmp.write_bytes(champion.read_bytes())
    os.replace(tmp, latest)
    return latest


def load_round_metrics(logs_dir: Path, round_id: int) -> RoundMetrics:
    """Read one round's metrics from logs/ and build RoundMetrics.

    The concrete log schema is owned by autoloop/loop.py. Validation
    perplexity must always come from the frozen holdout, never from a
    re-split.
    """
    metrics_path = logs_dir / "round_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"round metrics log missing: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row: dict[str, Any] = json.loads(line)
            if int(row.get("round_id", -1)) == round_id:
                return RoundMetrics(
                    round_id=round_id,
                    val_ppl=row.get("val_ppl"),
                    diversity_ratio=row.get("diversity_ratio"),
                    champion_val_ppl=row.get("champion_val_ppl"),
                    champion_diversity=row.get("champion_diversity"),
                )
    raise KeyError(f"round {round_id} not found in {metrics_path}")
