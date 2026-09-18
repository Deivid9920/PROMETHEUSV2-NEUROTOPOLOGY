"""Single-round runner for the self-improvement loop.

Thin CLI wrapper over :func:`prometheus_ns.autoloop.loop.run_round`
for operating one round at a time (cron-style) instead of a batch.
"""

from __future__ import annotations

import argparse
import json

from prometheus_ns import load_config
from prometheus_ns.autoloop.loop import next_round_id, repo_path, run_round


def main() -> None:
    """CLI entry point: ``python scripts/run_round.py --config PATH [--round-id N]``."""
    parser = argparse.ArgumentParser(description="Prometheus-NS single round runner")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--round-id", type=int, default=None, help="explicit round id (default: next)")
    parser.add_argument("--auto-continue", action="store_true", help="skip the human gate explicitly")
    parser.add_argument("--max-tokens", default=None, help="token budget override for this round")
    args = parser.parse_args()
    cfg = load_config(args.config)
    round_id = args.round_id or next_round_id(repo_path(cfg, "logs"))
    result = run_round(
        cfg,
        round_id=round_id,
        auto_continue=args.auto_continue,
        max_tokens=float(args.max_tokens) if args.max_tokens else None,
    )
    print(json.dumps({
        "round_id": result["round_id"],
        "decision": result["decision"].cause,
        "promoted": result["decision"].promote and not result["decision"].abort,
        "candidate_ppl": result["candidate_ppl"],
        "champion_ppl": result["champion_ppl"],
        "diversity_ratio": result["diversity_ratio"],
        "duration_s": result["duration_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
