"""Parameter count report per component and total.

The total is verified against the contracted ranges: nano in
[12M, 18M], small in [85M, 115M], large in [260M, 310M]. The large
profile is counted analytically by default so the check does not need
to allocate ~270M parameters on constrained machines; ``--materialize``
forces the real sum.
"""

from __future__ import annotations

import argparse
import json
import sys

from prometheus_ns import cfg_get, load_config
from prometheus_ns.model.config import ModelConfig, analytic_param_count, model_config_from_yaml

RANGES = {
    "nano": (12_000_000, 18_000_000),
    "small": (85_000_000, 115_000_000),
    "large": (260_000_000, 310_000_000),
}


def component_table(mc: ModelConfig) -> dict[str, int]:
    """Parameter count broken down by model component (analytic)."""
    d = mc.d_model
    head_dim = mc.head_dim
    return {
        "tok_emb" + (" (tied with lm_head)" if mc.tie_embeddings else ""): mc.vocab * d,
        "lm_head" + (" (tied, counted once)" if mc.tie_embeddings else ""): 0 if mc.tie_embeddings else mc.vocab * d,
        "attention.q": mc.n_layer * (d * d),
        "attention.k+v": mc.n_layer * (2 * d * mc.n_kv_head * head_dim),
        "attention.out": mc.n_layer * (d * d),
        "swiglu.gate+up+down": mc.n_layer * (3 * d * mc.d_ff),
        "norms": mc.n_layer * 2 * d + d,
    }


def count_profile(cfg: dict, profile: str, materialize: bool = False) -> dict:
    """Return the parameter report for one profile."""
    mc = model_config_from_yaml(cfg, profile)
    total = analytic_param_count(mc)
    if materialize:
        from prometheus_ns.model.model import TransformerLM

        total = TransformerLM(mc).num_parameters()
    low, high = RANGES.get(profile, (0, sys.maxsize))
    return {
        "profile": profile,
        "total": total,
        "contracted_range": [low, high],
        "in_range": low <= total <= high,
        "components": component_table(mc),
    }


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.model.count_params --config PATH [--profile NAME]``."""
    parser = argparse.ArgumentParser(description="Prometheus-NS parameter count")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--profile", default=None, help="profile override (nano|small|large)")
    parser.add_argument("--materialize", action="store_true", help="instantiate the model for an exact sum")
    args = parser.parse_args()
    cfg = load_config(args.config)
    profile = args.profile or cfg_get(cfg, "model.profile", "nano")
    report = count_profile(cfg, profile, materialize=args.materialize)
    print(json.dumps({"profile": report["profile"], "parameters": report["total"]}, indent=2))
    print(f"{'component':<34}{'params':>14}")
    for name, value in report["components"].items():
        print(f"{name:<34}{value:>14,}")
    low, high = report["contracted_range"]
    status = "OK" if report["in_range"] else "OUT OF RANGE"
    print(f"\ntotal: {report['total']:,} (contract [{low:,}, {high:,}] -> {status})")


if __name__ == "__main__":
    main()
