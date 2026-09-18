"""Domain prompt suite runner.

Consumes ``eval/prompts.yaml`` (schema_version 1): the ``defaults``
block supplies generation parameters, per-prompt entries may override
any default, and outputs are written to the schema's ``output_path``.
Prompts flagged ``diagnostic: true`` are reported exactly as observed —
failures stay failures, no post-hoc corrections. Prompts with
``apply_symbolic_check`` (default true) get their output validated
against the symbolic consistency graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from prometheus_ns import cfg_get, ensure_dir, load_config, repo_path
from prometheus_ns.eval.perplexity import _load_model_from_checkpoint  # shared loader
from prometheus_ns.model.model import EOS_ID
from prometheus_ns.symbolic.generator import check_sentences

PROMPTS_PATH = "eval/prompts.yaml"
REQUIRED_FIELDS = ("id", "category", "prompt", "review_note")


def load_prompt_schema(cfg: dict) -> dict:
    """Load and validate the prompt suite schema."""
    path = repo_path(cfg, PROMPTS_PATH)
    with path.open("r", encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)
    if schema.get("schema_version") != 1:
        raise ValueError(f"{PROMPTS_PATH}: unsupported schema_version {schema.get('schema_version')}")
    if "defaults" not in schema or "prompts" not in schema:
        raise ValueError(f"{PROMPTS_PATH}: 'defaults' and 'prompts' sections are required")
    ids = [p.get("id") for p in schema["prompts"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{PROMPTS_PATH}: duplicate prompt ids")
    for prompt in schema["prompts"]:
        missing = [f for f in REQUIRED_FIELDS if f not in prompt]
        if missing:
            raise ValueError(f"{PROMPTS_PATH}: prompt {prompt.get('id')} missing fields {missing}")
    return schema


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    params: dict,
    device: torch.device,
) -> str:
    """Sample one continuation with repetition penalty, top-k and top-p.

    Greedy decoding collapses small models into repetition loops and
    raw sampling at high temperature produces word salad; the defaults
    in prompts.yaml sit in between. Generation stops at ``<|eos|>``.
    """
    max_new = int(params.get("max_new_tokens", 60))
    temperature = float(params.get("temperature", 0.8))
    top_k = int(params.get("top_k", 40))
    top_p = float(params.get("top_p", 0.9))
    rep_penalty = float(params.get("repetition_penalty", 1.15))

    bos_id = int(tokenizer.token_to_id("<|bos|>"))
    eos_id = int(tokenizer.token_to_id("<|eos|>"))
    ids = [bos_id] + tokenizer.encode(prompt).ids
    max_seq = model.cfg.max_seq
    generated: list[int] = []
    input_ids = torch.tensor([ids[-max_seq:]], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(input_ids[:, -max_seq:])
        next_logits = logits[0, -1, :].float().clone()
        for token_id in set(generated + ids):
            if next_logits[token_id] > 0:
                next_logits[token_id] /= rep_penalty
            else:
                next_logits[token_id] *= rep_penalty
        next_logits /= max(temperature, 1e-5)
        if top_k > 0:
            kth = torch.topk(next_logits, min(top_k, next_logits.shape[-1])).values[-1]
            next_logits[next_logits < kth] = float("-inf")
        probs = torch.softmax(next_logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[1:] = cutoff[:-1].clone()
            cutoff[0] = False
            probs[sorted_idx[cutoff]] = 0.0
            probs = probs / probs.sum()
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        if next_token == eos_id:
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(generated)


def run_suite(cfg: dict, checkpoint_path: Path | None = None) -> list[dict]:
    """Run every prompt and write the outputs to the schema's output_path."""
    from prometheus_ns.device import get_device

    schema = load_prompt_schema(cfg)
    tokenizer_module = _load_tokenizer_module()
    tokenizer = tokenizer_module.load_tokenizer(cfg)
    checkpoints_dir = repo_path(cfg, "artifacts", "checkpoints")
    checkpoint_path = checkpoint_path or (
        checkpoints_dir / "champion.pt" if (checkpoints_dir / "champion.pt").exists() else checkpoints_dir / "latest.pt"
    )
    if not checkpoint_path.exists():
        raise RuntimeError(f"no checkpoint found: {checkpoint_path} (run make train first)")
    device = get_device()
    model, _ = _load_model_from_checkpoint(cfg, checkpoint_path, device)

    defaults = dict(schema["defaults"])
    records: list[dict] = []
    for entry in schema["prompts"]:
        params = {**defaults, **{k: v for k, v in entry.items() if k in defaults}}
        output = generate_continuation(model, tokenizer, entry["prompt"], params, device)
        record = {
            "id": entry["id"],
            "category": entry["category"],
            "prompt": entry["prompt"],
            "output": output,
            "params": params,
            "diagnostic": bool(entry.get("diagnostic", False)),
            "review_note": entry.get("review_note", ""),
        }
        if entry.get("apply_symbolic_check", True) and output.strip():
            verdicts = check_sentences([output], cfg)
            record["symbolic_status"] = verdicts[0]["status"] if verdicts else "empty"
        else:
            record["symbolic_status"] = "not-applied"
        records.append(record)

    output_path = repo_path(
        cfg,
        schema.get("output_path") or defaults.get("output_path") or "eval/prompt_outputs.jsonl",
    )
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"prompt suite written: {output_path} ({len(records)} prompts)")
    return records


def _load_tokenizer_module():
    from prometheus_ns.model import tokenizer_train

    return tokenizer_train


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.eval.prompt_suite --config PATH``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS domain prompt suite")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for record in run_suite(cfg):
        print(f"[{record['id']}] {record['prompt']} -> {record['output'][:80]!r}")


if __name__ == "__main__":
    main()
