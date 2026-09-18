"""Held-out perplexity on the frozen holdout, in blocks of 256 docs.

The holdout digest is re-verified before any measurement: an
evaluation whose data cannot be proven intact is worthless. The same
module powers the int8-vs-fp32 comparison used by the quantization
check (dynamic int8 quantizes ``nn.Linear`` weights; small models are
sensitive to it, so the measured delta is reported, never assumed).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.autoloop.promotion import compute_holdout_digest
from prometheus_ns.model.config import ModelConfig
from prometheus_ns.model.model import PAD_ID, TransformerLM
from prometheus_ns.model.tokenizer_train import load_tokenizer

BLOCK_DOCS = 256


def _load_model_from_checkpoint(cfg: dict, checkpoint_path: Path, device: torch.device) -> tuple[TransformerLM, dict]:
    """Instantiate the model stored in a checkpoint and load its weights."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    profile = payload.get("profile")
    if profile:
        mc = ModelConfig(**profile)
    else:
        from prometheus_ns.model.config import model_config_from_yaml

        mc = model_config_from_yaml(cfg)
    model = TransformerLM(mc).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


@torch.no_grad()
def perplexity_of_docs(
    model: TransformerLM,
    tokenizer,
    docs: list[Path],
    max_seq: int,
    device: torch.device,
    block_docs: int = BLOCK_DOCS,
) -> tuple[float, int]:
    """Perplexity over documents, processed in blocks of ``block_docs``.

    Documents are tokenized on the fly, concatenated with the ``eos``
    separator, cut into ``max_seq`` windows and evaluated under the
    causal loss. Returns ``(ppl, evaluated_token_count)``.
    """
    eos_id = int(tokenizer.token_to_id("<|eos|>"))
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(docs), block_docs):
        block = docs[start:start + block_docs]
        ids: list[int] = []
        for path in block:
            ids.extend(tokenizer.encode(path.read_text(encoding="utf-8")).ids)
            ids.append(eos_id)
        if len(ids) < 2:
            continue
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        windows = tokens.split(max_seq)
        for window in windows:
            if window.numel() < 2:
                continue
            x = window.unsqueeze(0)
            logits = model(x)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            shift_targets = x[:, 1:].reshape(-1)
            nll = torch.nn.functional.cross_entropy(shift_logits, shift_targets, ignore_index=PAD_ID, reduction="sum")
            total_nll += nll.item()
            total_tokens += int((shift_targets != PAD_ID).sum().item())
    if total_tokens == 0:
        raise RuntimeError("no tokens evaluated: holdout documents are empty")
    return math.exp(total_nll / total_tokens), total_tokens


def evaluate_checkpoint_on_holdout(cfg: dict, tokenizer, checkpoint_path: Path, int8: bool = False) -> dict:
    """Evaluate one checkpoint on the frozen holdout with integrity check.

    Returns:
        Dictionary with ``ppl``, ``tokens``, ``docs`` and the verified
        holdout digest.
    """
    holdout_dir = repo_path(cfg, "data_clean", "holdout")
    expected = cfg_get(cfg, "loop.holdout_sha256")
    if expected is None:
        raise RuntimeError("holdout is not frozen: run scripts/freeze_holdout.py once before evaluating")
    digest = compute_holdout_digest(holdout_dir)
    if digest != expected:
        raise RuntimeError(f"holdout hash mismatch ({digest} != {expected}): refusing to evaluate")
    docs = sorted(p for p in holdout_dir.glob("**/*.txt") if p.is_file())
    if not docs:
        raise RuntimeError("frozen holdout is empty")

    from prometheus_ns.device import get_device

    device = get_device()
    model, _payload = _load_model_from_checkpoint(cfg, checkpoint_path, device)
    if int8:
        model = quantize_dynamic(model, {nn.Linear})
    profile = cfg_get(cfg, "model.profile", "nano")
    max_seq = int(cfg_get(cfg, f"model.{profile}.max_seq", 256))
    ppl, tokens = perplexity_of_docs(model, tokenizer, docs, max_seq, device)
    return {"ppl": round(ppl, 4), "tokens": tokens, "docs": len(docs), "int8": int8, "holdout_digest": digest}


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.eval.perplexity --config PATH [--checkpoint PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS held-out perplexity")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (default: champion.pt, else latest.pt)")
    parser.add_argument("--int8", action="store_true", help="also evaluate a dynamic int8 copy")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg)
    checkpoints_dir = repo_path(cfg, "artifacts", "checkpoints")
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        checkpoints_dir / "champion.pt" if (checkpoints_dir / "champion.pt").exists() else checkpoints_dir / "latest.pt"
    )
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint found: {checkpoint} (run make train first)")

    result = evaluate_checkpoint_on_holdout(cfg, tokenizer, checkpoint)
    result["checkpoint"] = str(checkpoint)
    if args.int8:
        int8_result = evaluate_checkpoint_on_holdout(cfg, tokenizer, checkpoint, int8=True)
        delta_pct = 100.0 * (int8_result["ppl"] - result["ppl"]) / result["ppl"]
        result["int8_ppl"] = int8_result["ppl"]
        result["int8_degradation_pct"] = round(delta_pct, 3)
        print(f"fp32 ppl: {result['ppl']:.4f} | int8 ppl: {int8_result['ppl']:.4f} | delta: {delta_pct:+.2f}%")

    logs_dir = repo_path(cfg, "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "eval_ppl.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
