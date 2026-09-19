"""Interactive chat REPL over the champion checkpoint.

Default path quantizes ``nn.Linear`` layers with dynamic int8, which
runs everywhere on CPU without extra dependencies; ``--fp32`` keeps
full precision for comparison. Generation uses the ``inference``
block of config.yaml: temperature, top-k, top-p and repetition
penalty, with a strict stop at ``<|eos|>``. The model is small; incoherent
generations are reported as they come, never polished.
"""

from __future__ import annotations

import argparse

import torch
from torch.ao.quantization import quantize_dynamic
from torch import nn

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.device import get_device
from prometheus_ns.eval.prompt_suite import generate_continuation
from prometheus_ns.eval.perplexity import _load_model_from_checkpoint
from prometheus_ns.model.tokenizer_train import load_tokenizer


def load_chat_model(cfg: dict, fp32: bool) -> tuple[torch.nn.Module, object, dict]:
    """Load the champion (or latest) checkpoint for chatting.

    Returns:
        ``(model, tokenizer, info)`` where info describes device and
        quantization for the session banner.
    """
    checkpoints_dir = repo_path(cfg, "artifacts", "checkpoints")
    checkpoint = checkpoints_dir / "champion.pt"
    if not checkpoint.exists():
        checkpoint = checkpoints_dir / "latest.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint found in {checkpoints_dir}: run make train first")
    device = get_device()
    model, _payload, _mc = _load_model_from_checkpoint(cfg, checkpoint, device)
    quantized = False
    if not fp32 and device.type == "cpu":
        model = quantize_dynamic(model, {nn.Linear})
        quantized = True
    if not fp32 and device.type == "cuda":
        model = model.to(torch.bfloat16)
    info = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "quantization": "fp32" if fp32 else ("dynamic int8" if quantized else "bf16"),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, load_tokenizer(cfg), info


def repl(cfg: dict, fp32: bool = False) -> None:
    """Run the interactive REPL; Ctrl-C or Ctrl-D exits."""
    model, tokenizer, info = load_chat_model(cfg, fp32)
    params = dict(cfg_get(cfg, "inference", {}) or {})
    print("Prometheus-NS chat")
    print(f"  checkpoint : {info['checkpoint']}")
    print(f"  device     : {info['device']} ({info['quantization']})")
    print(f"  parameters : {info['parameters']:,}")
    print("  the model is tiny; expect limited coherence. Empty line exits.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        output = generate_continuation(model, tokenizer, user, params, get_device())
        print(f"model> {output}\n")


def main() -> None:
    """CLI entry point: ``python scripts/chat.py --config PATH (--int8 | --fp32)``."""
    parser = argparse.ArgumentParser(description="Prometheus-NS chat REPL")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--int8", action="store_true", help="dynamic int8 quantization (default on CPU)")
    quant.add_argument("--fp32", action="store_true", help="full precision reference")
    args = parser.parse_args()
    cfg = load_config(args.config)
    repl(cfg, fp32=args.fp32)


if __name__ == "__main__":
    main()
