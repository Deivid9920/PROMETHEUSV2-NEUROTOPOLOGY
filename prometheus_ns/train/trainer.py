"""CPU-first trainer with atomic checkpointing and exact resume.

Determinism contract: an interrupted run resumed from ``latest.pt``
reproduces the uninterrupted run's loss to 1e-6. The checkpoint
therefore serializes, besides model and optimizer state:

- torch/numpy/python RNG states;
- the optimizer step and token cursor (the packed-window stream is
  read sequentially, so the cursor fully determines future batches);
- the schedule parameters (total steps and peak lr) used by the run.

Checkpoint writes are atomic (``.tmp`` file plus ``os.replace``); the
Windows file-lock caveat is documented in docs/gpu_migration.md and
the project targets POSIX/WSL2.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from prometheus_ns import cfg_get, ensure_dir, load_config, repo_path
from prometheus_ns.device import autocast_ctx, get_device, setup_compute
from prometheus_ns.model.config import ModelConfig, model_config_from_yaml
from prometheus_ns.model.model import EOS_ID, PAD_ID, TransformerLM
from prometheus_ns.train.dataset import PackedDataset, build_packed_dataset
from prometheus_ns.train.schedules import WarmupCosine


def _atomic_torch_save(payload: dict, target: Path) -> None:
    """Serialize ``payload`` to ``target`` through a temp file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _window_batch(dataset: PackedDataset, index: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``batch`` packed windows starting at window ``index``.

    The stream is finite; when the cursor reaches the end it wraps
    around deterministically so both a continuous run and a resumed run
    visit identical windows at identical steps.
    """
    total = len(dataset)
    rows = [(index + i) % total for i in range(batch)]
    tokens = dataset.tokens[rows]
    flags = dataset.flags[rows]
    return torch.from_numpy(np.asarray(tokens, dtype=np.int64)), torch.from_numpy(np.asarray(flags, dtype=np.float32))


@torch.no_grad()
def evaluate(model: TransformerLM, dataset: PackedDataset, device: torch.device, max_windows: int, batch: int) -> float:
    """Average next-token loss over the first ``max_windows`` val windows."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, min(max_windows, len(dataset)), batch):
        tokens, flags = _window_batch(dataset, start, batch)
        targets = tokens.clone()
        _, loss = model(tokens.to(device), doc_flags=flags.to(device), targets=targets.to(device))
        n = int((targets[:, 1:] != PAD_ID).sum().item())
        if n == 0:
            continue
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def train(
    cfg: dict,
    profile: str,
    tokenizer,
    max_tokens: float | None = None,
    resume_path: str | Path | None = None,
    init_from: str | Path | None = None,
    continue_lr_frac: float | None = None,
    timeout_s: float | None = None,
    checkpoints_dir: Path | None = None,
    metrics_path: Path | None = None,
    val_windows: int = 8,
    interrupt_at_step: int | None = None,
) -> dict:
    """Run one pretraining pass and return the run summary.

    Args:
        cfg: Parsed project configuration.
        profile: Model profile name (nano|small|large).
        tokenizer: Trained ``tokenizers.Tokenizer`` for packing.
        max_tokens: Token budget override; defaults to the
            ``tokens_per_round_{cpu,gpu}`` config value for the device.
        resume_path: ``latest.pt`` to resume exactly (state and RNG).
        init_from: Checkpoint whose weights seed a fresh optimizer
            (continued pretraining round of the autoloop).
        continue_lr_frac: Peak-lr fraction override for continued runs.
        timeout_s: Wall-clock budget; on expiry the run checkpoints and
            returns partial state instead of exceeding the budget.
        checkpoints_dir: Where ``latest.pt`` is written.
        metrics_path: JSONL sink for ``{step, loss, val_loss, ppl,
            tokens_s, rss_mb}`` rows.
        val_windows: Validation windows evaluated per eval point.
        interrupt_at_step: Simulated crash point for the resume
            contract test; the run checkpoints and returns early when
            the step counter reaches it.

    Returns:
        Dictionary with ``history`` (per-step losses), ``val_history``
        (per-eval losses/ppl), ``tokens_seen``, ``step`` and
        ``interrupted`` flags.
    """
    summary = setup_compute(cfg)
    seed = int(cfg_get(cfg, "seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = get_device()
    mc: ModelConfig = model_config_from_yaml(cfg, profile)
    model = TransformerLM(mc).to(device)

    warmup_frac = float(cfg_get(cfg, "train.warmup_frac", 0.02))
    min_lr_frac = float(cfg_get(cfg, "train.min_lr_frac", 0.10))
    weight_decay = float(cfg_get(cfg, "train.weight_decay", 0.1))
    grad_clip = float(cfg_get(cfg, "train.grad_clip", 1.0))
    eval_every = int(cfg_get(cfg, "train.eval_every_steps", 500))
    checkpoint_every = int(cfg_get(cfg, "train.checkpoint_every_steps", 2000))
    micro_batch = int(cfg_get(cfg, f"train.micro_batch.{device.type}", 16))
    grad_accum = int(cfg_get(cfg, f"train.grad_accum.{device.type}", 4))
    use_bf16 = bool(cfg_get(cfg, "train.use_bf16_gpu", True)) and device.type == "cuda"

    if device.type == "cpu":
        base_lr = float(cfg_get(cfg, "train.lr_nano" if profile == "nano" else "train.lr_small", 3.0e-4))
    else:
        base_lr = float(cfg_get(cfg, "train.lr_nano" if profile == "nano" else "train.lr_small", 2.5e-4))
    if continue_lr_frac is None:
        continue_lr_frac = float(cfg_get(cfg, "train.continue_lr_frac", 0.3))

    if max_tokens is None:
        per_device_key = "train.tokens_per_round_gpu" if device.type == "cuda" else "train.tokens_per_round_cpu"
        max_tokens = float(cfg_get(cfg, per_device_key, 5.0e7))
    max_tokens = float(max_tokens)

    if checkpoints_dir is None:
        checkpoints_dir = repo_path(cfg, "artifacts", "checkpoints")
    ensure_dir(checkpoints_dir)
    if metrics_path is None:
        metrics_path = repo_path(cfg, "logs", "metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Packed data: build once, reuse across rounds.
    packed_dir = repo_path(cfg, "data_clean", "packed")
    train_prefix, val_prefix = packed_dir / "train", packed_dir / "val"
    if not (train_prefix.parent / "train.bin.npy").exists() or not (val_prefix.parent / "val.bin.npy").exists():
        build_packed_dataset(cfg, tokenizer)
    train_set = PackedDataset(train_prefix)
    val_set = PackedDataset(val_prefix)

    tokens_per_step = micro_batch * grad_accum * mc.max_seq
    total_steps = max(1, int(math.ceil(max_tokens / tokens_per_step)))

    step = 0
    tokens_seen = 0
    history: list[tuple[int, float]] = []
    val_history: list[tuple[int, float, float]] = []

    optimizer = torch.optim.AdamW(
        _param_groups(model, weight_decay),
        lr=base_lr * (continue_lr_frac if init_from else 1.0),
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def log_row(row: dict) -> None:
        row.update({"rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1)})
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def evaluate_and_log(loss_val: float) -> float:
        val_loss = evaluate(model, val_set, device, val_windows, micro_batch)
        ppl = math.exp(min(val_loss, 20.0))
        val_history.append((step, val_loss, ppl))
        log_row({"step": step, "loss": round(loss_val, 6), "val_loss": round(val_loss, 6), "ppl": round(ppl, 4), "tokens_s": tokens_per_second(), "phase": "eval"})
        return val_loss

    started = time.time()
    tokens_window_start = started

    def tokens_per_second() -> float:
        elapsed = max(time.time() - tokens_window_start, 1e-6)
        return round(tokens_seen / elapsed, 1) if tokens_seen else 0.0

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        # The schedule is rebuilt from the stored run parameters so the
        # resumed trajectory is bit-identical to the interrupted one.
        total_steps = ckpt["total_steps"]
        base_lr = ckpt["base_lr"]
        for group, target in zip(optimizer.param_groups, ckpt["param_lr"], strict=True):
            group["lr"] = target
        step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        history = [tuple(x) for x in ckpt.get("history", [])]
        val_history = [tuple(x) for x in ckpt.get("val_history", [])]
        print(f"resumed from {resume_path} at step {step}, tokens_seen={tokens_seen}")
    elif init_from is not None:
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"initialized weights from {init_from} (fresh optimizer, continued schedule)")

    scheduler = WarmupCosine(total_steps, base_lr, warmup_frac, min_lr_frac)

    def apply_lr() -> None:
        lr = scheduler(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

    def save_latest() -> None:
        payload = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "total_steps": total_steps,
            "base_lr": base_lr,
            "param_lr": [g["lr"] for g in optimizer.param_groups],
            "profile": asdict(mc),
            "history": history,
            "val_history": val_history,
        }
        _atomic_torch_save(payload, checkpoints_dir / "latest.pt")

    interrupted = False
    model.train()
    apply_lr()
    if step == 0:
        log_row({"step": 0, "device_summary": summary, "phase": "start", "total_steps": total_steps, "tokens_per_step": tokens_per_step})

    while step < total_steps:
        if timeout_s is not None and time.time() - started > timeout_s:
            interrupted = True
            print(f"round timeout ({timeout_s:.0f}s): checkpointing partial state at step {step}")
            break
        if interrupt_at_step is not None and step >= interrupt_at_step:
            interrupted = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum):
            index = (step * grad_accum + micro) * micro_batch
            tokens, flags = _window_batch(train_set, index, micro_batch)
            targets = tokens.clone()
            with autocast_ctx(device, enabled=use_bf16):
                _, loss = model(tokens.to(device), doc_flags=flags.to(device), targets=targets.to(device))
            (loss / grad_accum).backward()
            loss_accum += loss.item() / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not math.isfinite(loss_accum) or not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite training state at step {step}: loss={loss_accum}, grad_norm={grad_norm}"
            )
        apply_lr()
        optimizer.step()
        step += 1
        tokens_seen += tokens_per_step
        history.append((step, loss_accum))
        if step % 20 == 0 or step == total_steps:
            log_row({"step": step, "loss": round(loss_accum, 6), "tokens_s": tokens_per_second(), "phase": "train", "tokens_seen": tokens_seen})
        if step % eval_every == 0:
            evaluate_and_log(loss_accum)
        if step % checkpoint_every == 0:
            save_latest()

    final_val = evaluate_and_log(history[-1][1] if history else 0.0)
    save_latest()
    duration = time.time() - started
    result = {
        "step": step,
        "tokens_seen": tokens_seen,
        "history": history,
        "val_history": val_history,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(min(final_val, 20.0)),
        "tokens_s": round(tokens_seen / max(duration, 1e-9), 1),
        "duration_s": round(duration, 1),
        "interrupted": interrupted,
        "device": str(device),
    }
    log_row({"step": step, "phase": "end", "tokens_s": result["tokens_s"], "duration_s": result["duration_s"], "final_val_ppl": result["final_val_ppl"]})
    return result


def promote_checkpoint(checkpoints_dir: Path) -> Path:
    """Atomically promote ``latest.pt`` to ``champion.pt``."""
    latest = checkpoints_dir / "latest.pt"
    champion = checkpoints_dir / "champion.pt"
    if not latest.exists():
        raise FileNotFoundError(f"no checkpoint to promote: {latest}")
    tmp = champion.with_suffix(".pt.tmp")
    tmp.write_bytes(latest.read_bytes())
    os.replace(tmp, champion)
    return champion


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.train.trainer --config PATH --profile NAME --max-tokens N``."""
    import argparse

    from prometheus_ns.model.tokenizer_train import load_tokenizer

    parser = argparse.ArgumentParser(description="Prometheus-NS trainer")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--profile", required=True, help="model profile name")
    parser.add_argument("--max-tokens", default=None, help="token budget override for this run")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume exactly")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg)
    result = train(
        cfg,
        args.profile,
        tokenizer,
        max_tokens=float(args.max_tokens) if args.max_tokens else None,
        resume_path=args.resume,
    )
    if result["val_history"]:
        first = result["val_history"][0]
        last = result["val_history"][-1]
        print(f"val loss {first[1]:.4f} -> {last[1]:.4f} (ppl {first[2]:.2f} -> {last[2]:.2f})")
    print(f"done: {result['step']} steps, {result['tokens_seen']} tokens, {result['tokens_s']} tokens/s on {result['device']}")


if __name__ == "__main__":
    main()
