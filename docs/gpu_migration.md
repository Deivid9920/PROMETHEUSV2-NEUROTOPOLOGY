# GPU Migration Playbook (Phase 8)

This document is written before the migration and executed once GPU
hardware is available. It is the single authorized procedure: any step
performed out of order invalidates the verification gates.

## Policy

The migration changes configuration and environment only. Application
code does not change, with one explicitly authorized exception
(optional flash-attention, step 7). All device-dependent behavior is
already centralized in `prometheus_ns/device.py`; nothing outside it
may probe `torch.cuda` or set thread counts.

## Preconditions (all mandatory)

- Phases 0-6 complete; `make test` green on the current CPU setup.
- `config.yaml` -> `loop.holdout_sha256` set and matching
  `data_clean/holdout/` (verified by `make eval`).
- Working tree clean; tag the pre-migration state:
  `git tag v0.1-cpu && git push --tags` (or local tag).
- NVIDIA driver >= 525 installed and `nvidia-smi` works.
- CUDA compute capability >= 8.0 (Ampere or newer) available. On
  Turing or older, see "bf16 policy" in Troubleshooting before
  starting.

## Step 1 — Reinstall torch with CUDA

Pitfall: if CPU torch (e.g. 2.4.1+cpu) is already installed, a plain
`pip install -r requirements-gpu.txt` is a no-op, because +cpu local
versions satisfy the `torch>=2.3,<2.5` constraint. Force the
reinstall:

```bash
pip install --force-reinstall -r requirements-gpu.txt
```

Verify BEFORE continuing (expected: version without a `+cpu` suffix,
`cuda.is_available()` True, device name matching your GPU):

```bash
python -c "import torch; print(torch.__version__); \
           print(torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0))"
```

If `is_available()` is False, stop: fix the driver/CUDA pairing
first. Nothing downstream can be validated without a working device.

## Step 2 — Configuration-only changes

Apply exactly this diff to `config.yaml`:

```yaml
model:
  profile: nano        ->  profile: small
train:
  # micro_batch / grad_accum: use the gpu values already present
  # (cpu: 16, gpu: 64 / cpu: 4, gpu: 2)
  # no change needed: use_bf16_gpu: true, grad_checkpoint_gpu: true
```

The `small` profile is the migration target. The `large` profile is a
second migration performed only after `small` completes one full loop
round successfully (step 6).

## Step 3 — Structural sanity gate

```bash
make test
python -m prometheus_ns.model.count_params --config config.yaml --profile small
```

Expected: all structural tests green; parameter count inside the
contracted range 85M-115M. Record the exact number for the README
metrics table.

## Step 4 — GPU smoke training

```bash
make train PROFILE=small TOKENS=5e6
```

Verify before continuing:

- Runs to completion without OOM; record peak VRAM (add
  `nvidia-smi --query-gpu=memory.used --format=csv -l 30` in a second
  terminal, or read `logs/metrics.jsonl` if the trainer logs it).
- `logs/metrics.jsonl` shows val loss decreasing across at least
  three evaluations.
- Throughput (tokens/s) recorded; compare against the CPU baseline
  from Phase 3. Expected speedup on a 16 GB consumer GPU: roughly
  10-40x over the CPU figure, but the recorded number is the only
  authoritative value.

## Step 5 — Evaluation integrity

```bash
make eval
```

`eval/perplexity.py` re-verifies `loop.holdout_sha256` before
measuring; the migration does not touch data, so the hash must pass.
If it fails, STOP and restore from `v0.1-cpu`: an evaluation baseline
that cannot be reproduced is worthless.

## Step 6 — One full loop round on GPU

```bash
make loop ROUNDS=1
```

The promotion gate logic is device-agnostic; round 1 must promote
only under the same rules as on CPU. Append the round's real metrics
to `docs/promotion_log.md` (the loop does this) and the README table.

## Step 7 — Optional optimizations (each independently revertible)

### flash-attention (authorized code exception)

`prometheus_ns/model/blocks.py` may add a guarded import and a config
flag. Nothing else changes:

```python
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
```

Add the optional key `model.use_flash_attn: true` to `config.yaml`.
This is permitted: `tests/test_structure.py` validates required keys,
not the absence of optional ones. flash-attn install failures are
frequent and non-blocking: the fallback attention path must remain
the default whenever the import fails.

### torch.compile

Only after step 6 passes, try `torch.compile(model)` behind the same
style of guarded flag. Validate one smoke run for numerical
stability (val loss curve comparable to eager mode) before keeping
it.

### Quantization policy on GPU

Dynamic int8 (`torch.ao.quantization.quantize_dynamic`) is a CPU
deployment path. On GPU, the chat REPL runs in bf16;
`scripts/chat.py` selects per device via `device.get_device()`. No
int8 on GPU.

## Step 8 — large profile (second migration)

After `small` completes a full round: repeat steps 2-6 with
`profile: large`. The profile already carries GQA (`n_kv_head: 8`),
`max_seq 1024` and `vocab 16000`. If OOM occurs at seq 1024, reduce
`train.micro_batch.gpu` from 64 in steps (32, 16) before touching
anything else; do not reduce `grad_accum` to compensate.

## Memory budget reference (estimates to verify, not guarantees)

| Profile | Params | Static (weights bf16 + grads + AdamW moments + fp32 master) | Activations (ckpt on) | Fits 16 GB |
|---------|--------|------|-----------------------|------------|
| small   | ~97M   | ~2.0 GB | ~3-6 GB @ bs64 x seq512 | yes |
| large   | ~285M  | ~5.5 GB | ~6-9 GB @ bs64 x seq1024 | yes |

Static = params x (2 bytes weights bf16 + 2 bytes grads + 8 bytes
AdamW moments fp32 + 4 bytes master fp32). Record measured peak VRAM
for both profiles in the README metrics table.

## Rollback

```bash
git checkout v0.1-cpu
pip install --force-reinstall "torch>=2.3,<2.5" --index-url https://download.pytorch.org/whl/cpu
make test
```

## Troubleshooting

| Symptom | First knob | Second knob | Never do |
|---------|-----------|-------------|----------|
| CUDA OOM at step 4 | reduce `train.micro_batch.gpu` (64->32->16) | verify `grad_checkpoint_gpu: true` | raise `grad_accum` to compensate |
| `is_available()` False | fix driver/CUDA wheel pairing (step 1) | reinstall with cu121 index | continue on CPU silently |
| bf16 errors or extreme slowdown | check compute capability >= 8.0 | on Turing: stay fp32 via `use_bf16_gpu: false` and accept slower runs | switch to fp16 outside the policy |
| flash-attn build failure | ignore; fallback path is the default | remove the `use_flash_attn` flag | make flash-attn a hard dependency |
| val loss NaN on GPU smoke | reduce LR 10x in a scratch run to isolate | verify `grad_clip` 1.0 is applied | blame the hardware first |

## Completion checklist

- Steps 1-6 done in order, each gate passed with recorded evidence.
- README metrics table updated with REAL numbers: param count, GPU
  tokens/s, peak VRAM, per-round promotion metrics.
- `git tag v0.2-gpu` on the verified state.
- Any deviation from this document reported in the commit message
  that introduces it.
