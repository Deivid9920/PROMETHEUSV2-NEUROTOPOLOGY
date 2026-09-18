# Prometheus-NS

A neuro-symbolic language-modeling system that collects its own
training data from the open web, trains a compact decoder-only
transformer under a CPU-first compute budget, and improves itself in
discrete rounds. Every promotion is gated by measured perplexity on a
frozen holdout, a corpus diversity floor as a model-collapse guard,
and a symbolic consistency filter that quarantines contradicting
documents.

## Architecture

```mermaid
flowchart LR
    subgraph Data pipeline
        A[crawler] --> B[extractor] --> C[cleaner] --> D[quality]
    end
    D --> E[tokenize]
    E --> F[trainer]
    subgraph Symbolic layer
        G[triplet extractor] --> H[consistency graph] --> Q[quarantine]
    end
    D --> G
    Q -. excluded documents .-> F
    F --> L{promotion gate}
    L --> P[perplexity + prompt suite]
    L --> R[chat REPL int8]
```

## Repository layout

```text
prometheus_ns/
  device.py        compute policy: device, threads, autocast (single source of truth)
  data/            crawler, extractor, cleaner, quality, reporter
  model/           config, blocks (RMSNorm/SwiGLU/RoPE/GQA), model, tokenizer, param count
  train/           streaming packed dataset, schedules, trainer with atomic checkpoints
  symbolic/        triplet extraction, consistency graph, output validation
  autoloop/        round orchestration, promotion gate, diversity guard
  eval/            held-out perplexity, domain prompt suite
scripts/           chat REPL, single-round runner, holdout freezing
tests/             structural contracts and unit tests
artifacts/         tokenizer (committed) and checkpoints (ignored)
```

## Quickstart

```bash
make setup            # venv + torch (CPU index) + pinned dependencies + spaCy model
make test             # structural contracts (Phase 0 verification)
make data             # crawl, extract, clean, report
python scripts/freeze_holdout.py --config config.yaml   # one-time holdout split
make tokenize         # BPE tokenizer for the active profile
make train PROFILE=nano TOKENS=5e6   # smoke pretraining run on CPU
make loop ROUNDS=1    # one self-improvement round
make eval             # held-out perplexity and prompt suite
make chat             # REPL with dynamic int8 quantization
```

## Usage targets

| Target | Purpose |
|--------|---------|
| `make data` | crawl, extract, clean, report |
| `make tokenize` | train the BPE tokenizer for the active profile |
| `make train PROFILE=nano TOKENS=5e6` | smoke pretraining run on CPU |
| `make loop ROUNDS=1` | one self-improvement round |
| `make loop ROUNDS=2 AUTO_CONTINUE=1` | skip the human gate deliberately |
| `make eval` | held-out perplexity and prompt suite |
| `make chat QUANT=0` | REPL in fp32 (default is dynamic int8) |

## Data policy

- Sources: Wikipedia API, Project Gutenberg, arXiv API, and domains
  whose **robots.txt** allows our agent; the crawler re-checks
  **robots.txt** per URL.
- Identified crawler (`User-Agent: PrometheusNS-Research/0.1`), one
  request per second per domain, exponential backoff on failures.
- Extraction: trafilatura with a readability-lxml fallback. Cleaning:
  unicode repair, exact and MinHash near-duplicate removal, heuristic
  quality filters (symbol ratio, punctuation line ratio, stopword
  ratio). New documents are deduplicated against the historical index
  AND the frozen holdout to prevent test leakage.

## Holdout integrity

The evaluation holdout is split exactly once
(`scripts/freeze_holdout.py`). Its SHA-256 is recorded in
`config.yaml` under `loop.holdout_sha256` and is verified before every
promotion decision; a mismatch aborts the round instead of producing
a comparison. The holdout directory is committed to the repository as
part of the integrity anchor.

## Model profiles

| Profile | d_model | layers | heads | kv-heads | d_ff | max_seq | vocab |
|---------|---------|--------|-------|----------|------|---------|-------|
| nano    | 384     | 6      | 6     | 6        | 1024 | 256     | 8000  |
| small   | 768     | 12     | 12    | 12       | 2048 | 512     | 16000 |
| large   | 1024    | 16     | 16    | 8 (GQA)  | 4096 | 1024    | 16000 |

## Measured metrics

No estimated numbers are reported in this project; the table below is
populated only from executed runs. The smoke values were measured on a
2-core / 3 GB CPU container (the reference environment of the
specification assumes 8-16 cores); Wikipedia API access was blocked
from that network (HTTP 403), so the smoke crawl covered Project
Gutenberg and arXiv only. The reduced-scale configuration used for the
smoke run is preserved in `config.smoke.yaml`; `config.yaml` keeps the
contracted defaults for real runs.

| Metric | Value | Source |
|--------|-------|--------|
| nano parameter count | 13,693,824 | `python -m prometheus_ns.model.count_params --profile nano` |
| small parameter count (analytic) | 97,241,856 | `python -m prometheus_ns.model.count_params --profile small` |
| large parameter count (analytic) | 268,076,032 | `python -m prometheus_ns.model.count_params --profile large` |
| training throughput (CPU smoke, 2 cores) | 1,504.8 tokens/s | `logs/metrics.jsonl` (500k-token run) |
| smoke-run val perplexity | 2430.10 -> 1367.43 (500k tokens, 31 steps) | trainer stdout + `logs/metrics.jsonl` |
| holdout perplexity fp32 | 2,928.72 | `logs/eval_ppl.json` |
| holdout perplexity int8 (dynamic) | 2,927.76 (-0.03% vs fp32) | `logs/eval_ppl.json` |
| tokenizer fertility (nano, 8k vocab) | 1.4114 tokens/word | `artifacts/tokenizer/report.json` |
| round 1 promotion | promoted: val_ppl 2,276.62, diversity 0.4720 | `docs/promotion_log.md` |
| smoke corpus (16 pages crawled, 2 sources) | 22 docs kept, 1.174 MB | `docs/data_report.json` |

## GPU migration

Documented in `docs/gpu_migration.md` (phase 8). The switch is
configuration-only: `model.profile`, batch sizes and bf16 flags live
in `config.yaml`, and all device logic lives in `device.py`.

## Status

| Phase | Scope |
|-------|-------|
| 0 | skeleton, contracts, environment |
| 1 | autonomous data pipeline |
| 2 | tokenizer and model |
| 3 | CPU pretraining |
| 4 | symbolic module |
| 5 | self-improvement loop |
| 6 | evaluation and inference |
| 7 | final documentation |
| 8 | GPU migration (documented, not executed) |

## Limitations

- The nano model is a ~14M parameter system trained on a small corpus:
  generations are often incoherent and factual claims are unreliable.
  The prompt suite records outputs as observed, without corrections.
- int8 dynamic quantization on a model this size can degrade
  perplexity measurably; the measured delta is reported by
  `eval/perplexity.py --int8` rather than assumed below any threshold.
- Packed windows mix documents under plain causal attention; the
  document-boundary flags enable block-diagonal masking, which the
  trainer uses, but the eval path uses plain causal windows.
- The symbolic layer uses curated disjunction lists; polysemous
  subjects ("jaguar" the animal vs the car) are only partially
  disambiguated by entity normalization, so recall of true
  contradictions is bounded by the extractor's pattern coverage.
- Checkpoint writes rely on POSIX atomic rename semantics; run the
  project on Linux/WSL2 (Windows locks `os.replace` targets held open
  by another process).

## Resumen en español

Prometheus-NS es un sistema neuro-simbólico que recolecta sus propios
datos de entrenamiento desde la web (Wikipedia, Project Gutenberg,
arXiv), limpia el corpus con deduplicación exacta y MinHash, entrena
un transformer decoder-only compacto en CPU y se auto-mejora en
rondas medibles. Cada ronda pasa por una compuerta de promoción con
tres reglas: (1) la perplejidad sobre el holdout congelado debe bajar
al menos 1% relativo, (2) la diversidad del corpus no debe caer más
de 20% frente a la ronda previa y (3) el hash SHA-256 del holdout
debe coincidir con el ancla registrada en `config.yaml`; cualquier
discrepancia aborta la ronda. Los documentos contradictorios
detectados por el grafo simbólico van a cuarentena y se excluyen del
pretraining siguiente, sin borrarse, para mantener la auditabilidad.

Comandos equivalentes a la guía rápida: `make setup`, `make test`,
`make data`, `python scripts/freeze_holdout.py --config config.yaml`,
`make tokenize`, `make train PROFILE=nano TOKENS=5e6`,
`make loop ROUNDS=1`, `make eval` y `make chat`.
