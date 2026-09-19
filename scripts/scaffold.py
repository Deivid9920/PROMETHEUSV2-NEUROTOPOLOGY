"""Repository scaffold: directory tree, package inits and module stubs.

Idempotent: existing files are kept untouched, so running it on a
fresh clone rebuilds any missing skeleton stub while leaving the
implemented modules alone. The anchor files listed in ``ANCHORS`` are
immutable contracts: they are never generated here and their presence
is only reported.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLAIN_DIRS = (
    "docs",
    "tests/fixtures",
    "artifacts/tokenizer",
    "artifacts/checkpoints",
    "data_raw",
    "data_clean",
    "data_clean/holdout",
    "triplets",
    "logs",
)

KEEP_EMPTY_DIRS = (
    "logs",
    "data_raw",
    "data_clean",
    "triplets",
    "artifacts/checkpoints",
)

PACKAGES = {
    "prometheus_ns": "Prometheus-NS package root: shared configuration utilities.",
    "prometheus_ns/data": "Autonomous data pipeline: crawl, extract, clean, score, report.",
    "prometheus_ns/model": "Tokenizer and decoder-only transformer model components.",
    "prometheus_ns/train": "Unsupervised pretraining: packed dataset, schedules, trainer.",
    "prometheus_ns/symbolic": "Symbolic layer: triplet extraction, consistency, validation.",
    "prometheus_ns/autoloop": "Self-improvement loop: diversity, promotion gate, rounds.",
    "prometheus_ns/eval": "Evaluation: held-out perplexity and the domain prompt suite.",
}

CLI_MAIN = (
    "",
    "",
    'if __name__ == "__main__":',
    "    main()",
)

STUBS: dict[str, tuple[str, bool]] = {
    "prometheus_ns/data/crawler.py": (
        "Autonomous crawler: per-domain rate limiting, robots.txt checks,\n"
        "exponential backoff retries and raw-HTML persistence.",
        True,
    ),
    "prometheus_ns/data/extractor.py": (
        "Main-content extraction: trafilatura primary, readability fallback,\n"
        "Gutenberg boilerplate stripping and challenge-page detection.",
        False,
    ),
    "prometheus_ns/data/cleaner.py": (
        "Streaming cleaner: unicode normalization, exact and MinHash\n"
        "near-duplicate removal, holdout-leak guard and Gopher filters.",
        True,
    ),
    "prometheus_ns/data/quality.py": (
        "Gopher-style quality heuristics and the weighted quality score\n"
        "used for the keep-quantile cut.",
        False,
    ),
    "prometheus_ns/data/reporter.py": (
        "Data pipeline report: runtime-counted per-source statistics\n"
        "written to docs/data_report.json.",
        True,
    ),
    "prometheus_ns/model/config.py": (
        "ModelConfig dataclass resolved from the active config.yaml\n"
        "profile, plus the analytic parameter count.",
        False,
    ),
    "prometheus_ns/model/blocks.py": (
        "Transformer building blocks: RMSNorm, SwiGLU, RoPE cache and\n"
        "causal attention with grouped-query heads.",
        False,
    ),
    "prometheus_ns/model/model.py": (
        "Decoder-only TransformerLM with tied embeddings and\n"
        "variance-scaled residual initialization.",
        False,
    ),
    "prometheus_ns/model/tokenizer_train.py": (
        "BPE tokenizer training with reserved special tokens, corpus\n"
        "sampling and fertility reporting.",
        True,
    ),
    "prometheus_ns/model/count_params.py": (
        "Parameter count per component and total, checked against the\n"
        "contracted per-profile ranges.",
        True,
    ),
    "prometheus_ns/train/dataset.py": (
        "Streaming packed dataset over numpy memmaps with document\n"
        "boundary flags for block-diagonal attention.",
        False,
    ),
    "prometheus_ns/train/trainer.py": (
        "CPU-first trainer: AdamW, grad accumulation, atomic checkpoints\n"
        "and bit-exact resume including RNG and stream cursor state.",
        True,
    ),
    "prometheus_ns/train/schedules.py": (
        "Learning-rate schedule: linear warmup plus cosine decay to a\n"
        "configurable floor.",
        False,
    ),
    "prometheus_ns/symbolic/extractor.py": (
        "spaCy triplet extraction: copula, verb, passive-voice and\n"
        "apposition patterns with normalized entities and confidence.",
        True,
    ),
    "prometheus_ns/symbolic/consistency.py": (
        "Subject-indexed consistency graph: is_a disjunction and binary\n"
        "exclusivity contradictions with document quarantine.",
        True,
    ),
    "prometheus_ns/symbolic/generator.py": (
        "Output validation: triplets of generated sentences checked\n"
        "against the consistency graph.",
        True,
    ),
    "prometheus_ns/autoloop/loop.py": (
        "Round orchestration: crawl, clean, triplets, quarantine,\n"
        "continued pretraining, evaluation and the promotion gate.",
        True,
    ),
    "prometheus_ns/autoloop/diversity.py": (
        "Distinct-bigram diversity baseline recorded per round.",
        False,
    ),
    "prometheus_ns/eval/perplexity.py": (
        "Held-out perplexity on the frozen holdout with digest\n"
        "verification and optional int8 comparison.",
        True,
    ),
    "prometheus_ns/eval/prompt_suite.py": (
        "Domain prompt suite runner honoring the eval/prompts.yaml\n"
        "schema (defaults, overrides, symbolic checks, diagnostics).",
        True,
    ),
    "scripts/chat.py": (
        "Interactive chat REPL over the champion checkpoint with dynamic\n"
        "int8 quantization and repetition-safe sampling.",
        True,
    ),
    "scripts/run_round.py": (
        "Single-round runner around the self-improvement loop.",
        True,
    ),
    "scripts/freeze_holdout.py": (
        "One-time holdout split with the SHA-256 anchor written into\n"
        "config.yaml; regeneration is forbidden.",
        True,
    ),
}

ANCHORS = (
    "config.yaml",
    "requirements.txt",
    "requirements-gpu.txt",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "Makefile",
    "scripts/scaffold.py",
    "tests/test_structure.py",
    "prometheus_ns/device.py",
    "prometheus_ns/autoloop/promotion.py",
    "eval/prompts.yaml",
    "docs/gpu_migration.md",
)


def stub_source(doc: str, cli: bool) -> str:
    lines = [f'"""{doc}"""', ""]
    if cli:
        lines.extend(CLI_MAIN)
    return "\n".join(lines) + "\n"


def main() -> None:
    created: list[str] = []
    skipped: list[str] = []

    for rel in PLAIN_DIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    for rel in KEEP_EMPTY_DIRS:
        directory = ROOT / rel
        directory.mkdir(parents=True, exist_ok=True)
        keep = directory / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
            created.append(str(keep.relative_to(ROOT)))

    for package, doc in PACKAGES.items():
        init = ROOT / package / "__init__.py"
        if init.exists():
            skipped.append(str(init.relative_to(ROOT)))
            continue
        init.write_text(f'"""{doc}"""\n', encoding="utf-8")
        created.append(str(init.relative_to(ROOT)))

    for rel, (doc, cli) in STUBS.items():
        target = ROOT / rel
        if target.exists():
            skipped.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stub_source(doc, cli), encoding="utf-8")
        created.append(rel)

    print("created:")
    for rel in created:
        print(f"  + {rel}")
    if skipped:
        print("kept existing:")
        for rel in skipped:
            print(f"  = {rel}")

    print("anchors:")
    for rel in ANCHORS:
        state = "present" if (ROOT / rel).is_file() else "MISSING: anchor contract not present"
        print(f"  {'ok' if state == 'present' else '!!'} {rel}: {state}")

    print("\nnext: make setup && make test")


if __name__ == "__main__":
    main()
