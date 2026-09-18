"""Tests for the prompt suite: schema validation and execution path."""

from __future__ import annotations

import json

import pytest
import yaml

from prometheus_ns.eval.prompt_suite import load_prompt_schema, run_suite

PROMPTS_PATH = "eval/prompts.yaml"


def test_repo_schema_is_valid(config) -> None:
    schema = load_prompt_schema(config)
    assert schema["schema_version"] == 1
    prompts = schema["prompts"]
    assert len(prompts) == 20
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids))
    categories = {p["category"] for p in prompts}
    assert {"factual", "listing", "commonsense", "narrative", "technical", "diagnostic"} <= categories
    assert all("review_note" in p for p in prompts)
    diagnostic = [p for p in prompts if p.get("diagnostic")]
    assert diagnostic, "the suite must include diagnostic probes"
    assert all("review_note" in p for p in diagnostic)


def test_repo_schema_has_defaults_and_output_path(config) -> None:
    schema = load_prompt_schema(config)
    assert {"max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"} <= set(schema["defaults"])
    assert schema["output_path"] == "eval/prompt_outputs.jsonl"


def test_schema_version_mismatch_rejected(tmp_path) -> None:
    bad = tmp_path / "prompts.yaml"
    bad.write_text("schema_version: 99\nprompts: []\n", encoding="utf-8")
    cfg = {"_config_path": str(tmp_path / "config.yaml")}
    import prometheus_ns.eval.prompt_suite as suite

    original = suite.PROMPTS_PATH
    suite.PROMPTS_PATH = "prompts.yaml"
    try:
        with pytest.raises(ValueError, match="schema_version"):
            load_prompt_schema(cfg)
    finally:
        suite.PROMPTS_PATH = original


def test_duplicate_ids_rejected(tiny_repo) -> None:
    eval_dir = tiny_repo["root"] / "eval"
    (eval_dir / "prompts.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "defaults": {"max_new_tokens": 4},
                "output_path": "eval/out.jsonl",
                "prompts": [
                    {"id": "a", "category": "factual", "prompt": "p", "review_note": "n"},
                    {"id": "a", "category": "factual", "prompt": "q", "review_note": "n"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_schema(tiny_repo["cfg"])


def test_run_suite_writes_outputs(tiny_repo, monkeypatch) -> None:
    import torch
    from dataclasses import asdict

    from prometheus_ns.model.config import ModelConfig
    from prometheus_ns.model.model import TransformerLM

    eval_dir = tiny_repo["root"] / "eval"
    (eval_dir / "prompts.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "defaults": {"max_new_tokens": 6, "temperature": 0.8, "top_k": 10, "top_p": 0.9,
                              "repetition_penalty": 1.1, "apply_symbolic_check": True},
                "output_path": "eval/out.jsonl",
                "prompts": [
                    {"id": "free-01", "category": "narrative", "prompt": "The old keeper walked",
                     "review_note": "free continuation", "apply_symbolic_check": False},
                    {"id": "fact-01", "category": "factual", "prompt": "The keeper lit",
                     "review_note": "checked against the graph"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ckpt_dir = tiny_repo["root"] / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mc = ModelConfig(64, 2, 4, 2, 128, 64, 512, tie_embeddings=True)
    torch.manual_seed(5)
    torch.save({"model_state": TransformerLM(mc).state_dict(), "profile": asdict(mc)}, ckpt_dir / "latest.pt")

    # The suite loads the persisted tokenizer artifact.
    tokenizer_dir = tiny_repo["root"] / "artifacts" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tiny_repo["tokenizer"].save(str(tokenizer_dir / "tokenizer.json"))

    records = run_suite(tiny_repo["cfg"])
    assert len(records) == 2
    assert records[0]["symbolic_status"] == "not-applied"  # per-prompt override honored
    out_path = tiny_repo["root"] / "eval" / "out.jsonl"
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert {r["id"] for r in rows} == {"free-01", "fact-01"}
    assert all("output" in r and "params" in r for r in rows)
