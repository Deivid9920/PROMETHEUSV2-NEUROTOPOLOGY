"""Tests for the data report generator: numbers must come from real artifacts."""

from __future__ import annotations

import json

from prometheus_ns.data.reporter import build_report


def _seed_repo(root) -> None:
    (root / "data_clean" / "gutenberg").mkdir(parents=True)
    (root / "data_clean" / "arxiv").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)

    doc_a = root / "data_clean" / "gutenberg" / "aaa.txt"
    doc_b = root / "data_clean" / "arxiv" / "bbb.txt"
    doc_a.write_text("x" * 1024, encoding="utf-8")
    doc_b.write_text("y" * 2048, encoding="utf-8")
    index = [
        {"hash": "a" * 64, "source": "gutenberg", "path": "gutenberg/aaa.txt", "chars": 1024},
        {"hash": "b" * 64, "source": "arxiv", "path": "arxiv/bbb.txt", "chars": 2048},
    ]
    (root / "data_clean" / "index.jsonl").write_text(
        "\n".join(json.dumps(e) for e in index) + "\n", encoding="utf-8"
    )
    crawl_rows = [
        {"url": "https://gutenberg.org/a", "status": 200, "bytes": 1024, "ts": 100.0},
        {"url": "https://arxiv.org/b", "status": 200, "bytes": 2048, "ts": 160.0},
    ]
    (root / "logs" / "crawl.jsonl").write_text(
        "\n".join(json.dumps(r) for r in crawl_rows) + "\n", encoding="utf-8"
    )
    (root / "logs" / "clean_stats.json").write_text(
        json.dumps({"duration_s": 30.0, "discarded": {"dup_exact": 3, "dup_near": 2, "symbol_ratio": 1}}),
        encoding="utf-8",
    )


def test_report_counts_from_artifacts(tmp_path) -> None:
    from prometheus_ns import load_config

    root = tmp_path / "repo"
    _seed_repo(root)
    cfg = load_config(root / "config.yaml") if (root / "config.yaml").exists() else {"_config_path": str(root / "config.yaml")}
    report = build_report(cfg)

    assert report["total_docs"] == 2
    assert set(report["por_fuente"]) == {"gutenberg", "arxiv"}
    assert report["por_fuente"]["gutenberg"]["docs"] == 1
    assert report["por_fuente"]["arxiv"]["docs"] == 1
    assert abs(report["por_fuente"]["arxiv"]["MB"] - 2048 / (1024 * 1024)) < 1e-3
    assert report["dup_exactos"] == 3
    assert report["dup_near"] == 2
    # 60 s of crawl span + 30 s of cleaning, measured from artifacts.
    assert report["duracion_s"] == 90.0
    assert "descartados_por_regla" in report["por_fuente"]["gutenberg"]
