"""Data pipeline report generator.

Reads the crawl log, the cleaner stats and the cleaned-corpus index and
writes ``docs/data_report.json``. Every number is counted at runtime
from real artifacts; nothing is estimated.
"""

from __future__ import annotations

import json
from pathlib import Path

from prometheus_ns import cfg_get, load_config, repo_path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def build_report(cfg: dict) -> dict:
    """Assemble the data report from runtime artifacts."""
    clean_dir = repo_path(cfg, "data_clean")
    index = _read_jsonl(clean_dir / "index.jsonl")
    crawl_log = _read_jsonl(repo_path(cfg, "logs", "crawl.jsonl"))

    per_source: dict[str, dict] = {}
    total_bytes = 0
    for entry in index:
        source = entry.get("source", "unknown")
        path = clean_dir / entry.get("path", "")
        size = path.stat().st_size if path.exists() else 0
        total_bytes += size
        bucket = per_source.setdefault(
            source,
            {"docs": 0, "MB": 0.0, "descartados_por_regla": {}},
        )
        bucket["docs"] += 1
        bucket["MB"] += size / (1024 * 1024)

    clean_stats: dict = {}
    stats_path = repo_path(cfg, "logs", "clean_stats.json")
    if stats_path.exists():
        clean_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    discarded = clean_stats.get("discarded", {})

    crawl_ts = [e.get("ts", 0.0) for e in crawl_log if e.get("ts")]
    crawl_s = (max(crawl_ts) - min(crawl_ts)) if len(crawl_ts) > 1 else 0.0

    report = {
        "por_fuente": {
            source: {
                "docs": bucket["docs"],
                "MB": round(bucket["MB"], 3),
                "descartados_por_regla": dict(discarded),
            }
            for source, bucket in sorted(per_source.items())
        },
        "total_docs": len(index),
        "total_MB": round(total_bytes / (1024 * 1024), 3),
        "dup_exactos": discarded.get("dup_exact", 0),
        "dup_near": discarded.get("dup_near", 0),
        "duracion_s": round(crawl_s + clean_stats.get("duration_s", 0.0), 1),
    }
    return report


def main() -> None:
    """CLI entry point: ``python -m prometheus_ns.data.reporter --config PATH``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS data report")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = build_report(cfg)
    out_path = repo_path(cfg, "docs", "data_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"data report written: {out_path} ({report['total_docs']} docs, {report['total_MB']} MB)")


if __name__ == "__main__":
    main()
