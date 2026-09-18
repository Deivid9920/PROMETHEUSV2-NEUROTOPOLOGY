"""Freeze the evaluation holdout exactly once.

Moves ``data.holdout_frac`` of the cleaned documents into
``data_clean/holdout/`` and records the directory's SHA-256 in
``config.yaml`` (``loop.holdout_sha256``). Regeneration is forbidden:
the script refuses to run when an anchor already exists, and the
promotion gate aborts any round whose holdout bytes changed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.autoloop.promotion import compute_holdout_digest


def freeze_holdout(cfg: dict) -> dict:
    """Split off the holdout and write its digest into config.yaml.

    The anchor is re-read from disk on every call, so stale in-memory
    configs can never split the holdout twice. The passed ``cfg`` dict
    is updated in place once the anchor is written.

    Returns:
        Summary with the count of moved docs and the recorded digest.
    """
    clean_dir = repo_path(cfg, "data_clean")
    holdout_dir = clean_dir / "holdout"
    config_path = Path(cfg["_config_path"])
    config_text = config_path.read_text(encoding="utf-8")
    fresh = yaml.safe_load(config_text)
    existing = (fresh.get("loop") or {}).get("holdout_sha256")

    holdout_dir.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        actual = compute_holdout_digest(holdout_dir)
        if actual == existing:
            return {"status": "already-frozen", "docs": len(list(holdout_dir.glob('*.txt'))), "sha256": existing}
        raise RuntimeError(
            f"holdout anchor exists but the directory hash differs "
            f"(config {existing} != dir {actual}): regeneration is forbidden"
        )

    index_path = clean_dir / "index.jsonl"
    if not index_path.exists():
        raise RuntimeError("no cleaned corpus index found: run make data first")

    entries = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise RuntimeError("cleaned corpus index is empty: run make data first")

    # Deterministic selection: sort by content hash and take the first
    # holdout_frac share. No RNG involvement, so any later re-sort or
    # re-run cannot produce a different split.
    frac = float(cfg_get(cfg, "data.holdout_frac", 0.02))
    entries.sort(key=lambda e: e["hash"])
    n_holdout = max(1, int(round(frac * len(entries))))
    selected = entries[:n_holdout]

    moved = 0
    remaining: list[str] = []
    selected_hashes = {e["hash"] for e in selected}
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["hash"] in selected_hashes:
                src = clean_dir / row["path"]
                if src.exists():
                    shutil.move(str(src), str(holdout_dir / src.name))
                    moved += 1
            else:
                remaining.append(line)
    with index_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(remaining) + ("\n" if remaining else ""))

    digest = compute_holdout_digest(holdout_dir)
    updated = re.sub(
        r"holdout_sha256:\s*null",
        f"holdout_sha256: {digest}",
        config_text,
        count=1,
    )
    if updated == config_text:
        raise RuntimeError("could not write loop.holdout_sha256 into config.yaml: anchor line not found")
    config_path.write_text(updated, encoding="utf-8")
    cfg.setdefault("loop", {})["holdout_sha256"] = digest
    return {"status": "frozen", "docs": moved, "sha256": digest}


def main() -> None:
    """CLI entry point: ``python scripts/freeze_holdout.py --config PATH``."""
    parser = argparse.ArgumentParser(description="Prometheus-NS holdout freezing")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = freeze_holdout(cfg)
    print(f"holdout {result['status']}: {result['docs']} docs, sha256={result['sha256']}")


if __name__ == "__main__":
    main()
