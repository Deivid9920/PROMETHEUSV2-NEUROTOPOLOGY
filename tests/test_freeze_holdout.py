"""Tests for the one-time holdout freeze script."""

from __future__ import annotations

import json

import pytest

from prometheus_ns.autoloop.promotion import compute_holdout_digest


@pytest.fixture
def frozen(tiny_repo):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.freeze_holdout import freeze_holdout

    tiny_repo["freeze_fn"] = freeze_holdout
    return tiny_repo


def test_freeze_moves_holdout_share_and_writes_anchor(frozen) -> None:
    result = frozen["freeze_fn"](frozen["cfg"])
    assert result["status"] == "frozen"
    assert result["docs"] == 1  # 2% of 24 docs, minimum one

    holdout_dir = frozen["root"] / "data_clean" / "holdout"
    moved = list(holdout_dir.glob("*.txt"))
    assert len(moved) == result["docs"]

    cfg_text = (frozen["root"] / "config.yaml").read_text(encoding="utf-8")
    assert f"holdout_sha256: {result['sha256']}" in cfg_text
    assert compute_holdout_digest(holdout_dir) == result["sha256"]

    # The moved doc must be gone from the main index and corpus.
    index_rows = [
        json.loads(line)
        for line in (frozen["root"] / "data_clean" / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    moved_stems = {p.stem for p in moved}
    assert not any(row["path"].split("/")[-1].replace(".txt", "") in moved_stems for row in index_rows)


def test_second_freeze_is_a_noop(frozen) -> None:
    first = frozen["freeze_fn"](frozen["cfg"])
    second = frozen["freeze_fn"](frozen["cfg"])
    assert first["status"] == "frozen"
    assert second["status"] == "already-frozen"
    assert second["sha256"] == first["sha256"]


def test_tampered_holdout_refuses_regeneration(frozen) -> None:
    frozen["freeze_fn"](frozen["cfg"])
    holdout_dir = frozen["root"] / "data_clean" / "holdout"
    next(holdout_dir.glob("*.txt")).write_text("changed bytes", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        frozen["freeze_fn"](frozen["cfg"])


def test_freeze_without_corpus_fails(tmp_path) -> None:
    import yaml
    from prometheus_ns import load_config

    root = tmp_path / "repo"
    (root / "data_clean").mkdir(parents=True)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"loop": {"holdout_sha256": None}, "data": {"holdout_frac": 0.02}}), encoding="utf-8")
    cfg = load_config(cfg_path)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.freeze_holdout import freeze_holdout

    with pytest.raises(RuntimeError, match="no cleaned corpus"):
        freeze_holdout(cfg)
