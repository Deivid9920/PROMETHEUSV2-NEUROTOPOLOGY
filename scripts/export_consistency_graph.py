#!/usr/bin/env python3
"""Export the PROMETHEUS-NS consistency graph for the CASSANDRA bridge.

Contract (defined by cassandra_ns/ensemble/symbolic_bridge.py):

  {"triplets": [["paris", "located_in", "france", true],
                ["paris", "located_in", "germany", false], ...]}

Every triplet in the project store (triplets/triplets.jsonl) is
emitted as supported (true). Pairs found contradicted by the
consistency analysis (triplets/contradictions.jsonl) are additionally
emitted as contradicted (false). Absent triplets count as unknown
(0.5) on the CASSANDRA side, so export is conservative: nothing that
the graph does not know is ever asserted.

Predicate-scheme adapter (documented interoperability, no new facts):
CASSANDRA's shallow claim extractor emits SURFACE verbs ("is", "wrote",
"born") while PROMETHEUS-NS stores LEMMATISED relations ("is_a",
"write", "bear"). For each graph triplet whose relation has a surface
alias in CASSANDRA's closed verb list, an alias entry with the SAME
subject/object and the SAME consistency value is added. The adapter
only renames predicates between the two projects' schemas; it never
changes the set of subjects/objects or the consistency verdicts.

Outputs:
  artifacts/prometheus_consistency_graph.json   (the bridge graph)
  logs/graph_export_stats.json                  (export statistics)

Usage:
  PYTHONPATH=. python scripts/export_consistency_graph.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# CASSANDRA claim-surface -> PROMETHEUS-NS lemma, for the closed verb
# list of cassandra_ns.ensemble.symbolic_bridge._VERBS. Only lemmas
# that actually occur in the store produce alias entries.
_SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "is_a": ("is", "are", "was", "were"),
    "write": ("wrote",),
    "invent": ("invented",),
    "discover": ("discovered",),
    "locate": ("located",),
    "die": ("died",),
    "live": ("lived",),
    "found": ("founded",),
    "bear": ("born",),
    "have": ("has", "had"),
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    from prometheus_ns import load_config, repo_path
    from prometheus_ns.symbolic.consistency import load_triplets

    cfg = load_config(args.config)
    triplets = load_triplets(repo_path(cfg, "triplets", "triplets.jsonl"))
    contrad_path = repo_path(cfg, "triplets", "contradictions.jsonl")
    contrad_pairs: set[tuple[str, str, str]] = set()
    if contrad_path.is_file():
        for line in contrad_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for side in ("triplet_a", "triplet_b"):
                t = row.get(side, {})
                contrad_pairs.add((t["sub"], t["rel"], t["obj"]))

    seen: set[tuple[str, str, str]] = set()
    rows: list[list] = []
    n_supported = n_contradicted = n_alias = 0
    for t in triplets:
        consistent = (t.sub, t.rel, t.obj) not in contrad_pairs
        entries = [(t.sub, t.rel, t.obj)]
        for surface in _SURFACE_ALIASES.get(t.rel, ()):
            entries.append((t.sub, surface, t.obj))
        for s, p, o in entries:
            if (s, p, o) in seen:
                continue
            seen.add((s, p, o))
            rows.append([s, p, o, consistent])
            if consistent:
                n_supported += 1
            else:
                n_contradicted += 1
            if p != t.rel:
                n_alias += 1

    rel_counts: dict[str, int] = {}
    for t in triplets:
        rel_counts[t.rel] = rel_counts.get(t.rel, 0) + 1

    doc = {
        "meta": {
            "source": "PROMETHEUS-NS symbolic store",
            "triplets_store": "triplets/triplets.jsonl",
            "n_store_triplets": len(triplets),
            "n_contradicted_pairs": len(contrad_pairs),
            "predicate_adapter": ("surface aliases of CASSANDRA _VERBS "
                                  "added for lemmatised relations; same "
                                  "subject/object and consistency"),
            "note": ("absent triplets are UNKNOWN on the CASSANDRA side "
                     "(signal 0.5); the graph never asserts them"),
        },
        "triplets": rows,
    }
    out = repo_path(cfg, "artifacts", "prometheus_consistency_graph.json")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    stats = {
        "store_triplets": len(triplets),
        "exported_rows": len(rows),
        "supported": n_supported,
        "contradicted": n_contradicted,
        "alias_rows": n_alias,
        "unique_subjects": len({t.sub for t in triplets}),
        "unique_objects": len({t.obj for t in triplets}),
        "relations": dict(sorted(rel_counts.items(),
                                 key=lambda kv: -kv[1])),
    }
    stats_path = repo_path(cfg, "logs", "graph_export_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
