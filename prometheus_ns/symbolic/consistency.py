"""Contradiction detection over the triplet graph.

Builds a NetworkX DiGraph indexed by subject and flags two kinds of
inconsistency:

(a) is_a disjunction: the same subject attached to two distinct
    classes of the curated ``symbolic.disjoint_classes`` list
    (bird/mammal/fish/vehicle/food/...). Polysemy ("jaguar" the animal
    vs "jaguar" the car) is mitigated by entity normalization plus the
    strict class whitelist: pairs outside the list never contradict;
(b) binary exclusivity: the same (subject, relation) pointing at two
    objects listed in ``symbolic.exclusive_pairs`` (alive/dead, ...).

Conflicting documents go to ``triplets/quarantine.jsonl`` and are
excluded from the next round's pretraining; nothing is deleted, so the
decision stays auditable. The graph is capped at
``symbolic.max_graph_entities`` nodes as a RAM guard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.symbolic.extractor import Triplet


@dataclass(frozen=True)
class Contradiction:
    """One detected conflict between two triplets."""

    triplet_a: dict
    triplet_b: dict
    contradice_con: str
    conf_media: float


def load_triplets(path: Path) -> list[Triplet]:
    """Read a triplets JSONL file into :class:`Triplet` objects."""
    triplets: list[Triplet] = []
    if not path.exists():
        return triplets
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            triplets.append(Triplet(row["sub"], row["rel"], row["obj"], float(row["conf"]), row.get("source_doc_hash", "")))
    return triplets


class ConsistencyGraph:
    """Subject-indexed DiGraph with a hard entity cap for RAM safety."""

    def __init__(self, max_entities: int = 200000) -> None:
        self.graph = nx.DiGraph()
        self.max_entities = max_entities
        self.capped = False

    def add(self, triplet: Triplet) -> None:
        sub, obj = triplet.sub, triplet.obj
        new_nodes = {sub, obj} - set(self.graph.nodes)
        if self.graph.number_of_nodes() + len(new_nodes) > self.max_entities:
            self.capped = True
            return
        self.graph.add_node(sub)
        self.graph.add_node(obj)
        if self.graph.has_edge(sub, obj):
            edge = self.graph[sub][obj]
            edge["confs"].append(triplet.conf)
            edge["docs"].append(triplet.source_doc_hash)
        else:
            self.graph.add_edge(sub, obj, rel=triplet.rel, confs=[triplet.conf], docs=[triplet.source_doc_hash])

    def subjects(self) -> set[str]:
        return set(self.graph.nodes)


def find_contradictions(
    triplets: list[Triplet],
    disjoint_classes: list[str],
    exclusive_pairs: list[list[str]],
    min_confidence: float,
    max_entities: int = 200000,
) -> tuple[list[Contradiction], set[str]]:
    """Flag conflicting triplets and the documents that asserted them.

    Returns:
        ``(contradictions, quarantined_doc_hashes)``. A document is
        quarantined only when BOTH sides of a detected conflict meet
        the confidence threshold, which limits false positives from
        noisy low-confidence extractions.
    """
    graph = ConsistencyGraph(max_entities)
    for triplet in triplets:
        graph.add(triplet)

    by_subject: dict[str, list[Triplet]] = {}
    for triplet in triplets:
        by_subject.setdefault(triplet.sub, []).append(triplet)

    disj = set(disjoint_classes)
    contradictions: list[Contradiction] = []
    quarantined: set[str] = set()

    for subject, items in by_subject.items():
        # (a) is_a disjunction over the curated class list.
        class_facts = [t for t in items if t.rel == "is_a" and t.obj in disj and t.conf >= min_confidence]
        seen_classes: dict[str, Triplet] = {}
        for fact in class_facts:
            if fact.obj in seen_classes:
                continue
            for other in seen_classes.values():
                contradictions.append(
                    Contradiction(
                        asdict(fact),
                        asdict(other),
                        "is_a_disjunction",
                        round((fact.conf + other.conf) / 2, 4),
                    )
                )
                if fact.source_doc_hash:
                    quarantined.add(fact.source_doc_hash)
                if other.source_doc_hash:
                    quarantined.add(other.source_doc_hash)
            seen_classes[fact.obj] = fact
        # (b) binary exclusivity per (subject, relation).
        pair_set = {frozenset(pair) for pair in exclusive_pairs}
        by_rel: dict[str, list[Triplet]] = {}
        for t in items:
            if t.rel == "is_a":
                continue
            by_rel.setdefault(t.rel, []).append(t)
        for rel, rel_items in by_rel.items():
            for i, a in enumerate(rel_items):
                for b in rel_items[i + 1:]:
                    if frozenset((a.obj, b.obj)) in pair_set and a.conf >= min_confidence and b.conf >= min_confidence:
                        contradictions.append(
                            Contradiction(asdict(a), asdict(b), "exclusive_pair", round((a.conf + b.conf) / 2, 4))
                        )
                        if a.source_doc_hash:
                            quarantined.add(a.source_doc_hash)
                        if b.source_doc_hash:
                            quarantined.add(b.source_doc_hash)
    return contradictions, quarantined


def run_consistency(cfg: dict) -> dict:
    """Detect contradictions over ``triplets/triplets.jsonl`` and quarantine.

    Writes ``triplets/contradictions.jsonl`` and
    ``triplets/quarantine.jsonl``; returns run stats.
    """
    triplets_dir = repo_path(cfg, "triplets")
    triplets = load_triplets(triplets_dir / "triplets.jsonl")
    min_confidence = float(cfg_get(cfg, "symbolic.min_confidence", 0.6))
    disjoint = list(cfg_get(cfg, "symbolic.disjoint_classes", []))
    exclusive = [list(p) for p in cfg_get(cfg, "symbolic.exclusive_pairs", [])]
    max_entities = int(cfg_get(cfg, "symbolic.max_graph_entities", 200000))

    contradictions, quarantined = find_contradictions(triplets, disjoint, exclusive, min_confidence, max_entities)

    triplets_dir.mkdir(parents=True, exist_ok=True)
    with (triplets_dir / "contradictions.jsonl").open("w", encoding="utf-8") as fh:
        for c in contradictions:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    with (triplets_dir / "quarantine.jsonl").open("w", encoding="utf-8") as fh:
        for dhash in sorted(quarantined):
            fh.write(json.dumps({"source_doc_hash": dhash, "reason": "symbolic_contradiction"}, ensure_ascii=False) + "\n")

    stats = {
        "triplets": len(triplets),
        "contradictions": len(contradictions),
        "quarantined_docs": len(quarantined),
        "graph_capped": False,
    }
    return stats


def main() -> None:
    """CLI entry point: contradiction detection over extracted triplets."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS consistency analysis")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(run_consistency(cfg), indent=2))


if __name__ == "__main__":
    main()
