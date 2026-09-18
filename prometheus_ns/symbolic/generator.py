"""Validation of model output sentences against the consistency graph.

At inference time, NER plus the same dependency patterns used during
extraction run over generated sentences; each resulting triplet is
checked against the graph and classified as ``ok`` (present with a
compatible object), ``contradicts`` (conflicts with a high-confidence
graph fact through the disjunction or exclusivity rules) or
``unverified`` (no graph evidence either way). The prompt suite uses
this to annotate outputs without correcting them post hoc.
"""

from __future__ import annotations

from dataclasses import asdict

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.symbolic.consistency import Contradiction, find_contradictions
from prometheus_ns.symbolic.extractor import Triplet, extract_triplets_from_doc, load_nlp


def check_sentences(
    sentences: list[str],
    cfg: dict,
    graph_triplets: list[Triplet] | None = None,
) -> list[dict]:
    """Classify each sentence's claims against the consistency graph.

    Args:
        sentences: Generated sentences to validate.
        cfg: Parsed project configuration.
        graph_triplets: Pre-extracted graph facts; loaded from
            ``triplets/triplets.jsonl`` when omitted.

    Returns:
        One record per sentence: ``{"sentence", "claims", "status",
        "detail"}`` where ``status`` is the worst verdict among its
        claims (contradicts > unverified > ok).
    """
    nlp = load_nlp(cfg)
    if graph_triplets is None:
        from prometheus_ns.symbolic.consistency import load_triplets

        graph_triplets = load_triplets(repo_path(cfg, "triplets", "triplets.jsonl"))

    min_confidence = float(cfg_get(cfg, "symbolic.min_confidence", 0.6))
    disjoint = list(cfg_get(cfg, "symbolic.disjoint_classes", []))
    exclusive = [list(p) for p in cfg_get(cfg, "symbolic.exclusive_pairs", [])]
    max_entities = int(cfg_get(cfg, "symbolic.max_graph_entities", 200000))

    known: dict[tuple[str, str], set[str]] = {}
    for t in graph_triplets:
        known.setdefault((t.sub, t.rel), set()).add(t.obj)

    records: list[dict] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        doc = nlp(sentence)
        claims = extract_triplets_from_doc(doc, min_confidence=0.0)
        if not claims:
            records.append({"sentence": sentence, "claims": [], "status": "ok", "detail": "no factual claims detected"})
            continue
        candidate, _ = find_contradictions(claims, disjoint, exclusive, min_confidence=0.0, max_entities=max_entities)
        statuses: list[str] = []
        details: list[str] = []
        for claim in claims:
            peers = known.get((claim.sub, claim.rel), set())
            if not peers:
                statuses.append("unverified")
                details.append(f"no graph evidence for ({claim.sub}, {claim.rel}, {claim.obj})")
                continue
            if claim.obj in peers:
                statuses.append("ok")
                details.append(f"matches graph: ({claim.sub}, {claim.rel}, {claim.obj})")
                continue
            contradicts = any(
                _conflict_class(claim.obj, peer, disjoint) or _conflict_pair(claim.obj, peer, exclusive)
                for peer in peers
            )
            statuses.append("contradicts" if contradicts else "unverified")
            details.append(f"conflicts with graph facts for ({claim.sub}, {claim.rel})")
        worst = "contradicts" if "contradicts" in statuses else ("unverified" if "unverified" in statuses else "ok")
        records.append(
            {
                "sentence": sentence,
                "claims": [asdict(c) for c in claims],
                "status": worst,
                "detail": "; ".join(details),
            }
        )
    return records


def _conflict_class(a: str, b: str, disjoint_classes: list[str]) -> bool:
    """True when both values are distinct members of the curated classes."""
    return a in disjoint_classes and b in disjoint_classes and a != b


def _conflict_pair(a: str, b: str, exclusive_pairs: list[list[str]]) -> bool:
    return sorted((a, b)) in [sorted(pair) for pair in exclusive_pairs]


def main() -> None:
    """CLI entry point: validate sentences read from stdin."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Prometheus-NS output validation")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sentences = [line.strip() for line in sys.stdin if line.strip()]
    for record in check_sentences(sentences, cfg):
        print(f"[{record['status']}] {record['sentence']} :: {record['detail']}")


if __name__ == "__main__":
    main()
