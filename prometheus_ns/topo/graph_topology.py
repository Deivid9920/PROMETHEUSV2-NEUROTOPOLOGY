"""Symbolic-graph topology sensors (V2.1). Implementation anchor.

Two sensors with explicitly different roles:

R3a (PRIMARY -- feeds the veto): census of directed simple cycles in the
is_a graph whose edges all carry conf >= graph_conf_min. A directed cycle
A->B->...->A with confident edges IS a structural taxonomic
contradiction: antisymmetry violated around a loop. This is what the S2
stress round targets and what rule R3 consumes (as graph_h1_count).

R3b (SECONDARY -- observation only): persistent H1 of the Vietoris-Rips
complex over the adjacency distance matrix d(u,v) = 1 - conf (missing
edge: 1). Under this filtration high-confidence cycles CLOSE EARLY (the
2-simplices fill them, so persistence is LOW) and weak cycles persist.
R3b is therefore a geometric indicator of graph reorganization, NOT a
contradiction sensor. An earlier design inverted this reading and would
have produced a false negative on S2; it is corrected here. Never feed
R3b to the veto.

Determinism: triplets are sorted and deduplicated before building the
graph, so cycle enumeration order is reproducible. The census uses a
BOUNDED DFS (depth <= max_cycle_len, work and count capped) because a
plain simple_cycles enumeration on a dense top-K graph yields an
astronomical number of long cycles that no CPU budget survives; either
cap being hit makes truncated=True (the reported count is a floor,
never silent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prometheus_ns import cfg_get, load_config, repo_path


def load_triplets(path: Path) -> list[dict]:
    """Load triplets.jsonl, fail-loud on malformed records.

    A record without sub/obj/conf is a pipeline defect upstream, never a
    silent zero: SystemExit with the offending line number.
    """
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{lineno}: invalid JSON ({exc}); the triplet "
                    "pipeline upstream is broken, fix it there") from exc
            for key in ("sub", "obj", "conf"):
                if key not in rec:
                    raise SystemExit(
                        f"{path}:{lineno}: triplet missing '{key}': the "
                        "extractor contract is violated, fail loud here")
            records.append(rec)
    records.sort(key=lambda r: (str(r["sub"]), str(r["obj"]),
                                -float(r["conf"]), str(r.get("triplet_id", ""))))
    return records


def build_isa_graph(triplets: list[dict], topk: int | None,
                    conf_min: float, relation: str = "is_a") -> dict:
    """Deterministic is_a digraph: g[u][v] = {'conf', 'triplet_id'}.

    Deduplicates (sub, obj) keeping the MAX-confidence record; drops
    edges below conf_min; optionally keeps the top-K entities by degree
    (ties broken by entity name) so the census budget is bounded.
    """
    best: dict[tuple[str, str], dict] = {}
    for rec in triplets:
        if str(rec.get("rel", relation)) != relation:
            continue
        conf = float(rec["conf"])
        if conf < conf_min:
            continue
        key = (str(rec["sub"]), str(rec["obj"]))
        prev = best.get(key)
        if prev is None or conf > float(prev["conf"]):
            best[key] = {"conf": conf,
                         "triplet_id": str(rec.get("triplet_id", ""))}

    if topk is not None and len(best) > 0:
        degree: dict[str, int] = {}
        for (u, v) in best:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        keep = set(sorted(degree, key=lambda n: (-degree[n], n))[:topk])
        best = {k: v for k, v in best.items() if k[0] in keep and k[1] in keep}

    graph: dict[str, dict[str, dict]] = {}
    for (u, v), payload in best.items():
        graph.setdefault(u, {})[v] = payload
    return graph


def census_directed_cycles(graph: dict, max_cycle_len: int = 5,
                           max_cycles: int = 10000,
                           max_work: int = 2_000_000) -> tuple[list[dict], bool]:
    """R3a census: directed simple cycles with min edge confidence.

    Bounded DFS (depth <= max_cycle_len) instead of unbounded
    simple_cycles enumeration: on a dense top-K graph the number of
    LONG simple cycles is astronomical and a generator that filters them
    still enumerates them, which no CPU budget survives. Each cycle is
    reported once at its minimal node (canonical rotation). Work and
    count are both capped; either cap being hit makes truncated=True
    (pinned semantics: the reported count is a floor, never silent).
    Returns (cycles, truncated); cycle order is deterministic.
    """
    nodes = sorted({u for u in graph} | {v for t in graph.values()
                                         for v in t})
    index = {name: i for i, name in enumerate(nodes)}
    successors = {
        index[u]: {index[v]: float(p["conf"]) for v, p in targets.items()}
        for u, targets in graph.items() for v in targets
    }

    cycles: list[dict] = []
    truncated = False
    work = 0
    for start in range(len(nodes)):
        # canonical rotation: a cycle is reported at its minimal node
        stack = [(start, [start], {start})]
        while stack:
            current, path, on_path = stack.pop()
            for nxt in sorted(successors.get(current, {}), reverse=True):
                work += 1
                if work > max_work:
                    truncated = True
                    stack.clear()
                    break
                if nxt == start and len(path) >= 2:
                    if len(cycles) < max_cycles:
                        confs = [successors[path[k]][path[(k + 1) % len(path)]]
                                 for k in range(len(path))]
                        cycles.append({
                            "path": [nodes[i] for i in path],
                            "min_conf": float(min(confs))})
                    else:
                        truncated = True
                        stack.clear()
                        break
                elif nxt > start and nxt not in on_path \
                        and len(path) < max_cycle_len:
                    stack.append((nxt, path + [nxt], on_path | {nxt}))
            if truncated:
                break
        if truncated:
            break

    truncated = truncated or len(cycles) >= max_cycles
    cycles.sort(key=lambda c: (len(c["path"]), tuple(c["path"]),
                               -c["min_conf"]))
    return cycles, truncated


def triangle_census(graph: dict) -> int:
    """EXACT count of directed triangles (3-cycles) = trace(A^3) / 3.

    The veto input for R3a. The general length-5 census is budget-bound
    on a dense top-K graph (both clean and degraded graphs saturate any
    work cap, so their floors measure the budget, not the graph); the
    triangle count is exact, deterministic and budget-free, and directed
    triangles are the dominant shape of real taxonomic contradictions
    (A->B->C->A). A planted or real confidence-closed triangle moves this
    number; the R3 ceiling (1.25x champion) compares exact counts.
    """
    nodes = sorted({u for u in graph} | {v for t in graph.values()
                                         for v in t})
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=np.int64)
    for u, targets in graph.items():
        for v in targets:
            adjacency[index[u], index[v]] = 1
    if n == 0:
        return 0
    return int(np.trace(adjacency @ adjacency @ adjacency) // 3)


def graph_cycle_count(graph: dict, max_cycle_len: int = 5,
                      max_cycles: int = 10000) -> tuple[float, dict]:
    """R3a scalar for the veto (graph_h1_count slot) + audit details."""
    cycles, truncated = census_directed_cycles(graph, max_cycle_len,
                                               max_cycles)
    details = {"cycles": cycles[:50], "truncated": truncated,
               "count": len(cycles)}
    return float(len(cycles)), details


def adjacency_distance_matrix(graph: dict) -> tuple[list[str], np.ndarray]:
    """R3b input: d(u,v) = 1 - conf over the SYMMETRIZED relation.

    Symmetrization keeps the STRONGER relation (higher confidence =
    smaller distance); missing edges sit at distance 1.0. Returns the
    sorted node list and the (n, n) distance matrix.
    """
    nodes = sorted({u for u in graph} | {v for t in graph.values()
                                         for v in t})
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    matrix = np.ones((n, n), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    for u, targets in graph.items():
        for v, payload in targets.items():
            i, j = index[u], index[v]
            distance = 1.0 - float(payload["conf"])
            matrix[i, j] = min(matrix[i, j], distance)
            matrix[j, i] = min(matrix[j, i], distance)
    return nodes, matrix


def rips_h1_persistence(graph: dict) -> tuple[np.ndarray, dict]:
    """R3b: persistent H1 of the Vietoris-Rips over the adjacency matrix.

    READING (honest, mandatory): under this filtration high-confidence
    cycles CLOSE EARLY (their 2-simplices fill them, so persistence is
    LOW) and weak cycles persist. R3b is a geometric indicator of graph
    reorganization, NOT a contradiction sensor; it is reported, never
    vetoed. Requires the persistence engine (Fase 1-V2).
    """
    from prometheus_ns.topo.persistence import compute_diagrams

    nodes, matrix = adjacency_distance_matrix(graph)
    diagrams = compute_diagrams(None, maxdim=1, distance_matrix=matrix)
    meta = {"nodes": nodes, "matrix": matrix, "n_nodes": len(nodes)}
    return diagrams[1], meta


def snapshot_current(config_path: str) -> dict:
    """CLI entry (make topo-graph): census the CURRENT symbolic graph."""
    cfg = load_config(config_path)
    triplets_path = repo_path(cfg, "triplets", "triplets.jsonl")
    if not triplets_path.is_file():
        raise SystemExit(f"missing {triplets_path}: run the extraction first")
    triplets = load_triplets(triplets_path)
    graph = build_isa_graph(
        triplets,
        topk=int(cfg_get(cfg, "topo.graph_topk", 800)),
        conf_min=float(cfg_get(cfg, "topo.graph_conf_min", 0.6)),
        relation=str(cfg_get(cfg, "topo.isa_relation", "is_a")),
    )
    count, details = graph_cycle_count(
        graph,
        max_cycle_len=int(cfg_get(cfg, "topo.max_cycle_len", 5)),
        max_cycles=int(cfg_get(cfg, "topo.max_cycles", 10000)),
    )
    out_dir = repo_path(cfg, "artifacts", "topo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph_current.json"
    out_path.write_text(json.dumps(
        {"count": count, "n_entities": len(set(graph) | {
            v for t in graph.values() for v in t}),
         "truncated": details["truncated"], "cycles": details["cycles"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"R3a census: {count:.0f} directed cycles "
          f"({details['truncated'] and 'truncated' or 'complete'}) "
          f"-> {out_path}")
    return {"count": count, "details": details}
