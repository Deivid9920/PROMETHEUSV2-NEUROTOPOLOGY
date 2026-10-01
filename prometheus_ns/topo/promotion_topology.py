"""Topological promotion veto for PROMETHEUS-V2. Immutable anchor (V2.1).

The V2 promotion gate is the conjunction of the inherited NS gate
(autoloop.promotion.decide_promotion, untouched) and the negation of the
veto computed here. This module receives ALREADY-COMPUTED measurements
(persistence diagrams as (n, 2) arrays of (birth, death), cycle counts)
and applies three closed rules:

R1 structural churn. The bottleneck distance between the candidate's and
the champion's model diagrams (max over H0 and H1) exceeds
churn_bottleneck while the relative perplexity gain is below
churn_gain_excuse: a large structural shift is only acceptable when the
functional gain is large.

R2 representational collapse. betti0_at_eps of the candidate falls below
b0_collapse_ratio times baseline_b0, which the CALLER supplies as the
RUNNING MINIMUM of betti0@eps across rounds: collapse means falling under
the best structure ever observed, so legitimate structural improvement
never creates a false floor. Fewer distinct representational clusters at
a fixed scale is the geometric signature of model collapse, visible
before perplexity notices.

R3 symbolic cycle growth (V2.1). graph_h1_count carries the R3a census —
directed simple cycles of the is_a graph with every edge above the
confidence floor, computed by topo/graph_topology.py. It vetoes when it
exceeds (1 + h1_growth_max) times the champion's count. The R3b Rips-H1
of the adjacency filtration is logged for observation but NEVER fed here.

Bottleneck distance (exact). The naive reduction to a sum-minimizing
assignment (scipy linear_sum_assignment) is WRONG for this metric:
bottleneck minimizes the MAXIMUM matched cost. This implementation is
exact: candidates are the pairwise L-infinity costs and the
half-persistences (diagonal fallback prices); feasibility of a radius
delta — every point whose half-persistence exceeds delta must be matched
to an opposite point within delta, with disjoint matchings — is decided
by a maximum bipartite matching (scipy.sparse.csgraph, Hopcroft-Karp in
C) under binary search over the sorted candidates.
tests/test_topo_promotion.py contains fixtures on which the
sum-minimizing implementation returns a strictly smaller, wrong value.

In round 1 (no champion measurements) the veto cannot fire: the
measurement ESTABLISHES the baseline instead. A veto never aborts a
round — it only blocks promotion; the loop continues with the previous
champion intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TopoRules:
    """Thresholds mirrored from config.yaml -> topo."""

    churn_bottleneck: float        # R1: distance that counts as churn
    churn_gain_excuse: float       # R1: relative ppl gain that excuses churn
    b0_collapse_ratio: float       # R2: floor as fraction of running-min betti0
    h1_growth_max: float           # R3: allowed relative growth of R3a cycles
    eps: float                     # fixed scale for betti0_at_eps
    h1_min_persistence: float      # reserved for R3b reporting, not the veto


@dataclass(frozen=True)
class TopoMetrics:
    """Measurements of the candidate round and of the reigning champion."""

    round_id: int
    model_h0: np.ndarray | None              # candidate, concatenated layers
    model_h1: np.ndarray | None
    graph_h1_count: float | None             # R3a directed-cycle census
    champion_model_h0: np.ndarray | None     # None in round 1
    champion_model_h1: np.ndarray | None
    champion_graph_h1_count: float | None
    baseline_b0: float | None                # running minimum of betti0@eps
    relative_ppl_gain: float = 0.0


@dataclass(frozen=True)
class TopoDecision:
    """Outcome of the topological veto for one round."""

    veto: bool
    establish_baseline: bool
    causes: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def _as_diagram(diag: np.ndarray | None) -> np.ndarray:
    if diag is None:
        return np.empty((0, 2))
    arr = np.asarray(diag, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2))
    return arr.reshape(-1, 2)


def _half_persistence(diag: np.ndarray) -> np.ndarray:
    """Diagonal fallback price of each point: (death - birth) / 2."""
    if len(diag) == 0:
        return np.empty(0)
    return (diag[:, 1] - diag[:, 0]) / 2.0


def _pairwise_linf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise L-infinity distances between two (n, 2) / (m, 2) arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    return np.maximum(
        np.abs(a[:, None, 0] - b[None, :, 0]),
        np.abs(a[:, None, 1] - b[None, :, 1]),
    )


def _feasible(costs: np.ndarray, hp_a: np.ndarray, hp_b: np.ndarray,
              delta: float) -> bool:
    """True iff a matching with every cost <= delta covers all points
    whose half-persistence exceeds delta (the rest may fall to diagonals).

    Exact reduction to a rectangular assignment (Jonker-Volgenant).
    Theorem used: a matching covering the required set exists iff SOME
    maximum matching covers it (any matching extends to a maximum one by
    augmenting paths, which never unmatch a vertex). The assignment
    therefore penalizes disallowed pairs with BIG (primary objective:
    maximum cardinality) and rewards pairs that cover a required vertex
    with -1 (secondary objective: maximum coverage). Feasible iff the
    optimal assignment's required coverage equals |required|.
    """
    forced_a = hp_a > delta
    forced_b = hp_b > delta
    if not forced_a.any() and not forced_b.any():
        return True
    n, m = costs.shape
    if n == 0 or m == 0:
        return False
    allowed = costs <= delta
    # Reward = number of required ENDPOINTS the pair covers (0, 1 or 2)
    covers = forced_a[:, None].astype(float) + forced_b[None, :].astype(float)
    big = n * m + int(forced_a.sum()) + int(forced_b.sum()) + 1
    score = np.where(allowed, -covers, float(big))
    row_ind, col_ind = linear_sum_assignment(score)
    total = score[row_ind, col_ind].sum()
    used_big = int(np.sum(~allowed[row_ind, col_ind]))
    coverage = -int(round(total - big * used_big))
    return coverage == int(forced_a.sum()) + int(forced_b.sum())


def bottleneck_distance(diag1: np.ndarray | None,
                        diag2: np.ndarray | None) -> float:
    """Exact bottleneck distance between two persistence diagrams.

    Minimizes the MAXIMUM matched cost, where each point may either be
    matched to an opposite point (L-infinity cost) or fall to its own
    diagonal (cost = half-persistence). Binary search over the sorted
    candidate radii; feasibility decided by maximum bipartite matching.
    """
    d1 = _as_diagram(diag1)
    d2 = _as_diagram(diag2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        return float(_half_persistence(d2).max())
    if len(d2) == 0:
        return float(_half_persistence(d1).max())
    hp_a = _half_persistence(d1)
    hp_b = _half_persistence(d2)
    costs = _pairwise_linf(d1, d2)
    candidates = np.unique(np.concatenate([costs.ravel(), hp_a, hp_b]))
    lo, hi = 0, len(candidates) - 1
    best = float(candidates[hi])
    while lo <= hi:
        mid = (lo + hi) // 2
        if _feasible(costs, hp_a, hp_b, float(candidates[mid])):
            best = float(candidates[mid])
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def betti0_at_eps(h0_diagram: np.ndarray, eps: float) -> int:
    """Number of H0 classes alive at scale eps: birth <= eps < death."""
    h0 = _as_diagram(h0_diagram)
    if len(h0) == 0:
        return 0
    return int(np.sum((h0[:, 0] <= eps) & (h0[:, 1] > eps)))


def persistent_h1_count(h1_diagram: np.ndarray, min_persistence: float) -> int:
    """Number of H1 classes whose persistence strictly exceeds the floor."""
    h1 = _as_diagram(h1_diagram)
    if len(h1) == 0:
        return 0
    persistence = h1[:, 1] - h1[:, 0]
    return int(np.sum(persistence > min_persistence))


def decide_topo_veto(metrics: TopoMetrics, rules: TopoRules) -> TopoDecision:
    """Apply rules R1/R2/R3 to already-computed measurements.

    Round 1 (no champion measurements) never vetoes: the measurement
    establishes the baseline. The returned checks carry the full numeric
    evidence for logs/topo.jsonl.
    """
    causes: list[str] = []
    checks: dict = {}

    if metrics.round_id == 1 or metrics.champion_model_h0 is None:
        return TopoDecision(
            veto=False,
            establish_baseline=True,
            causes=["first round: measurements establish the baseline"],
            checks=checks,
        )

    b0_h0 = bottleneck_distance(metrics.champion_model_h0, metrics.model_h0)
    b1_h1 = bottleneck_distance(metrics.champion_model_h1, metrics.model_h1)
    observed = max(b0_h0, b1_h1)
    churn = observed > rules.churn_bottleneck
    excused = metrics.relative_ppl_gain >= rules.churn_gain_excuse
    checks["R1_churn"] = {
        "bottleneck_h0": b0_h0, "bottleneck_h1": b1_h1,
        "observed": observed, "threshold": rules.churn_bottleneck,
        "relative_ppl_gain": metrics.relative_ppl_gain,
        "excuse": rules.churn_gain_excuse, "churn": churn, "excused": excused,
    }
    if churn and not excused:
        causes.append(
            f"R1 structural churn: bottleneck {observed:.4f} > "
            f"{rules.churn_bottleneck} with relative ppl gain "
            f"{metrics.relative_ppl_gain:.4f} < {rules.churn_gain_excuse}"
        )

    b0_current = betti0_at_eps(metrics.model_h0, rules.eps)
    if metrics.baseline_b0 is None or metrics.baseline_b0 <= 0:
        collapse = False
        checks["R2_collapse"] = {"b0_current": b0_current,
                                 "baseline": metrics.baseline_b0,
                                 "note": "no usable baseline; rule skipped"}
    else:
        floor = rules.b0_collapse_ratio * metrics.baseline_b0
        collapse = b0_current < floor
        checks["R2_collapse"] = {
            "b0_current": b0_current, "floor": floor,
            "ratio": rules.b0_collapse_ratio,
            "baseline_b0": metrics.baseline_b0, "collapse": collapse,
        }
        if collapse:
            causes.append(
                f"R2 representational collapse: betti0@eps={b0_current} < "
                f"floor {floor:.2f} (baseline {metrics.baseline_b0})"
            )

    if (metrics.champion_graph_h1_count is None
            or metrics.graph_h1_count is None):
        graph = False
        checks["R3_graph"] = {"note": "graph measurements missing; skipped"}
    else:
        ceiling = (1.0 + rules.h1_growth_max) * metrics.champion_graph_h1_count
        graph = metrics.graph_h1_count > ceiling
        checks["R3_graph"] = {
            "candidate": metrics.graph_h1_count,
            "champion": metrics.champion_graph_h1_count,
            "ceiling": ceiling, "growth_max": rules.h1_growth_max,
            "growth": graph,
        }
        if graph:
            causes.append(
                f"R3 symbolic cycle growth: persistent H1 "
                f"{metrics.graph_h1_count} > ceiling {ceiling:.2f} "
                f"(champion {metrics.champion_graph_h1_count})"
            )

    return TopoDecision(veto=bool(causes), establish_baseline=False,
                        causes=causes, checks=checks)
