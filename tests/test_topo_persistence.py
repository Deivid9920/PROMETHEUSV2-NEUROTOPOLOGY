"""Persistence-engine anchors against known topological facts.

These tests pin the ripser wrapper (topo/persistence.py) to textbook
results. They are the numerical certification that the engine measures
what topology says it should; a failing test here invalidates every
downstream gate decision. Values derived analytically:

- Square of side 1 (4 corners): exactly one H1 class, born when the
  cycle closes at radius 1, dying at the circumradius sqrt(2)/2...
  using the standard Vietoris-Rips on Euclidean distances: born at 1,
  dies at sqrt(2) -> persistence sqrt(2) - 1.
- Two tight clusters far apart: at scale eps between cluster diameter
  and separation, betti0 = 2.
- Single point: H0 has one essential class (infinite death).
"""

from __future__ import annotations

import numpy as np
import pytest

from prometheus_ns.topo.persistence import (
    compute_diagrams, betti0_at_eps, persistent_h1_count,
)


@pytest.fixture(autouse=True)
def _require_persistence_engine():
    """Fase 0-V2: the wrapper is a contracted stub until Fase 1-V2;
    these anchors activate the moment the engine lands."""
    from prometheus_ns.topo import persistence as _persistence
    if not getattr(_persistence, "PERSISTENCE_IMPLEMENTED", False):
        pytest.skip("persistence engine pending (Fase 1-V2)")


def test_square_h1_persistence_matches_theory() -> None:
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diagrams = compute_diagrams(corners, maxdim=1)
    h1 = diagrams[1]
    # exactly one essential-cycle class with the analytic persistence
    persistent = h1[h1[:, 1] - h1[:, 0] > 0.1]
    assert len(persistent) == 1
    assert persistent[0, 1] - persistent[0, 0] == pytest.approx(
        np.sqrt(2) - 1, abs=1e-6)


def test_two_clusters_betti0() -> None:
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(0.0, 0.05, size=(50, 2))
    cluster_b = rng.normal(10.0, 0.05, size=(50, 2))
    points = np.vstack([cluster_a, cluster_b])
    diagrams = compute_diagrams(points, maxdim=1)
    assert betti0_at_eps(diagrams[0], eps=1.0) == 2
    assert betti0_at_eps(diagrams[0], eps=11.0) == 1


def test_single_point_has_one_essential_h0_class() -> None:
    diagrams = compute_diagrams(np.array([[0.0, 0.0]]), maxdim=0)
    h0 = diagrams[0]
    assert (h0[:, 1] == np.inf).sum() == 1


def test_persistent_h1_count_threshold() -> None:
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diagrams = compute_diagrams(corners, maxdim=1)
    assert persistent_h1_count(diagrams[1], min_persistence=0.1) == 1
    assert persistent_h1_count(diagrams[1],
                               min_persistence=np.sqrt(2)) == 0


def test_subsample_determinism() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(1000, 3))
    d1 = compute_diagrams(points, maxdim=1, subsample=500, seed=42)
    d2 = compute_diagrams(points, maxdim=1, subsample=500, seed=42)
    assert np.array_equal(d1[0], d2[0])
    assert np.array_equal(d1[1], d2[1])
