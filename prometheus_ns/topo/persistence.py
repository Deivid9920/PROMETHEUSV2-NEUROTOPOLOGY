"""Persistence engine over ripser. Contract (stub until Fase 1-V2).

compute_diagrams(points, maxdim=1, subsample=None, seed=42,
distance_matrix=None) -> list[np.ndarray] of diagrams, with deterministic
subsampling and distance-matrix input for graph filtrations (points=None).
Applies the FROZEN geometry policy (config.topo.geometry: PCA
n_components or cosine, plus normalization) exactly once, here,
identically for every round. betti0_at_eps and persistent_h1_count
DELEGATE to promotion_topology (the anchor owns the definitions). Round
cache artifacts/topo/round_{N}.npz keyed by the sha256 of the
state_dict TENSOR BYTES in canonical order (never the .pt file: optimizer
state changes on re-save). Anchors tests/test_topo_persistence.py pin
the numerics: square -> H1 persistence sqrt(2)-1; two clusters ->
betti0=2 at mid scale; determinism under equal seeds.
"""

PERSISTENCE_IMPLEMENTED = False


def compute_diagrams(*args, **kwargs):
    raise NotImplementedError(
        "persistence engine pending (Fase 1-V2): wrapper over ripser with "
        "the frozen geometry policy, deterministic subsampling and "
        "distance-matrix input")


def betti0_at_eps(*args, **kwargs):
    from prometheus_ns.topo.promotion_topology import betti0_at_eps as _f
    return _f(*args, **kwargs)


def persistent_h1_count(*args, **kwargs):
    from prometheus_ns.topo.promotion_topology import (
        persistent_h1_count as _f)
    return _f(*args, **kwargs)
