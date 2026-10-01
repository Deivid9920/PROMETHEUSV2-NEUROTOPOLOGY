"""Persistence engine over ripser (Fase 1-V2).

compute_diagrams(points, maxdim=1, subsample=None, seed=42,
distance_matrix=None, geometry=None) -> list[np.ndarray] of diagrams,
with deterministic subsampling and distance-matrix input for graph
filtrations (points=None). The FROZEN geometry policy
(config.topo.geometry: PCA n_components or cosine, plus normalization)
is applied exactly once, HERE, identically for every round: tests pin
the raw engine to textbook results (square -> H1 persistence sqrt(2)-1;
two clusters -> betti0=2; single point -> one essential class), while
the round path passes the frozen policy explicitly.

betti0_at_eps and persistent_h1_count DELEGATE to promotion_topology
(the anchor owns the definitions). Diagram caches are keyed by the
sha256 of the checkpoint's state_dict TENSOR BYTES in canonical order
(never the .pt file: optimizer state changes on re-save).
"""

from __future__ import annotations

import numpy as np
from ripser import ripser

PERSISTENCE_IMPLEMENTED = True


def apply_geometry(points: np.ndarray, geometry: dict) -> np.ndarray:
    """Frozen geometry policy (A1/A2), applied identically every round.

    normalize=True: center the cloud and scale by the MEAN NORM, which
    neutralizes the residual-stream norm growth that would otherwise
    move betti0@eps without any structural change. method='pca':
    deterministic SVD projection to n_components in [20, 50] (sign-fixed
    for reproducibility). method='cosine': the caller receives unit-row
    points and cosine DISTANCES (chord-equivalent) via the distance
    path of compute_diagrams.
    """
    pts = np.asarray(points, dtype=np.float64)
    if geometry.get("normalize", True):
        centre = pts.mean(axis=0, keepdims=True)
        pts = pts - centre
        norms = np.linalg.norm(pts, axis=1)
        scale = float(norms.mean())
        if scale > 0.0:
            pts = pts / scale
    if geometry.get("method") == "pca":
        k = int(geometry.get("n_components", 32))
        k = min(k, pts.shape[1], pts.shape[0])
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        components = vt[:k]
        # deterministic sign: the largest-|component| entry is positive
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row[:] = -row
        return pts @ components.T
    return pts


def _cosine_distance_matrix(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = points / norms
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = 1.0 - cos
    np.fill_diagonal(distance, 0.0)
    return distance


def deterministic_subsample(points: np.ndarray, subsample: int | None,
                            seed: int) -> np.ndarray:
    """Deterministic subsample: same seed -> the same subset, always."""
    n = len(points)
    if subsample is None or subsample >= n:
        return points
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=int(subsample), replace=False))
    return points[idx]


def compute_diagrams(points=None, maxdim: int = 1, subsample: int | None = None,
                     seed: int = 42, distance_matrix=None,
                     geometry: dict | None = None) -> list[np.ndarray]:
    """Persistence diagrams [H0, H1, ...] as (n_i, 2) (birth, death) arrays.

    points path: geometry policy (when given) then deterministic
    subsampling then ripser on Euclidean distances.
    distance_matrix path (points=None): optional subsample of the matrix
    indices, then ripser with distance_matrix=True (R3b graph filtration).
    """
    if distance_matrix is not None:
        matrix = np.asarray(distance_matrix, dtype=np.float64)
        if subsample is not None and subsample < len(matrix):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(matrix), size=int(subsample),
                                     replace=False))
            matrix = matrix[np.ix_(idx, idx)]
        result = ripser(matrix, maxdim=maxdim, distance_matrix=True)
        return [np.asarray(d) for d in result["dgms"][:maxdim + 1]]
    if points is None:
        raise ValueError("compute_diagrams needs points or distance_matrix")
    pts = np.asarray(points, dtype=np.float64)
    if geometry:
        pts = apply_geometry(pts, geometry)
        if geometry.get("method") == "cosine" and len(pts) > 0:
            pts = deterministic_subsample(pts, subsample, seed)
            matrix = _cosine_distance_matrix(pts)
            result = ripser(matrix, maxdim=maxdim, distance_matrix=True)
            return [np.asarray(d) for d in result["dgms"][:maxdim + 1]]
    pts = deterministic_subsample(pts, subsample, seed)
    result = ripser(pts, maxdim=maxdim)
    return [np.asarray(d) for d in result["dgms"][:maxdim + 1]]


def betti0_at_eps(h0_diagram, eps: float) -> int:
    """DELEGATES to the anchor: definitions live there and only there."""
    from prometheus_ns.topo.promotion_topology import betti0_at_eps as _f
    return _f(h0_diagram, eps)


def persistent_h1_count(h1_diagram, min_persistence: float) -> int:
    """DELEGATES to the anchor: definitions live there and only there."""
    from prometheus_ns.topo.promotion_topology import (
        persistent_h1_count as _f)
    return _f(h1_diagram, min_persistence)


def save_diagram_cache(path, tensor_sha256: str, geometry: dict,
                       per_layer: dict) -> None:
    """Round cache keyed by the state_dict TENSOR sha256 (C5)."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tensor_sha256": tensor_sha256,
               "geometry": geometry, "layers": {}}
    for layer, data in per_layer.items():
        payload["layers"][layer] = {
            "h0": np.asarray(data["h0"], dtype=np.float64),
            "h1": np.asarray(data["h1"], dtype=np.float64),
            "betti0": data["betti0"],
        }
    np.savez_compressed(path,
                        meta=json.dumps(payload),
                        **{f"layer_{layer}_h0": data["h0"]
                           for layer, data in per_layer.items()},
                        **{f"layer_{layer}_h1": data["h1"]
                           for layer, data in per_layer.items()})


def load_diagram_cache(path) -> dict | None:
    """Return the cached payload only when the tensor hash matches."""
    import json
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        payload = json.loads(str(data["meta"]))
        layers = {}
        for key in data.files:
            if key.endswith("_h0"):
                layer = key[len("layer_"):-len("_h0")]
                layers[layer] = {
                    "h0": data[key],
                    "h1": data[f"layer_{layer}_h1"],
                    "betti0": payload["layers"][layer]["betti0"],
                }
        return {"tensor_sha256": payload["tensor_sha256"],
                "geometry": payload["geometry"], "layers": layers}
    return None
