"""Sensor stability anchors (risk A3). Immutable anchor (V2.1).

A veto threshold below the subsampling noise floor vetoes every round
and kills the loop; one above the real signal vetoes nothing. These
tests pin both ends:

- Mechanics tests (always run, synthetic clouds): same seed reproduces
  the same subset and identical diagrams; two independent subsamples of
  the SAME cloud move the diagram less than a genuinely different cloud
  does (the sensor must be quieter than its signal).
- Calibration-gate test (activates when artifacts/topo/self_noise.json
  exists, produced by the Fase 1-V2 verification on the real champion):
  recomputes the reported p95 from the stored pairwise matrix and
  enforces p95 < config.topo.churn_bottleneck. Until that artifact
  exists the sensor is NOT calibrated and stress rounds must not start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from prometheus_ns.topo.persistence import compute_diagrams
from prometheus_ns.topo.promotion_topology import bottleneck_distance

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _require_persistence_engine():
    """Fase 0-V2: mechanics tests need the wrapper (Fase 1-V2); the
    calibration-gate test activates with artifacts/topo/self_noise.json."""
    from prometheus_ns.topo import persistence as _persistence
    if not getattr(_persistence, "PERSISTENCE_IMPLEMENTED", False):
        pytest.skip("persistence engine pending (Fase 1-V2)")


def test_same_seed_same_subset_identical_diagrams() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, size=(1200, 8))
    d1 = compute_diagrams(points, maxdim=0, subsample=500, seed=42)
    d2 = compute_diagrams(points, maxdim=0, subsample=500, seed=42)
    assert np.array_equal(d1[0], d2[0])


def test_self_noise_quieter_than_signal() -> None:
    rng = np.random.default_rng(1)
    cloud = rng.normal(0.0, 1.0, size=(2000, 6))
    shifted = cloud + np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d_a = compute_diagrams(cloud, maxdim=0, subsample=500, seed=1)
    d_b = compute_diagrams(cloud, maxdim=0, subsample=500, seed=2)
    d_c = compute_diagrams(shifted, maxdim=0, subsample=500, seed=3)
    self_noise = bottleneck_distance(d_a[0], d_b[0])
    cross = bottleneck_distance(d_a[0], d_c[0])
    assert self_noise > 0.0          # subsampling does move the diagram
    assert self_noise < cross        # but far less than a real change


def test_self_noise_artifact_calibration_gate() -> None:
    """Fase 1-V2 gate: p95 of the self-bottleneck distribution must sit
    below the churn threshold. Skips until the artifact exists."""
    artifact = REPO_ROOT / "artifacts/topo/self_noise.json"
    if not artifact.is_file():
        pytest.skip("self_noise.json missing: run the Fase 1-V2 "
                    "stability verification on the real champion first")
    data = __import__("json").loads(artifact.read_text(encoding="utf-8"))
    seeds = data["seeds"]
    assert len(seeds) >= 5, "stability needs at least 5 subsample seeds"
    values = []
    for layer, block in data["layers"].items():
        matrix = block["pairwise"]
        assert len(matrix) == len(seeds)
        for i in range(len(seeds)):
            assert len(matrix[i]) == len(seeds)
            for j in range(i + 1, len(seeds)):
                values.append(matrix[i][j])
    recomputed_p95 = float(np.percentile(values, 95))
    assert recomputed_p95 == pytest.approx(data["p95_overall"], abs=1e-9)
    with (REPO_ROOT / "config.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    threshold = float(cfg["topo"]["churn_bottleneck"])
    assert recomputed_p95 < threshold, (
        f"sensor noise floor p95={recomputed_p95:.4f} >= churn threshold "
        f"{threshold}: recalibrate BEFORE round N1 (legitimate N0 "
        "calibration), never after seeing stress results"
    )
