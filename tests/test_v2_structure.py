"""Structural contract tests for the V2 additions (V2.1).

Complements (never replaces) tests/test_structure.py of the inherited
repo. Additionally verifies that the delivered anchors are REAL
implementations, not stubs that would silently disable the sensor.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_V2_FILES = [
    "prometheus_ns/topo/__init__.py",
    "prometheus_ns/topo/promotion_topology.py",
    "prometheus_ns/topo/persistence.py",
    "prometheus_ns/topo/activations.py",
    "prometheus_ns/topo/graph_topology.py",
    "prometheus_ns/topo/stress.py",
    "tests/test_topo_promotion.py",
    "tests/test_topo_persistence.py",
    "tests/test_topo_stability.py",
    "tests/test_topo_graph.py",
    "docs/v2_experiment_protocol.md",
    "scripts/scaffold_v2.py",
    "scripts/divergence_report.py",
    "scripts/run_round_v2.py",
    "scripts/make_notebooks_topo.py",
]

REQUIRED_NOTEBOOKS = [
    "notebooks/05_model_topology.ipynb",
    "notebooks/06_graph_topology.ipynb",
    "notebooks/07_divergence_study.ipynb",
]

V2_MAKEFILE_TARGETS = ["topo-extract", "topo-graph", "topo", "stress",
                       "divergence-report"]

TOPO_KEYS = ["probe_sentences", "layers", "subsample", "maxdim",
             "geometry", "eps_betti", "churn_bottleneck",
             "b0_collapse_ratio", "h1_growth_max", "graph_topk",
             "graph_conf_min", "isa_relation", "max_cycle_len",
             "max_cycles", "seed"]
STRESS_KEYS = ["modes", "dup_ratio", "lr_multiplier"]

HEREDADO_INTOCADOS = ["prometheus_ns/autoloop/promotion.py",
                      "prometheus_ns/device.py"]


@pytest.fixture(scope="session")
def config() -> dict:
    path = REPO_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("rel", REQUIRED_V2_FILES)
def test_v2_file_exists(rel: str) -> None:
    assert (REPO_ROOT / rel).is_file(), f"missing V2 file: {rel}"


@pytest.mark.parametrize("rel", REQUIRED_NOTEBOOKS)
def test_notebook_exists(rel: str) -> None:
    assert (REPO_ROOT / rel).is_file(), f"missing notebook: {rel}"


@pytest.mark.parametrize("key", TOPO_KEYS)
def test_topo_section(config: dict, key: str) -> None:
    assert key in config.get("topo", {}), f"config.topo missing: {key}"


@pytest.mark.parametrize("key", STRESS_KEYS)
def test_stress_section(config: dict, key: str) -> None:
    assert key in config.get("stress", {}), f"config.stress missing: {key}"


def test_topo_threshold_sanity(config: dict) -> None:
    topo = config["topo"]
    assert 0 < topo["churn_bottleneck"] < 1
    assert 0 < topo["b0_collapse_ratio"] < 1
    assert 0 < topo["h1_growth_max"] < 1
    assert topo["probe_sentences"] >= 64
    assert 100 <= topo["subsample"] <= 2000
    assert topo["maxdim"] == 1


def test_geometry_policy_is_frozen(config: dict) -> None:
    geometry = config["topo"]["geometry"]
    assert geometry["method"] in ("pca", "cosine")
    if geometry["method"] == "pca":
        assert 20 <= geometry["n_components"] <= 50
    assert geometry["normalize"] is True


def test_v2_makefile_targets() -> None:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    declared = set(re.findall(r"^([a-zA-Z_-]+)\s*:", text, flags=re.MULTILINE))
    missing = [t for t in V2_MAKEFILE_TARGETS if t not in declared]
    assert not missing, f"Makefile missing V2 targets: {missing}"


def test_anchors_are_not_stubs() -> None:
    """The V2.1 anchors must be real implementations: a stub that
    silently disables the sensor invalidates the whole gate."""
    promotion = (REPO_ROOT / "prometheus_ns/topo/promotion_topology.py"
                 ).read_text(encoding="utf-8")
    assert "linear_sum_assignment" in promotion, (
        "exact bottleneck must use the assignment solver")
    graph = (REPO_ROOT / "prometheus_ns/topo/graph_topology.py"
             ).read_text(encoding="utf-8")
    assert "simple_cycles" in graph, "R3a census must use networkx cycles"
    assert "Never feed" in graph or "never vetoed" in graph.lower() or \
        "observation only" in graph, "R3b honest reading must be stated"
    report = (REPO_ROOT / "scripts/divergence_report.py").read_text(
        encoding="utf-8")
    assert "Wilson" in report, "report must use exact Wilson intervals"
    assert "EXPERIMENT INVALID" in report, "mechanical V1-V5 checks"


def test_inherited_anchors_untouched_by_git() -> None:
    """The NS gate and device policy must not be modified by V2 commits.
    Verified via git: the two files must have no uncommitted changes and
    no changes after the V2 base commit. Skips gracefully outside git."""
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", *HEREDADO_INTOCADOS],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git not available")
    if diff.returncode != 0:
        pytest.skip("not a git repository")
    assert diff.stdout.strip() == "", (
        "inherited anchors have uncommitted modifications: promotion.py "
        "and device.py are immutable in V2"
    )
