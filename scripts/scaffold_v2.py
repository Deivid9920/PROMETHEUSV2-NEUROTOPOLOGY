#!/usr/bin/env python3
"""Create the PROMETHEUS-V2 skeleton over the inherited NS repository.

V2.1: graph_topology.py and divergence_report.py are delivered as
IMPLEMENTED anchors (not stubs) and are only reported. Stub contracts
carry the corrected requirements: geometry policy frozen (A1/A2),
probe sentences from data_clean excluding the holdout (H3), bit-exact
extraction determinism (C1), tensor-sha256 cache key (C5),
treatment_sha256 manifests (D2).

Usage:
    python3 scripts/scaffold_v2.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUBS: dict[str, str] = {
    "prometheus_ns/topo/persistence.py": (
        "Persistence engine over ripser. Contract: compute_diagrams(points"
        ", maxdim=1, subsample=None, seed=42, distance_matrix=None) -> "
        "list[np.ndarray] of diagrams, with deterministic subsampling and "
        "distance-matrix input for graph filtrations (points=None). "
        "Applies the FROZEN geometry policy (config.topo.geometry: PCA "
        "n_components or cosine, plus normalization) exactly once, here, "
        "identically for every round. betti0_at_eps and "
        "persistent_h1_count DELEGATE to promotion_topology (the anchor "
        "owns the definitions). Round cache artifacts/topo/round_{N}.npz "
        "keyed by the sha256 of the state_dict TENSOR BYTES in canonical "
        "order (never the .pt file: optimizer state changes on re-save). "
        "Anchors tests/test_topo_persistence.py pin the numerics: square "
        "-> H1 persistence sqrt(2)-1; two clusters -> betti0=2 at mid "
        "scale; determinism under equal seeds."
    ),
    "prometheus_ns/topo/activations.py": (
        "Probe-activation extraction. Contract: "
        "create_probe_sentences(cfg) -- ONE-TIME: sample config.topo."
        "probe_sentences sentences from data_clean/ EXCLUDING data_clean/"
        "holdout/ (the holdout stays blind to topology), dropping any "
        "longer than model max_seq with the discarded count recorded, "
        "writing data/processed/probe_sentences.json with sha256. "
        "load_probe_sentences(cfg) fail-loud if missing. "
        "extract_probe_activations(cfg, checkpoint_path) -> {layer: "
        "np.ndarray [subsample, d_model]}: model.eval() + torch.no_grad(), "
        "fp32, forward hooks at config.topo.layers, LAST REAL token "
        "vector (lengths tracked, fail-loud on overlong input), batch 32, "
        "deterministic subsample with config.topo.seed. Determinism "
        "contract: two full extractions of the same checkpoint are BIT "
        "IDENTICAL (np.array_equal, no tolerance). "
        "tensor_sha256(checkpoint_path) -> cache key over state_dict "
        "tensor bytes. self_noise_report(cfg, checkpoint_path, n_seeds=5) "
        "-> writes artifacts/topo/self_noise.json {seeds, layers: {layer: "
        "{pairwise 5x5}}, p95_overall} -- the A3 calibration gate consumed "
        "by tests/test_topo_stability.py."
    ),
    "prometheus_ns/topo/stress.py": (
        "Deliberate-degradation treatments (Fase 4-V2). Contract: "
        "build_treatment(cfg, mode, round_id) -> manifest with "
        "treatment_sha256 (sha256 over corpus bytes + triplets file + lr "
        "manifest, D2) and the degraded material: dup_flood duplicates "
        "stress.dup_ratio of the packed corpus with near-duplicate noise; "
        "contradiction_flood plants is_a contradictions in the triplets "
        "targeting R3a; lr_spike multiplies the lr by stress.lr_multiplier "
        "and shortens the corpus 10x. All treatments are deterministic "
        "under config.topo.seed, write under data_stress/, and never "
        "touch champion.pt (D1: stress trains on stress_candidate.pt)."
    ),
}

ANCHORS = [
    "prometheus_ns/topo/promotion_topology.py",
    "prometheus_ns/topo/graph_topology.py",
    "tests/test_topo_promotion.py",
    "tests/test_topo_persistence.py",
    "tests/test_topo_stability.py",
    "tests/test_topo_graph.py",
    "tests/test_v2_structure.py",
    "docs/v2_experiment_protocol.md",
    "scripts/divergence_report.py",
    "scripts/make_notebooks_topo.py",
    "scripts/run_round_v2.py",
]

CLI_MAIN = """

if __name__ == "__main__":
    pass
"""


def main() -> None:
    created: list[str] = []
    kept: list[str] = []

    for rel in ["prometheus_ns/topo", "artifacts/topo", "data_stress",
                "notebooks"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
        keep = ROOT / rel / ".gitkeep"
        if not keep.exists() and rel != "prometheus_ns/topo":
            keep.write_text("", encoding="utf-8")
            created.append(str(keep))

    init = ROOT / "prometheus_ns/topo/__init__.py"
    if not init.exists():
        init.write_text(
            '"""Neurotopology layer: sensors with veto power."""\n',
            encoding="utf-8")
        created.append("prometheus_ns/topo/__init__.py")

    for rel, contract in STUBS.items():
        target = ROOT / rel
        if target.exists():
            kept.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'"""{contract}"""\n', encoding="utf-8")
        created.append(rel)

    print("created:")
    for rel in created:
        print(f"  + {rel}")
    if kept:
        print("kept existing:")
        for rel in kept:
            print(f"  = {rel}")

    print("anchors:")
    for rel in ANCHORS:
        ok = (ROOT / rel).is_file()
        state = "present" if ok else "MISSING: copy it from the spec package"
        print(f"  {'ok' if ok else '!!'} {rel}: {state}")

    print("\nnext: apply config/Makefile/requirements diffs, then "
          "make setup && make test")


if __name__ == "__main__":
    main()
