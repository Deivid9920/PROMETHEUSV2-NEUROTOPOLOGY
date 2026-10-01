#!/usr/bin/env python3
"""PROMETHEUS-V2 round runner (the Makefile V2 contract).

Normal round : run_round_v2.py --config config.yaml --round-id N
Topo only    : run_round_v2.py --config config.yaml --topo-only
Stress round : run_round_v2.py --config config.yaml --stress MODE --round-id N

Round flow (V2.1, after clean/quality/triplets as NS):
  1. continued pretraining -> candidate (stress: on stress_candidate.pt, D1)
  2. eval NS (ppl on the frozen holdout, digest check, diversity) - intact
  3. topo-extract: probe activations of candidate AND champion ->
     diagrams per layer under the FROZEN geometry policy; bottleneck
     distances via the exact anchor
  4. topo-graph: R3a directed-cycle census + R3b Rips-H1 (observation)
  5. topo_veto = decide_topo_veto(metrics, rules)   [ANCLA, no editar]
     with baseline_b0 = running minimum (D3) and graph_h1_count = R3a
  6. FAIL-CLOSED (1.6): round >= 2 without a topo measurement -> error
  7. PROMOTION = decide_promotion(NS).promote AND NOT topo_veto.veto
  8. logs/topo.jsonl (config_sha256 + treatment_sha256 + evidence),
     logs/promotion_decisions.jsonl (structured), promotion_log.md

Corpus policy (documented in docs/v2_experiment_protocol.md): the web
crawler is NOT part of V2 rounds. The seeded crawler re-fetches identical
URLs every round and would mutate the corpus, breaking the mechanical
treatment_sha256 comparability the divergence report enforces (V2).
clean/extract run deterministically over the frozen data_clean snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.autoloop.diversity import measure_and_record
from prometheus_ns.autoloop.promotion import (
    RoundMetrics, compute_holdout_digest, decide_promotion,
    rollback_to_champion, write_promotion_log,
)
from prometheus_ns.data.cleaner import clean_corpus
from prometheus_ns.symbolic.consistency import run_consistency
from prometheus_ns.symbolic.extractor import extract_corpus
from prometheus_ns.topo.graph_topology import (
    adjacency_distance_matrix, build_isa_graph, graph_cycle_count,
    load_triplets, rips_h1_persistence, triangle_census,
)
from prometheus_ns.topo.promotion_topology import (
    TopoMetrics, TopoRules, bottleneck_distance, betti0_at_eps,
    decide_topo_veto, persistent_h1_count,
)
from prometheus_ns.train.dataset import build_packed_dataset
from prometheus_ns.train.trainer import promote_checkpoint, train


def verify_holdout(cfg: dict) -> str:
    """Recompute the holdout digest and compare with the config anchor
    (the inherited integrity gate, mirrored here for the V2 runner)."""
    holdout_dir = repo_path(cfg, "data_clean", "holdout")
    expected = cfg_get(cfg, "loop.holdout_sha256")
    if expected is None:
        raise RuntimeError("loop.holdout_sha256 is not set")
    actual = compute_holdout_digest(holdout_dir)
    if actual != expected:
        raise RuntimeError(
            f"holdout hash mismatch: config {expected} != dir {actual}; "
            "aborting without evaluating (integrity gate)")
    return actual


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_inventory_sha256(cfg: dict) -> str:
    """sha256 over the data_clean inventory (name + size), the frozen
    holdout digest and the symbolic config: identifies the corpus
    snapshot that produced the cached triplets."""
    clean_dir = repo_path(cfg, "data_clean")
    h = hashlib.sha256()
    for path in sorted(clean_dir.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(clean_dir)).encode())
            h.update(str(path.stat().st_size).encode())
    h.update(compute_holdout_digest(clean_dir / "holdout").encode())
    h.update(json.dumps(cfg.get("symbolic", {}), sort_keys=True).encode())
    return h.hexdigest()


def triplets_with_cache(cfg: dict) -> dict:
    """Re-extract only when the corpus inventory changed (runner-level
    cache; the inherited extractor itself is cache-free)."""
    logs_dir = repo_path(cfg, "logs")
    cache_path = logs_dir / "corpus_cache.json"
    inventory = corpus_inventory_sha256(cfg)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("inventory") == inventory and \
                repo_path(cfg, "triplets", "triplets.jsonl").is_file():
            return {"docs": cached.get("docs"), "cached": True}
    stats = extract_corpus(cfg)
    run_consistency(cfg)
    cache_path.write_text(json.dumps(
        {"inventory": inventory, "docs": stats.get("docs")}),
        encoding="utf-8")
    return {**stats, "cached": False}


def treatment_sha256(cfg: dict, stress_mode: str | None) -> str:
    """sha256 over the EFFECTIVE training inputs (D2): corpus inventory +
    triplets file bytes + lr manifest. Mode-agnostic: S2 poisons the
    triplets, S3 the lr manifest, S1 the corpus - all change this hash."""
    h = hashlib.sha256()
    h.update(corpus_inventory_sha256(cfg).encode())
    triplets_path = repo_path(cfg, "triplets", "triplets.jsonl")
    if triplets_path.is_file():
        h.update(_sha256_file(triplets_path).encode())
    lr = {"lr_multiplier": 1 if stress_mode != "lr_spike" else
          int(cfg_get(cfg, "stress.lr_multiplier", 10)),
          "mode": stress_mode or "normal"}
    h.update(json.dumps(lr, sort_keys=True).encode())
    if stress_mode and (repo_path(cfg, "data_stress") / stress_mode).is_dir():
        stress_dir = repo_path(cfg, "data_stress") / stress_mode
        for path in sorted(stress_dir.rglob("*")):
            if path.is_file():
                h.update(str(path.relative_to(stress_dir)).encode())
                h.update(_sha256_file(path).encode())
    return h.hexdigest()


def topo_rules_from_config(cfg: dict) -> TopoRules:
    topo = cfg["topo"]
    return TopoRules(
        churn_bottleneck=float(topo["churn_bottleneck"]),
        churn_gain_excuse=float(0.02),
        b0_collapse_ratio=float(topo["b0_collapse_ratio"]),
        h1_growth_max=float(topo["h1_growth_max"]),
        eps=float(topo["eps_betti"]),
        h1_min_persistence=float(0.1),
    )


def _champion_sha256(cfg: dict) -> str | None:
    champion = repo_path(cfg, "artifacts", "checkpoints", "champion.pt")
    return _sha256_file(champion) if champion.is_file() else None


def measure_checkpoint_topology(cfg: dict, checkpoint: Path) -> dict:
    """Diagrams + betti0 for one checkpoint under the FROZEN geometry.

    The gate input (anchor contract) is the diagram of the CONCATENATED
    per-layer activation cloud: 256 probes x |layers| points, geometry-
    projected (PCA 32 + normalization, A1/A2), subsampled to
    config.topo.subsample. Per-layer diagrams are kept for the notebook
    cache only.
    """
    from prometheus_ns.topo.activations import (
        extract_probe_activations, tensor_sha256,
    )
    from prometheus_ns.topo.persistence import compute_diagrams

    topo_cfg = cfg["topo"]
    geometry = dict(topo_cfg["geometry"])
    tensor_hash = tensor_sha256(checkpoint)
    activations = extract_probe_activations(cfg, checkpoint)
    per_layer = {}
    for layer, points in activations.items():
        diagrams = compute_diagrams(
            points, maxdim=int(topo_cfg["maxdim"]),
            subsample=int(topo_cfg["subsample"]), seed=int(topo_cfg["seed"]),
            geometry=geometry)
        per_layer[str(layer)] = {
            "h0": diagrams[0], "h1": diagrams[1],
            "betti0": betti0_at_eps(diagrams[0], float(topo_cfg["eps_betti"])),
        }
    concat = np.concatenate(
        [activations[str(l)] for l in sorted(activations,
                                             key=lambda x: int(x))], axis=0)
    gate_diagrams = compute_diagrams(
        concat, maxdim=int(topo_cfg["maxdim"]),
        subsample=int(topo_cfg["subsample"]), seed=int(topo_cfg["seed"]),
        geometry=geometry)
    gate_betti0 = betti0_at_eps(gate_diagrams[0],
                                float(topo_cfg["eps_betti"]))
    return {"tensor_sha256": tensor_hash, "layers": per_layer,
            "h0": gate_diagrams[0], "h1": gate_diagrams[1],
            "betti0": gate_betti0,
            "n_points": int(len(concat))}


def measure_graph_topology(cfg: dict, triplets_path: Path | None = None,
                           champion_baseline: bool = False) -> tuple[float, float | None, dict]:
    """R3a census (veto input) + R3b Rips-H1 (observation only).

    triplets_path overrides the triplet source: the S2 stress round
    measures the CANDIDATE graph over the poisoned triplets (the
    effective symbolic material of that round, D2)."""
    topo_cfg = cfg["topo"]
    if triplets_path is None:
        triplets_path = repo_path(cfg, "triplets", "triplets.jsonl")
    triplets = load_triplets(triplets_path)
    graph = build_isa_graph(
        triplets,
        topk=int(topo_cfg["graph_topk"]),
        conf_min=float(topo_cfg["graph_conf_min"]),
        relation=str(topo_cfg["isa_relation"]))
    count, details = graph_cycle_count(
        graph,
        max_cycle_len=int(topo_cfg["max_cycle_len"]),
        max_cycles=int(topo_cfg["max_cycles"]))
    try:
        h1_rips, _ = rips_h1_persistence(graph)
        rips_count = float(persistent_h1_count(
            h1_rips, float(topo_cfg.get("h1_min_persistence", 0.1))))
    except NotImplementedError:
        rips_count = None
    return count, rips_count, details


def running_min_betti0(cfg: dict) -> float | None:
    """D3: the R2 floor compares against the running minimum of betti0
    across rounds (colapso = caer bajo el minimo historico)."""
    topo_log = repo_path(cfg, "logs", "topo.jsonl")
    values = []
    if topo_log.is_file():
        for line in topo_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("betti0_eps"), (int, float)) \
                        and row["betti0_eps"] > 0:
                    values.append(row["betti0_eps"])
    return min(values) if values else None


def fail_closed_check(cfg: dict, round_id: int) -> None:
    """(1.6): round >= 2 without a topo measurement is an error."""
    if round_id < 2:
        return
    topo_log = repo_path(cfg, "logs", "topo.jsonl")
    if topo_log.is_file():
        for line in topo_log.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("round_id") == round_id:
                return
    raise SystemExit(
        f"FAIL-CLOSED (1.6): round {round_id} has no topological "
        "measurement in logs/topo.jsonl; sensor ausente != sensor conforme")


def run_topo_only(cfg: dict) -> dict:
    """make topo-extract: measure the CURRENT champion (or establish the
    round-1 baseline when no champion exists)."""
    from prometheus_ns.model.tokenizer_train import load_tokenizer

    started = time.time()
    verify_holdout(cfg)
    tokenizer = load_tokenizer(cfg)
    checkpoints = repo_path(cfg, "artifacts", "checkpoints")
    champion = checkpoints / "champion.pt"
    if not champion.is_file():
        raise SystemExit("no champion.pt: train the inherited NS round first")

    measurement = measure_checkpoint_topology(cfg, champion)
    betti0 = measurement["betti0"]
    clean_graph = build_isa_graph(
        load_triplets(repo_path(cfg, "triplets", "triplets.jsonl")),
        topk=int(cfg["topo"]["graph_topk"]),
        conf_min=float(cfg["topo"]["graph_conf_min"]),
        relation=str(cfg["topo"]["isa_relation"]))
    cycles = float(triangle_census(clean_graph))
    _, rips_count, details = measure_graph_topology(cfg)
    row = {
        "round_id": 0, "stress_mode": None, "kind": "topo-only",
        "config_sha256": _sha256_file(Path(cfg["_config_path"])),
        "treatment_sha256": treatment_sha256(cfg, None),
        "champion_sha256": _sha256_file(champion),
        "candidate_sha256": _sha256_file(champion),
        "relative_ppl_gain": 0.0,
        "bottleneck_h0": 0.0, "bottleneck_h1": 0.0,
        "betti0_eps": betti0,
        "graph_cycle_count": cycles, "graph_h1_rips_count": rips_count,
        "veto": False, "establish_baseline": True,
        "causes": ["topo-only measurement: establishes/refreshes baseline"],
        "checks": {"R1_churn": {"note": "topo-only"}, "duration_s":
                   round(time.time() - started, 1)},
        "graph_details_truncated": details["truncated"],
    }
    _append_jsonl(repo_path(cfg, "logs", "topo.jsonl"), row)
    print(f"topo-only: betti0@eps={betti0} R3a={cycles:.0f} "
          f"R3b={rips_count}")
    return row


def classify_damage(cfg: dict, manifest: dict | None,
                    candidate_path: Path) -> str | None:
    """B5 post-hoc classification with evidence EXTERNAL to the gate:
    perplexity of the stress candidate vs the champion on a secondary
    heldout drawn from the stress corpus. 'damage evidence' when the
    candidate is clearly worse on that secondary set; 'no verdict'
    otherwise (a difference of criteria, not proven damage)."""
    if manifest is None:
        return None
    import random

    import torch

    from prometheus_ns.eval.perplexity import (_load_model_from_checkpoint,
                                               perplexity_of_docs)
    from prometheus_ns.model.tokenizer_train import load_tokenizer

    docs = [Path(p) for p in manifest.get("treated_docs", [])]
    if not docs or not docs[0].is_file():
        return None
    rng = random.Random(int(cfg["topo"]["seed"]) + 777)
    sample = sorted(rng.sample(docs, min(12, len(docs))))
    secondary = repo_path(cfg, "data_stress", manifest["mode"], "secondary")
    secondary.mkdir(parents=True, exist_ok=True)
    for doc in sample:
        (secondary / doc.name).write_text(doc.read_text(encoding="utf-8"),
                                          encoding="utf-8")
    tokenizer = load_tokenizer(cfg)
    champion = repo_path(cfg, "artifacts", "checkpoints", "champion.pt")
    if not champion.is_file():
        return None
    device = torch.device("cpu")
    champion_model = _load_model_from_checkpoint(cfg, champion, device)[0]
    candidate_model = _load_model_from_checkpoint(cfg, candidate_path,
                                                  device)[0]
    max_seq = int(champion_model.cfg.max_seq)
    champion_ppl, _ = perplexity_of_docs(champion_model, tokenizer,
                                         sorted(secondary.glob("*.txt")),
                                         max_seq, device)
    candidate_ppl, _ = perplexity_of_docs(candidate_model, tokenizer,
                                          sorted(secondary.glob("*.txt")),
                                          max_seq, device)
    if candidate_ppl > champion_ppl * 1.05:
        return "damage evidence"
    return "no verdict"


def run_v2_round(cfg: dict, round_id: int, stress_mode: str | None = None,
                 max_tokens: float | None = None) -> dict:
    """One full V2 round: NS machinery + topological sensors + composite
    gate. The NS decision logic is called INTACT (composition, not
    modification)."""
    from prometheus_ns.eval.perplexity import evaluate_checkpoint_on_holdout
    from prometheus_ns.model.tokenizer_train import load_tokenizer
    from prometheus_ns.topo.activations import extract_probe_activations

    started = time.time()
    logs_dir = repo_path(cfg, "logs")
    checkpoints = repo_path(cfg, "artifacts", "checkpoints")
    champion_path = checkpoints / "champion.pt"
    digest = verify_holdout(cfg)
    config_sha = _sha256_file(Path(cfg["_config_path"]))

    # stress treatment (D1/D2): degraded material + separate checkpoint
    # directory so champion.pt can only change by a normal-round gate
    manifest = None
    stress_checkpoints = checkpoints
    treated_docs = None
    poisoned_triplets = None
    if stress_mode:
        from prometheus_ns.topo.stress import build_treatment
        manifest = build_treatment(cfg, stress_mode, round_id)
        stress_checkpoints = repo_path(cfg, "data_stress", stress_mode,
                                       "checkpoints")
        treated_docs = [Path(p) for p in manifest.get("treated_docs", [])]
        poisoned = manifest.get("material", {}).get("poisoned_triplets")
        poisoned_triplets = Path(poisoned) if poisoned else None

    _append_jsonl(logs_dir / "loop.jsonl",
                  {"round_id": round_id, "phase": "v2_start",
                   "stress_mode": stress_mode, "holdout": digest})

    # 1. corpus (frozen snapshot; crawl excluded by protocol) + triplets
    clean_stats = clean_corpus(cfg)
    triplet_stats = triplets_with_cache(cfg)

    # 2. treatment manifest (D2)
    treatment = treatment_sha256(cfg, stress_mode)

    # 3. diversity of the round's training corpus (inherited machinery)
    tokenizer = load_tokenizer(cfg)
    from prometheus_ns.train.dataset import split_docs_for_training
    if treated_docs is not None:
        train_docs = treated_docs
    else:
        train_docs, _val = split_docs_for_training(cfg, tokenizer)
    sample_dir = repo_path(cfg, "docs", "_diversity_sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("*.txt"):
        old.unlink()
    for doc in train_docs[:10000]:
        (sample_dir / doc.name).write_text(doc.read_text(encoding="utf-8"),
                                           encoding="utf-8")
    diversity_ratio = measure_and_record(sample_dir, round_id, logs_dir)

    # 4. champion baseline on the frozen holdout (inherited eval, intact)
    champion_ppl = None
    if champion_path.is_file():
        champion_ppl = evaluate_checkpoint_on_holdout(
            cfg, tokenizer, champion_path)["ppl"]

    # 5. continued pretraining -> candidate (D1: stress trains on a
    #    separate checkpoint directory; champion changes only via the
    #    gate on normal rounds)
    build_packed_dataset(cfg, tokenizer, train_docs=train_docs)
    target = stress_checkpoints / "latest.pt"
    lr_multiplier = int((manifest or {}).get("lr_multiplier", 1))
    if lr_multiplier != 1:
        # lr_spike (S3): in-memory lr mutation, no inherited file edited;
        # the treatment manifest carries the multiplier for the audit
        for key in ("lr_nano", "lr_small", "lr_large"):
            if key in cfg.get("train", {}):
                cfg["train"][key] = float(cfg["train"][key]) * lr_multiplier
    train_result = train(
        cfg, cfg_get(cfg, "model.profile", "nano"), tokenizer,
        max_tokens=max_tokens,
        init_from=champion_path if champion_path.is_file() else
        (target if target.is_file() else None),
        continue_lr_frac=0.3 if (champion_path.is_file() or
                                 target.is_file()) else None,
        timeout_s=max(float(cfg_get(cfg, "loop.round_timeout_s", 10800))
                      - (time.time() - started), 60.0),
        checkpoints_dir=stress_checkpoints,
        metrics_path=logs_dir / "metrics.jsonl")
    candidate_path = target

    # 6. NS evaluation on the frozen holdout (intact)
    candidate_eval = evaluate_checkpoint_on_holdout(cfg, tokenizer,
                                                    candidate_path)
    candidate_ppl = candidate_eval["ppl"]
    relative_gain = (0.0 if not champion_ppl
                     else (champion_ppl - candidate_ppl) / champion_ppl)

    candidate = RoundMetrics(round_id=round_id, val_ppl=candidate_ppl,
                             diversity_ratio=diversity_ratio,
                             champion_val_ppl=champion_ppl,
                             champion_diversity=None)
    champion_metrics = RoundMetrics(
        round_id=round_id - 1, val_ppl=champion_ppl,
        diversity_ratio=None) if champion_ppl is not None else None
    ns_decision = decide_promotion(
        candidate, champion_metrics,
        float(cfg_get(cfg, "loop.min_ppl_gain", 0.01)),
        float(cfg_get(cfg, "loop.max_diversity_drop", 0.20)),
        digest, cfg_get(cfg, "loop.holdout_sha256"))

    # 7. topological sensors (candidate AND champion)
    candidate_topo = measure_checkpoint_topology(cfg, candidate_path)
    if champion_path.is_file():
        champion_topo = measure_checkpoint_topology(cfg, champion_path)
        b0_h0 = bottleneck_distance(champion_topo["h0"], candidate_topo["h0"])
        b1_h1 = bottleneck_distance(champion_topo["h1"], candidate_topo["h1"])
        champion_betti0 = champion_topo["betti0"]
    else:
        champion_topo = None
        b0_h0 = b1_h1 = 0.0
        champion_betti0 = None
    candidate_betti0 = candidate_topo["betti0"]
    # S2: the candidate's symbolic census reads the POISONED triplets (the
    # round's effective symbolic material); the champion baseline keeps
    # the clean graph from the previous round's log
    cycles, rips_count, details = measure_graph_topology(
        cfg, triplets_path=poisoned_triplets)
    # R3a veto input: the EXACT triangle census (budget-free); the
    # enumerated floor and R3b stay in the audit details
    cycles = float(triangle_census(build_isa_graph(
        load_triplets(poisoned_triplets or repo_path(
            cfg, "triplets", "triplets.jsonl")),
        topk=int(cfg["topo"]["graph_topk"]),
        conf_min=float(cfg["topo"]["graph_conf_min"]),
        relation=str(cfg["topo"]["isa_relation"]))))

    topo_metrics = TopoMetrics(
        round_id=round_id,
        model_h0=candidate_topo["h0"], model_h1=candidate_topo["h1"],
        graph_h1_count=cycles,
        champion_model_h0=champion_topo["h0"] if champion_topo else None,
        champion_model_h1=champion_topo["h1"] if champion_topo else None,
        champion_graph_h1_count=None,  # filled from logs below
        baseline_b0=running_min_betti0(cfg) or champion_betti0,
        relative_ppl_gain=relative_gain)
    if champion_topo is None:
        topo_decision = decide_topo_veto(topo_metrics,
                                         topo_rules_from_config(cfg))
        topo_decision.checks["note"] = "round 1: establishes the baseline"
    else:
        # champion graph census comes from the previous round's log (the
        # champion graph is unchanged since then)
        prev = None
        topo_log = logs_dir / "topo.jsonl"
        if topo_log.is_file():
            for line in topo_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("graph_cycle_count") is not None:
                        prev = row
            if prev is not None:
                topo_metrics = replace(topo_metrics,
                                       champion_graph_h1_count=prev["graph_cycle_count"])
        topo_decision = decide_topo_veto(topo_metrics,
                                         topo_rules_from_config(cfg))

    # 8. FAIL-CLOSED audit + composite gate (composition, not modification)
    if round_id >= 2 and champion_topo is None:
        raise SystemExit(
            f"FAIL-CLOSED (1.6): round {round_id} without topological "
            "measurements; sensor ausente != sensor conforme")
    final_promote = bool(ns_decision.promote and not ns_decision.abort
                         and not topo_decision.veto)

    # 9. checkpoint lifecycle (D1: stress candidates never touch champion)
    if final_promote and not stress_mode:
        promote_checkpoint(checkpoints)
    elif not final_promote and champion_path.is_file() and not stress_mode:
        rollback_to_champion(checkpoints)

    # 10. structured writes (B5 damage classification on divergence)
    damage = None
    if topo_decision.veto and ns_decision.promote:
        damage = classify_damage(cfg, manifest, candidate_path)
    topo_row = {
        "round_id": round_id, "stress_mode": stress_mode,
        "config_sha256": config_sha, "treatment_sha256": treatment,
        "champion_sha256": _champion_sha256(cfg),
        "candidate_sha256": _sha256_file(candidate_path),
        "holdout_sha256": digest,
        "relative_ppl_gain": relative_gain,
        "bottleneck_h0": b0_h0, "bottleneck_h1": b1_h1,
        "betti0_eps": candidate_betti0,
        "graph_cycle_count": cycles, "graph_h1_rips_count": rips_count,
        "veto": topo_decision.veto,
        "establish_baseline": topo_decision.establish_baseline,
        "causes": topo_decision.causes, "checks": topo_decision.checks,
        "graph_details_truncated": details["truncated"],
        "ns_decision": {"promote": ns_decision.promote,
                        "abort": ns_decision.abort,
                        "cause": ns_decision.cause},
        "train": {"step": train_result["step"],
                  "tokens_seen": train_result["tokens_seen"]},
        "duration_s": round(time.time() - started, 1),
    }
    _append_jsonl(logs_dir / "topo.jsonl", topo_row)
    _append_jsonl(logs_dir / "promotion_decisions.jsonl", {
        "round_id": round_id, "ns_promote": bool(ns_decision.promote),
        "ns_cause": ns_decision.cause, "topo_veto": topo_decision.veto,
        "final_promoted": final_promote,
        "damage_classification": damage,
    })
    candidate_metrics = RoundMetrics(
        round_id=round_id, val_ppl=candidate_ppl,
        diversity_ratio=diversity_ratio,
        champion_val_ppl=champion_ppl, champion_diversity=None)
    write_promotion_log(repo_path(cfg, "docs"),
                        type("D", (), {"promote": final_promote,
                                       "abort": ns_decision.abort,
                                       "cause": ns_decision.cause})(),
                        candidate_metrics)
    if topo_decision.veto and ns_decision.promote:
        print(f"STAR CASE: round {round_id}: NS promoted, the topology "
              f"vetoed: {topo_decision.causes}")
    print(f"v2 round {round_id}: final_promoted={final_promote} "
          f"ns={ns_decision.promote} topo_veto={topo_decision.veto} "
          f"ppl={candidate_ppl:.4f} R3a={cycles:.0f}"
          + (f" damage={damage}" if damage else ""))
    return topo_row


def main() -> None:
    parser = argparse.ArgumentParser(description="PROMETHEUS-V2 runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--topo-only", action="store_true")
    parser.add_argument("--stress", choices=["dup_flood",
                                             "contradiction_flood",
                                             "lr_spike"])
    parser.add_argument("--round-id", type=int, default=None)
    parser.add_argument("--max-tokens", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    max_tokens = float(args.max_tokens) if args.max_tokens else None
    if args.topo_only:
        run_topo_only(cfg)
        return
    round_id = args.round_id
    if round_id is None:
        raise SystemExit("--round-id is required for formal rounds")
    if args.stress:
        from prometheus_ns.topo.stress import build_treatment
        build_treatment(cfg, args.stress, round_id)
    run_v2_round(cfg, round_id, stress_mode=args.stress,
                 max_tokens=max_tokens)


if __name__ == "__main__":
    main()
