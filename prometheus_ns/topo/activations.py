"""Probe-activation extraction (Fase 1-V2).

create_probe_sentences(cfg) -- ONE-TIME: sample config.topo.probe_sentences
sentences from data_clean/ EXCLUDING data_clean/holdout/ (the holdout
stays blind to topology, H3), dropping any longer than model max_seq
with the discarded count recorded, writing
data/processed/probe_sentences.json with sha256. load_probe_sentences(cfg)
fails loud if missing.

extract_probe_activations(cfg, checkpoint_path) -> {layer: np.ndarray
[n_probes, d_model]}: model.eval() + torch.no_grad(), fp32, forward
hooks at config.topo.layers (block OUTPUT = residual stream,
pre-final-norm, C2 policy, identical for champion and candidate), LAST
REAL token vector (lengths tracked, fail-loud on overlong input), batch
32, deterministic order. Determinism contract (C1): two full extractions
of the same checkpoint are BIT IDENTICAL (np.array_equal, no tolerance).

tensor_sha256(checkpoint_path) -> cache key over state_dict tensor bytes
in canonical (sorted-name) order (C5).

self_noise_report(cfg, checkpoint_path, n_seeds=5) -> writes
artifacts/topo/self_noise.json {seeds, layers: {layer: {pairwise}},
p95_overall} -- the A3 calibration gate consumed by
tests/test_topo_stability.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from prometheus_ns import cfg_get, load_config, repo_path
from prometheus_ns.model.config import ModelConfig
from prometheus_ns.model.model import PAD_ID, TransformerLM
from prometheus_ns.model.tokenizer_train import load_tokenizer
from prometheus_ns.topo.promotion_topology import bottleneck_distance

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_TOKENS = 8


def _probe_path(cfg: dict) -> Path:
    return repo_path(cfg, "data", "processed", "probe_sentences.json")


def _active_max_seq(cfg: dict) -> int:
    profile = cfg_get(cfg, "model.profile", "nano")
    if profile == "auto":
        profile = cfg_get(cfg, "autofit.cpu_max_profile", "nano")
    return int(cfg["model"][profile]["max_seq"])


def create_probe_sentences(cfg: dict) -> dict:
    """ONE-TIME probe creation (H3 hygiene). Refuses to regenerate."""
    out_path = _probe_path(cfg)
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(cfg)
    max_seq = _active_max_seq(cfg)
    n_wanted = int(cfg["topo"]["probe_sentences"])
    seed = int(cfg["topo"]["seed"])
    clean_dir = repo_path(cfg, "data_clean")
    holdout = {p.resolve() for p in (clean_dir / "holdout").glob("**/*.txt")}

    candidates: list[str] = []
    seen: set[str] = set()
    discarded_overlong = 0
    docs = sorted(p for p in clean_dir.glob("**/*.txt")
                  if p.resolve() not in holdout)
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for raw in _SENTENCE_SPLIT.split(text):
            sentence = " ".join(raw.split())
            if len(sentence) < 40 or sentence in seen:
                continue
            ids = tokenizer.encode(sentence).ids
            if len(ids) > max_seq:
                discarded_overlong += 1
                continue
            if len(ids) < _MIN_SENTENCE_TOKENS:
                continue
            seen.add(sentence)
            candidates.append(sentence)

    if len(candidates) < n_wanted:
        raise RuntimeError(
            f"only {len(candidates)} probe candidates available, need "
            f"{n_wanted}: the cleaned corpus is too small")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(candidates), size=n_wanted,
                               replace=False).tolist())
    sentences = [candidates[i] for i in chosen]
    payload = {
        "sentences": sentences,
        "sha256": hashlib.sha256(
            "\n".join(sentences).encode("utf-8")).hexdigest(),
        "discarded_overlong": discarded_overlong,
        "candidates": len(candidates),
        "max_seq": max_seq,
        "seed": seed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"probe sentences: {len(sentences)} written -> {out_path} "
          f"(discarded overlong: {discarded_overlong})")
    return payload


def load_probe_sentences(cfg: dict) -> list[str]:
    """Fail-loud loader: the probe set is the project's fixed patient."""
    path = _probe_path(cfg)
    if not path.is_file():
        raise SystemExit(
            f"missing {path}: create the probe sentences once with "
            "create_probe_sentences() before any topological measurement")
    return json.loads(path.read_text(encoding="utf-8"))["sentences"]


def tensor_sha256(checkpoint_path) -> str:
    """C5 cache key: sha256 over state_dict tensor BYTES in canonical
    (sorted-name) order - never the .pt file (optimizer state changes on
    re-save and would break the cache)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state = checkpoint["model_state"]
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def extract_probe_activations(cfg: dict, checkpoint_path) -> dict:
    """Residual-stream activations of the fixed probes per hooked layer.

    Returns {str(layer): np.ndarray [n_probes, d_model]} in fp32. C1
    determinism: model.eval() + torch.no_grad(), fp32, fixed sentence
    order, batch 32 with right padding (causal attention + RoPE make
    padding after position len-1 inert, so the LAST REAL token vector of
    a sentence inside a batch equals the standalone vector).
    """
    probes = load_probe_sentences(cfg)
    layers = [int(x) for x in cfg["topo"]["layers"]]
    tokenizer = load_tokenizer(cfg)
    max_seq = _active_max_seq(cfg)

    encoded = []
    for sentence in probes:
        ids = tokenizer.encode(sentence).ids
        if len(ids) > max_seq:
            raise SystemExit(
                f"probe sentence exceeds max_seq={max_seq} "
                f"({len(ids)} tokens): the fail-loud contract (C3) fires; "
                "recreate probe_sentences.json only via the documented "
                "one-time procedure")
        encoded.append(ids)

    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    model = TransformerLM(ModelConfig(**checkpoint["profile"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # C1: dropout off, deterministic forward

    captured: dict[int, list] = {}

    def _hook(layer_index: int):
        def hook(_module, _inputs, output):
            captured.setdefault(layer_index, []).append(
                output.detach().to(torch.float32).cpu())
        return hook

    handles = [model.blocks[layer].register_forward_hook(_hook(layer))
               for layer in layers]

    result: dict[int, list] = {}
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            lengths = [len(ids) for ids in batch]
            width = max(lengths)
            input_ids = torch.full((len(batch), width), PAD_ID,
                                   dtype=torch.long)
            for i, ids in enumerate(batch):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            captured.clear()
            model(input_ids)
            for layer in layers:
                block_out = torch.cat(captured[layer], dim=0)  # [B, T, d]
                for i, length in enumerate(lengths):
                    result.setdefault(layer, []).append(
                        block_out[i, length - 1, :].numpy())
    for handle in handles:
        handle.remove()

    return {str(layer): np.stack(result[layer]).astype(np.float32)
            for layer in layers}


def self_noise_report(cfg: dict, checkpoint_path, n_seeds: int = 5) -> dict:
    """A3 calibration gate: self-bottleneck distribution of the sensor.

    The veto's actual input is the CONCATENATED per-layer cloud (anchor
    contract: model_h0 = candidate, concatenated layers), so the noise
    floor is measured there: same checkpoint, n_seeds subsample seeds,
    pairwise bottleneck distances, p95 vs churn_bottleneck.
    """
    from prometheus_ns.topo.persistence import compute_diagrams

    topo_cfg = cfg["topo"]
    layers = [str(x) for x in topo_cfg["layers"]]
    acts = extract_probe_activations(cfg, checkpoint_path)
    concat = np.concatenate([acts[layer] for layer in layers], axis=0)
    seeds = [int(topo_cfg["seed"]) + i for i in range(n_seeds)]
    geometry = dict(topo_cfg["geometry"])

    diagrams = []
    for seed in seeds:
        diagrams.append(compute_diagrams(
            concat, maxdim=int(topo_cfg["maxdim"]),
            subsample=int(topo_cfg["subsample"]), seed=seed,
            geometry=geometry))
    pairwise = [[0.0] * n_seeds for _ in range(n_seeds)]
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            distance = bottleneck_distance(diagrams[i][0], diagrams[j][0])
            pairwise[i][j] = distance
            pairwise[j][i] = distance
    values = [pairwise[i][j] for i in range(n_seeds)
              for j in range(i + 1, n_seeds)]
    report = {
        "checkpoint_sha256": tensor_sha256(checkpoint_path),
        "seeds": seeds,
        "layers": {"concat": {"pairwise": pairwise,
                              "n_points": int(len(concat)),
                              "subsample": int(topo_cfg["subsample"])}},
        "p95_overall": float(np.percentile(values, 95)),
        "values": values,
    }
    out = repo_path(cfg, "artifacts", "topo", "self_noise.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"self-noise: p95={report['p95_overall']:.4f} over {len(values)} "
          f"pairs -> {out}")
    return report
