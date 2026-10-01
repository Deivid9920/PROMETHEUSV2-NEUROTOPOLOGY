"""Deliberate-degradation treatments (Fase 4-V2).

build_treatment(cfg, mode, round_id) -> manifest with treatment_sha256
(sha256 over corpus bytes + triplets file + lr manifest, D2) and the
degraded material:

- dup_flood: duplicates the training docs deterministically until
  stress.dup_ratio of the FINAL corpus is duplicates (near-duplicates:
  the repetition is the degradation the R2 sensor watches).
- contradiction_flood: plants is_a contradictions in the triplets
  (confident edges closing directed cycles) targeting R3a; the poisoned
  file lives under data_stress/ and the round's census reads it.
- lr_spike: multiplies the lr by stress.lr_multiplier and shortens the
  corpus 10x (deterministic subsample) targeting R1.

All treatments are deterministic under config.topo.seed, write under
data_stress/, and never touch champion.pt (D1: stress rounds train in a
separate checkpoint directory; champion.pt only changes by gate
decision on normal rounds).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from prometheus_ns import cfg_get, repo_path
from prometheus_ns.train.dataset import split_docs_for_training
from prometheus_ns.model.tokenizer_train import load_tokenizer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _manifest_dir(cfg: dict, mode: str) -> Path:
    path = repo_path(cfg, "data_stress", mode)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_docs(cfg: dict) -> list[Path]:
    tokenizer = load_tokenizer(cfg)
    docs, _val = split_docs_for_training(cfg, tokenizer)
    return docs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_treatment(cfg: dict, mode: str, round_id: int) -> dict:
    """Build one stress treatment; returns the manifest (also written to
    data_stress/<mode>/manifest.json)."""
    if mode not in ("dup_flood", "contradiction_flood", "lr_spike"):
        raise SystemExit(f"unknown stress mode: {mode}")
    out = _manifest_dir(cfg, mode)
    docs = _train_docs(cfg)
    seed = int(cfg["topo"]["seed"])
    lr_multiplier = 1
    corpus_frac = 1.0
    treated_docs = list(docs)  # default: the corpus itself (S2 keeps it)

    if mode == "dup_flood":
        # duplicate deterministically until dup_ratio of the final corpus
        # is duplicated material: n_final = n_orig / (1 - dup_ratio)
        ratio = float(cfg["stress"]["dup_ratio"])
        n_final = int(len(docs) / max(1.0 - ratio, 1e-9))
        extra_needed = max(0, n_final - len(docs))
        rng = random.Random(seed + round_id)
        duplicates = []
        i = 0
        while len(duplicates) < extra_needed:
            source = docs[i % len(docs)]
            copy_path = out / f"dup_{i:05d}_{source.name}"
            if not copy_path.is_file():
                copy_path.write_text(source.read_text(encoding="utf-8"),
                                     encoding="utf-8")
            duplicates.append(copy_path)
            i += 1
        treated_docs = list(docs) + duplicates
        note = (f"{len(duplicates)} near-duplicate docs injected "
                f"(dup_ratio {ratio}); {len(treated_docs)} docs total")
        material = {"duplicates": [str(p) for p in duplicates]}
    elif mode == "contradiction_flood":
        # plant confident is_a contradictions AT the entity set the census
        # measures (the top-K by degree): complete directed triangles
        # (A->B->C->A) at conf 0.99, simulating a corrupted extraction
        # that closes taxonomic loops over the popular entities
        triplets_path = repo_path(cfg, "triplets", "triplets.jsonl")
        poisoned = out / "triplets_poisoned.jsonl"
        records = [json.loads(line) for line in
                   triplets_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        adjacency: dict[str, list[str]] = {}
        for rec in records:
            if rec.get("rel") == "is_a":
                adjacency.setdefault(str(rec["sub"]), []).append(str(rec["obj"]))
        degree: dict[str, int] = {}
        for u, vs in adjacency.items():
            for v in vs:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1
        topk = int(cfg["topo"]["graph_topk"])
        top_entities = sorted(degree, key=lambda n: (-degree[n], n))[:topk]
        rng = random.Random(seed + round_id)
        planted = []
        n_cycles = min(600, max(50, len(top_entities) // 2))
        for k in range(n_cycles):
            a, b, c = rng.sample(top_entities, 3)
            for sub, obj in ((a, b), (b, c), (c, a)):
                planted.append({"sub": sub, "rel": "is_a", "obj": obj,
                                "conf": 0.99,
                                "triplet_id": f"planted_cyc{k}_{sub[:8]}_{obj[:8]}",
                                "planted": True})
        with poisoned.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for rec in planted:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        note = (f"{n_cycles} planted directed triangles ({len(planted)} "
                f"edges at conf 0.99) over the top-{topk} entity set "
                "- R3a target")
        material = {"poisoned_triplets": str(poisoned), "planted": planted}
    else:  # lr_spike
        lr_multiplier = int(cfg["stress"]["lr_multiplier"])
        corpus_frac = 0.1
        rng = random.Random(seed + round_id)
        kept = sorted(docs, key=lambda p: p.name)
        keep_n = max(1, int(len(kept) * corpus_frac))
        step = len(kept) / keep_n
        treated_docs = [kept[int(i * step)] for i in range(keep_n)]
        note = (f"lr x{lr_multiplier} on a 10x shorter corpus "
                f"({len(treated_docs)} of {len(kept)} docs)")
        material = {"docs": [str(p) for p in treated_docs],
                    "lr_multiplier": lr_multiplier}

    manifest = {
        "mode": mode, "round_id": round_id, "note": note,
        "lr_multiplier": lr_multiplier, "corpus_frac": corpus_frac,
        "treated_docs": [str(p) for p in treated_docs],
        "material": material,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
    digest = hashlib.sha256()
    digest.update(_sha(manifest_path).encode())
    digest.update(_sha(repo_path(cfg, "triplets", "triplets.jsonl")).encode())
    manifest["treatment_sha256"] = digest.hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
    print(f"stress treatment [{mode}]: {note}")
    return manifest
