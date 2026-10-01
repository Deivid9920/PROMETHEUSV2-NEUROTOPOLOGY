#!/usr/bin/env python3
"""Generate the three PROMETHEUS-V2 TDA notebooks (V2.1).

Idempotent (never overwrites executed notebooks). Cells read ONLY the
structured JSON logs (logs/topo.jsonl, logs/promotion_decisions.jsonl,
artifacts/topo/*.npz, artifacts/topo/self_noise.json) — never markdown
reports (E1). Cells import the fixed V2 package API; notebooks fail with
ImportError — not wrong numbers — until their phase is implemented.

Usage:
    python3 scripts/make_notebooks_topo.py --out notebooks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

SETUP_CELL = """\
# PROMETHEUS-V2 setup cell — run FIRST. On Colab with GPU desired:
# Runtime > Change runtime type > GPU, then run this.
import os, subprocess, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/REPLACE-USER/PROMETHEUS-NS.git"  # <- edit

if IN_COLAB:
    if not Path("/content/PROMETHEUS-NS/.git").exists():
        subprocess.run(["git", "clone", REPO_URL, "/content/PROMETHEUS-NS"],
                       check=True)
    os.chdir("/content/PROMETHEUS-NS")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)
    print("Colab setup complete.")
else:
    try:
        import torch, ripser, scipy, networkx  # noqa: F401
        print("Local environment OK.")
    except ImportError as exc:
        raise SystemExit(f"Missing dependency ({exc}). Run `make setup`.")
"""

IMPORTS_CELL = """\
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd().resolve()
while not (REPO / "config.yaml").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
os_cwd = Path.cwd()

FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_figure(fig, name):
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure -> {out}")


TOPO = load_jsonl(REPO / "logs" / "topo.jsonl")
DECISIONS = load_jsonl(REPO / "logs" / "promotion_decisions.jsonl")
print(f"topo.jsonl rows: {len(TOPO)} | decisions: {len(DECISIONS)}")
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


NB5 = [
    md("# 05 - Model topology: persistence diagrams per layer and round\n\n"
       "Numbers come ONLY from logs/topo.jsonl and artifacts/topo caches "
       "(E1: structured JSON, never markdown parsing). The sensor scans "
       "fixed probe sentences extracted from data_clean EXCLUDING the "
       "frozen holdout (H3 hygiene)."),
    code(SETUP_CELL),
    code(IMPORTS_CELL),
    md("## betti0@eps per round with the running minimum (D3 baseline)\n\n"
       "The R2 collapse floor compares against the RUNNING MINIMUM of "
       "betti0@eps: a legitimately improving structure never creates a "
       "false floor."),
    code("""\
rows = [{"round": r["round_id"], "stress": r.get("stress_mode"),
         "betti0": r["betti0_eps"],
         "baseline": r.get("establish_baseline", False)}
        for r in TOPO if isinstance(r.get("betti0_eps"), (int, float))]
betti = pd.DataFrame(rows)
if betti.empty:
    raise SystemExit("no topological measurements yet: run the V2 rounds")
betti["running_min"] = betti["betti0"].cummin()
print(betti.to_string(index=False))"""),
    code("""\
fig, ax = plt.subplots(figsize=(8, 4.5))
kinds = betti["stress"].fillna("normal")
colors = {"normal": "#1f77b4"}
for i, (idx, row) in enumerate(betti.iterrows()):
    color = "tab:red" if isinstance(row["stress"], str) else "tab:blue"
    ax.scatter(row["round"], row["betti0"], color=color, zorder=3, s=55)
ax.step(betti["round"], betti["running_min"], where="post",
        ls="--", lw=1.0, label="running minimum (D3 floor)")
ax.set_xlabel("round"); ax.set_ylabel("betti0 @ eps = eps_betti")
ax.set_title("representational clusters per round")
ax.legend(); save_figure(fig, "nb5_betti0_per_round"); plt.show()"""),
    md("## Self-noise distribution of the sensor (Fase 1-V2, risk A3)\n\n"
       "The churn threshold only means something above the subsampling "
       "noise floor: 5 seeds, pairwise self-bottleneck per layer."),
    code("""\
sn_path = REPO / "artifacts" / "topo" / "self_noise.json"
if not sn_path.is_file():
    print("self_noise.json missing: the sensor is NOT calibrated yet "
          "(Fase 1-V2 verification on the real champion)")
else:
    sn = json.loads(sn_path.read_text(encoding="utf-8"))
    values = [sn["layers"][layer]["pairwise"][i][j]
              for layer in sn["layers"]
              for i in range(len(sn["seeds"]))
              for j in range(i + 1, len(sn["seeds"]))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=12, color="#4c78a8")
    ax.axvline(sn["p95_overall"], ls="--", lw=1.2,
               label=f"p95 = {sn['p95_overall']:.4f}")
    import yaml
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    thr = cfg["topo"]["churn_bottleneck"]
    ax.axvline(thr, color="tab:red", ls="--", lw=1.2,
               label=f"churn_bottleneck = {thr}")
    ax.set_xlabel("self-bottleneck (same checkpoint, different subsample)")
    ax.set_title("sensor noise floor vs threshold")
    ax.legend(); save_figure(fig, "nb5_self_noise"); plt.show()"""),
    md("## Interpretation and limits\n\n"
       "- The topological sensor DESCRIBES geometry; it does not cause "
       "quality (descriptive, not causal).\n"
       "- One model, one geometry policy, one seed: comparisons across "
       "rounds with an identical pipeline only."),
]

NB6 = [
    md("# 06 - Symbolic-graph topology: R3a census and R3b filtration\n\n"
       "R3a (PRIMARY, feeds the veto): directed simple cycles with every "
       "edge above the confidence floor - a structural taxonomic "
       "contradiction.\n\n"
       "R3b (SECONDARY, observation only): persistent H1 of the Rips "
       "filtration over d(u,v) = 1 - conf. Under this filtration "
       "high-confidence cycles CLOSE EARLY (low persistence) and weak "
       "cycles persist - R3b is NOT a contradiction sensor."),
    code(SETUP_CELL),
    code(IMPORTS_CELL),
    code("""\
from prometheus_ns.topo.graph_topology import (
    build_isa_graph, census_directed_cycles, adjacency_distance_matrix,
    load_triplets, rips_h1_persistence)
import yaml
cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
triplets = load_triplets(REPO / "triplets" / "triplets.jsonl")
graph = build_isa_graph(triplets,
                        topk=cfg["topo"]["graph_topk"],
                        conf_min=cfg["topo"]["graph_conf_min"],
                        relation=cfg["topo"]["isa_relation"])
cycles, truncated = census_directed_cycles(
    graph, cfg["topo"]["max_cycle_len"], cfg["topo"]["max_cycles"])
print(f"R3a census: {len(cycles)} directed cycles "
      f"(truncated={truncated})")"""),
    md("## The census, with edges and confidences"),
    code("""\
if cycles:
    cyc = pd.DataFrame([{"cycle": " -> ".join(c["path"] + [c["path"][0]]),
                         "min_conf": c["min_conf"]} for c in cycles])
    display(cyc.head(25))
else:
    print("no directed cycles above the confidence floor")"""),
    code("""\
counts = [r.get("graph_cycle_count") for r in TOPO
          if r.get("graph_cycle_count") is not None]
rounds = [r["round_id"] for r in TOPO
          if r.get("graph_cycle_count") is not None]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rounds, counts, marker="o")
ax.set_xlabel("round"); ax.set_ylabel("R3a directed-cycle census")
ax.set_title("symbolic cycle growth across rounds")
save_figure(fig, "nb6_r3a_census"); plt.show()"""),
    md("## R3b panel (secondary, honest reading)\n\n"
       "The 4-cycle hand value anchors the filtration: edge distance 0.5 "
       "is born at 0.5 and dies at 1.0 (K4 completion) - persistence 0.5."),
    code("""\
try:
    h1, meta = rips_h1_persistence(graph)
    persistence = h1[:, 1] - h1[:, 0]
    persistent = persistence[persistence > 0.1]
    print(f"R3b persistent H1 classes (persistence > 0.1): {len(persistent)}")
    print("READING: a geometric indicator of graph reorganization - "
          "R3b NEVER feeds the veto.")
except NotImplementedError:
    print("persistence engine pending (Fase 1-V2)")"""),
    md("## Cross with the NS quarantine\n\n"
       "Do the directed cycles capture taxonomic contradictions that the "
       "pairwise detection quarantined, or do they see NEW damage?"),
    code("""\
quarantine_path = REPO / "triplets" / "quarantine.jsonl"
quarantined = load_jsonl(quarantine_path)
print(f"quarantined docs (NS pairwise detection): {len(quarantined)}")
cycle_entities = {e for c in cycles for e in c["path"]}
print(f"entities inside R3a cycles: {len(cycle_entities)}")
print("crossing the two sets is a qualitative readout: the census sees "
      "graph-wide structure, the pair detector sees local conflicts")"""),
]

NB7 = [
    md("# 07 - Divergence study: does the topological veto inform?\n\n"
       "Divergence = NS promotes AND the topology vetoes. Rates per arm "
       "with exact Wilson intervals (n is small: the interval IS the "
       "result). A zero rate is a valid finding and is discussed, never "
       "hidden."),
    code(SETUP_CELL),
    code(IMPORTS_CELL),
    code("""\
from statistics import NormalDist
import math


def wilson(successes, total, ci=0.95):
    if total == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - (1 - ci) / 2)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0, centre - spread), min(1, centre + spread)


dec_by_round = {d["round_id"]: d for d in DECISIONS}
rows = []
for r in TOPO:
    dec = dec_by_round.get(r["round_id"], {})
    rows.append({
        "round": r["round_id"],
        "stress": r.get("stress_mode"),
        "ns_promoted": bool(dec.get("ns_promote", False)),
        "topo_veto": bool(r.get("veto", False)),
        "establish_baseline": bool(r.get("establish_baseline", False)),
        "ppl_gain": r.get("relative_ppl_gain", 0.0),
    })
table = pd.DataFrame(rows)
if table.empty:
    raise SystemExit("no V2 logs yet: run the experiment rounds")
table["divergence"] = table["ns_promoted"] & table["topo_veto"] \\
    & ~table["establish_baseline"]
print(table.to_string(index=False))"""),
    md("## Divergence rates per arm (case-study framing)"),
    code("""\
for arm, sel in (("normal", ~table["stress"].notna() | table["establish_baseline"]),
                 ("stress", table["stress"].notna())):
    arm_rows = table[sel & ~table["establish_baseline"]]
    div = int(arm_rows["divergence"].sum())
    lo, hi = wilson(div, len(arm_rows))
    print(f"{arm}: {div}/{len(arm_rows)} divergence | "
          f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
print("\\nBoth rates zero IS a valid result: the sensor was live "
      "(diagrams committed) and the honest reading follows.")"""),
    md("## Perplexity vs structural distance\n\n"
       "Rounds where ppl moves little but the bottleneck moves a lot are "
       "the veto's target population - each point is one round."),
    code("""\
obs = [r["checks"]["R1_churn"]["observed"] for r in TOPO
       if "R1_churn" in (r.get("checks") or {})
       and "observed" in r["checks"]["R1_churn"]]
gains = [r.get("relative_ppl_gain", 0) for r in TOPO
         if "R1_churn" in (r.get("checks") or {})
         and "observed" in r["checks"]["R1_churn"]]
kinds = ["stress" if r.get("stress_mode") else "normal" for r in TOPO
         if "R1_churn" in (r.get("checks") or {})
         and "observed" in r["checks"]["R1_churn"]]
fig, ax = plt.subplots(figsize=(7, 5))
for kind in set(kinds):
    xs = [g for g, k in zip(gains, kinds) if k == kind]
    ys = [o for o, k in zip(obs, kinds) if k == kind]
    ax.scatter(xs, ys, label=kind, s=45)
import yaml
cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
ax.axhline(cfg["topo"]["churn_bottleneck"], ls="--", lw=0.8)
ax.set_xlabel("relative ppl gain"); ax.set_ylabel("bottleneck distance")
ax.legend(); save_figure(fig, "nb7_ppl_vs_bottleneck"); plt.show()"""),
    md("## Interpretation and limits\n\n"
       "- What the measured divergence rates do and do not establish: "
       "round count is small, one model, one seed, one geometry policy.\n"
       "- Thresholds were frozen before S1 (commit "
       "`chore(v2): freeze thresholds before stress`); any post-hoc "
       "tuning marks the experiment invalid (config_sha256 audit).\n"
       "- Divergence without a damage verdict proves a difference of "
       "criteria, not that the veto is right.\n"
       "- The veto is a guardrail, not a proof of quality."),
]

NOTEBOOKS = {
    "05_model_topology.ipynb": NB5,
    "06_graph_topology.ipynb": NB6,
    "07_divergence_study.ipynb": NB7,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        target = out_dir / name
        if target.exists():
            print(f"  = {name}: kept existing (delete to regenerate)")
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}
        nbf.write(nb, str(target))
        print(f"  + {name}")


if __name__ == "__main__":
    main()
