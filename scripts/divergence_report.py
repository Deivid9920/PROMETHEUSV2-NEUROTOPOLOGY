#!/usr/bin/env python3
"""Divergence report with mechanical experiment-integrity checks.

Immutable anchor (V2.1). Implements the verifications the master prompt
declares mandatory; without them the divergence experiment can be a
fiction: silently retuned thresholds, stress rounds whose degraded corpus
was never actually used, a champion mutated behind a veto.

Expected log schemas (written by the V2 loop, one JSON object per line):

logs/topo.jsonl:
  {round_id, stress_mode|null, config_sha256, treatment_sha256,
   champion_sha256, candidate_sha256, relative_ppl_gain,
   bottleneck_h0, bottleneck_h1, betti0_eps, graph_cycle_count,
   graph_h1_rips_count|null, veto, establish_baseline, causes[], checks{}}

logs/promotion_decisions.jsonl:
  {round_id, ns_promote, ns_cause, topo_veto, final_promoted,
   damage_classification|null}

treatment_sha256 = sha256 over the EFFECTIVE training inputs of the
round (corpus bytes + triplets file + lr manifest). Normal rounds share
one value; every stress round must differ from it (D2). This is
mode-agnostic: S2 poisons triplets rather than the corpus, and a plain
corpus hash would false-flag it.

Verifications (any violation prints the EXPERIMENT INVALID banner and
exits 1; the report is still rendered for the record):
  V1 threshold drift (B1): all rounds share one config_sha256.
  V2 treatment integrity (D2): normal rounds share treatment_sha256;
      every stress round's differs.
  V3 champion integrity (D1): after a vetoed round, the next round's
      champion_sha256 equals the vetoed round's.
  V4 fail-closed audit (1.6): every decision round >= 2 has a topo
      measurement; every topo measurement has a decision entry.
  V5 baseline uniqueness: exactly one establish_baseline round, the first.

Statistics: divergence = ns_promote AND topo_veto. Rates for normal and
stress rounds with exact Wilson score intervals (stdlib NormalDist, no
scipy needed). Missing joins render as 'missing', never invented.
Per-round damage_classification (B5) is rendered when present; the
protocol owns filling it with external evidence.

Usage:
    python3 scripts/divergence_report.py [--logs logs]
        [--out docs/divergence_report.md] [--ci 0.95]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def wilson_interval(successes: int, total: int, ci: float) -> tuple[float, float]:
    """Exact Wilson score interval (stdlib only)."""
    if total == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1.0 - (1.0 - ci) / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / total
                                     + z * z / (4.0 * total * total))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def verify_integrity(topo_rows: list[dict], decisions: list[dict]) -> list[str]:
    """Mechanical verifications V1-V5; returns violation strings."""
    violations: list[str] = []
    by_round_topo = {int(r["round_id"]): r for r in topo_rows if "round_id" in r}
    by_round_dec = {int(d["round_id"]): d for d in decisions if "round_id" in d}

    # V1 threshold drift (B1)
    configs = {r.get("config_sha256") for r in topo_rows}
    if len(configs) > 1:
        violations.append(
            f"V1 threshold drift (B1): {len(configs)} distinct config_sha256 "
            "values across rounds -> EXPERIMENT INVALID: threshold drift")
    elif topo_rows:
        missing = [r.get("round_id") for r in topo_rows
                   if not r.get("config_sha256")]
        if missing:
            violations.append(f"V1: rounds missing config_sha256: {missing}")

    # V2 treatment integrity (D2)
    normal = [r for r in topo_rows if not r.get("stress_mode")]
    stress = [r for r in topo_rows if r.get("stress_mode")]
    normal_treatments = {r.get("treatment_sha256") for r in normal}
    if len(normal_treatments) > 1:
        violations.append(
            "V2 treatment integrity (D2): normal rounds do NOT share one "
            "treatment_sha256 -> the 'normal condition' is not a condition")
    stress_treatments = {r.get("treatment_sha256") for r in stress}
    if stress and stress_treatments & normal_treatments:
        violations.append(
            "V2: a stress round shares treatment_sha256 with the normal "
            "condition -> the degradation never reached training (D2)")

    # V3 champion integrity (D1)
    ordered = sorted(by_round_topo)
    for prev_id, next_id in zip(ordered, ordered[1:]):
        prev, nxt = by_round_topo[prev_id], by_round_topo[next_id]
        if prev.get("veto") and not prev.get("establish_baseline"):
            if prev.get("champion_sha256") != nxt.get("champion_sha256"):
                violations.append(
                    f"V3 champion integrity (D1): round {next_id} trained "
                    f"from a champion mutated behind the round-{prev_id} veto")

    # V4 fail-closed audit (1.6)
    decision_rounds = {int(d) for d in by_round_dec}
    for rid in decision_rounds:
        if rid >= 2 and rid not in by_round_topo:
            violations.append(
                f"V4 fail-closed (1.6): decision round {rid} has NO topo "
                "measurement in logs/topo.jsonl")
    for rid in by_round_topo:
        if rid not in decision_rounds:
            violations.append(
                f"V4: topo measurement for round {rid} has NO decision entry "
                "in logs/promotion_decisions.jsonl")

    # V5 baseline uniqueness
    baselines = [r for r in topo_rows if r.get("establish_baseline")]
    if len(baselines) > 1:
        violations.append(f"V5: {len(baselines)} establish_baseline rounds")
    if topo_rows and baselines and min(ordered) != baselines[0].get("round_id"):
        violations.append("V5: the establish_baseline round is not the first")
    return violations


def render_report(topo_rows: list[dict], decisions: list[dict],
                  violations: list[str], ci: float) -> str:
    by_round_dec = {int(d["round_id"]): d for d in decisions
                    if "round_id" in d}
    lines = ["# Divergence report (PROMETHEUS-V2)", "",
             "Generated mechanically from logs/topo.jsonl + "
             "logs/promotion_decisions.jsonl. Zero rates are a valid "
             "result.", ""]

    if violations:
        lines += ["## EXPERIMENT INVALID", ""]
        lines += [f"- {v}" for v in violations]
        lines.append("")

    lines += ["## Per-round table", "",
              "| round | type | NS promote | topo veto | divergence | "
              "causes | damage classification |",
              "|---|---|---|---|---|---|---|"]
    for rid in sorted({int(r["round_id"]) for r in topo_rows
                       if "round_id" in r}
                      | {int(d["round_id"]) for d in decisions
                         if "round_id" in d}):
        topo = next((r for r in topo_rows
                     if int(r.get("round_id", -1)) == rid), None)
        dec = by_round_dec.get(rid)
        kind = (topo or {}).get("stress_mode") or "normal"
        if (topo or {}).get("establish_baseline"):
            kind = "baseline"
        ns = "missing" if dec is None else str(bool(dec.get("ns_promote")))
        veto = "missing" if topo is None else str(bool(topo.get("veto")))
        div = ("missing" if dec is None or topo is None
               else str(bool(dec.get("ns_promote")) and bool(topo.get("veto"))))
        causes = "; ".join((topo or {}).get("causes", [])) or "-"
        damage = ("missing" if dec is None
                  else str(dec.get("damage_classification") or "pending"))
        lines.append(f"| {rid} | {kind} | {ns} | {veto} | {div} | "
                     f"{causes} | {damage} |")

    for arm, label in (("normal", "Normal rounds"), ("stress", "Stress rounds")):
        rows = [r for r in topo_rows
                if bool(r.get("stress_mode")) == (arm == "stress")
                and not r.get("establish_baseline")]
        divergences = 0
        for r in rows:
            dec = by_round_dec.get(int(r["round_id"]))
            if dec and dec.get("ns_promote") and r.get("veto"):
                divergences += 1
        lo, hi = wilson_interval(divergences, len(rows), ci)
        lines += ["", f"## {label}", "",
                  f"- divergence rate: **{divergences}/{len(rows)} = "
                  f"{(divergences / len(rows)) if rows else 0.0:.2f}**",
                  f"- Wilson {int(ci * 100)}% CI: [{lo:.3f}, {hi:.3f}]",
                  "- framing: case study (n is small; the interval IS the "
                  "result, not a population rate)"]

    lines += ["", "## Interpretation limits", "",
              "- The veto is a guardrail, not a proof of quality.",
              "- Divergence without a damage verdict proves a difference "
              "of criteria, not that the veto is right.",
              "- One model, one seed, one geometry policy: the numbers "
              "describe this champion, not the technique in general."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--out", default="docs/divergence_report.md")
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    logs_dir = Path(args.logs)
    topo_rows = load_jsonl(logs_dir / "topo.jsonl")
    decisions = load_jsonl(logs_dir / "promotion_decisions.jsonl")
    if not topo_rows and not decisions:
        raise SystemExit("no V2 logs found: run the experiment rounds first")

    violations = verify_integrity(topo_rows, decisions)
    report = render_report(topo_rows, decisions, violations, args.ci)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"report written: {out_path}")
    if violations:
        for v in violations:
            print(f"VIOLATION: {v}")
        raise SystemExit("EXPERIMENT INVALID: fix the process, not the report")


if __name__ == "__main__":
    main()
