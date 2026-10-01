# V2 Experiment Protocol: Deliberate Divergence (V2.1)

The central deliverable of PROMETHEUS-V2. Written before execution and
executed once; deviations must be committed with an explanation and
invalidate the affected measurements. Integrity is enforced MECHANICALLY
by scripts/divergence_report.py, not by good intentions.

## Hypothesis under test

The topological veto is informative: it blocks promotion in rounds where
the inherited NS metrics (holdout perplexity, bigram diversity) see no
problem, more often under deliberately degraded training than under
normal rounds. If it never fires, or fires exactly when NS already
rejects, that is the honest result and is reported identically.

## Sensor calibration prerequisite (A3)

BEFORE any formal round: run the Fase 1-V2 stability verification on the
current champion (5 subsample seeds, pairwise self-bottleneck matrix) and
commit artifacts/topo/self_noise.json. The churn threshold must exceed
the p95 of that distribution; tests/test_topo_stability.py enforces it.
An uncalibrated sensor invalidates everything downstream.

## Phase P — pilots (B4)

For each stress mode: one 1-epoch run, NO gate decision, NO threshold
edits. Purpose: verify the mode moves SOMETHING measurable (internal
metric or symbolic census). An inert mode is replaced BEFORE the formal
rounds, with the replacement documented. A pilot that changes no internal
measurement predicts a redundant formal round.

## Phase N — normal rounds N1-N3 (calibration window, B2)

Full rounds on the inherited pipeline. Threshold adjustments are allowed
ONLY in this window, each one a commit with its rationale. These rounds
also measure the false-veto rate of an honest sensor.

## Freeze gate (B1)

After N3: commit "chore(v2): freeze thresholds before stress". From S1
on, ANY config change invalidates the experiment — detected mechanically:
every logs/topo.jsonl line carries config_sha256, and the divergence
report flags drift as EXPERIMENT INVALID. Retuning after seeing stress
results requires restarting the schedule from N1 and is marked in the
report.

## Phase S — stress rounds S1-S3

| Round | Mode                | Degradation                             | Primary rule targeted          |
|-------|---------------------|------------------------------------------|--------------------------------|
| S1    | dup_flood           | 90% near-duplicate corpus                | R2 (representational collapse) |
| S2    | contradiction_flood | planted is_a contradictions in triplets  | R3a (directed-cycle census)    |
| S3    | lr_spike            | lr x multiplier, 10x shorter corpus      | R1 (structural churn)          |

Guarantees, verified mechanically by the report:

- D1: stress rounds train on stress_candidate.pt; champion.pt changes
  only through a gate promotion. After a vetoed round, the next round's
  champion hash must equal the vetoed round's (V3).
- D2: every round logs treatment_sha256 (corpus + triplets + lr
  manifest); stress rounds must differ from the normal condition (V2).

## Measurements

Per round, two JSONL entries (schemas in divergence_report.py): full
topological evidence (bottlenecks, betti0, R3a census, R3b count as
observation) in logs/topo.jsonl; gate decisions in
logs/promotion_decisions.jsonl. Cached diagrams under artifacts/topo/.

## Report and statistics (B3)

make divergence-report renders docs/divergence_report.md: per-round
table and divergence rates with exact Wilson intervals. With n = 3 per
arm the intervals are wide by construction (33% spans roughly [1%, 91%]):
read this as a CASE STUDY. Claim language is forbidden beyond what the
intervals support.

## Damage classification (B5)

A divergence proves a difference of criteria, not that the veto is right.
Each divergent round is classified post hoc with evidence EXTERNAL to the
gate: perplexity on a secondary heldout drawn from the stress corpus, and
qualitative inspection of generations. The verdict — "damage evidence" or
"no verdict" — is recorded in promotion_decisions.jsonl ->
damage_classification and rendered by the report. Unclassified rounds
render as 'pending', never silently resolved.

## Honest-negative protocol

If divergence rates are zero: state it, commit the per-round diagrams
proving the sensor was live (diagrams computed, distances measured,
self-noise distribution documented), and discuss candidate causes
(thresholds too loose, stress modes too weak, metrics not actually
superficial at this model scale). Do NOT retune and rerun silently: a
retuned threshold restarts the schedule from N1 and is marked in the
report.
