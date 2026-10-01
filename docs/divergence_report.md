# Divergence report (PROMETHEUS-V2)

Generated mechanically from logs/topo.jsonl + logs/promotion_decisions.jsonl. Zero rates are a valid result.

## Per-round table

| round | type | NS promote | topo veto | divergence | causes | damage classification |
|---|---|---|---|---|---|---|
| 1 | baseline | True | False | False | first round: measurements establish the baseline | pending |
| 2 | normal | True | False | False | - | pending |
| 3 | normal | True | False | False | - | pending |
| 4 | dup_flood | True | False | False | - | pending |
| 5 | contradiction_flood | True | True | True | R3 symbolic cycle growth: persistent H1 1483.0 > ceiling 1275.00 (champion 1020.0) | damage evidence |
| 6 | lr_spike | False | True | False | R1 structural churn: bottleneck 0.7345 > 0.15 with relative ppl gain -33.9363 < 0.02; R2 representational collapse: betti0@eps=4 < floor 36.00 (baseline 72) | pending |

## Normal rounds

- divergence rate: **0/2 = 0.00**
- Wilson 95% CI: [0.000, 0.658]
- framing: case study (n is small; the interval IS the result, not a population rate)

## Stress rounds

- divergence rate: **1/3 = 0.33**
- Wilson 95% CI: [0.061, 0.792]
- framing: case study (n is small; the interval IS the result, not a population rate)

## Interpretation limits

- The veto is a guardrail, not a proof of quality.
- Divergence without a damage verdict proves a difference of criteria, not that the veto is right.
- One model, one seed, one geometry policy: the numbers describe this champion, not the technique in general.
