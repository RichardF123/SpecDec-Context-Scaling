# H2O v3a — controlled V-FP8 × H2O interaction audit

> **Final-status note (2026-10-02).** The paired audit has completed. The
> canonical outputs are the `_final` files archived under
> [`results/kv_closeout_2026-10-02/`](../results/kv_closeout_2026-10-02/).

## Why this audit is necessary

The earlier v2 and v3 smoke tests used different prompt strings. Therefore,
comparing a v2 BF16-H2O row directly with a v3 V-FP8-H2O row does **not**
establish an interaction effect.

The v3 combined measurements themselves remain valid. What was not yet valid
was the causal inference that V-FP8 changed the H2O retention trajectory.

This audit fixes that by running the two H2O arms on the **same exact prompt,
context, budget, model weights, page size, k, and generation length**.

## Files

REPLACE:
- `specdec/engine/spec_dec_h2o.py`

ADD:
- `specdec/engine/h2o_v3a_interaction_audit.py`
- `scripts/run_h2o_v3a_interaction_audit.sh`
- `docs/h2o_v3a_interaction_audit_runbook.md`

The replacement only adds an optional observation-only prune trace callback.
With tracing disabled, H2O selection behavior is unchanged.

## What is compared

For each exact prompt/context pair:

1. BF16 target SpecDec, no H2O
2. V-FP8 target SpecDec, no H2O
3. BF16 target + H2O
4. V-FP8 target + H2O

The H2O arms use identical heavy/recent budgets.

## Trace evidence

At each committed round/layer, the audit records:

- absolute resident positions before pruning
- accumulated H2O scores
- heavy-hitter ranking
- recent positions
- final keep positions
- evicted positions
- generated-prefix length at the prune boundary

Then it reports:

- first score-state difference
- first heavy-ranking difference
- first keep-set difference
- keep-set Jaccard overlap
- first generated-token difference
- whether keep-set divergence occurred early enough to affect that output token

## Interpretation

Strong interaction evidence requires:

1. V-FP8-only and BF16-only outputs are still identical on the paired input;
2. BF16-H2O and V-FP8-H2O retention sets diverge;
3. that retention divergence occurs before the first H2O output divergence.

If condition 1 fails, quantization itself is a confounder.
If output changes but no earlier keep-set difference is observed, the proposed
retention-mediated explanation is not established by this audit.

## Run

```bash
chmod +x scripts/run_h2o_v3a_interaction_audit.sh
bash scripts/run_h2o_v3a_interaction_audit.sh
```

Outputs:

- `h2o_v3a_interaction_summary.csv`
- `h2o_v3a_interaction_detail.json`

## Final result

All 12 paired cases retained quantization-only exact output (`12/12`), while
all `12/12` showed a retention-state or keep-set change. Classification was:

- `retention_diverged_before_output`: 3 cases, all at aggressive ratio `0.25`;
- `retention_changed_output_same`: 9 cases.

This is retention-mediated interaction evidence: V-FP8 can change the H2O
score/retention trajectory, and in some aggressive-retention cases the
retention divergence precedes output divergence. It does not prove that V-FP8
always changes H2O output or establish a complete causal mechanism.
