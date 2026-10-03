# H2O v2 — SpecDec committed-history integration

> **Historical stage record.** This file preserves the v2 gate and its
> then-current exclusions. The final closeout also covers the V-FP8 target,
> final quality/payload evaluation, and the paired interaction audit. H2O score
> capture and compaction remain reference/materialized implementations; no H2O
> latency improvement is claimed. See
> [the authoritative closeout](kv_optimization_closeout_2026-10-02.md).

## Scope

This batch integrates H2O with speculative decoding without allowing
speculative evidence to mutate committed retention state.

### Core invariants

1. **Absolute position != resident KV length**

   RoPE positions use committed `seen_tokens`.

2. **No pruning inside a speculative round**

   Proposal and target-verification tails remain intact until accept/mismatch
   resolution.

3. **Rejected speculative score rows are discarded**

   They are captured temporarily for evaluation but never update heavy-hitter
   state.

4. **Only committed history is eligible for H2O**

   Accepted candidates and mismatch repair tokens update state; then H2O prunes
   once at the round boundary.

5. **Rollback is resident-local**

   Because pruning never happens mid-round, rollback remains:
   `resident_prefix + accepted_prefix`.

## Files

REPLACE:
- `specdec/engine/model_self.py`
- `specdec/engine/h2o_policy.py`

ADD:
- `specdec/engine/spec_dec_h2o.py`
- `specdec/engine/h2o_specdec_test.py`
- `specdec/engine/h2o_specdec_quality.py`
- `scripts/run_h2o_v2_specdec.sh`

## Current supported configuration

- paged_v0
- BF16/base target
- BF16/base draft
- batch size 1
- k=4 tested
- direct paged q_len=1 path remains available
- target verify remains existing materialized path
- reference H2O score collection

Not yet included:
- V-FP8 target + H2O
- page-native H2O compaction
- fused CUDA score accumulation
- H2O performance/speedup claim

## Run

```bash
chmod +x scripts/run_h2o_v2_specdec.sh
bash scripts/run_h2o_v2_specdec.sh
```

### Important correctness gate

`h2o_specdec_test` first runs an oversized H2O budget.

Expected:
- standard SpecDec IDs == H2O integration IDs
- standard SpecDec stats == H2O integration stats
- target cache tensors exactly equal
- draft cache tensors exactly equal
- prune count == 0

Then it forces active pruning and checks:
- `seen_tokens > resident_tokens`
- target and draft remain within budget
- H2O state aligns with resident cache
- rejected speculative score rows are explicitly discarded
- mismatch/rollback still works after logical/resident separation

## Quality smoke

`h2o_specdec_quality.py` compares active-pruning H2O/SpecDec against the
unpruned SpecDec baseline across deterministic prompt families.

Approximate retention is allowed to change output.  The purpose is to measure
the budget/quality curve, not to demand losslessness.
