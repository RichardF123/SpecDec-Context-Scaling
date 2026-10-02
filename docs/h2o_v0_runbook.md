# H2O v0 — first dynamic-KV-retention batch

This starts the H2O line after PagedAttention v1d.

## What v0 does

- cumulative q_len=1 decode attention mass;
- retain heavy hitters + recent tokens;
- fixed resident KV budget;
- per-layer state;
- absolute processed-token positions remain separate from resident-cache length,
  so RoPE keeps advancing after eviction;
- paged-KV compaction and memory accounting.

## Important scope

This is a global-token GQA adaptation of H2O, not an exact per-head
reproduction. v0 intentionally uses reference K materialization for score
capture and reference materialize/gather/repack for eviction. It is a
correctness/controller prototype, not a latency optimization.

Not in v0:
- SpecDec + H2O
- V-FP8 + H2O
- q_len > 1 H2O
- CUDA score accumulation
- page-native compaction

## Files

REPLACE:
- specdec/engine/model_self.py
- specdec/engine/paged_cache.py

ADD:
- specdec/engine/h2o_policy.py
- specdec/engine/h2o_generate.py
- specdec/engine/h2o_v0_test.py
- specdec/engine/h2o_v0_bench.py
- scripts/run_h2o_v0.sh

## Run

```bash
chmod +x scripts/run_h2o_v0.sh
bash scripts/run_h2o_v0.sh
```

The script first reruns the key PagedAttention regressions, then H2O v0
correctness and a retention/quality smoke benchmark.
