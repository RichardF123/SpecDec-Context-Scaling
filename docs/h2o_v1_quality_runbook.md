# H2O v1 quality gate

## Why this batch exists

H2O v0 correctness passed, but one smoke case (`ctx=256`, 25% budget) diverged
strongly while several other cases were exact. That is not automatically a
bug: H2O is an approximate retention method. Before adding speculative-decoding
state transitions, we need to know whether the mechanism has a sensible
quality/budget curve and whether heavy-hitter retention helps relative to a
simple recent-only cache.

## This batch does NOT modify the H2O core

ADD only:

- `specdec/engine/h2o_v1_quality_eval.py`
- `scripts/run_h2o_v1_quality.sh`

The first gate also proves:

`plain paged/direct greedy == H2O score capture with no eviction`

including exact final cache tensors.

## Main comparison

At identical token budgets:

- `h2o_balanced`: 50% heavy hitters + 50% recent
- `recent_only`: 100% recent

Across three deterministic prompt families and two context lengths.

Metrics:

- token agreement with no-prune baseline
- common generated prefix
- exact generated-suffix match
- resident/payload ratio
- prune count

## Run

```bash
chmod +x scripts/run_h2o_v1_quality.sh
bash scripts/run_h2o_v1_quality.sh
```

Outputs:

- `h2o_v1_quality_raw.csv`
- `h2o_v1_quality_summary.csv`
- `h2o_v1_divergences.json`

## Decision rule for the next step

The purpose is not to require perfect agreement under aggressive pruning.

Proceed to SpecDec + committed-history H2O if:

1. no-eviction parity is exact;
2. budget enforcement remains valid;
3. quality degradation is budget/workload dependent rather than a universal
   failure;
4. H2O balanced is at least competitive with recent-only in enough cases to
   justify carrying the mechanism forward.

If H2O is consistently worse than recent-only, fix the score aggregation /
retention split before SpecDec integration.
