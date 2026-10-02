# H2O v3 — canonical V-FP8 target combination

## Purpose

Combine the project's established asymmetric target cache:

- target: K BF16 + V FP8
- draft: BF16

with the H2O committed-history controller from v2.

## Important mechanism point

H2O importance uses QK attention mass.  In the target's V-FP8 format, K remains
BF16, so score capture does not need to dequantize V and can use the same K
importance calculation.

H2O pruning uses `paged_cache.select_indices`, which already keeps V payload
and V scale tensors token-aligned.

## Files

REPLACE:
- `specdec/engine/model_self.py`
- `specdec/engine/h2o_policy.py`
- `specdec/engine/spec_dec_h2o.py`

ADD:
- `specdec/engine/h2o_v3_combo_test.py`
- `specdec/engine/h2o_v3_combo_eval.py`
- `scripts/run_h2o_v3_combo.sh`

## Run

```bash
chmod +x scripts/run_h2o_v3_combo.sh
bash scripts/run_h2o_v3_combo.sh
```

## Gates

1. Re-run H2O v2 correctness.
2. Exact no-prune parity for canonical V-FP8 target + BF16 draft.
3. Real active H2O pruning with V-FP8 target.
4. Verify V payload and V-scale token alignment after pruning.
5. Combined quality/payload evaluation.

## Combined evaluation

For each prompt/context:

1. full BF16 target+draft baseline;
2. V-FP8 target + BF16 draft, no H2O;
3. V-FP8 target + BF16 draft + H2O.

It reports:
- quantization-only agreement vs BF16;
- combined agreement vs V-FP8 baseline;
- combined agreement vs BF16;
- total target+draft paged payload;
- payload ratios;
- H2O resident lengths.

This remains a quality + payload experiment, not an H2O speed benchmark.
Reference H2O score collection and compaction still materialize tensors.
