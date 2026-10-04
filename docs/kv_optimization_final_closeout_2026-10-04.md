# KV Optimization Final Closeout — 2026-10-04

## 1. Scope

This document consolidates the completed KV optimization engineering extension.
It adds the final 4K/8K PagedAttention validation and operational latency
measurements for the current H2O-inspired reference implementation to the
earlier 2026-10-02 closeout.

It does not redefine the frozen course experiment. The formal scope remains
the six main BF16 conditions plus two 8K V-only FP8 supplement conditions in
`proposal_freeze_2026-10-01.md`. Production BF16 AR-versus-SpecDec strict
token-level consistency remains a separate unresolved project-level issue.

## 2. Final implementation stack

- paged_v0 KV storage with page size 16;
- BF16 and asymmetric target V-FP8 storage, with BF16 K, FP8 E4M3FN V, and
  FP32 per-token/head V scale;
- direct page-aware CUDA attention for `q_len=1..4`;
- direct SpecDec propose, verify, and repair routing;
- BF16 draft cache;
- H2O-inspired global-token/GQA committed-history retention;
- logical seen length separated from resident cache length;
- rejected speculative rows excluded from committed H2O state.

Paged storage, direct attention, V-FP8, and H2O remain opt-in. The historical
contiguous/eager path remains the default.

## 3. Correctness status

The 4K/8K real-model validation used batch 1, a
Qwen/Qwen2.5-1.5B-Instruct target, a Qwen/Qwen2.5-0.5B-Instruct draft,
SpecDec `k=4`, 16 generated tokens, chunked prefill 512, and H2O disabled.
For both BF16 and V-FP8 target caches, all 3/3 direct runs passed at 4K and 8K.
Each condition matched its materialized reference on generated output and
SpecDec statistics. Direct propose and verify routing was exercised without
verify fallback.

This validates the final direct path through 8K for the tested configuration.
It does not cover arbitrary models, batches, generation lengths, hardware, or
contexts above 8K.

## 4. Final A–G quality/memory ablation

The accepted 2026-10-02 result remains unchanged:

| ID | Configuration | Exact | Agreement | Prefix /16 | Payload ratio | Resident ratio |
|---|---|---:|---:|---:|---:|---:|
| A | BF16 contiguous baseline | 1.000 | 1.000 | 16.00 | 1.000 | 1.000 |
| B | + Paged KV storage | 1.000 | 1.000 | 16.00 | 1.000 | 1.000 |
| C | + Direct PagedAttention | 1.000 | 1.000 | 16.00 | 1.000 | 1.000 |
| D | + V-FP8 target | 1.000 | 1.000 | 16.00 | 0.830 | 1.000 |
| E | BF16 + H2O 50% | 0.667 | 0.875 | 14.00 | 0.458 | 0.458 |
| F | V-FP8 + H2O 50% | 0.667 | 0.875 | 14.00 | 0.380 | 0.458 |
| G | V-FP8 + H2O 25% stress | 0.000 | 0.500 | 7.67 | 0.190 | 0.229 |

Paged storage preserved semantics in these cases. V-FP8 reduced payload while
preserving the tested outputs. H2O 50% provided a better memory/quality tradeoff
than 25%, but was not universally lossless. The 25% arm remains an aggressive
stress regime.

## 5. Direct PagedAttention latency and temporary memory

The final operator benchmark compares direct page-aware CUDA against the
materialized CUDA reference. `Materialized/direct` below is a latency ratio;
values below one mean direct was slower.

| Format | Context | q_len | Materialized/direct | Temporary MiB saved |
|---|---:|---:|---:|---:|
| BF16 | 4096 | 1 | 0.741x | 4.000 |
| BF16 | 4096 | 4 | 0.745x | 4.000 |
| BF16 | 8192 | 1 | 0.802x | 8.000 |
| BF16 | 8192 | 4 | 0.830x | 8.000 |
| V-FP8 | 4096 | 1 | 0.404x | 13.028 |
| V-FP8 | 4096 | 4 | 0.402x | 13.020 |
| V-FP8 | 8192 | 1 | 0.385x | 26.060 |
| V-FP8 | 8192 | 4 | 0.390x | 26.051 |

The ratios and memory differences are derived from
`paged_attention_final_bench_4k8k.csv`. Direct page-aware access removes the
full-history materialization temporary allocation, and the saving grows with
context length. The current custom direct kernel did not provide a latency
speedup on the tested RTX 4060.

Earlier v1d measurements used a different implementation and scope. They are
retained as historical engineering evidence, but the final q_len 1–4 and
V-FP8 benchmark supersedes them for current performance claims.

## 6. Formal 4K/8K PagedAttention stability validation

| Context | Target KV | Direct runs | Peak allocated GiB | Materialized | Output/stats parity |
|---:|---|---:|---:|---|---:|
| 4096 | BF16 | 3/3 PASS | 4.411 | PASS | 1 |
| 4096 | V-FP8 | 3/3 PASS | 4.387 | PASS | 1 |
| 8192 | BF16 | 3/3 PASS | 4.858 | PASS | 1 |
| 8192 | V-FP8 | 3/3 PASS | 4.811 | PASS | 1 |

These are real dual-model runs, distinct from the synthetic operator
microbenchmark in the previous section. The recorded peak allocation is a
whole-run PyTorch metric for this harness; operator temporary deltas are a
different measurement and must not be interchanged.

## 7. H2O operational latency

The 512/2048 regime used systems, reasoning, and code prompts with three paired
repeats per prompt, giving nine valid comparisons per target KV/context. The
corrected 4096/8192 regime used the systems prompt only, three paired repeats,
and chunked prefill 512. Do not pool the regimes as identically sampled data.

| Context | Target KV | Pairs | H2O/standard decode | Resident/seen |
|---:|---|---:|---:|---:|
| 512 | BF16 | 9 | 2.316x | 0.485 |
| 512 | V-FP8 | 9 | 2.382x | 0.485 |
| 2048 | BF16 | 9 | 3.079x | 0.496 |
| 2048 | V-FP8 | 9 | 3.111x | 0.496 |
| 4096 | BF16 | 3 | 3.186x | 0.498 |
| 4096 | V-FP8 | 3 | 3.239x | 0.498 |
| 8192 | BF16 | 3 | 4.206x | 0.499 |
| 8192 | V-FP8 | 3 | 3.465x | 0.499 |

At 8K BF16, the three ratios were 4.015x, 6.617x, and 4.206x. At 8K
V-FP8 they were 3.298x, 3.718x, and 3.465x. The variability is not hidden.

The first harness's 4K rows reported `NO_VALID_PAIRS` after single-shot
prefill attempts. Its raw rows contain OOM statuses, but those rows diagnose an
unsuitable harness for long contexts, not a scientific failure or unsupported
H2O configuration. The corrected chunked-prefill 4K/8K benchmark passed every
requested cell with exact paired output and equal SpecDec statistics.

The current reference score-capture and materialize/gather/repack compaction
path introduces substantial operational latency overhead. These numbers do not
measure an intrinsic theoretical lower bound of the H2O retention policy.

## 8. Interaction and composition conclusions

The earlier paired V-FP8 × H2O audit remains valid: quantization-only output was
exact in 12/12 cases; retention state or keep set changed in 12/12; nine cases
were `retention_changed_output_same`; and three aggressive 25% cases were
`retention_diverged_before_output`. This is bounded retention-mediated
interaction evidence, not proof that V-FP8 always changes H2O output or a
complete causal mechanism.

The composed system therefore establishes:

- storage and direct-route correctness for the tested PagedAttention cases;
- payload reduction from V-FP8;
- a quality/memory tradeoff from H2O retention;
- substantial latency overhead in the current H2O reference implementation.

## 9. Known limitations

- The direct kernel is a memory optimization, not a demonstrated latency win.
- V-FP8 exactness is limited to tested configurations.
- H2O 50% is not universally lossless; 25% is a stress regime.
- The H2O policy is a global-token/GQA adaptation inspired by H2O, not a claim
  of exact reproduction of the original per-head method.
- H2O score capture is not fused and may materialize K.
- H2O compaction remains a reference materialize/gather/repack path.
- H2O latency measures current implementation overhead, not intrinsic policy
  cost.
- 4K/8K stability evidence is limited to the tested batch-1, 1.5B target plus
  0.5B draft, 16-token configuration on the recorded RTX 4060.
- The separate production BF16 AR/SpecDec long-context consistency issue is
  not resolved by these KV experiments.

## 10. Reproduction commands

The archived runners are retained for provenance; the completed GPU work must
not be rerun as part of documentation maintenance:

```bash
bash scripts/run_paged_attention_4k8k_final.sh
bash scripts/run_h2o_latency_final.sh
bash scripts/run_h2o_latency_longctx.sh
```

## 11. Evidence locations

- Earlier closeout and A–G evidence: `results/kv_closeout_2026-10-02/`
- Final 4K/8K PagedAttention evidence:
  `results/paged_attention_4k8k_2026-10-04/`
- H2O operational latency evidence: `results/h2o_latency_2026-10-04/`
- Long-context harness: `specdec/engine/paged_attention_long_context.py`
- H2O latency harnesses: `specdec/engine/h2o_latency_final.py` and
  `specdec/engine/h2o_latency_longctx.py`

## 12. Final claim boundaries

Paged storage is not a speed claim. Final direct PagedAttention removes
materialization temporary allocation but is slower than the materialized
reference in the reported operator benchmark. H2O bounds resident history near
half at a 50% nominal budget, while the current reference implementation adds
substantial decode overhead. All correctness, memory, and latency conclusions
apply only to their documented configurations and evidence files.
