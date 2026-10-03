# Final KV Optimization Ablation

## Setup

- Target: `Qwen/Qwen2.5-1.5B-Instruct`
- Draft: `Qwen/Qwen2.5-0.5B-Instruct`
- Contexts: `[128, 256]`
- Prompt families: `['systems', 'reasoning', 'code']`
- Generated tokens per case: `16`
- Page size: `16`
- SpecDec `k=4`
- Quality metrics are relative to configuration A on the exact same input.
- Payload is final allocated target+draft KV payload.
- H2O latency is intentionally excluded: current H2O score capture and compaction are reference implementations.

## Overall quality / payload summary

| ID | Configuration | Exact rate | Token agreement | Common prefix | Payload ratio vs A | Mean resident/seen |
|---|---|---:|---:|---:|---:|---:|
| A | BF16 contiguous baseline | 1.000 | 1.000 | 16.00/16 | 1.000 | 1.000 |
| B | + Paged KV storage | 1.000 | 1.000 | 16.00/16 | 1.000 | 1.000 |
| C | + Direct PagedAttention | 1.000 | 1.000 | 16.00/16 | 1.000 | 1.000 |
| D | + V-FP8 target | 1.000 | 1.000 | 16.00/16 | 0.830 | 1.000 |
| E | BF16 + H2O 50% | 0.667 | 0.875 | 14.00/16 | 0.458 | 0.458 |
| F | V-FP8 + H2O 50% | 0.667 | 0.875 | 14.00/16 | 0.380 | 0.458 |
| G | V-FP8 + H2O 25% (stress) | 0.000 | 0.500 | 7.67/16 | 0.190 | 0.229 |

## Memory by context

| Config | Context | Mean payload MiB | Ratio vs A | Target resident | Draft resident | Seen tokens |
|---|---:|---:|---:|---:|---:|---:|
| A | 128 | 5.625 | 1.000 | 144.0 | 144.0 | 144.0 |
| A | 256 | 10.625 | 1.000 | 272.0 | 272.0 | 272.0 |
| B | 128 | 5.625 | 1.000 | 144.0 | 144.0 | 144.0 |
| B | 256 | 10.625 | 1.000 | 272.0 | 272.0 | 272.0 |
| C | 128 | 5.625 | 1.000 | 144.0 | 144.0 | 144.0 |
| C | 256 | 10.625 | 1.000 | 272.0 | 272.0 | 272.0 |
| D | 128 | 4.671 | 0.830 | 144.0 | 144.0 | 144.0 |
| D | 256 | 8.824 | 0.830 | 272.0 | 272.0 | 272.0 |
| E | 128 | 2.500 | 0.444 | 64.0 | 64.0 | 144.0 |
| E | 256 | 5.000 | 0.471 | 128.0 | 128.0 | 272.0 |
| F | 128 | 2.076 | 0.369 | 64.0 | 64.0 | 144.0 |
| F | 256 | 4.152 | 0.391 | 128.0 | 128.0 | 272.0 |
| G | 128 | 1.038 | 0.185 | 32.0 | 32.0 | 144.0 |
| G | 256 | 2.076 | 0.195 | 64.0 | 64.0 | 272.0 |

## SpecDec behavior

| ID | Mean proposed | Mean accepted | Acceptance/proposal | Mean mismatch rounds | Mean rollbacks |
|---|---:|---:|---:|---:|---:|
| A | 16.00 | 16.00 | 1.000 | 0.00 | 0.00 |
| B | 16.00 | 16.00 | 1.000 | 0.00 | 0.00 |
| C | 16.00 | 16.00 | 1.000 | 0.00 | 0.00 |
| D | 16.00 | 16.00 | 1.000 | 0.00 | 0.00 |
| E | 17.17 | 14.83 | 0.864 | 1.17 | 1.17 |
| F | 17.17 | 14.83 | 0.864 | 1.17 | 1.17 |
| G | 23.00 | 11.67 | 0.507 | 4.33 | 4.33 |

## Interpretation boundaries

- A→B evaluates paged storage semantics/allocated payload, not a speed claim.
- Direct PagedAttention latency should be supported by the dedicated repeated v1d benchmark (`materialized_cuda` vs `direct_cuda`), not inferred from this table.
- D isolates the established V-FP8 target cache on these inputs.
- E/F use the practical 50% H2O operating point.
- G is an aggressive 25% stress setting; output changes are expected and must be reported rather than treated as a correctness failure.
- V-FP8 × H2O interaction evidence belongs to the separate paired v3a interaction audit.
