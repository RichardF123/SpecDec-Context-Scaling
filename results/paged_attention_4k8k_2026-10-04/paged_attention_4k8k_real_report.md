# Final PagedAttention 4K/8K real dual-model validation

| Context | Target KV | Direct runs | Stable | Prefill med s | Decode med s | Decode tok/s | Peak alloc GiB | Materialized | Direct=materialized |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 4096 | base_bf16 | 3/3 | PASS | 2.739 | 1.670 | 9.580 | 4.411 | PASS | 1 |
| 4096 | v_fp8 | 3/3 | PASS | 3.226 | 2.060 | 7.769 | 4.387 | PASS | 1 |
| 8192 | base_bf16 | 3/3 | PASS | 6.192 | 2.454 | 6.520 | 4.858 | PASS | 1 |
| 8192 | v_fp8 | 3/3 | PASS | 6.940 | 3.294 | 4.857 | 4.811 | PASS | 1 |

Stable=PASS requires 3/3 fresh direct runs, deterministic output/stats, direct propose+verify routing, and zero verify fallback.
H2O is disabled here; this isolates the PagedAttention 4K/8K question.
