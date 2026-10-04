# H2O Long-Context Operational Latency

Chunked prefill: 512

| Context | Target KV | Budget | Pairs | Status | H2O/Std decode | Std/H2O speed | H2O tok/s | Peak alloc GiB | Resident/seen | Exact output | Stats equal |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | base_bf16 | 0.50 | 3/3 | PASS | 3.186x | 0.314x | 2.185 | 4.411 | 0.498 | 1.000 | 1.000 |
| 4096 | v_fp8 | 0.50 | 3/3 | PASS | 3.239x | 0.309x | 2.591 | 4.387 | 0.498 | 1.000 | 1.000 |
| 8192 | base_bf16 | 0.50 | 3/3 | PASS | 4.206x | 0.238x | 1.631 | 4.858 | 0.499 | 1.000 | 1.000 |
| 8192 | v_fp8 | 0.50 | 3/3 | PASS | 3.465x | 0.289x | 1.453 | 4.811 | 0.499 | 1.000 | 1.000 |

Boundary: this measures the current reference H2O scoring/compaction implementation, not a fused/optimized H2O kernel.
