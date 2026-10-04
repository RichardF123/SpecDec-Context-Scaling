# H2O Operational Latency Benchmark

Measures the current reference H2O score-capture + compaction path.

`H2O/standard decode > 1` means H2O is slower.

| Context | Target KV | Budget | Pairs | H2O/Std decode | Std/H2O speed | H2O tok/s | Peak alloc GiB | Resident/seen | Exact output | Stats equal |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | base_bf16 | 0.50 | 9 | 2.316x | 0.432x | 8.166 | 4.132 | 0.485 | 1.000 | 1.000 |
| 512 | v_fp8 | 0.50 | 9 | 2.382x | 0.420x | 8.327 | 4.129 | 0.485 | 1.000 | 1.000 |
| 2048 | base_bf16 | 0.50 | 9 | 3.079x | 0.325x | 5.051 | 5.078 | 0.496 | 1.000 | 1.000 |
| 2048 | v_fp8 | 0.50 | 9 | 3.111x | 0.321x | 4.052 | 5.066 | 0.496 | 1.000 | 1.000 |
| 4096 | base_bf16 | 0.50 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4096 | v_fp8 | 0.50 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Boundary

- This is an operational benchmark of the current Python/reference H2O path.
- It is not a fused/optimized H2O kernel benchmark.
- A slowdown here measures the present score-capture + cache-compaction implementation.
- If outputs or SpecDec stats differ, latency ratios are operational rather than path-identical.
