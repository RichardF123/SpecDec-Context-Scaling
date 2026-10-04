# KV Optimization Final GPU Closeout — 2026-10-02

> **Final successor (2026-10-04).** This document remains the authoritative
> record for the 2026-10-02 GPU closeout. The later 4K/8K validation and H2O
> operational-latency measurements are consolidated in
> [the 2026-10-04 final closeout](kv_optimization_final_closeout_2026-10-04.md).
> They extend rather than replace the accepted A–G results.

## Status and scope

The RTX 4060 GPU closeout for the KV optimization engineering extension is
complete. Canonical data is archived in
[`results/kv_closeout_2026-10-02/`](../results/kv_closeout_2026-10-02/).

This closeout does **not** redefine the frozen course study. The formal scope in
[`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md) remains six
main conditions (2K/4K/8K × AR/Spec, base BF16) plus two 8K V-only FP8
supplement conditions. The closeout is a supplementary KV optimization study.

Production BF16 AR-vs-SpecDec strict token-level correctness remains a
separate unresolved/ongoing issue. Nothing in this closeout resolves or
supersedes that investigation.

## Recorded closeout environment

| Item | Recorded value |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| torch CUDA runtime | 12.8 |
| Transformers | 4.57.3 |
| GPU | NVIDIA GeForce RTX 4060 |
| Compute capability | `(8, 9)` |
| GPU memory (`nvidia-smi`) | 8188 MiB |
| NVIDIA driver | 595.84 |
| `nvidia-smi` CUDA version | 13.2 |

The proposal target remains “RTX 4060 Laptop GPU 8GB.” The closeout environment
reported “NVIDIA GeForce RTX 4060”; this document does not infer the “Laptop”
qualifier.

## Final PagedAttention state

The completed implementation covers paged KV storage, BF16 direct paged CUDA
attention for `q_len=1..4`, and direct decode/propose/verify/repair routes,
including SpecDec target verification at `q_len=2..4`. It also covers the
asymmetric target V-FP8 direct route: K is BF16, V is FP8 E4M3FN, and scale is
FP32 per token/head. K/V/V-scale CUDA pointer tables are reused. Real-model and
SpecDec direct-route parity gates completed. Prefill remains on the existing
materialized/fused path and is outside the direct paged decode/block claim.

The final microbenchmark compared direct page-aware CUDA against materialized
CUDA for contexts 512/1024/2048, `q_len=1/4`, BF16/V-FP8, with three repeats.
Raw parity error was zero in every recorded row. At context 2048:

| Format | q_len | Materialized temporary MiB | Direct temporary MiB | Reduction | Materialized ms | Direct ms |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 2.005859375 | 0.005859375 | 99.708% | 0.412722 | 0.587286 |
| BF16 | 4 | 2.0234375 | 0.0234375 | 98.842% | 0.427825 | 0.607743 |
| V-FP8 | 1 | 6.5185546875 | 0.005859375 | 99.910% | 0.938067 | 1.960299 |
| V-FP8 | 4 | 6.52734375 | 0.0234375 | 99.641% | 0.741228 | 2.030486 |

“Temporary MiB” is the attention-call temporary allocation / peak delta, not
whole-system GPU VRAM. Across all 12 reported format/context/query
configurations, direct CUDA was slower than the materialized CUDA reference.
The verified benefit is avoidance of full-history materialization and a large
attention-path temporary-allocation reduction, **not** latency speedup.

## Final A–G ablation

This small closeout ablation used target `Qwen/Qwen2.5-1.5B-Instruct`, draft
`Qwen/Qwen2.5-0.5B-Instruct`, contexts 128/256, systems/reasoning/code prompt
families, 16 generated tokens, page size 16, and SpecDec `k=4`. There were six
exact-input cases per configuration. It is not the formal 2K/4K/8K factorial.

| ID | Configuration | Exact rate | Token agreement | Mean common prefix | Payload ratio | Resident/seen ratio |
|---|---|---:|---:|---:|---:|---:|
| A | BF16 contiguous baseline | 1.0000000 | 1.000 | 16.0000 | 1.00000000 | 1.00000000 |
| B | + paged KV storage | 1.0000000 | 1.000 | 16.0000 | 1.00000000 | 1.00000000 |
| C | + direct PagedAttention | 1.0000000 | 1.000 | 16.0000 | 1.00000000 | 1.00000000 |
| D | + V-FP8 target | 1.0000000 | 1.000 | 16.0000 | 0.83046875 | 1.00000000 |
| E | BF16 + H2O 50% | 0.6666667 | 0.875 | 14.0000 | 0.45751634 | 0.45751634 |
| F | V-FP8 + H2O 50% | 0.6666667 | 0.875 | 14.0000 | 0.37995302 | 0.45751634 |
| G | V-FP8 + H2O 25% stress | 0.0000000 | 0.500 | 7.6667 | 0.18997651 | 0.22875817 |

A→B establishes paged-storage semantic parity, not speed. B→C preserves output
parity in this ablation; latency comes from the independent microbenchmark.
V-FP8 alone preserved exact output in these six cases while reducing combined
target+draft final payload to about `0.830×` baseline. H2O 50% reduced resident
KV but changed output in two cases; it is not lossless. V-FP8 + H2O 50%
reduced combined payload to about `0.380×` with the same measured quality
change. The 25% arm is an aggressive stress condition with substantial quality
degradation, not a lossless operating point.

## V-FP8 × H2O paired interaction audit

The final audit contains 12 paired cases. Quantization-only output was exact in
`12/12`; retention state or keep set changed in `12/12`. Nine cases were
classified `retention_changed_output_same`. Three were classified
`retention_diverged_before_output`, all under aggressive ratio `0.25`.

This shows that V-FP8 can change the H2O attention-score/retention trajectory.
In some aggressive-retention cases, retention divergence preceded output
divergence, providing retention-mediated interaction evidence. Most changed
retention sets still produced the same 16-token output. This audit does not
show that V-FP8 always changes H2O output and does not prove a complete causal
mechanism.

## H2O claim boundary

The closeout covers heavy-hitter + recent retention, committed-history
integration, proposal/verify/repair handling, exclusion of rejected speculative
score rows from committed state, logical-seen versus resident length, round
boundary pruning, BF16, V-FP8 target, SpecDec integration, quality/payload
evaluation, and the paired interaction audit.

H2O score capture is still a reference/materialized implementation, and
compaction remains materialize → gather → repack. H2O results establish
retention behavior, quality/memory trade-offs, and SpecDec integration
correctness; they do not establish an H2O latency improvement.

## Provenance and authoritative files

The source archive `SPECDEC_FINAL_GPU_CLOSEOUT_2026-10-02.zip` has SHA-256
`4cc6445201db1987236d01bbee5e014e58894ce898e1d3779013b6be29019358`.
Every source entry in its `SHA256SUMS.txt` verified before import. Files with a
`_final` suffix are authoritative; older files are intermediate evidence.

The archive's cloud directory had no `.git`. Its `code_snapshot/` is therefore
an experiment-time source snapshot, not Git history. Only canonical results,
environment evidence, an executed historical runbook, and documentation-level
consistency corrections were imported; no source tree was overlaid.
