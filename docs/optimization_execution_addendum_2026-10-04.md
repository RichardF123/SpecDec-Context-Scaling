# Optimization Execution Addendum — 2026-10-04

## Scope and provenance

This addendum integrates three completed GPU experiment archives into the
`feat/paged-attention-v1` engineering line. It is supplementary engineering evidence and does
**not** redefine the frozen 6+2 course experiment or resolve the separate
production BF16 AR-vs-SpecDec strict-consistency issue.

Integration target at preparation time:

- branch: `feat/paged-attention-v1`
- GitHub head: `c2640d74d74de8196b6026d2c229506f3c4ec673`

The three source archives passed their own internal SHA256 manifests before
integration:

- Phase12: `cc03a341c0be5fcd46e4c46709ba7d6afae4e38b65a9b00093b7a5a3487be363`
- Phase13: `cc5958e04bb60cc1e2d51265324f02b522653281ad1b2da4b6a349439bcc7992`
- Pure FA2: `23b6960766e4697465a5ec9aff842d7febe50863705ee1616e1d2b5d589d1240`

The archives do not record a Git commit SHA for the exact runtime checkout, so
the hashes above are the provenance anchors for these experiment snapshots.

## 1. Phase12 — H2O execution optimization

The H2O **policy is unchanged**. Phase12 changes execution only:

1. page-native compaction replaces materialize → gather → repack;
2. score capture is integrated with the paged attention kernel path, while
   still revisiting K to emit normalized token mass. It is not described as a
   zero-extra-QK single-pass fusion.

The correctness gate passed for BF16 and V-FP8 compaction and for real SpecDec
reference-vs-optimized output, statistics, and retained positions.

Optimized H2O is faster than the previous reference implementation, but it is
still slower than standard direct SpecDec in the tested harness.

| Target KV | Context | Optimized vs reference | Optimized / standard | Resident / seen |
|---|---:|---:|---:|---:|
| base_bf16 | 512 | 1.303x | 1.801x | 0.485 |
| base_bf16 | 2048 | 1.459x | 1.890x | 0.496 |
| base_bf16 | 4096 | 1.938x | 1.900x | 0.498 |
| base_bf16 | 8192 | 2.017x | 2.055x | 0.499 |
| v_fp8 | 512 | 1.580x | 1.531x | 0.485 |
| v_fp8 | 2048 | 1.495x | 2.043x | 0.496 |
| v_fp8 | 4096 | 1.892x | 1.733x | 0.498 |
| v_fp8 | 8192 | 1.811x | 1.892x | 0.499 |

### Phase12 tiled-attention feasibility gate

The existing direct PagedAttention already used online softmax. Phase12 tested
whether the repository's mature contiguous Triton tiled online-softmax kernel,
after paying paged-KV materialization cost, could beat current page-native
direct attention end-to-end.

The pre-registered q_len=4 target criterion produced **0/3
wins**, so the real SpecDec Triton probe was correctly skipped. Contiguous
Triton compute was faster than the original paged-direct kernel at long
contexts, but paged → materialize → Triton was slower end-to-end. This result
motivated a page-native tiled kernel rather than a materialize-first backend.

## 2. Phase13 — custom Flash-style page-native PagedAttention

Phase13 adds a **custom Flash-style / tiled online-softmax page-native CUDA
kernel**. It is not the official Dao-AILab FlashAttention package.

The final route keeps K/V paged and adds `flash_paged_cuda` for q_len 1..4,
with BF16 and asymmetric V-FP8 target support. The operator gate passed.

For BF16 target q_len=4, the Phase13 microbenchmark reported:

| Context | Old paged-direct | Flash-paged | Kernel speedup |
|---:|---:|---:|---:|
| 2048 | 0.5548 ms | 0.5019 ms | 1.105x |
| 4096 | 1.0654 ms | 0.9642 ms | 1.105x |
| 8192 | 2.1210 ms | 1.9102 ms | 1.110x |

Real SpecDec summary:

| Target KV | Context | Direct tok/s | Flash-paged tok/s | Summary speedup | Exact | Stats |
|---|---:|---:|---:|---:|---:|---:|
| base_bf16 | 2048 | 11.209 | 13.029 | 1.162x | 1.000 | 1.000 |
| base_bf16 | 4096 | 9.503 | 9.676 | 1.018x | 1.000 | 1.000 |
| base_bf16 | 8192 | 7.302 | 8.349 | 1.143x | 1.000 | 1.000 |
| v_fp8 | 2048 | 11.566 | 12.144 | 1.050x | 1.000 | 1.000 |
| v_fp8 | 4096 | 8.076 | 8.838 | 1.094x | 1.000 | 1.000 |
| v_fp8 | 8192 | 3.124 | 5.434 | 1.739x | 1.000 | 1.000 |

The raw paired runs are retained. Some individual repeats were noisier than the
summary and included regressions, especially at shorter BF16 contexts; do not
upgrade the table into a claim that every repeat is faster. The strongest
Phase13 result is that the page-native tiled kernel materially improves the old
direct page traversal while preserving the tested output/statistics.

## 3. Pure official FlashAttention-2

This is a separate BF16 backend using the official `flash-attn` package with a
preallocated **contiguous** KV cache and `flash_attn_with_kvcache`. It is
deliberately not paged and does not claim V-FP8/H2O support.

The tested runtime archive records FlashAttention 2.8.3 on RTX 4060 / sm_89.
The operator correctness gate passed.

Target q_len=4 microbenchmark:

| Context | Paged direct | Official FA2 KV-cache | Speedup vs paged |
|---:|---:|---:|---:|
| 2048 | 0.5542 ms | 0.0808 ms | 6.856x |
| 4096 | 1.0658 ms | 0.1080 ms | 9.868x |
| 8192 | 2.0923 ms | 0.1617 ms | 12.941x |

Real SpecDec:

| Context | Paged tok/s | Pure FA2 tok/s | Decode speedup | Exact | Stats |
|---:|---:|---:|---:|---:|---:|
| 2048 | 12.883 | 21.872 | 1.698x | 1.000 | 1.000 |
| 4096 | 9.208 | 22.244 | 2.416x | 1.000 | 1.000 |
| 8192 | 7.343 | 26.777 | 3.647x | 1.000 | 1.000 |

Pure FA2 is the strongest BF16 latency result in these three experiment
archives, but it answers a different systems question from the paged/V-FP8
line: it trades page-native storage and custom V-FP8 compatibility for a mature
contiguous FlashAttention inference backend.

## 4. Integration decision

- **Keep Phase12 H2O execution changes** as the current H2O implementation.
  They reduce reference overhead without changing the frozen policy, but H2O
  remains slower than standard direct SpecDec in the tested implementation.
- **Keep Phase13 `flash_paged_cuda` as an opt-in paged route.** It supersedes
  the older direct kernel for the tested Flash-paged experiment but is still a
  custom kernel, not official FlashAttention.
- **Keep Pure FA2 as a separate opt-in BF16 runtime/benchmark backend.** It is
  the performance reference for contiguous BF16 KV and should not be described
  as V-FP8, paged, or H2O.
- Historical result files are not rewritten. The three final source ZIPs are
  retained under `results/` as immutable provenance archives.

## 5. Code merge rule

Phase13 was executed after Phase12. For the four files present in both archives
(`model_self.py`, `ops.py`, `build.py`, `bind.cpp`), the Phase13 snapshot is
used because it retains the Phase12 hooks and adds the Phase13 Flash-paged
route. Phase12-only H2O/page-native files are taken from the Phase12 archive.
Pure FA2 adds independent Python modules and does not replace these core files.

The old experiment installer/restore/package wrappers are intentionally **not**
promoted as final top-level repository entrypoints. Their exact snapshots remain
inside the immutable result archives. This avoids committing rollback tools
that can overwrite the newly integrated core and avoids making the optional
FlashAttention installer a mandatory project dependency.

## 6. Claim boundaries

These are engineering-extension results on the recorded RTX 4060 setup and
their documented model/context configurations. They do not resolve or replace
the frozen formal experiment, do not prove general performance across hardware
or models, and do not make H2O universally lossless.
