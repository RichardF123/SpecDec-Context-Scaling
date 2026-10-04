# PagedAttention 4K/8K validation — 2026-10-04

This directory preserves the final long-context validation supplied in
`SPECDEC_4K8K_VALIDATION_ARCHIVE_2026-10-04.zip`.

## Provenance

- Source archive SHA-256:
  `5b670fe267a8d1e6cddec91308684bfe7c09ce90321d18e0adab313e7f55c52b`
- The source archive manifest verified every listed file.
- The source environment recorded Python 3.12.14, PyTorch 2.11.0+cu128,
  CUDA runtime/toolkit 12.8, Transformers 4.57.1, NVIDIA driver 595.91.07,
  and an NVIDIA GeForce RTX 4060 with 8188 MiB reported by `nvidia-smi`.
- Model revisions are preserved under `environment/HF_MODEL_REVISIONS.txt`.
- The source directory had no Git worktree. Core source snapshots were checked
  against branch HEAD; every existing implementation file was byte-identical.
  Only the long-context harness and runner were new and were integrated into
  the repository.

## Canonical results

- `paged_attention_final_bench_4k8k.csv`: operator-path medians.
- `paged_attention_final_bench_4k8k_raw.csv`: all operator repeats.
- `paged_attention_4k8k_real_raw.csv`: canonical real-model run table.
- `paged_attention_4k8k_real_summary.csv`: canonical real-model summary.
- `paged_attention_4k8k_real_report.md`: generated report.
- `workers/`: fresh-process JSON evidence.

The canonical real-model timing table predates several worker-file reruns in
the source package. Nine of the 16 worker JSON files have timing fields that do
not match `paged_attention_4k8k_real_raw.csv`; their timestamps show they were
rewritten after the canonical CSV/report were produced. Keep the worker files
as supplemental fresh-process evidence, but use the canonical raw/summary CSVs
for the reported timing medians.

## Operator benchmark

`latency ratio` means materialized latency divided by direct latency. Values
below one mean the direct kernel was slower.

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

These values are derived directly from
`paged_attention_final_bench_4k8k.csv`. The result establishes growing
temporary-allocation savings, not a latency speedup.

## Real dual-model validation

Configuration: batch 1, Qwen2.5-1.5B-Instruct target,
Qwen2.5-0.5B-Instruct draft, SpecDec `k=4`, 16 generated tokens, page size 16,
chunked prefill 512, paged_v0, direct PagedAttention, and H2O disabled.

| Context | Target KV | Direct runs | Peak allocated GiB | Materialized reference | Output/stats parity |
|---:|---|---:|---:|---|---:|
| 4096 | BF16 | 3/3 PASS | 4.411 | PASS | 1 |
| 4096 | V-FP8 | 3/3 PASS | 4.387 | PASS | 1 |
| 8192 | BF16 | 3/3 PASS | 4.858 | PASS | 1 |
| 8192 | V-FP8 | 3/3 PASS | 4.811 | PASS | 1 |

The final direct PagedAttention path was validated through 8K for this tested
configuration. This evidence does not generalize to other model sizes,
batches, generation lengths, hardware, or contexts above 8K.

`SHA256SUMS.txt` covers the integrated directory except the manifest itself.
`SOURCE_ARCHIVE_SHA256SUMS.txt` preserves the source archive's manifest.
