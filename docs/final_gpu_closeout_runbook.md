# Final 4060 Closeout Run — historical execution runbook

> **Completed historical procedure.** This runbook came from the verified
> `SPECDEC_FINAL_GPU_CLOSEOUT_2026-10-02.zip`. It documents how the already
> completed closeout was executed; it is not a request to rerun GPU work. The
> authoritative outcome is
> [KV Optimization Final GPU Closeout](kv_optimization_closeout_2026-10-02.md).

This was the last GPU round for the KV/PagedAttention subtask.

## Final scope

The bundle closed the two remaining optimized attention-path gaps:

1. BF16 direct paged attention expanded from `q_len=1` to `q_len=1..4`,
   including SpecDec target verification.
2. The canonical asymmetric target V-FP8 format received a direct paged CUDA
   path: K stays BF16; V stays E4M3FN in pages; FP32 scale pages are read
   directly.

The V-FP8 kernel BF16-rounds dequantized V values before accumulation to match
the existing materialized reference path.

H2O score collection and H2O compaction intentionally remain reference
implementations. This closeout does not claim H2O latency speedup.

## Historical command

The source package instructed the operator to extract the closeout bundle at
the repository root and run:

```bash
chmod +x scripts/run_final_gpu_closeout.sh
bash scripts/run_final_gpu_closeout.sh
```

That runner applied the core patch, rebuilt the extension, ran final gates,
refreshed the interaction audit and A–G ablation, benchmarked BF16 and V-FP8
direct paths, and created the shutdown archive.

## Historical success condition

The closeout required the runner to print `FINAL_GPU_CLOSEOUT_PASS` and create
`/workspace/SPECDEC_FINAL_GPU_CLOSEOUT_2026-10-02.zip`. The resulting archive
has since been downloaded and checksum-verified. Do not execute this historical
procedure as part of documentation maintenance.
