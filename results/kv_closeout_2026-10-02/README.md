# KV Optimization Final GPU Closeout — 2026-10-02

This directory is the stable repository archive of the completed KV
optimization engineering extension. It is separate from the formal G5 course
experiment defined by `docs/proposal_freeze_2026-10-01.md`.

## Provenance

- Source archive: `SPECDEC_FINAL_GPU_CLOSEOUT_2026-10-02.zip`
- Source archive SHA-256:
  `4cc6445201db1987236d01bbee5e014e58894ce898e1d3779013b6be29019358`
- Closeout date: 2026-10-02
- Archive verification: every entry listed by the source
  `SHA256SUMS.txt` verified successfully before import.
- Repository copy checksums: [`SHA256SUMS.txt`](SHA256SUMS.txt)

The source archive's cloud working directory had no `.git`; its
`code_snapshot/` is an experiment-time source-tree snapshot, not Git history.
No source tree was copied wholesale into this repository.

## Recorded environment

- Python 3.12.14
- PyTorch 2.11.0+cu128; torch CUDA runtime 12.8
- Transformers 4.57.3
- NVIDIA GeForce RTX 4060; compute capability `(8, 9)`
- 8188 MiB reported by `nvidia-smi`
- NVIDIA driver 595.84; `nvidia-smi` CUDA version 13.2

This is the recorded closeout GPU name. The proposal's target hardware remains
“RTX 4060 Laptop GPU 8GB”; the two descriptions are intentionally not merged.

## Canonical result files

- `paged_attention_final_bench.csv`
- `paged_attention_final_bench_raw.csv`
- `h2o_v3a_interaction_summary_final.csv`
- `h2o_v3a_interaction_detail_final.json`
- `kv_final_ablation_raw_final.csv`
- `kv_final_ablation_summary_final.csv`
- `kv_final_ablation_report_final.md`
- `kv_final_ablation_manifest_final.json`

The `environment/` subdirectory preserves the relevant runtime, package, GPU,
compiler, and model-revision snapshots. Files without the `_final` suffix from
the source package were intermediate evidence and are deliberately not copied
here.

## Claim boundaries

- Direct PagedAttention removes full-history materialization and substantially
  reduces attention-call temporary allocation/peak delta. It did not improve
  latency in the reported RTX 4060 configurations.
- H2O establishes retention behavior, quality/memory trade-offs, and SpecDec
  integration correctness. Its score capture and compaction are still
  reference/materialized paths; no H2O latency improvement is established.
- The V-FP8 × H2O audit provides bounded retention-mediated interaction
  evidence, not proof that V-FP8 always changes output or a complete causal
  mechanism.
- The A–G ablation is a small 128/256-context KV optimization study, not the
  formal 2K/4K/8K course factorial.
- Production BF16 AR-vs-SpecDec strict token-level correctness remains a
  separate unresolved/ongoing issue.
