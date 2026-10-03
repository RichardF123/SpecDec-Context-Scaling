# Final KV Optimization Ablation

> **Completed 2026-10-02.** This runbook is retained as the historical
> standalone execution protocol. Its runner writes the unsuffixed default
> output names listed below. The authoritative closeout copies are separate
> `_final` artifacts under
> [`results/kv_closeout_2026-10-02/`](../results/kv_closeout_2026-10-02/).

This is the freeze-point evaluation for the KV-optimization subtask.

## ADD only

- `specdec/engine/kv_final_ablation.py`
- `scripts/run_kv_final_ablation.sh`
- `docs/kv_final_ablation_runbook.md`

No core implementation file is replaced.

## Configurations

| ID | Configuration |
|---|---|
| A | BF16 contiguous SpecDec baseline |
| B | + paged KV storage |
| C | + direct page-aware attention (`q_len=1..4`) |
| D | + V-FP8 target |
| E | BF16 + H2O 50% |
| F | V-FP8 + H2O 50% |
| G | V-FP8 + H2O 25% aggressive stress |

The table is intentionally not presented as seven strictly cumulative steps:
E is the BF16-H2O branch used to separate retention from quantization, while F
is the practical combined configuration.

## Metrics

Quality versus A on identical inputs:
- exact generated-suffix match
- token agreement
- common prefix
- first divergence position

Memory:
- final allocated target + draft KV payload
- payload ratio versus A
- target/draft resident token counts
- resident / logical-seen ratio

SpecDec behavior:
- rounds
- proposed
- accepted
- mismatch rounds
- rollbacks
- H2O prune rounds
- discarded speculative score rows

## Why latency is not included

The current H2O controller still uses reference attention-score capture and
reference materialize/gather/repack compaction.  Timing it would mix algorithmic
retention behavior with intentionally unoptimized controller overhead.

Use the final PagedAttention benchmark for the direct paged-attention latency
result. It found a large attention-call temporary-allocation reduction but no
latency speedup in the reported RTX 4060 configurations.

## Run

```bash
chmod +x scripts/run_kv_final_ablation.sh
bash scripts/run_kv_final_ablation.sh
```

## Historical/default runner outputs

- `kv_final_ablation_raw.csv`
- `kv_final_ablation_summary.csv`
- `kv_final_ablation_report.md`
- `kv_final_ablation_manifest.json`

The Markdown file is intended to be directly usable when writing the project
report/result section.

## Canonical final-closeout copies

The completed closeout archived authoritative copies under
`results/kv_closeout_2026-10-02/` with these names:

- `kv_final_ablation_raw_final.csv`
- `kv_final_ablation_summary_final.csv`
- `kv_final_ablation_report_final.md`
- `kv_final_ablation_manifest_final.json`

The `_final` suffix identifies the authoritative closeout artifacts; it is not
the default naming behavior of the historical standalone runner.
