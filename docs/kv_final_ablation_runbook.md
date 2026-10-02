# Final KV Optimization Ablation

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
| C | + direct page-aware q_len=1 attention |
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

Use the already-completed PagedAttention v1d repeated benchmark for the direct
paged-attention latency result.

## Run

```bash
chmod +x scripts/run_kv_final_ablation.sh
bash scripts/run_kv_final_ablation.sh
```

## Outputs

- `kv_final_ablation_raw.csv`
- `kv_final_ablation_summary.csv`
- `kv_final_ablation_report.md`
- `kv_final_ablation_manifest.json`

The Markdown file is intended to be directly usable when writing the project
report/result section.
