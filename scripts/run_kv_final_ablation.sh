#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps

echo "=== quick regression: canonical V-FP8 + H2O combination ==="
python -m specdec.engine.h2o_v3_combo_test

echo "=== final KV optimization ablation ==="
python -m specdec.engine.kv_final_ablation \
  --contexts 128,256 \
  --steps 16 \
  --page-size 16 \
  --raw-csv kv_final_ablation_raw.csv \
  --summary-csv kv_final_ablation_summary.csv \
  --report kv_final_ablation_report.md \
  --manifest kv_final_ablation_manifest.json
