#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps

echo "=== regression: H2O v3 canonical combination ==="
python -m specdec.engine.h2o_v3_combo_test

echo "=== H2O v3a: controlled paired interaction audit ==="
python -m specdec.engine.h2o_v3a_interaction_audit \
  --contexts 128,256 \
  --ratios 0.25,0.5 \
  --steps 16 \
  --page-size 16 \
  --csv h2o_v3a_interaction_summary.csv \
  --json h2o_v3a_interaction_detail.json
