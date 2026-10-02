#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps

echo "=== regression: H2O v2 committed-history correctness ==="
python -m specdec.engine.h2o_specdec_test

echo "=== H2O v3: canonical V-FP8 target correctness ==="
python -m specdec.engine.h2o_v3_combo_test

echo "=== H2O v3: combined quality/payload evaluation ==="
python -m specdec.engine.h2o_v3_combo_eval \
  --contexts 128,256 \
  --ratios 0.25,0.5 \
  --steps 16 \
  --page-size 16 \
  --csv h2o_v3_combo.csv
