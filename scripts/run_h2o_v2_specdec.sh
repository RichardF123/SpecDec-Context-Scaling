#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps

echo "=== regression: H2O v0 ==="
python -m specdec.engine.h2o_v0_test

echo "=== regression: H2O v1 quality ==="
python -m specdec.engine.h2o_v1_quality_eval \
  --contexts 256,512 \
  --budget-ratios 0.25,0.5 \
  --steps 16 \
  --page-size 16 \
  --csv h2o_v1_quality_raw.csv \
  --summary-csv h2o_v1_quality_summary.csv \
  --divergences-json h2o_v1_divergences.json

echo "=== H2O v2: SpecDec committed-history correctness ==="
python -m specdec.engine.h2o_specdec_test

echo "=== H2O v2: SpecDec quality smoke ==="
python -m specdec.engine.h2o_specdec_quality \
  --contexts 128,256 \
  --ratios 0.25,0.5 \
  --steps 16 \
  --page-size 16 \
  --csv h2o_v2_specdec_quality.csv
