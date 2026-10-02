#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps

echo "=== regression: pointer table ==="
python -m specdec.engine.paged_pointer_table_test

echo "=== regression: paged operator ==="
python -m specdec.engine.paged_attention_operator_test

echo "=== regression: real model ==="
python -m specdec.engine.paged_attention_model_test

echo "=== regression: SpecDec paged attention ==="
python -m specdec.engine.paged_attention_specdec_test

echo "=== regression: paged v0 default ==="
python -m specdec.engine.paged_cache_parity   --mode base   --page-size 4   --max-new-tokens 12

echo "=== H2O v0 correctness ==="
python -m specdec.engine.h2o_v0_test

echo "=== H2O v0 retention/quality smoke ==="
python -m specdec.engine.h2o_v0_bench   --contexts 256,512   --budget-ratios 0.25,0.5   --steps 16   --page-size 16   --csv h2o_v0_bench.csv
