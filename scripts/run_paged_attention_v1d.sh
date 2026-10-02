#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e . --no-deps
rm -rf .cache/torch_extensions

python -m specdec.engine.paged_pointer_table_test
python -m specdec.engine.paged_attention_operator_test
python -m specdec.engine.paged_attention_model_test
python -m specdec.engine.paged_attention_specdec_test
python -m specdec.engine.paged_cache_parity --mode base --page-size 4 --max-new-tokens 12

python -m specdec.engine.paged_attention_bench \
  --model small \
  --contexts 512,1024,2048 \
  --page-sizes 8,16,32 \
  --steps 24 \
  --warmup 4 \
  --repeats 3 \
  --csv paged_attention_v1d_small.csv
