#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SpecDec-feat-paged-attention-v1

echo "============================================================"
echo " FINAL H2O OPERATIONAL LATENCY BENCHMARK"
echo "============================================================"

python -m pip install -e . --no-deps

python -m specdec.engine.h2o_latency_final \
  --contexts 512,2048,4096 \
  --target-kvs base_bf16,v_fp8 \
  --prompt-families systems,reasoning,code \
  --ratios 0.5 \
  --steps 16 \
  --page-size 16 \
  --repeats 3 \
  --outdir output/h2o_latency_final

echo
echo "============================================================"
echo " H2O_LATENCY_BENCH_PASS"
echo "============================================================"
