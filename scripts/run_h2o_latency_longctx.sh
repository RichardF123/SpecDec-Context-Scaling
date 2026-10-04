#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SpecDec-feat-paged-attention-v1

echo "============================================================"
echo " H2O 4K/8K LATENCY — CHUNKED PREFILL FIX"
echo "============================================================"

python -m pip install -e . --no-deps

# Long-context extension only. We already have valid multi-prompt 512/2K data.
# Use one deterministic prompt family and 3 paired repeats to avoid wasting GPU.
python -m specdec.engine.h2o_latency_longctx \
  --contexts 4096,8192 \
  --target-kvs base_bf16,v_fp8 \
  --prompt-families systems \
  --ratios 0.5 \
  --steps 16 \
  --page-size 16 \
  --chunk 512 \
  --repeats 3 \
  --outdir output/h2o_latency_longctx

echo
echo "============================================================"
echo " H2O_LONGCTX_LATENCY_PASS"
echo "============================================================"
