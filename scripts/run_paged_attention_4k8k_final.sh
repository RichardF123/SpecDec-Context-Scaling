#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SpecDec-feat-paged-attention-v1

echo "============================================================"
echo " FINAL 4K/8K PAGEDATTENTION VALIDATION"
echo "============================================================"

python -m pip install -e . --no-deps

echo
echo "=== A. 4K/8K operator-path microbenchmark ==="
python -m specdec.engine.paged_attention_final_bench \
  --contexts 4096,8192 \
  --qlens 1,4 \
  --page-size 16 \
  --warmup 4 \
  --iters 20 \
  --repeats 3 \
  --csv paged_attention_final_bench_4k8k.csv

echo
echo "=== B. Real dual-model fresh-process 4K/8K SpecDec stability ==="
python -m specdec.engine.paged_attention_long_context \
  --contexts 4096,8192 \
  --target-kvs base_bf16,v_fp8 \
  --steps 16 \
  --chunk 512 \
  --page-size 16 \
  --k 4 \
  --direct-repeats 3 \
  --materialized-reference \
  --outdir output/paged_attention_4k8k_final

echo
echo "============================================================"
echo " PAGED_ATTENTION_4K8K_VALIDATION_PASS"
echo "============================================================"
