# PagedAttention v1c bundle

> **Historical stage record.** This runbook describes the v1c support boundary
> at that time. It is superseded by the 2026-10-02 final closeout, which covers
> BF16 and asymmetric V-FP8 direct paged CUDA for `q_len=1..4`, including
> SpecDec target verification. See
> [the authoritative closeout](kv_optimization_closeout_2026-10-02.md).

This bundle continues from the already-passing v1b state.

## What this batch adds

1. Phase-gated direct paged attention
   - direct BF16 q_len=1 remains enabled for: decode / propose / repair / other
   - target verify is explicitly excluded, even when kp == 1
   - prefill is explicitly excluded
   - V-FP8 and multi-token calls remain on the existing materialized path

2. Explicit SpecDec attention phases
   - prefill
   - propose
   - verify
   - repair

3. Optional final-cache return for strict tests
   - `spec_decode_greedy(..., return_caches=False)` preserves existing callers
   - tests may use `return_caches=True`

4. SpecDec parity gate
   - canonical target=v_fp8, draft=base_bf16
   - all-BF16 target/draft
   - compares generated IDs, full SpecDec stats, final target/draft cache tensors
   - checks routing: propose/repair may be direct; verify/prefill must not be direct

5. Whole-model decode benchmark
   - materialized_cuda vs direct_cuda
   - latency, tokens/s, and extra peak allocated CUDA memory
   - CSV output option

## Files to replace/add

REPLACE:
- `specdec/engine/model_self.py`
- `specdec/engine/spec_dec.py`

ADD:
- `specdec/engine/paged_attention_specdec_test.py`
- `specdec/engine/paged_attention_bench.py`

Do NOT replace the already-passing v1a/v1b files:
- `specdec/kernels/cuda/paged_attention.cu`
- `specdec/kernels/cuda/bind.cpp`
- `specdec/kernels/build.py`
- `specdec/kernels/ops.py`
- `specdec/engine/paged_attention_operator_test.py`
- `specdec/engine/paged_attention_model_test.py`
- `specdec/engine/paged_cache.py`
- `specdec/engine/paged_cache_parity.py`

## Run order

From repository root:

```bash
python -m pip install -e . --no-deps
rm -rf .cache/torch_extensions
```

### Gate 1 — operator regression

```bash
python -m specdec.engine.paged_attention_operator_test
```

Expected: PASS.

### Gate 2 — real-model regression

```bash
python -m specdec.engine.paged_attention_model_test
```

Expected: 0.5B and 1.5B PASS.

### Gate 3 — paged-v0 default regression

```bash
python -m specdec.engine.paged_cache_parity   --mode base   --page-size 4   --max-new-tokens 12
```

Expected: contiguous == paged_v0 PASS.

### Gate 4 — NEW SpecDec integration gate

```bash
python -m specdec.engine.paged_attention_specdec_test
```

Expected:
- canonical-vfp8-target PASS
- all-bf16 PASS
- `direct_by_phase` contains propose, and repair when mismatch occurs
- `direct_by_phase` must NOT contain verify or prefill
- final `paged_attention_specdec_test: PASS`

### Gate 5 — quick benchmark

```bash
python -m specdec.engine.paged_attention_bench   --model small   --contexts 128,512,1024   --steps 16   --page-size 16   --csv paged_attention_bench_small.csv
```

For a fuller run:

```bash
python -m specdec.engine.paged_attention_bench   --model small   --contexts 256,512,1024,2048   --steps 32   --page-size 16   --csv paged_attention_bench_small_full.csv
```

Optional 1.5B run:

```bash
python -m specdec.engine.paged_attention_bench   --model target   --contexts 128,512,1024   --steps 16   --page-size 16   --csv paged_attention_bench_target.csv
```

## Historical v1c claims if its gates pass

- paged_v0 storage semantics remain preserved.
- BF16 q_len=1 page-aware CUDA attention matches the materialized CUDA path in operator and real-model tests.
- SpecDec can use direct page-aware attention for eligible single-token propose/repair calls while keeping prefill/verify on the validated reference/materialized path.
- Final SpecDec outputs, statistics, and caches match the materialized CUDA baseline in the included tests.

Do NOT claim a speedup or memory improvement until benchmark output actually shows it.
Do NOT call V-FP8 direct paged attention implemented; it is intentionally not included in this batch.
Do NOT call q_len=2..4 page-aware verify implemented; it is intentionally excluded.
