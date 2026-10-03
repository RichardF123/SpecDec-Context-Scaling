# PagedAttention v1d — pointer-table reuse + repeated benchmark

> **Historical stage record.** The scope below is the v1d boundary, not the
> current implementation boundary. The 2026-10-02 final closeout subsequently
> added BF16 `q_len=1..4` verification and asymmetric V-FP8 direct paged CUDA.
> See [the authoritative closeout](kv_optimization_closeout_2026-10-02.md).

This batch continues from the already-passing v1c state:

- Paged KV v0 storage parity: PASS
- direct paged CUDA operator parity: PASS
- real Qwen 0.5B/1.5B model parity: PASS
- SpecDec propose/repair integration parity: PASS
- first latency/memory probe completed

## Scope

v1d does **not** change the numerical attention algorithm or expand support to
V-FP8 direct attention / q_len=2..4 verification.  It only reduces metadata
overhead and makes the performance evaluation more reliable.

### Changes

1. Lazy CUDA page-pointer table cache in each `paged_v0` layer.
   - First direct use builds the tiny int64 address tables.
   - Appends inside an existing physical page reuse them.
   - Allocation of a new physical page invalidates them.
   - `trim_layer()` returns a fresh cache without inherited pointer metadata.

2. Direct model fast path skips a duplicate full consistency scan.
   - `append_current()` has already validated the cache in the same forward.
   - External/direct operator calls still validate by default.

3. Repeated benchmark.
   - alternating mode order to reduce order/thermal bias;
   - median, p10, p90;
   - page-size sweep;
   - pointer-table build/hit counters;
   - raw + summary CSV.

## Files

REPLACE:

- `specdec/engine/paged_cache.py`
- `specdec/kernels/ops.py`
- `specdec/engine/model_self.py`
- `specdec/engine/paged_attention_bench.py`

ADD:

- `specdec/engine/paged_pointer_table_test.py`
- `docs/paged_attention_v1d_runbook.md`
- `scripts/run_paged_attention_v1d.sh`

Keep the already-passing CUDA kernel/binding files unchanged.

## Run order

From repository root:

```bash
python -m pip install -e . --no-deps
rm -rf .cache/torch_extensions
```

### 1. Focused v1d pointer-table gate

```bash
python -m specdec.engine.paged_pointer_table_test
```

Expected final line:

```text
paged_pointer_table_test: PASS
```

### 2. Regression gates

```bash
python -m specdec.engine.paged_attention_operator_test
python -m specdec.engine.paged_attention_model_test
python -m specdec.engine.paged_attention_specdec_test
python -m specdec.engine.paged_cache_parity --mode base --page-size 4 --max-new-tokens 12
```

All must remain PASS.

### 3. Quick repeated benchmark

```bash
python -m specdec.engine.paged_attention_bench \
  --model small \
  --contexts 512,1024,2048 \
  --page-sizes 8,16,32 \
  --steps 24 \
  --warmup 4 \
  --repeats 3 \
  --csv paged_attention_v1d_small.csv
```

This writes:

- `paged_attention_v1d_small.csv` — aggregated summary
- `paged_attention_v1d_small_raw.csv` — every repeat

### 4. Optional long-context confirmation

If time/GPU budget allows:

```bash
python -m specdec.engine.paged_attention_bench \
  --model small \
  --contexts 1024,2048,4096 \
  --page-sizes 8,16,32 \
  --steps 32 \
  --warmup 4 \
  --repeats 5 \
  --csv paged_attention_v1d_small_long.csv
```

Optional 1.5B confirmation:

```bash
python -m specdec.engine.paged_attention_bench \
  --model target \
  --contexts 512,1024,2048 \
  --page-sizes 8,16,32 \
  --steps 24 \
  --warmup 4 \
  --repeats 3 \
  --csv paged_attention_v1d_target.csv
```

## Interpretation

Correctness gates decide whether v1d is valid.  Benchmark results decide only
performance claims.  Do not claim a speedup from one configuration unless the
repeated median and spread support it.

The primary mechanism claim remains narrower and stronger: direct page-aware
attention avoids full-history K/V materialization, and its temporary memory
cost should grow much more slowly with context length than the materialized
baseline.

If v1d passes, freeze the PagedAttention line and move to H2O/retention work.
