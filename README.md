# SpecDec — from-scratch small-LLM inference engine (course project)

> From-scratch Qwen2-family inference engine + KV-cache decoding + optimization
> ladder (quantization → speculative decoding), measured on an 8 GB laptop GPU.
> Docs of record: `docs/plan_v6.md`, `docs/stage_report_l1_l2.md`, `docs/data_ledger.md`.

## Project status

- ✅ **L1 — from-scratch forward** (RMSNorm / RoPE / GQA attention / SwiGLU / 24-layer
  assembly / tied lm_head): alignment gate **max|Δ| = 0.000** vs transformers (fp32).
- ✅ **L2 — KV-cache decode + greedy generate**: cache-equivalence ≈ 2e-5 (fp noise),
  **generation alignment PASS** (full token sequences equal official model math).
- 🔄 L4 (int8/int4 quantization) is next; L4b (KV-quant) / L5 (speculative decoding)
  follow; CUDA kernel rail is a controlled bonus (see `docs/plan_v6.md`).
- Performance baseline (Qwen2.5-0.5B): CPU fp32 decode ≈ 9.2 tok/s; GPU fp16 decode
  ≈ 10.6 tok/s (latency-bound, naive decode — the baseline later stages improve on).

## Quickstart

```powershell
# 1) env (option A: conda / option B: venv)
conda create -n specdec python=3.11 -y ; conda activate specdec
# or:
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1

# 2) install (CUDA build REQUIRED — CPU wheel silently gives cuda=False)
pip install -U pip
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-local.txt
pip install -e .

# 3) verify machine (Step-0 gate)
python -m specdec.cli env-check
```
Expected: `cuda available: True`, `NVIDIA GeForce RTX 4060 Laptop GPU`, VRAM ~8 GiB.

> If HF downloads are slow/fail: `set HF_HOME=D:\hf-cache` and
> `set HF_ENDPOINT=https://hf-mirror.com` before the acceptance commands.
> Windows note: official `triton` has no Windows wheels — kernel work (L3 rail)
> goes under WSL2 or stays eager per the go/no-go rule (see `docs/plan_v6.md`).

## Acceptance commands (run from repo root, venv active)

```powershell
python -m specdec.engine.align_ours          # L1 gate — expect 0.000, PASS
python -m specdec.engine.test_cache_equiv    # L2 cache==full — expect < 1e-3, PASS
python -m specdec.engine.align_generate      # L2 generation alignment — expect PASS
python -m specdec.engine.bench_basic         # perf baseline (CPU + GPU if present)
```

## Docs

- `docs/plan_v6.md` — authoritative project plan (MUST/SHOULD, gates, timeline)
- `docs/stage_report_l1_l2.md` — phase summary (decisions, debugging saga, lessons)
- `docs/data_ledger.md` — measured-data ledger (correctness + perf, with conditions)
- `docs/L1_code_teaching.md` — line-by-line walkthrough of `engine/model_self.py`
- `docs/debug_lesson_alignment.md` — the differential-debugging methodology

## Package layout

```
specdec/
  cli.py          CLI: env-check, run
  config.py       RunConfig + config_hash / run_id (reproducibility chain)
  results.py      append-only results.jsonl ledger
  engine/         model_self.py (OURS) + align/test/bench harnesses
  kernels/        (L3 rail, optional)
  quant/          Phase 4: int8/int4 (OURS)
  kv_quant/       Phase 4b: KV quantization (OURS)
  spec_dec/       Phase 5: speculative decoding (OURS)
  bench/          Phase 7: benchmark harness
```

Honesty rule: never append fake rows to `results.jsonl` — only real measured
metrics. Engine code is written by the team; only weight loading / tokenizer /
test harnesses borrow libraries (OURS vs LIB boundary is documented in plan_v6).
