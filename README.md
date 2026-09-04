# SpecDec — from-scratch small-LLM inference engine (course project)

Phase 0 scaffold. Docs of record: `D:\RealDSH\specdec_final_plan_v3.md`,
`specdec_technical_proposal_v2.md`, `specdec_design_deliverable.md`,
`specdec_ai_runbook.md`.

## Quickstart (Step 0 of the runbook)

```powershell
# 1) create env (option A: conda / option B: venv)
conda create -n specdec python=3.11 -y ; conda activate specdec
# or:
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1

# 2) install (PyPI torch wheels bundle CUDA; verify below)
pip install -U pip
pip install -r requirements-local.txt
pip install -e .

# 3) verify the machine (Step-0 gate)
specdec env-check
```

Expected: `cuda available: True`, device `NVIDIA GeForce RTX 4060 Laptop GPU`,
VRAM ~8 GiB, capability `8.9`.

> **Windows + Triton note:** official `triton` does not ship Windows wheels.
> Either (a) run the kernel work under **WSL2 (Ubuntu)** — recommended, or
> (b) try `pip install triton-windows`. Per the go/no-go rule: if Triton is
> unavailable or slow on your machine, LOCK EAGER, report it honestly, and
> spend the L3 budget on L4/L4b/L5 — the MUST ladder does not depend on Triton.

## Commands

```powershell
specdec env-check                 # Step-0 gate
specdec run --stage L1 --seed 42  # scaffold stub until Phase 1 engine lands
```

## Package layout

```
specdec/
  cli.py        CLI: env-check, run (stub)
  config.py     RunConfig + config_hash / run_id (reproducibility chain)
  results.py    append-only results.jsonl ledger
  engine/       Phase 1+: forward, attention, KV-cache, sampling (OURS)
  kernels/      Phase 3+: Triton kernels (OURS)
  quant/        Phase 4+: int8/int4 (OURS)
  kv_quant/     Phase 4b: KV quantization (OURS)
  spec_dec/     Phase 5: draft + verify (OURS)
  bench/        Phase 7: benchmark harness
```

Honesty rule: never append fake rows to `results.jsonl` — only real measured
metrics starting in Phase 1+.
