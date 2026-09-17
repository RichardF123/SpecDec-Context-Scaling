# FlashAttention × G4 RTX-4060 runbook

This runbook deliberately has stop gates.  Do not run tuning or the final
systems ablation when an earlier correctness stage fails.

## Frozen references

- integration base: `main-fresh@d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897`
- Flash reference: PR #2 `b1904b5d867dd6eb7262f733d1818c940145c819`
- branch: `experiment/flash-g4-integration`
- target: Qwen2.5-1.5B-Instruct
- draft: Qwen2.5-0.5B-Instruct
- required device for performance decisions: RTX 4060 8GB

Run from the repository root.  Keep the same PyTorch/CUDA/Triton environment
for every paired comparison and save `output/flash_g4/` after each stage.

## Stage 0 — environment and eager G4 regression

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
import triton
print("triton:", triton.__version__)
PY

python -m specdec.engine.g4_b5_eos_branch_tests
python -m specdec.engine.g4_b4_cache_tests
python -m specdec.engine.g4_b5_helper_tests
python -m specdec.engine.g4_b5_smoke
```

Stop if an existing eager test regresses.  The integration does not modify
`spec_dec.py` or `spec_cache.py`, so a failure here is not an acceptable Flash
trade-off.

## Stage 1 — integrated Flash correctness

First run the attention-core ladder, including no-cache/cached q=1/2/4/8 and
q=4 verify → rollback → corrective q=1 cases:

```bash
python -m specdec.engine.flash_attention_g4_core_probe
```

Required final line:

```text
decision=PASS_G4_CAUSAL_ALIGNMENT
```

Then run a cheap end-to-end smoke before the full contexts:

```bash
FLASH_G4_COMPAT_CTX=512 \
FLASH_G4_COMPAT_MAXNEW=8 \
python -m specdec.engine.flash_attention_g4_specdec_compat
```

If it passes, run the full AR + SpecDec compatibility gate:

```bash
FLASH_G4_COMPAT_CTX=512,2048,8192 \
FLASH_G4_COMPAT_MAXNEW=32 \
python -m specdec.engine.flash_attention_g4_specdec_compat
```

Required decision:

```text
PASS_FLASH_G4_COMPATIBILITY
```

The gate checks generated tokens, AR argmax, SpecDec statistics and trace,
q_len=4 observation, cache invariants, and constructed m=0..4 rollback payloads.

Finally re-run the frozen PPL equivalence gate through the integrated selector:

```bash
python -m specdec.engine.flash_attention_g4_ppl_gate
```

Required decision:

```text
PASS_FLASHATTENTION_PPL_EQUIVALENCE
```

Stop and report the JSON immediately if any Stage 1 command does not pass.

## Stage 2 — bounded small-query tuning

Only after Stage 1 passes:

```bash
FLASH_TUNE_WARMUP=3 \
FLASH_TUNE_REPS=9 \
python -m specdec.engine.flash_attention_small_q_tune
```

This does not alter production dispatch.  Send the resulting
`flash_attention_small_q_tune_*.json` for review.  The selected q=1, q=4, and
large-prefill configurations must be committed separately, then Stage 1 must be
re-run before performance data are collected.

## Stage 3 — final supporting systems data

Run only after the selected dispatch commit and repeated Stage 1 PASS:

```bash
FLASH_G4_CTX=512,2048,8192 \
FLASH_G4_REPS=3 \
FLASH_G4_WARM=1 \
FLASH_G4_MAXNEW=64 \
python -m specdec.engine.flash_attention_g4_ablation
```

This is a supporting BF16 eager/Flash ablation; it does not modify the frozen
G4 KV × decode matrix.  Both 1.5B target and 0.5B draft remain GPU-resident.

Then run the isolated 8K bottleneck diagnostic:

```bash
FLASH_PROFILE_CTX=8192 \
FLASH_G4_MAXNEW=64 \
python -m specdec.engine.flash_attention_g4_bottleneck
```

Interpret the CUDA-event breakdown together with its stated residual caveat.
Do not claim that `torch.cat` or KV movement is the primary bottleneck unless
its measured contribution supports that conclusion.

## Evidence files

Expected files under `output/flash_g4/`:

- `flash_attention_g4_core_probe.json`
- `flash_attention_g4_specdec_compat.json`
- `flash_attention_two_pass_ppl_gate.json`
- `flash_attention_small_q_tune_<hash>.json`
- `flash_g4_ablation_<hash>.jsonl`
- `flash_attention_g4_bottleneck.json`

Raw outputs are not source commits.  Final reviewed summaries and provenance
belong in the data ledger/results document after the runs are complete.
