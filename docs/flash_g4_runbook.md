# FlashAttention × G4 RTX-4060 runbook

This runbook deliberately follows the unchanged latest-main G4 evidence
protocol.  The repository records exact same-backend BF16 SpecDec-versus-AR
behavior, but it does not treat a BF16 near-tie as standalone implementation
evidence: latest main reports the BF16 behavior (documented as 19/20), uses
`g4_b6_tie_diag.py` for signal-versus-noise attribution, and uses
`g4_b6_fp32_correctness.py` as the high-precision implementation gate.
Cross-backend eager/Flash token, statistic, or trace equality remains
diagnostic evidence only.

## Recorded 2026-09-17 outcome

Stage 0 and the isolated Flash core passed.  An initial integration diagnostic
then required eager and Flash to preserve identical AR output, acceptance
statistics, and full SpecDec traces.  That cross-backend requirement was
stricter than latest-main `g4_b6_correctness.py`, whose reference is same-engine
target-only cached AR.  The old result remains useful numerical-sensitivity
evidence, but it does not decide Flash G4 correctness.

The bounded launch/exp sweep result was:

```text
decision=STOP_NO_NUMERIC_CANDIDATE
```

This means only that no candidate preserved strict cross-backend identity.  It
does not alter the latest-main BF16-attribution/high-precision protocol below.
PyTorch Flash SDPA was used only as a sanity reference and is not a repository
backend or proposed implementation.

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

Then run the full AR + SpecDec same-backend BF16 behavior diagnostic:

```bash
FLASH_G4_COMPAT_CTX=512,2048,8192 \
FLASH_G4_COMPAT_MAXNEW=32 \
FLASH_G4_COMPAT_CHUNK=512 \
python -m specdec.engine.flash_attention_g4_specdec_compat
```

An exact result is recorded as:

```text
PASS_FLASH_G4_SAME_BACKEND_CORRECTNESS
```

For each backend, the diagnostic records whether SpecDec generated tokens equal
that same backend's target-only cached AR tokens.  It also checks Flash
determinism, finite values, non-vacuous acceptance, q_len=4 observation, cache
invariants, and constructed m=0..4 rollback payloads.  Eager-versus-Flash
output, stats, and traces do not affect the result.

A BF16 token mismatch is not automatically relabelled PASS, but neither is it
automatically an implementation failure.  Apply the same repository-owned
classification used by `g4_b6_tie_diag.py`: record the exact BF16 outcome and
compare decision signal with observed shape/path noise.  The unchanged
`g4_b6_fp32_correctness.py` remains the state-machine implementation evidence.
Any non-near-tie mismatch, non-finite result, cache failure, nondeterminism, or
causal-alignment failure remains a blocker.

Finally re-run the frozen PPL equivalence gate through the integrated selector:

```bash
python -m specdec.engine.flash_attention_g4_ppl_gate
```

Required decision:

```text
PASS_FLASHATTENTION_PPL_EQUIVALENCE
```

Stop and inspect the JSON immediately for any hard failure above.  Do not turn
a BF16 mismatch into a pass without the repository's attribution evidence.

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
