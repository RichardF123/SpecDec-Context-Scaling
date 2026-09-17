# FlashAttention × G4 change ownership

Status: integration evaluated; end-to-end result CHECK
Integration branch: `experiment/flash-g4-integration`

## Recorded topology

| Role | Commit |
|---|---|
| latest `main-fresh` | `d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897` |
| PR #2 / `experiment/streaming-kv-probe` | `b1904b5d867dd6eb7262f733d1818c940145c819` |
| common merge base | `da2fac76ba0d011a5f0c5b097a485eceac5d54b8` |

The two heads diverged after the recorded merge base.  This branch starts from
`main-fresh`; that starting topology does not imply that main owns newer Flash
functionality.  The integration is a three-way semantic merge, not a whole-file
choice and not a merge/cherry-pick of PR #2.

## Ownership map

| Type | File(s) | Main changed? | PR #2 changed? | Functional owner | Merge action |
|---|---|---:|---:|---|---|
| A | `specdec/engine/spec_dec.py` | yes | no | latest G4 | keep main exactly; Flash adapts to its fixed-K block verification and accounting |
| A | `specdec/engine/spec_cache.py` | yes | no | latest G4 | keep main exactly; validate Flash caches through `cache_length`, `trim_cache`, and `check_consistent` |
| A | `specdec/engine/g4_*.py` | yes | no | latest G4 | keep main tests and semantics; add separate Flash compatibility coverage |
| A | `specdec/engine/g4_part4_pilot_measure.py` | yes | no | latest G4 measurement | reuse its timing, repetition, memory, prompt, metadata, and OOM discipline |
| A | `README.md`, G4 protocol/design/report docs, `requirements-local.txt`, report generators | yes | no | latest main | retain main; do not replace with older PR #2 snapshots |
| B | `specdec/engine/flash_attention_two_pass_triton.py` | no | yes | validated PR #2 Flash | migrate the latest two-pass BF16 GQA kernel; production integration may add separately measured shape dispatch only after gates pass |
| B | `flash_attention_two_pass_{core_probe,engine_diagnostic,full_trial,layer_diag,ppl_gate}.py` and `flash_attention_integration_smoke.py` | no | yes | validated PR #2 Flash evidence | preserve or adapt as regression references; do not weaken their numerical/behavioral gates |
| C | `specdec/engine/model_self.py` | no after merge base | yes | shared boundary | use merge-base/latest-main engine as source of truth; transplant only the opt-in backend state, strict selector, lazy import, and validated two-pass call; preserve main cache mutation and multi-token forward semantics |
| C | `docs/data_ledger.md` | yes | yes | shared evidence ledger | semantic append-only merge later; retain both G4 records and Flash positive/negative evidence without overwriting either side |
| D | `flash_attention_triton.py`, `flash_attention_core_probe.py`, `flash_attention_engine_trial.py` | no | yes | PR #2 negative one-pass evidence | keep available for reproducibility if migrated, but never select from production and never use its core-only pass as an end-to-end correctness claim |
| D | `kv_streaming_*`, `kv_triton_engine_trial.py` | no | yes | PR #2 failed/diagnostic V-FP8 fusion work | leave out of the production engine; reference as negative evidence and optional future work only |
| D | `kv_policy_*`, `kv_vint4_*`, PR #2-only head-reduction/K-INT8/V-INT4 paths | no | yes | PR #2 supporting KV exploration | do not transplant into latest-main `model_self.py`; they are outside the frozen G4 design and are not required by BF16 Flash integration |
| D | `docs/gqa_kv_policy_probe.md`, `docs/gqa_kv_final_resolution.md` | no | yes | PR #2 supporting evidence | preserve the conclusions/provenance; do not let them redefine the latest G4 formal matrix |

## `model_self.py` semantic merge decision

The merge-base and latest `main-fresh` versions of `model_self.py` are byte-for-
byte identical.  PR #2 adds both (a) the validated Flash selector and (b) many
unrelated experimental KV representations.  Importing the whole PR #2 file
would therefore expand the latest production engine with rejected or optional
research paths and would obscure which behavior G4 was validated against.

Only these PR #2 production semantics are eligible for the integrated file:

1. per-layer backend state defaults to `eager`;
2. `QwenLM.set_attention_backend()` accepts exactly `eager` and
   `triton_bf16_two_pass`;
3. Triton is imported lazily only when explicitly selected;
4. the Triton call consumes native-head BF16 Q/K/V after the existing cache
   append, so cached `q_len=1` and block verification `q_len=4` share the same
   cache contract;
5. unsupported device, dtype, head dimension, KV format, or head layout raises
   an explicit error; there is no silent fallback;
6. eager remains the unchanged default for CPU, Windows, V-FP8, and the frozen
   formal G4 conditions.

The integration must not modify `spec_dec.py` or `spec_cache.py` to accommodate
Flash.  Compatibility is established by tests that exercise cached multi-token
causal alignment and every feasible accept/rollback outcome.

## Evaluation outcome

The semantic merge preserved this ownership map, and neither `main-fresh` nor
PR #2 was modified.  Isolated causal/cache tests passed, but the real-prompt
end-to-end AR/SpecDec equality gate did not.  The backend therefore remains
explicit experimental code on this branch; eager remains the only accepted G4
backend.  See `docs/flash_g4_latest_g4_results.md` for the bounded follow-up and
stop decision.
