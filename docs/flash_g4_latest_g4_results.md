# FlashAttention × latest G4 integration result

Status: **CHECK — Flash q_len=1 serial and q_len=4 block semantics differ**
Run date: 2026-09-17  
Base: `main-fresh@d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897`  
GPU: NVIDIA GeForce RTX 4060 8GB  
Runtime: PyTorch 2.11.0+cu128, CUDA runtime 12.8, Triton 3.6.0

## Decision

The migrated two-pass BF16 GQA Triton backend passes isolated attention,
cached multi-token causal alignment, and cache rollback tests.  Early
diagnostics found that it does not preserve eager backend tokens and traces for
every real WikiText prompt.  Cross-backend equality is not a latest-main gate.
For same-backend behavior, latest main records exact BF16 SpecDec-versus-AR
results but explicitly treats its own 19/20 BF16 outcome through
signal-versus-noise attribution and a separate exact FP32 implementation gate.

No correctness threshold was relaxed.  The frozen G4 12-condition matrix was
not changed.  `spec_dec.py` and `spec_cache.py` were not modified.

| Required result | Outcome |
|---|---|
| eager versus Flash exact AR identity | **CHECK (diagnostic only)** |
| eager versus Flash exact SpecDec trace | **CHECK (diagnostic only)** |
| Flash SpecDec versus Flash target-only AR (BF16 behavior) | **CHECK; attribution in progress** |
| latest-main FP32 SpecDec implementation gate | **unchanged reference** |
| cache rollback compatibility | **PASS** |
| cached `q_len=4` causal correctness | **PASS** |
| default backend | **eager (unchanged)** |

The corrected 512-token smoke gate has now run.  Eager SpecDec reproduced
eager target-only AR, but Flash SpecDec did not reproduce Flash target-only
AR.  Cross-backend SpecDec happened to agree in this run.  The BF16 mismatch is
recorded as CHECK and requires the same signal-versus-noise classification used
by latest main; it is not silently accepted and is not automatically promoted
to an implementation failure solely because exact long-prefix tokens differ.

Observed smoke result:

```text
ctx=512, max_new=8, K=4
constructed rollback coverage m=0..4: PASS
eager SpecDec == eager AR: PASS
Flash SpecDec == Flash AR: CHECK
Flash SpecDec repeat deterministic: PASS
Flash/eager SpecDec diagnostic: tokens/stats/trace equal in this run
decision=CHECK_FLASH_G4_SAME_BACKEND_CORRECTNESS
```

This localizes the next investigation to cached small-query behavior rather
than the G4 state machine or rollback contract.  The production policy uses
`BLOCK_M=16` for q_len=1 but `BLOCK_M=64` for q_len=4.  The new bounded
`flash_attention_g4_block_serial_screen.py` therefore advances identical
target prefixes through serial q_len=1 and block q_len=4 calls while screening
small-query launch policies.  It is diagnostic-only and does not change the
production selector.

That launch-policy screen has now completed.  All eight candidates failed at
the same first-round, third same-context output row, choosing token 1602 on the
serial path and token 883 on the block path.  The candidates covered unified
`BLOCK_M` values 8/16/32/64, `BLOCK_N` values 32/64/128, the retained mixed
16/64 policy, and libdevice exponentiation.  The result was:

```text
decision=STOP_NO_BLOCK_SERIAL_CANDIDATE
```

Because the failure location and token pair were invariant across the bounded
launch sweep, further tile-size tuning is not justified as a correctness fix.
The next diagnostic isolates custom/eager attention on the serial and block
paths, plus a serialized-custom q_len=4 upper bound, to distinguish an
attention-kernel defect from numerical differences in the full-model block
execution path.

The first version of that diagnostic labelled any eager-online failure after a
shared custom-Flash prefill as `CHECK_MAINLINE_CONTROL`.  That label was too
strong: it was not a pure eager-prefill control, and latest main already
documents BF16 block/serial shape drift.  Version 2 separates eager and custom
prefill states and treats exact BF16 drift as behavior evidence requiring
attribution, consistent with the repository protocol.

Version 2 isolated the custom q_len=4 path:

| Route | Outcome | First relevant observation |
|---|---|---|
| custom prefill + custom serial + custom block | CHECK | round 0 row 2, 1602 vs 883 |
| custom prefill + custom serial + eager block | PASS | 16/16 tested positions |
| custom prefill + eager serial + custom block | CHECK | round 3 row 2 |
| custom prefill + eager serial + eager block | CHECK | BF16 shape drift |
| eager prefill + eager serial + eager block | CHECK | BF16 shape drift |
| custom serial + serialized-custom block API | PASS | exact zero error |

The decision was `CUSTOM_Q4_ATTENTION_SUSPECT`.  This does not establish that
every eager q_len=4 block is exact in BF16; it shows that, from the identical
custom-prefill state used in the failing route, replacing only custom q_len=4
attention with the repository eager expression removed the tested divergence.

The next bounded candidate is therefore an explicit runtime-only hybrid:
custom Triton for prefill and q_len=1, repository eager attention for cached
q_len=2..4.  It is named and tested explicitly, uses no official FlashAttention,
and does not change production dispatch.  Only after full-context correctness
and performance measurement will it be considered as an integration boundary.

## Correctness ladder

### Existing latest-main G4 regression

All pre-existing gates completed successfully before evaluating Flash:

- B5 EOS branch: PASS;
- B4 cache API: PASS;
- B5 helper tests: PASS;
- B5 structural smoke for `base_bf16` and `v_fp8`: PASS.

### Isolated Flash core

`flash_attention_g4_core_probe.py` passed:

- cached and non-cached `q_len` 1, 2, 4, and 8;
- cached `q_len=4`, KV length 2052;
- decode `q_len=1`, KV length 8192;
- `q_len=4` verification followed by rollback at `m=0..3` and corrective
  `q_len=1` append.

The end-to-end constructed state-machine test separately covered `m=0..4` and
verified cache payloads as well as lengths.  All cases passed.

### End-to-end real-prompt behavior

The first 512-context/8-token smoke passed.  Extending generation to 32 tokens
exposed cumulative numerical differences.

With the retained PR #2 launch (`BLOCK_N=64`):

| Context | AR | SpecDec | Note |
|---:|---|---|---|
| 512 | CHECK | CHECK | AR tokens diverged; SpecDec tokens matched but trace differed |
| 2048 | CHECK | CHECK | AR tokens and SpecDec behavior diverged |
| 8192 | PASS | OOM/CHECK | AR matched; dual-model eager prefill used chunk 2048 and OOMed |

The 2048 AR first divergence was at generated step 2 on an identical prefix:

- eager token: 572;
- Flash token: 1221;
- logit max absolute difference: 0.220703;
- logit relative L2: 0.017107.

The eager top-1 margin at that point was zero at BF16 resolution, so a
different reduction order could choose a different tied token.  This alone was
not treated as sufficient evidence of a semantic defect or sufficient reason
to relax the equality gate.

### Bounded numerical follow-up

A candidate `BLOCK_N=128` matched the four-token 2048 screen.  It was then
tested end-to-end with a 512-token prefill chunk, which removed the dual-model
8K OOM:

| Context | AR | SpecDec | Diagnostic timings (eager → Flash) |
|---:|---|---|---|
| 512 | CHECK | CHECK | AR diverged; SpecDec trace differed |
| 2048 | PASS | PASS | prefill 0.348s → 0.249s; decode 1.092s → 1.078s |
| 8192 | PASS | PASS | prefill 2.731s → 1.286s; decode 1.124s → 1.089s; SpecDec 6.381s → 4.921s |

These timings came from a correctness diagnostic without the final benchmark's
warm-up/repetition discipline.  They are feasibility signals, not reportable
performance estimates.

The final 512-context, 32-token screen compared:

- fast `BLOCK_N` 32, 64, and 128;
- prefill/decode mixes 128/64 and 64/128;
- libdevice-exp candidates at `BLOCK_N` 32, 64, and 128;
- prefill-only and decode-only libdevice exponentiation.

Every candidate changed at least one greedy argmax.  The 32/64 candidates first
diverged at step 15 with an eager top-1 margin of 0.125, demonstrating that the
failure was not limited to exact logit ties.  The decision was:

```text
STOP_NO_NUMERIC_CANDIDATE
```

### Phase-isolation follow-up

The next bounded diagnostic separated prompt prefill from online AR decode /
SpecDec block verification.  It used a 512-token chunk, 32 generated tokens,
and the same real nested WikiText prefixes.  `F` denotes the retained custom
two-pass Flash kernel; `S` denotes PyTorch SDPA with `FLASH_ATTENTION` forced
and no math fallback.

| Context | Route | AR exact | SpecDec exact | First AR argmax difference |
|---:|---|---:|---:|---:|
| 512 | EE | PASS | PASS | none |
| 512 | FE | CHECK | CHECK | 15 |
| 512 | EF | CHECK | CHECK | 15 |
| 512 | FF | CHECK | CHECK | 3 |
| 512 | SS | PASS | CHECK | none |
| 2048 | EE | PASS | PASS | none |
| 2048 | FE | CHECK | CHECK | 2 |
| 2048 | EF | CHECK | CHECK | 2 |
| 2048 | FF | PASS | PASS | none |
| 2048 | SS | CHECK | CHECK | 6 |

The cross-backend diagnostic result was:

```text
NO_STRICT_PHASE_CANDIDATE
```

Both Flash-prefill/eager-online and eager-prefill/Flash-online routes diverged.
At 2048, using the custom kernel in both phases happened to preserve the tested
trace even though either hybrid diverged, showing context-dependent numerical
interaction rather than a stable single-phase defect.  PyTorch's forced Flash
SDPA reference also failed strict SpecDec trace equality.  A separate core
sanity probe is required before using the SDPA result to judge whether strict
token/trace identity is an appropriate cross-kernel contract.  The SDPA probe
subsequently passed this core sanity check, but official SDPA remains reference
evidence only and is not part of the implementation.

## Interpretation

The evidence separates three claims:

1. **Attention/cache semantics are correct.** Cached multi-token masking,
   native GQA mapping, trim/rollback, and corrective append all pass isolated
   and constructed tests.
2. **The kernel is numerically close but not behaviorally identical.** Small
   per-layer differences from exponential approximation and reduction order
   accumulate through 28 transformer layers and can change greedy argmax and
   speculative acceptance.
3. **Launch tuning is not a correctness fix.** Larger tiles and higher-accuracy
   exponentiation changed which prompts matched, but no tested configuration
   preserved all required behavior.
4. **No simple phase hybrid fixes cross-backend strict identity.** Prefill-only and
   online-only substitution both changed behavior, while the full route could
   pass or fail depending on context because its numerical errors interact.
5. **Strict cross-backend identity is not the mainline G4 gate.** The mainline
   reference compares SpecDec with target-only AR from the same engine/backend;
   round statistics and traces need to be valid, not identical to eager.

The old PR #2 positive results remain valid for their recorded prompts and
environment.  This latest-G4 result narrows their scope: they do not establish
universal generated-token or SpecDec-trace equivalence on the new real-prompt
gate.

## Repository action

- retain eager as the default while the corrected same-backend gate is pending;
- keep the Flash selector explicit and experimental only on this branch;
- restore the validated PR #2 `BLOCK_N=64` launch instead of promoting the
  failed BN128 candidate;
- preserve the numerical screens as negative-result diagnostics;
- rerun compatibility using same-backend SpecDec-versus-AR correctness;
- do not run or report tuning/ablation stages as passed until that gate passes;
- do not merge this branch as a production G4 optimization until the unchanged
  mainline correctness contract passes and the result is reviewed.
