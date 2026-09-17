# FlashAttention × latest G4 integration result

Status: **CHECK — experimental backend only; do not enable by default**  
Run date: 2026-09-17  
Base: `main-fresh@d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897`  
GPU: NVIDIA GeForce RTX 4060 8GB  
Runtime: PyTorch 2.11.0+cu128, CUDA runtime 12.8, Triton 3.6.0

## Decision

The migrated two-pass BF16 GQA Triton backend passes isolated attention,
cached multi-token causal alignment, and cache rollback tests.  It does **not**
preserve greedy and speculative-decoding behavior for every tested real
WikiText prompt.  It therefore remains opt-in research code and must not
replace eager attention in the frozen G4 workflow.

No correctness threshold was relaxed.  The frozen G4 12-condition matrix was
not changed.  `spec_dec.py` and `spec_cache.py` were not modified.

| Required result | Outcome |
|---|---|
| latest main + Flash + AR | **CHECK** |
| latest main + Flash + SpecDec | **CHECK** |
| cache rollback compatibility | **PASS** |
| cached `q_len=4` causal correctness | **PASS** |
| default backend | **eager (unchanged)** |

Because end-to-end behavior did not pass, the PPL gate, production small-Q
tuning, formal supporting ablation, and bottleneck claims are blocked by the
pre-registered stop rule.

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

The result was:

```text
NO_STRICT_PHASE_CANDIDATE
```

Both Flash-prefill/eager-online and eager-prefill/Flash-online routes diverged.
At 2048, using the custom kernel in both phases happened to preserve the tested
trace even though either hybrid diverged, showing context-dependent numerical
interaction rather than a stable single-phase defect.  PyTorch's forced Flash
SDPA reference also failed strict SpecDec trace equality.  A separate core
sanity probe is required before using the SDPA result to judge whether strict
token/trace identity is an appropriate cross-kernel contract.

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
4. **No simple phase hybrid fixes strict identity.** Prefill-only and
   online-only substitution both changed behavior, while the full route could
   pass or fail depending on context because its numerical errors interact.

The old PR #2 positive results remain valid for their recorded prompts and
environment.  This latest-G4 result narrows their scope: they do not establish
universal generated-token or SpecDec-trace equivalence on the new real-prompt
gate.

## Repository action

- retain eager as the default and formal G4 backend;
- keep the Flash selector explicit and experimental only on this branch;
- restore the validated PR #2 `BLOCK_N=64` launch instead of promoting the
  failed BN128 candidate;
- preserve the numerical screens as negative-result diagnostics;
- do not run or report the blocked tuning/ablation stages as if integration
  had passed;
- do not merge this branch as a production G4 optimization without changing
  the stated correctness contract and re-reviewing that change explicitly.
