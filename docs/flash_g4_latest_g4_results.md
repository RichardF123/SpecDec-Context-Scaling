# FlashAttention × latest G4 integration result

Status: **SUPPORTING RESULT COMPLETE — useful long-context prefill optimization; experimental only**
Run dates: 2026-09-17--18
Base: `main-fresh@d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897`  
GPU: NVIDIA GeForce RTX 4060 8GB  
Runtime: PyTorch 2.11.0+cu128, CUDA runtime 12.8, Triton 3.6.0

## Decision

The migrated two-pass BF16 GQA Triton backend is compatible with the latest G4
cache contract at the operator and state-machine levels.  Pure custom
`q_len=4` verification was numerically fragile, so the final bounded candidate
is an **explicit hybrid**: custom Triton for prompt prefill and `q_len=1`, and
the repository eager attention expression for cached `q_len=2..4`.  This is a
named experimental policy, not a silent fallback and not official
FlashAttention.

No correctness threshold was relaxed.  The frozen G4 12-condition matrix was
not changed, `spec_dec.py` and `spec_cache.py` were not modified, and eager
remains the default backend.

| Required result | Outcome |
|---|---|
| isolated cached causal alignment, `q_len=1/2/4/8` | **PASS** |
| constructed rollback payload/length coverage, `m=0..4` | **PASS** |
| frozen WikiText-2 PPL equivalence | **PASS** |
| 8K, 64-token same-backend SpecDec versus AR | **PASS** |
| 512/2K, 64-token exact BF16 behavior | **CHECK; path-sensitive numerical decisions** |
| 2K FP32 attribution | **BF16_PATH_UNDECIDABLE; FP32 paths consistent** |
| cross-backend exact token/trace identity | **diagnostic only; not a mainline gate** |
| production selector/default backend | **unchanged (`eager`)** |

The candidate is therefore suitable as a **supporting systems ablation and
reviewable prototype**, but not as a production-default replacement.  Its
reportable result is strongest at 8K: it materially improves prefill while
leaving SpecDec decode essentially unchanged.

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

The next bounded candidate was therefore an explicit runtime-only hybrid:
custom Triton for prefill and q_len=1, repository eager attention for cached
q_len=2..4.  It is named and tested explicitly, uses no official FlashAttention,
and does not change production dispatch. Full-context correctness,
quality, performance, and profiling results for that boundary are recorded
below.

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

## Final explicit-hybrid correctness evidence

The operator-isolation result justified one bounded integration boundary:
custom Triton for prefill and `q_len=1`, repository eager attention for cached
`q_len=2..4`. The hybrid preserves the latest G4 cache API and uses the
unchanged target/draft state machine.

At 2K/32 generated tokens, one BF16 decision differed between SpecDec and AR.
The repository-style attribution found:

```text
first difference=29, SpecDec=264, AR=22844
BF16 gaps: AR=-0.125000, SpecDec=+0.125000
FP32 signal=+0.409622, observed BF16 path noise=0.534622
FP32 paths consistent=True
decision=BF16_PATH_UNDECIDABLE
```

Thus the high-precision implementation evidence is consistent, while the BF16
decision is smaller than the measured path noise. This is recorded as CHECK,
not relabelled as exact equivalence. At 8K/64 tokens, both eager and hybrid
SpecDec matched their same-backend AR result, and the eager/hybrid tokens,
statistics, and traces also matched for that tested prompt.

The frozen quality gate on WikiText-2 windows 40..79 passed:

| Backend | NLL | PPL |
|---|---:|---:|
| eager BF16 | 2.312986 | 10.1046 |
| hybrid Flash BF16 | 2.312847 | 10.1031 |

Paired `dNLL=-0.0001392`, one-sided 95% bound `0.0004992`, below the frozen
`0.001` margin; relative PPL delta was `-0.0139%`.

## Supporting performance ablation

The final measurement used an RTX 4060 8GB, real nested WikiText prompts,
both Qwen2.5-1.5B target and Qwen2.5-0.5B draft resident, one discarded warm-up,
three measured repetitions, CUDA synchronization, rotated order, and
controlled replay of the same workload. Configuration hash:
`94fc2fb08e3e`.

| Context | Mode | Prefill speedup | Decode speedup | Total speedup | Peak allocated delta | Peak reserved delta |
|---:|---|---:|---:|---:|---:|---:|
| 512 | AR | 1.053x | 1.019x | 1.018x | -1.5 MiB | -26 MiB |
| 512 | SpecDec | 1.081x | 0.988x | 0.989x | +0.0 MiB | -8 MiB |
| 2048 | AR | 1.338x | 1.030x | 1.061x | -10.2 MiB | -242 MiB |
| 2048 | SpecDec | 1.202x | 1.085x | 1.091x | +0.0 MiB | -72 MiB |
| 8192 | AR | 2.064x | 1.055x | 1.336x | -380.6 MiB | -2836 MiB |
| 8192 | SpecDec | 1.408x | 0.991x | 1.145x | +1.0 MiB | +64 MiB |

Peak allocated is the primary memory statistic; reserved deltas are allocator
state and are reported separately. The 512/2K controlled-replay timings are
valid performance comparisons, but their natural 64-token BF16 paths did not
remain exactly equal and must not be presented as end-to-end equality results.
The 8K natural validation did remain equal.

## 8K decode bottleneck diagnosis

The final profiler used three repetitions and CUDA events. It also calculated
the exact bytes written by the current full-cache `torch.cat` append contract.

| Workload/backend | Median decode | Target attention + KV | Draft layers | Cache-cat output written |
|---|---:|---:|---:|---:|
| AR eager | 3707.02 ms | 1282.63 ms | 0 | 13.84 GiB |
| AR hybrid | 3634.26 ms | 991.93 ms | 0 | 13.84 GiB |
| SpecDec eager | 6290.21 ms | 664.07 ms | 4112.07 ms | 14.06 GiB |
| SpecDec hybrid | 6425.66 ms | 766.72 ms | 4164.24 ms | 14.06 GiB |

For AR, the hybrid reduced the target attention-plus-KV segment by about
`290.7 ms` (`22.7%`), but QKV projection, RoPE, MLP, logits, and full-cache
copying remained. That is why the end-to-end decode gain was only about 2% in
this instrumented run (and 5.5% in the lower-overhead supporting benchmark).

For SpecDec, draft-model layers consumed about 65% of decode time. Across
target and draft, cache append materialized `14.06 GiB` of K/V outputs over one
64-token workload (`6.15 GiB` target plus `7.91 GiB` draft). Flash does not
change either cost, and the custom path does not improve the eager `q_len=2..4`
verification boundary. SpecDec bookkeeping itself was only 11--12 ms and is
not the bottleneck.

This supports a concrete next-stage design: preallocated K/V storage with a
logical cache length, in-place append, and logical rollback compatible with
`spec_cache.trim_cache()`. Paged KV is a later option. Such work should be a
separate prototype and must not rewrite the frozen G4 cache contract in this
Flash PR.

## Interpretation

The evidence supports six scoped claims:

1. **Attention/cache semantics are correct.** Cached multi-token masking,
   native GQA mapping, trim/rollback, and corrective append pass isolated and
   constructed tests.
2. **Pure custom q_len=4 is not ready.** Its small numerical differences can
   change a BF16 greedy decision. Tile and exponent sweeps did not provide a
   stable correctness fix.
3. **The explicit hybrid is the defensible boundary.** It retains custom
   long-prefill and q_len=1 paths while using the repository expression for
   G4 block verification. It does not change cache or SpecDec semantics.
4. **Quality equivalence passed.** The frozen PPL gate is comfortably within
   its predeclared margin.
5. **The useful performance result is long-context prefill.** At 8K, AR total
   improved 1.336x and SpecDec total 1.145x, driven by 2.064x/1.408x prefill.
   Decode gains were small or absent.
6. **The next bottleneck is outside Flash attention.** SpecDec is dominated by
   draft-model execution and repeated full-cache movement; bookkeeping is
   negligible. A rollback-aware preallocated cache is the evidence-based next
   prototype.

The old PR #2 positive results remain valid for their recorded prompts and
environment. This latest-G4 study adds block-verification/rollback evidence,
documents the pure-custom q4 negative result, and narrows the implementation to
an explicit hybrid supporting ablation.

## Repository action

- keep eager as the production default;
- keep the hybrid selector explicit and experimental;
- preserve pure-custom q4 and numerical-sweep failures as negative results;
- present the final table only as a supporting systems ablation, never as a
  fourth factor in the frozen G4 experiment;
- keep 512/2K long-horizon token differences labelled CHECK and retain the FP32
  attribution evidence;
- open this work as a draft/review PR rather than silently merging it;
- pursue preallocated/logical-length KV cache work on a separate branch and
  re-run rollback, AR, SpecDec, PPL, memory, and latency gates before adoption.
