# PR #2 / #3 review and isolated research import

Date: 2026-10-02. This review preserves useful collaborator research without
changing the current engine, defaults, cache contract, dependencies, proposal,
or acceptance gates. Neither PR is approved for wholesale production merge.

## Sources and import boundary

| PR | Reviewed head | Date | Scope |
|---|---|---|---|
| [#2](https://github.com/RichardF123/SpecDec/pull/2) | `ca8b68c5ad5ebefae7d813a6477208a86e3d3dd4` | 2026-09-14 | KV head reduction, K-INT8, channel-block K, packed V-INT4 |
| [#3](https://github.com/RichardF123/SpecDec/pull/3) | `13b57b12360f70dd331ad68e75511eaaa0b733e4` | 2026-09-18 | Two-pass Triton attention, G4 hybrid experiments and profiling |

GitHub PR refs were fetched and source reviewed at these exact heads. PR API
status was not available to this session; fetching a PR ref does not establish
whether the PR is currently open or closed. These heads are not the same work
as the already-merged PR #4 custom CUDA/Triton operator pack.

Source snapshots and original reports are in [the research archive](research_archive/README.md).
The archives contain files changed in each PR relative to its merge base,
at their original repository paths, plus a manifest of Git blob IDs and
SHA-256 hashes. They are NOT standalone runnable packages. In particular,
their old `model_self.py` and `data_ledger.md` must never be extracted over
current main. Reproduction requires the original commit environment or a
separately reviewed port. No archived scripts are installed or imported by
the current engine.

This is a selective research-artifact import, NOT a Git merge of either PR.
Original PR heads and failures remain intact; no production-readiness or
fresh-GPU-verification claim is implied.

## Findings: PR #2

### P1: channel-block K representation violates the current SpecDec cache contract

The candidate stores quantized K tokens, per-block scales and a separate BF16
residual. Current `spec_cache.cache_length()` counts only `k.shape[2]`, while
`trim_cache()` slices every tensor on dimension 2 as if it were a token axis.
For the candidate, scale dimension 2 is blocks, not tokens, and the residual
also needs separate handling. Merely adding the candidate format to the engine
would give incorrect length/rollback semantics.

CPU reproduction using the PR's actual helper functions: append 33 tokens with
block size 32 -> K length 32, scale length 1, residual length 1, V length 33.
Current `cache_length()` reports 32 and `check_consistent()` raises. This blocks
direct adoption into current SpecDec, not the original standalone AR study.

### P2: original results are trade-offs, not an upgrade to the frozen policy

The collaborator report records:

| Candidate | Quality / storage | Operational result and interpretation |
|---|---|---|
| Direct 2-to-1 KV head merging | pilot dNLL >= +7.44 | Changes pretrained model semantics; rejected |
| Channel-block K-INT8 + V-FP8 | confirmation dNLL +0.013580; relative dPPL +1.367%; 1.855x logical compression vs BF16 | Compared with V-FP8, peak allocated fell 17.8 MiB but decode was 13.9% slower; negative/trade-off evidence |
| BF16-K + packed V-INT4 g32 | confirmation dNLL +0.009427; relative dPPL +0.947% vs BF16; 1.524x logical compression | Compared with V-FP8 at 8K, logical KV fell 22.75 MiB and peak allocated 45.2 MiB, but decode was 26.2% slower |
| V-INT4 g64/g128 | g64 failed its confirmation criteria; g128 rejected at screening | Do not promote a wider group based on screening alone |

These numbers are quoted from the PR report, not independently rerun here.
The V-INT4 quality denominator is BF16 in the confirmation figure above; its
memory/latency comparison is against V-FP8. They must not be conflated.

Value: preserve the negative head-reduction result and the quality-memory-
latency comparison; INT4 packing helpers may be reusable in a future isolated
port. Keep K-BF16 + V-FP8 as the present policy. No new formal factor is added.

## Findings: PR #3

### P1: the historical integration does not satisfy the current exact-token gate

The report explicitly records 512/2K BF16 CHECK outcomes and a pure-custom
q_len=4 failure. One tested 8K prompt passing is not a general AR/Spec
correctness guarantee. It cannot resolve or waive the current F13 failure.

### P1: the old selector cannot be merged over the current attention path

The PR adds `set_attention_backend()` and replaces the old attention dispatch.
Current main additionally has phase-aware cuDNN selection, diagnostic precision
controls, and backend accounting. Whole-file replacement loses those features;
simply copying runners also fails because they expect the historical selector.
The custom kernel supports head_dim=128 and uncompressed BF16 KV; its benchmark
keeps the head_dim=64 draft eager. It is not a universal draft/target or V-FP8
replacement.

### P2: reported speedups are supporting replay evidence, not formal Spec/AR speedups

The collaborator used PyTorch 2.11.0+cu128 / Triton 3.6.0; this checkout uses
PyTorch 2.6.0+cu124. The supporting runner defaults to chunk=512 rather than
the current diagnostic chunk=2048. It loads both models even for its AR row,
and timing replays fixed tokens/traces instead of measuring natural generation
with acceptance decisions. These are useful controlled backend comparisons,
not the target-only AR baseline or natural SpecDec speedup of the final study.

The report's 8K numbers compare hybrid against eager within each mode:
AR prefill 2.064x, AR total 1.336x; Spec prefill 1.408x, Spec decode 0.991x,
Spec total 1.145x. They do NOT mean Spec beats AR by 1.145x.

The runner also permits summaries/comparisons after partial measured-run OOM
when each backend has at least one result. Its minimum configured repetitions
is not an enforced completeness gate after failures. Do not reuse it as the
current formal measurement runner without correcting that boundary.

Value: the two-pass kernel deliberately preserves BF16 score/probability
materialization boundaries while avoiding a full score matrix. The report's
draft-layer and full-cache `torch.cat` analysis offers a useful future profiling
hypothesis. Its 14.06 GiB figure is cumulative cache-output bytes over the
workload, NOT peak VRAM. Preallocated cache is a separate future change needing
rollback tests, not a fix established by this review.

## Verification performed in this review

- Parsed all changed Python sources: PR #2 14 files, PR #3 17 files.
- Ran only PR #2 quantizer helper functions on CPU: zero INT4 roundtrip and
  two-values-per-byte packing passed.
- Reproduced the channel-block cache incompatibility against current
  `spec_cache` with a 33-token CPU example.
- No GPU kernels, model loading, PPL, speed measurements, new dependencies,
  or acceptance-gate changes were performed.
- Original reports refer to external experiment artifacts; this import does
  not supply or independently validate those raw GPU artifacts.

## Decision and next priority

Preserve both PRs as isolated, source-attributed research artifacts. Do not
wholesale merge their engine changes, enable historical backends, or modify
the main 6 + supplemental 2 study. Continue the current cache-history
correctness investigation first. Any later V-INT4 or Triton port must be a
separate opt-in change with cache/rollback, dtype/dispatch, native-device
correctness and measurement-completeness tests before use.
