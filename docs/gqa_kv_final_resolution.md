# Final resolution: GQA-aware KV compression and fused consumption

Status: **supporting investigation complete; optional BF16 Triton backend validated**  
Date: 2026-09-14

## 1. Question and scope

Qwen2.5-1.5B has 12 query heads and 2 native KV heads. GQA therefore already
uses one sixth as many stored KV heads as a 12-KV-head MHA layout. This
investigation asked two bounded questions:

1. Can a different head-count/quantization combination reduce KV storage beyond
   the frozen K-BF16 + V-FP8 policy without an unacceptable quality or latency
   cost?
2. If logical compression does not translate into whole-model peak reduction,
   can blockwise or fused consumption avoid full V dequantization and the 2-to-12
   GQA expansion?

The conclusion is specific to the tested Qwen2.5-1.5B eager engine, BF16 compute,
8K context, and RTX 4060 8GB. It is not a universal impossibility claim about
quantized-KV kernels, trained GQA-to-MQA conversion, or other architectures.

## 2. Representation-level results

The completed GQA-aware policy probe established the following.

| Candidate | Confirmed/operational result | Decision |
|---|---|---|
| Direct native KV-head reduction, 2 to 1 | dNLL at least +7.44 in screening | Reject; changes the pretrained attention function without uptraining |
| Per-token K-INT8 + V-FP8 | Relative PPL +18.541% in screening | Reject |
| Per-channel/block-32 K-INT8 + V-FP8 | Logical KV 28.9% below current; confirmed relative PPL +1.327%; 8K peak allocated only 17.8 MiB lower; decode 13.9% slower | Diagnostic negative result |
| Same K-INT8 with recent-BF16 residual-128 | Pilot improvement did not reproduce on fresh windows 80..119 | Reject |
| K-BF16 + packed V-INT4 group-32 | Logical KV 13.4% below current; peak allocated 45.2 MiB lower; relative PPL +0.924%; decode 26.2% slower | Optional memory-priority mode only |
| K-BF16 + V-FP8 | Near-lossless confirmed quality and best eager latency-quality-memory balance | Keep as default |

Therefore, no tested representation is a Pareto-improving drop-in replacement
for K-BF16 + V-FP8. V-INT4 is useful only when a user explicitly accepts lower
quality and slower decode for a small additional allocation reduction.

## 3. Evidence ladder for blockwise/fused consumption

The kernel investigation was kept separate from the representation decision.
It used the same Qwen-shaped dimensions (batch 1, 12 query heads, 2 KV heads,
head dimension 128) and did not alter the frozen cache format.

### 3.1 PyTorch blockwise mechanism probe

At 8K with current V-FP8, block-512 streaming reduced attention-core temporary
allocation from approximately 100.57 MiB to 4.82 MiB (95.2%), but was 3.66 times
as slow as eager. This verified that full-cache materialization was removable,
while also showing that a Python loop over blocks was not an operational solution.

### 3.2 Triton attention-core probe

Environment: NVIDIA GeForce RTX 4060 8GB; PyTorch 2.11.0+cu128; CUDA 12.8;
Triton 3.6.0; BF16 compute.

Latency and peak-memory measurements were separated: steady-state latency was
measured with a warm allocator, while peak allocation used a separate clean run.
At 8K, block 128 was the selected point:

| Metric | Eager V-FP8 | Triton block-128 | Change |
|---|---:|---:|---:|
| Attention-core latency | 1.017 ms | 0.126 ms | 0.123x latency, approximately 8.10x faster |
| Temporary allocation | 100.57 MiB | 0.384 MiB | -99.6% |
| max absolute output difference | - | 0.000488 | Within the pre-registered 1e-3 gate |
| relative L2 output difference | - | 0.005941 | Within the pre-registered 1e-2 gate |

This result proves the mechanism: fused block consumption can remove nearly all
of the temporary materialization and can accelerate the isolated attention core.
It does not establish end-to-end model correctness or speedup.

### 3.3 First V-FP8 fused whole-model trial (negative result)

The production `model_self.py` remained unchanged. A runtime-only monkeypatch
redirected native-head, cached, single-token V-FP8 decode to the Triton block-128
kernel. Eager and Triton modes were measured over four balanced-order 8K runs,
each decoding 64 greedy tokens.

| Metric | Eager V-FP8 | Triton V-FP8 | Change |
|---|---:|---:|---:|
| Median prefill | 3.038 s | 3.031 s | No meaningful change; prefill remained eager |
| Median decode | 4.088 s | 3.787 s | 1.079x speedup; -7.36% latency |
| Median ms/token | 63.88 | 59.17 | -4.70 ms/token |
| Decode peak allocated | 3.855 GiB | 3.800 GiB | -56.1 MiB |
| Decode peak extra allocation | 228.5 MiB | 172.4 MiB | -56.1 MiB |
| Decode peak reserved | 4.230 GiB | 4.230 GiB | No change |
| Prefill/final logical cache | 169.75 / 171.08 MiB | 169.75 / 171.08 MiB | No change |

The first Triton run included approximately 47 seconds of one-time JIT
compilation and was not representative of steady-state decode. In the three
warmed paired rounds, Triton reduced decode latency by 5.8% to 11.3%, consistent
with the median result.

Operational stability alone passed: both backends were finite and deterministic
across their own repeated runs. Cross-backend numerical/behavioral equivalence
did not pass:

- the first logit comparison already had max absolute difference 3.21875 and
  relative L2 difference 0.2246;
- median per-step logit relative L2 difference was 0.2193;
- the first logit argmax disagreement occurred at zero-based comparison step 20;
- greedy tokens first diverged at zero-based token index 21 (the 22nd generated
  token);
- all later logits cannot be interpreted as paired numerical comparisons after
  autoregressive trajectories diverge, but the large pre-divergence drift is
  independently sufficient to fail the gate.

### 3.4 From-scratch BF16 GQA FlashAttention follow-up

The failed V-FP8 decode kernel established that an attention-core pass is not
enough. A separate from-scratch BF16 GQA implementation therefore tested full
causal prefill, cached/chunked prefill, and decode before any engine integration.
The first one-pass online-softmax version passed random-tensor core thresholds,
but small per-layer differences accumulated through the model and caused the
8K greedy trajectory to diverge. Runtime KV strides were also initially marked
as compile-time constants, causing repeated Triton compilation as cache length
changed; making those strides runtime arguments removed the 30--47 second
first-decode anomaly.

The final kernel uses a two-pass, globally normalized design. Pass 1 stores one
FP32 maximum and denominator per query row. Pass 2 recomputes tiled QK scores,
normalizes against those global statistics, converts the final probability to
BF16, and accumulates P @ V. It still never materializes the quadratic score or
probability matrix and maps 12 query heads directly to 2 native KV heads. The
extra storage is two FP32 scalars per query row. This is a correctness-focused
two-pass FlashAttention variant, not a claim of bitwise identity with a
canonical single-pass FlashAttention implementation.

Against eager attention, the two-pass kernel reduced core relative-L2 error
from 0.0048--0.0074 to 0.0000--0.000094 across full-prefill, cached-prefill, and
8K decode cases. A 2K layer-growth diagnostic found maximum same-input local
attention relative-L2 error 0.000108; later-layer amplification affected logits
but did not indicate a layer-specific kernel discontinuity.

Whole-model RTX 4060 results used Qwen2.5-1.5B, BF16 KV, chunk 2048, 64 greedy
decode tokens, and four balanced-order rounds:

| Context | Prefill speedup | Prefill peak allocated change | Decode speedup | Decode peak allocated change | Greedy tokens |
|---:|---:|---:|---:|---:|---|
| 2K | 1.567x | 0.0 MiB | 1.076x | -12.3 MiB | 64/64 identical |
| 4K | 1.834x | -412.0 MiB | 1.244x | -24.6 MiB | 64/64 identical |
| 8K | 2.299x | -1404.0 MiB | 1.021x | -48.5 MiB | 64/64 identical |

The 2K repeated-text prompt exceeded the deliberately strict 0.02 last-logit
relative-L2 gate (0.0491) while retaining the same argmax and all 64 generated
tokens. The 4K and 8K cases passed that strict gate (0.0197 and 0.0159).
Accordingly, quality was decided by a separately frozen corpus-level gate
rather than by relaxing the prompt-specific threshold after observing it.

On WikiText-2 windows 40..79 (40 paired non-overlapping windows, 81,880 scored
tokens per backend, chunk 512), eager BF16 obtained NLL/PPL 2.312986/10.1046 and
Triton obtained 2.312847/10.1031. Paired delta-NLL was -0.0001392 with SE
0.0001837; the two-sided 95% equivalence bound was 0.0004992, below the frozen
0.001 margin. Relative delta-PPL was -0.0139%. The quality gate therefore
**passed**.

Finally, the production selector itself was smoke-tested at 2K/chunk-512 with
16 decode tokens: last-logit relative-L2 0.01594, same argmax, identical tokens,
and `PASS_INTEGRATION_SMOKE`. The un-warmed smoke timing includes first-use JIT
and is not a steady-state performance result.

## 4. Final engineering decision

The first V-FP8 fused decode kernel remains a negative result and is not
integrated. The validated two-pass **BF16** kernel is integrated only as an
explicit opt-in backend named `triton_bf16_two_pass`. The default remains
`eager`, so existing CPU, native-Windows, quantized-KV, and formal experiments
are unchanged. Triton is imported lazily only when the optional backend is
selected. Unsupported combinations raise a clear error instead of silently
falling back.

The opt-in backend is intentionally restricted to CUDA BF16, head dimension
128, checkpoint-native GQA heads, and `fmt="fp16"`. It does not fuse FP8 or INT4
dequantization and must not be described as a result for compressed KV cache.
This preserves the distinction between two conclusions: representation-level
V-FP8 remains the default cache policy, while BF16 FlashAttention is a validated
system-level memory/prefill optimization.

The adopted policy remains:

- default: **K-BF16 + V-FP8 in the validated eager engine**;
- optional BF16 attention backend: **two-pass Triton GQA FlashAttention**;
- optional memory-priority ablation: **K-BF16 + packed V-INT4 group-32**;
- rejected: direct KV-head reduction, both K-INT8 variants, recent-BF16 residual,
  Python blockwise attention, and the first V-FP8 Triton whole-model integration;
- frozen G3 and the formal 3 x 2 x 2 experimental design remain unchanged.

## 5. Answer to the original question

GQA-aware quantization can reduce logical KV bytes further, but none of the
tested representation changes simultaneously improves quality, latency, and
whole-model peak memory enough to replace K-BF16 + V-FP8 as the default. The
first fused V-FP8 attempt also failed behavioral equivalence.

For uncompressed BF16 KV, however, the from-scratch two-pass Triton backend is a
validated solution: it preserves frozen-window PPL and greedy output, reduces
8K whole-model prefill peak allocation by 1404 MiB, and accelerates 8K prefill
by 2.299x. This does not make KV compression unnecessary and does not establish
a fused compressed-KV kernel; it is a separate, useful system optimization.

## 6. Project plan after closure

1. Review the GQA-aware representation result and the BF16 FlashAttention result
   as separate conclusions within this supporting PR.
2. Keep `eager` as the default; use
   `model.set_attention_backend("triton_bf16_two_pass")` only for the validated
   BF16/native-GQA configuration.
3. Do not claim that the BF16 kernel supports V-FP8 or V-INT4. A fused
   quantized-KV backend would require a new implementation and validation gate.
4. Return to the frozen main study: context length x KV representation x decoding
   method on RTX 4060 8GB, with speculative-decoding acceptance, latency,
   throughput, speedup, and peak VRAM as planned outcomes.
5. Present this investigation as supporting system evidence rather than changing
   the project's formal 3 x 2 x 2 design.

## 7. Evidence provenance

The source archive supplied after the RTX 4060 runs has SHA-256:

`0450c52a87f74d533ed6fe44051a6f3997c6201367af0a09f7690132fa265228`

Contained evidence:

| Artifact | SHA-256 |
|---|---|
| `14_streaming_attention_probe.json` | `af38b53c5a41e972763b0bec98047f8e5c36d15b49ebb6a000acba47e6a19a9b` |
| `14_streaming_attention_probe.log` | `30e42dfe8cf9f369b100f33f342e88fcb16252308b1b5fb7a51e2c0df517e5c0` |
| `15_streaming_attention_triton_probe.json` | `228b4424aa826974a59dee97582e30a6ee0d416a0362593e654d75dd24107f7b` |
| `15_streaming_attention_triton_probe.log` | `ccb12e035af33f6b2154f6bd1c6596d3d29e203ff73337e8c9a180881d2528f9` |
| `16_triton_vfp8_engine_trial.json` | `450c41b7b58e04feb7b33a9dd59e95630699a61c47eb62ddcdd543de0a8489ea` |
| `16_triton_vfp8_engine_trial.log` | `17226df67df6352eb61f6752e8c24589e3f8b354683a774caef2f9f3cec61280` |

Raw logs remain external run artifacts; repository documentation records only
the reviewed summary and provenance hashes.

The final BF16 FlashAttention evidence archive has SHA-256:

`6c3c5aa1c5f9f57df6a088b6fbcb459bfac618a1d91dd1dfc946a1afc073e6e2`

Key reviewed JSON hashes:

| Artifact | SHA-256 |
|---|---|
| `flash_attention_two_pass_core_probe.json` | `9f5d3159d076cc27a7e44f3dd7a6c23eea502c57430358db9ac87cc067c8a62b` |
| `flash_attention_two_pass_full_trial.json` | `c22c4ed951c071d151dcb1fee6a6f11836a9b1c5442f216c0c1126a98ef7d737` |
| `flash_attention_two_pass_ppl_gate.json` | `0491d0fb74899fc7cc73db355e515096a54cd590433b828db23e7635f7ceabdf` |
