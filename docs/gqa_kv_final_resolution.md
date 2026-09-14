# Final resolution: GQA-aware KV compression and fused consumption

Status: **supporting investigation complete; no production-policy change**  
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

### 3.3 Isolated whole-model trial

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

## 4. Final engineering decision

The project will **not integrate the current Triton kernel into the production
engine**.

The attention-core result is real, but system-level benefit is modest: 7.36%
decode-latency reduction and 56.1 MiB lower allocated peak, with no reserved-memory
or logical-cache reduction. More importantly, the kernel does not preserve the
current engine's numerical behavior. Matching the eager BF16 score, probability,
value, and accumulation semantics would require a more complex multi-stage
kernel and another quality campaign; that effort is disproportionate to the
observed whole-model gain and is outside the course project's core speculative-
decoding question.

The adopted policy remains:

- default: **K-BF16 + V-FP8 in the validated eager engine**;
- optional memory-priority ablation: **K-BF16 + packed V-INT4 group-32**;
- rejected: direct KV-head reduction, both K-INT8 variants, recent-BF16 residual,
  Python blockwise attention, and the current Triton whole-model integration;
- frozen G3 and the formal 3 x 2 x 2 experimental design remain unchanged.

## 5. Answer to the original question

GQA-aware quantization can reduce logical KV bytes further, and fused blockwise
consumption can almost eliminate temporary attention allocation. However, among
the tested methods, **none simultaneously improves whole-model memory, latency,
and output fidelity enough to replace K-BF16 + V-FP8 as the default**. The
remaining bottleneck is no longer merely how the cache is encoded; it is the
interaction between kernel execution, allocator-visible peak memory, and the
engine's numerical semantics.

This is a useful negative result rather than an absence of a solution: it closes
unmotivated head/group sweeps, provides an optional V-INT4 trade-off, validates
the fused-consumption mechanism, and supplies an evidence-based stopping rule.

## 6. Project plan after closure

1. Submit and review the existing GQA-aware policy PR without adding Triton code.
2. Preserve the Triton files and whole-model harness only on the experimental
   branch for reproducibility.
3. Do not spend further project time on compatibility-kernel engineering unless
   the team explicitly changes scope.
4. Return to the frozen main study: context length x KV representation x decoding
   method on RTX 4060 8GB, with speculative-decoding acceptance, latency,
   throughput, speedup, and peak VRAM as planned outcomes.
5. In the report, present this investigation as supporting mechanism evidence,
   not as the main contribution and not as a universal negative claim.

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
