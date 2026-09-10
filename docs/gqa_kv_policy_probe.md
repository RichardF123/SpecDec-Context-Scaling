# GQA KV-head x quantization probe (pilot protocol)

Status: **experimental screening only**. This does not replace the frozen
Asymmetric V-only FP8 result unless an independent confirmation run supports an
explicit protocol amendment.

## Question

Qwen2.5-1.5B has 12 query heads but only 2 native KV heads, so GQA has already
reduced KV storage by 6x relative to a 12-KV-head MHA cache. Can the remaining
cache be reduced beyond the current K-BF16 + V-FP8 policy without reproducing the
large quality loss observed for K-FP8?

## Mechanism and scope

- Directly reducing 2 native KV heads to 1 changes the pretrained attention
  function (GQA -> MQA). It is not a lossless cache encoding. The GQA conversion
  literature uses uptraining; therefore the no-training mean-merge variants here
  are diagnostic ablations only.
- Existing project evidence localizes the large loss to K-FP8. The primary new
  candidate keeps both native heads and stores K as symmetric RTN INT8 while
  retaining the frozen per-token/per-head V-FP8 representation. INT8 has the same
  one-byte payload as FP8 but much finer resolution near zero.
- KIVI and KVQuant motivate different treatment of K and V (especially
  per-channel/pre-RoPE handling for aggressive K quantization). This first probe
  deliberately tests the smallest implementation change. If native-head K-INT8
  is still inaccurate, the next experiment is blockwise per-channel K
  quantization, not another FP8 group-size sweep.

## Frozen pilot matrix

The probe evaluates the Cartesian product below on the existing pilot windows
0..3 only:

- format: current `v_fp8` or candidate `k_int8_v_fp8`;
- stored K heads: 2 (native) or 1 (deterministic mean merge);
- stored V heads: 2 (native) or 1 (deterministic mean merge).

A native-head BF16 baseline is run once. The confirmation windows 40..79 are not
read during screening.

| Policy | Logical bytes/token/layer | Compression vs BF16 | Interpretation |
|---|---:|---:|---|
| K-BF16(2) + V-BF16(2) | 1024 | 1.00x | baseline |
| K-BF16(2) + V-FP8(2) | 776 | 1.320x | current frozen policy |
| K-INT8(2) + V-FP8(2) | 528 | 1.939x | primary new candidate |
| K-BF16(1) + V-FP8(2) | 520 | 1.969x | K-head reduction diagnostic |
| K-BF16(2) + V-FP8(1) | 644 | 1.590x | V-head reduction diagnostic |
| K-BF16(1) + V-FP8(1) | 388 | 2.639x | full head reduction diagnostic |
| K-INT8(1) + V-FP8(2), or K-INT8(2) + V-FP8(1) | 396 | 2.586x | combined diagnostics |
| K-INT8(1) + V-FP8(1) | 264 | 3.879x | most aggressive diagnostic |

Scale metadata is included: 4 bytes per token per stored head for each quantized
matrix. These are logical cache bytes, not whole-process peak VRAM.

## Evaluation and decision discipline

1. Run `python -m specdec.engine.kv_policy_v1`. Stop on any structural failure.
2. Run `python -m specdec.engine.kv_policy_pilot` once on the target RTX 4060.
3. Compare mean dNLL first, then cache bytes. Do not interpret a small negative
   dNLL as an improvement and do not average these results with another GPU.
4. Select at most one non-baseline candidate based on the pilot. Freeze its exact
   format/head policy before reading windows 40..79.
5. Run the selected candidate against base and the current V-only policy on the
   independent 40-window confirmation set. Only then decide whether an amendment
   is justified.

The pilot does not measure or claim speedup. Eager dequantization may make a more
compressed representation slower; latency belongs to a separate target-hardware
measurement after quality selection.

## References

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints*, EMNLP 2023: https://arxiv.org/abs/2305.13245
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*,
  ICML 2024: https://arxiv.org/abs/2402.02750
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with
  KV Cache Quantization*, NeurIPS 2024: https://arxiv.org/abs/2401.18079
