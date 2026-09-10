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

## Round-1 pilot outcome (RTX 4060 8GB)

The native-head current policy remained near-lossless (`dNLL=+0.0016`,
`relative dPPL=+0.161%`, `1.320x` logical KV compression). Direct 2-to-1 K/V
head merging was catastrophic (`dNLL >= +7.44`) and is rejected without an
explicit uptraining procedure. Per-token/per-head K-INT8 + V-FP8 improved
substantially over the earlier K-FP8 negative ablation but still produced
`dNLL=+0.1701` / `relative dPPL=+18.541%`; it is not promoted to formal
confirmation.

## Round-2 targeted candidates

Round 2 retains both native KV heads and tests only mechanisms motivated by the
round-1 localization:

- `v_int4`: K remains BF16; V uses symmetric packed INT4 with per-token,
  per-head, group-32 scales.  Logical storage is 672 B/token/layer (`1.524x`
  vs BF16), compared with 776 B for the current V-FP8 policy.
- `k_pc_int8_v_fp8`: completed 32-token K blocks use per-channel INT8 scales;
  an incomplete K tail remains BF16, while V retains the current FP8 encoding.
  At a block boundary this is 552 B/token/layer (`1.855x` vs BF16).  The BF16
  residual is always fewer than 32 tokens and makes incremental decode valid.

Run `kv_policy_v2` first, then `kv_policy_pilot_v2`.  The second pilot repeats
the baseline/current rows for paired comparison but excludes all rejected
head-reduction and per-token K-INT8 variants, so it evaluates only 16
policy-windows.  Formal windows 40..79 remain untouched.

### Round-2 selection freeze (before reading confirmation windows)

RTX 4060 pilot results on windows 0..3 were:

| Policy | dNLL | relative dPPL | Compression vs BF16 |
|---|---:|---:|---:|
| current V-FP8 | +0.001600 | +0.161% | 1.320x |
| packed V-INT4 group32 | +0.006937 | +0.696% | 1.524x |
| K per-channel INT8 block32 + V-FP8 | +0.008800 | +0.884% | 1.855x |

The **single selected confirmation candidate** is K per-channel INT8 block32 +
V-FP8 with both native KV heads.  It uses 552 vs 776 B/token/layer for the
current policy (28.9% fewer logical KV bytes), while its pilot quality difference
is under 1% relative PPL.  V-INT4 is not advanced because it saves only 13.4%
over the current policy while providing a small 0.188 percentage-point pilot
quality advantage over the selected candidate.  Run `kv_policy_confirm_v2` once
on offsets 40..79; only after reviewing that result may the 8K operational VRAM
comparison be run.

### K per-channel INT8 confirmation and operation

The frozen 40-window confirmation produced `dNLL=+0.013580` and relative
`dPPL=+1.367%` versus BF16 (`paired dNLL SE=0.001292`).  The current V-FP8
policy produced `dNLL=+0.000397` / `dPPL=+0.040%`.  The selected candidate is
therefore a statistically stable, modest quality-memory trade-off, not a
near-lossless replacement for the current policy.  Its logical cache remains
552 B/token/layer (`1.855x` vs BF16; 28.9% lower than current).

At 8K, the candidate reduced logical cache by 49.00 MiB versus current V-FP8,
but median peak allocated memory fell only 17.8 MiB; median reserved memory rose
76.0 MiB and decode was 13.9% slower.  Prefill was unchanged (+0.4%).  It is
therefore retained only as a negative/diagnostic ablation.

### Recent-BF16 K residual follow-up

Keeping 128 recent K tokens in BF16 improved the four-window pilot from
`dPPL=+0.884%` to `+0.571%` at `1.809x` compression; residual 256 was dominated
on both quality and storage.  A new, untouched 80..119 confirmation set removed
the apparent gain: residual-128 produced `dNLL=+0.013270`, `dPPL=+1.336%`, and
paired SE `0.001009`, essentially the same as the no-residual candidate on its
independent set.  This demonstrates why the four-window pilot must not be used
as final evidence.  Residual-128 was rejected without another 8K run.

### Packed V-INT4 confirmation and operation

The remaining mechanism keeps sensitive K entirely in BF16 and packs V as
symmetric INT4 with per-token/per-head/group-32 scales.  Its independent 32-
window confirmation used untouched offsets 4..35 and produced
`dNLL=+0.009427`, relative `dPPL=+0.947%`, and paired SE `0.001018` versus BF16.
It uses 672 B/token/layer (`1.524x` vs BF16; 13.4% below current V-FP8).

On RTX 4060 8GB at 8K context, three rotated-order rounds produced:

| Policy | Peak allocated | Peak reserved | Logical 8K KV | Prefill | Decode |
|---|---:|---:|---:|---:|---:|
| BF16 KV | 5.225 GiB | 7.123 GiB | 224.00 MiB | 3.03 s | 2.31 s |
| Current V-FP8 | 5.146 GiB | 7.123 GiB | 169.75 MiB | 3.05 s | 2.97 s |
| BF16-K + packed V-INT4 | 5.102 GiB | 7.125 GiB | 147.00 MiB | 3.05 s | 3.75 s |

All policies were finite and token-stable across three repetitions.  Relative
to current V-FP8, V-INT4 reduced peak allocated memory by 45.2 MiB and logical
cache by 22.75 MiB; reserved memory changed by only +2.0 MiB and prefill by
+0.3%, but decode was 26.2% slower.

### Final decision

- Keep **K-BF16 + V-FP8** as the default: it remains essentially lossless and
  has the best practical latency-quality balance in this eager engine.
- Keep **K-BF16 + packed V-INT4 group32** as an optional memory-priority mode:
  it gives the largest observed incremental peak-allocation reduction, at a
  confirmed ~0.9% PPL and 26.2% decode cost versus current V-FP8.
- Do not reduce pretrained GQA heads directly and do not promote either K-INT8
  variant.  Further material peak reduction requires blockwise/fused consumption
  of quantized KV rather than another head/group sweep; that execution work is
  outside this completed probe.

## References

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints*, EMNLP 2023: https://arxiv.org/abs/2305.13245
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*,
  ICML 2024: https://arxiv.org/abs/2402.02750
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with
  KV Cache Quantization*, NeurIPS 2024: https://arxiv.org/abs/2401.18079
