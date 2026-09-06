# Course Project Proposal — DSAI 4207

**Project title:** SpecDec: Where Does Speculative Decoding Stop Paying Off on an 8 GB GPU — and Does KV-Cache Quantization Extend It?

**Team:** [Student 1 — …, Student 2 — …, Student 3 — …] *(fill names/IDs)*

---

## 1. What is your question?

> **On an 8 GB consumer GPU (RTX 4060), how much does each stage of LLM-inference optimization (KV caching → weight quantization → KV-cache quantization → fixed-chain speculative decoding) improve decode latency and memory — and at what context length does speculative decoding stop paying off, with vs. without a quantized (fp8) KV cache?**

We answer it with one controlled experiment at a time, measured on the same model (Qwen2.5-1.5B), same prompts, same seeds.

> **Incremental contribution (explicit):** we propose **no new method**. Our contribution is a reproducible, per-rung measurement and extension of known techniques (KV caching, int8/int4 weights, fp8 KV caching, fixed-chain speculative decoding), implemented from scratch in eager PyTorch on one 8 GB consumer GPU — including the interaction of fp8 KV × speculative decoding and measured break-even context lengths for this configuration. Note: the binding resources here are **VRAM (8 GB) and latency**, not money — local inference is $0, which is the realistic "low-resource" deployment setting.

## 2. Why is this interesting?

Small models running on laptop GPUs are the practical way most people deploy LLMs, but "use KV cache / quantize / try speculative decoding" are slogans — practitioners lack a single, reproducible measurement of *how much each technique is worth and when it stops helping* on an 8 GB device. Prior work (MagicDec, QuantSpec) reports acceptance-rate decay with context, but on datacenter-scale hardware and models. We provide a controlled empirical answer on the 8 GB consumer setting, with an honesty-first methodology (reproducibility, external baselines, documented failure cases).

## 3. What will you build or test?

- **Model:** Qwen2.5-1.5B (open-weight, small; also Qwen2.5-0.5B for fast iteration).
- **System (from scratch):** a minimal inference engine implementing forward pass, RoPE, GQA attention, KV-cache decode loop, greedy generation — written by the team, **not** wrappers around `transformers.generate`. (Status: engine + KV-cache + generation complete; single-forward alignment vs. the reference implementation is bit-exact, max|Δ|=0; generated sequences match official greedy decoding token-for-token.)
- **Optimization ladder as controlled experiments (one variable per rung):**
  - Baseline: naive autoregressive decode (no KV cache).
  - Rung 1: KV-cache decode loop.
  - Rung 2: int8 / int4 weight quantization.
  - Rung 3: fp8 KV-cache quantization (KIVI-style, per-channel keys / per-token values).
  - Rung 4: fixed-chain speculative decoding with a vocab-sharing draft model — **fixed choice: Qwen2.5-0.5B (shares the 151936-tokenizer with the 1.5B target; no training)**; a self-speculative layer-skip draft is only a fallback comparison.
- **Variables:** context length {2K, 4K, 8K} × KV dtype {fp16, fp8}; acceptance rate and net speedup vs. length.
- **Baselines / references:** our own no-cache baseline; the same model via `transformers` (official math, full-recompute greedy) as the correctness reference; published numbers of EAGLE-2 / MagicDec / QuantSpec as literature context.

## 4. How will you evaluate it?

Metrics are fixed **before** running: decode tokens/s and time-to-first-token (batch=1); peak GPU memory (VRAM) and KV bytes/token; acceptance rate and net speedup ratio for speculative decoding; **quality preservation** via perplexity and a small held-out task set (we verify the optimizations do not degrade the model, since a speedup that breaks quality is meaningless).

> **Metric roles (clarified):** ppl / task-set accuracy measure **quantization quality loss** (fp16 vs fp8 KV; int8/int4 weights). Speculative-decoding quality is measured by **acceptance rate + net speedup**, not ppl — greedy verify is lossless in expectation, so its ppl change should be ≈ 0 and is only a sanity check.

Reproducibility: fixed seeds, config-hash run ids, append-only `results.jsonl`, external reference comparison. **Failure analysis** (graded heavily): we will report *where* each technique stops helping — e.g., acceptance-rate collapse at long context, near-tie greedy flips, KV-quantization quality drift — with concrete examples.

## 5. Is it feasible?

- **Resources:** RTX 4060 Laptop 8 GB + i7-12700H + 32 GB RAM; free Colab as fallback. 1.5B QLoRA-grade training is NOT required (no fine-tuning in the core plan). Cost ≈ $0 (local inference); optional small hosted/rented runs ≤ $30.
- **Effort already demonstrated:** engine, KV cache, and generation are implemented and validated (alignment 0.000; generation alignment PASS); time budget for the remaining ladder ≈ within 7 weeks at 15–20 h/week (plan `docs/plan_v6.md`).
- **Division of work (2–3 members; names to be finalized by 10/7; contribution statement will be concrete):**
  - Member A (engine/architecture): forward pass, KV-cache, generation loop; correctness alignment.
  - Member B (quantization + measurement): int8/int4 and fp8-KV implementation; benchmark harness and cost instrumentation.
  - Member C (speculative decoding + analysis): draft+verify loop, acceptance-rate study, failure-case and report analysis. *(Adjust roles if 2 members.)*
- **Repo/reproducibility:** public repo with README (install/data/reproduce), pinned requirements, scripts and results directory; large checkpoints not uploaded (downloaded by script).

> *Note on prior work (transparency): this project was scoped and implemented during the course. No other-course submission is planned for the same work.*

---

*This proposal is a plan, not a contract — if early experiments reveal a more interesting question (e.g., KV-quantization behavior dominates), we will adjust and document why.*
