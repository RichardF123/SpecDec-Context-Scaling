# When Does Speculative Decoding Stop Paying Off? Context Length and FP8-Compressed KV Cache on an 8GB GPU

Fang Chuming 23097386D    Deng Xiyue 23110509D    Cheng Shuyue 23097089D

## 1. Research Question & Motivation

This project studies the question: **How does context length affect speculative-decoding speedup on an 8GB consumer GPU, and does FP8-compressed KV cache change the context range in which speculative decoding remains beneficial?** It also examines whether any observed degradation is mainly associated with reduced accepted-token benefit or increased draft/verification overhead.

Speculative decoding can reduce generation latency by using a smaller draft model to propose several tokens and a larger target model to verify them. Its practical benefit may depend strongly on context length. As context grows, KV-cache memory usage and verification cost increase, which may reduce the advantage of accepting multiple draft tokens per target-model step. This issue is especially relevant for small open-weight models running on consumer GPUs, where memory pressure can determine whether long-context inference is feasible. The project is an Efficient LLM Inference empirical systems evaluation focused on controlled measurement of context length, FP8-compressed KV cache, and speculative decoding under an 8GB GPU constraint.

## 2. Method / System

The system uses a custom PyTorch eager inference engine for Qwen2.5 models. The target model is **Qwen2.5-1.5B-Instruct**, and the draft/development model is **Qwen2.5-0.5B-Instruct**. Existing development-stage validation on the 0.5B model includes full-forward correctness validation, KV-cache correctness validation, greedy generation correctness, chunked prefill implementation, 4K chunked-prefill equivalence, identification of the memory cause of raw 8K eager-prefill out-of-memory behavior, and a baseline sanity comparison against transformers eager. A Hugging Face Transformers eager implementation will also be used as an external sanity reference to check that the custom engine operates in a reasonable performance range.

Formal benchmarking will start after the corresponding correctness checks are completed for the 1.5B target model. The tested system includes KV caching, chunked prefill for long contexts, unquantized BF16 KV cache, FP8-compressed KV cache, standard autoregressive decoding, and fixed-chain speculative decoding with proposal length **k = 4** under greedy generation. FP8-compressed KV refers to storing KV cache tensors in compressed FP8 form and dequantizing them for eager attention, rather than implementing native FP8 attention.

## 3. Experimental Design & Evaluation

The central experiment is a **3 x 2 x 2 controlled factorial design** with three factors: context length, KV representation, and decoding method. Context length will be approximately 2K, 4K, and 8K tokens. KV representation will be FP16 or FP8-compressed. Decoding will be either standard autoregressive decoding or speculative decoding. This produces 12 experimental conditions.

Prompts will be constructed as paired prompt families. For each family, the 2K, 4K, and 8K contexts will be nested prefixes of the same base text, and the continuation task will remain the same across lengths. Contexts will target approximately 2K, 4K, and 8K tokens with a length tolerance of at most 32 tokens; actual token counts will be recorded and no synthetic filler will be inserted. A pilot run will use 4 prompt families to check feasibility, instrumentation, timing stability, and correctness. The formal experiment will use at least 16 prompt families.

Performance evaluation will report decode latency, end-to-end latency, tokens per second, speculative speedup, and peak VRAM. Decode-only speculative speedup is the primary performance metric. End-to-end speedup is secondary and includes target and draft prefill costs. Speedup comparisons will stay within the same KV representation: FP16 speculative decoding will be compared with FP16 autoregressive decoding, and FP8 speculative decoding will be compared with FP8 autoregressive decoding. Each condition will be measured over repeated runs, with performance summarized across prompt families rather than using best-case timing.

Speculative-decoding behavior will be measured using acceptance rate, accepted tokens per verification, draft latency, and verification latency. These metrics support empirical attribution of speedup changes across context lengths. The analysis will examine whether long-context degradation is more closely associated with accepted-token benefit, draft overhead, or verification overhead. KV bandwidth may be used as supporting interpretation, but it will not be treated as an independently isolated measured component.

FP8-compressed KV quality will be evaluated on the WikiText-2 test set using at least 32 fixed, non-overlapping windows per condition, with perplexity and Delta PPL reported relative to the unquantized BF16 KV baseline. Optional logit or top-1 sanity checks may also be used if useful. Perplexity is used to assess quality degradation from FP8 KV storage, not speculative-decoding performance. If speculative speedup crosses 1x between tested context lengths, one or two intermediate context lengths may be evaluated to estimate the break-even region.

## 4. Feasibility & Expected Outcomes

The project will run on an **NVIDIA RTX 4060 Laptop GPU with 8GB VRAM**. It uses small open-weight models and an already functioning custom inference engine, keeping the required compute resources within the scale of a course project. No model training or paid API usage is required; the main computational cost is local inference and benchmarking on a single RTX 4060 Laptop GPU. The remaining implementation, benchmarking, and analysis effort is estimated at approximately 140-175 person-hours across the three team members. The remaining core work includes 1.5B target correctness validation, formal 8K operational validation, FP8-compressed KV implementation, speculative-decoding implementation, instrumentation, pilot experiments, the final factorial experiment, and analysis.

The scope is restricted to protect the quality of evaluation and interpretation. Optional CUDA-kernel work and weight quantization are not part of the committed core project. The experiments will examine whether the memory savings from FP8-compressed KV translate into practical latency improvements and whether they shift the context range in which speculative decoding remains beneficial. Negative or mixed results would still be informative by clarifying the conditions under which FP8-compressed KV or speculative decoding provides limited practical benefit.

## 5. Team Contributions

**Fang Chuming**

Fang Chuming will take primary responsibility for custom inference-engine maintenance, Qwen2.5-1.5B target correctness and regression validation, KV-cache correctness, chunked-prefill integration, asymmetric V-only FP8 KV maintenance, FP8 scaling/dequantization and memory accounting, VRAM profiling, and KV-quality evaluation. Fang Chuming will also support system integration, reproducibility checks, failure diagnosis, and the final report.

**Deng Xiyue**

Deng Xiyue will take primary responsibility for the fixed-chain speculative-decoding implementation, target/draft cache synchronization, cache trimming and rollback, block-verification alignment, speculative-decoding correctness validation including cache synchronization, verification alignment, and output equivalence, acceptance instrumentation, and decode and end-to-end performance measurement. Deng Xiyue will also support experiment automation, failure diagnosis, reproducibility checks, and quantitative analysis.

**Cheng Shuyue**

Cheng Shuyue will take primary responsibility for paired prompt-family construction, pilot and final factorial experiment execution, experiment instrumentation and results-schema implementation, experiment configuration and results validation, repeated-run consistency checks, external-reference benchmarking, statistical aggregation, break-even analysis, figures and tables, and coordination of report and presentation materials. Cheng Shuyue will also independently review correctness evidence and contribute to reproducibility and failure-case analysis.

**Joint responsibilities**

All three members will jointly finalize the experimental protocol, review code and formal results, interpret negative or mixed findings, prepare the final report, and prepare the presentation.
