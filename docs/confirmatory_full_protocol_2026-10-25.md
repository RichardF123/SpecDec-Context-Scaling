# SpecDec independent confirmatory measurement protocol

Date: 2026-10-25  
Status: **frozen candidate; must be committed before GPU execution**  
Scope: FF01–FF16, BF16 main line, 2K/4K/8K × AR/SpecDec.

## 1. Purpose and independence

The first formal collection remains immutable. This new batch independently measures the same six conditions after the development-family pilot showed that time-interleaved scheduling substantially reduced the old 4K block-era drift. Pilot families F09/F03/F11/F04 are not used in the full batch, and no formal family is selected according to pilot latency, acceptance or output equality.

## 2. Frozen model and generation configuration

- Target: Qwen2.5-1.5B-Instruct; draft: Qwen2.5-0.5B-Instruct.
- BF16 compute; target and draft use uncompressed BF16 KV (`fp16` engine label).
- `cudnn_prefill` only for prefill; decode, propose, verify and repair remain eager.
- Target RoPE uses the existing CPU-FP32 inverse-frequency source; draft retains its existing path.
- Contexts: 2048, 4096 and 8192; chunk 2048; speculative block `k=4`.
- Greedy decoding; `max_new=64`; EOS 151645.
- No CUDA Graph, SDPA, FP8, alternate `k`, alternate chunk or model change enters this batch.

## 3. Inputs and measurement count

- Formal families are exactly FF01–FF16 from `docs/prompt_families_formal_manifest.json`.
- Each family provides nested 2K/4K/8K prefixes reconstructed from the frozen source and verified against the manifest hash.
- Each `(family, context, mode)` has five measured repetitions: block 0 has three and block 1 has two.
- Each process performs one discarded warm-up before measured repetitions.
- Total: 16 families × 3 contexts × 2 modes × 2 blocks = **192 process tasks and 480 measured records**.

## 4. Scheduling

For every `(family, context, block)`, AR and SpecDec form one adjacent two-process pair. They remain independent processes and do not share models, caches or allocator state.

- AB means AR then SpecDec; BA means SpecDec then AR.
- Within each context, exactly eight families use AB and eight use BA. The assignment is fixed by seed 4207 and reused in both blocks.
- Family/context pair order is independently shuffled inside each block.
- The global schedule alternates complete block-0 and block-1 pair units. Thus pair members remain adjacent while block identity does not encode an early or late run era.
- A 10-second cooldown remains between process tasks.

## 5. Provenance and execution gate

GPU execution is refused unless all of the following hold:

1. the worktree is clean;
2. `--expected-commit` equals `HEAD`;
3. the formal manifest contains exactly FF01–FF16;
4. the entry commit-memory gate has at least 18 GiB free;
5. the plan validates as 192 tasks / 480 measured, with 8/8 AB/BA per context and 96 adjacent pair units.

The batch records the commit, runtime information, manifest SHA256, this protocol's SHA256, runner/model source SHA256 values, complete planned order and actual start/end timestamps.

## 6. Hard-stop conditions

Stop without automatic retry after OOM, native crash, environment refusal, NaN/Inf, input hash mismatch, weight mismatch, cache/round invariant failure, missing/duplicate task, same-mode output inconsistency or an 8K resource failure. Preserve all evidence. AR/Spec output divergence is recorded and is not a hard failure.

## 7. Analysis unit and estimands

The statistical unit is the family, not an individual repetition.

1. Within each family/context/mode, combine the five repetitions by their median decode time.
2. For each family/context, compute `R = T_AR,decode / T_Spec,decode`; `R>1` means SpecDec is faster.
3. Report the 16 family ratios, their median and range. Any confidence interval or bootstrap procedure must operate over the 16 paired families, never over the 80 individual records.
4. Report prefill and end-to-end time separately from decode time.
5. Report acceptance, rounds, proposed/accepted tokens, verify/repair calls, generated length and memory beside latency.
6. Report within-process, between-block and boundary checks separately. Do not delete or replace a family because it receives CHECK.

## 8. Output divergence and claim boundary

The primary timing table includes all completed trajectories and reports output equality, common-prefix length and first divergence. If AR and SpecDec outputs differ, the ratio compares the observed latency of two generation strategies on divergent trajectories. It must not be called lossless or identical-output speedup.

An identical-output subset may be shown as a secondary descriptive analysis, with its family count displayed. It is not substituted for the full frozen family set.

## 9. 8K rule

The old batch remains under Route C. In this independent batch, an 8K family ratio is available only when its six timing checks are all OK. Families that do not satisfy this remain visible through absolute latency, workload and resource observations and are labelled restricted. The analysis must display the number of reportable and restricted 8K families.

## 10. Reporting discipline

- The pilot and first formal batch are historical supporting evidence, not pooled with this batch.
- Profiled Nsight/CUDA Graph times are mechanism diagnostics and are not mixed with formal unprofiled latency.
- A median ratio below one is described as observed slowdown, not failed correctness.
- Context-length trends are reported with family-level distributions and their uncertainty; no universal hardware/model claim is made.
- Missing, refused and failed conditions remain in the denominator of the execution audit.

