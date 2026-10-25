# Evidence Map: Confirmatory Speculative-Decoding Result

This working map separates formal performance evidence from mechanism diagnostics and records the boundary of every claim.

| Claim | Level | Supporting artifact | Supports | Does not support |
|---|---|---|---|---|
| Full batch completed | L1 direct | `measurements/confirmatory_evidence_manifest_2026-10-25.json` | 192/192 jobs, 480/480 measured records, frozen commit and hashes | Scientific interpretation by itself |
| SpecDec slower at 2K/4K | L1 direct | `measurements/confirmatory_pair_summary_2026-10-25.csv`; `confirmatory_context_summary_2026-10-25.csv` | 16/16 ratios below one; medians and intervals | Other hardware, batch sizes, or model pairs |
| 8K remains restricted | L1 direct | Same pair/context summaries and `confirmatory_boundary_counts_2026-10-25.csv` | Four reportable families, 12 restricted; all 16 descriptive | Unrestricted population estimate at 8K |
| Output paths can diverge | L1 direct | `confirmatory_pair_summary_2026-10-25.csv` | Same-output counts and deterministic within-mode repetition | Quality equivalence or quality degradation |
| Confirmatory schedule reduces 4K block confound | L1 direct comparison | `block_shift_summary_2026-10-24.csv`; `confirmatory_block_shifts_2026-10-25.csv` | Earlier and confirmatory block-shift distributions | Elimination of every temporal/environmental confound |
| Acceptance is associated with ratio | L1 derived | `confirmatory_pair_summary_2026-10-25.csv` | Pearson association across 16 families per context | Causality or an isolated acceptance intervention |
| Draft and repair dominate measured stage time | L1 diagnostic | `cost_profile_2k_2026-10-23.csv`; `operator_evidence_2026-10-24.csv` | Stage shares for four 2K development families | Formal held-out-family latency decomposition |
| Proposal path has large uncovered gaps | L1 diagnostic | `operator_evidence_2026-10-24.csv` | Kernel-covered versus uncovered span for F09 2K | Complete attribution of gaps to CPU, memory, or synchronization |
| CUDA Graph exposes replayable overhead | L1 diagnostic control | `cudagraph_chain_corrected_2026-10-24.csv` | Same-adapter 16-step eager versus replay timing | End-to-end SpecDec speedup or production integration result |

## Contribution-to-evidence trace

1. **Controlled negative result:** formal pair/context summaries and the evidence manifest.
2. **Robustness against earlier block confounding:** old and confirmatory block-shift tables plus AB/BA order effects.
3. **Mechanistic explanation:** stage ledger, Nsight proposal trace, and CUDA Graph diagnostic.
4. **Correctness boundary:** same-output, common-prefix, repeatability, and timing-qualification fields in the pair summary.

## Writing discipline

- Use “under the frozen configuration,” not “speculative decoding is slower.”
- Use “associated with,” not “caused by,” for acceptance correlations.
- Use “uncovered trace gaps,” not “CPU overhead,” unless a CPU-side trace is added.
- Use “diagnostic optimization headroom,” not “CUDA Graph speedup,” for the isolated graph experiment.
- State `FORMAL_COMPLETE_WITH_RESTRICTED_8K` wherever a single overall verdict is required.
