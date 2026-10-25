# SpecDec acceptance-position and margin diagnostics

Date: 2026-10-26  
Status: development diagnostic; not a timing benchmark and not a formal-family result

## Purpose

Phase O removed most Python-side synchronization from the speculative loop, but only
improved decode time by about 3.8--4.5%.  This diagnostic asks a different question:
where does the fixed `k=4` proposal block fail, and are target rejections mostly
near-ties or substantive target/draft disagreements?

## Frozen configuration

- Target: Qwen2.5-1.5B-Instruct; draft: Qwen2.5-0.5B-Instruct.
- BF16 compute and base-BF16 KV for both models.
- `cudnn_prefill` for prefill only; eager execution for decode/propose/verify/repair.
- Greedy decoding, `k=4`, chunk size 2048, at most 64 generated tokens.
- Development-only families F09, F03, F11 and F04 at 2048, 4096 and 8192 tokens.

The runner first executes the uninstrumented Phase-O implementation as a reference.
It then replays the same case while retaining the draft and target decision rows.
Top-k, rank and margin kernels run only after generation has finished.  This ordering
matters: an earlier prototype computed top-k inside the generation loop and changed
BF16 near-tie trajectories.  That prototype is invalid and is not included here.

## Correctness and observer-effect gate

- Reference token equality: **12/12 cases** (all 64 generated tokens equal).
- Reference counter equality: **12/12 cases**.
- Target/draft cache invariants: **12/12 cases**.
- P-state preflight: qualified (median SM 2445 MHz; memory 8000 MHz).

Therefore the retained diagnostic rows describe the same trajectories as the
uninstrumented implementation. Diagnostic runtime is intentionally excluded from
all speed comparisons.

## Main results

| Context | Families | Median proposal acceptance | Median rounds | Median mismatch rounds |
|---:|---:|---:|---:|---:|
| 2048 | 4 | 0.579 | 23.0 | 13.0 |
| 4096 | 4 | 0.599 | 22.0 | 13.5 |
| 8192 | 4 | 0.429 | 26.0 | 19.5 |

The 8K development cases have lower median acceptance and require more rounds, but
the effect is not uniform across content.  F09 falls from 0.658 at 2K to 0.309 at
8K, whereas F03 rises from 0.516 to 0.776.  With four development families, this is
evidence of strong family-by-context interaction, not proof that acceptance decreases
monotonically with context length.

### Conditional acceptance by proposal position

| Context | Position 1 | Position 2 | Position 3 | Position 4 |
|---:|---:|---:|---:|---:|
| 2048 | 0.820 (89) | 0.803 (71) | 0.719 (57) | 0.795 (39) |
| 4096 | 0.818 (88) | 0.750 (72) | 0.865 (52) | 0.750 (44) |
| 8192 | 0.686 (102) | 0.686 (70) | 0.771 (48) | 0.861 (36) |

Parentheses give the number of decision-relevant rows reaching that position.  These
rates are conditional: position 4 is observed only after positions 1--3 were accepted.
Its high value must not be interpreted as unconditional evidence that later proposals
are easier.

### Target margin and draft-token rank at first mismatch

| Context | Mismatches | Median target top1-top2 margin | Margin <=0.1 | Margin <=0.3 | Draft rank median | Draft token in target top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 54 | 0.500 | 7 (13.0%) | 19 (35.2%) | 3 | 81.5% |
| 4096 | 52 | 0.375 | 8 (15.4%) | 23 (44.2%) | 2 | 88.5% |
| 8192 | 70 | 0.500 | 6 (8.6%) | 26 (37.1%) | 3 | 81.4% |

Only 21/176 first mismatches (11.9%) have a target top-1/top-2 margin at most 0.1,
while 108/176 (61.4%) exceed 0.3.  Thus BF16 near-ties exist but cannot explain most
rejections.  The dominant failure is genuine model disagreement: the 0.5B draft often
proposes a plausible token (81--88% are still in the target top-5), but not the target
argmax required by greedy speculative decoding.

## Implications

1. The lack of speedup has two independent causes.  Phase O shows substantial fixed
   eager-launch/dual-model overhead remains; this diagnostic shows that rejection and
   repair work also remains substantial, especially in three of four 8K families.
2. Increasing `k` alone is unlikely to solve the problem.  A longer block saves target
   calls only when the early draft tokens are accepted; otherwise it creates more
   counterfactual draft/verification work.
3. The next high-value experiment is mechanism-oriented, not another broad matrix:
   compare a stronger or better-aligned draft method while retaining this exact
   acceptance-position schema.  EAGLE-style feature drafting is a plausible research
   branch, but it is not a drop-in replacement and should remain separate from the
   completed fixed-model-pair study.
4. Formal conclusions must use the frozen formal families.  These four development
   families can motivate hypotheses and explain mechanisms, but cannot estimate a
   population effect by themselves.

## Reproduction and artifacts

```powershell
python -X utf8 -m experiments.acceptance_diagnostics.run_dev
python -X utf8 -m experiments.acceptance_diagnostics.analyze <run-directory>
```

- Runner: `experiments/acceptance_diagnostics/run_dev.py`
- Analyzer: `experiments/acceptance_diagnostics/analyze.py`
- CPU tests: `experiments/acceptance_diagnostics/test_analyze_cpu.py`
- Figure: `docs/figures/acceptance_diagnostics/acceptance_diagnostics_dev.{png,pdf,svg}`
- Tables: `docs/figures/acceptance_diagnostics/*_dev.csv`
- Raw local run: `output/acceptance_diagnostics/20261025T164950Z_71c351c9/`

