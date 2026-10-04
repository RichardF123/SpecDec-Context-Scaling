# H2O operational latency — 2026-10-04

This directory preserves the latency measurements supplied in
`SPECDEC_H2O_LATENCY_FINAL_2026-10-04.zip`.

## Provenance

- Source archive SHA-256:
  `596d455490f9c48e3eff3f7d8fb795315417c2441f40d64beda31eca2120fb14`
- The source manifest contains a known packaging defect: it lists
  `SHA256SUMS.txt` with the SHA-256 of an empty file. All other 26 listed files
  verified successfully. The original manifest is retained as
  `SOURCE_ARCHIVE_SHA256SUMS.txt`; this directory has a corrected independent
  `SHA256SUMS.txt` that omits self-hashing.
- The recorded environment used PyTorch 2.11.0+cu128, CUDA runtime 12.8,
  Transformers 4.57.1, NVIDIA driver 595.91.07, and an NVIDIA GeForce RTX 4060
  with 8188 MiB reported by `nvidia-smi`.
- The archived core H2O files were byte-identical to branch HEAD. Only the two
  benchmark modules and two runners were new.

## Experimental regimes

### `short_mid_context/`

- Contexts 512 and 2048 provide the retained final measurements.
- Prompt families: systems, reasoning, and code.
- Three paired repeats per family, giving nine valid pairs per KV format and
  context.
- The same first harness attempted 4096 with single-shot prefill and produced
  `NO_VALID_PAIRS`; its raw worker rows record OOM statuses. Those rows diagnose
  the unsuitable long-context harness and are not evidence that H2O itself is
  unsupported or failed at 4K.

### `long_context/`

- Contexts 4096 and 8192 use corrected chunked prefill with chunk size 512.
- Prompt family: systems only.
- Three paired repeats per KV format and context.
- Every requested cell produced 3/3 valid pairs with exact output and equal
  SpecDec statistics.

Do not pool these regimes as if they used the same prompt sampling.

## Final operational result

| Context | Target KV | Valid pairs | H2O/standard decode | Resident/seen |
|---:|---|---:|---:|---:|
| 512 | BF16 | 9 | 2.316x | 0.485 |
| 512 | V-FP8 | 9 | 2.382x | 0.485 |
| 2048 | BF16 | 9 | 3.079x | 0.496 |
| 2048 | V-FP8 | 9 | 3.111x | 0.496 |
| 4096 | BF16 | 3 | 3.186x | 0.498 |
| 4096 | V-FP8 | 3 | 3.239x | 0.498 |
| 8192 | BF16 | 3 | 4.206x | 0.499 |
| 8192 | V-FP8 | 3 | 3.465x | 0.499 |

At 8K BF16, individual decode-overhead ratios were 4.015x, 6.617x, and
4.206x. At 8K V-FP8 they were 3.298x, 3.718x, and 3.465x. The nontrivial
variation is part of the evidence.

These measurements describe the current reference score-capture and
materialize/gather/repack compaction implementation. They do not establish an
intrinsic latency cost for the H2O retention policy and do not constitute a
latency speedup claim.
