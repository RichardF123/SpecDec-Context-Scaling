# Formal raw-collection entry, 2026-10-23

## Authorization and scope

The user explicitly requested moving directly to formal experiments and delegated the integration on 2026-10-23. This records authorization for raw collection, not certification of acceleration or same-output production BF16 behavior. Earlier documents saying formal collection was not authorized describe the earlier state and are retained.

The entry is `python -m experiments.formal.run --formal --execute`. Without `--execute` it only prints the plan. `--formal`, `--pilot` and `--limit` are mutually exclusive. The formal input manifest remains the previously frozen `docs/prompt_families_formal_manifest.json` (`formal-v1`), FF01-FF16, with no outcome-driven selection. The six main conditions are unchanged: 2048/4096/8192 tokens by target-only AR / dual-model SpecDec; BF16, unquantized KV, k=4, chunk=2048, max_new=64 and EOS inclusion unchanged. Two randomized blocks, seed 4207, measured repeats 3+2, one discarded warm-up per process, serial execution and 10-second cooldown remain unchanged.

Totals: 192 independent worker jobs, 480 measured repetitions, 192 discarded warm-ups and 192 separate diagnostic passes. No pilot record is relabeled formal, and no FP8 condition is added.

## Implementation baseline

This revision changes orchestration, input routing, provenance and record validation, not model math. The new runner baseline is the clean HEAD captured in each formal batch's `expected_commit` and `runtime.git_commit`; each worker must match that commit and have a clean worktree before model loading. This is an explicit runner-baseline update, not a claim that the old document-only commit exemption applies. The actual full hash is stored automatically to avoid a self-referential commit hash in this file.

`input_sets.py` selects the manifest and allowed family set. `worker.load_input` reconstructs token IDs through the existing hash-verified loader and corrects its historical hard-coded manifest-path metadata to the selected path, adding a SHA256 digest. Formal and pilot family IDs cannot be mixed.

Formal raw results use `experiment_stage=formal`, separate `output/formal_batches`, `output/formal_launches` and `output/formal_worker` namespaces. Pilot defaults are unchanged. The parent still verifies every job. Formal records must have formal family/input identity, verified input hashes, clean code provenance and a matching expected commit. The batch stores its manifest hash. A worker cannot accidentally run formal collection without an expected commit from the batch entry.

## Observation and declaration policy fixed before collection

The completed pilot found timing variability without an identified repair. For 8K, use the pre-registered section 11b route C for reporting: retain its raw timing and resource evidence, but do not publish an 8K speedup factor as a formal result. All six conditions remain in collection and in the complete report. This is a reporting limitation, not deletion of C5/C6 or a claim of OOM. New data do not automatically lift this restriction; any later change must be separately justified, not chosen from favorable runs.

For all contexts, retain all repeats and report median/min/max/MAD/IQR and the paired-block differences. Use the existing throughput-based within-process spread and symmetric block difference normalized by the larger block median. Values above 20% are CHECK diagnostic flags, not permission to delete or retry samples. Unstable units cannot support definite speedup claims or silently disappear from tables. There is no automatic retry or resume feature in this entry.

Boundary telemetry remains outside timed intervals, using the already integrated monitor_start/end audit. It is accepted as the protocol's boundary-audit evidence; continuous high frequency is not a new admission requirement. Boundary samples do not prove the entire interval stayed at a frequency. Missing samples remain unavailable; low boundary clocks are explicitly retained for evaluation and are not alone proof of whole-interval environment failure. No new sampling thresholds or model precision switches are introduced.

All raw records and batch summaries start with `speedup_claim_allowed=false` (in worker `collection` metadata). Execution/record verification success is not measurement validity or a statistical conclusion. Formal paired analysis must still check status, inputs, work volume, environment evidence and divergence; no blanket lossless same-output claim is permitted. Eager-era quality numbers are not promoted into current-backend evidence. FP8 and optional generation-quality studies remain separate.

## Failure policy and resource checks

Batch entry requires at least 18 GiB free system commit; each worker retains the 16 GiB gate and GPU load preflight. Version mismatch, dirty worktree, OOM, crashes, nonfinite decisions, control/cache failure, incomplete repetitions, or measured-vs-own-diagnostic token differences stop the batch and preserve evidence. Environment refusal is not bypassed. No background applications are killed or system settings modified by the runner.

## Verification performed before launch

- CPU tests include a complete 192-job fake-worker batch, producing and validating 480 synthetic records under a temporary directory that is removed afterward. Synthetic data never enter the real result directories.
- The real formal dry-run prints 192 jobs / 480 measured and does not load models or tokenize formal inputs.
- The actual launch, environment refusal or completion must be reported separately; this document is not evidence that GPU collection has happened.
