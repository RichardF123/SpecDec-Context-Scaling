# Confirmatory full-run readiness record

Date: 2026-10-25  
Branch: `codex/confirmatory-full-freeze`  
Clean worktree: `D:\RealDSH\specdec-confirmatory-freeze`

## Outcome

The confirmatory measurement implementation is ready for a clean, independently versioned full run. No full GPU batch was started while preparing this record.

## Corrections completed after the pilot

1. The runner now accepts the frozen `formal` input set (FF01–FF16); pilot remains the default to prevent accidental full execution.
2. AR and SpecDec are adjacent processes inside each `(family, context, block)` pair.
3. Complete pair units, rather than individual tasks, alternate between block 0 and block 1.
4. AB/BA is stratified by context: each context has exactly 8 AB and 8 BA formal families.
5. Formal execution requires a clean worktree and an explicit `--expected-commit` equal to `HEAD`.
6. The batch records the formal manifest, full protocol and source hashes.
7. The original pilot source and preregistration were snapshotted before these corrections because the pilot ran from an uncommitted worktree.
8. A separate formal analyzer now enforces 192 tasks / 480 measured records, pools the raw 3+2 repetitions before each family median, and computes uncertainty by resampling families rather than repetitions.

## CPU-only verification

- Confirmatory unit/dry-run tests: 29 passed, 0 failed.
- Pilot-analysis tests: 5 passed, 0 failed.
- Full-analysis synthetic tests: 4 passed, 0 failed.
- Formal input reconstruction: FF01–FF16 × 2K/4K/8K = 48/48 hashes reproduced.
- Formal plan: 192 tasks, 480 measured records and 48 family-context pairs.
- AB/BA: 24/24 globally and 8/8 in every context.
- Pair adjacency: 96/96 block-pairs adjacent.
- Pair-unit block schedule: max same-block run 2; 95 block transitions.
- Formal dry-run printed 192/192 tasks without loading a model, touching the GPU or creating a batch directory.
- Readiness audit includes the formal analyzer and its CPU test; `execution_ready_now=true` only in the clean worktree.

Readiness evidence:

`output/confirmatory_readiness/2026-10-25/full_readiness.json`

## Frozen protocol and command

Protocol: `docs/confirmatory_full_protocol_2026-10-25.md`

Before execution, read the current clean commit:

```powershell
cd D:\RealDSH\specdec-confirmatory-freeze
$confirmatoryCommit = git rev-parse HEAD
git status --short   # must print nothing
```

Dry-run:

```powershell
D:\RealDSH\specdec\.venv\Scripts\python.exe -X utf8 `
  -m experiments.confirmatory.run --input-set formal
```

Full GPU run (only after reviewing the 192-task dry-run):

```powershell
D:\RealDSH\specdec\.venv\Scripts\python.exe -X utf8 `
  -m experiments.confirmatory.run --input-set formal --execute `
  --expected-commit $confirmatoryCommit
```

The runner refuses a dirty worktree, a missing expected commit or a commit mismatch before loading a model.

After a complete batch, run the CPU-only formal analyzer:

```powershell
D:\RealDSH\specdec\.venv\Scripts\python.exe -X utf8 `
  -m experiments.confirmatory.analyze_full <batch-directory>
```

It writes family-level timing, block-shift, AB/BA-order, pair and context summaries,
plus SVG/PNG/PDF stability figures. An incomplete or aborted batch is refused.

## Claim boundary

The full batch measures strategy latency. AR/Spec output divergence remains explicitly recorded. A ratio on divergent trajectories cannot be called lossless or identical-output speedup. The old formal batch and its Route C status remain unchanged.

