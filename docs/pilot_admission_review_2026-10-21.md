# Pilot admission review, 2026-10-21

## Evidence

- Source batch: `1f6b884a4b5946b09d4ee9a381507b21`, runner commit `a8a8ba4`.
- All 48 tasks and 120 measured records revalidated. No failed task; all outputs have 64 tokens.
- All task preflights qualified. This is not measured-phase P-state evidence.
- All measured outputs match their own-mode diagnostics; same-mode outputs are repeatable across blocks. AR/Spec equality is not universal.
- Original analysis v1 used latency spread and an asymmetric block ratio interval. Those are descriptive, not the existing report's exact admission metrics.
- Protocol-aligned v2 follows `scripts/g5_report_from_run.py`: within-process throughput range divided by median; between-block throughput median difference divided by the larger median.
- Protocol-aligned counts: 12/48 tasks above the 20% within-process diagnostic line; 12/24 family/condition units above the between-block line. The v1 counts of 13/48 and 13/24 are retained only for their explicitly different metrics.
- One selected monitored diagnostic completed with 7.3% latency spread. It did not reproduce the original large variation and cannot establish its cause. It is not a replacement sample.

## Admission decision

Execution and record-validation integration passed. Formal measurement admission is NOT established.

Under measurement protocol sections 7, 7b and 7c, execution success is separate from measurement validity. Unstable observations remain visible, receive CHECK and cannot support definite speedup claims. The 20% line is not a reason to delete samples or repeat until a desired outcome appears.

The current runner lacks the measured-rep runtime P-state audit required by section 7b.3. Its preflight occurs before model loading. Environment qualification for the measured phase must therefore remain unestablished, not silently OK or falsely ENV_INVALID.

Section 11b route A is not supported by this pilot's variation. No identified repair mechanism supports route B yet. Route C is the applicable proposed restricted-reporting route for 8K if no repair is pursued: retain 8K resource evidence but do not publish an 8K speedup factor. This review proposes that route; it does not silently amend the proposal or authorize formal collection. Unstable 2K/4K units also remain subject to section 7.

## Bounded next work

1. Add an offline, layered measurement assessment: execution, within-process, between-block, environment evidence availability and explicit claim blockers. Preserve source JSON and report median/min/max/MAD/IQR. Pilot data remain pilot data regardless of statistical stability.
2. Resolve section 7b.3 before formal collection: implement a precisely bounded runtime audit outside measured timing, or explicitly approve a versioned protocol amendment. Idle snapshots must not be represented as active-load evidence. Continuous in-timing polling remains diagnostic only.
3. Review and register the 8K route and the actual runner baseline. Do not rerun the entire pilot or launch the formal matrix by default.

No new algorithm correctness localization, model precision changes, selected slow-sample deletion, automatic retries, or formal collection are authorized by this document.

## Boundary-audit implementation follow-up

The new worker captures two timestamped snapshots outside each measured call and stores `runtime_environment_audit` (`boundary_audit_v1`). The new record validator requires this field by default, checks timestamp ordering, scope, evidence flags and the clock verdict recomputed from the snapshots. Missing telemetry is retained as unavailable; a low boundary clock does not prove a measured-interval environment failure. Boundary clock verdicts use the existing process-configured thresholds, not a new threshold.

The offline report recognizes complete boundary evidence as `BOUNDARY_ONLY`, not continuous active-load verification. This is an analysis-only status schema, not a silent amendment of canonical formal statuses. Old pilot evidence remains missing; historical reads require the explicit `require_runtime_audit=False` option. No source JSON is rewritten.

Integration case `output/formal_development/2ae421c369c64a6893e0db185bd44560` (F09/4096/SpecDec/block0) has three records accepted by the new validator. All six boundary snapshots are available and show SM 2535 MHz and memory 8000 MHz. This is development evidence from a dirty worktree, not formal data or a replacement for the pilot.

CPU regression: 42 tests passed, including scope/ordering/verdict forgery rejection, explicit legacy behavior, missing telemetry and low clocks. Latest full-pilot derived report: `output/pilot_analysis/1f6b884a_stability_v4_boundary_schema/`. Formal collection remains unauthorized; resolving interpretation of boundary-only evidence against section 7b.3 and registering the 8K route still require an explicit decision.
