"""Read-only pilot audit. Writes derived reports, never changes source runs."""
import argparse
import csv
import json
import statistics as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.formal.plan import Condition, Job
from experiments.formal.records import verify_job_results
from specdec.results_schema import mad, iqr
from experiments.formal.runtime_audit import validate_boundary_audit


def assess_pilot_unit(within_spreads, between_spread, preflight_ok, boundary_state="UNAVAILABLE"):
    """Assess existing pilot evidence; never infer measured-phase qualification."""
    within = "CHECK" if any(v > .2 for v in within_spreads) else "OK"
    between = "CHECK" if between_spread > .2 else "OK"
    blockers = ["pilot_only_not_formal_collection"]
    if boundary_state != "OK":
        blockers.append("runtime_boundary_evidence_missing" if boundary_state == "UNAVAILABLE" else "boundary_clock_check")
    if within == "CHECK":
        blockers.append("within_run_spread_gt_20pct")
    if between == "CHECK":
        blockers.append("between_block_difference_gt_20pct")
    if not preflight_ok:
        blockers.append("preflight_not_qualified")
    return dict(execution_status="SUCCESS", within_run_status=within,
                between_block_status=between,
                boundary_environment_status=boundary_state,
                environment_status="BOUNDARY_ONLY" if preflight_ok and boundary_state == "OK" else ("NOT_ESTABLISHED" if preflight_ok else "ENV_INVALID"),
                measurement_status="NOT_ESTABLISHED" if preflight_ok else "ENV_INVALID",
                timing_stability_status="CHECK" if "CHECK" in (within, between) else "OK",
                speedup_claim_allowed=False, speedup_claim_blockers=blockers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    batch = json.loads((args.batch / "summary.json").read_text(encoding="utf-8"))
    tasks, grouped = [], {}
    for entry in batch["completed"]:
        j = batch["jobs"][entry["index"]]
        job = Job(j["family_id"], Condition(**j["condition"]),
                  j["measurement_block"], j["measured_reps"])
        directory = ROOT / "output/formal_launcher_development" / entry["launch_id"] / "worker_results"
        # Legacy pilot remains readable, but never gains missing evidence.
        verify_job_results(directory, job, 151645, require_runtime_audit=False)
        records = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
        records.sort(key=lambda r: r["rep_index"])
        times = [r["decode_s"] for r in records]
        rates = [r["decode_tokens_per_s"] for r in records]
        env = records[0]["environment"]
        gpu = env["gpu"]["pstate_preflight"]
        audits = [r.get("runtime_environment_audit") for r in records]
        for audit in audits:
            if audit is not None:
                validate_boundary_audit(audit)
        boundary_state = (
            "UNAVAILABLE" if any(a is None or a["evidence_status"] != "COMPLETE" for a in audits)
            else ("OK" if all(a["boundary_clock_qualified"] is True for a in audits) else "CHECK")
        )
        row = dict(order=entry["index"] + 1, family=job.family_id,
                   boundary_environment_status=boundary_state,
                   ctx=job.condition.context_tokens, mode=job.condition.mode,
                   block=job.measurement_block, reps=len(records),
                   decode_median_s=st.median(times),
                   decode_min_s=min(times), decode_max_s=max(times),
                   decode_mad_s=mad(times), decode_iqr_s=iqr(times),
                   spread_fraction=(max(times)-min(times))/st.median(times),
                   protocol_throughput_spread_fraction=(max(rates)-min(rates))/st.median(rates),
                   decode_sequence_s=";".join(str(t) for t in times),
                   preflight_qualified=gpu["qualified"],
                   preflight_sm_mhz=gpu["sm_median"],
                   preflight_temp_c=gpu["temp_median"],
                   free_commit_gib=env["system_memory"]["commit_free_mib"]/1024,
                   launch_id=entry["launch_id"])
        tasks.append(row)
        grouped.setdefault((row["family"], row["ctx"], row["mode"]), []).append((row, records))
    pairs = []
    for key, entries in sorted(grouped.items()):
        by_block = {row["block"]: row for row, _ in entries}
        rate_medians = {row["block"]: st.median(r["decode_tokens_per_s"] for r in rr)
                        for row, rr in entries}
        hi, lo = max(rate_medians.values()), min(rate_medians.values())
        all_times = [r["decode_s"] for _, rr in entries for r in rr]
        assessment = assess_pilot_unit(
            [row["protocol_throughput_spread_fraction"] for row, _ in entries],
            (hi-lo)/hi, all(row["preflight_qualified"] is True for row, _ in entries),
            "UNAVAILABLE" if any(row["boundary_environment_status"] == "UNAVAILABLE" for row, _ in entries)
            else ("CHECK" if any(row["boundary_environment_status"] == "CHECK" for row, _ in entries) else "OK"))
        pairs.append(dict(family=key[0], ctx=key[1], mode=key[2],
                          decode_median_s=st.median(all_times),
                          decode_min_s=min(all_times), decode_max_s=max(all_times),
                          decode_mad_s=mad(all_times), decode_iqr_s=iqr(all_times),
                          **assessment,
                          block1_over_block0=by_block[1]["decode_median_s"]/by_block[0]["decode_median_s"],
                          protocol_between_block_fraction=(hi-lo)/hi,
                          unique_token_sequences=len({tuple(r["generated_ids"]) for _, rr in entries for r in rr})))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"assessment.json").write_text(json.dumps({
        "source_batch": args.batch.name, "analysis_only": True,
        "formal_admission_established": False,
        "status_schema": "pilot_offline_assessment_v1_not_canonical",
        "notes": "NOT_ESTABLISHED denotes missing evidence, not a measured environment failure. Per-block IQR is unavailable below four samples, following results_schema.iqr.",
        "units": pairs,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, rows in (("tasks.csv", tasks), ("blocks.csv", pairs)):
        with (args.out/name).open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    worst = sorted(tasks, key=lambda r: r["spread_fraction"], reverse=True)[:8]
    lines = ["# Pilot stability audit", "",
             f"Source batch: `{args.batch.name}`", "",
             f"Revalidated tasks: {len(tasks)} / planned {len(batch['jobs'])}.",
             f"Measured records: {sum(r['reps'] for r in tasks)}.",
             f"Within-task spread >20%: {sum(r['spread_fraction'] > .2 for r in tasks)} / {len(tasks)}.",
             f"Block median ratio outside [0.8, 1.2]: {sum(not .8 <= p['block1_over_block0'] <= 1.2 for p in pairs)} / {len(pairs)}.",
             f"Preflight qualified: {sum(r['preflight_qualified'] is True for r in tasks)} / {len(tasks)}.",
             f"Protocol throughput within-task spread >20%: {sum(r['protocol_throughput_spread_fraction'] > .2 for r in tasks)} / {len(tasks)}.",
             f"Protocol between-block relative difference >20%: {sum(p['protocol_between_block_fraction'] > .2 for p in pairs)} / {len(pairs)}.",
             "", "## Largest within-task spreads", "",
             "| Order | Family | Context | Mode | Block | Spread | Decode seconds in rep order |",
             "|---|---|---|---|---|---|---|"]
    for r in worst:
        lines.append(f"| {r['order']} | {r['family']} | {r['ctx']} | {r['mode']} | {r['block']} | {r['spread_fraction']:.1%} | {r['decode_sequence_s']} |")
    lines += ["", "## Layered assessment", "",
              "This is a new offline assessment schema, not a canonical formal result or a change to the frozen protocol.",
              f"Timing-stability CHECK units: {sum(p['timing_stability_status'] == 'CHECK' for p in pairs)} / {len(pairs)}.",
              f"Units with complete qualified boundary evidence: {sum(p['boundary_environment_status'] == 'OK' for p in pairs)} / {len(pairs)}. Boundary evidence is not continuous active-load verification.",
              "Missing boundary evidence remains NOT_ESTABLISHED. Qualified boundaries are BOUNDARY_ONLY; overall formal admission is not inferred.",
              "All speedup_claim_allowed flags are false: these are pilot data, not formal collection.",
              "See assessment.json for separate within-run/between-block states, descriptive statistics and explicit claim blockers."]
    lines += ["", "## Interpretation limits", "",
              "20% is a descriptive diagnostic line, not a new exclusion or admission rule.",
              "All observations are retained. No filtering or replacement runs are performed.",
              "Preflight clocks and temperatures precede model loading; they are not telemetry during measured decoding.",
              "These records cannot attribute variation to thermal throttling, clock changes, paging, or background work.",
              "Block medians use three and two repeats respectively; their ratios are descriptive, not significance tests.",
              "A token-stable run can still have unstable timing. No formal speedup conclusion is established."]
    (args.out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines[:11]))
    print("Reports:", args.out.resolve())


if __name__ == "__main__":
    main()
