"""CPU-only analysis for the frozen 16-family confirmatory experiment.

Usage:
    python -m experiments.confirmatory.analyze_full <batch_dir>

The statistical unit is one family at one context length.  Each mode pools the
three block-0 and two block-1 measured repetitions before taking its median.
Individual repetitions are never treated as independent families.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from experiments.formal.input_sets import FORMAL_FAMILIES, select_input_set

from .analyze_pilot import (
    CHECK_SPREAD,
    CONTEXTS,
    MODES,
    load_batch,
    plot_stability,
    validate_task,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TASKS = 192
EXPECTED_MEASURED = 480
EXPECTED_REPS = 5
BOOTSTRAP_SEED = 4207
BOOTSTRAP_SAMPLES = 10_000


def bootstrap_median_ci(values, *, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES):
    """Family-level percentile bootstrap CI for a context-level median."""
    x = np.asarray([float(v) for v in values], dtype=np.float64)
    if x.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(samples, x.size), replace=True)
    medians = np.median(draws, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def _unit_rows(task_rows, data):
    grouped = {}
    for row in task_rows:
        grouped.setdefault((row["family"], row["ctx"], row["mode"]), []).append(row)

    records_by_key = {}
    for item in data.values():
        meta = item["meta"]
        key = (meta["family_id"], meta["context_tokens"], meta["mode"])
        records_by_key.setdefault(key, []).extend(item["records"])

    units = []
    for key in sorted(grouped):
        family, ctx, mode = key
        rows = grouped[key]
        blocks = {r["block"] for r in rows}
        records = records_by_key[key]
        if blocks != {0, 1} or len(rows) != 2 or len(records) != EXPECTED_REPS:
            raise AssertionError(
                f"{family}/{ctx}/{mode}: expected blocks 0+1 and five records; "
                f"got blocks={blocks}, tasks={len(rows)}, records={len(records)}"
            )
        decode = [float(r["decode_s"]) for r in records]
        prefill = [float(r["prefill_s"]) for r in records]
        total = [float(r["total_s"]) for r in records]
        generated = [tuple(r["generated_ids"]) for r in records]
        medians = [float(r["throughput"]) for r in rows]
        between = (max(medians) - min(medians)) / max(medians)
        spec_records = [r.get("spec_stats") for r in records] if mode == "SpecDec" else []
        accepted = sum(s["accepted"] for s in spec_records) if spec_records else None
        proposed = sum(s["proposed"] for s in spec_records) if spec_records else None
        units.append({
            "family": family,
            "ctx": ctx,
            "mode": mode,
            "n": len(records),
            "decode_s": st.median(decode),
            "decode_min": min(decode),
            "decode_max": max(decode),
            "prefill_s": st.median(prefill),
            "total_s": st.median(total),
            "within_status": "CHECK" if any(r["within_status"] == "CHECK" for r in rows) else "OK",
            "between_status": "CHECK" if between > CHECK_SPREAD else "OK",
            "between_difference_pct": between * 100,
            "boundary_status": "OK" if all(r["boundary_status"] == "OK" for r in rows) else "CHECK",
            "boundary_evidence_complete": all(r["boundary_evidence_complete"] for r in rows),
            "mode_outputs_consistent": len(set(generated)) == 1,
            "generated_ids": list(generated[0]),
            "spec_acceptance": (accepted / proposed if proposed else None),
            "spec_rounds_median": (st.median(s["rounds"] for s in spec_records) if spec_records else None),
            "spec_mismatch_median": (st.median(s["mismatch_rounds"] for s in spec_records)
                                     if spec_records else None),
            "spec_target_verify_calls_median": (
                st.median(s["target_block_calls"] for s in spec_records) if spec_records else None
            ),
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
            "block0_decode_s": next(r["decode_s"] for r in rows if r["block"] == 0),
            "block1_decode_s": next(r["decode_s"] for r in rows if r["block"] == 1),
        })
    return units


def analyze(batch: Path, out: Path, manifest_path: Path) -> dict:
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {entry["family_id"]: entry for entry in manifest_doc["families"]}
    plan, batch_summary, planned, data = load_batch(batch)

    if batch_summary.get("status") not in {"formal_batch_verified", "confirmatory_full_verified"}:
        raise AssertionError(f"batch is not formally complete: {batch_summary.get('status')}")
    if set(data) != set(planned):
        raise AssertionError(f"completed task set differs from plan: {len(data)} != {len(planned)}")

    task_rows = [validate_task(item["meta"], item["records"], manifest)
                 for item in data.values()]
    task_rows.sort(key=lambda r: (r["ctx"], r["family"], r["mode"], r["block"]))
    n_tasks = len(task_rows)
    n_measured = sum(r["reps"] for r in task_rows)
    families_seen = {r["family"] for r in task_rows}
    if n_tasks != EXPECTED_TASKS or n_measured != EXPECTED_MEASURED:
        raise AssertionError(
            f"formal batch incomplete: tasks={n_tasks}/{EXPECTED_TASKS}, "
            f"measured={n_measured}/{EXPECTED_MEASURED}"
        )
    if families_seen != set(FORMAL_FAMILIES):
        raise AssertionError(f"formal family set mismatch: {sorted(families_seen)}")

    units = _unit_rows(task_rows, data)
    unit_map = {(r["family"], r["ctx"], r["mode"]): r for r in units}
    if len(units) != 96:
        raise AssertionError(f"expected 96 family/context/mode units, got {len(units)}")

    pairs = []
    for family in FORMAL_FAMILIES:
        for ctx in CONTEXTS:
            ar = unit_map[(family, ctx, "AR")]
            spec = unit_map[(family, ctx, "SpecDec")]
            a_ids, s_ids = ar["generated_ids"], spec["generated_ids"]
            common = next((i for i, (a, s) in enumerate(zip(a_ids, s_ids)) if a != s),
                          min(len(a_ids), len(s_ids)))
            six_ok = all(ar[k] == "OK" for k in
                         ("within_status", "between_status", "boundary_status")) and all(
                             spec[k] == "OK" for k in
                             ("within_status", "between_status", "boundary_status"))
            ratio = ar["decode_s"] / spec["decode_s"]
            restricted = ctx == 8192 and not six_ok
            pairs.append({
                "family": family,
                "ctx": ctx,
                "ar_decode_s": ar["decode_s"],
                "spec_decode_s": spec["decode_s"],
                "decode_ratio": ratio,
                "ratio_reportable": not restricted,
                "same_output": a_ids == s_ids,
                "common_prefix": common,
                "ar_repeat_output_consistent": ar["mode_outputs_consistent"],
                "spec_repeat_output_consistent": spec["mode_outputs_consistent"],
                "timing_six_checks_ok": six_ok,
                "claim_status": "8K_RESTRICTED" if restricted else "DESCRIPTIVE_FORMAL",
                "spec_acceptance": spec["spec_acceptance"],
                "spec_rounds_median": spec["spec_rounds_median"],
                "spec_mismatch_median": spec["spec_mismatch_median"],
                "spec_target_verify_calls_median": spec["spec_target_verify_calls_median"],
                "ar_peak_allocated_gib": ar["peak_allocated_gib"],
                "spec_peak_allocated_gib": spec["peak_allocated_gib"],
            })

    contexts = []
    for ctx_i, ctx in enumerate(CONTEXTS):
        subset = [r for r in pairs if r["ctx"] == ctx]
        reportable = [r["decode_ratio"] for r in subset if r["ratio_reportable"]]
        all_ratios = [r["decode_ratio"] for r in subset]
        low, high = bootstrap_median_ci(reportable, seed=BOOTSTRAP_SEED + ctx_i)
        contexts.append({
            "ctx": ctx,
            "families": len(subset),
            "all_six_checks_ok": sum(r["timing_six_checks_ok"] for r in subset),
            "reportable_ratio_families": len(reportable),
            "restricted_ratio_families": len(subset) - len(reportable),
            "same_output": sum(r["same_output"] for r in subset),
            "repeat_outputs_consistent": sum(
                r["ar_repeat_output_consistent"] and r["spec_repeat_output_consistent"]
                for r in subset
            ),
            "spec_slower_families_all": sum(r < 1 for r in all_ratios),
            "ratio_median_reportable": st.median(reportable) if reportable else None,
            "ratio_bootstrap95_low": low,
            "ratio_bootstrap95_high": high,
            "ar_decode_median_s": st.median(r["ar_decode_s"] for r in subset),
            "spec_decode_median_s": st.median(r["spec_decode_s"] for r in subset),
        })

    shifts = []
    for row in units:
        b0, b1 = row["block0_decode_s"], row["block1_decode_s"]
        shifts.append({
            "family": row["family"], "ctx": row["ctx"], "mode": row["mode"],
            "block0_decode_s": b0, "block1_decode_s": b1,
            "block1_vs_block0_pct": 100 * (b1 / b0 - 1),
        })

    order_rows = []
    for ctx in CONTEXTS:
        for mode in MODES:
            entries = [r for r in task_rows if r["ctx"] == ctx and r["mode"] == mode]
            first = [r["decode_s"] for r in entries if r["pair_position"] == 0]
            second = [r["decode_s"] for r in entries if r["pair_position"] == 1]
            if len(first) != 16 or len(second) != 16:
                raise AssertionError(f"{ctx}/{mode}: expected 16 first and 16 second tasks")
            mf, ms = st.median(first), st.median(second)
            order_rows.append({
                "ctx": ctx, "mode": mode, "n_first": len(first), "n_second": len(second),
                "median_first_s": mf, "median_second_s": ms,
                "first_vs_second_pct": 100 * (mf / ms - 1),
            })

    boundary_complete = all(
        record["runtime_environment_audit"]["evidence_status"] == "COMPLETE"
        for item in data.values() for record in item["records"]
    )
    repeat_consistent = all(
        r["ar_repeat_output_consistent"] and r["spec_repeat_output_consistent"] for r in pairs
    )
    restricted_8k = sum(r["ratio_reportable"] is False for r in pairs if r["ctx"] == 8192)
    verdict = "FORMAL_COMPLETE" if restricted_8k == 0 else "FORMAL_COMPLETE_WITH_RESTRICTED_8K"
    payload = {
        "schema_version": "confirmatory-full-analysis-v1",
        "batch": batch.name,
        "n_tasks": n_tasks,
        "n_measured": n_measured,
        "analysis_unit": "family-within-context; median of five measured repetitions",
        "bootstrap": {"unit": "family", "seed": BOOTSTRAP_SEED,
                      "samples": BOOTSTRAP_SAMPLES, "interval": "percentile 95%"},
        "contexts": contexts,
        "pairs": pairs,
        "boundary_all_complete": boundary_complete,
        "repeat_outputs_consistent": repeat_consistent,
        "restricted_8k_families": restricted_8k,
        "verdict": verdict,
        "claim_boundary": (
            "Ratios compare each strategy's observed trajectory. AR/Spec output divergence "
            "does not support a lossless identical-output speedup claim. Individual measured "
            "repetitions are not treated as independent samples."
        ),
    }

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "formal_tasks.csv",
              [{k: v for k, v in row.items() if k not in {"generated_ids", "spec_stats"}}
               for row in task_rows])
    write_csv(out / "formal_family_timing.csv",
              [{k: v for k, v in row.items() if k != "generated_ids"} for row in units])
    write_csv(out / "formal_block_shifts.csv", shifts)
    write_csv(out / "formal_order_ab_ba.csv", order_rows)
    write_csv(out / "formal_pair_summary.csv", pairs)
    write_csv(out / "formal_context_summary.csv", contexts)
    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_stability(pairs, shifts, out / "formal_stability")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    args = parser.parse_args()
    default_manifest, _ = select_input_set("formal")
    out = args.out or args.batch / "analysis"
    payload = analyze(args.batch, out, args.input_manifest or default_manifest)
    print(json.dumps({
        "out": str(out.resolve()),
        "n_tasks": payload["n_tasks"],
        "n_measured": payload["n_measured"],
        "contexts": payload["contexts"],
        "verdict": payload["verdict"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

