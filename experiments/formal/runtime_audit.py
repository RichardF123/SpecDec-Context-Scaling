"""Outside-timing boundary telemetry; not continuous active-load monitoring."""
import math
import time

from specdec.engine import g5_driver


def capture_boundary():
    start = time.perf_counter()
    snapshot = g5_driver.system_snapshot()
    return {
        "snapshot": snapshot,
        "query_start_monotonic_s": start,
        "query_end_monotonic_s": time.perf_counter(),
        "sampling_scope": "outside_timing_boundary",
        "active_load_verified": False,
    }


def assess_boundaries(before, after):
    samples = [point["snapshot"] for point in (before, after)]
    fields = ("clock_sm_MHz", "clock_mem_MHz")
    complete = all(
        sample.get("available") is True
        and all(type(sample.get(k)) in (int, float)
                and math.isfinite(sample[k]) for k in fields)
        for sample in samples
    )
    verdict = g5_driver.pstate_verdict(samples if complete else [])
    qualified = verdict["qualified"] if complete else None
    return {
        "schema_version": "boundary_audit_v1",
        "monitor_start": before,
        "monitor_end": after,
        "boundary_clock_verdict": verdict,
        "boundary_clock_qualified": qualified,
        "evidence_status": "COMPLETE" if complete else "UNAVAILABLE",
        "sampling_scope": "outside_timing_boundary",
        "continuous_active_load_verified": False,
        "interpretation": "Boundary clocks only. Low boundary clocks do not prove low clocks during measured work; high clocks do not certify the whole interval.",
    }


def validate_boundary_audit(audit):
    if not isinstance(audit, dict) or audit.get("schema_version") != "boundary_audit_v1":
        raise ValueError("Missing or unsupported runtime boundary audit")
    points = []
    for name in ("monitor_start", "monitor_end"):
        point = audit.get(name)
        if not isinstance(point, dict) or not isinstance(point.get("snapshot"), dict):
            raise ValueError("Missing boundary snapshot")
        if point.get("sampling_scope") != "outside_timing_boundary" or point.get("active_load_verified") is not False:
            raise ValueError("Boundary scope or active-load claim is invalid")
        stamps = [point.get(k) for k in ("query_start_monotonic_s", "query_end_monotonic_s")]
        if not all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in stamps):
            raise ValueError("Invalid boundary timestamps")
        if stamps[0] > stamps[1]:
            raise ValueError("Boundary query timestamps are reversed")
        points.append(point)
    if points[0]["query_end_monotonic_s"] > points[1]["query_start_monotonic_s"]:
        raise ValueError("Start and end boundary queries overlap or are reversed")
    expected = assess_boundaries(*points)
    for key in ("sampling_scope", "continuous_active_load_verified", "boundary_clock_qualified", "evidence_status"):
        if type(audit.get(key)) is not type(expected[key]) or audit.get(key) != expected[key]:
            raise ValueError(f"Inconsistent boundary assessment: {key}")
    if audit.get("boundary_clock_verdict") != expected["boundary_clock_verdict"]:
        raise ValueError("Boundary verdict does not match snapshots")
