"""CPU synthetic-batch tests for the 16-family formal analyzer."""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from experiments.formal.input_sets import FORMAL_FAMILIES, select_input_set

from . import analyze_full
from .plan import CONTEXTS, build_plan

N_REPS = {0: 3, 1: 2}
_passed = 0
_failed = 0


def check(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001
        _failed += 1
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")


def _record(task, token_sha1, rep, *, large_8k_shift=False):
    base = 0.50 if task.mode == "AR" else 1.00
    shift = 0.30 if (large_8k_shift and task.context_tokens == 8192
                     and task.measurement_block == 1 and task.family_id == "FF01") else 0.0
    decode = base * (1 + shift) + rep * 0.004
    record = {
        "schema_version": "worker-record-v1", "run_kind": "measured",
        "run_id": "synthetic-formal", "configuration": {},
        "input": {"token_sha1": token_sha1, "hash_verified": True,
                  "manifest_token_sha1": token_sha1},
        "input_tokens": task.context_tokens, "prefill_len": task.context_tokens,
        "decode_len": 64, "total_len": task.context_tokens + 64,
        "generated_ids": list(range(64)), "generated_length": 64,
        "stop_reason": "max_new_reached", "prefill_s": 0.9,
        "decode_s": decode, "total_s": 0.9 + decode,
        "decode_tokens_per_s": 64 / decode,
        "total_tokens_per_s": (task.context_tokens + 64) / (0.9 + decode),
        "peak_allocated_bytes": 3 * 2**30, "peak_reserved_bytes": 4 * 2**30,
        "same_mode_diagnostic_tokens_equal": True,
        "runtime_environment_audit": {"evidence_status": "COMPLETE",
                                      "boundary_clock_qualified": True},
    }
    if task.mode == "SpecDec":
        record["spec_stats"] = {
            "rounds": 56, "target_block_calls": 56, "mismatch_rounds": 24,
            "all_accept_rounds": 32, "proposed": 224, "draft_calls": 224,
            "accepted": 40,
        }
    return record


def build_batch(root: Path, *, status="formal_batch_verified", large_8k_shift=False):
    batch = root / "synthetic_formal"
    (batch / "tasks").mkdir(parents=True)
    manifest_path, _ = select_input_set("formal")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    families = {f["family_id"]: f for f in manifest["families"]}
    tasks = build_plan(FORMAL_FAMILIES, contexts=CONTEXTS, seed=4207)
    completed = []
    for task in tasks:
        dirname = (f"{task.planned_order:03d}_{task.family_id}_{task.context_tokens}_"
                   f"{task.mode}_b{task.measurement_block}")
        task_dir = batch / "tasks" / dirname
        results = task_dir / "worker_results"
        results.mkdir(parents=True)
        meta = asdict(task)
        (task_dir / "task_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        token_sha1 = families[task.family_id]["prefixes"][str(task.context_tokens)]["token_sha1"]
        for rep in range(N_REPS[task.measurement_block]):
            record = _record(task, token_sha1, rep, large_8k_shift=large_8k_shift)
            (results / f"measured_{rep:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8")
        completed.append({"planned_order": task.planned_order, "task_dir": str(task_dir),
                          "actual_start_utc": "2026-10-25T00:00:00Z",
                          "actual_end_utc": "2026-10-25T00:00:01Z"})
    (batch / "plan.json").write_text(
        json.dumps({"tasks": [asdict(t) for t in tasks]}, ensure_ascii=False),
        encoding="utf-8")
    (batch / "summary.json").write_text(
        json.dumps({"status": status, "completed": completed}, ensure_ascii=False),
        encoding="utf-8")
    return batch, manifest_path


def t_complete_outputs_and_pooled_median():
    with tempfile.TemporaryDirectory() as tmp:
        batch, manifest = build_batch(Path(tmp))
        out = batch / "analysis"
        result = analyze_full.analyze(batch, out, manifest)
        assert result["verdict"] == "FORMAL_COMPLETE"
        assert result["n_tasks"] == 192 and result["n_measured"] == 480
        assert len(result["pairs"]) == 48 and len(result["contexts"]) == 3
        unit = next(r for r in result["pairs"] if r["family"] == "FF01" and r["ctx"] == 2048)
        assert abs(unit["ar_decode_s"] - 0.504) < 1e-12, unit
        for name in ("formal_tasks.csv", "formal_family_timing.csv",
                     "formal_block_shifts.csv", "formal_order_ab_ba.csv",
                     "formal_pair_summary.csv", "formal_context_summary.csv", "summary.json"):
            assert (out / name).exists(), name
        for ext in ("svg", "png", "pdf"):
            assert (out / f"formal_stability.{ext}").exists(), ext


def t_unstable_8k_is_restricted_not_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        batch, manifest = build_batch(Path(tmp), large_8k_shift=True)
        result = analyze_full.analyze(batch, batch / "analysis", manifest)
        row = next(r for r in result["pairs"] if r["family"] == "FF01" and r["ctx"] == 8192)
        assert row["ratio_reportable"] is False
        assert row["claim_status"] == "8K_RESTRICTED"
        assert result["restricted_8k_families"] == 1
        assert result["verdict"] == "FORMAL_COMPLETE_WITH_RESTRICTED_8K"


def t_nonfinal_batch_refused():
    with tempfile.TemporaryDirectory() as tmp:
        batch, manifest = build_batch(Path(tmp), status="aborted")
        try:
            analyze_full.analyze(batch, batch / "analysis", manifest)
        except AssertionError as exc:
            assert "not formally complete" in str(exc)
            return
        raise AssertionError("aborted batch was accepted")


def t_bootstrap_is_reproducible():
    a = analyze_full.bootstrap_median_ci([0.4, 0.5, 0.6, 0.7])
    b = analyze_full.bootstrap_median_ci([0.4, 0.5, 0.6, 0.7])
    assert a == b and a[0] <= 0.55 <= a[1]


if __name__ == "__main__":
    check("T1 full outputs + five-rep pooled median", t_complete_outputs_and_pooled_median)
    check("T2 unstable 8K ratio restricted", t_unstable_8k_is_restricted_not_deleted)
    check("T3 non-final batch refused", t_nonfinal_batch_refused)
    check("T4 family bootstrap reproducible", t_bootstrap_is_reproducible)
    print(f"\n{_passed} passed, {_failed} failed")
    raise SystemExit(0 if _failed == 0 else 1)

