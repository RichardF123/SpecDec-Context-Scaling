"""experiments/confirmatory/test_analyze_pilot_cpu.py — pilot 分析器的 CPU 合成批测试。

合成批明确标注为 SYNTHETIC（不是 GPU 测量），只用于验证：
1. 结构校验（计数/对账/hash/SPEC 不变量）；
2. 六项检查与 verdict 分支（READY / NEEDS_MEASUREMENT_FIX / ENVIRONMENT_BLOCKED）；
3. CSV/图输出齐全。
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

from experiments.confirmatory import analyze_pilot
from experiments.confirmatory.analyze_pilot import CONTEXTS, DEV_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/prompt_families_manifest.json"
MODES = ("AR", "SpecDec")
N_REPS = {0: 3, 1: 2}

_passed = 0
_failed = 0


def check(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 - 测试聚合器
        _failed += 1
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def make_record(meta, ctx, mode, idx, block_shift=0.0):
    base = 0.50 if mode == "AR" else 1.12
    decode = base * (1 + block_shift) + (idx - 1) * 0.004
    man = {entry["family_id"]: entry for entry in manifest()["families"]}
    fam = meta["family_id"]
    token_sha1 = man[fam]["prefixes"][str(ctx)]["token_sha1"]
    ids = [100 + i for i in range(64)]
    record = dict(
        schema_version="worker-record-v1", run_kind="measured", run_id="synthetic-run",
        configuration={"attention_strategy": "cudnn_prefill", "compute_dtype": "bfloat16",
                       "target_verify_mode": "block"},
        input={"token_sha1": token_sha1, "hash_verified": True, "manifest_token_sha1": token_sha1},
        input_tokens=ctx, prefill_len=ctx, decode_len=64, total_len=ctx + 64,
        generated_ids=ids, generated_length=64, stop_reason="max_new_reached",
        prefill_s=0.9, decode_s=decode, total_s=0.9 + decode,
        decode_tokens_per_s=64 / decode, total_tokens_per_s=(ctx + 64) / (0.9 + decode),
        peak_allocated_bytes=3 * 2**30, peak_reserved_bytes=4 * 2**30,
        same_mode_diagnostic_tokens_equal=True,
        runtime_environment_audit={"evidence_status": "COMPLETE",
                                   "boundary_clock_qualified": True},
        worker_started_utc="2026-10-24T00:00:00Z",
    )
    if mode == "SpecDec":
        record["spec_stats"] = dict(
            rounds=56, target_block_calls=56, mismatch_rounds=24, all_accept_rounds=32,
            proposed=224, draft_calls=224, accepted=40)
    return record


def build_batch(tmp: Path, *, incomplete_boundary=False, drop_task=False,
                block_shift_by_ctx=None):
    batch = tmp / "synthetic_batch"
    (batch / "tasks").mkdir(parents=True)
    tasks = []
    index = 0
    for ctx in CONTEXTS:
        for mode in MODES:
            for family in DEV_FAMILIES:
                pair_id = f"{family}_{ctx}"
                pos = 0 if family in ("F09", "F03") else 1
                for block in (0, 1):
                    shift = block_shift_by_ctx.get(ctx, 0.0) if (block == 1 and block_shift_by_ctx) else 0.0
                    task = dict(
                        family_id=family, context_tokens=ctx, mode=mode,
                        measurement_block=block, measured_reps=N_REPS[block],
                        pair_id=pair_id,
                        pair_order=("AB" if pos == 0 else "BA"), pair_position=pos,
                        planned_order=index)
                    index += 1
                    dirname = f"{task['planned_order']:02d}_{family}_{ctx}_{mode}_b{block}"
                    task_dir = batch / "tasks" / dirname
                    task_dir.mkdir()
                    (task_dir / "task_meta.json").write_text(
                        json.dumps(task, ensure_ascii=False), encoding="utf-8")
                    results = task_dir / "worker_results"
                    results.mkdir()
                    for rep in range(N_REPS[block]):
                        record = make_record(task, ctx, mode, rep, block_shift=shift)
                        if incomplete_boundary and family == "F09" and ctx == 2048:
                            record["runtime_environment_audit"]["evidence_status"] = "INCOMPLETE"
                        (results / f"measured_{rep:02d}.json").write_text(
                            json.dumps(record, ensure_ascii=False), encoding="utf-8")
                    if drop_task and family == "F04" and ctx == 8192 and block == 0:
                        shutil.rmtree(task_dir)
                        tasks.append(dict(task, dropped=True))
                        continue
                    tasks.append(task)
    summary = dict(status="development_batch_verified",
                   completed=[dict(task_dir=str(batch / "tasks" / f"{t['planned_order']:02d}_{t['family_id']}_{t['context_tokens']}_{t['mode']}_b{t['measurement_block']}"),
                                   actual_start_utc="2026-10-24T00:00:00Z",
                                   actual_end_utc="2026-10-24T00:00:00Z")
                              for t in tasks if not t.get("dropped")])
    (batch / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    (batch / "plan.json").write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    return batch


def _call_main(batch, out):
    import sys
    argv = sys.argv
    sys.argv = ["analyze_pilot", str(batch), "--out", str(out),
                "--input-manifest", str(MANIFEST)]
    try:
        rc = analyze_pilot.main()
    finally:
        sys.argv = argv
    return rc


def t_ready():
    with tempfile.TemporaryDirectory() as tmp:
        batch = build_batch(Path(tmp))
        rc = _call_main(batch, batch / "analysis")
        assert rc == 0
        summary = json.loads((batch / "analysis" / "summary.json").read_text(encoding="utf-8"))
        assert summary["verdict"] == "READY_FOR_CONFIRMATORY_FULL", summary["verdict"]
        assert summary["n_tasks"] == 48 and summary["n_measured"] == 120
        assert all(row["all_six_checks_ok"] == 4 for row in summary["contexts"])
        for name in ("pilot_tasks.csv", "pilot_family_timing.csv", "pilot_block_shifts.csv",
                     "pilot_order_ab_ba.csv", "pilot_pair_summary.csv", "pilot_context_checks.csv"):
            path = batch / "analysis" / name
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) > 0, name
        for ext in ("svg", "png", "pdf"):
            assert (batch / "analysis" / f"pilot_stability.{ext}").exists()
        pairs = list(csv.DictReader((batch / "analysis" / "pilot_pair_summary.csv").open(encoding="utf-8")))
        assert all(float(p["decode_ratio"]) < 1 for p in pairs)  # AR 比 SpecDec 快
        assert all(p["same_output"] == "True" for p in pairs)
        assert all(p["claim_status"] == "DESCRIPTIVE_PILOT" for p in pairs)


def t_missing_task_environment_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        batch = build_batch(Path(tmp), drop_task=True)
        rc = _call_main(batch, batch / "analysis")
        assert rc == 0
        summary = json.loads((batch / "analysis" / "summary.json").read_text(encoding="utf-8"))
        assert summary["n_tasks"] < 48
        assert summary["n_measured"] < 120
        assert summary["verdict"] == "ENVIRONMENT_BLOCKED"


def t_boundary_incomplete_needs_fix():
    with tempfile.TemporaryDirectory() as tmp:
        batch = build_batch(Path(tmp), incomplete_boundary=True)
        rc = _call_main(batch, batch / "analysis")
        assert rc == 0
        summary = json.loads((batch / "analysis" / "summary.json").read_text(encoding="utf-8"))
        assert summary["boundary_all_complete"] is False
        assert summary["verdict"] == "NEEDS_MEASUREMENT_FIX"


def t_big_block_shift_triggers_fix():
    with tempfile.TemporaryDirectory() as tmp:
        batch = build_batch(Path(tmp), block_shift_by_ctx={4096: 0.30})
        rc = _call_main(batch, batch / "analysis")
        assert rc == 0
        summary = json.loads((batch / "analysis" / "summary.json").read_text(encoding="utf-8"))
        ctx4 = next(row for row in summary["contexts"] if row["ctx"] == 4096)
        assert ctx4["all_six_checks_ok"] == 0
        assert summary["verdict"] == "NEEDS_MEASUREMENT_FIX"


def t_hash_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        batch = build_batch(Path(tmp))
        path = sorted((batch / "tasks").glob("00_*"))[0] / "worker_results" / "measured_00.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["input"]["token_sha1"] = "0" * 40
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        try:
            _call_main(batch, batch / "analysis")
        except AssertionError as exc:
            assert "hash" in str(exc)
            return
        raise AssertionError("hash 不一致未被拒绝")


if __name__ == "__main__":
    check("T1 合成批 READY 且 CSV/图齐全", t_ready)
    check("T2 缺任务 => ENVIRONMENT_BLOCKED", t_missing_task_environment_blocked)
    check("T3 boundary 证据不完整 => NEEDS_MEASUREMENT_FIX", t_boundary_incomplete_needs_fix)
    check("T4 4K 大块漂移 => NEEDS_MEASUREMENT_FIX", t_big_block_shift_triggers_fix)
    check("T5 输入 hash 不一致 => 拒绝", t_hash_mismatch_raises)
    print(f"\n{_passed} passed, {_failed} failed")
    raise SystemExit(0 if _failed == 0 else 1)
