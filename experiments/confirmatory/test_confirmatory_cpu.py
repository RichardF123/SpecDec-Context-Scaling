"""experiments/confirmatory/test_confirmatory_cpu.py — confirmatory pilot 的 CPU/dry-run 自测。

覆盖：
| # | 检查 |
| --- | --- |
| T1 | 同 seed 生成完全相同的计划 |
| T2 | 不同 seed 不改变任务集合、但改变执行顺序 |
| T3 | 无重复/缺失任务（重复或遗漏计划会被拒绝）|
| T4 | 每个 (family,ctx,block) 恰含 AR 与 SpecDec 各一 |
| T5 | AB/BA 数量差 ≤ 1；同配对 AB/BA 一致 |
| T6 | 两个 block 在执行序列中确实交错（不退化前后两段）|
| T7 | measured 次数正确（block0=3、block1=2，共 120）|
| T8 | 输入 hash 与原 pilot manifest 一致（4 个开发 family × 3 ctx）|
| T9 | dry-run 不加载模型、不触碰 GPU、不创建输出目录 |
| T10 | 计划 JSON 读回并通过校验 |
| T11 | worker dry-run 与计划逐字段一致；篡改 meta 会被拒绝 |
| T12 | 默认生产路径不变（策略/rope 未被改动）|
| T13 | 正式 FF01–FF16 计划为 192 tasks / 480 measured，按 context 8/8 平衡且 pair 相邻 |
| T14 | FF01–FF16 的 2K/4K/8K 输入 hash 全部复现 |
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.formal import worker as formal_worker
from experiments.formal.input_sets import select_input_set
from specdec.engine import family_input, model_self

from . import plan as plan_mod
from . import run as run_mod
from . import worker as worker_mod

CHECKS: list = []
DEV_FAMILIES = ("F09", "F03", "F11", "F04")
CTXS = (2048, 4096, 8192)


def ck(name: str, ok: bool, extra=None) -> None:
    CHECKS.append((name, bool(ok), extra))


def expect_raises(name: str, fn, exc_type=Exception) -> None:
    try:
        fn()
    except exc_type as exc:                                        # noqa: BLE001
        ck(name, True, f"拦截：{type(exc).__name__}")
    except Exception as exc:                                       # noqa: BLE001
        ck(name, False, f"错误类型 {type(exc).__name__}: {exc}")
    else:
        ck(name, False, "没有抛错")


def main() -> int:
    # ---------------- T1/T2/T3/T4/T5/T6/T7：计划 ----------------
    tasks = plan_mod.build_plan(DEV_FAMILIES, contexts=CTXS, seed=4207)
    tasks2 = plan_mod.build_plan(DEV_FAMILIES, contexts=CTXS, seed=4207)
    ck("T1 同 seed 计划完全相同", tasks == tasks2, None)
    ck("T1 同 seed 序列化一致",
       [t.planned_order for t in tasks] == [t.planned_order for t in tasks2], None)

    tasks_other = plan_mod.build_plan(DEV_FAMILIES, contexts=CTXS, seed=99)
    keys = lambda ts: {(t.family_id, t.context_tokens, t.mode, t.measurement_block) for t in ts}
    orders = lambda ts: [t.planned_order for t in ts]
    families_of_order = lambda ts: [(t.family_id, t.context_tokens, t.mode, t.measurement_block)
                                    for t in sorted(ts, key=lambda t: t.planned_order)]
    ck("T2 不同 seed 任务集合相同", keys(tasks) == keys(tasks_other), None)
    ck("T2 不同 seed 执行顺序不同", families_of_order(tasks) != families_of_order(tasks_other), None)

    summary = plan_mod.validate_plan(tasks, DEV_FAMILIES, contexts=CTXS)
    ck("T3 合法计划通过校验（48 任务 / 120 measured）",
       summary["n_tasks"] == 48 and summary["n_measured"] == 120, summary)
    expect_raises("T3 重复任务被拒",
                  lambda: plan_mod.validate_plan(tasks + [tasks[0]], DEV_FAMILIES,
                                                 contexts=CTXS), ValueError)
    expect_raises("T3 遗漏任务被拒",
                  lambda: plan_mod.validate_plan(tasks[1:], DEV_FAMILIES,
                                                 contexts=CTXS), ValueError)

    pair_modes = {}
    for task in tasks:
        pair_modes.setdefault((task.family_id, task.context_tokens,
                               task.measurement_block), set()).add(task.mode)
    ck("T4 每个 (family,ctx,block) 含 AR 与 SpecDec 各一",
       all(v == {"AR", "SpecDec"} for v in pair_modes.values())
       and len(pair_modes) == 24, len(pair_modes))

    ck("T5 AB/BA 数量差 ≤ 1 且 12 配对", summary["n_ab"] + summary["n_ba"] == 12
       and abs(summary["n_ab"] - summary["n_ba"]) <= 1,
       (summary["n_ab"], summary["n_ba"]))
    ck("T5 AB/BA 按 context 分层平衡",
       all(v == {"AB": 2, "BA": 2}
           for v in summary["pair_order_by_context"].values()),
       summary["pair_order_by_context"])

    blocks = summary["block_interleave"]["blocks"]
    ck("T6 block 交错（不是前后两段）",
       summary["block_interleave"]["max_same_block_run"] <= 2
       and summary["block_interleave"]["transitions"] >= 12
       and set(blocks[:8]) == {0, 1} and set(blocks[-8:]) == {0, 1},
       summary["block_interleave"])
    ck("T6 AR/Spec 以完整相邻 pair 为调度单位",
       summary["adjacent_pair_units"] == 24, summary["adjacent_pair_units"])

    reps_ok = all(t.measured_reps == (3 if t.measurement_block == 0 else 2) for t in tasks)
    ck("T7 measured reps 正确（0→3，1→2，总 120）",
       reps_ok and sum(t.measured_reps for t in tasks) == 120, None)

    # ---------------- T8：输入 hash 与 pilot manifest 一致 ----------------
    manifest_path, families = select_input_set("pilot")
    pilot = family_input.load_manifest(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(
        family_input.TOKENIZER_ID, revision=family_input.TOKENIZER_REVISION,
        local_files_only=True)
    hash_ok = True
    hash_detail = {}
    for family in DEV_FAMILIES:
        for ctx in CTXS:
            try:
                ids, meta = family_input.load_family_ids(
                    tokenizer, family, manifest=pilot, ctx=ctx, verify_hash=True)
                expected = meta["manifest_token_sha1"]
                actual = meta["token_sha1"]
                if expected is not None and actual != expected:
                    hash_ok = False
                    hash_detail[f"{family}_{ctx}"] = (actual, expected)
            except Exception as exc:                                # noqa: BLE001
                hash_ok = False
                hash_detail[f"{family}_{ctx}"] = repr(exc)
    ck("T8 4 family × 3 ctx 输入 hash 与 manifest 一致", hash_ok, hash_detail)

    # ---------------- T9：dry-run 不触碰 GPU/不建目录 ----------------
    before = sorted(p.name for p in run_mod.OUTPUT_ROOT.glob("*")) if run_mod.OUTPUT_ROOT.exists() else []
    saved_argv = sys.argv
    sys.argv = ["run"]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_mod.main()
    finally:
        sys.argv = saved_argv
    after = sorted(p.name for p in run_mod.OUTPUT_ROOT.glob("*")) if run_mod.OUTPUT_ROOT.exists() else []
    import re

    text = buf.getvalue()
    ck("T9 dry-run 返回 0", rc == 0, rc)
    lines = [line for line in text.splitlines() if re.match(r"^\d+\.\s", line.strip())]
    ck("T9 dry-run 打印 48 个任务", len(lines) == 48, len(lines))
    ck("T9 dry-run 不创建输出目录", before == after, (before, after))
    ck("T9 dry-run 不加载模型（无 CUDA 分配）",
       torch.cuda.memory_allocated() == 0 if torch.cuda.is_available() else True, None)

    # ---------------- T10：计划 JSON 读回 ----------------
    with tempfile.TemporaryDirectory(prefix="confirmatory_cpu_") as tmp:
        path = Path(tmp) / "plan.json"
        plan_mod.save_plan(tasks, DEV_FAMILIES, path, contexts=CTXS, seed=4207,
                           batch_id="test_batch",
                           input_manifest=manifest_path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        ck("T10 计划 JSON 读回并通过校验",
           saved["n_tasks"] == 48 and saved["n_measured"] == 120
           and len(saved["tasks"]) == 48
           and [t["planned_order"] for t in saved["tasks"]] == list(range(48)), None)
        expect_raises("T10 计划文件重复写入被拒",
                      lambda: plan_mod.save_plan(tasks, DEV_FAMILIES, path, contexts=CTXS),
                      FileExistsError)

    # ---------------- T11：worker dry-run 与计划一致；篡改被拒 ----------------
    with tempfile.TemporaryDirectory(prefix="confirmatory_cpu_") as tmp:
        target_task = sorted(tasks, key=lambda t: t.planned_order)[0]
        meta = dict(
            meta_schema="confirmatory-task-meta-v1",
            confirmatory_batch_id="test_batch",
            plan_seed=4207, schedule_seed=4207,
            families=list(DEV_FAMILIES), contexts=list(CTXS),
            family_id=target_task.family_id, context_tokens=target_task.context_tokens,
            mode=target_task.mode, measurement_block=target_task.measurement_block,
            measured_reps=target_task.measured_reps, pair_id=target_task.pair_id,
            pair_order=target_task.pair_order, pair_position=target_task.pair_position,
            planned_order=target_task.planned_order,
            actual_start_utc="2026-10-24T00:00:00Z",
            input_set="pilot", expected_commit=None,
        )
        meta_path = Path(tmp) / "meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        saved_argv = sys.argv
        sys.argv = ["worker", "--family", target_task.family_id,
                    "--ctx", str(target_task.context_tokens), "--mode", target_task.mode,
                    "--block", str(target_task.measurement_block),
                    "--input-set", "pilot", "--meta", str(meta_path),
                    "--output-dir", str(Path(tmp) / "out"), "--dry-run"]
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = worker_mod.main()
        finally:
            sys.argv = saved_argv
        ck("T11 worker dry-run 返回 0 且与计划一致",
           rc == 0 and str(target_task.family_id) in buf.getvalue(), buf.getvalue()[:120])

        bad = dict(meta)
        bad["planned_order"] = (meta["planned_order"] + 3) % 48
        bad_path = Path(tmp) / "bad_meta.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        sys.argv = ["worker", "--family", target_task.family_id,
                    "--ctx", str(target_task.context_tokens), "--mode", target_task.mode,
                    "--block", str(target_task.measurement_block),
                    "--input-set", "pilot", "--meta", str(bad_path),
                    "--output-dir", str(Path(tmp) / "out2"), "--dry-run"]
        try:
            raised = False
            try:
                worker_mod.main()
            except RuntimeError:
                raised = True
        finally:
            sys.argv = saved_argv
        ck("T11 篡改 planned_order 的 meta 被拒", raised, None)

    # ---------------- T12：默认生产路径未变 ----------------
    ck("T12 默认策略仍为 eager", model_self.attention_strategy() == "eager",
       model_self.attention_strategy())
    ck("T12 正式 worker 函数未被替换",
       formal_worker.run_once.__module__ == "experiments.formal.worker", None)
    ck("T12 rope 已恢复",
       model_self.Build_Rope_cache.__name__ == "Build_Rope_cache",
       getattr(model_self.Build_Rope_cache, "__name__", None))

    # ---------------- T13/T14：正式计划与输入，仅 CPU ----------------
    formal_manifest_path, formal_families = select_input_set("formal")
    formal_tasks = plan_mod.build_plan(formal_families, contexts=CTXS, seed=4207)
    formal_summary = plan_mod.validate_plan(formal_tasks, formal_families, contexts=CTXS)
    ck("T13 正式计划 192 tasks / 480 measured / 48 pairs",
       formal_summary["n_tasks"] == 192
       and formal_summary["n_measured"] == 480
       and formal_summary["n_pairs"] == 48,
       {k: formal_summary[k] for k in ("n_tasks", "n_measured", "n_pairs")})
    ck("T13 正式 AB/BA 每个 context 均为 8/8",
       all(v == {"AB": 8, "BA": 8}
           for v in formal_summary["pair_order_by_context"].values()),
       formal_summary["pair_order_by_context"])
    ck("T13 正式 96 个 block-pair 全部相邻且 block 交错",
       formal_summary["adjacent_pair_units"] == 96
       and formal_summary["block_interleave"]["max_same_block_run"] <= 2,
       {"adjacent": formal_summary["adjacent_pair_units"],
        "block": formal_summary["block_interleave"]})

    formal_manifest = family_input.load_manifest(formal_manifest_path)
    formal_hash_ok = True
    formal_hash_detail = {}
    for family in formal_families:
        for ctx in CTXS:
            try:
                _, meta = family_input.load_family_ids(
                    tokenizer, family, manifest=formal_manifest, ctx=ctx, verify_hash=True)
                if meta["manifest_token_sha1"] is not None \
                        and meta["token_sha1"] != meta["manifest_token_sha1"]:
                    formal_hash_ok = False
                    formal_hash_detail[f"{family}_{ctx}"] = (
                        meta["token_sha1"], meta["manifest_token_sha1"])
            except Exception as exc:                                # noqa: BLE001
                formal_hash_ok = False
                formal_hash_detail[f"{family}_{ctx}"] = repr(exc)
    ck("T14 FF01–FF16 × 3 ctx 输入 hash 全部复现",
       formal_hash_ok, formal_hash_detail)

    # ---------------- T15：跨 block 同模式输出签名 ----------------
    with tempfile.TemporaryDirectory(prefix="confirmatory_signature_") as tmp:
        results = Path(tmp) / "worker_results"
        results.mkdir()
        for index in range(2):
            (results / f"measured_{index:02d}.json").write_text(
                json.dumps({"run_kind": "measured", "generated_ids": [1, 2, 3]}),
                encoding="utf-8")
        ck("T15 一致 measured 输出签名可读",
           run_mod.measured_output_signature(Path(tmp)) == (1, 2, 3), None)
        (results / "measured_01.json").write_text(
            json.dumps({"run_kind": "measured", "generated_ids": [1, 2, 4]}),
            encoding="utf-8")
        expect_raises("T15 同任务输出不一致被拒",
                      lambda: run_mod.measured_output_signature(Path(tmp)), ValueError)

    npass = sum(1 for _, ok, _ in CHECKS if ok)
    for name, ok, extra in CHECKS:
        if not ok:
            print(f"  [FAIL] {name}  {extra}")
    print(f"confirmatory CPU 自测：PASS {npass} / FAIL {len(CHECKS) - npass}")
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
