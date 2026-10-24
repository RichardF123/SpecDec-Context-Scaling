"""experiments/confirmatory/run.py — confirmatory 批次编排器（串行、失败即停）。

用法：
    python -m experiments.confirmatory.run                         # pilot dry-run
    python -m experiments.confirmatory.run --input-set formal      # 192-task formal dry-run
    python -m experiments.confirmatory.run --input-set formal --execute \
        --expected-commit <clean-frozen-commit>                    # 完整确认实验

产物：output/confirmatory_pilot/<batch_id>/
    plan.json / batch.json / summary.json
    tasks/<NN>_<family>_<ctx>_<mode>_b<block>/{task_meta.json,launch.json,worker.log,worker_results,exit.json}

不改正式默认：worker 复用 experiments/formal/worker.py 的全部模型计算与生成逻辑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from experiments.formal.environment import EnvironmentRefused, check_environment
from experiments.formal.input_sets import select_input_set
from experiments.formal.records import verify_job_results
from experiments.formal.runtime_info import collect_runtime_info, require_formal_version

from .plan import CONTEXTS, build_plan, save_plan, validate_plan

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOTS = {
    "pilot": ROOT / "output" / "confirmatory_pilot",
    "formal": ROOT / "output" / "confirmatory_full",
}
OUTPUT_ROOT = OUTPUT_ROOTS["pilot"]  # 兼容既有 CPU 测试；正式路径由 input-set 选择。
COOLDOWN_S = 10
ENTRY_MIN_COMMIT_GIB = 18
EOS_ID = 151645
DEV_FAMILIES = ("F09", "F03", "F11", "F04")
PREREGISTRATION = ROOT / "docs" / "confirmatory_full_protocol_2026-10-25.md"
SOURCE_FILES = (
    "experiments/confirmatory/plan.py",
    "experiments/confirmatory/worker.py",
    "experiments/confirmatory/run.py",
    "experiments/confirmatory/analyze_pilot.py",
    "experiments/confirmatory/audit_full_ready.py",
    "docs/confirmatory_full_protocol_2026-10-25.md",
    "docs/prompt_families_formal_manifest.json",
    "experiments/formal/worker.py",
    "experiments/formal/diagnostics.py",
    "experiments/formal/records.py",
    "experiments/formal/runtime_audit.py",
    "experiments/formal/environment.py",
    "specdec/engine/model_self.py",
    "specdec/engine/spec_dec.py",
    "specdec/engine/spec_cache.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_sha256() -> dict:
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            if (ROOT / rel).exists() else None for rel in SOURCE_FILES}


def task_dirname(task) -> str:
    return f"{task.planned_order:02d}_{task.family_id}_{task.context_tokens}_{task.mode}_b{task.measurement_block}"


def print_plan(tasks) -> None:
    for task in tasks:
        print(f"  {task.planned_order:2d}. {task.family_id} ctx={task.context_tokens} "
              f"mode={task.mode:<7} block={task.measurement_block} reps={task.measured_reps} "
              f"pair={task.pair_id} order={task.pair_order} pos={task.pair_position}")


def launch_task(task, batch_dir: Path, *, seed: int, families, contexts,
                input_set: str, expected_commit: str | None) -> dict:
    task_dir = batch_dir / "tasks" / task_dirname(task)
    task_dir.mkdir(parents=True, exist_ok=False)
    result_dir = task_dir / "worker_results"
    meta = dict(
        meta_schema="confirmatory-task-meta-v1",
        confirmatory_batch_id=batch_dir.name,
        plan_seed=seed,
        schedule_seed=seed,
        families=list(families),
        contexts=list(contexts),
        family_id=task.family_id,
        context_tokens=task.context_tokens,
        mode=task.mode,
        measurement_block=task.measurement_block,
        measured_reps=task.measured_reps,
        pair_id=task.pair_id,
        pair_order=task.pair_order,
        pair_position=task.pair_position,
        planned_order=task.planned_order,
        actual_start_utc=utc_now(),
        input_set=input_set,
        expected_commit=expected_commit,
    )
    meta_path = task_dir / "task_meta.json"
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    command = [
        sys.executable, "-X", "utf8",
        "-m", "experiments.confirmatory.worker",
        "--family", task.family_id,
        "--ctx", str(task.context_tokens),
        "--mode", task.mode,
        "--block", str(task.measurement_block),
        "--input-set", input_set,
        "--meta", str(meta_path),
        "--output-dir", str(result_dir),
    ]
    launch = {
        "task": dict(planned_order=task.planned_order, family_id=task.family_id,
                     context_tokens=task.context_tokens, mode=task.mode,
                     measurement_block=task.measurement_block,
                     measured_reps=task.measured_reps, pair_id=task.pair_id,
                     pair_order=task.pair_order, pair_position=task.pair_position),
        "command": command,
        "actual_start_utc": meta["actual_start_utc"],
        "status": "planned",
    }
    with (task_dir / "launch.json").open("x", encoding="utf-8") as handle:
        json.dump(launch, handle, ensure_ascii=False, indent=2)

    stage = "launch_worker"
    try:
        with (task_dir / "worker.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        launch["returncode"] = completed.returncode
        if completed.returncode != 0:
            launch["status"] = "worker_failed"
            raise SystemExit(1)
        stage = "verify_results"
        experiment_stage = "formal" if input_set == "formal" else "development"
        launch["result_check"] = verify_job_results(
            result_dir, task.to_job(), eos_id=EOS_ID,
            experiment_stage=experiment_stage)
        launch["status"] = f"{experiment_stage}_job_verified"
        launch["actual_end_utc"] = utc_now()
    except (Exception, SystemExit) as exc:
        if launch["status"] != "worker_failed":
            launch["status"] = ("result_verification_failed" if stage == "verify_results"
                                else "launch_failed")
        launch["error"] = repr(exc)
        exc.task_dir = str(task_dir)
        raise
    finally:
        with (task_dir / "exit.json").open("x", encoding="utf-8") as handle:
            json.dump(launch, handle, ensure_ascii=False, indent=2)
        print("任务结束：", launch["status"], "|", task_dir / "exit.json", flush=True)
    return launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-set", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--expected-commit",
                        help="formal --execute 必填；必须等于 clean worktree 的 HEAD")
    parser.add_argument("--seed", type=int, default=4207)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=list(CONTEXTS))
    args = parser.parse_args()

    manifest_path, registered_families = select_input_set(args.input_set)
    families = tuple(args.families) if args.families is not None else tuple(registered_families)
    if families != tuple(registered_families):
        raise SystemExit(
            f"--families 必须与 {args.input_set} 冻结集合及顺序一致：{registered_families}")
    contexts = tuple(args.contexts)
    tasks = build_plan(families, contexts=contexts, seed=args.seed)
    summary = validate_plan(tasks, families, contexts=contexts)

    print("输入集合：", args.input_set, families, "清单：", manifest_path)
    print(f"上下文：{contexts}；mode：AR/SpecDec；block：0=3 reps, 1=2 reps")
    print(f"任务数：{summary['n_tasks']}；measured 总数：{summary['n_measured']}；"
          f"配对：{summary['n_pairs']}（AB={summary['n_ab']} BA={summary['n_ba']}）")
    print("按 context 的 AB/BA：", summary["pair_order_by_context"])
    print("相邻 pair 单元：", summary["adjacent_pair_units"])
    print(f"block 交错：transitions={summary['block_interleave']['transitions']}，"
          f"max_same_block_run={summary['block_interleave']['max_same_block_run']}")
    print("执行序列：")
    print_plan(tasks)

    if not args.execute:
        print("Dry-run：未加载模型、未触碰 GPU、未创建任何目录")
        return 0

    if args.input_set == "formal" and not args.expected_commit:
        raise SystemExit("formal --execute 必须提供 --expected-commit <clean-frozen-commit>")
    runtime = collect_runtime_info()
    if args.input_set == "formal":
        require_formal_version(runtime, args.expected_commit)
    output_root = OUTPUT_ROOTS[args.input_set]
    try:
        entry = check_environment(min_commit_gib=ENTRY_MIN_COMMIT_GIB)
    except EnvironmentRefused as exc:
        refusal_dir = output_root / "refusals"
        refusal_dir.mkdir(parents=True, exist_ok=True)
        batch_id = f"{utc_now().replace(':', '').replace('-', '')}_{uuid4().hex[:8]}"
        path = refusal_dir / f"{batch_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(schema_version="confirmatory-refusal-v1", status="ENV_REFUSED",
                           stage="entry_gate", detail=str(exc),
                           requested=dict(min_commit_gib=ENTRY_MIN_COMMIT_GIB),
                           runtime=runtime, input_set=args.input_set,
                           families=list(families),
                           contexts=list(args.contexts)), handle,
                      ensure_ascii=False, indent=2)
        print(f"入口环境门槛拒绝：{exc}", file=sys.stderr, flush=True)
        print(f"证据：{path}", file=sys.stderr, flush=True)
        raise

    batch_id = f"{utc_now().replace(':', '').replace('-', '')}_{uuid4().hex[:8]}"
    batch_dir = output_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    plan_path = save_plan(
        tasks, families, batch_dir / "plan.json", contexts=contexts, seed=args.seed,
        batch_id=batch_id, input_manifest=manifest_path,
        provenance={
            "git": {"git_commit": runtime["git_commit"],
                    "dirty_worktree": runtime["dirty_worktree"]},
            "runtime": runtime,
            "source_sha256": source_sha256(),
            "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
            "note": (f"confirmatory {args.input_set}：context 分层 AB/BA + 相邻 pair + "
                     "block pair-unit 时间交错；"
                     "worker 复用正式 worker 的模型计算与生成逻辑"),
        })
    batch = {
        "schema_version": "confirmatory-batch-v1",
        "batch_id": batch_id,
        "experiment": f"specdec_confirmatory_{args.input_set}",
        "phase": "confirmatory_formal" if args.input_set == "formal" else "confirmatory_pilot",
        "input_set": args.input_set,
        "expected_commit": args.expected_commit,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "seed": args.seed,
        "families": list(families),
        "contexts": list(contexts),
        "plan_json": str(plan_path),
        "entry_environment": entry,
        "runtime": runtime,
        "source_sha256": source_sha256(),
        "n_tasks": len(tasks),
        "n_measured": sum(t.measured_reps for t in tasks),
        "completed": [],
        "active_index": None,
        "status": "running",
        "speedup_claim_allowed": False,
    }
    with (batch_dir / "batch.json").open("x", encoding="utf-8") as handle:
        json.dump(batch, handle, ensure_ascii=False, indent=2)
    print("批次目录：", batch_dir, flush=True)
    try:
        for index, task in enumerate(tasks):
            if index > 0:
                time.sleep(COOLDOWN_S)
            print(f"任务 {index + 1}/{len(tasks)}：{task.family_id} ctx={task.context_tokens} "
                  f"{task.mode} b{task.measurement_block}（planned_order={task.planned_order}）",
                  flush=True)
            batch["active_index"] = index
            launch = launch_task(task, batch_dir, seed=args.seed,
                                 families=families, contexts=contexts,
                                 input_set=args.input_set,
                                 expected_commit=args.expected_commit)
            batch["completed"].append({"index": index, "planned_order": task.planned_order,
                                       "status": launch["status"],
                                       "task_dir": str(batch_dir / "tasks" / task_dirname(task)),
                                       "actual_start_utc": launch["actual_start_utc"],
                                       "actual_end_utc": launch.get("actual_end_utc")})
            batch["active_index"] = None
        batch["status"] = ("formal_batch_verified" if args.input_set == "formal"
                           else "development_batch_verified")
        return 0
    except (Exception, SystemExit) as exc:
        batch["failed"] = {
            "index": batch["active_index"],
            "task": dict(planned_order=tasks[batch["active_index"]].planned_order,
                         family_id=tasks[batch["active_index"]].family_id,
                         context_tokens=tasks[batch["active_index"]].context_tokens,
                         mode=tasks[batch["active_index"]].mode,
                         measurement_block=tasks[batch["active_index"]].measurement_block)
            if batch["active_index"] is not None else None,
            "error_type": type(exc).__name__, "error": str(exc),
            "task_dir": getattr(exc, "task_dir", None),
        }
        raise
    finally:
        if batch["status"] == "running":
            batch["status"] = "aborted"
        with (batch_dir / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(batch, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
