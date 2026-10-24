"""experiments/confirmatory/worker.py — confirmatory 单任务 worker。

模型计算与生成逻辑**全部复用** `experiments/formal/worker.py` 的函数
（load_input / load_target / load_draft / configure_target / check_runtime_config /
run_diagnostic / run_once / capture_boundary / assess_boundaries / save_record /
save_diagnostic / save_failure / 环境门与 runtime 采集），本文件只负责：
* 读取任务元数据（confirmatory_batch_id / pair_id / pair_order / pair_position /
  planned_order / schedule_seed / actual_start_utc）；
* 与本地重建的 confirmatory 计划逐字段交叉校验（防止运行期换任务）；
* 在每条 measured record / diagnostic / failure 上附加上述元数据字段；
* 记录 worker 自身启动时间。

用法（由 experiments.confirmatory.run 调度）：
    python -m experiments.confirmatory.worker --family F09 --ctx 2048 --mode AR \
        --block 0 --input-set pilot --meta <task_meta.json> --output-dir <dir>
dry-run（不加载模型、不触碰 GPU）：
    python -m experiments.confirmatory.worker ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from experiments.formal import worker as formal_worker
from experiments.formal.environment import EnvironmentRefused, check_environment, check_gpu_environment
from experiments.formal.input_sets import select_input_set
from experiments.formal.records import save_diagnostic, save_failure, save_record
from experiments.formal.runtime_audit import assess_boundaries, capture_boundary
from experiments.formal.runtime_info import collect_runtime_info, require_formal_version
from specdec.engine import target_rope_scope

from .plan import build_plan

META_FIELDS = ("confirmatory_batch_id", "pair_id", "pair_order", "pair_position",
               "planned_order", "schedule_seed", "actual_start_utc", "input_set",
               "expected_commit")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-set", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--family", required=True)
    parser.add_argument("--ctx", required=True, type=int, choices=(2048, 4096, 8192))
    parser.add_argument("--mode", required=True, choices=("AR", "SpecDec"))
    parser.add_argument("--block", required=True, type=int, choices=(0, 1))
    parser.add_argument("--meta", type=Path, required=True,
                        help="confirmatory 任务元数据 JSON（由 confirmatory.run 写出）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_meta(path: Path) -> dict:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in META_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"任务元数据缺少字段：{missing}")
    return meta


def find_planned_task(meta: dict, family: str, ctx: int, mode: str, block: int):
    """用元数据里的 seed/families/contexts 重建计划并定位本任务，逐字段交叉校验。"""
    tasks = build_plan(meta["families"], contexts=meta["contexts"], seed=meta["schedule_seed"])
    matches = [t for t in tasks
               if t.family_id == family and t.context_tokens == ctx
               and t.mode == mode and t.measurement_block == block]
    if len(matches) != 1:
        raise RuntimeError(f"任务在计划中命中 {len(matches)} 次（应为 1）")
    task = matches[0]
    expected = {
        "pair_id": task.pair_id, "pair_order": task.pair_order,
        "pair_position": task.pair_position, "planned_order": task.planned_order,
        "measured_reps": task.measured_reps,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"任务元数据与计划不一致（{key}）：meta={meta.get(key)!r} 计划={value!r}")
    if meta.get("mode") != mode or meta.get("context_tokens") != ctx \
            or meta.get("family_id") != family or meta.get("measurement_block") != block:
        raise RuntimeError("任务元数据（family/ctx/mode/block）与命令行不一致")
    return task


def main() -> int:
    args = parse_args()
    meta = load_meta(args.meta)
    if meta["input_set"] != args.input_set:
        raise RuntimeError(
            f"任务元数据 input_set={meta['input_set']!r} 与命令行 {args.input_set!r} 不一致")
    task = find_planned_task(meta, args.family, args.ctx, args.mode, args.block)
    job = task.to_job()

    if args.dry_run:
        print(json.dumps({"planned": asdict(task), "job": asdict(job)},
                         ensure_ascii=False, indent=2))
        return 0

    manifest_path, families = select_input_set(args.input_set)
    if args.family not in families:
        raise SystemExit(f"{args.family} 不属于输入集合 {args.input_set}")

    worker_started = utc_now()
    run_id = uuid4().hex
    experiment_stage = "formal" if args.input_set == "formal" else "development"
    output_dir = Path(args.output_dir)
    progress = {"stage": "environment"}
    environment = None
    rope_installed = False
    meta_fields = {key: meta[key] for key in META_FIELDS}
    meta_fields["worker_started_utc"] = worker_started

    try:
        environment = check_environment()
        progress["stage"] = "runtime_info"
        runtime_info = collect_runtime_info()
        if experiment_stage == "formal":
            if not meta.get("expected_commit"):
                raise RuntimeError("formal worker 缺少 expected_commit")
            require_formal_version(runtime_info, meta["expected_commit"])
        progress["stage"] = "gpu_preflight"
        environment["gpu"] = check_gpu_environment()
        progress["stage"] = "load_input"
        ids, metadata = formal_worker.load_input(
            job.family_id, job.condition.context_tokens, input_set=args.input_set)

        progress["stage"] = "load_target"
        model, tokenizer, load_info = formal_worker.load_target()
        draft = None
        draft_info = None
        if job.condition.mode == "SpecDec":
            progress["stage"] = "load_draft"
            draft, draft_tokenizer, draft_info = formal_worker.load_draft()
            if draft_tokenizer.eos_token_id != tokenizer.eos_token_id:
                raise ValueError("Target 与 Draft 的 EOS ID 不一致")

        progress["stage"] = "configure"
        rope_info = formal_worker.configure_target(model)
        rope_installed = True
        config_info = formal_worker.check_runtime_config(model, draft, job)

        provenance = {
            "input": metadata,
            "load": load_info,
            "draft_load": draft_info,
            "rope": rope_info,
            "environment": environment,
            "runtime": runtime_info,
            "configuration": config_info,
            "collection": {
                "input_set": args.input_set,
                "speedup_claim_allowed": False,
                "claim_status": "confirmatory_raw_pending_analysis",
                **meta_fields,
            },
        }

        progress["stage"] = "diagnostic"
        diagnostic = formal_worker.run_diagnostic(model, draft, tokenizer, ids, job)
        diagnostic.update(provenance)
        diagnostic["run_id"] = run_id
        diagnostic["job"] = asdict(job)
        diagnostic["experiment_stage"] = experiment_stage
        diagnostic.update(meta_fields)
        progress["stage"] = "save_diagnostic"
        save_diagnostic(diagnostic, output_dir / "diagnostics" / "result.json")

        progress["stage"] = "warmup"
        warmup = formal_worker.run_once(model, draft, tokenizer, ids, job)
        del warmup

        for rep_index in range(job.measured_reps):
            progress["rep_index"] = rep_index
            progress["stage"] = "monitor_start"
            monitor_start = capture_boundary()
            progress["stage"] = "measured"
            result = formal_worker.run_once(model, draft, tokenizer, ids, job)
            progress["stage"] = "monitor_end"
            monitor_end = capture_boundary()
            result["runtime_environment_audit"] = assess_boundaries(monitor_start, monitor_end)
            result.update(provenance)
            result["diagnostic_file"] = "diagnostics/result.json"
            result["same_mode_diagnostic_tokens_equal"] = (
                result["generated_ids"] == diagnostic["generated_ids"])
            result["run_id"] = run_id
            result["rep_index"] = rep_index
            result["run_kind"] = "measured"
            result["experiment_stage"] = experiment_stage
            result["eos_id"] = tokenizer.eos_token_id
            result.update(meta_fields)
            filename = (f"{job.family_id}_{job.condition.condition_id}"
                        f"_b{job.measurement_block}_r{rep_index}.json")
            progress["stage"] = "save_record"
            save_record(result, output_dir / filename, tokenizer.eos_token_id)
            progress["stage"] = "compare_diagnostic"
            if not result["same_mode_diagnostic_tokens_equal"]:
                raise RuntimeError("同模式诊断与计时输出不一致，保留结果并停止")
            print(f"Measured {rep_index + 1} / {job.measured_reps}: "
                  f"prefill={result['prefill_s']:.3f}s decode={result['decode_s']:.3f}s",
                  flush=True)
        print(f"confirmatory worker 完成：{run_id} records={job.measured_reps}", flush=True)
        return 0
    except Exception as exc:
        failure_class = ("ENV_REFUSED" if isinstance(exc, EnvironmentRefused)
                         else "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError)
                         else "ERROR")
        failure = {
            "run_id": run_id, "job": asdict(job),
            **progress,
            "failure_class": failure_class,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "environment": environment,
            "experiment_stage": experiment_stage,
            "environment_failure_evidence": getattr(exc, "evidence", None),
            **meta_fields,
        }
        save_failure(failure, output_dir / "failure.json")
        raise
    finally:
        if rope_installed:
            target_rope_scope.restore()


if __name__ == "__main__":
    raise SystemExit(main())
