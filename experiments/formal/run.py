import argparse
import subprocess
import sys
import time
from pathlib import Path
from .plan import build_jobs, validate_plan
import json
from dataclasses import asdict
from .records import verify_job_results
from uuid import uuid4
from .environment import check_environment
from .input_sets import select_input_set
from .runtime_info import collect_runtime_info, require_formal_version
import hashlib

def launch_job(job, command, root, *, experiment_stage="development"):
    command = list(command)
    launch_id = uuid4().hex
    launch_dir = root / "output" / ("formal_launches" if experiment_stage == "formal" else "formal_launcher_development") / launch_id
    launch_dir.mkdir(parents=True, exist_ok=False)
    result_dir = launch_dir / "worker_results"
    command.extend(["--output-dir", str(result_dir)])

    metadata = {
        "launch_id": launch_id,
        "experiment_stage": experiment_stage,
        "command": command,
        "cwd": str(root),
        "status": "planned",
    }

    with (launch_dir / "launch.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("父进程日志目录：", launch_dir, flush=True)
    stage = "launch_worker"
    try:
        with (launch_dir / "worker.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd = root,
                stdout = log,
                stderr = subprocess.STDOUT,
                check = False,
            )
        metadata["returncode"] = completed.returncode
        if completed.returncode != 0:
           metadata["status"] = "worker_failed"
           raise SystemExit(1)

        stage = "verify_results"
        metadata["result_check"] = verify_job_results(
            result_dir, job, eos_id = 151645, experiment_stage=experiment_stage,
        )
        metadata["status"] = f"{experiment_stage}_job_verified"

    except (Exception, SystemExit) as exc:
        if metadata["status"] != "worker_failed":
            metadata["status"] = (
            "result_verification_failed"
            if stage == "verify_results"
            else "launch_failed"
        )
        metadata["error"] = repr(exc)
        exc.launch_id = launch_id
        exc.launch_dir = str(launch_dir)
        raise
    finally:
        with (launch_dir / "exit.json").open("x", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print("任务结束：", 
              metadata["status"],
              "| 记录：",
              launch_dir / "exit.json",
              flush = True,)
    return metadata   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=4207)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, choices=(1, 2))
    scope.add_argument("--pilot", action="store_true")
    scope.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    input_set = "formal" if args.formal else "pilot"
    experiment_stage = "formal" if args.formal else "development"
    manifest_path, families = select_input_set(input_set)
    jobs = build_jobs(families, seed=args.seed)
    validate_plan(jobs, families)

    root = Path(__file__).resolve().parents[2]
    if args.pilot or args.formal:
        selected = jobs
        scope_name = input_set
    else:
        limit = 1 if args.limit is None else args.limit
        selected = jobs[:limit]
        scope_name = "development"

    print("完整计划任务数：", len(jobs))
    print("调度范围：", scope_name)
    print("本次任务数：", len(selected))
    print("Measured 次数：", sum(j.measured_reps for j in selected))

    batch = None
    if args.execute:
        entry_environment = check_environment(min_commit_gib=18)
        runtime = collect_runtime_info()
        expected_commit = require_formal_version(runtime) if args.formal else None
        batch_id = uuid4().hex
        batch_dir = (
            root / "output" / ("formal_batches" if args.formal else "formal_batch_development") / batch_id
        )
        batch_dir.mkdir(parents = True, exist_ok = False)
        batch = {
            "batch_id": batch_id,
            "experiment_stage": experiment_stage,
            "seed": args.seed,
            "execution_scope": scope_name,
            "entry_environment": entry_environment,
            "runtime": runtime,
            "expected_commit": expected_commit,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "speedup_claim_allowed": False,
            "jobs": [asdict(job) for job in selected],
            "completed": [],
            "active_index": None,
            "status": "running",
        }
        with (batch_dir / "plan.json").open("x", encoding="utf-8") as file:
            json.dump(batch, file, ensure_ascii=False, indent=2)
        print("批次目录：", batch_dir, flush=True)

    try:
        for index, job in enumerate(selected):
            command = [
                sys.executable, "-X", "utf8",
                "-m", "experiments.formal.worker",
                "--family", job.family_id,
                "--ctx", str(job.condition.context_tokens),
                "--mode", job.condition.mode,
                "--block", str(job.measurement_block),
                "--input-set", input_set,
            ]
            if args.execute and args.formal:
                command.extend(["--expected-commit", expected_commit])
            print(f"任务 {index + 1}：", job)
            print("命令：", subprocess.list2cmdline(command))

            if not args.execute:
                continue

            batch["active_index"] = index
            if index > 0:
                time.sleep(10)
            try:
                metadata = launch_job(job, command, root, experiment_stage=experiment_stage)
            except (Exception, SystemExit) as exc:
                batch["failed"] = {
                    "index": index,
                    "job": asdict(job),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "launch_id": getattr(exc, "launch_id", None),
                    "launch_dir": getattr(exc, "launch_dir", None),
                }
                raise
            batch["completed"].append({
                "index": index,
                "launch_id": metadata["launch_id"],
                "status": metadata["status"],
            })
            batch["active_index"] = None

        if args.execute:
            batch["status"] = f"{experiment_stage}_batch_verified"
        else:
            print("Dry-run：未启动任何 worker")

    finally:
        if batch is not None:
            if batch["status"] == "running":
                batch["status"] = "aborted"
            with (batch_dir / "summary.json").open("x", encoding="utf-8") as file:
                json.dump(batch, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
