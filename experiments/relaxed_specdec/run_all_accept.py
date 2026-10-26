"""Perfect-Accept Latency Floor 计时运行器（机制诊断）。

条件：4 families × 3 contexts × {AR（target-only）, all_accept_oracle（SpecDec，100% acceptance）}
= 12 + 12 = 24 个条件；每条件独立进程、串行、warm-up 1 + measured 3。
AR 在同一次实验环境下重新运行（不跨日期拼旧 AR）。
timed pass 不做 rank/top-k/文本评分/逐 token 打印；finite 检查只在单独 diagnostic 运行。
输出 <run>/runs.jsonl 的每行携带：
    experiment_class="mechanism_diagnostic", accept_mode="all_accept_oracle"（oracle 行）,
    quality_claim_allowed=False, formal_result_allowed=False。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.execution_optimization import config  # noqa: E402
from experiments.relaxed_specdec.run_dev import wait_for_memory  # noqa: E402

FAMILIES = config.FROZEN["dev_families"]
CONTEXTS = config.FROZEN["contexts"]
K = 4
MAX_NEW = 64
REPS = 3
WARMUP = 1
COOLDOWN_S = 10
ENTRY_MIN_GIB = 16


def condition_key(role: str, family: str, ctx: int) -> str:
    return f"{family}_{ctx}_{role}"


def run_condition(role: str, family: str, ctx: int, out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.formal.runtime_info import collect_runtime_info
    from experiments.formal.runtime_audit import capture_boundary, assess_boundaries
    from experiments.relaxed_specdec import all_accept_oracle as A
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=False)
    progress = {"stage": "environment"}
    result = dict(schema_version="all-accept-condition-v1", role=role,
                  family=family, context=ctx, utc=datetime.now(timezone.utc).isoformat(),
                  frozen=config.FROZEN, execution_mode=config.execution_mode(),
                  experiment_class="mechanism_diagnostic",
                  accept_mode=("all_accept_oracle" if role == "oracle" else None),
                  quality_claim_allowed=False, formal_result_allowed=False)
    installed = False
    try:
        environment = check_environment(ENTRY_MIN_GIB)
        environment["gpu"] = check_gpu_environment()
        result["environment"] = environment
        result["runtime"] = collect_runtime_info()

        progress["stage"] = "load_target"
        target, tokenizer, target_info = W.load_target()
        draft, dtok, draft_info = None, None, None
        if role == "oracle":
            progress["stage"] = "load_draft"
            draft, dtok, draft_info = W.load_draft()
            if dtok.eos_token_id != tokenizer.eos_token_id:
                raise RuntimeError("EOS mismatch")
        rope = W.configure_target(target)
        installed = True
        result["load"] = dict(target=target_info, draft=draft_info)
        result["rope"] = rope

        progress["stage"] = "load_input"
        ids, input_meta = W.load_input(family, ctx)
        eos = tokenizer.eos_token_id
        result["input"] = input_meta
        result["eos_id"] = eos
        prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
        mode = "AR" if role == "ar" else "SpecDec"
        job = Job(family, Condition(f"L{ctx}_{mode}_base_bf16", ctx, mode, "base_bf16"), 0, 1)
        result["configuration"] = W.check_runtime_config(target, draft, job)

        progress["stage"] = "warmup"
        if role == "ar":
            W.run_ar_once(target, tokenizer, ids, job)
        else:
            A.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos)

        reps = []
        progress["stage"] = "measured"
        for rep in range(REPS):
            monitor_start = capture_boundary()
            if role == "ar":
                record = W.run_ar_once(target, tokenizer, ids, job)
            else:
                model_self.reset_attention_stats()
                out_run = A.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                   measure_timing=True, capture_cache=True)
                model_self.assert_strategy_used()
                timing = out_run["timing"]
                generated = out_run["gen"]
                record = dict(
                    max_new=MAX_NEW, generated_ids=generated, generated_length=len(generated),
                    stop_reason=("eos" if generated and generated[-1] == eos
                                 else "max_new_reached"),
                    prefill_s=timing["prefill_s"], decode_s=timing["decode_s"],
                    total_s=timing["total_s"],
                    decode_tokens_per_s=len(generated) / timing["decode_s"],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    spec_stats=out_run["stats"], ledger=out_run["ledger"],
                    control=out_run["control"], final_cache=out_run["final_cache"],
                    attention_stats=model_self.attention_stats(),
                    experiment_class=out_run["experiment_class"],
                    accept_mode=out_run["accept_mode"],
                    quality_claim_allowed=out_run["quality_claim_allowed"],
                    formal_result_allowed=out_run["formal_result_allowed"],
                )
            monitor_end = capture_boundary()
            record["runtime_environment_audit"] = assess_boundaries(monitor_start, monitor_end)
            record["rep"] = rep
            reps.append(record)
        result["reps"] = reps

        progress["stage"] = "diagnostic"
        if role == "oracle":
            model_self.reset_attention_stats()
            diag = A.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                            finite_check=True, capture_cache=True)
            model_self.assert_strategy_used()
            result["diagnostic"] = dict(generated_ids=diag["gen"], stats=diag["stats"],
                                        final_cache=diag["final_cache"])
            finite_ok = True  # finite_check 未抛异常
        else:
            diag = None
            finite_ok = True

        first = reps[0]["generated_ids"]
        result["outputs_consistent"] = all(r["generated_ids"] == first for r in reps)
        if role == "oracle":
            result["outputs_consistent"] = (result["outputs_consistent"]
                                            and reps[0]["generated_ids"] == diag["gen"])
        if not result["outputs_consistent"]:
            raise RuntimeError("同条件多次运行输出不一致")
        result["generated_ids"] = first
        result["generated_length"] = len(first)
        result["stop_reason"] = reps[0]["stop_reason"]
        result["finite"] = finite_ok
        if role == "oracle":
            fc = reps[0]["final_cache"]
            result["cache_invariants_ok"] = bool(
                fc["tgt_consistent"] and fc["drf_consistent"]
                and fc["tgt_len"] == ctx + len(first) and fc["drf_len"] == fc["tgt_len"])
            result["attention_strategy_ok"] = True
            result["spec_stats"] = reps[0]["spec_stats"]
        result["status"] = "ok"
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
        return result
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["progress"] = progress
        (out / "failure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                    default=str), encoding="utf-8")
        raise
    finally:
        if installed:
            target_rope_scope.restore()


def build_conditions() -> list:
    conds = []
    for family in FAMILIES:
        for ctx in CONTEXTS:
            conds.append(("ar", family, ctx))
            conds.append(("oracle", family, ctx))
    return conds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--role", choices=("ar", "oracle"))
    parser.add_argument("--family")
    parser.add_argument("--ctx", type=int)
    args = parser.parse_args()

    if args.role:
        run_condition(args.role, args.family, args.ctx, args.out)
        return 0

    run_dir = args.out or (ROOT / "output" / "all_accept_oracle" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    manifest = dict(schema_version="all-accept-run-manifest-v1",
                    utc=datetime.now(timezone.utc).isoformat(),
                    experiment_class="mechanism_diagnostic",
                    accept_mode="all_accept_oracle",
                    quality_claim_allowed=False, formal_result_allowed=False,
                    families=list(FAMILIES), contexts=list(CONTEXTS), k=K, max_new=MAX_NEW,
                    reps=REPS, warmup=WARMUP,
                    conditions=[dict(key=condition_key(*c), role=c[0], family=c[1],
                                     context=c[2]) for c in conditions],
                    notes=["AR 与 oracle 在同一实验环境重新运行（不跨日期拼旧 AR）",
                           "timed pass 不含 rank/top-k/文本评分；finite 在独立 diagnostic 运行",
                           "all-accept 输出无质量声明：quality_claim_allowed=False"])
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)

    runs_jsonl = run_dir / "runs.jsonl"
    failures_jsonl = run_dir / "failures.jsonl"
    for i, (role, family, ctx) in enumerate(conditions, 1):
        key = condition_key(role, family, ctx)
        cond_dir = run_dir / key
        if (cond_dir / "result.json").exists():
            print(f"[{i}/{len(conditions)}] {key} 已完成，跳过", flush=True)
            continue
        print(f"[{i}/{len(conditions)}] {key}", flush=True)
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--role", role, "--family", family, "--ctx", str(ctx), "--out", str(cond_dir)]
        log_path = run_dir / f"{key}.log"
        attempts = 0
        while True:
            attempts += 1
            free = wait_for_memory()
            print(f"    commit free={free:.2f} GiB", flush=True)
            if cond_dir.exists():
                result_file = cond_dir / "result.json"
                if result_file.exists():
                    raise RuntimeError(f"{cond_dir} 已有 result.json，拒绝覆盖")
                import shutil
                shutil.rmtree(cond_dir)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"--- attempt {attempts} at {datetime.now(timezone.utc).isoformat()} ---\n")
                with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                    rc = proc.wait()
            if rc == 0:
                result = json.loads((cond_dir / "result.json").read_text(encoding="utf-8"))
                for rep in result["reps"]:
                    row = dict(key=key, role=role, family=family, context=ctx, rep=rep["rep"],
                               experiment_class="mechanism_diagnostic",
                               accept_mode=("all_accept_oracle" if role == "oracle" else None),
                               quality_claim_allowed=False, formal_result_allowed=False,
                               input_sha1=result["input"]["token_sha1"],
                               generated_ids=rep["generated_ids"],
                               generated_length=rep["generated_length"],
                               stop_reason=rep["stop_reason"],
                               prefill_s=rep["prefill_s"], decode_s=rep["decode_s"],
                               total_s=rep["total_s"],
                               decode_tokens_per_s=rep["decode_tokens_per_s"],
                               peak_allocated_bytes=rep["peak_allocated_bytes"],
                               peak_reserved_bytes=rep["peak_reserved_bytes"],
                               boundary_audit=rep["runtime_environment_audit"],
                               spec_stats=rep.get("spec_stats"),
                               finite=result.get("finite"),
                               cache_invariants_ok=result.get("cache_invariants_ok"),
                               attention_strategy_ok=result.get("attention_strategy_ok"),
                               environment=result["environment"].get("gpu", {}).get("pstate_preflight"),
                               configuration=result["configuration"])
                    with runs_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                break
            failure = None
            fpath = cond_dir / "failure.json"
            if fpath.exists():
                failure = json.loads(fpath.read_text(encoding="utf-8"))
            if failure is not None:
                with failures_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(key=key, attempt=attempts,
                                                 error=failure.get("error"),
                                                 traceback=failure.get("traceback")),
                                            ensure_ascii=False, default=str) + "\n")
            if failure and "EnvironmentRefused" in str(failure.get("error")):
                print(f"    ENV_REFUSED (attempt {attempts}): {failure.get('error')}", flush=True)
                if attempts >= 8:
                    raise RuntimeError(f"{key} env refused 8x; see {log_path}")
                time.sleep(60)
                continue
            raise RuntimeError(f"{key} failed rc={rc}; see {log_path}")
        time.sleep(COOLDOWN_S)
    print("COMPLETE:", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
