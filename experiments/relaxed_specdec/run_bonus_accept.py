"""Bonus-Aware Perfect-Accept Oracle 计时运行器（同批 36 条件）。

三种模式（同一批、同一环境、独立进程、串行）：
  ar       : target-only（AR 基线）
  nobonus  : all_accept_oracle（原无 bonus 契约，16 轮）
  bonus    : bonus_accept_oracle carry-forward（pending-bonus，13 轮，timed path 无额外 target 单步）
每条件 warm-up 1 + measured 3；timed pass 不含 rank/top-k/文本评分；finite 在独立 diagnostic 运行。
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
ROLES = ("ar", "nobonus", "bonus")


def condition_key(role: str, family: str, ctx: int) -> str:
    return f"{family}_{ctx}_{role}"


def run_condition(role: str, family: str, ctx: int, out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.formal.runtime_info import collect_runtime_info
    from experiments.formal.runtime_audit import capture_boundary, assess_boundaries
    from experiments.relaxed_specdec import all_accept_oracle as A0
    from experiments.relaxed_specdec import bonus_accept_oracle as B
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=False)
    progress = {"stage": "environment"}
    result = dict(schema_version="bonus-accept-condition-v1", role=role,
                  family=family, context=ctx, utc=datetime.now(timezone.utc).isoformat(),
                  frozen=config.FROZEN, execution_mode=config.execution_mode(),
                  experiment_class="mechanism_diagnostic",
                  accept_mode=("all_accept_bonus_oracle" if role == "bonus"
                               else "all_accept_oracle" if role == "nobonus" else None),
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
        if role != "ar":
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
        mode_name = "AR" if role == "ar" else "SpecDec"
        job = Job(family, Condition(f"L{ctx}_{mode_name}_base_bf16", ctx, mode_name, "base_bf16"), 0, 1)
        result["configuration"] = W.check_runtime_config(target, draft, job)

        def run_oracle(which, measure):
            if which == "nobonus":
                return A0.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                 measure_timing=measure, capture_cache=True)
            return B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos, mode="carry",
                            measure_timing=measure, capture_cache=True)

        progress["stage"] = "warmup"
        if role == "ar":
            W.run_ar_once(target, tokenizer, ids, job)
        else:
            run_oracle(role, False)

        reps = []
        progress["stage"] = "measured"
        for rep in range(REPS):
            monitor_start = capture_boundary()
            if role == "ar":
                record = W.run_ar_once(target, tokenizer, ids, job)
            else:
                model_self.reset_attention_stats()
                out_run = run_oracle(role, True)
                model_self.assert_strategy_used()
                timing = out_run["timing"]
                generated = out_run["gen"]
                st = out_run["stats"]
                record = dict(
                    max_new=MAX_NEW, generated_ids=generated, generated_length=len(generated),
                    stop_reason=("eos" if generated and generated[-1] == eos
                                 else "max_new_reached"),
                    prefill_s=timing["prefill_s"], decode_s=timing["decode_s"],
                    total_s=timing["total_s"],
                    decode_tokens_per_s=len(generated) / timing["decode_s"],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    spec_stats=st, ledger=out_run["ledger"],
                    control=out_run.get("control"),
                    final_cache=out_run["final_cache"],
                    pending_at_stop=out_run.get("pending_at_stop"),
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
        if role != "ar":
            model_self.reset_attention_stats()
            # 显式 diagnostic：finite 检查（不进计时）
            if role == "nobonus":
                diag = A0.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                 finite_check=True, capture_cache=True)
            else:
                diag = B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos, mode="carry",
                                finite_check=True, capture_cache=True)
            model_self.assert_strategy_used()
            result["diagnostic"] = dict(generated_ids=diag["gen"], stats=diag["stats"],
                                        final_cache=diag["final_cache"],
                                        control=diag.get("control"),
                                        pending_at_stop=diag.get("pending_at_stop"))
            finite_ok = True
        else:
            finite_ok = True

        first = reps[0]["generated_ids"]
        result["outputs_consistent"] = all(r["generated_ids"] == first for r in reps)
        if role != "ar":
            result["outputs_consistent"] = result["outputs_consistent"] and diag["gen"] == first
        if not result["outputs_consistent"]:
            raise RuntimeError("同条件多次运行输出不一致")
        result["generated_ids"] = first
        result["generated_length"] = len(first)
        result["stop_reason"] = reps[0]["stop_reason"]
        result["finite"] = finite_ok
        if role != "ar":
            fc = reps[0]["final_cache"]
            result["cache_invariants_ok"] = bool(fc["tgt_consistent"] and fc["drf_consistent"])
            result["attention_strategy_ok"] = True
            result["spec_stats"] = reps[0]["spec_stats"]
            result["control"] = reps[0]["control"]
            result["pending_at_stop"] = reps[0]["pending_at_stop"]
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
            for role in ROLES:
                conds.append((role, family, ctx))
    return conds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--family")
    parser.add_argument("--ctx", type=int)
    args = parser.parse_args()

    if args.role:
        run_condition(args.role, args.family, args.ctx, args.out)
        return 0

    run_dir = args.out or (ROOT / "output" / "bonus_accept_oracle" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    manifest = dict(schema_version="bonus-accept-run-manifest-v1",
                    utc=datetime.now(timezone.utc).isoformat(),
                    experiment_class="mechanism_diagnostic",
                    accept_mode="all_accept_bonus_oracle",
                    quality_claim_allowed=False, formal_result_allowed=False,
                    families=list(FAMILIES), contexts=list(CONTEXTS), k=K, max_new=MAX_NEW,
                    reps=REPS, warmup=WARMUP,
                    conditions=[dict(key=condition_key(*c), role=c[0], family=c[1],
                                     context=c[2]) for c in conditions],
                    notes=["AR / nobonus / bonus 同批同环境重跑（不跨日期拼旧数据）",
                           "bonus 使用 pending-bonus carry-forward：target 只做 block call，timed path 无额外 target 单步",
                           "timed pass 不含 rank/top-k/文本评分；finite 在独立 diagnostic 运行",
                           "bonus-aware 输出无质量声明：quality_claim_allowed=False"])
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
                               accept_mode=result.get("accept_mode"),
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
                               ledger=rep.get("ledger"),
                               control=rep.get("control"),
                               pending_at_stop=rep.get("pending_at_stop"),
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
