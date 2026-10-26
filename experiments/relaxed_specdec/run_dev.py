"""relaxed_specdec 开发集运行器（Stage A / Stage B）。

设计（与预注册口径一致）：
  * 条件：AR（4 families × 3 contexts，独立进程）与 relaxed SpecDec（4×3×5 delta，独立进程）；
  * 每条件：warm-up 1 次（丢弃）+ measured 3 次；串行、不并行、不共享 cache；
  * 主计时：decode_s 由 warm-up 后的 measured 运行测得（AR 用 W.run_ar_once 口径；
    relaxed 用 relaxed_spec.decode 的 measure_timing 口径，prefill 分块=2048）；
  * 观察者效应：measured 运行**不**做 finite 检查；finite/rank 类检查在单独 diagnostic 运行中完成，
    不进入计时；每个 rep 记录 runtime_environment_audit（boundary clock）；
  * 不自动重试、不挑选结果；只有 EnvironmentRefused 时等待自然恢复后重试同一条件（记录证据）。

用法：
  python -m experiments.relaxed_specdec.run_dev --stage a --out <dir>
  python -m experiments.relaxed_specdec.run_dev --stage b --out <dir>
  python -m experiments.relaxed_specdec.run_dev --role relaxed --family F09 --ctx 2048 --delta 0.5 --out <cond_dir>
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

FAMILIES = config.FROZEN["dev_families"]
CONTEXTS = config.FROZEN["contexts"]
DELTAS = (0.0, 0.1, 0.3, 0.5, 1.0)
STAGE_A_DELTAS = (0.0, 0.3, 0.5, 1.0)
STAGE_A_FAMILY = "F09"
STAGE_A_CTX = 2048
REPS = 3
WARMUP = 1
COOLDOWN_S = 10
ENTRY_MIN_GIB = 16


def wait_for_memory(min_gib: float = 16.1, timeout_s: int = 3600) -> float:
    from specdec.engine.g5_8k_spec_variance_probe import system_memory
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        m = system_memory()
        if m.get("available"):
            last = m["commit_free_mib"] / 1024.0
            if last >= min_gib:
                return last
        time.sleep(20)
    raise RuntimeError(f"等待内存恢复超时（最后 commit free={last} GiB）")


def run_condition(role: str, family: str, ctx: int, delta, out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.formal.runtime_info import collect_runtime_info
    from experiments.formal.runtime_audit import capture_boundary, assess_boundaries
    from experiments.relaxed_specdec import relaxed_spec as R
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=False)
    progress = {"stage": "environment"}
    result = dict(schema_version="relaxed-condition-v1", role=role, family=family,
                  context=ctx, delta=delta,
                  utc=datetime.now(timezone.utc).isoformat(),
                  frozen=config.FROZEN, execution_mode=config.execution_mode())
    installed = False
    try:
        environment = check_environment(ENTRY_MIN_GIB)
        environment["gpu"] = check_gpu_environment()
        result["environment"] = environment
        result["runtime"] = collect_runtime_info()

        progress["stage"] = "load_target"
        target, tokenizer, target_info = W.load_target()
        draft = None
        draft_info = None
        if role == "relaxed":
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

        # warm-up
        progress["stage"] = "warmup"
        if role == "ar":
            W.run_ar_once(target, tokenizer, ids, job)
        else:
            R.decode(target, draft, prompt, 64, k=4, delta=delta, eos_id=eos)

        reps = []
        progress["stage"] = "measured"
        for rep in range(REPS):
            monitor_start = capture_boundary()
            if role == "ar":
                record = W.run_ar_once(target, tokenizer, ids, job)
            else:
                model_self.reset_attention_stats()
                out_run = R.decode(target, draft, prompt, 64, k=4, delta=delta, eos_id=eos,
                                   measure_timing=True, capture_cache=True)
                model_self.assert_strategy_used()
                timing = out_run["timing"]
                generated = out_run["gen"]
                record = dict(
                    max_new=64, generated_ids=generated, generated_length=len(generated),
                    stop_reason=("eos" if generated and generated[-1] == eos
                                 else "max_new_reached"),
                    prefill_s=timing["prefill_s"], decode_s=timing["decode_s"],
                    total_s=timing["total_s"],
                    decode_tokens_per_s=len(generated) / timing["decode_s"],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    spec_stats=out_run["stats"], control=out_run["control"],
                    ledger=out_run["ledger"], final_cache=out_run["final_cache"],
                    attention_stats=model_self.attention_stats(),
                )
            monitor_end = capture_boundary()
            record["runtime_environment_audit"] = assess_boundaries(monitor_start, monitor_end)
            record["rep"] = rep
            reps.append(record)
        result["reps"] = reps

        # 诊断运行（不进计时）：finite 检查 + 输出一致性
        progress["stage"] = "diagnostic"
        if role == "relaxed":
            model_self.reset_attention_stats()
            diag = R.decode(target, draft, prompt, 64, k=4, delta=delta, eos_id=eos,
                            finite_check=True, capture_cache=True)
            model_self.assert_strategy_used()
            result["diagnostic"] = dict(generated_ids=diag["gen"],
                                        final_cache=diag["final_cache"],
                                        stats=diag["stats"])
            finite_ok = True  # finite_check 未抛异常即视为通过
        else:
            diag = None
            finite_ok = True

        first = reps[0]["generated_ids"]
        result["outputs_consistent"] = all(r["generated_ids"] == first for r in reps)
        if role == "relaxed":
            result["outputs_consistent"] = (result["outputs_consistent"]
                                            and reps[0]["generated_ids"] == diag["gen"])
        if not result["outputs_consistent"]:
            raise RuntimeError("同条件多次运行输出不一致")
        result["generated_ids"] = first
        result["generated_length"] = len(first)
        result["stop_reason"] = reps[0]["stop_reason"]
        result["finite"] = finite_ok
        if role == "relaxed":
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


def condition_key(role: str, family: str, ctx: int, delta) -> str:
    if role == "ar":
        return f"{family}_{ctx}_ar"
    return f"{family}_{ctx}_relaxed_d{str(delta).replace('.', 'p')}"


def build_conditions(stage: str) -> list:
    if stage == "a":
        conds = [("ar", STAGE_A_FAMILY, STAGE_A_CTX, None)]
        for delta in STAGE_A_DELTAS:
            conds.append(("relaxed", STAGE_A_FAMILY, STAGE_A_CTX, delta))
        return conds
    conds = []
    for family in FAMILIES:
        for ctx in CONTEXTS:
            conds.append(("ar", family, ctx, None))
            for delta in DELTAS:
                conds.append(("relaxed", family, ctx, delta))
    return conds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b"), default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--role", choices=("ar", "relaxed"))
    parser.add_argument("--family")
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--delta", type=float)
    args = parser.parse_args()

    if args.role:
        delta = args.delta if args.role == "relaxed" else None
        run_condition(args.role, args.family, args.ctx, delta, args.out)
        return 0

    stage = args.stage or "a"
    run_dir = args.out or (ROOT / "output" / "relaxed_specdec" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(stage)
    manifest_path = run_dir / "manifest.json"
    manifest = dict(schema_version="relaxed-run-manifest-v1",
                    utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                    families=list(FAMILIES), contexts=list(CONTEXTS), deltas=list(DELTAS),
                    stage_a=dict(family=STAGE_A_FAMILY, context=STAGE_A_CTX,
                                 deltas=list(STAGE_A_DELTAS)),
                    reps=REPS, warmup=WARMUP,
                    conditions=[dict(key=condition_key(*c), role=c[0], family=c[1],
                                     context=c[2], delta=c[3]) for c in conditions],
                    notes=["主计时 = measured 运行的 decode_s（AR 用 run_ar_once 口径；relaxed 用 decode measure_timing）",
                           "finite/rank 检查只在单独 diagnostic 运行，不进计时",
                           "环境拒绝仅等待自然恢复后重试，不改参数/门槛"])
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = {c["key"]: c for c in old["conditions"]}
        for c in manifest["conditions"]:
            merged.setdefault(c["key"], c)
        manifest["conditions"] = list(merged.values())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)

    runs_jsonl = run_dir / "runs.jsonl"
    failures_jsonl = run_dir / "failures.jsonl"
    for i, (role, family, ctx, delta) in enumerate(conditions, 1):
        key = condition_key(role, family, ctx, delta)
        cond_dir = run_dir / key
        if (cond_dir / "result.json").exists():
            print(f"[{i}/{len(conditions)}] {key} 已完成，跳过", flush=True)
            continue
        print(f"[{i}/{len(conditions)}] {key}", flush=True)
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--role", role, "--family", family, "--ctx", str(ctx), "--out", str(cond_dir)]
        if role == "relaxed":
            cmd += ["--delta", repr(float(delta))]
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
                    row = dict(key=key, role=role, family=family, context=ctx, delta=delta,
                               rep=rep["rep"],
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
