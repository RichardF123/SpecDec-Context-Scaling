"""Phase O 开发集基准：4 开发 family × 3 上下文 × {AR, reference SpecDec, optimized SpecDec}。

每个条件一个**独立进程**、串行执行：
  * warm-up 1 次（丢弃）；
  * measured 3 次（主计时：decode_s = total - prefill，与正式口径一致，不加任何阶段事件同步）；
  * 另外 1 次**诊断计时**（带 CUDA Event 阶段账本，单独保存，不进主比例；其输出必须与
    measured/reference 一致，否则报错）。
  * 记录 boundary-clock 资质、峰值显存、P-state 证据、输入 hash。
AR 使用 target-only 独立进程；reference 与 optimized 各自独立进程。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
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
REPS = 3
WARMUP = 1
COOLDOWN_S = 10


@contextmanager
def phase_scope(spec_dec_mod, target, draft, timed=True):
    """与 scripts/profile_specdec_cost_2k.py 同口径的 CUDA Event 阶段账本（诊断用）。"""
    names = ("_propose", "_verify", "_step")
    originals = {name: getattr(spec_dec_mod, name) for name in names}
    events = []

    def wrap(name, function):
        def wrapped(*args, **kwargs):
            role = "target" if args[0] is target else "draft" if args[0] is draft else "unknown"
            phase = {"_propose": "draft_propose", "_verify": "target_verify",
                     "_step": "repair_" + role}[name]
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            result = function(*args, **kwargs)
            end.record()
            end.synchronize()
            events.append(dict(phase=phase, wall_s=time.perf_counter() - start,
                               cuda_interval_ms=begin.elapsed_time(end)))
            return result
        return wrapped

    try:
        for name, function in originals.items():
            setattr(spec_dec_mod, name, wrap(name, function))
        yield events
    finally:
        for name, function in originals.items():
            setattr(spec_dec_mod, name, function)


def phase_totals(events):
    totals = {}
    for e in events:
        t = totals.setdefault(e["phase"], {"calls": 0, "wall_s": 0.0, "cuda_interval_ms": 0.0})
        t["calls"] += 1
        t["wall_s"] += e["wall_s"]
        t["cuda_interval_ms"] += e["cuda_interval_ms"]
    return totals


def build_prefill(events):
    """返回 spec_dec._prefill 的计时包装（与 spec_prefill_scope 同语义，但并入阶段账本）。"""
    from experiments.formal.worker import test_chunked_prefill

    def adapted_prefill(model, input_ids):
        role = "target" if model is target_ref[0] else "draft" if model is draft_ref[0] else "unknown"
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        next_logits, cache = test_chunked_prefill(model, input_ids, chunk=2048, diagnostic=False)
        end.record()
        end.synchronize()
        events.append(dict(phase="prefill_" + role, wall_s=time.perf_counter() - start,
                           cuda_interval_ms=begin.elapsed_time(end)))
        return next_logits.unsqueeze(1), cache

    return adapted_prefill


target_ref = [None]
draft_ref = [None]


def run_condition(role: str, family: str, ctx: int, out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.formal.runtime_info import collect_runtime_info
    from experiments.formal.runtime_audit import capture_boundary, assess_boundaries
    from specdec.engine import model_self, spec_dec, target_rope_scope

    out.mkdir(parents=True, exist_ok=False)
    progress = {"stage": "environment"}
    result = dict(schema_version="ot-bench-condition-v1",
                  role=role, family=family, context=ctx,
                  utc=datetime.now(timezone.utc).isoformat(),
                  frozen=config.FROZEN, execution_mode=config.execution_mode(),
                  flags=config.optimization_flags() if role == "opt" else None)
    installed = False
    try:
        environment = check_environment()
        environment["gpu"] = check_gpu_environment()
        result["environment"] = environment
        result["runtime"] = collect_runtime_info()

        progress["stage"] = "load_target"
        target, tokenizer, target_info = W.load_target()
        draft = None
        draft_info = None
        if role != "ar":
            progress["stage"] = "load_draft"
            draft, dtok, draft_info = W.load_draft()
            if dtok.eos_token_id != tokenizer.eos_token_id:
                raise RuntimeError("EOS mismatch")
        target_ref[0], draft_ref[0] = target, draft
        rope = W.configure_target(target)
        installed = True
        result["load"] = dict(target=target_info, draft=draft_info)
        result["rope"] = rope

        progress["stage"] = "load_input"
        ids, input_meta = W.load_input(family, ctx)
        mode = {"ar": "AR", "ref": "SpecDec", "opt": "SpecDec"}[role]
        job = Job(family, Condition(f"L{ctx}_{mode}_base_bf16", ctx, mode, "base_bf16"), 0, 1)
        result["input"] = input_meta
        result["configuration"] = W.check_runtime_config(target, draft, job)
        result["eos_id"] = tokenizer.eos_token_id

        # warm-up
        if role == "ar":
            W.run_once(target, None, tokenizer, ids, job)
        else:
            W.run_once(target, draft, tokenizer, ids, job)

        # measured：主计时（无阶段事件同步）
        reps = []
        progress["stage"] = "measured"
        for rep in range(REPS):
            monitor_start = capture_boundary()
            if role == "ar":
                record = W.run_once(target, None, tokenizer, ids, job)
            elif role == "ref":
                record = W.run_once(target, draft, tokenizer, ids, job)
            else:
                record = run_optimized_once(target, draft, tokenizer, ids)
            monitor_end = capture_boundary()
            record["runtime_environment_audit"] = assess_boundaries(monitor_start, monitor_end)
            record["rep"] = rep
            reps.append(record)
        result["reps"] = reps

        # 诊断计时：阶段账本（仅一次；输出必须与 measured 一致）
        progress["stage"] = "stage_ledger"
        if role == "ar":
            ledger = dict(calls=None, note="AR 无 propose/verify/repair 阶段")
        else:
            events = []
            originals = {n: getattr(spec_dec, n) for n in ("_prefill", "_propose", "_verify", "_step")}
            prefill_orig = spec_dec._prefill
            events_s = []
            try:
                model_self.reset_attention_stats()
                prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
                if role == "ref":
                    spec_dec._prefill = build_prefill(events)
                    with W.spec_generation_scope():
                        with phase_scope(spec_dec, target, draft, timed=True) as events_s:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            run_out = spec_dec.spec_decode_greedy(
                                target, draft, prompt, 64, k=4, eos_id=tokenizer.eos_token_id,
                                target_kv="base_bf16", draft_kv="base_bf16")
                            torch.cuda.synchronize()
                            total = time.perf_counter() - start
                    totals = phase_totals(events_s + events)
                else:
                    from experiments.execution_optimization.fast_spec_dec import (
                        spec_decode_greedy_optimized, StageTimer)
                    timer = StageTimer(enabled=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    run_out = spec_decode_greedy_optimized(
                        target, draft, prompt, 64, k=4, eos_id=tokenizer.eos_token_id,
                        timer=timer)
                    torch.cuda.synchronize()
                    total = time.perf_counter() - start
                    stage = run_out["ledger"]["stage_s"]
                    totals = dict(
                        prefill_target=dict(calls=None, wall_s=stage.get("prefill_target", 0.0),
                                            cuda_interval_ms=None),
                        prefill_draft=dict(calls=None, wall_s=stage.get("prefill_draft", 0.0),
                                           cuda_interval_ms=None),
                        draft_propose=dict(calls=None, wall_s=stage.get("propose", 0.0),
                                           cuda_interval_ms=None),
                        target_verify=dict(calls=None, wall_s=stage.get("verify", 0.0),
                                           cuda_interval_ms=None),
                        repair_target=dict(calls=None, wall_s=stage.get("repair", 0.0),
                                           cuda_interval_ms=None),
                    )
                model_self.assert_strategy_used()
                ledger = dict(
                    calls={k: v["calls"] for k, v in totals.items()},
                    wall_s={k: v["wall_s"] for k, v in totals.items()},
                    cuda_interval_ms={k: v["cuda_interval_ms"] for k, v in totals.items()},
                    total_s=total,
                    gen=run_out["gen"],
                )
            finally:
                spec_dec._prefill = prefill_orig
                for n, f in originals.items():
                    setattr(spec_dec, n, f)
        result["stage_ledger"] = ledger

        # 输出一致性：measured 全部相同且与账本 run 相同
        first = reps[0]["generated_ids"]
        result["outputs_consistent"] = all(r["generated_ids"] == first for r in reps)
        if ledger.get("gen") is not None:
            result["outputs_consistent"] = result["outputs_consistent"] and ledger["gen"] == first
        if not result["outputs_consistent"]:
            raise RuntimeError("同一条件的多次运行输出不一致")
        result["generated_ids"] = first
        result["generated_length"] = len(first)
        result["stop_reason"] = reps[0]["stop_reason"]
        if role != "ar":
            result["spec_stats"] = reps[0].get("spec_stats")
        result["attention_stats"] = reps[0].get("attention_stats")
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


def run_optimized_once(target, draft, tokenizer, token_ids):
    """与 experiments.formal.worker.run_spec_once 同口径，但调用优化执行器。

    计时：执行器内部用与 formal 相同的同步边界（prefill 前后、decode 后一次同步）分别测
    prefill 与 decode；decode_s = 解码循环的 wall，prefill_s = target+draft 分块 prefill 之和。
    """
    from specdec.engine import model_self, spec_dec
    from experiments.execution_optimization.fast_spec_dec import spec_decode_greedy_optimized

    if spec_dec.target_verify_mode() != "block":
        raise ValueError("优化执行器只支持 block verify 计时")
    if spec_dec.verify_wrap_state()["active"]:
        raise RuntimeError("不能使用诊断 verify wrapper 计时")
    device = next(target.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model_self.reset_attention_stats()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = spec_decode_greedy_optimized(
        target, draft, input_ids, 64, k=4, eos_id=tokenizer.eos_token_id,
        measure_timing=True)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model_self.assert_strategy_used()
    timing = out["timing"]
    generated = out["gen"]
    stop = "eos" if generated and generated[-1] == tokenizer.eos_token_id else "max_new_reached"
    return {
        "max_new": 64,
        "generated_ids": generated, "generated_length": len(generated),
        "stop_reason": stop,
        "prefill_s": timing["prefill_s"], "decode_s": timing["decode_s"],
        "total_s": timing["total_s"],
        "decode_tokens_per_s": len(generated) / timing["decode_s"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "diagnostic_checks_executed": False,
        "prefill_components": timing["prefill_components"],
        "spec_stats": out["stats"],
        "attention_stats": model_self.attention_stats(),
        "execution_ledger": out["ledger"],
    }


def main_child(role, family, ctx, out):
    run_condition(role, family, ctx, out)


def wait_for_memory(min_gib: float = 17.3, timeout_s: int = 3600) -> float:
    """等待自然恢复（不放松门槛、不改参数）：commit 可用量 ≥ min_gib 才轮到下一个条件。"""
    from specdec.engine.g5_8k_spec_variance_probe import system_memory
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        m = system_memory()
        if m.get("available"):
            free = m["commit_free_mib"] / 1024.0
            last = free
            if free >= min_gib:
                return free
        time.sleep(20)
    raise RuntimeError(f"等待内存恢复超时（{timeout_s}s，最后 commit free={last} GiB）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--role", choices=("ar", "ref", "opt"))
    parser.add_argument("--family")
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--families", default=None,
                        help="逗号分隔；默认 4 个开发 family")
    parser.add_argument("--contexts", default=None,
                        help="逗号分隔；默认 2048,4096,8192")
    args = parser.parse_args()

    if args.role:
        run_condition(args.role, args.family, args.ctx, args.out)
        return 0

    families = tuple(args.families.split(",")) if args.families else FAMILIES
    contexts = tuple(int(x) for x in args.contexts.split(",")) if args.contexts else CONTEXTS
    run_dir = args.out or (ROOT / "output" / "specdec_execution_optimization" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = [(role, fam, ctx) for fam in families for ctx in contexts for role in ("ar", "ref", "opt")]
    manifest = dict(schema_version="ot-bench-manifest-v1",
                    utc=datetime.now(timezone.utc).isoformat(),
                    families=list(families), contexts=list(contexts),
                    roles=["ar", "ref", "opt"], reps=REPS, warmup=WARMUP,
                    conditions=[dict(role=r, family=f, context=c,
                                     dir=str(run_dir / f"{f}_{c}_{r}")) for r, f, c in conditions],
                    flags=config.optimization_flags(),
                    notes=["主计时与正式口径一致：decode_s = total - prefill，无阶段事件同步",
                           "阶段账本来自每条件额外 1 次诊断运行（CUDA Event），不进主比例",
                           "AR 为 target-only 独立进程"])
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)
    for i, (role, family, ctx) in enumerate(conditions, 1):
        out = run_dir / f"{family}_{ctx}_{role}"
        if (out / "result.json").exists():
            print(f"[{i}/{len(conditions)}] {family} ctx={ctx} role={role} 已完成，跳过", flush=True)
            continue
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--role", role, "--family", family, "--ctx", str(ctx), "--out", str(out)]
        log_path = run_dir / f"{family}_{ctx}_{role}.log"
        print(f"[{i}/{len(conditions)}] {family} ctx={ctx} role={role}", flush=True)
        attempts = 0
        while True:
            attempts += 1
            free = wait_for_memory()
            print(f"    commit free={free:.2f} GiB", flush=True)
            # 环境拒绝后的重试：清除上一次未完成的输出目录（其中只有 failure.json）
            if out.exists():
                result_file = out / "result.json"
                if result_file.exists():
                    raise RuntimeError(f"{out} 已有 result.json，拒绝覆盖")
                import shutil
                shutil.rmtree(out)
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
                break
            failure = None
            fpath = out / "failure.json"
            if fpath.exists():
                failure = json.loads(fpath.read_text(encoding="utf-8"))
            if failure and "EnvironmentRefused" in str(failure.get("error")):
                # 环境拒绝：保留证据，等待自然恢复后重试（不放宽门槛、不改参数）
                print(f"    ENV_REFUSED (attempt {attempts}): {failure.get('error')}",
                      flush=True)
                if attempts >= 8:
                    raise RuntimeError(f"{family}_{ctx}_{role} env refused 8x; see {log_path}")
                time.sleep(60)
                continue
            raise RuntimeError(f"{family}_{ctx}_{role} failed rc={rc}; see {log_path}")
        time.sleep(COOLDOWN_S)
    print("COMPLETE:", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
