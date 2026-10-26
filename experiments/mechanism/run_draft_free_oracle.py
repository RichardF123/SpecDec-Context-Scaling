"""Draft-Free Perfect-Accept Bonus Oracle 运行器（smoke / full 4×3 诊断矩阵）。

模式：ar（target-only 基线）、bonus（perfect-accept + bonus oracle，同批对照）、
draft_free（保留 target verify/bonus/cache/控制，计时区间内 0 次 draft forward）。
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
MODES = ("ar", "bonus", "draft_free")


def _mem():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated_gib=torch.cuda.memory_allocated() / 2**30,
                reserved_gib=torch.cuda.memory_reserved() / 2**30,
                free_gib=free / 2**30, total_gib=total / 2**30)


def run_child(mode: str, family: str, ctx: int, out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.formal.worker import test_chunked_prefill
    from experiments.relaxed_specdec import bonus_accept_oracle as B
    from experiments.mechanism import draft_free_oracle as DFO
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=True)
    env = check_environment(16)
    env["gpu"] = check_gpu_environment()
    target, tokenizer, tinfo = W.load_target()
    draft, dtok, dinfo = None, None, None
    draft_resident = False
    draft_note = None
    if mode != "ar":
        try:
            draft, dtok, dinfo = W.load_draft()
            if dtok.eos_token_id != tokenizer.eos_token_id:
                raise RuntimeError("EOS mismatch")
            draft_resident = True
        except torch.OutOfMemoryError as exc:   # 保留证据，不静默卸载
            draft_note = f"draft load OOM: {exc}"
            torch.cuda.empty_cache()
    rope = W.configure_target(target)
    installed = True
    result = dict(schema_version="draft-free-oracle-v1",
                  experiment_class="mechanism_diagnostic",
                  formal_result_allowed=False, quality_claim_allowed=False,
                  deployable=False, scope="development diagnostic (F09/F03/F11/F04 × 2K/4K/8K)",
                  mode=mode, family=family, context=ctx,
                  utc=datetime.now(timezone.utc).isoformat(),
                  environment=env, input_meta=None,
                  load=dict(target=tinfo, draft=dinfo), draft_resident=draft_resident,
                  draft_note=draft_note)
    try:
        ids, meta = W.load_input(family, ctx)
        result["input_meta"] = meta
        prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
        # 保留双模型显存环境：bonus 模式 draft 必须真实 prefill（不计时）
        if mode == "bonus" and draft is not None:
            test_chunked_prefill(draft, prompt, chunk=2048, diagnostic=False)
        # draft 不在场时（8K target-only lower bound）按 AR 配置校验；语义不变
        job_mode = "AR" if (mode == "ar" or draft is None) else "SpecDec"
        job = Job(family, Condition(f"L{ctx}_{mode}", ctx, job_mode, "base_bf16"), 0, 1)
        result["target_only_lower_bound"] = bool(mode == "draft_free" and draft is None)
        result["configuration"] = W.check_runtime_config(target, draft, job)

        if mode == "ar":
            W.run_ar_once(target, tokenizer, ids, job)
            reps = []
            for rep in range(REPS):
                r = W.run_ar_once(target, tokenizer, ids, job)
                reps.append(dict(rep=rep, prefill_s=r["prefill_s"], decode_s=r["decode_s"],
                                 total_s=r["total_s"], gen=r["generated_ids"]))
            result["reps"] = reps
            result["generated_ids"] = reps[0]["gen"]
        elif mode == "bonus":
            if draft is None:
                raise RuntimeError("bonus 模式需要 draft")
            B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=tokenizer.eos_token_id, mode="carry")
            reps = []
            for rep in range(REPS):
                model_self.reset_attention_stats()
                o = B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=tokenizer.eos_token_id,
                             mode="carry", measure_timing=True, capture_cache=True)
                model_self.assert_strategy_used()
                reps.append(dict(rep=rep, prefill_s=o["timing"]["prefill_s"],
                                 decode_s=o["timing"]["decode_s"],
                                 total_s=o["timing"]["total_s"], gen=o["gen"],
                                 stats=o["stats"], final_cache=o["final_cache"],
                                 mem=_mem()))
            result["reps"] = reps
            result["generated_ids"] = reps[0]["gen"]
        else:
            # draft 保持 resident + 真实 prefill；若 8K 等导致 OOM：记录并显式降级为
            # target-only lower bound（draft_resident=False），不静默卸载、不改上下文/长度
            try:
                if draft is not None:
                    test_chunked_prefill(draft, prompt, chunk=2048, diagnostic=False)
                import hashlib
                frozen = None
                if ctx >= 8192:
                    ar_path = out.parent / f"{family}_{ctx}_ar" / "result.json"
                    if ar_path.exists():
                        ar_ref = json.loads(ar_path.read_text(encoding="utf-8"))
                        gen = [int(x) for x in ar_ref["generated_ids"]]
                        frozen = dict(gen=gen,
                                      sha1=hashlib.sha1(bytes(str(gen), "utf-8")).hexdigest(),
                                      source=f"ar_result_file:{ar_path.name}")
                if frozen is None:
                    ar_freeze_job = Job(family, Condition(f"L{ctx}_arfreeze", ctx, "AR", "base_bf16"), 0, 1)
                    frozen = DFO.freeze_via_ar(target, tokenizer, ids, ar_freeze_job)
                    frozen["source"] = "ar_freeze_run"
                torch.cuda.empty_cache()
                DFO.decode_draft_free(target, draft, prompt, frozen["gen"], k=K,
                                      eos_id=tokenizer.eos_token_id)
            except torch.OutOfMemoryError as exc:
                draft_note = f"draft_resident path OOM: {repr(exc)[:200]}"
                result["draft_note"] = draft_note
                result["draft_resident"] = False
                draft_resident = False
                if draft is not None:
                    del draft
                    draft = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                ar_freeze_job = Job(family, Condition(f"L{ctx}_arfreeze", ctx, "AR", "base_bf16"), 0, 1)
                frozen = DFO.freeze_via_ar(target, tokenizer, ids, ar_freeze_job)
                torch.cuda.empty_cache()
                DFO.decode_draft_free(target, None, prompt, frozen["gen"], k=K,
                                      eos_id=tokenizer.eos_token_id)
            layout = DFO.candidate_layout(frozen["gen"], K, MAX_NEW)
            result["frozen"] = frozen
            result["layout"] = layout
            # warmup
            DFO.decode_draft_free(target, draft, prompt, frozen["gen"], k=K,
                                  eos_id=tokenizer.eos_token_id)
            reps = []
            for rep in range(REPS):
                guard = DFO.DraftCallGuard(draft) if draft is not None else None
                with guard if guard is not None else _null():
                    o = DFO.decode_draft_free(target, draft, prompt, frozen["gen"], k=K,
                                              eos_id=tokenizer.eos_token_id,
                                              measure_timing=True, capture_cache=True,
                                              finite_check=True)
                calls = guard.calls if guard is not None else 0
                if calls != 0:
                    raise RuntimeError(f"draft forward 进入计时区间：calls={calls}")
                reps.append(dict(rep=rep, prefill_s=o["timing"]["prefill_s"],
                                 decode_s=o["timing"]["decode_s"],
                                 total_s=o["timing"]["total_s"], gen=o["gen"],
                                 stats=o["stats"], control=o["control"],
                                 final_cache=o["final_cache"], draft_forward_calls=calls,
                                 mem=_mem()))
            result["reps"] = reps
            result["generated_ids"] = reps[0]["gen"]
            # 结构门
            g0 = reps[0]
            checks = dict(
                gen_equals_frozen=g0["gen"] == frozen["gen"],
                rounds_ok=g0["stats"]["rounds"] == 13,
                candidates_ok=g0["stats"]["candidate_tokens"] == 52,
                bonus_ok=g0["stats"]["bonus_tokens"] == 12,
                len_ok=len(g0["gen"]) == 64,
                draft_calls_zero=g0["draft_forward_calls"] == 0,
                cache_len_ok=(g0["final_cache"]["tgt_len"] == ctx + len(g0["gen"])),
                cache_consistent=g0["final_cache"]["tgt_consistent"],
                bonus_argmax_vs_frozen_mismatch=g0["stats"]["bonus_mismatch_rounds"],
                repeat_consistent=all(r["gen"] == g0["gen"] for r in reps),
                target_block_calls=g0["stats"]["target_block_calls"],
                verification_rounds=g0["stats"]["rounds"],
            )
            result["structure_checks"] = checks
            gate = (checks["gen_equals_frozen"] and checks["rounds_ok"] and checks["candidates_ok"]
                    and checks["bonus_ok"] and checks["len_ok"] and checks["draft_calls_zero"]
                    and checks["cache_len_ok"] and checks["cache_consistent"]
                    and checks["repeat_consistent"])
            result["gate_ok"] = bool(gate)
            if not gate:
                (out / "failure.json").write_text(json.dumps(
                    dict(error="DRAFT_FREE_STRUCTURE_GATE_FAILED", checks=checks,
                         family=family, context=ctx), ensure_ascii=False, indent=2),
                    encoding="utf-8")
                raise RuntimeError(f"DRAFT_FREE_STRUCTURE_GATE failed: {checks}")
        result["status"] = "ok"
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
        return result
    except Exception as exc:  # noqa: BLE001
        import traceback
        (out / "failure.json").write_text(json.dumps(dict(
            mode=mode, family=family, context=ctx, error=repr(exc),
            traceback=traceback.format_exc()), ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if installed:
            target_rope_scope.restore()


class _null:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stage", choices=("smoke", "full"))
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--family")
    ap.add_argument("--ctx", type=int)
    args = ap.parse_args()
    if args.mode:
        run_child(args.mode, args.family, args.ctx, args.out)
        return 0
    stage = args.stage or "smoke"
    run_dir = args.out or (ROOT / "output" / "draft_free_oracle" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "smoke":
        combos = [("F09", 2048, m) for m in MODES]
    else:
        combos = [(f, c, m) for f in FAMILIES for c in CONTEXTS for m in MODES]
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        schema_version="draft-free-oracle-manifest-v1",
        experiment_class="mechanism_diagnostic", deployable=False,
        stage=stage, family_scope=("F09 only" if stage == "smoke" else list(FAMILIES)),
        contexts=list(CONTEXTS), k=K, max_new=MAX_NEW, reps=REPS,
        modes=list(MODES),
        combos=[dict(family=f, context=c, mode=m, dir=str(run_dir / f"{f}_{c}_{m}"))
                for f, c, m in combos]), ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)
    for i, (family, ctx, mode) in enumerate(combos, 1):
        out = run_dir / f"{family}_{ctx}_{mode}"
        if (out / "result.json").exists():
            print(f"[{i}/{len(combos)}] {family}_{ctx}_{mode} 已完成，跳过", flush=True)
            continue
        free = wait_for_memory(min_gib=17.2)
        print(f"[{i}/{len(combos)}] {family}_{ctx}_{mode} commit free={free:.2f} GiB", flush=True)
        log = run_dir / f"{family}_{ctx}_{mode}.log"
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--mode", mode, "--family", family, "--ctx", str(ctx), "--out", str(out)]
        attempts = 0
        while True:
            attempts += 1
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"--- attempt {attempts} ---\n")
                with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        handle.write(line)
                        handle.flush()
                    rc = proc.wait()
            if rc == 0 or (out / "result.json").exists():
                break
            failure = None
            if (out / "failure.json").exists():
                failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            if failure and "EnvironmentRefused" in str(failure.get("error")):
                print(f"    ENV_REFUSED (attempt {attempts})；等待自然恢复", flush=True)
                if attempts >= 8:
                    raise RuntimeError(f"{family}_{ctx}_{mode} env refused 8x")
                time.sleep(60)
                continue
            raise RuntimeError(f"{family}_{ctx}_{mode} failed rc={rc}; see {log}")
        time.sleep(3)
    print("COMPLETE:", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
