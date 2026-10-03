"""row_aligned_stage2.py — **Stage 2/3** 运行器：在既有 8K smoke harness 之上加一层"候选开关"。

## 为什么这样做
Stage 2/3 必须复用**既有**案例与规则（manifest family、token hash、ctx、chunk、k、max_new、EOS、
token 接受规则、停止条件），**不得**另造一套。仓库里实现这些规则的是 `g5_8k_smoke_cudnn`
（AR/Spec 两条生成路径 + 分块 prefill + 结构/有限性判据 + 逐轮记账）。因此本运行器**只做三件事**：

1. 由 `RA2_*` 环境变量推导出该 harness 需要的 `SMOKE_*` 环境（**在 import harness 之前**设置）；
2. `RA2_MODE=aligned` 时安装 `spec_dec.set_verify_wrap(row_aligned_verify)` —— 候选模式
   **只**包住 target block verify（AR/prefill 不经过该挂载点）；其它模式不安装；
3. 跑完把该次运行的**生成 token IDs 先落盘**（`.pt`），再写本进程的摘要 JSON（新 tag）。

配置逐项沿用 F13 已知失败记录的规则：ctx=8192 / chunk=2048 / k=4 / max_new=64 / base_bf16 /
arm=eager / target RoPE=cpu_fp32 / `SMOKE_BF16_RED_REDUCTION=1` / `SMOKE_STRUCT_CHECK=1`。
**不改** token 接受公式、**不改**停止条件、**不改** draft、**不共享 cache、不自动重试**。

## 用法
```powershell
# 一个进程 = 一个 (case, mode, rep)
$env:RA2_CASE="F13"; $env:RA2_MODE="aligned"; $env:RA2_REP="1"
python -X utf8 -m specdec.engine.row_aligned_stage2            # 跑一次生成
python -X utf8 -m specdec.engine.row_aligned_stage2 --aggregate # 只读汇总（不加载模型）
```
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("RA2_TAG", "div8209_ra2")
CASE = os.environ.get("RA2_CASE", "F13").strip()
MODE_RAW = os.environ.get("RA2_MODE", "ar").strip().lower()
# 协议别名（2026-10-03 冻结）：`prod` = production（原 `spec`）；`diag` = diagnostic oracle（原 `aligned`）。
# 旧名保持不变 ⇒ 既有产物与命令逐字可用；本改动只是**新增别名与字段**，不改任何数值路径。
ALIAS = {"prod": "spec", "diag": "aligned"}
MODE = ALIAS.get(MODE_RAW, MODE_RAW)
REP = int(os.environ.get("RA2_REP", "1"))
MAXNEW = int(os.environ.get("RA2_MAXNEW", "64"))
CTX = int(os.environ.get("RA2_CTX", "8192"))
CHUNK = int(os.environ.get("RA2_CHUNK", "2048"))
K = int(os.environ.get("RA2_K", "4"))
KV = os.environ.get("RA2_KV", "base_bf16")
ARM = os.environ.get("RA2_ARM", "eager")
TARGET_ROPE = os.environ.get("RA2_TARGET_ROPE", "cpu_fp32")
MODES = ("ar", "spec", "aligned")


def _smoke_env() -> dict:
    """由 `RA2_*` 推导该 harness 需要的 `SMOKE_*`（**必须在 import harness 之前**生效）。"""
    return dict(
        SMOKE_MODE=("ar" if MODE == "ar" else "spec"),
        SMOKE_ARM=ARM, SMOKE_FAMILY=CASE, SMOKE_CTX=str(CTX), SMOKE_CHUNK=str(CHUNK),
        SMOKE_K=str(K), SMOKE_MAXNEW=str(MAXNEW), SMOKE_KV=KV,
        SMOKE_TARGET_ROPE=TARGET_ROPE, SMOKE_BF16_RED_REDUCTION="1", SMOKE_STRUCT_CHECK="1",
        SMOKE_PREFILL_DIGEST="1")


def run_one() -> int:
    assert MODE in MODES, f"未知模式：{MODE}"
    from .. import results_schema as S
    from . import spec_dec
    from . import row_aligned as RA

    env = _smoke_env()
    os.environ.update(env)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(tool="row_aligned_stage2", tag=TAG, run_id=run_id, case=CASE, mode=MODE,
                     mode_requested=MODE_RAW, rep=REP,
                     diagnostic_only=bool(MODE == "aligned"),
                     smoke_env=env, stage="start",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     rules="既有 F13 记录规则：ctx=8192/chunk=2048/k=4/max_new=64/base_bf16/eager/"
                           "target_rope=cpu_fp32/bf16_red_red=1/struct_check=1；接受公式与停止条件未改",
                     protocol="docs/correctness_protocol_post_shape_localization_2026-10-03.md")
    # family provenance（manifest 已登记记录；input token 由 harness 的 family_input 重建并校验 sha1）
    try:
        from . import family_input as FI
        rec["provenance"] = FI.family_record(CASE)
    except Exception as e:                                                # noqa: BLE001
        rec["provenance"] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
    if MODE == "aligned":
        rec["verify_wrap_install"] = spec_dec.set_verify_wrap(RA.row_aligned_verify)
        RA.clear_verify_log()
    else:
        rec["verify_wrap_install"] = dict(installed=False, had_previous=False,
                                          note="非对齐模式 ⇒ **不安装**候选（AR/原始 Spec 路径不受影响）")
    rec["verify_wrap_before_run"] = spec_dec.verify_wrap_state()
    fallback_pt = OUTDIR / f"{TAG}_{CASE}_{MODE}_rep{REP}_{run_id}.pt"
    try:
        from . import g5_8k_smoke_cudnn as H          # ← import 即读取上面的 SMOKE_* 环境
        rec["smoke_config"] = dict(ctx=H.CTX, chunk=H.CHUNK, k=H.K, max_new=H.MAXNEW, kv=H.KV,
                                   arm=H.ARM, mode=H.MODE, family=CASE, target_rope=H.TARGET_ROPE)
        rc = H.main()
        rec["smoke_rc"] = int(rc)
        # 找到该 harness 刚写下的产物（按 mtime 取最新，并把路径记进本记录）
        cands = sorted(glob.glob(str(OUTDIR / "smoke8k_*.json")), key=os.path.getmtime)
        smoke_path = Path(cands[-1]) if cands else None
        rec["smoke_artifact"] = str(smoke_path)
        if smoke_path is not None:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            rec["smoke"] = dict(
                stage=smoke.get("stage"), run_id=smoke.get("run_id"),
                gen=list((smoke.get("run") or {}).get("gen") or []),
                gen_len=(smoke.get("run") or {}).get("gen_len"),
                stop_reason=(smoke.get("run") or {}).get("stop_reason"),
                gen_ids_in_vocab=(smoke.get("run") or {}).get("gen_ids_in_vocab"),
                prefill_logits_finite=(smoke.get("run") or {}).get("prefill_logits_finite"),
                specdec_stats=(smoke.get("run") or {}).get("specdec_stats"),
                wall_s=(smoke.get("run") or {}).get("wall_s"),
                prefill_s=(smoke.get("run") or {}).get("prefill_s"),
                end_state=smoke.get("end_state"),
                checks_passed=((smoke.get("checks") or {}).get("passed")),
                decision_coverage=(smoke.get("decision_coverage") or {}).get("ok"),
                input=(smoke.get("input") or {}),
                mem_timeline_last=(smoke.get("mem_timeline") or [{}])[-1],
                n_rounds=len(smoke.get("rounds") or []),
                rounds=list(smoke.get("rounds") or []),
                pstate=smoke.get("pstate"),
                gpu_consumers_before=smoke.get("gpu_consumers_before"),
                resident_models=smoke.get("resident_models"),
                weights_bitmatch=smoke.get("weights_bitmatch"),
                kv_format=smoke.get("kv_format"),
                attention_stats=(smoke.get("run") or {}).get("attention_stats"))
    except Exception as e:                                            # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        rc = 2
    finally:
        try:
            from . import model_self as M
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            from . import spec_dec as _sd
            _sd.set_verify_wrap(None)
        except Exception:                                             # noqa: BLE001
            pass
    rec["verify_wrap_after_run"] = None
    try:
        from . import spec_dec as _sd
        rec["verify_wrap_after_run"] = _sd.verify_wrap_state()
    except Exception:                                                 # noqa: BLE001
        pass
    if MODE == "aligned":
        from . import row_aligned as RA
        log = RA.verify_log()
        tbc = int((((rec.get("smoke") or {}).get("specdec_stats") or {}).get("target_block_calls") or 0))
        rec["candidate"] = dict(n_verify_calls_logged=len(log), target_block_calls=tbc,
                                counts_match=bool(len(log) == tbc),
                                per_call=log,
                                qk_applied_per_call=sorted({e.get("qk_applied") for e in log}),
                                lin_applied_per_call=sorted({e.get("lin_applied") for e in log}))
    else:
        from . import row_aligned as RA
        rec["candidate"] = dict(n_verify_calls_logged=len(RA.verify_log()), installed=False)
    rec["stage"] = "complete" if rc == 0 else "smoke_rc_nonzero"
    # ★ 生成 IDs 先落盘（控制 18），再写摘要
    torch.save(dict(gen=(rec.get("smoke") or {}).get("gen"),
                    case=CASE, mode=MODE, rep=REP,
                    meta=dict(run_id=run_id, smoke_artifact=rec.get("smoke_artifact"),
                              stop_reason=((rec.get("smoke") or {}).get("stop_reason")),
                              gen_len=((rec.get("smoke") or {}).get("gen_len")))),
               str(fallback_pt))
    rec["artifact_pt"] = str(fallback_pt)
    p = OUTDIR / f"{TAG}_{CASE}_{MODE}_rep{REP}_{run_id}.json"
    rec["artifact_json"] = str(p)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    g = (rec.get("smoke") or {}).get("gen") or []
    print(f"[ra2] case={CASE} mode={MODE} rep={REP} rc={rc} gen_len={len(g)} "
          f"stop={((rec.get('smoke') or {}).get('stop_reason'))} -> {p.name}", flush=True)
    return int(rc)


# ------------------------------------------------------------------ 离线汇总（CPU、不加载模型）
def _load_runs() -> list:
    out = []
    for f in sorted(glob.glob(str(OUTDIR / f"{TAG}_*_rep*_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("tool") == "row_aligned_stage2":
            d["_file"] = f
            out.append(d)
    return out


def aggregate() -> int:
    from .. import results_schema as S
    runs = _load_runs()
    cases = sorted({r["case"] for r in runs})
    by = {}
    for r in runs:
        by.setdefault(r["case"], {}).setdefault(r["mode"], []).append(r)
    summary: dict = dict(tool="row_aligned_stage2_summary", tag=TAG,
                         run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                         git=S.git_meta(ROOT), cases=cases,
                         rules="既有 F13/F06 manifest 窗口与既有生成规则；token 接受公式/停止条件未改",
                         runs=[dict(file=Path(r["_file"]).name, case=r["case"], mode=r["mode"], rep=r["rep"],
                                    gen=list((r.get("smoke") or {}).get("gen") or []),
                                    gen_len=((r.get("smoke") or {}).get("gen_len")),
                                    stop_reason=((r.get("smoke") or {}).get("stop_reason")),
                                    gen_ids_in_vocab=((r.get("smoke") or {}).get("gen_ids_in_vocab")),
                                    prefill_finite=((r.get("smoke") or {}).get("prefill_logits_finite")),
                                    checks_passed=((r.get("smoke") or {}).get("checks_passed")),
                                    struct_ok=((r.get("smoke") or {}).get("decision_coverage")),
                                    wall_s=((r.get("smoke") or {}).get("wall_s")),
                                    prefill_s=((r.get("smoke") or {}).get("prefill_s")),
                                    stats=((r.get("smoke") or {}).get("specdec_stats")),
                                    n_rounds=((r.get("smoke") or {}).get("n_rounds")),
                                    peak_alloc_mib=((r.get("smoke") or {}).get("mem_timeline_last") or {}
                                                    ).get("pt_peak_allocated_mib"),
                                    peak_res_mib=((r.get("smoke") or {}).get("mem_timeline_last") or {}
                                                  ).get("pt_peak_reserved_mib"),
                                    candidate=r.get("candidate"), smoke_artifact=r.get("smoke_artifact"))
                               for r in runs],
                         note="每个 (case, mode, rep) 一个独立进程；进程内不共享 cache")
    checks: dict = {}
    per_case: dict = {}
    for c in cases:
        d = by.get(c, {})
        ar = d.get("ar", [])
        sp = d.get("spec", [])
        al = d.get("aligned", [])
        gens = lambda lst: [list((r.get("smoke") or {}).get("gen") or []) for r in lst]
        ar_g, sp_g, al_g = gens(ar), gens(sp), gens(al)
        pc = dict(n_reps=dict(ar=len(ar), spec=len(sp), aligned=len(al)),
                  ar_gen_len=[len(g) for g in ar_g], spec_gen_len=[len(g) for g in sp_g],
                  aligned_gen_len=[len(g) for g in al_g],
                  reps_consistent=dict(
                      ar=bool(len(set(map(tuple, ar_g))) == 1), spec=bool(len(set(map(tuple, sp_g))) == 1),
                      aligned=bool(len(set(map(tuple, al_g))) == 1)),
                  stop_reasons=dict(ar=[((r.get("smoke") or {}).get("stop_reason")) for r in ar],
                                    spec=[((r.get("smoke") or {}).get("stop_reason")) for r in sp],
                                    aligned=[((r.get("smoke") or {}).get("stop_reason")) for r in al]),
                  no_other_short=bool(all(((r.get("smoke") or {}).get("stop_reason")) != "other_short"
                                          for r in ar + sp + al)),
                  all_finite=bool(all(((r.get("smoke") or {}).get("prefill_logits_finite")) is not False
                                      and ((r.get("smoke") or {}).get("gen_ids_in_vocab")) is not False
                                      for r in ar + sp + al)),
                  struct_invariants_ok=bool(all(((r.get("smoke") or {}).get("decision_coverage")) is not False
                                                for r in sp + al)),
                  smoke_checks_passed=bool(all(((r.get("smoke") or {}).get("checks_passed")) is not False
                                               for r in ar + sp + al)),
                  oom=None)
        # token equality：候选 vs AR（**硬门**）；原始 Spec 只作对照
        pc["original_spec_fails"] = bool(sp_g and ar_g and sp_g[0] != ar_g[0])
        pc["candidate_equals_ar"] = bool(al_g and ar_g and all(g == ar_g[0] for g in al_g))
        pc["first_divergence_original_spec"] = next(
            (i for i, (a, b) in enumerate(zip(sp_g[0], ar_g[0])) if a != b), None) if (sp_g and ar_g) else None
        pc["first_divergence_candidate"] = next(
            (i for i, (a, b) in enumerate(zip(al_g[0], ar_g[0])) if a != b), None) if (al_g and ar_g) else None
        pc["candidate_applied"] = [r.get("candidate") for r in al]
        pc["candidate_counts_match_block_calls"] = bool(
            al and all((r.get("candidate") or {}).get("counts_match") is True for r in al))
        per_case[c] = pc
        checks[f"case_{c}_candidate_token_equality"] = bool(pc["candidate_equals_ar"])
        checks[f"case_{c}_reps_consistent"] = bool(all(pc["reps_consistent"].values()))
        checks[f"case_{c}_no_other_short"] = pc["no_other_short"]
        checks[f"case_{c}_finite"] = pc["all_finite"]
        checks[f"case_{c}_struct"] = pc["struct_invariants_ok"]
        checks[f"case_{c}_candidate_counts"] = pc["candidate_counts_match_block_calls"]
    summary["per_case"] = per_case
    summary["checks"] = checks
    summary["stage2_hard_gate"] = ("PASS" if (checks and all(checks.values())) else "FAIL")
    # ★ 先落盘（含全部生成 IDs），再打印
    sp_pt = OUTDIR / f"{TAG}_stage2_summary_{summary['run_utc']}.pt"
    torch.save({f"{r['case']}|{r['mode']}|rep{r['rep']}": r.get("gen") for r in summary["runs"]}, str(sp_pt))
    sp = OUTDIR / f"{TAG}_stage2_summary_{summary['run_utc']}.json"
    summary["artifact_pt"] = str(sp_pt)
    summary["artifact_json"] = str(sp)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("== Stage 2 汇总 ==")
    for c, pc in per_case.items():
        print(f"  [{c}] reps={pc['n_reps']}；gen_len AR={pc['ar_gen_len']} Spec={pc['spec_gen_len']} "
              f"Aligned={pc['aligned_gen_len']}")
        print(f"        原始 Spec 失败={pc['original_spec_fails']}（首分歧 index="
              f"{pc['first_divergence_original_spec']}）；**候选 == AR：{pc['candidate_equals_ar']}**"
              f"（首分歧 index={pc['first_divergence_candidate']}）")
        print(f"        两次重复一致={pc['reps_consistent']}；stop={pc['stop_reasons']}")
        print(f"        候选 applied 计数与 block calls 一致={pc['candidate_counts_match_block_calls']}")
    for k, v in checks.items():
        print(f"   [check] {k}: {v}")
    print(f"  ★ Stage 2 硬门 = {summary['stage2_hard_gate']}")
    print(f"\n[ra2] RESULT_JSON={sp}")
    print(f"[ra2] RESULT_PT={sp_pt}")
    return 0 if summary["stage2_hard_gate"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate()
    return run_one()


if __name__ == "__main__":
    raise SystemExit(main())
