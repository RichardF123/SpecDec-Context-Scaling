"""serial_oracle_gpu.py — **串行 target verification oracle** 的 GPU 运行器（Stage 1/2）。

## 它只做四件事（其余全部复用既有 harness）
1. 由 `SO_*` 环境变量推导该 harness 需要的 `SMOKE_*`（**必须在 import harness 之前**设置）；
2. 在 harness 之前把 **阶段感知的调用几何审计**装到 `QwenLM._run` 上（见 `serial_oracle.py`），
   `SO_MODE=serial|ar` 时**强制**非 `prefill` 阶段必须 `seq_len==1`（违反即抛错，不降级），
   `SO_MODE=prod` 时只**记录**（用于取得生产 block 路径的几何作对照）；
3. `SO_MODE=serial` 时 `spec_dec.set_target_verify_mode("serial")`（**不**安装 row-aligned wrapper）；
4. 跑完先落盘生成 token IDs（`.pt`），再写本进程摘要 JSON（新 tag），并核对**记账恒等式**。

## 冻结配置（与 F13 已知失败记录、24 轮协议逐项一致）
`ctx=8192 / chunk=2048 / k=4 / max_new=64 / base_bf16 / arm=eager / target_rope=cpu_fp32 /
SMOKE_BF16_RED_REDUCTION=1 / SMOKE_STRUCT_CHECK=1`；
**不改**接受规则、**不改**停止条件、**不改** draft、**不共享 cache、不自动重试**、每进程一次运行。

## 记账恒等式（非空泛的证据；`target_calls` = 审计器实测的 target generation 调用数）
* `ar`    ：`target_calls == stats.decode_steps`
* `serial`：`target_calls == stats.proposed + stats.rollbacks`（每轮 kp 次逐 token 校验 + 每次回滚一次修复步）
* `prod`  ：`target_calls == stats.target_block_calls + stats.rollbacks`（每轮一次 block + 每次回滚一次修复步）
任一等式不成立 ⇒ 审计器没覆盖住全部调用 ⇒ 本次运行的几何结论**无效**（运行器会明确标出）。

## 用法
```powershell
$env:SO_CASE="F13"; $env:SO_MODE="serial"; $env:SO_REP="1"
python -X utf8 -m specdec.engine.serial_oracle_gpu          # 一个进程 = 一个 (case, mode, rep)
```
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("SO_TAG", "div8209_so")
CASE = os.environ.get("SO_CASE", "F13").strip()
MODE = os.environ.get("SO_MODE", "serial").strip().lower()
REP = int(os.environ.get("SO_REP", "1"))
MAXNEW = int(os.environ.get("SO_MAXNEW", "64"))
CTX = int(os.environ.get("SO_CTX", "8192"))
CHUNK = os.environ.get("SO_CHUNK", "2048")
K = int(os.environ.get("SO_K", "4"))
KV = os.environ.get("SO_KV", "base_bf16")
ARM = os.environ.get("SO_ARM", "eager")
TARGET_ROPE = os.environ.get("SO_TARGET_ROPE", "cpu_fp32")
MODES = ("ar", "prod", "serial")
EXPECTED_INSTANCES = dict(ar=1, prod=2, serial=2)


def _smoke_env() -> dict:
    """由 `SO_*` 推导 `SMOKE_*`（**必须在 import harness 之前**生效）。"""
    return dict(
        SMOKE_MODE=("ar" if MODE == "ar" else "spec"),
        SMOKE_ARM=ARM, SMOKE_FAMILY=CASE, SMOKE_CTX=str(CTX), SMOKE_CHUNK=str(CHUNK),
        SMOKE_K=str(K), SMOKE_MAXNEW=str(MAXNEW), SMOKE_KV=KV,
        SMOKE_TARGET_ROPE=TARGET_ROPE, SMOKE_BF16_RED_REDUCTION="1", SMOKE_STRUCT_CHECK="1",
        SMOKE_PREFILL_DIGEST="1")


def _accounting(mode: str, stats: dict, end_state: dict, trun: dict) -> dict:
    """核对"审计器实测的 target generation 调用数"与解码器记账的恒等式。"""
    tc = (trun.get("target") or {}).get("n_generation_calls")
    if mode == "ar":
        exp = int((stats or {}).get("decode_steps") or 0)
        why = "AR：每个解码步恰好一次 s=1 的 target 前向"
    elif mode == "serial":
        exp = int((stats or {}).get("proposed") or 0) + int((stats or {}).get("rollbacks") or 0)
        why = "serial：每轮 kp 次逐 token 校验 + 每次回滚一次修复步（`_step`）"
    else:
        exp = int((stats or {}).get("target_block_calls") or 0) + int((stats or {}).get("rollbacks") or 0)
        why = "prod：每轮一次 block 校验 + 每次回滚一次修复步（`_step`）"
    return dict(mode=mode, target_generation_calls_measured=tc, expected=exp, why=why,
                matches=bool(tc is not None and int(tc) == exp),
                note="不成立 ⇒ 审计器未覆盖全部 target 调用 ⇒ 几何结论无效")


def run_one() -> int:
    assert MODE in MODES, f"未知模式：{MODE}（可选 {MODES}）"
    from .. import results_schema as S
    from . import model_self as M
    from . import serial_oracle as SO
    from . import spec_dec

    env = _smoke_env()
    os.environ.update(env)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        tool="serial_oracle_gpu", tag=TAG, run_id=run_id, case=CASE, mode=MODE, rep=REP,
        diagnostic_only=True,
        what=("把本轮那一次 `s=Kp` 的 target block 校验换成**逐 token `_step`（s=1）**，"
              "其余控制逻辑（accept/replacement/rollback/EOS/截断/cache 提交）逐字复用；"
              "`prod` = 生产 block 校验（对照）；`ar` = target-only 自回归（基线）"),
        semantics="三种模式跑同一冻结配置、同一 family、独立进程；token IDs 与 AR 的相等性是硬门",
        smoke_env=env, stage="start",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=S.git_meta(ROOT),
        rules=("冻结配置：ctx=8192/chunk=2048/k=4/max_new=64/base_bf16/eager/target_rope=cpu_fp32/"
               "bf16_red_red=1/struct_check=1；接受公式与停止条件未改；draft 未改；cache 不共享；不自动重试"),
        design_doc="docs/serial_target_oracle_design_2026-10-03.md",
        accounting_identity=("ar: target_calls==decode_steps；serial: target_calls==proposed+rollbacks；"
                             "prod: target_calls==target_block_calls+rollbacks"))
    try:
        from . import family_input as FI
        rec["provenance"] = FI.family_record(CASE)
    except Exception as e:                                                # noqa: BLE001
        rec["provenance"] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")

    # ---- 1) 调用几何审计：**必须装在 harness 之前**（TRS/DecisionHooks 会继续往里套）----
    enforce = bool(MODE in ("ar", "serial"))
    aud = SO.install_run_audit(M.QwenLM, enforce_m1_outside_prefill=enforce)
    rec["audit_install"] = dict(enforce_m1_outside_prefill=enforce, owner="model_self.QwenLM",
                                installed_before_harness_import=True,
                                why=("serial/ar：非 prefill 阶段任何 s>1 的 target 前向立即抛错；"
                                     "prod：只记录，用于取得对照几何"))
    # ---- 2) verifier 模式（serial = 仅诊断；与 row-aligned wrapper 互斥）----
    rec["verify_wrap_install"] = spec_dec.set_verify_wrap(None)
    rec["verify_mode_install"] = spec_dec.set_target_verify_mode("serial" if MODE == "serial" else "block")
    rec["verify_mode_before_run"] = spec_dec.target_verify_mode()
    rec["verify_wrap_before_run"] = spec_dec.verify_wrap_state()
    rec["row_aligned_wrap_active"] = bool(rec["verify_wrap_before_run"].get("active"))

    spec_dec.clear_serial_trace()
    fallback_pt = OUTDIR / f"{TAG}_{CASE}_{MODE}_rep{REP}_{run_id}.pt"
    rc = 2
    try:
        from . import g5_8k_smoke_cudnn as H
        rec["smoke_config"] = dict(ctx=H.CTX, chunk=H.CHUNK, k=H.K, max_new=H.MAXNEW, kv=H.KV,
                                   arm=H.ARM, mode=H.MODE, family=CASE, target_rope=H.TARGET_ROPE,
                                   bf16_red_red=H.RED_RED, struct_check=H.STRUCT_CHECK)
        rc = int(H.main())
        rec["smoke_rc"] = rc
        cands = sorted(glob.glob(str(OUTDIR / "smoke8k_*.json")), key=os.path.getmtime)
        smoke_path = Path(cands[-1]) if cands else None
        rec["smoke_artifact"] = str(smoke_path)
        if smoke_path is not None:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            run = smoke.get("run") or {}
            rec["smoke"] = dict(
                stage=smoke.get("stage"), run_id=smoke.get("run_id"),
                gen=list(run.get("gen") or []), gen_len=run.get("gen_len"),
                stop_reason=run.get("stop_reason"), gen_ids_in_vocab=run.get("gen_ids_in_vocab"),
                prefill_logits_finite=run.get("prefill_logits_finite"),
                specdec_stats=run.get("specdec_stats"), wall_s=run.get("wall_s"),
                prefill_s=run.get("prefill_s"),
                end_state=smoke.get("end_state"), checks_passed=((smoke.get("checks") or {}).get("passed")),
                checks_failures=[f.get("name") for f in ((smoke.get("checks") or {}).get("failures") or [])],
                checks=dict(smoke.get("checks") or {}),
                verdict=smoke.get("verdict"),
                cache_structure_ok=smoke.get("cache_structure_ok"),
                cache_structure_n_checks=((smoke.get("cache_structure_check") or {}).get("n_checks")),
                cache_structure_checks=((smoke.get("cache_structure_check") or {}).get("checks")),
                finiteness_n_checked=((smoke.get("checks") or {}).get("n_checked")),
                decision_coverage=(smoke.get("decision_coverage") or {}).get("ok"),
                decision_coverage_detail=(smoke.get("decision_coverage") or {}),
                input=(smoke.get("input") or {}),
                mem_timeline_last=(smoke.get("mem_timeline") or [{}])[-1],
                n_rounds=len(smoke.get("rounds") or []),
                rounds=list(smoke.get("rounds") or []),
                round_invariants=smoke.get("round_invariants"),
                pstate=smoke.get("pstate"),
                gpu_consumers_before=smoke.get("gpu_consumers_before"),
                resident_models=smoke.get("resident_models"),
                weights_bitmatch=smoke.get("weights_bitmatch"),
                kv_format=smoke.get("kv_format"),
                prefill_digest=smoke.get("prefill_digest"),
                target_rope_scope=smoke.get("target_rope_scope"),
                attention_stats=run.get("attention_stats"),
                commit_guard=smoke.get("commit_guard"))
    except Exception as e:                                                # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = f"{type(e).__name__}: {str(e)}"
        rec["traceback"] = traceback.format_exc()
        rc = 2

    # ---- 3) 运行后：审计汇总 + 串行 trace + 模式/包装恢复核查 ----
    # ⚠️ 后处理**不得**丢失运行记录：首版这里写成 `aud.uninstall`（应为 `aud["uninstall"]`）⇒
    #    AttributeError 直接抛出、连 JSON 都没写。现在整段包起来，异常也照常落盘并标出。
    try:
        rec["verify_mode_after_run"] = spec_dec.target_verify_mode()
        rec["verify_wrap_after_run"] = spec_dec.verify_wrap_state()
        rec["serial_verify_dispatched"] = bool(rec["verify_mode_before_run"] == "serial")
        tr_ = spec_dec.serial_trace()
        rec["serial_trace"] = tr_
        rec["serial_trace_summary"] = dict(
            n_rounds_traced=len(tr_),
            kp_per_round=[t.get("kp") for t in tr_],
            pre_per_round=[t.get("pre") for t in tr_],
            step_cache_lens_progression=[t.get("step_cache_lens") for t in tr_],
            step_argmax=[t.get("step_argmax") for t in tr_],
            note="serial 模式下每轮恰好一条（由 `_verify` 内部追加）；prod/ar 恒为空")
        trun = SO.run_audit_summary(aud["state"])
        rec["audit"] = trun
        rec["audit_uninstall"] = aud["uninstall"]()
        try:
            spec_dec.set_target_verify_mode("block")
        except Exception:                                                 # noqa: BLE001
            pass
        stats = ((rec.get("smoke") or {}).get("specdec_stats") or {})
        end_state = ((rec.get("smoke") or {}).get("end_state") or {})
        rec["accounting"] = _accounting(MODE, stats, end_state, trun)
        cg = ((rec.get("smoke") or {}).get("commit_guard") or {})
        rec["env_gate"] = cg
        rec["env_qualified"] = bool(cg.get("ok", True))
        ran = bool((rec.get("smoke") or {}).get("run") or (rec.get("smoke") or {}).get("gen"))
        rec["model_run_happened"] = bool(ran)
        # ---- 4) 本进程的自检（**每一步都必须为真**，否则本次运行结论无效）----
        n_inst = int(trun.get("n_instances") or 0)
        n_rounds = int(((rec.get("smoke") or {}).get("n_rounds") or 0))
        checks = dict(
            environment_qualified=bool(rec["env_qualified"]),
            model_run_happened=bool(ran),
            audit_sane=(True if ran else None),
            audit_instances_as_expected=(bool(n_inst == EXPECTED_INSTANCES[MODE]) if ran else None),
            audit_target_identified=(bool(trun.get("target_identified")) if ran else None),
            audit_no_violations=(bool(not trun.get("violations")) if ran else None),
            audit_certifies_target_generation_m1=(None if (MODE == "prod" or not ran)
                                                  else bool(trun.get("certifies_m1"))),
            accounting_matches=(bool((rec.get("accounting") or {}).get("matches")) if ran else None),
            serial_trace_present_iff_serial=(bool((len(tr_) > 0) == (MODE == "serial")) if ran else None),
            serial_trace_one_entry_per_round=((None if MODE != "serial"
                                               else bool(n_rounds > 0 and len(tr_) == n_rounds))
                                              if ran else None),
            no_row_aligned_wrap=bool(not rec["row_aligned_wrap_active"]),
            mode_restored_to_block=bool(rec["verify_mode_after_run"] == "block"),
            smoke_checks_passed=(bool(((rec.get("smoke") or {}).get("checks_passed")) is True)
                                 if ran else None),
            smoke_rc_zero=bool(rc == 0),
        )
    except Exception as e:                                                # noqa: BLE001
        checks = dict(post_processing_failed=False)
        trun = {}
        rec["post_processing_error"] = f"{type(e).__name__}: {str(e)}"
        rec["post_processing_traceback"] = traceback.format_exc()
    rec["self_checks"] = checks
    rec["self_checks_all_ok"] = bool(all(v is not False for v in checks.values()))
    if rc == 0 and rec["self_checks_all_ok"]:
        rec["stage"] = "complete"
    elif not rec.get("env_qualified", True):
        rec["stage"] = "env_unqualified"
        rec["blocked_by"] = "environment"
    elif rc == 0:
        rec["stage"] = "complete_with_selfcheck_failures"
    elif not rec.get("model_run_happened", False):
        rec["stage"] = "no_model_run"
        rec["blocked_by"] = rec.get("blocked_by") or "unknown_no_run"
    else:
        rec["stage"] = "smoke_rc_nonzero"
    # ★ 生成 IDs 先落盘，再写摘要（后处理失败时也照常写，字段缺失不影响记录存在）
    acc = rec.get("accounting") or {}
    torch.save(dict(gen=(rec.get("smoke") or {}).get("gen"), case=CASE, mode=MODE, rep=REP,
                    meta=dict(run_id=run_id, smoke_artifact=rec.get("smoke_artifact"),
                              stop_reason=((rec.get("smoke") or {}).get("stop_reason")),
                              gen_len=((rec.get("smoke") or {}).get("gen_len")),
                              target_generation_calls=acc.get("target_generation_calls_measured"),
                              verify_mode=rec.get("verify_mode_before_run"),
                              env_qualified=rec.get("env_qualified"),
                              stage=rec.get("stage"))),
               str(fallback_pt))
    rec["artifact_pt"] = str(fallback_pt)
    p = OUTDIR / f"{TAG}_{CASE}_{MODE}_rep{REP}_{run_id}.json"
    rec["artifact_json"] = str(p)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    g = (rec.get("smoke") or {}).get("gen") or []
    print(f"[so] case={CASE} mode={MODE} rep={REP} rc={rc} stage={rec.get('stage')} "
          f"env_qualified={rec.get('env_qualified')} gen_len={len(g)} "
          f"stop={(rec.get('smoke') or {}).get('stop_reason')} "
          f"tgt_gen_calls={acc.get('target_generation_calls_measured')}/{acc.get('expected')} "
          f"m1={(rec.get('audit') or {}).get('target_all_generation_m1')} "
          f"selfchk={rec['self_checks_all_ok']} -> {p.name}", flush=True)
    return int(rc)


def main() -> int:
    return run_one()


if __name__ == "__main__":
    raise SystemExit(main())
