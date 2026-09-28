"""g5_8k_qk32_intervention.py — QK FP32 最小精度干预测试（控制组 vs 干预组，完整 64-token 生成）。

问题
  仅把 **target** 的 QK 打分改成 fp32（`DIAG_QK_FP32`）能否改善 8K 下 AR 与 SpecDec 的输出不一致？

配置（与**原失败实验**一致，未改任何一项）
  ctx=8192、chunk=2048、max_new=**64**、k=**4**、greedy；同一 prompt/token ids（记录 hash）；
  attention backend 沿用 `cudnn_prefill`（**不因启用开关而换 backend**）；
  target KV ∈ {base_bf16, v_fp8}，draft KV 固定 `base_bf16`。

两组
  `QK_GROUP=control`  ：`set_diag_preset("eager")`，无干预；
  `QK_GROUP=qk32`     ：**仅 target** 走 QK FP32（draft 保持原精度）。

干预范围（由 `diag_qk_scope.static_scope_proof()` 静态证明并写入结果）
  * 开关在 `model_self` 里**只有一个影响计算的读取点**，位于 eager 分支；
  * **融合 prefill 绕过该开关**（`_fused_prefill_attention` 源码里没有该开关名）
    ⇒ 生效阶段是 target 的 **decode / verify / repair**，**不是**全链路升精度；
  * 不改 softmax / AV / o_proj / MLP / RMSNorm / lm_head / 权重 / 缓存 dtype / 并列规则 / 验收阈值。

做法：`QKScopeHook` 按**模型实例**限定开关（不改 `model_self.py`），
      因而不需要为启用开关而切换 backend，也不影响 draft。

纪律：不做性能测量（计时只作诊断记录）；不改验收协议；不自动重试。
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_fields as DF
from . import diag_qk_scope as QS
from . import model_self as M
from . import spec_cache
from . import spec_dec
from .smoke_checks import (CheckRegistry, evaluate, prefill_finiteness_live,
                           verify_decision_coverage)
from .smoke_hooks import DecisionHooks
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

GROUP = os.environ.get("QK_GROUP", "control").strip().lower()
KV = os.environ.get("QK_KV", "base_bf16").strip()
MODE = os.environ.get("QK_MODE", "ar").strip().lower()
TAG = os.environ.get("QK_TAG", "qk32")
CTX = int(os.environ.get("QK_CTX", "8192"))
CHUNK = int(os.environ.get("QK_CHUNK", "2048"))
MAXNEW = int(os.environ.get("QK_MAXNEW", "64"))
K = int(os.environ.get("QK_K", "4"))
ARM = os.environ.get("QK_ARM", "cudnn_prefill")     # 与原实验同一个 backend 策略
assert GROUP in ("control", "qk32"), GROUP
assert MODE in ("ar", "spec"), MODE
assert KV in ("base_bf16", "v_fp8"), KV
assert MAXNEW == 64 and K == 4, "本轮必须保持 max_new=64 / k=4"

# 首个分歧的候选观察位（前 2×2 已定位到 index 31；这里仍全序列扫描）
WATCH = [1083, 5880, 5230, 22909]


def _write(rec: dict, run_id: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{TAG}_{GROUP}_{KV}_{MODE}_{run_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _mib(x: float) -> float:
    return round(float(x) / 2 ** 20, 3)


def _row(t: torch.Tensor) -> dict:
    """一行 logits 的关键字段（fp32，CPU）。"""
    r = t.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=WATCH)
    idx = int(f["argmax_id"])
    top2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(top2[0] - top2[1])
    f["tie_at_top"] = bool(float(top2[0]) == float(top2[1]))
    f["_argmax"] = idx
    return f


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        run_id=run_id, tool="g5_8k_qk32_intervention", tag=TAG,
        group=GROUP, kv=KV, mode=MODE,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=S.git_meta(ROOT),
        config=dict(ctx=CTX, chunk=CHUNK, max_new=MAXNEW, k=K, greedy=True,
                    attention_strategy=ARM, target_kv=KV, draft_kv="base_bf16",
                    group=GROUP),
        intervention=("control：生产精度（set_diag_preset('eager')），target 与 draft 都不启用任何精度干预"
                      if GROUP == "control" else
                      "qk32：**仅 target** 启用 QK FP32（DIAG_QK_FP32）；draft 保持生产精度；"
                      "prefill 走融合 cuDNN ⇒ **不受影响**"),
        scope="最小精度干预测试：控制组 vs 干预组，各自完整 64-token 生成。"
              "不做性能测量、不改默认生产配置、不改验收协议。")
    print(f"[qk32] run_id={run_id} group={GROUP} kv={KV} mode={MODE} tag={TAG}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)

    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    rec["gpu_consumers_before"] = P5V.gpu_consumers()
    print(f"[qk32] commit: {why}")
    print(f"[qk32] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        rec["failure"] = dict(stage="env_gate", reason=f"commit_ok={ok} pstate={pf.get('qualified')}")
        _write(rec, run_id)
        print("[qk32] !! 环境不合格，未加载")
        return 3

    reg: CheckRegistry | None = None
    qk_hook = None
    hooks = None
    try:
        # ---- 精度策略：**先设成干净基线**，再由 per-model hook 打开 target ----
        rec["diag_precision_baseline"] = M.set_diag_preset("eager")
        rec["attention_scope_proof"] = QS.static_scope_proof()
        print(f"[qk32] 干预范围：生效={rec['attention_scope_proof']['effective_stages']} "
              f"未生效={rec['attention_scope_proof']['not_effective_stages']}")

        from .lowmem_load import load_ours_bf16
        tgt, tok, tinfo = load_ours_bf16(TARGET)
        if MODE == "ar":
            drf, dinfo = None, dict(load_path="not_loaded", weights_bitmatch=True,
                                    mismatch_keys=[], n_params_checked=None,
                                    cpu_peak_note="AR 模式不加载 draft")
            rec["resident_models"] = ["target"]
        else:
            drf, _t2, dinfo = load_ours_bf16(DRAFT)
            rec["resident_models"] = ["target", "draft"]
        rec["load"] = dict(target=tinfo, draft=dinfo)
        rec["weights_bitmatch"] = bool(tinfo["weights_bitmatch"] and dinfo["weights_bitmatch"])
        if not rec["weights_bitmatch"]:
            rec["stage"] = "aborted_weight_mismatch"
            rec["failure"] = dict(stage="load", reason="权重逐张量校验不一致")
            _write(rec, run_id)
            return 2

        tgt.set_kv_format(spec_dec._fmt(KV))
        if drf is not None:
            drf.set_kv_format("fp16")
        rec["kv_format"] = dict(target=spec_dec._fmt(KV),
                                draft=("fp16" if drf is not None else "not_loaded"))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()

        full = P5V.load_ids(tok)
        ids = full[:, :CTX].contiguous()
        prompt_len = int(ids.shape[1])
        rec["input"] = dict(ctx=prompt_len, ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]),
                            source="wikitext-2-raw test 前缀（与其它 G5 工具同源）",
                            vocab_size=int(tgt.embed_tokens.weight.shape[0]))
        print(f"[qk32] 输入 {prompt_len} token hash={rec['input']['ids_hash']}；开始 prefill ...",
              flush=True)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        acc: dict = {}
        inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
        reg = CheckRegistry()
        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner)
        hooks.install()
        # ★ per-model 作用域：**只**让 target 吃到 QK FP32（控制组则谁都不开）
        qk_hook = QS.QKScopeHook(tgt=tgt, enabled=(GROUP == "qk32"))
        qk_hook.install()

        rounds: list[dict] = []
        verify_rows: list[dict] = []
        repair_lg: list[dict] = []
        ar_rows: list[dict] = []
        ar_meta: list[dict] = []
        prefill_row: dict | None = None
        eos_id = tok.eos_token_id

        # ---- 记录真实验证行/修复行（**只读**包装，不改计算）----
        # ⚠️ 签名自检必须看 **hooks 保存的原函数**：此刻 `spec_dec._verify` 已被
        # `DecisionHooks` 换成 `(self, *a, **k)` 包装器，直接 inspect 它只会看到 ['a','k']。
        # 真实语义是 `_verify(model, cache, cands)`（不是 `(cache, cands)`）——
        # 首跑把形参写错，`cache` 绑到了模型实例上，`cache[0]` 直接 TypeError。
        import inspect as _inspect
        _vp = list(_inspect.signature(hooks._orig["verify"]).parameters)
        _sp = list(_inspect.signature(hooks._orig["step"]).parameters)
        assert len(_vp) >= 3 and len(_sp) >= 3, f"_verify/_step 签名异常: {_vp} {_sp}"
        rec["spy_signatures"] = dict(verify=_vp, step=_sp)
        _orig_verify, _orig_step = spec_dec._verify, spec_dec._step

        def _verify_spy(model, cache, cands, *a, **k):
            pre = spec_cache.cache_length(cache)
            rows, blk = _orig_verify(model, cache, cands, *a, **k)
            verify_rows.append(dict(round=reg.round, pre=pre, kp=len(cands),
                                    cands=[int(x) for x in cands],
                                    rows=rows.detach().float().cpu()))
            return rows, blk

        def _step_spy(model, token, cache, *a, **k):
            lg, c = _orig_step(model, token, cache, *a, **k)
            if model is tgt and lg is not None:
                repair_lg.append(dict(row=lg.detach().float().cpu().reshape(-1)))
            return lg, c

        if MODE == "spec":
            spec_dec._verify, spec_dec._step = _verify_spy, _step_spy

        try:
            if MODE == "ar":
                torch.cuda.synchronize()
                with torch.inference_mode(), M.attention_phase("prefill"):
                    lg, cache = inner(tgt, ids)
                reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg",
                                 where="AR prefill 返回：首步 argmax 的输入")
                prefill_row = _row(lg.reshape(-1, lg.shape[-1])[-1])
                nxt = prefill_row["_argmax"]
                gen_ids: list[int] = []
                decode_runs = 0
                cur_row = prefill_row
                for i in range(MAXNEW):
                    ar_rows.append(dict(gen_index=i, input_position=prompt_len - 1 + decode_runs,
                                        **{k: v for k, v in cur_row.items()
                                           if k != "_argmax"}))
                    gen_ids.append(nxt)
                    ar_meta.append(dict(gen_index=i, cache_len=spec_cache.cache_length(cache),
                                        pt_reserved_mib=_mib(torch.cuda.memory_reserved())))
                    if eos_id is not None and nxt == eos_id:
                        break
                    if i == MAXNEW - 1:
                        break
                    with torch.inference_mode(), M.attention_phase("decode"):
                        lg, cache = tgt._run(
                            tgt.embed_tokens(torch.tensor([[nxt]], device=ids.device)),
                            cache, need_logits=True)
                    decode_runs += 1
                    reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                                     round_=i, step=decode_runs,
                                     where=f"AR 第 {decode_runs} 次解码返回：紧接被 argmax 使用")
                    cur_row = _row(lg.reshape(-1, lg.shape[-1])[-1])
                    nxt = cur_row["_argmax"]
                torch.cuda.synchronize()
                out = dict(gen=gen_ids, stats=dict(mode="AR", decode_steps=decode_runs,
                                                   rounds=len(gen_ids)))
            else:
                torch.cuda.synchronize()
                out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos_id,
                                                  target_kv=KV, draft_kv="base_bf16",
                                                  log=lambda d: rounds.append(dict(d)))
                torch.cuda.synchronize()
                if not verify_rows:      # 首轮 prefill 的 saved_next 也要留证
                    pass
        finally:
            spec_dec._verify, spec_dec._step = _orig_verify, _orig_step
            wall = time.perf_counter() - t0
            if qk_hook is not None:
                rec["qk_scope_report"] = qk_hook.report()
                qk_hook.remove()
            hooks.remove()

        rec["precision_scope_effective"] = dict(
            group=GROUP,
            target_qk_fp32=bool(GROUP == "qk32"),
            draft_qk_fp32=False,
            prefill_affected=False,
            effective_stages=rec["attention_scope_proof"]["effective_stages"],
            hook=rec.get("qk_scope_report"),
            note="`qk_scope_report.calls_target` 与 `calls_other` 分别对应该模型实例上的开关状态；"
                 "干预组要求 target 全为 flag_on、draft 全为 flag_off",
        )

        gen = [int(x) for x in out["gen"]]
        gen_len = len(gen)
        vocab = int(tgt.embed_tokens.weight.shape[0])
        if gen and eos_id is not None and gen[-1] == eos_id:
            stop_reason = "eos"
        elif gen_len >= MAXNEW:
            stop_reason = "max_new_reached"
        else:
            stop_reason = "other_short"
        try:
            M.assert_strategy_used()
            attn_ok = "ok"
        except RuntimeError as e:
            attn_ok = f"RuntimeError: {e}"
        a_stats = M.attention_stats()
        fused_by_phase = a_stats.get("fused_by_phase", {})
        eager_by_phase = a_stats.get("eager_by_phase", {})
        dispatch_ok = (sorted(k for k, v in fused_by_phase.items() if v > 0) == ["prefill"]
                       and all(fused_by_phase.get(p, 0) == 0
                               for p in ("propose", "verify", "repair", "decode")))
        rec["rounds"] = rounds
        # ⚠️ 修正：旧写法 `all(reg.calls.get(("prefill", m), {}).get("ok", False)
        #                    for m in ("target","draft"))` 会把"draft 未驻留 ⇒ 该检查不存在"
        # 当成失败 ⇒ **AR 恒报 False**（与同记录的 checks.passed=True 自相矛盾）。
        # 现在只对**实际驻留并执行过** prefill 的模型求合取，未驻留记 not_applicable。
        pf_finite = prefill_finiteness_live(reg, rec["resident_models"])
        rec["run"] = dict(
            gen=gen, gen_len=gen_len, max_new=MAXNEW, eos_id=eos_id, stop_reason=stop_reason,
            gen_ids_in_vocab=bool(all(0 <= g < vocab for g in gen)),
            prefill_logits_finite=pf_finite["all_executed_ok"],
            prefill_logits_finite_by_model=pf_finite,
            prefill_logits_finite_semantics=pf_finite["semantics"],
            attention_ok=attn_ok, attention_stats=a_stats,
            dispatch_only_prefill=dispatch_ok,
            wall_s=wall, specdec_stats=out["stats"],
            peak_allocated_mib=_mib(torch.cuda.max_memory_allocated()),
            peak_reserved_mib=_mib(torch.cuda.max_memory_reserved()),
            mem_now=dict(allocated_mib=_mib(torch.cuda.memory_allocated()),
                         reserved_mib=_mib(torch.cuda.memory_reserved()),
                         cuda_free_mib=_mib(torch.cuda.mem_get_info()[0])),
            timing_note="计时含插桩，仅作诊断记录；本轮**不做**加速/减速声明。",
        )
        # 决策行落盘（供跨组在同一共同前缀下比较）
        dumps: dict = dict(mode=MODE, prompt_len=prompt_len)
        if prefill_row is not None and MODE == "ar":
            dumps["prefill_row"] = {k: v for k, v in prefill_row.items() if k != "_argmax"}
        if MODE == "ar":
            dumps["ar_rows"] = ar_rows
            dumps["ar_meta"] = ar_meta
        else:
            dumps["verify_rows"] = [{k: v for k, v in vr.items() if k != "rows"} |
                                    dict(rows=[_row(r) for r in vr["rows"]])
                                    for vr in verify_rows]
            dumps["repair_lg"] = [_row(x["row"]) for x in repair_lg]
            dumps["reconstruction_note"] = (
                "saved_next(round r) = prefill 行（r=0）；否则若上一轮 m==Kp 用其 rows[Kp-1]，"
                "否则用上一轮 repair 行 —— 与 spec_dec 的实现一致，可据 emitted 交叉核对")
        dp = OUTDIR / f"{TAG}_{GROUP}_{KV}_{MODE}_{run_id}.rows.pt"
        rows_all = {k: v for k, v in dumps.items() if k != "verify_rows"}
        torch.save(rows_all, str(dp))
        rec["rows_dump"] = str(dp)
        rec["decision_rows"] = dumps

        if MODE == "ar":
            dlg = reg.calls.get(("decode", "target"), {}).get("role_counts", {}).get("decode_lg", 0)
            extra = [dict(name="ar_decode_step_coverage", ok=(dlg == int(out["stats"]["decode_steps"])),
                          detail=f"decode_lg 检查={dlg} vs 解码步数={out['stats']['decode_steps']}")]
        else:
            cov = verify_decision_coverage(reg,
                                           draft_calls_expected=int(out["stats"].get("draft_calls", 0)),
                                           rounds_expected=int(out["stats"].get("rounds", 0)))
            cov["name"] = "decision_step_coverage"
            extra = [cov]
        rec["decision_coverage"] = extra[0]
        rec["checks"] = evaluate(registry=reg,
                                 mismatch_rounds=int(out["stats"].get("mismatch_rounds", 0)),
                                 extra_required=extra, mode=MODE, weights_ok=rec["weights_bitmatch"],
                                 weights_detail="逐张量校验通过",
                                 invariants_ok=True, invariants_detail="本轮不变量见 rounds/ar_meta",
                                 dispatch_ok=dispatch_ok,
                                 dispatch_detail=f"fused={fused_by_phase} eager={eager_by_phase}",
                                 attention_assert=attn_ok, gen_len=gen_len, max_new=MAXNEW,
                                 eos_id=eos_id, gen=gen, stop_reason=stop_reason,
                                 exception=None, alloc_ok=True)
        passed = bool(rec["checks"].get("passed"))
        rec["stage"] = "complete" if passed else "checks_failed"
        rec["failure"] = None if passed else dict(stage="checks", reason="checks.passed=False")
        p = _write(rec, run_id)
        print(f"[qk32] gen_len={gen_len} stop={stop_reason} "
              f"peak_alloc={rec['run']['peak_allocated_mib']}MiB "
              f"peak_resv={rec['run']['peak_reserved_mib']}MiB")
        print(f"[qk32] qk_scope: {json.dumps(rec.get('qk_scope_report'), ensure_ascii=False)}")
        print(f"[qk32] gen={gen}")
        print(f"[qk32] RESULT_JSON={p}")
        return 0 if passed else 5
    except Exception:
        rec["stage"] = "crashed"
        rec["failure"] = dict(stage="exception", reason=traceback.format_exc().strip().splitlines()[-1])
        rec["traceback"] = traceback.format_exc()
        try:
            if qk_hook is not None:
                qk_hook.remove()
            if hooks is not None:
                hooks.remove()
        except Exception:                                       # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        rec["alloc"] = P5V.allocator_state()
        p = _write(rec, run_id)
        print(f"[qk32] !! 崩溃（已落盘 {p}）：\n{rec['traceback']}")
        print(f"[qk32] RESULT_JSON={p}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
