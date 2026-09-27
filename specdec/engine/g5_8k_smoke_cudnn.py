"""g5_8k_smoke_cudnn.py — **8K 单次 smoke**（条件性工具；只回答"能不能跑、峰值多少"）。

它**不是**吞吐测量，也**不是**稳定性验证。纪律（违反即数据作废）
  - **单次**运行，`prefill + 少量 decode`，不启动正式矩阵、不做长时间循环；
  - 上下文 / chunk / k / dtype 全部**冻结**，本工具只提供默认值，不在此处调参；
  - `allocated` **不是**全部物理显存占用；`reserved` 超过显存**也不单独**作为分页证据
    ⇒ 同时记录 PyTorch allocated/reserved/peak **与** 设备级 `nvidia-smi` 占用，两者分开陈述；
  - 计时字段仅供诊断（**含插桩**），**禁止**进入任何 speedup 表；
  - 发现结构错误 / 质量明显异常 / OOM ⇒ **保存结果并停止**，不升级测试；
  - 双模型都用**降峰值加载器**（`lowmem_load`，权重逐位校验）以提高一次跑成的概率；
    该选择写入结果，不与其他工具的加载路径混为一谈。

用法（交互式 GPU 终端）
  scripts\\run_g5_8k_smoke.cmd              （= 本文档默认配置）
环境变量
  SMOKE_CTX=8192 SMOKE_CHUNK=2048 SMOKE_K=4 SMOKE_MAXNEW=16 SMOKE_KV=base_bf16
  SMOKE_ARM=cudnn_prefill|eager   默认 cudnn_prefill（eager 在 8K 上历史上必 OOM，仅作对照）
  SMOKE_ALLOW_UNQUALIFIED=1       允许 commit/P-state 不合格仍跑（**数据只能算探索性**）
  SMOKE_DRY_RUN=1                 只打印计划，不加载、不占显存
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V
from . import g5_driver as D
from .smoke_checks import (CheckRegistry, FinitenessError, evaluate,
                           verify_decision_coverage)
from .smoke_hooks import DecisionHooks
from . import spec_cache

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

CTX = int(os.environ.get("SMOKE_CTX", "8192"))
CHUNK = int(os.environ.get("SMOKE_CHUNK", "2048"))
K = int(os.environ.get("SMOKE_K", "4"))
MAXNEW = int(os.environ.get("SMOKE_MAXNEW", "16"))
KV = os.environ.get("SMOKE_KV", "base_bf16")
ARM = os.environ.get("SMOKE_ARM", "cudnn_prefill")
# SMOKE_MODE=spec（SpecDec，default）| ar（**target-only** 自回归；不是 dual-resident）
MODE = os.environ.get("SMOKE_MODE", "spec").strip().lower()
ALLOW_UNQUALIFIED = S.env_flag("SMOKE_ALLOW_UNQUALIFIED", False)
DRY_RUN = S.env_flag("SMOKE_DRY_RUN", False)
# 轻量回归：注入故障以验证"检查失败 ⇒ 落盘 + 非零退出"（**不加载模型**）
SELFTEST = os.environ.get("SMOKE_SELFTEST", "").strip()


def _cache_len(cache) -> int:
    """KV cache 的当前长度（用引擎自己的 spec_cache.cache_length，不另造口径）。"""
    return int(spec_cache.cache_length(cache))


def mem(tag: str, rec: dict) -> None:
    """同时记录 **PyTorch 侧** 与 **设备侧**；两者含义不同，报告里必须分开写。"""
    free, total = torch.cuda.mem_get_info()
    smi = D.system_snapshot()
    blk = dict(tag=tag,
               pt_allocated_mib=torch.cuda.memory_allocated() / 2 ** 20,
               pt_reserved_mib=torch.cuda.memory_reserved() / 2 ** 20,
               pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
               pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
               cuda_free_mib=free / 2 ** 20, cuda_total_mib=total / 2 ** 20,
               smi_mem_used_MiB=smi.get("mem_used_MiB"),
               smi_mem_total_MiB=smi.get("mem_total_MiB"),
               smi_util_pct=smi.get("util_pct"),
               note="pt_* 是 PyTorch 记账（allocated≠全部物理占用；reserved 是缓存池上限）")
    rec.setdefault("mem_timeline", []).append(blk)
    print(f"[smoke] mem {tag:<14} pt_alloc={blk['pt_allocated_mib']:7.1f} "
          f"pt_reserved={blk['pt_reserved_mib']:7.1f} "
          f"pt_peak_res={blk['pt_peak_reserved_mib']:7.1f} "
          f"cuda_free={blk['cuda_free_mib']:7.1f} smi_used={blk['smi_mem_used_MiB']}", flush=True)


def _out_path(run_id: str) -> Path:
    return OUTDIR / f"smoke8k_{ARM}_{run_id}.json"


def _write(rec: dict, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _out_path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def _selftest(scenario: str, rec: dict, run_id: str) -> int:
    """**不加载模型**地验证"检查失败 ⇒ 落盘 + 非零退出"这条路径。

    为什么需要它：任务要求"轻量回归…不要为了测试错误分支加载大模型"，
    同时要证明"错误会被拦住、结果能落盘、runner 不会继续推进"。
    本函数复用与真实运行**同一套** `smoke_checks` 判定逻辑与 `_write`，
    只是把故障**注入**进来（而非真的跑出故障）。
    """
    reg = CheckRegistry()
    ok = torch.zeros(1, 4)
    rec["selftest"] = dict(scenario=scenario,
                           note="**非真实运行**：故障为注入，用于回归判定/退出码/落盘路径")
    exc = None
    non_finite = None
    if scenario == "nonfinite":
        try:
            reg.check_finite(ok, phase="prefill", model="target", role="prefill_lg")
            reg.check_finite(torch.tensor([1.0, float("nan")]), phase="verify",
                             model="target", role="rows", where="selftest 注入")
        except FinitenessError as e:
            exc = f"FinitenessError: {e}"
            non_finite = dict(error=str(e), first_bad=reg.first_bad, coverage=reg.coverage())
    else:
        for (ph, md, role) in (("prefill", "target", "prefill_lg"), ("prefill", "draft", "prefill_lg"),
                               ("propose", "draft", "input_lg"), ("verify", "target", "rows"),
                               ("repair", "target", "saved_next"), ("repair", "draft", "saved_next")):
            if scenario == "missing" and ph in ("propose", "verify"):
                continue                       # 注入"预期检查缺失"
            reg.check_finite(ok, phase=ph, model=md, role=role)
    rec["checks"] = evaluate(
        registry=reg, mismatch_rounds=1,
        weights_ok=True, weights_detail="338+290 逐位校验（selftest 注入）",
        invariants_ok=(scenario != "invariantfail"),
        invariants_detail="selftest 注入" + ("：不变量失败" if scenario == "invariantfail" else "：通过"),
        dispatch_ok=(scenario != "dispatchfail"),
        dispatch_detail="selftest 注入" + ("：融合出现在 verify" if scenario == "dispatchfail" else "：仅 prefill"),
        attention_assert="ok" if scenario != "dispatchfail" else "RuntimeError: 静默不生效",
        gen_len=16, max_new=16, eos_id=151645, gen=[1] * 16,
        stop_reason="max_new_reached", exception=exc,
        alloc_ok=True, alloc_detail="selftest 注入")
    if non_finite:
        rec["non_finite"] = non_finite
    rec["stage"] = "run_ended" if rec["checks"]["passed"] else "run_ended_checks_failed"
    rec["verdict"] = dict(ran=True, passed=bool(rec["checks"]["passed"]), selftest=True,
                          claim="selftest：只验证判定/退出码/落盘路径，**不是**任何 GPU 结论")
    _write(rec, run_id)
    rc = 0 if rec["checks"]["passed"] else 5
    print(f"[smoke/selftest] scenario={scenario} passed={rec['checks']['passed']} "
          f"failures={[f['name'] for f in rec['checks']['failures']]} "
          f"uncovered={rec['checks']['uncovered']} exit={rc}")
    return rc


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_8k_smoke_cudnn",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     config=dict(mode=MODE, ctx=CTX, chunk=CHUNK, k=K, max_new=MAXNEW, kv=KV, arm=ARM,
                                 target_model=TARGET, draft_model=DRAFT,
                                 load_path="lowmem_load.load_ours_bf16 (双模型, 权重逐位校验)"),
                     scope="**单次 smoke**：只回答能不能跑 + 峰值多少。不是吞吐测量，不是稳定性验证。"
                     if False else "**单次 smoke**：只回答能不能跑 + 峰值多少。不是吞吐测量，不是稳定性验证。",
                     timing_policy="计时字段含插桩，仅供诊断，禁止进入 speedup 表")
    print(f"[smoke] run_id={run_id} arm={ARM} ctx={CTX} chunk={CHUNK} k={K} "
          f"max_new={MAXNEW} kv={KV}")
    if SELFTEST:
        return _selftest(SELFTEST, rec, run_id)
    if DRY_RUN:
        rec["stage"] = "dry_run"
        _write(rec, run_id)
        print(f"[smoke] DRY RUN（未加载、未占显存）-> {_out_path(run_id)}")
        return 0

    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=ok, why=why, system=P5V.system_memory(),
                               line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    rec["gpu_consumers_before"] = P5V.gpu_consumers()
    print(f"[smoke] commit: {why}")
    print(f"[smoke] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')} "
          f"mem={pf.get('mem_median')}")
    print(f"[smoke] GPU 上其它 C+G 进程={rec['gpu_consumers_before'].get('count')}")
    if (not ok or pf.get("qualified") is not True) and not ALLOW_UNQUALIFIED:
        rec["stage"] = "refused"
        rec["skipped"] = f"环境不合格：commit_ok={ok} pstate={pf.get('qualified')}"
        _write(rec, run_id)
        print(f"[smoke] !! 环境不合格，未加载：{rec['skipped']}")
        return 3
    rec["allow_unqualified"] = ALLOW_UNQUALIFIED

    from .lowmem_load import load_ours_bf16
    reg: CheckRegistry | None = None      # 先声明：FinitenessError 处理器需要它（避免 NameError 掩盖原始错误）
    try:
        # **关闭所有诊断精度开关**：本 smoke 只回答"能否完成完整双模型流程"，
        # 不许把诊断用的精度干预混进来（它们只在 §9 的定位实验里启用）。
        rec["diag_precision"] = M.set_diag_preset("eager")
        rec["profiler_used"] = False              # 本工具不使用 profiler
        rec["default_strategy_at_start"] = M.attention_strategy()

        mem("before_load", rec)
        t0 = time.perf_counter()
        tgt, tok, tinfo = load_ours_bf16(TARGET)
        rec["stage_target_loaded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mem("after_target_load", rec)
        _write(rec, run_id)                       # 逐阶段落盘：target 加载完成
        if MODE == "ar":
            # **AR 必须是 target-only**：不加载 draft ⇒ 不可能是 dual-resident 冒充的基线
            drf, dinfo = None, dict(load_path="not_loaded", weights_bitmatch=True,
                                    mismatch_keys=[], n_params_checked=None,
                                    cpu_peak_note="AR 模式不加载 draft")
            rec["resident_models"] = ["target"]
        else:
            drf, _t2, dinfo = load_ours_bf16(DRAFT)
            rec["resident_models"] = ["target", "draft"]
        rec["load"] = dict(target=tinfo, draft=dinfo, load_s=time.perf_counter() - t0)
        rec["weights_bitmatch"] = bool(tinfo["weights_bitmatch"] and dinfo["weights_bitmatch"])
        rec["weights_checked"] = dict(target=tinfo["n_params_checked"],
                                      draft=dinfo["n_params_checked"])
        rec["stage"] = "loaded"
        rec["stage_draft_loaded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mem("after_load", rec)
        _write(rec, run_id)                       # 加载是历史崩溃高发点
        if not rec["weights_bitmatch"]:
            print("[smoke] !! 权重校验未通过 -> 停止（不产生后续数字）")
            rec["stage"] = "aborted_weight_mismatch"
            _write(rec, run_id)
            return 2

        tgt.set_kv_format(spec_dec._fmt(KV))
        # `set_kv_format` 收的是**引擎名**（'fp16'/'v_fp8'/'fp8'），不是实验名（'base_bf16'）
        if drf is not None:
            drf.set_kv_format("fp16")   # draft head_dim=64 < GROUP_V=128 ⇒ 只能 base(bf16 K/V)
        rec["kv_format"] = dict(target=spec_dec._fmt(KV),
                                draft=("fp16" if drf is not None else "not_loaded"))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()

        full = P5V.load_ids(tok)
        ids = full[:, :CTX].contiguous()
        prompt_len = int(ids.shape[1])
        rec["input"] = dict(ctx=int(ids.shape[1]), ids_hash=P5V._gen_hash(
            [int(x) for x in ids[0].tolist()]),
            source="wikitext-2-raw test 前缀（与其它 G5 工具同源）",
            vocab_size=int(tgt.embed_tokens.weight.shape[0]),
            dtypes=dict(model=str(next(tgt.parameters()).dtype),
                        target_kv_payload=KV, draft_kv_payload="base_bf16"))
        print(f"[smoke] 输入 {ids.shape[1]} token; 开始分块 prefill (chunk={CHUNK}) ...", flush=True)

        # ---- 阶段标注 + 决策点有限性检查：**统一用 smoke_hooks**（与轻量集成测试同一套代码）----
        # 为什么要抽出去：任务要求"每个实际用于候选选择的 logits 必须在 argmax 之前检查"，
        # 而 `spec_dec._propose` 循环内的中间步**只能**通过包装 `QwenLM._run` 覆盖；
        # 抽成模块后可用 stub 模型做**不加载大模型**的集成测试（smoke_midstep_selftest_cpu.py）。
        acc: dict = {}                           # 分块 prefill 的累计耗时（诊断用）
        inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
        reg = CheckRegistry()
        stage: dict = {}

        def on_prefill(key, dt):
            stage[f"{key}_prefill_s"] = dt
            mem(f"after_{key}_prefill", rec)
            stage[f"{key}_prefill_done"] = True
            rec["stage"] = f"{key}_prefill_done"
            _write(rec, run_id)                   # 逐阶段落盘

        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner,
                              on_prefill=on_prefill)
        hooks.install()
        rounds: list[dict] = []
        eos_id = tok.eos_token_id
        gen_ids: list[int] = []
        ar_inv: list[dict] = []
        try:
            if MODE == "ar":
                # ---------------- AR（target-only，**不是** dual-resident）----------------
                # 单 token 自回归：每步 `_run` 的 logits 立即被 argmax 使用 ⇒ 必须先检查。
                #
                # ⚠️ **必须与生产 AR 语义一致**：引擎自己的 AR 单步 `spec_dec._step` 是在
                # `torch.inference_mode()` 里调 `model._run` 的（`spec_dec.py:45-51`），
                # 分块 prefill（`P5V.chunked_prefill`）同样在 inference_mode 内。
                # 本工具首跑 8K AR 时**漏了这一步**，于是同一个前向在 inference_mode 之外执行，
                # 8K 解码到中途 `pt_reserved` 从 5722 MiB 涨到 22924 MiB 并 OOM
                # （见 output/g5/attn/smoke8k_cudnn_prefill_20260927T134045Z.json，已保留为失败样本）。
                # 这里补齐 inference_mode **不是**放宽条件，而是**恢复**生产计算语义。
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.inference_mode(), M.attention_phase("prefill"):
                    lg, cache = inner(tgt, ids)
                reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg",
                                 where="AR prefill 返回：首步 argmax 的输入")
                stage["target_prefill_done"] = True
                stage["target_prefill_s"] = time.perf_counter() - t1
                mem("after_target_prefill", rec)
                rec["stage"] = "target_prefill_done"
                _write(rec, run_id)
                nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
                # ⚠️ **记账顺序必须与生产 AR 一致**：引擎的 `QwenLM.generate` 与 8K 探针的 AR
                # （`g5_8k_spec_variance_probe.formal_pass`）都是**先记录本步 token、再喂给下一步**：
                #     tok = argmax(cur); gen.append(tok); ... cur, cache = _step(tgt, tok, cache)
                # 本工具首版写成"先 `_run` 再记录"，于是 **prefill 决策出的第一个 token 从未被记录**，
                # 整条序列错位一位 —— 表现出来就是"AR 与 SpecDec 在 index 0 就分歧"。
                # 那是**工具记账缺陷**（已保留失败/错位样本为证），**不是**引擎或 cuDNN 分歧；
                # 修法是恢复生产记账顺序，**不改任何计算语义**。
                decode_runs = 0
                for i in range(MAXNEW):
                    gen_ids.append(nxt)                      # ← 先记录（本步 token 来自上一步的 logits）
                    clen = _cache_len(cache)
                    ar_inv.append(dict(step=i, gen_len=len(gen_ids), decode_runs=decode_runs,
                                       cache_len=clen, expected=prompt_len + decode_runs,
                                       pt_allocated_mib=torch.cuda.memory_allocated() / 2 ** 20,
                                       pt_reserved_mib=torch.cuda.memory_reserved() / 2 ** 20,
                                       cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                       ok=(clen == prompt_len + decode_runs)))
                    if eos_id is not None and nxt == eos_id:
                        break
                    if i == MAXNEW - 1:
                        break
                    with torch.inference_mode(), M.attention_phase("decode"):
                        lg, cache = tgt._run(
                            tgt.embed_tokens(torch.tensor([[nxt]], device=ids.device)),
                            cache, need_logits=True)
                    decode_runs += 1
                    # **决策前检查**：下一次 argmax 用的就是这份 logits
                    reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                                     round_=i, step=decode_runs,
                                     where=f"AR 第 {decode_runs} 次解码 forward 返回："
                                           f"紧接被 argmax 使用")
                    nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
                torch.cuda.synchronize()
                wall = time.perf_counter() - t1
                out = dict(gen=gen_ids, ids=list(ids[0].tolist()) + gen_ids,
                           stats=dict(mode="AR", decode_steps=decode_runs,
                                      rounds=len(gen_ids)))
            else:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos_id,
                                                  target_kv=KV, draft_kv="base_bf16",
                                                  log=lambda d: rounds.append(dict(d)))
                torch.cuda.synchronize()
                wall = time.perf_counter() - t1
        finally:
            hooks.remove()

        mem("after_decode", rec)
        rec["stage_decode_done"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["stage_flags"] = dict(stage)
        rec["rounds"] = rounds
        # ---- 冻结的逐轮不变量（与 spec_dec 的契约一致）----
        inv = []
        for r in rounds:
            pre, kp, m = r["pre"], r["Kp"], r["m"]
            exp_tgt = pre + (m + 1 if m < kp else kp)
            inv.append(dict(round=r["round"], pre=pre, Kp=kp, m=m,
                            tgt_len=r["tgt_len"], drf_len=r["drf_len"],
                            tgt_len_ok=(r["tgt_len"] == exp_tgt),
                            drf_len_ok=(r["drf_len"] == exp_tgt),
                            lengths_equal=(r["tgt_len"] == r["drf_len"])))
        rec["round_invariants"] = dict(mode=MODE, per_round=inv,
                                       all_ok=bool(inv) and all(x["tgt_len_ok"] and x["drf_len_ok"]
                                                                and x["lengths_equal"] for x in inv),
                                       rule="tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)")
        if MODE == "ar":
            # AR：不变量 = cache 长度 == prompt + 已生成数（逐门步核对）
            inv = ar_inv
            rec["round_invariants"] = dict(mode="ar", per_round=inv,
                                           all_ok=bool(inv) and all(x["ok"] for x in inv),
                                           rule="cache_len == prompt_len + len(gen)（每步）")
        gen = [int(x) for x in out["gen"]]
        vocab = int(tgt.embed_tokens.weight.shape[0])
        attn_ok = None
        try:
            M.assert_strategy_used()
            attn_ok = "ok"
        except RuntimeError as e:
            attn_ok = f"RuntimeError: {e}"
        a_stats = M.attention_stats()
        # **确认新路径确实启用、且没有错误扩展到验证阶段**
        fused_by_phase = a_stats.get("fused_by_phase", {})
        eager_by_phase = a_stats.get("eager_by_phase", {})
        if MODE == "ar":
            dispatch_ok = (sorted(k for k, v in fused_by_phase.items() if v > 0) == ["prefill"]
                           and all(fused_by_phase.get(p, 0) == 0
                                   for p in ("propose", "verify", "repair", "decode"))
                           and eager_by_phase.get("decode", 0) > 0)
        else:
            dispatch_ok = (sorted(k for k, v in fused_by_phase.items() if v > 0) == ["prefill"]
                           and all(fused_by_phase.get(p, 0) == 0
                                   for p in ("propose", "verify", "repair", "decode"))
                           and eager_by_phase.get("verify", 0) > 0)
        # ---- 停止原因：**不能少生成却当固定长度完成** ----
        gen_len = len(gen)
        if gen and eos_id is not None and gen[-1] == eos_id:
            stop_reason = "eos"
        elif gen_len >= MAXNEW:
            stop_reason = "max_new_reached"
        else:
            stop_reason = "other_short"       # 既没到 max_new 也没 EOS ⇒ 必须判失败
        rec["run"] = dict(gen=gen, gen_len=gen_len, max_new=MAXNEW, eos_id=eos_id,
                          stop_reason=stop_reason,
                          gen_ids_in_vocab=bool(all(0 <= g < vocab for g in gen)),
                          # 历史字段口径澄清：旧的 `logits_all_finite` **只**覆盖两模型 prefill 的 logits，
                          # 名称夸大覆盖范围 ⇒ 新运行改用下面的显式命名（**旧 JSON 不改写**）
                          prefill_logits_finite=all(
                              reg.calls.get(("prefill", m), {}).get("ok", False)
                              for m in ("target", "draft")),
                          prefill_s=acc.get("t"),
                          prefill_target_s=acc.get("target"), prefill_draft_s=acc.get("draft"),
                          stage_prefill_s=rec.get("stage_flags"),
                          wall_s=wall, specdec_stats=out["stats"],
                          attention_ok=attn_ok, attention_stats=a_stats,
                          dispatch_only_prefill=dispatch_ok,
                          s_max_by_phase=a_stats.get("s_max_by_phase"),
                          eager_reason_kinds=a_stats.get("eager_reason_kinds"),
                          note="计时含插桩，仅诊断；禁止进入任何 speedup 表。"
                               "父子阶段耗时**不可相加**（prefill 是 wall 的子集）。")
        # ---- 通过判定：**执行结束 ≠ 验证通过** ----
        # ---- 决策步覆盖核对（任务要求：阶段出现过一次检查 ≠ 完整覆盖）----
        if MODE == "ar":
            dlg = reg.calls.get(("decode", "target"), {}).get("role_counts", {}).get("decode_lg", 0)
            steps_n = int(out["stats"].get("decode_steps", len(gen_ids)))
            cov = dict(name="ar_decode_step_coverage", ok=(dlg == steps_n),
                       detail=f"decode_lg 检查={dlg} vs 实际解码步数={steps_n}")
            extra = [cov]
        else:
            cov = verify_decision_coverage(
                reg, draft_calls_expected=int(out["stats"].get("draft_calls", 0)),
                rounds_expected=int(out["stats"].get("rounds", 0)))
            cov["name"] = "decision_step_coverage"
            extra = [cov]
        rec["decision_coverage"] = cov
        rec["checks"] = evaluate(
            registry=reg, mismatch_rounds=int(out["stats"].get("mismatch_rounds", 0)),
            extra_required=extra, mode=MODE,
            weights_ok=rec["weights_bitmatch"],
            weights_detail=f"target {tinfo['n_params_checked']} + draft {dinfo['n_params_checked']} 逐位校验",
            invariants_ok=rec["round_invariants"]["all_ok"],
            invariants_detail=f"{len(inv)} 轮，全部满足 tgt_len==drf_len==pre+(m+1 if m<Kp else Kp)"
                              if inv else "无轮次记录",
            dispatch_ok=dispatch_ok,
            dispatch_detail=f"fused={fused_by_phase} eager={eager_by_phase}",
            attention_assert=attn_ok, gen_len=gen_len, max_new=MAXNEW, eos_id=eos_id,
            gen=gen, stop_reason=stop_reason, exception=None,
            alloc_ok=bool(rec.get("mem_timeline")),
            alloc_detail=f"时间线 {len(rec.get('mem_timeline', []))} 个时点")
        passed = bool(rec["checks"]["passed"])
        print(f"[smoke] 完成：gen={gen_len} stop_reason={stop_reason} "
              f"有限性检查 {reg.n_checked} 次（覆盖 {sorted(reg.calls)}）")
        print(f"[smoke] 阶段分流：fused_by_phase={fused_by_phase} "
              f"eager_by_phase={eager_by_phase} 仅prefill融合={dispatch_ok}")
        print(f"[smoke] 逐轮不变量 all_ok={rec['round_invariants']['all_ok']} "
              f"rounds={len(rounds)}")
        print(f"[smoke] 验证：passed={passed} 必需检查={rec['checks']['n_required']} "
              f"未覆盖={rec['checks']['uncovered']} 不适用={rec['checks']['not_applicable']} "
              f"失败={[f['name'] for f in rec['checks']['failures']]}")
        rec["stage"] = "run_ended" if passed else "run_ended_checks_failed"
        rec["verdict"] = dict(
            ran=True, passed=passed,
            weights_bitmatch=rec["weights_bitmatch"],
            prefill_logits_finite=rec["run"]["prefill_logits_finite"],
            finiteness_coverage_note="新运行覆盖 prefill/propose/verify/repair 的**决策用** logits；"
                                     "旧 JSON 的 `logits_all_finite` 仅指 prefill，口径已在此澄清",
            gen_in_vocab=rec["run"]["gen_ids_in_vocab"],
            stop_reason=stop_reason, gen_len=gen_len,
            attention_assert=attn_ok,
            dispatch_only_prefill=dispatch_ok,
            round_invariants_ok=rec["round_invariants"]["all_ok"],
            pt_peak_reserved_mib=rec["mem_timeline"][-1]["pt_peak_reserved_mib"],
            pt_peak_allocated_mib=rec["mem_timeline"][-1]["pt_peak_allocated_mib"],
            physical_limit_mib=rec["mem_timeline"][-1]["cuda_total_mib"],
            # 预登记口径：这只是"PyTorch 记账峰值 ≤ 物理容量"的**事实记录**，
            # **不得**据此推断整卡安全（另有 20+ 消费者；见 smi_mem_used / cuda_free 各时点）。
            pt_reserved_within_physical_line=bool(
                rec["mem_timeline"][-1]["pt_peak_reserved_mib"]
                <= rec["mem_timeline"][-1]["cuda_total_mib"]),
            not_a_whole_card_safety_verdict=True,
            claim="**单次跑完只代表『能跑』**：单一 run ⇒ 不构成正确性/稳定性/性能结论；"
                  "allocated ≠ 整卡物理占用，reserved 超物理容量不单独证明分页。")
        _write(rec, run_id)
        del tgt, drf
        torch.cuda.empty_cache()
        if not passed:
            print("[smoke] !! 验证未通过（执行已结束）："
                  + ", ".join(f["name"] for f in rec["checks"]["failures"]))
            return 5
    except FinitenessError as e:
        # 非有限值：**立即**停止（已在 check_finite 里落记录），落盘并非零退出
        M.set_attention_strategy("eager")
        M.reset_attention_stats()
        rec["stage"] = "finiteness_failed"
        rec["checks"] = evaluate(
            registry=reg if reg is not None else CheckRegistry(),
            mismatch_rounds=-1,          # -1 = "未知"（run 被中止）⇒ 不作有条件必需
            weights_ok=False, weights_detail="未完成",
            invariants_ok=False, invariants_detail="未完成", dispatch_ok=False,
            dispatch_detail="未完成", attention_assert="not_reached", gen_len=0, max_new=MAXNEW,
            eos_id=None, gen=[], stop_reason="aborted_non_finite",
            exception=f"FinitenessError: {e}", alloc_ok=bool(rec.get("mem_timeline")),
            alloc_detail=f"{len(rec.get('mem_timeline', []))} 个时点")
        rec["non_finite"] = dict(error=str(e), first_bad=(reg.first_bad if reg else None),
                                 coverage=(reg.coverage() if reg else None))
        mem("on_non_finite", rec)
        _write(rec, run_id)
        print(f"[smoke] !! 非有限值，已在决策前停止并落盘：{e}")
        return 5
    except torch.OutOfMemoryError as e:
        rec["stage"] = "oom"
        rec["oom"] = str(e)
        rec["traceback"] = traceback.format_exc()      # 缺这条时无法定位到具体语句（首跑教训）
        rec["alloc_on_oom"] = P5V.allocator_state()
        _write(rec, run_id)
        print(f"[smoke] !! OOM，已落盘并停止（不升级测试）：{e}")
        return 2
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        mem("on_crash", rec)
        _write(rec, run_id)
        print(f"[smoke] !! 崩溃（已落盘）：\n{rec['traceback']}")
        return 2
    finally:
        M.set_attention_strategy("eager")
        M.reset_attention_stats()

    _write(rec, run_id)
    print(f"[smoke] -> {_out_path(run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
