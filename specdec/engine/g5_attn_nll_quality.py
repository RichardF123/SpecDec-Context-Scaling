"""g5_attn_nll_quality.py — attention 后端的**诊断性质量检查**（mean NLL / PPL，GPU）。

定位：这是**小规模诊断**，**不是**正式质量结论。目的只有一个 ——
在真实文本上回答"换 attention 后端之后，语言建模质量变了多少？"
并与 **HF 参考**（明确指定 attention 实现）以及**未经量化的同一路径**对照。

固定量：同一段 token ids、同一窗口、同一预测位置、同一计分方式。

预测器
  hf_eager   HF `attn_implementation="eager"`（**显式指定并记录**）
  hf_sdpa    HF `attn_implementation="sdpa"` —— **只表示"已配置为 sdpa"**：本工具**不核实** HF
             内部实际选中了哪个 kernel（flash / mem-efficient / cudnn / math 之一）
             ⇒ 它只提供"另一条后端路径"与"后端差异的自然量级"参照，
             **不得**当作"已确认的 cuDNN 对照"
  ours_eager / ours_cudnn       自建引擎，`base_bf16`
  ours_eager_vfp8 / ours_cudnn_vfp8  自建引擎，`v_fp8`（**与对应的未量化路径比**）
  每个 ours 预测器各跑 `chunk=L`（单块、无历史）与 `chunk=L/2`（两块、非方阵 offset mask）

计分（**唯一口径，写死并断言**）
  `logits[0, t]` 预测 `ids[0, t+1]`；
  只对 `t ∈ [W-1, W-1+N)` 计分 ⇒ 被评分的 token 恰好是 `ids[0, W : W+N]`，共 **N** 个。
  脚本会断言：各预测器的 **N 相同**、**被评分 token 的 id 相同（同一 hash）**、**logits 全有限**。

因果/泄漏检查（诊断，不设人造阈值）
  同一模型两次前向：长度 `W` 与长度 `L`，比较二者在位置 `W-1` 的 logits。
  位置 `W-1` 两次都只能看到 `≤ W-1` 的 token，因此差异反映的是**不同序列长度下 kernel 归约**的差别；
  如果这个数达到 O(1)，说明因果掩码有问题。**只报数，不判 PASS/FAIL。**

纪律
  - 不覆盖也不重新解释旧的预登记判据；本脚本产出**新的独立诊断量**；
  - 默认策略仍为 `eager`；不碰 chunk/k/量化方案/logits 策略/allocator/缓存布局；
  - 未量化基线口径：`v_fp8` 只与 **同后端的 `base_bf16`** 比（HF 不参与 v_fp8 的对照）。

用法
  python -m specdec.engine.g5_attn_nll_quality
环境变量
  NLL_W=256（上下文长度）NLL_N=128（评分 token 数）NLL_KV=base_bf16,v_fp8
  NLL_HF=1 是否跑 HF 对照（默认 1）NLL_ALLOW_UNQUALIFIED=1
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import results_schema as S
from . import model_self as M
from . import g5_8k_spec_variance_probe as P5V
from .g4_b4_cache_tests import _from_pretrained, load_ours

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET = P5V.TARGET

W = int(os.environ.get("NLL_W", "256"))          # 上下文长度
N = int(os.environ.get("NLL_N", "128"))          # 评分 token 数
L = W + N
KVS = [x.strip() for x in os.environ.get("NLL_KV", "base_bf16,v_fp8").split(",") if x.strip()]
RUN_HF = S.env_flag("NLL_HF", True)
LOWMEM = S.env_flag("NLL_LOWMEM_LOAD", True)   # 降峰值加载器（权重逐位校验，见 lowmem_load.py）
ALLOW_UNQUALIFIED = S.env_flag("NLL_ALLOW_UNQUALIFIED", False)


def _hash_ids(ids) -> str:
    s = ",".join(str(int(x)) for x in ids)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def nll_of(logits: torch.Tensor, ids: torch.Tensor, *, logit_offset: int,
           local_start: int, n: int) -> torch.Tensor:
    """返回每 token NLL（CPU fp32）。

    **必须显式给偏移**，因为分块 prefill 的 `_run` 只在最后一块返回 logits：
      `logits[0, t]` 对应**全局位置** `logit_offset + t`，它预测 `ids[0, logit_offset + t + 1]`。
    评分范围：局部 `[local_start, local_start+n)` ⇒ 全局位置 `[W-1, W-1+N)`
    （全局位置 p 的 logits 预测 p+1）⇒ 被评分的 token 恰好是 `ids[0, W : W+N]`。
    """
    lp = F.log_softmax(logits[0, local_start:local_start + n].float(), dim=-1)
    t0 = logit_offset + local_start + 1
    tgt = ids[0, t0:t0 + n]
    assert tgt.shape[0] == n and lp.shape[0] == n, (tgt.shape, lp.shape, n, logit_offset, local_start)
    return (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).detach().cpu()


def _summarise(nll: torch.Tensor, ids, n: int) -> dict:
    return dict(n=n, mean_nll=float(nll.mean()), ppl=float(torch.exp(nll.mean())),
                mean_nll_repr=repr(float(nll.mean())),
                min_nll=float(nll.min()), max_nll=float(nll.max()),
                nll_sum=float(nll.sum()),
                target_ids_hash=_hash_ids(ids[0, W:W + n].tolist()),
                per_token_nll=[float(x) for x in nll])


# ------------------------------------------------------------------ HF
def _free_report(tag: str) -> dict:
    """显式报告显存与"PyTorch 还占着多少"，避免"以为释放了其实没释放"。

    首跑就是在这一步暴露的：两个 HF pass 跑完后 PyTorch 仍报告 2.81 GiB allocated，
    随后 `ours.to("cuda")` 只因 28 MiB 就 OOM（WDDM 超额订阅下 free 数字并不等于可用）。
    """
    a = torch.cuda.memory_allocated() / 2 ** 30
    r = torch.cuda.memory_reserved() / 2 ** 30
    free, total = torch.cuda.mem_get_info()
    print(f"[nll] mem {tag}: allocated={a:.2f} GiB reserved={r:.2f} GiB "
          f"cuda_free={free / 2 ** 30:.2f} GiB / {total / 2 ** 30:.2f} GiB", flush=True)
    return dict(tag=tag, allocated_gib=a, reserved_gib=r, cuda_free_gib=free / 2 ** 30,
                cuda_total_gib=total / 2 ** 30)


def hf_pass(ids, attn_impl: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = _from_pretrained(AutoConfig, TARGET, trust_remote_code=True)
    t0 = time.perf_counter()
    hf = _from_pretrained(AutoModelForCausalLM, TARGET, config=cfg,
                          attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
                          trust_remote_code=True)
    load_s = time.perf_counter() - t0
    actual = getattr(hf.config, "_attn_implementation", None)
    hf = hf.to("cuda").eval()
    try:
        with torch.inference_mode():
            out = hf(input_ids=ids.to("cuda"))
        logits = out.logits
        finite = bool(torch.isfinite(logits).all())
        nll = nll_of(logits, ids.to("cuda"), logit_offset=0, local_start=W - 1, n=N)
        r = dict(predictor=f"hf_{attn_impl}", engine="huggingface",
                 attn_implementation_requested=attn_impl, attn_implementation_actual=actual,
                 attn_kernel_verified=False,
                 attn_kernel_note="只记录并核对了 attn_implementation；"
                                  "**未**核实 HF 内部实际选中的 kernel"
                                  "（flash/mem-efficient/cudnn/math）",
                 dtype="bfloat16", load_s=load_s, logits_finite=finite,
                 logit_offset=0, logits_shape=list(logits.shape),
                 **_summarise(nll, ids.to("cuda"), N))
        del logits, out
    finally:
        del hf
        gc.collect()
        gc.collect()                     # 循环引用：HF 模块树需要多轮回收
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        r["mem_after_free"] = _free_report(f"hf_{attn_impl}_after_free")
    return r


# ------------------------------------------------------------------ ours
def ours_pass(model, ids, *, kv: str, strategy: str, chunk: int) -> dict:
    from . import spec_dec
    engine_fmt = spec_dec._fmt(kv)
    model.set_kv_format(engine_fmt)
    M.set_attention_strategy(strategy)
    M.reset_attention_stats()
    t0 = time.perf_counter()
    with torch.inference_mode(), M.attention_phase("prefill"):
        lg, cache, last_st = None, None, 0
        for st in range(0, L, chunk):
            last_st = st
            lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                   need_logits=(st + chunk >= L))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    st_attn = M.attention_stats()
    finite = bool(torch.isfinite(lg).all())
    # 分块时 `lg` 只覆盖全局位置 [last_st, L)；被评分的全局位置是 [W-1, W-1+N)
    local_start = (W - 1) - last_st
    assert 0 <= local_start and local_start + N <= lg.shape[1], \
        f"logits 覆盖范围不足：last_st={last_st} lg_len={lg.shape[1]} local_start={local_start} N={N}"
    nll = nll_of(lg, ids, logit_offset=last_st, local_start=local_start, n=N)
    asserted = None
    if strategy != "eager":
        try:
            M.assert_strategy_used()
            asserted = "ok"
        except RuntimeError as e:
            asserted = f"RuntimeError: {e}"
    return dict(predictor=f"ours_{strategy}" + ("" if kv == "base_bf16" else f"_{kv}"),
                engine="self", kv=kv, engine_kv_format=engine_fmt, strategy=strategy,
                chunk=chunk, attn_implementation=("cudnn_prefill" if strategy != "eager" else "eager"),
                forward_s=dt, logits_finite=finite, logits_shape=list(lg.shape),
                logit_offset=last_st, local_start_in_logits=local_start,
                attention_stats=st_attn, assert_strategy_used=asserted,
                **_summarise(nll, ids, N))


def causality_check(model, ids) -> dict:
    """**真·泄漏检查**：位置 `W-1` 的 logits 是否受"位置 ≥ W 的 token"影响。

    三次前向，读同一个全局位置 `W-1` 的 logits：
      A 长度 W            （没有未来 token）
      C 长度 L，未来 token = 原文
      B 长度 L，未来 token = **随机替换**
    位置 `W-1` 在三次里都只能看到 `≤ W-1`，因此：
      `|A − C|` = 纯粹的"序列长度不同 ⇒ kernel 归约不同"的量（不是泄漏）
      `|B − C|` = **长度相同、只有未来 token 不同** ⇒ 这个差就是**泄漏**（若掩码正确应为 0 / ULP 级）
    只报数并把差值折算成该位置参考幅度的 ULP，**不设人造阈值**。
    """
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids_b = ids.clone()
    ids_b[0, W:] = torch.randint(0, 1000, (L - W,), generator=g).to(ids.device)
    with torch.inference_mode(), M.attention_phase("prefill"):
        la, _ = model._run(model.embed_tokens(ids[:, :W]), None, need_logits=True)
        lc, _ = model._run(model.embed_tokens(ids[:, :L]), None, need_logits=True)
        lb, _ = model._run(model.embed_tokens(ids_b[:, :L]), None, need_logits=True)
    a, b, c = la[0, W - 1].float(), lb[0, W - 1].float(), lc[0, W - 1].float()
    ref = float(c.abs().max())
    top = torch.topk(c, 2).values
    ulp = max(ref, 1e-30) * 2 ** -8          # bf16 在该幅度下的一个 ULP
    return dict(position=W - 1, A_len=W, B_len=L, C_len=L,
                future_tokens_replaced=int(L - W),
                A_vs_C_max_abs=float((a - c).abs().max()),
                B_vs_C_max_abs=float((b - c).abs().max()),      # ← 泄漏指标
                B_vs_C_in_bf16_ulp=float((b - c).abs().max()) / ulp,
                A_vs_C_in_bf16_ulp=float((a - c).abs().max()) / ulp,
                ref_absmax=ref,
                argmax_A=int(a.argmax()), argmax_B=int(b.argmax()), argmax_C=int(c.argmax()),
                argmax_all_same=bool(int(a.argmax()) == int(b.argmax()) == int(c.argmax())),
                c_top1_minus_top2=float(top[0] - top[1]))


# ------------------------------------------------------------------ main
def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_attn_nll_quality",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     config=dict(W=W, N=N, L=L, kvs=KVS, run_hf=RUN_HF,
                                 scoring="logits[t] predicts ids[t+1]; score t in [W-1, W-1+N)"
                                         " => scored tokens = ids[W:W+N]",
                                 scope="**小规模诊断**，不是正式质量结论"),
                     status="not_a_formal_quality_conclusion")
    print(f"[nll] run_id={run_id} W={W} N={N} L={L} kvs={KVS} hf={RUN_HF}")
    guard_line = float(P5V.MIN_COMMIT_GIB)
    rec["commit_guard_line_gib"] = guard_line
    rec["commit_guard_overridden"] = bool(guard_line < 16.0)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=ok, why=why, system=P5V.system_memory())
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    print(f"[nll] commit: {why}")
    print(f"[nll] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')} "
          f"mem={pf.get('mem_median')}")
    if (not ok or pf.get("qualified") is not True) and not ALLOW_UNQUALIFIED:
        rec["skipped"] = f"环境不合格：commit_ok={ok} pstate={pf.get('qualified')}"
        _write(rec, run_id)
        print(f"[nll] !! 环境不合格，未加载模型：{rec['skipped']}")
        return 3
    rec["allow_unqualified"] = ALLOW_UNQUALIFIED

    preds = []
    try:
        # ---- 输入（一次取好，全程复用同一段 token）----
        from transformers import AutoTokenizer
        tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
        full = P5V.load_ids(tok)
        assert full.shape[1] >= L, (full.shape, L)
        ids = full[:, :L].contiguous().to("cuda")
        rec["input"] = dict(source="wikitext-2-raw test 前 %d token（与其它 G5 工具同源）" % L,
                            L=L, ids_hash=_hash_ids(ids[0].tolist()),
                            scored_hash_expected=_hash_ids(ids[0, W:W + N].tolist()),
                            first10=[int(x) for x in ids[0, :10].tolist()])
        print(f"[nll] 输入 ids_hash={rec['input']['ids_hash']} scored={rec['input']['scored_hash_expected']}")

        if RUN_HF:
            for impl in ("eager", "sdpa"):
                r = hf_pass(ids.cpu(), impl)
                preds.append(r)
                print(f"[nll] hf_{impl:<5} actual={r['attn_implementation_actual']} "
                      f"mean_nll={r['mean_nll']:.6f} ppl={r['ppl']:.4f} "
                      f"finite={r['logits_finite']}")
        # ---- ours（模型只加载一次；HF 已释放）----
        if LOWMEM:
            from .lowmem_load import load_ours_bf16
            rec["mem_before_ours_load"] = _free_report("before_ours_load")
            model, _tok2, load_info = load_ours_bf16(TARGET)
        else:
            model, _tok2 = load_ours(TARGET)
            load_info = dict(load_path="g4_b4_cache_tests.load_ours", weights_bitmatch=None,
                             mismatch_keys=None, n_params_checked=None,
                             cpu_peak_note="共享加载器（不降峰值）")
        rec["load"] = load_info
        rec["model_dtype"] = str(next(model.parameters()).dtype)
        rec["alloc_after_load"] = P5V.allocator_state()
        rec["causality_check"] = causality_check(model, ids)
        c = rec["causality_check"]
        print(f"[nll] 泄漏检查 @位置{W-1}：**|B−C|={c['B_vs_C_max_abs']:.3e} "
              f"（{c['B_vs_C_in_bf16_ulp']:.2f} ULP，未来 token 不同、长度相同 ⇒ 泄漏指标）**；"
              f"|A−C|={c['A_vs_C_max_abs']:.3e}（{c['A_vs_C_in_bf16_ulp']:.2f} ULP，长度效应）; "
              f"argmax 三者同={c['argmax_all_same']} ref_absmax={c['ref_absmax']:.3f}")
        for kv in KVS:
            for strategy in ("eager", "cudnn_prefill"):
                for chunk in (L, L // 2):
                    r = ours_pass(model, ids, kv=kv, strategy=strategy, chunk=chunk)
                    preds.append(r)
                    print(f"[nll] {r['predictor']:<22} chunk={chunk:<4} "
                          f"mean_nll={r['mean_nll']:.6f} ppl={r['ppl']:.4f} "
                          f"finite={r['logits_finite']} "
                          f"fused={r['attention_stats']['fused_by_phase'].get('prefill', 0)}")
        del model
        torch.cuda.empty_cache()
        rec["stage"] = "complete"
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        _write(rec, run_id)
        print(f"[nll] !! 崩溃（已落盘）：\n{rec['traceback']}")
        return 2
    finally:
        M.set_attention_strategy("eager")
        M.reset_attention_stats()

    rec["predictors"] = preds
    _consistency(preds, rec)
    _write(rec, run_id)
    _print(rec)
    print(f"[nll] -> {_out_path(run_id)}")
    return 0


def _consistency(preds: list[dict], rec: dict) -> None:
    """口径一致性断言：N 相同、被评分 token 的 id 相同、logits 全有限。"""
    ns = {p["n"] for p in preds}
    hs = {p["target_ids_hash"] for p in preds}
    fin = all(p["logits_finite"] for p in preds)
    rec["consistency"] = dict(n_values=sorted(ns), n_consistent=(len(ns) == 1),
                              target_hash_values=sorted(hs),
                              target_ids_consistent=(len(hs) == 1),
                              all_logits_finite=fin,
                              predictors=len(preds))
    print(f"[nll] 一致性：N={sorted(ns)} 被评分 token hash={sorted(hs)} 全有限={fin}")


def _print(rec: dict) -> None:
    print("\n== mean NLL / PPL（小规模诊断，**不是**正式质量结论）==")
    print(f"  {'预测器':<24}{'chunk':>6}{'mean NLL':>12}{'PPL':>10}")
    for p in rec["predictors"]:
        print(f"  {p['predictor']:<24}{str(p.get('chunk', '-')):>6}"
              f"{p['mean_nll']:>12.6f}{p['ppl']:>10.4f}")
    by = {p["predictor"]: p for p in rec["predictors"]}
    print("\n== 关键对照（Δmean NLL，正数 = 前者更差）==")
    def d(a, b):
        if a in by and b in by:
            return f"  {a} − {b}: {by[a]['mean_nll'] - by[b]['mean_nll']:+.6f}"
        return None
    for pair in (("hf_eager", "hf_sdpa"),
                 ("ours_eager", "ours_cudnn_prefill"),
                 ("ours_eager", "hf_eager"),
                 ("ours_eager_vfp8", "ours_eager"),
                 ("ours_cudnn_prefill_vfp8", "ours_cudnn_prefill")):
        line = d(*pair)
        if line:
            print(line)


def _out_path(run_id: str) -> Path:
    return OUTDIR / f"nll_quality_{run_id}.json"


def _write(rec: dict, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _out_path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


if __name__ == "__main__":
    raise SystemExit(main())
