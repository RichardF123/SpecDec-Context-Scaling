"""g4_b5_helper_tests.py — B5 分段自测: 先验证 spec_dec 的"零件", 再写主循环

用法: 你按顺序实现 spec_dec.py 里的 4 个小函数, 每实现一个就跑一次本脚本:
    _fmt(kv) / _prefill(model, ids) / _propose(draft, cache, lg, k_eff, eos_id) / _verify(target, cache, cands)
未实现的函数会被跳过(打印 SKIP), 已实现的会跑断言 —— 这样你可以"写完一个验一个"。

测试内容(核心是 T4 的 off-by-one 检查):
  T1 _fmt:          base_bf16->fp16, v_fp8->v_fp8
  T2 _prefill:      logits 形状/finite/cache 长度 == prompt 长度;与 model.forward 末行一致
  T3 _propose:      draft 提议的前 k 个 token == draft.generate 的前 k 个
  T4 _verify:       rows[j] (输入 cands[j]) 预测的正是 cands[j+1](j=0..Kp-2);
                    rows[Kp-1] 预测"全接受之后"的下一个;block cache 长度 == pre+Kp 且一致
Run（交互式 GPU 终端, 单跑; 会加载双模型）: python -m specdec.engine.g4_b5_helper_tests
"""
from __future__ import annotations

import gc
import traceback

import torch

from . import spec_cache
from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
K = 4
PROMPT = "The capital of France is"

# 允许你自己起名(左=测试期望名, 右=候选名列表)
ALIASES = {
    "_fmt": ["_fmt", "fmt_of", "_kv_fmt"],
    "_prefill": ["_prefill", "prefill"],
    "_propose": ["_propose", "propose"],
    "_verify": ["_verify", "verify", "target_block"],
}


def _resolve(mod, key):
    for name in ALIASES[key]:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, None


def main() -> int:
    import importlib
    from transformers import AutoTokenizer
    spec_dec = importlib.import_module("specdec.engine.spec_dec")
    torch.manual_seed(0)

    ok = True
    results = {}

    print("[stage] loading target 1.5B ...", flush=True)
    tgt, tok = load_ours(TARGET)
    print("[stage] loading draft 0.5B ...", flush=True)
    drf, _ = load_ours(DRAFT)
    eos = tok.eos_token_id
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    # ---- T1 _fmt ----
    fn, name = _resolve(spec_dec, "_fmt")
    if fn is None:
        print("[T1] SKIP (未实现 _fmt)", flush=True)
    else:
        t1 = (fn("base_bf16") == "fp16" and fn("v_fp8") == "v_fp8")
        print(f"[T1] {name}: {t1}", flush=True)
        ok &= t1
        results["T1"] = t1

    # ---- T2 _prefill ----
    fn, name = _resolve(spec_dec, "_prefill")
    if fn is None:
        print("[T2] SKIP (未实现 _prefill)", flush=True)
    else:
        lg, cache = fn(tgt, ids)
        L = spec_cache.cache_length(cache)
        spec_cache.check_consistent(cache)
        fin = bool(torch.isfinite(lg).all().item())
        with torch.inference_mode():
            ref = tgt.forward(ids)[:, -1]
        d = (lg[:, -1].to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        t2 = (tuple(lg.shape[:2]) == (1, ids.shape[1]) and fin
              and L == ids.shape[1] and d < 1e-1)
        print(f"[T2] {name}: shape={tuple(lg.shape)} finite={fin} cache_len={L}"
              f"(exp {ids.shape[1]}) vs-forward max|Δ|={d:.3e} -> {t2}", flush=True)
        ok &= t2
        results["T2"] = t2
        del lg, cache, ref
        torch.cuda.empty_cache()

    # draft 的 prefill(供 T3/T4 使用)
    drf_lg, drf_cache = None, None
    if _resolve(spec_dec, "_prefill")[0] is not None:
        drf_lg, drf_cache = _resolve(spec_dec, "_prefill")[0](drf, ids)

    # ---- T3 _propose ----
    fn_p, name_p = _resolve(spec_dec, "_propose")
    if fn_p is None or drf_lg is None:
        print("[T3] SKIP (未实现 _propose 或 _prefill)", flush=True)
    else:
        cands, drf_cache2, prop_lg = fn_p(drf, drf_cache, drf_lg.clone(), K, eos)
        ref_gen = drf.generate(ids, K, eos_id=eos)[0, ids.shape[1]:].tolist()
        t3 = cands == ref_gen
        print(f"[T3] {name_p}: cands={[tok.decode([c])[:6] for c in cands]} "
              f"ref={[tok.decode([c])[:6] for c in ref_gen]} -> {t3}", flush=True)
        ok &= t3
        results["T3"] = t3

    # ---- T4 _verify (对齐/off-by-one 检查) ----
    fn_v, name_v = _resolve(spec_dec, "_verify")
    fn_pf, _ = _resolve(spec_dec, "_prefill")
    if fn_v is None or fn_pf is None:
        print("[T4] SKIP (未实现 _verify 或 _prefill)", flush=True)
    else:
        # 候选 = target 自己的贪心 4 个 -> 期望"全接受"
        cands = tgt.generate(ids, K, eos_id=eos)[0, ids.shape[1]:].tolist()
        # 两份独立 prefill: 一份给 block 验证, 一份给串行参照(引擎 cache 原地推进)
        lg_a, cache_a = fn_pf(tgt, ids)
        lg_b, cache_b = fn_pf(tgt, ids)
        saved_next = lg_a[:, -1]
        pre = spec_cache.cache_length(cache_a)
        rows, blk = fn_v(tgt, cache_a, cands)
        Lb = spec_cache.cache_length(blk)
        spec_cache.check_consistent(blk)
        serial = []
        cur = cache_b
        with torch.inference_mode():
            for t in cands:
                lg_s, cur = tgt._run(
                    tgt.embed_tokens(torch.tensor([[t]], device="cuda")), cur,
                    need_logits=True)
                serial.append(lg_s[0, -1].clone())
        # 判定规则: verify[j] = [saved_next, rows[0..j-1]] 判定 cands[j]
        verify = [saved_next[0]] + [rows[j] for j in range(len(cands) - 1)]
        g = [int(v.argmax()) for v in verify]
        accept_ok = (g == cands)
        # 同上下文对齐: rows[j] (上下文 prefix+cands[0..j]) == serial[j]
        same_ctx = [int(rows[j].argmax()) == int(serial[j].argmax())
                    for j in range(len(cands))]
        diffs = [float((rows[j].to(torch.float32) - serial[j].to(torch.float32)).abs().max())
                 for j in range(len(cands))]
        t4 = accept_ok and all(same_ctx) and Lb == pre + len(cands)
        print(f"[T4] {name_v}: verify-rule all-accept={accept_ok} (g==cands) | "
              f"rows[j] vs serial[j] argmax-same={same_ctx} | block cache "
              f"{Lb}(exp {pre + len(cands)}) | same-context max|Δ| "
              f"{[f'{d:.1e}' for d in diffs]} -> {t4}", flush=True)
        ok &= t4
        results["T4"] = t4
        del rows, blk, serial, cache_a, cache_b, lg_a, lg_b, cur
        torch.cuda.empty_cache()

    del tgt, drf
    gc.collect()
    torch.cuda.empty_cache()
    done = {k: v for k, v in results.items()}
    print(f"\nB5 helpers: {done}")
    print("B5-helpers:", "PASS" if (done and all(done.values())) else
          ("PARTIAL (未完成为 SKIP, 继续实现)" if not results else "CHECK"))
    return 0 if (results and all(results.values())) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
