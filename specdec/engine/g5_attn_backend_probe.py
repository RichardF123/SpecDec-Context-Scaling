"""g5_attn_backend_probe.py — attention 后端的**显存 / 时间 / 数值等价性**微基准。

**不加载模型、不改引擎**：只造 q/k/v，量"换实现之后单笔显存与耗时各降多少"。

要回答的问题
  `model_self.py:156-170` 现行路径是手写的
      scores = q @ k^T / sqrt(d)                     (bf16)
      scores.masked_fill(偏移因果掩码, -inf)
      attn   = softmax(scores, fp32).to(bf16)        ← **8K 下这一笔 806–896 MiB，OOM 就崩在这**
      out    = attn @ v
  换成 `F.scaled_dot_product_attention` 之后能省多少、且**必须数值等价**。

⚠️ 两个已实测的语义陷阱（本探针就是为了把它们钉死）
  1. **`is_causal=True` 不是我们的掩码**：PyTorch 的 `is_causal` 等价于
     `triu(ones(L,S), diagonal=1)`（**左上**对齐），而分块 prefill 的 query 是全长 KV 的
     **最后 s 个** token，需要 `diagonal=T-s+1`（**右下**对齐）。
     实测（CPU, fp64）：`s=8,T=8` 差 2.1e-07 ✅；`s=5,T=32` 差 **2.157** ❌；`s=1,T=32` 差 **2.156** ❌。
     ⇒ 只有 `s == T`（全长一次前向）时才能用 `is_causal`。
  2. **SDPA 的 bool mask 语义与 `masked_fill` 相反**：SDPA 里 **True = 参与**
     （而 `masked_fill` 里 True = 屏蔽）⇒ 必须传 `keep = ~triu(...)`。

安全性
  只造 q/k/v，不加载权重、不建 8K KV、不跑双模型 ⇒ 峰值约 1–2 GiB，
  远低于会触发 WDDM 超订 / commit 压力的那条路径（那次整机硬挂的条件）。
  手写路径在 s=8192 时单笔约 3 GiB ⇒ 默认**不测**，需显式 `--include-full-prefill`。

用法
  python -m specdec.engine.g5_attn_backend_probe --dry-run    # 只做 CPU 自检 + 打印计划
  python -m specdec.engine.g5_attn_backend_probe              # GPU 微基准（约 2 分钟）
可选：--reps 20   --include-full-prefill   --tol 1e-2
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

CASES = [
    dict(name="prefill_chunk_target", s=2048, T=8192, heads=12, kv_heads=2, head_dim=128,
         note="分块 prefill 最后一个 chunk（手写路径 806 MiB）"),
    dict(name="prefill_chunk_draft", s=2048, T=8192, heads=14, kv_heads=2, head_dim=64,
         note="draft 同形（手写路径 896 MiB —— OOM 崩点）"),
    dict(name="block_verify_target", s=5, T=8192, heads=12, kv_heads=2, head_dim=128,
         note="SpecDec 块验证（k+1=5）"),
    dict(name="decode_target", s=1, T=8192, heads=12, kv_heads=2, head_dim=128,
         note="AR 单步"),
    dict(name="full_prefill_target", s=8192, T=8192, heads=12, kv_heads=2, head_dim=128,
         note="引擎默认 `_prefill` 全长一次前向（手写路径约 3 GiB）", opt_in=True),
]

# 变体：(名称, 掩码形式, 是否需要 attn_mask)
VARIANTS = ("manual", "sdpa_is_causal", "sdpa_mask_default", "sdpa_mask_flash",
            "sdpa_mask_mem_efficient", "sdpa_mask_cudnn")


def manual_attention(q, k, v, full_length: bool):
    """**逐行复刻** model_self.py:156-170（含 fp32 softmax 与 bf16 回写）。"""
    h_q, h_kv = q.shape[1], k.shape[1]
    if h_q != h_kv:
        k = k.repeat_interleave(h_q // h_kv, dim=1)
        v = v.repeat_interleave(h_q // h_kv, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    s, T = q.shape[2], k.shape[2]
    if full_length:
        mask = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    elif s == 1:
        pass
    else:
        mask = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool),
                          diagonal=(T - s) + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


def keep_mask(q, k) -> torch.Tensor:
    """偏移因果的**保留**掩码（True = 参与）——注意与 masked_fill 语义相反。"""
    s, T = q.shape[2], k.shape[2]
    if s == T:
        drop = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool), diagonal=1)
    elif s == 1:
        drop = torch.zeros(s, T, device=q.device, dtype=torch.bool)
    else:
        drop = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool),
                          diagonal=(T - s) + 1)
    return ~drop


def sdpa_call(q, k, v, *, use_mask: bool, backend=None):
    kw = dict(dropout_p=0.0, scale=q.shape[-1] ** -0.5,
              enable_gqa=(q.shape[1] != k.shape[1]))
    if use_mask:
        kw["attn_mask"] = keep_mask(q, k)
    else:
        kw["is_causal"] = True
    import torch.nn.attention as A
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, **kw)
    with A.sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(q, k, v, **kw)


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 2 ** 20


def variants_for(case: dict, full_length: bool):
    out = ["manual"]
    if full_length:
        out.append("sdpa_is_causal")          # 只有 s==T 才合法
    out += ["sdpa_mask_default", "sdpa_mask_flash", "sdpa_mask_mem_efficient",
            "sdpa_mask_cudnn"]
    return out


def run_case(case: dict, reps: int, tol: float) -> dict:
    s, T, h, kv, d = case["s"], case["T"], case["heads"], case["kv_heads"], case["head_dim"]
    dev, dt = "cuda", torch.bfloat16
    full_length = (s == T)
    torch.manual_seed(0)
    q = torch.randn(1, h, s, d, device=dev, dtype=dt)
    k = torch.randn(1, kv, T, d, device=dev, dtype=dt)
    v = torch.randn(1, kv, T, d, device=dev, dtype=dt)
    torch.cuda.synchronize()
    base_mib = torch.cuda.memory_allocated() / 2 ** 20

    import torch.nn.attention as A
    be_map = {"sdpa_mask_flash": A.SDPBackend.FLASH_ATTENTION,
              "sdpa_mask_mem_efficient": A.SDPBackend.EFFICIENT_ATTENTION,
              "sdpa_mask_cudnn": A.SDPBackend.CUDNN_ATTENTION}

    out = dict(**{kk: case[kk] for kk in ("name", "s", "T", "heads", "kv_heads",
                                         "head_dim", "note")}, results={})
    ref = None
    for var in variants_for(case, full_length):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                if var == "manual":
                    fn = lambda: manual_attention(q, k, v, full_length)      # noqa: E731
                elif var == "sdpa_is_causal":
                    fn = lambda: sdpa_call(q, k, v, use_mask=False)          # noqa: E731
                else:
                    b = be_map.get(var)
                    fn = lambda: sdpa_call(q, k, v, use_mask=True, backend=b)  # noqa: E731
                res = fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    res = fn()
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                peak = max(_peak_mib() - base_mib, 0.0)
                if var == "manual":
                    ref = res.clone()
                cmp_ = None
                if ref is not None:
                    diff = (res.float() - ref.float()).abs().max().item()
                    cmp_ = dict(max_abs_diff=diff, equivalent=bool(diff <= tol))
                out["results"][var] = dict(
                    peak_transient_mib=peak, median_ms=statistics.median(ts) * 1000.0,
                    vs_manual=cmp_)
        except Exception as e:                                      # noqa: BLE001
            out["results"][var] = dict(error=f"{type(e).__name__}: {e}")
    del q, k, v
    torch.cuda.empty_cache()
    return out


def cpu_selfcheck(tol: float) -> list[tuple[str, bool, str]]:
    """**CPU 上**先证明两件事：`is_causal` 只对 s==T 合法；`keep_mask` 形式等价。"""
    rows = []
    torch.manual_seed(0)
    for (s, T, h, kv, d) in ((8, 8, 4, 2, 16), (5, 32, 4, 2, 16), (1, 32, 4, 2, 16)):
        q = torch.randn(1, h, s, d, dtype=torch.float64)
        k = torch.randn(1, kv, T, d, dtype=torch.float64)
        v = torch.randn(1, kv, T, d, dtype=torch.float64)
        ref = manual_attention(q, k, v, s == T)
        d_mask = (ref - sdpa_call(q, k, v, use_mask=True)).abs().max().item()
        rows.append((f"keep-mask 形式  s={s:<4} T={T:<4}", d_mask <= tol,
                     f"max|Δ|={d_mask:.3e}"))
        if s == T:
            d_ic = (ref - sdpa_call(q, k, v, use_mask=False)).abs().max().item()
            rows.append((f"is_causal 形式  s={s:<4} T={T:<4}", d_ic <= tol,
                         f"max|Δ|={d_ic:.3e}"))
        else:
            d_ic = (ref - sdpa_call(q, k, v, use_mask=False)).abs().max().item()
            rows.append((f"is_causal 形式  s={s:<4} T={T:<4}", d_ic > tol,
                         f"max|Δ|={d_ic:.3e} ⇒ 已证**不合法**（左上对齐）"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="attention 后端微基准（不加载模型、不改引擎）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--include-full-prefill", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="与手写路径的最大绝对差容忍（bf16 输出，量级 ~1）")
    args = ap.parse_args()

    print("== CPU 语义自检（先证明候选形式对不对，再上 GPU）==")
    bad = 0
    for name, ok, detail in cpu_selfcheck(args.tol):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}  {detail}")
        bad += (0 if ok else 1)
    if bad:
        print(f"  !! 有 {bad} 项不符合预期 —— 先修形式，别上 GPU")
        return 3

    cases = [c for c in CASES if (not c.get("opt_in")) or args.include_full_prefill]
    print(f"\n将测试 {len(cases)} 组形状，reps={args.reps}，容差={args.tol:g}")
    for c in cases:
        print(f"  s={c['s']:<5} T={c['T']:<5} heads={c['heads']:<3} kv={c['kv_heads']} "
              f"d={c['head_dim']:<4} {c['name']:<22} {c['note']}")
    if args.dry_run:
        print("\n--dry-run：未使用 GPU、未写文件")
        return 0
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return 2

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows = []
    for c in cases:
        r = run_case(c, args.reps, args.tol)
        rows.append(r)
        print(f"\n== {r['name']}  s={r['s']} T={r['T']} heads={r['heads']}/{r['kv_heads']} "
              f"d={r['head_dim']} ==")
        print(f"  {'variant':<26}{'peak瞬态MiB':>13}{'median_ms':>11}{'max|Δ|':>11}{'等价':>7}")
        for var, val in r["results"].items():
            if "error" in val:
                print(f"  {var:<26}  ERROR: {val['error'][:90]}")
                continue
            cmp_ = val.get("vs_manual") or {}
            eq = "—" if not cmp_ else ("是" if cmp_["equivalent"] else "**否**")
            md = "—" if not cmp_ else f"{cmp_['max_abs_diff']:.3e}"
            print(f"  {var:<26}{val['peak_transient_mib']:>13.1f}"
                  f"{val['median_ms']:>11.2f}{md:>11}{eq:>7}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"attn_backend_{run_id}.json"
    if p.exists():
        raise SystemExit(f"[拒绝覆盖] {p}")
    p.write_text(json.dumps(dict(run_id=run_id, torch=torch.__version__,
                                 device=torch.cuda.get_device_name(0),
                                 tol=args.tol, cases=rows), ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"\n写出 {p}")
    print("注：**算子级微基准**（不加载模型、不改引擎），用于判断是否值得提 amendment；"
          "不是吞吐结论，也不进正式主表。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
