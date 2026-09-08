"""fp8_scale_diag.py — 量化误差归因诊断(零引擎改动): 决定 B(分组 scale)是否值得

输入: 真实 bf16 K/V(base bf16 chunk1024 prefill 快照, 若干层)。
对每层 K、V 计算反量化误差, 比较三种 scale 方案:
  1) per-head-per-token H (现状): 每 (token, head) 一个 scale
  2) group=32 G (假想 B): 每 (token, head, 2x32) 一个 scale
  3) per-element E (参照上界): fp32 scale 可无损编码 bf16 值 -> 误差≈0, 表示
     "去掉一切组内动态范围失配"能到多好(真实分组只能逼近它, 不可达)
判读:
  G 的 abs 误差明显 < H => 组内动态范围失配是误差主因 -> B 值得实施
  G ~ H => 把 64 维劈两半不改善 => 误差来自别处(整向量量级一致的少数大元素 +
      E4M3 3-bit 网格) -> B 收益有限, 转 A 或校准式 per-channel 静态 scale(需批准)
指标: 每层 相对误差 max、归一绝对误差 max、绝对误差 max(absmax), K/V 分开。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_scale_diag
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .fp8_kv_v3_localize import prefill, snapshot
from .g2_operational import make_text
from .model_self import E4M3_MAX

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
PROMPT = 4096
LAYERS = (0, 7, 14, 21, 27)


def quant_err(x: torch.Tensor, group: int):
    """x (1,2,T,64) fp32 真值; group=64 -> per head-token, 32 -> 2x32, 1 -> per-element.
    返回相对误差 max(每元素 |err|/|x| 对 x!=0) 与 绝对误差(相对 x 最大值)."""
    xf = x.to(torch.float32)
    d = xf.shape[-1]
    g = d // group
    xg = xf.view(*xf.shape[:-1], g, group)
    m = xg.abs().amax(dim=-1, keepdim=True)
    s = m / E4M3_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (xg / s).to(torch.float8_e4m3fn)
    hat = (q.to(torch.float32) * s).view_as(xf)
    err = (hat - xf).abs()
    denom = xf.abs() + 1e-12
    rel = (err / denom).amax().item()
    # 相对"该层该矩阵最大幅值"的归一绝对误差(与 V3 的 max|Delta| 可比口径)
    norm = err.amax().item() / (xf.abs().amax().item() + 1e-12)
    return rel, norm, err.amax().item()


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = make_text(tok, PROMPT).to("cuda")

    from .fp8_kv_v3_localize import load_model as _lm
    ours = _lm()
    cache, _ = prefill(ours, ids, "fp16", 1024)   # base bf16, KV 即真值(bf16)
    snap = snapshot(cache)
    ours = None
    del cache
    torch.cuda.empty_cache()

    print(f"true KV from base bf16 chunk1024 prefill; layers {LAYERS}\n")
    print("schemes: H=per-head-token(64)  G=group32(2x32)  E=per-element(lossless bound)")
    for li in LAYERS:
        for key in ("k", "v"):
            x = snap[li][key].to(torch.float32)
            r_h, a_h, abs_h = quant_err(x, 64)
            r_g, a_g, abs_g = quant_err(x, 32)
            r_e, a_e, abs_e = quant_err(x, 1)
            print(f"L{li:2d} {key}: H rel={r_h:.3f} abs_norm={a_h:.4f} absmax={abs_h:.2f} | "
                  f"G rel={r_g:.3f} abs_norm={a_g:.4f} absmax={abs_g:.2f} | "
                  f"E rel={r_e:.3f} abs_norm={a_e:.4f} absmax={abs_e:.2f}")
    print("\nreadout: absmax_G << absmax_H  => dynamic-range-bound, B worth trying;")
    print("         absmax_G ~ absmax_H    => grouping ineffective (B low value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
