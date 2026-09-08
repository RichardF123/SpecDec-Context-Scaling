"""fp8_kv_v1_roundtrip.py — G3 验证 V1 (B 版): groupwise(32) FP8 KV 往返单测（CUDA）

直接 import 引擎实现 quantize_kv_fp8/dequantize_kv_fp8/GROUP_SIZE（被测 = 生产代码）。
检查（无人工精度阈值, 只有硬不变量 + 记录量级）:
  1. payload: shape 与输入同形、dtype=float8_e4m3fn、无 NaN/Inf、max|q|<=448
  2. scale: shape (B,kv,S,g), g=D//GROUP_SIZE; 全零组 scale=1
  3. 确定性: 同输入两次量化逐位一致
  4. 全零组往返严格为 0（且 finite）
  5. 每组最大元素映射到恰好 |448|
  6. 离群隔离: 大元素只影响其所在 32 维组; 同 head 其它组误差按自身组动态范围
     <= 组 max/16（E4M3 rel<=2^-4）; 全局 roundtrip max|Δ| <= 全局 max/16
报告 head_dim / GROUP_SIZE / groups 实际值。
Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v1_roundtrip
"""
from __future__ import annotations

import torch

from .model_self import GROUP_SIZE, dequantize_kv_fp8, quantize_kv_fp8


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    print(f"device={dev}  {torch.cuda.get_device_name(0)}", flush=True)

    B, KV, S, D = 1, 2, 256, 128          # 1.5B 真实 D=128 -> g = 4
    g = D // GROUP_SIZE
    print(f"head_dim={D}  GROUP_SIZE={GROUP_SIZE}  groups/head={g}", flush=True)

    # log-uniform 幅值 1e-2..1e3, 跨 token/head; 边界覆盖
    base = 10.0 ** (torch.rand(B * KV * S, device=dev) * 5 - 2).view(B, KV, S, 1)
    x = base * torch.randn(B, KV, S, D, device=dev)
    x[:1, :1, :8] = 0.0                    # 若干全零 token（两组皆零）
    x[0, 0, 8, 0] = 1e4                    # 离群: head0/token8/group0
    x = x.to(torch.bfloat16)

    ok = True

    q, scale = quantize_kv_fp8(x)
    # [1] payload 结构
    p_ok = (q.shape == x.shape and q.dtype == torch.float8_e4m3fn)
    qv = q.to(torch.float32)
    fin = bool(torch.isfinite(qv).all().item() and torch.isfinite(scale).all().item())
    over = qv.abs().max().item()
    print(f"[1] payload: shape_ok={p_ok} max|q|={over:.1f} (<=448) finite={fin}")
    ok &= p_ok and fin and over <= 448.0

    # [2] scale 结构
    s_ok = (tuple(scale.shape) == (B, KV, S, g) and scale.dtype == torch.float32)
    zscale = scale[0, 0, 0].tolist()
    print(f"[2] scale shape={tuple(scale.shape)} (expect (B,kv,S,g))  dtype={scale.dtype}  "
          f"zero-token scales={zscale} (expect [1,1,1,1])")
    ok &= s_ok and all(abs(v - 1.0) < 1e-6 for v in zscale)

    # [3] 确定性
    q2, s2 = quantize_kv_fp8(x)
    det = (q.to(torch.float32) == q2.to(torch.float32)).all().item() and torch.equal(scale, s2)
    print(f"[3] deterministic: {det}")
    ok &= det

    # [4] 全零 token 严格还原
    hat0 = dequantize_kv_fp8(q, scale, torch.bfloat16)[0, 0, :8]
    zero_ok = bool((hat0.abs().max().item() == 0) and torch.isfinite(hat0).all().item())
    print(f"[4] all-zero tokens roundtrip exact & finite: {zero_ok}")
    ok &= zero_ok

    # [5] 每组最大元素 -> |448|
    qg = q.to(torch.float32).view(B, KV, S, g, GROUP_SIZE)
    pmax = qg.abs().amax(dim=-1)          # (B,KV,S,g)
    # 排除全零组（其 max 应为 0）
    nonz = (qg.abs().amax(dim=-1) > 0)
    map448 = bool((pmax[nonz] == 448.0).all().item() if nonz.any() else True)
    print(f"[5] each nonzero group max maps to exactly 448: {map448}")
    ok &= map448

    # [6] 离群隔离 + 全局界
    hat = dequantize_kv_fp8(q, scale, torch.bfloat16).to(torch.float32)
    xf = x.to(torch.float32)
    err = (hat - xf).abs()
    max_abs = xf.abs().amax().item()
    max_err = err.amax().item()
    # 离群 token: head0 group1.. 的元素误差须 <= 自身组 max/16
    tok = xf[0, 0, 8]
    other_groups = torch.cat([tok[GROUP_SIZE:]], dim=0)     # group 1..g-1 元素
    og_max = other_groups.abs().max().item()
    err_og = (hat[0, 0, 8][GROUP_SIZE:] - tok[GROUP_SIZE:]).abs().max().item()
    isol = err_og <= og_max / 16.0 + 1e-3
    print(f"[6] outlier in group0: other-group max|Δ|={err_og:.3e} "
          f"<= their max/16 ({og_max/16:.3e}): {isol}")
    print(f"    global roundtrip max|Δ|={max_err:.3e} <= global max/16 "
          f"({max_abs/16:.3e}): {max_err <= max_abs/16}")
    ok &= isol and max_err <= max_abs / 16

    print("\nV1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
