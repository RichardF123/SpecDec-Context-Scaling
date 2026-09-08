"""fp8_kv_v1_roundtrip.py — G3 验证 V1: FP8 KV 量化/反量化往返单测（CUDA）

依据 design_freeze_fp8_kv.md D4（FROZEN v1.0）:
    s   = max(|x|, dim=-1) / 448        # per (…, kv-head, token); max=0 -> s=1
    q   = (x_fp32 / s).to(float8_e4m3fn) # 单次舍入（fp32 除法避免 double-rounding）
    hat = q.to(fp16) * s                 # 反量化

本文件内置 quantize_kv_fp8 / dequant_kv_fp8 **参考实现**（引擎 helper 必须与之数值一致）;
检查项（无人工精度阈值, 只记录量级 + 硬性不变量）:
  1. payload 无 NaN / Inf; max|q| <= 448（E4M3FN 无 inf, 构造上不溢出）
  2. 往返 max|Δ| 量级（理论界: 逐元素相对误差 <= 2^-4 → 全局 abs err <= max|v|/16）
  3. 确定性: 同输入两次量化逐位一致
  4. 边界: 全零组（s=1 路径）; 单点离群组（per-token scale 应吸收）; 幅值跨 5 个数量级

Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v1_roundtrip
"""
from __future__ import annotations

import torch

E4M3_MAX = 448.0


def quantize_kv_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (…, head_dim) fp16/fp32 → (payload fp8 同形, scale (…) fp32). D4 参考实现."""
    m = x.abs().amax(dim=-1, keepdim=True)          # (…, 1)
    s = m / E4M3_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)  # 全零组防除零
    x32 = x.to(torch.float32) / s                   # fp32 除法 → 单次舍入到 fp8
    q = x32.to(torch.float8_e4m3fn)
    return q, s.squeeze(-1)


def dequant_kv_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """q fp8 (…, d), scale (…) fp32 → fp16 反量化. D4 参考实现."""
    return q.to(torch.float16) * scale.unsqueeze(-1)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    print(f"device={dev}  {torch.cuda.get_device_name(0)}", flush=True)

    T, H, D = 2048, 2, 64

    # ---- 幅值跨 5 个数量级的组（模拟跨层/跨 token 的 K/V 动态）----
    base = 10.0 ** torch.rand(T * H, device=dev)     # 1e0..1e1? 用 log-uniform:
    base = 10.0 ** (torch.rand(T * H, device=dev) * 5 - 2)   # 1e-2 .. 1e3
    base = base.view(T, H, 1)
    x = base * torch.randn(T, H, D, device=dev)

    # ---- 边界: 全零组 / 单点离群组 ----
    x[:2] = 0.0                                       # 全零
    x[2, 0, 0] = 1e4                                  # 离群（组内 max 被它主导）
    x = x.to(torch.float16)

    ok = True
    # 1) 不变量
    q, s = quantize_kv_fp8(x)
    qv = q.to(torch.float16).to(torch.float32)
    nan_inf = not (torch.isfinite(qv).all().item() and torch.isfinite(s).all().item())
    over = qv.abs().max().item()
    print(f"[1] payload: max|q|={over:.1f} (<=448: {over <= 448.0})  "
          f"any_nan_inf={nan_inf}")
    ok &= (not nan_inf) and (over <= 448.0)

    # 2) 往返误差量级
    hat = dequant_kv_fp8(q, s)
    xf = x.to(torch.float32)
    err = (hat.to(torch.float32) - xf).abs()
    max_abs = xf.abs().amax().item()
    max_err = err.amax().item()
    rel = (err / (xf.abs() + 1e-12)).amax().item()
    # 正确理论界: E4M3 逐元素相对误差 <= 2^-4  →  全局 abs err <= max|v|/16
    # （旧版误写 max|v|/(448*16)=1.4, 把"最高 binade 半步长"当全局上界 —— harness 修正）
    th = max_abs / 16.0
    print(f"[2] roundtrip: max|Δ|={max_err:.3e}  max|v|={max_abs:.3e}  "
          f"理论界=max/16≈{th:.3e}  超界={max_err > th}")
    print(f"    max rel|Δ|={rel:.3e}（接近 1 仅来自近零小元素, abs 才是记账量; "
          f"正常幅值处 rel<=2^-4≈0.0625）")
    ok &= max_err <= th

    # 3) 确定性
    q2, s2 = quantize_kv_fp8(x)
    det = (q.to(torch.float16) == q2.to(torch.float16)).all().item() and \
          torch.equal(s, s2)
    print(f"[3] deterministic (bitwise): {det}")
    ok &= det

    # 4) 边界行为细查
    zero_grp_hat = dequant_kv_fp8(*quantize_kv_fp8(x[:2]))
    print(f"[4] all-zero group roundtrip max|Δ|={zero_grp_hat.abs().max().item():.2e} "
          f"(==0 且无 NaN/Inf: "
          f"{zero_grp_hat.abs().max().item() == 0 and torch.isfinite(zero_grp_hat).all().item()})")
    ok &= zero_grp_hat.abs().max().item() == 0
    # 离群组: 组内其它元素不应被离群毁掉（per-token-per-head scale 的意义）。
    # 正确检查对象 = head 0 的 63 个非离群元素（旧版误取 head 1 —— harness 修正）。
    # 判据用相对误差: 它们幅值可达离群的一半, abs 误差上百是 E4M3 数学预期。
    q3, s3 = quantize_kv_fp8(x[2:3])
    hat3 = dequant_kv_fp8(q3, s3)
    others = x[2:3][:, 0, 1:]                      # head 0, 排除第 0 维离群点
    others_hat = hat3[:, 0, 1:]
    rel_other = ((others_hat - others.to(torch.float32)).abs()
                 / (others.abs().to(torch.float32) + 1e-12)).max().item()
    print(f"[4] outlier-group head0 others: max rel|Δ|={rel_other:.4f} "
          f"（<0.1: {rel_other < 0.1}）")
    ok &= rel_other < 0.1

    # 5) 极值映射: max 恰好映射到 448
    xm = torch.full((1, 1, 4), 448.0, device=dev, dtype=torch.float16)
    qm, _ = quantize_kv_fp8(xm)
    print(f"[5] max-magnitude maps to exactly 448: "
          f"{qm.to(torch.float16).abs().max().item() == 448.0}")

    print("\nV1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
