"""attn_l27_tail_ab_selftest_cpu.py — A/B 末端传播实验的**运行前 CPU 自测**。

覆盖（对应任务 §四「控制与断言」）：
  1. `as_row3`/`require_bf16_exact`：形状、dtype、finite、形状不等禁止广播、fp32 记录的可复用性；
  2. `bf16_add_row`：**必须**在 bf16 下做加法（用 `1.0 + 2^-9` 证明它与「FP32 求和」给出不同结果），
     拒绝 fp32 输入、拒绝形状不等、拒绝 device 不一致、拒绝非有限；
  3. `check_residual_identity`：正确取「L27 层输入」时通过；**误用 attention 之后的残差时必然失败**
     （即「重复加 attention」会被拦住）；
  4. 阶段比较：shape 不等 / NaN / 源张量被原地修改都必须抛错停止。

运行：`python -m specdec.engine.attn_l27_tail_ab_selftest_cpu`
"""
from __future__ import annotations

import torch

from . import attn_tail_common as T
from . import diag_checks as DC

H = T.HIDDEN


def main() -> int:
    npass = nfail = 0

    def check(name, cond):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}")

    def raises(fn):
        try:
            fn()
            return False
        except (DC.CheckError, ValueError, TypeError):
            return True

    print("== 1. row3 归一化：形状 / dtype / finite ==")
    r3 = torch.randn(1, 1, H).to(torch.bfloat16)
    r1 = r3.reshape(H)
    check("(1,1,1536) bf16 通过", tuple(T.as_row3(r3, "r3").shape) == (1, 1, H))
    check("(1536,) bf16 通过并归一为 (1,1,1536)", tuple(T.as_row3(r1, "r1").shape) == (1, 1, H))
    check("★ fp32 张量被拒绝（禁止用 FP32 值替代真实 bf16）",
          raises(lambda: T.as_row3(r3.float(), "fp32")))
    check("★ 形状 (1,4,1536) 被拒绝（禁止广播/错形状）",
          raises(lambda: T.as_row3(torch.randn(1, 4, H).to(torch.bfloat16), "wrong")))
    check("非张量被拒绝", raises(lambda: T.as_row3([1.0], "list")))
    bad = r3.clone()
    bad[0, 0, 5] = float("nan")
    check("★ 含 NaN 被拒绝（对完整张量实算 isfinite）", raises(lambda: T.as_row3(bad, "nan")))
    check("bf16 可精确记录的 fp32 值通过 require_bf16_exact",
          raises(lambda: T.require_bf16_exact(r3.float().cpu(), "rec")) is False)
    check("★ 非 bf16 可表示的 fp32 值被 require_bf16_exact 拒绝",
          raises(lambda: T.require_bf16_exact(torch.tensor([[[1.0009765625] * H]]), "notbf16")))

    print("== 2. bf16 残差加法必须真的在 bf16 下做 ==")
    a = torch.zeros(1, 1, H, dtype=torch.bfloat16)
    a[0, 0, 0] = 1.0
    b = torch.zeros(1, 1, H, dtype=torch.bfloat16)
    b[0, 0, 0] = 2.0 ** -9                       # bf16 在 1.0 处的间距是 2^-8 ⇒ 应舍入回 1.0
    out = T.bf16_add_row(a, b, "tiny_add")
    check("★ 1.0 + 2^-9 在 bf16 下 == 1.0（证明加法真的发生在 bf16，而非 fp32 求和）",
          float(out[0, 0, 0]) == 1.0)
    fp32_sum = float((a.double() + b.double())[0, 0, 0])
    check("★ 同一对被 FP32 求和会得到 1.001953125（与上一条不同 ⇒ 两条路径可区分）",
          abs(fp32_sum - 1.001953125) < 1e-12)
    check("结果 dtype 仍是 bf16", str(out.dtype) == "torch.bfloat16")
    check("★ 左侧为 fp32 时被拒绝（禁止 FP32 求和替代）",
          raises(lambda: T.bf16_add_row(a.float(), b, "fp32_left")))
    check("★ 形状不等被拒绝（(1,1,1536) vs (1,1536) 不得广播）",
          raises(lambda: T.bf16_add_row(a, b.reshape(1, H), "shape")))
    check("★ 形状 (1,4,1536) 被拒绝（本实验只允许 (1,1,1536)）",
          raises(lambda: T.bf16_add_row(a, torch.randn(1, 4, H).to(torch.bfloat16), "shape4")))
    nanb = b.clone()
    nanb[0, 0, 7] = float("inf")
    check("★ 含 Inf 的输入被拒绝", raises(lambda: T.bf16_add_row(a, nanb, "inf")))

    print("== 3. 残差来源必须正确（拦住「重复加 attention」） ==")
    layer_in = torch.randn(1, 1, H).to(torch.bfloat16)
    attn_out = torch.randn(1, 1, H).to(torch.bfloat16)
    post_res = layer_in + attn_out                     # 生产：x = x + atten_out（bf16）
    ok = T.check_residual_identity(layer_in, attn_out, post_res, "correct")
    check("★ 用「L27 层输入」作残差 ⇒ 恒等式成立", ok["identity_holds"] is True)
    check("★ 用「attention 之后的残差」当残差 ⇒ 恒等式必然失败（重复加 attention 被拦住）",
          raises(lambda: T.check_residual_identity(post_res, attn_out, post_res, "double_add")))
    check("参考残差不是 bf16 时被拒绝",
          raises(lambda: T.check_residual_identity(layer_in, attn_out, post_res.float(), "fp32_ref")))

    print("== 4. 阶段比较与控制 ==")
    x = torch.randn(1, 1, H).to(torch.bfloat16)
    y = x.clone()
    c = T.stage_cmp(x, y, "same")
    check("相同张量 ⇒ elementwise_equal 且 max|Δ|=0",
          c["elementwise_equal"] is True and c["abs_err_max"] == 0.0)
    check("★ shape 不等 ⇒ 抛 CheckError（禁止广播比较）",
          raises(lambda: T.stage_cmp(x, x.reshape(1, H), "shape")))
    bad2 = x.clone()
    bad2[0, 0, 9] = float("nan")
    check("★ 含 NaN ⇒ 抛 CheckError", raises(lambda: T.stage_cmp(x, bad2, "nan")))
    src = x.clone()
    snap = DC.snapshot(src)
    src[0, 0, 0] = src[0, 0, 0] + 1
    check("★ 源张量被原地修改 ⇒ 抛 CheckError",
          raises(lambda: DC.cmp_tensor(src, y, "polluted", ref_snapshot=snap, src=src)))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
