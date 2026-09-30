"""attn_l27_g012_selftest_cpu.py — G0/G1/G2 运行前的 CPU 测试（不加载模型、不用 GPU）。

覆盖（对应任务 §六「运行前 CPU 测试」）：
  1. Q/K/V 维度、对齐行与绝对位置；
  2. mask 对未来位置的屏蔽（含对齐行恰可见 keys 0..8216 = 8217 个）；
  3. 相同有效前缀的 K/V 裁剪（纯切片、前 N 项逐元素相同、有效范围不变）；
  4. shape 不等禁止广播（`(1,12,1,128)` vs `(1,12,4,128)` 等）；
  5. 缺失检查、NaN/Inf、快照污染 ⇒ 必须**抛错停止**（用 `diag_checks.require` 与显式断言）；
  6. 有效位置有限性检查与「被 mask 位置 = -inf」**分开**判定，`-inf` 不得被误报为非有限失败。

运行：`python -m specdec.engine.attn_l27_g012_selftest_cpu`（期望 PASS 全部 / FAIL 0）
"""
from __future__ import annotations

import torch

from . import attn_l27_core as C
from . import diag_checks as DC

NB, NH, NKV, HD = 1, 12, 2, 128
PRE, S = 8215, 4              # 轮前 cache 长度 / 块查询行数
T_BLK = PRE + S               # 8219
ALIGN = 1                     # 对齐块行


def _qk(s, T, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(NB, NH, s, HD, generator=g).to(dtype)
    k = torch.randn(NB, NKV, T, HD, generator=g).to(dtype)
    v = torch.randn(NB, NKV, T, HD, generator=g).to(dtype)
    return q, k, v


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

    print("== 1. Q/K/V 维度、对齐行与绝对位置 ==")
    q, k, v = _qk(S, T_BLK)
    check("q 形状 (1,12,4,128)", tuple(q.shape) == (NB, NH, S, HD))
    check("k/v 形状 (1,2,8219,128)", tuple(k.shape) == (NB, NKV, T_BLK, HD))
    keep = C.block_mask(S, T_BLK)
    check("mask 形状 (4,8219) bool", tuple(keep.shape) == (S, T_BLK) and keep.dtype == torch.bool)
    check("块行 i 可见 key ≤ 8215+i",
          [C.last_visible_key(keep[i]) for i in range(S)] == [PRE, PRE + 1, PRE + 2, PRE + 3])
    align_row = PRE + ALIGN
    check("★ 对齐行（行 1）最后可见 key == 8216（输入位置 8216）",
          C.last_visible_key(keep[ALIGN]) == align_row == 8216)
    check("★ 对齐行可见 key 数 == 8217（keys 0..8216）", C.visible_count(keep[ALIGN]) == 8217)
    check("块行 0 可见 8216 个（keys 0..8215）", C.visible_count(keep[0]) == 8216)

    print("== 2. mask 对未来位置的屏蔽 ==")
    check("★ 对齐行看不到位置 8217/8218",
          (not bool(keep[ALIGN][8217])) and (not bool(keep[ALIGN][8218])))
    check("对齐行能看到 0..8216 全部",
          all(bool(keep[ALIGN][j]) for j in range(0, 8217)))
    check("块行 3 可见 keys 0..8218（8219 个）", C.visible_count(keep[3]) == 8219)
    sc = C.scores_from(q, k)
    pr = C.probs_from(sc, keep, q.dtype)
    masked_vals = sc.masked_fill(~keep, float("-inf"))
    keepF = keep.expand_as(masked_vals)          # 布尔索引需要与张量同形（不广播索引）
    valid_ok = bool(torch.isfinite(masked_vals).masked_select(keepF).all())
    neginf_ok = bool((masked_vals.masked_select(~keepF) == float("-inf")).all())
    check("★ 有效位置 scores 全有限（只在有效位置检查）", valid_ok)
    check("★ 被 mask 位置恰为 -inf（预期值，不是非有限失败）", neginf_ok)
    check("概率在有效位置之和 ≈ 1",
          bool(torch.allclose(pr.float().sum(-1), torch.ones(NB, NH, S), atol=2e-2)))
    check("被 mask 位置概率为 0", bool((pr.float().masked_select(~keepF) == 0).all()))

    print("== 3. 相同有效前缀的 K/V 裁剪 ==")
    k2, v2 = C.truncate_kv(k, v, 8217)
    check("截断后形状 (1,2,8217,128)", tuple(k2.shape) == (NB, NKV, 8217, HD))
    check("★ 前 8217 个 K 逐元素相同（纯切片）", torch.equal(k[:, :, :8217, :], k2))
    check("★ 前 8217 个 V 逐元素相同（纯切片）", torch.equal(v[:, :, :8217, :], v2))
    check("截断是视图（共享存储）", k2._base is not None or k2.data_ptr() == k.data_ptr())
    keep2 = C.block_mask(1, 8217)          # 单查询、T=8217：全部可见
    check("★ T=8217 时单查询可见 8217 个 key（0..8216），有效范围与 G1 一致",
          C.visible_count(keep2[0]) == 8217 and C.last_visible_key(keep2[0]) == 8216)
    check("★ G1 行 mask 截到 8217 后与 G2 行 mask 逐元素相同（有效范围一致）",
          torch.equal(keep[ALIGN][:8217], keep2[0]))
    check("两者有效可见范围同为 keys 0..8216（8217 个）",
          C.last_visible_key(keep[ALIGN]) == C.last_visible_key(keep2[0]) == 8216)
    try:
        C.truncate_kv(k, v, 9000)
        check("★ 截断超过现有长度必须抛错", False)
    except ValueError:
        check("★ 截断超过现有长度必须抛错", True)

    print("== 4. shape 不等禁止广播 ==")
    try:
        DC.cmp_tensor(q[:, :, ALIGN:ALIGN + 1, :].reshape(-1), q.reshape(-1), "1row_vs_4row")
        check("★ (1,12,1,128) vs (1,12,4,128) 直接比较必须抛 CheckError", False)
    except DC.CheckError:
        check("★ (1,12,1,128) vs (1,12,4,128) 直接比较必须抛 CheckError", True)
    try:
        C.probs_from(sc, C.block_mask(2, T_BLK), q.dtype)      # s=2 的 mask 配 s=4 的 scores
        check("★ mask 尾两维与 scores 不匹配必须抛错", False)
    except ValueError:
        check("★ mask 尾两维与 scores 不匹配必须抛错", True)
    try:
        C.scores_from(q, k[:, :, :, :64])                       # head_dim 不一致
        check("★ head_dim 不一致必须抛错", False)
    except ValueError:
        check("★ head_dim 不一致必须抛错", True)
    try:
        C.scores_from(q.float(), k)                             # dtype 不一致
        check("★ dtype 不一致必须抛错", False)
    except ValueError:
        check("★ dtype 不一致必须抛错", True)

    print("== 5. 缺失检查 / NaN / Inf / 快照污染 ⇒ 必须停止 ==")
    try:
        DC.require(False, "捕获缺少 RoPE 后 q", "未拿到 L27 的 q")
        check("★ 缺失观察点必须抛 CheckError", False)
    except DC.CheckError:
        check("★ 缺失观察点必须抛 CheckError", True)
    bad = k.clone()
    bad[0, 0, 7, 3] = float("nan")
    try:
        DC.cmp_tensor(k2, bad[:, :, :8217, :], "nan_in_k")
        check("★ K 中出现 NaN 必须抛 CheckError（覆盖完整张量，非首元素）", False)
    except DC.CheckError:
        check("★ K 中出现 NaN 必须抛 CheckError（覆盖完整张量，非首元素）", True)
    check("finite_all 对含 Inf 的概率返回 False",
          DC.finite_all(torch.tensor([1.0, float("inf")])) is False)
    src = k.clone()
    snap = DC.snapshot(src)
    src[0, 0, 0, 0] = src[0, 0, 0, 0] + 1
    try:
        DC.cmp_tensor(src[:, :, :8217, :], k2, "polluted", ref_snapshot=snap, src=src)
        check("★ 源张量被原地污染必须抛 CheckError（与独立快照比）", False)
    except DC.CheckError:
        check("★ 源张量被原地污染必须抛 CheckError（与独立快照比）", True)

    print("== 6. 功能式核心在未 mask 位置等价于参考实现（自洽性） ==")
    ref = torch.softmax((torch.matmul(q.float(),
                                      C.M.kv_repeat(k, 6).float().transpose(-2, -1))
                         / (HD ** 0.5)).masked_fill(~keep, float("-inf")), dim=-1)
    check("概率与 fp32 参考实现接近（bf16 容差内）",
          bool(torch.allclose(pr.float(), ref, atol=2e-2, rtol=2e-2)))
    av4 = C.core_forward(q, k, v, keep)["av"]
    av1 = C.core_forward(q[:, :, ALIGN:ALIGN + 1, :], k, v, keep[ALIGN:ALIGN + 1])["av"]
    check("单行调用的核心输出形状 (1,12,1,128)", tuple(av1.shape) == (NB, NH, 1, HD))
    check("展平后 (1,1,1536) vs (1,4,1536)",
          tuple(C.flatten_core(av1).shape) == (NB, 1, NH * HD)
          and tuple(C.flatten_core(av4).shape) == (NB, S, NH * HD))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
