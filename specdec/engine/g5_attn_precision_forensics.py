"""g5_attn_precision_forensics.py — 定位"eager vs 融合"数值差异来源的**专项取证**（GPU，不加载权重）。

背景（`docs/cudnn_prefill_stage_report.md` §6）：模型级 A/B 的 logits 差达 **12–60% 相对**，
而算子级用 `randn` 测却只有 2–3 ULP。本工具回答"到底是哪一侧、哪个环节"。

它检验并区分 4 个假说
  H1 **分数矩阵的 bf16 舍入**（eager 在 softmax 前把分数舍入成 bf16；
     **不宣称** cuDNN 内部精度 —— 本工具只测"融合侧误差对输入幅度不敏感"，不解释其内部实现）
  H2 掩码语义不一致（融合侧忽略 offset-causal 掩码）
  H3 GQA 广播错误（`enable_gqa=True` 与手工 `kv_repeat` 结果不同）
  H4 缩放被重复施加（`scale` 之外 kernel 又乘了一次 1/√d）

判据是**可观测的对比**，不是自证：
  * H1 ⇒ eager 误差**随分数幅度线性增长**，fused 误差**与幅度无关**，且 `|eager−fused| ≈ |eager−fp32|`；
  * H2 ⇒ 融合结果应≈"**未加掩码**"的 eager（专门构造"未来 key 分数更高"的输入来暴露）；
  * H3 ⇒ `enable_gqa=True` 与"手工展开的实体 k/v"结果**不同**（若逐位相同则 H3 不成立）；
  * H4 ⇒ 融合结果应≈"用另一个 scale 算的 eager"（若与正确 scale 的 eager 一致则 H4 不成立）。

用法
  python -m specdec.engine.g5_attn_precision_forensics
  FORENSICS_DEVICE=cpu 可强制 CPU（但 CPU 无 CUDNN 后端 ⇒ 融合列会整体失败，用于自检报错行为）
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from . import model_self as M

DEV = os.environ.get("FORENSICS_DEVICE", "cuda")
DT = torch.bfloat16
SEED = 0


def _eager(q, k, v, head_dim, has_cache=False, *, mask=True, scale=None) -> torch.Tensor:
    """eager 参照。`mask=False` 故意**不加掩码**（用于 H2）；`scale` 可换（用于 H4）。"""
    n_rep = q.shape[1] // k.shape[1]
    k_rep, v_rep = M.kv_repeat(k, n_rep), M.kv_repeat(v, n_rep)
    s, T = q.shape[2], k.shape[2]
    sc = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5 if scale is None else scale)
    if mask:
        if not has_cache or s > 1:
            diag = 1 if not has_cache else (T - s) + 1
            sc = sc.masked_fill(torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool),
                                           diagonal=diag), float("-inf"))
    return torch.matmul(torch.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype), v_rep)


def _fused(q, k, v, head_dim, *, gqa=True, mask=True, scale=None) -> torch.Tensor:
    """强制 CUDNN 后端的融合路径（**不吞异常**：不可用就让它抛）。"""
    s, T = q.shape[2], k.shape[2]
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        kw = dict(dropout_p=0.0, scale=head_dim ** -0.5 if scale is None else scale)
        if mask:
            kw["attn_mask"] = M._kv_keep_mask(s, T, q.device)
        if gqa and q.shape[1] != k.shape[1]:
            kw["enable_gqa"] = True
            return F.scaled_dot_product_attention(q, k, v, **kw)
        return F.scaled_dot_product_attention(
            q, M.kv_repeat(k, q.shape[1] // k.shape[1]),
            M.kv_repeat(v, q.shape[1] // k.shape[1]), **kw)


def _d(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _randn(shape, mult=1.0, dev=DEV):
    return (torch.randn(*shape, device=dev, dtype=DT) * mult)


def h1_score_rounding() -> bool:
    """H1：eager 误差随分数幅度线性增长，fused 不随幅度增长，且 |eager−fused| ≈ |eager−fp32|。"""
    print("\n== H1 分数矩阵的 bf16 舍入 ==")
    print(f"{'q倍数':>6}{'分数 bf16vs32':>15}{'eager vs fp32':>15}{'fused vs fp32':>15}"
          f"{'eager vs fused':>16}{'|ef|/|e32|':>12}")
    rows = []
    for mult in (1, 3, 10, 30):
        torch.manual_seed(SEED)
        q = _randn((1, 12, 128, 128), mult)
        k, v = _randn((1, 2, 128, 128)), _randn((1, 2, 128, 128))
        kr = M.kv_repeat(k, 6)
        sc32 = torch.matmul(q.float(), kr.float().transpose(-2, -1)) / 128 ** 0.5
        sc16 = torch.matmul(q, kr.transpose(-2, -1)) / 128 ** 0.5
        ref = _eager(q.float(), k.float(), v.float(), 128)
        e, f = _eager(q, k, v, 128), _fused(q, k, v, 128)
        row = dict(mult=mult, score_bf16_vs_fp32=_d(sc32, sc16),
                   eager_vs_fp32=_d(ref, e), fused_vs_fp32=_d(ref, f), eager_vs_fused=_d(e, f))
        row["ratio_diff_over_eager_err"] = row["eager_vs_fused"] / max(row["eager_vs_fp32"], 1e-30)
        rows.append(row)
        print(f"{mult:>6}{row['score_bf16_vs_fp32']:>15.3e}{row['eager_vs_fp32']:>15.3e}"
              f"{row['fused_vs_fp32']:>15.3e}{row['eager_vs_fused']:>16.3e}"
              f"{row['ratio_diff_over_eager_err']:>12.3f}")
    ok = (all(r["eager_vs_fp32"] > r["fused_vs_fp32"] for r in rows)
          and rows[-1]["eager_vs_fp32"] > 10 * rows[0]["eager_vs_fp32"]
          and rows[-1]["fused_vs_fp32"] < 3 * rows[0]["fused_vs_fp32"]
          and all(r["ratio_diff_over_eager_err"] > 0.9 for r in rows))
    print(f"  ⇒ H1 {'成立' if ok else '不成立'}：eager 误差随幅度增长而 fused 不增长，"
          f"且差异 ≈ eager 自身误差")
    return ok, rows


def h2_mask() -> tuple[bool, dict]:
    """H2：构造"未来 key 的 value 极大"的输入 —— 若融合侧忽略掩码，它会把未来 token 带进输出。

    **构造要点（首版写错过，这里逐条说明）**：
      ① `v` 的巨大值只放在**最后 4 个位置**，且**从不**被后续赋值覆盖
         （首版用 `v[i]=1.0` 把同一位置刚设的 `100.0` 覆盖掉了 ⇒ 退化，两个参照物相等）；
      ② `q`、`k` 都非零，保证分数不是全 0（全 0 也会让测试退化）；
      ③ 查询 i ≤ 27 时，位置 ≥ 28 全在未来 ⇒ 带掩码应得 ~1.0，无掩码应得 ~13.4。
    """
    print("\n== H2 掩码语义（未来 key 的 value 极大）==")
    torch.manual_seed(SEED)
    s = T = 32
    q = torch.zeros(1, 4, s, 8, device=DEV, dtype=DT)
    k = torch.zeros(1, 4, T, 8, device=DEV, dtype=DT)
    v = torch.zeros(1, 4, T, 8, device=DEV, dtype=DT)
    q[0, :, :, 0] = 1.0
    k[0, :, :, 0] = 1.0
    v[0, :, :, 0] = 1.0
    v[0, :, T - 4:, 0] = 100.0          # 只在最后 4 个位置放大，之后不再改写
    e_masked = _eager(q, k, v, 8)
    e_open = _eager(q, k, v, 8, mask=False)
    f = _fused(q, k, v, 8)
    res = dict(eager_masked_max=float(e_masked.abs().max()),
               eager_unmasked_max=float(e_open.abs().max()),
               fused_max=float(f.abs().max()),
               masked_vs_unmasked=_d(e_masked, e_open),
               fused_vs_masked=_d(e_masked, f), fused_vs_unmasked=_d(e_open, f))
    print(f"  eager(带掩码) max={res['eager_masked_max']:.3f} | "
          f"eager(无掩码) max={res['eager_unmasked_max']:.3f} | fused max={res['fused_max']:.3f}")
    print(f"  |eager带掩码 − eager无掩码| = {res['masked_vs_unmasked']:.3e}  ← **测试有效性前提**")
    print(f"  |fused − eager带掩码| = {res['fused_vs_masked']:.3e} ; "
          f"|fused − eager无掩码| = {res['fused_vs_unmasked']:.3e}")
    degenerate = res["masked_vs_unmasked"] < 1.0
    if degenerate:
        print("  ⇒ 构造退化（两个参照物几乎相同）⇒ **本项测试无效**，不得据此下结论")
        res["degenerate"] = True
        return False, res
    res["degenerate"] = False
    # 判据必须是**比较**（融合更贴哪一个参照物），不能用绝对阈值：
    # 本测试故意构造极端输入，bf16 舍入误差本身就比常规大（实测 3.1e-2 / max 13.375 = 0.6 ULP）。
    h2 = res["fused_vs_unmasked"] < res["fused_vs_masked"]
    print(f"  ⇒ H2（融合侧忽略掩码）{'成立' if h2 else '**不成立**'}："
          f"融合结果贴着{'**无掩码**' if h2 else '**带掩码**'}的 eager"
          f"（{res['fused_vs_masked']:.3e} vs {res['fused_vs_unmasked']:.3e}）")
    return h2, res


def h3_gqa() -> tuple[bool, dict]:
    """H3：`enable_gqa=True` vs 手工展开成实体 k/v —— 逐位相同则 GQA 广播无差错。"""
    print("\n== H3 GQA 广播（enable_gqa vs 手工 kv_repeat）==")
    res = {}
    for tag, (qh, kh, hd) in (("target 12/2/d128", (12, 2, 128)), ("draft 14/2/d64", (14, 2, 64))):
        torch.manual_seed(SEED)
        q, k, v = _randn((1, qh, 64, hd), 10), _randn((1, kh, 64, hd)), _randn((1, kh, 64, hd))
        a = _fused(q, k, v, hd, gqa=True)
        b = _fused(q, k, v, hd, gqa=False)
        e = _eager(q, k, v, hd)
        res[tag] = dict(gqa_vs_manual=_d(a, b), eager_vs_manual=_d(e, b), eager_vs_gqa=_d(e, a))
        print(f"  {tag:<18} |gqa − manual| = {res[tag]['gqa_vs_manual']:.3e} ; "
              f"|eager − manual| = {res[tag]['eager_vs_manual']:.3e} ; "
              f"|eager − gqa| = {res[tag]['eager_vs_gqa']:.3e}")
    h3 = any(r["gqa_vs_manual"] > 1e-6 for r in res.values())
    print(f"  ⇒ H3（GQA 广播出错）{'成立' if h3 else '**不成立**'}：两条路线逐位相同")
    return h3, res


def h4_scale() -> tuple[bool, dict]:
    """H4：融合结果最贴近哪一个 **eager 变体**（不同 scale）。

    **首版写错过**：把"fused vs fp32"也塞进同一个 dict 再取 `min` ⇒ 必然选中 fp32 那条
    （量纲不同、数值天然最小），得出"H4 成立"的错误结论。这里只在**同量纲的 eager 变体**之间比较。
    """
    print("\n== H4 缩放是否被重复施加 ==")
    torch.manual_seed(SEED)
    q, k, v = _randn((1, 12, 64, 128), 10), _randn((1, 2, 64, 128)), _randn((1, 2, 64, 128))
    f = _fused(q, k, v, 128)
    variants = {"正确 scale 1/√d": _eager(q, k, v, 128),
                "scale=1.0": _eager(q, k, v, 128, scale=1.0),
                "多乘一次 1/√d": _eager(q, k, v, 128, scale=(128 ** -0.5) * (128 ** -0.5))}
    res = {name: _d(t, f) for name, t in variants.items()}
    # 独立参考：fused 与 fp32 真值的距离（**只作对照，不参与 argmin**）
    res_ref = _d(_eager(q.float(), k.float(), v.float(), 128), f)
    for name, val in res.items():
        print(f"  |fused − eager[{name}]| = {val:.3e}")
    print(f"  （对照，不参与 argmin）|fused − eager_fp32[正确 scale]| = {res_ref:.3e}")
    best = min(res, key=res.get)
    h4 = best != "正确 scale 1/√d"
    print(f"  ⇒ 同量纲里最接近的是 **{best}** ⇒ H4（重复缩放）{'成立' if h4 else '**不成立**'}")
    return h4, dict(comparisons=res, best=best, fused_vs_fp32_correct_scale=res_ref)


def main() -> int:
    print(f"device={DEV} torch={torch.__version__} cuda={torch.version.cuda} "
          f"cudnn={torch.backends.cudnn.version()} "
          f"cudnn_sdp_enabled={torch.backends.cuda.cudnn_sdp_enabled()}")
    if DEV.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(0)}")
    h1, r1 = h1_score_rounding()
    h2, r2 = h2_mask()
    h3, r3 = h3_gqa()
    h4, r4 = h4_scale()
    print("\n== 汇总结论 ==")
    print(f"  H1 分数 bf16 舍入（eager 侧） : {'**成立**' if h1 else '不成立'}")
    print(f"  H2 掩码被忽略                : {'成立' if h2 else '不成立'}")
    print(f"  H3 GQA 广播错误              : {'成立' if h3 else '不成立'}")
    print(f"  H4 缩放重复施加              : {'成立' if h4 else '不成立'}")
    if h1 and not (h2 or h3 or h4):
        print("  ⇒ 差异来源 = **eager 基线在 softmax 前把分数舍入成 bf16**；"
              "融合路径在尖峰注意力下更接近 fp32。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
