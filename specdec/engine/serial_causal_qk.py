"""serial_causal_qk.py — QKᵀ 的**逐行 M=1 causal-prefix** 诊断构造（默认关闭；纯逻辑、可 CPU 测）。

## 目的（本轮唯一问题）
在 B/verify 路径的 L0 上，把 QK 改成"**按每个 query 行分别以 M=1 + 真实 causal key 前缀**计算，
再拼回原 block"，其余算子完全不变，用来判定 **QK 的执行形状敏感性**是否是
当前位置 8209 的 AR/Spec token 翻转的**主要持续来源**。

## 生产原式（`model_self.SelfAttention.forward`，**不改**）
```python
scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
```
B 路径：`q = (1,12,4,128)`、`k_rep = (1,12,8212,128)`、`p = T - s = 8208`、`T = 8212`；
原有 causal mask 为 `triu(ones(s,T), diagonal=p+1)` ⇒ 第 `i` 行可见 `j <= p+i`，即**前缀长度 `p+i+1`**。

## 诊断构造（**逐位规则**）
对每个 `i = 0..s-1`：`visible_i = p+i+1`；
```python
qi = q[:, :, i:i+1, :]                      # M = 1（真的以 M=1 执行，不是先算 M=4 再切）
ki = k_rep[:, :, :visible_i, :]             # 真实 causal key 前缀，绝不读未来 key
raw_i    = torch.matmul(qi, ki.transpose(-2, -1))
scaled_i = raw_i / (head_dim ** 0.5)        # 与生产**同一个** matmul 表达式 + 同一个标量除法
```
拼回 `scores_serial_causal: (1,12,s,T)`：第 `i` 行的 `[0:visible_i]` = `scaled_i`（**逐位保持原值**，
只是切片赋值、不做任何算术），`[visible_i:T]` = **有限占位值**（默认 `0.0`）——
这些位置**恰好**是原有 causal mask 覆盖的位置（由本模块的 CPU 门证明逐列对应），
随后由生产原有的 `masked_fill(mask, -inf)` 变成 `-inf`。

## 与生产的差异（只说这一处）
`raw_i` 与 `scaled_i` 被写成两句（生产是一句复合表达式）——与本仓库 `qk_shape_2x2.scores_pair`
**同一写法**，并且该写法在上一轮已被**逐位证明**等于生产保存的 `qk_scores_pre_mask`
（QK 2×2 复现门：`A` 的 scaled 与真实 S 整张量 `n_diff=0`）。

## 布局说明（为什么用 `.contiguous()`）
`q[:, :, i:i+1, :]` 与 `k_rep[:, :, :visible_i, :]` 都是 **view**，其 stride 与 S 路径（s=1）里
那次 matmul 收到的张量**不一定相同**（S 路径的 `q` 来自 s=1 的 reshape，`k_rep` 是 `kv_repeat` 的连续输出）。
`contiguous_rows=True`（默认）会把每行的输入**复制成标准连续布局**，使第 0 行的 matmul 收到的
形状/步长与 S 路径**完全一致** —— 上一轮 QK 2×2 的 A 格正是这种构造，并且**逐位复现**了真实 S。
它只是内存布局规范化（一次拷贝），**不是**公式/dtype/精度改变；实际布局由工具逐行记录。
"""
from __future__ import annotations

import torch

from . import diag_checks as DC

HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128        # 本模型实测值（仅用于自测/默认参数）
PLACEHOLDER = 0.0                             # 未来位置的有限占位值（随后被原有 mask 变成 -inf）


# ------------------------------------------------------------------ 纯逻辑
def row_visible(p: int, i: int, T: int) -> int:
    """第 `i` 行真实可见的前缀长度 = `p + i + 1`（并防御性夹到 `T`）。"""
    return int(min(p + i + 1, T))


def row_scores(q, k_rep, i: int, *, p: int, head_dim: int, contiguous_rows: bool = True) -> dict:
    """**单行**（M=1）QKᵀ×缩放：只用 `k_rep[:, :, :visible_i, :]`。返回原始/scaled 与元数据。"""
    DC.require(q.dim() == 4 and k_rep.dim() == 4, "q/k_rep 必须是 4 维",
               f"{tuple(q.shape)} {tuple(k_rep.shape)}")
    s, T = int(q.shape[2]), int(k_rep.shape[2])
    DC.require(0 <= i < s, f"行号越界：i={i} s={s}")
    DC.require(int(q.shape[1]) == int(k_rep.shape[1]), "q 与 k_rep 的 head 维必须一致",
               f"{tuple(q.shape)} {tuple(k_rep.shape)}")
    DC.require(int(q.shape[3]) == int(k_rep.shape[3]) == int(head_dim), "head_dim 不一致",
               f"{tuple(q.shape)} {tuple(k_rep.shape)} head_dim={head_dim}")
    vis = row_visible(p, i, T)
    qi = q[:, :, i:i + 1, :]
    ki = k_rep[:, :, :vis, :]
    lay_q, lay_k = _layout(qi), _layout(ki)
    if contiguous_rows:
        qi = qi.contiguous()
        ki = ki.contiguous()
    raw = torch.matmul(qi, ki.transpose(-2, -1))          # ← 与生产**同一个** matmul 表达式
    scaled = raw / (head_dim ** 0.5)                      # ← 与生产**同一个**标量除法
    DC.require(tuple(raw.shape) == (int(q.shape[0]), int(q.shape[1]), 1, vis),
               f"行 {i} 的 raw 形状不符", f"{tuple(raw.shape)} 期望 (...,1,{vis})")
    return dict(i=int(i), visible=vis, p=int(p), T=T,
                raw=raw, scaled=scaled,
                layout_q=lay_q, layout_k=lay_k,
                layout_q_used=_layout(qi), layout_k_used=_layout(ki),
                contiguous_rows=bool(contiguous_rows),
                finite=bool(torch.isfinite(raw).all()))


def serial_causal_scores(q, k_rep, *, p: int, head_dim: int, contiguous_rows: bool = True,
                         placeholder: float = PLACEHOLDER):
    """逐行 M=1 + 真实 causal 前缀 → 拼回 `(1,H,s,T)`；返回 `(scores, meta)`。

    `scores[:, :, i, :visible_i]` **逐位**等于第 `i` 行单行计算的 scaled 值；
    `scores[:, :, i, visible_i:]` 全为**有限**占位值，且这些位置恰好被原有 causal mask 覆盖。
    """
    DC.require(q.dim() == 4 and k_rep.dim() == 4, "q/k_rep 必须是 4 维")
    b, h, s, _ = q.shape
    T = int(k_rep.shape[2])
    DC.require(s >= 1 and p == T - s, "口径要求 p == T - s", f"p={p} s={s} T={T}")
    out = torch.empty((b, h, s, T), dtype=q.dtype, device=q.device)
    rows = []
    for i in range(int(s)):
        r = row_scores(q, k_rep, i, p=p, head_dim=head_dim, contiguous_rows=contiguous_rows)
        vis = r["visible"]
        out[:, :, i, :vis] = r["scaled"][:, :, 0, :]       # 逐位赋值（无算术）
        out[:, :, i, vis:] = placeholder                   # 有限占位（随后被 mask 变成 -inf）
        rows.append(r)
    meta = dict(p=int(p), s=int(s), T=T, head_dim=int(head_dim),
                visible=[int(r["visible"]) for r in rows],
                expected_visible=[row_visible(p, i, T) for i in range(int(s))],
                padded=[int(T - r["visible"]) for r in rows],
                placeholder=float(placeholder), contiguous_rows=bool(contiguous_rows),
                rows=rows,
                note="第 i 行只用 K[:p+i+1]；未来位置为有限占位值，随后必须被原有 causal mask 变成 -inf")
    DC.require(meta["visible"] == meta["expected_visible"], "每行可见长度必须等于 p+i+1", str(meta["visible"]))
    return out, meta


def production_like_scores(q, k_rep, *, head_dim: int):
    """**仅用于 CPU 门**：复刻生产原式（一条复合表达式），用来证明 `s=1` 时两者逐位一致。"""
    return torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)


def causal_mask(s: int, T: int, *, device, cache_is_none: bool = False):
    """复刻 `model_self` 的 causal mask 规则（**只用于 CPU 门**，不参与生产路径）。

    * `cache is None`（prefill）：`triu(diagonal=1)`；
    * `s == 1`（decode）：**不加掩码**（返回 `None`）；
    * 否则（verify）：`triu(diagonal=p+1)`，`p = T - s`。
    """
    if cache_is_none:
        return torch.triu(torch.ones(s, T, device=device, dtype=torch.bool), diagonal=1)
    if s == 1:
        return None
    p = T - s
    return torch.triu(torch.ones(s, T, device=device, dtype=torch.bool), diagonal=p + 1)


def apply_mask_like_production(scores, *, cache_is_none: bool = False):
    """按生产的 mask 分支施加掩码（**只用于 CPU 门**）。"""
    s, T = int(scores.shape[2]), int(scores.shape[3])
    m = causal_mask(s, T, device=scores.device, cache_is_none=cache_is_none)
    return scores if m is None else scores.masked_fill(m, float("-inf"))


def _layout(t) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()))


# ------------------------------------------------------------------ H1–H4 分类（纯函数）
def classify_h1(s0_row, b1_row, *, vis: int, qk_coords_expected=None) -> dict:
    """**H1**：S0 与 B1 的 QK 第 0 行在共同可见范围 `0:vis` 上是否逐位一致。

    ⚠️ 口径（CPU 门抓到过的错）：输入是 **4 维** scores `(1,H,s,T)`，
    第 0 行 + 共同可见 key 的切片是 `[0, :, 0, :vis]`（`(H, vis)`）；
    写成 `[0, :, :vis]` 会切到 **s 维**，形状看着"像"却比错了东西。
    """
    DC.require(s0_row.dim() == 4 and b1_row.dim() == 4,
               "classify_h1 需要 4 维 scores (1,H,s,T)", f"{s0_row.dim()}/{b1_row.dim()}")
    DC.require(int(s0_row.shape[3]) >= vis and int(b1_row.shape[3]) >= vis,
               "可见范围超出 scores 长度", f"{tuple(s0_row.shape)} {tuple(b1_row.shape)} vis={vis}")
    a, b = s0_row[0, :, 0, :vis], b1_row[0, :, 0, :vis]
    DC.assert_same_shape(a, b, "H1_scores_row0")
    neq = (a != b)
    coords = [[int(x) for x in c] for c in torch.nonzero(neq).tolist()[:64]]
    out = dict(name="H1_qk_row0_S0_vs_B1", vis=int(vis), shape=[int(x) for x in a.shape],
               exact_equal=bool(torch.equal(a, b)), n_diff=int(neq.sum()),
               max_abs=float((a.float() - b.float()).abs().max()),
               coords=coords, coords_total=int(neq.sum()),
               finite_both=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
               sha16_s0=None, sha16_b1=None,
               reading="逐位一致 ⇒ 干预把 QK 几何恢复成 S 路径（H1 通过）；否则干预实现无效，停止因果解释")
    from . import sb_align as A
    out["sha16_s0"], out["sha16_b1"] = A.stable_sha(a), A.stable_sha(b)
    if qk_coords_expected is not None:
        out["expected_zero_diff"] = True
        out["matches_expected"] = bool(out["exact_equal"])
    return out


def classify_h2(nodes_s0: dict, nodes_b1: dict, chain, chain_out: list, *, vis: int,
                first_div: dict | None = None) -> dict:
    """**H2**：S0 与 B1 的 L0 输出是否恢复；若未恢复，给出**第一个剩余**的独立分歧算子。"""
    zs = nodes_s0["layer_output"][0, -1]
    zb = nodes_b1["layer_output"][0, 0]
    neq = (zs != zb)
    out = dict(name="H2_L0_output_S0_vs_B1", shape=[int(x) for x in zs.shape],
               exact_equal=bool(torch.equal(zs, zb)), n_diff=int(neq.sum()),
               max_abs=float((zs.float() - zb.float()).abs().max()),
               coords=[[int(x) for x in c] for c in torch.nonzero(neq).tolist()[:64]])
    if out["exact_equal"]:
        out["class"] = "L0_restored"
        out["reading"] = "逐位一致 ⇒ QK 对齐**足以**消除本例 L0 的差异"
    else:
        out["class"] = "L0_not_restored"
        out["first_remaining_divergence"] = first_div
        out["reading"] = ("仍不同 ⇒ QK 是一个真实来源，但 L0 内**还有独立**的 shape-sensitive 算子"
                          "（见 `first_remaining_divergence`）；不继续修复")
    return out


def classify_h3(rows_s0: dict, rows_b1: dict, *, n_layers: int) -> dict:
    """**H3**：逐层比较 S0 与 B1（层输入行 / 层输出行），找**第一层**重新出现差异的位置。

    * 层 `i`：输入行逐位相同、输出行不同 ⇒ 该层是**新的独立** shape-sensitive 边界；
    * 层 `i`：输入行已不同 ⇒ 只是**传递**上游差异；
    * 全层输入输出都逐位相同 ⇒ 后面没有任何层重新引入差异。
    """
    layers = []
    first_independent = None
    first_any_diff = None
    for i in range(int(n_layers)):
        a_in, b_in = rows_s0[i]["input"], rows_b1[i]["input"]
        a_out, b_out = rows_s0[i]["output"], rows_b1[i]["output"]
        in_eq = bool(torch.equal(a_in, b_in))
        out_eq = bool(torch.equal(a_out, b_out))
        d_in = int((a_in != b_in).sum())
        d_out = int((a_out != b_out).sum())
        layers.append(dict(layer=int(i), input_exact_equal=in_eq, output_exact_equal=out_eq,
                           input_n_diff=d_in, output_n_diff=d_out,
                           kind=("same" if (in_eq and out_eq) else
                                 ("new_independent_boundary" if in_eq else "transmits_upstream"))))
        if not out_eq and first_any_diff is None:
            first_any_diff = int(i)
        if in_eq and not out_eq and first_independent is None:
            first_independent = int(i)
    return dict(name="H3_layerwise_S0_vs_B1", n_layers=int(n_layers), layers=layers,
                first_layer_with_any_output_diff=first_any_diff,
                first_layer_with_equal_input_and_different_output=first_independent,
                all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"]
                                              for l in layers)),
                note="逐层给出「输入相同/不同」与「输出相同/不同」；**不**用『传递』一词代替证据")


def classify_h4(logits_s0, logits_b1) -> dict:
    """**H4**：最终 logits 与 argmax token 的恢复类别（1/2/3）。"""
    a, b = logits_s0.reshape(-1), logits_b1.reshape(-1)
    DC.assert_same_shape(a, b, "H4_logits")
    same_logits = bool(torch.equal(a, b))
    ta, tb = int(a.argmax()), int(b.argmax())
    cls = 1 if (same_logits and ta == tb) else (2 if ta == tb else 3)
    return dict(name="H4_final_logits_S0_vs_B1", vocab=int(a.numel()),
                logits_exact_equal=same_logits, logits_n_diff=int((a != b).sum()),
                logits_max_abs=float((a.float() - b.float()).abs().max()),
                token_s0=ta, token_b1=tb, token_equal=bool(ta == tb), category=int(cls),
                reading={1: "logits 逐位一致且 token 一致 ⇒ 该局部轨迹被**完全恢复**",
                         2: "logits 不同但 token 一致 ⇒ 只能说 **token 结果恢复**，数值路径仍不同",
                         3: "logits 与 token 都不同 ⇒ **QK 不是唯一来源**"}[cls])


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest() -> int:
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:240]}" if extra is not None else ""))

    torch.manual_seed(0)
    # ---- 纯逻辑：p=3, s=4 的逐行 causal 前缀 ------------------------------------
    p, s, T, hd, H = 3, 4, 7, 4, 2
    q = torch.randn(1, H, s, hd).to(torch.bfloat16)
    k = torch.randn(1, H, T, hd).to(torch.bfloat16)
    sc, meta = serial_causal_scores(q, k, p=p, head_dim=hd)
    check("★ 每行可见长度 == p+i+1（p=3,s=4 ⇒ 4,5,6,7）",
          meta["visible"] == [4, 5, 6, 7] == meta["expected_visible"], meta["visible"])
    check("★ 输出形状 == (1,H,s,T)，且每行可见部分逐位等于该行单行 M=1 计算",
          tuple(sc.shape) == (1, H, s, T)
          and all(torch.equal(sc[:, :, i, :meta["visible"][i]],
                              row_scores(q, k, i, p=p, head_dim=hd)["scaled"][:, :, 0, :])
                  for i in range(s)))
    # 逐行「不读未来 key」的哨兵测试：把**该行前缀之后**的 key 全换成 NaN，
    # 再走同一条逐行路径；若实现真的读到未来位置，NaN 会污染该行的可见区域。
    def _poisoned(i):
        kp = k.clone()
        kp[:, :, meta["visible"][i]:, :] = float("nan")
        return kp

    check("★ 每行**只**用 K[:p+i+1]（把该行未来 key 换成 NaN ⇒ 可见部分仍逐位不变且有限）",
          all(torch.equal(row_scores(q, _poisoned(i), i, p=p, head_dim=hd)["scaled"],
                          row_scores(q, k, i, p=p, head_dim=hd)["scaled"])
              and row_scores(q, _poisoned(i), i, p=p, head_dim=hd)["finite"]
              for i in range(s)))
    check("★ 未来位置 == 有限占位值（0.0），且列范围与 p+i+1 严格互补",
          all((sc[0, 0, i, meta["visible"][i]:].numel() == 0
               or (float(sc[0, 0, i, meta["visible"][i]:].abs().max()) == 0.0
                   and bool(torch.isfinite(sc[0, 0, i, meta["visible"][i]:]).all())))
              for i in range(s))
          and meta["padded"] == [T - v for v in meta["visible"]], meta["padded"])
    m = causal_mask(s, T, device=sc.device)
    check("★ 原有 causal mask 恰好覆盖**且仅覆盖**这些占位位置（逐列对应）",
          all(bool(m[i, meta["visible"][i]:].all()) and not bool(m[i, :meta["visible"][i]].any())
              for i in range(s)))
    post = apply_mask_like_production(sc)
    check("★ mask 后：未来位置全部 -inf，且可见部分逐位未变",
          bool(torch.isinf(post[:, :, :, :]).sum() == H * sum(meta["padded"]))
          and all(torch.equal(post[:, :, i, :meta["visible"][i]], sc[:, :, i, :meta["visible"][i]])
                  for i in range(s)))
    r0 = row_scores(q, k, 0, p=p, head_dim=hd)
    check("★ 第 0 行严格是 M=1、N=p+1（raw 形状 (1,H,1,p+1)）",
          tuple(r0["raw"].shape) == (1, H, 1, p + 1) and r0["visible"] == p + 1)
    # ---- s=1：必须与生产原式逐位一致 ------------------------------------------
    q1 = torch.randn(1, H, 1, hd).to(torch.bfloat16)
    sc1, meta1 = serial_causal_scores(q1, k, p=T - 1, head_dim=hd)
    check("★ s=1 时与**生产原式**逐位一致（同一张量、同一表达式）",
          torch.equal(sc1, production_like_scores(q1, k, head_dim=hd)) and meta1["padded"] == [0])
    # ---- 布局：contiguous_rows 的效果与取值不变 --------------------------------
    kview = k.repeat(1, 1, 2, 1)[:, :, :T, :]          # 制造一个非连续 view（stride 不同）
    a_sc, _ = serial_causal_scores(q, kview, p=p, head_dim=hd, contiguous_rows=True)
    b_sc, _ = serial_causal_scores(q, kview.contiguous(), p=p, head_dim=hd, contiguous_rows=True)
    check("★ `.contiguous()` 只改布局、逐位不改值（view 与已连续输入结果相同）",
          torch.equal(a_sc, b_sc)
          and row_scores(q, kview, 0, p=p, head_dim=hd)["layout_k_used"]["stride"]
          == row_scores(q, kview.contiguous(), 0, p=p, head_dim=hd)["layout_k_used"]["stride"])
    check("★ contiguous_rows=False 时保留 view 布局（供工具记录/对照）",
          row_scores(q, kview, 0, p=p, head_dim=hd, contiguous_rows=False)["layout_k_used"]["is_contiguous"]
          is False)
    # ---- 口径断言 --------------------------------------------------------------
    bad = 0
    try:
        serial_causal_scores(q, k, p=p + 1, head_dim=hd)
    except DC.CheckError:
        bad += 1
    check("★ 口径错误（p ≠ T-s）必须报错", bad == 1)
    bad = 0
    try:
        row_scores(q, k, s, p=p, head_dim=hd)
    except DC.CheckError:
        bad += 1
    check("★ 行号越界必须报错", bad == 1)
    # ---- H1–H4 分类 -----------------------------------------------------------
    h1 = classify_h1(sc, sc.clone(), vis=meta["visible"][0])
    check("★ H1：逐位相同 ⇒ exact_equal=True", h1["exact_equal"] and h1["n_diff"] == 0)
    sc_bad = sc.clone()
    sc_bad[0, 1, 0, 2] = sc_bad[0, 1, 0, 2] + torch.tensor(1.0, dtype=sc_bad.dtype)
    h1b = classify_h1(sc, sc_bad, vis=meta["visible"][0])
    check("★ H1：人为注入 1 个坐标差异 ⇒ 判否且坐标可定位（口径 = [0,:,0,:vis]）",
          (not h1b["exact_equal"]) and h1b["n_diff"] == 1 and h1b["coords"] == [[1, 2]]
          and list(h1b["shape"]) == [H, meta["visible"][0]], h1b["coords"])
    n_s0 = dict(layer_output=torch.randn(1, 1, 8).to(torch.bfloat16))
    n_b1 = dict(layer_output=n_s0["layer_output"].clone())
    h2 = classify_h2(n_s0, n_b1, None, None, vis=4, first_div=None)
    check("★ H2：L0 输出逐位相同 ⇒ 分类 L0_restored", h2["class"] == "L0_restored")
    n_b1b = dict(layer_output=n_s0["layer_output"].clone())
    n_b1b["layer_output"][0, 0, 3] = n_b1b["layer_output"][0, 0, 3] + torch.tensor(1.0, dtype=torch.bfloat16)
    h2b = classify_h2(n_s0, n_b1b, None, None, vis=4, first_div=dict(first_divergence_operator="x"))
    check("★ H2：L0 输出不同 ⇒ 分类 L0_not_restored 并带首个剩余边界",
          h2b["class"] == "L0_not_restored" and h2b["n_diff"] == 1
          and h2b["first_remaining_divergence"]["first_divergence_operator"] == "x")
    base = torch.randn(8).to(torch.bfloat16)
    rows_s0 = {i: dict(input=base.clone(), output=base.clone()) for i in range(4)}
    rows_b1 = {i: dict(input=base.clone(), output=base.clone()) for i in range(4)}
    rows_b1[0]["output"] = rows_b1[0]["output"] + torch.tensor(1.0, dtype=torch.bfloat16)
    rows_b1[1]["input"] = rows_b1[1]["input"] + torch.tensor(1.0, dtype=torch.bfloat16)
    h3 = classify_h3(rows_s0, rows_b1, n_layers=4)
    check("★ H3：层 0「输入同、输出不同」⇒ 新独立边界；层 1「输入已不同」⇒ 传递",
          h3["first_layer_with_equal_input_and_different_output"] == 0
          and h3["layers"][1]["kind"] == "transmits_upstream"
          and h3["layers"][2]["kind"] == "same")
    lg = torch.randn(16)
    check("★ H4 三类：全同 ⇒ 1；仅 token 同 ⇒ 2；token 不同 ⇒ 3",
          classify_h4(lg, lg.clone())["category"] == 1
          and classify_h4(lg, (lg + 0.001 * torch.randn(16)))["category"] == 2
          and classify_h4(lg, -lg)["category"] == 3)
    print(f"\nQK 逐行 M=1 causal-prefix（CPU 门）：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def main() -> int:
    print("== QK 逐行 M=1 causal-prefix 纯逻辑 CPU 门（**不加载模型**）==")
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
