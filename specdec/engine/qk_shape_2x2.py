"""qk_shape_2x2.py — 位置 8209 的 **QKᵀ 2×2 执行形状分解**（纯逻辑，CPU 可测）。

## 目标（**只**看 QK^T / sqrt(d) 这一个算子）
固定 Q/K 数值，只改变 `torch.matmul(q, k_rep.transpose(-2,-1))` 的**整体执行形状**：

| cell | Query 行数 M | Key 长度 N | Q 来源 | K 来源 |
| --- | --- | --- | --- | --- |
| A | 1 | 8209 | `q1`（真实 S 的 q_rope，(1,12,1,128)） | `k8209`（(1,2,8209,128)） |
| B | 1 | 8212 | `q1` | `k8212`（(1,2,8212,128)） |
| C | 4 | 8209 | `q4_real`（真实 B 的 q_rope，(1,12,4,128)） | `k8209` |
| D | 4 | 8212 | `q4_real` | `k8212` |
| E | 4 | 8209 | `q4_repeat`（`q1` 在 query-row 维重复 4 次） | `k8209` |
| F | 4 | 8212 | `q4_repeat` | `k8212` |

`A` 必须**逐位**复现真实 S 路径的 `qk_scores_pre_mask`；`D` 必须逐位复现真实 B 路径的该张量
（**复现门**；不过门 ⇒ 停止解释，**不**换 layout/公式/dtype/参数重试）。

## 表达式（与 `model_self.py` **完全一致**，不得换成 einsum/手写循环/FP32 累加）
```python
k_rep = kv_repeat(k_all, heads // kv_heads)          # ★ 复用生产实现
scores_raw    = torch.matmul(q, k_rep.transpose(-2, -1))
scores_scaled = scores_raw / (head_dim ** 0.5)       # 生产写法（不是 math.sqrt）
```
`raw` 与 `scaled` **分别记录**：若 raw 已不同 ⇒ 边界只能定位为 **matmul**；
若 raw 相同而 scaled 不同 ⇒ 定位为**标量缩放**。

## 统一比较口径
batch 0、**全部 12 个 query head**、**query row 0**、**key 列 0:8209**
（B 的 3 个额外候选 key 与 C/D/E/F 的其余 3 个 query 行都**不进**比较区）。
"""
from __future__ import annotations

import math

import torch

from . import diag_checks as DC
from . import model_self as M
from . import sb_align as A

HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
N_COMMON = 8209                     # 共同可见 key 数（key 位置 0..8208）


class QKError(RuntimeError):
    """2×2 构造/复现门/分类的实现或契约错误（不是引擎错误）。"""


def q_source(kind: str, q1, q4_real):
    """按 cell 定义取 Q（**不改数值**，只做重复/切片）。"""
    if kind == "q1":
        return q1
    if kind == "q4_real":
        return q4_real
    if kind == "q4_repeat":
        # ★ 内容控制：把 q1（1 行）在 query-row 维重复 4 次 ⇒ 其余 3 行内容**与第 0 行完全相同**
        return q1.repeat(1, 1, 4, 1)
    raise QKError(f"未知 Q 来源：{kind}")


def k_source(kind: str, k8209, k8212):
    if kind == "k8209":
        return k8209
    if kind == "k8212":
        return k8212
    raise QKError(f"未知 K 来源：{kind}")


CELLS: tuple = (
    ("A", 1, N_COMMON, "q1", "k8209"),
    ("B", 1, 8212, "q1", "k8212"),
    ("C", 4, N_COMMON, "q4_real", "k8209"),
    ("D", 4, 8212, "q4_real", "k8212"),
    ("E", 4, N_COMMON, "q4_repeat", "k8209"),
    ("F", 4, 8212, "q4_repeat", "k8212"),
)


def k_rep_prod(k, *, heads: int = HEADS, kv_heads: int = KV_HEADS):
    """用**生产实现**的 `kv_repeat` 把 K 扩到 `heads` 个头（不得手写另一套广播）。"""
    DC.require(heads % kv_heads == 0, "heads 必须能被 kv_heads 整除", f"{heads}/{kv_heads}")
    rep = M.kv_repeat(k, heads // kv_heads)
    DC.require(int(rep.shape[1]) == heads, "kv_repeat 输出头数不符", f"{tuple(rep.shape)}")
    # 逐头核对映射：第 h 个 query head 必须等于第 h // repeat 个 KV head
    rep_f = heads // kv_heads
    for h in (0, rep_f - 1, rep_f, heads - 1):
        DC.require(bool(torch.equal(rep[:, h], k[:, h // rep_f])),
                   "kv_repeat 映射与 h//repeat 不符", f"h={h}")
    return rep


def scores_pair(q, k_rep, *, head_dim: int = HEAD_DIM):
    """与 `model_self.py` **完全一致**的两步表达式；返回 `(raw, scaled)`（**保留各自真实 dtype**）。"""
    DC.require(q.dim() == 4 and k_rep.dim() == 4, "q/k 必须 4 维", f"{q.shape}/{k_rep.shape}")
    raw = torch.matmul(q, k_rep.transpose(-2, -1))
    scaled = raw / (head_dim ** 0.5)
    return raw, scaled


def row0_prefix(scores):
    """统一比较区：`[0, :, 0, :8209]`（batch 0、12 头、query row 0、共同可见 key）。"""
    DC.require(scores.dim() == 4, "scores 必须 4 维", str(tuple(scores.shape)))
    DC.require(scores.shape[2] >= 1, "scores 的 query 维为空", str(tuple(scores.shape)))
    DC.require(scores.shape[3] >= N_COMMON, "scores 的 key 维不足 8209", str(tuple(scores.shape)))
    return scores[0, :, 0, :N_COMMON]


def _layout(t) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()), n_elem=int(t.numel()))


def _per_head(a, b) -> list:
    """按 query head 的差异分布（比较区已是 `(heads, N)`）。"""
    out = []
    for h in range(int(a.shape[0])):
        x, y = a[h].float(), b[h].float()
        d = (x - y).abs()
        out.append(dict(head=h, n_diff=int((a[h] != b[h]).sum()), max_abs=float(d.max()),
                        mean_abs=float(d.mean()), exact_equal=bool(torch.equal(a[h], b[h]))))
    return out


def compare(a, b, name: str, *, max_coords: int = 64) -> dict:
    """两项的**完整**比较记录（原始 dtype；仅统计时转 FP32）。"""
    DC.assert_same_shape(a, b, name)
    DC.require(str(a.dtype) == str(b.dtype), f"{name}: dtype 不同", f"{a.dtype}/{b.dtype}")
    x, y = a.float(), b.float()
    d = (x - y).abs()
    neq = (a != b)
    flat = int(d.reshape(-1).argmax())
    idx = [int(v) for v in torch.unravel_index(torch.tensor(flat), d.shape)]
    coords = [[int(v) for v in c] for c in torch.nonzero(neq, as_tuple=False)][:max_coords]
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    DC.require(finite, f"{name}: 含非有限元素")
    return dict(name=name, a_layout=_layout(a), b_layout=_layout(b),
                dtype=str(a.dtype), exact_equal=bool(torch.equal(a, b)),
                n_diff=int(neq.sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                max_diff_index=idx, a_at_max=float(x[tuple(idx)]), b_at_max=float(y[tuple(idx)]),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b),
                n_coords_saved=len(coords), coords=coords,
                coords_total=int(neq.sum()),
                per_head=_per_head(a, b), finite=finite)


def classify(cmp: dict) -> dict:
    """按用户给定的规则判定 M 效应 / N 效应 / 交互（**逐位**确定性，不用显著性措辞）。

    * M 效应：`A≠C`（N=8209）与 `B≠D`（N=8212）**都**成立 ⇒ 支持"query 行数改变 QK kernel 数值路径"；
    * N 效应：`A≠B`（M=1）与 `C≠D`（M=4）**都**成立 ⇒ 支持"key 长度改变 QK kernel 数值路径"；
    * 只要某一侧的 M 效应只在**一个** N 下出现、或两侧方向不同 ⇒ 记 **interaction**；
    * 另报"哪个因素（或组合）复现真实 S/B 的那 3 个差异"——用 `A/D` 的真实对照做参照。
    """
    def ne(k):
        return bool(cmp[k]["exact_equal"] is False)

    m_at_N8209 = ne("A_vs_C")      # M 效应（N 固定 8209）
    m_at_N8212 = ne("B_vs_D")      # M 效应（N 固定 8212）
    n_at_M1 = ne("A_vs_B")         # N 效应（M 固定 1）
    n_at_M4 = ne("C_vs_D")         # N 效应（M 固定 4）
    m_effect = bool(m_at_N8209 and m_at_N8212)
    n_effect = bool(n_at_M1 and n_at_M4)
    interaction = bool((m_at_N8209 != m_at_N8212) or (n_at_M1 != n_at_M4))
    return dict(
        m_effect=m_effect, n_effect=n_effect, interaction=interaction,
        detail=dict(m_effect_at_N8209=m_at_N8209, m_effect_at_N8212=m_at_N8212,
                    n_effect_at_M1=n_at_M1, n_effect_at_M4=n_at_M4),
        real_S_vs_B=dict(n_diff=cmp["A_vs_D"]["n_diff"], max_abs=cmp["A_vs_D"]["max_abs"]),
        m_only_comparisons=dict(A_vs_B=dict(n_diff=cmp["A_vs_B"]["n_diff"],
                                            max_abs=cmp["A_vs_B"]["max_abs"]),
                                C_vs_D=dict(n_diff=cmp["C_vs_D"]["n_diff"],
                                            max_abs=cmp["C_vs_D"]["max_abs"])),
        n_only_comparisons=dict(A_vs_C=dict(n_diff=cmp["A_vs_C"]["n_diff"],
                                            max_abs=cmp["A_vs_C"]["max_abs"]),
                                B_vs_D=dict(n_diff=cmp["B_vs_D"]["n_diff"],
                                            max_abs=cmp["B_vs_D"]["max_abs"])),
        content_control=dict(C_vs_E=dict(n_diff=cmp["C_vs_E"]["n_diff"],
                                         max_abs=cmp["C_vs_E"]["max_abs"]),
                             D_vs_F=dict(n_diff=cmp["D_vs_F"]["n_diff"],
                                         max_abs=cmp["D_vs_F"]["max_abs"]),
                             row0_unaffected_by_other_rows=bool(
                                 cmp["C_vs_E"]["exact_equal"] and cmp["D_vs_F"]["exact_equal"])),
        reading=("逐位确定性实验：`!=` 即「该形状变化改变了这一区域的输出」，`==` 即「没改变」；"
                 "**不**使用统计显著性措辞；shape-sensitive **不**等同于实现 bug。"))


def attr_value(cmp: dict, key: str):
    """从比较记录取字段（供汇总脚本用，避免各处重复写 key）。"""
    return cmp[key]


def expect_fail_gate(gate: dict) -> bool:
    """复现门（A vs 真实S、D vs 真实B）是否通过（两处都必须 `exact_equal`）。"""
    return bool(gate.get("A_vs_real_S", {}).get("exact_equal") is True
                and gate.get("D_vs_real_B", {}).get("exact_equal") is True)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest() -> int:
    np_ = nf = 0

    def check(name, cond, extra=None):
        nonlocal np_, nf
        if cond:
            np_ += 1
            print(f"  [ok] {name}")
        else:
            nf += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    torch.manual_seed(0)
    print("== 1) cell 构造 / 复现门所需的形状 ==")
    q1 = torch.randn(1, HEADS, 1, HEAD_DIM).to(torch.bfloat16)
    q4 = torch.randn(1, HEADS, 4, HEAD_DIM).to(torch.bfloat16)
    q4[0, :, 0, :] = q1[0, :, 0, :]                       # 真实 B 的第 0 行 == S 的行
    k8209 = torch.randn(1, KV_HEADS, N_COMMON, HEAD_DIM).to(torch.bfloat16)
    k8212 = torch.cat([k8209, torch.randn(1, KV_HEADS, 3, HEAD_DIM).to(torch.bfloat16)], dim=2)
    check("★ 输入形状符合约定：q1 (1,12,1,128)、q4 (1,12,4,128)、k (1,2,8209|8212,128)",
          tuple(q1.shape) == (1, 12, 1, 128) and tuple(q4.shape) == (1, 12, 4, 128)
          and tuple(k8209.shape) == (1, 2, 8209, 128) and tuple(k8212.shape) == (1, 2, 8212, 128))
    check("★ 控制 1/2：`q1 == q4[:,:,0:1,:]`、`k8209 == k8212[:,:,:8209,:]` 逐位成立",
          bool(torch.equal(q1, q4[:, :, 0:1, :])) and bool(torch.equal(k8209, k8212[:, :, :8209, :])))
    q4_rep = q_source("q4_repeat", q1, q4)
    check("★ 内容控制 `q4_repeat`：4 行完全同内容（= q1 行重复）",
          tuple(q4_rep.shape) == (1, 12, 4, 128)
          and all(bool(torch.equal(q4_rep[:, :, j], q1[:, :, 0])) for j in range(4)))

    print("== 2) GQA repeat（用生产 `kv_repeat`）==")
    kr = k_rep_prod(k8209)
    kr2 = M.kv_repeat(k8209, HEADS // KV_HEADS)
    check("★ `k_rep_prod` 就是生产 `kv_repeat`（逐位相同）且头数=12",
          bool(torch.equal(kr, kr2)) and int(kr.shape[1]) == 12)
    check("★ 第 h 个 query head 对应第 h//6 个 KV head（抽 4 个位置核对）",
          all(bool(torch.equal(kr[:, h], k8209[:, h // 6])) for h in (0, 5, 6, 11)))

    print("== 3) raw / scaled 两步与 row0 前缀切片 ==")
    raw_a, sc_a = scores_pair(q1, kr)
    check("★ raw 是 matmul 结果、scaled = raw / (head_dim ** 0.5)（生产写法）",
          tuple(raw_a.shape) == (1, 12, 1, N_COMMON)
          and bool(torch.equal(sc_a, raw_a / (HEAD_DIM ** 0.5))))
    check("★ `head_dim ** 0.5` 与 `math.sqrt(head_dim)` 在本模型取同一值（记录该事实）",
          (HEAD_DIM ** 0.5) == math.sqrt(HEAD_DIM), (HEAD_DIM ** 0.5, math.sqrt(HEAD_DIM)))
    s = row0_prefix(raw_a)
    check("★ row0 前缀切片 = `[0,:,0,:8209]`（形状 (12,8209)）",
          tuple(s.shape) == (12, N_COMMON) and bool(torch.equal(s, raw_a[0, :, 0, :N_COMMON])))
    bad = 0
    try:
        row0_prefix(torch.zeros(1, 12, 1, 8000))
    except DC.CheckError:
        bad += 1
    check("★ key 维不足 8209 时**报错**（不静默截断）", bad == 1)

    print("== 4) 六格执行 + 比较字段 ==")
    got = {}
    for name, m_, n_, qk, kk in CELLS:
        q = q_source(qk, q1, q4)
        k = k_source(kk, k8209, k8212)
        raw, sc = scores_pair(q, k_rep_prod(k))
        got[name] = dict(raw=raw, scaled=sc,
                         raw_row0=row0_prefix(raw), scaled_row0=row0_prefix(sc))
    check("★ 六个 cell 全部构造成 (1,12,M,N)（M/N 与定义一致）",
          all(tuple(got[n]["raw"].shape) == (1, 12, m_, N)
              for n, m_, N, _q, _k in CELLS), {n: tuple(got[n]["raw"].shape) for n, *_ in CELLS})
    c = compare(got["A"]["raw_row0"], got["A"]["raw_row0"].clone(), "A_vs_A")
    check("★ 比较记录含全部必需字段（layout/exact/n_diff/max/mean/index/两侧值/sha16/坐标/分头）",
          all(k in c for k in ("a_layout", "b_layout", "dtype", "exact_equal", "n_diff", "max_abs",
                              "mean_abs", "max_diff_index", "a_at_max", "b_at_max", "sha16_a",
                              "sha16_b", "coords", "per_head", "finite"))
          and c["exact_equal"] is True and len(c["per_head"]) == 12, sorted(c))
    c2 = compare(got["A"]["raw_row0"], got["B"]["raw_row0"], "A_vs_B")
    check("★ 差异坐标列表可用（A vs B 的 `coords_total == n_diff`，且列表不超上限）",
          c2["coords_total"] == c2["n_diff"] and len(c2["coords"]) <= 64, c2["n_diff"])

    print("== 5) 主效应 / 交互分类（合成数据） ==")
    def mk(exact):
        return dict(exact_equal=exact, n_diff=0 if exact else 1, max_abs=0.0 if exact else 1.0)
    only_M = {"A_vs_C": mk(False), "B_vs_D": mk(False), "A_vs_B": mk(True), "C_vs_D": mk(True),
              "A_vs_D": mk(False), "C_vs_E": mk(True), "D_vs_F": mk(True)}
    r = classify(only_M)
    check("★ 只有 M 效应：`m_effect=True, n_effect=False, interaction=False`",
          r["m_effect"] and not r["n_effect"] and not r["interaction"], r["detail"])
    only_N = {"A_vs_C": mk(True), "B_vs_D": mk(True), "A_vs_B": mk(False), "C_vs_D": mk(False),
              "A_vs_D": mk(False), "C_vs_E": mk(True), "D_vs_F": mk(True)}
    r2 = classify(only_N)
    check("★ 只有 N 效应：`n_effect=True, m_effect=False, interaction=False`",
          r2["n_effect"] and not r2["m_effect"] and not r2["interaction"])
    both = {"A_vs_C": mk(False), "B_vs_D": mk(False), "A_vs_B": mk(False), "C_vs_D": mk(False),
            "A_vs_D": mk(False), "C_vs_E": mk(True), "D_vs_F": mk(True)}
    r3 = classify(both)
    check("★ M 与 N 都有影响：两者都报为 True（**不**强行选一个）",
          r3["m_effect"] and r3["n_effect"] and not r3["interaction"])
    inter = {"A_vs_C": mk(False), "B_vs_D": mk(True), "A_vs_B": mk(True), "C_vs_D": mk(True),
             "A_vs_D": mk(False), "C_vs_E": mk(True), "D_vs_F": mk(True)}
    r4 = classify(inter)
    check("★ 交互：某一侧 M 效应只在**一个** N 下出现 ⇒ `interaction=True`",
          r4["interaction"] is True, r4["detail"])

    print("== 6) 复现门与内容控制行为 ==")
    check("★ 复现门：A/D 都逐位相等 ⇒ 通过；任一不等 ⇒ 不通过",
          expect_fail_gate({"A_vs_real_S": dict(exact_equal=True), "D_vs_real_B": dict(exact_equal=True)}) is True
          and expect_fail_gate({"A_vs_real_S": dict(exact_equal=True), "D_vs_real_B": dict(exact_equal=False)}) is False
          and expect_fail_gate({}) is False)
    r5 = classify({"A_vs_C": mk(False), "B_vs_D": mk(False), "A_vs_B": mk(False), "C_vs_D": mk(False),
                   "A_vs_D": mk(False), "C_vs_E": mk(False), "D_vs_F": mk(True)})
    check("★ 内容控制失败（C≠E）⇒ 如实标出 `row0_unaffected_by_other_rows=False`",
          r5["content_control"]["row0_unaffected_by_other_rows"] is False)
    print(f"\nQK 形状 2×2（CPU 门）：PASS {np_} / FAIL {nf}")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
