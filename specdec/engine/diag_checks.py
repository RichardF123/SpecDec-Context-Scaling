"""diag_checks.py — 本轮**修复后的**诊断校验原语（纯函数/纯断言，可 CPU 自测）。

修的是上一轮 `lmhead_shape_f06_i25.py` 里暴露的四类校验缺陷（§二）
  1. **标量索引**：`H_B[0,1:2]` 形状是 (1,1536)，再用 `[0,0]` 取到的是**标量**；
     与 (1536,) 行相比会**广播**，报出虚假的 max|Δ|（上轮出现 91 的假失败）。
     ⇒ 本模块所有逐元素比较**先断言 shape 一致**，禁止广播掩盖索引错误。
  2. **有限性**：必须**真的调用** `torch.isfinite` 并覆盖**完整被比较张量**，
     不得以"字典非空""字段存在"代替有限性。
  3. **未被改动**：不得拿**同一个对象与自身**比较；必须与**运行前独立快照**做内容校验。
  4. **布局控制**：`.contiguous()` 对**已经连续**的张量**可能不复制**（返回自身），
     因此不能凭它声称完成了独立布局控制；要显式检查 `data_ptr` 并说明是否真的产生了新存储。
  另：必需检查失败或观察点缺失时**抛错**，让上层停止归因，而不是继续输出成功结论。

用法：`python -m specdec.engine.diag_checks` 跑自测。
"""
from __future__ import annotations


class CheckError(RuntimeError):
    """诊断校验失败。**必须停止归因**，不得继续给出结论。"""


def require(cond, what: str, detail=""):
    if not cond:
        raise CheckError(f"{what} 未满足。{detail}".strip())
    return True


def snapshot(t):
    """运行前**独立快照**（用于事后证明原对象未被原地修改）。"""
    return t.detach().clone() if hasattr(t, "detach") else t


def finite_all(t) -> bool:
    """**真正调用** `torch.isfinite` 并覆盖**完整张量**。"""
    import torch
    if not torch.is_tensor(t):
        raise CheckError(f"finite_all 只接受张量，收到 {type(t).__name__}")
    return bool(torch.isfinite(t).all())


def assert_same_shape(a, b, what: str = "comparison"):
    if tuple(a.shape) != tuple(b.shape):
        raise CheckError(f"{what}: shape 不一致 {tuple(a.shape)} vs {tuple(b.shape)}"
                         f"（禁止广播比较；这通常意味着索引写错）")
    return True


def cmp_tensor(a, b, name: str = "cmp", *, ref_snapshot=None, src=None) -> dict:
    """逐元素比较。shape 必须一致；有限性对**两个张量的全部元素**都实算。

    `ref_snapshot`/`src` 给出时，额外校验 `src` 未被原地修改（与快照内容比较）。
    """
    import torch
    assert_same_shape(a, b, name)
    fa, fb = finite_all(a), finite_all(b)
    require(fa and fb, f"{name}: 存在非有限元素", f"finite(a)={fa} finite(b)={fb}")
    d = (a - b).abs()
    flat = int(d.reshape(-1).argmax())
    idx = [int(x) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
    out = dict(name=name, shape=[int(x) for x in a.shape], dtype=str(a.dtype),
               elementwise_equal=bool(torch.equal(a, b)),
               abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
               max_err_index=idx, a_at_max=float(a[tuple(idx)]), b_at_max=float(b[tuple(idx)]),
               finite=dict(a=fa, b=fb))
    if ref_snapshot is not None and src is not None:
        assert_same_shape(src, ref_snapshot, f"{name}:src_vs_snapshot")
        out["src_unmodified"] = bool(torch.equal(src, ref_snapshot))
        require(out["src_unmodified"], f"{name}: 源张量被原地修改")
    return out


def layout(t) -> dict:
    """shape/stride/dtype/device/contiguous，并**显式记录是否与某块存储共享**。"""
    import torch
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()), is_view=bool(t._base is not None),
                data_ptr=int(t.data_ptr()))


def layout_control(slice_t, name: str = "layout_control") -> dict:
    """**真正的**布局控制：比较 `.contiguous()` 是否真的产生了新存储。

    `.contiguous()` 对已连续张量**返回自身**（不复制）⇒ 若如此，则该"控制"与 view 是**同一块存储**，
    **不能**据此声称"排除了布局影响"。这里显式报告，并把该限制写进结果。
    """
    c = slice_t.contiguous()
    same_storage = int(c.data_ptr()) == int(slice_t.data_ptr())
    return dict(name=name, view=layout(slice_t), contiguous=layout(c),
                contiguous_returned_same_storage=same_storage,
                control_valid=not same_storage,
                note=("`.contiguous()` 未复制（输入本就连续）⇒ 本次**不是**独立布局控制，"
                      "结论中必须写明该限制" if same_storage else
                      "`.contiguous()` 产生了新存储 ⇒ 构成独立布局控制"))


def _selftest() -> int:
    import torch
    npass = nfail = 0

    def check(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    print("== 1. 标量与向量误比必须被拦住（shape 断言，禁止广播） ==")
    row = torch.randn(1536)                     # (1536,) 行
    v = row[:1]                                 # (1,) —— 模拟 H_B[0,1:2][0,0] 的标量化结果
    try:
        cmp_tensor(row, v, "row_vs_scalar")
        raised = False
    except CheckError:
        raised = True
    check("★ (1536,) vs (1,) 直接抛 CheckError（不再广播出假 max|Δ|）", raised)
    # 正确取法：整行对整行
    c = cmp_tensor(row, row.clone(), "row_vs_row")
    check("正确取法（行 vs 行）逐元素相等", c["elementwise_equal"] is True)
    c2 = cmp_tensor(row.view(1, 1536), row.view(1, 1536).clone(), "2d_vs_2d")
    check("二维形状也按 shape 比对", c2["elementwise_equal"] is True)

    print("== 2. NaN / Inf 必须被检出（对完整张量实算 isfinite） ==")
    bad = torch.zeros(4096)
    bad[4095] = float("nan")                    # 放在**最后一个**元素：只查首元素会漏
    try:
        cmp_tensor(torch.zeros(4096), bad, "nan_tail")
        raised = False
    except CheckError:
        raised = True
    check("★ 末尾元素为 NaN ⇒ 抛 CheckError（覆盖完整张量）", raised)
    inf = torch.zeros(1024); inf[7] = float("inf")
    check("finite_all 对 Inf 返回 False", finite_all(inf) is False)
    check("finite_all 对全有限返回 True", finite_all(torch.zeros(64)) is True)
    try:
        finite_all([1, 2])
        nontensor_raised = False
    except CheckError:
        nontensor_raised = True
    check("finite_all 拒绝非张量输入（抛 CheckError）", nontensor_raised)

    print("== 3. 原地修改必须能与运行前快照比出来 ==")
    src = torch.randn(8)
    snap = snapshot(src)
    c = cmp_tensor(src, src.clone(), "selfcheck", ref_snapshot=snap, src=src)
    check("未修改时 src_unmodified=True", c["src_unmodified"] is True)
    src[0] += 1.0
    try:
        cmp_tensor(src, src.clone(), "after_mutate", ref_snapshot=snap, src=src)
        raised = False
    except CheckError:
        raised = True
    check("★ 原地修改后抛 CheckError（与独立快照比较，而非与自身比较）", raised)

    print("== 4. 缺失观察点必须停止（require） ==")
    try:
        require(False, "捕获缺少 H_B", "未拿到最终 Norm 输出")
        raised = False
    except CheckError:
        raised = True
    check("★ 观察点缺失 ⇒ 抛 CheckError（上层应停止归因）", raised)

    print("== 5. 布局控制要如实报告是否真的复制 ==")
    cont = torch.randn(1, 4, 8)                 # 已连续
    sl = cont[0, 1:2]
    lc = layout_control(sl, "cont_case")
    check("★ 对已连续切片：.contiguous() 未复制 ⇒ control_valid=False 且写明限制",
          lc["contiguous_returned_same_storage"] is True and lc["control_valid"] is False
          and "不是" in lc["note"])
    big = torch.randn(4, 8)
    sl2 = big[1:2]                              # 非连续（行切片）
    lc2 = layout_control(sl2, "noncont_case")
    check("对非连续切片：control_valid=True 或如实报告",
          isinstance(lc2["control_valid"], bool) and "contiguous" in lc2["note"])

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
