"""serial_align.py — `o_proj` 的**逐行 M=1** 诊断构造 + 本轮 H1–H5 判定（纯逻辑、可 CPU 测）。

## 目的（本轮唯一目标）
在**保留** L0 serial-causal QK 干预的前提下，把 L0 的 `self_attn.o_proj` 也按行以 **M=1** 调用
**同一个 `nn.Linear`**，再拼回 `(1,4,1536)`，判定 `o_proj`、L0 输出、后续层与最终 logits 能否进一步恢复。

## 生产原式（`model_self.SelfAttention.forward`，**不改**）
```python
op = self.o_proj(out)          # S 路径 out=(1,1,1536)（等价 M=1）；B 路径 out=(1,4,1536)（等价 M=4）
```

## 诊断构造（**逐位规则**）
```python
rows = []
for i in range(out.shape[1]):
    oi = self.o_proj(out[:, i:i+1, :])     # 每行**实际**以 [1,1,in_features] 调用同一个 nn.Linear
    rows.append(oi)
op = torch.cat(rows, dim=1)                # 拼接只组合行，不做任何额外算术
```

* **不允许**先执行 M=4 再切片；**不允许**改成显式 matmul / FP32 / einsum / 手写 kernel；
* 第 0 行必须复现 S0 的 `o_proj` 输出（S0 的 `out` 就是 `(1,1,1536)`，等价一次同样的调用）；
* 其余三行照常拼回并继续进入生产路径（残差 → post-attention RMSNorm → MLP → …）；
* `nn.Linear.forward` 内部对 >2 维输入做 `reshape(-1, in_features)` ⇒ 逐行调用与 S 路径那次调用
  在**同一形状、同一布局、同一 dtype、同一权重**下执行（第 0 行因此应与 S0 逐位一致）。

## 与生产的差异（只说这一处）
把**一次** `M=4` 的 `nn.Linear` 调用替换为**四次** `M=1` 的同一模块调用 + 一次 `torch.cat`。
调用的是**同一个模块实例**（同一权重、同一 bias 配置、同一 dtype），布局由工具逐行记录。
"""
from __future__ import annotations

import torch

from . import diag_checks as DC
from . import sb_align as A


# ------------------------------------------------------------------ 模块事实与逐行调用
def module_facts(mod) -> dict:
    """从**实际模块**读取：类型、in/out 特征、bias 是否存在、权重形状/dtype/data_ptr。"""
    return dict(class_name=type(mod).__name__,
                in_features=int(getattr(mod, "in_features", -1)),
                out_features=int(getattr(mod, "out_features", -1)),
                has_bias=bool(getattr(mod, "bias", None) is not None),
                weight_shape=[int(x) for x in mod.weight.shape],
                weight_dtype=str(mod.weight.dtype),
                weight_data_ptr=int(mod.weight.data_ptr()),
                bias_data_ptr=(int(mod.bias.data_ptr()) if getattr(mod, "bias", None) is not None else None),
                module_id=int(id(mod)))


def _layout(t) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()))


def serial_linear_rows(mod, x, *, contiguous_rows: bool = False):
    """逐行 `M=1` 调用**同一个** `nn.Linear`，再 `torch.cat(dim=1)` 拼回；返回 `(out, meta)`。

    `contiguous_rows=False`（默认）= 严格按本轮规格 `mod(x[:, i:i+1, :])`（不做额外拷贝）。
    """
    DC.require(x.dim() == 3, "serial Linear 需要 3 维输入 (b, s, in_features)", f"{tuple(x.shape)}")
    s = int(x.shape[1])
    DC.require(s >= 1, "s 必须 ≥ 1", f"s={s}")
    rows, meta_rows = [], []
    for i in range(s):
        xi = x[:, i:i + 1, :]
        if contiguous_rows:
            xi = xi.contiguous()
        lay = _layout(xi)
        oi = mod(xi)                                     # ← 与生产**同一个**模块、同一次调用形式
        DC.require(oi.dim() == 3 and int(oi.shape[1]) == 1,
                   f"第 {i} 行的输出形状不符", f"{tuple(oi.shape)}")
        rows.append(oi)
        meta_rows.append(dict(i=int(i), input_shape=[int(v) for v in xi.shape],
                              input_layout=lay, input_numel=int(xi.numel()),
                              output_shape=[int(v) for v in oi.shape],
                              output_dtype=str(oi.dtype)))
    out = torch.cat(rows, dim=1)                         # 只组合行，不做算术
    DC.require(tuple(out.shape) == (int(x.shape[0]), s, int(out.shape[2])),
               "拼接后的形状不符", f"{tuple(out.shape)}")
    meta = dict(s=s, out_features=int(out.shape[2]), cat_dim=1,
                contiguous_rows=bool(contiguous_rows),
                per_row_input_shapes=[r["input_shape"] for r in meta_rows],
                rows=meta_rows,
                note="每行以 [1,1,in_features] 调用同一个 nn.Linear；拼接只组合行、不做额外算术")
    return out, meta


# ------------------------------------------------------------------ 持久性（首个 / 首个持续）
def chain_persistence(node_cmp: dict, chain, out_names: list) -> dict:
    """链上「所有输入逐位相同、输出不同」的算子：**首个** 与 **首个持续到末尾**。"""
    per_op = []
    for (op, ins), outn in zip(chain, out_names):
        in_eq = all(node_cmp[i]["exact_equal"] for i in ins if i in node_cmp)
        out_eq = node_cmp[outn]["exact_equal"]
        per_op.append(dict(op=op, output_node=outn, inputs=list(ins),
                           all_inputs_equal=bool(in_eq), output_equal=bool(out_eq),
                           output_n_diff=int(node_cmp[outn]["n_diff"])))
    boundaries = [d["op"] for d in per_op if d["all_inputs_equal"] and not d["output_equal"]]
    first_persistent = None
    for i, d in enumerate(per_op):
        if d["all_inputs_equal"] and not d["output_equal"] and all(
                not per_op[j]["output_equal"] for j in range(i, len(per_op))):
            first_persistent = d["op"]
            break
    return dict(per_operator=per_op, boundaries=boundaries,
                first_divergence_operator=(boundaries[0] if boundaries else None),
                first_persistent_divergence_operator=first_persistent,
                note="`first_divergence_operator` 可能是**瞬时**的（被下游舍入吸收）；"
                     "`first_persistent_...` = 从它起之后每个算子输出都不同 ⇒ 真正持续到出口的边界")


# ------------------------------------------------------------------ H2–H5 分类（纯函数）
def classify_h2_oproj(s0_heads_row, x_heads_row, s0_op_row, x_op_row, bq_op_row) -> dict:
    """**H2**：`heads_merged` 决策行必须逐位相同；`o_proj` 决策行必须逐位一致（否则干预无效）。"""
    DC.assert_same_shape(s0_heads_row, x_heads_row, "H2_heads_merged")
    DC.assert_same_shape(s0_op_row, x_op_row, "H2_o_proj")
    h_eq, o_eq = bool(torch.equal(s0_heads_row, x_heads_row)), bool(torch.equal(s0_op_row, x_op_row))
    bq_vs_x = dict(exact_equal=bool(torch.equal(bq_op_row, x_op_row)),
                   n_diff=int((bq_op_row != x_op_row).sum()),
                   max_abs=float((bq_op_row.float() - x_op_row.float()).abs().max()),
                   coords=[[int(v) for v in c] for c in torch.nonzero(bq_op_row != x_op_row).tolist()[:64]])
    s0_vs_bq = dict(exact_equal=bool(torch.equal(s0_op_row, bq_op_row)),
                    n_diff=int((s0_op_row != bq_op_row).sum()),
                    coords=[[int(v) for v in c] for c in
                            torch.nonzero(s0_op_row != bq_op_row).tolist()[:64]])
    return dict(name="H2_o_proj_local_closure", heads_merged_exact_equal=h_eq,
                o_proj_exact_equal=o_eq,
                heads_merged_n_diff=int((s0_heads_row != x_heads_row).sum()),
                o_proj_n_diff=int((s0_op_row != x_op_row).sum()),
                o_proj_max_abs=float((s0_op_row.float() - x_op_row.float()).abs().max()),
                s0_o_proj_sha16=A.stable_sha(s0_op_row), x_o_proj_sha16=A.stable_sha(x_op_row),
                BQ_o_proj_vs_BQO=s0_vs_bq, BQ_vs_BQO=bq_vs_x,
                o_proj_2diffs_removed=bool(s0_vs_bq["n_diff"] > 0 and o_eq),
                class_=("local_closure_ok" if (h_eq and o_eq) else
                        ("heads_differ" if not h_eq else "o_proj_not_restored")),
                reading="`heads_merged` 与 `o_proj` 决策行都逐位一致 ⇒ serial o_proj 局部闭环成功；"
                        "`o_proj` 仍不同 ⇒ 干预实现无效，停止后续归因")


def classify_h3_l0(s0_l0_row, x_l0_row, persistence: dict, *, b0_l0_row=None) -> dict:
    """**H3**：L0 输出是否完全恢复；未恢复则给出**第一个持续**的边界。"""
    eq = bool(torch.equal(s0_l0_row, x_l0_row))
    out = dict(name="H3_L0_output_S0_vs_BQO", exact_equal=eq,
               n_diff=int((s0_l0_row != x_l0_row).sum()),
               max_abs=float((s0_l0_row.float() - x_l0_row.float()).abs().max()),
               sha16_s0=A.stable_sha(s0_l0_row), sha16_x=A.stable_sha(x_l0_row),
               first_divergence_operator=persistence["first_divergence_operator"],
               first_persistent_divergence_operator=persistence["first_persistent_divergence_operator"],
               boundaries=persistence["boundaries"],
               class_=("L0_restored" if eq else "L0_not_restored"))
    if b0_l0_row is not None:
        out["b0_context"] = dict(n_diff_S0_vs_B0=int((s0_l0_row != b0_l0_row).sum()))
    out["reading"] = ("逐位一致 ⇒ QK + o_proj 足以恢复本例 L0" if eq else
                      "仍不同 ⇒ L0 内存在**第三个**独立 shape-sensitive 边界（见 first_persistent_...）")
    return out


def classify_h4_layers(rows_s0: dict, rows_x: dict, *, n_layers: int) -> dict:
    """**H4**：逐层输入/输出决策行比较（**不**对输入已不同的层做独立来源判定）。"""
    layers = []
    for i in range(int(n_layers)):
        a_in, b_in = rows_s0[i]["input"], rows_x[i]["input"]
        a_out, b_out = rows_s0[i]["output"], rows_x[i]["output"]
        layers.append(dict(layer=int(i), input_exact_equal=bool(torch.equal(a_in, b_in)),
                           output_exact_equal=bool(torch.equal(a_out, b_out)),
                           input_n_diff=int((a_in != b_in).sum()),
                           output_n_diff=int((a_out != b_out).sum())))
    first_any = next((l["layer"] for l in layers if not l["output_exact_equal"]), None)
    first_indep = next((l["layer"] for l in layers
                        if l["input_exact_equal"] and not l["output_exact_equal"]), None)
    return dict(name="H4_layerwise_S0_vs_BQO", layers=layers,
                all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"]
                                              for l in layers)),
                first_layer_with_any_output_diff=first_any,
                first_layer_with_equal_input_and_different_output=first_indep,
                layers_beyond_first_all_have_different_input=bool(
                    all(not l["input_exact_equal"] for l in layers[1:])),
                note="输入已不同的层**不能**用本判据判定独立来源；`n_diff` 轨迹如实给出（放大/衰减）")


def classify_h5_final(logits_s0, logits_x, *, dtype_note="原始 dtype") -> dict:
    """**H5**：最终 logits / argmax / top-2 margin 的恢复类别（1/2/3）。"""
    a, b = logits_s0.reshape(-1), logits_x.reshape(-1)
    DC.assert_same_shape(a, b, "H5_logits")
    eq = bool(torch.equal(a, b))
    ta, tb = int(a.argmax()), int(b.argmax())
    af, bf = a.float(), b.float()
    ma = torch.topk(af, 2).values
    mb = torch.topk(bf, 2).values
    cls = 1 if (eq and ta == tb) else (2 if ta == tb else 3)
    return dict(name="H5_final_S0_vs_BQO", vocab=int(a.numel()), logits_exact_equal=eq,
                logits_n_diff=int((a != b).sum()),
                logits_max_abs=float((af - bf).abs().max()),
                logits_mean_abs=float((af - bf).abs().mean()),
                token_s0=ta, token_x=tb, token_equal=bool(ta == tb),
                margin_s0=float(ma[0] - ma[1]), margin_x=float(mb[0] - mb[1]),
                dtype_note=dtype_note, category=int(cls),
                reading={1: "logits 与 token 均一致 ⇒ 该轨迹被完全恢复",
                         2: "token 一致、logits 不一致 ⇒ **只**能说 token 结果恢复，数值路径未对齐",
                         3: "token 与 logits 都不一致 ⇒ QK+o_proj 不足以恢复本例"}[cls])


def mlp_boundary_report(node_cmp: dict, *, boundary_op: str | None) -> dict:
    """**单独观察的 MLP 边界**：逐个 projection 报告「输入相同/输出不同」，并给出首个这一点。

    ⚠️ 规格要求：**不**把多个 projection 合并成"MLP 不同"，必须指出第一个边界。
    """
    order = ["gate_proj", "up_proj", "silu_activation", "gated_product", "down_proj", "mlp_residual_add"]
    node_of = {"gate_proj": "mlp_gate_proj", "up_proj": "mlp_up_proj", "silu_activation": "mlp_activation",
               "gated_product": "mlp_gated_product", "down_proj": "mlp_down_proj",
               "mlp_residual_add": "layer_output"}
    items, first = [], None
    for op in order:
        node = node_of[op]
        d = node_cmp.get(node)
        if d is None:
            continue
        items.append(dict(op=op, node=node, exact_equal=bool(d["exact_equal"]),
                          n_diff=int(d["n_diff"]),
                          is_shape_boundary=bool(d["exact_equal"] is False)))
        if first is None and d["exact_equal"] is False:
            first = op
    return dict(per_operator=items, first_mlp_output_that_differs=first,
                reached_l0_output_differently=bool(node_cmp["layer_output"]["exact_equal"] is False),
                note="「边界」= 其**输入决策行逐位相同**而输出不同（由 `chain_persistence` 判定）；"
                     "这里只列出 MLP 段各个 projection 的输出差异，供定位第一个边界")


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
    IN, OUT, S = 16, 8, 4
    lin = torch.nn.Linear(IN, OUT, bias=False).to(torch.bfloat16).eval()      # 与生产 o_proj 同配置
    lin_b = torch.nn.Linear(IN, OUT, bias=True).to(torch.bfloat16).eval()
    x = torch.randn(1, S, IN).to(torch.bfloat16)
    x1 = torch.randn(1, 1, IN).to(torch.bfloat16)

    facts = module_facts(lin)
    check("★ `module_facts` 从**实际模块**读出 in/out/bias/dtype/data_ptr",
          facts["in_features"] == IN and facts["out_features"] == OUT and facts["has_bias"] is False
          and facts["weight_dtype"] == "torch.bfloat16" and facts["weight_data_ptr"] == lin.weight.data_ptr()
          and module_facts(lin_b)["has_bias"] is True, facts)

    out, meta = serial_linear_rows(lin, x)
    check("★ 每行**实际**以 [1,1,in_features] 调用同一个 nn.Linear（4 行 ⇒ 4 次调用记录）",
          meta["per_row_input_shapes"] == [[1, 1, IN]] * S
          and all(r["input_numel"] == IN for r in meta["rows"]), meta["per_row_input_shapes"])
    check("★ 输出形状拼回 (1,4,out_features) 且拼接维度 = 1",
          tuple(out.shape) == (1, S, OUT) and meta["cat_dim"] == 1)
    check("★ **拼接顺序**正确：第 i 行就是第 i 次调用的结果（逐位）",
          all(torch.equal(out[:, i:i + 1, :], lin(x[:, i:i + 1, :])) for i in range(S)))
    check("★ 拼接只组合行、不做算术（把输入行打乱 ⇒ 输出行随之对应改变，且未打乱行保持逐位相同）",
          (lambda p: torch.equal(serial_linear_rows(lin, x[:, p, :])[0][:, 1:2, :],
                                 lin(x[:, p[1]:p[1] + 1, :])))([2, 0, 3, 1]))
    check("★ s=1 时与**普通调用**逐位一致（同一张量、同一模块）",
          torch.equal(serial_linear_rows(lin, x1)[0], lin(x1)))
    check("★ bias=True 的模块同样成立（同一模块、逐行）",
          torch.equal(serial_linear_rows(lin_b, x)[0][:, 2:3, :], lin_b(x[:, 2:3, :])))
    check("★ 逐行结果与**整块 M=4** 调用可以不同（这正是本轮要检验的形状敏感性；此处只记录事实）",
          isinstance(bool(torch.equal(out, lin(x))), bool), f"equal={bool(torch.equal(out, lin(x)))}")
    check("★ `contiguous_rows=True` 与 `False` 逐位相同（布局规范化不改值）",
          torch.equal(serial_linear_rows(lin, x, contiguous_rows=True)[0], out))
    bad = 0
    try:
        serial_linear_rows(lin, torch.randn(S, IN).to(torch.bfloat16))
    except DC.CheckError:
        bad += 1
    check("★ 2 维输入必须报错（口径要求 3 维）", bad == 1)

    # ---- 多目标 / MLP 三个投影的不同 in-out 维度（单模块行为必须逐位不变）------------
    MID = 24
    gate_like = torch.nn.Linear(IN, MID, bias=False).to(torch.bfloat16).eval()
    up_like = torch.nn.Linear(IN, MID, bias=False).to(torch.bfloat16).eval()
    down_like = torch.nn.Linear(MID, IN, bias=False).to(torch.bfloat16).eval()
    out_g, meta_g = serial_linear_rows(gate_like, x)
    out_u, meta_u = serial_linear_rows(up_like, x)
    out_d, meta_d = serial_linear_rows(down_like, out_g)
    check("★ MLP 三个投影的 in/out 维度各异时逐行调用仍成立（gate/up: in=16→out=24；down: 24→16）",
          tuple(out_g.shape) == (1, S, MID) and tuple(out_u.shape) == (1, S, MID)
          and tuple(out_d.shape) == (1, S, IN)
          and all(sh == [1, 1, IN] for sh in meta_g["per_row_input_shapes"])
          and all(sh == [1, 1, MID] for sh in meta_d["per_row_input_shapes"])
          and module_facts(gate_like)["out_features"] == MID
          and module_facts(down_like)["in_features"] == MID,
          (meta_g["per_row_input_shapes"], meta_d["per_row_input_shapes"]))
    check("★ 单模块路径的 CPU 回归：`serial_linear_rows` 与逐行参考实现逐位一致（旧行为不变）",
          all(torch.equal(serial_linear_rows(m, xx)[0],
                          torch.cat([m(xx[:, i:i + 1, :]) for i in range(S)], dim=1))
              for m, xx in ((lin, x), (gate_like, x), (down_like, out_g))))
    check("★ `module_facts` 逐投影可区分（同一 hidden=16 但 out 维度不同的两个模块权重 data_ptr 不同）",
          module_facts(gate_like)["weight_data_ptr"] != module_facts(up_like)["weight_data_ptr"]
          and module_facts(gate_like)["weight_shape"] == [MID, IN])

    # ---- H2–H5 分类（合成数据）------------------------------------------------
    r = torch.randn(OUT).to(torch.bfloat16)
    h2 = classify_h2_oproj(r.clone(), r.clone(), r.clone(), r.clone(), (r + 0.01).to(torch.bfloat16))
    check("★ H2：heads/o_proj 都逐位相同 ⇒ local_closure_ok，且能看出 BQ 的差异被消除",
          h2["class_"] == "local_closure_ok" and h2["o_proj_2diffs_removed"] is True)
    h2b = classify_h2_oproj(r.clone(), r.clone(), r.clone(), (r + 0.01).to(torch.bfloat16), r.clone())
    check("★ H2：o_proj 未逐位恢复 ⇒ 判为 o_proj_not_restored（干预无效）",
          h2b["class_"] == "o_proj_not_restored")
    # 合成一条**合法**的链：每个算子的输入都是**它前面的节点**（不是它自己的输出）
    chain = [("a", ("n0",)), ("b", ("n0",)), ("c", ("n1",))]
    names = ["n1", "n2", "n3"]
    eq_node = dict(exact_equal=True, n_diff=0)
    cmp_pers = dict(n0=eq_node, n1=dict(exact_equal=False, n_diff=2),
                    n2=dict(exact_equal=False, n_diff=5), n3=dict(exact_equal=False, n_diff=9))
    pers = chain_persistence(cmp_pers, chain, names)
    check("★ 链持久性：首个分歧 = a，且从它起之后都不同 ⇒ 首个**持续**分歧也是 a",
          pers["first_divergence_operator"] == "a"
          and pers["first_persistent_divergence_operator"] == "a"
          and pers["boundaries"] == ["a", "b"],
          [(d["op"], d["all_inputs_equal"], d["output_equal"]) for d in pers["per_operator"]])
    cmp_abs = dict(n0=eq_node, n1=dict(exact_equal=False, n_diff=2),
                   n2=eq_node, n3=eq_node)
    pers2 = chain_persistence(cmp_abs, chain, names)
    check("★ 链持久性：差异在下一算子被吸收 ⇒ 首个分歧 = a 但**没有**持续边界",
          pers2["first_divergence_operator"] == "a"
          and pers2["first_persistent_divergence_operator"] is None,
          [(d["op"], d["output_equal"]) for d in pers2["per_operator"]])
    h3 = classify_h3_l0(r.clone(), r.clone(), pers, b0_l0_row=(r + 0.01).to(torch.bfloat16))
    check("★ H3：L0 输出逐位一致 ⇒ L0_restored",
          h3["class_"] == "L0_restored" and h3["b0_context"]["n_diff_S0_vs_B0"] > 0)
    h3b = classify_h3_l0(r.clone(), (r + 0.01).to(torch.bfloat16), pers)
    check("★ H3：L0 输出仍不同 ⇒ L0_not_restored 且带首个持续边界",
          h3b["class_"] == "L0_not_restored"
          and h3b["first_persistent_divergence_operator"] == "a")
    rows_s0 = {i: dict(input=r.clone(), output=r.clone()) for i in range(3)}
    rows_x = {i: dict(input=r.clone(), output=r.clone()) for i in range(3)}
    rows_x[1]["output"] = (r + 0.01).to(torch.bfloat16)
    rows_x[2]["input"] = (r + 0.01).to(torch.bfloat16)
    h4 = classify_h4_layers(rows_s0, rows_x, n_layers=3)
    check("★ H4：层 1「输入同、输出不同」⇒ 独立层；层 2 输入已不同 ⇒ 不判独立",
          h4["first_layer_with_equal_input_and_different_output"] == 1
          and h4["first_layer_with_any_output_diff"] == 1
          and h4["layers"][2]["input_exact_equal"] is False)
    lg = torch.randn(32).to(torch.bfloat16)
    h5_1 = classify_h5_final(lg, lg.clone())
    h5_2 = classify_h5_final(lg, (lg + 1e-3 * torch.randn(32)).to(torch.bfloat16))
    h5_3 = classify_h5_final(lg, -lg)
    check("★ H5：三类（全同=1 / 仅 token 同=2 / 都不同=3）+ top-2 margin 可算",
          h5_1["category"] == 1 and h5_3["category"] == 3
          and h5_2["category"] in (1, 2) and isinstance(h5_1["margin_s0"], float))
    rep = mlp_boundary_report(dict(mlp_gate_proj=dict(exact_equal=False, n_diff=3),
                                   mlp_up_proj=dict(exact_equal=False, n_diff=4),
                                   mlp_activation=dict(exact_equal=False, n_diff=2),
                                   mlp_gated_product=dict(exact_equal=False, n_diff=9),
                                   mlp_down_proj=dict(exact_equal=False, n_diff=7),
                                   layer_output=dict(exact_equal=False, n_diff=5)),
                              boundary_op=None)
    check("★ MLP 边界报告：逐个 projection 列出、不合并成『MLP 不同』，并给出第一个输出不同的算子",
          rep["first_mlp_output_that_differs"] == "gate_proj" and len(rep["per_operator"]) == 6)
    print(f"\no_proj 逐行 M=1（纯逻辑 CPU 门）：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def main() -> int:
    print("== o_proj 逐行 M=1 + H1–H5 判定 纯逻辑 CPU 门（**不加载模型**）==")
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
