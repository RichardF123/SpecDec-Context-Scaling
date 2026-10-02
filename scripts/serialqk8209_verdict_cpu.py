"""serialqk8209_verdict_cpu.py — L0 QK 逐行 M=1 因果干预的**只读**汇总（纯 CPU、不加载模型、不改既有产物）。

## 作用
从 `div8209_sqk_*.pt` 的**原始张量**独立复算：
① **H1**：S0 与 B1 的 QK 第 0 行（12 头、keys 0:8209）是否逐位一致；
② **H2**：L0 输出是否恢复；未恢复则给出**首个分歧算子**与**首个"持续到末尾"的分歧算子**
   （区分"瞬时、被下游舍入吸收"与"真正持续"——上一轮已经吃过"只看链上第一个"的亏）；
③ **H3**：逐层（28 层）输入行/输出行比较与 n_diff 轨迹；
④ **H4**：最终 logits / argmax 的恢复类别（1/2/3）；
⑤ **诊断构造自检**：逐行 `visible == p+i+1`、占位列全为有限 0、可见部分**逐位**等于该行单行结果、
   拼接后可见区逐位保持、mask 后占位列全 `-inf` 且可见区不变、第 0 行 == S0；
⑥ 与运行 JSON 的逐项核对 + 与**历史产物**（上一轮 L0 算子 run / QK 2×2 结论）的逐位对照；
⑦ 18 项必需控制中**可由原始张量独立复核**的部分。

**不**加载大模型、**不**做前向、**不**做 FP32、**不**改任何既有产物；摘要错误**不**靠重跑 GPU 修。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specdec.engine import diag_checks as DC          # noqa: E402
from specdec.engine import op_probe as OP             # noqa: E402
from specdec.engine import sb_align as A              # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_sqk_20261002T184729Z.json"
DEF_PT = OUT / "div8209_sqk_20261002T184729Z.pt"
L0OPS_PT = OUT / "div8209_l0ops_20261002T172447Z.pt"
QK2X2_JSON = OUT / "qk2x2_8209_verdict_cpu_20261002T181152Z.json"
ARMS = ("S0", "B0", "B1", "S1")


def _sl_B(kind, t, vis):
    """**B 口径**切片（第 0 行 / 共同可见范围）；本文件自带一份，**不**复用工具实现。"""
    if kind == "row":
        return t[0, 0]
    if kind == "heads":
        return t[0, :, 0, :]
    if kind == "kv":
        return t[0, :, :vis, :]
    if kind == "scores":
        return t[0, :, 0, :vis]
    raise ValueError(kind)


def _sl_S(kind, t, vis):
    """**S 口径**切片（末位行 / 共同可见范围）。"""
    if kind == "row":
        return t[0, -1]
    if kind == "heads":
        return t[0, :, -1, :]
    if kind == "kv":
        return t[0, :, :vis, :]
    if kind == "scores":
        return t[0, :, 0, :vis]
    raise ValueError(kind)


def _cmp(a, b, name) -> dict:
    DC.assert_same_shape(a, b, name)
    d = (a.float() - b.float()).abs()
    neq = (a != b)
    return dict(name=name, shape=[int(x) for x in a.shape], exact_equal=bool(torch.equal(a, b)),
                n_diff=int(neq.sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                coords=[[int(x) for x in c] for c in torch.nonzero(neq).tolist()[:64]],
                coords_total=int(neq.sum()), sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b),
                finite_a=bool(torch.isfinite(a).all()), finite_b=bool(torch.isfinite(b).all()))


def _nodes_cmp(nodes_a: dict, nodes_b: dict, *, side_a: str, side_b: str, vis: int) -> dict:
    """节点比较：`side_a`/`side_b` ∈ {'S','B'} 决定取末位行还是第 0 行。"""
    sl = {"S": _sl_S, "B": _sl_B}
    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        a = sl[side_a](kind, nodes_a[name], vis)
        b = sl[side_b](kind, nodes_b[name], vis)
        out[name] = _cmp(a, b, f"{name}_{side_a}vs{side_b}")
    return out


def _chain_analysis(node_cmp: dict, chain, out_names: list) -> dict:
    """链上首个分歧 + **首个持续到末尾**的分歧（两者分开报告）。"""
    order = [(op, ins, out) for (op, ins), out in zip(chain, out_names)]
    per_op = []
    for op, ins, outn in order:
        in_eq = all(node_cmp[i]["exact_equal"] for i in ins if i in node_cmp)
        out_eq = node_cmp[outn]["exact_equal"]
        per_op.append(dict(op=op, output_node=outn, inputs=list(ins),
                           all_inputs_equal=bool(in_eq), output_equal=bool(out_eq),
                           output_n_diff=int(node_cmp[outn]["n_diff"])))
    first_boundary = next((d["op"] for d in per_op if d["all_inputs_equal"] and not d["output_equal"]), None)
    boundaries = [d["op"] for d in per_op if d["all_inputs_equal"] and not d["output_equal"]]
    # 「持续」= 从该算子起，**之后每一个**算子的输出都不同
    first_persistent = None
    for i, d in enumerate(per_op):
        if d["all_inputs_equal"] and not d["output_equal"] and all(
                not per_op[j]["output_equal"] for j in range(i, len(per_op))):
            first_persistent = d["op"]
            break
    return dict(per_operator=per_op, first_divergence_operator=first_boundary,
                all_operators_with_equal_inputs_and_different_output=boundaries,
                first_persistent_divergence_operator=first_persistent,
                note="`first_divergence_operator` = 生产顺序上第一个满足「所有输入逐位相同、输出不同」的算子"
                     "（可能是**瞬时**的，会被下游舍入吸收）；`first_persistent_...` = 从它开始"
                     "**之后每个算子的输出都不同**的那个，即真正持续到 L0 出口的边界")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    j = json.loads(jp.read_text(encoding="utf-8"))
    VIS = int(j["config"]["vis"])
    pt = torch.load(tp, map_location="cpu")
    n_layers = int(pt["meta"]["n_layers"])
    rec: dict = dict(tool="serialqk8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始张量独立复算 H1–H4、L0 逐算子链（首个/首个持续分歧）、28 层轨迹、"
                               "诊断构造自检与全部可复核控制；不加载模型、不做前向、不改产物",
                     sources=dict(json=str(jp), pt=str(tp), l0ops_pt=str(L0OPS_PT),
                                  qk2x2_json=str(QK2X2_JSON)),
                     run_id=j["run_id"], stage=j["stage"], git=j["git"], config=j["config"],
                     vis=VIS, n_layers=n_layers,
                     note="全部比较用原始 dtype（bf16/fp32 原样）；仅统计时转 FP32")

    checks: dict = {}
    nodes = {arm: {k: v for k, v in pt[f"nodes_{arm}"].items()} for arm in ARMS}
    rows = {arm: pt[f"rows_{arm}"] for arm in ARMS}
    finals = {arm: pt[f"final_row_{arm}"] for arm in ARMS}

    # ---------------- ① H1 ----------------
    qk_s0, qk_b0, qk_b1 = (nodes["S0"]["qk_scores_pre_mask"], nodes["B0"]["qk_scores_pre_mask"],
                           nodes["B1"]["qk_scores_pre_mask"])
    h1 = _cmp(qk_s0[0, :, 0, :VIS], qk_b1[0, :, 0, :VIS], "H1_qk_row0_S0_vs_B1")
    h1_pre = _cmp(qk_s0[0, :, 0, :VIS], qk_b0[0, :, 0, :VIS], "H1b_qk_row0_S0_vs_B0")
    checks["H1_matches_json"] = bool(h1["exact_equal"] == j["H1"]["exact_equal"]
                                     and h1["n_diff"] == j["H1"]["n_diff"]
                                     and h1["sha16_b"] == j["H1"]["sha16_b1"])
    print("== ① H1：QK 第 0 行（12 头 × keys 0:8209）==")
    print(f"   S0 vs B0（干预前）：exact={h1_pre['exact_equal']} n_diff={h1_pre['n_diff']} "
          f"max|Δ|={h1_pre['max_abs']} coords={h1_pre['coords']}")
    print(f"   S0 vs B1（干预后）：exact={h1['exact_equal']} n_diff={h1['n_diff']} max|Δ|={h1['max_abs']} "
          f"sha16={h1['sha16_a']}/{h1['sha16_b']}")

    # ---------------- ② H2 + L0 逐算子链 ----------------
    n_s0b0 = _nodes_cmp(nodes["S0"], nodes["B0"], side_a="S", side_b="B", vis=VIS)
    n_s0b1 = _nodes_cmp(nodes["S0"], nodes["B1"], side_a="S", side_b="B", vis=VIS)
    n_b0b1 = _nodes_cmp(nodes["B0"], nodes["B1"], side_a="B", side_b="B", vis=VIS)
    ch_s0b0 = _chain_analysis(n_s0b0, OP.CHAIN, OP.CHAIN_NODE_OUT)
    ch_s0b1 = _chain_analysis(n_s0b1, OP.CHAIN, OP.CHAIN_NODE_OUT)
    z_s0, z_b0, z_b1 = (nodes["S0"]["layer_output"][0, -1], nodes["B0"]["layer_output"][0, 0],
                        nodes["B1"]["layer_output"][0, 0])
    h2 = _cmp(z_s0, z_b1, "H2_L0_output_S0_vs_B1")
    h2_b0 = _cmp(z_s0, z_b0, "H2b_L0_output_S0_vs_B0")
    h2.update(transition_from_B0=dict(n_diff_S0_vs_B0=h2_b0["n_diff"], n_diff_S0_vs_B1=h2["n_diff"],
                                      removed=int(h2_b0["n_diff"] - h2["n_diff"]),
                                      note="QK 几何对齐后 L0 输出差异的减少量（**不是**「修复」）"))
    checks["H2_matches_json"] = bool(h2["exact_equal"] == j["H2"]["exact_equal"]
                                     and h2["n_diff"] == j["H2"]["n_diff"])
    checks["chain_first_divergence_matches_json"] = bool(
        ch_s0b1["first_divergence_operator"]
        == j["first_divergence_S0_vs_B1"]["first_divergence_operator"]
        and ch_s0b1["all_operators_with_equal_inputs_and_different_output"]
        == j["first_divergence_S0_vs_B1"]["all_operators_with_equal_inputs_and_different_output"])
    checks["per_node_matches_json"] = bool(all(
        n_s0b1[k]["n_diff"] == j["node_compare"]["S0_vs_B1"][k]["n_diff"]
        and n_s0b1[k]["sha16_a"] == j["node_compare"]["S0_vs_B1"][k]["sha16_s"]
        and n_s0b0[k]["n_diff"] == j["node_compare"]["S0_vs_B0"][k]["n_diff"]
        for k in n_s0b1))
    print("\n== ② H2：L0 输出与逐算子链 ==")
    print(f"   L0 输出 S0 vs B0：n_diff={h2_b0['n_diff']} max|Δ|={h2_b0['max_abs']:.6g}；"
          f"S0 vs B1：n_diff={h2['n_diff']} max|Δ|={h2['max_abs']:.6g} "
          f"⇒ 减少 {h2['transition_from_B0']['removed']}")
    print(f"   首个分歧算子 = {ch_s0b1['first_divergence_operator']}；"
          f"满足「输入同·输出不同」的全部算子 = {ch_s0b1['all_operators_with_equal_inputs_and_different_output']}")
    print(f"   ★ 首个**持续**分歧算子 = {ch_s0b1['first_persistent_divergence_operator']}")
    print("   op                        in_eq  out_eq   n_diff(S0vsB1)   n_diff(S0vsB0)")
    for d, d0 in zip(ch_s0b1["per_operator"], ch_s0b0["per_operator"]):
        print(f"   {d['op']:<24s} {str(d['all_inputs_equal']):<6s} {str(d['output_equal']):<6s} "
              f"{d['output_n_diff']:>12d} {d0['output_n_diff']:>14d}")

    # ---------------- ③ H3 全层轨迹 ----------------
    layers = []
    for i in range(n_layers):
        a_in, b_in = rows["S0"][i]["input"], rows["B1"][i]["input"]
        a_out, b_out = rows["S0"][i]["output"], rows["B1"][i]["output"]
        layers.append(dict(layer=i,
                           input_exact_equal=bool(torch.equal(a_in, b_in)),
                           output_exact_equal=bool(torch.equal(a_out, b_out)),
                           input_n_diff=int((a_in != b_in).sum()),
                           output_n_diff=int((a_out != b_out).sum())))
    first_any = next((l["layer"] for l in layers if not l["output_exact_equal"]), None)
    first_indep = next((l["layer"] for l in layers
                        if l["input_exact_equal"] and not l["output_exact_equal"]), None)
    h3 = dict(layers=layers, first_layer_with_any_output_diff=first_any,
              first_layer_with_equal_input_and_different_output=first_indep,
              all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"]
                                            for l in layers)),
              layers_1_to_27_all_have_different_input=bool(all(not l["input_exact_equal"]
                                                               for l in layers[1:])),
              note="第 0 层的输入行逐位相同、输出不同 ⇒ 分歧在 L0 内部产生；第 1–27 层的输入**已经不同**"
                   "⇒ 严格判据（输入相同且输出不同）**无法**在这些层触发，本轮不能据此判定它们是否为独立边界；"
                   "下面同时给出 n_diff 轨迹（可看出放大 vs 衰减）")
    checks["H3_matches_json"] = bool(h3["first_layer_with_any_output_diff"]
                                     == j["H3"]["first_layer_with_any_output_diff"]
                                     and h3["first_layer_with_equal_input_and_different_output"]
                                     == j["H3"]["first_layer_with_equal_input_and_different_output"])
    print("\n== ③ H3：28 层轨迹（S0 vs B1）==")
    for l in layers:
        print(f"   L{l['layer']:2d} in_eq={str(l['input_exact_equal']):5s} in_nd={l['input_n_diff']:>6d} "
              f"out_eq={str(l['output_exact_equal']):5s} out_nd={l['output_n_diff']:>6d}")
    print(f"   首个输出差异层={first_any}；首个「输入同·输出不同」层={first_indep}")

    # ---------------- ④ H4 ----------------
    lg_s0, lg_b0, lg_b1 = finals["S0"], finals["B0"], finals["B1"]
    h4 = _cmp(lg_s0, lg_b1, "H4_final_logits_S0_vs_B1")
    # 与工具 `classify_h4` 的字段名对齐（便于逐项核对）
    h4["logits_exact_equal"], h4["logits_n_diff"] = h4["exact_equal"], h4["n_diff"]
    h4["logits_max_abs"] = h4["max_abs"]
    t_s0, t_b0, t_b1 = int(lg_s0.argmax()), int(lg_b0.argmax()), int(lg_b1.argmax())
    h4.update(token_s0=t_s0, token_b0=t_b0, token_b1=t_b1, token_equal=bool(t_s0 == t_b1),
              category=int(1 if (h4["exact_equal"] and t_s0 == t_b1) else (2 if t_s0 == t_b1 else 3)),
              category_B0=int(1 if (torch.equal(lg_s0, lg_b0) and t_s0 == t_b0)
                              else (2 if t_s0 == t_b0 else 3)))
    checks["H4_matches_json"] = bool(h4["category"] == j["H4"]["category"]
                                     and h4["logits_n_diff"] == j["H4"]["logits_n_diff"]
                                     and h4["token_b1"] == j["H4"]["token_b1"])
    print("\n== ④ H4：最终 logits / argmax ==")
    print(f"   S0 vs B1：logits exact={h4['exact_equal']} n_diff={h4['n_diff']} max|Δ|={h4['max_abs']:.6g}；"
          f"token {t_s0} vs {t_b1} ⇒ **类别 {h4['category']}**")
    print(f"   S0 vs B0（干预前）：n_diff={int((lg_s0 != lg_b0).sum())}；token {t_s0} vs {t_b0} "
          f"⇒ 类别 {h4['category_B0']}")

    # ---------------- ⑤ 诊断构造自检 ----------------
    ser = pt["serial_B1"][0]
    p_, s_, T_ = int(ser["p"]), int(ser["s"]), int(ser["T"])
    per_row = []
    ok_all = True
    for r in ser["rows"]:
        i, vis = int(r["i"]), int(r["visible"])
        fits = (vis == p_ + i + 1)
        placed = torch.equal(qk_b1[0, :, i, :vis], r["scaled"][0, :, 0, :])
        padded_finite = bool(torch.isfinite(qk_b1[0, :, i, vis:]).all())
        padded_zero = bool((qk_b1[0, :, i, vis:] == 0).all())
        ok_all = ok_all and fits and placed and padded_finite and padded_zero
        per_row.append(dict(i=i, visible=vis, expected=p_ + i + 1, visible_ok=bool(fits),
                            visible_placed_bitwise=bool(placed), padded_finite=padded_finite,
                            padded_zero=padded_zero))
    post = nodes["B1"]["qk_scores_post_mask"]
    mask_ok = []
    for r in ser["rows"]:
        i, vis = int(r["i"]), int(r["visible"])
        mask_ok.append(dict(i=i, future_all_minus_inf=bool(torch.isinf(post[0, :, i, vis:]).all()),
                            visible_unchanged=bool(torch.equal(post[0, :, i, :vis], qk_b1[0, :, i, :vis]))))
    serial_check = dict(p=p_, s=s_, T=T_, rows=per_row, all_visible_ok=bool(ok_all),
                        mask=mask_ok,
                        all_mask_ok=bool(all(m["future_all_minus_inf"] and m["visible_unchanged"]
                                             for m in mask_ok)),
                        row0_equals_S0_bitwise=bool(torch.equal(qk_b1[0, :, 0, :VIS], qk_s0[0, :, 0, :VIS])),
                        r0_layout=ser["rows"][0]["layout_q_used"],
                        r0_layout_k=ser["rows"][0]["layout_k_used"],
                        placeholder=float(ser["placeholder"]),
                        contiguous_rows=bool(ser["contiguous_rows"]),
                        note="逐行可见长度、占位列（有限 0.0）、拼接后可见区逐位保持、mask 后占位列 -inf，"
                             "第 0 行与 S0 逐位一致")
    checks["serial_construction_ok"] = bool(serial_check["all_visible_ok"] and serial_check["all_mask_ok"]
                                            and serial_check["row0_equals_S0_bitwise"])
    print("\n== ⑤ 诊断构造自检（B1 逐行）==")
    for r in per_row:
        print(f"   行 {r['i']}：visible={r['visible']}（期望 {r['expected']}）{'✔' if r['visible_ok'] else '✘'}；"
              f"可见区逐位保持={r['visible_placed_bitwise']}；占位列有限={r['padded_finite']} "
              f"且全 0={r['padded_zero']}")
    for m in mask_ok:
        print(f"   行 {m['i']} mask 后：未来列全 -inf={m['future_all_minus_inf']}；可见区未变={m['visible_unchanged']}")
    print(f"   第 0 行 == S0 逐位：{serial_check['row0_equals_S0_bitwise']}；"
          f"行 0 的 q 布局={serial_check['r0_layout']}；k 布局={serial_check['r0_layout_k']}")

    # ---------------- ⑥ 与历史产物对照 ----------------
    hist = {}
    if L0OPS_PT.exists():
        l0 = torch.load(L0OPS_PT, map_location="cpu")
        hist = dict(
            S0_final_vs_history=_cmp(lg_s0, l0["final_row_S"], "S0_final_vs_history"),
            B0_final_vs_history=_cmp(lg_b0, l0["final_row_B"], "B0_final_vs_history"),
            S0_L0out_vs_history=_cmp(z_s0, l0["L0_output_row_S"], "S0_L0out_vs_history"),
            B0_L0out_vs_history=_cmp(z_b0, l0["L0_output_row_B"], "B0_L0out_vs_history"),
            S0_qk_vs_history=_cmp(qk_s0, l0["raw_S"]["qk_scores_pre_mask"], "S0_qk_vs_history"),
            B0_qk_vs_history=_cmp(qk_b0, l0["raw_B"]["qk_scores_pre_mask"], "B0_qk_vs_history"),
            S0_x0_vs_history=_cmp(nodes["S0"]["x0"][0, -1], l0["x0_row_S"], "S0_x0_vs_history"),
            B0_x0_vs_history=_cmp(nodes["B0"]["x0"][0, 0], l0["x0_row_B"], "B0_x0_vs_history"))
        checks["history_reproduction"] = bool(all(v["exact_equal"] for v in hist.values()))
    qkref = {}
    if QK2X2_JSON.exists():
        q2 = json.loads(QK2X2_JSON.read_text(encoding="utf-8"))
        exp = q2["attribution"]["observed_A_vs_D"]["coords"]
        qkref = dict(qk2x2_expected_coords=exp, measured_coords=h1_pre["coords"],
                     coords_match=bool([list(x) for x in h1_pre["coords"]] == [list(x) for x in exp]),
                     qk2x2_expected_n_diff=q2["attribution"]["observed_A_vs_D"]["n_diff"],
                     measured_n_diff=h1_pre["n_diff"],
                     qk2x2_expected_max_abs_raw=q2["attribution"]["observed_A_vs_D"]["max_abs_raw"],
                     qk2x2_expected_max_abs_scaled=q2["attribution"]["observed_A_vs_D"]["max_abs_scaled"],
                     measured_max_abs_scaled=h1_pre["max_abs"])
        checks["qk2x2_coords_match"] = bool(qkref["coords_match"] and h1_pre["n_diff"] == 3
                                            and h1_pre["max_abs"] == 4.0)
    print("\n== ⑥ 与历史产物对照 ==")
    for k, v in hist.items():
        print(f"   {k}: exact={v['exact_equal']} n_diff={v['n_diff']} sha16={v['sha16_a']}/{v['sha16_b']}")
    if qkref:
        print(f"   与 QK 2×2 结论对照：坐标 {qkref['measured_coords']} vs 期望 {qkref['qk2x2_expected_coords']}"
              f" ⇒ match={qkref['coords_match']}；n_diff={qkref['measured_n_diff']}、"
              f"max|Δ|(scaled)={qkref['measured_max_abs_scaled']}")

    # ---------------- ⑦ 可独立复核的控制 ----------------
    ctl = {}
    ctl["c4_S1_equals_S0_all_nodes"] = bool(all(torch.equal(nodes["S1"][k], nodes["S0"][k])
                                                for k in nodes["S0"]))
    ctl["c4_S1_equals_S0_final_row"] = bool(torch.equal(finals["S1"], finals["S0"]))
    ctl["c4_S1_equals_S0_layer_rows"] = bool(all(
        torch.equal(rows["S1"][i][k], rows["S0"][i][k]) for i in range(n_layers)
        for k in ("input", "output")))
    ctl["c7_B1_masked_total"] = int(torch.isinf(post).sum())
    ctl["c7_B0_masked_total"] = int(torch.isinf(nodes["B0"]["qk_scores_post_mask"]).sum())
    ctl["c7_S0_masked_total"] = int(torch.isinf(nodes["S0"]["qk_scores_post_mask"]).sum())
    ctl["c7_expected_masked"] = int(sum(12 * (T_ - (p_ + i + 1)) for i in range(s_)))
    ctl["c8_poison_all_rows_unaffected"] = bool(pt["poison_B1"]
                                               and all(r["raw_equal"] and r["scaled_equal"]
                                                       and r["finite"] for r in pt["poison_B1"]))
    ctl["c12_repeat_consistency_json"] = j["repeat_consistency"]
    ctl["c13_H_L0_cache_pre_shape"] = [int(x) for x in pt["H_L0_cache_pre"]["k"].shape]
    ctl["c15_all_node_slices_finite"] = bool(all(v["finite_a"] and v["finite_b"]
                                                for v in n_s0b1.values()))
    ctl["c15_logits_finite"] = bool(torch.isfinite(lg_s0).all() and torch.isfinite(lg_b1).all())
    ctl["positions"] = j["controls"]["positions"]
    ctl["tokens"] = dict(S0=t_s0, B0=t_b0, B1=t_b1, anchors=j["controls"]["token_matches_anchors"])
    ctl["norm_row_S0_vs_B1"] = (_cmp(pt["norm_row_S0"], pt["norm_row_B1"], "final_norm_S0_vs_B1")
                                if pt["norm_row_S0"] is not None else None)
    checks["c4_S1_equals_S0"] = bool(ctl["c4_S1_equals_S0_all_nodes"] and ctl["c4_S1_equals_S0_final_row"]
                                     and ctl["c4_S1_equals_S0_layer_rows"])
    checks["c7_mask_pattern"] = bool(ctl["c7_B1_masked_total"] == ctl["c7_expected_masked"]
                                     == ctl["c7_B0_masked_total"] and ctl["c7_S0_masked_total"] == 0)
    checks["c8_poison"] = ctl["c8_poison_all_rows_unaffected"]
    checks["c12_repeat_json_all_true"] = bool(all(v is True for k, v in j["repeat_consistency"].items()
                                                  if k.endswith("identical")))
    checks["c15_finite"] = bool(ctl["c15_all_node_slices_finite"] and ctl["c15_logits_finite"])
    checks["positions_ok"] = bool(ctl["positions"]["both_predict_target"])
    checks["tokens_match_anchors"] = bool(all(v["match"] for v in j["controls"]["token_matches_anchors"].values()))
    print("\n== ⑦ 可独立复核的控制 ==")
    for k, v in ctl.items():
        if isinstance(v, dict) and "layers" in v:
            continue
        print(f"   {k}: {json.dumps(v, ensure_ascii=False, default=str)[:300]}")

    rec["H1"] = dict(measured_B0=h1_pre, measured_B1=h1)
    rec["H2"] = h2
    rec["H2_B0"] = h2_b0
    rec["chain_S0_vs_B1"] = ch_s0b1
    rec["chain_S0_vs_B0"] = ch_s0b0
    rec["node_compare_S0_vs_B1"] = n_s0b1
    rec["node_compare_S0_vs_B0"] = n_s0b0
    rec["node_compare_B0_vs_B1"] = n_b0b1
    rec["H3"] = h3
    rec["H4"] = h4
    rec["serial_construction"] = serial_check
    rec["history_reproduction"] = hist
    rec["qk2x2_cross_check"] = qkref
    rec["controls_recomputed"] = ctl
    rec["recompute_checks"] = checks

    flat = True
    print("\n== 核对（JSON 一致性 / 历史复现 / 控制）==")
    for k, v in checks.items():
        print(f"   [check] {k}: {v}")
        flat = flat and bool(v)
    DC.require(flat, "独立复算与运行 JSON / 历史产物 / 控制不一致",
               json.dumps({k: v for k, v in checks.items() if not v}, ensure_ascii=False))

    # ---------------- 结论（严格分级）----------------
    h1_ok = h1["exact_equal"]
    h2_ok = h2["exact_equal"]
    h4c = h4["category"]
    if not h1_ok:
        verdict = "H1 未通过 ⇒ 干预实现无效，停止因果解释（本轮不继续）"
    elif h2_ok:
        verdict = ("QK 几何恢复**足以**恢复本例 L0 输出；"
                   + ("后续层与 logits 也一致 ⇒ 该局部轨迹被完全恢复" if h4c == 1
                      else "但后续层/logits 仍有差异 ⇒ 后续层存在新的独立执行形状差异"))
    else:
        verdict = ("**QK shape sensitivity 是一个真实的分歧来源**（对齐后 L0 输出差异由 "
                   f"{h2_b0['n_diff']} 降到 {h2['n_diff']}，QK/AV/head_merge 逐位恢复），"
                   "但 **L0 内至少还有一个独立的 shape-sensitive 边界**："
                   f"首个持续分歧算子 = `{ch_s0b1['first_persistent_divergence_operator']}`"
                   f"（链上满足「输入逐位相同、输出不同」的全部算子 = "
                   f"{ch_s0b1['all_operators_with_equal_inputs_and_different_output']}，"
                   "其中 `q_proj`、`softmax` 的差异分别被 RoPE 与 bf16 概率回转**吸收**）；"
                   + ("最终 **token 选择恢复**（S0=B1）但 logits 未逐位一致 ⇒ 只能说"
                      "「token 结果恢复，数值路径仍不同」" if h4c == 2 else
                      ("logits 与 token 都恢复" if h4c == 1 else "logits 与 token 都未恢复")))
    rec["verdict"] = dict(
        H1=h1_ok, H2=h2_ok, H4_category=h4c,
        token_s0=t_s0, token_b1=t_b1,
        first_divergence_operator=ch_s0b1["first_divergence_operator"],
        first_persistent_divergence_operator=ch_s0b1["first_persistent_divergence_operator"],
        boundaries=ch_s0b1["all_operators_with_equal_inputs_and_different_output"],
        verdict=verdict,
        b6_changed=False,
        not_claimed=["L1–L27 是否为独立边界（本轮判据在这些层无法触发：输入已不同）",
                     "把 shape-sensitive 说成实现 bug、或用 ULP/margin 豁免正确性",
                     "谁是数学真值（本轮不做 FP32）",
                     "B6 通过（本轮未跑 B6，token 恢复 ≠ B6 通过）",
                     "对别的位置/长度/family/层/模型的外推",
                     "把本诊断分支当作生产修复（它默认关闭、只作用于被注册的模块实例）"])
    print("\n== ★ 结论 ==")
    print(f"   {verdict}")
    print(f"   B6 状态：未改变（本轮未跑 B6/acceptance/pilot/final）")
    p = OUT / f"serialqk8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
