"""serialalign8209_verdict_cpu.py — QK + o_proj 联合对齐的**只读**汇总（纯 CPU、不加载模型、不改产物）。

## 作用
从 `div8209_sla_*.pt` 的**原始张量**独立复算：
① **H1**：S0/B0 对历史、BQ 对**上一轮 QK-only 干预**、S1 对 S0 的逐位一致；
② **H2**：`heads_merged` / `o_proj` 决策行的局部闭环，以及 BQ 的 2 个差异是否被完全消除；
③ **H3**：L0 输出是否完全恢复；未恢复则独立给出**首个分歧**与**首个持续到出口**的边界；
④ **H4**：28 层输入/输出决策行轨迹；
⑤ **H5**：logits / token / top-2 margin 的恢复类别；
⑥ **MLP 边界**：逐个 projection（gate/up/SiLU/gated/down/residual）独立报告，**不**合并成"MLP 不同"；
⑦ **serial o_proj 自检**：逐行输入形状 `[1,1,in_features]`、拼接顺序与逐位正确、s=1 等于普通调用；
⑧ 20 项控制中**可由原始张量独立复核**的部分 + 与 JSON 摘要逐项核对。

**不**加载模型、**不**做前向、**不**做 FP32、**不**改任何既有产物；摘要错误**不**靠重跑 GPU 修。
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

from specdec.engine import diag_checks as DC        # noqa: E402
from specdec.engine import op_probe as OP           # noqa: E402
from specdec.engine import sb_align as A            # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_sla_20261002T192540Z.json"
DEF_PT = OUT / "div8209_sla_20261002T192540Z.pt"
L0OPS_PT = OUT / "div8209_l0ops_20261002T172447Z.pt"
PREV_PT = OUT / "div8209_sqk_20261002T184729Z.pt"
PREV_JSON = OUT / "div8209_sqk_20261002T184729Z.json"
ARMS = ("S0", "B0", "BQ", "BQO", "S1")


def _cmp(a, b, name) -> dict:
    DC.assert_same_shape(a, b, name)
    d = (a.float() - b.float()).abs()
    neq = (a != b)
    return dict(name=name, shape=[int(x) for x in a.shape], exact_equal=bool(torch.equal(a, b)),
                n_diff=int(neq.sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                coords=[[int(x) for x in c] for c in torch.nonzero(neq).tolist()[:64]],
                coords_total=int(neq.sum()), sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b),
                finite_a=bool(torch.isfinite(a).all()), finite_b=bool(torch.isfinite(b).all()))


def _slice(kind, t, vis, side):
    """`side` ∈ {'S','B'}：S 取末位行、B 取第 0 行（**本文件自带**，不复用工具实现）。"""
    last = (side == "S")
    if kind == "row":
        return t[0, -1] if last else t[0, 0]
    if kind == "heads":
        return t[0, :, -1, :] if last else t[0, :, 0, :]
    if kind == "kv":
        return t[0, :, :vis, :]
    if kind == "scores":
        return t[0, :, 0, :vis]
    raise ValueError(kind)


def _nodes_cmp(a_nodes, b_nodes, *, side_a, side_b, vis) -> dict:
    return {name: _cmp(_slice(kind, a_nodes[name], vis, side_a),
                       _slice(kind, b_nodes[name], vis, side_b), f"{name}_{side_a}vs{side_b}")
            for name, (who, kind) in OP.SPECS.items() if kind != "shape_only"}


def _persistence(node_cmp: dict, chain, out_names: list) -> dict:
    """链上「所有输入逐位相同、输出不同」的算子：首个与**首个持续到末尾**。"""
    per_op = []
    for (op, ins), outn in zip(chain, out_names):
        in_eq = all(node_cmp[i]["exact_equal"] for i in ins if i in node_cmp)
        out_eq = node_cmp[outn]["exact_equal"]
        per_op.append(dict(op=op, node=outn, all_inputs_equal=bool(in_eq), output_equal=bool(out_eq),
                           output_n_diff=int(node_cmp[outn]["n_diff"])))
    bounds = [d["op"] for d in per_op if d["all_inputs_equal"] and not d["output_equal"]]
    first_persistent = None
    for i, d in enumerate(per_op):
        if d["all_inputs_equal"] and not d["output_equal"] and all(
                not per_op[j]["output_equal"] for j in range(i, len(per_op))):
            first_persistent = d["op"]
            break
    return dict(per_operator=per_op, boundaries=bounds,
                first_divergence_operator=(bounds[0] if bounds else None),
                first_persistent_divergence_operator=first_persistent)


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
    rec: dict = dict(tool="serialalign8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始张量独立复算 H1–H5、L0 逐算子链（首个/首个持续）、MLP 逐 projection 边界、"
                               "28 层轨迹、serial o_proj 自检与控制复核；不加载模型、不做前向、不改产物",
                     sources=dict(json=str(jp), pt=str(tp), l0ops_pt=str(L0OPS_PT), prev_pt=str(PREV_PT),
                                  prev_json=str(PREV_JSON)),
                     run_id=j["run_id"], stage=j["stage"], git=j["git"], config=j["config"],
                     oproj_facts=pt.get("oproj_facts"), vis=VIS, n_layers=n_layers,
                     note="全部比较用原始 dtype；仅统计时转 FP32")

    checks: dict = {}
    nodes = {arm: dict(pt[f"nodes_{arm}"]) for arm in ARMS}
    rows = {arm: pt[f"rows_{arm}"] for arm in ARMS}
    finals = {arm: pt[f"final_row_{arm}"] for arm in ARMS}
    n_s0b0 = _nodes_cmp(nodes["S0"], nodes["B0"], side_a="S", side_b="B", vis=VIS)
    n_s0bq = _nodes_cmp(nodes["S0"], nodes["BQ"], side_a="S", side_b="B", vis=VIS)
    n_s0bqo = _nodes_cmp(nodes["S0"], nodes["BQO"], side_a="S", side_b="B", vis=VIS)
    n_bqbqo = _nodes_cmp(nodes["BQ"], nodes["BQO"], side_a="B", side_b="B", vis=VIS)
    cache_pre = dict(exact_equal=True, n_diff=0)
    pers = _persistence(dict(n_s0bqo, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)
    pers_b0 = _persistence(dict(n_s0b0, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)

    # ---------------- ① H1 ----------------
    l0 = torch.load(L0OPS_PT, map_location="cpu") if L0OPS_PT.exists() else {}
    prev = torch.load(PREV_PT, map_location="cpu") if PREV_PT.exists() else {}
    h1: dict = {}
    if l0:
        h1["S0_history"] = dict(
            final=_cmp(finals["S0"], l0["final_row_S"], "S0_final_vs_history"),
            L0_output=_cmp(nodes["S0"]["layer_output"][0, -1], l0["L0_output_row_S"], "S0_L0out_vs_history"),
            qk=_cmp(nodes["S0"]["qk_scores_pre_mask"], l0["raw_S"]["qk_scores_pre_mask"], "S0_qk_vs_history"),
            x0=_cmp(nodes["S0"]["x0"][0, -1], l0["x0_row_S"], "S0_x0_vs_history"))
        h1["B0_history"] = dict(
            final=_cmp(finals["B0"], l0["final_row_B"], "B0_final_vs_history"),
            L0_output=_cmp(nodes["B0"]["layer_output"][0, 0], l0["L0_output_row_B"], "B0_L0out_vs_history"),
            qk=_cmp(nodes["B0"]["qk_scores_pre_mask"], l0["raw_B"]["qk_scores_pre_mask"], "B0_qk_vs_history"),
            x0=_cmp(nodes["B0"]["x0"][0, 0], l0["x0_row_B"], "B0_x0_vs_history"))
    if prev:
        h1["BQ_prev_round"] = dict(
            nodes_all_identical=bool(all(torch.equal(nodes["BQ"][k], prev["nodes_B1"][k])
                                         for k in prev["nodes_B1"] if k in nodes["BQ"])),
            final=_cmp(finals["BQ"], prev["final_row_B1"], "BQ_final_vs_prev"),
            layer_rows_identical=bool(all(
                torch.equal(rows["BQ"][i][kk], prev["rows_B1"][i][kk])
                for i in range(n_layers) for kk in ("input", "output"))),
            L0_output=_cmp(nodes["BQ"]["layer_output"][0, 0], prev["nodes_B1"]["layer_output"][0, 0],
                           "BQ_L0out_vs_prev"),
            o_proj_vs_S0=_cmp(nodes["S0"]["o_proj"][0, -1], nodes["BQ"]["o_proj"][0, 0],
                              "BQ_oproj_vs_S0"))
    h1["S1_vs_S0"] = dict(
        nodes_identical=bool(all(torch.equal(nodes["S1"][k], nodes["S0"][k]) for k in nodes["S0"])),
        final=_cmp(finals["S1"], finals["S0"], "S1_final_vs_S0"),
        layer_rows_identical=bool(all(
            torch.equal(rows["S1"][i][kk], rows["S0"][i][kk])
            for i in range(n_layers) for kk in ("input", "output"))),
        serial_qk_calls=len(pt["serial_qk_S1"] or []), serial_lin_calls=len(pt["serial_lin_S1"] or []))
    checks["H1_reproductions"] = bool(
        (not h1.get("S0_history") or all(v["exact_equal"] for v in h1["S0_history"].values()))
        and (not h1.get("B0_history") or all(v["exact_equal"] for v in h1["B0_history"].values()))
        and (not h1.get("BQ_prev_round") or (h1["BQ_prev_round"]["nodes_all_identical"]
                                            and h1["BQ_prev_round"]["final"]["exact_equal"]
                                            and h1["BQ_prev_round"]["layer_rows_identical"]
                                            and h1["BQ_prev_round"]["L0_output"]["exact_equal"]))
        and h1["S1_vs_S0"]["nodes_identical"] and h1["S1_vs_S0"]["final"]["exact_equal"]
        and h1["S1_vs_S0"]["layer_rows_identical"])
    print("== ① H1：旧结果复现 ==")
    for k in ("S0_history", "B0_history"):
        if k in h1:
            print(f"   {k}: " + "；".join(f"{kk}:exact={v['exact_equal']}(n={v['n_diff']})"
                                          for kk, v in h1[k].items()))
    if "BQ_prev_round" in h1:
        b = h1["BQ_prev_round"]
        print(f"   BQ ≡ 上一轮：节点={b['nodes_all_identical']} 层行={b['layer_rows_identical']} "
              f"final={b['final']['exact_equal']} L0out={b['L0_output']['exact_equal']}；"
              f"BQ 的 o_proj vs S0 n_diff={b['o_proj_vs_S0']['n_diff']}")
    print(f"   S1 ≡ S0：节点={h1['S1_vs_S0']['nodes_identical']} final={h1['S1_vs_S0']['final']['exact_equal']} "
          f"层行={h1['S1_vs_S0']['layer_rows_identical']}（QK 调用={h1['S1_vs_S0']['serial_qk_calls']}、"
          f"o_proj 调用={h1['S1_vs_S0']['serial_lin_calls']}）")

    # ---------------- ② H2 ----------------
    heads = _cmp(nodes["S0"]["heads_merged"][0, -1], nodes["BQO"]["heads_merged"][0, 0], "H2_heads_merged")
    oproj = _cmp(nodes["S0"]["o_proj"][0, -1], nodes["BQO"]["o_proj"][0, 0], "H2_o_proj")
    oproj_bq_vs_bqo = _cmp(nodes["BQ"]["o_proj"][0, 0], nodes["BQO"]["o_proj"][0, 0], "H2_BQ_vs_BQO_o_proj")
    h2 = dict(heads_merged=heads, o_proj=oproj, BQ_vs_BQO=oproj_bq_vs_bqo,
              local_closure_ok=bool(heads["exact_equal"] and oproj["exact_equal"]),
              bq_two_diffs_removed=bool(not oproj_bq_vs_bqo["exact_equal"] and oproj["exact_equal"]),
              class_=("local_closure_ok" if (heads["exact_equal"] and oproj["exact_equal"])
                      else ("heads_differ" if not heads["exact_equal"] else "o_proj_not_restored")))
    checks["H2_matches_json"] = bool(h2["local_closure_ok"] == j["H2"]["o_proj_exact_equal"]
                                     and h2["o_proj"]["n_diff"] == j["H2"]["o_proj_n_diff"])
    print("\n== ② H2：o_proj 局部闭环 ==")
    print(f"   heads_merged 决策行：exact={heads['exact_equal']} n_diff={heads['n_diff']}")
    print(f"   o_proj 决策行：exact={oproj['exact_equal']} n_diff={oproj['n_diff']} "
          f"max|Δ|={oproj['max_abs']} sha16={oproj['sha16_a']}/{oproj['sha16_b']}")
    print(f"   BQ 的 o_proj vs BQO：n_diff={oproj_bq_vs_bqo['n_diff']} "
          f"coords={oproj_bq_vs_bqo['coords']} ⇒ 2 个差异被消除={h2['bq_two_diffs_removed']}"
          f"；class={h2['class_']}")

    # ---------------- ③ H3 ----------------
    l0_s0, l0_b0 = nodes["S0"]["layer_output"][0, -1], nodes["B0"]["layer_output"][0, 0]
    l0_bq, l0_bqo = nodes["BQ"]["layer_output"][0, 0], nodes["BQO"]["layer_output"][0, 0]
    h3 = dict(S0_vs_BQO=_cmp(l0_s0, l0_bqo, "H3_L0out_S0_vs_BQO"),
              S0_vs_BQ=_cmp(l0_s0, l0_bq, "H3_L0out_S0_vs_BQ"),
              S0_vs_B0=_cmp(l0_s0, l0_b0, "H3_L0out_S0_vs_B0"),
              BQ_vs_BQO=_cmp(l0_bq, l0_bqo, "H3_L0out_BQ_vs_BQO"))
    h3["class_"] = "L0_restored" if h3["S0_vs_BQO"]["exact_equal"] else "L0_not_restored"
    h3["persistence"] = pers
    h3["n_diff_trajectory"] = dict(B0=h3["S0_vs_B0"]["n_diff"], BQ=h3["S0_vs_BQ"]["n_diff"],
                                   BQO=h3["S0_vs_BQO"]["n_diff"])
    checks["H3_matches_json"] = bool(h3["S0_vs_BQO"]["n_diff"] == j["H3"]["n_diff"]
                                     and h3["class_"] == j["H3"]["class_"])
    checks["H3_first_persistent_matches_json"] = bool(
        pers["first_persistent_divergence_operator"] == j["H3"]["first_persistent_divergence_operator"]
        and pers["boundaries"] == j["H3"]["boundaries"])
    print("\n== ③ H3：L0 输出 ==")
    print(f"   n_diff 轨迹：B0={h3['S0_vs_B0']['n_diff']} → BQ={h3['S0_vs_BQ']['n_diff']} → "
          f"BQO={h3['S0_vs_BQO']['n_diff']}（max|Δ|={h3['S0_vs_BQO']['max_abs']}）⇒ **{h3['class_']}**")
    print(f"   首个分歧算子 = {pers['first_divergence_operator']}；满足「输入同·输出不同」的全部算子 = "
          f"{pers['boundaries']}")
    print(f"   ★ 首个**持续**分歧算子 = {pers['first_persistent_divergence_operator']}")

    # ---------------- ④ 逐算子表 ----------------
    print("\n== ④ L0 逐算子（S0vsBQO | S0vsBQ | S0vsB0 的 n_diff）==")
    for d, dq, d0 in zip(pers["per_operator"], _persistence(dict(n_s0bq, cache_pre=cache_pre), OP.CHAIN,
                                                            OP.CHAIN_NODE_OUT)["per_operator"],
                         pers_b0["per_operator"]):
        print(f"   {d['op']:<24s} in_eq={str(d['all_inputs_equal']):<5s} out_eq={str(d['output_equal']):<5s} "
              f"n_diff={d['output_n_diff']:>6d} | BQ {dq['output_n_diff']:>6d} | B0 {d0['output_n_diff']:>6d}")

    # ---------------- ⑤ H4 ----------------
    layers = []
    for i in range(n_layers):
        a_in, b_in = rows["S0"][i]["input"], rows["BQO"][i]["input"]
        a_out, b_out = rows["S0"][i]["output"], rows["BQO"][i]["output"]
        layers.append(dict(layer=i, input_exact_equal=bool(torch.equal(a_in, b_in)),
                           output_exact_equal=bool(torch.equal(a_out, b_out)),
                           input_n_diff=int((a_in != b_in).sum()),
                           output_n_diff=int((a_out != b_out).sum())))
    h4 = dict(layers=layers,
              all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"] for l in layers)),
              first_layer_with_any_output_diff=next((l["layer"] for l in layers if not l["output_exact_equal"]), None),
              first_layer_with_equal_input_and_different_output=next(
                  (l["layer"] for l in layers if l["input_exact_equal"] and not l["output_exact_equal"]), None),
              layers_beyond_0_all_have_different_input=bool(all(not l["input_exact_equal"] for l in layers[1:])))
    checks["H4_matches_json"] = bool(h4["first_layer_with_any_output_diff"]
                                     == j["H4"]["first_layer_with_any_output_diff"]
                                     and h4["layers_beyond_0_all_have_different_input"]
                                     == j["H4"]["layers_beyond_first_all_have_different_input"])
    print("\n== ⑤ H4：28 层轨迹（S0 vs BQO）==")
    for l in layers:
        print(f"   L{l['layer']:2d} in_eq={str(l['input_exact_equal']):5s} in_nd={l['input_n_diff']:>6d} "
              f"out_eq={str(l['output_exact_equal']):5s} out_nd={l['output_n_diff']:>6d}")

    # ---------------- ⑥ H5 ----------------
    lg_s0, lg_bqo, lg_b0, lg_bq = (finals["S0"], finals["BQO"], finals["B0"], finals["BQ"])
    h5 = _cmp(lg_s0, lg_bqo, "H5_final_S0_vs_BQO")
    t_s0, t_bqo, t_b0, t_bq = (int(x.argmax()) for x in (lg_s0, lg_bqo, lg_b0, lg_bq))
    af, bf = lg_s0.float(), lg_bqo.float()
    m_s0 = torch.topk(af, 2).values
    m_bqo = torch.topk(bf, 2).values
    h5.update(token_s0=t_s0, token_bqo=t_bqo, token_equal=bool(t_s0 == t_bqo),
              margin_s0=float(m_s0[0] - m_s0[1]), margin_bqo=float(m_bqo[0] - m_bqo[1]),
              category=int(1 if (h5["exact_equal"] and t_s0 == t_bqo) else (2 if t_s0 == t_bqo else 3)),
              reference=dict(n_diff_S0_vs_B0=_cmp(lg_s0, lg_b0, "H5_S0_vs_B0")["n_diff"],
                             n_diff_S0_vs_BQ=_cmp(lg_s0, lg_bq, "H5_S0_vs_BQ")["n_diff"],
                             token_B0=t_b0, token_BQ=t_bq))
    checks["H5_matches_json"] = bool(h5["category"] == j["H5"]["category"]
                                     and h5["n_diff"] == j["H5"]["logits_n_diff"]
                                     and h5["token_bqo"] == j["H5"]["token_x"])
    print("\n== ⑥ H5：最终 logits / token ==")
    print(f"   S0 vs BQO：exact={h5['exact_equal']} n_diff={h5['n_diff']} max|Δ|={h5['max_abs']:.6g} "
          f"mean|Δ|={h5['mean_abs']:.6g}")
    print(f"   token：S0={t_s0}、BQO={t_bqo}（一致={h5['token_equal']}）、B0={t_b0}、BQ={t_bq}；"
          f"margin：S0={h5['margin_s0']:.6g}、BQO={h5['margin_bqo']:.6g} ⇒ **类别 {h5['category']}**")
    print(f"   对照：S0 vs B0 logits n_diff={h5['reference']['n_diff_S0_vs_B0']}、"
          f"S0 vs BQ n_diff={h5['reference']['n_diff_S0_vs_BQ']}")

    # ---------------- ⑦ MLP 边界（逐 projection）----------------
    mlp_order = [("gate_proj", "mlp_gate_proj", ("post_attention_rmsnorm",)),
                 ("up_proj", "mlp_up_proj", ("post_attention_rmsnorm",)),
                 ("silu_activation", "mlp_activation", ("mlp_gate_proj",)),
                 ("gated_product", "mlp_gated_product", ("mlp_activation", "mlp_up_proj")),
                 ("down_proj", "mlp_down_proj", ("mlp_gated_product",)),
                 ("mlp_residual_add", "layer_output", ("attention_residual", "mlp_down_proj"))]
    mlp_items = []
    for op, node, ins in mlp_order:
        in_eq = all(n_s0bqo[i]["exact_equal"] for i in ins)
        mlp_items.append(dict(op=op, node=node, inputs=list(ins), all_inputs_equal=bool(in_eq),
                              output_exact_equal=bool(n_s0bqo[node]["exact_equal"]),
                              output_n_diff=int(n_s0bqo[node]["n_diff"]),
                              is_boundary=bool(in_eq and not n_s0bqo[node]["exact_equal"])))
    h_mlp = dict(per_operator=mlp_items,
                 attention_residual_exact_equal=bool(n_s0bqo["attention_residual"]["exact_equal"]),
                 post_attention_rmsnorm_exact_equal=bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"]),
                 first_boundary=next((m["op"] for m in mlp_items if m["is_boundary"]), None),
                 boundaries=[m["op"] for m in mlp_items if m["is_boundary"]],
                 note="**不**把多个 projection 合并成『MLP 不同』：逐个给出「输入逐位相同、输出不同」的判定")
    checks["MLP_matches_json"] = bool(h_mlp["first_boundary"] == j["MLP_boundary"]["first_mlp_output_that_differs"]
                                      and h_mlp["attention_residual_exact_equal"]
                                      == j["MLP_boundary"]["attention_residual_exact_equal"])
    print("\n== ⑦ MLP 边界（逐 projection，S0 vs BQO）==")
    print(f"   attention_residual 逐位相同={h_mlp['attention_residual_exact_equal']}；"
          f"post_attention_rmsnorm 逐位相同={h_mlp['post_attention_rmsnorm_exact_equal']}")
    for m in mlp_items:
        print(f"   {m['op']:<20s} in_eq={str(m['all_inputs_equal']):<5s} "
              f"out_eq={str(m['output_exact_equal']):<5s} n_diff={m['output_n_diff']:>5d} "
              f"边界={m['is_boundary']}")
    print(f"   首个 MLP 边界 = {h_mlp['first_boundary']}（全部 {h_mlp['boundaries']}）")

    # ---------------- ⑧ serial o_proj 自检 ----------------
    sl_bqo = pt["serial_lin_BQO"]
    sl_s1 = pt["serial_lin_S1"]
    inf = pt["oproj_facts"]["in_features"]
    sl_checks = dict(
        n_calls_bqo=len(sl_bqo), n_calls_s1=len(sl_s1),
        per_row_input_shapes=[c["per_row_input_shapes"] for c in sl_bqo],
        all_rows_1x1_in=bool(all(sh == [1, 1, inf] for c in sl_bqo for sh in c["per_row_input_shapes"])),
        cat_dim=[c["cat_dim"] for c in sl_bqo],
        module_ids=[c["module_id"] for c in sl_bqo],
        same_module_as_oproj=bool(all(c["module_id"] == j["oproj_facts"]["module_id"] for c in sl_bqo)),
        row_input_dtypes=sorted({r["input_layout"]["dtype"] for c in sl_bqo for r in c["rows"]}),
        per_row_input_numel=sorted({r["input_numel"] for c in sl_bqo for r in c["rows"]}),
        # 「拼接只组合行」的可核部分：进入生产路径的张量就是这次 cat 的结果（与探针记录同一对象/同值），
        # 且 **四行都参与**（BQ 与 BQO 的 o_proj 逐行比较可见 4 行里的差异）
        serial_out_matches_probe=bool(torch.equal(sl_bqo[0]["out"], nodes["BQO"]["o_proj"])),
        per_row_bq_vs_bqo_n_diff=[int((nodes["BQ"]["o_proj"][0, i, :] != nodes["BQO"]["o_proj"][0, i, :]).sum())
                                  for i in range(int(nodes["BQO"]["o_proj"].shape[1]))],
        o_proj_row0_equals_S0=bool(torch.equal(nodes["BQO"]["o_proj"][0, 0, :], nodes["S0"]["o_proj"][0, -1, :])))
    checks["serial_oproj_selfcheck"] = bool(
        sl_checks["n_calls_bqo"] == 1 and sl_checks["all_rows_1x1_in"]
        and sl_checks["cat_dim"] == [1] and sl_checks["same_module_as_oproj"]
        and sl_checks["row_input_dtypes"] == ["torch.bfloat16"]
        and sl_checks["per_row_input_numel"] == [inf] and sl_checks["o_proj_row0_equals_S0"]
        and sl_checks["serial_out_matches_probe"])
    print("\n== ⑧ serial o_proj 自检 ==")
    print(f"   逐行输入形状={sl_checks['per_row_input_shapes']}（全部 [1,1,{inf}]={sl_checks['all_rows_1x1_in']}）；"
          f"cat_dim={sl_checks['cat_dim']}")
    print(f"   同一模块实例={sl_checks['same_module_as_oproj']}（module_id={sl_checks['module_ids'][0]}）；"
          f"输入 dtype={sl_checks['row_input_dtypes']}；逐行 numel={sl_checks['per_row_input_numel']}")
    print(f"   BQO 的 o_proj 第 0 行 == S0 决策行：{sl_checks['o_proj_row0_equals_S0']}")

    # ---------------- ⑨ 控制与一致性核对 ----------------
    ctl = dict(
        repeat_consistency=j["repeat_consistency"],
        repeat_all_true=bool(all(v is True for k, v in j["repeat_consistency"].items()
                                 if k.endswith("identical"))),
        finiteness=j["finiteness"],
        c5_qk_row0=j["controls"]["c5_BQO_qk_row0_equals_S0"]["exact_equal"],
        c8_per_row=j["controls"]["c8_serial_oproj_per_row_shape"]["calls"][0]["all_rows_1x1_in"],
        c9_same_module_weight=j["controls"]["c9_serial_oproj_same_module_weight_dtype"],
        c10_only_L0=j["controls"]["c10_intervention_only_L0_oproj"],
        c14_restored=j["controls"]["c14_hooks_and_context_restored"],
        c17_tf32=j["controls"]["c17_allow_tf32_false"],
        c18_draft=j["controls"]["c18_draft_never_loaded"],
        tokens=j["controls"]["token_matches_anchors"],
        final_norm=j["controls"]["final_norm_row"],
        node_json_match=bool(all(
            n_s0bqo[k]["n_diff"] == j["node_compare"]["S0_vs_BQO"][k]["n_diff"]
            and n_s0bqo[k]["sha16_a"] == j["node_compare"]["S0_vs_BQO"][k]["sha16_s"] for k in n_s0bqo)),
    )
    fn = ctl["final_norm"]
    ctl["final_norm_local"] = dict(
        S0_vs_BQO_n_diff=int((pt["norm_row_S0"] != pt["norm_row_BQO"]).sum()),
        S0_vs_B0_n_diff=int((pt["norm_row_S0"] != pt["norm_row_B0"]).sum()),
        json_matches=bool(int((pt["norm_row_S0"] != pt["norm_row_BQO"]).sum()) == fn["S0_vs_BQO"]["n_diff"]))
    checks.update(node_json_match=ctl["node_json_match"], repeat_all_true=ctl["repeat_all_true"],
                  final_norm_matches_json=ctl["final_norm_local"]["json_matches"],
                  finiteness=bool(not j["finiteness"]["nodes_with_nonfinite_slice"]))
    checks["controls_all_true"] = bool(
        ctl["c5_qk_row0"] and ctl["c8_per_row"]
        and ctl["c10_only_L0"]["applied_is_L0_oproj"]
        and ctl["c10_only_L0"]["seen_plus_applied_covers_all"]
        and ctl["c14_restored"]["probe_active"] is False
        and ctl["c14_restored"]["serial_qk_active"] is False
        and ctl["c14_restored"]["serial_linear_active"] is False
        and ctl["c14_restored"]["hooks_removed"]
        and ctl["c17_tf32"]["after"] is False
        and ctl["c18_draft"]["draft_loaded"] is False
        and all(v["match"] for v in ctl["tokens"].values()))
    print("\n== ⑨ 控制与一致性 ==")
    print(f"   重复一致（五臂 × 2 pass）={ctl['repeat_all_true']}；逐节点与 JSON 一致={ctl['node_json_match']}")
    print(f"   控制 5/8/10/14：QK row0={ctl['c5_qk_row0']}、逐行 [1,1,in]={ctl['c8_per_row']}、"
          f"只作用 L0={ctl['c10_only_L0']['applied_is_L0_oproj']}、"
          f"上下文恢复={ctl['c14_restored']}")
    print(f"   最终 RMSNorm：S0 vs BQO n_diff={ctl['final_norm_local']['S0_vs_BQO_n_diff']}、"
          f"S0 vs B0 n_diff={ctl['final_norm_local']['S0_vs_B0_n_diff']}")
    print(f"   有限性：{ctl['finiteness']}")

    rec.update(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, MLP_boundary=h_mlp,
               serial_oproj_selfcheck=sl_checks, node_compare=dict(S0_vs_B0=n_s0b0, S0_vs_BQ=n_s0bq,
                                                                   S0_vs_BQO=n_s0bqo, BQ_vs_BQO=n_bqbqo),
               controls_recomputed=ctl, recompute_checks=checks)

    print("\n== 核对 ==")
    flat = True
    for k, v in checks.items():
        print(f"   [check] {k}: {v}")
        flat = flat and bool(v)
    DC.require(flat, "独立复算与运行 JSON / 历史产物不一致",
               json.dumps({k: v for k, v in checks.items() if not v}, ensure_ascii=False))

    # ---------------- 结论（严格分级）----------------
    if not h2["local_closure_ok"]:
        verdict = "H2 未通过 ⇒ serial o_proj 干预实现无效，停止后续归因"
    elif h3["class_"] == "L0_restored":
        verdict = ("QK + o_proj 联合对齐**足以**恢复本例 L0 输出（逐位一致）"
                   + ("；后续层与 logits 也一致 ⇒ 该轨迹被完全恢复" if h5["category"] == 1
                      else "；但后续层/logits 仍有差异 ⇒ 后续层存在新的独立 shape-sensitive 边界"))
    else:
        verdict = (f"**QK + o_proj 联合对齐把 L0 输出差异从 111 降到 "
                   f"{h3['S0_vs_BQO']['n_diff']}**（路径：B0={h3['n_diff_trajectory']['B0']} → "
                   f"BQ={h3['n_diff_trajectory']['BQ']} → BQO={h3['n_diff_trajectory']['BQO']}），"
                   "attention 子块（QK/AV/head merge/o_proj/残差/post-attn RMSNorm）已**逐位恢复**；"
                   f"但 L0 **未**完全恢复 ⇒ **第三个独立 shape-sensitive 边界 = "
                   f"`{pers['first_persistent_divergence_operator']}`**"
                   f"（全部满足「输入逐位相同、输出不同」的算子：{pers['boundaries']}）；"
                   + ("最终 **token 恢复**（S0=BQO）但 logits 未逐位一致 ⇒ 只能说「token 结果恢复、"
                      "数值路径仍未对齐」" if h5["category"] == 2 else
                      ("logits 与 token 都恢复" if h5["category"] == 1 else "logits 与 token 都未恢复")))
    rec["verdict"] = dict(H2_local_closure=h2["local_closure_ok"], H3_class=h3["class_"],
                          H5_category=h5["category"], token_s0=t_s0, token_bqo=t_bqo,
                          l0_n_diff_trajectory=h3["n_diff_trajectory"],
                          first_persistent_boundary=pers["first_persistent_divergence_operator"],
                          boundaries=pers["boundaries"],
                          mlp_first_boundary=h_mlp["first_boundary"],
                          verdict=verdict, b6_changed=False,
                          not_claimed=["L1–L27 是否为独立边界（这些层输入已不同，判据无法触发）",
                                       "gate_proj/up_proj 内部的具体 kernel/tile/归约顺序（未做 profiler）",
                                       "谁是数学真值（未做 FP32）",
                                       "B6 通过（未跑 B6；token 恢复 ≠ B6 通过）",
                                       "对别的位置/长度/family/层/模型的外推",
                                       "把诊断分支当作生产修复（两者默认关闭、只作用于注册实例）"])
    print("\n== ★ 结论 ==")
    print(f"   {verdict}")
    print("   B6 状态：未改变（本轮未跑 B6/acceptance/pilot/final）")
    p = OUT / f"serialalign8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
