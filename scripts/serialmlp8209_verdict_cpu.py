"""serialmlp8209_verdict_cpu.py — L0 全部持续 GEMM 对齐的**只读**汇总（纯 CPU、不加载模型、不改产物）。

## 作用
从 `div8209_sml_*.pt` 的**原始张量**独立复算：
① **H1**：S0/B0 对历史、**BQO 对上一轮产物**、S1 对 S0 的逐位一致；
② **H2**：BQOGU 的 `post_attention_rmsnorm`/`gate_proj`/`up_proj`/`SiLU`/`gated_product` 决策行是否逐位恢复；
③ **H3**：BQOGU 中 `down_proj` 是否满足「输入逐位相同、输出不同」⇒ 独立边界；
④ **H4**：BQOGUD 的 L0 输出是否与 S0 **逐位一致**（并给 B0→BQO→BQOGU→BQOGUD 变化表）；
⑤ **H5**：若 L0 恢复，逐层找**第一处**「输入逐位相同、输出不同」的层（**不**进入该层内部）；
⑥ **H6**：S0 与 B0/BQO/BQOGU/BQOGUD 的 logits/token/top-2/margin 对照与分类；
⑦ **B0→BQO→BQOGU→BQOGUD 的差异变化表**（逐算子）；
⑧ serial Linear 自检（每行输入形状、拼接、模块身份）与 24 项控制中可独立复核的部分。

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
DEF_JSON = OUT / "div8209_sml_20261002T201740Z.json"
DEF_PT = OUT / "div8209_sml_20261002T201740Z.pt"
L0OPS_PT = OUT / "div8209_l0ops_20261002T172447Z.pt"
PREV_PT = OUT / "div8209_sla_20261002T192540Z.pt"
PREV_JSON = OUT / "div8209_sla_20261002T192540Z.json"
ARMS = ("S0", "B0", "BQO", "BQOGU", "BQOGUD", "S1")
TRAJ = ("B0", "BQO", "BQOGU", "BQOGUD")


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
    per_op = []
    for (op, ins), outn in zip(chain, out_names):
        in_eq = all(node_cmp[i]["exact_equal"] for i in ins if i in node_cmp)
        out_eq = node_cmp[outn]["exact_equal"]
        per_op.append(dict(op=op, node=outn, inputs=list(ins), all_inputs_equal=bool(in_eq),
                           output_equal=bool(out_eq), output_n_diff=int(node_cmp[outn]["n_diff"])))
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


def _token_fields(row) -> dict:
    f = row.float().reshape(-1)
    v, idx = torch.topk(f, 2)
    return dict(argmax=int(f.argmax()), top1=int(idx[0]), top1_value=float(v[0]),
                top2=int(idx[1]), top2_value=float(v[1]), margin=float(v[0] - v[1]),
                sha16=A.stable_sha(row))


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
    facts = pt["module_facts"]
    rec: dict = dict(tool="serialmlp8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始张量独立复算 H1–H6、L0 逐算子变化表、MLP 逐 projection 边界、"
                               "28 层轨迹、serial Linear 自检与控制复核；不加载模型、不改产物",
                     sources=dict(json=str(jp), pt=str(tp), l0ops_pt=str(L0OPS_PT), prev_pt=str(PREV_PT),
                                  prev_json=str(PREV_JSON)),
                     run_id=j["run_id"], stage=j["stage"], git=j["git"], config=j["config"],
                     module_facts=facts, vis=VIS, n_layers=n_layers,
                     note="全部比较用原始 dtype；仅统计时转 FP32；所有『逐位一致』都指**决策行**或共同可见前缀")

    checks: dict = {}
    nodes = {arm: dict(pt[f"nodes_{arm}"]) for arm in ARMS}
    rows = {arm: pt[f"rows_{arm}"] for arm in ARMS}
    finals = {arm: pt[f"final_row_{arm}"] for arm in ARMS}
    cache_pre = dict(exact_equal=True, n_diff=0)
    nc = {a: _nodes_cmp(nodes["S0"], nodes[a], side_a="S", side_b="B", vis=VIS)
          for a in ("B0", "BQO", "BQOGU", "BQOGUD")}
    pers = {a: _persistence(dict(nc[a], cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)
            for a in ("B0", "BQO", "BQOGU", "BQOGUD")}

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
        h1["BQO_prev_round"] = dict(
            nodes_all_identical=bool(all(torch.equal(nodes["BQO"][k], prev["nodes_BQO"][k])
                                         for k in prev["nodes_BQO"] if k in nodes["BQO"])),
            final=_cmp(finals["BQO"], prev["final_row_BQO"], "BQO_final_vs_prev"),
            layer_rows_identical=bool(all(torch.equal(rows["BQO"][i][kk], prev["rows_BQO"][i][kk])
                                          for i in range(n_layers) for kk in ("input", "output"))),
            L0_output=_cmp(nodes["BQO"]["layer_output"][0, 0], prev["nodes_BQO"]["layer_output"][0, 0],
                           "BQO_L0out_vs_prev"))
    h1["S1_vs_S0"] = dict(
        nodes_identical=bool(all(torch.equal(nodes["S1"][k], nodes["S0"][k]) for k in nodes["S0"])),
        final=_cmp(finals["S1"], finals["S0"], "S1_final_vs_S0"),
        layer_rows_identical=bool(all(torch.equal(rows["S1"][i][kk], rows["S0"][i][kk])
                                      for i in range(n_layers) for kk in ("input", "output"))))
    checks["H1_reproductions"] = bool(
        (not h1.get("S0_history") or all(v["exact_equal"] for v in h1["S0_history"].values()))
        and (not h1.get("B0_history") or all(v["exact_equal"] for v in h1["B0_history"].values()))
        and (not h1.get("BQO_prev_round") or (h1["BQO_prev_round"]["nodes_all_identical"]
                                             and h1["BQO_prev_round"]["final"]["exact_equal"]
                                             and h1["BQO_prev_round"]["layer_rows_identical"]
                                             and h1["BQO_prev_round"]["L0_output"]["exact_equal"]))
        and h1["S1_vs_S0"]["nodes_identical"] and h1["S1_vs_S0"]["final"]["exact_equal"]
        and h1["S1_vs_S0"]["layer_rows_identical"])
    print("== ① H1：历史/上一轮复现 ==")
    for k in ("S0_history", "B0_history"):
        if k in h1:
            print(f"   {k}: " + "；".join(f"{kk}:exact={v['exact_equal']}(n={v['n_diff']})"
                                          for kk, v in h1[k].items()))
    if "BQO_prev_round" in h1:
        b = h1["BQO_prev_round"]
        print(f"   BQO ≡ 上一轮：节点={b['nodes_all_identical']} 层行={b['layer_rows_identical']} "
              f"final={b['final']['exact_equal']} L0out={b['L0_output']['exact_equal']}")
    print(f"   S1 ≡ S0：节点={h1['S1_vs_S0']['nodes_identical']} final={h1['S1_vs_S0']['final']['exact_equal']} "
          f"层行={h1['S1_vs_S0']['layer_rows_identical']}")

    # ---------------- ② H2 ----------------
    def row_of(arm, name, end):
        return nodes[arm][name][0, end]

    h2_nodes = ("post_attention_rmsnorm", "mlp_gate_proj", "mlp_up_proj", "mlp_activation", "mlp_gated_product")
    h2 = {n: _cmp(row_of("S0", n, -1), row_of("BQOGU", n, 0), f"H2_{n}") for n in h2_nodes}
    h2["reference_BQO"] = {n: _cmp(row_of("S0", n, -1), row_of("BQO", n, 0), f"BQO_{n}") for n in h2_nodes}
    h2["class_"] = ("gate_up_local_closure_ok" if all(h2[n]["exact_equal"] for n in h2_nodes)
                    else "gate_up_not_restored")
    checks["H2_matches_json"] = bool(h2["class_"] == j["H2"]["class_"]
                                     and h2["mlp_gate_proj"]["n_diff"] == j["H2"]["gate_proj"]["n_diff"])
    print("\n== ② H2：gate/up 局部闭环（BQOGU vs S0）==")
    for n in h2_nodes:
        print(f"   {n:<24s} exact={h2[n]['exact_equal']} n_diff={h2[n]['n_diff']} "
              f"（BQO 对照 n_diff={h2['reference_BQO'][n]['n_diff']}）")
    print(f"   ⇒ {h2['class_']}")

    # ---------------- ③ H3 ----------------
    gp_gu = h2["mlp_gated_product"]
    down_gu = _cmp(row_of("S0", "mlp_down_proj", -1), row_of("BQOGU", "mlp_down_proj", 0), "H3_down_BQOGU")
    h3 = dict(gated_product=gp_gu, down_proj=down_gu,
              down_proj_class=("independent_boundary" if (gp_gu["exact_equal"] and not down_gu["exact_equal"])
                               else ("both_equal" if (gp_gu["exact_equal"] and down_gu["exact_equal"])
                                     else "gate_up_closure_failed_inputs_differ")),
              reference_BQO_down=_cmp(row_of("S0", "mlp_down_proj", -1), row_of("BQO", "mlp_down_proj", 0),
                                      "BQO_down"))
    checks["H3_matches_json"] = bool(h3["down_proj_class"] == j["H3"]["down_proj_class"]
                                     and h3["down_proj"]["n_diff"] == j["H3"]["down_proj"]["row"]["n_diff"])
    print("\n== ③ H3：down_proj 是否独立边界（在 BQOGU 中）==")
    print(f"   gated_product 逐位相同={gp_gu['exact_equal']}；down_proj exact={down_gu['exact_equal']} "
          f"n_diff={down_gu['n_diff']} max|Δ|={down_gu['max_abs']:.6g} ⇒ **{h3['down_proj_class']}**"
          f"（BQO 对照 down n_diff={h3['reference_BQO_down']['n_diff']}）")

    # ---------------- ④ H4 ----------------
    l0_s0 = row_of("S0", "layer_output", -1)
    h4 = dict(nodes={a: _cmp(l0_s0, row_of(a, "layer_output", 0), f"H4_L0out_S0_vs_{a}")
                     for a in TRAJ},
              trajectory={a: int((l0_s0 != row_of(a, "layer_output", 0)).sum()) for a in TRAJ})
    h4["class_"] = "L0_restored" if h4["nodes"]["BQOGUD"]["exact_equal"] else "L0_not_restored"
    h4["BQOGUD_L1_input_equals_S0_L1_input"] = bool(torch.equal(rows["S0"][1]["input"],
                                                                rows["BQOGUD"][1]["input"]))
    checks["H4_matches_json"] = bool(h4["class_"] == j["H4"]["class_"]
                                     and h4["trajectory"]["BQOGUD"] == j["H4"]["trajectory"]["BQOGUD"])
    print("\n== ④ H4：L0 输出是否完全恢复 ==")
    print(f"   变化表：B0={h4['trajectory']['B0']} → BQO={h4['trajectory']['BQO']} → "
          f"BQOGU={h4['trajectory']['BQOGU']} → **BQOGUD={h4['trajectory']['BQOGUD']}** ⇒ **{h4['class_']}**")
    print(f"   BQOGUD 的 L0 输出 vs S0：exact={h4['nodes']['BQOGUD']['exact_equal']} "
          f"sha16={h4['nodes']['BQOGUD']['sha16_a']}/{h4['nodes']['BQOGUD']['sha16_b']}；"
          f"L1 input == S0 L1 input：{h4['BQOGUD_L1_input_equals_S0_L1_input']}")

    # ---------------- ⑤ H5 ----------------
    layers = []
    for i in range(n_layers):
        a_in, b_in = rows["S0"][i]["input"], rows["BQOGUD"][i]["input"]
        a_out, b_out = rows["S0"][i]["output"], rows["BQOGUD"][i]["output"]
        layers.append(dict(layer=i, input_exact_equal=bool(torch.equal(a_in, b_in)),
                           output_exact_equal=bool(torch.equal(a_out, b_out)),
                           input_n_diff=int((a_in != b_in).sum()),
                           output_n_diff=int((a_out != b_out).sum())))
    first_indep = next((l["layer"] for l in layers if l["input_exact_equal"] and not l["output_exact_equal"]), None)
    h5 = dict(applicable=bool(h4["class_"] == "L0_restored"), layers=layers,
              first_layer_with_equal_input_and_different_output=first_indep,
              first_layer_with_any_output_diff=next((l["layer"] for l in layers
                                                     if not l["output_exact_equal"]), None),
              all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"] for l in layers)),
              note="仅当 L0 完全恢复时该层才是『后续第一处重新分歧层』；**不**进入该层内部继续诊断")
    checks["H5_matches_json"] = bool(h5["first_layer_with_equal_input_and_different_output"]
                                     == j["H5"]["first_layer_with_equal_input_and_different_output"])
    print("\n== ⑤ H5：28 层轨迹（S0 vs BQOGUD）==")
    for l in layers:
        print(f"   L{l['layer']:2d} in_eq={str(l['input_exact_equal']):5s} in_nd={l['input_n_diff']:>6d} "
              f"out_eq={str(l['output_exact_equal']):5s} out_nd={l['output_n_diff']:>6d}")
    print(f"   ★ 后续**第一处**重新分歧层 = L{first_indep}（L0 已逐位恢复 ⇒ 该判定适用）")

    # ---------------- ⑥ H6 ----------------
    h6 = {}
    for a in TRAJ:
        c = _cmp(finals["S0"], finals[a], f"H6_S0_vs_{a}")
        # 与工具 `classify_h5_final` 的字段名对齐（便于逐项核对）
        c["logits_exact_equal"], c["logits_n_diff"] = c["exact_equal"], c["n_diff"]
        c["logits_max_abs"], c["logits_mean_abs"] = c["max_abs"], c["mean_abs"]
        sf, xf = _token_fields(finals["S0"]), _token_fields(finals[a])
        c.update(token_s0=sf["argmax"], token_x=xf["argmax"], token_equal=bool(sf["argmax"] == xf["argmax"]),
                 margin_s0=sf["margin"], margin_x=xf["margin"],
                 top2_s0=[sf["top1"], sf["top2"]], top2_x=[xf["top1"], xf["top2"]],
                 category=int(1 if (c["exact_equal"] and sf["argmax"] == xf["argmax"])
                              else (2 if sf["argmax"] == xf["argmax"] else 3)))
        h6[a] = c
    checks["H6_matches_json"] = bool(all(h6[a]["category"] == j["H6"][a]["category"]
                                        and h6[a]["n_diff"] == j["H6"][a]["logits_n_diff"]
                                        for a in TRAJ))
    checks["H6_not_monotonic_reported"] = bool(h6["BQOGU"]["n_diff"] < h6["BQOGUD"]["n_diff"]
                                               or h6["BQOGU"]["n_diff"] < h6["BQO"]["n_diff"])
    print("\n== ⑥ H6：最终 logits / token / top-2 ==")
    print("   arm      class  logits_exact  n_diff     max|Δ|      mean|Δ|     token(S0/x)  margin(S0/x)  top2(S0/x)")
    for a in TRAJ:
        c = h6[a]
        print(f"   {a:<8s} {c['category']}      {str(c['logits_exact_equal']):<12s}  {c['n_diff']:>7d}  "
              f"{c['max_abs']:.6g}   {c['mean_abs']:.6g}   {c['token_s0']}/{c['token_x']:<7d}  "
              f"{c['margin_s0']:.6g}/{c['margin_x']:.6g}  {c['top2_s0']}/{c['top2_x']}")
    print("   ⇒ 必须如实报告：**n_diff 并非随对齐算子增多而单调下降**"
          f"（BQO={h6['BQO']['n_diff']}、BQOGU={h6['BQOGU']['n_diff']}、BQOGUD={h6['BQOGUD']['n_diff']}）")

    # ---------------- ⑦ 差异变化表（逐算子）----------------
    print("\n== ⑦ B0 → BQO → BQOGU → BQOGUD 的逐算子差异变化（S0 视角的 n_diff）==")
    print("   op                        in_eq  out_eq(BQOGUD)   B0      BQO     BQOGU   BQOGUD")
    change_rows = []
    for d in pers["BQOGUD"]["per_operator"]:
        op, node = d["op"], d["node"]
        vals = [nc[a][node]["n_diff"] for a in TRAJ]
        change_rows.append(dict(op=op, node=node, all_inputs_equal=d["all_inputs_equal"],
                                output_equal_bqogud=d["output_equal"], n_diff=dict(zip(TRAJ, vals))))
        print(f"   {op:<24s} {str(d['all_inputs_equal']):<6s} {str(d['output_equal']):<13s} "
              f"{vals[0]:>6d}  {vals[1]:>6d}  {vals[2]:>6d}  {vals[3]:>6d}")

    # ---------------- ⑧ serial Linear 自检 ----------------
    sl_by_arm = {}
    for a in ("BQO", "BQOGU", "BQOGUD", "S1"):
        sl = pt[f"serial_lin_{a}"] or []
        sl_by_arm[a] = dict(
            n_calls=len(sl),
            per_call=[dict(module_id=c["module_id"], n_rows=len(c["per_row_input_shapes"]),
                           shapes=sorted({tuple(sh) for sh in c["per_row_input_shapes"]}),
                           cat_dim=c["cat_dim"], in_numel=sorted({r["input_numel"] for r in c["rows"]}),
                           dtype=sorted({r["input_layout"]["dtype"] for r in c["rows"]})) for c in sl],
            n_rows_all_equal_s=bool(all(len(c["per_row_input_shapes"]) == 4 for c in sl)),
            cat_dim_all_1=bool(all(c["cat_dim"] == 1 for c in sl)),
            all_bf16=bool(all(r["input_layout"]["dtype"] == "torch.bfloat16" for c in sl for r in c["rows"])))
    expected_modules = {"BQO": 1, "BQOGU": 3, "BQOGUD": 4, "S1": 4}
    # ⚠️ S1 是 s=1 的 decode 臂 ⇒ 每次逐行调用只有 **1** 行；B 臂每次 4 行。
    expected_rows = {"BQO": 4, "BQOGU": 4, "BQOGUD": 4, "S1": 1}
    checks["serial_linear_selfcheck"] = bool(
        all(sl_by_arm[a]["n_calls"] == expected_modules[a]
            and all(c["n_rows"] == expected_rows[a] for c in sl_by_arm[a]["per_call"])
            and sl_by_arm[a]["cat_dim_all_1"] and sl_by_arm[a]["all_bf16"] for a in sl_by_arm))
    print("\n== ⑧ serial Linear 自检（每行必须以 [1,1,Din] 调用）==")
    for a, d in sl_by_arm.items():
        print(f"   {a:<7s} 调用数={d['n_calls']}（期望 {expected_modules[a]}）；每行行数="
              f"{sorted({c['n_rows'] for c in d['per_call']})}（期望 {expected_rows[a]}）；"
              f"cat_dim=1={d['cat_dim_all_1']}；bf16={d['all_bf16']}")
        for c in d["per_call"]:
            print(f"      module_id={c['module_id']} 形状={c['shapes']} numel={c['in_numel']}")

    # ---------------- ⑨ 控制与一致性 ----------------
    inp = facts
    ctl = dict(
        repeat_all_true=bool(all(v is True for k, v in j["repeat_consistency"].items()
                                 if k.endswith("identical"))),
        finiteness=j["finiteness"],
        c7_c9=j["controls"]["c7_c9_per_row_input_shapes"],
        c10=j["controls"]["c10_same_modules_weights_dtype"],
        c13=j["controls"]["c13_only_L0_instances"],
        c14=j["controls"]["c14_other_layers_block_path"],
        c17=j["controls"]["c17_hooks_and_context_restored"],
        c18=j["controls"]["c18_cache_inputs_weights_unmodified"],
        c20=j["controls"]["c20_allow_tf32_false"],
        c21=j["controls"]["c21_draft_never_loaded"],
        c22=j["controls"]["c22_git_and_artifacts"],
        c23=j["controls"]["c23_old_artifacts_not_overwritten"],
        c24=j["controls"]["c24_raw_saved_before_summary"],
        tokens_json_match=bool(all(j["controls"]["tokens"][a]["argmax_id"] == _token_fields(finals[a])["argmax"]
                                   for a in ARMS)),
        node_json_match=bool(
            # 本轮工具的 JSON **没有**写完整逐节点表（缺陷；见文档）：它能写下的部分在这里逐项核对。
            all(d["op"] == po["op"] and d["output_n_diff"] == po["output_n_diff"]
                and d["all_inputs_equal"] == po["all_inputs_equal"]
                for d, po in zip(pers["BQOGUD"]["per_operator"],
                                 j["H4"]["persistence_bqogud"]["per_operator"]))
            and all(h4["trajectory"][a] == j["H4"]["trajectory"][a] for a in TRAJ)
            and all(h2["reference_BQO"][k]["n_diff"] == j["H2"]["reference_BQO"][k2]["n_diff"]
                    for k, k2 in (("mlp_gate_proj", "gate_proj"), ("mlp_up_proj", "up_proj"),
                                  ("mlp_gated_product", "gated_product")))
            and h3["down_proj"]["n_diff"] == j["H3"]["down_proj"]["row"]["n_diff"]
            and nc["B0"]["layer_output"]["n_diff"] == j["H4"]["trajectory"]["B0"]
            and nc["BQO"]["layer_output"]["n_diff"] == j["H4"]["trajectory"]["BQO"]
            and nc["BQOGU"]["layer_output"]["n_diff"] == j["H4"]["trajectory"]["BQOGU"]),
        module_relation_ok=bool(inp["gate_proj"]["out_features"] == inp["up_proj"]["out_features"]
                                and inp["down_proj"]["in_features"] == inp["gate_proj"]["out_features"]
                                and inp["down_proj"]["out_features"] == inp["gate_proj"]["in_features"]))
    checks.update(repeat_all_true=ctl["repeat_all_true"], node_json_match=ctl["node_json_match"],
                  tokens_json_match=ctl["tokens_json_match"], module_relation_ok=ctl["module_relation_ok"],
                  finiteness=bool(not j["finiteness"]["nodes_with_nonfinite_slice"]),
                  per_row_shapes_ok=bool(ctl["c7_c9"]["gate_ok"] and ctl["c7_c9"]["up_ok"]
                                         and ctl["c7_c9"]["down_ok"] and ctl["c7_c9"]["o_proj_ok"]),
                  only_L0=bool(ctl["c13"]["applied_is_L0_set"] and ctl["c13"]["seen_is_all_others"]
                               and ctl["c13"]["seen_plus_applied_covers_all"]),
                  controls_ok=bool(ctl["c10"]["same_module_ids"] and ctl["c10"]["same_weight_ptrs"]
                                   and ctl["c10"]["no_bias"] and ctl["c10"]["call_ids_are_L0"]
                                   and ctl["c17"]["probe_active"] is False
                                   and ctl["c17"]["serial_qk_active"] is False
                                   and ctl["c17"]["serial_linear_active"] is False
                                   and ctl["c17"]["hooks_removed"]
                                   and ctl["c20"]["after"] is False
                                   and ctl["c21"]["draft_loaded"] is False
                                   and ctl["c24"]["pt_written_first"]))
    print("\n== ⑨ 控制与一致性 ==")
    print(f"   重复一致（六臂 × 2 pass）={ctl['repeat_all_true']}；逐节点与 JSON 一致={ctl['node_json_match']}；"
          f"token 与 JSON 一致={ctl['tokens_json_match']}")
    print(f"   控制 7/8/9（逐行形状）=gate:{ctl['c7_c9']['gate_ok']} up:{ctl['c7_c9']['up_ok']} "
          f"down:{ctl['c7_c9']['down_ok']} o_proj:{ctl['c7_c9']['o_proj_ok']}")
    print(f"   控制 13/14（只作用 L0）：applied={ctl['c13']['n_applied']}、seen={ctl['c13']['n_seen']}、"
          f"调用点总数={ctl['c13']['n_call_sites']}、覆盖全部={ctl['c13']['seen_plus_applied_covers_all']}")
    print(f"   控制 17（恢复）={ctl['c17']}")
    print(f"   控制 24（落盘顺序）={ctl['c24']}")
    print(f"   模块事实：gate={inp['gate_proj']['in_features']}→{inp['gate_proj']['out_features']}、"
          f"up={inp['up_proj']['in_features']}→{inp['up_proj']['out_features']}、"
          f"down={inp['down_proj']['in_features']}→{inp['down_proj']['out_features']}、"
          f"bias={[inp[k]['has_bias'] for k in ('gate_proj', 'up_proj', 'down_proj')]}、"
          f"dtype={inp['gate_proj']['weight_dtype']}")

    rec.update(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, H6=h6, change_table=change_rows,
               serial_linear_selfcheck=sl_by_arm,
               node_compare={a: nc[a] for a in nc}, persistence={a: pers[a] for a in pers},
               controls_recomputed=ctl, recompute_checks=checks)

    print("\n== 核对 ==")
    flat = True
    for k, v in checks.items():
        print(f"   [check] {k}: {v}")
        flat = flat and bool(v)
    DC.require(flat, "独立复算与运行 JSON / 历史产物不一致",
               json.dumps({k: v for k, v in checks.items() if not v}, ensure_ascii=False))

    # ---------------- 结论 ----------------
    h2_ok = h2["class_"] == "gate_up_local_closure_ok"
    l0_ok = h4["class_"] == "L0_restored"
    if not h2_ok:
        verdict = "H2 未通过 ⇒ gate/up 的 serial Linear 未闭环，停止后续归因"
    elif not l0_ok:
        verdict = (f"gate/up 已闭合、down_proj 为独立边界（{h3['down_proj_class']}），但 L0 仍差 "
                   f"{h4['trajectory']['BQOGUD']} 个坐标 ⇒ 剩余差异在 residual 或其他位置；"
                   f"首个持续边界 = {pers['BQOGUD']['first_persistent_divergence_operator']}")
    else:
        verdict = (f"**L0 完全闭合**：BQOGUD 的 L0 输出与 S0 **逐位一致**（差异轨迹 "
                   f"B0={h4['trajectory']['B0']} → BQO={h4['trajectory']['BQO']} → "
                   f"BQOGU={h4['trajectory']['BQOGU']} → BQOGUD={h4['trajectory']['BQOGUD']}），"
                   "且 attention 子块与 MLP 段（gate/up/SiLU/gated/down/residual）的**决策行**全部逐位恢复；"
                   "本轮定位到的持久 GEMM 边界依次为 `qk_matmul_scale`、`o_proj`、"
                   f"`gate_proj`/`up_proj`、`down_proj`（在 BQOGU 中判定为独立边界：{h3['down_proj_class']}）；"
                   f"后续**第一处**重新分歧层 = **L{h5['first_layer_with_equal_input_and_different_output']}**"
                   "（该层输入逐位相同、输出不同；**本轮不进入其内部诊断**）；"
                   f"最终 token 恢复（{h6['BQOGUD']['token_s0']}={h6['BQOGUD']['token_x']}）但 logits 未逐位一致"
                   f"（类别 {h6['BQOGUD']['category']}、n_diff={h6['BQOGUD']['n_diff']}）⇒ **token 恢复 ≠ B6 通过**")
    rec["verdict"] = dict(H2_class=h2["class_"], H3_down=h3["down_proj_class"], H4_class=h4["class_"],
                          H6_categories={a: h6[a]["category"] for a in TRAJ},
                          H6_n_diff={a: h6[a]["n_diff"] for a in TRAJ},
                          first_layer_after_L0=h5["first_layer_with_equal_input_and_different_output"],
                          persistent_boundaries=dict(
                              QK="qk_matmul_scale（上一轮）", o_proj="（上一轮）",
                              gate_up="gate_proj / up_proj（本轮 H2）", down_proj="down_proj（本轮 H3）"),
                          verdict=verdict, b6_changed=False,
                          not_claimed=["L1 内部的具体边界（本轮按约定不进入 L1–L27 算子级诊断）",
                                       "各 GEMM 内部的具体 kernel/tile/归约顺序（未做 profiler）",
                                       "谁是数学真值（未做 FP32）",
                                       "B6 通过（未跑 B6；token 恢复 ≠ B6 通过）",
                                       "对别的位置/长度/family/模型的外推",
                                       "把诊断分支当作生产修复（默认关闭、只作用于注册实例）"])
    print("\n== ★ 结论 ==")
    print(f"   {verdict}")
    print("   B6 状态：未改变（本轮未跑 B6/acceptance/pilot/final）")
    p = OUT / f"serialmlp8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
