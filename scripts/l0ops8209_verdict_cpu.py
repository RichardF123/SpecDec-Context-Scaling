"""l0ops8209_verdict_cpu.py — 位置 8209 的 **L0 内部算子边界**只读汇总（纯 CPU、不加载模型、不改既有产物）。

## 作用
从 `div8209_l0ops_*.pt` 的**原始张量**独立复算：
① 25 个节点（+`cache_pre` 虚拟节点）的 S/B 对应决策切片比较（exact / n_diff / max|Δ| / mean|Δ| / sha16）；
② 每个算子"**所有**输入逐位相同 + 输出首次不同"的判定与 `first_divergence_operator`；
③ 与运行 JSON 的逐节点摘要逐项核对、`slices` 与从 `raw_*` 重算的切片逐位核对；
④ 与上一轮全层捕获的 L0 输入/输出决策行逐位核对；与第二次完整 run 的原始张量逐位核对（跨 run 确定性）；
⑤ 输出层级表格与结论（无歧义字段名）。

**不**加载大模型、**不**做前向、**不**做 FP32 投影、**不**改任何既有产物。
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

from specdec.engine import op_probe as OP            # noqa: E402
from specdec.engine import sb_align as A             # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_l0ops_20261002T172447Z.json"
DEF_PT = OUT / "div8209_l0ops_20261002T172447Z.pt"
PT2 = OUT / "div8209_l0ops_20261002T172301Z.pt"
CHAIN_PT = OUT / "div8209_chain_20261002T164558Z.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    j = json.loads(jp.read_text(encoding="utf-8"))
    VIS = int(j["config"]["vis"])
    pt = torch.load(tp, map_location="cpu")
    rec: dict = dict(tool="l0ops8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始张量独立复算 L0 内部 25 个节点的对应决策切片比较与首次分歧判定；"
                               "不做前向、不做 FP32、不改既有产物",
                     sources=dict(json=str(jp), pt=str(tp), pt2=str(PT2), chain_pt=str(CHAIN_PT)),
                     vis=VIS, note="全部比较用原始 dtype；仅统计时转 FP32")

    # ---- ① 独立复算（从 raw_* 重算，并核对 slices）----
    node = {}
    slice_mismatch = []
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        c = OP.cmp_node(name, kind, pt["raw_S"][name], pt["raw_B"][name], vis=VIS)
        node[name] = c
        a_s, b_s, _r = OP.slice_pair(kind, pt["raw_S"][name], pt["raw_B"][name], vis=VIS)
        if not (torch.equal(a_s, pt["slices"][f"{name}_S"]) and torch.equal(b_s, pt["slices"][f"{name}_B"])):
            slice_mismatch.append(name)
    cache_pre = dict(exact_equal=bool(torch.equal(pt["cache_pre_S"]["k"][:, :, :VIS],
                                                  pt["cache_pre_B"]["k"][:, :, :VIS])),
                     n_diff=int((pt["cache_pre_S"]["k"][:, :, :VIS]
                                 != pt["cache_pre_B"]["k"][:, :, :VIS]).sum()),
                     note="虚拟节点：L0 轮前缓存 K（共同可见范围 0..8208）")
    rec["node_compare"] = node
    rec["cache_pre"] = cache_pre
    rec["slices_match_raw_recompute"] = bool(not slice_mismatch)

    print("== ① L0 内部逐节点比较（独立复算：原始 dtype 张量 + 语义切片）==")
    print("   node                     exact  n_diff   max|Δ|        max_idx    S@max        B@max        sha16(S/B)")
    for (op, ins), outn in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
        c = node[outn]
        print(f"   {outn:24s} {str(c['exact_equal']):5s} {c['n_diff']:>6d}  {c['max_abs']:<12.6g} "
              f"{str(c['max_diff_index']):<11s} {c['s_at_max']:<12.6g} {c['b_at_max']:<12.6g} "
              f"{c['sha16_s']}/{c['sha16_b']}")

    # ---- ② 首次分歧判定（同一纯函数）----
    cls = OP.classify(OP.CHAIN, dict(node, cache_pre=cache_pre), OP.CHAIN_NODE_OUT)
    rec["first_divergence"] = cls
    print("\n== ② 首次分歧判定 ==")
    print(f"   first_divergence_operator = {cls['first_divergence_operator']}")
    print(f"   满足条件（所有输入逐位相同 且 输出不同）的算子 = "
          f"{cls['all_operators_with_equal_inputs_and_different_output']}")
    for d in cls["per_operator"]:
        if d["all_inputs_equal"] and d["output_equal"] is False:
            print(f"     · {d['op']:<22s} inputs={d['inputs']} output={d['output']}")

    # ---- ③ 与 JSON / 上一轮 / 第二次 run 核对 ----
    checks = dict(
        json_node_match={k: bool(node[k]["exact_equal"] == j["node_compare"][k]["exact_equal"]
                                 and node[k]["n_diff"] == j["node_compare"][k]["n_diff"]
                                 and node[k]["sha16_s"] == j["node_compare"][k]["sha16_s"]
                                 and node[k]["sha16_b"] == j["node_compare"][k]["sha16_b"])
                         for k in node},
        json_first_divergence_match=bool(cls["first_divergence_operator"]
                                         == j["first_divergence"]["first_divergence_operator"]
                                         and cls["all_operators_with_equal_inputs_and_different_output"]
                                         == j["first_divergence"]["all_operators_with_equal_inputs_and_different_output"]),
        slices_match=bool(not slice_mismatch),
    )
    if Path(CHAIN_PT).exists():
        ch = torch.load(CHAIN_PT, map_location="cpu")
        l0_in = ch["rows_S_raw"][0]["input"]
        l0_out_s = ch["rows_S_raw"][0]["output"]
        l0_out_b = ch["rows_B_raw"][0]["output"]
        checks["L0_input_equals_prev_round"] = bool(torch.equal(pt["x0_row_S"], l0_in)
                                                    and torch.equal(pt["x0_row_B"], l0_in))
        checks["L0_output_equals_prev_round"] = bool(torch.equal(pt["L0_output_row_S"], l0_out_s)
                                                     and torch.equal(pt["L0_output_row_B"], l0_out_b))
        checks["L0_output_prev_round_n_diff_is_111"] = bool(
            int((l0_out_s.reshape(-1) != l0_out_b.reshape(-1)).sum()) == 111)
        checks["final_rows_equals_prev_round"] = bool(
            torch.equal(pt["final_row_S"], ch["final_row_S"])
            and torch.equal(pt["final_row_B"], ch["final_row_B"]))
    if Path(PT2).exists():
        pt2 = torch.load(PT2, map_location="cpu")
        checks["second_run_all_nodes_bitwise_equal"] = bool(all(
            torch.equal(pt["raw_S"][k], pt2["raw_S"][k]) and torch.equal(pt["raw_B"][k], pt2["raw_B"][k])
            for k in pt["raw_S"]))
    rec["recompute_checks"] = checks
    print("\n== ③ 与 JSON / 上一轮 / 第二次 run 核对 ==")
    flat = True
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"   {k}: 全真={all(v.values())}（{sum(v.values())}/{len(v)}）")
            flat = flat and all(v.values())
        else:
            print(f"   {k}: {v}")
            flat = flat and bool(v)
    DC = __import__("specdec.engine.diag_checks", fromlist=["x"])
    DC.require(flat, "独立复算与运行 JSON/上一轮/第二次 run 不一致",
               json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not vv})
                           for k, v in checks.items()}, ensure_ascii=False))

    # ---- ④ 结论（无歧义字段名）----
    q = node["q_linear"]
    qk = node["qk_scores_pre_mask"]
    rec["verdict"] = dict(
        first_divergence_operator=cls["first_divergence_operator"],
        also_satisfying=cls["all_operators_with_equal_inputs_and_different_output"],
        q_proj=dict(inputs=("input_rmsnorm",), input_exact_equal=node["input_rmsnorm"]["exact_equal"],
                    output_exact_equal=q["exact_equal"], n_diff=q["n_diff"], max_abs=q["max_abs"],
                    max_diff_index=q["max_diff_index"], s_at_max=q["s_at_max"], b_at_max=q["b_at_max"],
                    sha16_s=q["sha16_s"], sha16_b=q["sha16_b"]),
        k_proj_contrast=dict(output_exact_equal=node["k_linear"]["exact_equal"],
                             n_diff=node["k_linear"]["n_diff"]),
        v_proj_contrast=dict(output_exact_equal=node["v_linear"]["exact_equal"],
                             n_diff=node["v_linear"]["n_diff"]),
        rope_q=dict(input_already_differs=bool(not node["q_reshaped"]["exact_equal"]),
                    output_exact_equal=node["q_rope"]["exact_equal"], n_diff=node["q_rope"]["n_diff"],
                    note="进入 RoPE 的 q 已经不逐位相同 ⇒ RoPE **不是**首次分歧边界；"
                         "而且本案例里 RoPE 的输出又变回**逐位相同**（差异被舍入吸收）"),
        qk_matmul=dict(inputs_exact_equal=bool(node["q_rope"]["exact_equal"]
                                               and node["k_visible"]["exact_equal"]),
                       n_diff=qk["n_diff"], max_abs=qk["max_abs"], max_diff_index=qk["max_diff_index"],
                       s_at_max=qk["s_at_max"], b_at_max=qk["b_at_max"],
                       note="输入（q_rope、k_visible）都逐位相同，输出不同 ⇒ **也**满足首次分歧条件"),
        wording=("`q_proj` 是本案例中**最早观测到的 shape-sensitive 数值分歧边界**："
                 "输入（input RMSNorm 输出）逐位相同、权重为同一张量、输出有 2/1536 个坐标差 1 个 bf16 ULP。"
                 "**不能**据此写成 PyTorch/cuBLAS/BF16 的实现缺陷：要区分 kernel、归约顺序与实现错误，"
                 "还需要独立控制（本轮不做）。"),
        not_claimed=["q_proj 内部的具体 kernel/归约差异（本轮只定位边界）",
                     "323/659 谁是数学真值（未做 FP32）",
                     "把数值差异直接判为实现 bug", "用 margin/ULP 豁免正确性", "B6 通过或修复完成",
                     "对别的位置/长度/历史的推广"])
    print("\n== ④ 结论 ==")
    for k, v in rec["verdict"].items():
        print(f"   {k}: {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False, default=str)}")
    p = OUT / f"l0ops8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
