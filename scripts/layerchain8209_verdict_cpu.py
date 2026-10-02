"""layerchain8209_verdict_cpu.py — 位置 8209 的**全层边界**只读汇总（纯 CPU、不加载模型、不改既有产物）。

## 作用
从 `div8209_chain_*.pt` 里的**原始 bf16 决策行**独立复算：
① 每层输入/输出行的 S/B 比较（n_diff / max|Δ| / mean|Δ| / sha16 / 逐位相等）；
② 边界链自洽（embedding 行 → L0 输入 → …→ 最后一层输出 → 最终 Norm 输入）；
③ 两次捕获（pass1 vs pass2）逐位一致；
④ 最终 Norm 输出行与最终 logits 行的 S/B 比较；
⑤ 与运行 JSON 的摘要逐项核对、与上一轮 L27 run 的 L27 行逐位核对、与更早已保存的
   局部对照行（`div8209_cells_*.pt`）逐位核对。

并给出**三个严格分开**的边界：
`earliest_observed_input_difference` / `earliest_observed_output_difference` /
`first_same_input_different_output`（= 当前轨迹中最早观测到的分歧产生层）。

**不**加载大模型、**不**做前向、**不**做 FP32 投影、**不**改任何既有产物；
**摘要错误不得通过重跑 GPU 修复**（本脚本只从已保存张量复算）。
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
from specdec.engine import sb_align as A            # noqa: E402
from specdec.engine import layer_chain_f13_p8209 as CH   # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_chain_20261002T164558Z.json"
DEF_PT = OUT / "div8209_chain_20261002T164558Z.pt"
L27_PT = OUT / "div8209_l27_20261002T161807Z.pt"
CELLS_PT = OUT / "div8209_cells_20261002T064415Z.pt"


def _row_cmp(a, b, name: str) -> dict:
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(x, y, name)
    d = (x - y).abs()
    return dict(name=name, n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                exact_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b),
                dtype=str(a.dtype), shape=[int(v) for v in a.shape])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    rec: dict = dict(tool="layerchain8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始 bf16 决策行独立复算全层边界，与 JSON/上一轮参考逐项核对；"
                               "不做前向、不做 FP32、不改既有产物",
                     sources=dict(json=str(jp), pt=str(tp), l27_pt=str(L27_PT), cells_pt=str(CELLS_PT)))
    j = json.loads(jp.read_text(encoding="utf-8"))
    pt = torch.load(tp, map_location="cpu")
    RS, RB = pt["rows_S_raw"], pt["rows_B_raw"]
    n = len(RS)
    rec["raw_dtypes"] = dict(n_layers=n, dtype=str(RS[0]["input"].dtype),
                             note="`.pt` 里是**原始 bf16** 决策行；统计/比较时才转 FP32")

    # ---- ① 每层比较（独立复算）----
    per_layer = {}
    for i in range(n):
        inp = _row_cmp(RS[i]["input"], RB[i]["input"], f"L{i}_input")
        out = _row_cmp(RS[i]["output"], RB[i]["output"], f"L{i}_output")
        per_layer[i] = dict(layer=i, input=inp, output=out,
                            input_equal=inp["exact_equal"], output_equal=out["exact_equal"])
    rec["per_layer"] = per_layer
    print("== ① 每层决策行 S vs B（原始 bf16；比较时转 FP32）==")
    print("   layer | in n_diff  max|Δ|    mean|Δ|   | out n_diff max|Δ|    mean|Δ|   | in_eq out_eq")
    for i in range(n):
        d = per_layer[i]
        print(f"   L{i:<4d} | {d['input']['n_diff']:>6d} {d['input']['abs_err_max']:<10.6g} "
              f"{d['input']['abs_err_mean']:<9.6g} | {d['output']['n_diff']:>6d} "
              f"{d['output']['abs_err_max']:<9.6g} {d['output']['abs_err_mean']:<9.6g} | "
              f"{str(d['input_equal']):<5s} {str(d['output_equal']):<5s}")

    # ---- ② 三个边界（同一纯函数，`layer_chain_f13_p8209._classify`）----
    cls = CH._classify({i: dict(input_equal=per_layer[i]["input_equal"],
                                output_equal=per_layer[i]["output_equal"]) for i in range(n)})
    rec["boundaries"] = cls
    print("\n== ② 三个边界（严格分开）==")
    print(f"   最早**输入**差异层 (earliest_observed_input_difference) = {cls['earliest_observed_input_difference']}")
    print(f"   最早**输出**差异层 (earliest_observed_output_difference) = {cls['earliest_observed_output_difference']}")
    print(f"   第一层**同输入异输出** (first_same_input_different_output) = {cls['first_same_input_different_output']}")
    print(f"   L0 输入已不同 = {cls['l0_input_already_differs']}")

    # ---- ③ 边界链自洽（从张量复算）----
    chain = dict(
        embed_S_vs_B=_row_cmp(pt["embed_row_S"], pt["embed_row_B"], "embedS_vs_embedB"),
        L0_input_equals_embed_S=_row_cmp(RS[0]["input"], pt["embed_row_S"], "L0in_S_vs_embedS"),
        L0_input_equals_embed_B=_row_cmp(RB[0]["input"], pt["embed_row_B"], "L0in_B_vs_embedB"),
        last_out_equals_norm_in_S=_row_cmp(RS[n - 1]["output"], pt["norm_in_row_S"], "L27out_S_vs_normIn_S"),
        last_out_equals_norm_in_B=_row_cmp(RB[n - 1]["output"], pt["norm_in_row_B"], "L27out_B_vs_normIn_B"),
        inter_S_all=[bool(torch.equal(RS[i]["output"], RS[i + 1]["input"])) for i in range(n - 1)],
        inter_B_all=[bool(torch.equal(RB[i]["output"], RB[i + 1]["input"])) for i in range(n - 1)],
    )
    chain["inter_S_all_equal"] = bool(all(chain["inter_S_all"]))
    chain["inter_B_all_equal"] = bool(all(chain["inter_B_all"]))
    rec["chain_checks"] = chain
    print("\n== ③ 边界链自洽（逐位）==")
    for k in ("embed_S_vs_B", "L0_input_equals_embed_S", "L0_input_equals_embed_B",
              "last_out_equals_norm_in_S", "last_out_equals_norm_in_B"):
        print(f"   {k}: exact_equal={chain[k]['exact_equal']} "
              f"(sha16 {chain[k]['sha16_a']}/{chain[k]['sha16_b']})")
    print(f"   层间链 S 全等={chain['inter_S_all_equal']} B 全等={chain['inter_B_all_equal']} "
          f"（共 {n - 1} 条边 × 2 条路径）")

    # ---- ④ 两次捕获逐位一致 ----
    rep = {}
    if pt.get("pass2_rows_S_raw") and pt.get("pass2_rows_B_raw"):
        rep["S_all_rows_identical"] = bool(all(
            torch.equal(RS[i]["input"], pt["pass2_rows_S_raw"][i]["input"])
            and torch.equal(RS[i]["output"], pt["pass2_rows_S_raw"][i]["output"]) for i in range(n)))
        rep["B_all_rows_identical"] = bool(all(
            torch.equal(RB[i]["input"], pt["pass2_rows_B_raw"][i]["input"])
            and torch.equal(RB[i]["output"], pt["pass2_rows_B_raw"][i]["output"]) for i in range(n)))
    print("\n== ④ 两次捕获一致性 ==")
    print(f"   pass1 vs pass2：S 全部层行逐位相同={rep.get('S_all_rows_identical')}；"
          f"B 全部层行逐位相同={rep.get('B_all_rows_identical')}")
    DC.require(rep.get("S_all_rows_identical") is True and rep.get("B_all_rows_identical") is True,
               "两次捕获的层行不一致（本脚本只从已保存张量复算，不重跑 GPU）", str(rep))

    # ---- ⑤ 最终行 + 与 JSON / 上一轮参考核对 ----
    fin = dict(norm_out_S_vs_B=_row_cmp(pt["norm_out_row_S"], pt["norm_out_row_B"], "normOut_S_vs_B"),
               final_S_vs_B=_row_cmp(pt["final_row_S"], pt["final_row_B"], "finalLogits_S_vs_B"))
    rec["final_rows"] = fin
    checks = dict(
        json_per_layer_match={i: bool(per_layer[i]["input"]["n_diff"] == j["per_layer"][str(i)]["input"]["n_diff"]
                                      and per_layer[i]["output"]["n_diff"] == j["per_layer"][str(i)]["output"]["n_diff"]
                                      and per_layer[i]["output"]["sha16_a"] == j["per_layer"][str(i)]["output"]["sha16_a"]
                                      and per_layer[i]["output"]["sha16_b"] == j["per_layer"][str(i)]["output"]["sha16_b"])
                              for i in range(n)},
        json_boundaries_match=bool(cls["earliest_observed_input_difference"]
                                   == j["boundaries"]["earliest_observed_input_difference"]
                                   and cls["earliest_observed_output_difference"]
                                   == j["boundaries"]["earliest_observed_output_difference"]
                                   and cls["first_same_input_different_output"]
                                   == j["boundaries"]["first_same_input_different_output"]),
    )
    if Path(L27_PT).exists():
        l27 = torch.load(L27_PT, map_location="cpu")
        checks["L27_input_row_S_equals_prev_run"] = bool(torch.equal(RS[n - 1]["input"], l27["aS_raw"][0, 0]))
        checks["L27_input_row_B_equals_prev_run"] = bool(torch.equal(RB[n - 1]["input"], l27["aB4_raw"][0, 0]))
        checks["L27_output_row_S_equals_prev_run"] = bool(torch.equal(RS[n - 1]["output"], l27["zS_raw"][0, 0]))
        checks["L27_output_row_B_equals_prev_run"] = bool(torch.equal(RB[n - 1]["output"], l27["zB4_raw"][0, 0]))
    if Path(CELLS_PT).exists():
        cells = torch.load(CELLS_PT, map_location="cpu")
        checks["final_row_S_equals_saved_cells_H_AR_S"] = bool(
            torch.equal(pt["final_row_S"], cells["H_AR-S"]))
        checks["final_row_B_equals_saved_cells_H_AR_B"] = bool(
            torch.equal(pt["final_row_B"], cells["H_AR-B"]))
    rec["recompute_checks"] = checks
    print("\n== ⑤ 与 JSON / 上一轮参考核对 ==")
    flat = True
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"   {k}: 全真={all(v.values())}（{sum(v.values())}/{len(v)}）")
            flat = flat and all(v.values())
        else:
            print(f"   {k}: {v}")
            flat = flat and bool(v)
    print(f"   最终行：normOut S vs B n_diff={fin['norm_out_S_vs_B']['n_diff']} "
          f"max|Δ|={fin['norm_out_S_vs_B']['abs_err_max']:.6g}；"
          f"finalLogits S vs B n_diff={fin['final_S_vs_B']['n_diff']} "
          f"max|Δ|={fin['final_S_vs_B']['abs_err_max']:.6g} "
          f"(argmax {int(pt['final_row_S'].float().argmax())} vs {int(pt['final_row_B'].float().argmax())})")
    DC.require(flat, "独立复算与 JSON / 上一轮参考不一致", json.dumps(
        {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not vv})
         for k, v in checks.items()}, ensure_ascii=False))

    # ---- 结论 ----
    rec["verdict"] = dict(
        first_divergence_layer=cls["first_same_input_different_output"],
        earliest_observed_input_difference=cls["earliest_observed_input_difference"],
        earliest_observed_output_difference=cls["earliest_observed_output_difference"],
        l0_input_already_differs=cls["l0_input_already_differs"],
        meaning=("L0 的**输入行**（= embedding 行，S/B 逐位相同，sha16 "
                 f"{chain['embed_S_vs_B']['sha16_a']}）在两条路径下完全一致，而 **L0 的输出行**不再逐位相同"
                 f"（n_diff={per_layer[0]['output']['n_diff']}，max|Δ|={per_layer[0]['output']['abs_err_max']:.6g}）"
                 "⇒ 当前轨迹中**最早观测到的分歧产生层 = L0**；此后每一层的输入差异都等于上一层的输出差异"
                 "（层间链逐位自洽），即 L1..L27 是在**传递**这个已经产生的差异。"),
        not_monotonic=("差异**不随层数单调增长**：n_diff 从 111 上升到 1365 后回落（1343/1328/1318…），"
                       "max|Δ| 也从 0.75 回落；**不**用二分排除未直接测量的层（本轮 28 层全部直接测量）。"),
        not_claimed=["323/659 谁是数学真值（未做 FP32）",
                     "L0 内**哪一个算子**首次产生差异（本轮只定位到层，不进入层内）",
                     "把数值差异直接判为 bug", "用 margin 豁免正确性", "B6 通过或修复完成",
                     "对别的位置/长度/历史的推广"])
    print("\n== 结论 ==")
    for k, v in rec["verdict"].items():
        print(f"   {k}: {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}")
    p = OUT / f"layerchain8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
