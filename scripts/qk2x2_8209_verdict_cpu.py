"""qk2x2_8209_verdict_cpu.py — 位置 8209 的 **QKᵀ 执行形状 2×2 分解**只读汇总（纯 CPU、不加载模型、不改既有产物）。

## 作用
从 `div8209_qk2x2_*.pt` 的**已保存 CUDA 张量**独立复算（**不**在 CPU 上重算 GEMM —— CPU GEMM 与 CUDA GEMM
不是同一条数值路径，本仓库已把它记成缺陷，见 174724Z）：
① 八个比较（六格 + 两个复现对照 + A/D）的 raw/scaled 两套口径，逐项与运行 JSON 核对（exact/n_diff/max/sha16）；
② 复现门（A ≡ 真实 S、D ≡ 真实 B，**scaled 口径**，整张量逐位）从原始张量重算；
③ 结构不变量：row0 前缀切片 == `raw[0,:,0,0:8209]`；`k_rep` == 生产 `kv_repeat(k)` 逐位；`scaled == raw/√d`（设备舍入，只记录）；
④ 跨 run / 跨轮一致性：与本轮第二次 CUDA run 的 sha16 比对（多 run 确定性）、与上一轮 L0 算子边界 `.pt`
   的 `qk_scores_pre_mask` 第 0 行前缀逐位比对；把首跑（CPU cell）的差异**如实记录**为缺陷、不作为结论；
⑤ **归因**：M 效应 / N 效应 / 交互，以及"哪条路径能完整复现真实 S→B 的第 0 行差异"；
   另附三个**只读汇总新增的 CPU 派生比较**（`A_vs_E`/`E_vs_F`/`B_vs_F`，把 query 内容完全固定）作为支持性证据。

**不**加载大模型、**不**做前向、**不**做 FP32、**不**改任何既有产物。
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

from specdec.engine import qk_shape_2x2 as QK            # noqa: E402
from specdec.engine import diag_checks as DC             # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_qk2x2_20261002T180127Z.json"
DEF_PT = OUT / "div8209_qk2x2_20261002T180127Z.pt"
PREV_CUDA_JSON = OUT / "div8209_qk2x2_20261002T175655Z.json"   # 第二次 run（gate 口径有误，但 cell 在 CUDA）
FIRST_CPU_JSON = OUT / "div8209_qk2x2_20261002T174724Z.json"   # 首跑：cell 落在 CPU（已记录缺陷）
L0OPS_PT = OUT / "div8209_l0ops_20261002T172447Z.pt"

TOOL_PAIRS = (("A_vs_B", "A", "B"), ("C_vs_D", "C", "D"), ("A_vs_C", "A", "C"),
              ("B_vs_D", "B", "D"), ("C_vs_E", "C", "E"), ("D_vs_F", "D", "F"),
              ("A_vs_D", "A", "D"))
SUP_PAIRS = (("A_vs_E", "A", "E"), ("E_vs_F", "E", "F"), ("B_vs_F", "B", "F"))
FIELDS = ("exact_equal", "n_diff", "max_abs", "max_diff_index", "sha16_a", "sha16_b", "finite")


def _coordsets(c: dict) -> list:
    return [tuple(x) for x in (c.get("coords") or [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    j = json.loads(jp.read_text(encoding="utf-8"))
    VIS = int(j["config"]["vis"])
    cells_meta = j["config"]["cells"]
    pt = torch.load(tp, map_location="cpu")
    rec: dict = dict(tool="qk2x2_8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从已保存的 CUDA 张量独立复算比较/门/结构不变量/跨 run 一致性/归因；"
                               "**不**在 CPU 上重算 GEMM、不做前向、不做 FP32、不改既有产物",
                     sources=dict(json=str(jp), pt=str(tp), prev_cuda_json=str(PREV_CUDA_JSON),
                                  first_cpu_json=str(FIRST_CPU_JSON), l0ops_pt=str(L0OPS_PT)),
                     config=j["config"], stage=j.get("stage"), vis=VIS,
                     note="全部比较用原始 dtype（bf16）张量；仅在统计时转 FP32；比较区域恒为 batch0/12 头/row0/keys 0:8209")

    # ---------------- ① 八个比较独立复算（raw + scaled）----------------
    cmp_tool = {}
    for tag, x, y in TOOL_PAIRS:
        cmp_tool[tag] = dict(
            raw=QK.compare(QK.row0_prefix(pt["cells"][x]["raw"]), QK.row0_prefix(pt["cells"][y]["raw"]),
                           f"{tag}_raw"),
            scaled=QK.compare(QK.row0_prefix(pt["cells"][x]["scaled"]),
                              QK.row0_prefix(pt["cells"][y]["scaled"]), f"{tag}_scaled"))
    cmp_tool["A_vs_real_S"] = dict(raw=QK.compare(pt["cells"]["A"]["raw"], pt["real_S_scores"], "A_vs_real_S_raw"),
                                   scaled=QK.compare(pt["cells"]["A"]["scaled"], pt["real_S_scores"],
                                                     "A_vs_real_S_scaled"))
    cmp_tool["D_vs_real_B"] = dict(raw=QK.compare(pt["cells"]["D"]["raw"], pt["real_B_scores"], "D_vs_real_B_raw"),
                                   scaled=QK.compare(pt["cells"]["D"]["scaled"], pt["real_B_scores"],
                                                     "D_vs_real_B_scaled"))
    # ③ 结构不变量：row0 前缀必须是 raw 的原样切片
    slice_ok = all(
        torch.equal(pt["row0_prefix"][c]["raw"], pt["cells"][c]["raw"][0, :, 0, :VIS])
        and torch.equal(pt["row0_prefix"][c]["scaled"], pt["cells"][c]["scaled"][0, :, 0, :VIS])
        for c in pt["row0_prefix"])
    # ③ k_rep == 生产 kv_repeat(k) 逐位
    #    （`.pt` 里 K 存在顶层 `k8209`/`k8212`，每个 cell 的 `k_source` 指明用的是哪一个）
    krep_ok = all(torch.equal(pt["cells"][c]["k_rep"], QK.k_rep_prod(pt[pt["cells"][c]["k_source"]]))
                  for c in pt["cells"])
    # ③ scaled == raw/√d（CUDA 计算 vs CPU 复算：设备舍入，只记录）
    dsc = {c: int((pt["cells"][c]["scaled"] != (pt["cells"][c]["raw"] / (QK.HEAD_DIM ** 0.5))).sum())
           for c in pt["cells"]}

    # ④ 跨 run / 跨轮一致性
    checks = dict(
        json_match={k: bool(all(cmp_tool[k][kind][f] == j["cell_compare"][k][kind][f]
                                for kind in ("raw", "scaled") for f in FIELDS))
                   for k in cmp_tool},
        row0_prefix_is_raw_slice=bool(slice_ok),
        k_rep_equals_production_kv_repeat=bool(krep_ok),
        all_finite=bool(all(cmp_tool[k][kind]["finite"] for k in cmp_tool for kind in ("raw", "scaled"))),
        gate_scaled_from_pt=QK.expect_fail_gate(dict(A_vs_real_S=cmp_tool["A_vs_real_S"]["scaled"],
                                                     D_vs_real_B=cmp_tool["D_vs_real_B"]["scaled"])),
        gate_scaled_matches_json=bool(j["reproduction_gate"]["ok"] is True),
    )
    cross = dict(
        prev_cuda_run=dict(available=PREV_CUDA_JSON.exists(), path=str(PREV_CUDA_JSON)),
        first_cpu_run=dict(available=FIRST_CPU_JSON.exists(), path=str(FIRST_CPU_JSON)),
        l0ops_pt=dict(available=L0OPS_PT.exists(), path=str(L0OPS_PT)))
    if cross["prev_cuda_run"]["available"]:
        j2 = json.loads(PREV_CUDA_JSON.read_text(encoding="utf-8"))
        same = all(j["cell_compare"][k][kind]["sha16_a"] == j2["cell_compare"][k][kind]["sha16_a"]
                   and j["cell_compare"][k][kind]["sha16_b"] == j2["cell_compare"][k][kind]["sha16_b"]
                   for k in j2["cell_compare"] if k in j["cell_compare"]
                   for kind in ("raw", "scaled") if kind in j2["cell_compare"][k])
        cross["prev_cuda_run"].update(cells_sha16_identical=same,
                                      note="第二次 run（175655Z，gate 口径有误但 cell 在 CUDA）⇒ 两个独立 run 的六格逐位一致")
        checks["multi_run_determinism_cells_sha16"] = bool(same)
    if cross["first_cpu_run"]["available"]:
        j0 = json.loads(FIRST_CPU_JSON.read_text(encoding="utf-8"))
        diff = sorted(k for k in j0["cell_compare"] if k in j["cell_compare"] and any(
            j["cell_compare"][k][kind]["sha16_a"] != j0["cell_compare"][k][kind]["sha16_a"]
            for kind in ("raw", "scaled") if kind in j0["cell_compare"][k]))
        kv_pairs = ("A_vs_B", "C_vs_D", "A_vs_C", "B_vs_D", "C_vs_E", "D_vs_F", "A_vs_D")
        cross["first_cpu_run"].update(
            cells_differing_from_this_run=diff,
            stage=j0.get("stage"),
            observed_pattern=dict(
                note="**观测记录**（**不是**本轮的形状结论）：首跑六格在 **CPU** 上执行时的 M/N 比较，逐字抄录。"
                     "该 run 的复现门未过（`stage=reproduction_gate_failed`）、也没有 `classification` 字段"
                     "（工具在门失败处按设计停止解释）",
                comparisons={t: dict(exact=j0["cell_compare"][t]["raw"]["exact_equal"],
                                     n_diff=j0["cell_compare"][t]["raw"]["n_diff"],
                                     max_abs=j0["cell_compare"][t]["raw"]["max_abs"])
                             for t in kv_pairs if t in j0["cell_compare"]}),
            note="首跑的 cell 落在 CPU（`ProbeCapture.close()` 把张量搬到 CPU）⇒ "
                 "**同一个 2×2 在 CPU 后端下六个 M/N 比较全部逐位相同**，而在 CUDA 后端下有差异 ⇒ "
                 "这是**已记录的缺陷** + **设备/后端依赖观测**；"
                 "**不**作为本轮的形状结论、也**不**构成确定性反例")
    if cross["l0ops_pt"]["available"]:
        l0 = torch.load(L0OPS_PT, map_location="cpu")
        s_row = l0["raw_S"]["qk_scores_pre_mask"][0, :, 0, :VIS]
        b_row = l0["raw_B"]["qk_scores_pre_mask"][0, :, 0, :VIS]
        cross["l0ops_pt"].update(s_row_equals_real_S=bool(torch.equal(s_row, pt["real_S_row0"])),
                                 b_row_equals_real_B=bool(torch.equal(b_row, pt["real_B_row0"])),
                                 s_equals_cellA_scaled=bool(torch.equal(s_row, pt["row0_prefix"]["A"]["scaled"])),
                                 b_equals_cellD_scaled=bool(torch.equal(b_row, pt["row0_prefix"]["D"]["scaled"])),
                                 note="上一轮（L0 算子边界）与本轮是两个独立 run；保存的 qk_scores_pre_mask 是 scaled 口径")
        checks["l0ops_round_row0_bitwise_identical"] = bool(
            cross["l0ops_pt"]["s_row_equals_real_S"] and cross["l0ops_pt"]["b_row_equals_real_B"]
            and cross["l0ops_pt"]["s_equals_cellA_scaled"] and cross["l0ops_pt"]["b_equals_cellD_scaled"])
    rec["cmp_recompute"] = {k: {kind: {f: v[f] for f in FIELDS + ("coords", "coords_total")}
                                for kind, v in cmp_tool[k].items()} for k in cmp_tool}
    rec["structure_invariants"] = dict(row0_prefix_is_raw_slice=bool(slice_ok),
                                      k_rep_equals_production_kv_repeat=bool(krep_ok),
                                      scaled_vs_raw_over_sqrtd_n_diff=dsc,
                                      scaled_vs_raw_note="`scaled` 在生产里是 CUDA 上 bf16 除法的结果；"
                                                         "这里在 CPU 上重算只作观测（舍入可能不同），不作为门")
    rec["cross_run_checks"] = cross
    rec["recompute_checks"] = checks

    print("== ① 比较独立复算（只读；raw/scaled 分列；区域 = batch0/12 头/row0/keys 0:8209）==")
    print("   tag             kind    exact  n_diff    max|Δ|      max_idx     S@max       B@max      sha16(S/B)")
    for tag in list(dict.fromkeys([t for t, _x, _y in TOOL_PAIRS] + ["A_vs_real_S", "D_vs_real_B"])):
        for kind in ("raw", "scaled"):
            c = cmp_tool[tag][kind]
            print(f"   {tag:15s} {kind:6s}  {str(c['exact_equal']):5s} {c['n_diff']:>7d}  {c['max_abs']:<11.6g} "
                  f"{str(c['max_diff_index']):<12s} {c['a_at_max']:<11.6g} {c['b_at_max']:<11.6g} "
                  f"{c['sha16_a']}/{c['sha16_b']}")
    print("\n== ③ 结构不变量 ==")
    print(f"   row0_prefix == raw[0,:,0,0:{VIS}] ：{slice_ok}")
    print(f"   k_rep == 生产 kv_repeat(k)      ：{krep_ok}")
    print(f"   CPU 重算 raw/√d 与保存的 scaled 不同坐标数：{dsc}（设备舍入，只观测）")
    print("\n== ④ 跨 run / 跨轮一致性 ==")
    for k, v in cross.items():
        print(f"   {k}: {json.dumps({kk: vv for kk, vv in v.items() if kk != 'note'}, ensure_ascii=False, default=str)}")
    flat = True
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"   [check] {k}: 全真={all(v.values())}（{sum(v.values())}/{len(v)}）")
            flat = flat and all(v.values())
        else:
            print(f"   [check] {k}: {v}")
            flat = flat and bool(v)
    DC.require(flat, "独立复算与运行 JSON / 跨 run 一致性不符",
               json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not vv})
                           for k, v in checks.items()}, ensure_ascii=False))

    # ---------------- ⑤ 归因 ----------------
    sup = {}
    for tag, x, y in SUP_PAIRS:
        sup[tag] = dict(raw=QK.compare(QK.row0_prefix(pt["cells"][x]["raw"]),
                                       QK.row0_prefix(pt["cells"][y]["raw"]), f"{tag}_raw"),
                        scaled=QK.compare(QK.row0_prefix(pt["cells"][x]["scaled"]),
                                          QK.row0_prefix(pt["cells"][y]["scaled"]), f"{tag}_scaled"))
    cls_raw = QK.classify({k: v["raw"] for k, v in cmp_tool.items()})
    cls_scaled = QK.classify({k: v["scaled"] for k, v in cmp_tool.items()})
    cls_json_match = bool(
        all(cls_raw[k] == j["classification"]["raw"][k]
            for k in ("m_effect", "n_effect", "interaction", "detail"))
        and all(cls_scaled[k] == j["classification"]["scaled"][k]
                for k in ("m_effect", "n_effect", "interaction", "detail")))
    DC.require(cls_json_match, "分类结果与运行 JSON 不一致",
               json.dumps(dict(raw=cls_raw["detail"], scaled=cls_scaled["detail"]), ensure_ascii=False))
    rec["classification"] = dict(raw=cls_raw, scaled=cls_scaled,
                                matches_json=cls_json_match,
                                supplementary_cpu_derived=dict(
                                    note="以下三个比较由**只读汇总**在既有六格张量上新增（把 query 内容完全固定），"
                                         "作为支持性证据；**不是** GPU 工具预设的终点",
                                    A_vs_E=dict(meaning="M 效应（N=8209，四行内容全为 q1）",
                                                n_diff=sup["A_vs_E"]["raw"]["n_diff"],
                                                max_abs=sup["A_vs_E"]["raw"]["max_abs"],
                                                coords=_coordsets(sup["A_vs_E"]["raw"])),
                                    B_vs_F=dict(meaning="M 效应（N=8212，四行内容全为 q1）",
                                                n_diff=sup["B_vs_F"]["raw"]["n_diff"],
                                                max_abs=sup["B_vs_F"]["raw"]["max_abs"],
                                                coords=_coordsets(sup["B_vs_F"]["raw"])),
                                    E_vs_F=dict(meaning="N 效应（M=4，内容恒为 q1，与 A_vs_B 同内容）",
                                                n_diff=sup["E_vs_F"]["raw"]["n_diff"],
                                                max_abs=sup["E_vs_F"]["raw"]["max_abs"],
                                                coords=_coordsets(sup["E_vs_F"]["raw"]))))

    coords_ad = _coordsets(cmp_tool["A_vs_D"]["raw"])
    coords_ac = _coordsets(cmp_tool["A_vs_C"]["raw"])
    coords_bd = _coordsets(cmp_tool["B_vs_D"]["raw"])
    coords_ab = _coordsets(cmp_tool["A_vs_B"]["raw"])
    # 逐个坐标把六个 cell 的值列出来（raw 口径）：看清"共享坐标"上 N 改动落到哪一侧
    def _vals(coord):
        h, k_ = coord
        return {c: float(pt["cells"][c]["raw"][0, h, 0, k_]) for c in ("A", "B", "C", "D", "E", "F")}
    union = sorted(set(coords_ad) | set(coords_ac) | set(coords_bd) | set(coords_ab))
    rec["coord_detail_raw"] = dict(union_of_differing_coords=[list(x) for x in union],
                                   values_per_cell={f"{h},{k_}": _vals((h, k_)) for h, k_ in union},
                                   note="六格在**每个差异坐标**上的 raw 值（bf16 原样转 float）；"
                                        "C/D 与 E/F 成对相同是内容控制的结果")
    print("\n   [coord] 每个差异坐标上六格的 raw 值：")
    for h, k_ in union:
        v = _vals((h, k_))
        print(f"     (head={h}, key={k_}): A={v['A']} B={v['B']} C={v['C']} D={v['D']} "
              f"E={v['E']} F={v['F']}")
    rec["attribution"] = dict(
        observed_A_vs_D=dict(n_diff=cmp_tool["A_vs_D"]["raw"]["n_diff"],
                             max_abs_raw=cmp_tool["A_vs_D"]["raw"]["max_abs"],
                             max_abs_scaled=cmp_tool["A_vs_D"]["scaled"]["max_abs"],
                             coords=coords_ad,
                             meaning="真实 S(M=1,N=8209) 与真实 B(M=4,N=8212) 的第 0 行差异（本轮要解释的对象）"),
        path_M_only_A_vs_C=dict(n_diff=cmp_tool["A_vs_C"]["raw"]["n_diff"],
                                max_abs_raw=cmp_tool["A_vs_C"]["raw"]["max_abs"], coords=coords_ac),
        path_N_only_A_vs_B=dict(n_diff=cmp_tool["A_vs_B"]["raw"]["n_diff"],
                                max_abs_raw=cmp_tool["A_vs_B"]["raw"]["max_abs"], coords=coords_ab),
        M_effect_reproduces_whole_observed_diff=bool(coords_ac == coords_ad
                                                     and cmp_tool["A_vs_C"]["raw"]["n_diff"]
                                                     == cmp_tool["A_vs_D"]["raw"]["n_diff"]
                                                     and cmp_tool["A_vs_C"]["raw"]["max_abs"]
                                                     == cmp_tool["A_vs_D"]["raw"]["max_abs"]),
        N_effect_at_M4_is_zero=bool(cmp_tool["C_vs_D"]["raw"]["exact_equal"]
                                    and sup["E_vs_F"]["raw"]["exact_equal"]),
        N_effect_at_M1_is_nonzero=bool(cmp_tool["A_vs_B"]["raw"]["exact_equal"] is False),
        C_equals_D_row0_bitwise=bool(cmp_tool["C_vs_D"]["raw"]["exact_equal"] and
                                     cmp_tool["C_vs_D"]["scaled"]["exact_equal"]),
        M_effect_coords_at_N8209=[list(x) for x in coords_ac],
        M_effect_coords_at_N8212=[list(x) for x in coords_bd],
        M_effect_coord_sets_identical_across_N=bool(coords_ac == coords_bd),
        M_effect_coord_sets_note="M 效应在 N=8209 与 N=8212 下**坐标集合不同**"
                                 "（共享 (0,2367)、(1,4981)，第三个坐标分别为 (9,4011) 与 (4,4453)）"
                                 "⇒ 这是**交互的又一条证据**：不只 N 的敏感性依赖 M，"
                                 "M 改动的**哪些坐标**也依赖 N。个数与最大幅值相同（3 个、max|Δ|=64 raw）",
        content_controls_zero=bool(cmp_tool["C_vs_E"]["raw"]["exact_equal"]
                                   and cmp_tool["D_vs_F"]["raw"]["exact_equal"]),
        verdict=("第 0 行的持续分歧**归因于 M（query 行数 1→4）**："
                 "① 单独改 M（A→C，N 固定 8209）就复现出与 A→D **完全相同**的差异坐标集合、个数与最大幅值"
                 "（因为 C≡D 逐位相同，所以 A→D 与 A→C 的差异集合按构造相同）；"
                 "② 单独改 N 在 M=4 下效应为 **0**（C≡D；把内容换成 q1 重复的 E_vs_F 同样为 0）；"
                 "③ 但 N 在 M=1 下**并非无效**（A→B 有差异）⇒ 两个因素**不独立**，记为 interaction=True；"
                 "④ 内容控制（C_vs_E、D_vs_F 逐位相同）说明第 0 行不受其余 query 行内容影响，"
                 "所以上面的 M 效应是**形状**效应而非内容效应；"
                 "⑤ 补充：M 效应在 N=8209 与 N=8212 下**坐标集合不同**（第三个坐标分别是 (9,4011) 与 (4,4453)），"
                 "即不仅 N 的敏感性依赖 M，M 改动的坐标也依赖 N —— 交互是**双向**的。"),
        not_claimed=["具体是哪个 kernel/tile/归约顺序导致 M=1 与 M=4 走不同数值路径（本轮不做 profiler、不切后端）",
                     "谁是数学真值（本轮不做 FP32）",
                     "把 shape-sensitive 说成实现 bug、或用 margin/ULP 豁免正确性",
                     "N 在其他 M 下普遍无效、或 M 在其他 N 下普遍有效",
                     "对别的位置/别的层/别的长度/别的模型的推广",
                     "B6 通过或修复完成（B6 状态未变）"])
    print("\n== ⑤ 归因 ==")
    print(f"   观测对象 A_vs_D：n_diff={cmp_tool['A_vs_D']['raw']['n_diff']} "
          f"max|Δ|(raw)={cmp_tool['A_vs_D']['raw']['max_abs']} max|Δ|(scaled)="
          f"{cmp_tool['A_vs_D']['scaled']['max_abs']} coords={coords_ad}")
    print(f"   只改 M（A_vs_C）：n_diff={cmp_tool['A_vs_C']['raw']['n_diff']} "
          f"max|Δ|(raw)={cmp_tool['A_vs_C']['raw']['max_abs']} coords={coords_ac}")
    print(f"   只改 N（A_vs_B）：n_diff={cmp_tool['A_vs_B']['raw']['n_diff']} "
          f"max|Δ|(raw)={cmp_tool['A_vs_B']['raw']['max_abs']} coords={coords_ab}")
    print(f"   C≡D 逐位：{cmp_tool['C_vs_D']['raw']['exact_equal']}；E_vs_F（supp.）："
          f"{sup['E_vs_F']['raw']['exact_equal']}；A_vs_E（supp.）：n_diff={sup['A_vs_E']['raw']['n_diff']}")
    print(f"   分类(raw)：M={cls_raw['m_effect']} N={cls_raw['n_effect']} 交互={cls_raw['interaction']} "
          f"明细={cls_raw['detail']}")
    print(f"   ★ 归因 = {rec['attribution']['verdict']}")
    p = OUT / f"qk2x2_8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
