"""layer8209_verdict_cpu.py — 位置 8209 的 **L27 边界**只读汇总（纯 CPU、不加载模型、不改既有产物）。

## 为什么需要
GPU run 的 JSON 里原始比较字段已经无歧义，但按本项目的惯例，**结论要能从保存的原始张量独立复算**：
本脚本只用 `.pt` 里的 **bf16 原始张量**重算
① L27 输入比较（`aS` vs `aB`）、② L27 输出比较（`zS` vs `zB4`）、
③ 四格第 0 行向量的两向对照、④ 四格尾部的判定行（argmax/margin/并列数/摘要）与两向对照，
并与 JSON 的对应字段逐项核对；再核对与**上一轮单行参考**（Norm 四格 `P_NA/P_ND`、LM head `HS_1/HB_1`）的逐位一致。

## 输入 / 输出
* 输入：`--json` / `--pt`（默认本轮的完整 L27 run）；只读。
* 输出：`output/g5/attn/layer8209_verdict_cpu_<ts>.json` + 屏幕表格。
* **不**改任何既有产物；**不**做前向；**不**做 FP32（只用原 bf16 张量做统计/比较）。
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

OUT = ROOT / "output" / "g5" / "attn"
DEF_JSON = OUT / "div8209_l27_20261002T161807Z.json"
DEF_PT = OUT / "div8209_l27_20261002T161807Z.pt"
NORM_PT = OUT / "div8209_norm_20261002T080941Z.pt"
LMH_PT = OUT / "div8209_lmh_20261002T071235Z.pt"


def _cmp(a, b, name: str) -> dict:
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(x, y, name)
    d = (x - y).abs()
    return dict(name=name, n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                elementwise_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                argmax_a=int(x.argmax()), argmax_b=int(y.argmax()),
                argmax_same=bool(int(x.argmax()) == int(y.argmax())))


def _flips(c: dict) -> bool:
    return bool(c["elementwise_equal"] is False and c["argmax_same"] is False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    rec: dict = dict(tool="layer8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始 bf16 张量独立重算 L27 边界的四格与两向对照，"
                               "并与 JSON 记录逐项核对；不做前向、不做 FP32、不改任何既有产物",
                     sources=dict(json=str(jp), pt=str(tp), norm_pt=str(NORM_PT), lmh_pt=str(LMH_PT)))
    j = json.loads(jp.read_text(encoding="utf-8"))
    pt = torch.load(tp, map_location="cpu")
    aS, aB4 = pt["aS_raw"], pt["aB4_raw"]
    zS, zB4 = pt["zS_raw"], pt["zB4_raw"]
    cells = {k: v for k, v in pt["cell_out_raw"].items()}
    rows = {k: v for k, v in pt["tail_rows_raw"].items()}
    rec["raw_dtypes"] = dict(aS=str(aS.dtype), aB4=str(aB4.dtype), zS=str(zS.dtype), zB4=str(zB4.dtype),
                             cells={k: str(v.dtype) for k, v in cells.items()},
                             tail_rows={k: str(v.dtype) for k, v in rows.items()},
                             note="`.pt` 里 `*_raw` 是**原始 bf16**；统计/比较时才转 FP32")

    # ---- 1) L27 输入/输出比较 ----
    c_in = _cmp(aS[0, 0], aB4[0, 0], "aS_row0_vs_aB_row0")
    c_out = _cmp(zS[0, 0], zB4[0, 0], "zS_row0_vs_zB4_row0")
    rec["boundary_compare"] = dict(l27_input=c_in, l27_output=c_out,
                                   l27_input_already_differs=bool(not c_in["elementwise_equal"]),
                                   note="`aS≠aB` 只证明上游已有差异；是否驱动翻转看 §2 的两向对照")
    print("== 1) L27 输入/输出（原始 bf16，比较时转 FP32）==")
    for k, c in (("aS vs aB（层输入）", c_in), ("zS vs zB4（层输出）", c_out)):
        print(f"  {k:<22} equal={c['elementwise_equal']} n_diff={c['n_diff']}/{c['shape'][0] if 'shape' in c else aS.shape[-1]} "
              f"max|Δ|={c['abs_err_max']:.6g} mean|Δ|={c['abs_err_mean']:.6g}")

    # ---- 2) L27 四格（第 0 行向量）----
    v_single = _cmp(cells["LA"][0, 0], cells["LB"][0, 0], "LA_vs_LB")
    v_four = _cmp(cells["LC"][0, 0], cells["LD"][0, 0], "LC_vs_LD")
    v_sh_aS = _cmp(cells["LA"][0, 0], cells["LC"][0, 0], "LA_vs_LC")
    v_sh_aB = _cmp(cells["LB"][0, 0], cells["LD"][0, 0], "LB_vs_LD")
    rec["cells_vector_compare"] = dict(fixed_single_row_change_input=v_single,
                                       fixed_four_row_change_input=v_four,
                                       fixed_input_aS_change_shape=v_sh_aS,
                                       fixed_input_aB_change_shape=v_sh_aB)
    print("\n== 2) L27 四格（第 0 行向量）==")
    for k, c in (("固定单行、换输入", v_single), ("固定四行、换输入", v_four),
                 ("固定输入 aS、改形状", v_sh_aS), ("固定输入 aB、改形状", v_sh_aB)):
        print(f"  {k:<20} equal={c['elementwise_equal']} n_diff={c['n_diff']} "
              f"max|Δ|={c['abs_err_max']:.6g}")

    # ---- 3) 尾部判定行（重算两向对照 + 与 JSON 核对）----
    p_single = _cmp(rows["LA"], rows["LB"], "P_LA_vs_P_LB")
    p_four = _cmp(rows["LC"], rows["LD"], "P_LC_vs_P_LD")
    p_sh_aS = _cmp(rows["LA"], rows["LC"], "P_LA_vs_P_LC")
    p_sh_aB = _cmp(rows["LB"], rows["LD"], "P_LB_vs_P_LD")
    rec["tail_compare"] = dict(fixed_single_row_change_input=p_single,
                               fixed_four_row_change_input=p_four,
                               fixed_input_aS_change_shape=p_sh_aS,
                               fixed_input_aB_change_shape=p_sh_aB)
    print("\n== 3) 尾部判定行（同一 Norm + 同一绑定权重，完整词表）==")
    for k, c in (("固定单行、换输入", p_single), ("固定四行、换输入", p_four),
                 ("固定输入 aS、改形状", p_sh_aS), ("固定输入 aB、改形状", p_sh_aB)):
        print(f"  {k:<20} same_argmax={c['argmax_same']} elementwise={c['elementwise_equal']} "
              f"n_diff={c['n_diff']} max|Δ|={c['abs_err_max']:.6g} ({c['argmax_a']} vs {c['argmax_b']})")

    # ---- 4) 与 JSON 记录 / 上一轮单行参考核对 ----
    checks = dict(
        json_tail_rows_match={k: bool(A.stable_sha(rows[k]) == j["tail"]["norm_rows"][k]["sha16"]
                                      and int(rows[k].float().argmax()) == j["tail"]["norm_rows"][k]["argmax_id"])
                              for k in ("LA", "LB", "LC", "LD")},
    )
    if Path(NORM_PT).exists():
        pr = torch.load(NORM_PT, map_location="cpu").get("proj_rows_raw", {})
        checks["P_LA_equals_prev_norm_P_NA"] = bool(torch.equal(rows["LA"], pr["NA"]))
        checks["P_LD_equals_prev_norm_P_ND"] = bool(torch.equal(rows["LD"], pr["ND"]))
    if Path(LMH_PT).exists():
        pr2 = torch.load(LMH_PT, map_location="cpu").get("rows", {})
        checks["P_LA_equals_prev_lmh_HS_1"] = bool(torch.equal(rows["LA"], pr2["HS_1"]))
        checks["P_LD_equals_prev_lmh_HB_1"] = bool(torch.equal(rows["LD"], pr2["HB_1"]))
        checks["P_LD_vs_prev_lmh_HB_4_n_diff"] = int((rows["LD"] != pr2["HB_4"]).sum())
    rec["recompute_checks"] = checks
    print("\n== 4) 与 JSON / 上一轮单行参考核对 ==")
    ok_all = True
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
            ok_all = ok_all and all(v.values())
        else:
            print(f"  {k}: {v}")
            ok_all = ok_all and (bool(v) if isinstance(v, bool) else True)
    DC.require(ok_all, "独立重算与原始记录/上一轮参考不一致", str(checks))

    # ---- 5) 结论（无歧义字段名）----
    rec["verdict"] = dict(
        l27_input_already_differs=bool(not c_in["elementwise_equal"]),
        l27_shape_change_changes_row0=dict(
            differs=bool(not (v_sh_aS["elementwise_equal"] and v_sh_aB["elementwise_equal"])),
            n_diff=[v_sh_aS["n_diff"], v_sh_aB["n_diff"]],
            max_abs=[v_sh_aS["abs_err_max"], v_sh_aB["abs_err_max"]]),
        fixed_shape_change_input_flips=dict(single_row=_flips(p_single), four_row=_flips(p_four)),
        fixed_input_change_shape_flips=dict(aS=_flips(p_sh_aS), aB=_flips(p_sh_aB)),
        meaning=("① L27 的**输入已经不同**（`aS≠aB`）；② 只改 L27 行数会**改变第 0 行向量**"
                 "（334/175 个坐标、max 0.5/0.125）与最终 logits 行（8.4e4–14.1e4 个坐标、max 0.125），"
                 "但**没有翻转**本例的 323/659；③ 固定形状换输入在**两种形状下都翻转** ⇒ "
                 "支持**输入来源效应**：L27 在本例中主要**传递**上游差异。"),
        not_claimed=["323/659 谁是数学真值（未做 FP32）",
                     "差异由哪个上游算子首次产生（需再上移边界；不假定逐层单调、不用二分排除更早层）",
                     "把数值差异直接判为 bug", "用 margin 豁免正确性", "B6 通过或修复完成",
                     "对别的位置/长度/历史的推广"])
    print("\n== 5) 结论（无歧义字段名）==")
    for k, v in rec["verdict"].items():
        print(f"  {k}: {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}")
    p = OUT / f"layer8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
