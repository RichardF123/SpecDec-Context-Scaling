"""norm8209_verdict_cpu.py — 位置 8209 的**最终 Norm 边界**只读汇总（纯 CPU、不加载模型、不写产物到 output 之外）。

## 为什么需要
`div8209_norm_20261002T080941Z.json` 里的**原始比较**（`input_compare` / `norm_vector_compare` /
`projection_compare` / `norm_cells_controls`）都是无歧义的；但它的**派生摘要**
`verdict.norm_shape_effect_on_vector.{fixed_xS,fixed_xB}` 存的是 `elementwise_equal`
（`True` = 两格**逐位相同** = **未观察到**形状效应）—— 键名容易被读反（代码里已改名）。
本脚本从**原始张量**（`.pt`，原始 bf16）**独立重算**四格的决策向量与固定形状投影，
用**无歧义字段名**给出结论，并核对与 JSON 中的原始比较一致。

## 输入 / 输出
* 输入：`--json` / `--pt`（默认本轮的完整 run）；只读。
* 输出：`output/g5/attn/norm8209_verdict_cpu_<ts>.json` + 屏幕表格。
* **不**改任何既有产物；**不**做任何前向、**不**做 FP32 投影（只用原 bf16 表达式复算）。
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
DEF_JSON = OUT / "div8209_norm_20261002T080941Z.json"
DEF_PT = OUT / "div8209_norm_20261002T080941Z.pt"


def _cmp(a, b, name: str) -> dict:
    out = DC.cmp_tensor(a.float().reshape(-1), b.float().reshape(-1), name)
    out["n_diff"] = int((a.reshape(-1) != b.reshape(-1)).sum())
    out["argmax_same"] = bool(int(a.reshape(-1).argmax()) == int(b.reshape(-1).argmax()))
    out["argmax_a"] = int(a.reshape(-1).argmax())
    out["argmax_b"] = int(b.reshape(-1).argmax())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEF_JSON))
    ap.add_argument("--pt", default=str(DEF_PT))
    ap.add_argument("--lmh-pt", default=str(OUT / "div8209_lmh_20261002T071235Z.pt"))
    a = ap.parse_args()
    jp, tp = Path(a.json), Path(a.pt)
    rec: dict = dict(tool="norm8209_verdict_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从原始 bf16 张量独立重算四格决策向量与固定形状投影，给无歧义结论；"
                               "不做前向、不做 FP32 投影、不改任何既有产物",
                     sources=dict(json=str(jp), pt=str(tp), lmh_pt=str(a.lmh_pt)))
    j = json.loads(jp.read_text(encoding="utf-8"))
    pt = torch.load(tp, map_location="cpu")
    W = json.loads(jp.read_text(encoding="utf-8"))["projection"]["rows"]        # 只用于核对 sha
    dv = {k: v for k, v in pt["norm_dv_raw"].items()}
    xS, xB4 = pt["xS_raw"], pt["xB4_raw"]
    hS, hB4 = pt["hS_raw"], pt["hB4_raw"]
    rec["raw_dtypes"] = dict(xS=str(xS.dtype), xB4=str(xB4.dtype), hS=str(hS.dtype),
                             hB4=str(hB4.dtype), dv={k: str(v.dtype) for k, v in dv.items()},
                             note="`.pt` 里的 `*_raw` 是**原始 bf16**；本脚本只在比较时转 FP32")

    # ---- 1) Norm 输入比较 ----
    cmp_in = _cmp(xS[0, 0], xB4[0, 0], "xS_row0_vs_xB_row0")
    cmp_h = _cmp(hS[0, 0], hB4[0, 0], "hS_row0_vs_hB_row0")
    rec["input_compare"] = dict(
        xS_vs_xB=cmp_in, hS_vs_hB=cmp_h,
        verdict_input_already_differs=bool(not cmp_in["elementwise_equal"]),
        note="`xS≠xB` 只证明差异在 Norm **之前**已存在；**不足以**证明它驱动翻转（需看 §2 的两向对照）")
    print("== 1) Norm 输入/输出比较（原始 bf16，比较时转 FP32）==")
    for k, c in (("xS vs xB", cmp_in), ("hS vs hB", cmp_h)):
        print(f"  {k:<10} equal={c['elementwise_equal']} n_diff={c['n_diff']}/{c['shape'][0]} "
              f"max|Δ|={c['abs_err_max']:.6g} mean|Δ|={c['abs_err_mean']:.6g}")

    # ---- 2) Norm 四格（向量级）----
    v_single_in = _cmp(dv["NA"][0, 0], dv["NB"][0, 0], "NA_vs_NB")
    v_four_in = _cmp(dv["NC"][0, 0], dv["ND"][0, 0], "NC_vs_ND")
    v_xS_shape = _cmp(dv["NA"][0, 0], dv["NC"][0, 0], "NA_vs_NC")
    v_xB_shape = _cmp(dv["NB"][0, 0], dv["ND"][0, 0], "NB_vs_ND")
    rec["norm_vector_compare"] = dict(
        fixed_single_row_change_input=v_single_in, fixed_four_row_change_input=v_four_in,
        fixed_input_xS_change_shape=v_xS_shape, fixed_input_xB_change_shape=v_xB_shape,
        shape_change_is_bitwise_inert_on_row0=bool(v_xS_shape["elementwise_equal"]
                                                   and v_xB_shape["elementwise_equal"]))
    print("\n== 2) Norm 四格（决策向量，向量级）==")
    print(f"  固定单行形状、换输入：equal={v_single_in['elementwise_equal']} n_diff={v_single_in['n_diff']} "
          f"max|Δ|={v_single_in['abs_err_max']:.6g}")
    print(f"  固定四行形状、换输入：equal={v_four_in['elementwise_equal']} n_diff={v_four_in['n_diff']} "
          f"max|Δ|={v_four_in['abs_err_max']:.6g}")
    print(f"  固定输入 xS、改形状：equal={v_xS_shape['elementwise_equal']} n_diff={v_xS_shape['n_diff']} "
          f"max|Δ|={v_xS_shape['abs_err_max']:.6g}")
    print(f"  固定输入 xB、改形状：equal={v_xB_shape['elementwise_equal']} n_diff={v_xB_shape['n_diff']} "
          f"max|Δ|={v_xB_shape['abs_err_max']:.6g}")

    # ---- 3) 与 JSON 的原始比较逐项核对（独立重算 vs 原始记录）----
    # ⚠️ 口径必须对齐：JSON 的 `projection.rows[k].sha16` 是**投影后的 logits 行**（151936 维）的摘要，
    #    不是决策向量（1536 维）的摘要。首版把两者混比 ⇒ 4 项 sha 核对全为 False（自己写错，不是数据问题）。
    proj_raw = {k: v for k, v in pt["proj_rows_raw"].items()}
    jj = json.loads(jp.read_text(encoding="utf-8"))
    checks = dict(
        xS_vs_xB_matches_json=bool(jj["input_compare"]["xS_vs_xB"]["n_diff"] == cmp_in["n_diff"]),
        hS_vs_hB_matches_json=bool(jj["input_compare"]["hS_vs_hB"]["n_diff"] == cmp_h["n_diff"]),
        proj_sha_matches_record={k: bool(A.stable_sha(proj_raw[k]) == jj["projection"]["rows"][k]["sha16"])
                                 for k in ("NA", "NB", "NC", "ND")},
        proj_argmax_matches_record={k: bool(int(proj_raw[k].float().argmax())
                                            == jj["projection"]["rows"][k]["argmax_id"])
                                    for k in ("NA", "NB", "NC", "ND")},
        dv_NC_identical_to_NA=bool(torch.equal(dv["NC"], dv["NA"])),
        dv_ND_identical_to_NB=bool(torch.equal(dv["ND"], dv["NB"])),
        proj_NC_identical_to_NA=bool(torch.equal(proj_raw["NC"], proj_raw["NA"])),
        proj_ND_identical_to_NB=bool(torch.equal(proj_raw["ND"], proj_raw["NB"])),
    )
    rec["recompute_checks"] = checks
    print("\n== 3) 与原始 JSON 的核对（独立重算）==")
    flat_ok = True
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
            flat_ok = flat_ok and all(v.values())
        else:
            print(f"  {k}: {v}")
            flat_ok = flat_ok and bool(v)
    DC.require(flat_ok, "独立重算与原始记录不一致", str(checks))

    # ---- 4) 汇总结论（无歧义字段名）----
    rec["verdict"] = dict(
        norm_input_already_differs=bool(not cmp_in["elementwise_equal"]),
        norm_shape_change_is_bitwise_inert=dict(
            on_decision_vector=bool(v_xS_shape["elementwise_equal"] and v_xB_shape["elementwise_equal"]),
            n_diff=[v_xS_shape["n_diff"], v_xB_shape["n_diff"]],
            note="固定输入、只把 Norm 的行数 1↔4 改变：决策向量**逐位相同** ⇒ 本例**未观察到** Norm 形状效应"),
        input_difference_remains_after_norm=bool(not cmp_h["elementwise_equal"]),
        meaning=("① Norm 的**输入**已经不同（`xS≠xB`）；② Norm 的**执行形状**在本例对决策向量"
                 "**逐位无影响**；③ 因此按规则：固定形状换输入会翻转、固定输入换形状不翻转 ⇒ "
                 "支持**输入差异驱动**，下一步转向 Decoder 堆栈（**本轮不扫层**）。"),
        not_claimed=["323/659 谁是数学真值（未做 FP32）",
                     "差异由哪个上游算子产生（需上移边界）",
                     "把'执行路径相关的数值差异'等同于'必然只是 BF16 舍入'",
                     "对别的位置/长度/历史的推广"])
    print("\n== 4) 结论（无歧义字段名）==")
    for k, v in rec["verdict"].items():
        print(f"  {k}: {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}")
    p = OUT / f"norm8209_verdict_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[verdict] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
