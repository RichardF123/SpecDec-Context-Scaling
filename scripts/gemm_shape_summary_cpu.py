"""gemm_shape_summary_cpu.py — 形状敏感性机制的**只读**汇总（纯 CPU、不加载模型、不改产物）。

## 作用
从 `div8209_gemm_*.json` + `.pt` 与 `div8209_backend_*.json` 独立整理：
① Step 0 复现（从 `.pt` 的 S/B 决策行**重算** argmax 与 323/659 的 logit）；
② 五类算子的 `M=1 vs M=4` 矩阵（`n_diff`/`max|Δ|`/第 0 行是否逐位一致）+ profiler 可用性结论；
③ layout/content 控制结论；
④ 精度/确定性旋钮结论；
⑤ 候选对照（L0 决策行 token / 是否闭合）；
⑥ 跨层最小矩阵（是否重复同类形状效应）；
⑦ attention 后端观测。
同时写出机器可读的 **JSONL**（每个 (层, 算子, 变体) 一行）。

**不**加载模型、**不**做前向、**不**改任何产物。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"


def _newest(pat: str):
    fs = sorted(glob.glob(str(OUT / pat)), key=lambda p: Path(p).stat().st_mtime)
    return Path(fs[-1]) if fs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--pt", default=None)
    a = ap.parse_args()
    jp = Path(a.json) if a.json else _newest("div8209_gemm_*.json")
    pt_p = Path(a.pt) if a.pt else jp.with_suffix(".pt")
    bk_p = _newest("div8209_backend_*.json")
    j = json.loads(jp.read_text(encoding="utf-8"))
    # ⚠️ torch 2.6 的 `torch.load` 默认 `weights_only=True`；本 .pt 里存了 `TorchVersion`（版本快照）
    #   ⇒ 需要显式 `weights_only=False`（产物由本仓库自己写入，来源可信）。
    pt = torch.load(pt_p, map_location="cpu", weights_only=False) if pt_p.exists() else {}
    bk = json.loads(bk_p.read_text(encoding="utf-8")) if bk_p and bk_p.exists() else {}
    rec: dict = dict(tool="gemm_shape_summary_cpu", run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：重算 Step 0 的 token/logit、整理五算子矩阵/控制/旋钮/候选/跨层/后端；不加载模型",
                     sources=dict(probe_json=jp.name, probe_pt=pt_p.name, backend_json=(bk_p.name if bk_p else None)),
                     versions=j.get("versions"), stage=j.get("stage"))
    checks: dict = {}
    # ① Step 0（从 .pt 重算）
    if pt:
        S, B = pt["S_final_row"].float(), pt["B_final_row"].float()
        s0 = dict(S_argmax=int(S.argmax()), S_logit_323=float(S[323]), S_logit_659=float(S[659]),
                  B_argmax=int(B.argmax()), B_logit_323=float(B[323]), B_logit_659=float(B[659]),
                  S_323_equals_659=bool(float(S[323]) == float(S[659])),
                  B_659_gt_323=bool(float(B[659]) > float(B[323])))
        rec["step0"] = s0
        checks["step0_reproduced"] = bool(s0["S_argmax"] == 323 and s0["B_argmax"] == 659
                                          and s0["S_323_equals_659"] and s0["B_659_gt_323"])
    # ② 五算子矩阵
    rows = []
    m4 = j["step1_op_matrix_L0"]
    for op, d in m4.items():
        ck = d.get("M4row0_vs_M1_sameN") or d.get("M4row0_vs_M1")
        ckc = d.get("M4row0_vs_M1_causalprefix") or d.get("M4row0_vs_M1contig")
        kk = (d.get("profile_all") or {}).get("cuda_kernels") or []
        kr = (d.get("profile_row") or {}).get("cuda_kernels") or []
        rows.append(dict(layer=0, op=op, variant="M4_row0_vs_M1", n_diff=ck["n_diff"],
                         max_abs=ck["max_abs"], exact=ck["exact_equal"],
                         variant2="M4_row0_vs_M1_causalprefix/contig",
                         n_diff2=(ckc or {}).get("n_diff"), max_abs2=(ckc or {}).get("max_abs"),
                         M1_eq_M1contig=(d.get("M1_vs_M1contig") or d.get("M1_sameN_vs_M1prefix") or {}
                                         ).get("exact_equal"),
                         kernels_available=bool(kk or kr),
                         aten_ops=[x["op"] for x in ((d.get("profile_all") or {}).get("aten_ops") or [])[:3]]))
    rec["op_matrix_L0"] = rows
    # ③ 控制
    ctl = j["step2_controls_L0"]
    rec["controls_L0"] = {k: dict(all_rows_same_vs_M1=v["all_rows_same_vs_M1"]["n_diff"],
                                 other_rows_perturbed_vs_M4row0=v["other_rows_perturbed_vs_M4row0"]["n_diff"],
                                 other_rows_perturbed_vs_M1=v["other_rows_perturbed_vs_M1"]["n_diff"],
                                 row_view_vs_contig=(v.get("row_view_vs_contig")
                                                     or v.get("q_row_view_vs_contig") or {}).get("n_diff"),
                                 stride_variant_n_diff=(v.get("k_stride_variant", {}).get("vs_M1")
                                                        or v.get("stride_variant", {}).get("vs_M4row0") or {}
                                                        ).get("n_diff"))
                          for k, v in ctl.items()}
    # ④ 旋钮
    rec["knobs"] = {k: dict(gate_n_diff=(v.get("gate_proj") or {}).get("n_diff"),
                            qk_n_diff=(v.get("QK") or {}).get("n_diff"),
                            state=(v.get("state") or {}), error=v.get("error"))
                    for k, v in j["step3_knobs"].items()}
    # ⑤ 候选
    c = j["step3_candidates_L0"]
    rec["candidates_L0"] = dict(M4_block=c["m4_block"], five_ops_row=c["five_ops_row"],
                                all_linears_row=c["all_linears_row"], s_step=c["s_step"])
    checks["five_ops_row_token_equals_S"] = bool(c["five_ops_row"]["token"] == c["s_step"]["argmax"])
    # ⑥ 跨层
    cross = j["step4_cross_layer"]
    rec["cross_layer"] = {k: dict(input_n_diff=v["input"]["n_diff"], output_n_diff=v["output"]["n_diff"],
                                  boundaries=v["boundaries"], minimal_M1_vs_M4=v["minimal_M1_vs_M4"])
                          for k, v in cross.items()}
    all_five_positive = all(all(v["n_diff"] > 0 for v in d["minimal_M1_vs_M4"].values())
                            for d in rec["cross_layer"].values())
    checks["same_mechanism_all_layers"] = bool(all_five_positive)
    checks["profiler_kernels_available"] = bool(any(r["kernels_available"] for r in rows))
    # ⑦ 后端
    if bk:
        rec["backends"] = {k: dict(S=(v.get("S") or {}).get("argmax"), B=(v.get("B") or {}).get("argmax"),
                                   tokens_match=v.get("tokens_match"),
                                   L0out_n_diff=(v.get("L0_output") or {}).get("n_diff"),
                                   fused_by_phase=((v.get("attention_stats") or {}).get("fused_by_phase")),
                                   eager_by_phase=((v.get("attention_stats") or {}).get("eager_by_phase")),
                                   error=v.get("error"))
                           for k, v in bk["backends"].items()}
    rec["checks"] = checks
    # ---- JSONL ----
    jl = OUT / f"gemm_shape_summary_{rec['run_utc']}.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dict(kind="op_matrix_L0", **r), ensure_ascii=False) + "\n")
        for lk, v in rec["cross_layer"].items():
            for op, d in v["minimal_M1_vs_M4"].items():
                fh.write(json.dumps(dict(kind="cross_layer_minimal", layer=lk, op=op, **d),
                                    ensure_ascii=False) + "\n")
        for k, v in rec["knobs"].items():
            fh.write(json.dumps(dict(kind="knob", name=k, **v), ensure_ascii=False) + "\n")
    rec["jsonl"] = str(jl)
    print("== ① Step 0（从 .pt 重算）==")
    if "step0" in rec:
        s = rec["step0"]
        print(f"   S argmax={s['S_argmax']}（323 的 logit={s['S_logit_323']}、659={s['S_logit_659']}，"
              f"并列={s['S_323_equals_659']}）")
        print(f"   B argmax={s['B_argmax']}（323={s['B_logit_323']}、659={s['B_logit_659']}，"
              f"659>323={s['B_659_gt_323']}）")
    print("\n== ② 五类算子（L0，M=4 的第 0 行 vs 独立 M=1）==")
    for r in rows:
        print(f"   {r['op']:<10} n_diff={r['n_diff']:>4d} max|Δ|={r['max_abs']:<11.6g} "
              f"（等价变体 n_diff={r['n_diff2']}）| kernel 可用={r['kernels_available']} "
              f"| ATen={r['aten_ops']}")
    print("\n== ③ content/layout 控制（L0）==")
    for k, v in rec["controls_L0"].items():
        print(f"   {k:<10} 四行同内容 vs M1={v['all_rows_same_vs_M1']}、"
              f"改其他行后第0行 vs 真实M4={v['other_rows_perturbed_vs_M4row0']}、"
              f"第0行 view vs contig={v['row_view_vs_contig']}、stride 变体={v['stride_variant_n_diff']}")
    print("\n== ④ 旋钮 ==")
    for k, v in rec["knobs"].items():
        print(f"   {k:<24} gate n_diff={v['gate_n_diff']} QK n_diff={v['qk_n_diff']} "
              f"{v['state'] or (v['error'] or '')[:60]}")
    print("\n== ⑤ 候选（L0 决策行）==")
    cc = rec["candidates_L0"]
    print(f"   单步 S={cc['s_step']['argmax']}；M4 块={cc['M4_block']['token']}；"
          f"五算子逐行={cc['five_ops_row']['token']}；全部 Linear 逐行={cc['all_linears_row']['token']}")
    checks["five_ops_row_token_equals_S"] = bool(cc["five_ops_row"]["token"] == cc["s_step"]["argmax"])
    checks["all_linears_no_extra_benefit_at_L0"] = bool(cc["all_linears_row"]["token"]
                                                       == cc["five_ops_row"]["token"])
    print("\n== ⑥ 跨层 ==")
    for k, v in rec["cross_layer"].items():
        print(f"   {k}: 输入 n_diff={v['input_n_diff']}、输出 n_diff={v['output_n_diff']}、"
              f"边界={v['boundaries']}、最小 M1vsM4={ {o: d['n_diff'] for o, d in v['minimal_M1_vs_M4'].items()} }")
    if "backends" in rec:
        print("\n== ⑦ attention 后端观测 ==")
        for k, v in rec["backends"].items():
            print(f"   {k:<14} S={v['S']} B={v['B']} 一致={v['tokens_match']} "
                  f"L0out n_diff={v['L0out_n_diff']} fused={v['fused_by_phase']} "
                  f"eager={v['eager_by_phase']} {v['error'] or ''}")
    print("\n== 核对 ==")
    for k, v in checks.items():
        print(f"   [check] {k}: {v}")
    p = OUT / f"gemm_shape_summary_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[summary] RESULT_JSON={p}")
    print(f"[summary] RESULT_JSONL={jl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
