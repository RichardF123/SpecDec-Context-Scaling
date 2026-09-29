"""analyze_f06_index25_cpu.py — F06/index25 取证结果的**纯 CPU** 汇总（不加载模型）。

回答任务 §3/§4 的问题：
  * 首个分歧是否精确复现（index / 两侧 token / 共同前缀）；
  * 实际决策行来自哪次调用、哪一行、预测哪个位置；
  * 两侧（AR 单步 vs SpecDec block 行）在**同一输入 token、同一输入位置**下的 logits 差异；
  * cache 对齐核验（长度、逐层、以及共同前缀位置是否逐位一致）。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import torch

from . import diag_cache_cmp as CC
from . import diag_fields as DF

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
DIV_INDEX = 25
EXPECT_AR, EXPECT_SPEC = 715, 576


def _newest(pattern: str) -> Path:
    return Path(sorted(glob.glob(str(OUTDIR / pattern)))[-1])


def _j(mode):
    return json.loads(_newest(f"{TAG}_{FAMILY}_{mode}_*.json").read_text(encoding="utf-8"))


def _rows(mode):
    return torch.load(_newest(f"{TAG}_{FAMILY}_{mode}_*.rows.pt"), map_location="cpu")


def _cell(row: torch.Tensor) -> dict:
    r = row.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=[EXPECT_AR, EXPECT_SPEC])
    t2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(t2[0] - t2[1])
    f["tie_at_top"] = bool(float(t2[0]) == float(t2[1]))
    return f


def main() -> int:
    a = _j("ar")
    b = _j("spec")
    ra = _rows("ar")
    ar_real = ra["real"]
    dec = b["decision"]
    w = [x for x in b["window_rounds"] if x["round"] == dec["deciding_round"]][0]
    spec_rows_all = _rows("spec")                          # {roundN_rows: tensor(Kp,V)}
    spec_row = spec_rows_all[f"round{w['round']}_rows"][dec["deciding_slot"] - 1]

    out = dict(tool="analyze_f06_index25_cpu", tag=TAG,
               repro=dict(
                   ar_argmax=a["ar"]["real_argmax"], ar_expected=EXPECT_AR,
                   ar_match=(a["ar"]["real_argmax"] == EXPECT_AR),
                   spec_emitted=dec["emitted_token"], spec_expected=EXPECT_SPEC,
                   spec_match=(dec["emitted_token"] == EXPECT_SPEC),
                   ar_common_prefix=a["ar"]["gen_head"][:DIV_INDEX],
               ),
               decision=dict(
                   side="spec", round=dec["deciding_round"], slot=dec["deciding_slot"],
                   source_kind=dec["source_kind"], source_detail=dec["source_detail"],
                   input_position=dec["input_position"], predicted_position=dec["predicted_position"],
                   fed_token_at_input_position=int(dec["cands"][dec["deciding_slot"] - 1]),
                   round_Kp=dec["Kp"], round_m=dec["m"], round_cands=dec["cands"],
                   round_g=dec["g"], round_emitted=dec["emitted"],
                   cache_len_before_verify=dec["pre"],
                   note="block 的第 j 行：输入位置 pre+j、预测 pre+j+1；本行 j=slot-1"),
               ar_decision=dict(side="ar", source_kind="decode_single_step",
                                input_position=a["ar"]["fed_token_position"],
                                predicted_position=a["ar"]["predicted_position"],
                                fed_token=a["ar"]["fed_token"],
                                cache_len_before=a["ar"]["cache_len_before"]),
               )
    # ---- 同一输入 token / 同一输入位置的 logits 对照 ----
    ba, sb = _cell(ar_real), _cell(spec_row)
    cmp_ = dict(DF.compare_rows(ar_real, spec_row))
    cmp_.update(ar=ba, spec=sb,
                note="两侧输入 token 与输入位置相同（见 alignment），但 cache 长度与计算形状不同")
    out["logits_compare_same_token_same_position"] = cmp_
    # ---- 关键候选的 logits ----
    out["key_candidate_logits"] = {
        str(t): dict(ar=float(ar_real[t]), spec=float(spec_row[t]),
                     delta=float(ar_real[t] - spec_row[t])) for t in (715, 576, 4710, 4882)}
    # ---- 对齐核验 ----
    out["alignment"] = dict(
        same_input_token=(int(a["ar"]["fed_token"]) == int(dec["cands"][dec["deciding_slot"] - 1])),
        ar_input_token=int(a["ar"]["fed_token"]),
        spec_input_token=int(dec["cands"][dec["deciding_slot"] - 1]),
        same_input_position=(int(a["ar"]["fed_token_position"]) == int(dec["input_position"])),
        ar_input_position=int(a["ar"]["fed_token_position"]),
        spec_input_position=int(dec["input_position"]),
        ar_cache_len_before=int(a["ar"]["cache_len_before"]),
        spec_cache_len_before=int(dec["pre"]),
        ar_shape="s=1, T=8217（无 mask 应用）",
        spec_shape=f"s={dec['Kp']}, T={dec['pre'] + dec['Kp']}（triu diagonal={dec['pre'] + 1}）",
        ar_mask=a["ar"]["mask"], spec_mask=w["mask"],
        ar_snapshot_integrity=dict((k, a["snapshot_integrity"].get(k))
                                   for k in ("all_equal", "max_abs_diff", "n_tensors_compared")),
        ar_replay_equals_real=dict((k, a["replay"].get(k))
                                   for k in ("elementwise_equal", "abs_err_max", "argmax_same")),
    )
    # ---- cache 逐张量比较（AR 8216 vs Spec 8215；只比较共同部分）----
    ca = torch.load(sorted(glob.glob(str(OUTDIR / f"{TAG}_{FAMILY}_ar_*.cache.pt")))[-1],
                    map_location="cpu")
    cs = torch.load(_newest(f"{TAG}_{FAMILY}_spec_*.cache.pt"),
                    map_location="cpu")[f"round{w['round']}_snap_pre"]
    la, ls = int(ca[0]["k"].shape[2]), int(cs[0]["k"].shape[2])
    n = min(la, ls)
    ca_t = spec_trim = None
    try:
        ca_t = [{k: (v[:, :, :n] if torch.is_tensor(v) and v.dim() >= 3 and v.shape[2] == la else v)
                 for k, v in d.items()} for d in ca]
        cs_t = [{k: (v[:, :, :n] if torch.is_tensor(v) and v.dim() >= 3 and v.shape[2] == ls else v)
                 for k, v in d.items()} for d in cs]
        hist = CC.compare_caches(ca_t, cs_t, dequantize=False)
        hist["note"] = (f"AR cache {la} vs Spec cache {ls}，**只比较共同前缀 {n}**；"
                        "base_bf16 ⇒ 原始 bf16 载荷比较")
        out["cache_history_compare_common_prefix"] = hist
    except Exception as e:                                        # noqa: BLE001
        out["cache_history_compare_common_prefix"] = dict(error=str(e))
    p = OUTDIR / f"{TAG}_analysis_cpu.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("== 复现 ==")
    print(f"  AR argmax={out['repro']['ar_argmax']} (expect {EXPECT_AR}) match="
          f"{out['repro']['ar_match']} | Spec emitted={out['repro']['spec_emitted']} "
          f"(expect {EXPECT_SPEC}) match={out['repro']['spec_match']}")
    print(f"  AR 共同前缀前 25 token = {out['repro']['ar_common_prefix']}")
    print("== 决策来源 ==")
    print(f"  SpecDec: round={dec['deciding_round']} slot={dec['deciding_slot']} "
          f"{dec['source_kind']} ({dec['source_detail']})")
    print(f"    输入位置 {dec['input_position']} -> 预测位置 {dec['predicted_position']}；"
          f"cache 送入长度 {dec['pre']}；Kp={dec['Kp']} m={dec['m']}")
    print(f"    cands={dec['cands']} g={dec['g']} emitted={dec['emitted']}")
    print("== 同 token 同位置 logits 对照 ==")
    for side in ("ar", "spec"):
        c = cmp_[side]
        print(f"  {side:4s}: argmax={c['argmax_id']} max={c['max_score']} "
              f"tied={c['n_max_tied']}{c['tied_max_tokens']} margin_top1_top2={c['margin_top1_top2']}")
    print(f"  逐元素相等={cmp_['elementwise_equal']} argmax_same={cmp_['argmax_same']} "
          f"max|Δ|={cmp_['abs_err_max']}")
    print("  关键候选 logits:", json.dumps(out["key_candidate_logits"], ensure_ascii=False))
    al = out["alignment"]
    print("== 对齐 ==")
    print(f"  同输入 token={al['same_input_token']} ({al['ar_input_token']} vs {al['spec_input_token']})；"
          f"同输入位置={al['same_input_position']} ({al['ar_input_position']} vs {al['spec_input_position']})")
    print(f"  cache 长度 AR={al['ar_cache_len_before']} vs Spec={al['spec_cache_len_before']}；"
          f"形状 AR={al['ar_shape']} vs Spec={al['spec_shape']}")
    hc = out["cache_history_compare_common_prefix"]
    if "error" not in hc:
        print(f"== cache 共同前缀比较（{hc['length']} vs {hc['length_b']}） ==")
        print(f"  逐位相同={hc['all_equal']} max|Δ|={hc['max_abs_diff']} n={hc['n_tensors_compared']}"
              f" 分键={json.dumps(hc['per_key'],ensure_ascii=False)}")
    print(f"[analysis] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
