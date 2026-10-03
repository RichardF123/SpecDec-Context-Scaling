"""row_aligned_decision_cpu.py — 全层 row-aligned 候选模式的**只读**项目级汇总（纯 CPU、不加载模型）。

## 作用
从**已落盘的产物**独立复算并汇总（不重跑 GPU、不改任何产物）：
① Stage 0：`row_aligned.py` 的门结果（由 `--stage0` 现跑一次纯 CPU 门，或读既有记录）；
② Stage 1：`div8209_ball_*.json` 的 1A 硬门与 1B 分类、逐层 `n_diff` 轨迹、候选 applied/seen 计数；
③ Stage 2：`{TAG}_stage2_summary_*.json` 的 token equality 硬门（候选 vs AR）、两次重复一致性；
④ Stage 3：`{TAG}_stage3_summary_*.json` 的性能观察（若已执行）；
⑤ **决策输出**：`correctness_gain` / `overhead` / `net_value` 与四选一
   （VIABLE_B6_CANDIDATE / CORRECT_BUT_NO_SPEED_VALUE / CORRECTNESS_STILL_FAILS / ENVIRONMENT_INVALID）。

**不**加载模型、**不**做前向、**不**改任何既有产物。
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

OUT = ROOT / "output" / "g5" / "attn"


def _newest(pattern: str):
    fs = sorted(glob.glob(str(OUT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(fs[-1]) if fs else None


def _stage1(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    layers = j["per_layer_S0_vs_BALL"]["layers"]
    c = j["class_1B"]
    return dict(
        artifact=path.name, stage=j.get("stage"), run_id=j.get("run_id"),
        expected=j.get("expected_counts"),
        gate_1A=dict(
            S0_history=bool(all((v is None) or v["exact_equal"] for v in j["gate_1A"]["S0_history"].values())),
            B0_history=bool(all((v is None) or v["exact_equal"] for v in j["gate_1A"]["B0_history"].values())),
            SALL_equals_S0=bool(j["gate_1A"]["SALL_vs_S0"]["final_row"]["exact_equal"]
                                and j["gate_1A"]["SALL_vs_S0"]["layer_rows_identical"]),
            qk_applied=j["gate_1A"]["coverage"]["qk_applied_ball"],
            qk_expected=j["gate_1A"]["coverage"]["expected_qk"],
            lin_applied=j["gate_1A"]["coverage"]["lin_applied_ball"],
            lin_expected=j["gate_1A"]["coverage"]["expected_linear"],
            draft_applied=j["gate_1A"]["coverage"]["draft_applied"],
            repeat_all_true=bool(all(v is True for k, v in j["repeat_consistency"].items()
                                     if k.endswith("identical")))),
        class_1B=dict(category=c["category"], logits_exact=c["logits"]["exact_equal"],
                      logits_n_diff=c["logits"]["n_diff"], logits_max_abs=c["logits"]["abs_err_max"],
                      token_s0=c["token_s0"]["argmax"], token_ball=c["token_ball"]["argmax"],
                      margin_s0=c["token_s0"]["margin"], margin_ball=c["token_ball"]["margin"]),
        per_layer=[dict(layer=l["layer"], in_eq=l["input_exact_equal"], in_nd=l["input_n_diff"],
                        out_eq=l["output_exact_equal"], out_nd=l["output_n_diff"]) for l in layers],
        first_layer_with_equal_input_and_different_output=j["per_layer_S0_vs_BALL"][
            "first_layer_with_equal_input_and_different_output"],
        all_layers_identical=j["per_layer_S0_vs_BALL"]["all_layers_identical"],
        sections_bitwise_equal_up_to=next((l["layer"] - 1 for l in layers if not l["output_exact_equal"]), None),
        controls=dict(positions_both_predict_target=j["controls"]["positions"]["both_predict_target"],
                      causal_prefix_ok=j["controls"]["causal_prefix"]["all_rows_ok"],
                      token_match_anchors=all(v["match"] for v in j["controls"]["token_match_anchors"].values()),
                      cache_struct=j["controls"]["cache_struct"]),
        finiteness=j["finiteness"], source_integrity=dict(
            H_snapshot_unchanged=j["source_integrity"]["H_snapshot_unchanged"],
            W_data_ptr_stable=j["source_integrity"]["W_data_ptr_stable"],
            probe_state_now=j["source_integrity"]["probe_state_now"]["active"],
            serial_qk_state_now=j["source_integrity"]["serial_qk_state_now"]["active"],
            serial_linear_state_now=j["source_integrity"]["serial_linear_state_now"]["active"],
            verify_wrap_now=j["source_integrity"]["verify_wrap_now"]["active"],
            hooks_removed=j["source_integrity"]["hooks_removed"]),
        candidate_counts={a: j["controls"].get("verify_only", {}) for a in ("BALL",)})


def _stage2(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    return dict(artifact=path.name, run_utc=j.get("run_utc"), cases=j.get("cases"),
                hard_gate=j.get("stage2_hard_gate"), checks=j.get("checks"), per_case=j.get("per_case"),
                runs=j.get("runs"))


def _stage3(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    return j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--stage3", default=None)
    a = ap.parse_args()
    s1p = Path(a.stage1) if a.stage1 else _newest("div8209_ball_*.json")
    s2p = Path(a.stage2) if a.stage2 else _newest("div8209_ra2_stage2_summary_*.json")
    s3p = Path(a.stage3) if a.stage3 else _newest("div8209_ra2_stage3_summary_*.json")
    rec: dict = dict(tool="row_aligned_decision_cpu",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从已落盘产物汇总 Stage 0–3 的门与代价，给出四选一决策；不加载模型、不改产物",
                     sources=dict(stage1=(s1p.name if s1p else None), stage2=(s2p.name if s2p else None),
                                  stage3=(s3p.name if s3p else None)))
    s1 = _stage1(s1p) if s1p and s1p.exists() else None
    s2 = _stage2(s2p) if s2p and s2p.exists() else None
    s3 = _stage3(s3p) if s3p and s3p.exists() else None
    rec["stage1"] = s1
    rec["stage2"] = s2
    rec["stage3"] = s3
    print("== ① Stage 1（位置 8209 全层局部门）==")
    if s1:
        g = s1["gate_1A"]
        print(f"   产物 {s1['artifact']}（stage={s1['stage']}）；预期 {s1['expected']}")
        print(f"   1A：S0复现历史={g['S0_history']} B0复现历史={g['B0_history']} SALL≡S0={g['SALL_equals_S0']} "
              f"QK applied={g['qk_applied']}/{g['qk_expected']} Linear applied={g['lin_applied']}/{g['lin_expected']} "
              f"draft applied={g['draft_applied']} 两次重复一致={g['repeat_all_true']}")
        c = s1["class_1B"]
        print(f"   1B：类别 {c['category']}；logits exact={c['logits_exact']} n_diff={c['logits_n_diff']} "
              f"max|Δ|={c['logits_max_abs']}；token {c['token_s0']} vs {c['token_ball']}；"
              f"margin {c['margin_s0']}/{c['margin_ball']}")
        print(f"   逐层：**L0–L{s1['sections_bitwise_equal_up_to']} 逐位一致**；"
              f"首个「输入同·输出不同」层 = L{s1['first_layer_with_equal_input_and_different_output']}；"
              f"全层一致={s1['all_layers_identical']}")
        print("   " + " ".join(f"L{l['layer']}:{l['out_nd']}" for l in s1["per_layer"][:8]) + " …")
    else:
        print("   （无 Stage 1 产物）")
    print("\n== ② Stage 2（已知案例 token equality 硬门）==")
    if s2:
        for c, pc in (s2["per_case"] or {}).items():
            print(f"   [{c}] 原始 Spec 失败={pc['original_spec_fails']}"
                  f"（首分歧 index={pc['first_divergence_original_spec']}）；"
                  f"**候选 == AR：{pc['candidate_equals_ar']}**（首分歧 index={pc['first_divergence_candidate']}）")
            print(f"        重复一致={pc['reps_consistent']}；stop={pc['stop_reasons']}；"
                  f"候选计数一致={pc['candidate_counts_match_block_calls']}")
        print(f"   ★ Stage 2 硬门 = {s2['hard_gate']}")
        for k, v in (s2["checks"] or {}).items():
            print(f"     [check] {k}: {v}")
    else:
        print("   （无 Stage 2 产物 ⇒ Stage 2 未执行或未完成）")
    print("\n== ③ Stage 3（性能观察）==")
    if s3:
        for k, v in (s3.get("comparison") or {}).items():
            print(f"   {k}: {json.dumps(v, ensure_ascii=False, default=str)[:400]}")
    else:
        print("   （无 Stage 3 产物 ⇒ Stage 3 未执行）")

    # ---- 决策 ----
    env_invalid = bool(s1 is None or (s1 and s1.get("finiteness", {}).get("all_logits_finite") is False))
    if s1 and s1["gate_1A"] and not all([s1["gate_1A"]["S0_history"], s1["gate_1A"]["B0_history"],
                                        s1["gate_1A"]["SALL_equals_S0"], s1["gate_1A"]["repeat_all_true"],
                                        s1["gate_1A"]["qk_applied"] == s1["gate_1A"]["qk_expected"],
                                        s1["gate_1A"]["lin_applied"] == s1["gate_1A"]["lin_expected"]]):
        env_invalid = True
    correctness_gain = None
    if s2:
        correctness_gain = dict(original_spec_fails={c: pc["original_spec_fails"]
                                                     for c, pc in s2["per_case"].items()},
                                aligned_spec_passes_token_equality={c: pc["candidate_equals_ar"]
                                                                    for c, pc in s2["per_case"].items()})
    overhead = None
    net_value = None
    if s3:
        overhead = s3.get("overhead")
        net_value = s3.get("net_value")
    if env_invalid:
        decision = "ENVIRONMENT_INVALID"
    elif s2 is None:
        decision = "CORRECTNESS_STILL_FAILS"      # Stage 2 未通过/未执行 ⇒ 不能作为候选
    elif s2["hard_gate"] != "PASS":
        decision = "CORRECTNESS_STILL_FAILS"
    elif overhead is None:
        decision = "CORRECT_BUT_NO_SPEED_VALUE"   # 正确性通过但还没测代价 ⇒ 不能判 VIABLE
    else:
        dec_ok = bool(overhead.get("aligned_decode_over_original") is not None
                      and net_value is not None and net_value.get("aligned_spec_vs_ar") is not None)
        faster_than_ar = bool(net_value and net_value.get("aligned_spec_vs_ar", 0) > 1.0)
        lower_overhead = bool(overhead and overhead.get("aligned_decode_over_original", 99) < 2.0)
        decision = ("VIABLE_B6_CANDIDATE" if (dec_ok and faster_than_ar and lower_overhead)
                    else "CORRECT_BUT_NO_SPEED_VALUE")
    rec["decision_inputs"] = dict(correctness_gain=correctness_gain, overhead=overhead,
                                  net_value=net_value, environment_invalid=env_invalid,
                                  stage1_class_1B=(s1["class_1B"]["category"] if s1 else None),
                                  stage2_hard_gate=(s2["hard_gate"] if s2 else None),
                                  stage3_executed=bool(s3))
    rec["decision"] = decision
    rec["decision_policy"] = ("不预设人工阈值：只有在 **token equality 硬门全过** 且 "
                              "**aligned Spec 相对 AR 仍有加速（net_value > 1）** 且 **相对原始 Spec 的"
                              "decode 时间代价 < 2×** 时才判 VIABLE_B6_CANDIDATE；"
                              "正确但无加速价值 ⇒ CORRECT_BUT_NO_SPEED_VALUE；"
                              "token equality 未过 ⇒ CORRECTNESS_STILL_FAILS；"
                              "环境/门不可信 ⇒ ENVIRONMENT_INVALID")
    rec["b6_unchanged"] = True
    print("\n== ★ 决策 ==")
    print(f"   correctness_gain = {json.dumps(correctness_gain, ensure_ascii=False, default=str)}")
    print(f"   overhead         = {json.dumps(overhead, ensure_ascii=False, default=str)}")
    print(f"   net_value        = {json.dumps(net_value, ensure_ascii=False, default=str)}")
    print(f"   ⇒ **{decision}**")
    print("   正式 B6 状态：**未改变**（本轮未跑正式矩阵；acceptance/pilot/final 仍禁止）")
    p = OUT / f"row_aligned_decision_cpu_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[decision] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
