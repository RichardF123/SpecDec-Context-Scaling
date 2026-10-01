"""f13_i50_corrections_cpu.py — F13/index50 诊断记录的**修正与重新标注**（纯 CPU，不新增 GPU 运行）。

保留全部历史产物，只**新增**一份 `…_corrected_<run_id>.json`，逐条给出修正依据：

1. **自由生成覆盖度**：AR 捕获的最后两步是**固定喂入** `[220,16]`，不能称为完全自由生成。
   离线检查：`row_8241.argmax == 16`（= 喂入的第二个 token）⇒ 喂入值与模型在该位置自己的选择一致，
   因此 8242 的决策行仍是**真实 AR 决策行**；但口径必须写成"末两步为固定喂入（已离线核对一致）"。
2. **不可比较项**：`probe_rows.ar_row8241_vs_sp_saved_next` 两侧**预测位置不同**（8241 vs 8240）⇒
   显式标为 `incomparable`，**不得**用于任何同位置比较（其 max_abs=12.86 无意义）。
3. **字段命名**：`execution_effect_*` / `history_effect_*` 实际存的是 `same_argmax`（布尔），
   命名会误导（同一个 `False` 在 H_AR 表示"effect 存在"、在 H_SP 表示不存在）。
   新增 `same_argmax` / `choice_changed` 明确字段；**不改写**旧 JSON，只在修正文件里声明旧名含义。
4. **margin 只作描述标签**：AR/Spec 不一致仍记 **FAIL**，不因 margin 小而豁免。
5. **mask/可见范围的证据级别**：本轮只做了**按位置公式推导**（`decision_rows.slot_table`），
   **未**取 attention mask 张量 ⇒ 不得写成"已验证实际 attention mask"。
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f13i50_20261001"
FEED = [220, 16]
TARGET_POS = 8242
AR_OLD, SP_OLD = 17, 20


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cap_ar_j, cap_sp_j = _newest(f"{TAG}_capture_ar_*.json"), _newest(f"{TAG}_capture_spec_*.json")
    cells_j = _newest(f"{TAG}_cells_*.json")
    ar = json.load(open(cap_ar_j, encoding="utf-8"))
    sp = json.load(open(cap_sp_j, encoding="utf-8"))
    ce = json.load(open(cells_j, encoding="utf-8"))
    rec = dict(run_id=run_id, tool="f13_i50_corrections_cpu",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               sources=dict(capture_ar=str(cap_ar_j), capture_spec=str(cap_sp_j),
                            cells=str(cells_j)),
               scope="只新增修正文件；**不修改**任何历史产物。")

    # ---- 1. 自由生成覆盖度 + 离线一致性检查 ----
    r8241 = ar["ar"]["row_8241"]
    r8242 = ar["ar"]["row_8242"]
    fed_matches = (int(r8241["argmax"]) == int(FEED[1]))
    rec["correction_1_free_generation_coverage"] = dict(
        claim_corrected_from="AR 捕获（完全自由生成）",
        claim_corrected_to="AR 捕获：到 cache=8240 之前的 token 由模型自决；**末两步为固定喂入 [220,16]**",
        fed=FEED,
        row_8241_argmax=int(r8241["argmax"]),
        fed_token_2=int(FEED[1]),
        fed_matches_model_choice=bool(fed_matches),
        row_8242_argmax=int(r8242["argmax"]),
        conclusion=("离线核对通过：喂入的第二个 token(16) 与模型在位置 8241 自己的 argmax 一致 ⇒ "
                    "8242 的决策行仍是**真实 AR 决策行**；但口径必须写成『末两步为固定喂入（已离线核对）』"),
        hard_assertion_added_to_tool="f13_i50_capture.py: require(row_8241.argmax == FEED[1])")
    assert fed_matches, "离线核对失败：喂入 token 与该行 argmax 不一致"

    # ---- 2. 不可比较项 ----
    probe = (ce.get("probe_rows") or {}).get("ar_row8241_vs_sp_saved_next")
    sp_pos = int(sp["align"]["pre"])
    rec["correction_2_incomparable"] = dict(
        field="probe_rows.ar_row8241_vs_sp_saved_next",
        a_predicted_pos=8241, b_predicted_pos=sp_pos,
        incomparable=True,
        reason=f"两侧**预测位置不同**（{8241} vs {sp_pos}）⇒ 不是同位置比较",
        recorded_max_abs=(probe or {}).get("max_abs"),
        action="显式标为 incomparable；该值**未**用于任何结论",
        note="该字段本轮**未**进入结论；此处只是把它正式标记为不可比较")

    # ---- 3. 字段命名修正（不改旧 JSON）----
    cmp_ = ce["compare"]
    norm = dict(
        same_history_S_vs_B_AR=dict(same_argmax=cmp_["same_history_S_vs_B_AR"]["same_argmax"],
                                    choice_changed=(not cmp_["same_history_S_vs_B_AR"]["same_argmax"]),
                                    a_argmax=cmp_["same_history_S_vs_B_AR"]["a_argmax"],
                                    b_argmax=cmp_["same_history_S_vs_B_AR"]["b_argmax"]),
        same_history_S_vs_B_SP=dict(same_argmax=cmp_["same_history_S_vs_B_SP"]["same_argmax"],
                                    choice_changed=(not cmp_["same_history_S_vs_B_SP"]["same_argmax"]),
                                    a_argmax=cmp_["same_history_S_vs_B_SP"]["a_argmax"],
                                    b_argmax=cmp_["same_history_S_vs_B_SP"]["b_argmax"]),
        same_exec_S_AR_vs_SP=dict(same_argmax=cmp_["same_exec_S_AR_vs_SP"]["same_argmax"],
                                  choice_changed=(not cmp_["same_exec_S_AR_vs_SP"]["same_argmax"]),
                                  a_argmax=cmp_["same_exec_S_AR_vs_SP"]["a_argmax"],
                                  b_argmax=cmp_["same_exec_S_AR_vs_SP"]["b_argmax"]),
        same_exec_B_AR_vs_SP=dict(same_argmax=cmp_["same_exec_B_AR_vs_SP"]["same_argmax"],
                                  choice_changed=(not cmp_["same_exec_B_AR_vs_SP"]["same_argmax"]),
                                  a_argmax=cmp_["same_exec_B_AR_vs_SP"]["a_argmax"],
                                  b_argmax=cmp_["same_exec_B_AR_vs_SP"]["b_argmax"]),
        old_field_semantics=dict(
            note="旧 `verdict.execution_effect_*` / `history_effect_*` **存的是 same_argmax**，"
                 "名字与含义不一致（同一个 False 在 H_AR 表示『效应存在』、在 H_SP 表示『不存在』）",
            old_values=dict(ce["verdict"])))
    rec["correction_3_normalized_fields"] = norm

    # ---- 4. margin 只作描述标签；不一致仍 FAIL ----
    rec["correction_4_margin_labeling"] = dict(
        margins={k: ce["cells"][k]["margin"] for k in ce["cells"]},
        label_policy="margin 只作**描述标签**（是否落在并列/1 ULP 量级）",
        status_policy="AR/Spec 完整序列不一致仍记 **FAIL**；不因 margin 小而豁免",
        mismatch_status="FAIL",
        mismatch_detail=dict(first_divergence_index=50, predicted_pos=TARGET_POS,
                             ar_token=AR_OLD, spec_token=SP_OLD))

    # ---- 5. mask/可见范围的证据级别 ----
    rec["correction_5_mask_evidence_level"] = dict(
        formula_derived=dict(source="decision_rows.slot_table",
                             claim=f"决策行可见最后 key = pre+1 = 8241（按位置公式推导）",
                             used_in_this_round=True),
        actual_tensor_checked=dict(checked=False,
                                   note="本轮 **未** 取 attention mask / scores 张量；"
                                        "**不得**表述为『已验证实际 attention mask』"),
        precedent=("更早的 L27 诊断（§9.30 等）确实检查过**实际** mask/`-inf`，"
                   "但那是**另一批运行**的证据，不能挪用来支持本轮 F13 cells"),
        action="在报告里把两者分开书写；本轮结论只依赖公式推导 + argmax/logits 比较")

    p = OUTDIR / f"{TAG}_corrected_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("== 修正 1：自由生成覆盖度 ==")
    print(f"  喂入 {FEED}；row_8241.argmax={r8241['argmax']}（与喂入一致={fed_matches}）；"
          f"row_8242.argmax={r8242['argmax']}")
    print("== 修正 2：不可比较项 ==")
    print(f"  AR row_8241 预测 8241 vs Spec saved_next 预测 {sp_pos} ⇒ incomparable=True")
    print("== 修正 3：字段命名 ==")
    for k, v in norm.items():
        if isinstance(v, dict) and "same_argmax" in v:
            print(f"  {k}: same_argmax={v['same_argmax']} choice_changed={v['choice_changed']} "
                  f"({v['a_argmax']} vs {v['b_argmax']})")
    print("== 修正 4：margin 标签 / 不一致仍 FAIL ==")
    print(f"  margins={rec['correction_4_margin_labeling']['margins']}；"
          f"status={rec['correction_4_margin_labeling']['mismatch_status']}")
    print("== 修正 5：mask 证据级别 ==")
    print(f"  实际 mask 张量检查 = {rec['correction_5_mask_evidence_level']['actual_tensor_checked']['checked']}"
          "（本轮只做公式推导）")
    print(f"\n[corr] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
