"""g5_8k_divergence_2x2_selftest_cpu.py — 2×2 诊断工具**纯逻辑**的 CPU 自测（不加载模型）。

覆盖本轮修的两个缺陷在"判定逻辑"层面的防线：
  A. 生成预算：`capture_verdict` 必须能识别"Kp 被削减"这种情况（v1 的失败模式），
     并据此拒绝套用旧归因（all_match=False）。
  B. 证据有效性：`argmax_same` **不得**被当作"两行相同"；
     并列（gap=0）与真正胜出（gap>0）必须能被区分。
"""
from __future__ import annotations

import torch

from . import diag_fields as DF
from .g5_8k_divergence_2x2 import capture_verdict, _row_extra, _cell, WATCH

EXPECTED_ROUND = dict(round=8, Kp=4, m=1, pre=8222, cands=[572, 5230, 304, 279],
                      g=[572, 5880, 304, 279], emitted=[572, 5880])
LOGGED = dict(round=8, Kp=4, m=1, pre=8222, cands=[572, 5230, 304, 279],
              g=[572, 5880, 304, 279], emitted=[572, 5880])


def _run() -> int:
    npass = nfail = 0

    def check(name, cond):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}")

    print("== A. 捕获条件判定：忠实复现 vs 被预算削减 ==")
    v = capture_verdict(EXPECTED_ROUND, actual_round=8, actual_cache_len=8222,
                        actual_cands=[572, 5230, 304, 279], logged=LOGGED)
    check("完全一致（Kp=4）⇒ all_match=True", v["all_match"] is True)
    check("记录到预测 token 位置 = 8223",
          v["expect_predicted_token_position"] == 8223)

    # v1 的失败模式：max_new=32 ⇒ k_eff=2
    v2 = capture_verdict(EXPECTED_ROUND, actual_round=8, actual_cache_len=8222,
                         actual_cands=[572, 5230],
                         logged=dict(LOGGED, Kp=2, cands=[572, 5230], g=[572, 5880],
                                     emitted=[572, 5880]))
    check("★ Kp=2（v1 的失败模式）⇒ all_match=False", v2["all_match"] is False)
    check("★ 且明确标出 matches_Kp=False", v2["matches_Kp"] is False)
    check("★ 给出 '没有执行原实验计算形状' 的说明", "kp_mismatch_note" in v2)

    v3 = capture_verdict(EXPECTED_ROUND, actual_round=8, actual_cache_len=8222,
                         actual_cands=[572, 5230, 304, 279],
                         logged=dict(LOGGED, g=[572, 1083, 304, 279]))
    check("候选相同但 g 不同 ⇒ all_match=False（且 matches_cands 仍为 True）",
          v3["all_match"] is False and v3["matches_cands"] is True
          and v3["matches_g"] is False)
    v4 = capture_verdict(EXPECTED_ROUND, actual_round=8, actual_cache_len=8224,
                         actual_cands=[572, 5230, 304, 279], logged=LOGGED)
    check("cache_len 不是 8222 ⇒ all_match=False", v4["all_match"] is False
          and v4["matches_cache_len"] is False)
    v5 = capture_verdict(EXPECTED_ROUND, actual_round=7, actual_cache_len=8218,
                         actual_cands=[10993, 659, 576, 5492], logged=None)
    check("轮号错（round 7）⇒ all_match=False 且 matches_g 为 False（logged=None）",
          v5["all_match"] is False and v5["matches_g"] is False)

    print("== B. argmax 相同 ≠ 两行相同；并列与胜出必须可分 ==")
    a = torch.zeros(16)
    b = torch.zeros(16)
    a[3], b[3] = 20.625, 20.625
    a[5], b[5] = 20.5, 20.375
    cmp_ = DF.compare_rows(a, b)
    check("argmax 相同但值不同 ⇒ argmax_same=True 且 elementwise_equal=False",
          cmp_["argmax_same"] is True and cmp_["elementwise_equal"] is False)
    check("max|Δ| 反映真实差异（0.125）", abs(cmp_["abs_err_max"] - 0.125) < 1e-6)
    c = a.clone()
    check("完全相同 ⇒ elementwise_equal=True",
          DF.compare_rows(a, c)["elementwise_equal"] is True)

    print("== C. 并列（margin=0）与胜出（margin>0）的区分 ==")
    tied = torch.zeros(16)
    tied[3], tied[5] = 20.625, 20.625
    e_tied = _row_extra(tied)
    check("精确并列 ⇒ margin_top1_top2 = 0.0 且 tie_at_top=True",
          e_tied["margin_top1_top2"] == 0.0 and e_tied["tie_at_top"] is True)
    check("★ 并列时 gap_to_next_distinct **不是** 0（它是到第三档的距离 20.625）"
          "⇒ 两个间距含义不同",
          abs(e_tied["gap_to_next_distinct"] - 20.625) < 1e-9)
    won = torch.zeros(16)
    won[3], won[5] = 20.625, 20.5
    e_won = _row_extra(won)
    check("非并列 ⇒ margin_top1_top2 = 0.125 且 tie_at_top=False",
          abs(e_won["margin_top1_top2"] - 0.125) < 1e-9 and e_won["tie_at_top"] is False)
    check("非并列时 gap_to_next_distinct = 0.125",
          abs(e_won["gap_to_next_distinct"] - 0.125) < 1e-9)

    print("== D. 单元格字段：argmax_id 与并列集合分开记录 ==")
    cell = _cell(tied, watch=[3, 5, 7])
    check("cell 记录 argmax_id（较小下标 3）", cell["argmax_id"] == 3)
    check("cell 记录完整并列集合 {3,5}", sorted(cell["tied_max_tokens"]) == [3, 5])
    check("cell 记录 n_max_tied=2", cell["n_max_tied"] == 2)
    check("cell 记录 finite=True", cell["finite"] is True)
    check("并列 cell 的 margin_top1_top2=0", cell["margin_top1_top2"] == 0.0)
    cell_won = _cell(won, watch=[3, 5, 7])
    check("非并列 cell 的并列集合只含 argmax",
          cell_won["tied_max_tokens"] == [cell_won["argmax_id"]]
          and cell_won["n_max_tied"] == 1)
    check("watch 按传入下标取值（不越界）",
          set(cell["watch_scores"].keys()) == {"3", "5", "7"})

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
