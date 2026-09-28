"""diag_fields_selftest_cpu.py — `diag_fields` 的轻量回归（**不加载大模型**）。

覆盖任务要求："先用无大模型测试覆盖**并列**、**非并列**、**位置编号**及**字段一致性**"。

用法：python -m specdec.engine.diag_fields_selftest_cpu
"""
from __future__ import annotations

import torch

from . import diag_fields as D

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main() -> int:
    print("== ① bf16 相邻间距（ULP）必须按**真实可表示值**算 ==")
    for x in (20.0, 20.625, 0.5, 1.0, 16.0, 31.5, -20.625):
        u = float(D.bf16_ulp(torch.tensor([x]))[0])
        print(f"  [info] bf16_ulp({x}) = {u}")
    check("20.625 附近间距 = 0.125（[16,32) 区间；**不是** 0.0625）",
          abs(float(D.bf16_ulp(torch.tensor([20.625]))[0]) - 0.125) < 1e-9)
    check("20.0 附近间距 = 0.125", abs(float(D.bf16_ulp(torch.tensor([20.0]))[0]) - 0.125) < 1e-9)
    check("1.0 附近间距 = 2^-8 = 0.00390625",
          abs(float(D.bf16_ulp(torch.tensor([1.0]))[0]) - 2 ** -8) < 1e-12)
    check("0.5 附近间距 = 2^-9", abs(float(D.bf16_ulp(torch.tensor([0.5]))[0]) - 2 ** -9) < 1e-12)
    check("负值取相邻间距的较小者，与正值同量级",
          abs(float(D.bf16_ulp(torch.tensor([-20.625]))[0]) - 0.125) < 1e-9)
    check("20.625 与 20.5 之间恰好是 **1 个 ULP**",
          abs((20.625 - 20.5) / float(D.bf16_ulp(torch.tensor([20.625]))[0]) - 1.0) < 1e-9)

    print("\n== ② 并列：argmax 与 topk 顺序必须分开记录 ==")
    V = 8
    row = torch.full((V,), 1.0)
    row[3] = 20.625
    row[5] = 20.625                      # 与 index 3 **精确并列**
    f = D.row_fields(row, watch=[3, 5, 1], k=4)
    check("② argmax_id = 较小下标 3（并列时 argmax 的取法）", f["argmax_id"] == 3, str(f["argmax_id"]))
    check("② max_score = 20.625（数值本身，与下标无关）", f["max_score"] == 20.625)
    check("② n_max_tied = 2（并列数）", f["n_max_tied"] == 2, str(f["n_max_tied"]))
    check("② tied_max_tokens = [3,5]（**完整**并列集合）", f["tied_max_tokens"] == [3, 5],
          str(f["tied_max_tokens"]))
    check("② topk 列表里含两个 20.625（但**不**宣称其顺序为解码 top1）",
          sum(1 for e in f["topk"] if e["score"] == 20.625) == 2 and "topk" in f["topk_note"])
    check("② watch_scores 给的是原始分数", f["watch_scores"] == {"3": 20.625, "5": 20.625, "1": 1.0})
    check("② finite=True", f["finite"] is True)

    print("\n== ③ 非并列 ==")
    row2 = torch.tensor([1.0, 2.0, 20.625, 20.5, 0.0, 0.0, 0.0, 0.0])
    f2 = D.row_fields(row2, watch=[2, 3], k=3)
    check("③ argmax_id = 2", f2["argmax_id"] == 2)
    check("③ n_max_tied = 1", f2["n_max_tied"] == 1)
    check("③ tied_max_tokens = [2]", f2["tied_max_tokens"] == [2])
    check("③ 20.625 与 20.5 相差 1 ULP（逐元素算）",
          abs(float(D.bf16_ulp(torch.tensor([20.625]))[0]) - 0.125) < 1e-9)

    print("\n== ④ 逐元素 ULP 比较（不得用单一幅度归一化整行）==")
    a = torch.tensor([20.625, 20.625, 1.0, 1.0])
    b = torch.tensor([20.5, 20.5, 1.0, 1.0])
    c = D.compare_rows(a, b)
    check("④ 原始绝对误差 = 0.125（优先报告）", abs(c["abs_err_max"] - 0.125) < 1e-9, str(c["abs_err_max"]))
    check("④ 逐元素 ULP 比 = 1.0（用各元素自身间距）", abs(c["ulp_err_max"] - 1.0) < 1e-6,
          str(c["ulp_err_max"]))
    d = D.compare_rows(torch.tensor([1.0]), torch.tensor([1.0 + 2 ** -8]))
    check("④ 小幅度处 1 个 2^-8 步长 ⇒ ULP 比 ≈1，而不是被大幅度 ULP 归一成 0.0078",
          abs(d["ulp_err_max"] - 1.0) < 1e-3, str(d["ulp_err_max"]))
    check("④ argmax_same 字段存在且正确", c["argmax_same"] is True)
    e = D.compare_rows(a, torch.tensor([20.5, 20.75, 1.0, 1.0]))
    check("④ 混合幅度行：报原始绝对误差 0.125（=1 ULP）且逐元素 ULP 也算出来",
          abs(e["abs_err_max"] - 0.125) < 1e-9 and e["ulp_err_max"] is not None,
          f"abs={e['abs_err_max']} ulp={e['ulp_err_max']}")

    print("\n== ④b 近零元素不得毒化 ULP 统计 ==")
    z = D.compare_rows(torch.tensor([1e-6, 20.625]), torch.tensor([0.15, 20.5]))
    check("④b 原始绝对误差为主口径，不被近零元素放大", abs(z["abs_err_max"] - 0.149999) < 1e-5,
          str(z["abs_err_max"]))
    check("④b ULP 统计排除 |x|<1 的近零元素 ⇒ 不会出现 1e6 量级",
          z["ulp_err_max"] is not None and z["ulp_err_max"] < 10,
          f"ulp_err_max={z['ulp_err_max']}")
    check("④b 明确写出归一化告警与最大误差所在元素",
          "不得" in z["normalization_caveat"] and z["abs_err_at_max_element"]["index"] == 0,
          str(z["abs_err_at_max_element"]))
    check("④b n_ulp_valid 只数了 |x|>=1 的元素（2 个里 1 个）", z["n_ulp_valid"] == 1,
          str(z["n_ulp_valid"]))

    print("\n== ⑤ 位置编号三套口径必须分开 ==")
    p = D.position_labels(prompt_len=8192, gen_index=31, cache_len_before=8222, kp=4)
    check("⑤ 输入位置 = 8222", p["input_position"] == 8222)
    check("⑤ 所预测 token 位置 = 8223", p["predicted_token_position"] == 8223)
    check("⑤ 生成序列 index = 31", p["gen_index"] == 31)
    check("⑤ block 的 4 行位置 = 8222..8225", p["block_rows_positions"] == [8222, 8223, 8224, 8225])
    check("⑤ 由 gen_index 反推的输入位置与实测一致（8222）", p["consistent"] is True)
    q = D.position_labels(prompt_len=8192, gen_index=30, cache_len_before=8222, kp=4)
    check("⑤ 若把 PREFIX_TOKENS=30 误当分歧下标 ⇒ 一致性检查**报错**",
          q["consistent"] is False,
          "即 30 对应输入位置 8221（产出 572 的那个决策），不是分歧点")

    print("\n== ⑥ 字段一致性 ==")
    f3 = D.row_fields(torch.arange(8, dtype=torch.float32), watch=[0, 7], k=8)
    check("⑥ argmax_id 与 topk[0].token 在本例一致（但语义仍分开记录）",
          f3["argmax_id"] == f3["topk"][0]["token"])
    check("⑥ max_score == topk[0].score == 该 token 的 watch_scores",
          f3["max_score"] == f3["topk"][0]["score"] == f3["watch_scores"]["7"])
    check("⑥ 并列场景下 argmax_id 允许与 topk[0].token 不同（② 已展示）", f["argmax_id"] in f["tied_max_tokens"])

    print(f"\nCPU 自测：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
