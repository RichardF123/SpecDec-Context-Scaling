"""decision_rows_selftest_cpu.py — 决策行对齐逻辑的 CPU 自测（先自测，再捕获）。

覆盖任务 §三 要求（决策行对齐部分）：
  * saved_next / block row / 预测位置对齐；
  * 被替换槽位来自 `rows[i-1]`（而不是 rows[i]、最后一行或 repair 输出）；
  * 决策行看不见未来候选（因果/掩码）；
  * cache 快照不被原地修改（能检出被改写）；
  * 独立深拷贝（共享存储被检出）；
  * 必需记录缺失必须阻断判定。
（同前缀验证 / mask 可见范围 / cache 截断 / 不同 token 历史不得比较 已在
 `sb_align_selftest_cpu` 覆盖，此处不重复。）

运行：`python -m specdec.engine.decision_rows_selftest_cpu`
"""
from __future__ import annotations

import torch

from . import decision_rows as D


def main() -> int:
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    def raises(fn):
        try:
            fn()
            return False
        except D.AlignError:
            return True

    print("== 1. 槽位表：saved_next / rows[j] ↔ 预测位置 ==")
    tab = D.slot_table(8240, 4)
    check("slot0 = saved_next，预测位置 = pre（8240）",
          tab[0]["source"] == "saved_next" and tab[0]["predicted_pos"] == 8240)
    check("rows[0] 预测 pre+1 = 8241", tab[1]["source"] == "rows[0]"
          and tab[1]["predicted_pos"] == 8241)
    check("rows[1] 预测 pre+2 = 8242（本轮决策位置）", tab[2]["source"] == "rows[1]"
          and tab[2]["predicted_pos"] == 8242)
    check("rows[j] 的可见最后 key = pre+j（rows[1] ⇒ 8241，不含未来候选）",
          tab[2]["visible_last_key"] == 8241)
    check("saved_next 的可见最后 key = pre-1", tab[0]["visible_last_key"] == 8239)

    print("== 2. 本轮 F13 round19 的对齐（真实记录）==")
    al = D.align_decision(pre=8240, kp=4, cands=[220, 16, 16, 1154], g=[220, 16, 20, 11],
                          emitted=[220, 16, 20])
    check("被替换槽位 m = 2", al["m"] == 2, al["m"])
    check("★ 决策行 = rows[1]（不是 rows[2]、最后一行或 repair 输出）",
          al["deciding_source"] == "rows[1]" and al["deciding_row_index"] == 1)
    check("★ 决策行预测位置 = 8242", al["deciding_predicted_pos"] == 8242)
    D.require_deciding_row_matches(al, expect_source="rows[1]", expect_row_index=1,
                                   expect_pred_pos=8242)
    check("require_deciding_row_matches 对正确的行通过", True)
    check("★ 错行会被拦截（声称是 rows[2]）",
          raises(lambda: D.require_deciding_row_matches(al, expect_source="rows[2]")))
    check("★ 错位置会被拦截（声称 8243）",
          raises(lambda: D.require_deciding_row_matches(al, expect_source="rows[1]",
                                                        expect_pred_pos=8243)))
    D.require_aligned_row_is_causal(al)
    check("★ 因果检查通过：可见 8241 < 目标 8242", True)

    print("== 3. 全接受 / 零接受 ==")
    al2 = D.align_decision(pre=8240, kp=4, cands=[1, 2, 3, 4], g=[1, 2, 3, 4],
                           emitted=[1, 2, 3, 4])
    check("全接受：deciding_slot = kp、all_accept=True、无决策行",
          al2["deciding_slot"] == 4 and al2["all_accept"] and al2["deciding_source"] is None)
    al3 = D.align_decision(pre=8240, kp=4, cands=[1, 2, 3, 4], g=[9, 2, 3, 4], emitted=[9])
    check("零接受（m=0）：决策来自 saved_next（不是 rows[0]）",
          al3["m"] == 0 and al3["deciding_source"] == "saved_next"
          and al3["deciding_predicted_pos"] == 8240)
    check("零接受时 emitted = [g[0]]", al3["emitted"] == [9])

    print("== 4. emitted 一致性必须被强制 ==")
    check("★ emitted 与 cands[:m]+[g[m]] 不符 ⇒ 抛错",
          raises(lambda: D.align_decision(pre=8240, kp=4, cands=[220, 16, 16, 1154],
                                          g=[220, 16, 20, 11], emitted=[220, 16, 16])))
    check("★ g 与 cands 长度不符 ⇒ 抛错",
          raises(lambda: D.deciding_slot([1, 2], [1, 2, 3])))

    print("== 5. 决策行的因果/可见范围必须成立 ==")
    bad = dict(al)
    bad["deciding_visible_last_key"] = 8242      # 假装能看见自己要决定的位置
    check("★ 决策行看见未来位置 ⇒ 抛错", raises(lambda: D.require_aligned_row_is_causal(bad)))
    bad2 = dict(al)
    bad2["deciding_visible_last_key"] = 8240     # 少看一个（位置算错）
    check("★ 可见范围不等于 target-1 ⇒ 抛错", raises(lambda: D.require_aligned_row_is_causal(bad2)))
    bad3 = dict(al)
    bad3["deciding_visible_last_key"] = 8243
    check("★ 能看见后续候选（8243）⇒ 抛错",
          raises(lambda: D.require_aligned_row_is_causal(bad3)))

    print("== 6. cache 快照：原地修改与共享存储都能被检出 ==")
    mk = lambda n=8: [dict(k=torch.zeros(1, 2, n, 128, dtype=torch.bfloat16),
                           v=torch.zeros(1, 2, n, 128, dtype=torch.bfloat16)) for _ in range(2)]
    c = mk()
    snap = D.snapshot(c)
    D.require_unmodified(c, snap, "未修改时应通过")
    check("未修改的 cache 通过 require_unmodified", True)
    c[1]["k"][0, 0, 0, 0] = 1.0
    check("★ cache 被原地改写 ⇒ 抛错", raises(lambda: D.require_unmodified(c, snap, "polluted")))
    a, b = mk(), mk()
    D.require_distinct_storage(a, b, "独立深拷贝")
    check("独立深拷贝通过 require_distinct_storage", True)
    check("★ 共享存储 ⇒ 抛错", raises(lambda: D.require_distinct_storage(a, a, "shared")))

    print("== 7. 快照必须是 CPU 且是真拷贝（不被后续原地改长影响）==")
    c2 = mk(4)
    snap2 = D.snapshot(c2)
    c2.append(dict(k=torch.zeros(1, 2, 9, 128, dtype=torch.bfloat16),
                   v=torch.zeros(1, 2, 9, 128, dtype=torch.bfloat16)))   # 模拟 cache 变长
    check("★ 快照层数不随 cache 变长而变", len(snap2) == 2 and len(c2) == 3)
    check("快照在 CPU 上", all(t.device.type == "cpu" for d in snap2 for t in d.values()))

    print("== 8. 必需记录缺失必须阻断（不静默继续）==")
    try:
        D.require(False, "缺少决策行 logits", "本轮必须重新捕获，不得从 token ID 反推 margin")
        check("★ 缺少决策 logits ⇒ 抛错", False)
    except D.AlignError:
        check("★ 缺少决策 logits ⇒ 抛错", True)
    try:
        D.require(False, "快照缺失", "AR 在 cache 长度 8240 的深拷贝不存在")
        check("★ 缺少快照 ⇒ 抛错", False)
    except D.AlignError:
        check("★ 缺少快照 ⇒ 抛错", True)

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
