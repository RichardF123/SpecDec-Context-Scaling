"""decision_rows.py — 决策行对齐的纯逻辑（CPU 可测）。

`_verify` 的索引语义（已由既有 16 条 CPU 哨兵钉死）：
  设 `pre` = verify 之前的 cache 长度、`s = Kp` = 候选个数、`T = pre + Kp`。
  * `saved_next`  = 位置 `pre-1` 的 logits ⇒ **预测位置 `pre`**，用于回答 `cands[0]`；
  * `rows[j]`     = 位置 `pre+j` 的 logits ⇒ **预测位置 `pre+j+1`**，用于回答 `cands[j+1]`；
  * 可见范围：`rows[j]` 能看见的最后一个 key = `(T - s) + j = pre + j`（**不含**位置 `pre+j+1` 的候选）。
  * `g = [argmax(saved_next)] + [argmax(rows[j]) for j in range(Kp-1)]`，与 `cands` 逐位比较，
    第一个不等处 `i` = 被替换的槽位 ⇒ `emitted = cands[:i] + [g[i]]`，`m = i`。
    **该槽位的 argmax 来自 `rows[i-1]`**（`i>=1`），或来自 `saved_next`（`i==0`）。

本模块把这些关系变成**可判定**的函数，避免"取错行 / 混用 repair 后输出 / 用错位置"。
"""
from __future__ import annotations

import torch


class AlignError(Exception):
    pass


def require(cond, what: str, detail=""):
    if not cond:
        raise AlignError(f"{what} 未满足。{detail}".strip())
    return True


def slot_table(pre: int, kp: int) -> list:
    """返回每个候选槽位的 `(slot, source, row_index, predicted_pos, visible_last_key)`。

    `source` ∈ {"saved_next", "rows[j]"}；`predicted_pos` 是该槽位要决定的**绝对位置**；
    `visible_last_key` 是该次调用能看到的最大 key 位置。
    """
    require(pre >= 0 and kp >= 1, "pre 必须 >=0 且 kp>=1", f"pre={pre} kp={kp}")
    out = []
    for slot in range(kp):
        if slot == 0:
            src, ridx, vis = "saved_next", None, pre - 1
        else:
            ridx = slot - 1
            src, vis = f"rows[{ridx}]", pre + ridx
        out.append(dict(slot=slot, source=src, row_index=ridx, predicted_pos=pre + slot,
                        visible_last_key=vis))
    return out


def deciding_slot(cands: list, g: list) -> int:
    """第一个 `g[i] != cands[i]` 的槽位（全接受则返回 `len(cands)`）。"""
    require(len(g) == len(cands), "g 与 cands 长度必须一致", f"{len(g)} vs {len(cands)}")
    for i, (c, gg) in enumerate(zip(cands, g)):
        if c != gg:
            return i
    return len(cands)


def align_decision(pre: int, kp: int, cands: list, g: list, emitted: list) -> dict:
    """把一次 verify 的**决策行**对齐出来，并做一致性硬检查。"""
    require(len(cands) == kp, "cands 长度必须等于 kp", f"{len(cands)} vs {kp}")
    i = deciding_slot(cands, g)
    if i == len(cands):                     # 全接受：没有"被替换的槽位"，emitted 应等于 cands
        require(list(emitted) == list(cands), "全接受时 emitted 必须等于 cands",
                f"{emitted} vs {cands}")
    else:
        require(list(emitted) == list(cands[:i]) + [g[i]],
                "emitted 必须等于 cands[:m] + [被替换 token]", f"{emitted}")
    tab = slot_table(pre, kp)
    slot = tab[i] if i < len(tab) else None
    rows_used = [] if slot is None or slot["row_index"] is None else [slot["row_index"]]
    return dict(pre=pre, kp=kp, m=i, cands=list(cands), g=list(g), emitted=list(emitted),
                deciding_slot=i, deciding_source=(slot["source"] if slot else None),
                deciding_row_index=(slot["row_index"] if slot else None),
                deciding_predicted_pos=(slot["predicted_pos"] if slot else None),
                deciding_visible_last_key=(slot["visible_last_key"] if slot else None),
                rows_used=rows_used, all_accept=(i == kp),
                table=tab,
                rule="saved_next↔cands[0]↔位置 pre；rows[j]↔cands[j+1]↔位置 pre+j+1；"
                     "被替换槽位 i 的 argmax 来自 rows[i-1]（i>=1）或 saved_next（i==0）")


def require_deciding_row_matches(align: dict, *, expect_source: str, expect_row_index=None,
                                 expect_pred_pos: int | None = None) -> None:
    """断言"我们捕获/比较的那一行"就是真正的决策行。"""
    require(align["deciding_source"] == expect_source,
            "决策行来源与预期不符", f"{align['deciding_source']} != {expect_source}")
    if expect_row_index is not None:
        require(align["deciding_row_index"] == expect_row_index,
                "决策行下标与预期不符", f"{align['deciding_row_index']} != {expect_row_index}")
    if expect_pred_pos is not None:
        require(align["deciding_predicted_pos"] == expect_pred_pos,
                "决策行预测位置与预期不符",
                f"{align['deciding_predicted_pos']} != {expect_pred_pos}")


def require_aligned_row_is_causal(align: dict) -> None:
    """决策行**不得**看见它自己要决定的那个位置或更后的候选。"""
    if align["deciding_source"] == "saved_next" or align["deciding_slot"] == 0:
        return
    vis = align["deciding_visible_last_key"]
    target = align["deciding_predicted_pos"]
    require(vis < target, "决策行看见了未来位置（违反因果/掩码）", f"visible={vis} target={target}")
    require(vis == target - 1, "决策行的可见范围应为 target-1", f"{vis} vs {target - 1}")


def snapshot(cache: list) -> list:
    """**深拷贝到 CPU**（`_verify`/`_step` 会原地改长 cache，必须先拷贝）。"""
    return [{k: (v.detach().clone().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
            for d in cache]


def require_unmodified(cache: list, snap: list, what: str) -> None:
    """证明运行前的快照未被原地改写。"""
    require(len(cache) == len(snap), f"{what}: 层数变化", f"{len(cache)} vs {len(snap)}")
    for i, (a, b) in enumerate(zip(cache, snap)):
        for kk in ("k", "v"):
            require(kk in a and kk in b, f"{what}: 第 {i} 层缺少 {kk!r}")
            require(tuple(a[kk].shape) == tuple(b[kk].shape),
                    f"{what}: L{i}.{kk} 形状被改", f"{tuple(a[kk].shape)} vs {tuple(b[kk].shape)}")
            require(bool(torch.equal(a[kk].detach().cpu(), b[kk])),
                    f"{what}: L{i}.{kk} 被原地修改")


def require_distinct_storage(a: list, b: list, what: str) -> None:
    require(len(a) == len(b), f"{what}: 层数不一致")
    shared = [f"L{i}_{kk}" for i, (da, db) in enumerate(zip(a, b)) for kk in ("k", "v")
              if da[kk].data_ptr() == db[kk].data_ptr()]
    require(not shared, f"{what}: 共享可变存储（必须是独立深拷贝）", str(shared[:6]))
