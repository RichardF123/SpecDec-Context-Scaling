"""sb_align.py — S/B 对照的**对齐与可比性**纯逻辑（CPU 可测；不含模型调用）。

为什么需要它
  之前用「预测位置 → 行」的字典比较，会把 **verify 行**与 **repair 行**的来源隐去，
  也会把「已被拒候选之后的行」当成同前缀行塞进误差统计。本模块把**行来源**、
  **可见 token 历史**、**可比性判定**变成显式数据，并对每条必需条件做硬检查。

核心概念
  * 一行（row）由一次调用产生，带：
      - `pred_pos`   : 该行预测的绝对位置（= 该行能看见的最后一个位置的**下一个**位置）
      - `n_visible`  : 该行能看见的 key 数（含自身）——**必须等于 pred_pos**
      - `source`     : `prefill` | `verify` | `repair` | `serial`
      - `detail`     : 例如 `round=11,j=1` 或 `step=24`
      - `hist`       : 该行可见的**绝对位置 ≥ start_pos 的 (位置, token)** 列表（升序）
      - `cache_len_after` : 调用后 cache 的物理长度
  * **可比性**：只有 `hist` 里每一个 token 都与「共同前缀」在同位置的 token 相同，且
    `pred_pos <= limit`，该行才可与另一条路径的同位置行比较；否则标为**不可比较**，
    **不进入**任何误差统计。
"""
from __future__ import annotations

import hashlib

import torch

ROW_SOURCES = ("prefill", "verify", "repair", "serial")


class AlignError(Exception):
    """对齐/可比性/校验失败。"""


def stable_sha(t: torch.Tensor, n: int = 16) -> str:
    """**跨进程可比**的内容 sha256（不用进程内 `hash()`）。"""
    if not torch.is_tensor(t):
        raise AlignError(f"stable_sha 只接受张量，收到 {type(t).__name__}")
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def require(cond, what: str, detail=""):
    if not cond:
        raise AlignError(f"{what} 未满足。{detail}".strip())
    return True


def make_row(*, pred_pos: int, n_visible: int, cache_len_after: int, source: str,
             detail: str, hist: list, row: torch.Tensor) -> dict:
    """构造一行并做**结构性硬检查**（位置/长度/来源/顺序/形状）。"""
    require(source in ROW_SOURCES, f"未知行来源 {source!r}", str(ROW_SOURCES))
    require(isinstance(pred_pos, int) and pred_pos > 0, "pred_pos 必须是正整数", str(pred_pos))
    require(n_visible == pred_pos,
            f"{source}:{detail} 可见 key 数必须等于预测位置（否则位置偏移）",
            f"n_visible={n_visible} pred_pos={pred_pos}")
    require(cache_len_after >= n_visible,
            f"{source}:{detail} cache 物理长度不得小于可见数",
            f"{cache_len_after} < {n_visible}")
    hs = [(int(p), int(t)) for p, t in hist]
    require(all(hs[i][0] < hs[i + 1][0] for i in range(len(hs) - 1)),
            f"{source}:{detail} hist 位置必须严格递增", str(hs[:6]))
    if hs:
        require(hs[-1][0] == pred_pos - 1,
                f"{source}:{detail} hist 最后一个位置必须 = pred_pos-1",
                f"last={hs[-1][0]} pred_pos={pred_pos}")
    r = row.detach().float().reshape(-1).cpu()
    require(torch.isfinite(r).all(), f"{source}:{detail} 行内含非有限值")
    return dict(pred_pos=pred_pos, n_visible=n_visible, cache_len_after=int(cache_len_after),
                source=source, detail=detail, hist=hs, row=r,
                argmax=int(torch.argmax(r)), row_sha=stable_sha(row))


def comparable(row: dict, common: dict, *, limit: int) -> tuple:
    """判该行是否与「共同前缀」一致：返回 `(bool, reason)`。

    `common`：绝对位置 → 共同前缀 token（只含**已确认共同**的位置）。
    """
    if row["pred_pos"] > limit:
        return False, f"预测位置 {row['pred_pos']} 超出共同前缀对照上限 {limit}"
    for pos, tok in row["hist"]:
        if pos not in common:
            return False, f"hist 位置 {pos} 无共同前缀记录（token={tok}）"
        if common[pos] != tok:
            return False, f"hist 位置 {pos} 的 token={tok} != 共同前缀 {common[pos]}"
    return True, "ok"


def compare_arms(rows_s: list, rows_b: list, common: dict, *, limit: int) -> dict:
    """**保留来源**的对照：按预测位置配对，逐行判可比性，不可比者不入统计。"""
    out = dict(limit=limit, pairs=[], excluded_rows=[], excluded_by_reason={},
               first_nonzero=None, first_argmax=None)
    by_pos_b: dict = {}
    for rb in rows_b:
        by_pos_b.setdefault(rb["pred_pos"], []).append(rb)
    paired_keys = set()
    for rs in rows_s:
        cands = by_pos_b.get(rs["pred_pos"], [])
        ok_s, why_s = comparable(rs, common, limit=limit)
        if not cands:
            # 「S 有行、B 无对应位置」由下面的**统一清点**登记，避免重复条目
            continue
        for rb in cands:
            ok_b, why_b = comparable(rb, common, limit=limit)
            if not (ok_s and ok_b):
                # ⚠️ **不在这里登记排除项**：本循环是「S×B 组合」，同一行会被登记多次
                # （实测 63 条里只有 51 行是不同行）。排除清单统一由下面的**逐行清点**产生。
                continue
            paired_keys.add(("S", f"{rs['source']}:{rs['detail']}", rs["pred_pos"]))
            paired_keys.add(("B", f"{rb['source']}:{rb['detail']}", rb["pred_pos"]))
            d = (rs["row"] - rb["row"]).abs()
            rec = dict(pred_pos=rs["pred_pos"],
                       s_source=f"{rs['source']}:{rs['detail']}",
                       b_source=f"{rb['source']}:{rb['detail']}",
                       s_cache_len=rs["cache_len_after"], b_cache_len=rb["cache_len_after"],
                       s_n_visible=rs["n_visible"], b_n_visible=rb["n_visible"],
                       prefix_len=len(rs["hist"]),
                       s_argmax=rs["argmax"], b_argmax=rb["argmax"],
                       same_argmax=bool(rs["argmax"] == rb["argmax"]),
                       elementwise_equal=bool(torch.equal(rs["row"], rb["row"])),
                       max_abs=float(d.max()), mean_abs=float(d.mean()),
                       n_diff=int((d > 0).sum()), n_elem=int(d.numel()))
            out["pairs"].append(rec)
    out["pairs"].sort(key=lambda x: (x["pred_pos"], x["b_source"]))
    # ---- 排除清单：**严格一行一条**（键 = which + source:detail + pred_pos，并断言唯一）----
    out.pop("incomparable", None)
    seen = set()
    for which, rows in (("S", rows_s), ("B", rows_b)):
        for r in rows:
            key = (which, f"{r['source']}:{r['detail']}", r["pred_pos"])
            if key in paired_keys or key in seen:
                continue
            seen.add(key)
            okr, why = comparable(r, common, limit=limit)
            if which == "B":
                has_s = any(rs["pred_pos"] == r["pred_pos"] for rs in rows_s)
                reason = why if not okr else ("S 臂无该预测位置的行" if not has_s
                                              else "S 臂该位置的行不可比")
            else:
                reason = why if not okr else "B 臂无该预测位置的行"
            out["excluded_rows"].append(dict(
                which=which, pred_pos=r["pred_pos"], source=f"{r['source']}:{r['detail']}",
                comparable=okr, reason=reason, cache_len=r["cache_len_after"],
                n_visible=r["n_visible"], prefix_len=len(r["hist"]), argmax=r["argmax"]))
    out["excluded_rows"].sort(key=lambda x: (x["pred_pos"], x["which"], x["source"]))
    keys = [(x["which"], x["source"], x["pred_pos"]) for x in out["excluded_rows"]]
    require(len(keys) == len(set(keys)),
            "排除清单出现**重复登记**（同一 which+source+pred_pos 多次）",
            str(sorted({k for k in keys if keys.count(k) > 1})[:4]))
    by_reason: dict = {}
    for x in out["excluded_rows"]:
        k = ("预测位置超出上限" if "上限" in x["reason"] else
             ("token 前缀与共同前缀不符" if "!=" in x["reason"] else
              ("无共同前缀记录" if "无共同前缀记录" in x["reason"] else x["reason"])))
        by_reason[k] = by_reason.get(k, 0) + 1
    out["excluded_by_reason"] = by_reason
    out["n_excluded_rows"] = len(out["excluded_rows"])
    out["n_excluded_distinct_pos"] = len({x["pred_pos"] for x in out["excluded_rows"]})
    out["dedup_note"] = ("`excluded_rows` 严格**一行一条**（键 = which + source:detail + pred_pos，"
                         "已 require 唯一）；`pairs` 只含可比配对。"
                         "上一版把「S×B 组合」也登记为条目 ⇒ 同一行被记多次（63 条里实际只有 51 行），已修。")
    out["counts"] = dict(s_rows=len(rows_s), b_rows=len(rows_b),
                         paired_s_rows=len({x["s_source"] for x in out["pairs"]}),
                         paired_b_rows=len({x["b_source"] for x in out["pairs"]}),
                         paired_positions=len({x["pred_pos"] for x in out["pairs"]}),
                         excluded_rows=out["n_excluded_rows"])
    for rec in out["pairs"]:
        if out["first_nonzero"] is None and not rec["elementwise_equal"]:
            out["first_nonzero"] = rec["pred_pos"]
        if out["first_argmax"] is None and not rec["same_argmax"]:
            out["first_argmax"] = rec["pred_pos"]
    return out


def require_full_cache(cache: list, *, expect_len: int, n_layers: int, what: str,
                       head_dim: int | None = None) -> dict:
    """全层 KV 的 shape/dtype/finite/长度**硬检查**。

    `head_dim=None` 时从第一层推断，并要求**所有层一致**
    （target 为 128、draft 为 64 —— 不得硬编码成 128）。
    """
    require(len(cache) == n_layers, f"{what}: 层数不符", f"{len(cache)} vs {n_layers}")
    if head_dim is None:
        require("k" in cache[0], f"{what}: 第 0 层缺少 'k'")
        head_dim = int(cache[0]["k"].shape[-1])
    info = {}
    for i, d in enumerate(cache):
        for kk in ("k", "v"):
            require(kk in d, f"{what}: 第 {i} 层缺少 {kk!r}")
            t = d[kk]
            require(t.dim() == 4, f"{what}: L{i}.{kk} 必须是 4 维", str(tuple(t.shape)))
            require(int(t.shape[0]) == 1 and int(t.shape[3]) == head_dim,
                    f"{what}: L{i}.{kk} 末维/批维不符（期望 head_dim={head_dim}）",
                    str(tuple(t.shape)))
            require(int(t.shape[2]) == expect_len,
                    f"{what}: L{i}.{kk} 长度不符", f"{t.shape[2]} vs {expect_len}")
            require(str(t.dtype) == "torch.bfloat16", f"{what}: L{i}.{kk} 必须是 bf16", str(t.dtype))
            require(bool(torch.isfinite(t).all()), f"{what}: L{i}.{kk} 含非有限值")
            info[f"L{i}_{kk}"] = dict(shape=[int(x) for x in t.shape], dtype=str(t.dtype),
                                      sha=stable_sha(t), sum=float(t.detach().float().sum()))
    return info


def cache_digest_diff(a: dict, b: dict) -> dict:
    """用**稳定摘要**比较两份全层摘要；并明确区分「摘要一致」与直接逐元素检查。"""
    keys = sorted(set(a) & set(b))
    same = [k for k in keys if a[k]["sha"] == b[k]["sha"]]
    diff = [k for k in keys if a[k]["sha"] != b[k]["sha"]]
    return dict(n=len(keys), n_same_digest=len(same), n_diff_digest=len(diff),
                diff_keys=diff[:8],
                semantics="摘要一致**不等于**逐元素检查；摘要不同即**确证**存在差异。"
                          "若只有 sum 可用，只能判「是否不同」，不能推断差异大小。")


def serial_expected_len(prompt_len: int, n_fed: int) -> int:
    """串行臂在喂入 `n_fed` 个 token 之后的**应有 cache 长度**。"""
    return int(prompt_len) + int(n_fed)


def initial_state_independent(a: list, b: list) -> dict:
    """S/B 起点检查：**逐元素相同** 且 **不共享可变存储**。"""
    require(len(a) == len(b), "初始状态层数不一致")
    shared, not_equal = [], []
    for i, (da, db) in enumerate(zip(a, b)):
        for kk in ("k", "v"):
            if da[kk].data_ptr() == db[kk].data_ptr():
                shared.append(f"L{i}_{kk}")
            if not torch.equal(da[kk], db[kk]):
                not_equal.append(f"L{i}_{kk}")
    require(not shared, "S/B 初始状态共享可变存储（必须为独立深拷贝）", str(shared[:6]))
    require(not not_equal, "S/B 初始状态并非逐元素相同", str(not_equal[:6]))
    return dict(elementwise_equal=True, shares_storage=False, n_tensors_checked=2 * len(a))
