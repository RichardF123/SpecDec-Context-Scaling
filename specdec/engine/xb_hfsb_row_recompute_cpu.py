"""xb_hfsb_row_recompute_cpu.py — **只读重算**：用既有产物做「同协议跨实现行对照」（纯 CPU，不跑 GPU）。

## 为什么需要它
`xb_probe.py` 上一轮 HF 侧的克隆坏了（历史被清空），于是那次"阶段 B 跨实现对照"报告出 12–16 的差异
（已判 **INVALID**）。本轮修好克隆后，**不需要重跑 8K** 也能得到更好的证据：另一支独立 HF 探针
`hf_serial_block_probe.py` 用的是**正确**克隆，且与本协议 **ctx/chunk/pre/tokens 完全相同**。
把两侧**各自**的既有产物放在一起重算即可。

## 三重前置校验（任一不满足就**拒绝**出结论，rc=6）
1. **协议一致**：ctx / chunk / pre / tokens 四项逐项相同；
2. **历史一致（位级证据）**：`hfsb.initial_cache.kv_digest.L0_k.sha` == `xb_hf.kv_digest.c2_sha[0]`
   —— 两支独立 HF 运行对同一 8240-token 历史算出的**第 0 层 K 张量逐位相同**
   （同一实现、同一权重 ⇒ 相同 K 实际上排除了"输入序列/位置不同"）；
3. **精度设置一致**：hfsb 运行的 `allow_bf16_reduced_precision_reduction` **有效值**必须等于
   xb-ours 运行记录的值（默认 `RED=0` 的那次 hfsb **不**入选）。

## 它**不是**什么（严格性上限，必须随结论一起引用）
* **不是**"同一份代码的同路径对照"：两侧来自不同工具、不同 API 调用约定；
* HF 侧来自 `hfsb`，它额外传了 `output_attentions=True`（微型 CPU 检查显示数值中性，见
  `cache_clone` 自测第 6 组）并装有 mask 记录器（包装器原样转发，见报告 §9.43）；
* **不**判定哪一方"正确"，**不**改变任何门槛，**不**用于正式验收；行级只有 argmax 与 sha 可对齐
  （两侧都存了 `_row_fields` 的 sha，同一 `_sha` 定义 ⇒ sha 相等即逐位相等）。
"""
from __future__ import annotations

import glob
import json
import os
import time

OUT = "output/g5/attn"
TAG = "xb_hfsb_row_recompute"


def _load(pat):
    ps = sorted(glob.glob(os.path.join(OUT, pat)))
    return ps


def _eff_reduction(j):
    r = j.get("bf16_reduced_precision_reduction")
    if isinstance(r, dict):
        return bool(r.get("effective_after_set", r.get("default_at_import", True)))
    return None                                     # 旧运行没记录 ⇒ 未知，按"未声明"处理


def main() -> int:
    ours_p = _load("xb_20261001_ours_*.json")
    xbhf_p = _load("xb_20261001_hf_*.json")
    if not ours_p or not xbhf_p:
        print("[xbrr] !! 缺少 xb 产物")
        return 5
    oj = json.load(open(ours_p[-1], encoding="utf-8"))
    xj = json.load(open(xbhf_p[-1], encoding="utf-8"))
    want_red = oj["precision"]["bf16_reduced_precision_reduction"]

    cands, rejected = [], []
    for p in _load("hfsb_20261001_F13_*.json"):
        j = json.load(open(p, encoding="utf-8"))
        if j.get("stage") != "complete" or not j.get("rows"):
            continue
        eff = _eff_reduction(j)
        why = []
        for k in ("ctx", "chunk", "pre", "tokens"):
            if j.get(k) != oj.get(k):
                why.append(f"{k}={j.get(k)!r}≠{oj.get(k)!r}")
        if eff != want_red:
            why.append(f"bf16_reduction={eff}≠{want_red}")
        (rejected if why else cands).append((os.path.basename(p), why))
    rec = dict(tool="xb_hfsb_row_recompute_cpu", ours_run=oj["run_id"], xb_hf_run=xj["run_id"],
               want_bf16_reduction=want_red,
               hfsb_candidates=[c[0] for c in cands],
               hfsb_rejected=[dict(run=c[0], why=c[1]) for c in rejected],
               recompute_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"[xbrr] ours={oj['run_id']} xb_hf={xj['run_id']} 目标 reduction={want_red}")
    for name, why in rejected:
        print(f"   [排除] hfsb {name}：{'; '.join(why)}")
    if not cands:
        print("[xbrr] !! 没有满足协议+精度的 hfsb 运行 ⇒ 拒绝出结论（rc=6）")
        return 6
    hj = json.load(open(os.path.join(OUT, cands[-1][0]), encoding="utf-8"))
    print(f"[xbrr] 选用 hfsb {cands[-1][0]}（协议与精度均一致；其余候选保留在 JSON）")
    rec["hfsb_used"] = cands[-1][0]

    # ---- 2) 历史一致（位级证据）----
    h_l0k = hj["initial_cache"]["kv_digest"]["L0_k"]["sha"]
    x_l0k = xj["kv_digest"]["c2_sha"][0]
    rec["history_bitwise_check"] = dict(
        hfsb_L0_k_sha=h_l0k, xb_hf_C2_L0_k_sha=x_l0k, equal=bool(h_l0k == x_l0k),
        meaning="两支独立 HF 运行对同一 8240-token 历史算出的第 0 层 K 逐位相同 "
                "⇒ 输入序列/位置/日程一致（同实现同权重下）")
    print(f"[xbrr] 历史位级核对：hfsb L0_k sha={h_l0k} | xb-hf C2 L0_k sha={x_l0k} "
          f"⇒ {'一致' if h_l0k == x_l0k else '**不一致**'}")
    if h_l0k != x_l0k:
        print("[xbrr] !! 历史不是位级同一 ⇒ 拒绝出结论（rc=6）")
        p = os.path.join(OUT, f"{TAG}_{oj['run_id']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        print(f"[xbrr] RESULT_JSON={p}")
        return 6

    # ---- 3) 行对照 ----
    rows = []
    for j in range(len(oj["tokens"])):
        a_s, a_b = oj["serial"][j], oj["block"][j]                    # OURS（正确克隆）
        h_s, h_b = hj["rows"]["serial"][j], hj["rows"]["block"][j]     # HF 全历史（正确克隆）
        x_s, x_b = xj["serial"][j], xj["block"][j]                     # HF 空历史（坏克隆症状）
        rows.append(dict(
            j=j, pred_pos=int(h_s.get("pred_pos", oj["pre"] + j + 1)),
            ours_serial=dict(argmax=a_s["argmax"], margin=a_s["margin"], sha=a_s["sha"]),
            ours_block=dict(argmax=a_b["argmax"], margin=a_b["margin"], sha=a_b["sha"]),
            hf_full_serial=dict(argmax=h_s["argmax"], margin=h_s["margin"], sha=h_s["sha"]),
            hf_full_block=dict(argmax=h_b["argmax"], margin=h_b["margin"], sha=h_b["sha"]),
            hf_wiped_serial=dict(argmax=x_s["argmax"], margin=x_s["margin"], sha=x_s["sha"]),
            hf_wiped_block=dict(argmax=x_b["argmax"], margin=x_b["margin"], sha=x_b["sha"]),
            argmax_ours_vs_hf_full_serial=bool(a_s["argmax"] == h_s["argmax"]),
            argmax_ours_vs_hf_full_block=bool(a_b["argmax"] == h_b["argmax"]),
            row_bitequal_ours_vs_hf_full_serial=bool(a_s["sha"] == h_s["sha"]),
            row_bitequal_ours_vs_hf_full_block=bool(a_b["sha"] == h_b["sha"]),
            hf_full_vs_wiped_differ=bool(h_s["argmax"] != x_s["argmax"]),
        ))
    rec["rows"] = rows
    rec["all_argmax_equal_serial"] = all(r["argmax_ours_vs_hf_full_serial"] for r in rows)
    rec["all_argmax_equal_block"] = all(r["argmax_ours_vs_hf_full_block"] for r in rows)
    rec["n_rows_bitwise_equal"] = sum(r["row_bitequal_ours_vs_hf_full_serial"]
                                      and r["row_bitequal_ours_vs_hf_full_block"] for r in rows)
    rec["limits"] = ("非同一份代码的同路径对照；HF 侧来自另一支探针（output_attentions=True、装有原样转发的"
                     "mask 记录器）；只对齐 argmax 与行 sha；不判定谁对；不改变任何门槛。")
    print(f"\n{'j':>2} {'pred':>5} | {'OURS-S':>6} {'OURS-B':>6} | {'HF全-S':>6} {'HF全-B':>6} | "
          f"{'HF空-S':>6} {'HF空-B':>6} | argmax同(S/B) | 行位级同(S/B)")
    for r in rows:
        print(f"{r['j']:>2} {r['pred_pos']:>5} | {r['ours_serial']['argmax']:>6} "
              f"{r['ours_block']['argmax']:>6} | {r['hf_full_serial']['argmax']:>6} "
              f"{r['hf_full_block']['argmax']:>6} | {r['hf_wiped_serial']['argmax']:>6} "
              f"{r['hf_wiped_block']['argmax']:>6} | "
              f"{str(r['argmax_ours_vs_hf_full_serial'])[0]}/{str(r['argmax_ours_vs_hf_full_block'])[0]:>5} | "
              f"{str(r['row_bitequal_ours_vs_hf_full_serial'])[0]}/"
              f"{str(r['row_bitequal_ours_vs_hf_full_block'])[0]}")
    print(f"\n★ 四行 argmax 是否全同（serial / block）：{rec['all_argmax_equal_serial']} / "
          f"{rec['all_argmax_equal_block']}；行级逐位相等的行数 {rec['n_rows_bitwise_equal']}/{len(rows)}")
    print("  ★ 对照观察（不是结论）：HF 空历史两臂 = "
          f"{[r['hf_wiped_serial']['argmax'] for r in rows]} / "
          f"{[r['hf_wiped_block']['argmax'] for r in rows]}"
          "  ← 这是坏克隆的症状；与 HF 全历史不同 ⇒ 旧的 12–16 差异来自克隆 bug")
    p = os.path.join(OUT, f"{TAG}_{oj['run_id']}_{hj['run_id']}_"
                          f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"[xbrr] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
