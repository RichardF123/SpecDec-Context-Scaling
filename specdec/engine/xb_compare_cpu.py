"""xb_compare_cpu.py — 跨实现（OURS vs HF）同路径对照的**只读**比较（纯 CPU）。

阶段顺序：
  C1 = prompt 的 4×2048 分块 prefill 之后（cache len 8192）
  C2 = 再喂 48 个 token 的一个 chunk 之后（cache len 8240）
  Serial / Block = 从 C2 的独立深拷贝出发，4 个 token 的单步 / 块前向（**仅 `XB_STAGE=ab` 才会执行**）

## 阶段闸门（本轮新增）
两侧 JSON 里的 `stage_gate.executed` 决定**哪些阶段可比**。只比较**两边都执行过**的阶段：
未执行的阶段记 `executed=False`，打印"未执行"，**绝不当作"相同"**，也不参与 `earliest_difference_stage`。
阶段 B 额外要求两侧都带 `arm_clone_check`（克隆不变量断言）；缺失即**拒绝**比较阶段 B。
旧产物（`stage_gate` 缺失）按各产物里的键推断范围，并标记 `legacy_artifact=true`（"范围是推断的、不是声明的"）。

## 措辞纪律
C1 只能说成 **"第一个已检查且发现差异的检查点"**；不能说"差异首次产生于 C1"
（C1 之前没有已检查的检查点被比较过）。
"""
from __future__ import annotations

import glob
import json
import os
import time

import torch

OUT = "output/g5/attn"
_PROTO_KEYS = ("ctx", "chunk", "pre", "tokens", "family")


def newest(pat):
    return sorted(glob.glob(os.path.join(OUT, pat)))[-1]


def fields(x):
    r = x.detach().float().reshape(-1)
    t2 = torch.topk(r, 2).values
    am = int(torch.argmax(r))
    tied = [int(i) for i in torch.nonzero(r == r.amax()).reshape(-1)]
    return dict(argmax=am, max_score=float(r.max()), margin=float(t2[0] - t2[1]),
                n_tied_max=len(tied))


def cand(a, b, label):
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    d = (x - y).abs()
    fa, fb = fields(x), fields(y)
    return dict(stage=label, bitwise_equal=bool(torch.equal(x, y)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()), n_elem=int(x.numel()),
                argmax_ours=fa["argmax"], argmax_hf=fb["argmax"],
                same_argmax=bool(fa["argmax"] == fb["argmax"]),
                margin_ours=fa["margin"], margin_hf=fb["margin"],
                tied_ours=fa["n_tied_max"], tied_hf=fb["n_tied_max"])


def kv_cand(ka, kb, label):
    """逐层 K/V：逐位相等数 + 最大绝对差（只报摘要）。"""
    rows = []
    for i in range(min(len(ka), len(kb))):
        for kk in ("k", "v"):
            if kk not in ka[i] or kk not in kb[i]:
                rows.append(dict(layer=i, attr=kk, missing_key=True))
                continue
            x, y = ka[i][kk].float(), kb[i][kk].float()
            if tuple(x.shape) != tuple(y.shape):
                rows.append(dict(layer=i, attr=kk, shape_mismatch=[list(x.shape), list(y.shape)]))
                continue
            d = (x - y).abs()
            rows.append(dict(layer=i, attr=kk, bitequal=bool(torch.equal(x, y)),
                             max_abs=float(d.max()), mean_abs=float(d.mean()),
                             n_diff=int((d > 0).sum())))
    return dict(stage=label, rows=rows, n_layers=len(rows) // 2,
                n_bitequal=sum(1 for r in rows if r.get("bitequal")))


def _declared_scope(j, o):
    """返回 (executed_set, legacy)。缺 `stage_gate` 时按键推断并标记 legacy。"""
    sg = j.get("stage_gate") or {}
    if sg.get("executed"):
        return set(sg["executed"]), False
    scope = set()
    if "c1_logits" in o:
        scope.add("C1")
    if "c2_logits" in o:
        scope.add("C2")
    if "serial_rows" in o and "block_rows" in o:
        scope |= {"Serial", "Block"}
    return scope, True


def main() -> int:
    o = torch.load(newest("xb_20261001_ours_*.pt"), map_location="cpu")
    h = torch.load(newest("xb_20261001_hf_*.pt"), map_location="cpu")
    oj = json.load(open(newest("xb_20261001_ours_*.json"), encoding="utf-8"))
    hj = json.load(open(newest("xb_20261001_hf_*.json"), encoding="utf-8"))
    rec = dict(tool="xb_compare_cpu", ours=oj["run_id"], hf=hj["run_id"],
               ours_json=os.path.basename(newest("xb_20261001_ours_*.json")),
               hf_json=os.path.basename(newest("xb_20261001_hf_*.json")),
               schedule=oj["schedule"], tokens=oj["tokens"],
               input=dict(seq_sha16_ours=oj["input"]["seq_sha16"], seq_sha16_hf=hj["input"]["seq_sha16"],
                          prefix_len=oj["input"]["prefix_len"]),
               precision=dict(ours=oj["precision"], hf=hj["precision"]))
    # ---- 协议必须逐项相同，否则根本不是"同路径"对照 ----
    assert rec["input"]["seq_sha16_ours"] == rec["input"]["seq_sha16_hf"], "两侧输入序列 hash 不一致"
    assert oj["schedule"] == hj["schedule"], f"两侧日程不同：{oj['schedule']} vs {hj['schedule']}"
    for k in _PROTO_KEYS:
        assert oj.get(k) == hj.get(k), f"两侧协议参数 {k} 不同：{oj.get(k)} vs {hj.get(k)}"
    assert rec["precision"]["ours"] == rec["precision"]["hf"], "两侧精度设置不同"

    # ---- 阶段闸门：只比较两边都执行过的阶段 ----
    eo, lo = _declared_scope(oj, o)
    eh, lh = _declared_scope(hj, h)
    common = eo & eh
    rec["stage_gate"] = dict(
        ours_declared=sorted(eo), hf_declared=sorted(eh), common=sorted(common),
        ours_scope_inferred=lo, hf_scope_inferred=lh,
        rule="只比较两边都执行过的阶段；未执行的阶段记 executed=False，**不当作相同**，也不参与最早差异判定。")
    for tag, s in (("OURS", eo), ("HF", eh)):
        if not s:
            print(f"[xbc] !! {tag} 产物没有可比较的阶段（scope={sorted(s)}）⇒ 拒绝比较")
            return 5
    print(f"[xbc] 阶段范围：OURS={sorted(eo)} HF={sorted(eh)} 可比={sorted(common)}"
          + ("（旧产物，范围按键推断）" if (lo or lh) else ""))

    have_c1 = "C1" in common
    have_c2 = "C2" in common
    have_ab = {"Serial", "Block"} <= common

    if have_c1:
        print("== 检查点 C1（prompt 4×2048 分块 prefill，len 8192）==")
        rec["C1_logits"] = cand(o["c1_logits"], h["c1_logits"], "C1_logits")
        rec["C1_kv"] = kv_cand(o["c1_kv"], h["c1_kv"], "C1_kv")
        rec["C1_len"] = dict(ours=oj["C1"]["cache_len"], hf=hj["C1"]["cache_len"])
        c1 = rec["C1_logits"]
        print(f"  logits: bitwise={c1['bitwise_equal']} max|Δ|={c1['max_abs']:.6g} "
              f"mean={c1['mean_abs']:.6g} n_diff={c1['n_diff']} | argmax {c1['argmax_ours']} vs "
              f"{c1['argmax_hf']} | margin {c1['margin_ours']:.6g} vs {c1['margin_hf']:.6g}")
        print(f"  KV: 逐位相等 {rec['C1_kv']['n_bitequal']}/{len(rec['C1_kv']['rows'])} 个张量；"
              f"cache len {rec['C1_len']}")
    else:
        for k in ("C1_logits", "C1_kv", "C1_len"):
            rec[k] = dict(executed=False, reason="有一侧未执行 C1")
        print("== 检查点 C1：未执行（有一侧未执行）⇒ 不比较、不当作相同 ==")

    if have_c2:
        print("== 检查点 C2（+48 token 一个 chunk，len 8240）==")
        rec["C2_logits"] = cand(o["c2_logits"], h["c2_logits"], "C2_logits")
        rec["C2_kv"] = kv_cand(o["c2_kv"], h["c2_kv"], "C2_kv")
        rec["C2_len"] = dict(ours=oj["C2"]["cache_len"], hf=hj["C2"]["cache_len"])
        c2 = rec["C2_logits"]
        print(f"  logits: bitwise={c2['bitwise_equal']} max|Δ|={c2['max_abs']:.6g} "
              f"mean={c2['mean_abs']:.6g} n_diff={c2['n_diff']} | argmax {c2['argmax_ours']} vs "
              f"{c2['argmax_hf']} | margin {c2['margin_ours']:.6g} vs {c2['margin_hf']:.6g}")
        print(f"  KV: 逐位相等 {rec['C2_kv']['n_bitequal']}/{len(rec['C2_kv']['rows'])} 个张量；"
              f"cache len {rec['C2_len']}")
    else:
        for k in ("C2_logits", "C2_kv", "C2_len"):
            rec[k] = dict(executed=False, reason="有一侧未执行 C2")
        print("== 检查点 C2：未执行（有一侧未执行）⇒ 不比较、不当作相同 ==")

    if have_ab:
        aco = oj.get("arm_clone_check")
        ach = hj.get("arm_clone_check")
        if not (aco and ach):
            rec["B"] = dict(executed=False, reason="缺少 arm_clone_check（克隆不变量断言）⇒ 拒绝比较阶段 B")
            print("== 阶段 B：**拒绝比较**（两侧没有都提供 arm_clone_check）==")
            have_ab = False
    if have_ab:
        print(f"== 阶段 B：同路径跨实现（serial 第 j 步 ↔ block 第 j 行，均预测 pre+j+1，pre={oj['pre']}）==")
        rec["B_serial"] = [cand(o["serial_rows"][j], h["serial_rows"][j], f"serial_j{j}")
                           for j in range(len(oj["tokens"]))]
        rec["B_block"] = [cand(o["block_rows"][j], h["block_rows"][j], f"block_j{j}")
                          for j in range(len(oj["tokens"]))]
        rec["within_ours_serial_vs_block"] = [cand(o["serial_rows"][j], o["block_rows"][j], f"ours_j{j}")
                                              for j in range(len(oj["tokens"]))]
        rec["within_hf_serial_vs_block"] = [cand(h["serial_rows"][j], h["block_rows"][j], f"hf_j{j}")
                                            for j in range(len(oj["tokens"]))]
        rec["B_clone_check"] = dict(ours=aco, hf=ach)
        for tag, key in (("OURS-Serial vs HF-Serial", "B_serial"),
                         ("OURS-Block  vs HF-Block ", "B_block"),
                         ("OURS serial vs block    ", "within_ours_serial_vs_block"),
                         ("HF   serial vs block    ", "within_hf_serial_vs_block")):
            for j, x in enumerate(rec[key]):
                print(f"  {tag} j={j} pred={oj['pre'] + j} bitwise={x['bitwise_equal']} "
                      f"max|Δ|={x['max_abs']:.6g} mean={x['mean_abs']:.6g} n_diff={x['n_diff']} "
                      f"argmax {x['argmax_ours']} vs {x['argmax_hf']} same={x['same_argmax']} "
                      f"margin {x['margin_ours']:.6g}/{x['margin_hf']:.6g}")
    elif "B" not in rec:
        rec["B"] = dict(executed=False, reason="有一侧未执行阶段 B（serial/block）")
        print("== 阶段 B：未执行（有一侧未执行）⇒ 不比较、不当作相同 ==")

    print("\n== 差异最早出现在哪个**已执行**的检查点 ==")
    earliest = None
    if have_c1 and not rec["C1_logits"]["bitwise_equal"]:
        earliest = "C1_logits：**第一个已检查且发现差异的检查点**（prompt prefill 之后就已不同；" \
                   "不能说成「差异首次产生于 C1」）"
    elif have_c1 and rec["C1_kv"]["n_bitequal"] != len(rec["C1_kv"]["rows"]):
        earliest = "C1_kv：**第一个已检查且发现差异的检查点**（prompt prefill 后各层 K/V 就已不同）"
    elif have_c2 and not rec["C2_logits"]["bitwise_equal"]:
        earliest = "C2_logits：C1 之后、喂入 48 token 之后出现差异"
    elif have_c2 and rec["C2_kv"]["n_bitequal"] != len(rec["C2_kv"]["rows"]):
        earliest = "C2_kv"
    elif have_ab and (not rec["B_serial"][0]["bitwise_equal"] or not rec["B_block"][0]["bitwise_equal"]):
        earliest = "阶段 B（pre 之后的 4 token 调用）"
    elif have_c1:
        earliest = "全部**已执行**的检查点都逐位相同"
    else:
        earliest = "没有可判定的已执行检查点"
    rec["earliest_difference_stage"] = earliest
    rec["wording_rule"] = ("C1 只能表述为「第一个已检查且发现差异的检查点」；"
                           "未执行的阶段不得当成「相同」。")
    print(f"  ★ {earliest}")
    p = os.path.join(OUT, f"xb_20261001_compare_{oj['run_id']}_{hj['run_id']}_"
                          f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    if os.path.exists(p):                       # ★ 绝不覆盖历史记录：同名则加序号
        k = 1
        while os.path.exists(p.replace(".json", f"_r{k}.json")):
            k += 1
        p = p.replace(".json", f"_r{k}.json")
    rec["artifact_naming"] = ("文件名带**写入时间戳**，并对已存在文件加序号；"
                              "本轮之前的版本用固定名 `..._{ours}_{hf}.json`，"
                              "重跑会覆盖历史派生记录（已在勘误中记录该过程错误）。")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1, default=str)
    print(f"[xbc] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
