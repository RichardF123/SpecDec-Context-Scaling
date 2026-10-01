"""xb_compare_cpu.py — 跨实现（OURS vs HF）同路径对照的**只读**比较（纯 CPU）。

阶段顺序（**差异最早出现在哪一段**由此判定）：
  C1 = prompt 的 4×2048 分块 prefill 之后（cache len 8192）
  C2 = 再喂 48 个 token 的一个 chunk 之后（cache len 8240）
  Serial / Block = 从 C2 的独立深拷贝出发，4 个 token 的单步 / 块前向
比较内容：logits（逐位/最大/平均绝对差、argmax、margin、并列集合）与各层 K/V（逐位 + 差异摘要）。
"""
from __future__ import annotations

import glob
import json
import os

import torch

OUT = "output/g5/attn"


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
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()), n_elem=int(d.numel()),
                argmax_ours=fa["argmax"], argmax_hf=fb["argmax"],
                same_argmax=bool(fa["argmax"] == fb["argmax"]),
                margin_ours=fa["margin"], margin_hf=fb["margin"],
                tied_ours=fa["n_tied_max"], tied_hf=fb["n_tied_max"])


def kv_cand(ka, kb, label):
    """逐层 K/V：逐位相等数 + 最大绝对差（只报摘要）。"""
    rows = []
    for i in range(min(len(ka), len(kb))):
        for kk in ("k", "v"):
            x, y = ka[i][kk].float(), kb[i][kk].float()
            if tuple(x.shape) != tuple(y.shape):
                rows.append(dict(layer=i, attr=kk, shape_mismatch=[list(x.shape), list(y.shape)]))
                continue
            d = (x - y).abs()
            rows.append(dict(layer=i, attr=kk, bitequal=bool(torch.equal(x, y)),
                             max_abs=float(d.max()), mean_abs=float(d.mean()),
                             n_diff=int((d > 0).sum())))
    return dict(stage=label, rows=rows,
                n_layers=len(rows) // 2, n_bitequal=sum(1 for r in rows if r.get("bitequal")))


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
    assert rec["input"]["seq_sha16_ours"] == rec["input"]["seq_sha16_hf"], "两侧输入序列 hash 不一致"

    print("== 阶段 A：C1（prompt 4×2048 分块 prefill，len 8192）==")
    rec["C1_logits"] = cand(o["c1_logits"], h["c1_logits"], "C1_logits")
    rec["C1_kv"] = kv_cand(o["c1_kv"], h["c1_kv"], "C1_kv")
    rec["C1_len"] = dict(ours=oj["C1"]["cache_len"], hf=hj["C1"]["cache_len"])
    c1 = rec["C1_logits"]
    print(f"  logits: bitwise={c1['bitwise_equal']} max|Δ|={c1['max_abs']:.6g} "
          f"mean={c1['mean_abs']:.6g} n_diff={c1['n_diff']} | argmax {c1['argmax_ours']} vs "
          f"{c1['argmax_hf']} | margin {c1['margin_ours']:.6g} vs {c1['margin_hf']:.6g}")
    print(f"  KV: 逐位相等 {rec['C1_kv']['n_bitequal']}/{len(rec['C1_kv']['rows'])} 个张量；"
          f"cache len {rec['C1_len']}")

    print("== 阶段 A：C2（+48 token 一个 chunk，len 8240）==")
    rec["C2_logits"] = cand(o["c2_logits"], h["c2_logits"], "C2_logits")
    rec["C2_kv"] = kv_cand(o["c2_kv"], h["c2_kv"], "C2_kv")
    rec["C2_len"] = dict(ours=oj["C2"]["cache_len"], hf=hj["C2"]["cache_len"])
    c2 = rec["C2_logits"]
    print(f"  logits: bitwise={c2['bitwise_equal']} max|Δ|={c2['max_abs']:.6g} "
          f"mean={c2['mean_abs']:.6g} n_diff={c2['n_diff']} | argmax {c2['argmax_ours']} vs "
          f"{c2['argmax_hf']} | margin {c2['margin_ours']:.6g} vs {c2['margin_hf']:.6g}")
    print(f"  KV: 逐位相等 {rec['C2_kv']['n_bitequal']}/{len(rec['C2_kv']['rows'])} 个张量；"
          f"cache len {rec['C2_len']}")

    print("== 阶段 B：同路径跨实现（serial 第 j 步 ↔ block 第 j 行，均预测 pre+j+1）==")
    rec["B_serial"] = [cand(o["serial_rows"][j], h["serial_rows"][j], f"serial_j{j}")
                       for j in range(len(oj["tokens"]))]
    rec["B_block"] = [cand(o["block_rows"][j], h["block_rows"][j], f"block_j{j}")
                      for j in range(len(oj["tokens"]))]
    rec["within_ours_serial_vs_block"] = [cand(o["serial_rows"][j], o["block_rows"][j], f"ours_j{j}")
                                          for j in range(len(oj["tokens"]))]
    rec["within_hf_serial_vs_block"] = [cand(h["serial_rows"][j], h["block_rows"][j], f"hf_j{j}")
                                        for j in range(len(oj["tokens"]))]
    for tag, key in (("OURS-Serial vs HF-Serial", "B_serial"),
                     ("OURS-Block  vs HF-Block ", "B_block"),
                     ("OURS serial vs block    ", "within_ours_serial_vs_block"),
                     ("HF   serial vs block    ", "within_hf_serial_vs_block")):
        for j, x in enumerate(rec[key]):
            print(f"  {tag} j={j} pred={8264 - 24 + j} bitwise={x['bitwise_equal']} "
                  f"max|Δ|={x['max_abs']:.6g} mean={x['mean_abs']:.6g} n_diff={x['n_diff']} "
                  f"argmax {x['argmax_ours']} vs {x['argmax_hf']} same={x['same_argmax']} "
                  f"margin {x['margin_ours']:.6g}/{x['margin_hf']:.6g}")

    print("\n== 差异最早出现在哪个已检查阶段 ==")
    order = [("C1_logits", rec["C1_logits"]), ("C1_kv", None), ("C2_logits", rec["C2_logits"]),
             ("C2_kv", None), ("B_serial_j0", rec["B_serial"][0]), ("B_block_j0", rec["B_block"][0])]
    earliest = None
    if not rec["C1_logits"]["bitwise_equal"]:
        earliest = "C1_logits（prompt prefill 之后就已不同）"
    elif rec["C1_kv"]["n_bitequal"] != len(rec["C1_kv"]["rows"]):
        earliest = "C1_kv（prompt prefill 后的各层 K/V 就已不同）"
    elif not rec["C2_logits"]["bitwise_equal"]:
        earliest = "C2_logits（喂入 48 token 之后不同）"
    elif rec["C2_kv"]["n_bitequal"] != len(rec["C2_kv"]["rows"]):
        earliest = "C2_kv"
    elif not rec["B_serial"][0]["bitwise_equal"] or not rec["B_block"][0]["bitwise_equal"]:
        earliest = "阶段 B（pre=8240 的 4 token 调用）"
    else:
        earliest = "全部已检查阶段都逐位相同"
    rec["earliest_difference_stage"] = earliest
    print(f"  ★ {earliest}")
    p = os.path.join(OUT, f"xb_20261001_compare_{oj['run_id']}_{hj['run_id']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1, default=str)
    print(f"[xbc] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
