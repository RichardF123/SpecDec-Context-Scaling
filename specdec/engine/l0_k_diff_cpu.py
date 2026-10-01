"""l0_k_diff_cpu.py — **零 GPU**：从既有两份 C1 快照里取出第 0 层 K/V 的全部差异清单（只读，不改任何历史产物）。

## 为什么要它
报告 §9.43.5 只给了"L0 K 有 60 个元素不同、L0 V 逐位相同"的**汇总**。要定位就必须知道
**这 60 个差异到底在哪些 (batch, kv_head, token_position, head_dim) 上、值是多少、落在哪个 prefill chunk**，
以及这些位置有没有结构（RoPE 的旋转是**成对**的：`dim` 与 `dim+64` 由同一对 cos/sin 驱动）。

## 纪律
* **固定原始文件路径**（不"自动选最新文件"）；只写**新的**派生 JSON，**绝不覆盖**任何既有产物。
* 存的 fp32 张量是否**精确等于** bf16：先断言 `t.to(bf16).float() == t`，否则"逐位比较"就没有意义。
* 数值间距（ULP）**按 dtype 实测**：给定 bf16 值，取其位模式 +1 得到相邻可表示值，差值即该值处的间距。
  报告同时给**绝对差**与**间距/ULP 比值**；**不**从差异元素数量反推扰动大小（那条推断已在勘误 §11 撤回）。
* 只消费**已消费过**的 F13（不引入任何独立验收样本）。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
# ---- 固定路径：绝对不"选最新" ----
OURS_PT = OUTDIR / "xb_20261001_ours_20261001T132915Z.pt"
HF_PT = OUTDIR / "xb_20261001_hf_20261001T132947Z.pt"
OURS_JSON = OUTDIR / "xb_20261001_ours_20261001T132915Z.json"
HF_JSON = OUTDIR / "xb_20261001_hf_20261001T132947Z.json"
PREFIX_PT = OUTDIR / "f13i50_20261001_capture_ar_20261001T072313Z.pt"
TAG = "l0k_diff"
CHUNK = 2048


def bf16_spacing(v: float) -> float:
    """bf16 在 |v| 处的相邻可表示值间距（符号-幅值位模式 +1）。"""
    t = torch.tensor([v], dtype=torch.float32).to(torch.bfloat16)
    bits = t.view(torch.int16).to(torch.int32) & 0xFFFF
    up = ((bits + 1) & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    return abs(float(up) - float(t.float()))


def bf16_bits(t: torch.Tensor) -> int:
    return int(t.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF)


def _finite_report(name: str, t: torch.Tensor) -> dict:
    fin = torch.isfinite(t)
    return dict(name=name, dtype=str(t.dtype), shape=list(t.shape), stride=list(t.stride()),
                numel=int(t.numel()), n_nonfinite=int((~fin).sum()), n_nan=int(torch.isnan(t).sum()),
                n_inf=int(torch.isinf(t).sum()), min=float(t.min()), max=float(t.max()))


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(tool="l0_k_diff_cpu", run_utc=ts, torch=torch.__version__,
                     semantics="零 GPU 只读派生：既有 C1 快照的第 0 层 K/V 差异清单；不覆盖任何历史产物",
                     fixed_paths=dict(ours_pt=str(OURS_PT), hf_pt=str(HF_PT), ours_json=str(OURS_JSON),
                                      hf_json=str(HF_JSON), prefix_pt=str(PREFIX_PT)))
    for p in (OURS_PT, HF_PT, OURS_JSON, HF_JSON, PREFIX_PT):
        if not p.exists():
            print(f"[l0k] !! 缺少固定路径文件：{p}")
            return 5
    oj = json.load(open(OURS_JSON, encoding="utf-8"))
    hj = json.load(open(HF_JSON, encoding="utf-8"))

    # ---------------- 1) 运行配置与输入来源核实 ----------------
    print("== 1) 运行配置 / 输入来源核实（来自两份 JSON，不用'最新文件'）==")
    cfg_rows = {}
    for tag, j in (("ours", oj), ("hf", hj)):
        cfg_rows[tag] = dict(run_id=j["run_id"], impl=j["impl"], ctx=j["ctx"], chunk=j["chunk"],
                             pre=j["pre"], tokens=j["tokens"], schedule=j["schedule"],
                             precision=j["precision"], versions=j["versions"], model=j["model"],
                             stage=j.get("stage"), family=j["input"]["family"]["family_id"],
                             seq_sha16=j["input"]["seq_sha16"], prefix_len=j["input"]["prefix_len"],
                             prefix_src=j["input"]["prefix_src"],
                             weights=j.get("weights"))
        print(f"  [{tag}] run={j['run_id']} impl={j['impl']} ctx={j['ctx']} chunk={j['chunk']} "
              f"pre={j['pre']} tokens={j['tokens']}")
        print(f"        schedule={j['schedule']} stage={j.get('stage')}")
        print(f"        precision={json.dumps(j['precision'], ensure_ascii=False)}")
        print(f"        seq_sha16={j['input']['seq_sha16']} prefix_len={j['input']['prefix_len']} "
              f"prefix_src={Path(j['input']['prefix_src']).name}")
        if j.get("weights"):
            w = j["weights"]
            print(f"        weights: bitmatch={w['bitmatch']} checked={w['n_checked']} "
                  f"mismatch={len(w['mismatch'])}")
    rec["config"] = cfg_rows
    assert cfg_rows["ours"]["seq_sha16"] == cfg_rows["hf"]["seq_sha16"], "两侧输入序列 hash 不同"
    for k in ("ctx", "chunk", "pre", "tokens", "schedule"):
        assert cfg_rows["ours"][k] == cfg_rows["hf"][k], f"两侧 {k} 不同"
    assert oj["impl"]["attn_implementation"] == "eager(engine)" and \
        hj["impl"]["attn_implementation"] == "eager", "两侧 attn 实现不是 eager"
    print("  [ok] 两侧输入 hash 相同、ctx/chunk/pre/tokens/schedule 相同、都是 eager")

    # 输入序列重算：token ids 必须能复现记录里的 seq_sha16
    from transformers import AutoTokenizer
    from . import family_input as FI
    tok = AutoTokenizer.from_pretrained(oj["model"], trust_remote_code=True)
    ids_list, fam_meta = FI.load_family_ids(tok, "F13", ctx=int(oj["ctx"]))
    cap = torch.load(PREFIX_PT, map_location="cpu")
    prefix = [int(x) for x in cap["prefix_tokens"]][: int(oj["pre"]) - int(oj["ctx"])]
    seq = list(ids_list) + prefix
    sha = hashlib.sha256(json.dumps(seq).encode("utf-8")).hexdigest()[:16]
    print(f"  重算输入：len={len(seq)} sha16={sha}（记录值 {oj['input']['seq_sha16']}）")
    rec["input_recheck"] = dict(len=len(seq), sha16=sha, recorded_sha16=oj["input"]["seq_sha16"],
                                match=bool(sha == oj["input"]["seq_sha16"]),
                                prefix_pt=str(PREFIX_PT), prefix_len=len(prefix))
    if sha != oj["input"]["seq_sha16"]:
        print("[l0k] !! 输入序列无法复现记录 hash ⇒ 停止（后续位置索引会失去意义）")
        return 6
    print("  [ok] 输入序列与记录逐位一致（sha16 相同）")
    seq_t = torch.tensor([seq], dtype=torch.long)
    # 治理：本工具**不读 manifest**，family 元数据来自所读的既有产物（记录下来以便守卫按范围检查）
    rec["input"] = dict(source_kind="derived_from_artifacts", family=oj["input"]["family"],
                        source_json=OURS_JSON.name, source_pt=OURS_PT.name,
                        seq_sha16=sha, prefix_pt=PREFIX_PT.name,
                        note="只读派生；不消费任何新样本；family 字段来自既有产物")

    # ---------------- 2) 两份快照的第 0 层 K/V ----------------
    print("\n== 2) 两份 C1 快照的第 0 层 K/V（形状/dtype/有限性）==")
    o, h = torch.load(OURS_PT, map_location="cpu"), torch.load(HF_PT, map_location="cpu")
    kA, kB = o["c1_kv"][0]["k"], h["c1_kv"][0]["k"]
    vA, vB = o["c1_kv"][0]["v"], h["c1_kv"][0]["v"]
    rec["shape"] = dict(ours_k=list(kA.shape), hf_k=list(kB.shape))
    assert tuple(kA.shape) == tuple(kB.shape) == (1, 2, int(oj["pre"]) - 48, 128), \
        f"第 0 层 K 形状不是 (1,2,8192,128)：{tuple(kA.shape)} vs {tuple(kB.shape)}"
    for nm, t in (("ours_k", kA), ("hf_k", kB), ("ours_v", vA), ("hf_v", vB)):
        r = _finite_report(nm, t)
        print(f"  {nm}: shape={r['shape']} stride={r['stride']} dtype={r['dtype']} "
              f"nonfinite={r['n_nonfinite']} (nan={r['n_nan']}, inf={r['n_inf']}) "
              f"range=[{r['min']:.6g}, {r['max']:.6g}]")
        rec.setdefault("tensors", {})[nm] = r
    rec["exact_bf16"] = {nm: bool(torch.equal(t.to(torch.bfloat16).float(), t))
                         for nm, t in (("ours_k", kA), ("hf_k", kB), ("ours_v", vA), ("hf_v", vB))}
    assert all(rec["exact_bf16"].values()), "存的 fp32 张量不是精确的 bf16 值 ⇒ 逐位比较无意义"
    print(f"  [ok] 四个张量都是**精确的 bf16 值**（fp32 存储，无额外舍入）⇒ 逐位比较成立")
    assert all(rec["tensors"][n]["n_nonfinite"] == 0 for n in
               ("ours_k", "hf_k", "ours_v", "hf_v")), "存在 NaN/Inf"

    print("\n== 3) V 是否逐位相同；K 差异计数 ==")
    rec["v_compare"] = dict(bitwise_equal=bool(torch.equal(vA, vB)),
                            n_diff=int((vA != vB).sum()), numel=int(vA.numel()),
                            max_abs=float((vA.float() - vB.float()).abs().max()))
    rec["k_compare"] = dict(bitwise_equal=bool(torch.equal(kA, kB)), numel=int(kA.numel()),
                            max_abs=float((kA.float() - kB.float()).abs().max()),
                            mean_abs=float((kA.float() - kB.float()).abs().mean()),
                            n_diff=int((kA != kB).sum()))
    print(f"  V: 逐位相同={rec['v_compare']['bitwise_equal']} 不同元素={rec['v_compare']['n_diff']}"
          f"/{rec['v_compare']['numel']} max|Δ|={rec['v_compare']['max_abs']:.6g}")
    print(f"  K: 逐位相同={rec['k_compare']['bitwise_equal']} 不同元素={rec['k_compare']['n_diff']}"
          f"/{rec['k_compare']['numel']} max|Δ|={rec['k_compare']['max_abs']:.6g} "
          f"mean|Δ|={rec['k_compare']['mean_abs']:.6g}")
    assert rec["v_compare"]["bitwise_equal"], "V 不再是逐位相同 ⇒ 与报告 §9.43.5 的前提不一致"

    # ---------------- 4) 60 个差异的位置与数值 ----------------
    print("\n== 4) K 差异的完整清单（(batch, kv_head, token_position, head_dim) + 值 + 间距）==")
    neq = (kA != kB)
    idx = torch.nonzero(neq, as_tuple=False)
    diffs = []
    for r in idx.tolist():
        b, hd, p, d = r
        va, vb = float(kA[b, hd, p, d]), float(kB[b, hd, p, d])
        sp = bf16_spacing(va)
        diffs.append(dict(batch=int(b), kv_head=int(hd), token_position=int(p), head_dim=int(d),
                          chunk=int(p) // CHUNK, token_id=int(seq_t[0, p]),
                          ours=va, hf=vb, abs_diff=abs(va - vb), bf16_spacing=sp,
                          diff_in_spacing=(abs(va - vb) / sp if sp else None),
                          ours_bits=bf16_bits(torch.tensor(va)), hf_bits=bf16_bits(torch.tensor(vb)),
                          dim_pair_index=int(d) % 64))
    rec["diffs"] = diffs
    for x in diffs:
        print(f"  b={x['batch']} kvh={x['kv_head']} pos={x['token_position']:>5} dim={x['head_dim']:>3} "
              f"chunk={x['chunk']} tok={x['token_id']:>6} | ours={x['ours']:+.9g} hf={x['hf']:+.9g} "
              f"|Δ|={x['abs_diff']:.6g} spacing={x['bf16_spacing']:.6g} "
              f"|Δ|/spacing={x['diff_in_spacing']:.4g} bits {x['ours_bits']:04x}->{x['hf_bits']:04x}")

    # ---------------- 5) 结构汇总 ----------------
    import collections
    by_chunk = collections.Counter(x["chunk"] for x in diffs)
    by_head = collections.Counter(x["kv_head"] for x in diffs)
    by_dim = collections.Counter(x["head_dim"] for x in diffs)
    by_pos = collections.Counter(x["token_position"] for x in diffs)
    by_pair = collections.Counter(x["dim_pair_index"] for x in diffs)
    both = [d for d in by_pair if (d + 64) in by_pair]
    rec["structure"] = dict(
        n_diff=len(diffs), by_chunk=dict(sorted(by_chunk.items())), by_kv_head=dict(sorted(by_head.items())),
        distinct_positions=len(by_pos), distinct_dims=len(by_dim), distinct_dim_pairs=len(by_pair),
        dim_pairs_with_both_members=len(both), by_dim=dict(sorted(by_dim.items())),
        spacing_values=sorted({x["bf16_spacing"] for x in diffs}),
        diff_in_spacing_values=sorted({round(x["diff_in_spacing"], 6) for x in diffs}),
        max_abs_diff=max(x["abs_diff"] for x in diffs),
        positions_with_multiple=dict(sorted((p, c) for p, c in by_pos.items() if c > 1)))
    print("\n== 5) 结构汇总 ==")
    print(f"  差异元素 {len(diffs)} 个；分布 chunk={dict(sorted(by_chunk.items()))} "
          f"kv_head={dict(sorted(by_head.items()))}")
    print(f"  不同位置 {len(by_pos)} 个、不同 head_dim {len(by_dim)} 个、"
          f"成对出现的 (dim,dim+64) 组合 {len(both)}/{len(by_pair)}")
    print(f"  bf16 间距取值集合={rec['structure']['spacing_values']}；"
          f"|Δ|/间距 取值集合={rec['structure']['diff_in_spacing_values']}")
    print(f"  涉及位置的 token 数（>1 表示同一位置多个维度差）: "
          f"{len(rec['structure']['positions_with_multiple'])} 个位置")
    print(f"  ★ chunk 覆盖：{sorted(by_chunk)}（>1 个 chunk 则 L0 重放必须覆盖这些 chunk）")

    p = OUTDIR / f"{TAG}_{oj['run_id']}_{hj['run_id']}_{ts}.json"
    if p.exists():
        p = p.with_name(p.stem + "_r1.json")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[l0k] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
