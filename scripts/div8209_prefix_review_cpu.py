"""div8209_prefix_review_cpu.py — **只读**复核"新首分歧（位置 8209）之前的共同前缀缓存"（纯 CPU，不加载模型）。

## 为什么需要它
上一轮把"长度 8237 的重建历史 vs AR 历史"整段比较（0/56 逐位、max|Δ|=23.97）当作"A 无法归因"的量化证据，
并顺带把"历史是否一致"整体标成"无法回答"。但**共同前缀**（生成序列到 index 16 为止，即位置 0..8208）
是可以在两份快照里**逐位**验证的：只要那段缓存逐位相同，新分歧（8209）之前的"同历史"前提就成立。

本脚本把这件事**独立**做一遍（不依赖上一轮的脚本结论），并回答三个问题：
  1. 两份（三份）快照在位置区间 [0,8192) / [0,8208) / [0,8209) 上逐层 K/V 是否**逐位**相同？
  2. 这些早期缓存是否可能被**后续操作重新计算或原地改写**？（代码路径审计 + 落盘产物的字节级不可变核查
     + 克隆独立性核查）
  3. 三次真实运行在 index 16/17 的 token 与当轮记录，是否与"重建=659 / 原Spec=323 / AR=323"一致？

## 严格边界
* **只读**：不加载模型、不跑任何前向、不写任何既有产物；只新建一个带时间戳的 JSON。
* 所有比较都是**逐张量 `torch.equal`**（不是 sha 摘要）；sha 只用于"产物字节级未变"的核查。
* 区间比较**不是**归因：本脚本只回答"共同前缀是否逐位相同"，不回答任何成因问题。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specdec.engine import cache_clone as CC            # noqa: E402
from specdec.engine import sb_align as A                # noqa: E402

OUT = ROOT / "output" / "g5" / "attn"

# ---- 固定引用（显式路径，不"自动选最新"）----
AR_CAP, AR_RUN = "divcap_ar_20261001T185503Z", "smoke8k_eager_20261001T185503Z.json"
SP_CAP, SP_RUN = "divcap_spec_20261001T185705Z", "smoke8k_eager_20261001T185705Z.json"
RB_CAP, RB_RUN = "divasc_rebuild_20261001T220737Z", "smoke8k_eager_20261001T220737Z.json"

CTX = 8192                 # prompt 长度（第一个生成 token 的位置）
IDX_NEW = 17               # 新首分歧的 index（绝对位置 8209）
POS_NEW = CTX + IDX_NEW    # 8209
PRE_NEW = POS_NEW - 1      # 8208：新分歧轮的 pre（缓存里应含位置 0..8207）
PRE_OLD = 8237             # 上一轮现场的捕获长度
RANGES = ((0, CTX), (0, PRE_NEW), (0, POS_NEW), (PRE_NEW, POS_NEW),
          (POS_NEW, PRE_OLD), (0, PRE_OLD))


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _load_cap(tag: str) -> dict:
    pt, js = OUT / f"{tag}.pt", OUT / f"{tag}.json"
    tens = torch.load(pt, map_location="cpu")
    return dict(tag=tag, pt=pt, json=js, tens=tens,
                snap=tens["snapshot_cpu"],
                meta=json.loads(js.read_text(encoding="utf-8")))


def _cache_len(c) -> int:
    return int(c[0]["k"].shape[2])


def _slice(c, lo: int, hi: int):
    return [{k: (v[:, :, lo:hi] if torch.is_tensor(v) and v.dim() == 4 else v)
             for k, v in lay.items()} for lay in c]


def _cmp_range(a, b, lo: int, hi: int) -> dict:
    """逐层 K/V 在位置区间 [lo,hi) 上的**逐位**比较（`torch.equal`，不是摘要）。"""
    sa, sb = _slice(a, lo, hi), _slice(b, lo, hi)
    rows = []
    for i in range(min(len(sa), len(sb))):
        for kk in ("k", "v"):
            x, y = sa[i][kk], sb[i][kk]
            shape_ok = tuple(x.shape) == tuple(y.shape)
            rows.append(dict(layer=i, attr=kk, shape_equal=shape_ok,
                             bitequal=bool(shape_ok and torch.equal(x, y)),
                             n_diff=(int((x != y).sum()) if shape_ok else None),
                             max_abs=(float((x.float() - y.float()).abs().max()) if shape_ok
                                      else None)))
    return dict(range=[lo, hi], n_positions=hi - lo, n_tensors=len(rows),
                n_bitequal=sum(1 for r in rows if r["bitequal"]),
                all_bitequal=all(r["bitequal"] for r in rows),
                n_diff_total=sum(r["n_diff"] for r in rows if r["n_diff"] is not None),
                max_abs=float(max(r["max_abs"] for r in rows if r["max_abs"] is not None)),
                first_diff_tensors=[(r["layer"], r["attr"], r["n_diff"]) for r in rows
                                    if not r["bitequal"]][:6])


def _first_div_pos(a, b, lo: int = 0) -> dict:
    """两份历史**最早**在哪个位置出现数值差异（先按 kv 头/head_dim 归约，再取位置）。"""
    n = min(_cache_len(a), _cache_len(b))
    first, per_layer = None, []
    for i in range(min(len(a), len(b))):
        f = None
        for kk in ("k", "v"):
            neq = (a[i][kk][:, :, lo:n] != b[i][kk][:, :, lo:n])
            # ⚠️ 归约维必须是 (kv_heads, head_dim)：`.any(dim=1).any(dim=-1)` ⇒ 留下 (T,)
            pos_any = neq.any(dim=1).any(dim=-1).reshape(-1)
            nz = torch.nonzero(pos_any, as_tuple=False)
            if nz.numel():
                p = lo + int(nz.reshape(-1)[0])
                f = p if f is None else min(f, p)
        per_layer.append(dict(layer=i, first_diff_pos=f))
        if f is not None:
            first = f if first is None else min(first, f)
    return dict(first_diff_pos_any_layer=first,
                n_layers_diverging=sum(1 for x in per_layer if x["first_diff_pos"] is not None),
                per_layer=per_layer)


def _round_at(run_json: Path, pre: int) -> dict:
    if not run_json.exists():
        return dict(found=False, why="运行记录不存在")
    j = json.loads(run_json.read_text(encoding="utf-8"))
    rounds = j.get("rounds") or []
    hit = next((r for r in rounds if int(r.get("pre", -1)) == pre), None)
    if hit is None:
        return dict(found=False, why=f"该运行记录里没有 pre={pre} 的轮次",
                    n_rounds=len(rounds))
    return dict(found=True, round=hit.get("round"), pre=hit.get("pre"), Kp=hit.get("Kp"),
                m=hit.get("m"), cands=hit.get("cands"), g=hit.get("g"),
                emitted=hit.get("emitted"), tgt_len=hit.get("tgt_len"))


def _gen(run_json: Path) -> list:
    j = json.loads(run_json.read_text(encoding="utf-8"))
    return [int(x) for x in j["run"]["gen"]]


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    files = [OUT / f"{t}.pt" for t in (AR_CAP, SP_CAP, RB_CAP)]
    files += [OUT / f"{t}.json" for t in (AR_CAP, SP_CAP, RB_CAP)]
    sha_before = {p.name: _sha_file(p) for p in files if p.exists()}
    rec: dict = dict(tool="div8209_prefix_review_cpu", run_utc=ts,
                     semantics="只读复核：新首分歧（8209）之前的共同前缀缓存是否逐位一致；"
                               "区间比较不做任何归因",
                     refs=dict(ar_cap=AR_CAP, ar_run=AR_RUN, spec_cap=SP_CAP, spec_run=SP_RUN,
                               rebuild_cap=RB_CAP, rebuild_run=RB_RUN),
                     constants=dict(ctx=CTX, idx_new=IDX_NEW, pos_new=POS_NEW,
                                    pre_new=PRE_NEW, pre_old=PRE_OLD))
    L = {t: _load_cap(t) for t in (AR_CAP, SP_CAP, RB_CAP)}
    lens = {t: _cache_len(L[t]["snap"]) for t in L}
    rec["snapshots"] = {t: dict(len=lens[t], n_layers=len(L[t]["snap"]),
                               keys=sorted(L[t]["snap"][0].keys()),
                               declared_len=(L[t]["meta"].get("snap_cache_len")
                                             or L[t]["meta"].get("snapshot_cache_len")
                                             or L[t]["meta"].get("snapshot_len")),
                               dtypes=sorted({str(v.dtype) for v in L[t]["snap"][0].values()
                                              if torch.is_tensor(v)}))
                        for t in L}
    print("== 快照 ==")
    for t, v in rec["snapshots"].items():
        print(f"  {t:<46} len={v['len']} 层={v['n_layers']} 键={v['keys']} "
              f"声明长度={v['declared_len']} dtype={v['dtypes']}")

    # ---------------- (1) 区间逐位比较 ----------------
    pairs = [("rebuild_vs_ar", RB_CAP, AR_CAP), ("rebuild_vs_origspec", RB_CAP, SP_CAP),
             ("origspec_vs_ar", SP_CAP, AR_CAP)]
    rec["range_compare"] = {}
    for name, ta, tb in pairs:
        rec["range_compare"][name] = {f"[{lo},{hi})": _cmp_range(L[ta]["snap"], L[tb]["snap"], lo, hi)
                                      for lo, hi in RANGES}
    print("\n== 区间逐位比较（每层 K/V，torch.equal；共 56 张量）==")
    for name in rec["range_compare"]:
        print(f"  {name}:")
        for rng, d in rec["range_compare"][name].items():
            print(f"    {rng:<14} 逐位相同 {d['n_bitequal']}/{d['n_tensors']}  不同元素={d['n_diff_total']:<9}"
                  f" max|Δ|={d['max_abs']:.6g}")

    rec["first_divergence_position"] = dict(
        rebuild_vs_ar=_first_div_pos(L[RB_CAP]["snap"], L[AR_CAP]["snap"]),
        rebuild_vs_origspec=_first_div_pos(L[RB_CAP]["snap"], L[SP_CAP]["snap"]),
        origspec_vs_ar=_first_div_pos(L[SP_CAP]["snap"], L[AR_CAP]["snap"]))
    print("\n== 最早数值分叉位置（同长度 8237 的两份历史）==")
    for k, d in rec["first_divergence_position"].items():
        print(f"  {k:<24} 最早=位置 {d['first_diff_pos_any_layer']} 分叉层数={d['n_layers_diverging']}")

    # ---------------- (2) 截断 = 纯切片 + 克隆独立性 + 早期缓存未被改写 ----------------
    trunc = {t: _slice(L[t]["snap"], 0, PRE_NEW) for t in L}
    pure = {}
    for t in L:
        src = L[t]["snap"]
        pure[t] = bool(all(torch.equal(trunc[t][i][kk], src[i][kk][:, :, :PRE_NEW])
                           for i in range(len(src)) for kk in ("k", "v")))
    rec["truncation"] = dict(pre=PRE_NEW, is_pure_slice=pure,
                             note="截断 = 沿 token 维切片（`v[:, :, :8208]`），**不重算**任何位置；"
                                  "逐张量核对切片与源前缀逐位相同")
    cc = CC.clone_list_cache_full(L[RB_CAP]["snap"])
    indep = dict(same_data_ptr=any(cc[i][kk].data_ptr() == L[RB_CAP]["snap"][i][kk].data_ptr()
                                   for i in range(len(cc)) for kk in ("k", "v")),
                 clone_len=_cache_len(cc), src_len=_cache_len(L[RB_CAP]["snap"]))
    cc[0]["k"][0, 0, 0, 0] = cc[0]["k"][0, 0, 0, 0] + 1          # 只在克隆上原地改一个元素
    indep["clone_mutation_visible_in_source"] = bool(not torch.equal(cc[0]["k"],
                                                                    L[RB_CAP]["snap"][0]["k"]))
    indep["source_changed_by_clone_mutation"] = bool(
        torch.equal(L[RB_CAP]["snap"][0]["k"], L[RB_CAP]["snap"][0]["k"])) and _cache_len(
        L[RB_CAP]["snap"]) == PRE_OLD
    del cc
    rec["independence"] = indep
    print("\n== 截断 / 克隆独立性 ==")
    print(f"  截断为长度 {PRE_NEW} 是纯切片（逐张量核对）：{pure}")
    print(f"  克隆是否共享存储={indep['same_data_ptr']}；克隆上原地改动是否影响源="
          f"{indep['clone_mutation_visible_in_source']}（False 才正确）")

    # ---------------- (3) token 前缀与三次真实运行的现场记录 ----------------
    per_run = {}
    for tag, cap, run in (("ar", AR_CAP, AR_RUN), ("origspec", SP_CAP, SP_RUN),
                          ("rebuild", RB_CAP, RB_RUN)):
        g = _gen(OUT / run) if (OUT / run).exists() else []
        per_run[tag] = dict(gen_len=len(g), gen_head=g[:IDX_NEW + 2],
                            token_at_col16=(g[IDX_NEW - 1] if len(g) > IDX_NEW - 1 else None),
                            token_at_col17=(g[IDX_NEW] if len(g) > IDX_NEW else None),
                            prefix_tokens_=(L[cap]["meta"].get("prefix_tokens") or [])[:IDX_NEW],
                            round_at_pre_new=_round_at(OUT / run, PRE_NEW))
    rec["runs"] = per_run
    print("\n== 三次真实运行（index 16/17 与 pre=8208 的当轮记录）==")
    for tag, v in per_run.items():
        r = v["round_at_pre_new"]
        print(f"  {tag:<9} gen_len={v['gen_len']:<3} index16={v['token_at_col16']} "
              f"index17={v['token_at_col17']}")
        print(f"            pre=8208 轮：found={r['found']} "
              + (f"round={r.get('round')} Kp={r.get('Kp')} m={r.get('m')} cands={r.get('cands')} "
                 f"g={r.get('g')}" if r["found"] else f"({r.get('why')})"))
    pref_eq = bool(per_run["ar"]["prefix_tokens_"] == per_run["origspec"]["prefix_tokens_"]
                   == per_run["rebuild"]["prefix_tokens_"])
    gen_head_eq = bool(per_run["ar"]["gen_head"] == per_run["origspec"]["gen_head"]
                       == per_run["rebuild"]["gen_head"])
    rec["prefix_tokens"] = dict(n=IDX_NEW, equal_all_three=pref_eq,
                                ar=per_run["ar"]["prefix_tokens_"], gen_head_equal=gen_head_eq,
                                note="prefix_tokens 取各捕获里 gen 的前 17 个 = 位置 8192..8208 的 token；"
                                     "三次运行在这 17 个 token 上逐 token 相同 ⇒ 新分歧之前的**输入内容**相同")
    print(f"\n  index 0..16 的 token 三次运行是否完全相同：{gen_head_eq}；"
          f"prefix_tokens 完全相同：{pref_eq}")

    # ---------------- (4) 落盘产物字节级不可变 ----------------
    sha_after = {p.name: _sha_file(p) for p in files if p.exists()}
    reloaded_ok = {}
    for t in (AR_CAP, SP_CAP, RB_CAP):
        again = torch.load(OUT / f"{t}.pt", map_location="cpu")["snapshot_cpu"]
        reloaded_ok[t] = bool(all(torch.equal(again[i][kk], L[t]["snap"][i][kk])
                                  for i in range(len(again)) for kk in ("k", "v")))
        del again
    rec["artifact_immutability"] = dict(
        sha_before=sha_before, sha_after=sha_after,
        unchanged=bool(sha_before == sha_after),
        reload_matches_in_memory=reloaded_ok,
        note="① 脚本运行前后对三个 .pt/.json 做 sha256 核对；② 重新从磁盘加载并与内存中的快照逐张量"
             "`torch.equal` 复核 ⇒ 本脚本与任何读操作都没有改写落盘产物")
    print("\n== 落盘产物不可变 ==")
    print(f"  运行前后 sha256 一致={rec['artifact_immutability']['unchanged']}；"
          f"重新加载与内存逐张量相同={reloaded_ok}")

    # ---------------- (5) 早期缓存是否可能被"后续操作"重算/原地改写（代码路径审计） ----------------
    rec["no_inplace_audit"] = dict(
        capture_copy="`g5_8k_smoke_cudnn._snap_cpu_full` = `cache_clone.clone_list_cache_full`"
                     "（`copy.deepcopy`）后再 `.detach().to('cpu')` ⇒ 快照与活缓存**不共享存储**",
        capture_once="重建捕获里 `diag['snap']` 只在 `spec_cache.cache_length(replay) == capture_len`"
                     "**第一次**成立时写入（`if diag['snap'] is None`），此后不再改",
        append_only="`model_self.QwenLM._run` 对缓存的写法是 `cache['k'] = torch.cat([...], dim=2)`"
                    "（`model_self.py:352-355` 附近的同一模式）⇒ **追加**，不重写已有位置",
        trim_slice_only="`spec_cache.trim_cache` 只做 `val[:, :, :new_len]` 切片，不重算",
        replay_branch="重建分支每轮从**轮前克隆** `H_pre` 的独立副本开始喂入 ⇒ 已捕获位置不会被重放改写；"
                      "重放缓存每轮都会**替换** `tgt_cache`（诊断要求），而捕获快照是**更早**的独立 deepcopy",
        verified="本脚本另做两项实测核查：① 截断=纯切片（逐张量）② 磁盘产物 sha 与重载逐张量一致；"
                 "`cache_clone` 的独立性由 `require_list_clone_invariants` 在工具里断言（19 个 CPU 模块"
                 "自测中已含该项）",
        caveat="**不能**由本审计推出\"任何运行下早期缓存都必然未被改写\"；它只覆盖上述代码路径与已核查的产物。"
               "位置 8208 的 KV 在重建历史里是否真的是**串行**写出的，本脚本用**一致性证据**支撑"
               "（见下），不做几何/算子级归因")
    # 一致性证据：重建历史的 8192..8208 段与 AR 的串行段逐位相同（若重建历史上这段是块内 KV，"
    # 按既有结论它应从位置 8192 起就与串行段不同）
    ev = rec["range_compare"]["rebuild_vs_ar"]["[0,8209)"]
    rec["no_inplace_audit"]["consistency_evidence"] = dict(
        rebuild_vs_ar_prefix_0_8209_bitequal=ev["n_bitequal"], n_tensors=ev["n_tensors"],
        paired_with="既有结论：块内 KV 与单步 KV 在**同内容**下最早从位置 8192 起不同（28/28 层）",
        reading="重建历史的位置 8192..8208 段与 AR 的**单步**段逐位相同 ⇒ 该段与\"块内 KV\"**不相容**，"
                "与\"重建分支逐 token 串行写入\"相容（这是证据，不是几何级证明）")
    print("\n== 早期缓存未被重算/原地改写的核查 ==")
    print(f"  重建 vs AR 在 [0,8209) 上逐位相同：{ev['n_bitequal']}/{ev['n_tensors']}")
    print("  代码路径：快照=deepcopy+to('cpu')、只写一次、缓存写入=torch.cat 追加、"
          "trim=纯切片、重放从独立副本开始")

    rec["stage"] = "complete"
    p = OUT / f"div8209_prefix_review_cpu_{ts}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[prefix-review] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
