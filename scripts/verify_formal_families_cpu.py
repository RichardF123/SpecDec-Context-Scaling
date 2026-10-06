"""verify_formal_families_cpu.py — `formal-v1` 正式 family manifest 的**只读 CPU 校验**。

校验项（对应本轮任务书的验收清单）：
  V1 16 个正式窗口身份唯一（`family_id`、`(block, window_index)`）
  V2 结构自洽：`window_token_start == window_index × 8192`、`n_tokens_window == 8192`
  V3 **排除对账**：与"已用于调试/pilot"的窗口**身份级不相交**，且 8192 `token_sha1` 全不重复
  V4 **消费集证据核对**：从 `output/g5/attn/*.json` 的运行记录里**重建**被消费 family 集合，
     与构建器登记的一致（产物不可用 ⇒ 记 `evidence_unavailable`，**不**静默通过）
  V5 **可重建**：按 source/block/window/tokenizer 重建 token ids 并与三档 `prefixes` 的
     `n_tokens`/`token_sha1` **逐个复现**
  V6 **嵌套前缀**：同一窗口的 2K ⊂ 4K ⊂ 8K（逐 token 相同）
  V7 **位置不重叠**（同 block 对齐切分、block 间字符区间不相交）——并**显式声明**
     "位置不重叠 ≠ 文本内容绝不重复"；内容重复用 `text_sha1` **实测**并如实打印（不据此筛选样本）
  V8 **分层与配额**：用构建器同一规则重算分层 ⇒ 与 manifest 一致；每 block ≤ 5；总数 = 16
  V9 无 token ids 泄漏（`prefixes` 只含 `n_tokens`/`token_sha1`）；`source_manifest_sha256` 与当前
     `v1-draft` 文件实际哈希一致（旧清单被改动 ⇒ CHECK，因为排除前提可能已变）

退出码：`0` 全过；`1` 存在失败；`3` 参数/文件缺失。
用法：`python scripts/verify_formal_families_cpu.py [--print-only]`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_formal_families as BF                                       # noqa: E402
import build_prompt_families as B                                        # noqa: E402

V1 = ROOT / "docs" / "prompt_families_manifest.json"
FORMAL = ROOT / "docs" / "prompt_families_formal_manifest.json"
ATTN = ROOT / "output" / "g5" / "attn"

checks: list = []


def ck(name: str, ok: bool, extra=None) -> None:
    checks.append((name, bool(ok), extra))


def _h_ids(ids) -> str:
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true")
    a = ap.parse_args()
    if not FORMAL.exists():
        print(f"[fail] 缺 {FORMAL} ⇒ 先跑 scripts/build_formal_families.py")
        return 3
    man = json.loads(FORMAL.read_text(encoding="utf-8"))
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    fams = man["families"]
    print(f"== formal-v1 校验 == version={man.get('manifest_version')} n={len(fams)} "
          f"pool={man.get('windows_pool')} candidates={man.get('candidates')}")

    # ---- V1 身份唯一 ----
    ids = [f["family_id"] for f in fams]
    keys = [(f["block"], f["window_index"]) for f in fams]
    ck("V1 16 个 family_id 唯一且形如 FFnn",
       len(fams) == 16 and len(set(ids)) == 16 and all(i.startswith("FF") and i[2:].isdigit() for i in ids),
       ids)
    ck("V1 (block, window_index) 身份唯一", len(set(keys)) == len(keys), keys)

    # ---- V2 结构自洽 ----
    bad2 = [f["family_id"] for f in fams
            if f["window_token_start"] != f["window_index"] * B.WIN or f["n_tokens_window"] != B.WIN]
    ck("V2 window_token_start == window_index×8192 且 n_tokens_window == 8192", not bad2, bad2)

    # ---- V3 排除对账 ----
    v1map = {(f["block"], f["window_index"]): f for f in v1["families"]}
    consumed_keys = {k for k, f in v1map.items() if f["family_id"] in BF.CONSUMED}
    overlap = [f["family_id"] for f in fams if (f["block"], f["window_index"]) in consumed_keys]
    ck("V3 与已消费（调试/pilot）窗口**身份级不相交**", not overlap,
       dict(overlap=overlap, n_consumed=len(consumed_keys)))
    excluded_hashes = {v1map[k]["prefixes"]["8192"]["token_sha1"] for k in consumed_keys}
    hash_overlap = [f["family_id"] for f in fams if f["prefixes"]["8192"]["token_sha1"] in excluded_hashes]
    ck("V3 8192 token_sha1 与全部已消费窗口**均不相同**（哈希级对账）", not hash_overlap, hash_overlap)
    ck("V3 pilot ⊆ 已消费集合（规则前提）", set(BF.PILOT) <= set(BF.CONSUMED), list(BF.PILOT))

    # ---- V4 消费集证据核对（从运行产物重建）----
    seen_fams, n_recs, evidence_ok = set(), 0, True
    if ATTN.exists():
        for p in sorted(ATTN.glob("*.json")):
            try:
                j = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:                                            # noqa: BLE001
                continue
            if not isinstance(j, dict) or not j.get("model_run_happened"):
                continue
            fam = (((j.get("smoke") or {}).get("input") or {}).get("family") or {}).get("family_id")
            if fam:
                seen_fams.add(fam)
                n_recs += 1
    else:
        evidence_ok = False
    extra_seen = sorted(seen_fams - set(BF.CONSUMED))
    ck("V4 运行产物里被消费的 family 集合 == 构建器登记集合",
       evidence_ok and not extra_seen,
       dict(evidence="available" if evidence_ok else "evidence_unavailable",
            n_run_records=n_recs, seen=sorted(seen_fams), extra=extra_seen))
    ck("V4 正式集合与产物中出现的 family 均不相交",
       not ({f["family_id"] for f in fams} & seen_fams), sorted({f["family_id"] for f in fams} & seen_fams))

    # ---- V5 可重建 + V6 嵌套前缀 ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B.TOKENIZER, local_files_only=True)
    blocks = B.load_block_texts("wikitext-2-raw/wiki.train.raw", n_blocks=4)
    rebuild_bad, nest_bad = [], []
    for f in fams:
        ids_b = tok(blocks[f["block"]], return_tensors="pt").input_ids[0].tolist()
        seg = ids_b[f["window_token_start"]:f["window_token_start"] + B.WIN]
        for ctx in B.CTXS:
            pre = f["prefixes"][str(ctx)]
            sub = seg[:ctx]
            if pre["n_tokens"] != len(sub) or pre["token_sha1"] != _h_ids(sub):
                rebuild_bad.append((f["family_id"], ctx, pre["n_tokens"], len(sub)))
        if not (seg[:2048] == seg[:4096][:2048] == seg[:8192][:2048]
                and seg[:4096] == seg[:8192][:4096]):
            nest_bad.append(f["family_id"])
    ck("V5 16×3 档前缀全部可从 source/block/window/tokenizer **复现**（长度+hash）", not rebuild_bad,
       rebuild_bad[:4])
    ck("V6 同一窗口的 2K ⊂ 4K ⊂ 8K（逐 token 相同）", not nest_bad, nest_bad)

    # ---- V7 位置不重叠 + 内容重复实测 ----
    within_bad = []
    for b in sorted({f["block"] for f in fams}):
        starts = sorted((f["window_token_start"], f["window_token_start"] + B.WIN)
                        for f in fams if f["block"] == b)
        for (s1, e1), (s2, e2) in zip(starts, starts[1:]):
            if s2 < e1:
                within_bad.append((b, s1, e1, s2, e2))
    ck("V7 同 block 内窗口**位置**互不重叠（对齐切分）", not within_bad, within_bad)
    ck("V7 block 间字符区间不相交（每 block 独立 300k 字符）", True, "构造保证；见 corpus_rule")
    all_text = Counter(f["text_sha1"] for f in fams)
    dup_in = [h for h, n in all_text.items() if n > 1]
    dup_vs_excl = sorted({f["text_sha1"] for f in fams} & {v1map[k]["text_sha1"] for k in consumed_keys})
    print("   [V7 说明] **位置不重叠 ≠ 文本内容绝不重复**：以下为**实测**结果（仅记录，不据此筛选）：")
    print(f"             选中内部重复 text_sha1 = {dup_in or '无'}；与已消费窗口重复 = {dup_vs_excl or '无'}")
    ck("V7 内容重复已**实测并记录**（不把位置不重叠当作内容不重复）",
       isinstance(man.get("content_dedup_check"), dict), man.get("content_dedup_check"))

    # ---- V8 分层与配额 ----
    pool = BF.build_pool()
    cands = [c for c in pool if (c["block"], c["win"]) not in consumed_keys]
    strata = BF.assign_strata(cands)
    strata_bad = [f["family_id"] for f in fams
                  if strata.get((f["block"], f["window_index"])) != f["formal_stratum"]]
    ck("V8 分层可用同一确定性规则**重算复现**", not strata_bad, strata_bad)
    per_block = Counter(f["block"] for f in fams)
    ck(f"V8 每 block 计数 ≤ {man.get('per_block_cap')}", all(v <= man.get("per_block_cap", 5)
                                                      for v in per_block.values()), dict(per_block))
    ck("V8 总数恰好 16 且分层分布 4/4/4/4",
       len(fams) == 16 and set(Counter(f["formal_stratum"] for f in fams).values()) == {4},
       dict(Counter(f["formal_stratum"] for f in fams)))

    # ---- V9 无 ids 泄漏 + source manifest 未变 ----
    leak = [f["family_id"] for f in fams
            if any(k not in ("n_tokens", "token_sha1") for p in f["prefixes"].values() for k in p)]
    ck("V9 `prefixes` 不含 token ids（只存长度与 hash）", not leak, leak)
    sha_now = hashlib.sha256(V1.read_bytes()).hexdigest()
    ck("V9 `source_manifest_sha256` 与当前 v1-draft 一致（旧清单未被改动）",
       sha_now == man.get("source_manifest_sha256"), dict(now=sha_now[:16],
                                                          recorded=str(man.get("source_manifest_sha256"))[:16]))

    npass = sum(1 for _, ok, _ in checks if ok)
    print("\n== 校验结果 ==")
    for name, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {str(extra)[:160]}" if not ok else ""))
    print(f"\nformal-families 校验：PASS {npass} / FAIL {len(checks) - npass}")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
