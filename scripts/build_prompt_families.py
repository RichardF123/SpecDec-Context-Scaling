"""build_prompt_families.py — 生成 >=16 个 nested-prefix prompt family 的版本化 manifest。

设计要点（对应协议 Amendment v7.2 / 提案 §3）
  1. **nested prefix**：一个 family = 一段 8192 token 的真实文本；2K/4K/8K 三个条件是它的
     前缀（2048 ⊂ 4096 ⊂ 8192），**同一个续写任务**，从而把"长度效应"与"内容效应"分开。
  2. **不插合成填充**：全部取自真实语料 `wikitext-2-raw-v1/wiki.train.raw`。
     **质量窗用 test.raw（G3 的 40 窗），family 用 train.raw ⇒ 两者天然不重叠**（比在同一文件里
     划保留区更干净，也不用维护偏移表）。
  3. **窗口规则可复现**：每个 block 内按 `[i*8192, (i+1)*8192)` **对齐切分**、互不重叠；
     block 之间独立（每 300k 字符单独 tokenize，避免超长编码告警；窗口不跨 block）。
  4. **预登记分层**（先验假设，pilot 负责证伪）：按窗口文本特征分 4 层，各 4 个
        S1_low_entropy  distinct-token 比例最低（模板化/重复文本 -> 预期接受率高）
        S2_high_markup  wikitext 标记符号比例最高（表格/信息框 -> 预期接受率低）
        S3_prose        纯散文（字母比例高、标记少）
        S4_mixed        其余
     假设的接受率排序：**S1 > S3 > S4 > S2**（写进 manifest 的 expected_acceptance_rank，
     pilot 后回填实测值；若被证伪，也是结论）。
  5. 每个 family 记录：实际 token 数、三段前缀的 token-id 哈希、文本哈希、特征值、分层。

用法：python scripts/build_prompt_families.py            # 写 docs/prompt_families_manifest.json
      python scripts/build_prompt_families.py --print    # 只打印表格，不写文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "wikitext-2-raw-v1.zip"
OUT = ROOT / "docs" / "prompt_families_manifest.json"
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"

BLOCK_CHARS = 300_000       # 每个 block 单独 tokenize（约 70k token，避免超长编码告警）
WIN = 8192                  # 一个 family 的完整窗口长度
CTXS = (2048, 4096, 8192)   # 三个嵌套前缀长度
N_FAMILIES = 16
MANIFEST_VERSION = "v1-draft"


def load_block_texts(entry: str, n_blocks: int) -> list[str]:
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read(entry).decode("utf-8", errors="ignore")
    blocks = []
    for i in range(n_blocks):
        s = raw[i * BLOCK_CHARS:(i + 1) * BLOCK_CHARS]
        if len(s) < 20_000:
            break
        blocks.append(s)
    return blocks


def features(text: str, tok_ids: list[int]) -> dict:
    n = max(len(text), 1)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    marks = sum(c in "|{}[]<>=" for c in text)
    nonascii = sum(ord(c) > 127 for c in text)
    distinct = len(set(tok_ids)) / max(len(tok_ids), 1)
    return {
        "letter_ratio": round(letters / n, 4),
        "digit_ratio": round(digits / n, 4),
        "markup_ratio": round(marks / n, 4),
        "nonascii_ratio": round(nonascii / n, 4),
        "distinct_token_ratio": round(distinct, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    blocks = load_block_texts("wikitext-2-raw/wiki.train.raw", n_blocks=4)
    print(f"[corpus] train.raw blocks={len(blocks)} chars={[len(b) for b in blocks]}",
          flush=True)

    cands = []
    for bi, btext in enumerate(blocks):
        ids = tok(btext, return_tensors="pt").input_ids[0].tolist()
        n_win = len(ids) // WIN
        print(f"[block {bi}] tokens={len(ids)} -> {n_win} aligned windows", flush=True)
        for i in range(n_win):
            seg = ids[i * WIN:(i + 1) * WIN]
            if len(seg) < WIN:
                continue
            text = tok.decode(seg, skip_special_tokens=True)
            cands.append(dict(block=bi, win=i, token_start=i * WIN, n_tokens=len(seg),
                              text=text, ids=seg, feats=features(text, seg)))
    print(f"[cands] {len(cands)} candidates", flush=True)
    if len(cands) < N_FAMILIES:
        print("!! 候选不足，需要更多 block")
        return 2

    by_distinct = sorted(cands, key=lambda c: c["feats"]["distinct_token_ratio"])
    by_markup = sorted(cands, key=lambda c: -c["feats"]["markup_ratio"])
    by_prose = sorted(cands, key=lambda c: -(c["feats"]["letter_ratio"]
                                             - 2 * c["feats"]["markup_ratio"]))
    strata: dict[int, str] = {}

    def assign(order, name, want):
        got = 0
        for c in order:
            k = id(c)
            if k in strata:
                continue
            strata[k] = name
            got += 1
            if got == want:
                return

    assign(by_distinct, "S1_low_entropy", 4)
    assign(by_markup, "S2_high_markup", 4)
    assign(by_prose, "S3_prose", 4)
    for c in cands:
        if id(c) not in strata:
            strata[id(c)] = "S4_mixed"

    rank = {"S1_low_entropy": 1, "S3_prose": 2, "S4_mixed": 3, "S2_high_markup": 4}
    per_stratum: dict[str, int] = {}
    per_block: dict[int, int] = {}
    cap = max(1, N_FAMILIES // max(len(blocks), 1))     # 每 block 均摊，避免集中在语料前段
    picked = []
    for c in sorted(cands, key=lambda c: (rank[strata[id(c)]], c["block"], c["win"])):
        s = strata[id(c)]
        if per_stratum.get(s, 0) >= 4 or per_block.get(c["block"], 0) >= cap:
            continue
        per_stratum[s] = per_stratum.get(s, 0) + 1
        per_block[c["block"]] = per_block.get(c["block"], 0) + 1
        picked.append(c)
        if len(picked) == N_FAMILIES:
            break
    if len(picked) < N_FAMILIES:      # 约束太紧时放宽 block 上限补齐
        for c in sorted(cands, key=lambda c: (rank[strata[id(c)]], c["block"], c["win"])):
            if c in picked or per_stratum.get(strata[id(c)], 0) >= 4:
                continue
            per_stratum[strata[id(c)]] = per_stratum.get(strata[id(c)], 0) + 1
            picked.append(c)
            if len(picked) == N_FAMILIES:
                break
    picked = sorted(picked, key=lambda c: (c["block"], c["win"]))

    families = []
    for fi, c in enumerate(picked, start=1):
        ids = c["ids"]                 # 用对齐切分的原始 id（不重新编码，避免往返漂移）
        pre = {}
        for ctx in CTXS:
            seg = ids[:ctx]
            pre[str(ctx)] = dict(
                n_tokens=len(seg),
                token_sha1=hashlib.sha1(json.dumps(seg).encode()).hexdigest()[:16])
        s = strata[id(c)]
        families.append(dict(
            family_id=f"F{fi:02d}",
            stratum=s,
            expected_acceptance_rank=rank[s],
            source="wikitext-2-raw-v1/wiki.train.raw",
            block=c["block"], window_index=c["win"],
            window_token_start=c["token_start"], n_tokens_window=c["n_tokens"],
            text_sha1=hashlib.sha1(c["text"].encode()).hexdigest()[:16],
            prefixes=pre,
            features=c["feats"],
            text_head=c["text"][:100].replace("\n", " "),
        ))

    per_block_realized: dict[str, int] = {}
    for f in families:
        k = str(f["block"])
        per_block_realized[k] = per_block_realized.get(k, 0) + 1

    manifest = dict(
        manifest_version=MANIFEST_VERSION,
        tokenizer=TOKENIZER,
        builder="scripts/build_prompt_families.py",
        corpus_rule=(f"train.raw 每 {BLOCK_CHARS} 字符一个 block，块内按 {WIN} token 对齐切分；"
                     f"窗口不跨 block；不插合成填充"),
        selection_rule=(f"按 (stratum 先验接受率序, block, window) 排序取数；每 stratum 上限 4、"
                        f"每 block 上限 {cap}；**联合约束不可行时放宽 block 上限补齐 16**"
                        f"（因部分 block 不含某层文本，实测分布见 realized_per_block）"),
        realized_per_block=per_block_realized,
        quality_corpus="wikitext-2-raw-v1/wiki.test.raw（G3 的 40 个质量窗）",
        non_overlap_argument="family 取自 train.raw，质量窗取自 test.raw ⇒ 无重叠",
        n_families=len(families), ctxs=list(CTXS),
        strata={s: sum(1 for f in families if f["stratum"] == s)
                for s in sorted(rank, key=lambda k: rank[k])},
        caveat=("markup_ratio 的层间分离较弱（S2 约 0.004–0.005 vs S3 约 0.002），"
                "因此 S2 更应视为**排序**而非独立文本体制；pilot 后按实测接受率重新分层"
                "（若与先验假设冲突，按 G3 的 pilot-driven amendment 流程登记）"),
        preregistered_hypothesis=("接受率排序 S1_low_entropy > S3_prose > S4_mixed > "
                                  "S2_high_markup（pilot 证伪）"),
        families=families)

    print(f"\n{'fam':<5}{'stratum':<18}{'blk':>4}{'win':>5}{'tok':>7}"
          f"{'distinct':>10}{'markup':>8}{'digit':>8}  head")
    for f in families:
        ft = f["features"]
        print(f"{f['family_id']:<5}{f['stratum']:<18}{f['block']:>4}"
              f"{f['window_index']:>5}{f['n_tokens_window']:>7}"
              f"{ft['distinct_token_ratio']:>10.3f}{ft['markup_ratio']:>8.3f}"
              f"{ft['digit_ratio']:>8.3f}  {f['text_head'][:38]}")
    print(f"\nstrata: {manifest['strata']}")
    print(f"per-block: {per_block_realized}")
    if not args.do_print:
        OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
