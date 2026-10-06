"""build_formal_families.py — 构建**正式评估**用的 16 个未使用窗口 manifest（`formal-v1`）。

## 为什么需要
正式评估要求"**未用于调试、未参与 pilot**"的输入（pilot = F09/F03/F11/F04）。
旧 manifest `docs/prompt_families_manifest.json`（`v1-draft`，16 family）中已有 10 个窗口被消费 ⇒
**本脚本从"窗口池中未被消费的候选"里按**预先写死的确定性规则**选出**恰好 16 个，
写入**新文件** `docs/prompt_families_formal_manifest.json`；**不修改** `v1-draft`。

## 规则（先写死，再执行；不含任何模型推理或结果驱动筛选）
**R0 语料与窗口**（与 `build_prompt_families.py` **同一规则**，逐字复用其函数）：
  `data/wikitext-2-raw-v1.zip` 的 `wikitext-2-raw/wiki.train.raw`；每 `BLOCK_CHARS=300000` 字符一个 block；
  块内按 `WIN=8192` token **对齐切分** `[i·8192,(i+1)·8192)`；窗口不跨 block；不插合成填充；
  tokenizer = `Qwen/Qwen2.5-1.5B-Instruct`（`local_files_only=True`）。⇒ 池 = 4 block × 8 窗 = **32**。

**R1 排除集（身份级，不可协商）**：`v1-draft` 中**已被任何工作消费**的 10 个窗口 ——
  F13（定位/历史）、F01/F02/F08/F10（协议轮）、F06（定点诊断）、F09/F03/F11/F04（Stage 2b；其中
  **F09/F03/F11/F04 即 pilot**）。⇒ 候选 = 32 − 10 = **22**（脚本会断言该数）。
  排除按 **(block, window_index)** 身份执行，并**同时**核对 8192 前缀的 `token_sha1` 不复用（哈希级对账见校验脚本）。

**R2 候选集合内分层（相对排序；标签带 `c` 后缀）**：对 **22 个候选**按 `features()` 排序
  （全部 tie-break 均为显式、确定性的）：
  * `S1c_low_entropy` = `distinct_token_ratio` **最小 4**；tie → `markup_ratio` 升序 → (block, window) 升序；
  * `S2c_high_markup` = 余下中 `markup_ratio` **最大 4**；tie → `distinct_token_ratio` 降序 → (block, window) 升序；
  * `S3c_prose`       = 余下中 `letter_ratio − 2·markup_ratio` **最大 4**；tie → `letter_ratio` 降序 → (block, window) 升序；
  * `S4c_mixed`       = 其余 **10**。
  ⚠️ 这些标签是**集合内相对排序**：**不得**与 `v1-draft` 的 `S1..S4` 直接比较，**不得**宣称"恢复了原分布"。

**R3 选取（恰好 16；每 block 上限 5）**：
  1. 先取 `S1c+S2c+S3c` 全部 **12** 个；
  2. 再从 `S4c_mixed` 按 `(block 升序, window 升序)` **贪心**补 4 个：仅当加入后该 block 计数 ≤ **5** 时加入，直到 **16**；
  3. 若上限导致 **无法凑满 16** ⇒ **报错退出（exit 3）并保留证据**；**不**偷偷换规则、**不**降到 15、**不**放宽到 6。

**R4 编号**：最终按 `(block, window)` 升序编号 `FF01..FF16`（`FF` 前缀**避免**与历史 `F01..F16` 在 tag/产物中混淆）。

**R5 内容去重（只记录、不假设）**：位置不重叠**不等于**文本内容绝不重复。脚本对
  (a) 16 个正式窗口之间、(b) 正式窗口 vs 被排除的 10 个窗口，比较 `text_sha1`，把结果写进 manifest 的
  `content_dedup_check`（不据此筛选样本）。

用法：`python scripts/build_formal_families.py --print`（只打印） / 无参数（写文件）
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
import build_prompt_families as B                                        # noqa: E402

V1 = ROOT / "docs" / "prompt_families_manifest.json"
OUT = ROOT / "docs" / "prompt_families_formal_manifest.json"
PER_BLOCK_CAP = 5
N_SELECT = 16
PILOT = ("F09", "F03", "F11", "F04")
CONSUMED = {
    "F13": "定位/历史（div8209 系列）",
    "F01": "协议轮（serial oracle Stage 2）",
    "F02": "协议轮（serial oracle Stage 2）",
    "F08": "协议轮（serial oracle Stage 2）",
    "F10": "协议轮（serial oracle Stage 2）",
    "F06": "定点诊断（L27/index25 取证）",
    "F09": "Stage 2b + B6 的 8K 融合复核 + **pilot**",
    "F03": "Stage 2b + **pilot**",
    "F11": "Stage 2b + **pilot**",
    "F04": "Stage 2b + **pilot**",
}
STRATA = ("S1c_low_entropy", "S2c_high_markup", "S3c_prose", "S4c_mixed")


def _key(c) -> tuple:
    return (c["block"], c["win"])


def build_pool() -> list:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B.TOKENIZER, local_files_only=True)
    blocks = B.load_block_texts("wikitext-2-raw/wiki.train.raw", n_blocks=4)
    pool = []
    for bi, btext in enumerate(blocks):
        ids = tok(btext, return_tensors="pt").input_ids[0].tolist()
        for i in range(len(ids) // B.WIN):
            seg = ids[i * B.WIN:(i + 1) * B.WIN]
            if len(seg) < B.WIN:
                continue
            text = tok.decode(seg, skip_special_tokens=True)
            pool.append(dict(block=bi, win=i, token_start=i * B.WIN, n_tokens=len(seg),
                             ids=seg, text=text, feats=B.features(text, seg),
                             token_sha1_8192=hashlib.sha1(
                                 json.dumps(seg[:8192]).encode()).hexdigest()[:16],
                             text_sha1=hashlib.sha1(text.encode()).hexdigest()[:16]))
    return pool


def assign_strata(cands: list) -> dict:
    """R2：候选集合内相对排序（每个 tie-break 都显式）。"""
    out: dict = {}
    rest = sorted(cands, key=lambda c: (c["feats"]["distinct_token_ratio"],
                                        c["feats"]["markup_ratio"], _key(c)))
    for c in rest[:4]:
        out[_key(c)] = STRATA[0]
    rest = [c for c in rest if _key(c) not in out]
    rest = sorted(rest, key=lambda c: (-c["feats"]["markup_ratio"],
                                       -c["feats"]["distinct_token_ratio"], _key(c)))
    for c in rest[:4]:
        out[_key(c)] = STRATA[1]
    rest = [c for c in rest if _key(c) not in out]
    rest = sorted(rest, key=lambda c: (-(c["feats"]["letter_ratio"] - 2 * c["feats"]["markup_ratio"]),
                                       -c["feats"]["letter_ratio"], _key(c)))
    for c in rest[:4]:
        out[_key(c)] = STRATA[2]
    for c in cands:
        out.setdefault(_key(c), STRATA[3])
    return out


def select(cands: list, strata: dict) -> tuple:
    """R3：先取 S1c+S2c+S3c，再用 (block, window) 升序贪心从 S4c 补齐（每 block ≤ 5）。"""
    picked = [c for c in cands if strata[_key(c)] != STRATA[3]]
    per_block = Counter(c["block"] for c in picked)
    for c in sorted([c for c in cands if strata[_key(c)] == STRATA[3]], key=_key):
        if len(picked) >= N_SELECT:
            break
        if per_block[c["block"]] >= PER_BLOCK_CAP:
            continue
        picked.append(c)
        per_block[c["block"]] += 1
    return sorted(picked, key=_key), dict(per_block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args()

    v1 = json.loads(V1.read_text(encoding="utf-8"))
    v1_by_key = {(f["block"], f["window_index"]): f for f in v1["families"]}
    pilot_ids = set(PILOT)
    consumed_keys = {k: f for k, f in v1_by_key.items() if f["family_id"] in CONSUMED}
    if not set(PILOT) <= set(CONSUMED):
        print(f"!! pilot {sorted(pilot_ids)} 不是已消费 family 的子集 ⇒ 规则前提不成立")
        return 3
    print(f"[R1] v1-draft families={len(v1['families'])}；已消费={len(consumed_keys)}（含 pilot {sorted(pilot_ids)}）")

    allw = build_pool()
    print(f"[R0] 窗口池 = {len(allw)}（每 block "
          f"{dict(Counter(c['block'] for c in allw))}）")
    cands = [c for c in allw if _key(c) not in consumed_keys]
    print(f"[R1] 候选（未消费）= {len(cands)}")
    if len(cands) != 22:
        print(f"!! 候选数 {len(cands)} ≠ 22（预期）⇒ 保留证据并停止，不偷偷改规则")
        return 3

    strata = assign_strata(cands)
    dist = Counter(strata[_key(c)] for c in cands)
    print(f"[R2] 候选分层（集合内相对）= {dict(dist)}")
    picked, per_block = select(cands, strata)
    print(f"[R3] 选中 = {len(picked)}；每 block 计数 = {per_block}")
    if len(picked) != N_SELECT:
        print(f"!! 每 block 上限 {PER_BLOCK_CAP} 下无法凑满 {N_SELECT}（得 {len(picked)}）"
              f" ⇒ 保留证据、报错退出；不放宽上限、不降到 15、不换规则")
        return 3
    if any(v > PER_BLOCK_CAP for v in per_block.values()):
        print(f"!! block 配额越界 {per_block} ⇒ 报错退出")
        return 3

    fams = []
    for fi, c in enumerate(picked, start=1):
        pre = {}
        for ctx in B.CTXS:
            seg = c["ids"][:ctx]
            pre[str(ctx)] = dict(n_tokens=len(seg),
                                 token_sha1=hashlib.sha1(json.dumps(seg).encode()).hexdigest()[:16])
        fams.append(dict(
            family_id=f"FF{fi:02d}",
            formal_stratum=strata[_key(c)],
            source="wikitext-2-raw-v1/wiki.train.raw",
            block=c["block"], window_index=c["win"], window_token_start=c["token_start"],
            n_tokens_window=c["n_tokens"], prefixes=pre, features=c["feats"],
            text_sha1=c["text_sha1"], text_head=c["text"][:100].replace("\n", " "),
        ))

    sel_keys = {_key(c) for c in picked}
    dup_sel = [h for h, n in Counter(f["text_sha1"] for f in fams).items() if n > 1]
    dup_vs_excl = sorted({f["text_sha1"] for f in fams}
                         & {v1_by_key[k]["text_sha1"] for k in consumed_keys if k in v1_by_key})
    pool_dup = sorted({h for h, n in Counter(c["text_sha1"] for c in allw).items() if n > 1})

    man = dict(
        manifest_version="formal-v1",
        purpose=("正式评估输入：**未用于调试、未参与 pilot** 的 16 个窗口。"
                 "与 `v1-draft` 并存；**不修改**旧文件。"),
        source_manifest=str(V1.relative_to(ROOT)).replace("\\", "/"),
        source_manifest_sha256=hashlib.sha256(V1.read_bytes()).hexdigest(),
        builder="scripts/build_formal_families.py",
        rules="见 builder 文件头 R0–R5（先写死再执行；无模型推理、无结果驱动筛选）",
        corpus="data/wikitext-2-raw-v1.zip :: wikitext-2-raw/wiki.train.raw",
        tokenizer=B.TOKENIZER,
        ctxs=list(B.CTXS),
        corpus_rule=f"每 {B.BLOCK_CHARS} 字符一个 block；块内按 {B.WIN} token 对齐切分；不跨 block；不插合成填充",
        selection_rule=(f"候选 = 池 − 已消费身份（{len(consumed_keys)} 个）；集合内分层 S1c/S2c/S3c 各 4、其余 S4c；"
                        f"先取 S1c+S2c+S3c（12），再按 (block, window) 升序从 S4c 贪心补齐至 {N_SELECT}；"
                        f"每 block 上限 {PER_BLOCK_CAP}；凑不满则报错退出"),
        strata_rule=("集合内相对排序：S1c=distinct_token_ratio 最小 4；S2c=余下 markup_ratio 最大 4；"
                     "S3c=余下 letter_ratio−2·markup_ratio 最大 4；S4c=其余"),
        strata_note=("⚠️ `c` 后缀表示**候选/集合内**相对排序：**不得**与 `v1-draft` 的 `S1..S4` 直接比较，"
                     "**不得**宣称恢复原分布或与原分层可比"),
        per_block_cap=PER_BLOCK_CAP,
        windows_pool=len(allw), candidates=len(cands), n_families=len(fams),
        excluded=[dict(window=list(k), family_id=v1_by_key[k]["family_id"], reason=CONSUMED[v1_by_key[k]["family_id"]])
                  for k in sorted(consumed_keys)],
        pilot_family_ids=list(PILOT),
        pilot_note="pilot 的 4 个 family 已被 R1 排除 ⇒ 正式集合与 pilot **身份级不相交**",
        positional_non_overlap=("同 block 内为 [i·8192,(i+1)·8192) 对齐切分 ⇒ 位置互不重叠；"
                               "block 间为不相交字符区间；family 取自 train.raw、质量窗取自 test.raw"),
        content_dedup_check=dict(
            note=("**位置不重叠 ≠ 文本内容绝不重复**：这里只**测量**内容重复，不据此筛选样本"),
            selected_pair_duplicate_text_sha1=dup_sel,
            selected_vs_excluded_duplicate_text_sha1=dup_vs_excl,
            whole_pool_duplicate_text_sha1=pool_dup,
            n_selected=len(fams)),
        no_token_ids_stored=("`prefixes` 只存 `n_tokens` 与 `token_sha1`（**不存 token IDs**）⇒ "
                             "运行器必须按 source/block/window/tokenizer 重建再核对"),
        families=fams)

    print(f"\n{'fam':<6}{'stratum(c)':<20}{'blk':>4}{'win':>5}{'tok':>7}{'distinct':>10}{'markup':>9}  head")
    for f in fams:
        ft = f["features"]
        print(f"{f['family_id']:<6}{f['formal_stratum']:<20}{f['block']:>4}{f['window_index']:>5}"
              f"{f['n_tokens_window']:>7}{ft['distinct_token_ratio']:>10.3f}{ft['markup_ratio']:>9.4f}"
              f"  {f['text_head'][:40]}")
    print(f"\n[分层分布] {dict(Counter(f['formal_stratum'] for f in fams))}")
    print(f"[每 block] {dict(Counter(f['block'] for f in fams))}")
    print(f"[内容去重] 选中内部重复={dup_sel or '无'}；与排除集重复={dup_vs_excl or '无'}；"
          f"整池重复组={len(pool_dup)}（仅记录）")
    if not a.do_print:
        OUT.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
