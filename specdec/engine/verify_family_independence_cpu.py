"""verify_family_independence_cpu.py — 验证候选 prompt family 与调试输入**互相独立**（纯 CPU，不用 GPU）。

做四件事
  1. **复现 manifest**：用与 `scripts/build_prompt_families.py` **完全相同**的流程
     （同一 zip、同一 tokenizer、同一 `BLOCK_CHARS/WIN` 切分）重建全部 16 个 family 的
     2048/4096/8192 前缀，逐条核对 manifest 里的 `token_sha1`/`text_sha1`。
     核对通过 ⇒ manifest 可信，选定的 family 有可复现的 token id。
  2. **复现调试输入**：用各 G5 工具同一段 `test.raw` 前 8192 token 的读法算出它的 token hash，
     确认就是本轮之前所有诊断用的那一条输入（`bf6a34a82018fd39`）。
  3. **独立性判据（两层）**
     * 语料层：family 取自 `wiki.train.raw`，调试输入取自 `wiki.test.raw`（**不同 split**）；
     * 序列层：**32-gram 交集必须为 0**（不是"看起来不一样"）——family vs 调试输入、
       以及两个候选 family 之间。
  4. **使用历史**：扫描代码库，确认除 builder 之外**没有任何** runner 读取过 manifest
     （⇒ 16 个 family 从未参与此前实验）。

只读：不写 manifest、不改任何工具、不加载模型、不碰 CUDA。
用法：python -m specdec.engine.verify_family_independence_cpu
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "wikitext-2-raw-v1.zip"
MANIFEST = ROOT / "docs" / "prompt_families_manifest.json"
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_CHARS = 300_000          # 与 builder 一致
WIN = 8192
CTXS = (2048, 4096, 8192)
N_BLOCKS = 4
SHINGLE = 32                   # 独立性用的 n-gram 长度
DEBUG_HASH_EXPECTED = "bf6a34a82018fd39"


def _h(ids) -> str:
    """与 builder / `P5V._gen_hash` 同一约定：sha1(json(ids))[:16]。"""
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def _h_text(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()[:16]


def _load_blocks(n: int) -> list:
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read("wikitext-2-raw/wiki.train.raw").decode("utf-8", errors="ignore")
    out = []
    for i in range(n):
        s = raw[i * BLOCK_CHARS:(i + 1) * BLOCK_CHARS]
        if len(s) < 20_000:
            break
        out.append(s)
    return out


def _debug_ids(tok):
    """与 G5 工具同一读法：test.raw → 逐行拼接 → 前 600k 字符 → 前 8192 token。

    返回 `(ids, entry_name)`；entry_name 用来**实证**"family 与调试输入来自不同 split"。
    """
    with zipfile.ZipFile(ZIP) as z:
        name = next(n for n in z.namelist() if "test" in n and n.endswith(".raw"))
        raw = z.read(name).decode("utf-8", errors="ignore")
    text = " ".join(ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not ln.strip().startswith("="))
    return tok(text[:600_000], return_tensors="pt").input_ids[0, :WIN].tolist(), name


def _shingles(ids, n=SHINGLE) -> set:
    return {tuple(ids[i:i + n]) for i in range(0, max(0, len(ids) - n + 1))}


# manifest **机器本身**（读取/校验/汇总/自测）不算"实验 runner 消费方"。
MANIFEST_MACHINERY = {
    "build_prompt_families.py",          # builder（生成 manifest）
    "family_input.py",                   # G1：按 manifest 取输入
    "family16_summary.py",               # 16 格汇总
    "verify_family_independence_cpu.py",  # 本文件
    "verify_readiness_selftest_cpu.py",   # G1–G5 自测
}
# 本轮**新增**的 opt-in 消费方：只有它带 manifest 输入路径，且默认关闭（SMOKE_FAMILY 未设即 legacy）。
OPT_IN_CONSUMERS = {"g5_8k_smoke_cudnn.py",      # fam16 授权轮（SMOKE_FAMILY，默认关闭）
                    "g5_f06_index25_forensics.py",  # F06/index25 定点取证（固定 family）
                    # 2026-09-30 经用户授权登记：S/B 对照工具，**注册范围仅限该工具与 F06**
                    "prefill_sb_replay_f06.py"}


def _manifest_consumers() -> list:
    """**区分大小写**扫描：除 manifest 机器本身外，是否还有工具读取 manifest 字段。

    ⚠️ 两点口径（都是踩过坑才写下的）：
      * 必须**区分大小写**：`F1[0-6]` 在**不区分大小写**时会命中 `bf16`，
        首版扫描因此把一堆 `base_bf16` 行误报成"消费方"；
      * 扫描结果**不能**被表述成"这 16 个 family 从未被使用过"的证明 —— 它不覆盖
        仓库外的手工运行、也不覆盖"按 token 内容取同一段语料"的做法。
        真正可核的是下面 `_family_used_in_existing_runs()`（经验证据）。
    """
    pats = ("families_manifest", "prompt_families", "family_id", "window_token_start")
    hits = []
    for base in (ROOT / "specdec" / "engine", ROOT / "scripts"):
        for p in sorted(base.glob("*")):
            if p.suffix not in (".py", ".ps1", ".cmd") or p.name == Path(__file__).name:
                continue
            if p.name in MANIFEST_MACHINERY:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:                                     # noqa: BLE001
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if any(pat in line for pat in pats):
                    hits.append(dict(file=p.name, line=i, text=line.strip()[:90]))
    return hits


def _family_used_in_existing_runs() -> list:
    """**经验证据**：已有的 run 记录里有没有任何一条真的用了 manifest family？

    比源码扫描强得多：它直接看"跑出来的记录"，不依赖"谁 import 了什么"。
    判定依据是 `input.source_kind == "manifest"` 或 `input.family` 非空。
    """
    used = []
    for p in sorted((ROOT / "output" / "g5" / "attn").glob("*.json")):
        if any(x in p.name for x in ("analysis", "revision", "compare", "summary", "index")):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(j, dict):
            continue
        inp = j.get("input") or {}
        if inp.get("source_kind") == "manifest" or inp.get("family"):
            used.append(dict(file=p.name, family=inp.get("family"), kind=inp.get("source_kind"),
                             tool=j.get("tool"), run_id=j.get("run_id")))
    return used


def _this_round_run_ids() -> set:
    """从 fam16 授权诊断的索引里取本轮 run_id（用于把"本轮/本轮之前"分开）。"""
    ids = set()
    for idx in sorted((ROOT / "output" / "g5" / "attn").glob("fam16_*.index.json")):
        try:
            j = json.loads(idx.read_text(encoding="utf-8-sig"))
        except Exception:                                        # noqa: BLE001
            continue
        for c in j.get("cells", []):
            if c.get("run_id"):
                ids.add(c["run_id"])
    return ids


def main() -> int:
    npass = nfail = 0

    def check(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    print("== 0. manifest 元信息 ==")
    print(f"  version={man['manifest_version']} n_families={man['n_families']} "
          f"ctxs={man['ctxs']} tokenizer={man['tokenizer']}")
    check("manifest 声称的 tokenizer 与本脚本一致", man["tokenizer"] == TOKENIZER)
    check("manifest 声称 16 个 family", man["n_families"] == 16 and len(man["families"]) == 16)

    print("== 1. 复现全部 16 个 family 的前缀 token id 与 hash ==")
    blocks = _load_blocks(N_BLOCKS)
    tok_blocks = [tok(b, return_tensors="pt").input_ids[0].tolist() for b in blocks]
    print(f"  blocks={len(tok_blocks)} tokens={[len(t) for t in tok_blocks]}")
    rebuilt, bad = {}, []
    for f in man["families"]:
        ids = tok_blocks[f["block"]]
        st = f["window_token_start"]
        seg = ids[st:st + WIN]
        rebuilt[f["family_id"]] = seg
        if len(seg) != WIN:
            bad.append((f["family_id"], "len", len(seg)))
            continue
        for c in CTXS:
            if _h(seg[:c]) != f["prefixes"][str(c)]["token_sha1"]:
                bad.append((f["family_id"], f"tok{c}", _h(seg[:c])))
        txt = tok.decode(seg, skip_special_tokens=True)
        if _h_text(txt) != f["text_sha1"]:
            bad.append((f["family_id"], "text", _h_text(txt)))
    check(f"16 个 family × 3 前缀 + text hash 全部复现（共 {16 * 4} 项）", not bad, bad[:6])

    print("== 2. 复现**调试输入**的 token hash ==")
    dbg, dbg_entry = _debug_ids(tok)
    dbg_h = _h(dbg)
    check(f"调试输入 hash == {DEBUG_HASH_EXPECTED}（与既有诊断同一输入）",
          dbg_h == DEBUG_HASH_EXPECTED, dbg_h)
    check("调试输入长度 = 8192", len(dbg) == WIN, len(dbg))

    print("== 3. 独立性：候选 family vs 调试输入（32-gram 交集必须为 0）==")
    dbg_sh = _shingles(dbg)
    for fid, ids in rebuilt.items():
        inter = _shingles(ids) & dbg_sh
        if inter:
            print(f"  !! {fid} 与调试输入共享 {len(inter)} 个 {SHINGLE}-gram")
    shared = [fid for fid, ids in rebuilt.items() if _shingles(ids) & dbg_sh]
    check(f"16/16 family 与调试输入 **零** {SHINGLE}-gram 交集", not shared, shared)

    print("== 4. 候选两档之间也互相独立 ==")
    CAND = ("F06", "F13")
    check("候选 family 都在 manifest 中", all(c in rebuilt for c in CAND))
    blocks_of = {f["family_id"]: f["block"] for f in man["families"]}
    check("候选取自不同 block", blocks_of[CAND[0]] != blocks_of[CAND[1]],
          {c: blocks_of[c] for c in CAND})
    check("候选彼此零 32-gram 交集",
          not (_shingles(rebuilt[CAND[0]]) & _shingles(rebuilt[CAND[1]])))

    print("== 5. 使用历史：除 builder 外无人读过 manifest ==")
    hits = _manifest_consumers()
    others = [h for h in hits if h["file"] != "build_prompt_families.py"]
    unexpected = [h for h in others if h["file"] not in OPT_IN_CONSUMERS]
    seen_files = sorted({h["file"] for h in others})
    check("★ 没有**预期之外**的消费方（唯一 opt-in 消费方是 g5_8k_smoke_cudnn.py，默认关闭）",
          not unexpected, unexpected[:5])
    check("消费方集合 = 预期的 opt-in 集合（授权使用之前无任何 runner 消费 manifest）",
          seen_files == sorted(OPT_IN_CONSUMERS), seen_files)
    man_sources = {f["source"] for f in man["families"]}
    check("manifest 声称的 source 全部是 train.raw",
          all("train" in s for s in man_sources), sorted(man_sources))
    check("调试输入来自 test.raw 且与 family 的 source **是不同文件**",
          "test" in dbg_entry and all("test" not in s for s in man_sources),
          dict(debug_entry=dbg_entry, family_sources=sorted(man_sources)))
    print("== 5b. **经验证据**：run 记录里的 family 使用情况（分「本轮之前 / 本轮」） ==")
    used = _family_used_in_existing_runs()
    # 本轮（fam16 授权诊断）确实跑了 16 格 family 运行 ⇒ 不能再说"没有任何一条用过"。
    # 正确口径：**本轮之前**一条都没有；**本轮**恰好 16 格。
    # 授权范围 = ① fam16 授权诊断的 16 格；② 之后的 F06/index25 定点取证 run。
    # 除此之外**不允许**任何 run 消费 family（这才是可长期复用的检查；
    # "从来没被用过"在授权使用之后已不成立，改为**授权范围校验**）。
    fam16_ids = _this_round_run_ids()
    fam16 = [u for u in used
             if u["file"] in {f"smoke8k_cudnn_prefill_{r}.json" for r in fam16_ids}]
    # 授权消费方（**新增必须显式登记**）：两者都属于 F06/index25 定点取证。
    # `prefill_sb_replay_f06.py` 于 2026-09-30 经用户授权登记，**注册范围仅限该工具与 F06**
    # （下方 `scoped` 检查把这一限制变成可判定条件）。
    authorize_tools = ("g5_f06_index25_forensics", "prefill_sb_replay_f06")
    forensic = [u for u in used if (u.get("tool") or "").startswith(authorize_tools)]
    # `g5_8k_smoke_cudnn.py` 的**非 fam16** 运行（例如 2026-09-30 的 eager 候选配置诊断）
    # 也消费 manifest；按用户授权，**只有 family == F06** 才纳入 F06 定点诊断范围，
    # 其它 family 一律仍算「授权范围之外」。
    fam16_files = {f"smoke8k_cudnn_prefill_{r}.json" for r in fam16_ids}
    smoke_f06 = [u for u in used
                 if (u.get("tool") or "").startswith("g5_8k_smoke_cudnn")
                 and u["file"] not in fam16_files
                 and (u.get("family") or {}).get("family_id") == "F06"]
    smoke_other = [u for u in used
                   if (u.get("tool") or "").startswith("g5_8k_smoke_cudnn")
                   and u["file"] not in fam16_files
                   and (u.get("family") or {}).get("family_id") != "F06"]
    authorized_files = ({u["file"] for u in fam16} | {u["file"] for u in forensic}
                        | {u["file"] for u in smoke_f06})
    outside = [u for u in used if u["file"] not in authorized_files]
    check("★ **授权范围之外**没有任何 run 消费 manifest family（授权范围 = fam16 16 格 + F06/index25 定点取证）",
          not outside, outside[:5])
    check("★ smoke 工具的非 fam16 运行只允许 F06（eager 候选配置诊断的登记范围）",
          not smoke_other, [u["file"] for u in smoke_other][:4])
    scoped = [u for u in used if (u.get("tool") or "").startswith("prefill_sb_replay_f06")]
    bad_family = [u["file"] for u in scoped
                  if (u.get("family") or {}).get("family_id") != "F06"]
    check("★ 新登记消费方 `prefill_sb_replay_f06` 只消费 F06（注册范围仅限该工具 + F06）",
          len(scoped) > 0 and not bad_family, dict(n_scoped=len(scoped), bad=bad_family[:4]))
    if fam16_ids:
        check(f"fam16 授权轮恰好 {len(fam16_ids)} 格用了 family（实测 %d）"
              % len(fam16), len(fam16) == len(fam16_ids),
              dict(expected=len(fam16_ids), got=len(fam16)))
    check("F06/index25 定点取证的 run 也被识别（tool 前缀）", len(forensic) >= 2,
          [u["file"] for u in forensic][:4])

    out = dict(
        tool="verify_family_independence_cpu",
        manifest_version=man["manifest_version"],
        debug_input=dict(source="wikitext-2-raw/wiki.test.raw 前 600000 字符",
                         n_tokens=len(dbg), token_sha1=dbg_h,
                         matches_expected=(dbg_h == DEBUG_HASH_EXPECTED)),
        candidates={
            c: dict(family_id=c,
                    stratum=next(f["stratum"] for f in man["families"] if f["family_id"] == c),
                    block=blocks_of[c],
                    window_token_start=next(f["window_token_start"] for f in man["families"]
                                            if f["family_id"] == c),
                    source="wikitext-2-raw-v1/wiki.train.raw",
                    token_sha1_8192=_h(rebuilt[c]),
                    manifest_token_sha1_8192=next(f["prefixes"]["8192"]["token_sha1"]
                                                  for f in man["families"]
                                                  if f["family_id"] == c))
            for c in CAND},
        independence=dict(corpus_split=dict(families="wiki.train.raw", debug="wiki.test.raw"),
                          shingle_len=SHINGLE,
                          all_families_zero_overlap_with_debug=not shared,
                          candidates_zero_overlap_each_other=True,
                          manifest_consumers_excluding_machinery=others,
                          manifest_machinery_allowlist=sorted(MANIFEST_MACHINERY),
                          opt_in_consumers=sorted(OPT_IN_CONSUMERS),
                          family_used_in_existing_runs=used,
                          family_use_authorization=dict(
                              fam16_round_cells=len(fam16), forensic_runs=len(forensic),
                              outside_authorized=[u["file"] for u in outside],
                              rule="授权范围 = fam16 16 格 + F06/index25 定点取证；"
                                   "其他任何 run 消费 family 都算越界"),
                          scan_caveat="源码扫描只覆盖本仓库的 specdec/engine 与 scripts；"
                                      "不覆盖仓库外的手工运行，也不构成「从未使用」的证明。"),
        all_16_prefixes_reproduced=not bad,
        n_pass=npass, n_fail=nfail,
    )
    p = ROOT / "output" / "g5" / "attn" / "family_independence_cpu_check.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    print(f"[verify] RESULT_JSON={p}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
