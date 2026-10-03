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
import re
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
                    "prefill_sb_replay_f06.py",
                    # 2026-10-01 经用户授权登记：F13/index50 定点取证，**注册范围仅限该工具与 F13**
                    "f13_i50_capture.py",
                    # 2026-10-01 经用户授权登记：UCG 原型诊断，**注册范围仅限该工具与 F13**
                    "unified_geometry_run.py",
                    # 2026-10-01 经用户授权登记：HF 外部对照，**注册范围仅限该工具与 F13**
                    "hf_serial_block_probe.py",
                    # 2026-10-01 经用户授权登记：自研 vs HF 同路径对齐，**注册范围仅限该工具与 F13**
                    "xb_probe.py",
                    # 2026-10-01 经用户授权登记：L0 K 有界定位（第 0 层重放），**注册范围仅限该工具与 F13**
                    "l0_k_replay.py",
                    # 2026-10-01 经用户授权登记：L0 K 差异清单（只读派生，只引用既有 F13 产物）
                    "l0_k_diff_cpu.py",
                    # 2026-10-01 经用户授权登记：AR/Spec 一致性比较（只读派生，只引用既有 F13 产物）
                    "ar_spec_consistency_cpu.py",
                    # 2026-10-03 经**本轮任务**（分层正确性协议的未见数据验证）登记：
                    # 决策位置 top-1/top-2/margin 提取 —— **注册范围仅限协议预登记的 4 个 family**
                    # （F01/F02/F08/F10；由下方 scoped 检查把这一限制变成可判定条件）
                    "correctness_protocol_margins.py",
                    # 2026-10-04 登记：**Stage 2b 工具**（判决/逐阶段核对/预检/自测）。
                    # 它们会**读取**记录里的 manifest 字段（`family_id` / `window_token_start` /
                    # `token_sha1` 等）与 manifest 本身（预检复现 token hash）⇒ 按 `_manifest_consumers()`
                    # 的口径属于消费方，必须显式登记（登记 + 下方 scoped 检查；**不放宽**任何既有检查）。
                    # 它们**不产生任何 run**，因此**不**影响 run 级授权检查。
                    "stage2b_verdict.py", "stage2b_preflight_cpu.py",
                    "stage2b_verdict_selftest_cpu.py"}
                    # 注：`stage_check.py` **不**登记 —— 它不引用 manifest 字段（`_manifest_consumers()`
                    # 的检测器扫不到它），而"消费方集合 == OPT_IN_CONSUMERS"是**等式**检查，
                    # 多登记一个检测不到的文件会直接判失败（首版即如此）。

# 2026-10-04：**Stage 2b 候选 family**（尚未授权运行；D-3 待批准）。
# 语义边界（必须写清，避免"登记=已授权"的误读）：
#   * 这里只是把这些 id 从"**未知**字面量"降级为"**已知候选**"，使**工具源码**可以通过源级检查；
#   * **run 级**授权**没有**放宽：任何 *run* 消费这四个 family 仍会被
#     `smoke_other` / `outside` 检查判为**越界**（下方另有专门的"候选 family 未被任何 run 消费"检查）；
#   * D-3 批准前**不得**用它们跑任何 GPU 任务。
STAGE2B_CANDIDATE_FAMILIES = ("F09", "F03", "F11", "F04")

# 2026-10-03：本轮协议**预登记**的未见数据 family（四 stratum 各取最小 id，排除 F13/F06）。
# 授权来源 = **用户在本轮任务中明确指定**"从 manifest 选 4 个未参与 8209/F13 定位的 family，
# 至少覆盖 3 个 stratum，运行前预登记"。它不是既有的 F06/F13 定点取证范围，
# 因此必须**显式登记**（登记 + scoped 检查；**不放宽**任何既有检查）。
PROTOCOL_ROUND_FAMILIES = ("F01", "F02", "F08", "F10")

# 2026-10-03（**串行 target verification oracle 轮**）：授权来源 = 本轮任务书明确指定
#   Stage 1 = F13；Stage 2 = 协议预登记的 F01/F02/F08/F10（**不重跑** production/row-aligned）。
#   ⇒ 授权集合 = {"F13"} ∪ PROTOCOL_ROUND_FAMILIES。
#
# 为什么该轮的运行器**不**进 `OPT_IN_CONSUMERS`：
#   * `OPT_IN_CONSUMERS` 的口径是"**直接**消费 manifest 字段的文件"，由 `_manifest_consumers()`
#     按 `families_manifest/prompt_families/family_id/window_token_start` 这些**字面量**识别，
#     并且有等式检查 `seen_files == sorted(OPT_IN_CONSUMERS)`（登记了但检测不到会直接判失败）。
#   * `serial_oracle_gpu.py` 只是**驱动** harness（经 `SO_CASE` → `SMOKE_FAMILY`），真正的取数仍在
#     已登记的 `g5_8k_smoke_cudnn` ⇒ 它产出的 `smoke8k_eager_*.json` 仍按 `smoke_f06`/`smoke_protocol`
#     分类并被授权范围覆盖。
#   * 但它自己的运行记录（family 嵌在 `smoke.input`）**不**被主检测器看到（主检测器只读顶层 `input`），
#     因此**另加**下方专项 scoped 检查（**只增检查、不改既有**）。
SERIAL_ORACLE_ROUND = dict(tag="div8209_so", tool="serial_oracle_gpu",
                           families=("F13",) + PROTOCOL_ROUND_FAMILIES)


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


def _serial_oracle_run_records(tag: str, tool: str) -> list:
    """读取**串行 oracle 轮**运行器自己写的记录（family 嵌在 `smoke.input`，主检测器看不到）。

    ⚠️ 口径：本函数只读文件、只提取字段；**不**改任何检查的严格性。空集合会被**如实**报出
    （GPU 运行开始之前必然为空 —— 不得把"空"说成"已验证"）。
    """
    out = []
    for p in sorted((ROOT / "output" / "g5" / "attn").glob(f"{tag}_*_rep*_*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(j, dict) or j.get("tool") != tool:
            continue
        inp = ((j.get("smoke") or {}).get("input") or {})
        fam = inp.get("family") or {}
        out.append(dict(file=p.name, case=j.get("case"), mode=j.get("mode"), rep=j.get("rep"),
                        kind=inp.get("source_kind"), family_id=fam.get("family_id"),
                        stage=j.get("stage"), blocked_by=j.get("blocked_by"),
                        env_qualified=j.get("env_qualified"),
                        model_run_happened=j.get("model_run_happened")))
    return out


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
    authorize_tools = ("g5_f06_index25_forensics", "prefill_sb_replay_f06", "f13_i50_capture",
                       "unified_geometry_run", "hf_serial_block_probe", "xb_probe",
                       "l0_k_replay", "l0_k_diff_cpu", "c1_rope_source_diag", "ar_spec_consistency_cpu")
    forensic = [u for u in used if (u.get("tool") or "").startswith(authorize_tools)]
    # `g5_8k_smoke_cudnn.py` 的**非 fam16** 运行（2026-09-30 的 eager 候选配置诊断、
    # 2026-10-01 的 F13 eager 扩展验证）也消费 manifest；按用户授权，
    # **只允许 F06 与 F13**（两者都是已参与调试的定点诊断 family），
    # 其它 family 一律仍算「授权范围之外」；**不**扩大到任何别的 family。
    REGISTERED_DIAG_FAMILIES = ("F06", "F13")
    fam16_files = {f"smoke8k_cudnn_prefill_{r}.json" for r in fam16_ids}
    smoke_f06 = [u for u in used
                 if (u.get("tool") or "").startswith("g5_8k_smoke_cudnn")
                 and u["file"] not in fam16_files
                 and (u.get("family") or {}).get("family_id") in REGISTERED_DIAG_FAMILIES]
    smoke_other = [u for u in used
                   if (u.get("tool") or "").startswith("g5_8k_smoke_cudnn")
                   and u["file"] not in fam16_files
                   and (u.get("family") or {}).get("family_id") not in REGISTERED_DIAG_FAMILIES
                   and (u.get("family") or {}).get("family_id") not in PROTOCOL_ROUND_FAMILIES]
    # 2026-10-03 协议验证轮：smoke 工具在该轮里**只允许**预登记的 4 个 family
    smoke_protocol = [u for u in used
                      if (u.get("tool") or "").startswith("g5_8k_smoke_cudnn")
                      and u["file"] not in fam16_files
                      and (u.get("family") or {}).get("family_id") in PROTOCOL_ROUND_FAMILIES]
    smoke_protocol_bad = [u for u in smoke_protocol
                          if (u.get("family") or {}).get("family_id") not in PROTOCOL_ROUND_FAMILIES]
    authorized_files = ({u["file"] for u in fam16} | {u["file"] for u in forensic}
                        | {u["file"] for u in smoke_f06}
                        # 2026-10-03：协议验证轮（未见数据）——**只允许预登记的 4 个 family**
                        | {u["file"] for u in smoke_protocol})
    outside = [u for u in used if u["file"] not in authorized_files]
    check("★ **授权范围之外**没有任何 run 消费 manifest family（授权范围 = fam16 16 格 + F06/index25 定点取证 "
          "+ 2026-10-03 协议验证轮的 4 个预登记 family）",
          not outside, outside[:5])
    check("★ smoke 工具的非 fam16 运行只允许已登记的定点诊断 family（F06 / F13）或协议预登记 family（F01/F02/F08/F10）",
          not smoke_other, [u["file"] for u in smoke_other][:4])
    check("★ 协议验证轮的 smoke 运行**只**消费预登记的 4 个 family（F01/F02/F08/F10）",
          not smoke_protocol_bad, [u["file"] for u in smoke_protocol_bad][:4])
    cm = [u for u in used if (u.get("tool") or "").startswith("correctness_protocol_margins")]
    bad_cm = [u["file"] for u in cm
              if (u.get("family") or {}).get("family_id") not in PROTOCOL_ROUND_FAMILIES]
    check("★ 新登记消费方 `correctness_protocol_margins` 只引用协议预登记的 4 个 family"
          "（注册范围仅限该工具 + F01/F02/F08/F10）",
          not bad_cm, dict(n_scoped=len(cm), bad=bad_cm[:4]))
    # 2026-10-03（串行 target verification oracle 轮）：授权集合 = {"F13"} ∪ {F01,F02,F08,F10}。
    # 记录级：读**运行器自己写的**记录（family 嵌在 `smoke.input`）。**空的含义要说清楚**：
    # GPU 运行开始之前（或被环境门拒绝时）这里必然为空 —— 空**不**等于"已验证在范围内"。
    so = _serial_oracle_run_records(SERIAL_ORACLE_ROUND["tag"], SERIAL_ORACLE_ROUND["tool"])
    so_fams = SERIAL_ORACLE_ROUND["families"]
    bad_so = [u["file"] for u in so
              if (u.get("family_id") and u["family_id"] not in so_fams)
              or (u.get("case") and u["case"] not in so_fams)]
    n_so_ran = sum(1 for u in so if u.get("model_run_happened"))
    n_so_env = sum(1 for u in so if u.get("blocked_by") == "environment")
    check("★ 本轮运行器 `serial_oracle_gpu`（串行 target verification oracle）的 family 只在 "
          "{F13} ∪ {F01,F02,F08,F10} 内（记录级；空集合**不算**证明，见 extra.note）",
          not bad_so,
          dict(n_scoped=len(so), n_model_runs=n_so_ran, n_env_blocked=n_so_env,
               families=list(so_fams), bad=bad_so[:4],
               note="记录级检查只覆盖**已存在**的运行记录；在 GPU 运行（且通过环境门）之前它为空，"
                    "此时它不构成任何结论。本轮 Stage 1/2 的 family 范围另由**来源**保证："
                    "`SO_CASE` 由调用方显式给定，本仓库的 `scripts/so_batch.ps1` 只被以 "
                    "F13 与 F01/F02/F08/F10 调用。"))
    # 源级：上述两个文件里**不得**出现授权集合之外的 family 字面量
    # ⚠️ 必须加**词边界**：`F\d{2}` 会命中 `BF16`/`bf16` 中的 "F16"（本仓库里 bf16 出现极多）。
    #    这正是本文件 `_manifest_consumers()` 注释里早已警告过的陷阱；本次仍踩了一次，
    #    首版 `re.findall(r"F\d{2}")` 因此误报 F16 ⇒ 改为 `(?<![A-Za-z0-9_])F\d{2}`。
    #    **修的是本检查的写法，不是被测文件**（不得为让检查通过而改被测代码）。
    _so_src = "\n".join((ROOT / "specdec" / "engine" / n).read_text(encoding="utf-8", errors="ignore")
                        for n in ("serial_oracle_gpu.py", "serial_oracle_aggregate.py",
                                  "serial_oracle_env_check.py",
                                  "serial_oracle_env_check_selftest_cpu.py")
                        if (ROOT / "specdec" / "engine" / n).exists())
    _so_ids = sorted({m for m in re.findall(r"(?<![A-Za-z0-9_])F\d{2}\b", _so_src)})
    _so_out_of_scope = [f for f in _so_ids if f not in so_fams]
    check("★ 本轮运行器/汇总器的**源码**里没有授权集合之外的 family 字面量（源级检查）",
          not _so_out_of_scope, dict(found=_so_ids, out_of_scope=_so_out_of_scope))
    # 源级（Stage 2b 工具，2026-10-04 新增）：允许的 family 字面量 = 已授权集合 ∪ **候选集合**。
    # 候选集合只用于"让工具源码可被检查"，**不构成 run 级授权**（见下方 run 级检查）。
    _s2b_src = "\n".join((ROOT / "specdec" / "engine" / n).read_text(encoding="utf-8", errors="ignore")
                         for n in ("stage2b_verdict.py", "stage_check.py",
                                   "stage2b_preflight_cpu.py", "stage2b_verdict_selftest_cpu.py")
                         if (ROOT / "specdec" / "engine" / n).exists())
    _s2b_allowed = set(so_fams) | set(STAGE2B_CANDIDATE_FAMILIES) | {"F06"}   # F06 = 已登记的定点诊断 family
    _s2b_ids = sorted({m for m in re.findall(r"(?<![A-Za-z0-9_])F\d{2}\b", _s2b_src)})
    _s2b_bad = [f for f in _s2b_ids if f not in _s2b_allowed]
    check("★ Stage 2b 工具的源码里只有'已授权 ∪ 候选'的 family 字面量（候选 = 待 D-3，未授权运行）",
          not _s2b_bad, dict(found=_s2b_ids, allowed=sorted(_s2b_allowed), out_of_scope=_s2b_bad))
    _cand_used = [u["file"] for u in used
                  if (u.get("family") or {}).get("family_id") in STAGE2B_CANDIDATE_FAMILIES]
    check("★ Stage 2b 候选 family **尚未被任何 run 消费**（D-3 批准前不得跑；run 级授权未放宽）",
          not _cand_used, dict(n_bad=len(_cand_used), bad=_cand_used[:4]))
    scoped = [u for u in used if (u.get("tool") or "").startswith("prefill_sb_replay_f06")]
    bad_family = [u["file"] for u in scoped
                  if (u.get("family") or {}).get("family_id") != "F06"]
    check("★ 新登记消费方 `prefill_sb_replay_f06` 只消费 F06（注册范围仅限该工具 + F06）",
          len(scoped) > 0 and not bad_family, dict(n_scoped=len(scoped), bad=bad_family[:4]))
    # 2026-10-01：`f13_i50_capture` 的注册范围**仅限 F13**
    scoped13 = [u for u in used if (u.get("tool") or "").startswith("f13_i50_capture")]
    bad13 = [u["file"] for u in scoped13
             if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `f13_i50_capture` 只消费 F13（注册范围仅限该工具 + F13）",
          len(scoped13) > 0 and not bad13, dict(n_scoped=len(scoped13), bad=bad13[:4]))
    # 2026-10-01：UCG 原型诊断的注册范围**仅限 F13**
    scoped_ug = [u for u in used if (u.get("tool") or "").startswith("unified_geometry_run")]
    bad_ug = [u["file"] for u in scoped_ug
              if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `unified_geometry_run` 只消费 F13（注册范围仅限该工具 + F13）",
          len(scoped_ug) > 0 and not bad_ug, dict(n_scoped=len(scoped_ug), bad=bad_ug[:4]))
    # 2026-10-01：HF 外部对照的注册范围**仅限 F13**
    scoped_hf = [u for u in used if (u.get("tool") or "").startswith("hf_serial_block_probe")]
    bad_hf = [u["file"] for u in scoped_hf
              if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `hf_serial_block_probe` 只消费 F13（注册范围仅限该工具 + F13）",
          len(scoped_hf) > 0 and not bad_hf, dict(n_scoped=len(scoped_hf), bad=bad_hf[:4]))
    # 2026-10-01：L0 K 有界定位（第 0 层重放）的注册范围**仅限 F13**
    scoped_l0 = [u for u in used if (u.get("tool") or "").startswith("l0_k_replay")]
    bad_l0 = [u["file"] for u in scoped_l0
              if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `l0_k_replay` 只消费 F13（注册范围仅限该工具 + F13）",
          len(scoped_l0) > 0 and not bad_l0, dict(n_scoped=len(scoped_l0), bad=bad_l0[:4]))
    # 2026-10-01：L0 K 差异清单（只读派生，只**引用**既有 F13 产物）的注册范围**仅限 F13**
    scoped_l0d = [u for u in used if (u.get("tool") or "").startswith("l0_k_diff_cpu")]
    bad_l0d = [u["file"] for u in scoped_l0d
               if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `l0_k_diff_cpu` 只引用 F13（注册范围仅限该工具 + F13）",
          len(scoped_l0d) > 0 and not bad_l0d, dict(n_scoped=len(scoped_l0d), bad=bad_l0d[:4]))
    # 2026-10-01：C1 频率来源对齐诊断（只消费已消费的 F13）的注册范围**仅限 F13**
    scoped_c1r = [u for u in used if (u.get("tool") or "").startswith("c1_rope_source_diag")]
    bad_c1r = [u["file"] for u in scoped_c1r
               if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `c1_rope_source_diag` 只消费 F13（注册范围仅限该工具 + F13）",
          len(scoped_c1r) > 0 and not bad_c1r, dict(n_scoped=len(scoped_c1r), bad=bad_c1r[:4]))
    # 2026-10-01：AR/Spec 一致性比较（只读派生，只**引用**既有 F13 产物）的注册范围**仅限 F13**
    scoped_asc = [u for u in used if (u.get("tool") or "").startswith("ar_spec_consistency_cpu")]
    bad_asc = [u["file"] for u in scoped_asc
               if (u.get("family") or {}).get("family_id") != "F13"]
    check("★ 新登记消费方 `ar_spec_consistency_cpu` 只引用 F13（注册范围仅限该工具 + F13）",
          len(scoped_asc) > 0 and not bad_asc, dict(n_scoped=len(scoped_asc), bad=bad_asc[:4]))
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
                          serial_oracle_round=dict(       # 2026-10-03：串行 target verification oracle 轮
                              tag=SERIAL_ORACLE_ROUND["tag"], tool=SERIAL_ORACLE_ROUND["tool"],
                              authorized_families=list(so_fams),
                              n_records=len(so), n_model_runs=n_so_ran, n_env_blocked=n_so_env,
                              records=so, out_of_scope=bad_so,
                              source_family_literals=_so_ids, source_out_of_scope=_so_out_of_scope,
                              vacuity_note="记录级检查在 GPU 运行（且通过环境门）之前为空；"
                                           "空集合**不**构成任何结论，此处如实记录 n_model_runs/n_env_blocked",
                              why_not_opt_in_consumer=("本运行器只驱动已登记的 g5_8k_smoke_cudnn"
                                                       "（取数仍在后者），按 `OPT_IN_CONSUMERS` 的"
                                                       "字面量口径它不进该集合；其 family 范围由本专项检查覆盖")),
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
