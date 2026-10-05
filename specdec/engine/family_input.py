"""family_input.py — G1：按 **manifest family** 取 8K 输入（显式启用，默认路径不变）。

原状（为什么需要它）
  `g5_8k_spec_variance_probe.load_ids()` 把 `wiki.test.raw` 的一段前缀**写死**，
  所以所有工具只能跑同一条输入 —— 验证准备清单里"两个未参与调试的 family"**根本没有入口**。

本模块只做三件事
  1. `resolve_input_plan()`：读环境变量，决定用**默认写死前缀**还是**manifest family**。
     `SMOKE_FAMILY` 未设 ⇒ `mode="legacy"`，调用方走原路径，**行为逐位不变**。
  2. `load_family_ids()`：按 manifest 的取用规则重建 token id，并把 **`token_sha1` 校验**作为
     **硬条件**（不匹配即抛 `FamilyInputError`，不降级、不警告放行）。
  3. `family_cells()`：16 格运行清单的**单一定义**（runner 文档与汇总器共用，避免两处口径）。

纪律
  * **不写** manifest、不改任何工具默认值；
  * tokenizer 用**显式**配置（`add_special_tokens` 显式给出、`revision` 可钉、`local_files_only`），
    不依赖库默认值漂移；
  * 只读 zip/JSON；不加载模型、不碰 CUDA。
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "wikitext-2-raw-v1.zip"
MANIFEST = ROOT / "docs" / "prompt_families_manifest.json"

BLOCK_CHARS = 300_000        # 与 scripts/build_prompt_families.py 一致
WIN = 8192
DEFAULT_CTX = 8192

# ---- tokenizer / EOS 的**显式**配置（不依赖库默认）----
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# manifest 里记录的 tokenizer id；builder 用 `AutoTokenizer.from_pretrained(id, local_files_only=True)`
# ⇒ 其 `add_special_tokens` 取库默认。这里把它**显式化**，避免"默认值变了而无人察觉"。
TOKENIZER_ADD_SPECIAL_TOKENS = True
# 本地缓存里没有可钉的 commit hash 时保持 None，并在记录里写明"revision 未钉"
TOKENIZER_REVISION = None
EOS_ID_EXPECTED = 151645     # Qwen2.5 系列；运行时仍以 tokenizer 实际值为准并核对


class FamilyInputError(RuntimeError):
    """manifest family 输入不可用（未知 family / hash 不匹配 / 结构异常）。**必须拒绝，不得降级。**"""


def _h(ids) -> str:
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def resolve_input_plan() -> dict:
    """按环境变量决定输入来源。**默认路径不变**；三模式互斥，歧义即显式报错。

    | 模式 | 触发条件 | 用途 |
    | --- | --- | --- |
    | `legacy` | 都没设 | 既有的写死 `test.raw` 前缀（历史默认行为，**逐位不变**） |
    | `manifest` | 只设 `SMOKE_FAMILY` | 8K manifest 窗口（Stage 1/2 轮） |
    | **`literal_prompt`**（2026-10-05 新增） | 只设 `SMOKE_PROMPT` | **固定字面用例**（B6 类型覆盖：自然语言/数学/代码/短/长/EOS/长度裁剪） |

    ⚠️ `SMOKE_PROMPT` 与 `SMOKE_FAMILY` **同时设置 ⇒ 抛 `FamilyInputError`**（不静默择一）。
    新增模式**只在显式设置 `SMOKE_PROMPT` 时生效** ⇒ 未指定参数时既有行为不变。
    """
    prompt = os.environ.get("SMOKE_PROMPT") or ""
    fam = (os.environ.get("SMOKE_FAMILY") or "").strip()
    man = os.environ.get("SMOKE_MANIFEST") or str(MANIFEST)
    if prompt and fam:
        raise FamilyInputError(
            "SMOKE_PROMPT 与 SMOKE_FAMILY **同时设置** ⇒ 输入来源歧义；显式拒绝（不静默择一）")
    if prompt:
        pid = (os.environ.get("SMOKE_PROMPT_ID") or "").strip() or None
        return dict(mode="literal_prompt", family_id=None, manifest=man, prompt=prompt,
                    prompt_id=pid,
                    prompt_sha1=hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16],
                    reason="SMOKE_PROMPT 已设置 ⇒ 使用**固定字面 prompt**（专项用例/类型覆盖）")
    if not fam:
        return dict(mode="legacy", family_id=None, manifest=man,
                    reason="SMOKE_FAMILY 未设置 ⇒ 使用既有的写死 test.raw 前缀（默认行为不变）")
    return dict(mode="manifest", family_id=fam, manifest=man,
                reason=f"SMOKE_FAMILY={fam} ⇒ 从 manifest 取该 family 的 8K 窗口")


def load_manifest(path=None) -> dict:
    p = Path(path or MANIFEST)
    if not p.exists():
        raise FamilyInputError(f"manifest 不存在：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def family_record(family_id: str, manifest: dict | None = None) -> dict:
    man = manifest or load_manifest()
    for f in man.get("families", []):
        if f.get("family_id") == family_id:
            return f
    raise FamilyInputError(
        f"manifest 里没有 family {family_id!r}；可用：{[f['family_id'] for f in man.get('families', [])]}")


def _resolve_entry(zf, declared: str) -> tuple:
    """把 manifest 的 `source` 解析成 zip 里**真实存在**的条目名。

    ⚠️ manifest 的 `source` 写作 `wikitext-2-raw-v1/wiki.train.raw`，而 zip 里真实条目是
    `wikitext-2-raw/wiki.train.raw`（builder 用的就是后者）⇒ **字符串不一致但指向同一语料**。
    这里按 basename 容错解析，并把两者都记进 meta，避免"看起来对不上"被当成 hash 问题。
    """
    names = zf.namelist()
    if declared in names:
        return declared, False
    base = declared.split("/")[-1]
    cands = [n for n in names if n.split("/")[-1] == base]
    if len(cands) == 1:
        return cands[0], True
    raise FamilyInputError(f"zip 里找不到 {declared!r} 对应的条目（候选 {cands}）")


def load_family_ids(tok, family_id: str, *, manifest: dict | None = None,
                    ctx: int = DEFAULT_CTX, verify_hash: bool = True):
    """按 manifest 规则重建该 family 的 token id。

    取用规则（与 builder 一致）：`block_text = raw_train[i*300000:(i+1)*300000]` →
    `tokenize(block_text)` → `ids[window_token_start : window_token_start+8192]` → 取前 `ctx`。

    `verify_hash=True` 时**必须**与 manifest 的 `prefixes[ctx].token_sha1` 一致，否则抛异常。
    返回 `(ids, meta)`；`meta` 是要写进 run JSON 的溯源字段。
    """
    man = manifest or load_manifest()
    f = family_record(family_id, man)
    declared = f["source"]
    with zipfile.ZipFile(ZIP) as z:
        entry, resolved = _resolve_entry(z, declared)
        raw = z.read(entry).decode("utf-8", errors="ignore")
    b = int(f["block"])
    block_text = raw[b * BLOCK_CHARS:(b + 1) * BLOCK_CHARS]
    if len(block_text) < 20_000:
        raise FamilyInputError(f"block {b} 文本过短（{len(block_text)} 字符）")
    ids_all = tok(block_text, return_tensors="pt",
                  add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS).input_ids[0].tolist()
    st = int(f["window_token_start"])
    win = ids_all[st:st + WIN]
    if len(win) != WIN:
        raise FamilyInputError(f"窗口长度 {len(win)} != {WIN}（window_token_start={st}）")
    ids = win[:ctx]
    got = _h(ids)
    px = (f.get("prefixes") or {}).get(str(ctx))
    want = px.get("token_sha1") if px else None
    ok = (got == want)
    if verify_hash and not ok:
        raise FamilyInputError(
            f"family {family_id} ctx={ctx} 的 token hash 不匹配：实测 {got} vs manifest {want}"
            f" —— **拒绝运行**（可能是 tokenizer/add_special_tokens/语料或 manifest 变更）")
    meta = dict(
        family_id=family_id, stratum=f.get("stratum"),
        expected_acceptance_rank=f.get("expected_acceptance_rank"),
        rank_provenance=("**先验序**：由文本特征 distinct_token_ratio 分层后**预登记**的接受率假设序"
                         "（manifest 的 preregistered_hypothesis），**不是**实测接受率"),
        source=entry, source_declared=declared, source_resolved_by_basename=resolved,
        source_note=("manifest 声明与 zip 真实条目名不同（`…-raw-v1/…` vs `…-raw/…`），"
                     "按 basename 解析到同一语料；不计入 hash 校验"),
        window_token_start=st, n_tokens_window=f.get("n_tokens_window"),
        ctx=ctx, token_sha1=got, manifest_token_sha1=want, hash_verified=bool(ok),
        manifest_version=man.get("manifest_version"),
        manifest_path=str(MANIFEST),
        tokenizer=dict(id=TOKENIZER_ID, add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS,
                       revision=TOKENIZER_REVISION,
                       revision_note=("revision 未钉（本地缓存未记录 commit）"
                                      if TOKENIZER_REVISION is None else "已钉"))
    )
    return ids, meta


def eos_config(tok) -> dict:
    """把 EOS 配置显式化并核对期望值（Qwen2.5 应为 151645）。"""
    got = tok.eos_token_id
    return dict(eos_id=got, eos_id_expected=EOS_ID_EXPECTED,
                matches_expected=(int(got) == EOS_ID_EXPECTED if got is not None else False),
                rule="命中 EOS 立即停止并记录实际长度；未命中且未达 max_new ⇒ stop_reason=other_short ⇒ 判失败",
                tokenizer=dict(id=TOKENIZER_ID, add_special_tokens=TOKENIZER_ADD_SPECIAL_TOKENS,
                               revision=TOKENIZER_REVISION))


# ---- 16 格运行清单：**单一定义** ----
FAMILIES = ("F06", "F13")
KVS = ("base_bf16", "v_fp8")
MODES = ("ar", "spec")
REPS = (1, 2)


def family_cells(families=FAMILIES, kvs=KVS, modes=MODES, reps=REPS) -> list:
    """返回 16 个格。顺序与编排脚本一致：family → kv → **rep → mode**
    （即 ar r1, spec r1, ar r2, spec r2 …）。

    这样 AR/Spec 配对总是"紧邻的同一 rep"，而重复比较总是"同 (family,kv,mode) 的 r1 vs r2"，
    两种比较都不会串到另一个 rep 上去。
    """
    cells = []
    n = 0
    for fam in families:
        for kv in kvs:
            for rep in reps:
                for mode in modes:
                    n += 1
                    cells.append(dict(n=n, family=fam, kv=kv, mode=mode, rep=rep,
                                      label=f"{fam}_{'base' if kv == 'base_bf16' else 'vfps'}_{mode}_r{rep}"))
    return cells


def expected_cell_keys(cells=None) -> list:
    return sorted((c["family"], c["kv"], c["mode"], c["rep"]) for c in (cells or family_cells()))
