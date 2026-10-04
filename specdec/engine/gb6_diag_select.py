"""gb6_diag_select.py — GB6 诊断的**产物配对选择**（纯 CPU、只读、可自测）。

## 为什么单独一个模块
反例 CF3：`correctness_protocol_margins.py` 原先按**旧模式名** `aligned`/`spec` 取产物，
而 Stage 2b 的模式名是 `ar`/`prod`/`serial`。找不到 divergent 侧时它把位置**退化为末 token**：

    真实 F13（tag=div8209_so）：旧代码 pos = 63（= len(ar)-1），真实首分歧 = 47。

这个选择/核对逻辑必须能被**纯 CPU 自测**覆盖（GPU 部分只负责重建 logits），
所以把它抽到这里：不 import torch、不碰 GPU、只读 JSON。

## 纪律
* 首分歧**必须**来自**同 tag、同 family** 的 `AR` 与 `production` 产物；
  任一侧缺失/来历不符 ⇒ 抛 `PairSelectionError`，**绝不**退回末 token。
* 核对输入 hash（`smoke.input.ids_hash` / `family.token_sha1`，且 `hash_verified` 为真）、
  上下文长度（`smoke.input.ctx` / `end_state.prompt_len`）、KV 格式（`smoke.kv_format`）、
  以及计算配置（`smoke_config` 去掉 `mode`/`family` 后的字典、`rules` 字符串）。
* 不借用旧 tag、旧 backend、其他 family 的 margin：文件名与记录内 `tag`/`case` 都必须匹配。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ENC = "utf-8-sig"
# `smoke_config` 里**按设计会不同**的键（各 arm 本来就不同），不参与"同一配置"核对
CONFIG_KEYS_EXEMPT = ("mode", "family")
# ⚠️ **真实产物实测**（div8209_so / F13）：
#     AR   : kv_format={"target":"fp16","draft":"not_loaded"}
#     prod : kv_format={"target":"fp16","draft":"fp16"}
#   AR 模式**不加载 draft**（协议如此）⇒ draft 侧字段按设计不同，**不能**拿来做"同一配置"判据，
#   否则会把所有正常配对误判为 `CONFIG_MISMATCH`（首次真实数据复核就撞上了）。
#   因此只核对 **target 侧**（模型 dtype 与 target KV payload）——而它必须一致。
DRAFT_SIDE_NOTE = ("AR 模式不加载 draft ⇒ `kv_format.draft` / `input.dtypes.draft_kv_payload` "
                   "按设计不同，不参与『同一配置』核对；参与核对的是 target 侧。")


class PairSelectionError(Exception):
    """配对失败：带**机器可读**的 `code` 与人类可读的 `detail`。"""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding=ENC))


def find_artifact(tag: str, family: str, mode: str, outdir: Path) -> Path:
    """找 `{tag}_{family}_{mode}_rep{rep}_{runid}.json` 中**最新**的一份（按文件名排序，只读）。"""
    fs = sorted(glob.glob(str(Path(outdir) / f"{tag}_{family}_{mode}_rep*_*.json")))
    fs = [f for f in fs if "aggregate" not in Path(f).name and "verdict" not in Path(f).name
          and "margins" not in Path(f).name]
    if not fs:
        raise PairSelectionError(
            "MISSING_ARTIFACT",
            f"tag={tag!r} family={family!r} mode={mode!r} 在 {Path(outdir).name} 下没有任何产物"
            f"（**不允许**用其它模式/其它 tag/family 代跑，也不允许退化为末 token）")
    return Path(fs[-1])


def _gen_of(rec: dict) -> list:
    return list(((rec.get("smoke") or {}).get("gen")) or [])


def load_gen(path: Path) -> list:
    """读取某份产物 `smoke.gen`（对外供 GPU 侧复用**被选中的那一份**，不再二次 glob）。"""
    return _gen_of(_load(Path(path)))


def first_divergence(a: list, b: list):
    """返回首个不同的**输出序列内下标**；完全相同或某一侧为空 ⇒ None。"""
    if not a or not b:
        return None
    i = 0
    for x, y in zip(a, b):
        if x != y:
            return i
        i += 1
    return None


def _round_index_of(rounds: list, out_index: int):
    """把输出内下标映射到**该 run 的第几轮**（由 `smoke.rounds` 顺序累加，属推导）。"""
    cum = 0
    for rr in (rounds or []):
        n = len(rr.get("emitted") or [])
        if cum <= out_index < cum + n:
            return rr.get("round")
        cum += n
    return None


def _config_identity(rec: dict) -> dict:
    sc = dict(rec.get("smoke_config") or {})
    for k in CONFIG_KEYS_EXEMPT:
        sc.pop(k, None)
    return dict(smoke_config=sc, rules=rec.get("rules"))


def _check_side(tag: str, family: str, mode: str, rec: dict, path: Path) -> list:
    """单侧产物的**来源核对**：tag/family/tool 必须与请求一致。"""
    bad = []
    if rec.get("tag") != tag:
        bad.append(f"{mode}: 记录内 tag={rec.get('tag')!r} ≠ 请求 tag={tag!r}（禁止借用其它 tag 的产物）")
    if rec.get("case") != family:
        bad.append(f"{mode}: 记录内 case={rec.get('case')!r} ≠ 请求 family={family!r}"
                   f"（禁止借用其它 family 的 margin）")
    fam_block = ((rec.get("smoke") or {}).get("input") or {}).get("family") or {}
    fid = fam_block.get("family_id")
    if fid is not None and fid != family:
        bad.append(f"{mode}: smoke.input.family.family_id={fid!r} ≠ {family!r}")
    if not path.name.startswith(f"{tag}_{family}_{mode}_"):
        bad.append(f"{mode}: 文件名 {path.name!r} 与 tag/family/mode 不匹配")
    return bad


def select_divergent_pair(*, tag: str, family: str, outdir: Path, ar_mode: str = "ar",
                          div_mode: str = "prod", ctx: int | None = None,
                          require_hash_verified: bool = True,
                          allow_last_index_when_equal: bool = False) -> dict:
    """选定 AR 与 divergent 侧产物并完成**同一配置**核对，返回首分歧位置及其来源证据。

    返回 dict：
      `ok` · `status`（`DIVERGENT` / `TOKEN_EQUAL`）· `first_divergence_index` ·
      `position`（== 首分歧下标；token 相等且未开 legacy 开关时为 None）·
      `position_origin`（消费端据此判断位置是否合法）· `round_index` ·
      `ar`/`div` 两个 `dict(file, mode, gen, gen_len, prompt_len)` · `config_checks` ·
      `prompt_len`（供 GB6 记账）· `ar_token`/`prod_token`（诊断侧与运行侧一致性用）。

    token 完全相等（无分歧）时：`status=TOKEN_EQUAL`、`position=None`
    ⇒ **GB6 不需要诊断**，调用方不得把它当成"首分歧"。
    """
    outdir = Path(outdir)
    checks: dict = {}
    ar_path = find_artifact(tag, family, ar_mode, outdir)
    div_path = find_artifact(tag, family, div_mode, outdir)
    ar, dv = _load(ar_path), _load(div_path)
    bad = _check_side(tag, family, ar_mode, ar, ar_path) + _check_side(tag, family, div_mode, dv, div_path)
    if bad:
        raise PairSelectionError("PROVENANCE_MISMATCH", "；".join(bad))
    ar_sm, dv_sm = (ar.get("smoke") or {}), (dv.get("smoke") or {})
    ar_in, dv_in = (ar_sm.get("input") or {}), (dv_sm.get("input") or {})
    ar_fam, dv_fam = (ar_in.get("family") or {}), (dv_in.get("family") or {})
    ar_es, dv_es = (ar_sm.get("end_state") or {}), (dv_sm.get("end_state") or {})

    # ---- 输入一致：ids_hash / family token_sha1 / ctx ----
    checks["ids_hash_equal"] = (ar_in.get("ids_hash") == dv_in.get("ids_hash")
                                and ar_in.get("ids_hash") is not None)
    checks["family_token_sha1_equal"] = (ar_fam.get("token_sha1") == dv_fam.get("token_sha1")
                                         and ar_fam.get("token_sha1") is not None)
    checks["ctx_equal"] = (ar_in.get("ctx") == dv_in.get("ctx") and ar_in.get("ctx") is not None)
    if require_hash_verified:
        checks["hash_verified_both"] = (ar_fam.get("hash_verified") is True
                                        and dv_fam.get("hash_verified") is True)
    # ---- 上下文长度一致（prompt_len）----
    pl_ar, pl_dv = ar_es.get("prompt_len"), dv_es.get("prompt_len")
    checks["prompt_len_equal"] = (pl_ar is not None and pl_ar == pl_dv)
    # ---- KV 格式与计算配置一致（**只比对 target 侧**；draft 侧按设计不同，见 DRAFT_SIDE_NOTE）----
    ar_kv, dv_kv = (ar_sm.get("kv_format") or {}), (dv_sm.get("kv_format") or {})
    checks["kv_format_target_equal"] = (bool(ar_kv.get("target"))
                                        and ar_kv.get("target") == dv_kv.get("target"))
    cfg_ar, cfg_dv = _config_identity(ar), _config_identity(dv)
    checks["compute_config_equal"] = (cfg_ar["smoke_config"] == cfg_dv["smoke_config"]
                                      and bool(cfg_ar["smoke_config"]))
    checks["rules_equal"] = (cfg_ar["rules"] == cfg_dv["rules"] and cfg_ar["rules"] is not None)
    dt_ar, dt_dv = (ar_in.get("dtypes") or {}), (dv_in.get("dtypes") or {})
    checks["dtypes_target_equal"] = (bool(dt_ar.get("model")) and
                                     {k: dt_ar.get(k) for k in ("model", "target_kv_payload")}
                                     == {k: dt_dv.get(k) for k in ("model", "target_kv_payload")})
    checks["same_ctx_requested"] = (ctx is None or ar_in.get("ctx") == ctx)

    failed = sorted(k for k, v in checks.items() if v is not True)
    if failed:
        detail = ("配对两侧**不是同一配置**："
                  + "；".join(f"{k}" for k in failed)
                  + f"（AR={ar_path.name} DIV={div_path.name}；"
                    f"ctx={ar_in.get('ctx')}/{dv_in.get('ctx')} "
                    f"prompt_len={pl_ar}/{pl_dv} "
                    f"ids_hash={ar_in.get('ids_hash')}/{dv_in.get('ids_hash')}）")
        raise PairSelectionError("CONFIG_MISMATCH", detail)

    ar_gen, dv_gen = _gen_of(ar), _gen_of(dv)
    d = first_divergence(dv_gen, ar_gen)
    if d is None:
        if not allow_last_index_when_equal:
            return dict(ok=True, status="TOKEN_EQUAL", first_divergence_index=None, position=None,
                        position_origin="token_equal（AR 与 production 逐 token 相同 ⇒ 无分歧，"
                                        "不记录位置；GB6 不需要诊断）",
                        round_index=None, prompt_len=(None if pl_ar is None else int(pl_ar)),
                        ar=dict(file=ar_path.name, mode=ar_mode, gen_len=len(ar_gen)),
                        div=dict(file=div_path.name, mode=div_mode, gen_len=len(dv_gen)),
                        config_checks=checks, config_notes=DRAFT_SIDE_NOTE, ar_token=None, prod_token=None)
        # **仅** legacy 诊断用途（显式开启）：位置=末位，且 `position_origin` 明确标为 legacy，
        # 消费端（GB6）会因为不含 `first_divergence` 而**拒绝**该记录。
        pos = max(0, len(ar_gen) - 1)
        return dict(ok=True, status="TOKEN_EQUAL_LEGACY_LAST_INDEX", first_divergence_index=None,
                    position=pos, position_origin="legacy_last_index（无分歧 ⇒ 末位；**非**首分歧）",
                    round_index=None, prompt_len=(None if pl_ar is None else int(pl_ar)),
                    ar=dict(file=ar_path.name, mode=ar_mode, gen_len=len(ar_gen)),
                    div=dict(file=div_path.name, mode=div_mode, gen_len=len(dv_gen)),
                    config_checks=checks, config_notes=DRAFT_SIDE_NOTE, ar_token=None, prod_token=None)
    rounds_dv = dv_sm.get("rounds") or []
    return dict(
        ok=True, status="DIVERGENT", first_divergence_index=int(d),
        position=int(d),
        position_origin=(f"first_divergence(ar_vs_{div_mode})：同 tag={tag}、同 family={family} 的 "
                         f"{ar_path.name} 与 {div_path.name} 首个不同 token 的输出内下标"),
        round_index=_round_index_of(rounds_dv, d),
        prompt_len=(None if pl_ar is None else int(pl_ar)),
        ar=dict(file=ar_path.name, mode=ar_mode, gen_len=len(ar_gen)),
        div=dict(file=div_path.name, mode=div_mode, gen_len=len(dv_gen)),
        config_checks=checks, config_notes=DRAFT_SIDE_NOTE,
        ar_token=(ar_gen[d] if d < len(ar_gen) else None),
        prod_token=(dv_gen[d] if d < len(dv_gen) else None),
        note="位置来自 AR 与 production 的**首分歧**（同 tag/同 family/同配置）；不进任何计时结论")
