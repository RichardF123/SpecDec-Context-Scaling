"""stage2b_verdict.py — **Stage 2b 判决工具**（显式 tag / family / 模式 / 重复；缺一即 `CHECK_NEEDS_REVIEW`）。

## 为什么另写一个模块（而不是继续用 `serial_oracle_aggregate --scope final`）
`serial_oracle_aggregate` 的 Stage 2 路径把 **tag（`div8209_so`）与 family 集合（F13/F01/F02/F08/F10）
写死在模块里**。用它来判 Stage 2b 会出现两个**静默错误**（本轮审计 B-1/B-2 已实测）：
① 换 tag 后 `_load_runs()` 读不到新产物（0 条）；② 即使读到，四个新 family 也**不在判决集合里**而被忽略
⇒ 可能给出"看起来通过、其实没看数据"的判决。因此本模块把三件事**变成必填参数**：
`--tag`（必填）· `--families`（必填）· `--modes`/`--reps`（每种模式的重复次数，必填）。

## 判决规则（写死）
* **覆盖性优先**：期望的 `(family, mode, rep)` 三元组**缺任何一个**、出现**重复编号冲突**、
  或出现**超范围/非指定的三元组** ⇒ `CHECK_NEEDS_REVIEW`（**绝不**静默 PASS）。
* tag 内存在**指定 collection 以外**的 family ⇒ 默认 `CHECK_NEEDS_REVIEW`（防止误读旧产物）；
  确属有意混放时须显式 `--allow-off-scope`（并记入产物）。
* **环境拒绝**记录只报告、不计入失败；但若因此造成必需重复数不足 ⇒ `CHECK_NEEDS_REVIEW`。
* **AR 基线**：
  - **有历史登记锚点的 family**（F13/F01/F02/F08/F10）⇒ 沿用既有判据（登记输出 hash + 上一轮产物逐位对照），**不改写**；
  - **新 family（无旧 AR 锚点）**⇒ 只做四项：两次（`reps` 次）完整 token IDs **自一致**、
    自报 `end_state.gen_hash`/长度/停止原因与本次独立计算的 hash 一致、`manifest` 输入 `token_sha1`
    与运行记录一致。**不得**套用旧 family 的登记输出 hash（否则会假报 `BASELINE_REGRESSION`）。
* **production 数值行为**：要求确定性重复、finite、cache/逐轮结构不变量、合法 token 与停止原因、无 OOM；
  几何上必须**出现** `s>1` 的 batched 前向（证明走的不是 serial）。与 AR 的 token 分歧**必须完整记录**
  （首位置、共同前缀、两侧 token、实际长度、接受/工作量），**不自动称无害、也不据此判失败**。
* **serial 硬检查**：18 项（含 `s==1`、0 次 `s>1`、审计非空、记账恒等、row-aligned 关闭、D6 清理后实测 `block`）。
* **判决次序**：`CHECK_NEEDS_REVIEW` → `BASELINE_REGRESSION` → `SERIAL_ORACLE_FAIL` → `SERIAL_ORACLE_PASS`。

**本模块不修改任何数值路径**，只读取运行产物；默认**不覆盖**任何历史文件（写新时间戳 JSON/JSONL）。
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from . import serial_oracle_aggregate as AGG

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TOOL = "serial_oracle_gpu"
LEGACY_TAG = AGG.TAG
EXPECTED_LEN = 64
LEGAL_STOPS = ("max_new_reached", "eos")
# 本轮（Stage 2b）**预登记**：模式集合必须精确等于三者；每种模式的独立重复数必须等于 2。
REQUIRED_MODES = ("ar", "prod", "serial")
PREREGISTERED_REPS = 2
# GB6 记账字段的两类来源（**不得**编造，也**不得**从别的 family/旧配置迁移）
GB6_FROM_RUNS = ("first_divergence_index", "common_prefix_len", "ar_token", "prod_token",
                 "ar_len", "prod_len", "absolute_position", "round_index",
                 "ar_acceptance_work", "prod_acceptance_work")
GB6_FROM_DIAGNOSTIC = ("ar_top1", "ar_top2", "margin", "tied_at_max", "cache_len_after")
# 有**历史登记输出锚点**的 family（沿用既有判据，不改写）；其余视为"新 family（无旧 AR 锚点）"
KNOWN_ANCHOR_FAMILIES = tuple(AGG.STAGE2_FAMILIES)


def _extra_fields(rec: dict) -> dict:
    """本模块额外需要的字段（不修改 `serial_oracle_aggregate._row`，避免影响既有判决路径）。"""
    sm = rec.get("smoke") or {}
    inp = sm.get("input") or {}
    fam = inp.get("family") or {}
    prov = rec.get("provenance") or {}
    return dict(input_family_id=fam.get("family_id"),
                input_family_token_sha1=fam.get("token_sha1"),
                input_hash_verified=fam.get("hash_verified"),
                input_source_kind=inp.get("source_kind"),
                provenance_hash_verified=(prov.get("hash_verified") if isinstance(prov, dict) else None),
                env_qualified=rec.get("env_qualified"),
                model_run_happened=rec.get("model_run_happened"))


def _row(rec: dict) -> dict:
    return {**AGG._row(rec), **_extra_fields(rec)}


def load_runs(tag: str, outdir: Path = OUTDIR) -> list:
    """只读**指定 tag** 的产物；绝不回退到其它 tag。"""
    out = []
    for f in sorted(glob.glob(str(Path(outdir) / f"{tag}_*_rep*_*.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        except Exception:                                                 # noqa: BLE001
            continue
        if d.get("tool") != TOOL:
            continue
        d["_file"] = Path(f).name
        out.append(d)
    return out


def _manifest_token_sha1(family: str, *, ctx: int = 8192) -> str | None:
    try:
        from . import family_input as FI
        rec = FI.family_record(family, FI.load_manifest())
        pref = rec.get("prefixes") or {}
        entry = pref.get(str(ctx)) or pref.get(ctx) or {}
        return entry.get("token_sha1")
    except Exception:                                                     # noqa: BLE001
        return None


# ---------------------------------------------------------------- 各阶段检查
def validate_arguments(tag: str, families: list, modes: list, reps) -> list:
    """**参数合法性**（CLI 与 `evaluate()` 两层都会调用；任何一条不满足即不可能 PASS）。

    规则来自本轮预登记：tag 非空 · family 集合非空且无重复、形如 `F\\d{2}` ·
    模式集合**精确等于** `ar,prod,serial`（不含重复、不多不少）· reps 为正整数且**等于**预登记值 2。
    """
    import re as _re
    errs: list = []
    if not str(tag or "").strip():
        errs.append("tag 为空")
    if not families:
        errs.append("family 集合为空（不允许：会导致『期望三元组=0 ⇒ 空泛 PASS』）")
    if len(set(families)) != len(families):
        errs.append(f"family 集合有重复：{families}")
    bad_ids = [f for f in families if not _re.fullmatch(r"F\d{2}", str(f))]
    if bad_ids:
        errs.append(f"family id 形状非法（应为 F\\d{{2}}）：{bad_ids}")
    if len(set(modes)) != len(modes):
        errs.append(f"模式集合有重复：{modes}")
    if set(modes) != set(REQUIRED_MODES):
        errs.append(f"模式必须**精确**包含 {list(REQUIRED_MODES)}（实得 {list(modes)}）")
    try:
        r = int(reps)
    except Exception:                                                     # noqa: BLE001
        r = -1
    if r <= 0:
        errs.append(f"reps 必须为正整数（实得 {reps!r}）")
    elif r != PREREGISTERED_REPS:
        errs.append(f"reps 必须等于本轮预登记值 {PREREGISTERED_REPS}（实得 {r}）")
    return errs


def _acceptance_work(r: dict) -> dict:
    st = r.get("specdec_stats") or {}
    return dict(rounds=st.get("rounds"), proposed=st.get("proposed"), accepted=st.get("accepted"),
                target_block_calls=st.get("target_block_calls"), draft_calls=st.get("draft_calls"),
                rollbacks=st.get("rollbacks"), actual_len=r.get("gen_len"),
                stop_reason=r.get("stop_reason"))


def _gb6_run_view(row: dict, raw: dict | None) -> dict:
    """把一份产物投影成 GB6 记账所需的**运行侧视图**（字段全部取自产物本身）。

    ⚠️ 缺陷修复（自测 CE4b 抓到**假 PASS**）：`gen` 必须走与聚合器唯一访问器 `_row` **同一路径**
    （`smoke.gen`）。首版 `_gb6_record` 直接读 runner 顶层 `rec["gen"]`，而真实产物**没有**顶层 `gen`
    ⇒ `gen` 恒为 None ⇒ `_first_div` 判不出分歧 ⇒ `divergent=False`
    ⇒ **production 与 AR 明明分歧却从未做 GB6 记账，判决却走 PASS 路径**（正是要堵的漏洞）。
    `prompt_len` 取 `smoke.end_state.prompt_len`（真实产物中存在，如 F13 prod = 8192），
    用于把输出内下标换算成**绝对位置**；缺失即如实记为 None，不猜、不迁移。
    """
    sm = ((raw or {}).get("smoke")) or {}
    es = sm.get("end_state") or {}
    gen = list(row.get("gen") or [])
    return dict(
        file=row.get("file"), rep=row.get("rep"), gen=gen, gen_len=len(gen),
        gen_hash=row.get("gen_hash"), rounds=sm.get("rounds") or [],
        specdec_stats=sm.get("specdec_stats") or {}, stop_reason=row.get("stop_reason"),
        prompt_len=(None if es.get("prompt_len") is None else int(es["prompt_len"])),
        end_state_gen_hash=es.get("gen_hash"), end_state_gen_len=es.get("gen_len"),
        end_state_n_rounds=es.get("n_rounds"), end_state_target_cache_len=es.get("target_cache_len"),
        end_state_draft_cache_len=es.get("draft_cache_len"))


def _round_index_of(rounds: list, out_index: int):
    """把"输出序列里的第 d 个 token"映射到**轮号**（由该 run 自己的 `rounds` 顺序累加得到，属推导而非编造）。"""
    rounds = rounds or []
    cum = 0
    for rr in rounds:
        n = len(rr.get("emitted") or [])
        if cum <= out_index < cum + n:
            return rr.get("round"), cum, n
        cum += n
    return None, None, None


def _gb6_record(fam: str, ar_views: list, pr_views: list, diag: dict | None) -> dict:
    """GB6 记账：**可从产物直接取得**的字段 vs **需要单独诊断 pass** 的字段，分开列清。"""
    out: dict = dict(family=fam, runs_derived={}, diagnostic={}, missing=[], source=None,
                     divergent=None, artifact_inconsistency={})
    if not ar_views or not pr_views:
        out["missing"] = list(GB6_FROM_RUNS) + list(GB6_FROM_DIAGNOSTIC)
        out["no_input"] = True
        return out
    ar, pr = ar_views[0], pr_views[0]
    ag, pg = list(ar.get("gen") or []), list(pr.get("gen") or [])
    # ---- 产物自洽性（与是否分歧无关，都要报）----
    if ar.get("prompt_len") is not None and pr.get("prompt_len") is not None \
            and int(ar["prompt_len"]) != int(pr["prompt_len"]):
        out["artifact_inconsistency"]["prompt_len"] = dict(ar=int(ar["prompt_len"]),
                                                           prod=int(pr["prompt_len"]))
    bad_hash = [v.get("file") for v in (ar_views + pr_views)
                if v.get("end_state_gen_hash") is not None
                and v.get("end_state_gen_hash") != v.get("gen_hash")]
    if bad_hash:
        out["artifact_inconsistency"]["gen_hash_vs_end_state"] = bad_hash
    d = AGG._first_div(pg, ag)
    if d is None:
        out["divergent"] = False
        return out
    out["divergent"] = True
    prompt_len = ar.get("prompt_len")
    # 轮号：分歧位于 **divergent（production）那一侧的第几轮** —— 该轮就是"做出该决策的轮"。
    # ⚠️ 修复：AR 模式产物**没有** `smoke.rounds`（AR 无投机轮）⇒ 只读 AR 的 rounds 会得到
    #    `round_index=None`（真实 F13 实测），于是"决策轮号"这一 GB6 必需字段永远缺失。
    #    现以 production 的 rounds 为准，并同时如实留下两侧推导值（不取舍其一而不说明）。
    pr_ridx, pr_cum, _prn = _round_index_of(pr.get("rounds"), d)
    ar_ridx, ar_cum, _arn = _round_index_of(ar.get("rounds"), d)
    ridx = pr_ridx if pr_ridx is not None else ar_ridx
    out["runs_derived"] = dict(
        first_divergence_index=d,
        common_prefix_len=d,
        ar_token=(ag[d] if d < len(ag) else None),
        prod_token=(pg[d] if d < len(pg) else None),
        ar_len=len(ag), prod_len=len(pg),
        absolute_position=(None if prompt_len is None else int(prompt_len) + int(d)),
        round_index=ridx,
        round_index_source=("prod" if pr_ridx is not None else ("ar" if ar_ridx is not None else None)),
        round_index_prod=pr_ridx, round_index_ar=ar_ridx,
        round_start_offset=(pr_cum if pr_ridx is not None else ar_cum),
        ar_acceptance_work=_acceptance_work(ar),
        prod_acceptance_work=_acceptance_work(pr),
        prod_hashes=[v.get("gen_hash") for v in pr_views],
        # 产物里可直接取得的 cache 长度（**补充证据**，不替代诊断 pass 的 `cache_len_after`）；
        # 真实产物 `end_state` 中存在；缺失则如实为 None，不推导、不迁移。
        ar_cache_len_artifact=ar.get("end_state_target_cache_len"),
        prod_cache_len_artifact=pr.get("end_state_target_cache_len"),
        prod_draft_cache_len_artifact=pr.get("end_state_draft_cache_len"))
    if diag:
        # ⚠️ 只接受**同一 family**且**同一 tag**的诊断记录；不做跨 family/旧配置迁移。
        #    本轮修复的两个"静默放行"点：
        #    (a) 真实诊断产物（如 `div8209_proto_margins_*.json`）把 `runs_tag` 放在**文件顶层**，
        #        每个 family 条目里没有 tag ⇒ 首版 `diag_tag=None` 被当作"无需核对"而**放行**。
        #        现改为：顶层 tag 由 `load_gb6_diagnostics` 下推到每条记录；**没有任何 tag 可核对
        #        即拒绝**（不能默认"同配置"，否则旧 margin 会被当成本轮结果迁移进来）。
        #    (b) 诊断的 `position` 是**输出序列内下标**（历史文件：F01 position=14、prompt_len=8192），
        #        首版却拿它与 `prompt_len + d`（绝对位置）比较 ⇒ 会**误判位置不符**（假拒绝）。
        #        现按同一坐标系比较：诊断若有 `prompt_len` ⇒ 换算成绝对位置再比；否则退化为下标比下标。
        same_family = (diag.get("family") == fam) or (diag.get("family_id") == fam)
        diag_tag = diag.get("runs_tag") or diag.get("tag")
        same_tag = (diag_tag is not None) and (diag_tag == _GB6_CTX.get("tag"))
        if not same_family:
            out["rejected_reason"] = (f"诊断记录的 family={diag.get('family') or diag.get('family_id')!r}"
                                      f" 与本 family 不符")
        elif diag_tag is None:
            out["rejected_reason"] = "诊断记录没有可核对的 tag（禁止默认『同配置』⇒ 不采用）"
        elif not same_tag:
            out["rejected_reason"] = f"诊断记录的 tag={diag_tag!r} 与本轮 tag 不符（禁止跨配置迁移）"
        else:
            out["source"] = diag.get("_source")
            out["diagnostic"] = {k: diag.get(k) for k in GB6_FROM_DIAGNOSTIC}
            # 位置交叉核对（同一坐标系）：生产者的口径是 **输出序列内下标** ——
            # `correctness_protocol_margins.py:142` 记的是 `pos = cp`（首分歧下标），
            # 历史产物亦如此（F01: position=14, prompt_len=8192）。故比较对象是 `common_prefix_len`；
            # 不把 `position` 当绝对位置（首版那样做会自相矛盾：同一个数两边都不对）。
            dpos = diag.get("position")
            d_common = out["runs_derived"].get("common_prefix_len")
            if dpos is not None and d_common is not None and int(dpos) != int(d_common):
                out["position_mismatch"] = dict(diagnostic=int(dpos), derived_common_prefix=int(d_common))
            out["position_frame"] = "output_index"
            # `prompt_len` 一致性（诊断与产物必须是同一输入配置）
            dpl = diag.get("prompt_len")
            if dpl is not None and prompt_len is not None and int(dpl) != int(prompt_len):
                out["prompt_len_mismatch"] = dict(diagnostic=int(dpl), derived=int(prompt_len))
    out["missing"] = ([k for k in GB6_FROM_RUNS if out["runs_derived"].get(k) in (None, {})]
                      + [k for k in GB6_FROM_DIAGNOSTIC if not out["diagnostic"].get(k, None) is not None
                         and out["diagnostic"].get(k) is None])
    return out


_GB6_CTX: dict = {}


def load_gb6_diagnostics(path: str | None) -> dict:
    """读取**独立诊断 pass** 的 GB6 记录（按 family 索引）。

    接受三种形态：`{family: {...}}`、`{"families": {family: {...}}}`、或记录列表（含 `family`/`family_id` 字段）。
    只做**读取与索引**；不填补缺失字段、不从别处迁移数值。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        j = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:                                                     # noqa: BLE001
        return {}
    out: dict = {}
    # ⚠️ 真实诊断产物把 `runs_tag` / `tag` 放在**文件顶层**，family 条目里没有 ⇒ 必须下推，
    #    否则 tag 核对会因"取不到 tag"而形同虚设（旧 margin 可被静默采用）。
    file_tag = (j.get("runs_tag") or j.get("tag")) if isinstance(j, dict) else None

    def _with_tag(v: dict) -> dict:
        v = {**v, "_source": p.name}
        if (v.get("runs_tag") or v.get("tag")) is None and file_tag is not None:
            v["runs_tag"] = file_tag
        return v

    if isinstance(j, dict) and isinstance(j.get("families"), dict):
        for k, v in j["families"].items():
            if isinstance(v, dict):
                v = _with_tag({**v, "family": v.get("family") or k})
                out[str(k)] = v
    elif isinstance(j, dict):
        for k, v in j.items():
            if isinstance(v, dict) and ("margin" in v or "ar_top1" in v or "position" in v):
                v = _with_tag({**v, "family": v.get("family") or k})
                out[str(k)] = v
    elif isinstance(j, list):
        for v in j:
            if isinstance(v, dict):
                k = v.get("family") or v.get("family_id")
                if k:
                    out[str(k)] = _with_tag(v)
    return out


def attach_internal(rec: dict) -> dict:
    """把行字典与该行**必须**携带的内部键一起返回（`_rounds`/`_trace`/`_cross`）。

    统一在一处构造，避免 `stage_check` 与 `stage2b_verdict` 两处各自漏挂（首版即漏挂 ⇒ 硬检查 8/10 误报）。
    """
    row = _row(rec)
    row["_rounds"] = AGG._stage1_rounds_fields(rec)
    row["_trace"] = AGG._trace_checks(rec)
    row["_cross"] = ((rec.get("audit") or {}).get("cross_check")) or {}
    return row


def _ar_baseline_known(fam: str, ar: list) -> dict:
    """有历史锚点的 family：沿用既有判据（登记 hash + 上一轮产物逐位对照）。"""
    prior = AGG._prior_ar(fam)
    anchor = AGG.STAGE2_AR_ANCHORS.get(fam) or {}
    ref = ar[0]["gen"] if ar else None
    checks = dict(anchor_class="registered_output_anchor",
                  reps_identical=bool(len(ar) >= 2 and len({tuple(r["gen"]) for r in ar}) == 1),
                  equals_registered_anchor=bool(ar and all(
                      r["gen_hash"] == anchor.get("hash") and r["gen_len"] == anchor.get("len")
                      and r["stop_reason"] == anchor.get("stop") for r in ar)),
                  self_reported_hash_agrees=bool(ar and all(
                      r["gen_hash"] == r.get("end_state_gen_hash") for r in ar)))
    if prior.get("available"):
        pg = prior["gens"][0] if prior.get("gens") else None
        checks.update(prior_source="prior_round_artifact",
                      equals_prior_tokens=bool(ref is not None and pg is not None and ref == pg),
                      equals_prior_hash=bool(ar and prior.get("hashes")
                                             and all(r["gen_hash"] == prior["hashes"][0] for r in ar)),
                      equals_prior_len=bool(ar and prior.get("lens")
                                            and all(r["gen_len"] == prior["lens"][0] for r in ar)),
                      equals_prior_stop=bool(ar and prior.get("stops")
                                             and all(r["stop_reason"] == prior["stops"][0] for r in ar)))
    else:
        checks.update(prior_source="registered_anchor_only（无上一轮产物）",
                      equals_prior_tokens=None, equals_prior_hash=None,
                      equals_prior_len=None, equals_prior_stop=None,
                      prior_note=prior.get("source"))
    # ⚠️ "没有上一轮产物" **不是**"缺证据"：旧 family（如 F13）本来就没有 prior artifact，
    #    只有登记输出锚点。若把 `equals_prior_* = None` 当成缺字段，会把真正的锚点不符
    #    误判成 `CHECK_NEEDS_REVIEW` 而不是 `BASELINE_REGRESSION`（自测 F 抓到）。
    missing_field: list = []
    ok = bool(checks["reps_identical"] and checks["equals_registered_anchor"]
              and checks["self_reported_hash_agrees"])
    return dict(checks=checks, ok=ok, missing_fields=missing_field)


def _ar_baseline_new(fam: str, ar: list, *, reps: int, ctx: int) -> dict:
    """**新 family（无旧 AR 锚点）**：只校验自一致 + 自报一致 + manifest 输入 hash。"""
    exp_sha = _manifest_token_sha1(fam, ctx=ctx)
    hashes = [r["gen_hash"] for r in ar]
    stops = [r["stop_reason"] for r in ar]
    input_shas = [r.get("input_family_token_sha1") for r in ar]
    checks = dict(anchor_class="no_output_anchor（新 family：不得套用旧 family 的登记 hash）",
                  reps_present=bool(len(ar) == reps),
                  reps_identical=bool(len(ar) >= 2 and len({tuple(r["gen"]) for r in ar}) == 1),
                  self_reported_hash_agrees=bool(ar and all(r["gen_hash"] == r.get("end_state_gen_hash")
                                                            for r in ar)),
                  self_reported_len_agrees=bool(ar and all(
                      r["gen_len"] == r.get("end_state_len") == EXPECTED_LEN for r in ar)),
                  stop_legal_and_consistent=bool(stops and len(set(stops)) == 1
                                                 and stops[0] in LEGAL_STOPS),
                  manifest_input_hash_available=bool(exp_sha),
                  manifest_input_hash_matches=bool(exp_sha and input_shas
                                                   and all(s == exp_sha for s in input_shas)),
                  manifest_expected_token_sha1=exp_sha,
                  observed_input_token_sha1=input_shas,
                  observed_hashes=hashes)
    missing = [k for k in ("manifest_input_hash_available", "manifest_input_hash_matches")
               if checks[k] is not True]
    hard_fail = [k for k in ("reps_identical", "self_reported_hash_agrees",
                             "self_reported_len_agrees", "stop_legal_and_consistent")
                 if checks[k] is not True]
    return dict(checks=checks, ok=bool(not hard_fail and not missing),
                inconsistency=hard_fail, missing_fields=missing)


def _production_checks(fam: str, pr: list, *, reps: int, ar_ref: list | None) -> dict:
    """production 数值行为：结构与重复性为门；与 AR 的分歧**记录**（不判失败）。"""
    gens = [r["gen"] for r in pr]
    lens = [set(r.get("target_gen_seq_lens") or []) for r in pr]
    batched = [bool(s and max(s) > 1) for s in lens]
    divs = [AGG._first_div(g, ar_ref) if ar_ref is not None else None for g in gens]
    checks = dict(reps_present=bool(len(pr) == reps),
                  reps_identical=bool(len(pr) >= 2 and len({tuple(g) for g in gens}) == 1),
                  checks_passed=bool(pr and all(r.get("checks_passed") is True for r in pr)),
                  structure_ok=bool(pr and all(r.get("cache_structure_ok") is True
                                               and (r.get("cache_structure_n_checks") or 0) > 0 for r in pr)),
                  round_invariants_ok=bool(pr and all(r.get("round_invariants_all_ok") is True for r in pr)),
                  finite=bool(pr and all(r.get("prefill_finite") is not False for r in pr)),
                  gen_ids_in_vocab=bool(pr and all(r.get("gen_ids_in_vocab") is not False for r in pr)),
                  stop_legal=bool(pr and all(r.get("stop_reason") in LEGAL_STOPS for r in pr)),
                  batched_geometry_present=bool(batched and all(batched)),
                  no_oom=bool(pr and all(r.get("smoke_rc_zero") is True for r in pr)))
    hard = [k for k, v in checks.items() if v is not True]
    divergent = [bool(d is not None) for d in divs]
    return dict(checks=checks, hard_failures=hard, divergences=divs,
                divergence_class=("NUMERICAL_DIVERGENCE_RECORDED" if any(divergent) else "TOKEN_EQUAL"),
                divergence_recorded=dict(
                    first_divergence=divs,
                    common_prefix=[(d if d is not None else len(g))
                                   for d, g in zip(divs, gens)],
                    ar_len=(len(ar_ref) if ar_ref is not None else None),
                    prod_len=[len(g) for g in gens],
                    prod_hashes=[r["gen_hash"] for r in pr],
                    note=("分歧**必须**在报告中分层解释：位置、完整输出、实际长度、接受率与工作量；"
                          "不按分歧位置筛选样本，也不据此判失败")),
                ok=bool(not hard))


# ---------------------------------------------------------------- 主判决
def evaluate(tag: str, families: list, modes: list, reps: int, *, outdir: Path = OUTDIR,
             allow_off_scope: bool = False, ctx: int = 8192,
             gb6_diagnostics: dict | None = None) -> dict:
    _GB6_CTX["tag"] = tag
    argument_errors = validate_arguments(tag, families, modes, reps)
    runs = load_runs(tag, outdir)
    usable, env_blocked = [], []
    for r in runs:
        if r.get("blocked_by") == "environment" or r.get("model_run_happened") is False:
            env_blocked.append(r)
        else:
            usable.append(r)
    rows, raw_by_key = [], {}
    for rec_i in usable:
        row = attach_internal(rec_i)
        rows.append(row)
        raw_by_key[(row.get("case"), row.get("mode"), int(row.get("rep") or 0))] = rec_i
    needs_review: list = []
    baseline_fail: list = []
    serial_fail: list = []
    # GB6（分歧记账）单独一条通道：它属于"证据不全、不能判 PASS"，但**不得掩盖**
    # 更决断的基线回归 / serial 硬失败（自测 F 抓到：AR 基线已错时再报 GB6 缺字段，
    # 会把 BASELINE_REGRESSION 降级成 NEEDS_REVIEW ⇒ 掩盖真实失败信号）。
    gb6_needs_review: list = []
    # ---- -1) 参数合法性（**空泛 PASS 的第一道防线**：空 family 集合会让期望三元组=0）----
    for e in argument_errors:
        needs_review.append(f"参数非法：{e}")
    # ---- -0.5) 环境拒绝：**单列报告、不计有效重复**；只有造成必需重复缺失才阻断（由下面的覆盖性检查负责）----
    env_blocked_records = [dict(file=r.get("_file"), case=r.get("case"), mode=r.get("mode"),
                               rep=r.get("rep"),
                               why=str((r.get("commit_guard") or {}).get("why"))[:120])
                           for r in env_blocked]

    # ---- 0) 覆盖性：缺 family/mode/rep、重复编号、超范围三元组 ----
    found: dict = {}
    for r in rows:
        found.setdefault((r.get("case"), r.get("mode"), int(r.get("rep") or 0)), []).append(r)
    expected = [(f, m, rep) for f in families for m in modes for rep in range(1, int(reps) + 1)]
    for key in expected:
        got = found.get(key)
        if not got:
            needs_review.append(f"缺运行：family={key[0]} mode={key[1]} rep={key[2]}")
        elif len(got) > 1:
            needs_review.append(f"重复编号冲突：(family={key[0]}, mode={key[1]}, rep={key[2]}) 有 "
                                f"{len(got)} 份产物 {[g['file'] for g in got]}")
    for key, got in sorted(found.items(), key=lambda kv: (str(kv[0]))):
        if key in expected:
            continue
        # 指定集合**之外**的 family 属于"越界产物"：由 `--allow-off-scope` 单独治理，
        # 此处不重复计入"非预期三元组"（否则显式允许后仍会被判 NEEDS_REVIEW）。
        if allow_off_scope and key[0] not in families:
            continue
        needs_review.append(f"非预期三元组：(family={key[0]}, mode={key[1]}, rep={key[2]})"
                            f"（编号超范围或不在指定 family/mode 集合内）")
    off_scope = sorted({r.get("case") for r in rows if r.get("case") not in families})
    if off_scope and not allow_off_scope:
        needs_review.append(f"tag={tag!r} 内存在指定集合之外的 family {off_scope}"
                            f" ⇒ 可能误读旧产物（确属有意请显式 --allow-off-scope）")

    # ---- 1) 每 family ----
    per_family: dict = {}
    for fam in families:
        by_mode = {m: sorted([r for r in rows if r.get("case") == fam and r.get("mode") == m],
                             key=lambda r: int(r.get("rep") or 0)) for m in modes}
        # ⚠️ 逐模式**存在性**检查（独立于 `expected` 三元组）：避免"某模式 0 次却被跳过"的放行漏洞
        #    （反例 CE2：只给 AR×2 + modes=['ar'] ⇒ 旧实现直接 PASS）。
        for m in REQUIRED_MODES:
            if not by_mode.get(m):
                needs_review.append(f"{fam}: {m} 模式没有任何可用产物 ⇒ 该阶段未执行（不得判 PASS）")
            elif len(by_mode[m]) < int(reps):
                needs_review.append(f"{fam}: {m} 只有 {len(by_mode[m])} 次 < 预登记 {reps} 次")
        ar, pr = by_mode.get("ar", []), by_mode.get("prod", [])
        se = by_mode.get("serial", [])
        ar_ref = ar[0]["gen"] if ar else None
        for r in se:
            r["_ar_stop"] = (ar[0]["stop_reason"] if ar else None)
            r["_first_div_vs_ar"] = (AGG._first_div(r["gen"], ar_ref) if ar_ref is not None else None)
        if fam in KNOWN_ANCHOR_FAMILIES:
            base = _ar_baseline_known(fam, ar)
        else:
            base = _ar_baseline_new(fam, ar, reps=int(reps), ctx=ctx)
        prod = _production_checks(fam, pr, reps=int(reps), ar_ref=ar_ref) if pr else \
            dict(checks={}, hard_failures=["production 运行缺失"], divergences=[], ok=False,
                 divergence_class=None, divergence_recorded=None)
        hard = {f"rep{r['rep']}": AGG._serial_hard_checks2(r, require_d6=True) for r in se}
        # 用**已附加**的内部键（不能再用 row 去调用 `_stage1_rounds_fields`：它读的是 `rec["smoke"]`）
        rounds = {f"rep{r['rep']}": r["_rounds"] for r in se}
        traces = {f"rep{r['rep']}": r["_trace"] for r in se}
        eq = bool(se and ar_ref is not None and all(r["gen"] == ar_ref for r in se))
        reps_ok = bool(len(se) >= 2 and len({tuple(r["gen"]) for r in se}) == 1)
        # ---- GB6：分歧时的记账完整性（可直接取得 vs 需独立诊断）----
        # ⚠️ 走 `_gb6_run_view`（`gen` 与 `_row` 同一路径 `smoke.gen`），不再直接读 runner 顶层 `gen`。
        ar_view = [_gb6_run_view(r, raw_by_key.get((fam, "ar", int(r["rep"])))) for r in ar]
        pr_view = [_gb6_run_view(r, raw_by_key.get((fam, "prod", int(r["rep"])))) for r in pr]
        diag = (gb6_diagnostics or {}).get(fam)
        gb6 = _gb6_record(fam, ar_view, pr_view, diag)
        if gb6.get("artifact_inconsistency"):
            gb6_needs_review.append(f"{fam}: GB6 产物自洽性异常（不因分歧与否而豁免）"
                                    f"{gb6['artifact_inconsistency']}")
        if gb6.get("divergent"):
            gb6["from_runs_available"] = sorted(GB6_FROM_RUNS)
            gb6["from_diagnostic_available"] = sorted(k for k in GB6_FROM_DIAGNOSTIC
                                                      if gb6["diagnostic"].get(k) is not None)
            gb6["from_diagnostic_missing"] = sorted(k for k in GB6_FROM_DIAGNOSTIC
                                                    if gb6["diagnostic"].get(k) is None)
            if gb6.get("rejected_reason"):
                gb6_needs_review.append(f"{fam}: GB6 诊断记录被拒绝 —— {gb6['rejected_reason']}")
            elif gb6["from_diagnostic_missing"]:
                gb6_needs_review.append(
                    f"{fam}: production 与 AR 分歧，但 GB6 必需字段未齐全（需**独立诊断 pass**）："
                    f"{gb6['from_diagnostic_missing']} ⇒ 不得判 PASS")
            if gb6.get("position_mismatch"):
                gb6_needs_review.append(
                    f"{fam}: GB6 诊断位置与产物推导不一致 {gb6['position_mismatch']}")
            if gb6.get("prompt_len_mismatch"):
                gb6_needs_review.append(
                    f"{fam}: GB6 诊断的 prompt_len 与产物不一致 {gb6['prompt_len_mismatch']}"
                    f"（疑似不同输入配置 ⇒ 不得采用）")
        per_family[fam] = dict(
            family=fam, anchor_class=base["checks"].get("anchor_class"),
            n_reps=dict(ar=len(ar), prod=len(pr), serial=len(se)),
            ar_checks=base["checks"], ar_baseline_ok=base["ok"],
            production_checks=prod["checks"], production_hard_failures=prod["hard_failures"],
            production_divergence_class=prod["divergence_class"],
            production_divergence=prod["divergence_recorded"],
            gb6=gb6,
            serial_equals_ar=eq, serial_reps_identical=reps_ok,
            first_divergence_vs_ar=[r["_first_div_vs_ar"] for r in se],
            serial_hashes=[r["gen_hash"] for r in se], ar_hashes=[r["gen_hash"] for r in ar],
            hard_checks=hard, independent_round_checks=rounds, independent_trace_checks=traces)
        if not base["ok"]:
            if base.get("missing_fields"):
                needs_review.append(f"{fam}: AR 基线**无法评估**（缺字段 {base['missing_fields']}）")
            else:
                baseline_fail.append(f"{fam}: AR 基线不成立（{base['checks']}）")
        for key, dd in hard.items():
            for name, v in dd.items():
                if v["ok"] is False:
                    serial_fail.append(f"{fam}/serial/{key}: `{name}` 失败 —— {v['detail']}")
                elif v["ok"] is None and not v.get("na"):
                    needs_review.append(f"{fam}/serial/{key}: 无法评估 `{name}`（字段缺失）")
        for key, rc in rounds.items():
            if not rc.get("ok"):
                serial_fail.append(f"{fam}/serial/{key}: 逐轮字段不完整 {rc.get('missing')}")
        for key, tc in traces.items():
            if tc.get("applicable") and not tc.get("ok"):
                serial_fail.append(f"{fam}/serial/{key}: trace 与轮记录不一致 {tc.get('bad')}")
        if se and not eq:
            serial_fail.append(f"{fam}: serial 与 AR token 不一致"
                               f"（首分歧 {per_family[fam]['first_divergence_vs_ar']}）")
        if se and not reps_ok:
            serial_fail.append(f"{fam}: 两次 serial 不一致")
        # production 缺失已由"逐模式存在性"检查覆盖；这里处理"有产物但硬门未过"
        if pr and prod["hard_failures"]:
            serial_fail.append(f"{fam}: production 检查失败 {prod['hard_failures']}")

    # 判决优先级（**任何一条都不会放行 PASS**，区别只在标签给出的人类下一步动作）：
    #   ① 参数非法 / 覆盖性不全 / AR 基线无法评估（缺字段）⇒ CHECK_NEEDS_REVIEW（证据不足，不声称失败）
    #   ② AR 基线**可评估且不成立** ⇒ BASELINE_REGRESSION（最决断：环境或模型锚点变了，先停机）
    #   ③ serial 硬检查 / token 等价 / production 结构失败 ⇒ SERIAL_ORACLE_FAIL
    #   ④ GB6 分歧记账不全 ⇒ CHECK_NEEDS_REVIEW（不得因"记账没做"而掩盖 ②③，也不得放行）
    #   ⑤ 全部通过 ⇒ SERIAL_ORACLE_PASS
    if needs_review:
        decision = "CHECK_NEEDS_REVIEW"
    elif baseline_fail:
        decision = "BASELINE_REGRESSION"
    elif serial_fail:
        decision = "SERIAL_ORACLE_FAIL"
    elif gb6_needs_review:
        decision = "CHECK_NEEDS_REVIEW"
    else:
        decision = "SERIAL_ORACLE_PASS"
    n_expected = len(families) * len(modes) * int(reps)
    return dict(tool="stage2b_verdict", tag=tag, families=list(families), modes=list(modes),
                reps=int(reps), allow_off_scope=bool(allow_off_scope),
                run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                argument_errors=argument_errors,
                expected_triples=[f"{f}|{m}|rep{r}" for f, m, r in expected],
                n_triples_expected=n_expected, n_triples_found=len(found),
                n_runs_found_total=len(runs), n_runs_used=len(rows),
                n_env_blocked=len(env_blocked), env_blocked_records=env_blocked_records,
                env_blocked_policy="环境拒绝**只报告、不计有效重复**；仅当其造成必需重复缺失时才阻断",
                off_scope_families=off_scope,
                gb6_diagnostics_used=sorted((gb6_diagnostics or {}).keys()),
                included_families=sorted(per_family.keys()),
                per_family=per_family, needs_review=needs_review + gb6_needs_review,
                gb6_needs_review=gb6_needs_review,
                baseline_failures=baseline_fail, serial_failures=serial_fail,
                decision=decision, decision_rule=(
                    "参数合法性（tag/family 非空无重复、模式精确 ar,prod,serial、reps==预登记值）⇒ CHECK_NEEDS_REVIEW；"
                    "覆盖性（缺 family/mode/rep、重复编号冲突、超范围三元组、越界 family）⇒ CHECK_NEEDS_REVIEW；"
                    "AR 基线（有锚点 family 用登记 hash + 上一轮产物；新 family 用自一致 + manifest 输入 hash）"
                    "⇒ BASELINE_REGRESSION；serial 硬检查/token 等价/production 结构 ⇒ SERIAL_ORACLE_FAIL；"
                    "**GB6 分歧记账不全 ⇒ CHECK_NEEDS_REVIEW**（优先级低于上面两种真实失败，高于 PASS；"
                    "理由：不得因『记账没做』掩盖基线回归/serial 失败，也不得在记账缺失时放行）；"
                    "否则 SERIAL_ORACLE_PASS。production 与 AR 的分歧**记录**为 NUMERICAL_DIVERGENCE_RECORDED，不判失败。"),
                next_round_allowed=(decision == "SERIAL_ORACLE_PASS"),
                note="本工具只读产物；不修改数值路径；不覆盖历史文件（写新时间戳）")


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Stage 2b 判决（显式 tag/family/mode/reps）")
    ap.add_argument("--tag", required=True, help="产物 tag（必填；不会回退到任何默认 tag）")
    ap.add_argument("--families", required=True, help="期望 family，逗号分隔（必填）")
    ap.add_argument("--modes", default="ar,prod,serial", help="期望模式，逗号分隔（默认 ar,prod,serial）")
    ap.add_argument("--reps", required=True, type=int, help="每种模式的独立重复次数（必填）")
    ap.add_argument("--allow-off-scope", action="store_true",
                    help="允许 tag 内存在指定集合之外的 family（默认不允许，存在即 NEEDS_REVIEW）")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--gb6", default=None,
                    help="GB6 **独立诊断 pass** 的产物（分歧时必需；缺则判 CHECK_NEEDS_REVIEW）")
    ap.add_argument("--no-write", action="store_true", help="只打印，不落盘")
    a = ap.parse_args()
    fams = [x.strip() for x in a.families.split(",") if x.strip()]
    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    # ⚠️ CLI 层校验：参数非法 ⇒ **非零退出且绝不写任何产物**（不得产出 PASS 记录）
    errs = validate_arguments(a.tag, fams, modes, a.reps)
    if errs:
        print("== Stage 2b 判决：**参数非法**，不评估、不落盘 ==")
        for e in errs:
            print(f"   [非法] {e}")
        print("   ★ 结论 = INVALID_ARGUMENTS（退出码 2；未写任何产物）")
        return 2
    diag = load_gb6_diagnostics(a.gb6)
    s = evaluate(a.tag, fams, modes, a.reps, outdir=Path(a.outdir),
                 allow_off_scope=a.allow_off_scope, gb6_diagnostics=diag)
    print(f"== Stage 2b 判决 ==  tag={s['tag']} families={s['families']} modes={s['modes']} reps={s['reps']}")
    print(f"  期望三元组 {s['n_triples_expected']} 个；实得 {s['n_triples_found']} 个；"
          f"读到产物 {s['n_runs_found_total']} 条（其中环境拒绝 {s['n_env_blocked']} 条）")
    for fam, pf in s["per_family"].items():
        print(f"  [{fam}] 锚点={pf['anchor_class']} reps={pf['n_reps']} AR基线ok={pf['ar_baseline_ok']} "
              f"serial==AR={pf['serial_equals_ar']} 两次一致={pf['serial_reps_identical']} "
              f"production={pf['production_divergence_class']} 首分歧={pf['first_divergence_vs_ar']}")
    for x in s["needs_review"]:
        print(f"  [需复核] {x}")
    for x in s["baseline_failures"]:
        print(f"  [基线回归] {x}")
    for x in s["serial_failures"]:
        print(f"  [serial 硬门失败] {x}")
    print(f"  ★ 判决 = {s['decision']}")
    if not a.no_write:
        # ⚠️ 修复：原先写死 `OUTDIR` ⇒ `--outdir` 对**落盘位置**无效（读 A 目录、写到默认目录）。
        #    本轮改为跟随 `--outdir`，与 `evaluate(outdir=...)` 的读取口径一致。
        od = Path(a.outdir)
        p = od / f"{s['tag']}_stage2b_verdict_{s['run_utc']}.json"
        pj = od / f"{s['tag']}_stage2b_verdict_{s['run_utc']}.jsonl"
        for r in usable_rows(a.tag, Path(a.outdir)):
            with pj.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        s["artifact_json"], s["artifact_jsonl"] = str(p), str(pj)
        p.write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[s2b] RESULT_JSON={p}\n[s2b] RESULT_JSONL={pj}")
    return 0 if s["decision"] == "SERIAL_ORACLE_PASS" else 1


def usable_rows(tag: str, outdir: Path = OUTDIR) -> list:
    out = []
    for r in load_runs(tag, outdir):
        if not (r.get("blocked_by") == "environment" or r.get("model_run_happened") is False):
            out.append(_row(r))
    return out


if __name__ == "__main__":
    raise SystemExit(_cli())
