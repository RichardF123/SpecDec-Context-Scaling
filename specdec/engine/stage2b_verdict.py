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
             allow_off_scope: bool = False, ctx: int = 8192) -> dict:
    runs = load_runs(tag, outdir)
    usable, env_blocked = [], []
    for r in runs:
        if r.get("blocked_by") == "environment" or r.get("model_run_happened") is False:
            env_blocked.append(r)
        else:
            usable.append(r)
    rows = []
    for rec_i in usable:
        row = _row(rec_i)
        # ⚠️ 与既有聚合器一致：`_serial_hard_checks` 依赖聚合器**附加**的这三个内部键
        #    （首版漏挂 ⇒ `08_identity_target_vs_draft` / `10_per_round_trace_complete` 误报失败）
        row["_rounds"] = AGG._stage1_rounds_fields(rec_i)
        row["_trace"] = AGG._trace_checks(rec_i)
        row["_cross"] = ((rec_i.get("audit") or {}).get("cross_check")) or {}
        rows.append(row)
    needs_review: list = []
    baseline_fail: list = []
    serial_fail: list = []

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
    for r in env_blocked:
        cg = r.get("commit_guard") or {}
        needs_review.append(f"环境拒绝（仅报告，若造成缺重复则已在上方计入）：{r['_file']} "
                            f"{str(cg.get('why'))[:90]}")

    # ---- 1) 每 family ----
    per_family: dict = {}
    for fam in families:
        by_mode = {m: sorted([r for r in rows if r.get("case") == fam and r.get("mode") == m],
                             key=lambda r: int(r.get("rep") or 0)) for m in modes}
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
        per_family[fam] = dict(
            family=fam, anchor_class=base["checks"].get("anchor_class"),
            n_reps=dict(ar=len(ar), prod=len(pr), serial=len(se)),
            ar_checks=base["checks"], ar_baseline_ok=base["ok"],
            production_checks=prod["checks"], production_hard_failures=prod["hard_failures"],
            production_divergence_class=prod["divergence_class"],
            production_divergence=prod["divergence_recorded"],
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
        if pr and not prod["ok"]:
            if prod["hard_failures"] == ["production 运行缺失"]:
                needs_review.append(f"{fam}: 缺 production 运行")
            else:
                serial_fail.append(f"{fam}: production 结构/重复性检查失败 {prod['hard_failures']}")

    if needs_review:
        decision = "CHECK_NEEDS_REVIEW"
    elif baseline_fail:
        decision = "BASELINE_REGRESSION"
    elif serial_fail:
        decision = "SERIAL_ORACLE_FAIL"
    else:
        decision = "SERIAL_ORACLE_PASS"
    n_expected = len(families) * len(modes) * int(reps)
    return dict(tool="stage2b_verdict", tag=tag, families=list(families), modes=list(modes),
                reps=int(reps), allow_off_scope=bool(allow_off_scope),
                run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                expected_triples=[f"{f}|{m}|rep{r}" for f, m, r in expected],
                n_triples_expected=n_expected, n_triples_found=len(found),
                n_runs_found_total=len(runs), n_runs_used=len(rows),
                n_env_blocked=len(env_blocked), off_scope_families=off_scope,
                included_families=sorted(per_family.keys()),
                per_family=per_family, needs_review=needs_review,
                baseline_failures=baseline_fail, serial_failures=serial_fail,
                decision=decision, decision_rule=(
                    "覆盖性（缺 family/mode/rep、重复编号冲突、超范围三元组、越界 family）⇒ CHECK_NEEDS_REVIEW；"
                    "AR 基线（有锚点 family 用登记 hash + 上一轮产物；新 family 用自一致 + manifest 输入 hash）"
                    "⇒ BASELINE_REGRESSION；serial 硬检查/token 等价/production 结构 ⇒ SERIAL_ORACLE_FAIL；"
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
    ap.add_argument("--no-write", action="store_true", help="只打印，不落盘")
    a = ap.parse_args()
    fams = [x.strip() for x in a.families.split(",") if x.strip()]
    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    s = evaluate(a.tag, fams, modes, a.reps, outdir=Path(a.outdir),
                 allow_off_scope=a.allow_off_scope)
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
        p = OUTDIR / f"{s['tag']}_stage2b_verdict_{s['run_utc']}.json"
        pj = OUTDIR / f"{s['tag']}_stage2b_verdict_{s['run_utc']}.jsonl"
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
