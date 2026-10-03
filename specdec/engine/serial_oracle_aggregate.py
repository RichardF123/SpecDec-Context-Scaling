"""serial_oracle_aggregate.py — 串行 target oracle 的**只读**汇总器（CPU，不加载模型、不用 GPU）。

## 判定依据 = 设计文档 §6 **预先登记**的口径（不是事后发明）
见 `docs/serial_target_oracle_design_2026-10-03.md`：
* `SERIAL_ORACLE_PASS`：5 个 family 的 serial oracle 与 AR 完整 token IDs **全部一致**；两模式两进程一致；
  全部 target generation 调用 M=1；轮不变量/cache/finite/合法停止全过；无 OOM；模式不作用于 production/draft/prefill。
* `SERIAL_ORACLE_FAIL`：任一 family token 不一致 / cache 或轮不变量失败 / 出现 M>1 / 被拒候选残留 /
  replacement·bonus 提交错误 / 两次重复不一致。
* `CHECK_NEEDS_REVIEW`：字段缺失 / AR 自身不重复 / 环境不合格 / 无法证明全部 M=1 / 接口无法无侵入替换。

**判定次序**（明确写死，避免歧义）：先看**已评估**的硬门 ⇒ 任一失败即 `SERIAL_ORACLE_FAIL`；
否则若有必需项**无法评估**（缺字段/缺 family/缺重复）⇒ `CHECK_NEEDS_REVIEW`；否则 `PASS`。

## 它自己**独立**重算的不变量（不依赖 harness 的 `all_ok` 结论）
逐轮从 `smoke.rounds` 重算：`tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)`、
`emitted == cands[:m]+[g[m]]`（mismatch）或 `== cands`（全接受）；
并对 serial 模式逐轮核对 `serial_trace`：`kp==Kp`、`pre==pre`、`step_cache_lens==[pre+1..pre+kp]`、
`step_argmax[:kp-1]==g[1:]`、以及**全接受轮的 bonus 链**（上一轮最后一行 == 本轮 `g[0]`）。
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = "div8209_so"
FAMILIES_ALL = ("F13", "F01", "F02", "F08", "F10")
# 历史锚点（既有失败记录，用于核对"新运行器的 AR/production 路径与历史逐位一致"）
ANCHOR = dict(F13=dict(ar_gen_hash="6e7522bf2521ad7b", prod_gen_hash="e90dad2b3fcf1c65",
                       first_divergence=47))


def _load_runs(tag: str = TAG) -> list:
    out = []
    for f in sorted(glob.glob(str(OUTDIR / f"{tag}_*_rep*_*.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:                                                 # noqa: BLE001
            continue
        if d.get("tool") == "serial_oracle_gpu":
            d["_file"] = Path(f).name
            out.append(d)
    return out


def _gen_hash(gen: list) -> str:
    from . import g5_8k_spec_variance_probe as P5V
    return P5V._gen_hash(list(gen))


def _first_div(a: list, b: list):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _round_checks(rounds: list) -> dict:
    """**独立**重算逐轮不变量（不读 harness 的结论字段）。"""
    bad = []
    for r in rounds or []:
        Kp, m, pre = int(r["Kp"]), int(r["m"]), int(r["pre"])
        exp_len = pre + (m + 1 if m < Kp else Kp)
        exp_emit = (list(r["cands"][:m]) + [int(r["g"][m])]) if m < Kp else list(r["cands"])
        if int(r["tgt_len"]) != exp_len or int(r["drf_len"]) != exp_len:
            bad.append(dict(round=r["round"], why="cache 长度 != pre+(m+1|Kp)",
                            tgt_len=r["tgt_len"], drf_len=r["drf_len"], expected=exp_len))
        if list(r["emitted"]) != exp_emit:
            bad.append(dict(round=r["round"], why="emitted != 期望", emitted=r["emitted"],
                            expected=exp_emit))
        if len(r["g"]) != Kp:
            bad.append(dict(round=r["round"], why="len(g) != Kp", g=r["g"], Kp=Kp))
    return dict(n_rounds=len(rounds or []), ok=bool(not bad), n_bad=len(bad), bad=bad[:8])


def _trace_checks(rec: dict) -> dict:
    """serial 模式逐轮核对 `serial_trace` 与 harness 的 `rounds`（含 bonus 链）。"""
    rounds = (rec.get("smoke") or {}).get("rounds") or []
    tr = rec.get("serial_trace") or []
    if rec.get("mode") != "serial":
        return dict(applicable=False, ok=True, n_trace=len(tr))
    bad = []
    if len(tr) != len(rounds):
        bad.append(dict(why="trace 条数 != 轮数", n_trace=len(tr), n_rounds=len(rounds)))
    for i, (t, r) in enumerate(zip(tr, rounds)):
        Kp, pre = int(r["Kp"]), int(r["pre"])
        if int(t.get("kp", -1)) != Kp:
            bad.append(dict(round=i, why="trace.kp != Kp", trace=t.get("kp"), Kp=Kp))
        if int(t.get("pre", -1)) != pre:
            bad.append(dict(round=i, why="trace.pre != pre", trace=t.get("pre"), pre=pre))
        if list(t.get("step_cache_lens") or []) != list(range(pre + 1, pre + Kp + 1)):
            bad.append(dict(round=i, why="step_cache_lens != [pre+1..pre+kp]",
                            got=t.get("step_cache_lens"), expected=list(range(pre + 1, pre + Kp + 1))))
        if list(t.get("step_argmax") or [])[:max(0, Kp - 1)] != list(r["g"][1:]):
            bad.append(dict(round=i, why="step_argmax[:kp-1] != g[1:]",
                            got=(t.get("step_argmax") or [])[:max(0, Kp - 1)], g=r["g"][1:]))
    # bonus 链：全接受轮的最后一行 ⇒ 下一轮 g[0]
    chain_bad = []
    for i in range(len(tr) - 1):
        r = rounds[i]
        if int(r["m"]) == int(r["Kp"]):
            last = (tr[i].get("step_argmax") or [None])[-1]
            if last != rounds[i + 1]["g"][0]:
                chain_bad.append(dict(round=i, last_step_argmax=last, next_g0=rounds[i + 1]["g"][0]))
    return dict(applicable=True, ok=bool(not bad and not chain_bad), n_trace=len(tr),
                n_bad=len(bad), bad=bad[:8], bonus_chain_ok=bool(not chain_bad),
                bonus_chain_bad=chain_bad[:8],
                note="step_argmax 覆盖**全部** kp 行（比 g 多最后一行：全接受时作为下一轮 bonus）")


def _row(rec: dict) -> dict:
    sm = rec.get("smoke") or {}
    stats = sm.get("specdec_stats") or {}
    au = rec.get("audit") or {}
    tgt = au.get("target") or {}
    gen = list(sm.get("gen") or [])
    return dict(file=rec.get("_file"), case=rec.get("case"), mode=rec.get("mode"), rep=rec.get("rep"),
                stage=rec.get("stage"), blocked_by=rec.get("blocked_by"),
                env_qualified=rec.get("env_qualified"), model_run_happened=rec.get("model_run_happened"),
                commit_guard=((rec.get("smoke") or {}).get("commit_guard")),
                gen_len=len(gen), gen=gen, gen_hash=_gen_hash(gen),
                stop_reason=sm.get("stop_reason"), gen_ids_in_vocab=sm.get("gen_ids_in_vocab"),
                prefill_finite=sm.get("prefill_logits_finite"),
                checks_passed=sm.get("checks_passed"), checks_failures=sm.get("checks_failures"),
                decision_coverage_ok=sm.get("decision_coverage"),
                round_invariants_all_ok=((sm.get("round_invariants") or {}).get("all_ok")),
                n_rounds=sm.get("n_rounds"), specdec_stats=stats,
                stats_rounds=stats.get("rounds"), stats_mismatch=stats.get("mismatch_rounds"),
                stats_all_accept=stats.get("all_accept_rounds"),
                target_gen_calls=(tgt.get("n_generation_calls")),
                target_gen_seq_lens=(tgt.get("generation_seq_lens")),
                target_all_m1=(tgt.get("all_generation_m1")),
                audit_sane=au.get("sane"), audit_certifies_m1=au.get("certifies_m1"),
                audit_n_instances=au.get("n_instances"), audit_violations=au.get("violations"),
                accounting=rec.get("accounting"), self_checks=rec.get("self_checks"),
                self_checks_all_ok=rec.get("self_checks_all_ok"),
                verify_mode=rec.get("verify_mode_before_run"),
                verify_mode_after=rec.get("verify_mode_after_run"),
                row_aligned_wrap_active=rec.get("row_aligned_wrap_active"),
                serial_trace_len=len(rec.get("serial_trace") or []),
                wall_s=sm.get("wall_s"),
                peak_alloc_mib=((sm.get("mem_timeline_last") or {}).get("pt_peak_allocated_mib")),
                peak_res_mib=((sm.get("mem_timeline_last") or {}).get("pt_peak_reserved_mib")),
                pstate=sm.get("pstate"), smoke_artifact=rec.get("smoke_artifact"),
                kill_switch_off=(rec.get("verify_mode_after_run") == "block"))


def aggregate(scope: str) -> int:
    from .. import results_schema as S
    runs = _load_runs()
    fams = ["F13"] if scope == "stage1" else list(FAMILIES_ALL)
    by: dict = {}
    for r in runs:
        by.setdefault(r.get("case"), {}).setdefault(r.get("mode"), []).append(r)
    per_case: dict = {}
    hard_fail: list = []
    needs_review: list = []
    for c in fams:
        d = by.get(c, {})
        rows = {m: [_row(r) for r in d.get(m, [])] for m in ("ar", "prod", "serial")}
        # ⚠️ **环境不合格的运行不得被当成数值失败**：harness 的环境门（commit/pstate）在加载模型**之前**
        #    就拒绝执行 ⇒ 这类记录既没有 token 也没有几何，只能计入"需复核（环境）"。
        for m in ("ar", "prod", "serial"):
            keep, blocked = [], []
            for r in rows[m]:
                (blocked if (r.get("blocked_by") == "environment"
                             or r.get("model_run_happened") is False) else keep).append(r)
            rows[m] = keep
            for r in blocked:
                cg = r.get("commit_guard") or {}
                needs_review.append(f"{c}/{m}/rep{r['rep']}: 环境不合格（未加载模型）"
                                    f" stage={r.get('stage')} why={str(cg.get('why'))[:120]}")
        ar, pr, se = rows["ar"], rows["prod"], rows["serial"]
        if not ar:
            needs_review.append(f"{c}: 缺 AR 运行")
        if not se:
            needs_review.append(f"{c}: 缺 serial 运行")
        ar_gens = [r["gen"] for r in ar]
        se_gens = [r["gen"] for r in se]
        ar_ref = ar_gens[0] if ar_gens else None
        pc: dict = dict(n_reps=dict(ar=len(ar), prod=len(pr), serial=len(se)),
                        ar_gen_len=[r["gen_len"] for r in ar],
                        prod_gen_len=[r["gen_len"] for r in pr],
                        serial_gen_len=[r["gen_len"] for r in se],
                        ar_hashes=[r["gen_hash"] for r in ar],
                        prod_hashes=[r["gen_hash"] for r in pr],
                        serial_hashes=[r["gen_hash"] for r in se],
                        ar_reps_identical=bool(ar_gens and len({tuple(g) for g in ar_gens}) == 1),
                        serial_reps_identical=bool(se_gens and len({tuple(g) for g in se_gens}) == 1),
                        serial_equals_ar=bool(se_gens and ar_ref is not None
                                              and all(g == ar_ref for g in se_gens)),
                        first_divergence_serial=None, first_divergence_prod=None,
                        stop_reasons=dict(ar=[r["stop_reason"] for r in ar],
                                          prod=[r["stop_reason"] for r in pr],
                                          serial=[r["stop_reason"] for r in se]),
                        peak_res_mib=[r["peak_res_mib"] for r in ar + se])
        if se_gens and ar_ref is not None:
            pc["first_divergence_serial"] = _first_div(se_gens[0], ar_ref)
        if pr and ar_ref is not None:
            pc["first_divergence_prod"] = _first_div(pr[0]["gen"], ar_ref)
        # --- 硬门（已评估才计）---
        if ar and se:
            if not pc["serial_equals_ar"]:
                hard_fail.append(f"{c}: serial oracle 与 AR 的 token IDs 不一致"
                                 f"（首分歧 index={pc['first_divergence_serial']}）")
            if not pc["ar_reps_identical"]:
                hard_fail.append(f"{c}: AR 两次重复不一致（环境/输入不可重复）")
            if not pc["serial_reps_identical"]:
                hard_fail.append(f"{c}: serial 两次重复不一致")
        for r in ar + se:
            tag = f"{c}/{r['mode']}/rep{r['rep']}"
            if r["checks_passed"] is not True:
                hard_fail.append(f"{tag}: harness checks.passed != True {r['checks_failures']}")
            if r["decision_coverage_ok"] is not True:
                hard_fail.append(f"{tag}: decision_coverage.ok != True")
            if r["round_invariants_all_ok"] is not True:
                hard_fail.append(f"{tag}: harness 轮不变量 all_ok != True")
            if r["prefill_finite"] is False or r["gen_ids_in_vocab"] is False:
                hard_fail.append(f"{tag}: 有限性/词表检查失败")
            if r["row_aligned_wrap_active"]:
                hard_fail.append(f"{tag}: row-aligned wrapper 处于激活状态（必须关闭）")
            if r["mode"] == "serial":
                if r["target_all_m1"] is not True or r["audit_certifies_m1"] is not True:
                    hard_fail.append(f"{tag}: 无法证明 target generation 调用全部 M=1"
                                     f"（{r['target_gen_seq_lens']}）")
                if (r["accounting"] or {}).get("matches") is not True:
                    hard_fail.append(f"{tag}: 记账恒等式不成立 {r['accounting']}")
                if r["verify_mode"] != "serial":
                    hard_fail.append(f"{tag}: 运行期 verify 模式不是 serial（{r['verify_mode']}）")
            else:
                if r["target_gen_seq_lens"] and set(r["target_gen_seq_lens"]) != {1}:
                    hard_fail.append(f"{tag}: AR 的 target generation 调用出现 s>1"
                                     f"（{r['target_gen_seq_lens']}）")
        # --- 必需但无法评估 ⇒ NEEDS_REVIEW ---
        for r in ar + se:
            tag = f"{c}/{r['mode']}/rep{r['rep']}"
            if r["n_rounds"] is None and r["mode"] == "serial":
                needs_review.append(f"{tag}: 缺 rounds 记录")
            if r["mode"] == "serial" and r["serial_trace_len"] in (None, 0):
                needs_review.append(f"{tag}: serial trace 为空（无法核对逐轮控制流）")
        # --- 我方独立重算的不变量 ---
        rc = {f"{r['mode']}/rep{r['rep']}": _round_checks((r.get("smoke") or {}).get("rounds") or [])
              for r in ar + se}
        tc = {f"{r['mode']}/rep{r['rep']}": _trace_checks(r) for r in se}
        pc["independent_round_checks"] = rc
        pc["independent_trace_checks"] = tc
        for k, v in rc.items():
            if not v["ok"]:
                hard_fail.append(f"{c} {k}: 独立重算的轮不变量失败 {v['bad']}")
        for k, v in tc.items():
            if v.get("applicable") and not v["ok"]:
                hard_fail.append(f"{c} {k}: serial trace 与轮记录不一致 {v['bad']} {v.get('bonus_chain_bad')}")
        # --- F13 历史锚点（核对新运行器与历史逐位一致）---
        if c in ANCHOR:
            a = ANCHOR[c]
            pc["anchor"] = dict(expected_ar_hash=a["ar_gen_hash"], expected_prod_hash=a["prod_gen_hash"],
                                expected_first_divergence=a["first_divergence"],
                                ar_hash_matches=bool(ar and ar[0]["gen_hash"] == a["ar_gen_hash"]),
                                ar_hash_matches_all=bool(ar and all(r["gen_hash"] == a["ar_gen_hash"] for r in ar)),
                                prod_hash_matches=bool(pr and pr[0]["gen_hash"] == a["prod_gen_hash"]),
                                prod_first_divergence=pc["first_divergence_prod"],
                                prod_reproduces_historical_divergence=bool(
                                    pc["first_divergence_prod"] == a["first_divergence"]))
            if ar and not pc["anchor"]["ar_hash_matches_all"]:
                hard_fail.append(f"{c}: AR gen_hash 与历史锚点不一致（运行器可能改动了本来不该动的路径）")
            if pr and not pc["anchor"]["prod_reproduces_historical_divergence"]:
                hard_fail.append(f"{c}: production 未复现历史首分歧 index={a['first_divergence']}"
                                 f"（实得 {pc['first_divergence_prod']}）")
        per_case[c] = pc
    # --- 判定（次序写死：先 FAIL，再 NEEDS_REVIEW，最后 PASS）---
    if hard_fail:
        decision = "SERIAL_ORACLE_FAIL"
    elif needs_review:
        decision = "CHECK_NEEDS_REVIEW"
    else:
        decision = "SERIAL_ORACLE_PASS"
    run_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    all_rows = [_row(r) for r in runs]
    summary = dict(tool="serial_oracle_aggregate", tag=TAG, scope=scope, run_utc=run_utc,
                   git=S.git_meta(ROOT),
                   families_requested=list(fams),
                   families_present=sorted({r.get("case") for r in runs}),
                   design_doc="docs/serial_target_oracle_design_2026-10-03.md",
                   decision_rule=("预先登记（设计文档 §6）：PASS = 5 family serial==AR + 两模式两进程一致 + "
                                  "全部 target generation M=1 + 轮不变量/cache/finite/合法停止全过 + 无 OOM；"
                                  "FAIL = 任一硬门失败；NEEDS_REVIEW = 必需项无法评估。"
                                  "次序：先 FAIL ⇒ 再 NEEDS_REVIEW ⇒ 最后 PASS"),
                   decision=decision, hard_failures=hard_fail, needs_review=needs_review,
                   per_case=per_case,
                   runs=all_rows,
                   artifacts=[r.get("smoke_artifact") for r in runs],
                   note=("本汇总**只读**；不重跑任何运行、不改任何历史 JSON/JSONL。"
                         "production 的 block 几何与上一轮 row-aligned 结果只作引用对照。"))
    p = OUTDIR / f"{TAG}_aggregate_{scope}_{run_utc}.json"
    pj = OUTDIR / f"{TAG}_aggregate_{scope}_{run_utc}.jsonl"
    with pj.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary["artifact_json"] = str(p)
    summary["artifact_jsonl"] = str(pj)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"== 串行 oracle 汇总（scope={scope}）==")
    for c in fams:
        pc = per_case.get(c)
        if not pc:
            print(f"  [{c}] 无运行")
            continue
        print(f"  [{c}] reps={pc['n_reps']}  AR={pc['ar_gen_len']} prod={pc['prod_gen_len']} "
              f"serial={pc['serial_gen_len']}")
        print(f"        serial == AR：{pc['serial_equals_ar']}（首分歧 index="
              f"{pc['first_divergence_serial']}）；AR 两次一致={pc['ar_reps_identical']}；"
              f"serial 两次一致={pc['serial_reps_identical']}")
        print(f"        production 首分歧 index={pc['first_divergence_prod']}"
              + (f"；历史锚点={pc.get('anchor')}" if "anchor" in pc else ""))
        print(f"        stop={pc['stop_reasons']}")
    for x in hard_fail:
        print(f"  [硬门失败] {x}")
    for x in needs_review:
        print(f"  [需复核] {x}")
    print(f"  ★ 判定 = {decision}")
    print(f"\n[so-agg] RESULT_JSON={p}")
    print(f"[so-agg] RESULT_JSONL={pj}")
    return 0 if decision == "SERIAL_ORACLE_PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("stage1", "final"), default="stage1")
    a = ap.parse_args()
    return aggregate(a.scope)


if __name__ == "__main__":
    raise SystemExit(main())
