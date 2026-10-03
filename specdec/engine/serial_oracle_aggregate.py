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


def _n_gen_s_gt1(tgt: dict) -> int | None:
    """**非 prefill 阶段**里 `seq_len > 1` 的 target 调用数（需求项 5 的"真正的 block 前向"口径）。

    ⚠️ 命名澄清（**必须写进报告**）：生产 `specdec_stats.target_block_calls` 的历史含义是
    "**每轮一次 verify 调用**"的计数器（`spec_dec` 主循环里 `st["target_block_calls"] += 1`），
    它**不是**"`s>1` 的 block 前向数"。serial 模式下该计数器仍等于**轮数**。
    因此需求项 5"`target_block_calls == 0`"的**实质**（不存在 block 前向）由本函数给出：
    从审计的逐阶段 `seq_len` 明细里数出非 prefill 且 `s>1` 的调用数，必须为 **0**。
    两者都如实记录，不做任何"改名冒充"。
    """
    if not tgt:
        return None
    n = 0
    for ph, d in (tgt.get("by_phase") or {}).items():
        if ph == "prefill":
            continue
        n += sum(1 for s in (d.get("seq_lens") or []) if int(s) > 1)
    return n


def _prefill_seq_lens(tgt: dict) -> list | None:
    """prefill 阶段的 `seq_len` 列表（需求项 9：分块 prefill 不得被误判为 generation）。"""
    if not tgt:
        return None
    return (tgt.get("by_phase") or {}).get("prefill", {}).get("seq_lens")


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
                target_label=(tgt.get("label")), target_label_source=(tgt.get("label_source")),
                target_n_layers=(tgt.get("n_layers_first")), target_n_params=(tgt.get("n_params")),
                target_by_phase=(tgt.get("by_phase")),
                target_n_prefill_calls=(tgt.get("n_prefill_calls")),
                # ★ 需求项 5：**真正**的 block（`s>1`）generation 前向计数（由审计的逐阶段 seq_len 明细数出来）
                target_n_gen_calls_with_s_gt_1=_n_gen_s_gt1(tgt),
                target_prefill_seq_lens=_prefill_seq_lens(tgt),
                audit_n_calls=(au.get("n_calls")), audit_labels=(au.get("labels")),
                audit_cross_check=(au.get("cross_check")),
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
                diagnostic_only=rec.get("diagnostic_only"),
                smoke_rc_zero=((rec.get("self_checks") or {}).get("smoke_rc_zero")),
                end_state_gen_hash=((sm.get("end_state") or {}).get("gen_hash")),
                end_state_len=((sm.get("end_state") or {}).get("gen_len")),
                end_state_stop=((sm.get("end_state") or {}).get("stop_reason")),
                eos_seen=((sm.get("end_state") or {}).get("eos_seen")),
                cache_structure_ok=sm.get("cache_structure_ok"),
                cache_structure_n_checks=sm.get("cache_structure_n_checks"),
                # ⚠️ 缺陷 D7：harness 把有限性计数放在 `checks.finiteness_coverage.n_checked`
                #    （首版误读 `checks.n_checked` ⇒ 恒为 None ⇒ 需求项 13 被**误判为失败**）。
                #    这里按记录里的**真实路径**读取，并附带 first_bad / 逐阶段 ok / uncovered。
                finiteness_n_checked=(((sm.get("checks") or {}).get("finiteness_coverage") or {})
                                      .get("n_checked")),
                finiteness_first_bad=(((sm.get("checks") or {}).get("finiteness_coverage") or {})
                                      .get("first_bad")),
                finiteness_by_phase_all_ok=all(
                    bool(v.get("ok")) for v in (((sm.get("checks") or {}).get("finiteness_coverage") or {})
                                                .get("by_phase_model") or {}).values()) or None,
                finiteness_by_phase=(((sm.get("checks") or {}).get("finiteness_coverage") or {})
                                     .get("by_phase_model")),
                checks_n_required=((sm.get("checks") or {}).get("n_required")),
                checks_uncovered=((sm.get("checks") or {}).get("uncovered")),
                kv_format=sm.get("kv_format"), weights_bitmatch=sm.get("weights_bitmatch"),
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


# ===========================================================================
# Stage 1（F13）专用判定：四类判决 + 17 项硬性检查
#   `SERIAL_ORACLE_STAGE1_PASS` / `SERIAL_ORACLE_STAGE1_FAIL` / `BASELINE_REGRESSION` /
#   `CHECK_NEEDS_REVIEW`（本轮任务书 §7 的规定）。
#   `--scope final`（5 family）仍沿用**设计文档 §6** 预先登记的
#   `SERIAL_ORACLE_PASS/FAIL/CHECK_NEEDS_REVIEW` 口径，两套口径并存、不互相替换。
#
# 判定次序（写死）：先"能否评估"（缺运行/环境/审计无法证明非空 ⇒ NEEDS_REVIEW）
#   ⇒ 再基线锚点（AR/production 复现失败 ⇒ BASELINE_REGRESSION）
#   ⇒ 再 serial 硬门（⇒ SERIAL_ORACLE_STAGE1_FAIL）⇒ 最后 PASS。
# ===========================================================================
STAGE1_FAMILY = "F13"
STAGE1_REQUIRED_REPS = 2
STAGE1_ANCHORS = dict(
    ar=dict(expected_gen_hash="6e7522bf2521ad7b", expected_len=64, expected_stop="max_new_reached"),
    prod=dict(expected_gen_hash="e90dad2b3fcf1c65", expected_len=64, expected_stop="max_new_reached",
              expected_first_divergence_vs_ar=47))
LEGAL_STOPS = ("max_new_reached", "eos")


def _serial_hard_checks(r: dict) -> dict:
    """**需求项 1–16**（逐 run）。返回 `{name: dict(ok=bool|None, detail=...)}`。

    `ok=None` 表示**无法评估**（字段缺失）⇒ 归入 `CHECK_NEEDS_REVIEW`，不得当作通过。
    """
    tr = r.get("_trace") or {}
    rc_ = r.get("_rounds") or {}
    cr = r.get("_cross") or {}
    need_fields = ("pre", "Kp", "m")
    out: dict = {}
    out["01_diagnostic_only"] = dict(ok=(r.get("diagnostic_only") is True),
                                     detail=f"diagnostic_only={r.get('diagnostic_only')}")
    out["02_verify_mode_serial"] = dict(
        ok=(r.get("verify_mode") == "serial"),
        detail=f"运行期 target_verify_mode={r.get('verify_mode')}；运行后={r.get('verify_mode_after')}")
    out["03_row_aligned_off"] = dict(ok=(r.get("row_aligned_wrap_active") is False),
                                     detail=f"row_aligned_wrap_active={r.get('row_aligned_wrap_active')}")
    out["04_all_target_gen_s_eq_1"] = dict(ok=(r.get("target_all_m1") is True),
                                          detail=f"generation_seq_lens={r.get('target_gen_seq_lens')}")
    out["05_no_block_generation_forward"] = dict(
        ok=(r.get("target_n_gen_calls_with_s_gt_1") == 0),
        detail=(f"非 prefill 且 s>1 的 target 调用数={r.get('target_n_gen_calls_with_s_gt_1')}"
                f"（注意 `specdec_stats.target_block_calls`={((r.get('specdec_stats') or {}).get('target_block_calls'))}"
                f" 是**每轮一次 verify 调用**的计数器，不是 block 前向数）"))
    out["06_audit_nonvacuous"] = dict(
        ok=bool(r.get("audit_sane") is True and (r.get("target_gen_calls") or 0) > 0),
        detail=f"audit_sane={r.get('audit_sane')} target_gen_calls={r.get('target_gen_calls')} "
               f"audit_n_calls={r.get('audit_n_calls')}")
    out["07_accounting_identity"] = dict(ok=((r.get("accounting") or {}).get("matches") is True),
                                         detail=str(r.get("accounting")))
    out["08_identity_target_vs_draft"] = dict(
        ok=bool(r.get("target_label") == "target"
                and r.get("target_label_source") == "smoke_hooks._HOOK_REGISTRY.tgt"
                and cr.get("layer_counts_differ") is True and cr.get("target_params_gt_draft") is True),
        detail=f"label={r.get('target_label')} source={r.get('target_label_source')} cross={cr}")
    _pref = r.get("target_prefill_seq_lens") or []
    out["09_prefill_chunked_not_generation"] = dict(
        ok=bool((r.get("target_n_prefill_calls") or 0) >= 1 and _pref and max(_pref) > 1
                and r.get("target_n_gen_calls_with_s_gt_1") == 0),
        detail=f"n_prefill_calls={r.get('target_n_prefill_calls')} prefill_seq_lens={_pref}"
               f"（分块 prefill 用大 s；generation 里 s>1 计数必须为 0）")
    n_rounds = r.get("n_rounds")
    out["10_per_round_trace_complete"] = dict(
        ok=bool(rc_.get("ok") is True and (r.get("serial_trace_len") or 0) == (n_rounds or -1)
                and n_rounds and all(k in rc_.get("fields", []) for k in
                                     ("pre", "Kp", "m", "cands", "g", "emitted", "tgt_len", "drf_len"))
                if rc_.get("fields") is not None else None),
        detail=(f"n_rounds={n_rounds} serial_trace_len={r.get('serial_trace_len')} "
                f"round_fields={rc_.get('fields')} missing={rc_.get('missing')} "
                f"trace_check={tr.get('ok')}"))
    out["11_round_invariants"] = dict(
        ok=bool(rc_.get("ok") is True and r.get("round_invariants_all_ok") is True),
        detail=f"独立重算 ok={rc_.get('ok')} bad={rc_.get('bad')}；"
               f"harness all_ok={r.get('round_invariants_all_ok')}")
    out["12_cache_structure"] = dict(
        ok=None if r.get("cache_structure_ok") is None
        else bool(r.get("cache_structure_ok") is True and (r.get("cache_structure_n_checks") or 0) > 0),
        detail=f"cache_structure_ok={r.get('cache_structure_ok')} "
               f"n_checks={r.get('cache_structure_n_checks')}")
    # 需求项 13（finite）：**加强**而非放宽 —— 除 ①prefill 有限 ②harness 判定通过 ③有限性检查数>0 外，
    # 还要求 ④无未覆盖项 ⑤逐阶段有限性全 ok ⑥first_bad 为空 ⑦cache 结构检查通过。
    # （缺陷 D7：首版把有限性计数读成 `checks.n_checked`（不存在）⇒ 误判为失败；改为真实路径后
    #  本条检查**更严**：serial 记录实测 210 次有限性检查、14/14 必需项、uncovered=[]、first_bad=null。）
    out["13_finite"] = dict(
        ok=bool(r.get("prefill_finite") is not False
                and r.get("checks_passed") is True
                and (r.get("finiteness_n_checked") or 0) > 0
                and (r.get("checks_uncovered") in ([], None))
                and r.get("finiteness_first_bad") is None
                and r.get("finiteness_by_phase_all_ok") is True
                and r.get("cache_structure_ok") is True),
        detail=f"prefill_finite={r.get('prefill_finite')} checks_passed={r.get('checks_passed')} "
               f"finiteness_n_checked={r.get('finiteness_n_checked')} "
               f"first_bad={r.get('finiteness_first_bad')} "
               f"by_phase_all_ok={r.get('finiteness_by_phase_all_ok')} "
               f"uncovered={r.get('checks_uncovered')} cache_structure_ok={r.get('cache_structure_ok')}")
    out["14_gen_ids_in_vocab"] = dict(ok=(r.get("gen_ids_in_vocab") is not False),
                                      detail=f"gen_ids_in_vocab={r.get('gen_ids_in_vocab')}")
    out["15_stop_reason_legal"] = dict(
        ok=bool(r.get("stop_reason") in LEGAL_STOPS and r.get("stop_reason") == r.get("_ar_stop")),
        detail=f"stop={r.get('stop_reason')} AR_stop={r.get('_ar_stop')} legal={LEGAL_STOPS}")
    out["16_no_oom"] = dict(ok=bool(r.get("smoke_rc_zero") is True and r.get("model_run_happened") is True
                                    and (r.get("stage") or "") not in ("crashed",)),
                            detail=f"stage={r.get('stage')} smoke_rc_zero={r.get('smoke_rc_zero')} "
                                   f"peak_res_mib={r.get('peak_res_mib')}")
    return out


def _stage1_rounds_fields(rec: dict) -> dict:
    """需求项 10：逐轮 trace 是否**字段完整**（并给出缺失项），以及 trace/round 一致性。"""
    rounds = (rec.get("smoke") or {}).get("rounds") or []
    tr = rec.get("serial_trace") or []
    need = ("pre", "Kp", "m", "cands", "g", "emitted", "tgt_len", "drf_len",
            "rollback", "repair", "correction", "all_accept", "tgt_len_after", "drf_len_after")
    missing = sorted({k for r in rounds for k in need if k not in r})
    trc = _trace_checks(rec)
    return dict(n_rounds=len(rounds), fields=(sorted(need) if not missing else None) if rounds else None,
                missing=missing, ok=bool(rounds and not missing), trace=trc,
                note=("逐轮字段：pre（轮前 cache 长度=target/draft 的 before）/ Kp / m / cands（proposals）/ "
                      "g（target predictions）/ emitted / tgt_len·drf_len（after）/ rollback·repair / "
                      "correction（replacement）/ all_accept（=bonus 情形，bonus 在下一轮 emit）；"
                      "serial_trace 另给每步 cache 长度与逐步 argmax"))


def aggregate_stage1() -> int:
    from .. import results_schema as S
    runs = _load_runs()
    rows_all = [_row(r) for r in runs]
    for r, raw in zip(rows_all, runs):
        r["_rounds"] = _stage1_rounds_fields(raw)
        r["_trace"] = _trace_checks(raw)
        r["_cross"] = (((raw.get("audit") or {}).get("cross_check")) or {})
    by: dict = {m: [] for m in ("ar", "prod", "serial")}
    env_blocked: list = []
    for r in rows_all:
        if r.get("blocked_by") == "environment" or r.get("model_run_happened") is False:
            env_blocked.append(r)
        else:
            by.setdefault(r.get("mode"), []).append(r)
    ar, pr, se = by.get("ar", []), by.get("prod", []), by.get("serial", [])
    ar_ref = ar[0]["gen"] if ar else None
    for r in pr + se:
        r["_ar_stop"] = (ar[0]["stop_reason"] if ar else None)
        r["_first_div_vs_ar"] = (_first_div(r["gen"], ar_ref) if ar_ref is not None else None)

    needs_review: list = []
    baseline_fail: list = []
    serial_fail: list = []
    # ---- 0) 能否评估 ----
    # ⚠️ 缺陷 D8（**判定范围**修正）：环境不合格的记录若来自**本轮之前的废弃尝试**，
    #    必须**如实列出**（`env_blocked_records`）但**不能**当成"本轮缺证据"——
    #    本轮的环境门是 ENVIRONMENT_OK、六次运行各自的进程内环境门也都合格。
    #    只有当它**导致必需重复数不足**（<2）时才算"无法评估"。这样既不放宽，
    #    也避免把历史探测记录误读成本轮结论。
    env_blocked_records = []
    for r in env_blocked:
        cg = r.get("commit_guard") or {}
        env_blocked_records.append(dict(
            file=r["file"], mode=r["mode"], rep=r["rep"], stage=r.get("stage"),
            why=str(cg.get("why"))[:140], run_id=None,
            note="本轮之前的废弃尝试记录（仅报告，不进入本轮判决）"))
    for m, lst in (("ar", ar), ("prod", pr), ("serial", se)):
        if len(lst) < STAGE1_REQUIRED_REPS:
            n_env = len([x for x in env_blocked if x.get("mode") == m])
            needs_review.append(f"{m}: 可用运行 {len(lst)}/{STAGE1_REQUIRED_REPS}"
                                f"（其中环境不合格 {n_env} 条）⇒ 缺重复，无法评估可重复性")
    for r in rows_all:
        if r.get("stage") == "crashed" or r.get("post_processing_error"):
            needs_review.append(f"{r['mode']}/rep{r['rep']}: 运行器自身失败 stage={r.get('stage')} "
                                f"{r.get('post_processing_error')}")
    # ---- 1) 基线锚点（AR / production）----
    ar_hash_ok = bool(ar and all(r["gen_hash"] == STAGE1_ANCHORS["ar"]["expected_gen_hash"] for r in ar))
    ar_len_ok = bool(ar and all(r["gen_len"] == STAGE1_ANCHORS["ar"]["expected_len"] for r in ar))
    ar_stop_ok = bool(ar and all(r["stop_reason"] == STAGE1_ANCHORS["ar"]["expected_stop"] for r in ar))
    ar_reps_ok = bool(len(ar) >= 2 and len({tuple(r["gen"]) for r in ar}) == 1)
    ar_selfhash_ok = bool(ar and all(r["gen_hash"] == r.get("end_state_gen_hash") for r in ar))
    if ar and not (ar_hash_ok and ar_len_ok and ar_stop_ok and ar_reps_ok):
        baseline_fail.append(f"AR 未复现历史锚点：hash_ok={ar_hash_ok}（实得 "
                             f"{[r['gen_hash'] for r in ar]}，期望 {STAGE1_ANCHORS['ar']['expected_gen_hash']}）"
                             f" len_ok={ar_len_ok} stop_ok={ar_stop_ok} reps_ok={ar_reps_ok}")
    prod_hash_ok = bool(pr and all(r["gen_hash"] == STAGE1_ANCHORS["prod"]["expected_gen_hash"] for r in pr))
    prod_len_ok = bool(pr and all(r["gen_len"] == STAGE1_ANCHORS["prod"]["expected_len"] for r in pr))
    prod_stop_ok = bool(pr and all(r["stop_reason"] == STAGE1_ANCHORS["prod"]["expected_stop"] for r in pr))
    prod_reps_ok = bool(len(pr) >= 2 and len({tuple(r["gen"]) for r in pr}) == 1)
    prod_div = (pr[0]["_first_div_vs_ar"] if pr else None)
    prod_div_ok = bool(prod_div == STAGE1_ANCHORS["prod"]["expected_first_divergence_vs_ar"])
    if pr and not (prod_hash_ok and prod_len_ok and prod_stop_ok and prod_reps_ok and prod_div_ok):
        baseline_fail.append(f"production 未复现历史锚点：hash_ok={prod_hash_ok}（实得 "
                             f"{[r['gen_hash'] for r in pr]}，期望 {STAGE1_ANCHORS['prod']['expected_gen_hash']}）"
                             f" len_ok={prod_len_ok} stop_ok={prod_stop_ok} reps_ok={prod_reps_ok} "
                             f"首分歧={prod_div}（期望 "
                             f"{STAGE1_ANCHORS['prod']['expected_first_divergence_vs_ar']}）")
    # ---- 2) serial 硬性检查（需求项 1–16）----
    hard: dict = {}
    for r in se:
        key = f"rep{r['rep']}"
        hard[key] = _serial_hard_checks(r)
        for name, d in hard[key].items():
            if d["ok"] is None:
                needs_review.append(f"serial/{key}: 无法评估 `{name}`（字段缺失）{d['detail']}")
            elif d["ok"] is False:
                serial_fail.append(f"serial/{key}: `{name}` 失败 —— {d['detail']}")
    # ---- 3) serial 与 AR 的 token equality（需求项 17 + 核心命题）----
    serial_eq_ar = bool(se and ar_ref is not None and all(r["gen"] == ar_ref for r in se))
    serial_reps_ok = bool(len(se) >= 2 and len({tuple(r["gen"]) for r in se}) == 1)
    div_rows = [dict(rep=r["rep"], first_divergence=r["_first_div_vs_ar"],
                     ar_token=(ar_ref[r["_first_div_vs_ar"]] if (ar_ref and r["_first_div_vs_ar"] is not None
                                                                 and r["_first_div_vs_ar"] < len(ar_ref)) else None),
                     serial_token=(r["gen"][r["_first_div_vs_ar"]] if (r["_first_div_vs_ar"] is not None
                                                                       and r["_first_div_vs_ar"] < len(r["gen"]))
                                   else None))
                for r in se]
    if se and ar_ref is not None and not serial_eq_ar:
        serial_fail.append(f"serial 与 AR 的 token IDs 不一致：{div_rows}")
    if se and not serial_reps_ok:
        serial_fail.append("两次 serial 运行不完全一致")
    # 首个分歧轮的**记录**（只记录，明确不做算子级定位）
    div_round = None
    if se and se[0]["_first_div_vs_ar"] is not None:
        idx = int(se[0]["_first_div_vs_ar"])
        cum = 0
        for rr in (se[0].get("_rounds") or {}).get("rounds_detail") or []:
            if cum <= idx < cum + len(rr.get("emitted") or []):
                div_round = rr
                break
            cum += len(rr.get("emitted") or [])
    # ---- 4) 判决（次序写死）----
    if needs_review:
        decision = "CHECK_NEEDS_REVIEW"
    elif baseline_fail:
        decision = "BASELINE_REGRESSION"
    elif serial_fail:
        decision = "SERIAL_ORACLE_STAGE1_FAIL"
    else:
        decision = "SERIAL_ORACLE_STAGE1_PASS"
    run_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary = dict(
        tool="serial_oracle_aggregate_stage1", tag=TAG, scope="stage1", family=STAGE1_FAMILY,
        run_utc=run_utc, git=S.git_meta(ROOT),
        decision_rule=("本轮任务书 §7：PASS = AR 两进程一致并复现历史 hash；production 两进程一致并复现 "
                       "hash 与 index 47；serial 两进程一致且 64/64 与 AR 相同；全部几何/缓存/轮/有限性/停止检查通过。"
                       "次序：先判（能否评估：缺运行/环境不合格/字段缺失 ⇒ CHECK_NEEDS_REVIEW）"
                       "⇒ 再判基线锚点（⇒ BASELINE_REGRESSION）⇒ 再判 serial 硬门"
                       "（⇒ SERIAL_ORACLE_STAGE1_FAIL）⇒ 最后 PASS"),
        pre_registered_anchors=STAGE1_ANCHORS, required_reps=STAGE1_REQUIRED_REPS,
        default_off="serial 模式仅作用于 target verification；row-aligned 必须关闭",
        n_runs_total=len(rows_all), n_runs_used=len(ar) + len(pr) + len(se),
        n_env_blocked=len(env_blocked),
        anchors=dict(ar=dict(hash_ok=ar_hash_ok, len_ok=ar_len_ok, stop_ok=ar_stop_ok,
                             reps_identical=ar_reps_ok, self_reported_hash_agrees=ar_selfhash_ok,
                             hashes=[r["gen_hash"] for r in ar], lens=[r["gen_len"] for r in ar],
                             stops=[r["stop_reason"] for r in ar],
                             expected=STAGE1_ANCHORS["ar"]),
                     prod=dict(hash_ok=prod_hash_ok, len_ok=prod_len_ok, stop_ok=prod_stop_ok,
                               reps_identical=prod_reps_ok,
                               first_divergence=prod_div, first_divergence_ok=prod_div_ok,
                               hashes=[r["gen_hash"] for r in pr], lens=[r["gen_len"] for r in pr],
                               stops=[r["stop_reason"] for r in pr],
                               expected=STAGE1_ANCHORS["prod"])),
        serial=dict(reps_identical=serial_reps_ok, equals_ar=serial_eq_ar,
                    hashes=[r["gen_hash"] for r in se], lens=[r["gen_len"] for r in se],
                    stops=[r["stop_reason"] for r in se],
                    first_divergence_vs_ar=[r["_first_div_vs_ar"] for r in se],
                    divergence_detail=div_rows,
                    geometry=dict(target_gen_calls=[r.get("target_gen_calls") for r in se],
                                  generation_seq_lens=[r.get("target_gen_seq_lens") for r in se],
                                  n_block_generation_forwards=[r.get("target_n_gen_calls_with_s_gt_1") for r in se],
                                  accounting_matches=[(r.get("accounting") or {}).get("matches") for r in se],
                                  prefill_seq_lens=[r.get("target_prefill_seq_lens") for r in se],
                                  specdec_target_block_calls=[(r.get("specdec_stats") or {}).get("target_block_calls")
                                                              for r in se])),
        hard_checks=hard,
        env_blocked_records=env_blocked_records,
        needs_review=needs_review, baseline_failures=baseline_fail, serial_failures=serial_fail,
        decision=decision, runs=rows_all,
        next_round_allowed=(decision == "SERIAL_ORACLE_STAGE1_PASS"),
        next_round_note=("仅当判决为 SERIAL_ORACLE_STAGE1_PASS 时才**建议**下一轮执行 F01/F02/F08/F10 Stage 2；"
                         "本轮**不执行** Stage 2"))
    p = OUTDIR / f"{TAG}_stage1_aggregate_{run_utc}.json"
    pj = OUTDIR / f"{TAG}_stage1_aggregate_{run_utc}.jsonl"
    with pj.open("w", encoding="utf-8") as fh:
        for r in rows_all:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary["artifact_json"] = str(p)
    summary["artifact_jsonl"] = str(pj)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"== Stage 1（F13）聚合 ==")
    print(f"  运行：total={len(rows_all)} used={len(ar) + len(pr) + len(se)} env_blocked={len(env_blocked)}"
          f"（AR {len(ar)} / prod {len(pr)} / serial {len(se)}）")
    print(f"  AR 锚点：hash_ok={ar_hash_ok} len_ok={ar_len_ok} stop_ok={ar_stop_ok} "
          f"reps={ar_reps_ok} hashes={[r['gen_hash'] for r in ar]}")
    print(f"  production 锚点：hash_ok={prod_hash_ok} len_ok={prod_len_ok} stop_ok={prod_stop_ok} "
          f"reps={prod_reps_ok} 首分歧={prod_div}（期望 "
          f"{STAGE1_ANCHORS['prod']['expected_first_divergence_vs_ar']}）")
    print(f"  serial 几何：seq_lens={[r.get('target_gen_seq_lens') for r in se]} "
          f"block_forwards={[r.get('target_n_gen_calls_with_s_gt_1') for r in se]}")
    print(f"  serial == AR：{serial_eq_ar}（首分歧 {[r['_first_div_vs_ar'] for r in se]}）；"
          f"两进程一致={serial_reps_ok}")
    for x in env_blocked_records:
        print(f"  [环境不合格·仅报告] {x['file']}（{x['mode']}/rep{x['rep']}）{x['why']}")
    for x in needs_review:
        print(f"  [需复核] {x}")
    for x in baseline_fail:
        print(f"  [基线回归] {x}")
    for x in serial_fail:
        print(f"  [serial 硬门失败] {x}")
    print(f"  ★ 判决 = {decision}")
    print(f"\n[so-s1] RESULT_JSON={p}")
    print(f"[so-s1] RESULT_JSONL={pj}")
    return 0 if decision == "SERIAL_ORACLE_STAGE1_PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("stage1", "final"), default="stage1")
    a = ap.parse_args()
    if a.scope == "stage1":
        return aggregate_stage1()
    return aggregate(a.scope)


if __name__ == "__main__":
    raise SystemExit(main())
