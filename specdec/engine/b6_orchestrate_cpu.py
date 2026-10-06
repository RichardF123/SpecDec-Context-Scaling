"""b6_orchestrate_cpu.py — B6 GPU 批次的**编排级**判定与进度清单（只读；纯 CPU；不加载模型、不使用 GPU）。

## 定位（**不是**新检查器，不重复任何取值规则）
逐项**取值**判定由 `b6_branch_coverage_cpu`（下称 CHK）负责；本模块 **import** 它并直接调用
`find_runs` / `evaluate_case` / `missing_required_fields` / `type_problems` / `config_violations` /
`hard_checks` / `consistency_checks` / `diag_precision_checks` / `backend_report`。
本模块只回答**编排问题**：

  1. 这批 run **能不能继续**跑下一组，还是**必须立刻停**（失败即停）；
  2. 停的原因是**哪一类**（机读类别），以及**已如实记录**但不构成硬失败的东西是什么；
  3. 当前 tag 下"**已完成 / 未完成**"清单（中断后照实列出，**不**把部分完成写成整体完成）。

## 分类规则（**预登记**，运行前写死；不因看到结果而放宽）
**硬失败 ⇒ STOP**（保留全部日志与产物，不打补丁、不自动重试、不替换参数/prompt）：
  * 子进程退出码非 0；`stage != "complete"`（含 crash / checks_failed）；环境门拒绝（`blocked_by`）；
    `model_run_happened` 非真；
  * 请求的 `(mode, rep)` 槽位**缺产物**，或**重复产物**，或产物内 `tag/case/mode` 与请求不符；
  * 必需字段缺失 / 类型非法 / 冻结配置违规（`k`·`chunk`·`kv`·`max_new`·`struct_check`·dtype）；
  * `arm` 与实际臂不符（本批声明 `arm=cudnn_prefill`；Stage 2b 的 `arm=eager` 遗迹正是靠这条拦住）；
  * 硬检查失败、自洽性失败、诊断精度验收失败；后端/融合**问题**（含**非 prefill 阶段融合**）；
  * 跨模式/跨重复的输入与配置不一致；同模式两次重复的**完整序列不相等**；
    serial 与 AR 的**完整序列不相等**；
  * `--require-fused-prefill` 时该组任何一条 run 的 prefill 融合计数为 0。

**只记录 ⇒ CONTINUE**（如实写入产物与报告，**不**判控制逻辑失败、**不**当通过）：
  * production 与 AR 的完整序列差异（含长度不同）——**发散轨迹记录**；
  * 声明分支未触发 / EOS 未命中——**覆盖缺口**（用例**不得**判 PASS，但不是硬失败）；
  * 用例声明 `expect_prefill_fused=None` 且既无融合也无回退理由——**分派规则证据边界**
    （CHK 的 `insufficient`；用例不得 PASS，但不是硬失败）。

退出码：`0` CONTINUE · `1` STOP · `3` 参数非法（不写产物）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import b6_branch_coverage_cpu as CHK

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = CHK.OUTDIR
MODES = CHK.MODES
REPS = CHK.REPS
# 本批声明的**固定臂**（候选后端；见 `docs/b6_bf16_coverage_audit_plan_2026-10-05.md` §4）
DEFAULT_ARM = "cudnn_prefill"

STOP_CATEGORIES: tuple = (
    "CHILD_RC_NONZERO", "ENV_BLOCKED", "NO_MODEL_RUN", "STAGE_NOT_COMPLETE",
    "ARTIFACT_MISSING", "ARTIFACT_DUPLICATE", "IDENTITY_OR_NAME_PROBLEM",
    "FIELD_MISSING", "TYPE_ILLEGAL", "CONFIG_VIOLATION", "ARM_MISMATCH",
    "HARD_CHECK_FAIL", "CONSISTENCY_FAIL", "DIAG_PRECISION_FAIL",
    "BACKEND_FUSION_PROBLEM", "FUSED_PREFILL_REQUIRED_BUT_ABSENT",
    "CROSS_MODE_OR_REP_MISMATCH", "REPEAT_MISMATCH", "SERIAL_VS_AR_MISMATCH",
)
RECORD_ONLY_KINDS: tuple = ("PRODUCTION_DIVERGENCE", "BRANCH_UNCOVERED",
                            "DISPATCH_RULE_EVIDENCE_BOUNDARY")


def group_slots(modes, reps) -> list:
    return [(str(m), int(r)) for m in modes for r in reps]


def _strict_rep(v):
    """严格 rep：非负整数（排除 bool）。非法 ⇒ None（调用方按"非法参数"处理）。"""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _rec_arm(rec: dict):
    return (rec.get("smoke_config") or {}).get("arm")


def _prefill_fused_count(rec: dict):
    """严格读取 `fused_by_phase['prefill']`：缺失 ⇒ 0；类型非法 ⇒ None（调用方按问题处理）。"""
    raw = CHK._dig(rec, "smoke.attention_stats.fused_by_phase")
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        return None
    v = raw.get("prefill")
    if v is None:
        return 0
    return CHK._strict_int(v)


def evaluate_group(tag: str, case: str, modes, reps, spec: dict, *, outdir=None,
                   expect_backend: str | None = None, expect_arm: str | None = DEFAULT_ARM,
                   require_fused_prefill: bool = False, child_rc=None,
                   expect_max_new: int | None = None) -> dict:
    """判定一组 `(case, modes × reps)` 能否继续。返回**机读**判定字典（含停止类别与只记录项）。"""
    outdir = Path(outdir or OUTDIR)
    spec = dict(spec or {})
    modes = [str(m) for m in modes]
    reps = [int(r) for r in reps]
    expect_max_new = spec.get("max_new") if expect_max_new is None else expect_max_new
    runs, run_problems = CHK.find_runs(tag, case, outdir)
    out: dict = dict(tool="b6_orchestrate_cpu", tag=tag, case=case, modes=modes, reps=reps,
                     expect_arm=expect_arm, expect_backend=expect_backend,
                     require_fused_prefill=bool(require_fused_prefill),
                     expect_max_new=expect_max_new, child_rc=child_rc,
                     declared=dict(spec), stop_categories=[], stop_reasons=[], findings=[],
                     notes=[], per_slot={}, decision=None, run_utc=None)
    stop: dict = {}          # category -> [理由]

    def add_stop(cat: str, reason: str) -> None:
        stop.setdefault(cat, []).append(reason)

    def add_finding(kind: str, detail: str) -> None:
        out["findings"].append(dict(kind=kind, detail=detail))

    # ---- 0) 子进程退出码 ----
    if child_rc is not None and int(child_rc) != 0:
        add_stop("CHILD_RC_NONZERO", f"子进程退出码={child_rc} ≠ 0 ⇒ 立即停止（保留日志与产物）")

    # ---- 1) 槽位完整性（缺产物 / 重复产物 / 身份不符）----
    for m, r in group_slots(modes, reps):
        key = (m, r)
        slot = dict(mode=m, rep=r, file=None, present=False, duplicate=False, stage=None,
                    arm=None, prefill_fused=None, blocked_by=None, model_run_happened=None)
        if key not in runs:
            add_stop("ARTIFACT_MISSING", f"{m}/rep{r}: 产物缺失（未找到 {tag}_{case}_{m}_rep{r}_*.json）")
            out["per_slot"][f"{m}/rep{r}"] = slot
            continue
        rec = runs[key]
        if rec is None:
            slot["duplicate"] = True
            add_stop("ARTIFACT_DUPLICATE",
                     f"{m}/rep{r}: **重复产物**（CHK.find_runs 不静默择一 ⇒ 必须人工处理）")
            out["per_slot"][f"{m}/rep{r}"] = slot
            continue
        slot.update(present=True, stage=rec.get("stage"), arm=_rec_arm(rec),
                    prefill_fused=_prefill_fused_count(rec), blocked_by=rec.get("blocked_by"),
                    model_run_happened=rec.get("model_run_happened"))
        # 环境门 / 是否真的发生了模型运行
        if rec.get("blocked_by"):
            add_stop("ENV_BLOCKED", f"{m}/rep{r}: blocked_by={rec.get('blocked_by')!r}（环境门拒绝）")
        if rec.get("model_run_happened") is not True:
            add_stop("NO_MODEL_RUN", f"{m}/rep{r}: model_run_happened={rec.get('model_run_happened')!r}"
                                     "（没有发生模型运行 ⇒ 不得当作证据）")
        if rec.get("stage") != "complete":
            add_stop("STAGE_NOT_COMPLETE", f"{m}/rep{r}: stage={rec.get('stage')!r}"
                                            "（期望 'complete'）")
        # 字段/类型/配置
        miss = CHK.missing_required_fields(rec)
        if miss:
            add_stop("FIELD_MISSING", f"{m}/rep{r}: 缺字段 {miss}")
        tp = CHK.type_problems(rec)
        if tp:
            add_stop("TYPE_ILLEGAL", f"{m}/rep{r}: {tp}")
        cfg = CHK.config_violations(rec, expect_max_new=expect_max_new)
        if cfg:
            add_stop("CONFIG_VIOLATION", f"{m}/rep{r}: {cfg}")
        if expect_arm is not None and _rec_arm(rec) != expect_arm:
            add_stop("ARM_MISMATCH", f"{m}/rep{r}: arm={_rec_arm(rec)!r} ≠ 本批声明的 {expect_arm!r}"
                                     "（臂漂移 ⇒ 该 run 不构成本批证据）")
        # 硬检查 / 自洽性 / 诊断精度（逐项取值，复用 CHK）
        for name, state, detail in CHK.hard_checks(rec):
            if state == "fail":
                add_stop("HARD_CHECK_FAIL", f"{m}/rep{r}: {name} → {detail}")
            elif state == "missing":
                add_stop("FIELD_MISSING", f"{m}/rep{r}: 硬检查 {name} → {detail}")
            elif state == "illegal":
                add_stop("TYPE_ILLEGAL", f"{m}/rep{r}: 硬检查 {name} → {detail}")
        for name, state, detail in CHK.consistency_checks(rec):
            if state == "fail":
                add_stop("CONSISTENCY_FAIL", f"{m}/rep{r}: {name} → {detail}")
            elif state == "missing":
                add_stop("FIELD_MISSING", f"{m}/rep{r}: 自洽性 {name} → {detail}")
            elif state == "illegal":
                add_stop("TYPE_ILLEGAL", f"{m}/rep{r}: 自洽性 {name} → {detail}")
        for name, state, detail in CHK.diag_precision_checks(rec):
            if state == "fail":
                add_stop("DIAG_PRECISION_FAIL", f"{m}/rep{r}: {name} → {detail}")
            elif state in ("missing", "illegal"):
                add_stop("DIAG_PRECISION_FAIL", f"{m}/rep{r}: 诊断精度{state}：{name} → {detail}")
        # 后端 / 融合策略（同一策略适用于 AR/Prod/Serial；禁止非 prefill 融合）
        br = CHK.backend_report(rec, expect_backend=expect_backend,
                                expect_prefill_fused=spec.get("expect_prefill_fused"))
        slot["backend"] = dict(strategy=br.get("strategy"), requested_backend=br.get("requested_backend"),
                               fused_by_phase=br.get("fused_by_phase"),
                               s_max_by_phase=br.get("s_max_by_phase"),
                               eager_reason_kinds=br.get("eager_reason_kinds"))
        if br["problems"]:
            add_stop("BACKEND_FUSION_PROBLEM", f"{m}/rep{r}: {br['problems']}")
        for nt in (br.get("notes") or []):
            out["notes"].append(f"{m}/rep{r}: {nt}")
        for ins in br.get("insufficient") or []:
            if spec.get("expect_prefill_fused") is None:
                add_finding("DISPATCH_RULE_EVIDENCE_BOUNDARY", f"{m}/rep{r}: {ins}")
            else:
                add_stop("BACKEND_FUSION_PROBLEM", f"{m}/rep{r}: {ins}")
        # 本组是否**要求**真的发生 prefill 融合（P-B 的 8K 复核目的）
        if require_fused_prefill:
            n = slot["prefill_fused"]
            if n is None:
                add_stop("FUSED_PREFILL_REQUIRED_BUT_ABSENT",
                         f"{m}/rep{r}: fused_by_phase['prefill'] 类型非法 ⇒ 无法证明融合")
            elif n <= 0:
                add_stop("FUSED_PREFILL_REQUIRED_BUT_ABSENT",
                         f"{m}/rep{r}: 本组要求**证明** prefill 融合，但 fused_by_phase="
                         f"{br.get('fused_by_phase')}（s_max_by_phase={br.get('s_max_by_phase')}）")
        out["per_slot"][f"{m}/rep{r}"] = slot

    # ---- 2) case 级：跨模式/跨重复/严格序列门（复用 CHK.evaluate_case）----
    res = CHK.evaluate_case(case, spec, runs, expect_backend=expect_backend,
                            expect_config=dict(max_new=expect_max_new, modes=modes),
                            run_problems=run_problems)
    out["case_verdict"] = res["verdict"]
    out["case_reasons"] = list(res["reasons"])
    # 覆盖记录（**只读**、不影响 decision；含"短块合法回退 ⇒ 融合路径未被覆盖"）
    out["coverage_notes"] = res.get("coverage_notes") or {}
    for mode, per_rep in out["coverage_notes"].items():
        for rep, notes in per_rep.items():
            for nt in notes:
                out["notes"].append(f"[覆盖记录] {mode}/{rep}: {nt}")
    out["branches"] = {b: v["verdict"] for b, v in (res.get("branches") or {}).items()}
    out["branch_observations"] = res.get("branch_observations_total")
    for p in res.get("duplicate_or_identity_problems") or []:
        add_stop("IDENTITY_OR_NAME_PROBLEM", p)
    gate = res.get("gate") or {}
    for rep, d in (gate.get("serial_vs_ar") or {}).items():
        if not d.get("equal"):
            add_stop("SERIAL_VS_AR_MISMATCH",
                     f"serial/{rep} 与 AR **完整序列不相等**（{d['kind']}：共同前缀 {d['common_prefix']}、"
                     f"len {d['len_a']} vs {d['len_b']}）")
    for mode, eq in (gate.get("repeats") or {}).items():
        if eq is False:
            add_stop("REPEAT_MISMATCH", f"{mode}: rep1 与 rep2 **完整序列不相等**")
    for p in (gate.get("comparability_problems") or []):
        add_stop("CROSS_MODE_OR_REP_MISMATCH", p)
    # production 差异：**只记录**
    for rep, d in (gate.get("production_vs_ar") or {}).items():
        if not d.get("equal"):
            add_finding("PRODUCTION_DIVERGENCE",
                        f"production/{rep} 与 AR 不同（{d['kind']}：共同前缀 {d['common_prefix']}、"
                        f"len {d['len_a']} vs {d['len_b']}、首个 token 差异位置 "
                        f"{d['first_token_diff']}）——只作发散轨迹记录")
        else:
            out["notes"].append(f"production/{rep} 与 AR 完整序列相等（仅本组记录）")
    for b, v in out["branches"].items():
        if v == "UNCOVERED":
            add_finding("BRANCH_UNCOVERED", f"声明分支 {b} 在本组范围内未触发（覆盖缺口，非硬失败）")

    # ---- 3) 结论 ----
    out["stop_categories"] = sorted(stop)
    for cat in out["stop_categories"]:
        out["stop_reasons"] += [f"[{cat}] {x}" for x in stop[cat]]
    out["decision"] = "STOP" if out["stop_categories"] else "CONTINUE"
    out["decision_reason_zh"] = (
        "存在硬失败 ⇒ **立即停止**（不重试、不替换参数、不覆盖产物；等待人工指示）"
        if out["decision"] == "STOP" else
        "无硬失败 ⇒ 可继续下一组（只记录项不影响继续，但**用例整体**是否 PASS 由 CHK 判定）")
    out["verdict_note"] = ("`decision` 只回答『能否继续编排』；**不**等价于用例通过。"
                           "用例是否 PASS 由 CHK 的 `case_verdict`（CASE_PASS/INCOMPLETE/FAIL）决定，"
                           "且**只记录项**（覆盖缺口/证据边界）会使用例**不得**判 PASS。")
    out["run_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return out


def progress(tag: str, plan: dict, *, outdir=None) -> dict:
    """**进度清单**：按计划列出"已完成 / 未完成"。只读；不做任何判定。"""
    outdir = Path(outdir or OUTDIR)
    cases = [str(c) for c in plan["cases"]]
    modes = [str(m) for m in plan["modes"]]
    reps = [int(r) for r in plan["reps"]]
    done, pending, problems = [], [], []
    per_case: dict = {}
    for case in cases:
        runs, probs = CHK.find_runs(tag, case, outdir)
        problems += [f"{case}: {p}" for p in probs]
        for m, r in group_slots(modes, reps):
            key = (m, r)
            slot = f"{case}/{m}/rep{r}"
            if key in runs and runs[key] is not None:
                done.append(slot)
                per_case.setdefault(case, {}).setdefault(m, []).append(r)
            elif key in runs:
                pending.append(slot + "（重复产物，需人工处理）")
            else:
                pending.append(slot)
    return dict(tool="b6_orchestrate_cpu:progress", tag=tag, plan=dict(cases=cases, modes=modes,
                 reps=reps), n_planned=len(cases) * len(modes) * len(reps),
                 n_completed=len(done), n_pending=len(pending), completed=done, pending=pending,
                 by_case=per_case, duplicate_or_identity_problems=problems,
                 note="只读进度：已完成/未完成照实列出；**部分完成不得写成整体完成**",
                 run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))


def _case_spec(case: str, declared_file: str | None) -> dict:
    from . import b6_cases as BC
    if declared_file:
        d = json.loads(Path(declared_file).read_text(encoding="utf-8-sig"))
        cm = d.get("cases") if isinstance(d.get("cases"), dict) else d
        if case not in cm:
            raise KeyError(f"声明文件里没有用例 {case!r}")
        return dict(cm[case])
    for c in BC.B6_CASES:
        if c["case_id"] == case:
            return dict(c)
    raise KeyError(f"内建用例表里没有 {case!r}（可用：{[c['case_id'] for c in BC.B6_CASES]}）")


def _write(outdir: Path, tag: str, payload: dict, name: str) -> str:
    p = Path(outdir) / name
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="B6 GPU 批次编排级判定（只读 CPU；不加载模型/不用 GPU）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--reps", default=",".join(str(r) for r in REPS))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--declared", default=None, help="用例声明 JSON（省略则用内建 b6_cases.py）")
    ap.add_argument("--expect-backend", default="CUDNN")
    ap.add_argument("--expect-arm", default=DEFAULT_ARM)
    ap.add_argument("--require-fused-prefill", action="store_true")
    ap.add_argument("--child-rc", type=int, default=None)
    ap.add_argument("--progress", action="store_true", help="只输出进度清单（需 --plan-cases）")
    ap.add_argument("--plan-cases", default=None)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    modes = [x.strip().lower() for x in a.modes.split(",") if x.strip()]
    reps = [x.strip() for x in a.reps.split(",") if x.strip()]
    if any(not r.isdigit() for r in reps) or not modes or not reps:
        print("[orch] 参数非法（modes/reps）⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    reps = [int(r) for r in reps]
    bad_mode = [m for m in modes if m not in MODES]
    if bad_mode:
        print(f"[orch] 非法 mode={bad_mode}（允许 {list(MODES)}）⇒ 退出码 3")
        return 3
    outdir = Path(a.outdir)
    if a.progress:
        if not a.plan_cases:
            print("[orch] --progress 需要 --plan-cases ⇒ 退出码 3")
            return 3
        p = progress(a.tag, dict(cases=a.plan_cases.split(","), modes=modes, reps=reps),
                     outdir=outdir)
        print(f"== 进度 tag={a.tag} 计划 {p['n_planned']} · 已完成 {p['n_completed']} · "
              f"未完成 {p['n_pending']} ==")
        for s in p["completed"]:
            print(f"  [done]    {s}")
        for s in p["pending"]:
            print(f"  [pending] {s}")
        for x in p["duplicate_or_identity_problems"]:
            print(f"  [warn]    {x}")
        if not a.no_write:
            print(f"[orch] PROGRESS_JSON={_write(outdir, a.tag, p, f'{a.tag}_b6_orchestrate_progress.json')}")
        return 0
    if not a.case:
        print("[orch] 需要一个 --case（或用 --progress --plan-cases）⇒ 退出码 3")
        return 3
    try:
        spec = _case_spec(a.case, a.declared)
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[orch] 用例声明不可用：{e} ⇒ 退出码 3")
        return 3
    d = evaluate_group(a.tag, a.case, modes, reps, spec, outdir=outdir,
                       expect_backend=a.expect_backend, expect_arm=(a.expect_arm or None),
                       require_fused_prefill=a.require_fused_prefill, child_rc=a.child_rc)
    print(f"== 编排判定 tag={a.tag} case={a.case} modes={modes} reps={reps} ⇒ "
          f"**{d['decision']}**（CHK 用例判定={d['case_verdict']}）==")
    for s in d["stop_reasons"]:
        print(f"  [STOP] {s}")
    for f in d["findings"]:
        print(f"  [record-only] [{f['kind']}] {f['detail']}")
    for n in d["notes"]:
        print(f"  [note] {n}")
    print(f"  {d['decision_reason_zh']}")
    if not a.no_write:
        rname = "{0}_{1}_b6_orchestrate_{2}.json".format(a.tag, a.case, d["run_utc"])
        print(f"[orch] RESULT_JSON={_write(outdir, a.tag, d, rname)}")
        ev = outdir / f"{a.tag}_b6_orchestrate_events.jsonl"
        with ev.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(run_utc=d["run_utc"], tag=a.tag, case=a.case, modes=modes,
                                     reps=reps, decision=d["decision"],
                                     stop_categories=d["stop_categories"],
                                     case_verdict=d["case_verdict"],
                                     n_findings=len(d["findings"]),
                                     findings_kinds=sorted({f["kind"] for f in d["findings"]})),
                                ensure_ascii=False) + "\n")
    return 0 if d["decision"] == "CONTINUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
