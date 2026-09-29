"""verify_readiness_selftest_cpu.py — G1–G5 与五类状态的 CPU 回归自测（不加载模型、不碰 CUDA）。

覆盖本轮要求的 8 类：
  1. 默认路径不变（未设 SMOKE_FAMILY ⇒ legacy，入口不改变原行为）
  2. manifest / hash 不匹配 ⇒ **拒绝**（不降级、不警告放行）
  3. 必需检查缺失 ⇒ **拒绝**（复用契约 v2 的 `prefill_finiteness`）
  4. AR/Spec 分歧 ⇒ 在 `on_divergence=record` 下**可记录继续**
  5. 结构失败 ⇒ **停止**
  6. 重复不一致 ⇒ **停止**（并标待查）
  7. 异常短输出 ⇒ **停止**（`stop_reason=other_short`）
  8. **缺失运行不得生成"完整成功"汇总**

另加：正式判据**结构上不可能**报 pass；环境拒绝与算法失败**分开分类**。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import family16_summary as SUM
from . import family_input as FI
from . import status_fields as SF
from .smoke_checks import prefill_finiteness, required_prefill_models


def _run(*, gen=None, stop="max_new_reached", stage="complete", max_new=64, checks_passed=True,
         round_inv_ok=True, cache_ok=None, family=None, kv="base_bf16", mode="ar",
         pf_all_required_ok=True, fail_names=None):
    gen = list(range(max_new)) if gen is None else gen
    return dict(
        stage=stage, mode=mode, kv=kv,
        run=dict(gen=gen, gen_len=len(gen), max_new=max_new, eos_id=151645,
                 stop_reason=stop,
                 prefill_logits_finite_by_model=dict(all_required_ok=pf_all_required_ok,
                                                     missing_records=[] if pf_all_required_ok else ["draft"])),
        checks=dict(passed=checks_passed, failures=[dict(name=n) for n in (fail_names or [])]),
        round_invariants=dict(all_ok=round_inv_ok),
        cache_structure_ok=cache_ok,
        input=dict(family_id=family, source_kind=("manifest" if family else "legacy")),
    )


def _run_() -> int:
    npass = nfail = 0

    def check(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    print("== 1. 默认路径不变（G1 的显式启用） ==")
    saved = os.environ.pop("SMOKE_FAMILY", None)
    try:
        plan = FI.resolve_input_plan()
        check("★ 未设 SMOKE_FAMILY ⇒ mode=legacy（原写死前缀路径不变）", plan["mode"] == "legacy")
        os.environ["SMOKE_FAMILY"] = "F06"
        check("设了 SMOKE_FAMILY ⇒ mode=manifest",
              FI.resolve_input_plan()["mode"] == "manifest")
    finally:
        os.environ.pop("SMOKE_FAMILY", None)
        if saved is not None:
            os.environ["SMOKE_FAMILY"] = saved
    check("16 格清单定义稳定（顺序固定、标签唯一）",
          len(FI.family_cells()) == 16
          and len({c["label"] for c in FI.family_cells()}) == 16
          and FI.family_cells()[0]["label"] == "F06_base_ar_r1")

    print("== 2. manifest / hash 不匹配 ⇒ 拒绝（不降级） ==")
    man = FI.load_manifest()
    # 用真实 tokenizer（纯 CPU）跑，才能真的走到 hash 校验那一步。
    # 首版传 tok=None，结果在 tokenize 处先抛 TypeError ⇒ 测的不是 hash 校验。教训记在此。
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FI.TOKENIZER_ID, local_files_only=True)

    ids_ok, meta_ok = FI.load_family_ids(tok, "F06")
    check("正对照：未篡改的 manifest ⇒ hash_verified=True 且长度 8192",
          meta_ok["hash_verified"] is True and len(ids_ok) == 8192)

    fake = json.loads(json.dumps(man))
    for f in fake["families"]:
        if f["family_id"] == "F06":
            f["prefixes"]["8192"]["token_sha1"] = "deadbeefdeadbeef"
    try:
        FI.load_family_ids(tok, "F06", manifest=fake)
        rejected = False
    except FI.FamilyInputError:
        rejected = True
    check("★ manifest hash 被篡改 ⇒ 抛 FamilyInputError（拒绝运行）", rejected is True)

    ids_x, meta_x = FI.load_family_ids(tok, "F06", manifest=fake, verify_hash=False)
    check("★ verify_hash=False 时仍**如实**记录 hash_verified=False（不谎报通过）",
          meta_x["hash_verified"] is False and meta_x["token_sha1"] != meta_x["manifest_token_sha1"])

    try:
        FI.family_record("F99", man)
        unknown_rejected = False
    except FI.FamilyInputError:
        unknown_rejected = True
    check("★ 未知 family ⇒ FamilyInputError", unknown_rejected)

    bad = json.loads(json.dumps(man))
    for f in bad["families"]:
        if f["family_id"] == "F13":
            f["block"] = 99
    try:
        FI.load_family_ids(tok, "F13", manifest=bad)
        block_rejected = False
    except Exception:                                        # noqa: BLE001
        block_rejected = True
    check("★ 越界 block ⇒ 拒绝（抛异常而非静默取空）", block_rejected)

    check("源码/manifest 的 source 名不一致时按 basename 解析并**记录**差异",
          meta_ok["source_resolved_by_basename"] is True
          and meta_ok["source_declared"] != meta_ok["source"])

    print("== 3. 必需检查缺失 ⇒ 拒绝（复用契约 v2） ==")
    pf = prefill_finiteness(
        [dict(name="finiteness:prefill/target", required=True, executed=True, ok=True)],
        required_models=required_prefill_models("spec"), not_applicable=[])
    check("★ spec 缺 draft 记录 ⇒ all_required_ok=False", pf["all_required_ok"] is False)
    check("缺失被记为 missing_records 而非 N/A", pf["missing_records"] == ["draft"])

    print("== 4. AR/Spec 分歧 ⇒ record 模式下可继续 ==")
    a = _run(gen=[1, 2, 3, 4], mode="ar")
    s = _run(gen=[1, 2, 9, 4], mode="spec")
    r_stop = SF.ar_spec_equality(a, s, on_divergence="stop")
    r_rec = SF.ar_spec_equality(a, s, on_divergence="record")
    check("分歧被识别：common_prefix=2、首差 index=2、两侧 token 3/9",
          r_rec["status"] == "different" and r_rec["comparison"]["common_prefix_len"] == 2
          and r_rec["comparison"]["first_divergence_index"] == 2
          and r_rec["comparison"]["a_token"] == 3 and r_rec["comparison"]["b_token"] == 9)
    check("★ on_divergence=stop ⇒ stops_advance=True", r_stop["stops_advance"] is True)
    check("★ on_divergence=record ⇒ stops_advance=False（记录并继续）",
          r_rec["stops_advance"] is False)
    check("record 模式的说明里明确『不自动归因、不自动豁免』",
          "不自动归因" in r_rec["reason"] and "不自动豁免" in r_rec["reason"])

    print("== 5. 结构失败 ⇒ 停止 ==")
    st_fail = SF.structural_status(_run(round_inv_ok=False))
    check("★ round_invariants 失败 ⇒ structural=fail", st_fail["status"] == "fail")
    check("★ 结构失败 ⇒ execution=algorithm_failed 且停止推进",
          SF.execution_status(_run(round_inv_ok=False))["status"] == SF.EXEC_ALGORITHM_FAILED
          and SF.execution_status(_run(round_inv_ok=False))["stops_advance"] is True)
    st_cache = SF.structural_status(_run(cache_ok=False))
    check("★ 跨层 cache 结构失败 ⇒ structural=fail", st_cache["status"] == "fail")
    check("★ 未启用结构检查 ⇒ not_checked（不得当 ok）",
          SF.structural_status(_run(cache_ok=None, round_inv_ok=None))["status"] == "not_checked")

    print("== 6. 重复不一致 ⇒ 停止（标待查） ==")
    rep = SF.repeatability_status(_run(gen=[1, 2, 3]), _run(gen=[1, 2, 4]))
    check("★ 两次重复不同 ⇒ status=different 且 stops_advance=True",
          rep["status"] == "different" and rep["stops_advance"] is True)
    check("说明里标『待查』且不重试", "待查" in rep["reason"] and "不重试" in rep["reason"])
    rep_ok = SF.repeatability_status(_run(gen=[1, 2, 3]), _run(gen=[1, 2, 3]))
    check("对照：两次相同 ⇒ identical 且不停止",
          rep_ok["status"] == "identical" and rep_ok["stops_advance"] is False)
    rep_na = SF.repeatability_status(_run(), None)
    check("★ 缺一次重复 ⇒ not_assessable 且停止推进（不得当通过）",
          rep_na["status"] == "not_assessable" and rep_na["stops_advance"] is True)

    print("== 7. 异常短输出 ⇒ 停止 ==")
    short = _run(gen=[1, 2, 3], stop="other_short")
    check("★ stop_reason=other_short ⇒ algorithm_failed",
          SF.execution_status(short)["status"] == SF.EXEC_ALGORITHM_FAILED)
    check("★ 且 failure_class=length_or_stop", SF.failure_class(short) == "length_or_stop")
    check("EOS 正常停止不算失败",
          SF.execution_status(_run(gen=[1, 2, 151645], stop="eos"))["status"] == SF.EXEC_COMPLETE)

    print("== 8. 缺失运行不得生成“完整成功”汇总 ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cells = []
        for c in FI.family_cells():
            if c["label"] == "F13_vfps_spec_r2":
                cells.append(dict(c, json=None, rc=0))          # 故意缺一格
                continue
            rec = _run(gen=[1, 2, 3], kv=c["kv"], mode=c["mode"], family=c["family"],
                       cache_ok=True)
            p = td / f"{c['label']}.json"
            p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            cells.append(dict(c, json=str(p), rc=0, run_id=f"rid_{c['label']}"))
        out = SUM.summarize(dict(tag="selftest", cells=cells), on_divergence="record")
        check("★ 缺 1 格 ⇒ diagnostic_completeness != complete",
              out["diagnostic_completeness"] != "complete", out["diagnostic_completeness"])
        check("★ 且 complete_success_claimable=False", out["complete_success_claimable"] is False)
        check("缺的那格被显式列出",
              [m["cell"]["label"] for m in out["missing_cells"]] == ["F13_vfps_spec_r2"])
        check("汇总里没有名为 passed/success 的总体字段",
              not any(k in ("passed", "success", "overall_pass") for k in out))
        check("★ 正式判据只能是 not_established（诊断结构上不可能报 pass）",
              out["formal_gate_status"]["status"] == "not_established"
              and out["formal_gate_status"]["never_pass_here"] is True)
        check("no_formal_pass 标志存在", out["no_formal_pass"] is True)

        # 补齐后仍不得出现"正式正确性 PASS"
        cells2 = [c for c in cells]
        rec = _run(gen=[1, 2, 3], kv="v_fp8", mode="spec", family="F13", cache_ok=True)
        p = td / "F13_vfps_spec_r2.json"
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        cells2[-1] = dict(cells2[-1], json=str(p))
        out2 = SUM.summarize(dict(tag="selftest", cells=cells2), on_divergence="record")
        check("补齐 16 格 ⇒ completeness=complete_diagnostics_only",
              out2["diagnostic_completeness"] == "complete_diagnostics_only",
              out2["diagnostic_completeness"])
        check("★ 即便 16 格齐全，formal_gate_status 仍是 not_established",
              out2["formal_gate_status"]["status"] == "not_established")
        check("★ 齐全时 complete_success_claimable=True，但其含义被显式限定为“仅诊断”",
              out2["complete_success_claimable"] is True
              and "不表示任何正式正确性结论" in out2["complete_success_meaning"])

        # 重复不一致 + 结构失败 + 短输出都要在汇总里体现为停止
        cells3 = [dict(c) for c in cells2]
        for c in cells3:
            if c["label"] == "F06_base_ar_r2":
                rec = _run(gen=[1, 2, 4], stop="other_short", kv="base_bf16", mode="ar",
                           family="F06", cache_ok=False)
                p = td / "bad.json"
                p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                c["json"] = str(p)
        out3 = SUM.summarize(dict(tag="selftest", cells=cells3), on_divergence="record")
        check("★ 有短输出/结构失败 ⇒ completeness=complete_with_hard_failures",
              out3["diagnostic_completeness"] == "complete_with_hard_failures",
              out3["diagnostic_completeness"])
        check("★ 停止推进的格子被列出", len(out3["stops_advance_cells"]) >= 1)
        check("★ 结构失败格子被列出", "F06_base_ar_r2" in out3["structural_fail_cells"])
        check("★ 重复不一致被列出（F06/base_bf16/ar）",
              any(m["family"] == "F06" and m["kv"] == "base_bf16" and m["mode"] == "ar"
                  for m in out3["repeatability_mismatches"]))

    print("== 8b. 工具词汇表：真实终止 stage 必须被认作 complete ==")
    # 本轮 16 次实跑暴露的缺陷：smoke 工具的终止 stage 是 `run_ended`，不是 `complete`，
    # 首版只认 "complete" ⇒ 16 个**全部通过**的运行被误判 incomplete。
    for st in ("complete", "run_ended"):
        r = SF.execution_status(_run(stage=st))
        check(f"★ stage={st!r} + checks.passed ⇒ complete", r["status"] == SF.EXEC_COMPLETE, r)
    for st in ("run_ended_checks_failed", "aborted_non_finite", "aborted_weight_mismatch"):
        check(f"stage={st!r} ⇒ algorithm_failed 且停止",
              SF.execution_status(_run(stage=st, checks_passed=False))["status"]
              == SF.EXEC_ALGORITHM_FAILED)
    check("★ 非终止 stage（如 loaded）⇒ incomplete 且停止推进",
          SF.execution_status(_run(stage="loaded"))["status"] == SF.EXEC_INCOMPLETE
          and SF.execution_status(_run(stage="loaded"))["stops_advance"] is True)
    check("对照：stage=run_ended 时 failure_class=none",
          SF.failure_class(_run(stage="run_ended")) == "none")

    print("== 9. 环境拒绝与算法失败分开分类，但都停止推进 ==")
    envrec = dict(_run(), stage="refused", skipped="commit 不足")
    check("★ 环境拒绝 ⇒ env_refused（与算法失败不同类）",
          SF.execution_status(envrec)["status"] == SF.EXEC_ENV_REFUSED)
    check("★ 两者都 stops_advance=True",
          SF.execution_status(envrec)["stops_advance"] is True
          and SF.execution_status(short)["stops_advance"] is True)
    check("两类分类名不同且 failure_class 不同",
          SF.execution_status(envrec)["status"] != SF.execution_status(short)["status"]
          and SF.execution_status(envrec)["failure_class"] != SF.execution_status(short)["failure_class"])
    crashed = dict(_run(), stage="crashed", traceback="X")
    check("崩溃 ⇒ crashed 且停止", SF.execution_status(crashed)["status"] == SF.EXEC_CRASHED
          and SF.execution_status(crashed)["stops_advance"] is True)
    check("必需检查缺失 ⇒ algorithm_failed（不是 env）",
          SF.execution_status(_run(pf_all_required_ok=False))["status"] == SF.EXEC_ALGORITHM_FAILED)

    print("== 10. summarize_run 不产生“总体 PASS”字段 ==")
    s = SF.summarize_run(_run())
    check("五类状态齐全",
          all(k in s for k in ("execution_status", "structural_status", "repeatability_status",
                              "ar_spec_equality", "formal_gate_status")))
    check("★ 显式声明没有总体 PASS 字段", s["no_aggregate_pass_field"] is True)
    check("★ formal_gate_status 永不 pass", s["formal_gate_status"]["status"] == "not_established")

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_())
