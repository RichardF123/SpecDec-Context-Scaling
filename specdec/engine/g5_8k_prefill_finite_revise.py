"""g5_8k_prefill_finite_revise.py — 从**已有 JSON 记录**离线复核 `prefill_logits_finite`（纯 CPU）。

不需要 GPU、不重新加载模型、**绝不覆盖原 JSON**；产物是新文件 `prefill_finite_revision_v2_*.json`。

两轮缺陷（本脚本的判据基准 = 契约 v2）
  A（上一轮修的）：旧表达式 `all(reg.calls.get(("prefill", m), {}).get("ok", False)
     for m in ("target","draft"))` 把"draft 未驻留 ⇒ 该检查不存在"当成失败 ⇒ **AR 恒报 False**。
  B（本轮修的）：契约 v1 只对"执行过"的模型求合取 ⇒ **Spec 下 draft 必需但未执行时仍报 True**（漏检）；
     且 live 入口把"registry 里没有记录"的模型直接标成 not_applicable ⇒ 把**漏执行**伪装成**不适用**。

契约 v2：必需集合按**模式**显式给出（AR={target}，Spec={target,draft}，由
`smoke_checks.required_prefill_models` 从 `expected_map` 派生）；必需模型必须有记录、
`executed=True`、`ok is True`；只有**显式声明**不适用才可排除；放行判据是 `all_required_ok`
（`all_executed_ok` 仅保留字面含义，**不得**当判据）。

约定：`REVISE_TAGS` 逗号分隔（默认两个相关 tag）；`REVISE_ALL=1` 扫描 output/g5/attn 下全部含
`checks.checks` 的记录（历史全量离线复核）。
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

from .smoke_checks import (contract_note, prefill_finiteness, required_prefill_models)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
DEFAULT_TAGS = ("qk32c_20260928T025500Z", "reshistc_20260928T154500Z")
SKIP = ("analysis", "revision", "compare")


def _files() -> list:
    if os.environ.get("REVISE_ALL", "") == "1":
        cands = sorted(glob.glob(str(OUTDIR / "*.json")))
    else:
        tags = [t.strip() for t in os.environ.get("REVISE_TAGS", ",".join(DEFAULT_TAGS)).split(",")
                if t.strip()]
        cands = []
        for t in tags:
            cands += sorted(glob.glob(str(OUTDIR / f"{t}_*.json")))
    out = []
    for p in cands:
        if any(s in Path(p).name for s in SKIP):
            continue
        try:
            j = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not isinstance(j, dict):          # 目录里也存在 list 形态的产物，跳过
            continue
        if isinstance(j.get("checks"), dict) and j["checks"].get("checks"):
            out.append(Path(p))
    return out


def revise_one(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    chk = j.get("checks") or {}
    # ⚠️ mode 可能记在顶层，也可能记在 config 里（不同工具不同）。**取不到就不判定**：
    #    若默认成 "spec"，会把 AR 记录误判成"缺 draft 必需检查"——首版全量扫描就是这样产生了
    #    10 条假缺口。宁可标 out_of_scope，也不产生假阳性。
    mode = j.get("mode") or (j.get("config") or {}).get("mode")
    kv = j.get("kv") or (j.get("config") or {}).get("kv")
    old = (j.get("run") or {}).get("prefill_logits_finite")
    if mode not in ("ar", "spec"):
        return dict(file=path.name, run_id=j.get("run_id"), group=j.get("group"), kv=kv,
                    mode=mode, stage=j.get("stage"), in_scope=False,
                    old_prefill_logits_finite=old, new_prefill_logits_finite=None,
                    changed=False, checks_passed=chk.get("passed"),
                    reason="记录未标注 mode ⇒ 无法确定必需模型集合，**不**做重新判定（避免假阳性）")
    req = required_prefill_models(mode)
    na_explicit = [n for n in chk.get("not_applicable", [])
                   if n.startswith("finiteness:prefill/")]
    pf = prefill_finiteness(chk.get("checks", []), required_models=req,
                            not_applicable=na_explicit)
    passed = chk.get("passed")
    new = pf["all_required_ok"]          # 放行判据（all_executed_ok 会漏检"必需但未执行"）
    return dict(
        file=path.name, run_id=j.get("run_id"), group=j.get("group"), kv=kv, mode=mode,
        stage=j.get("stage"), in_scope=True,
        required_models=req, na_declared_prefill=na_explicit,
        old_prefill_logits_finite=old,
        old_field_absent=(old is None),
        new_prefill_logits_finite=new,
        new_all_executed_ok=pf["all_executed_ok"],
        changed=(old != new),
        missing_records=pf["missing_records"],
        not_executed_required=pf["not_executed_required"],
        bad_ok_required=pf["bad_ok_required"],
        conflicts=pf["conflicts"],
        checks_passed=passed,
        # ⚠️ 不能用 `new == passed` 判"一致"：`passed` 是**整体**门（聚合全部检查），
        #    而 `all_required_ok` 只管 prefill 有限性。prefill 有限但别的检查失败是完全正常的
        #    （stage=run_ended_checks_failed）。有意义的**不变式**只有一个方向：
        #    **整体通过 ⇒ prefill 必须通过**。
        gate_implication_ok=(not (passed is True and new is not True)),
        overall_failures=[f.get("name") for f in (chk.get("failures") or [])][:8],
        old_violated_implication=(bool(passed is True and old is not True) if old is not None
                                  else None),
        gate_field=pf["gate_field"],
        basis="契约 v2：必需模型(%s) 必须有记录 + executed=True + ok is True；NA 仅取显式声明"
              % ",".join(req),
    )


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    items = [revise_one(p) for p in _files()]
    scope = [i for i in items if i.get("in_scope")]
    skipped = [i for i in items if not i.get("in_scope")]
    changed = [i for i in scope if i["changed"]]
    contradictory = [i for i in scope if i["old_violated_implication"]]
    inconsistent = [i for i in scope if not i["gate_implication_ok"]]
    leak = [i for i in scope if i["not_executed_required"] or i["missing_records"]
            or i["bad_ok_required"] or i["conflicts"]]
    out = dict(
        tool="g5_8k_prefill_finite_revise", contract_version=2,
        run_id=run_id, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scope=("REVISE_ALL=1 全量扫描 output/g5/attn" if os.environ.get("REVISE_ALL", "") == "1"
               else "tags=" + os.environ.get("REVISE_TAGS", ",".join(DEFAULT_TAGS))),
        n_files=len(items), n_in_scope=len(scope), n_out_of_scope=len(skipped),
        n_changed=len(changed),
        n_old_violating_implication=len(contradictory),
        n_gate_implication_violations=len(inconsistent),
        n_records_with_required_gap=len(leak),
        invariant="整体通过(passed=True) ⇒ prefill 必需检查必须通过；反之不要求"
                  "（prefill 有限但别的检查失败是正常情况）",
        out_of_scope_reason="记录未标注 mode（顶层与 config 都没有）⇒ 不做重新判定，避免假阳性",
        n_old_field_absent=len([i for i in scope if i.get("old_field_absent")]),
        defects=[
            dict(id="A", round="上一轮", symptom="AR 恒报 prefill_logits_finite=False",
                 root_cause="把'必需检查不存在'当成失败（AR 不加载 draft）",
                 fix="必需集合按模式显式给出；未驻留记 not_applicable"),
            dict(id="B", round="本轮",
                 symptom="Spec 下必需模型未执行时汇总仍为 True（漏检）；live 把缺失记录标成 N/A",
                 root_cause="只对'执行过'的模型求合取 + 从缺失推断 N/A",
                 fix="必需模型必须 present+executed+ok is True；NA 只取显式声明；放行判据改 all_required_ok"),
        ],
        contract=contract_note(),
        gate_field="all_required_ok",
        items=items,
        note="原 JSON 一律未改动；本文件是**修订分析**（新文件），依据见每项的 basis 字段。"
             "`all_executed_ok` 仅保留字面含义，**不得**用作放行判据。",
    )
    p = OUTDIR / f"prefill_finite_revision_v2_{run_id}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"== 离线复核（契约 v2）候选 {len(items)} 份；**可判定** {len(scope)}；"
          f"因未标注 mode 而跳过 {len(skipped)}；变化 {len(changed)}；"
          f"旧值自相矛盾 {len(contradictory)}；修正后不一致 {len(inconsistent)}；"
          f"存在必需缺口 {len(leak)}")
    for i in scope:
        flag = "CHANGED" if i["changed"] else "same   "
        gap = ((f" missing={i['missing_records']}" if i["missing_records"] else "")
               + (f" not_exec={i['not_executed_required']}" if i["not_executed_required"] else "")
               + (f" bad_ok={i['bad_ok_required']}" if i["bad_ok_required"] else ""))
        print(f"  {flag} {str(i['group']):8s}/{i['kv']}/{i['mode']:5s} "
              f"old={i['old_prefill_logits_finite']} new={i['new_prefill_logits_finite']} "
              f"passed={i['checks_passed']} req={i['required_models']}{gap}  {i['file'][:52]}")
    if inconsistent:
        print(f"  !! {len(inconsistent)} 份修正后仍与 checks.passed 不一致，需排查")
    print(f"[revise] RESULT_JSON={p}")
    return 0 if not inconsistent else 5


if __name__ == "__main__":
    raise SystemExit(main())
