"""g5_8k_prefill_finite_revise.py — 从**已有 JSON 记录**重算 `prefill_logits_finite`（纯 CPU）。

背景（本轮修复的缺陷）
  旧汇总 `all(reg.calls.get(("prefill", m), {}).get("ok", False) for m in ("target","draft"))`
  把"draft 未驻留 ⇒ 该检查根本不存在"当成**失败**，于是**所有 AR 运行恒报
  `prefill_logits_finite=False`**，而同一份记录里 `checks.passed=True`、`uncovered=[]`、
  `failures=[]`，且 `finiteness:prefill/draft` 明确列在 `not_applicable` —— 自相矛盾。

本脚本
  * **只读**原 JSON（绝不覆盖）；
  * 用 `smoke_checks.prefill_finiteness()`（与修复后的工具同一份逻辑）重算；
  * 输出修订分析 JSON，逐 run 给出 `old → new`、依据（哪些模型执行过、哪些是 not_applicable）
    以及是否与 `checks.passed` 一致。

用法：
  python -m specdec.engine.g5_8k_prefill_finite_revise            # 默认 tag=qk32c_20260928T025500Z
  REVISE_TAG=... python -m specdec.engine.g5_8k_prefill_finite_revise
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

from .smoke_checks import prefill_finiteness

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = os.environ.get("REVISE_TAG", "qk32c_20260928T025500Z")


def revise_one(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    checks = (j.get("checks") or {}).get("checks", [])
    na = (j.get("checks") or {}).get("not_applicable", [])
    pf = prefill_finiteness(checks, na)
    old = (j.get("run") or {}).get("prefill_logits_finite")
    passed = (j.get("checks") or {}).get("passed")
    new = pf["all_executed_ok"]
    return dict(
        file=path.name, run_id=j.get("run_id"), group=j.get("group"), kv=j.get("kv"),
        mode=j.get("mode"), stage=j.get("stage"),
        resident_models=j.get("resident_models"),
        old_prefill_logits_finite=old, new_prefill_logits_finite=new,
        changed=(old != new),
        checks_passed=passed,
        new_consistent_with_checks_passed=(bool(new) == bool(passed)),
        old_was_contradictory=(bool(old) != bool(passed)) if old is not None else None,
        executed_models=pf["executed_models"],
        not_applicable_models=pf["not_applicable_models"],
        by_model=pf["by_model"],
        semantics=pf["semantics"],
        reason=("draft 在 AR 模式下未驻留（not_applicable），旧汇总把它当成失败"
                if new != old else "无变化"),
    )


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    files = sorted(p for p in glob.glob(str(OUTDIR / f"{TAG}_*.json"))
                   if "analysis" not in p and "revision" not in p)
    items = [revise_one(Path(p)) for p in files]
    changed = [i for i in items if i["changed"]]
    contradictory = [i for i in items if i["old_was_contradictory"]]
    inconsistent_new = [i for i in items if not i["new_consistent_with_checks_passed"]]
    out = dict(
        tool="g5_8k_prefill_finite_revise", run_id=run_id, tag=TAG,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_files=len(items),
        n_changed=len(changed),
        n_old_contradictory=len(contradictory),
        n_new_inconsistent=len(inconsistent_new),
        defect=dict(
            name="prefill_logits_finite AR 误报 True→False",
            old_expression='all(reg.calls.get(("prefill", m), {}).get("ok", False) '
                           'for m in ("target","draft"))',
            why_wrong="AR 不加载 draft ⇒ ('prefill','draft') 不存在 ⇒ 取到 False "
                      "⇒ 恒报 False；且与同一记录的 checks.passed=True 矛盾",
            fix="只对**实际驻留并执行过** prefill 的模型求合取；未驻留记 not_applicable",
            fix_location="smoke_checks.prefill_finiteness / prefill_finiteness_live",
            originals_unchanged=True,
        ),
        items=items,
        note="原 JSON 一律未改动；本文件是**修订分析**，供引用。"
             "未驻留（not_applicable）**不是**失败。",
    )
    p = OUTDIR / f"{TAG}_prefill_finite_revision_{run_id}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"== 修订分析 tag={TAG} ==  文件 {len(items)} 个，变化 {len(changed)} 个，"
          f"旧值自相矛盾 {len(contradictory)} 个")
    for i in items:
        flag = "CHANGED" if i["changed"] else "same   "
        print(f"  {flag} {i['group']}/{i['kv']}/{i['mode']:5s} "
              f"old={i['old_prefill_logits_finite']} new={i['new_prefill_logits_finite']} "
              f"passed={i['checks_passed']} executed={i['executed_models']} "
              f"na={i['not_applicable_models']}")
    if inconsistent_new:
        print(f"  !! 仍有 {len(inconsistent_new)} 个与 checks.passed 不一致，需排查")
    print(f"[revise] RESULT_JSON={p}")
    return 0 if not inconsistent_new else 5


if __name__ == "__main__":
    raise SystemExit(main())
