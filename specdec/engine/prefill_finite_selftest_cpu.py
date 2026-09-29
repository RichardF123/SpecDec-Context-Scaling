"""prefill_finite_selftest_cpu.py — `prefill_finiteness` 契约 v2 的 CPU 回归自测（不加载模型）。

被修正的两处缺陷（本文件逐个钉死）
  * **漏检**：契约 v1 只对"执行过"的模型求合取 ⇒ Spec 下 `draft` 必需但未执行时，
    `all_executed_ok` 仍返回 **True**（制造了一个不会被抓住的通过）。
  * **把漏执行伪装成不适用**：`prefill_finiteness_live` v1 把 registry 里**没有记录**的模型
    直接写进 `not_applicable` ⇒ "必需检查漏执行"被降级成"模型不适用"。

正确合同（v2）
  1) 必需集合按**模式**显式给出：AR={target}，Spec={target,draft}（由 `required_prefill_models` 派生）。
  2) 必需模型必须有记录、`executed=True`、`ok is True`，否则失败。
  3) 只有**显式声明**不适用才可排除；**不**从缺失记录推断 N/A。
  4) 没有任何有效检查（必需集合为空）⇒ 不通过。
  5) `all_executed_ok` 保留字面含义，**不是**放行判据；放行一律读 `all_required_ok`。
"""
from __future__ import annotations

import torch

from .smoke_checks import (CheckRegistry, na_prefill_models, prefill_finiteness,
                           prefill_finiteness_live, required_prefill_models)

# ---- 真实历史记录的形状（照抄 output/g5/attn/qk32c_*.json，字段一致）----
AR_CHECKS = [dict(name="finiteness:prefill/target", required=True, executed=True, ok=True,
                  detail="AR 模式无条件必需; 调用 1 次, 角色=['prefill_lg']")]
SPEC_CHECKS = [
    dict(name="finiteness:prefill/draft", required=True, executed=True, ok=True,
         detail="无条件必需（该分支必然执行）; 调用 1 次"),
    dict(name="finiteness:prefill/target", required=True, executed=True, ok=True,
         detail="无条件必需（该分支必然执行）; 调用 1 次"),
]


def _run() -> int:
    npass = nfail = 0

    def check(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    def pf(checks, mode, na=None):
        return prefill_finiteness(checks, required_models=required_prefill_models(mode),
                                  not_applicable=(na if na is not None
                                                  else [f"finiteness:prefill/{m}"
                                                        for m in na_prefill_models(mode)]))

    print("== 0. 必需集合由 expected_map 派生（与 evaluate 同一份判据） ==")
    check("AR 必需 prefill 模型 = ['target']", required_prefill_models("ar") == ["target"],
          required_prefill_models("ar"))
    check("Spec 必需 prefill 模型 = ['draft','target']",
          required_prefill_models("spec") == ["draft", "target"], required_prefill_models("spec"))
    check("AR 明确不适用 = ['draft']", na_prefill_models("ar") == ["draft"],
          na_prefill_models("ar"))
    check("Spec 没有不适用的 prefill 模型", na_prefill_models("spec") == [])

    print("== 1. AR target 正常、draft 明确不适用 ⇒ 通过 ==")
    r = pf(AR_CHECKS, "ar")
    check("★ AR：all_required_ok=True", r["all_required_ok"] is True, r["missing_records"])
    check("AR：draft 只在 excluded_models（显式声明），不在必需里",
          r["excluded_models"] == ["draft"] and "draft" not in r["required_models"])
    check("AR：无缺失记录、无冲突", r["missing_records"] == [] and r["conflicts"] == [])

    print("== 2. Spec 两模型正常 ⇒ 通过 ==")
    s = pf(SPEC_CHECKS, "spec")
    check("★ Spec：all_required_ok=True", s["all_required_ok"] is True)
    check("Spec：executed_models = ['draft','target']",
          s["executed_models"] == ["draft", "target"], s["executed_models"])
    check("Spec：excluded_models 为空", s["excluded_models"] == [])

    print("== 3. ★ Spec draft 必需但未执行 ⇒ 必须失败（v1 漏检） ==")
    c3 = [SPEC_CHECKS[0] | dict(executed=False, ok=False), SPEC_CHECKS[1]]
    r3 = pf(c3, "spec")
    check("★ all_required_ok=False（v1 会返回 True）", r3["all_required_ok"] is False)
    check("not_executed_required 标出 draft", r3["not_executed_required"] == ["draft"],
          r3["not_executed_required"])
    check("字面字段 all_executed_ok 仍为 True（明文标注不得当判据）",
          r3["all_executed_ok"] is True and "不得" in r3["all_executed_ok_warning"])
    check("gate_field 指向 all_required_ok", r3["gate_field"] == "all_required_ok")

    print("== 4. ★ 必需模型整条记录缺失 ⇒ 必须失败 ==")
    r4 = pf([SPEC_CHECKS[1]], "spec", na=[])
    check("★ all_required_ok=False", r4["all_required_ok"] is False)
    check("missing_records 标出 draft", r4["missing_records"] == ["draft"], r4["missing_records"])
    check("★ 缺失**不**被记成 excluded/NA", "draft" not in r4["excluded_models"])
    r4b = pf([], "spec", na=[])
    check("两个必需模型都缺失 ⇒ 仍失败", r4b["all_required_ok"] is False
          and r4b["missing_records"] == ["draft", "target"])

    print("== 5. 执行过但 ok=False 或 None ⇒ 必须失败 ==")
    r5a = pf([AR_CHECKS[0] | dict(ok=False)], "ar")
    check("★ ok=False ⇒ all_required_ok=False", r5a["all_required_ok"] is False)
    check("bad_ok_required 标出 target", r5a["bad_ok_required"] == ["target"], r5a["bad_ok_required"])
    r5b = pf([AR_CHECKS[0] | dict(ok=None)], "ar")
    check("★ ok=None ⇒ all_required_ok=False（None 不等于通过）", r5b["all_required_ok"] is False)
    r5c = pf([AR_CHECKS[0] | dict(ok=True)], "ar")
    check("对照：ok=True ⇒ 通过", r5c["all_required_ok"] is True)

    print("== 6. 空记录 ⇒ 不能通过 ==")
    r6 = pf([], "ar")
    check("★ 空记录 all_required_ok=False", r6["all_required_ok"] is False)
    check("空记录 missing_records=['target']", r6["missing_records"] == ["target"])
    r6b = pf([], "ar", na=[])
    check("★ 连 N/A 声明都没有也照样失败（不能凭没跑当通过）",
          r6b["all_required_ok"] is False)

    print("== 7. required 与 N/A 冲突 ⇒ 必须失败 ==")
    r7 = pf(SPEC_CHECKS, "spec", na=["finiteness:prefill/draft"])
    check("★ draft 既必需又被列为 N/A ⇒ all_required_ok=False", r7["all_required_ok"] is False)
    check("conflicts 记录 required_but_listed_not_applicable",
          any(c["model"] == "draft" and c["kind"] == "required_but_listed_not_applicable"
              for c in r7["conflicts"]), r7["conflicts"])
    r7b = pf([AR_CHECKS[0] | dict(required=False)], "ar")
    check("记录把必需模型标成 required=False 也算冲突 ⇒ 失败",
          r7b["all_required_ok"] is False
          and any(c["kind"] == "required_model_recorded_as_not_required"
                  for c in r7b["conflicts"]))

    print("== 8. ★ live 入口：必需记录缺失不得转成 N/A（v1 缺陷） ==")
    reg = CheckRegistry()
    reg.check_finite(torch.ones(2), phase="prefill", model="target", role="prefill_lg")
    lv = prefill_finiteness_live(reg, "spec")
    check("★ live(spec) 缺 draft 记录 ⇒ all_required_ok=False", lv["all_required_ok"] is False)
    check("★ 且 draft 出现在 missing_records，**不**在 excluded/NA",
          lv["missing_records"] == ["draft"] and "draft" not in lv["excluded_models"])
    lv_ar = prefill_finiteness_live(reg, "ar")
    check("live(ar) 只需 target ⇒ 通过，且 draft 由**显式声明**排除",
          lv_ar["all_required_ok"] is True and lv_ar["excluded_models"] == ["draft"])
    check("live 入口标出 entry/mode", lv_ar.get("entry") == "live" and lv_ar.get("mode") == "ar")

    print("== 9. 修复后，正常历史样本的汇总不变 ==")
    r9a = pf(AR_CHECKS, "ar", na=["finiteness:prefill/draft"])
    check("历史 AR 样本：all_required_ok=True（与修订分析一致）",
          r9a["all_required_ok"] is True)
    check("历史 AR 样本：all_executed_ok=True（字面值也未被改变）",
          r9a["all_executed_ok"] is True)
    r9b = pf(SPEC_CHECKS, "spec", na=[])
    check("历史 Spec 样本：all_required_ok=True 且 all_executed_ok=True",
          r9b["all_required_ok"] is True and r9b["all_executed_ok"] is True)

    print("== 10. 不相关检查名不得被误计入 prefill ==")
    noise = AR_CHECKS + [dict(name="finiteness:verify/target", required=True, executed=True,
                              ok=False, detail="故意造的非 prefill 失败")]
    nn = pf(noise, "ar")
    check("verify 的失败不影响 prefill 汇总", nn["all_required_ok"] is True)
    check("汇总只含 prefill 的必需模型", nn["required_models"] == ["target"])

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
