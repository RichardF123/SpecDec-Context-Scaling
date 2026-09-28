"""prefill_finite_selftest_cpu.py — `prefill_finiteness()` 的回归自测（纯 CPU，不加载模型）。

回归对象：**AR 运行的 `prefill_logits_finite` 恒为 False**。
旧汇总把"draft 未驻留 ⇒ 该检查根本不存在"当成"检查失败"：
    `all(reg.calls.get(("prefill", m), {}).get("ok", False) for m in ("target","draft"))`
而同一份记录里 `checks.passed=True`、`not_applicable` 已明确列出 `finiteness:prefill/draft`。
下面的用例直接照抄真实 JSON 的形状（tag `qk32c_20260928T025500Z` 的 AR / spec 两类记录）。
"""
from __future__ import annotations

from .smoke_checks import prefill_finiteness

# ---- 真实记录的形状（从 output/g5/attn/qk32c_*.json 抄来，字段一致）----
AR_CHECKS = [dict(name="finiteness:prefill/target", required=True, executed=True, ok=True,
                  detail="AR 模式无条件必需; 调用 1 次, 角色=['prefill_lg']")]
AR_NA = ["finiteness:prefill/draft", "finiteness:propose/draft", "finiteness:propose/target",
         "finiteness:repair/draft", "finiteness:repair/target", "finiteness:verify/draft",
         "finiteness:verify/target"]

SPEC_CHECKS = [
    dict(name="finiteness:prefill/draft", required=True, executed=True, ok=True,
         detail="无条件必需（该分支必然执行）; 调用 1 次, 角色=['prefill_lg']"),
    dict(name="finiteness:prefill/target", required=True, executed=True, ok=True,
         detail="无条件必需（该分支必然执行）; 调用 1 次, 角色=['prefill_lg']"),
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

    print("== 1. AR 记录（draft 未驻留）⇒ 修正后应为 True ==")
    r = prefill_finiteness(AR_CHECKS, AR_NA)
    check("★ AR：all_executed_ok=True（旧口径错报 False）", r["all_executed_ok"] is True)
    check("AR：只有 target 被算作执行过", r["executed_models"] == ["target"], r["executed_models"])
    check("★ AR：draft 被记入 not_applicable_models（不是失败）",
          r["not_applicable_models"] == ["draft"], r["not_applicable_models"])
    check("AR：by_model['draft'].executed=False 且 ok=None",
          r["by_model"]["draft"]["executed"] is False and r["by_model"]["draft"]["ok"] is None)
    check("AR：语义写明未驻留不计失败", "不计入失败" in r["by_model"]["draft"]["detail"])

    print("== 2. Spec 记录（两模型都驻留）⇒ True，且两个都算执行过 ==")
    s = prefill_finiteness(SPEC_CHECKS, [])
    check("Spec：all_executed_ok=True", s["all_executed_ok"] is True)
    check("Spec：executed_models = [draft, target]",
          s["executed_models"] == ["draft", "target"], s["executed_models"])
    check("Spec：没有 not_applicable 模型", s["not_applicable_models"] == [])

    print("== 3. 真失败仍然必须被抓住（修正不能变成放宽） ==")
    bad = [dict(AR_CHECKS[0], ok=False)]
    check("★ target prefill 非有限 ⇒ all_executed_ok=False",
          prefill_finiteness(bad, AR_NA)["all_executed_ok"] is False)
    badmixed = [dict(SPEC_CHECKS[0]), dict(SPEC_CHECKS[1], ok=False)]
    check("★ Spec 中任一模型非有限 ⇒ False",
          prefill_finiteness(badmixed, [])["all_executed_ok"] is False)

    print("== 4. 边界：一个都没执行 ⇒ False（不能凭「没跑」当通过） ==")
    empty = prefill_finiteness([], AR_NA)
    check("★ 没有执行过任何 prefill ⇒ all_executed_ok=False",
          empty["all_executed_ok"] is False)
    check("空记录时 executed_models=[]", empty["executed_models"] == [])

    print("== 5. 与 checks.passed 不再自相矛盾 ==")
    check("AR：修正值 True 与 checks.passed=True 一致（旧值 False 与之矛盾）",
          prefill_finiteness(AR_CHECKS, AR_NA)["all_executed_ok"] is True)

    print("== 6. 不相关检查名不得被误计入 prefill ==")
    noise = AR_CHECKS + [dict(name="finiteness:verify/target", required=True, executed=True,
                              ok=False, detail="故意造的非 prefill 失败")]
    nn = prefill_finiteness(noise, AR_NA)
    check("verify 的失败不影响 prefill 汇总", nn["all_executed_ok"] is True)
    check("汇总里只有 prefill/* 的模型", sorted(nn["by_model"]) == ["draft", "target"],
          sorted(nn["by_model"]))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
