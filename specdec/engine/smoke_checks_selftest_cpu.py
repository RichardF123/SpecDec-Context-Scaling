"""smoke_checks_selftest_cpu.py — 8K smoke **检查逻辑**的轻量回归（**不加载任何模型**）。

覆盖任务要求的五种情形
  ① 正常通过：所有必需检查齐备且通过 ⇒ passed=True
  ② 非有限值：决策前检出 NaN/Inf ⇒ 抛 FinitenessError 且记录被保留
  ③ 分流失败：融合出现在 verify / 非 prefill ⇒ passed=False
  ④ cache 不变量失败 ⇒ passed=False
  ⑤ **缺失检查**：预期必需但未执行（含"空过"）⇒ passed=False

另测：`uncovered` 语义（未执行的可选分支既不算通过也不算失败，但必须被列出）；
     生成长度/停止原因（少生成不得当成功）；以及"执行结束 ≠ 验证通过"。

用法：python -m specdec.engine.smoke_checks_selftest_cpu
"""
from __future__ import annotations

import torch

from .smoke_checks import CheckRegistry, FinitenessError, evaluate

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _good_registry() -> CheckRegistry:
    """造一份"四阶段都跑过且全有限"的注册表（含 mismatch ⇒ repair 必需）。"""
    r = CheckRegistry()
    ok = torch.zeros(1, 4)
    r.check_finite(ok, phase="prefill", model="target", role="prefill_lg")
    r.check_finite(ok, phase="prefill", model="draft", role="prefill_lg")
    r.check_finite(ok, phase="propose", model="draft", role="input_lg")
    r.check_finite(ok, phase="propose", model="draft", role="return_lg")
    r.check_finite(ok, phase="verify", model="target", role="rows")
    r.check_finite(ok, phase="repair", model="target", role="saved_next")
    r.check_finite(ok, phase="repair", model="draft", role="saved_next")
    return r


def _ev(reg, **kw):
    base = dict(registry=reg, mismatch_rounds=1, weights_ok=True, weights_detail="338+290 ✓",
                invariants_ok=True, invariants_detail="5/5 ✓", dispatch_ok=True,
                dispatch_detail="fused 仅 prefill", attention_assert="ok",
                gen_len=16, max_new=16, eos_id=151645,
                gen=[1] * 16, stop_reason="max_new_reached", exception=None,
                alloc_ok=True, alloc_detail="peak_reserved 记入")
    base.update(kw)
    return evaluate(**base)


def main() -> int:
    print("== ① 正常通过 ==")
    v = _ev(_good_registry())
    check("① 全齐备 ⇒ passed=True", v["passed"] is True, f"failures={v['failures']}")
    check("① 必需检查数 ≥ 8", v["n_required"] >= 8, f"n_required={v['n_required']}")
    check("① 未覆盖列表为空", v["uncovered"] == [], str(v["uncovered"]))
    check("① 结构上不存在的组合被**显式列出**（不静默当作已覆盖）",
          sorted(v["not_applicable"]) == ["finiteness:propose/target", "finiteness:verify/draft"],
          str(v["not_applicable"]))

    print("\n== ② 非有限值：必须在决策前被拦住 ==")
    r = _good_registry()
    try:
        r.check_finite(torch.tensor([1.0, float("nan")]), phase="verify", model="target",
                       role="rows", where="第 3 轮")
        check("② NaN ⇒ 抛 FinitenessError", False, "没有抛")
    except FinitenessError as e:
        check("② NaN ⇒ 抛 FinitenessError", "在 argmax/接受判断**之前**检出" in str(e), str(e)[:70])
    check("② 失败记录被保留（first_bad 含 NaN 计数）",
          r.first_bad is not None and r.first_bad["n_nan"] == 1 and r.first_bad["phase"] == "verify",
          str(r.first_bad and {k: r.first_bad[k] for k in ("phase", "model", "n_nan", "n_inf")}))
    check("② 该相位被标记为不 ok", r.calls[("verify", "target")]["ok"] is False)
    r2 = CheckRegistry()
    try:
        r2.check_finite(torch.tensor([float("inf")]), phase="prefill", model="draft", role="prefill_lg")
        check("② Inf ⇒ 抛 FinitenessError", False)
    except FinitenessError as e:
        check("② Inf ⇒ 抛 FinitenessError", "Inf=1" in str(e))
    check("② 有限张量不抛且计数 +1",
          (lambda rr: (rr.check_finite(torch.zeros(2), phase="prefill", model="target",
                                       role="prefill_lg") is True and rr.n_checked == 1))(CheckRegistry()))

    print("\n== ③ 分流失败 ==")
    v = _ev(_good_registry(), dispatch_ok=False, dispatch_detail="fused 出现在 verify")
    check("③ 分流失败 ⇒ passed=False", v["passed"] is False)
    check("③ 失败项被点名", any(f["name"] == "dispatch_only_prefill" for f in v["failures"]),
          str([f["name"] for f in v["failures"]]))

    print("\n== ④ cache 不变量失败 ==")
    v = _ev(_good_registry(), invariants_ok=False, invariants_detail="第 2 轮 tgt_len != drf_len")
    check("④ 不变量失败 ⇒ passed=False", v["passed"] is False)
    check("④ 失败项被点名", any(f["name"] == "cache_invariants" for f in v["failures"]))

    print("\n== ⑤ 缺失检查（含「空过」）==")
    r = CheckRegistry()
    ok = torch.zeros(1, 4)
    r.check_finite(ok, phase="prefill", model="target", role="prefill_lg")
    r.check_finite(ok, phase="prefill", model="draft", role="prefill_lg")
    v = _ev(r)
    check("⑤ 缺 propose/verify ⇒ passed=False", v["passed"] is False)
    missing = [f["name"] for f in v["failures"]]
    check("⑤ 缺失项被逐个点名",
          any("propose" in m for m in missing) and any("verify" in m for m in missing), str(missing))
    check("⑤ 已执行的 prefill 仍记为 ok（不因缺失而整体作废）",
          all(c["ok"] is True for c in v["checks"]
              if c["executed"] and c["name"].startswith("finiteness:prefill")))

    print("\n== 未覆盖语义：全接受 run 没有 repair ==")
    r = _good_registry()
    r.calls.pop(("repair", "target"), None)
    r.calls.pop(("repair", "draft"), None)
    v = _ev(r, mismatch_rounds=0, invariants_detail="全接受")
    check("未执行的可选分支 ⇒ 不算失败", v["passed"] is True, f"failures={v['failures']}")
    check("但必须列入 uncovered（不能空过）",
          any("repair" in u for u in v["uncovered"]), str(v["uncovered"]))
    v2 = _ev(r, mismatch_rounds=1)
    check("同一份注册表在 mismatch>0 时 ⇒ repair 转为**必需**并判失败",
          v2["passed"] is False and any("repair" in f["name"] for f in v2["failures"]))

    print("\n== 生成长度 / 停止原因 ==")
    check("少生成但声称 max_new ⇒ 失败",
          _ev(_good_registry(), gen_len=7, gen=[1] * 7,
              stop_reason="max_new_reached")["passed"] is False)
    check("EOS 提前停止且最后一个是 eos ⇒ 通过",
          _ev(_good_registry(), gen_len=5, gen=[1, 2, 3, 4, 151645],
              stop_reason="eos")["passed"] is True)
    check("EOS 停止但末位不是 eos ⇒ 失败",
          _ev(_good_registry(), gen_len=5, gen=[1, 2, 3, 4, 5],
              stop_reason="eos")["passed"] is False)
    check("无理由的短生成(other_short) ⇒ 失败",
          _ev(_good_registry(), gen_len=3, gen=[1, 2, 3],
              stop_reason="other_short")["passed"] is False)

    print("\n== 其余必需项 ==")
    check("权重校验失败 ⇒ passed=False",
          _ev(_good_registry(), weights_ok=False, weights_detail="mismatch")["passed"] is False)
    check("attention_assert 非 ok ⇒ passed=False",
          _ev(_good_registry(), attention_assert="RuntimeError: 策略已切但 zero fused")["passed"] is False)
    check("运行期异常 ⇒ passed=False",
          _ev(_good_registry(), exception="RuntimeError: boom")["passed"] is False)
    check("内存字段缺失 ⇒ passed=False",
          _ev(_good_registry(), alloc_ok=False, alloc_detail="未记录 peak")["passed"] is False)

    print(f"\nCPU 自测：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
