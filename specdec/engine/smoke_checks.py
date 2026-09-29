"""smoke_checks.py — 8K smoke 的**检查注册表与通过判定**（纯函数；可在 CPU 上做轻量回归）。

为什么单独成模块
  1. 任务要求"检查必须在 argmax/接受判断**之前**发生；发现 NaN/Inf **立即**记录并停止"；
  2. 任务要求"区分**执行结束**与**验证通过**"——检查失败必须落盘并**非零退出**；
  3. 这些规则**必须能在不加载大模型的前提下被测试**（轻量回归要覆盖
     正常通过 / 非有限值 / 分流失败 / cache 不变量失败 / 缺失检查 五种情形）。
  ⇒ 把"检查什么、怎么判定"抽成纯逻辑，工具只负责在正确时机调用。

语义约定（**关键**）
  * `executed=False` 表示**该分支本次没有执行**（例如全接受的 run 不会有 repair）⇒ 记为**未覆盖**，
    **不算通过也不算失败**，但必须出现在报告的 `uncovered` 里（"不能空过"）。
  * `required=True` 且 `executed=False` ⇒ **失败**（"预期检查缺失不得判通过"）。
  * 非有限值 ⇒ 抛 `FinitenessError`，调用方必须落盘并非零退出，**不得**继续 argmax/接受判断。
"""
from __future__ import annotations

import torch

PHASES = ("prefill", "propose", "verify", "repair")
# 引擎里各阶段实际会跑的 (phase, model) 组合（由 spec_dec 的调用结构决定）
ALWAYS_REQUIRED = (
    ("prefill", "target"), ("prefill", "draft"),
    ("propose", "draft"), ("verify", "target"),
)
REQUIRED_IF_MISMATCH = (("repair", "target"), ("repair", "draft"))
NEVER_USED = (("propose", "target"), ("verify", "draft"))   # 结构上不会发生；
#                                                            出现即为异常，另行报告


class FinitenessError(RuntimeError):
    """检测到 NaN/Inf。**必须**立即停止：不得进入 argmax / 接受判断。"""


class CheckRegistry:
    """记录每次有限性检查的 (阶段, 模型, 角色)，并给出覆盖情况。

    **轮次/步号记账**：`round` 由 verify（每个解码轮调用一次）推进；
    `propose_step` 每次进入 `_propose` 归零、每次 propose 阶段的 `_run` 自增。
    这样"中间步检查"能带上 (round, 候选步编号)，而不只是"这个阶段出现过检查"。
    """

    def __init__(self) -> None:
        self.calls: dict[tuple[str, str], dict] = {}
        self.events: list[dict] = []
        self.n_checked = 0
        self.first_bad: dict | None = None
        self.round = -1
        self.propose_step = 0

    def begin_round(self) -> int:
        self.round += 1
        return self.round

    def begin_propose(self) -> None:
        self.propose_step = 0

    def next_propose_step(self) -> int:
        self.propose_step += 1
        return self.propose_step

    def _slot(self, phase: str, model: str) -> dict:
        return self.calls.setdefault((phase, model), dict(n=0, roles=[], role_counts={},
                                                          ok=True, bad=[], steps=[]))

    def check_finite(self, t, *, phase: str, model: str, role: str, where: str = "",
                     round_: int | None = None, step: int | None = None) -> bool:
        """在**决策之前**检查。返回是否有限；非有限时抛 `FinitenessError`（先落记录再抛）。"""
        if t is None:
            self.events.append(dict(phase=phase, model=model, role=role, where=where,
                                    checked=False, round=round_, step=step,
                                    note="张量为 None（该分支未产生决策用 logits）"))
            return False
        finite = bool(torch.isfinite(t).all())
        slot = self._slot(phase, model)
        slot["n"] += 1
        slot["role_counts"][role] = slot["role_counts"].get(role, 0) + 1
        if role not in slot["roles"]:
            slot["roles"].append(role)
        if step is not None:
            slot["steps"].append(dict(round=round_, step=step, role=role))
        self.n_checked += 1
        rec = dict(phase=phase, model=model, role=role, where=where, checked=True,
                   finite=finite, round=round_, step=step,
                   shape=[int(x) for x in t.shape], dtype=str(t.dtype))
        self.events.append(rec)
        if not finite:
            slot["ok"] = False
            slot["bad"].append(rec)
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            self.first_bad = dict(rec, n_nan=n_nan, n_inf=n_inf)
            raise FinitenessError(
                f"非有限值：phase={phase} model={model} role={role} {where} "
                f"round={round_} step={step} "
                f"shape={list(t.shape)} NaN={n_nan} Inf={n_inf}"
                f"（在 argmax/接受判断**之前**检出，已停止）")
        return True

    def coverage(self) -> dict:
        return {"n_checked": self.n_checked,
                "by_phase_model": {f"{p}/{m}": dict(v) for (p, m), v in sorted(self.calls.items())},
                "events": self.events,
                "first_bad": self.first_bad}


def verify_decision_coverage(registry: CheckRegistry, *, draft_calls_expected: int,
                             rounds_expected: int) -> dict:
    """
ounds_expected = **进入 _propose 的次数**（真实 SpecDec 里等于解码轮次）。"""
    """**核对"实际决策步数"与"检查覆盖"**（任务要求：阶段出现过一次检查 ≠ 完整覆盖）。

    论证（对应 `spec_dec._propose` 的真实结构）
      * 循环内第 j 次 `draft._run` 产出的 logits，会在第 j+1 次迭代被 `argmax` 使用
        ⇒ 必须在它**返回时**检查（role=`mid_lg`，逐 (round, 步号) 记录）。
      * 每轮进入 `_propose` 时传入的 lg 会被**第一次** argmax 使用（role=`input_lg`）。
      ⇒ 因此本轮要求：`mid_lg 次数 == 解码器报告的 draft 前向次数`，
        且 `input_lg 次数 == 轮次`。任一不符即判**未完整覆盖**。
    """
    slot = registry.calls.get(("propose", "draft"), {})
    rc = slot.get("role_counts", {})
    mid = rc.get("mid_lg", 0)
    entry = rc.get("input_lg", 0)
    ok = (mid == draft_calls_expected) and (rounds_expected < 0 or entry == rounds_expected)
    return dict(ok=bool(ok), mid_step_checks=mid, entry_checks=entry,
                draft_calls_reported=draft_calls_expected, rounds_reported=rounds_expected,
                detail=f"mid_lg 检查={mid} vs 解码器 draft 前向={draft_calls_expected}；"
                       f"input_lg 检查={entry} vs 轮次={rounds_expected}",
                steps=slot.get("steps", [])[:64])


AR_REQUIRED = (("prefill", "target"), ("decode", "target"))
AR_NOT_APPLICABLE = (("prefill", "draft"), ("propose", "draft"),
                     ("verify", "target"), ("repair", "target"), ("repair", "draft"))


def expected_map(*, mismatch_rounds: int, mode: str = "spec") -> dict[tuple[str, str], dict]:
    """本次 run **应当**有哪些检查（由执行结构决定，不由结果决定）。

    mode="ar"（target-only 自回归）不使用 draft/propose/verify/repair ⇒ 那些分支记为
    **不适用**（显式列出），而不是要求它们存在或静默当作通过。
    """
    if mode == "ar":
        out: dict[tuple[str, str], dict] = {}
        for key in AR_REQUIRED:
            out[key] = dict(required=True, why="AR 模式无条件必需")
        for key in AR_NOT_APPLICABLE:
            out[key] = dict(required=False, na=True, why="AR 模式不使用该分支")
        for key in NEVER_USED:
            out[key] = dict(required=False, na=True, why="按结构不应出现")
        return out
    out: dict[tuple[str, str], dict] = {}
    for key in ALWAYS_REQUIRED:
        out[key] = dict(required=True, why="无条件必需（该分支必然执行）")
    for key in REQUIRED_IF_MISMATCH:
        out[key] = dict(required=mismatch_rounds > 0,
                        why="存在 mismatch 轮 ⇒ repair 必然执行" if mismatch_rounds > 0
                            else "本次无 mismatch 轮 ⇒ repair 未执行（记为未覆盖）")
    for key in NEVER_USED:
        out[key] = dict(required=False, why="按 spec_dec 的调用结构，该组合不应出现")
    return out


def evaluate(*, registry: CheckRegistry, mismatch_rounds: int, weights_ok: bool,
             weights_detail: str, invariants_ok: bool, invariants_detail: str,
             dispatch_ok: bool, dispatch_detail: str, attention_assert: str,
             gen_len: int, max_new: int, eos_id, gen: list, stop_reason: str,
             exception: str | None, alloc_ok: bool = True,
             alloc_detail: str = "", extra_required: list | None = None,
             mode: str = "spec") -> dict:
    """把"执行结束"升级为"验证通过/不通过"。**只有 passed=True 才允许 exit 0。**"""
    exp = expected_map(mismatch_rounds=mismatch_rounds, mode=mode)
    checks: list[dict] = []
    failures: list[dict] = []
    uncovered: list[str] = []
    not_applicable: list[str] = []

    # ---- 1) 有限性检查的**覆盖与结果** ----
    for (phase, model), meta in sorted(exp.items()):
        name = f"finiteness:{phase}/{model}"
        if (phase, model) in NEVER_USED or meta.get("na"):
            # 结构上不存在的组合：**不算未覆盖**（它不是"该跑没跑"，而是"根本不会跑"），
            # 但必须**显式列出**以免被误读成"已覆盖且通过"。
            not_applicable.append(name)
            if (phase, model) in registry.calls:
                checks.append(dict(name=name, required=False, executed=True,
                                   ok=bool(registry.calls[(phase, model)]["ok"]),
                                   detail="**异常**：该组合按 spec_dec 的调用结构不应出现，请排查"))
            continue
        slot = registry.calls.get((phase, model))
        executed = slot is not None
        ok = None
        detail = meta["why"]
        if executed:
            ok = bool(slot["ok"])
            detail = f"{detail}; 调用 {slot['n']} 次, 角色={slot['roles']}"
        if not executed:
            if meta["required"]:
                ok = False
                failures.append(dict(name=name, detail=f"预期检查缺失：{meta['why']}"))
            else:
                uncovered.append(name)
        checks.append(dict(name=name, required=bool(meta["required"]), executed=executed,
                           ok=ok, detail=detail))

    # ---- 2) 其余必需检查 ----
    def req(name: str, ok: bool, detail: str) -> None:
        checks.append(dict(name=name, required=True, executed=True, ok=bool(ok), detail=detail))
        if not ok:
            failures.append(dict(name=name, detail=detail))

    req("run_no_exception", exception is None, f"exception={exception}")
    req("weights_bitmatch", bool(weights_ok), weights_detail)
    req("cache_invariants", bool(invariants_ok), invariants_detail)
    req("dispatch_only_prefill", bool(dispatch_ok), dispatch_detail)
    req("attention_assert", attention_assert == "ok", f"attention_assert={attention_assert!r}")
    req("alloc_recorded", bool(alloc_ok), alloc_detail)

    # ---- 3) 生成长度与停止原因（**不能少生成却当固定长度完成**）----
    len_ok = (stop_reason == "max_new_reached" and gen_len == max_new) or \
             (stop_reason == "eos" and 0 < gen_len <= max_new
              and eos_id is not None and gen and gen[-1] == eos_id)
    # ---- 2b) 调用方追加的必需检查（如决策步覆盖核对）----
    for item in (extra_required or []):
        req(item["name"], bool(item["ok"]), item.get("detail", ""))

    req("generation_length_and_stop_reason", len_ok,
        f"gen_len={gen_len} max_new={max_new} stop_reason={stop_reason} "
        f"eos_id={eos_id} last={gen[-1] if gen else None}")

    passed = not failures
    return dict(passed=passed, checks=checks, failures=failures, uncovered=uncovered,
                not_applicable=not_applicable,
                n_checks=len(checks), n_required=sum(1 for c in checks if c["required"]),
                finiteness_coverage=registry.coverage(),
                rule="只有 passed=True 才允许 exit 0；执行结束(stage=run_ended) ≠ 验证通过")


def required_prefill_models(mode: str) -> list:
    """本模式下**必需**执行 prefill 的模型 —— 直接从 `expected_map` 派生，不另立判据。

    `mode="ar"` ⇒ `["target"]`；`mode="spec"` ⇒ `["draft", "target"]`。
    这样"必需集合"与 `evaluate` 用的是**同一张表**，不会出现第二套互相矛盾的判据。
    """
    exp = expected_map(mismatch_rounds=0, mode=mode)
    return sorted(m for (p, m), meta in exp.items() if p == "prefill" and meta.get("required"))


def na_prefill_models(mode: str) -> list:
    """本模式下**明确不适用**的 prefill 模型（来自 `expected_map` 的显式声明）。

    ⚠️ 这是**声明**，不是从"记录缺失"推断出来的。缺失记录一律按失败处理。
    """
    exp = expected_map(mismatch_rounds=0, mode=mode)
    return sorted(m for (p, m), meta in exp.items() if p == "prefill" and not meta.get("required"))


def contract_note() -> dict:
    """`prefill_finiteness` 契约 v2 的机器可读摘要（写进结果 JSON，便于事后核对口径）。"""
    return dict(
        contract_version=2,
        gate_field="all_required_ok",
        required_by_mode={"ar": required_prefill_models("ar"),
                          "spec": required_prefill_models("spec")},
        na_by_mode={"ar": na_prefill_models("ar"), "spec": na_prefill_models("spec")},
        rules=[
            "必需模型必须有记录、executed=True 且 ok is True，否则汇总失败",
            "只有显式声明不适用（记录 required=False 或出现在 not_applicable）才可排除",
            "不得从'记录缺失'推断 not_applicable",
            "必需集合为空 ⇒ 不通过",
            "all_executed_ok 仅字面含义，不得当作放行判据",
        ],
        derivation="required/na 均来自 smoke_checks.expected_map（与 evaluate 同一份判据）",
    )


def prefill_finiteness(checks: list, *, required_models, not_applicable=None) -> dict:
    """按**模式必需模型**汇总 `prefill logits` 有限性。契约版本 2。

    合同（逐条对应本轮要求）
      1) 必需集合由调用方**显式**给出（通常用 `required_prefill_models(mode)`）：AR={target}，
         Spec={target,draft}。
      2) 每个必需模型必须有记录、`executed is True`、`ok is True`，否则汇总失败。
      3) **只有显式声明不适用的模型**才可排除（记录里 `required=False`，或出现在 `not_applicable`
         列表里）；**不得**因为"记录缺失"就推断为 N/A。
      4) 没有任何有效检查（必需集合为空）⇒ 不通过。
      5) `all_executed_ok` 保留**字面**含义（执行过的模型是否都 ok），**不是**放行判据；
         调用方一律读 `all_required_ok`。

    修正的两处缺陷（均已在 CPU 上复现）
      * **漏检**：旧实现只检查"执行过"的模型 ⇒ Spec 下 `draft` 必需但未执行时，
        `all_executed_ok` 仍返回 True（制造了一个不会被抓住的通过）。
      * **把漏执行伪装成不适用**：`prefill_finiteness_live` 旧版把"registry 里没有记录"
        的模型直接写进 `not_applicable` ⇒ "必需检查漏执行"被降级成"模型不适用"。
    """
    req = sorted(set(required_models or ()))
    na_names = set(not_applicable or ())
    recs: dict = {}
    for e in (checks or []):
        name = e.get("name", "")
        if name.startswith("finiteness:prefill/"):
            recs[name.split("/", 1)[1]] = e

    by_model: dict = {}
    missing: list = []
    not_executed: list = []
    bad_ok: list = []
    conflicts: list = []

    for m in req:
        e = recs.get(m)
        in_na = f"finiteness:prefill/{m}" in na_names
        if e is None:
            missing.append(m)
            by_model[m] = dict(required=True, present=False, executed=False, ok=None,
                               passes=False,
                               detail="**必需模型整条记录缺失** ⇒ 判失败（不得当作 N/A）")
        else:
            executed = bool(e.get("executed"))
            ok = e.get("ok")
            by_model[m] = dict(required=True, present=True, executed=executed, ok=ok,
                               recorded_required=bool(e.get("required")),
                               passes=bool(executed and ok is True),
                               detail=e.get("detail", ""))
            if not executed:
                not_executed.append(m)
            elif ok is not True:
                bad_ok.append(m)
            if not e.get("required"):
                conflicts.append(dict(model=m, kind="required_model_recorded_as_not_required"))
        if in_na:
            conflicts.append(dict(model=m, kind="required_but_listed_not_applicable"))

    # 可排除的模型：**必须**有显式声明（记录 required=False，或出现在 na 列表），且不在必需集合
    excluded: list = []
    for m, e in recs.items():
        if m not in req and not e.get("required"):
            excluded.append(m)
    for name in na_names:
        if name.startswith("finiteness:prefill/"):
            m = name.split("/", 1)[1]
            if m not in req:
                excluded.append(m)
    excluded = sorted(set(excluded))

    executed_all = sorted(m for m, v in by_model.items() if v["executed"] and m in req)
    all_executed_ok = bool(executed_all) and all(by_model[m]["ok"] is True for m in executed_all)
    all_required_ok = bool(req) and not (missing or not_executed or bad_ok or conflicts)
    return dict(
        contract_version=2,
        required_models=req,
        by_model=by_model,
        excluded_models=excluded,
        missing_records=sorted(missing),
        not_executed_required=sorted(not_executed),
        bad_ok_required=sorted(bad_ok),
        conflicts=conflicts,
        executed_models=executed_all,
        all_required_ok=all_required_ok,
        all_executed_ok=all_executed_ok,
        all_executed_ok_warning=("字面含义：**已执行**的必需模型是否都 ok。"
                                 "**不得**用它判断必需检查完整性 —— 必需模型没执行时它仍可能为 True。"
                                 "调用方一律读 `all_required_ok`。"),
        gate_field="all_required_ok",
        semantics=("必需模型必须有记录、executed=True 且 ok is True；缺失记录 / 未执行 / ok!=True / "
                   "required 与 N/A 冲突 一律判失败；未执行**不**自动推断为 N/A；"
                   "必需集合为空 ⇒ 不通过。"),
        derivation="必需/不适用集合来自 smoke_checks.expected_map（与 evaluate 同一份判据）",
    )


def prefill_finiteness_live(registry: "CheckRegistry", mode: str, *,
                            required_models=None, not_applicable=None) -> dict:
    """`prefill_finiteness` 的**运行中**入口。

    * 必需集合默认由 `mode` 派生（AR={target}，Spec={target,draft}）；
    * registry 里**没有**记录的必需模型 ⇒ **保持缺失**，不做任何 N/A 转换（这是缺陷 2 的修法）；
    * 不适用集合来自 `expected_map` 的**显式声明**。
    """
    req = (sorted(required_models) if required_models is not None
           else required_prefill_models(mode))
    na = (list(not_applicable) if not_applicable is not None
          else [f"finiteness:prefill/{m}" for m in na_prefill_models(mode)])
    checks = []
    for m in req:
        slot = registry.calls.get(("prefill", m))
        if slot is None:
            continue                      # ★ 缺失就是缺失，不转成 N/A
        checks.append(dict(name=f"finiteness:prefill/{m}", required=True, executed=True,
                           ok=bool(slot.get("ok")), detail=f"调用 {slot.get('n')} 次"))
    out = prefill_finiteness(checks, required_models=req, not_applicable=na)
    out.update(entry="live", mode=mode)
    return out


