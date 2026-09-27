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
