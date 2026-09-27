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
    """记录每次有限性检查的 (阶段, 模型, 角色)，并给出覆盖情况。"""

    def __init__(self) -> None:
        self.calls: dict[tuple[str, str], dict] = {}
        self.events: list[dict] = []
        self.n_checked = 0
        self.first_bad: dict | None = None

    def _slot(self, phase: str, model: str) -> dict:
        return self.calls.setdefault((phase, model), dict(n=0, roles=[], ok=True, bad=[]))

    def check_finite(self, t, *, phase: str, model: str, role: str, where: str = "") -> bool:
        """在**决策之前**检查。返回是否有限；非有限时抛 `FinitenessError`（先落记录再抛）。"""
        if t is None:
            self.events.append(dict(phase=phase, model=model, role=role, where=where,
                                    checked=False, note="张量为 None（该分支未产生决策用 logits）"))
            return False
        finite = bool(torch.isfinite(t).all())
        slot = self._slot(phase, model)
        slot["n"] += 1
        if role not in slot["roles"]:
            slot["roles"].append(role)
        self.n_checked += 1
        rec = dict(phase=phase, model=model, role=role, where=where, checked=True,
                   finite=finite, shape=[int(x) for x in t.shape], dtype=str(t.dtype))
        self.events.append(rec)
        if not finite:
            slot["ok"] = False
            slot["bad"].append(rec)
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            self.first_bad = dict(rec, n_nan=n_nan, n_inf=n_inf)
            raise FinitenessError(
                f"非有限值：phase={phase} model={model} role={role} {where} "
                f"shape={list(t.shape)} NaN={n_nan} Inf={n_inf}"
                f"（在 argmax/接受判断**之前**检出，已停止）")
        return True

    def coverage(self) -> dict:
        return {"n_checked": self.n_checked,
                "by_phase_model": {f"{p}/{m}": dict(v) for (p, m), v in sorted(self.calls.items())},
                "events": self.events,
                "first_bad": self.first_bad}


def expected_map(*, mismatch_rounds: int) -> dict[tuple[str, str], dict]:
    """本次 run **应当**有哪些检查（由执行结构决定，不由结果决定）。"""
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
             alloc_detail: str = "") -> dict:
    """把"执行结束"升级为"验证通过/不通过"。**只有 passed=True 才允许 exit 0。**"""
    exp = expected_map(mismatch_rounds=mismatch_rounds)
    checks: list[dict] = []
    failures: list[dict] = []
    uncovered: list[str] = []
    not_applicable: list[str] = []

    # ---- 1) 有限性检查的**覆盖与结果** ----
    for (phase, model), meta in sorted(exp.items()):
        name = f"finiteness:{phase}/{model}"
        if (phase, model) in NEVER_USED:
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
    req("generation_length_and_stop_reason", len_ok,
        f"gen_len={gen_len} max_new={max_new} stop_reason={stop_reason} "
        f"eos_id={eos_id} last={gen[-1] if gen else None}")

    passed = not failures
    return dict(passed=passed, checks=checks, failures=failures, uncovered=uncovered,
                not_applicable=not_applicable,
                n_checks=len(checks), n_required=sum(1 for c in checks if c["required"]),
                finiteness_coverage=registry.coverage(),
                rule="只有 passed=True 才允许 exit 0；执行结束(stage=run_ended) ≠ 验证通过")
