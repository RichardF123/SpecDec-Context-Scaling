"""smoke_hooks.py — 诊断用 hook：**阶段标注** + **决策点有限性检查**（含 `_propose` 中间步）。

设计约束（来自任务要求）
  1. **不复制一套投机算法、不改变生产计算语义**：只包装 `spec_dec` 的四个入口与
     `QwenLM._run`，在**返回值上**做检查（纯读取），不插入任何计算；
  2. **早于 argmax**：`spec_dec._propose` 的循环结构是
     `t = argmax(lg)` → `lg, cache = draft._run(...)` → 下一轮再用 `lg`
     ⇒ 循环内每次 `draft._run` 的返回值必须**在返回时就检查**，否则它会在
     **下一次迭代的 argmax** 处被使用而未经检查。本模块通过包装 `QwenLM._run`
     并把阶段限定为 `propose` 来覆盖这些**中间步**；
  3. **仅在诊断 harness 中启用**：hook 只在本模块 `install()` 期间存在，生产默认不装。
"""
from __future__ import annotations

import time

import torch

from . import model_self as M
from . import spec_dec
from .smoke_checks import CheckRegistry


def _sync() -> None:
    """计时边界同步（CPU 测试环境下 CUDA 可能未初始化 ⇒ 失败即忽略，不改变语义）。"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                          # noqa: BLE001
        pass


# `_run` hook **必须是普通函数**：若把绑定方法赋给类，`instance._run(...)` 不会重新绑定，
# 于是模型实例会被塞进第一个形参（轻量集成测试正是这样抓到了这个 bug）。
_HOOK_REGISTRY: dict[type, "DecisionHooks"] = {}


def _hook_for(model) -> "DecisionHooks | None":
    h = _HOOK_REGISTRY.get(type(model))
    if h is not None:
        return h
    for cls, hh in _HOOK_REGISTRY.items():
        if isinstance(model, cls):
            return hh
    return None


def _run_hook(model, x, cache_list=None, need_logits=True):
    """`QwenLM._run`（或测试用 stub 的 `_run`）的包装：检查后原样返回。"""
    h = _hook_for(model)
    if h is None:                                  # 理论上不会发生（已卸载）
        raise RuntimeError("DecisionHooks 的 _run hook 被调用但注册表里没有对应实例")
    lg, new_cache = h._orig["run"](model, x, cache_list, need_logits)
    # **只在 propose 阶段**做中间步检查：这正是"会被下一次 argmax 使用"的那份 logits。
    if lg is not None and M.current_attention_phase() == "propose":
        step = h.reg.next_propose_step()
        h.reg.check_finite(
            lg, phase="propose", model=h._model_key(model), role="mid_lg",
            round_=h.reg.round + 1, step=step,
            where=f"propose 循环内第 {step} 次 _run 的返回："
                  f"将被第 {step + 1} 次迭代的 argmax 使用")
    return lg, new_cache


class DecisionHooks:
    """把阶段标注与决策点检查装到 `spec_dec` 的四个入口 + `QwenLM._run` 上。

    参数
      tgt / drf        用于把 `_run` 的 self 映射成 "target"/"draft"
      registry         `CheckRegistry`（记账 + 非有限即抛）
      prefill_impl     真实分块 prefill 实现（替代 `spec_dec._prefill` 的内部）
      on_prefill       `(model_key, seconds) -> None`，供上层做阶段落盘/计时
      on_round_start   每次进入 `_verify`（= 新一轮）时调用
    """

    def __init__(self, *, tgt, drf, registry: CheckRegistry, prefill_impl,
                 on_prefill=None, on_round_start=None, run_owner=None,
                 on_verify_pre=None, on_verify_post=None):
        self.tgt, self.drf, self.reg = tgt, drf, registry
        self.prefill_impl = prefill_impl
        self.on_prefill = on_prefill
        self.on_round_start = on_round_start
        # `on_verify_pre(cache, cands)` 在**调用真 `_verify` 之前**触发 —— 这是唯一能拿到
        # "本轮开始时、未被就地修改过的 target cache"的时机（`_verify` 会就地改写传入的 dict）。
        self.on_verify_pre = on_verify_pre
        self.on_verify_post = on_verify_post
        # `_run` hook 装在**哪个类**上：生产用 `QwenLM`；轻量集成测试用 stub 类，
        # 这样测试跑的是**同一套 hook 代码**而无需加载大模型，且完全不碰生产类。
        self.run_owner = run_owner if run_owner is not None else M.QwenLM
        self._orig: dict = {}
        self.installed = False

    # ------------------------------------------------------------------ 包装器
    def _model_key(self, model) -> str:
        return "target" if model is self.tgt else "draft"

    def _prefill(self, model, ids_):
        key = self._model_key(model)
        _sync()
        t0 = time.perf_counter()
        with M.attention_phase("prefill"):
            lg, cache = self.prefill_impl(model, ids_)
        _sync()
        dt = time.perf_counter() - t0
        self.reg.check_finite(lg, phase="prefill", model=key, role="prefill_lg",
                              where="prefill 返回：供 saved_next/首轮 argmax 使用")
        if self.on_prefill is not None:
            self.on_prefill(key, dt)
        return lg, cache

    def _propose(self, *a, **k):
        self.reg.begin_propose()             # 每次进入 _propose 重置候选步号
        if len(a) > 2:
            self.reg.check_finite(a[2], phase="propose", model="draft", role="input_lg",
                                  round_=self.reg.round + 1, step=0,
                                  where="propose 入口 lg：**第一次** argmax 的输入")
        with M.attention_phase("propose"):
            cands, cache, lg_out = self._orig["propose"](*a, **k)
        # 循环内的中间步已由 `_run` hook 逐个检查；这里再核对返回值本身（冗余但便宜）
        self.reg.check_finite(lg_out, phase="propose", model="draft", role="return_lg",
                              round_=self.reg.round + 1, step="return",
                              where="propose 返回的 draft 末位 logits（下一轮输入）")
        return cands, cache, lg_out

    def _verify(self, *a, **k):
        self.reg.begin_round()               # 每轮 verify 一次 ⇒ 轮次在此推进
        if self.on_round_start is not None:
            self.on_round_start(self.reg.round)
        if self.on_verify_pre is not None and len(a) > 2:
            self.on_verify_pre(a[1], a[2])   # (cache_before_mutation, cands)
        with M.attention_phase("verify"):
            rows, blk = self._orig["verify"](*a, **k)
        if self.on_verify_post is not None:
            self.on_verify_post(rows)
        self.reg.check_finite(rows, phase="verify", model="target", role="rows",
                              round_=self.reg.round,
                              where="verify 返回候选行：**紧接**被 argmax 用于接受判断")
        return rows, blk

    def _step(self, *a, **k):
        key = "target" if (a and a[0] is self.tgt) else "draft"
        with M.attention_phase("repair"):
            lg_out, cache = self._orig["step"](*a, **k)
        self.reg.check_finite(lg_out, phase="repair", model=key, role="saved_next",
                              round_=self.reg.round,
                              where="repair 返回的末位 logits（下一轮 g[0] 的来源）")
        return lg_out, cache

    # ------------------------------------------------------------------ 安装/卸载
    def install(self) -> None:
        assert not self.installed, "已经安装过"
        self._orig = dict(prefill=spec_dec._prefill, propose=spec_dec._propose,
                          verify=spec_dec._verify, step=spec_dec._step,
                          run=self.run_owner._run)
        spec_dec._prefill = self._prefill
        spec_dec._propose = self._propose
        spec_dec._verify = self._verify
        spec_dec._step = self._step
        _HOOK_REGISTRY[self.run_owner] = self
        self.run_owner._run = _run_hook          # **普通函数**，才能正确绑定模型实例
        self.installed = True

    def remove(self) -> None:
        if not self.installed:
            return
        spec_dec._prefill = self._orig["prefill"]
        spec_dec._propose = self._orig["propose"]
        spec_dec._verify = self._orig["verify"]
        spec_dec._step = self._orig["step"]
        self.run_owner._run = self._orig["run"]
        self.installed = False
