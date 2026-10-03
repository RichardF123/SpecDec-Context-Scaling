"""serial_oracle.py — 串行 target oracle 的**调用几何审计**（默认不安装；只读、零副作用）。

## 目的
硬性证明：在 serial 模式下，generation 阶段的**每一次** target 前向都是 `seq_len == 1`
（与 AR 相同），且**没有** block `_run(s>1)`、没有 row-aligned wrapper、没有"内部先算 M=4 再切第 0 行"。

## 实现
临时包装 `model._run`（**只包 target**；draft 单独记录），逐个记录 `x.shape[1]`：
* 允许**恰好一次** prefill 调用（`s == prompt_len`；`attention_phase == "prefill"` 或首个大 `s`）；
* 其余（generation）调用必须 `s == 1`，否则**立即抛错**（不降级、不重试）；
* `exit` 时（含异常）恢复原 `_run`（`is` 同一对象核对）。

**不参与任何计算**：包装只记录形状后调用原函数。
"""
from __future__ import annotations

import contextlib

import torch


class CallGeometryError(RuntimeError):
    """调用几何违反串行 oracle 契约（硬失败）。"""


class _Audit:
    def __init__(self, model, *, role: str, allow_first_large: bool = True, prompt_len: int | None = None):
        self.model = model
        self.role = role
        self.allow_first_large = bool(allow_first_large)
        self.prompt_len = prompt_len
        self.calls: list = []
        self.n_large = 0
        self.orig = None

    def __call__(self, x, cache, *a, **kw):
        s = int(x.shape[1])
        self.calls.append(s)
        is_first = (len(self.calls) == 1)
        if s > 1:
            if not (is_first and self.allow_first_large):
                raise CallGeometryError(
                    f"{self.role}: generation 阶段出现 seq_len={s} > 1 的 target 前向（第 {len(self.calls)} 次调用）"
                    f" ⇒ 违反串行 oracle 契约（禁止 block _run(s>1)；禁止先算 M=4 再切第 0 行）")
            self.n_large += 1
        return self.orig(x, cache, *a, **kw)

    def summary(self) -> dict:
        gen = self.calls[1:] if self.calls else []
        return dict(role=self.role, n_calls=len(self.calls), per_call_seq_len=list(self.calls),
                    prefill_calls=int(self.n_large), generation_calls=len(gen),
                    generation_seq_lens=sorted(set(gen)),
                    all_generation_m1=bool(all(s == 1 for s in gen)),
                    max_generation_seq_len=(max(gen) if gen else None))


@contextlib.contextmanager
def target_call_audit(target, *, prompt_len: int | None = None, role: str = "target"):
    """包装 **target** 的 `_run`，强制 generation 调用 `s == 1`；退出（含异常）恢复原函数。"""
    aud = _Audit(target, role=role, allow_first_large=True, prompt_len=prompt_len)
    orig = target._run
    aud.orig = orig
    target._run = aud
    try:
        yield aud
    finally:
        if target._run is aud:
            target._run = orig
        else:                                                             # 被别的包装覆盖过：如实报错
            raise CallGeometryError(f"{role}: `_run` 未被本审计恢复（存在嵌套包装？）")


@contextlib.contextmanager
def draft_call_audit(draft, *, role: str = "draft"):
    """只**记录** draft 的调用形状（不受 `s==1` 断言约束）。"""
    aud = _Audit(draft, role=role, allow_first_large=True)
    orig = draft._run
    aud.orig = orig
    draft._run = aud
    try:
        yield aud
    finally:
        if draft._run is aud:
            draft._run = orig
