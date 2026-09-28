"""diag_qk_scope.py — 把 **QK FP32 诊断开关限定到指定模型实例**，并给出**干预范围**的静态/动态证明。

为什么需要它
  `model_self.DIAG_QK_FP32` 是**模块级全局**，而它只有一个**读取点**：
  `model_self.QwenLM.forward` 的 **eager 分支**里 `scores = torch.matmul(...)`。
  在没有 per-model 覆盖机制的情况下，直接 `set_diag_precision(qk32=True)` 会**同时**作用于
  target 与 draft —— 那就不再是"仅 target 启用 QK FP32"的干预，归因会被污染。

本模块的做法（**不改 `model_self.py`**）
  包一层 `QwenLM._run`：进入时按"被调用的是不是 target"把全局开关设成 True/False，
  调用结束（含异常）后**恢复原值**。因为 `model_self` 的 attention 在 `_run` 内部读取该全局量，
  所以这一层包装就实现了**按模型实例**的作用域，且对 draft 保持原精度。
  与 `smoke_hooks.DecisionHooks` 的 `_run` 包装可共存（安装顺序二者皆可，见自测）。

范围（静态可证）
  * 开关只有**一个读取点**，位于 eager 分支 ⇒ 只影响 **eager 路径**；
  * `decide_attention_path` 只在 `phase=="prefill"` 且 `s>=2` 时返回 `fused`，
    而**融合 prefill 不读该开关**（`_fused_prefill_attention` 源码里没有该名字）
    ⇒ **融合 prefill 保持不变，本干预不是"全链路升精度"**；
  * 因此实际生效的是 target 的 **decode / verify / repair**（以及 draft 的 propose，
    但本工具把 draft 排除在作用域外）。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from . import model_self as M

PHASES = ("prefill", "decode", "propose", "verify", "repair")


def _read_sites(name: str) -> list:
    """用 AST 找出 `name` 在**模块源码**里被**读取**（Load）的位置及其所在函数名。

    只看 Load 语境 ⇒ 定义/赋值不算读取点。
    """
    src = Path(M.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):                       # noqa: N802
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Name(self, node):                              # noqa: N802
            if node.id == name and isinstance(node.ctx, ast.Load):
                funcs.append(dict(line=node.lineno, func=self.stack[-1] if self.stack else None))
            self.generic_visit(node)

    V().visit(tree)
    return funcs


def static_scope_proof() -> dict:
    """把"开关到底改变哪些算子、哪些阶段生效"变成可复查的结构化事实。"""
    fused_src = inspect.getsource(M._fused_prefill_attention)
    qk_sites = _read_sites("DIAG_QK_FP32")
    prob_sites = _read_sites("DIAG_PROB_FP32")
    # `diag_precision()` 是**读取器**（它把开关读出来报告），不算"影响计算的读取点"。
    ACCESSORS = ("diag_precision", "set_diag_precision", "set_diag_preset")
    compute_sites = [s for s in qk_sites if s["func"] not in ACCESSORS]
    coverage = {}
    for ph in PHASES:
        for s in (1, 4):
            path, reason = M.decide_attention_path(ph, s, "cudnn_prefill")
            coverage[f"{ph}_s{s}"] = dict(path=path, reason=reason,
                                          switch_effective=(path == "eager"))
    return dict(
        qk_read_sites=qk_sites,
        qk_compute_read_sites=compute_sites,
        prob_read_sites=prob_sites,
        n_qk_read_sites=len(qk_sites),
        n_qk_compute_read_sites=len(compute_sites),
        qk_read_only_in_eager_branch=bool(
            len(compute_sites) == 1 and compute_sites[0]["func"] == "forward"),
        accessor_note="`diag_precision()` 只是读取并报告状态，不参与计算；"
                      "**影响计算的读取点**见 `qk_compute_read_sites`",
        fused_bypasses_qk=("DIAG_QK_FP32" not in fused_src),
        fused_bypasses_prob=("DIAG_PROB_FP32" not in fused_src),
        fused_impl="model_self._fused_prefill_attention",
        coverage_by_phase=coverage,
        effective_stages=["decode(AR target)", "verify(SpecDec target)",
                          "repair(SpecDec target)"],
        not_effective_stages=["prefill（融合 cuDNN，绕过该开关）"],
        changes_ops="QK 打分：`matmul(q,k^T)` 与 `1/sqrt(head_dim)` 缩放改为 fp32；"
                    "softmax 的**输入**因此不再是 bf16 舍入后的 scores",
        unchanged_ops=["softmax 本身（两条路径本来就以 float32 计算）",
                       "attn→bf16 回转（DIAG_PROB_FP32 关闭，保持历史口径）",
                       "AV matmul（bf16）", "o_proj / MLP / RMSNorm / lm_head / 权重 / 缓存 dtype",
                       "并列规则 / 验收阈值 / SpecDec 算法"],
        output_dtype_unchanged=True,
        scope_note="**不是**全链路升精度：融合 prefill 未变；draft 在本工具中未启用干预。",
    )


class QKScopeHook:
    """按**模型实例**限定 QK FP32 开关；对 draft（或任何非 target）保持原精度。"""

    def __init__(self, tgt, enabled: bool, run_owner=None):
        self.tgt = tgt
        self.enabled = bool(enabled)
        self.run_owner = run_owner if run_owner is not None else M.QwenLM
        self._orig = None
        self.installed = False
        self.stats = dict(calls_total=0, calls_target=0, calls_other=0,
                          flag_on_calls=0, flag_off_calls=0, max_flag_seen=0)
        self._other_classes: list = []

    # ------------------------------------------------------------------ 安装/卸载
    def install(self) -> None:
        assert not self.installed, "已经安装过"
        cur = M.diag_precision()
        assert not cur["qk_fp32"] and not cur["prob_fp32"], (
            f"安装 QKScopeHook 前诊断开关必须是干净的 (False,False)，当前={cur}")
        self._orig = self.run_owner._run
        hook, orig = self, self._orig

        # ⚠️ 必须安装**普通函数**而不是绑定方法：把 `self._wrapper`（绑定方法）赋给类，
        # `instance._run(x)` **不会**重新绑定，`x` 会被塞进第一个形参
        # （`smoke_hooks` 早就踩过同一个坑，见其文件头注释）。
        def _wrapper(model, x, cache_list=None, need_logits=True):
            return hook._dispatch(model, x, cache_list, need_logits, orig)

        self.run_owner._run = _wrapper
        self.installed = True

    def _dispatch(self, model, x, cache_list, need_logits, orig):
        is_tgt = model is self.tgt
        if not is_tgt:
            self._other_classes.append(type(model).__name__)
        on = bool(self.enabled and is_tgt)
        prev = bool(M.diag_precision()["qk_fp32"])
        self.stats["calls_total"] += 1
        self.stats["calls_target" if is_tgt else "calls_other"] += 1
        if on:
            self.stats["flag_on_calls"] += 1
        else:
            self.stats["flag_off_calls"] += 1
        self.stats["max_flag_seen"] = max(self.stats["max_flag_seen"], int(on))
        if on != prev:
            M.set_diag_precision(qk32=on)
        try:
            return orig(model, x, cache_list, need_logits)
        finally:
            if on != prev:
                M.set_diag_precision(qk32=prev)      # 异常也必须复位

    def remove(self) -> None:
        if not self.installed:
            return
        self.run_owner._run = self._orig
        self.installed = False
        M.set_diag_precision(qk32=False, prob32=False)   # 卸载后确保干净

    def report(self) -> dict:
        return dict(enabled=self.enabled, run_owner=self.run_owner.__name__,
                    scope="target-only" if self.enabled else "control(no intervention)",
                    **self.stats,
                    other_model_classes=sorted(set(self._other_classes)),
                    note="calls_target 全部应在 flag_on（干预组）或全部 flag_off（控制组）；"
                         "calls_other 必须全部 flag_off（draft 保持原精度）")
