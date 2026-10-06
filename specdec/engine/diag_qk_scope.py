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


ACCESSOR_FUNCS = ("diag_precision", "set_diag_precision", "set_diag_preset")


def read_site_roles(name: str = "DIAG_QK_FP32", *, co_switch: str = "_SERIAL_QK",
                    source: str | None = None) -> list:
    """把 `name` 的读取点**按角色**分类（静态、可复查）。

    | role | 判据 | 含义 |
    | --- | --- | --- |
    | `accessor` | 位于 `diag_precision()` / `set_diag_precision()` / `set_diag_preset()` | 只读取并**报告**状态，不参与计算 |
    | `precision` | 读取点**本身**就是某个 `if`/`elif` 的测试（`elif DIAG_QK_FP32:`） | 直接决定**精度表达式**（QK FP32 分支） |
    | `guard` | 读取点位于某个 `BoolOp`（and/or）内 | **路径选择/联合条件**；若该 `BoolOp` 是 `and` 且**首个操作数**测试 `co_switch`（`_SERIAL_QK`），则默认（`co_switch is None`）时**短路**、不参与任何决策 |
    | `unclassified` | 以上都不是 | ⇒ **必须为 0**（防止未来静默新增一个语义未定的读取点） |

    为什么要分类：`DIAG_QK_FP32` 的"作用范围"由**两类**读取点共同决定 —— 一类改精度，另一类只选路径；
    把它们混成一个计数（首版自测的做法）既会在新增 guard 时误报，也会掩盖"guard 是否默认惰性"这个真问题。

    `source` 可传入**合成源码**（默认读 `model_self.py`）⇒ 分类器本身可被回归测试
    （见 `qk_scope_selftest_cpu` 的 §1b：`if S:` ⇒ precision；`(_C is not None and not S)` ⇒ 惰性 guard；
    `(S and _C is not None)` ⇒ guard 但**不**惰性；`S + 1` ⇒ unclassified）。
    """
    src = source if source is not None else Path(M.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    parents: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    out: list = []

    def _names(node) -> set:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):                       # noqa: N802
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Name(self, node):                              # noqa: N802
            if node.id != name or not isinstance(node.ctx, ast.Load):
                self.generic_visit(node)
                return
            func = self.stack[-1] if self.stack else None
            cur, role, detail = node, None, {}
            while True:
                par = parents.get(id(cur))
                if par is None:
                    break
                if isinstance(par, ast.BoolOp):
                    first = par.values[0]
                    ops = {type(o).__name__ for o in par.values[1:]}
                    detail = dict(
                        boolop=type(par.op).__name__,
                        first_operand_names=sorted(_names(first)),
                        co_switch=co_switch,
                        co_switch_in_same_boolop=co_switch in _names(par),
                        target_in_first_operand=name in _names(first),
                        other_operands=sorted(ops))
                    role = "guard"
                    break
                if isinstance(par, ast.If) and par.test is node:
                    detail = dict(test_is_switch=True)
                    role = "precision"
                    break
                cur = par
            if role is None:
                role = "accessor" if func in ACCESSOR_FUNCS else "unclassified"
                detail = dict(why=("读取器：只报告状态" if role == "accessor"
                                   else "未归类：既非 if/elif 测试、也不在 BoolOp 内"))
            inert = bool(role == "guard" and detail.get("boolop") == "And"
                         and not detail.get("target_in_first_operand")
                         and co_switch in detail.get("first_operand_names", []))
            out.append(dict(line=node.lineno, func=func, role=role, inert_by_default=inert,
                            **detail))
            self.generic_visit(node)

    V().visit(tree)
    return sorted(out, key=lambda r: r["line"])


def static_scope_proof() -> dict:
    """把"开关到底改变哪些算子、哪些阶段生效"变成可复查的结构化事实。"""
    fused_src = inspect.getsource(M._fused_prefill_attention)
    qk_sites = _read_sites("DIAG_QK_FP32")
    prob_sites = _read_sites("DIAG_PROB_FP32")
    # `diag_precision()` 是**读取器**（它把开关读出来报告），不算"影响计算的读取点"。
    ACCESSORS = ("diag_precision", "set_diag_precision", "set_diag_preset")
    compute_sites = [s for s in qk_sites if s["func"] not in ACCESSORS]
    # ---- 2026-10-05：按**角色**分类（`n_qk_compute_read_sites` 的含义**不变**：仍是非读取器的读取点）----
    roles = read_site_roles("DIAG_QK_FP32")
    precision = [r for r in roles if r["role"] == "precision"]
    guards = [r for r in roles if r["role"] == "guard"]
    accessor_sites = [r for r in roles if r["role"] == "accessor"]
    unclassified = [r for r in roles if r["role"] == "unclassified"]
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
        qk_read_roles=roles,
        qk_precision_read_sites=precision,
        n_qk_precision_read_sites=len(precision),
        qk_guard_read_sites=guards,
        n_qk_guard_read_sites=len(guards),
        qk_accessor_read_sites=accessor_sites,
        n_qk_accessor_read_sites=len(accessor_sites),
        qk_unclassified_read_sites=unclassified,
        n_qk_unclassified_read_sites=len(unclassified),
        all_non_accessor_sites_classified=not unclassified,
        guards_inert_by_default=all(g["inert_by_default"] for g in guards) if guards else True,
        role_note="`precision` = 直接决定精度表达式（`elif DIAG_QK_FP32:`）；"
                  "`guard` = 只在 `BoolOp` 里与 `_SERIAL_QK` 联合判定的**路径选择**读取点"
                  "（默认 `_SERIAL_QK is None` ⇒ 短路，不参与任何决策）；"
                  "`unclassified` 必须为 0。两类读取点的**总数**仍等于 `n_qk_compute_read_sites`。",
        qk_read_only_in_eager_branch=bool(
            len(precision) == 1 and precision[0]["func"] == "forward"),
        accessor_note="`diag_precision()` 只是读取并报告状态，不参与计算；"
                      "**影响计算的读取点**见 `qk_compute_read_sites`（= precision + guard）",
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
