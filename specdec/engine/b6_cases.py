"""b6_cases.py — B6 **固定用例表**（单一来源；2026-10-05）。

## 为什么需要
`design_freeze_g4_specdec.md:83` 要求 B6 覆盖 10 类 prompt 形态。原实现把这些用例写在
`g4_b6_correctness.py:CASES` 里，但那里的参照是**生产路径 vs AR（旧判据）**。
本轮要把它们跑在**新口径（serial oracle vs AR）**下，且必须：
  * 输入**逐字不变**（文本与 `n` 不得擅自改动）；
  * 每个用例声明"**它应当证明什么**"（类型 + 分支 + 融合期望），以便用 trace 判定覆盖；
  * 用例表只有**一处**定义，避免"计划里的用例"与"实际跑的用例"漂移。

## 纪律
* 文本与 `n` 与 `g4_b6_correctness.py:CASES` **逐字一致**（由 CPU 自测 `ast` 解析比对强制）；
* 分支声明**只声明该用例在设计上应当触发的**；未触发的分支由检查器如实记 **未覆盖**（不得当 PASS）；
* `expect_prefill_fused`：`True` = 该用例**应当**触发 prefill 融合；`None` = **按实现的
  分派规则**（例如极短输入可能合理回退 eager），**不**作硬性要求。
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_TOOL = ROOT / "specdec" / "engine" / "g4_b6_correctness.py"

# 用例表（`kind` 对应用户要求的类型清单；`branches` = 该用例**应当**触发的分支）
B6_CASES: tuple = (
    dict(case_id="natural-1", kind="natural_language", text="The capital of France is",
         max_new=16, branches=(), expect_prefill_fused=True),
    dict(case_id="natural-2", kind="natural_language", text="Photosynthesis is the process by which",
         max_new=16, branches=(), expect_prefill_fused=True),
    dict(case_id="math", kind="math", text="3 + 5 = 8. And 7 times 6 is",
         max_new=16, branches=(), expect_prefill_fused=True),
    dict(case_id="code", kind="code", text="def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
         max_new=16, branches=(), expect_prefill_fused=True),
    dict(case_id="short", kind="short_input", text="Hello",
         max_new=16, branches=(), expect_prefill_fused=None),
    dict(case_id="long", kind="long_input",
         text=("The history of artificial intelligence and neural networks spans many "
               "decades of research into learning systems that process language with "
               "attention over long sequences of tokens in large models."),
         max_new=16, branches=(), expect_prefill_fused=True),
    dict(case_id="len13", kind="length_not_multiple_of_k", text="Q: what is the largest planet?\nA:",
         max_new=13, branches=("trim_last_round", "len_not_multiple_of_k"), expect_prefill_fused=True),
    dict(case_id="len5", kind="length_not_multiple_of_k", text="Once upon a time",
         max_new=5, branches=("trim_last_round", "len_not_multiple_of_k"), expect_prefill_fused=None),
    dict(case_id="len2", kind="length_lt_k", text="The answer is",
         max_new=2, branches=("trim_last_round", "len_not_multiple_of_k"), expect_prefill_fused=None),
    dict(case_id="eos-prone", kind="eos", text="Repeat the word cat forever: cat cat cat cat cat cat",
         max_new=16, branches=("eos_stop",), expect_prefill_fused=True),
)

# **集合级**要求：跨全部用例的 trace 合计必须覆盖这些分支（否则覆盖不完整）
SET_REQUIRED_BRANCHES: tuple = ("all_accept", "first_reject", "mid_reject", "rollback",
                                "trim_last_round")

SET_REQUIRED_TYPES: tuple = ("natural_language", "math", "code", "short_input", "long_input",
                            "length_not_multiple_of_k", "length_lt_k", "eos")


def declared_document() -> dict:
    """给 `b6_branch_coverage_cpu --declared` 用的声明文档（可直接落盘为 JSON）。"""
    return dict(
        note="B6 固定用例声明（单一来源 = specdec/engine/b6_cases.py）；分支未触发即记未覆盖",
        cases={c["case_id"]: dict(kind=c["kind"], text_sha1=None, max_new=c["max_new"],
                                  branches=list(c["branches"]),
                                  expect_prefill_fused=c["expect_prefill_fused"])
               for c in B6_CASES},
        set_requirements=dict(branches=list(SET_REQUIRED_BRANCHES), types=list(SET_REQUIRED_TYPES)))


def legacy_cases() -> list:
    """用 `ast` 解析旧工具里的 `CASES`（**不 import**，避免拉起 torch/模型依赖）。"""
    tree = ast.parse(LEGACY_TOOL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets):
            return [tuple(x) for x in ast.literal_eval(node.value)]
    raise RuntimeError("未能在 g4_b6_correctness.py 中找到 CASES")
