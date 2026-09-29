"""status_fields.py — **五种状态分开记录**（纯函数，可 CPU 自测）。

为什么必须分开（本轮要求）
  把"跑完了"当成"正式正确性通过"是本项目最危险的一类错误。因此在记录层面就把五件事拆开，
  并且让**正式门（frozen formal gate）在本工具里结构上不可能报 pass**：

    execution_status      这次进程/算法执行本身怎么样（含**环境拒绝**与**算法失败**分开）
    structural_status     结构不变量与跨层 cache 结构检查
    repeatability_status  同配置两次独立重复是否一致
    ar_spec_equality      同 family/同 KV 的 AR 与 SpecDec 的 token 序列关系
    formal_gate_status    冻结的正式判据（本工具只可能是 not_established / fail）

**诊断完成 ≠ 正式 correctness PASS**：本模块提供 `DIAGNOSTIC_ONLY_NOTE` 并要求调用方原样写入结果 JSON。

环境拒绝与算法失败**分开分类**（`env_refused` vs `algorithm_failed`），但二者**都**停止本轮推进
（`stops_advance()` 只对 `complete` 返回 False）。
"""
from __future__ import annotations

# ---------------- execution_status 取值 ----------------
EXEC_MISSING = "missing"                      # 记录不存在
EXEC_ENV_REFUSED = "env_refused"              # 环境门不合格（护栏/P-state），未加载模型
EXEC_CRASHED = "crashed"                      # 未捕获异常 / OOM
EXEC_ALGORITHM_FAILED = "algorithm_failed"    # 算法或检查失败（NaN、必需检查缺失、长度异常…）
EXEC_INCOMPLETE = "incomplete"                # 跑完但证据不完整（无法判定）
EXEC_COMPLETE = "complete"                    # 本工具意义上"跑完且证据自洽"

STOP_REASONS_OK = ("max_new_reached", "eos")

DIAGNOSTIC_ONLY_NOTE = (
    "这些是**诊断**字段。`execution_status=complete` 只表示「这一次按计划跑完且证据自洽」，"
    "**不构成**任何正式正确性结论；正式判据见 formal_gate_status。"
)


def _checks(rec: dict) -> dict:
    return (rec or {}).get("checks") or {}


def failure_class(rec: dict | None) -> str:
    """从**已有字段**派生失败分类（不新增判据来源）。"""
    if not rec:
        return "missing"
    stage = str(rec.get("stage", ""))
    if stage == "refused":
        return "env"
    if rec.get("traceback") or stage == "crashed":
        return "crash"
    if rec.get("oom") or rec.get("alloc_on_oom"):
        return "oom"
    if rec.get("non_finite"):
        return "finite"
    chk = _checks(rec)
    pf = (rec.get("run") or {}).get("prefill_logits_finite_by_model") or {}
    if pf and not pf.get("all_required_ok", True):
        return "coverage"
    if any(str(f.get("name", "")).startswith("finiteness") for f in (chk.get("failures") or [])):
        return "finite"
    if not chk.get("passed", False):
        return "coverage_or_checks"
    run = rec.get("run") or {}
    if run.get("stop_reason") not in STOP_REASONS_OK:
        return "length_or_stop"
    inv = rec.get("round_invariants") or {}
    if inv and not inv.get("all_ok", True):
        return "structure"
    if rec.get("cache_structure_ok") is False:
        return "structure"
    return "none"


def execution_status(rec: dict | None) -> dict:
    """执行层状态。**环境拒绝与算法失败分开**，但都属于"停止推进"。"""
    if not rec:
        return dict(status=EXEC_MISSING, reason="记录不存在", stops_advance=True,
                    failure_class="missing")
    fc = failure_class(rec)
    stage = str(rec.get("stage", ""))
    run = rec.get("run") or {}
    chk = _checks(rec)
    if stage == "refused" or fc == "env":
        return dict(status=EXEC_ENV_REFUSED, reason=rec.get("skipped") or "环境门不合格",
                    stops_advance=True, failure_class=fc)
    if fc in ("crash", "oom"):
        return dict(status=EXEC_CRASHED, reason=f"failure_class={fc}", stops_advance=True,
                    failure_class=fc)
    if fc in ("finite", "coverage", "structure", "length_or_stop", "coverage_or_checks"):
        return dict(status=EXEC_ALGORITHM_FAILED, reason=f"failure_class={fc}",
                    stops_advance=True, failure_class=fc)
    if stage != "complete":
        return dict(status=EXEC_INCOMPLETE, reason=f"stage={stage!r}", stops_advance=True,
                    failure_class=fc or "unknown")
    if not chk.get("passed", False):
        return dict(status=EXEC_ALGORITHM_FAILED, reason="checks.passed=False",
                    stops_advance=True, failure_class="coverage_or_checks")
    if run.get("stop_reason") not in STOP_REASONS_OK:
        return dict(status=EXEC_ALGORITHM_FAILED,
                    reason=f"stop_reason={run.get('stop_reason')!r}", stops_advance=True,
                    failure_class="length_or_stop")
    return dict(status=EXEC_COMPLETE, reason="stage=complete 且 checks.passed=True",
                stops_advance=False, failure_class="none")


def structural_status(rec: dict | None) -> dict:
    """结构层状态。**未做检查时返回 not_checked，而不是 ok**（不得把"没查"当"通过"）。"""
    if not rec:
        return dict(status="missing", reason="记录不存在")
    inv = rec.get("round_invariants")
    # ⚠️ `all_ok=None` 是"**没记录**"，不是"失败"。首版用 `bool(...)` 把 None 折成 False，
    #    于是"未记录不变量"被误判为结构失败（自测第 5 例抓到）。
    inv_ok = None
    if inv:
        raw = inv.get("all_ok")
        inv_ok = None if raw is None else bool(raw)
    cache_ok = rec.get("cache_structure_ok")          # None = 本 run 未启用结构检查
    if inv_ok is False or cache_ok is False:
        return dict(status="fail",
                    reason=f"round_invariants.all_ok={inv_ok} cache_structure_ok={cache_ok}",
                    round_invariants_ok=inv_ok, cache_structure_ok=cache_ok)
    if inv_ok is None and cache_ok is None:
        return dict(status="not_checked", reason="本 run 未记录结构检查（需显式启用）",
                    round_invariants_ok=inv_ok, cache_structure_ok=cache_ok)
    if inv_ok is True and cache_ok is True:
        return dict(status="ok", reason="跨层 cache 结构与逐轮不变量均通过",
                    round_invariants_ok=inv_ok, cache_structure_ok=cache_ok)
    return dict(status="partial", reason=f"round_invariants_ok={inv_ok} cache_structure_ok={cache_ok}",
                round_invariants_ok=inv_ok, cache_structure_ok=cache_ok)


def _gen(rec: dict | None):
    if not rec:
        return None
    g = (rec.get("run") or {}).get("gen")
    return [int(x) for x in g] if g is not None else None


def compare_sequences(a, b) -> dict:
    """两条 token 序列的关系（共同前缀 / 首个分歧 / 两侧 token）。纯函数。"""
    if a is None or b is None:
        return dict(assessable=False, reason="有一侧缺记录或 gen 缺失")
    n = min(len(a), len(b))
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    if p == n and len(a) == len(b):
        return dict(assessable=True, identical=True, common_prefix_len=p,
                    first_divergence_index=None, a_token=None, b_token=None,
                    a_len=len(a), b_len=len(b))
    return dict(assessable=True, identical=False, common_prefix_len=p,
                first_divergence_index=p,
                a_token=(a[p] if p < len(a) else None),
                b_token=(b[p] if p < len(b) else None),
                a_len=len(a), b_len=len(b),
                length_mismatch_only=(p == n and len(a) != len(b)))


def repeatability_status(rec1: dict | None, rec2: dict | None) -> dict:
    """同配置两次独立重复的一致性。**不一致 ⇒ 标待查并停止推进**（不是错误归因，也不是豁免）。"""
    if not rec1 or not rec2:
        return dict(status="not_assessable", reason="缺少一次重复的记录",
                    stops_advance=True)
    a, b = _gen(rec1), _gen(rec2)
    cmp_ = compare_sequences(a, b)
    if not cmp_["assessable"]:
        return dict(status="not_assessable", reason=cmp_["reason"], stops_advance=True)
    st = "identical" if cmp_["identical"] else "different"
    return dict(status=st, stops_advance=(st != "identical"),
                reason=("两次重复逐 token 相同" if cmp_["identical"]
                        else "两次重复不同 ⇒ 标**待查**并停止推进，不重试、不挑样本"),
                comparison=cmp_)


def ar_spec_equality(ar_rec: dict | None, spec_rec: dict | None, *,
                     on_divergence: str = "stop") -> dict:
    """AR 与 SpecDec 的 token 关系。

    `on_divergence="record"` 时**分歧只记录、不停止推进**（诊断模式下分歧是观测对象）；
    但它**不等于豁免**：本字段只是"相等性记录"，正式判据另见 `formal_gate_status`。
    """
    a, b = _gen(ar_rec), _gen(spec_rec)
    cmp_ = compare_sequences(a, b)
    if not cmp_["assessable"]:
        return dict(status="not_assessable", reason=cmp_["reason"], stops_advance=True,
                    on_divergence=on_divergence)
    if cmp_["identical"]:
        return dict(status="identical", stops_advance=False, on_divergence=on_divergence,
                    comparison=cmp_, reason="AR 与 SpecDec 逐 token 相同")
    stop = (on_divergence != "record")
    return dict(status="different", stops_advance=stop, on_divergence=on_divergence,
                comparison=cmp_,
                reason=("分歧已记录，按 on_divergence=record **继续**；"
                        "不自动归因为浮点误差、不自动豁免（仍待评估）"
                        if not stop else
                        "分歧：按现行默认（stop）停止推进"))


FORMAL_GATE = dict(
    name="B6 exact-id（冻结正式判据）",
    sources=["docs/design_freeze_g4_specdec.md §8（B6：SpecDec ids == target-only greedy ids，exact）",
             "docs/g5_measurement_protocol.md §8（主 Gate：FP32 B6 逐 token 精确相等）"],
    requires=["prompt 家族覆盖（B6 的 10 类）× 两档 KV",
              "同 family/同 KV 的 AR 与 SpecDec token 序列 **exact 相等**",
              "全部决策输入 finite",
              "多次确定性一致",
              "结构不变量通过"],
    note="单次、单模式、单 family 的运行**结构上不可能**满足上述全部条件 "
         "⇒ 本工具下 `formal_gate_status` **永远不得**为 pass。",
)


def formal_gate_status(*, ar_rec=None, spec_rec=None, family_coverage=None,
                       determinism_ok=None) -> dict:
    """冻结正式判据的状态。**只可能是 not_established 或 fail**（结构性保证）。

    这里刻意**不**实现"如何通过"——只实现"在证据不足时不得报通过"。
    """
    missing = []
    if ar_rec is None:
        missing.append("AR 记录")
    if spec_rec is None:
        missing.append("SpecDec 记录")
    # 家族覆盖**永远**不足：B6 要求 10 类 prompt 家族 × 两档 KV，本工具每次只跑一个 family/一种 mode。
    # 因此这里无条件计入缺失（不是 `or True` 的笔误）——见上方 FORMAL_GATE["note"]。
    if family_coverage is None or int(family_coverage) < 10:
        missing.append("prompt 家族覆盖（B6 十类，本工具不产生）")
    if determinism_ok is not True:
        missing.append("多次确定性一致（本工具单次运行不产生）")
    hard_fail = (failure_class(ar_rec) not in ("none", "missing")
                 or failure_class(spec_rec) not in ("none", "missing"))
    if hard_fail:
        return dict(status="fail", gate=FORMAL_GATE["name"],
                    reason="存在硬失败 ⇒ 正式判据直接 fail", missing_evidence=missing,
                    never_pass_here=True)
    return dict(status="not_established", gate=FORMAL_GATE["name"],
                reason="证据不足：本工具只做单次/单模式/单 family 诊断，无法评估冻结正式判据",
                missing_evidence=missing, never_pass_here=True)


def summarize_run(rec: dict | None, *, ar_rec=None, spec_rec=None, rep_rec=None,
                  on_divergence: str = "stop") -> dict:
    """把五类状态一次性算出来（写进结果 JSON 用）。**不产生任何"总体 PASS"字段。**"""
    ex = execution_status(rec)
    return dict(
        execution_status=ex,
        structural_status=structural_status(rec),
        repeatability_status=repeatability_status(rec, rep_rec),
        ar_spec_equality=(ar_spec_equality(ar_rec, spec_rec, on_divergence=on_divergence)
                          if (ar_rec is not None or spec_rec is not None) else
                          dict(status="not_assessable", reason="本 run 未配对", stops_advance=False)),
        formal_gate_status=formal_gate_status(ar_rec=ar_rec, spec_rec=spec_rec),
        failure_class=failure_class(rec),
        note=DIAGNOSTIC_ONLY_NOTE,
        no_aggregate_pass_field=True,
    )
