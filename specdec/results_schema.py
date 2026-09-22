"""results_schema.py — G5 单一结果 schema（版本化、可审计、可向后兼容旧 Part 4 JSONL）。

为什么要有独立一层
  Part 4 的 JSONL 是**测量核心**产出的原始行（字段名偏实现、缺少 run 级元数据）。
  G5 正式结果需要**统一身份 + 统一可用性语义**：run 级元数据（git commit、seed、条件顺序、
  schema 版本）+ 逐字段可用性（缺失写 null 且 available=false，**不得用 0 冒充测量值**）。
  因此本模块只做"映射与校验"，不去改动已冻结的测量代码，也不改写任何历史文件。

三条硬约束（由 validate_row / write_g5_run 强制）
  1. 任何值为 None 的字段，必须在 `availability` 里显式为 false；
  2. 遥测字段（显存/温度/功耗/时钟）不得出现字符串；取不到只能是 null + available=false；
  3. 写文件前检查目标路径不存在，**拒绝覆盖**（唯一文件名 = schema + config hash + UTC + run_id）。

近贴边判定（Task E）的固定证据门槛见 `classify_bf16_compare()`：四个条件**全部**满足才允许
写 `BF16_NEAR_TIE`，否则必须是 `NEEDS_INSPECTION`（不允许自动豁免）。

用法（一般由 probe/sweep 调用）：
    row = build_g5_row(part4_row_or_summary, meta)
    write_g5_run(Path("output/g5/...jsonl"), row)
旧数据读取：read_legacy_part4(path) -> list[dict]（只读，不改写）
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

SCHEMA_VERSION = "g5-results-1.2"
# 版本历史（只读兼容层：旧版本行永远可读，**绝不原地改写**）
#   v1.0 -> v1.1：新增资源口径字段并区分 AR 主基线 / 诊断控制；
#   v1.1 -> v1.2（不兼容变更，2026-09-22）：
#     ① summary 行按 `*_med` 显式映射（此前 10 个统计字段全部为 null）；
#     ② 跨进程 AR↔SpecDec 等价性由**父进程**填写，并区分 available / certified；
#     ③ 状态分层：within_run_status / between_block_status / environment_status；
#     ④ canonical 自洽校验 validate_canonical()（写盘前强制）。
#   旧版本行仍可由 build_g5_row() 只读映射，缺失字段一律 null + availability=false（**不猜**）。
READ_VERSIONS = ("g5-results-1.0", "g5-results-1.1", "g5-results-1.2")

MODE_CHOICES = ("AR", "SpecDec")
KV_CHOICES = ("base_bf16", "v_fp8")           # 冻结的两种 KV 口径
CTX_CHOICES = (2048, 4096, 8192)              # 冻结的 12 条件 factorial 长度
AR_RESIDENCY_CHOICES = ("target_only", "dual_control")
PRIMARY_AR_RESIDENCY = "target_only"          # 正式主基线
DIAGNOSTIC_AR_RESIDENCY = "dual_control"      # 诊断控制（不计入 factorial）
PRIMARY_ROLE = "primary"                      # 正式 factorial 成员
DIAGNOSTIC_ROLE = "diagnostic_control"        # 仅诊断，永无声明资格
BETWEEN_BLOCK_REL_LIMIT = 20.0                # 预登记诊断线：两随机 block 中位数相对差

# ---------------------------------------------------------------- 字段定义
# name -> (kind, 说明)；kind ∈ {"str","int","float","bool","list","dict"}
FIELDS: dict[str, tuple[str, str]] = {
    "schema_version": ("str", "schema 版本"),
    "run_id": ("str", "本次聚合 run 的唯一 id（= RUN_ID，含 UTC）"),
    "block_index": ("int", "随机化 block 序号"),
    "condition_index": ("int", "该 block 内条件序号（随机化后的执行序）"),
    "timestamp_utc": ("str", "该 rep 结束时的 UTC 时间戳"),
    "git_commit": ("str", "git HEAD"),
    "dirty_worktree": ("bool", "运行时代码是否 dirty"),
    "prompt_family": ("str", "family id（探索阶段可为 null）"),
    "prompt_hash": ("str", "prompt 文本/ids 的稳定哈希"),
    "input_token_hash": ("str", "实际喂入 token ids 的哈希"),
    "generated_token_hash": ("str", "生成 token ids 的哈希"),
    "context_requested": ("int", "请求的 context 长度"),
    "context_actual": ("int", "实际 token 数"),
    "max_new_tokens": ("int", "生成长度上限"),
    "target_model": ("str", ""),
    "draft_model": ("str", ""),
    "compute_dtype": ("str", ""),
    "target_kv": ("str", ""),
    "draft_kv": ("str", ""),
    "mode": ("str", "AR | SpecDec"),
    "k": ("int", "每轮提议候选数"),
    "chunk_size": ("int", "分块 prefill 块大小"),
    "seed": ("int", "随机化 seed（None=未随机化）"),
    "rep_index": ("int", "该条件的第几次 measured rep"),
    "condition_order": ("list", "该 block 的条件执行顺序（label 列表）"),
    "warmup": ("int", "丢弃的 warm-up 次数"),
    "prefill_seconds": ("float", ""),
    "decode_seconds": ("float", ""),
    "total_seconds": ("float", ""),
    "prefill_source": ("str", "prefill_seconds 的来源：reported_median | derived（默认 null）"),
    "decode_tok_per_s": ("float", ""),
    "e2e_tok_per_s": ("float", ""),
    "n_generated": ("int", ""),
    "acceptance": ("float", "**恒等式** accepted/proposed（AR 为 null，不猜）"),
    "acceptance_reported": ("float", "子进程汇报的 accept_rate 中位数（仅供审计，不参与恒等式）"),
    "accepted": ("int", ""),
    "proposed": ("int", ""),
    "rounds": ("int", ""),
    "draft_calls": ("int", ""),
    "target_block_calls": ("int", ""),
    "mismatch_rounds": ("int", ""),
    "all_accept_rounds": ("int", ""),
    "eos_id": ("int", "EOS token id（跨进程等价性认证前提之一）"),
    "accepted_length_histogram": ("dict", "{m: count}"),
    "peak_allocated_gib": ("float", "主显存指标（PyTorch 实际张量分配）"),
    "peak_reserved_gib": ("float", "分配器池（含空闲块/虚拟预留，≠物理占用）"),
    "nvidia_smi_used_mib": ("float", "驱动视角物理占用（辅助）"),
    "temperature_c": ("float", ""),
    "power_w": ("float", ""),
    "sm_clock_mhz": ("float", ""),
    "memory_clock_mhz": ("float", ""),
    "tokens_finite": ("bool", "prefill/AR 步 logits finite；SpecDec 验证行未扫描（见 notes）"),
    "ar_equal_available": ("bool", "是否有 AR 对照可比较"),
    "ar_equal": ("bool", "生成序列是否与同档 AR 完全相同（无对照时 null）"),
    "ar_equal_certified": ("bool", "认证前提（ctx/kv/input_hash/max_new/eos/n_gen/ids）是否全部满足"),
    "equality_source": ("str", "等价性判定来源：parent_cross_process | in_process | reference_row"),
    "equality_reason": ("str", "为何为 null / 为何未认证的**具体**原因"),
    "first_diff_index": ("int", "首个不同 token 位置（不等时）"),
    "status": ("str", "[兼容字段] 执行失败时=OOM/ERROR，否则=measurement_status"),
    "execution_status": ("str", "SUCCESS | ERROR | OOM（程序是否跑完）"),
    "measurement_status": ("str", "OK | CHECK | ENV_INVALID | NOT_MEASURED（测量是否有效）"),
    "within_run_status": ("str", "OK | CHECK | NOT_MEASURED：**仅**同一个进程内 reps 的稳定性"),
    "within_run_check_reason": ("str", "**仅**within-run 被判 CHECK 的原因（来自子进程 spread_flag）"),
    "within_run_rel_spread_pct": ("float", "子进程内的相对离散（decode/t_decode 取大者）"),
    "between_block_status": ("str", "OK | CHECK | None：两个随机顺序 block 之间的一致性"),
    "between_block_check_reason": ("str", "**仅**between-block 被判 CHECK 的原因"),
    "between_block_rel_diff_pct": ("float", "同 cell 两 block 中位数相对差"),
    "between_block_check": ("bool", "between-block 是否超过预登记诊断线"),
    "environment_status": ("str", "OK | ENV_INVALID | None：P-state / 供电 / 环境资格"),
    "reps_measured": ("int", "实际用于统计的 measured reps 数"),
    "reps_expected": ("int", "协议要求的 measured reps 数"),
    "active_sm_clock_mhz": ("float", "**负载期间**采样的 SM 时钟中位数"),
    "active_memory_clock_mhz": ("float", "负载期间显存时钟中位数"),
    "active_power_w": ("float", "负载期间功耗中位数"),
    "pstate_qualified": ("bool", "该 cell 是否处于合格 active P-state"),
    "pstate_reason": ("str", "P-state 判定原因（含实测时钟/阈值）"),
    "env_qualified": ("bool", "整批/该 block 的环境资格（预检 + 全 cell P-state）"),
    "speedup_claim_blockers": ("list", "不允许 speedup 声明的具体原因列表"),
    # ---- 资源口径（v1.1 新增）：谁是主要条件、谁占了多少资源、谁配当分母 ----
    "resident_models": ("list", "实际驻留的模型，如 ['target'] 或 ['target','draft']"),
    "ar_residency": ("str", "AR 的驻留模式：target_only | dual_control（SpecDec 为 null）"),
    "condition_role": ("str", "primary | diagnostic_control"),
    "formal_factorial_member": ("bool", "是否属于正式 3x2x2 factorial；诊断控制为 false"),
    "speedup_denominator_eligible": ("bool", "能否作为正式 speedup 分母；仅 AR target_only 为 true"),
    "speedup_claim_allowed": ("bool", "该 cell 是否允许产生**确定的**加速声明（不稳定 cell 为 false）"),
    "profile": ("dict", "profiling 分解（默认 null；profiling 数据不作为正式吞吐）"),
    "availability": ("dict", "field -> bool；值为 null 的字段必须显式为 false"),
    "notes": ("str", ""),
}

TELEMETRY_FIELDS = ("nvidia_smi_used_mib", "temperature_c", "power_w", "sm_clock_mhz",
                    "memory_clock_mhz", "peak_allocated_gib", "peak_reserved_gib")
NUMERIC_FIELDS = tuple(k for k, (kind, _) in FIELDS.items() if kind in ("int", "float"))
# `availability` 是**元字段**，不参与自身的一致性检查（否则会往 availability 里写 availability）
NULLABLE_FIELDS = tuple(k for k in FIELDS if k != "availability")


def resolve_condition_role(mode: str, ar_residency: str | None) -> dict:
    """**显式枚举**决定驻留与角色（不做字符串包含判断）。非法组合直接抛 ValueError。

    正式主基线：`AR + target_only`（只驻留 target；真实部署口径；正式 speedup 分母）
    诊断控制  ：`AR + dual_control`（draft 仅常驻、不参与 prefill/decode；**不入 factorial**）
    SpecDec   ：必须 target+draft 双驻留（算法自身资源需求）；`ar_residency` 记为 null
    """
    if mode not in MODE_CHOICES:
        raise ValueError(f"非法 mode: {mode!r}（可选 {MODE_CHOICES}）")
    if ar_residency is not None and ar_residency not in AR_RESIDENCY_CHOICES:
        raise ValueError(f"非法 ar_residency: {ar_residency!r}（可选 {AR_RESIDENCY_CHOICES}）")
    if mode == "AR":
        if ar_residency == PRIMARY_AR_RESIDENCY:
            return dict(ar_residency=PRIMARY_AR_RESIDENCY, resident_models=["target"],
                        condition_role="primary", formal_factorial_member=True,
                        speedup_denominator_eligible=True)
        return dict(ar_residency=DIAGNOSTIC_AR_RESIDENCY,
                    resident_models=["target", "draft"],
                    condition_role="diagnostic_control", formal_factorial_member=False,
                    speedup_denominator_eligible=False)
    return dict(ar_residency=None, resident_models=["target", "draft"],
                condition_role="primary", formal_factorial_member=True,
                speedup_denominator_eligible=False)


def build_primary_matrix(ctxs, kvs, modes=MODE_CHOICES,
                         ar_residency: str = PRIMARY_AR_RESIDENCY) -> list[tuple]:
    """正式 primary factorial cells：(ctx, kv, mode)，其中 AR 一律 target-only。"""
    return [(c, kv, m) for c in ctxs for kv in kvs for m in modes]


def build_dual_controls(ctxs, kvs) -> list[tuple]:
    """可选诊断控制 cells：AR + dual_control（**不计入 primary factorial**）。"""
    return [(c, kv, "AR") for c in ctxs for kv in kvs]


def compute_speedup(ar_row: dict, spec_row: dict, *, rel_limit: float = 20.0) -> dict:
    """统一方向：speedup = T_AR / T_Spec（>1 表示 SpecDec 更快，<1 表示更慢）。

    强制：AR 行必须是 **target_only primary**（否则抛错，防止诊断控制混入分母）；
    `speedup_claim_allowed=False` 时不产出确定性结论（只给数字与 CHECK 标注）。
    """
    if ar_row.get("mode") != "AR":
        raise ValueError("speedup 分母必须是 AR 行")
    if ar_row.get("ar_residency") != PRIMARY_AR_RESIDENCY or \
            not ar_row.get("speedup_denominator_eligible"):
        raise ValueError("speedup 分母必须是 AR + target_only（primary）；"
                         "dual_control 不得作为分母")
    if spec_row.get("mode") != "SpecDec":
        raise ValueError("speedup 分子必须是 SpecDec 行")
    if (ar_row.get("context_actual"), ar_row.get("target_kv")) != \
            (spec_row.get("context_actual"), spec_row.get("target_kv")):
        raise ValueError("speedup 只能同 context、同 KV 比较")
    out = dict(context_actual=ar_row.get("context_actual"), target_kv=ar_row.get("target_kv"))
    for field, key in (("total_seconds", "e2e"), ("decode_seconds", "decode")):
        t_ar, t_spec = ar_row.get(field), spec_row.get(field)
        if isinstance(t_ar, (int, float)) and isinstance(t_spec, (int, float)) and t_spec:
            out[f"speedup_{key}"] = t_ar / t_spec
        else:
            out[f"speedup_{key}"] = None
    claim = speedup_claim(ar_row, spec_row, rel_limit=rel_limit)
    out["speedup_claim_allowed"] = claim["allowed"]
    out["speedup_claim_blockers"] = claim["reasons"]
    return out


def residency_penalty(ar_primary: dict, ar_dual: dict) -> dict:
    """`penalty = T_AR_target_only / T_AR_dual_control`。

    **方向语义（写死）**：`penalty < 1` ⇒ 双驻留下 AR 更慢（即"额外模型驻留本身就有代价"）；
    等价地 `slowdown = T_dual / T_target = 1 / penalty > 1` 表示拖慢倍数。
    """
    if ar_primary.get("ar_residency") != PRIMARY_AR_RESIDENCY:
        raise ValueError("第一个参数必须是 AR target_only")
    if ar_dual.get("ar_residency") != DIAGNOSTIC_AR_RESIDENCY:
        raise ValueError("第二个参数必须是 AR dual_control")
    out = dict(context_actual=ar_primary.get("context_actual"),
               target_kv=ar_primary.get("target_kv"))
    for field, key in (("total_seconds", "e2e"), ("decode_seconds", "decode")):
        a, b = ar_primary.get(field), ar_dual.get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b and a:
            out[f"penalty_{key}"] = a / b
            out[f"slowdown_{key}"] = b / a
        else:
            out[f"penalty_{key}"] = None
            out[f"slowdown_{key}"] = None
    out["interpretation"] = "penalty<1（slowdown>1）⇒ 双驻留本身拖慢了 AR"
    return out


def plan_process(conditions, ar_residency: str | None = None) -> dict:
    """给定"一个进程里要跑的条件列表"，决定该进程需要驻留哪些模型，并做硬约束检查。

    `conditions`: [(ctx, kv, mode), ...]；`ar_residency`: AR 条件用的驻留模式。

    **硬约束**：`AR + target_only`（正式主基线）要求该进程**从未加载 draft**，因此不能与 SpecDec
    条件同进程（否则 AR 实际上是双驻留）。违反直接抛 ValueError。
    """
    if ar_residency is not None and ar_residency not in AR_RESIDENCY_CHOICES:
        raise ValueError(f"非法 ar_residency: {ar_residency!r}（可选 {AR_RESIDENCY_CHOICES}）")
    modes = {m for _, _, m in conditions}
    for m in modes:
        if m not in MODE_CHOICES:
            raise ValueError(f"非法 mode: {m!r}（可选 {MODE_CHOICES}）")
    if ar_residency == PRIMARY_AR_RESIDENCY and "AR" in modes and "SpecDec" in modes:
        raise ValueError("AR target_only 不能与 SpecDec 同进程（draft 一旦加载就不再是 "
                         "target-only）：请按 mode 过滤或用逐条件独立进程")
    need_draft = ("SpecDec" in modes) or \
                 (ar_residency == DIAGNOSTIC_AR_RESIDENCY and "AR" in modes)
    return dict(load_draft=need_draft,
                resident_models=["target"] + (["draft"] if need_draft else []),
                ar_residency=ar_residency, n_conditions=len(conditions))


EXECUTION_STATUSES = ("SUCCESS", "ERROR", "OOM")
MEASUREMENT_STATUSES = ("OK", "CHECK", "ENV_INVALID", "NOT_MEASURED")


def derive_statuses(*, returncode: int | None, oom: bool, spread_flag: str | None,
                    pstate_qualified: bool | None = None) -> dict:
    """把"程序跑完了吗"与"测量有效吗"**彻底分开**，并把测量问题**分层**。

    分层（v1.2，回答"到底是哪一层坏了"）：
      - `within_run_status`：**同一个进程内** reps 之间的稳定性（子进程 spread_flag）
      - `environment_status`：P-state / 供电 / 环境资格
      - `between_block_status`：两个随机顺序 block 之间的一致性（需父进程在全部 block 后填）
      - `measurement_status`：三层的**最坏值**（ENV_INVALID > CHECK > OK）

    **关键**：`returncode == 0` 只说明程序成功完成，**不代表**性能稳定 —— 子进程的
    `spread_flag='CHECK:relative-spread>20%'` 必须把 measurement_status 降为 CHECK。
    `within_run_check_reason` **只**保存 within-run 的原因；between-block 的原因另存
    `between_block_check_reason`，绝不拼接进同一个字段。
    """
    if oom or (returncode is not None and returncode not in (0,)):
        exec_status = "OOM" if oom else "ERROR"
    elif returncode is None:
        exec_status = "ERROR"
    else:
        exec_status = "SUCCESS"
    if exec_status != "SUCCESS":
        within, env = "NOT_MEASURED", None
        reason = f"执行未成功（{exec_status}）"
    else:
        within = "CHECK" if (spread_flag and "CHECK" in str(spread_flag)) else "OK"
        reason = str(spread_flag) if within == "CHECK" else None
        env = "ENV_INVALID" if pstate_qualified is False else "OK"
    if exec_status != "SUCCESS":
        meas = "NOT_MEASURED"
    elif env == "ENV_INVALID":
        meas = "ENV_INVALID"
    elif within == "CHECK":
        meas = "CHECK"
    else:
        meas = "OK"
    return dict(execution_status=exec_status, measurement_status=meas,
                within_run_status=within, environment_status=env,
                status=(meas if exec_status == "SUCCESS" else exec_status),
                within_run_check_reason=reason)


# ---------------------------------------------------------------- 跨进程等价性（v1.2）
# 配对 key：**必须是同一 block 内、同 context、同 KV、同输入 token、同生成长度上限**。
# 只比较 `AR target_only primary` 与 `SpecDec primary`；**dual-control 绝不作为正确性参照**。
EQUALITY_KEY_FIELDS = ("block_index", "context_actual", "target_kv",
                       "input_token_hash", "max_new_tokens")


def equality_key(row: dict) -> tuple:
    return tuple(row.get(f) for f in EQUALITY_KEY_FIELDS)


def refresh_inplace(row: dict) -> dict:
    """原地重建 row 的内容（保持对象身份不变）。

    为什么需要：`finalize_row` 返回的是**副本**。父进程在别处还持有该 row 的引用
    （如 `pending[i]['row']`），如果只替换列表元素，持有者看到的仍是旧 availability，
    于是"内存中一致、写盘时才一致"这种半吊子状态会反复咬人。这里直接原地更新。
    """
    fresh = finalize_row(row)
    row.clear()
    row.update(fresh)
    return row


def cross_process_equality(ar_row: dict | None, spec_row: dict | None) -> dict:
    """父进程填写的 AR↔SpecDec 生成序列等价性判定。

    三条纪律：
      ① 缺对照**绝不返回 False**：`ar_equal=None, ar_equal_available=False`（"没比过" ≠ "不相等"）；
      ② 认证前提必须齐全（ctx / KV / input_token_hash / max_new / EOS / n_generated /
         generated_token_ids 或 hash），缺一项就 `ar_equal_certified=False` 并写明原因；
      ③ 判定用 `generated_token_ids` 逐位相同（有 ids 时必须用 ids，不降级为 hash 比较）。
    """
    out = {"ar_equal": None, "ar_equal_available": False, "ar_equal_certified": False,
           "equality_source": "parent_cross_process", "equality_reason": None,
           "first_diff_index": None}
    if ar_row is None or spec_row is None:
        out["equality_source"] = None
        out["equality_reason"] = "缺少 AR target-only primary 对照，本 block 无法判定等价性"
        return out
    checks = {
        "context_equal": (ar_row.get("context_actual") is not None
                          and ar_row.get("context_actual") == spec_row.get("context_actual")),
        "kv_equal": (ar_row.get("target_kv") is not None
                     and ar_row.get("target_kv") == spec_row.get("target_kv")),
        "max_new_equal": (ar_row.get("max_new_tokens") is not None
                          and ar_row.get("max_new_tokens") == spec_row.get("max_new_tokens")),
        "eos_equal": (ar_row.get("eos_id") is not None
                      and ar_row.get("eos_id") == spec_row.get("eos_id")),
        "input_token_hash_equal": (ar_row.get("input_token_hash") is not None
                                   and ar_row.get("input_token_hash")
                                   == spec_row.get("input_token_hash")),
        "n_generated_recorded": (ar_row.get("n_generated") is not None
                                 and spec_row.get("n_generated") is not None),
    }
    a_ids, s_ids = ar_row.get("generated_token_ids"), spec_row.get("generated_token_ids")
    a_h = ar_row.get("generated_token_hash")
    s_h = spec_row.get("generated_token_hash")
    checks["token_evidence_recorded"] = (a_ids is not None and s_ids is not None) or \
                                        (a_h is not None and s_h is not None)
    missing = [k for k, v in checks.items() if not v]
    if missing:
        out["equality_reason"] = "认证前提不满足：" + ", ".join(missing)
        return out
    out["ar_equal_available"] = True
    out["ar_equal_certified"] = True
    if a_ids is not None and s_ids is not None:
        out["ar_equal"] = [int(x) for x in a_ids] == [int(x) for x in s_ids]
        out["equality_reason"] = "逐 token id 比较"
        if out["ar_equal"] is False:
            for i, (x, y) in enumerate(zip(a_ids, s_ids)):
                if x != y:
                    out["first_diff_index"] = i
                    break
            else:
                out["first_diff_index"] = min(len(a_ids), len(s_ids))
    else:                                  # 只有 hash：可判定但降级标注
        out["ar_equal"] = (a_h == s_h)
        out["equality_reason"] = "仅有 hash 证据（无 generated_token_ids），降级为 hash 比较"
    return out


def fill_cross_process_equality(rows: list[dict]) -> dict:
    """在**父进程**里为一批 canonical 行填写跨进程等价性（原地修改并返回汇总）。

    对每个 `SpecDec primary` 行，找**同 block + 同 key** 的 `AR target_only primary` 行做对照；
    AR 参照行自身标 `equality_source='reference_row'`（它不与自己比较）。
    找不到对照时写 `ar_equal=None / available=False`，**绝不写 False**。
    """
    summary = {"pairs_compared": 0, "pairs_missing": 0, "detail": []}
    ar_index: dict = {}
    for r in rows:
        if r.get("mode") == "AR" and r.get("ar_residency") == PRIMARY_AR_RESIDENCY \
                and r.get("condition_role") == PRIMARY_ROLE:
            ar_index[equality_key(r)] = r
    for r in rows:
        # 每次原地改写后都立刻刷新：availability 必须**马上**与值一致，
        # 否则内存中的行会短暂处于"值有 but availability=false"的自相矛盾状态。
        if r.get("mode") == "AR" and r.get("ar_residency") == PRIMARY_AR_RESIDENCY:
            r["equality_source"] = "reference_row"
            r["equality_reason"] = "AR 参照行本身不参与相等性判定（它是被比较的一方）"
            refresh_inplace(r)
            continue
        if r.get("mode") != "SpecDec" or r.get("condition_role") != PRIMARY_ROLE:
            continue
        ref = ar_index.get(equality_key(r))
        c = cross_process_equality(ref, r)
        for k, v in c.items():
            r[k] = v
        refresh_inplace(r)
        if ref is None:
            summary["pairs_missing"] += 1
        else:
            summary["pairs_compared"] += 1
        summary["detail"].append(dict(key=equality_key(r), ar_equal=c["ar_equal"],
                                      available=c["ar_equal_available"],
                                      certified=c["ar_equal_certified"],
                                      reason=c["equality_reason"]))
    return summary


def speedup_claim(ar: dict, spec: dict, *, rel_limit: float = 20.0) -> dict:
    """**正式 speedup 声明资格**（10 条全满足才允许）。返回 allowed + 具体拒绝原因。

    **措辞纪律（v1.2）**：within-run 与 between-block 的原因**分开报**。
    一个 within-run 明明 OK、只是两个 block 不一致的 cell，
    只能说 `between-block CHECK`，**不能**说成 `within-run CHECK`。
    """
    r: list[str] = []
    if ar.get("mode") != "AR" or spec.get("mode") != "SpecDec":
        r.append("NO CLAIM：分子/分母模式不对（需 AR ÷ SpecDec）")
    if ar.get("ar_residency") != PRIMARY_AR_RESIDENCY:
        r.append("NO CLAIM：AR 不是 target_only（正式主基线口径）")
    if not ar.get("speedup_denominator_eligible"):
        r.append("NO CLAIM：AR 无分母资格")
    if DIAGNOSTIC_ROLE in (ar.get("condition_role"), spec.get("condition_role")):
        r.append("NO CLAIM：dual-resident 诊断控制不得参与正式 speedup")
    for fld, name in (("context_actual", "context"), ("target_kv", "KV"),
                      ("max_new_tokens", "max_new"), ("input_token_hash", "input_token_hash")):
        a, s = ar.get(fld), spec.get(fld)
        if a is None or s is None:
            r.append(f"NO CLAIM：{name} 缺失，可比性无法确认")
        elif a != s:
            r.append(f"NO CLAIM：{name} 不一致（AR={a!r} / Spec={s!r}）")
    for row, who in ((ar, "AR"), (spec, "SpecDec")):
        if row.get("reps_expected") and row.get("reps_measured") is not None \
                and row["reps_measured"] < row["reps_expected"]:
            r.append(f"NO CLAIM：{who} measured reps 不完整"
                     f"（{row['reps_measured']}/{row['reps_expected']}）")
        # ---- 环境层 ----
        if row.get("environment_status") == "ENV_INVALID" \
                or row.get("measurement_status") == "ENV_INVALID":
            r.append(f"NO CLAIM：{who} ENV_INVALID（P-state/环境不合格）")
        # ---- within-run 层（只看 within_run_status，不看被 between-block 拉低后的总状态）----
        wrs = row.get("within_run_status")
        if wrs is None and row.get("within_run_check_reason"):
            wrs = "CHECK"
        if wrs == "CHECK":
            r.append(f"NO CLAIM：{who} within-run CHECK"
                     f"（{row.get('within_run_check_reason') or '未记录原因'}）")
        elif wrs == "NOT_MEASURED":
            r.append(f"NO CLAIM：{who} 执行未成功，未获得测量")
        # ---- between-block 层 ----
        bb = row.get("between_block_check")
        if bb is None and row.get("between_block_status") == "CHECK":
            bb = True
        if bb:
            rel = row.get("between_block_rel_diff_pct")
            r.append(f"NO CLAIM：{who} between-block CHECK"
                     f"（{rel:.1f}% > {rel_limit:g}%）" if isinstance(rel, (int, float))
                     else f"NO CLAIM：{who} between-block CHECK")
        if row.get("execution_status") in ("ERROR", "OOM"):
            r.append(f"NO CLAIM：{who} 执行 {row['execution_status']}")
        for fld in ("decode_seconds", "total_seconds"):
            v = row.get(fld)
            if isinstance(v, float) and v != v:
                r.append(f"NO CLAIM：{who} {fld} 为 NaN")
        if row.get("tokens_finite") is not True:
            r.append(f"NO CLAIM：{who} 正确性状态未记录或非 finite")
        if row.get("profile"):
            r.append(f"NO CLAIM：{who} 行内混入 profiling 数据")
    return dict(allowed=(not r), reasons=r, rel_limit_pct=rel_limit)


def speedup_claim_over_blocks(ar_rows: list[dict], spec_rows: list[dict], *,
                              pooled_seconds: dict | None = None,
                              rel_limit: float = 20.0) -> dict:
    """按 block 配对求 speedup，并把声明资格收敛到"**每个 block 都必须合格**"。

    `pooled_seconds` 可选：`{"ar_decode":…, "spec_decode":…, "ar_total":…, "spec_total":…}`
    由调用方用**全部 rep 合并中位数**算好（canonical 里只有逐 block 行，没有 rep 级秒数）。
    """
    out = {"per_block": [], "pooled_decode": None, "pooled_e2e": None,
           "allowed": False, "blockers": [], "blocks_paired": 0}
    ar_by_b = {r.get("block_index"): r for r in ar_rows}
    sp_by_b = {r.get("block_index"): r for r in spec_rows}
    blockers: list[str] = []
    for b in sorted(set(ar_by_b) & set(sp_by_b)):
        try:
            su = compute_speedup(ar_by_b[b], sp_by_b[b], rel_limit=rel_limit)
        except ValueError as e:                                    # noqa: PERF203
            out["per_block"].append(dict(block_index=b, error=str(e)))
            blockers.append(f"NO CLAIM：block{b} 无法比较（{e}）")
            continue
        out["blocks_paired"] += 1
        out["per_block"].append(dict(block_index=b, speedup_decode=su["speedup_decode"],
                                     speedup_e2e=su["speedup_e2e"],
                                     allowed=su["speedup_claim_allowed"]))
        blockers.extend(f"block{b}: {x}" for x in su["speedup_claim_blockers"])
    if pooled_seconds:
        ad, sd = pooled_seconds.get("ar_decode"), pooled_seconds.get("spec_decode")
        at, st = pooled_seconds.get("ar_total"), pooled_seconds.get("spec_total")
        if isinstance(ad, (int, float)) and isinstance(sd, (int, float)) and sd:
            out["pooled_decode"] = ad / sd
        if isinstance(at, (int, float)) and isinstance(st, (int, float)) and st:
            out["pooled_e2e"] = at / st
    out["blockers"] = list(dict.fromkeys(blockers))       # 去重但保持首次出现顺序
    out["allowed"] = bool(out["blocks_paired"] and not out["blockers"])
    return out


def finalize_row(row: dict) -> dict:
    """补齐 schema 字段，并**由值重算** availability（field -> bool）。

    availability 的语义被写死为"**该字段是否有值**"，因此它必须是值的纯函数：
    值非 null ⇒ true，值为 null ⇒ false。旧版用 `setdefault` 保留调用方传入的 availability，
    于是"先在值为 null 时 finalize、之后父进程再填值"就会留下
    `值有 but availability=false` 的自相矛盾行（跨进程等价性字段正好踩到这个坑）。
    """
    out = dict(row)
    for name in FIELDS:
        out.setdefault(name, None)
    out["schema_version"] = row.get("schema_version") or SCHEMA_VERSION
    avail = {name: (out.get(name) is not None) for name in NULLABLE_FIELDS}
    avail["availability"] = True
    out["availability"] = avail
    return out


def validate_row(row: dict) -> list[str]:
    """返回错误列表（空 = 通过）。只做机械可判的检查，不猜业务含义。"""
    errs: list[str] = []
    if row.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version 必须是 {SCHEMA_VERSION}，实际 {row.get('schema_version')!r}")
    for name in FIELDS:
        if name not in row:
            errs.append(f"缺字段 {name}")
    avail = row.get("availability")
    if not isinstance(avail, dict):
        errs.append("availability 必须是 dict")
        return errs
    if avail.get("availability") is not True:
        errs.append("availability['availability'] 必须为 True（元字段自指）")
    for name in NULLABLE_FIELDS:
        v = row.get(name)
        if v is None and avail.get(name) is not False:
            errs.append(f"{name} 为 null 但 availability 未显式置 false")
        if v is not None and avail.get(name) is False:
            errs.append(f"{name} 有值但 availability 为 false（自相矛盾）")
    for name in TELEMETRY_FIELDS:
        v = row.get(name)
        if v is not None and not isinstance(v, (int, float)):
            errs.append(f"遥测字段 {name} 非数值（不得用字符串/0 冒充）：{v!r}")
        if isinstance(v, float) and math.isnan(v):
            errs.append(f"遥测字段 {name} 是 NaN；取不到必须写 null + available=false")
    for name in NUMERIC_FIELDS:
        v = row.get(name)
        if v is not None and isinstance(v, bool):
            errs.append(f"{name} 是 bool，疑似误填")
    for name, (kind, _) in FIELDS.items():
        v = row.get(name)
        if v is None:
            continue
        ok = (kind == "str" and isinstance(v, str)) or \
             (kind == "int" and isinstance(v, int) and not isinstance(v, bool)) or \
             (kind == "float" and isinstance(v, (int, float)) and not isinstance(v, bool)) or \
             (kind == "bool" and isinstance(v, bool)) or \
             (kind == "list" and isinstance(v, list)) or \
             (kind == "dict" and isinstance(v, dict))
        if not ok:
            errs.append(f"{name} 类型应为 {kind}，实际 {type(v).__name__}")
    # ---- 资源口径一致性（v1.1）：矛盾组合必须被拒绝 ----
    mode = row.get("mode")
    res = row.get("ar_residency")
    rm = row.get("resident_models")
    if mode in MODE_CHOICES and res is not None and isinstance(rm, list):
        try:
            want = resolve_condition_role(mode, res)
        except ValueError as e:  # noqa: PERF203
            errs.append(f"mode/residency 组合非法: {e}")
        else:
            if rm != want["resident_models"]:
                errs.append(f"resident_models {rm} 与 mode={mode}/residency={res} 矛盾"
                            f"（应为 {want['resident_models']}）")
            for field in ("condition_role", "formal_factorial_member",
                          "speedup_denominator_eligible"):
                v = row.get(field)
                if v is not None and v != want[field]:
                    errs.append(f"{field}={v!r} 与 mode={mode}/residency={res} 矛盾"
                                f"（应为 {want[field]!r}）")
    if mode == "AR" and res == DIAGNOSTIC_AR_RESIDENCY:
        if row.get("formal_factorial_member") is True:
            errs.append("dual_control 不得标 formal_factorial_member=true")
        if row.get("speedup_denominator_eligible") is True:
            errs.append("dual_control 不得作为 speedup 分母")
    if mode == "AR" and res == PRIMARY_AR_RESIDENCY and isinstance(rm, list) and "draft" in rm:
        errs.append("AR target_only 的 resident_models 不得包含 draft")
    if mode == "SpecDec" and res is not None:
        errs.append("SpecDec 的 ar_residency 必须为 null")
    if row.get("status") not in (None, "OK") and row.get("speedup_claim_allowed") is True:
        errs.append(f"status={row.get('status')} 的 cell 不得 speedup_claim_allowed=true")
    # ---- 状态分层 / 环境资格 / 加速声明资格（v1.1 第二轮）----
    ex, ms = row.get("execution_status"), row.get("measurement_status")
    if ex is not None and ex not in EXECUTION_STATUSES:
        errs.append(f"execution_status 非法: {ex!r}（可选 {EXECUTION_STATUSES}）")
    if ms is not None and ms not in MEASUREMENT_STATUSES:
        errs.append(f"measurement_status 非法: {ms!r}（可选 {MEASUREMENT_STATUSES}）")
    if ex is not None and ms is not None:
        want = ms if ex == "SUCCESS" else ex
        if row.get("status") != want:
            errs.append(f"status={row.get('status')!r} 与 execution_status={ex}/"
                        f"measurement_status={ms} 不一致（应为 {want!r}）")
        if ex != "SUCCESS" and ms != "NOT_MEASURED":
            errs.append(f"执行未成功时 measurement_status 必须为 NOT_MEASURED，实际 {ms!r}")
    if row.get("pstate_qualified") is False and ms == "OK":
        errs.append("pstate_qualified=false 时 measurement_status 不得为 OK")
    if row.get("speedup_claim_allowed") is True:
        if ms != "OK":
            errs.append(f"measurement_status={ms} 不得 speedup_claim_allowed=true")
        if row.get("between_block_check") is True:
            errs.append("between_block CHECK 不得 speedup_claim_allowed=true")
        if row.get("env_qualified") is False:
            errs.append("env_qualified=false 不得 speedup_claim_allowed=true")
    blockers = row.get("speedup_claim_blockers")
    if isinstance(blockers, list) and blockers and row.get("speedup_claim_allowed") is True:
        errs.append("存在 speedup_claim_blockers 时不得 speedup_claim_allowed=true")
    rp, re_ = row.get("reps_measured"), row.get("reps_expected")
    if isinstance(rp, int) and isinstance(re_, int) and rp > re_:
        errs.append(f"reps_measured({rp}) > reps_expected({re_})")
    # ---- dual-control 的声明纪律（v1.2）----
    if row.get("condition_role") == DIAGNOSTIC_ROLE:
        if row.get("speedup_claim_allowed") is True:
            errs.append("diagnostic_control 永远不得 speedup_claim_allowed=true")
        if row.get("speedup_denominator_eligible") is True:
            errs.append("diagnostic_control 不得 speedup_denominator_eligible=true")
        if row.get("formal_factorial_member") is True:
            errs.append("diagnostic_control 不得 formal_factorial_member=true")
    return errs


def validate_canonical(row: dict, *, ref_row: dict | None = None) -> list[str]:
    """**canonical 自洽校验**（写盘前强制）：抓自相矛盾的行，而不是等分析阶段才发现。

    覆盖需求里的 12 条：summary 完整性 / SpecDec 统计完整性 / accepted-proposed-acceptance 恒等式 /
    hash 与 ids 一致 / ar_equal 与 hash 一致 / 认证前提齐全 / dual-control 无声明资格 /
    CHECK 必须有结构化原因 / within-run OK 时原因只能指向 between-block / availability 双向一致。
    """
    errs: list[str] = []
    ms = row.get("measurement_status")
    measured = ms not in (None, "NOT_MEASURED")
    # 1. 成功测得的行必须有三段时间
    if measured:
        for fld in ("prefill_seconds", "decode_seconds", "total_seconds"):
            if row.get(fld) is None:
                errs.append(f"已测得但缺 {fld}（summary 必须映射 *_med；不得用 total-decode 偷算）")
    # 2/3. SpecDec 的投机统计与恒等式
    if row.get("mode") == "SpecDec" and measured:
        for fld in ("acceptance", "accepted", "proposed", "rounds"):
            if row.get(fld) is None:
                errs.append(f"SpecDec 已测得但缺 {fld}（summary 统计未映射）")
        acc, ac, pr = row.get("acceptance"), row.get("accepted"), row.get("proposed")
        if isinstance(ac, int) and isinstance(pr, int) and pr > 0 and isinstance(acc, (int, float)):
            if abs(acc - ac / pr) > 1e-9:
                errs.append(f"acceptance({acc}) != accepted/proposed({ac}/{pr})")
        ng = row.get("n_generated")
        mm = row.get("mismatch_rounds")
        if isinstance(ng, int) and isinstance(ac, int) and isinstance(mm, int) \
                and ac + mm != ng:
            errs.append(f"协议恒等式违反：accepted({ac}) + mismatch_rounds({mm})"
                        f" != n_generated({ng})")
    if row.get("mode") == "AR" and measured:
        for fld in ("acceptance", "accepted", "proposed", "rounds"):
            if row.get(fld) is not None:
                errs.append(f"AR 行不得携带 SpecDec 专属统计量 {fld}（语义不存在，应为 null）")
    # 4. hash 与 ids 一致
    ids, h = row.get("generated_token_ids"), row.get("generated_token_hash")
    if ids is not None and h is not None:
        want = token_ids_hash(ids)
        if want != h:
            errs.append(f"generated_token_hash({h}) 与 generated_token_ids 不一致"
                        f"（按 ids 应为 {want}）")
    # 5/6/7. ar_equal 与双方 hash、认证前提
    ae, avail, cert = row.get("ar_equal"), row.get("ar_equal_available"), row.get("ar_equal_certified")
    if cert is True and avail is not True:
        errs.append("ar_equal_certified=true 但 ar_equal_available 不为 true")
    if cert is True:
        for fld in ("input_token_hash", "max_new_tokens", "n_generated", "equality_source"):
            if row.get(fld) is None:
                errs.append(f"ar_equal_certified=true 但缺认证前提字段 {fld}")
        if row.get("generated_token_hash") is None and row.get("generated_token_ids") is None:
            errs.append("ar_equal_certified=true 但无 generated_token_hash/ids 证据")
    if ae is True and cert is True and row.get("mode") == "AR" \
            and row.get("equality_source") == "parent_cross_process":
        errs.append("AR 参照行不应被标为跨进程比较结果")
    # 5/6. 有对照行时：ar_equal 必须与双方 hash 一致（**这条能直接抓住"两个相同 hash 却写 False"**）
    if ref_row is not None:
        a_h = ref_row.get("generated_token_hash")
        s_h = row.get("generated_token_hash")
        if a_h is not None and s_h is not None:
            if ae is True and a_h != s_h:
                errs.append(f"ar_equal=true 但两侧 hash 不同（AR={a_h} / 本行={s_h}）")
            if ae is False and a_h == s_h:
                errs.append(f"ar_equal=false 但两侧 hash 相同（{a_h}）—— 判定与证据矛盾")
    # 8. dual-control
    if row.get("condition_role") == DIAGNOSTIC_ROLE and row.get("speedup_claim_allowed") is True:
        errs.append("dual-control 行 speedup_claim_allowed 必须为 false")
    # 9. CHECK 必须有结构化原因
    if ms == "CHECK":
        if not (row.get("within_run_check_reason") or row.get("between_block_check_reason")):
            errs.append("measurement_status=CHECK 但没有任何结构化原因")
    # 10. within-run OK + between CHECK：原因只能指向 between-block
    if row.get("within_run_status") == "OK" and row.get("within_run_check_reason"):
        errs.append("within_run_status=OK 却带 within_run_check_reason（原因归属错误）")
    if row.get("between_block_status") == "CHECK":
        if not row.get("between_block_check_reason"):
            errs.append("between_block_status=CHECK 但缺 between_block_check_reason")
        for why in row.get("speedup_claim_blockers") or []:
            if "within-run" in str(why):
                errs.append(f"blockers 把 between-block 问题写成 within-run：{why}")
    # 11/12. availability 双向一致（validate_row 已查，这里再确保 CHECK 行整体可解释）
    for fld in ("prefill_seconds", "decode_seconds", "total_seconds"):
        if measured and row.get(fld) is not None \
                and (row.get("availability") or {}).get(fld) is False:
            errs.append(f"{fld} 有值但 availability=false")
    return errs


def write_g5_run(path: Path, row: dict, *, append: bool = False,
                 ref_row: dict | None = None, strict_canonical: bool = True) -> Path:
    """写入一条 G5 行。**默认拒绝覆盖既有文件**；append=True 仅用于同一 run 的多条件追加。

    `strict_canonical=True`（默认）时额外跑 canonical 自洽校验：**宁可写不出来，
    也不写出一份自相矛盾的 JSONL**。`ref_row` 给配对行可让校验额外核对两侧 hash。
    """
    row = finalize_row(row)
    errs = validate_row(row)
    if errs:
        raise ValueError("schema 校验失败: " + "; ".join(errs[:8]))
    if strict_canonical:
        cerrs = validate_canonical(row, ref_row=ref_row)
        if cerrs:
            raise ValueError("canonical 自洽校验失败: " + "; ".join(cerrs[:8]))
    if path.exists() and not append:
        raise SystemExit(f"[拒绝覆盖] {path} 已存在；换 run_id 或显式 append")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if path.exists() else "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
    return path


def read_g5_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- 旧 Part 4 兼容层
LEGACY_MAP = {
    "t_prefill_s": "prefill_seconds",
    "t_decode_s": "decode_seconds",
    "t_total_s": "total_seconds",
    "decode_tok_s": "decode_tok_per_s",
    "e2e_tok_s": "e2e_tok_per_s",
    "peak_alloc_GiB": "peak_allocated_gib",
    "peak_resv_GiB": "peak_reserved_gib",
    "gen_sha1": "generated_token_hash",
    "n_gen": "n_generated",
    "accept_rate": "acceptance",
    "target_kv": "target_kv",
    "input_token_hash": "input_token_hash",
}

# ---- 显式字段映射表（**不做泛化字符串替换**：删 `_med` 会误伤 `accept_rate_med` 这类语义不同的量）----
# `kind == "run"`：每 rep 一行的原始字段。
RUN_FIELD_MAP: dict[str, str] = {
    "t_prefill_s": "prefill_seconds",
    "t_decode_s": "decode_seconds",
    "t_total_s": "total_seconds",
    "decode_tok_s": "decode_tok_per_s",
    "e2e_tok_s": "e2e_tok_per_s",
    "peak_alloc_GiB": "peak_allocated_gib",
    "peak_resv_GiB": "peak_reserved_gib",
    "accept_rate": "acceptance_reported",
    "accepted": "accepted",
    "proposed": "proposed",
    "rounds": "rounds",
    "draft_calls": "draft_calls",
    "target_block_calls": "target_block_calls",
    "mismatch_rounds": "mismatch_rounds",
    "all_accept_rounds": "all_accept_rounds",
}

# `kind == "summary"`：`*_med` 是各统计量的中位数（**唯一可信来源**）。
SUMMARY_FIELD_MAP: dict[str, str] = {
    "t_prefill_s_med": "prefill_seconds",
    "t_decode_s_med": "decode_seconds",
    "t_total_s_med": "total_seconds",
    "decode_tok_s_med": "decode_tok_per_s",
    "e2e_tok_s_med": "e2e_tok_per_s",
    "peak_alloc_GiB_med": "peak_allocated_gib",
    "peak_resv_GiB_med": "peak_reserved_gib",
    "accept_rate_med": "acceptance_reported",
    "accepted_med": "accepted",
    "proposed_med": "proposed",
    "rounds_med": "rounds",
    "draft_calls_med": "draft_calls",
    "target_block_calls_med": "target_block_calls",
    "mismatch_rounds_med": "mismatch_rounds",
    "all_accept_rounds_med": "all_accept_rounds",
}

# AR 模式下这些量**语义上不存在**（不是 0，也不是缺失）：一律 null，避免把 0 当成测量值。
AR_NOT_APPLICABLE: tuple[str, ...] = (
    "acceptance", "acceptance_reported", "accepted", "proposed", "rounds",
    "draft_calls", "target_block_calls", "mismatch_rounds", "all_accept_rounds",
    "accepted_length_histogram",
)


def token_ids_hash(ids) -> str | None:
    """生成序列哈希。**必须**与 `g4_part4_pilot_measure.gen_sha1` 完全一致（自测断言两者相等）。"""
    if ids is None:
        return None
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


def _finite(v):
    """NaN/Inf 一律视为"没有值"，避免把 nan 写进 canonical（availability 会自相矛盾）。"""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def source_kind(src: dict) -> str:
    """判定来源行类型：summary / run / 未知。"""
    k = src.get("kind")
    if k in ("summary", "run"):
        return k
    # 旧文件可能不写 kind：有 *_med 就当 summary
    return "summary" if any(k2.endswith("_med") for k2 in src) else "run"


def map_stat_fields(src: dict) -> tuple[dict, list[str]]:
    """按显式映射表取出统计字段。返回 (映射结果, 实际命中的源字段名列表)。

    summary 优先 `*_med`；缺失时**不回退**到"用 total-decode 偷算 prefill"这类派生，
    只把结果留空（available=false），由 validate_canonical 在写盘前拦下真正该有值的行。
    """
    kind = source_kind(src)
    table = SUMMARY_FIELD_MAP if kind == "summary" else RUN_FIELD_MAP
    out: dict = {}
    used: list[str] = []
    for src_key, dst_key in table.items():
        if src_key in src and src[src_key] is not None:
            v = _finite(src[src_key])
            if v is not None:
                out[dst_key] = v
                used.append(src_key)
    if kind == "summary":
        for k in ("generated_token_hash", "generated_token_ids", "input_token_hash",
                  "n_generated", "n_gen", "eos_id", "input_tokens"):
            if src.get(k) is not None:
                out[k] = src[k]
                used.append(k)
    if "acceptance" not in out and isinstance(out.get("accepted"), int) \
            and isinstance(out.get("proposed"), int) and out["proposed"] > 0:
        out["acceptance"] = out["accepted"] / out["proposed"]     # 恒等式，非派生估计
    return out, used


def build_g5_row(src: dict, meta: dict | None = None) -> dict:
    """把 Part 4 的 run/summary 行（新旧皆可）映射为 G5 行。缺失字段 -> None + available=false。

    v1.2 的三条硬规则：
      ① **summary 走 `*_med` 显式映射**（`SUMMARY_FIELD_MAP`），run 行走 `RUN_FIELD_MAP`；
      ② **绝不**用 `total - decode` 偷算 prefill（只认汇报的中位数；来源记在 `prefill_source`）；
      ③ AR 行里 SpecDec 专属统计量（acceptance/accepted/proposed/rounds…）语义上不存在 ⇒ null，
         而不是把子进程汇报的 0 当成测量值。
    """
    meta = dict(meta or {})
    row: dict = {"schema_version": SCHEMA_VERSION}
    kind = source_kind(src)
    mapped, _used = map_stat_fields(src)
    row.update(mapped)
    # 直通字段（非统计量）
    for k in ("ctx", "context_actual", "context_requested", "mode", "generated_token_ids",
              "generated_token_hash", "input_token_hash", "target_kv", "eos_id",
              "ar_equal", "ar_equal_available", "ar_equal_certified", "equality_source",
              "equality_reason", "first_diff_index", "resident_models"):
        if src.get(k) is not None:
            row[k] = src[k]
    # 旧字段名兼容（v1.0/v1.1 的 run 行）：只在还没有值时才补
    for lk, gk in LEGACY_MAP.items():
        if gk not in row and src.get(lk) is not None:
            v = _finite(src[lk])
            if v is not None:
                row[gk] = v
    if "ctx" in row and "context_actual" not in row:
        row["context_actual"] = row["ctx"]
    # m_hist -> accepted_length_histogram
    hist_src = src.get("m_hist", src.get("accepted_length_histogram"))
    if hist_src is not None:
        hist: dict = {}
        for m in hist_src:
            hist[str(m)] = hist.get(str(m), 0) + 1
        row["accepted_length_histogram"] = hist
    # 子进程的 certified 标志只作为**回退**；v1.2 的权威判定来自父进程（equality_source 存在即权威）
    if row.get("ar_equal") is not None and row.get("ar_equal_certified") is None \
            and "certified" in src:
        row["ar_equal_certified"] = bool(src["certified"])
    if kind == "summary" and row.get("prefill_seconds") is not None:
        row["prefill_source"] = "reported_median"
    elif kind == "run" and row.get("prefill_seconds") is not None:
        row["prefill_source"] = "reported_per_rep"
    # 遥测：来自 monitor_start/end 快照（旧版为字符串，这里只做数值强转，失败即 None）
    for src_key, dst_key, when in (
            ("monitor_end", "nvidia_smi_used_mib", "mem_used_MiB"),
            ("monitor_end", "temperature_c", "temp_C"),
            ("monitor_end", "power_w", "power_W"),
            ("monitor_end", "sm_clock_mhz", "clock_sm_MHz"),
            ("monitor_end", "memory_clock_mhz", "clock_mem_MHz")):
        d = src.get(src_key) or {}
        if d.get("available"):
            try:
                row[dst_key] = _finite(float(d[when]))
            except (KeyError, TypeError, ValueError):
                row[dst_key] = None
        else:
            row[dst_key] = None
    row["tokens_finite"] = src.get("logits_finite") if "logits_finite" in src else \
        src.get("prefill_logits_finite")
    # ---- 状态分层：**exit=0 不等于测量有效** ----
    rels = [_finite(v) for v in (src.get("decode_tok_s_rel_spread_pct"),
                                 src.get("t_decode_s_rel_spread_pct"))]
    rels = [v for v in rels if isinstance(v, (int, float))]
    row["within_run_rel_spread_pct"] = max(rels) if rels else None
    row["reps_measured"] = src.get("reps", meta.get("reps_measured"))
    row["reps_expected"] = meta.get("reps_expected", src.get("reps_expected"))
    row["k"] = src.get("k", meta.get("k"))
    row["warmup"] = src.get("warm", meta.get("warmup"))
    row["chunk_size"] = src.get("chunk", meta.get("chunk_size"))
    row["notes"] = ("tokens_finite 范围：prefill + AR 每步；SpecDec 验证块行未扫描"
                    "（需引擎插桩）。由 results_schema.build_g5_row 从 Part 4 行映射。")
    # meta 先合并，**之后再解析角色**：否则 meta 的 mode/ar_residency 覆盖 src 后角色不重算，
    # 会产生"mode=AR 但角色按 SpecDec 填"的自相矛盾行（自测已抓到）。
    row.update({k: v for k, v in meta.items() if k in FIELDS and v is not None})
    mode_final = row.get("mode")
    res_final = row.get("ar_residency")
    role_src = meta.get("condition_role")
    if mode_final == "SpecDec":
        want = resolve_condition_role("SpecDec", None)       # SpecDec 必须双驻留（可确定）
        row.update({"ar_residency": None, "condition_role": want["condition_role"],
                    "formal_factorial_member": want["formal_factorial_member"],
                    "speedup_denominator_eligible": want["speedup_denominator_eligible"]})
        if row.get("resident_models") is None:
            row["resident_models"] = want["resident_models"]
    elif mode_final == "AR" and res_final is not None:
        want = resolve_condition_role("AR", res_final)
        row.update({"ar_residency": want["ar_residency"],
                    "resident_models": src.get("resident_models", meta.get("resident_models"))
                    or want["resident_models"],
                    "condition_role": want["condition_role"],
                    "formal_factorial_member": want["formal_factorial_member"],
                    "speedup_denominator_eligible": want["speedup_denominator_eligible"]})
    else:
        # AR 但 residency 未知（v1.0 / exploratory 数据）：**不猜**，一律 null
        for fld in ("ar_residency", "resident_models", "condition_role",
                    "formal_factorial_member", "speedup_denominator_eligible"):
            row[fld] = None
        if role_src is not None:
            row["condition_role"] = role_src
    # AR 行：SpecDec 专属统计量语义上不存在 -> 明确置 null（不把 0 当测量值）
    if row.get("mode") == "AR":
        for fld in AR_NOT_APPLICABLE:
            if fld in row:
                row[fld] = None
    # 状态分层（在 meta 合并之后算，使用最终 spread_flag / pstate 信息）
    st = derive_statuses(returncode=meta.get("returncode", 0),
                         oom=bool(src.get("oom")), spread_flag=src.get("spread_flag"),
                         pstate_qualified=row.get("pstate_qualified"))
    row.update(st)
    # between-block 是**独立一层**：它只能把 measurement_status 拉低到 CHECK，
    # 但**绝不**写进 within_run_check_reason（自测 ㉑/㉒ 抓的正是这个措辞错误）。
    bb = row.get("between_block_check")
    if bb is None:
        row["between_block_status"] = None
    else:
        row["between_block_status"] = "CHECK" if bb else "OK"
    rel_bb = row.get("between_block_rel_diff_pct")
    if bb:
        row["between_block_check_reason"] = (
            f"between-block 相对差 {rel_bb:.1f}% > {BETWEEN_BLOCK_REL_LIMIT:g}%"
            if isinstance(rel_bb, (int, float)) else "between-block 相对差超过预登记诊断线")
    else:
        row["between_block_check_reason"] = None
    if row["execution_status"] != "SUCCESS":
        row["measurement_status"] = "NOT_MEASURED"
    elif row.get("environment_status") == "ENV_INVALID":
        row["measurement_status"] = "ENV_INVALID"
    elif row.get("within_run_status") == "CHECK" or row["between_block_status"] == "CHECK":
        row["measurement_status"] = "CHECK"
    else:
        row["measurement_status"] = "OK"
    row["status"] = (row["measurement_status"] if row["execution_status"] == "SUCCESS"
                     else row["execution_status"])
    # 声明资格：状态必须 OK 且**不是诊断控制**（dual-control 永远没有声明资格）
    row["speedup_claim_allowed"] = bool(
        row["measurement_status"] == "OK"
        and row.get("between_block_check") is not True
        and row.get("env_qualified") is not False
        and row.get("condition_role") != DIAGNOSTIC_ROLE
        and row.get("formal_factorial_member") is not False)
    return finalize_row(row)


def read_legacy_part4(path: Path, meta: dict | None = None) -> list[dict]:
    """读旧 Part 4 JSONL 并映射为 G5 行（**只读，绝不改写原文件**）。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "summary" or r.get("kind") is None:
            out.append(build_g5_row(r, meta))
    return out


# ---------------------------------------------------------------- 统计工具
def mad(values) -> float | None:
    """中位数绝对偏差（MAD），抗离群；样本不足或全非有限 -> None（不填 0）。"""
    vals = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    return statistics.median([abs(v - med) for v in vals])


def iqr(values) -> float | None:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and v == v)
    if len(vals) < 4:
        return None
    n = len(vals)
    q1 = statistics.median(vals[:n // 2])
    q3 = statistics.median(vals[(n + 1) // 2:])
    return q3 - q1


# ---------------------------------------------------------------- BF16 近贴边判定（Task E）
def classify_bf16_compare(ev: dict, *, noise_floor: float = 0.0) -> str:
    """按固定证据门槛判定 BF16 差异性质。**四个条件全部满足**才允许 BF16_NEAR_TIE。

    需要 ev 提供：
      invariant_violations (int)     结构不变量违反数，必须为 0
      off_by_one (bool)              判定行是否更贴近相邻位置（结构错位），必须 False
      own_max_abs (float)            同位置 block-vs-serial max|Δ|（现场噪声）
      prev_max_abs / next_max_abs    相邻位置 max|Δ|
      fp32_margin (float|None)       该决策点 fp32 的真实间隔
    """
    if ev.get("invariant_violations"):
        return "CACHE_ROLLBACK_OR_BOOKKEEPING"
    if ev.get("off_by_one"):
        return "TARGET_BLOCK_LOGIT_INDEX"
    own = ev.get("own_max_abs")
    nb = [ev.get(k) for k in ("prev_max_abs", "next_max_abs")
          if isinstance(ev.get(k), (int, float))]
    if not isinstance(own, (int, float)) or not nb:
        return "NEEDS_INSPECTION"          # 缺证据不得豁免
    if not (own * 1.0 < min(nb)):          # 同位置必须明显更近
        return "NEEDS_INSPECTION"
    sig = ev.get("fp32_margin")
    if sig is None:
        return "NEEDS_INSPECTION"
    if abs(sig) <= max(own, noise_floor):
        return "BF16_NEAR_TIE"
    return "NEEDS_INSPECTION"


def git_meta(cwd: Path | None = None) -> dict:
    """run 级元数据：commit / dirty。取不到 -> None（不编造）。

    注意：Windows 上必须显式 `encoding='utf-8'`，否则 git 输出的 UTF-8 路径会被系统代码页
    （GBK）解码而在读取线程里抛 UnicodeDecodeError。
    """
    import subprocess
    cwd = str(cwd or Path.cwd())
    out = {"git_commit": None, "dirty_worktree": None}

    def _run(args):
        try:
            return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=15)
        except Exception:  # noqa: BLE001
            return None

    c = _run(["git", "rev-parse", "HEAD"])
    if c is not None and c.returncode == 0:
        out["git_commit"] = c.stdout.strip()
    d = _run(["git", "status", "--porcelain"])
    if d is not None and d.returncode == 0:
        out["dirty_worktree"] = bool(d.stdout.strip())
    return out


def default_run_id(prefix: str = "g5") -> str:
    import time
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def g5_path(outdir: Path, config_hash: str, run_id: str) -> Path:
    return Path(outdir) / f"g5_{SCHEMA_VERSION}_{config_hash}_{run_id}.jsonl"


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip() not in ("", "0", "false", "False", "no")
