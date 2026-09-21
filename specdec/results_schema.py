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

import json
import math
import os
import statistics
from pathlib import Path

SCHEMA_VERSION = "g5-results-1.1"
# v1.0 -> v1.1（不兼容变更）：新增资源口径字段并区分 AR 主基线/诊断控制；
#   v1.0 行仍可由 build_g5_row() 只读映射，新字段一律 null + availability=false（**不猜**）。

MODE_CHOICES = ("AR", "SpecDec")
AR_RESIDENCY_CHOICES = ("target_only", "dual_control")
PRIMARY_AR_RESIDENCY = "target_only"          # 正式主基线
DIAGNOSTIC_AR_RESIDENCY = "dual_control"      # 诊断控制（不计入 factorial）

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
    "decode_tok_per_s": ("float", ""),
    "e2e_tok_per_s": ("float", ""),
    "n_generated": ("int", ""),
    "acceptance": ("float", "accepted/proposed（AR 为 null）"),
    "accepted": ("int", ""),
    "proposed": ("int", ""),
    "rounds": ("int", ""),
    "draft_calls": ("int", ""),
    "target_block_calls": ("int", ""),
    "mismatch_rounds": ("int", ""),
    "all_accept_rounds": ("int", ""),
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
    "ar_equal_certified": ("bool", "前提（ctx/max_new/eos/input_token_hash）是否全部满足"),
    "first_diff_index": ("int", "首个不同 token 位置（不等时）"),
    "status": ("str", "OK | OOM | CHECK | SKIP"),
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


def compute_speedup(ar_row: dict, spec_row: dict) -> dict:
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
    out["speedup_claim_allowed"] = bool(
        ar_row.get("speedup_claim_allowed") and spec_row.get("speedup_claim_allowed"))
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


def finalize_row(row: dict) -> dict:
    """补齐 schema 字段：缺失 -> None；值为 None 的字段在 availability 里记 false。"""
    out = dict(row)
    for name in FIELDS:
        out.setdefault(name, None)
    out["schema_version"] = row.get("schema_version") or SCHEMA_VERSION
    avail = {k: v for k, v in (row.get("availability") or {}).items() if k in FIELDS}
    for name in NULLABLE_FIELDS:
        if out[name] is None:
            avail.setdefault(name, False)
        else:
            avail.setdefault(name, bool(avail.get(name, True)))
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
    return errs


def write_g5_run(path: Path, row: dict, *, append: bool = False) -> Path:
    """写入一条 G5 行。**默认拒绝覆盖既有文件**；append=True 仅用于同一 run 的多条件追加。"""
    row = finalize_row(row)
    errs = validate_row(row)
    if errs:
        raise ValueError("schema 校验失败: " + "; ".join(errs[:8]))
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


def build_g5_row(src: dict, meta: dict | None = None) -> dict:
    """把 Part 4 的 run/summary 行（新旧皆可）映射为 G5 行。缺失字段 -> None + available=false。"""
    meta = dict(meta or {})
    row: dict = {"schema_version": SCHEMA_VERSION}
    for lk, gk in LEGACY_MAP.items():
        if lk in src and src[lk] is not None:
            row[gk] = src[lk]
    for k in ("ctx", "context_actual", "context_requested", "mode", "accepted", "proposed",
              "rounds", "draft_calls", "target_block_calls", "mismatch_rounds",
              "all_accept_rounds", "peak_allocated_gib", "n_generated", "m_hist",
              "generated_token_ids", "ar_equal", "ar_equal_available", "certified"):
        if k in src and src[k] is not None:
            row[k] = src[k]
    # 字段改名与派生
    if "ctx" in row and "context_actual" not in row:
        row["context_actual"] = row["ctx"]
    if "m_hist" in row:
        hist = row.pop("m_hist")
        row["accepted_length_histogram"] = {}
        for m in hist:
            row["accepted_length_histogram"][str(m)] = \
                row["accepted_length_histogram"].get(str(m), 0) + 1
    if row.get("ar_equal") is not None and "ar_equal_certified" not in row:
        row["ar_equal_certified"] = bool(src.get("certified"))
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
                row[dst_key] = float(d[when])
            except (KeyError, TypeError, ValueError):
                row[dst_key] = None
        else:
            row[dst_key] = None
    row["tokens_finite"] = src.get("logits_finite") if "logits_finite" in src else \
        src.get("prefill_logits_finite")
    row["status"] = "OOM" if src.get("oom") else "OK"
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
        row.update({"ar_residency": None, "resident_models": want["resident_models"],
                    "condition_role": want["condition_role"],
                    "formal_factorial_member": want["formal_factorial_member"],
                    "speedup_denominator_eligible": want["speedup_denominator_eligible"]})
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
    row["speedup_claim_allowed"] = (row["status"] == "OK")
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
    """run 级元数据：commit / dirty。取不到 -> None（不编造）。"""
    import subprocess
    cwd = str(cwd or Path.cwd())
    out = {"git_commit": None, "dirty_worktree": None}
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                           text=True, timeout=15)
        if c.returncode == 0:
            out["git_commit"] = c.stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                           text=True, timeout=15)
        if d.returncode == 0:
            out["dirty_worktree"] = bool(d.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
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
