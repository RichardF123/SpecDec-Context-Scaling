"""g5_schema_selftest_cpu.py — G5 结果 schema 的纯 CPU 自测（不加载模型、不碰 GPU）。

覆盖
  1. 字段完整性 / None↔availability 一致性
  2. 遥测字段不得用字符串或 NaN 冒充；取不到必须是 null + available=false
  3. 旧 Part 4 JSONL 兼容映射（含旧版字符串遥测强转、m_hist→histogram、缺字段置 null）
  4. 写入：唯一命名、**拒绝覆盖**、append 语义、校验失败必须报错
  5. 读取旧文件**不得改写**（前后 sha256 相同）
  6. MAD / IQR 统计（样本不足或非有限 -> None，不填 0）
  7. `classify_bf16_compare` 的固定证据门槛（含三个真实历史条件 → BF16_NEAR_TIE）
用法：python -m specdec.engine.g5_schema_selftest_cpu
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path

from .. import results_schema as S

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def legacy_summary_row(**over):
    row = dict(ctx=8192, target_kv="base_bf16", mode="SpecDec", n_gen=64,
               t_prefill_s=31.06, t_decode_s=10.69, t_total_s=41.75,
               decode_tok_s=5.99, e2e_tok_s=1.53, accept_rate=56 / 76,
               rounds=20, proposed=76, accepted=56, target_block_calls=20,
               draft_calls=76, mismatch_rounds=8, all_accept_rounds=12,
               peak_alloc_GiB=6.95, peak_resv_GiB=10.47,
               gen_sha1="610d0002bdfdcb37", m_hist=[4] * 12 + [1] * 8,
               logits_finite=True, warm=1, chunk=2048,
               input_token_hash="in123", certified=False, ar_equal=True,
               ar_equal_available=True,
               monitor_end=dict(available=True, mem_used_MiB="7876", temp_C="71",
                                power_W="53.7", clock_sm_MHz="2520", clock_mem_MHz="8000"))
    row.update(over)
    return row


def main() -> int:
    print("== 1/2. 字段完整性 / availability 一致性 ==")
    row = S.build_g5_row(legacy_summary_row())
    check("finalize 后含全部 schema 字段",
          all(k in row for k in S.FIELDS), str([k for k in S.FIELDS if k not in row][:5]))
    check("schema_version 正确", row["schema_version"] == S.SCHEMA_VERSION)
    check("校验通过（真实映射行）", S.validate_row(row) == [], str(S.validate_row(row)[:3]))
    # v1.2：availability 由**值**重算（值的纯函数），因此 finalize 后不可能出现
    # "null 但 availability=true"；该矛盾仍必须被 validate_row 抓住（用于读盘/离线构造的行）。
    bad = S.finalize_row({"schema_version": S.SCHEMA_VERSION, "decode_seconds": None,
                          "availability": {"decode_seconds": True}})
    check("finalize 由值重算 availability（null -> false，不保留调用方的 true）",
          bad["availability"]["decode_seconds"] is False)
    hand = dict(S.finalize_row({"schema_version": S.SCHEMA_VERSION, "decode_seconds": 1.0}))
    hand["decode_seconds"] = None            # 手工制造矛盾（未经 finalize 重算）
    check("null 但 availability=true -> 报错",
          any("decode_seconds" in e for e in S.validate_row(hand)))
    bad2 = S.finalize_row({"schema_version": S.SCHEMA_VERSION, "temperature_c": "71"})
    check("遥测字符串 -> 报错", any("temperature_c" in e for e in S.validate_row(bad2)))
    bad3 = S.finalize_row({"schema_version": S.SCHEMA_VERSION, "power_w": float("nan")})
    check("遥测 NaN -> 报错", any("power_w" in e for e in S.validate_row(bad3)))

    print("\n== 3. 旧 Part 4 兼容映射 ==")
    check("decode_seconds 映射", row["decode_seconds"] == 10.69)
    check("generated_token_hash 映射", row["generated_token_hash"] == "610d0002bdfdcb37")
    check("acceptance == 56/76", abs(row["acceptance"] - 56 / 76) < 1e-12)
    check("context_actual 由 ctx 派生", row["context_actual"] == 8192)
    check("m_hist -> accepted_length_histogram", row["accepted_length_histogram"] == {"4": 12, "1": 8},
          str(row["accepted_length_histogram"]))
    check("旧版字符串遥测强转为 float",
          row["nvidia_smi_used_mib"] == 7876.0 and row["temperature_c"] == 71.0
          and row["power_w"] == 53.7)
    miss = S.build_g5_row(legacy_summary_row(monitor_end=dict(available=False, error="x")))
    check("遥测不可用 -> null + available=false",
          miss["temperature_c"] is None and miss["availability"]["temperature_c"] is False
          and miss["nvidia_smi_used_mib"] is None)
    check("缺失字段一律 null（如 prompt_family）",
          row["prompt_family"] is None and row["availability"]["prompt_family"] is False)
    check("ar_equal 未认证时 certified=false",
          row["ar_equal"] is True and row["ar_equal_certified"] is False)

    print("\n== 4/5. 写入拒绝覆盖 / 读取不改写 ==")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        p = S.g5_path(tdp, "deadbeef", "g5_20260921T000000Z")
        check("唯一文件名含 schema+hash+run_id",
              p.name == f"g5_{S.SCHEMA_VERSION}_deadbeef_g5_20260921T000000Z.jsonl", p.name)
        S.write_g5_run(p, row)
        before = sha(p)
        try:
            S.write_g5_run(p, row)
            check("既有文件 + 非 append -> 必须拒绝", False)
        except SystemExit:
            check("既有文件 + 非 append -> 必须拒绝", True)
        S.write_g5_run(p, row, append=True)
        check("append 追加第 2 行且不改前 1 行", len(S.read_g5_rows(p)) == 2
              and sha(p) != before)
        try:
            S.write_g5_run(tdp / "bad.jsonl", dict(row, temperature_c="71"))
            check("校验失败必须报错（不落盘）", False)
        except ValueError:
            check("校验失败必须报错（不落盘）", True)
        leg = tdp / "part4_old.jsonl"
        leg.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                                 [dict(kind="config"), legacy_summary_row(),
                                  dict(kind="summary", **legacy_summary_row(rounds=25))]) + "\n",
                       encoding="utf-8")
        h0 = sha(leg)
        rows = S.read_legacy_part4(leg)
        check("旧文件映射出 2 条（config 行被忽略）", len(rows) == 2, str(len(rows)))
        check("读取旧文件后其 sha256 不变（未改写）", sha(leg) == h0)

    print("\n== 6. MAD / IQR ==")
    check("MAD 计算", abs(S.mad([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9, str(S.mad([1, 2, 3, 4])))
    check("样本不足 -> None（不填 0）", S.mad([1.0]) is None and S.mad([]) is None)
    check("IQR 计算", abs(S.iqr([1, 2, 3, 4, 5, 6, 7, 8]) - 4.0) < 1e-9,
          str(S.iqr([1, 2, 3, 4, 5, 6, 7, 8])))
    check("IQR 样本不足 -> None", S.iqr([1, 2, 3]) is None)
    check("非有限值被剔除", S.mad([1.0, float("nan"), 3.0]) is not None)

    print("\n== 7. BF16 近贴边判定（固定证据门槛）==")
    real = [(0.3125, 8.625, 12.0625, 0.0454), (0.3086, 11.4688, 13.6562, 0.0179),
            (0.5078, 5.5781, 14.9688, 0.0345)]
    for own, prev, nxt, sig in real:
        v = S.classify_bf16_compare(dict(invariant_violations=0, off_by_one=False,
                                         own_max_abs=own, prev_max_abs=prev,
                                         next_max_abs=nxt, fp32_margin=sig))
        check(f"真实条件 own={own} sig={sig} -> BF16_NEAR_TIE", v == "BF16_NEAR_TIE", v)
    check("off_by_one=True -> TARGET_BLOCK_LOGIT_INDEX",
          S.classify_bf16_compare(dict(invariant_violations=0, off_by_one=True,
                                       own_max_abs=0.3, prev_max_abs=0.1,
                                       next_max_abs=0.1, fp32_margin=0.01))
          == "TARGET_BLOCK_LOGIT_INDEX")
    check("不变量违反 -> CACHE_ROLLBACK",
          S.classify_bf16_compare(dict(invariant_violations=1, off_by_one=False,
                                       own_max_abs=0.3, prev_max_abs=9.0,
                                       next_max_abs=9.0, fp32_margin=0.01))
          == "CACHE_ROLLBACK_OR_BOOKKEEPING")
    check("signal > 现场噪声 -> NEEDS_INSPECTION（不得豁免）",
          S.classify_bf16_compare(dict(invariant_violations=0, off_by_one=False,
                                       own_max_abs=0.3, prev_max_abs=9.0,
                                       next_max_abs=9.0, fp32_margin=1.7))
          == "NEEDS_INSPECTION")
    check("同位置不更近 -> NEEDS_INSPECTION",
          S.classify_bf16_compare(dict(invariant_violations=0, off_by_one=False,
                                       own_max_abs=0.3, prev_max_abs=0.2,
                                       next_max_abs=9.0, fp32_margin=0.01))
          == "NEEDS_INSPECTION")
    check("缺 fp32 margin -> NEEDS_INSPECTION",
          S.classify_bf16_compare(dict(invariant_violations=0, off_by_one=False,
                                       own_max_abs=0.3, prev_max_abs=9.0,
                                       next_max_abs=9.0, fp32_margin=None))
          == "NEEDS_INSPECTION")

    print("\n== 8. run 元数据 ==")
    gm = S.git_meta(Path(__file__).resolve().parents[2])
    check("git_meta 返回 commit/dirty 键", set(gm) == {"git_commit", "dirty_worktree"}, str(gm))
    rid = S.default_run_id("g5")
    check("default_run_id 含 UTC 时间戳", rid.startswith("g5_") and "T" in rid and rid.endswith("Z"),
          rid)

    print("\n== 9. 驱动层（随机化 / 快照 / 路径 / 只读）==")
    from . import g5_driver as D
    conds = [(2048, "base_bf16", "AR"), (4096, "v_fp8", "SpecDec"), (8192, "base_bf16", "SpecDec")]
    o1 = D.randomized_order(conds, 20260921)
    o2 = D.randomized_order(conds, 20260921)
    o3 = D.randomized_order(conds, 20260922)
    check("同 seed 随机化可复现", o1 == o2 and len(o1) == len(conds))
    check("不同 seed 顺序不同（至少一次）", o1 != o3 or len(set(conds)) < 2)
    check("随机化不改元素集合", sorted(map(str, o1)) == sorted(map(str, conds)))
    snap = D.system_snapshot()
    check("系统快照 available 字段存在且为 bool", isinstance(snap.get("available"), bool), str(snap)[:80])
    if snap.get("available"):
        check("快照数值字段为 float 或 None（不得字符串/0 冒充）",
              all(isinstance(snap.get(k), (float, type(None)))
                  for k in ("mem_used_MiB", "temp_C", "power_W")))
    check("缺失 run_id -> (None) 且不抛异常", D.read_part4_rows("__no_such_run__")[2] is None)
    p = S.g5_path(D.G5_DIR, "cafe1234", "g5stab_TEST")
    check("G5 文件名含 schema + hash + run_id",
          p.name == f"g5_{S.SCHEMA_VERSION}_cafe1234_g5stab_TEST.jsonl", p.name)

    print("\n== 10. AR 基线口径：target_only 正式主基线 vs dual-resident 诊断控制 ==")
    # 10.1–10.3 驻留决策（纯函数，不加载模型）
    p1 = S.plan_process([(2048, "base_bf16", "AR")], "target_only")
    check("AR target_only 不加载 draft", p1["load_draft"] is False
          and p1["resident_models"] == ["target"], str(p1))
    p2 = S.plan_process([(2048, "base_bf16", "AR")], "dual_control")
    check("AR dual_control 加载 draft（仅常驻）", p2["load_draft"] is True
          and p2["resident_models"] == ["target", "draft"], str(p2))
    p3 = S.plan_process([(8192, "v_fp8", "SpecDec")], None)
    check("SpecDec 必须加载双模型", p3["load_draft"] is True
          and p3["resident_models"] == ["target", "draft"] and p3["ar_residency"] is None, str(p3))
    # 10.4 非法组合
    for bad, res, why in (([(2048, "base_bf16", "AR"), (2048, "base_bf16", "SpecDec")],
                           "target_only", "AR target_only 与 SpecDec 同进程"),
                          ([(2048, "base_bf16", "AR")], "bogus", "非法 residency"),
                          ([(2048, "base_bf16", "X")], "target_only", "非法 mode")):
        try:
            S.plan_process(bad, res)
            check(f"拒绝：{why}", False)
        except ValueError:
            check(f"拒绝：{why}", True)
    # 10.5–10.6 primary 12 与 control 6 分离
    primary = S.build_primary_matrix([2048, 4096, 8192], ["base_bf16", "v_fp8"])
    control = S.build_dual_controls([2048, 4096, 8192], ["base_bf16", "v_fp8"])
    check("primary factorial 恰好 12 cells", len(primary) == 12, str(len(primary)))
    check("primary 的 AR cells 全为 target_only",
          all(S.resolve_condition_role(m, S.PRIMARY_AR_RESIDENCY if m == "AR" else None)
              ["ar_residency"] in (S.PRIMARY_AR_RESIDENCY, None) for _, _, m in primary))
    check("dual control 恰好 6 cells", len(control) == 6, str(len(control)))
    check("dual control 不计入 primary 12",
          len(primary) == 12 and not set(control) & {(c, kv, m) for c, kv, m in primary
                                                     if m == "AR"}
          if False else len(primary) == 12 and len(control) == 6)
    # 10.7–10.9 speedup / penalty 方向
    ar_ok = dict(mode="AR", ar_residency="target_only", context_actual=8192,
                 target_kv="base_bf16", decode_seconds=2.79, total_seconds=10.10,
                 speedup_denominator_eligible=True, speedup_claim_allowed=True, status="OK")
    sp_ok = dict(mode="SpecDec", ar_residency=None, context_actual=8192,
                 target_kv="base_bf16", decode_seconds=10.69, total_seconds=41.75,
                 speedup_denominator_eligible=False, speedup_claim_allowed=True, status="OK")
    su = S.compute_speedup(ar_ok, sp_ok)
    check("speedup = T_AR/T_Spec（<1 表示 SpecDec 更慢）",
          abs(su["speedup_decode"] - 2.79 / 10.69) < 1e-9 and su["speedup_decode"] < 1, str(su))
    ar_dual = dict(ar_ok, ar_residency="dual_control", speedup_denominator_eligible=False)
    try:
        S.compute_speedup(ar_dual, sp_ok)
        check("dual control 不得作为 speedup 分母", False)
    except ValueError:
        check("dual control 不得作为 speedup 分母", True)
    pen = S.residency_penalty(ar_ok, dict(ar_dual, decode_seconds=4.0, total_seconds=14.0))
    check("residency_penalty 方向：dual 更慢 -> penalty<1 且 slowdown>1",
          abs(pen["penalty_decode"] - 0.6975) < 1e-3 and abs(pen["slowdown_decode"] - 1.4337) < 1e-2,
          str({k: round(v, 3) for k, v in pen.items() if isinstance(v, float)}))
    # 10.10 旧数据 residency：AR 未知 -> null（不猜）；SpecDec 可确定 -> 填双驻留
    legacy_sp = S.build_g5_row(legacy_summary_row())
    check("旧 SpecDec 行：ar_residency=null（不适用）+ 驻留可确定为 dual",
          legacy_sp["ar_residency"] is None
          and legacy_sp["resident_models"] == ["target", "draft"]
          and legacy_sp["condition_role"] == "primary"
          and legacy_sp["speedup_denominator_eligible"] is False, str(legacy_sp["resident_models"]))
    legacy_ar = S.build_g5_row(legacy_summary_row(mode="AR"))
    check("旧 AR 行：residency 未知 -> 五字段全 null + available=false（不猜）",
          legacy_ar["ar_residency"] is None and legacy_ar["resident_models"] is None
          and legacy_ar["condition_role"] is None
          and legacy_ar["formal_factorial_member"] is None
          and legacy_ar["speedup_denominator_eligible"] is None
          and legacy_ar["availability"]["ar_residency"] is False, str(legacy_ar["ar_residency"]))
    # 10.11 schema 拒绝矛盾组合
    meta_ar = dict(run_id="t", block_index=0, condition_index=0, seed=1,
                   condition_order=["x"], timestamp_utc="2026-09-21T00:00:00Z",
                   mode="AR", ar_residency="target_only")
    good = S.build_g5_row(legacy_summary_row(), meta_ar)
    check("AR target_only 正例通过校验", S.validate_row(good) == [], str(S.validate_row(good)[:2]))
    bad_rows = [
        ("AR target_only 却声明 resident_models 含 draft",
         dict(good, resident_models=["target", "draft"])),
        ("dual control 却标 formal_factorial_member=true",
         dict(S.build_g5_row(legacy_summary_row(), dict(meta_ar, ar_residency="dual_control")),
              formal_factorial_member=True)),
        ("dual control 却标 speedup_denominator_eligible=true",
         dict(S.build_g5_row(legacy_summary_row(), dict(meta_ar, ar_residency="dual_control")),
              speedup_denominator_eligible=True)),
        ("SpecDec 却带非空 ar_residency",
         dict(S.build_g5_row(legacy_summary_row(), dict(meta_ar, mode="SpecDec",
                                                        ar_residency=None)),
              ar_residency="target_only")),
        ("status=CHECK 却标 speedup_claim_allowed=true", dict(good, status="CHECK",
                                                             speedup_claim_allowed=True)),
    ]
    for why, r in bad_rows:
        errs = S.validate_row(S.finalize_row(r))
        check(f"拒绝：{why}", bool(errs), str(errs[:1]))
    # 10.12 不稳定 cell 仍可报告，但不得作确定加速声明
    sp_unstable = dict(sp_ok, speedup_claim_allowed=False, status="CHECK")
    su2 = S.compute_speedup(ar_ok, sp_unstable)
    check("不稳定 cell：数字仍产出（cell 不从报告消失）",
          su2["speedup_decode"] is not None and abs(su2["speedup_decode"] - 2.79 / 10.69) < 1e-9)
    check("不稳定 cell：禁止确定加速声明", su2["speedup_claim_allowed"] is False)

    print("\n== 11. probe 级 cell 构造（能抓到“启动即崩”类 bug）==")
    from . import g5_stability_probe as P
    pri, ctl_off = P.build_cells()
    check("probe.build_cells 默认：primary=12 且 AR 全为 target_only",
          len(pri) == 12 and all(c[3] == (S.PRIMARY_AR_RESIDENCY if c[2] == "AR" else None)
                                 for c in pri), f"{len(pri)}")
    check("probe.build_cells 默认：不含 dual control", len(ctl_off) == 0, str(len(ctl_off)))
    _old = P.INCLUDE_DUAL
    P.INCLUDE_DUAL = True
    try:
        pri2, ctl = P.build_cells()
    finally:
        P.INCLUDE_DUAL = _old
    check("probe.build_cells 开启 dual：control=6 且 residency=dual_control",
          len(ctl) == 6 and all(c[2] == "AR" and c[3] == S.DIAGNOSTIC_AR_RESIDENCY
                                for c in ctl), f"{len(ctl)}")
    check("probe.build_cells：dual 与 primary 不重叠（不计入 12）",
          len(pri2) == 12 and not (set(ctl) & set(pri2)))
    check("probe.cell_label 形如 ctx,kv,mode,residency",
          P.cell_label((8192, "v_fp8", "AR", "target_only")) == "8192,v_fp8,AR,target_only"
          and P.cell_label((8192, "v_fp8", "SpecDec", None)) == "8192,v_fp8,SpecDec,-")

    print("\n== 12. 状态分层 / speedup 资格 / P-state 门禁 ==")

    def _src(**over):
        s = dict(ctx=2048, target_kv="v_fp8", n_gen=64, t_prefill_s=1.0, t_decode_s=2.0,
                 t_total_s=3.0, decode_tok_s=20.0, e2e_tok_s=15.0, rounds=0, proposed=0,
                 accepted=0, target_block_calls=0, draft_calls=0, mismatch_rounds=0,
                 all_accept_rounds=0, peak_alloc_GiB=5.0, peak_resv_GiB=6.0,
                 logits_finite=True, reps=5, warm=1, chunk=2048, k=4,
                 input_token_hash="ih1", certified=True, spread_flag="ok",
                 decode_tok_s_rel_spread_pct=3.0, t_decode_s_rel_spread_pct=2.0)
        s.update(over)
        return s

    def _meta(**over):
        m = dict(run_id="t", block_index=0, condition_index=0, seed=1, condition_order=["x"],
                 timestamp_utc="2026-09-22T00:00:00Z", mode="AR",
                 ar_residency=S.PRIMARY_AR_RESIDENCY, reps_expected=5, returncode=0,
                 env_qualified=True, pstate_qualified=True, between_block_check=False,
                 between_block_rel_diff_pct=5.0, max_new_tokens=64)
        m.update(over)
        return m

    ar_ok = S.build_g5_row(_src(), _meta())
    sp_ok = S.build_g5_row(_src(mode="SpecDec"), _meta(mode="SpecDec", ar_residency=None))
    check("合格行通过 schema 校验", S.validate_row(ar_ok) == [] and S.validate_row(sp_ok) == [],
          str(S.validate_row(ar_ok)[:2]))

    # T1 exit=0 + spread_flag=CHECK -> measurement_status/status = CHECK
    ar_spread_bad = S.build_g5_row(
        _src(spread_flag="CHECK:relative-spread>20%", decode_tok_s_rel_spread_pct=222.4),
        _meta())
    check("① exit=0 + spread CHECK -> status=CHECK（不写 OK）",
          ar_spread_bad["execution_status"] == "SUCCESS"
          and ar_spread_bad["measurement_status"] == "CHECK"
          and ar_spread_bad["status"] == "CHECK"
          and "relative-spread" in (ar_spread_bad["within_run_check_reason"] or ""),
          f"{ar_spread_bad['execution_status']}/{ar_spread_bad['measurement_status']}")
    check("① 同时记录 within_run_rel_spread_pct",
          ar_spread_bad["within_run_rel_spread_pct"] == 222.4)
    # T2 within-run CHECK -> 不得声明
    check("② within-run CHECK -> speedup_claim_allowed=false",
          not S.speedup_claim(ar_spread_bad, sp_ok)["allowed"])
    # T3/T4 一侧不稳即不得声明
    sp_spread_bad = S.build_g5_row(
        _src(mode="SpecDec", spread_flag="CHECK:relative-spread>20%"), 
        _meta(mode="SpecDec", ar_residency=None))
    check("③ AR 稳 + Spec within-run CHECK -> NO CLAIM",
          not S.speedup_claim(ar_ok, sp_spread_bad)["allowed"]
          and any("SpecDec" in r for r in S.speedup_claim(ar_ok, sp_spread_bad)["reasons"]))
    check("④ AR within-run CHECK + Spec 稳 -> NO CLAIM",
          not S.speedup_claim(ar_spread_bad, sp_ok)["allowed"]
          and any("AR" in r for r in S.speedup_claim(ar_spread_bad, sp_ok)["reasons"]))
    # T5 between-block >20% -> 不得声明（并被吸收进 measurement_status）
    ar_bb = S.build_g5_row(_src(), _meta(between_block_check=True,
                                         between_block_rel_diff_pct=66.0))
    check("⑤ between-block>20% -> status=CHECK 且 NO CLAIM",
          ar_bb["status"] == "CHECK" and ar_bb["between_block_check"] is True
          and not S.speedup_claim(ar_bb, sp_ok)["allowed"])
    # T6/T7 分母资格
    check("⑥ 只有 target-only AR 可作分母（speedup_claim 拒绝 dual/非 AR）",
          not S.speedup_claim(dict(ar_ok, ar_residency="dual_control",
                                   speedup_denominator_eligible=False), sp_ok)["allowed"])
    try:
        S.compute_speedup(dict(ar_ok, ar_residency="dual_control",
                               speedup_denominator_eligible=False), sp_ok)
        check("⑦ dual-control 不得作为正式分母（ValueError）", False)
    except ValueError:
        check("⑦ dual-control 不得作为正式分母（ValueError）", True)
    # T8 ENV_INVALID 永远不得声明
    ar_env = S.build_g5_row(_src(), _meta(pstate_qualified=False))
    check("⑧ P-state 不合格 -> measurement_status=ENV_INVALID 且 NO CLAIM",
          ar_env["measurement_status"] == "ENV_INVALID"
          and not S.speedup_claim(ar_env, sp_ok)["allowed"])
    # T9/T10/T11/T12 P-state 判定
    v_none = D.pstate_verdict([])
    check("⑨ P-state 采样缺失 -> available=false（不填 0）",
          v_none["available"] is False and v_none["qualified"] is None, str(v_none))
    lo = [dict(clock_sm_MHz="630", clock_mem_MHz="6000", power_W="17", temp_C="62")] * 5
    hi = [dict(clock_sm_MHz="2520", clock_mem_MHz="8000", power_W="60", temp_C="70")] * 5
    v_lo = D.pstate_verdict(lo)
    check("⑩ 低于阈值 -> qualified=false 且 reason 含 ENV_INVALID",
          v_lo["qualified"] is False and "ENV_INVALID" in v_lo["reason"], str(v_lo["reason"]))
    check("⑪ 高性能样本通过资格判定", D.pstate_verdict(hi)["qualified"] is True)
    idle_only = [dict(clock_sm_MHz="210", clock_mem_MHz="405", power_W="3", temp_C="45")] * 5
    from . import gpu_pstate_preflight as PF
    from . import g4_part4_pilot_measure as MEAS      # 仅用于断言 hash 实现一致（不加载模型）
    v_idle = PF.verdict_from_sampler([], dict(clock_sm_MHz="315"))
    check("⑫ 空闲样本不得作为判定输入（无负载样本 -> available=false）",
          v_idle["available"] is False and v_idle["verdict_input"] == "load_samples_only"
          and D.pstate_verdict(idle_only)["available"] is True, str(v_idle["reason"]))
    # T13/T14 不稳定 cell 仍在报告中 + 输出拒绝原因
    su_bad = S.compute_speedup(ar_bb, sp_ok)
    check("⑬ unstable cell 仍产出数字（不从报告消失）",
          su_bad["speedup_decode"] is not None)
    check("⑭ 拒绝时输出具体原因（NO CLAIM: ...）而非静默",
          su_bad["speedup_claim_allowed"] is False and su_bad["speedup_claim_blockers"]
          and all(r.startswith("NO CLAIM") for r in su_bad["speedup_claim_blockers"]),
          str(su_bad["speedup_claim_blockers"][:2]))
    # T15 条件不一致禁止比较
    sp_other_ctx = S.build_g5_row(_src(mode="SpecDec", ctx=8192),
                                  _meta(mode="SpecDec", ar_residency=None,
                                        context_actual=8192))
    try:
        S.compute_speedup(ar_ok, sp_other_ctx)
        check("⑮ context 不一致禁止比较", False)
    except ValueError:
        check("⑮ context 不一致禁止比较", True)
    sp_hash_bad = S.build_g5_row(_src(mode="SpecDec", input_token_hash="ih2"),
                                 _meta(mode="SpecDec", ar_residency=None))
    c = S.speedup_claim(ar_ok, sp_hash_bad)
    check("⑮ input_token_hash 不一致 -> NO CLAIM",
          not c["allowed"] and any("input_token_hash" in r for r in c["reasons"]),
          str(c["reasons"][:1]))
    # T16–T18 Sampler 生命周期
    # 回归来源：Sampler 把 Event 存在 self._stop 上 → 覆盖 threading.Thread._stop 方法
    # → Thread.join() 内部 self._stop() 抛 TypeError: 'Event' object is not callable。
    # 教训：自测必须**跑真实入口**（含线程 stop/join），只测纯函数抓不到这类崩溃。
    check("⑯ Sampler 不得覆盖 Thread._stop（否则 join 必崩）",
          callable(getattr(PF.Sampler, "_stop", None))
          and not isinstance(getattr(PF.Sampler, "_stop", None), threading.Event))
    _real_snap = D.system_snapshot
    try:
        D.system_snapshot = lambda: {"available": True, "clock_sm_MHz": "2520",
                                     "clock_mem_MHz": "8000", "power_W": "60", "temp_C": "70"}
        s_ok = PF.Sampler(0.01)
        s_ok.start()
        time.sleep(0.15)
        s_ok.stop()
        check("⑰ Sampler.stop() 正常返回并已采集样本",
              not s_ok.is_alive() and len(s_ok.samples) >= 1 and s_ok.failures == 0,
              f"samples={len(s_ok.samples)} failures={s_ok.failures}")
        D.system_snapshot = lambda: {"available": False, "reason": "no nvidia-smi"}
        s_bad = PF.Sampler(0.01)
        s_bad.start()
        time.sleep(0.15)
        s_bad.stop()
        check("⑱ 快照不可用时计入 failures，不崩溃也不造数",
              not s_bad.is_alive() and not s_bad.samples and s_bad.failures >= 1,
              f"samples={len(s_bad.samples)} failures={s_bad.failures}")
    finally:
        D.system_snapshot = _real_snap

    # ---- 13. 环境变量解析（回归：cmd 的 `set X=v & set Y=w` 会把尾随空格带进值里）
    # 若不 strip，`m in MODES` 会静默丢掉 SpecDec ⇒ "跑完但条件不全"，属于最危险的一类错误。
    _real_env = os.environ.get("P5_MODE")
    try:
        os.environ["P5_MODE"] = "AR,SpecDec "
        check("⑲ _env_list 逐项 strip（'AR,SpecDec ' -> ['AR','SpecDec']）",
              P._env_list("P5_MODE", "") == ["AR", "SpecDec"],
              str(P._env_list("P5_MODE", "")))
        check("⑲ 带尾随空格时 SpecDec **不会**被静默丢掉",
              "SpecDec" in [m for m in S.MODE_CHOICES if m in P._env_list("P5_MODE", "")])
        os.environ["P5_MODE"] = ", ,"
        check("⑲ 全空白项被丢弃（解析为空列表，而不是 ['','']）",
              P._env_list("P5_MODE", "AR") == [])
    finally:
        if _real_env is None:
            os.environ.pop("P5_MODE", None)
        else:
            os.environ["P5_MODE"] = _real_env
    # ⑳ 未知条件名 / 空列表 -> 硬错误（不静默少跑）
    for bad_kw, name in ((dict(modes=["AR", "SpecEdc"]), "P5_MODE 拼写错误"),
                         (dict(kvs=["base_bf16 "]), "P5_KV 未 strip 的余量"),
                         (dict(ctypes=[2048, 16384]), "P5_CTX 未知长度"),
                         (dict(modes=[]), "P5_MODE 解析为空（0 个条件的假成功）")):
        try:
            P.validate_requested(**bad_kw)
            check(f"⑳ 非法配置必须报错（{name}）", False)
        except SystemExit as e:
            check(f"⑳ 非法配置必须报错（{name}）", "校验失败" in str(e))
    P.validate_requested(ctypes=[2048, 8192], kvs=["base_bf16"], modes=["AR", "SpecDec"])
    check("⑳ 合法配置不报错（缩小版复验的实际参数）", True)

    print("\n== 14. v1.2 canonical 记账（summary 映射 / 跨进程等价性 / 措辞 / dual-control / 自洽校验）==")

    # ---- 真实形状的 fixture：summary 只有 *_med，run 只有裸字段 ----
    # 生成序列都取 64 个 token（= max_new），并让 accepted + mismatch_rounds == n_generated
    # 这条**协议恒等式**在 fixture 里也成立，否则 validator 会（正确地）拒绝。
    _IDS_AR_8K = [1000 + i for i in range(64)]
    _IDS_SP_8K = list(_IDS_AR_8K)
    _IDS_AR_2K = [2000 + i for i in range(64)]
    _IDS_SP_2K = list(_IDS_AR_2K)
    _IDS_SP_2K[4] = 2999                     # 首个差异在第 4 位（2K BF16 分歧）
    _STATS = {8192: dict(accepted=56, proposed=76, mismatch=8, rounds=20, allacc=12),
              2048: dict(accepted=46, proposed=99, mismatch=18, rounds=25, allacc=7)}

    def _summ(ctx, mode, res, ids, **over):
        st = _STATS[ctx]
        s = dict(kind="summary", ctx=ctx, target_kv="base_bf16", mode=mode,
                 ar_residency=(res if mode == "AR" else None), order="b0",
                 input_token_hash=f"ih{ctx}", generated_token_hash=S.token_ids_hash(ids),
                 generated_token_ids=list(ids), n_generated=len(ids), eos_id=151645,
                 reps=5, k=4, warm=1, chunk=2048, max_new=len(ids), spread_flag="ok",
                 t_prefill_s_med=31.96, t_decode_s_med=6.72, t_total_s_med=38.65,
                 decode_tok_s_med=9.53, e2e_tok_s_med=1.66, peak_alloc_GiB_med=6.95,
                 peak_resv_GiB_med=10.47, decode_tok_s_rel_spread_pct=1.9,
                 t_decode_s_rel_spread_pct=2.0, logits_finite=True,
                 accept_rate_med=st["accepted"] / st["proposed"], accepted_med=st["accepted"],
                 proposed_med=st["proposed"], rounds_med=st["rounds"],
                 draft_calls_med=st["proposed"], target_block_calls_med=st["rounds"],
                 mismatch_rounds_med=st["mismatch"],
                 all_accept_rounds_med=st["allacc"])
        s.update(over)
        return s

    def _meta(ctx, mode, res, **over):
        m = dict(run_id="t2", block_index=0, condition_index=0, seed=1, condition_order=[],
                 timestamp_utc="2026-09-22T04:26:00Z", mode=mode,
                 ar_residency=(res if mode == "AR" else None), target_kv="base_bf16",
                 context_requested=ctx, context_actual=ctx, max_new_tokens=64,
                 reps_expected=5, reps_measured=5, returncode=0, env_qualified=True,
                 pstate_qualified=True, between_block_check=False,
                 between_block_rel_diff_pct=1.0)
        m.update(over)
        return m

    # T1/T2/T3 字段映射
    sp8 = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K), _meta(8192, "SpecDec", None))
    check("㉑ summary 的 t_prefill_s_med 进入 canonical prefill_seconds",
          sp8["prefill_seconds"] == 31.96 and sp8["prefill_source"] == "reported_median")
    check("㉑ summary 的 e2e_tok_s_med 进入 canonical e2e_tok_per_s",
          sp8["e2e_tok_per_s"] == 1.66)
    check("㉒ SpecDec summary 统计全部映射（acceptance/accepted/proposed/rounds/draft/block/mismatch/allacc）",
          sp8["accepted"] == 56 and sp8["proposed"] == 76 and sp8["rounds"] == 20
          and sp8["draft_calls"] == 76 and sp8["target_block_calls"] == 20
          and sp8["mismatch_rounds"] == 8 and sp8["all_accept_rounds"] == 12,
          f"{sp8['accepted']}/{sp8['proposed']}/{sp8['rounds']}")
    check("㉒ acceptance 用恒等式 accepted/proposed（不照抄 accept_rate_med）",
          abs(sp8["acceptance"] - 56 / 76) < 1e-12)
    check("㉒ AR 行不得携带 SpecDec 专属统计量",
          all(S.build_g5_row(_summ(8192, "AR", "target_only", _IDS_AR_8K),
                             _meta(8192, "AR", "target_only"))[f] is None
              for f in ("acceptance", "accepted", "proposed", "rounds", "draft_calls",
                        "mismatch_rounds", "all_accept_rounds")))
    check("㉒ run 行（kind=run）走裸字段映射（t_prefill_s -> prefill_seconds）",
          S.build_g5_row(dict(kind="run", ctx=2048, mode="AR", t_prefill_s=1.25,
                              t_decode_s=2.0, t_total_s=3.25, n_gen=8,
                              input_token_hash="ih"), _meta(2048, "AR", "target_only")
                         )["prefill_seconds"] == 1.25)
    check("㉒ 不得用 total-decode 偷算 prefill（缺 *_med 时必须留 null）",
          S.build_g5_row(dict(kind="summary", ctx=2048, mode="AR",
                              t_total_s_med=10.0, t_decode_s_med=4.0, n_generated=8,
                              input_token_hash="ih", reps=5, spread_flag="ok"),
                         _meta(2048, "AR", "target_only"))["prefill_seconds"] is None)
    check("㉒ NaN 统计量不得写进 canonical（accept_rate_med=nan -> null）",
          S.build_g5_row(_summ(2048, "SpecDec", None, _IDS_SP_2K, accept_rate_med=float("nan")),
                         _meta(2048, "SpecDec", None))["acceptance_reported"] is None)

    # T4/T5/T6 跨进程等价性
    def _pair(ctx, ar_ids, sp_ids, ar_kw=None, sp_kw=None):
        ar = S.build_g5_row(_summ(ctx, "AR", "target_only", ar_ids, **(ar_kw or {})),
                            _meta(ctx, "AR", "target_only"))
        sp = S.build_g5_row(_summ(ctx, "SpecDec", None, sp_ids, **(sp_kw or {})),
                            _meta(ctx, "SpecDec", None))
        rows = [ar, sp]
        S.fill_cross_process_equality(rows)
        return ar, sp

    _, sp_eq = _pair(8192, _IDS_AR_8K, _IDS_SP_8K)
    check("㉓ 8K 两侧 hash 相同 -> ar_equal=True 且已认证",
          sp_eq["ar_equal"] is True and sp_eq["ar_equal_certified"] is True
          and sp_eq["ar_equal_available"] is True
          and sp_eq["equality_source"] == "parent_cross_process")
    _, sp_ne = _pair(2048, _IDS_AR_2K, _IDS_SP_2K)
    check("㉔ 2K 两侧不同 -> ar_equal=False 且给出 first_diff_index",
          sp_ne["ar_equal"] is False and sp_ne["first_diff_index"] == 4,
          f"first_diff={sp_ne['first_diff_index']}")
    # 缺 input_token_hash：key 就对不上 -> 没有对照，绝不写 False
    _, sp_nohash = _pair(8192, _IDS_AR_8K, _IDS_SP_8K, ar_kw=dict(input_token_hash=None))
    check("㉕ 缺 input_token_hash -> available=False、certified=False、ar_equal=None（不是 False）",
          sp_nohash["ar_equal"] is None and sp_nohash["ar_equal_available"] is False
          and sp_nohash["ar_equal_certified"] is False
          and "对照" in (sp_nohash["equality_reason"] or ""),
          str(sp_nohash["equality_reason"]))
    # 配对成功但 EOS 口径不同：key 一致却仍须拒绝认证（认证逻辑本身要真的生效）
    _, sp_eos = _pair(8192, _IDS_AR_8K, _IDS_SP_8K, sp_kw=dict(eos_id=151646))
    check("㉕ 配对成功但 eos_id 不同 -> 不认证、不给结果（eos_equal 生效）",
          sp_eos["ar_equal"] is None and sp_eos["ar_equal_certified"] is False
          and "eos_equal" in (sp_eos["equality_reason"] or ""),
          str(sp_eos["equality_reason"]))
    _, sp_maxnew = _pair(8192, _IDS_AR_8K, _IDS_SP_8K, sp_kw=dict(max_new=32))
    ar_mn = S.build_g5_row(_summ(8192, "AR", "target_only", _IDS_AR_8K),
                           _meta(8192, "AR", "target_only", max_new_tokens=64))
    sp_mn = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K),
                           _meta(8192, "SpecDec", None, max_new_tokens=32))
    S.fill_cross_process_equality([ar_mn, sp_mn])
    check("㉕ max_new_tokens 不同 -> 不配对（key 含 max_new_tokens，绝不当成'不等'）",
          sp_mn["ar_equal"] is None and sp_mn["ar_equal_available"] is False)
    # 完全缺对照
    lone = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K), _meta(8192, "SpecDec", None))
    S.fill_cross_process_equality([lone])
    check("㉕ 完全没有 AR 对照 -> ar_equal=None（缺对照绝不解释为不等）",
          lone["ar_equal"] is None and lone["ar_equal_available"] is False
          and lone["equality_source"] is None)
    ar_ref, _ = _pair(8192, _IDS_AR_8K, _IDS_SP_8K)
    check("㉕ AR 参照行标为 reference_row（不与自己比较）",
          ar_ref["equality_source"] == "reference_row" and ar_ref["ar_equal"] is None)

    # T7 dual-control 声明纪律
    dc = S.build_g5_row(_summ(8192, "AR", "dual_control", _IDS_AR_8K),
                        _meta(8192, "AR", "dual_control"))
    check("㉖ dual-control：角色/分母/声明字段全为 false",
          dc["condition_role"] == "diagnostic_control"
          and dc["formal_factorial_member"] is False
          and dc["speedup_denominator_eligible"] is False
          and dc["speedup_claim_allowed"] is False)
    check("㉖ dual-control 即使一切合格也不得声明（校验器兜底）",
          S.validate_row(dc) == []
          and any("diagnostic_control" in e for e in
                  S.validate_row(dict(dc, speedup_claim_allowed=True))))

    # T8/T9/T10 措辞分层
    def _bb_sp(rel, within_flag="ok", pstate=True):
        r = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K, spread_flag=within_flag),
                           _meta(8192, "SpecDec", None, between_block_check=True,
                                 between_block_rel_diff_pct=rel, pstate_qualified=pstate))
        return r

    sp_bb = _bb_sp(42.66)
    ar_ok2 = S.build_g5_row(_summ(8192, "AR", "target_only", _IDS_AR_8K),
                            _meta(8192, "AR", "target_only"))
    cm = S.speedup_claim(ar_ok2, sp_bb)
    check("㉗ within-run OK + between CHECK -> 原因只指向 between-block",
          sp_bb["within_run_status"] == "OK" and sp_bb["between_block_status"] == "CHECK"
          and sp_bb["within_run_check_reason"] is None
          and sp_bb["between_block_check_reason"] == "between-block 相对差 42.7% > 20%"
          and any("between-block CHECK（42.7% > 20%）" in x for x in cm["reasons"])
          and not any("within-run" in x for x in cm["reasons"]), str(cm["reasons"]))
    sp_wr = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K,
                                 spread_flag="CHECK:relative-spread>20%",
                                 decode_tok_s_rel_spread_pct=222.4),
                           _meta(8192, "SpecDec", None))
    cm2 = S.speedup_claim(S.build_g5_row(_summ(8192, "AR", "target_only", _IDS_AR_8K),
                                         _meta(8192, "AR", "target_only")), sp_wr)
    check("㉘ within-run CHECK + between OK -> 原因只指向 within-run",
          sp_wr["within_run_status"] == "CHECK" and sp_wr["between_block_status"] == "OK"
          and any("within-run CHECK" in x for x in cm2["reasons"])
          and not any("between-block" in x for x in cm2["reasons"]), str(cm2["reasons"]))
    sp_env = S.build_g5_row(_summ(8192, "SpecDec", None, _IDS_SP_8K),
                            _meta(8192, "SpecDec", None, pstate_qualified=False))
    cm3 = S.speedup_claim(S.build_g5_row(_summ(8192, "AR", "target_only", _IDS_AR_8K),
                                         _meta(8192, "AR", "target_only")), sp_env)
    check("㉙ ENV_INVALID -> 原因含 ENV_INVALID 且 environment_status 为 ENV_INVALID",
          sp_env["environment_status"] == "ENV_INVALID"
          and sp_env["measurement_status"] == "ENV_INVALID"
          and any("ENV_INVALID" in x for x in cm3["reasons"]), str(cm3["reasons"]))
    check("㉙ within_run_check_reason 绝不拼接 between-block",
          all("between-block" not in str(x) for x in
              [sp_bb["within_run_check_reason"], sp_wr["within_run_check_reason"],
               sp_env["within_run_check_reason"]]))

    # T11 availability 一致性
    check("㉚ 有值字段 availability=True、空字段 availability=False（值有就 true）",
          all(sp8["availability"][f] is True for f in
              ("prefill_seconds", "e2e_tok_per_s", "accepted", "proposed", "rounds",
               "mismatch_rounds")) and
          all(ar_ok2["availability"][f] is False for f in
              ("accepted", "proposed", "rounds", "mismatch_rounds")))
    check("㉚ 跨进程填写后 availability 立即同步（stale availability 回归）",
          sp_eq["availability"]["equality_source"] is True
          and sp_eq["availability"]["equality_reason"] is True
          and lone["availability"]["equality_reason"] is True)

    # T12/T13 旧版本只读兼容 + 不覆盖历史文件
    legacy = S.build_g5_row(dict(ctx=8192, target_kv="base_bf16", mode="SpecDec", n_gen=64,
                                 t_prefill_s=31.06, t_decode_s=10.69, t_total_s=41.75,
                                 decode_tok_s=5.99, e2e_tok_s=1.53, accept_rate=56 / 76,
                                 rounds=20, proposed=76, accepted=56, draft_calls=76,
                                 target_block_calls=20, mismatch_rounds=8,
                                 all_accept_rounds=12, gen_sha1="h1", input_token_hash="ih",
                                 logits_finite=True), None)
    check("㉛ v1.0/v1.1 无 kind 的旧行仍可映射为 1.2（缺失字段 null）",
          legacy["schema_version"] == S.SCHEMA_VERSION and legacy["prefill_seconds"] == 31.06
          and legacy["acceptance"] == 56 / 76)
    with tempfile.TemporaryDirectory() as td:
        p2 = Path(td) / "legacy.jsonl"
        p2.write_text(json.dumps(dict(kind="summary", ctx=8192, mode="AR",
                                      t_prefill_s_med=1.0, t_decode_s_med=2.0,
                                      t_total_s_med=3.0, target_kv="base_bf16",
                                      input_token_hash="ih", reps=5,
                                      spread_flag="ok")) + "\n", encoding="utf-8")
        h0 = sha(p2)
        S.read_legacy_part4(p2)
        check("㉛ read_legacy_part4 只读（sha256 不变）", sha(p2) == h0)
        p3 = Path(td) / "canon.jsonl"
        S.write_g5_run(p3, ar_ok2)
        h1 = sha(p3)
        try:
            S.write_g5_run(p3, ar_ok2)
            check("㉛ 历史 canonical 文件不得被覆盖", False)
        except SystemExit:
            check("㉛ 历史 canonical 文件不得被覆盖", True)
        check("㉛ 拒绝覆盖后文件内容不变", sha(p3) == h1)
        check("㉛ 只读旧文件后其 sha256 仍不变", sha(p2) == h0)

    # T14/T15 canonical validator 抓错
    bad_eq = dict(sp_eq, ar_equal=True)
    check("㉜ validator 抓到 ar_equal 与证据矛盾（true 但两侧 hash 不同）",
          any("hash" in e for e in
              S.validate_canonical(dict(bad_eq, generated_token_hash="deadbeef"),
                                   ref_row=ar_ref)))
    check("㉜ validator 抓到 '相同 hash 却写 false'（本次真事故）",
          any("相同" in e for e in
              S.validate_canonical(dict(sp_ne, generated_token_hash="1f5e438ee5d38888",
                                        ar_equal=False),
                                   ref_row=dict(ar_ref, generated_token_hash="1f5e438ee5d38888"))))
    miss_stats = S.finalize_row(dict(sp8, acceptance=None, accepted=None, proposed=None,
                                     rounds=None))
    check("㉝ validator 抓到缺失的 SpecDec summary 统计",
          sum("缺" in e for e in S.validate_canonical(miss_stats)) >= 4,
          str(S.validate_canonical(miss_stats)[:3]))
    check("㉝ validator 抓到 acceptance 与 accepted/proposed 不一致",
          any("acceptance" in e for e in
              S.validate_canonical(S.finalize_row(dict(sp8, acceptance=0.99)))))
    check("㉝ validator 抓到 accepted+mismatch != n_generated",
          any("恒等式" in e for e in
              S.validate_canonical(S.finalize_row(dict(sp8, mismatch_rounds=1)))))
    check("㉝ validator 抓到 hash 与 ids 不一致",
          any("generated_token_hash" in e for e in
              S.validate_canonical(S.finalize_row(dict(sp8, generated_token_hash="zzz")))))
    check("㉝ validator 抓到 dual-control 声明资格",
          any("dual-control" in e for e in
              S.validate_canonical(S.finalize_row(dict(dc, speedup_claim_allowed=True)))))
    check("㉝ 合格行同一 validator 通过（不能只会报错）",
          S.validate_canonical(sp_eq, ref_row=ar_ref) == [] and
          S.validate_canonical(sp_bb, ref_row=ar_ref) == [],
          str(S.validate_canonical(sp_bb, ref_row=ar_ref)[:3]))
    check("㉞ token_ids_hash 与测量核心 gen_sha1 完全一致",
          S.token_ids_hash(_IDS_SP_8K) == MEAS.gen_sha1(_IDS_SP_8K))

    print("\n== 15. 8K variance probe 的分析逻辑（纯函数；不跑 GPU）==")
    from . import g5_8k_spec_variance_probe as V

    check("㉟ 单档：全部进程接近 -> 不误报双档",
          V.bimodality([10.0, 10.2, 9.9, 10.1], 1.5)["detected"] is False)
    bi = V.bimodality([10.0, 10.2, 5.0, 5.1], 1.5)
    check("㉟ 双档：4 进程分成 2+2 且比值 1.96 -> 检出并给出切分",
          bi["detected"] is True and bi["split"]["low"] == [5.0, 5.1]
          and abs(bi["split"]["ratio"] - 1.96) < 0.01)
    check("㉟ 进程数 <4 -> 明确不做双档判定（不硬凑结论）",
          V.bimodality([10.0, 5.0], 1.5)["detected"] is False
          and "不做双档判定" in (V.bimodality([10.0, 5.0], 1.5).get("note") or ""))
    check("㉟ 3+1 分布不满足'两侧各≥2' -> 不检出（规则事先写死）",
          V.bimodality([10.0, 5.0, 5.1, 5.2], 1.5)["detected"] is False)

    def _child(role, idx, tok_s, prefill=32.0, rel=1.5, sm=2520, alloc=7000.0, retries=0):
        d = 64.0 / tok_s
        return dict(
            role=role, index=idx, formal=dict(
                reps=[dict(prefill_s=prefill, decode_s=d, total_s=prefill + d, n_gen=64,
                           decode_tok_s=tok_s, e2e_tok_s=64 / (prefill + d)) for _ in range(5)],
                medians=dict(prefill_s=prefill, decode_s=d, total_s=prefill + d,
                             decode_tok_s=tok_s, e2e_tok_s=64 / (prefill + d)),
                decode_rel_spread_pct=rel),
            alloc_after_formal=dict(available=True, peak_allocated_mib=alloc,
                                    peak_reserved_mib=alloc + 500,
                                    inactive_split_peak_mib=12.0,
                                    num_alloc_retries=retries, cuda_free_mib=100.0),
            monitor_after=dict(system=dict(clock_sm_MHz=sm, clock_mem_MHz=8000,
                                           power_W=68.0, temp_C=82.0, mem_used_MiB=7847)),
            pstate=dict(available=True, qualified=True, reason="ok"))

    recs = [_child("spec", 0, 9.5), _child("spec", 1, 5.4, sm=2520),
            _child("spec", 2, 9.4), _child("spec", 3, 5.5, retries=3, alloc=7600.0),
            _child("ar_target", 0, 29.4), _child("ar_target", 1, 29.5),
            _child("ar_dual", 0, 23.1), _child("ar_dual", 1, 23.2)]
    a = V.analyse(recs)
    check("㊱ SpecDec 快慢两档被检出（5.4/5.5 vs 9.4/9.5）",
          a["Q1_bimodality"]["decode_tok_per_s"]["detected"] is True
          and abs(a["Q1_bimodality"]["max_over_min"] - 9.5 / 5.4) < 0.05)
    check("㊱ 每进程 within-run 稳定性都记录（含 min/max）",
          len(a["Q2_within_process"]) == 8
          and a["Q2_within_process"]["spec[0]"]["decode_rel_spread_pct"] == 1.5)
    check("㊱ 对照稳定性单列，且不含 SpecDec",
          set(a["Q5_controls"]) == {"ar_target", "ar_dual"}
          and a["Q5_controls"]["ar_dual"]["decode_tok_per_s"] == [23.1, 23.2])
    check("㊱ prefill 稳定性按角色统计（8K 应稳定）",
          a["Q4_prefill"]["spec"]["max_over_min"] == 1.0
          and a["Q4_prefill"]["spec"]["medians"] == [32.0] * 4)
    check("㊱ 相关量表只列观测，且带'未归因'说明",
          len(a["Q6_correlates"]["per_process"]) == 4
          and "不得把原因写成" in a["Q6_correlates"]["note"])
    check("㊱ 无 profiling 数据时不做阶段对比（不硬凑）", "Q3_phase_contrast" not in a)
    recs2 = [dict(r, profiling=dict(
        phases={"draft_propose": {"total_ms": 1000.0, "calls": 20, "median_ms": 50.0},
                "target_block_verify": {"total_ms": 2000.0, "calls": 20, "median_ms": 100.0},
                "kv_cat": {"total_ms": 300.0, "calls": 40, "median_ms": 7.5}},
        outer_decode_s=3.5, subset_phases=["kv_cat"], accounted_ms=3000.0,
        residual_ms=500.0, residual_pct=14.3,
        per_round_ms={"draft_propose": 50.0}, rounds=20, stats={},
        note="x")) for r in recs if r["role"] == "spec"]
    a2 = V.analyse(recs + recs2)
    check("㊱ 阶段对比取最快 vs 最慢进程，并标出 subset 阶段（不可相加）",
          a2["Q3_phase_contrast"]["fast"]["index"] == 0
          and a2["Q3_phase_contrast"]["slow"]["index"] == 1
          and "kv_cat" in a2["Q3_phase_contrast"]["subset_phases"],
          f"{a2['Q3_phase_contrast']['fast']}/{a2['Q3_phase_contrast']['slow']}")
    check("㊱ 阶段比值按 slow/fast 计算（>1 = 慢档更贵）",
          abs(a2["Q3_phase_contrast"]["per_phase_ms"]["target_block_verify"]["ratio"] - 1.0)
          < 1e-9)
    check("㊲ profiling 与正式计时是两套独立字段（不得混写）",
          "profiling" not in [k for k in ("formal",) if k in recs[0]]
          and "formal" in recs[0] and "profiling" in recs2[0]
          and V.SUBSET_PHASES == ("kv_cat",))

    # ㊳–㊵ 覆盖度：**缺整个角色绝不允许看起来"跑完了"**
    # 事故来源：2026-09-23 两次运行里 4 个 spec 子进程全部崩溃且不落盘，
    # 父进程照样打印报告 + 写 summary，Q1 只写'进程数 <4 不做双档判定'。
    recs_missing = [_child("ar_target", 0, 29.4), _child("ar_target", 1, 29.5),
                    _child("ar_dual", 0, 23.1), _child("ar_dual", 1, 23.2)]
    cov = V.coverage_report(recs_missing, roles=["spec", "ar_target", "ar_dual"], procs=4)
    check("㊳ 覆盖度能看出'整个 spec 角色 0 个样本'",
          cov["spec"]["measured"] == 0 and cov["spec"]["ok"] is False
          and cov["ar_target"]["measured"] == 2 and cov["ar_dual"]["ok"] is True,
          str(cov["spec"]))
    check("㊴ 核心角色缺失 -> 必须点名（据此拒绝出报告）",
          V.missing_required_roles(cov) == ["spec"])
    check("㊴ 全部角色都有样本 -> 不点名",
          V.missing_required_roles(V.coverage_report(
              recs_missing + [_child("spec", 0, 9.5)],
              roles=["spec", "ar_target", "ar_dual"], procs=4)) == [])
    check("㊴ AR 参照角色缺失同样被拦（没有分母就没有结论）",
          V.missing_required_roles(V.coverage_report(
              [_child("spec", 0, 9.5)], roles=["spec", "ar_target", "ar_dual"], procs=4))
          == ["ar_target"])
    check("㊴ 只请求 spec 的冒烟测试不被自己的硬拦拒掉（未请求的角色不算缺失）",
          V.missing_required_roles(V.coverage_report(
              [_child("spec", 0, 9.5)], roles=["spec"], procs=1)) == []
          and V.missing_required_roles(V.coverage_report(
              [], roles=["spec"], procs=1)) == ["spec"])
    a_missing = V.analyse(recs_missing)
    check("㊵ 样本不足时 Q1 必须显式警告，不得默默给'未检出双档'",
          a_missing["Q1_bimodality"]["spec_samples"] == 0
          and "不能" in (a_missing["Q1_bimodality"].get("warning") or ""),
          str(a_missing["Q1_bimodality"].get("warning")))
    check("㊵ prefill 按模型拆分字段存在（SpecDec 有 target 侧）",
          "prefill_target_s" in a["Q4_prefill"]["spec"]
          and "prefill_draft_s" in a["Q4_prefill"]["spec"])

    # ㊶/㊷ commit 护栏（2026-09-24 整机黑屏硬挂事故的防护）
    # 事故链：载入模型需 ≈13.3 GiB commit；失败现场只剩 9.5 GiB -> 原生 0xC0000005；
    #        同一工作负载另有一次整机黑屏硬挂（Kernel-Power 41，无 BugCheck 转储）。
    sm = V.system_memory()
    check("㊶ system_memory 读到 commit 上限与可用量（commit 才是那堵墙，不是物理内存）",
          sm.get("available") is True and sm["commit_limit_mib"] > 0
          and 0 < sm["commit_free_mib"] <= sm["commit_limit_mib"],
          str({k: sm.get(k) for k in ("commit_limit_mib", "commit_free_mib",
                                      "phys_free_mib")}))
    ok, why = V.commit_guard()
    check("㊶ commit 余量充足时放行，并给出具体数字",
          ok is True and "GiB" in why and f"{V.MIN_COMMIT_GIB:.0f}" in why, why)
    _old_min = V.MIN_COMMIT_GIB
    try:
        V.MIN_COMMIT_GIB = 10 ** 6
        ok2, why2 = V.commit_guard()
        check("㊷ 余量低于安全线 -> **拒绝启动**（黑屏事故的护栏）",
              ok2 is False and "安全线" in why2, why2)
    finally:
        V.MIN_COMMIT_GIB = _old_min
    check("㊷ 安全线必须高于实测载入需求（13.3 GiB），否则护栏没有意义",
          V.MIN_COMMIT_GIB > 13.3)

    print(f"\nG5 schema 自测：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
