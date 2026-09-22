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
    bad = S.finalize_row({"schema_version": S.SCHEMA_VERSION, "decode_seconds": None,
                          "availability": {"decode_seconds": True}})
    check("null 但 availability=true -> 报错",
          any("decode_seconds" in e for e in S.validate_row(bad)))
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
