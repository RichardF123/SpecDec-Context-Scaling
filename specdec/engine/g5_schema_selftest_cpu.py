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
import tempfile
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
