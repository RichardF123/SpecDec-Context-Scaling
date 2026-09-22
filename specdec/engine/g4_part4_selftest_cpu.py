"""g4_part4_selftest_cpu.py — Part 4 测量工具链的 **CPU 自测**（不加载模型、不碰 GPU）。

覆盖（对应本轮规格 §11）
  1. `P4_KV` / `P4_MODE` 条件过滤与非法值拒绝
  2. 实际 condition count 计算
  3. subset **不冒充** 12 条件（标签里不得出现 "12-condition"）
  4. 只有完整正式集合（ctx={2048,4096,8192}）才允许 formal per-family 外推
  5. 同名输出**不被静默覆盖**（resolve_jsonl 在既有文件上必须报错，除非同 RUN_ID 组）
  6. summary 含 `mismatch_rounds_med` / `all_accept_rounds_med`（历史 KeyError 回归）
  7. 监控字段缺失时 `available=false`，**不得用 0 冒充**
  8. rel_spread 计算与 >20% → CHECK 规则
  9. `cross_process_token_equal` 跨进程 token 比较
用法：python -m specdec.engine.g4_part4_selftest_cpu
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import traceback
from pathlib import Path

from . import g4_part4_pilot_measure as M

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def fake_run(ctx=8192, kv="base_bf16", mode="AR", dec_s=1.0, gen_hash="h1", n_gen=64):
    return dict(mode=mode, target_kv=kv, ctx=ctx, n_gen=n_gen, gen_sha1=gen_hash,
                t_prefill_s=1.0, t_decode_s=dec_s, t_total_s=1.0 + dec_s,
                peak_alloc_GiB=6.0, peak_resv_GiB=8.0, weights_resident_GiB=3.8,
                prefill_logits_finite=True, step_logits_finite=True,
                decode_tok_s=(n_gen / dec_s if dec_s else float("nan")),
                e2e_tok_s=n_gen / (1.0 + dec_s),
                accept_rate=0.7 if mode == "SpecDec" else float("nan"),
                rounds=20 if mode == "SpecDec" else 0,
                proposed=76 if mode == "SpecDec" else 0,
                accepted=56 if mode == "SpecDec" else 0,
                target_block_calls=20 if mode == "SpecDec" else 0,
                draft_calls=76 if mode == "SpecDec" else 0,
                mismatch_rounds=8 if mode == "SpecDec" else 0,
                all_accept_rounds=12 if mode == "SpecDec" else 0,
                target_decode_calls=63 if mode == "AR" else 0,
                m_hist=[4] * 12 + [1] * 8 if mode == "SpecDec" else [],
                order="A", run_id="t")


def main() -> int:
    print("== 1/2. 条件过滤与实际 condition count ==")
    ctxs, kvs, modes = M.parse_filters(None, None, [8192])
    conds = M.build_matrix(ctxs, kvs, modes)
    check("无过滤：1 ctx × 2 kv × 2 mode = 4 conditions", len(conds) == 4, str(conds))
    ctxs, kvs, modes = M.parse_filters(None, None, [512, 2048, 8192])
    check("无过滤 + 3 ctx = 12 conditions", len(M.build_matrix(ctxs, kvs, modes)) == 12)
    ctxs, kvs, modes = M.parse_filters("base_bf16", "AR", [8192])
    conds = M.build_matrix(ctxs, kvs, modes)
    check("P4_KV+P4_MODE 过滤 -> 1 condition", conds == [(8192, "base_bf16", "AR")], str(conds))
    ctxs, kvs, modes = M.parse_filters("v_fp8", None, [2048, 8192])
    check("P4_KV 过滤 -> 2 conditions",
          M.build_matrix(ctxs, kvs, modes) == [(2048, "v_fp8", "AR"), (2048, "v_fp8", "SpecDec"),
                                               (8192, "v_fp8", "AR"), (8192, "v_fp8", "SpecDec")])
    ctxs, kvs, modes = M.parse_filters(None, "SpecDec", [8192], reverse=True)
    check("P4_REVERSE 只影响 kv 顺序", [c[1] for c in M.build_matrix(ctxs, kvs, modes)]
          == ["v_fp8", "base_bf16"])
    for bad, what in ((("bogus", None), "P4_KV"), ((None, "Sampling"), "P4_MODE")):
        try:
            M.parse_filters(bad[0], bad[1], [8192])
            check(f"非法 {what} 必须报错", False)
        except SystemExit:
            check(f"非法 {what} 必须报错", True)

    print("\n== 3/4. subset 标签与外推许可 ==")
    lab, ok = M.classify_matrix([8192], ["base_bf16", "v_fp8"], ["AR", "SpecDec"], 4)
    check("8K 4 条件标签不含 '12-condition'", "12-condition" not in lab, lab)
    check("8K 4 条件不允许外推", ok is False and "4-condition" in lab, lab)
    lab, ok = M.classify_matrix([2048, 4096, 8192], ["base_bf16", "v_fp8"], ["AR", "SpecDec"], 12)
    check("正式集合 -> formal 12-condition 且允许外推",
          ok is True and lab == "formal 12-condition per-family cost", lab)
    lab, ok = M.classify_matrix([512, 2048, 8192], ["base_bf16", "v_fp8"], ["AR", "SpecDec"], 12)
    check("探索集合 -> exploratory 12-condition 且允许外推",
          ok is True and lab == "exploratory 12-condition cost", lab)
    lab, ok = M.classify_matrix([8192], ["v_fp8"], ["AR"], 1)
    check("单条件 subset -> 允许?" and "subset" in lab, ok is False, lab)

    print("\n== 5. 同名输出不覆盖 ==")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        p = M.resolve_jsonl(tdp, "deadbeef", "run1", append_ok=False)
        check("首次定名成功", p.name == "part4_deadbeef_run1.jsonl", p.name)
        p.write_text("x", encoding="utf-8")
        try:
            M.resolve_jsonl(tdp, "deadbeef", "run1", append_ok=False)
            check("既有文件 + 非同组 -> 必须拒绝", False)
        except SystemExit:
            check("既有文件 + 非同组 -> 必须拒绝", True)
        try:
            p2 = M.resolve_jsonl(tdp, "deadbeef", "run1", append_ok=True)
            check("同 RUN_ID 组内 -> 允许追加", p2 == p)
        except SystemExit:
            check("同 RUN_ID 组内 -> 允许追加", False)

    print("\n== 6. summary 字段完整性（历史 KeyError 回归）==")
    s = M.summarize([fake_run(mode="SpecDec"), fake_run(mode="SpecDec", dec_s=1.2)])
    for k in ("mismatch_rounds_med", "all_accept_rounds_med", "target_block_calls_med",
              "draft_calls_med", "accept_rate_med", "t_decode_spread",
              "n_gen_consistent", "gen_sha1_consistent", "stats_consistent",
              "logits_finite", "spread_flag"):
        check(f"summary 含 {k}", k in s)
    check("accept_rate_med = 各 rep accept_rate 的中位数(0.7)",
          abs(s["accept_rate_med"] - 0.7) < 1e-9, str(s["accept_rate_med"]))
    check("summary 保留原始计数以便重算接受率 (56/76)",
          s["accepted_med"] == 56 and s["proposed_med"] == 76
          and abs(s["accepted_med"] / s["proposed_med"] - 56 / 76) < 1e-12)

    print("\n== 7. 监控字段缺失 -> available=false（不得填 0）==")
    real_run = M.subprocess.run

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")
    M.subprocess.run = boom
    try:
        snap = M.nvml_snapshot()
    finally:
        M.subprocess.run = real_run
    check("nvidia-smi 不存在 -> available=False", snap.get("available") is False, str(snap)[:80])
    check("缺失字段未用 0 冒充", "mem_used_MiB" not in snap and "temp_C" not in snap)
    check("附带 error 说明", "error" in snap)
    alloc = M.allocator_info()
    check("allocator 信息含 empty_cache_in_timed_region=False",
          alloc.get("empty_cache_in_timed_region") is False)

    print("\n== 8. rel_spread 与 >20% CHECK 规则 ==")
    r = M.rel_spread([1.0, 1.05, 1.1])
    check("rel_spread 计算", abs(r["spread"] - 0.1) < 1e-9 and abs(r["rel_spread_pct"] - 9.52) < 0.1,
          str(r))
    s_ok = M.summarize([fake_run(dec_s=1.0), fake_run(dec_s=1.02), fake_run(dec_s=1.01)])
    check("稳定条件 flag=ok", s_ok["spread_flag"] == "ok", s_ok["spread_flag"])
    s_bad = M.summarize([fake_run(dec_s=1.0), fake_run(dec_s=2.0), fake_run(dec_s=1.9)])
    check("不稳定条件 flag=CHECK", "CHECK" in s_bad["spread_flag"], s_bad["spread_flag"])

    print("\n== 9. 跨条件 token 比较（token_compare / compare_across_conditions）==")
    base = dict(ctx=8192, max_new=64, eos_id=3, input_token_hash="in1", target_kv="base_bf16")
    ar_ok = dict(base, n_generated=64, generated_token_hash="aa",
                 generated_token_ids=[1] * 64, mode="AR", order="A")
    sp_ok = dict(base, n_generated=64, generated_token_hash="aa",
                 generated_token_ids=[1] * 64, mode="SpecDec", order="A")
    c = M.token_compare(ar_ok, sp_ok)
    check("相等 + 前提齐全 -> ar_equal True 且 certified", c["ar_equal"] is True and c["certified"])
    sp_bad = dict(base, n_generated=64, generated_token_hash="bb",
                  generated_token_ids=[1] * 63 + [9], mode="SpecDec", order="A")
    c = M.token_compare(ar_ok, sp_bad)
    check("不等 -> first_diff_index=63 且两侧 token id",
          c["ar_equal"] is False and c["first_diff_index"] == 63
          and c["ar_token_id"] == 1 and c["spec_token_id"] == 9, str(c["first_diff_index"]))
    c = M.token_compare(ar_ok, None)
    check("缺 AR 对照 -> ar_equal=None 且 available=False（绝不 False）",
          c["ar_equal"] is None and c["ar_equal_available"] is False and c["first_diff_index"] is None)
    c = M.token_compare(dict(ar_ok, input_token_hash=None), sp_ok)
    check("input_token_hash 缺失 -> 仍报相等但 certified=False 并给出原因",
          c["ar_equal"] is True and c["certified"] is False and "input_token_hash" in c["certified_reason"])
    c = M.token_compare(ar_ok, dict(sp_ok, ctx=512))
    check("ctx 不同 -> 前提不满足被点名", "ctx_equal" in c["certified_reason"])
    c = M.token_compare(dict(ar_ok, generated_token_ids=None), dict(sp_bad, generated_token_ids=None))
    check("缺 ids -> 不给 first_diff_index 并说明原因",
          c["first_diff_index"] is None and "generated_token_ids" in c["certified_reason"])
    pairs = M.compare_across_conditions([ar_ok, sp_bad, dict(ar_ok, target_kv="v_fp8"),
                                         dict(sp_bad, target_kv="v_fp8")])
    check("compare_across_conditions 按 (ctx,kv,order) 配对",
          len(pairs) == 2 and all(v["ar_equal_available"] for v in pairs.values()))

    print("\n== 9b. 进程内等价性接缝（2026-09-22 canonical ar_equal=False 事故回归）==")
    # 事故根因：驱动器"一条件一进程"只给单一 P4_MODE，SpecDec 进程里 ref_gen 为空，
    # 旧代码仍拿 gen_sha1([]) 去比 -> ar_equal=False 且 certified=True（凭空否定）。
    sp_runs = [dict(n_gen=64, generated_token_hash=M.gen_sha1([1] * 64),
                    generated_token_ids=[1] * 64, input_token_hash="in1")]
    empty = M.in_process_equality({}, 8192, "base_bf16", sp_runs, 3, max_new=64)
    check("ref_gen 为空 -> ar_equal=None、available=False、certified=False（绝不 False）",
          empty["ar_equal"] is None and empty["ar_equal_available"] is False
          and empty["certified"] is False, str(empty["ar_equal"]))
    check("ref_gen 为空 -> 给出'本进程无对照'的原因且 equality_source 为 None",
          "本进程未运行同档 AR 对照" in (empty["certified_reason"] or "")
          and empty["equality_source"] is None)
    check("ref_gen 有其它 (ctx,kv) 的 AR 也不算对照",
          M.in_process_equality({(2048, "base_bf16"): [1] * 64}, 8192, "base_bf16",
                                sp_runs, 3, max_new=64)["ar_equal"] is None)
    same = M.in_process_equality({(8192, "base_bf16"): [1] * 64}, 8192, "base_bf16",
                                 sp_runs, 3, max_new=64)
    check("ref_gen 有同档 AR 且序列相同 -> True 且 certified",
          same["ar_equal"] is True and same["certified"] is True
          and same["equality_source"] == "in_process")
    diff_runs = [dict(sp_runs[0], generated_token_hash=M.gen_sha1([1] * 63 + [9]),
                      generated_token_ids=[1] * 63 + [9])]
    diff = M.in_process_equality({(8192, "base_bf16"): [1] * 64}, 8192, "base_bf16",
                                 diff_runs, 3, max_new=64)
    check("ref_gen 有同档 AR 但序列不同 -> False 且有 first_diff_index",
          diff["ar_equal"] is False and diff["first_diff_index"] == 63)
    check("原因拼接不再产生以 ' | ' 开头的半截字符串（旧版 certified_reason=' | 缺 ...'）",
          not any(str(M.in_process_equality({(8192, "base_bf16"): [1] * 64}, 8192, "base_bf16",
                                             [dict(sp_runs[0], generated_token_ids=None)], 3
                                             ).get("certified_reason") or "").startswith(" | ")
                  for _ in range(1)))

    print("\n== 10. 报表措辞：subset 不得打印 12 conditions；==AR 不得误报 False ==")
    summary = {}
    for kv in ("base_bf16", "v_fp8"):
        for mode in ("AR", "SpecDec"):
            s = M.summarize([fake_run(kv=kv, mode=mode)])
            s["oom"] = False
            summary[(8192, kv, mode)] = s
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.print_report(summary, [8192], ("base_bf16", "v_fp8"))
    out = buf.getvalue()
    check("打印 '实际运行 4 conditions'", "实际运行 4 conditions" in out)
    check("未打印 formal 12-condition", "formal 12-condition" not in out)
    check("未打印 exploratory 12-condition", "exploratory 12-condition" not in out)
    check("明确声明本 subset 不外推", "本 subset 不做 per-family 外推" in out)
    check("外推假设与实测分开显示", "纯假设，不是测量值" in out)
    spec_lines = [ln for ln in out.splitlines()
                  if re.match(r"^8192\s+\S+\s+SpecDec\s", ln)]   # 只取汇总表的 SpecDec 行
    check("单条件缺 AR 对照时列显示 n/a（不得显示 False）",
          bool(spec_lines) and all("n/a" in ln and "False" not in ln for ln in spec_lines),
          spec_lines[0] if spec_lines else "(无 SpecDec 汇总行)")

    print("\n== 11. 有限值聚合（aggregate_finite）三种情形 ==")
    a = M.aggregate_finite([3.0, float("nan"), None, "x", 9.0])
    check("有效+非有限混合 -> 只用有效值并记录剔除数",
          a["available"] and a["n"] == 2 and a["dropped_nonfinite"] == 3
          and a["min"] == 3.0 and a["max"] == 9.0, str(a))
    a = M.aggregate_finite([float("nan"), None, "n/a"])
    check("全部缺失 -> available=False 且 median=None（不填 0/NaN）",
          a["available"] is False and a["median"] is None and a["n"] == 0, str(a))
    a = M.aggregate_finite([7.5])
    check("仅一个有效值 -> median=min=max=7.5 且 rel=0", a["available"] and a["median"] == 7.5
          and a["max_rel_spread_pct"] == 0.0, str(a))
    a = M.aggregate_finite([])
    check("空列表 -> available=False", a["available"] is False)

    print("\n== 12. 跨顺序稳定性（order_stability）==")
    st = M.order_stability({"A": 5.99, "B": 8.42}, 20.0)
    check("跨顺序 28.9% -> CHECK 且 order_ratio 正确",
          st["available"] and "CHECK" in st["flag"]
          and abs(st["relative_order_difference_pct"] - 28.9) < 0.2
          and abs(st["order_ratio"] - 5.99 / 8.42) < 1e-6, str(st))
    st = M.order_stability({"A": 22.90, "B": 23.78}, 20.0)
    check("跨顺序 3.7% -> ok", st["available"] and st["flag"] == "ok", str(st))
    st = M.order_stability({"A": 5.0}, 20.0)
    check("只有一个顺序 -> available=False", st["available"] is False)
    st = M.order_stability({"A": 5.0, "B": float("nan")}, 20.0)
    check("含 NaN -> 视为只有一个有效顺序", st["available"] is False)

    print("\n== 13. nvidia-smi 解析与旧版字符串兼容 ==")
    snap = M.nvml_snapshot()
    check("真实快照可用或明确 unavailable",
          snap.get("available") in (True, False) and ("error" in snap or snap["available"]))
    if snap.get("available"):
        check("数值字段是 float 而非字符串",
              isinstance(snap.get("temp_C"), (float, type(None)))
              and isinstance(snap.get("mem_used_MiB"), (float, type(None))))

    def fake_smi(*a, **k):
        class R:
            returncode = 0
            stdout = "NVIDIA RTX, 7875, 8188, [N/A], 42.99, 2520, 8000, 555.97\n"
            stderr = ""
        return R()
    real = M.subprocess.run
    M.subprocess.run = fake_smi
    try:
        s2 = M.nvml_snapshot()
    finally:
        M.subprocess.run = real
    check("[N/A] -> None 且点名 unparsed_fields（不填 0）",
          s2["temp_C"] is None and "temp_C" in s2["unparsed_fields"]
          and s2["power_W"] == 42.99, str(s2))

    import specdec.engine.g4_part4_8k_isolated_probe as P
    legacy = dict(target_kv="v_fp8", mode="AR", order="A",
                  monitor_start={"available": True, "mem_used_MiB": "7875", "temp_C": "71"})
    check("probe.monitor 兼容旧版字符串并转 float",
          P.monitor(legacy, "start", "mem_used_MiB") == 7875.0
          and P.monitor(legacy, "start", "temp_C") == 71.0)
    check("probe.monitor 对不可解析值返回 None",
          P.monitor(dict(legacy, monitor_start={"available": True, "temp_C": "[N/A]"}),
                    "start", "temp_C") is None)
    check("probe.monitor 对 available=False 返回 None",
          P.monitor({"monitor_start": {"available": False, "temp_C": "71"}}, "start", "temp_C") is None)

    print(f"\nPart4 自测：PASS {len(PASS)} / FAIL {len(FAIL)}")
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
