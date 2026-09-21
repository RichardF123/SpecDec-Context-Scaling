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

    print("\n== 9. 跨进程 token 比较 ==")
    rows = [dict(ctx=8192, target_kv="base_bf16", mode="AR", order="A", gen_sha1="aa"),
            dict(ctx=8192, target_kv="base_bf16", mode="SpecDec", order="A", gen_sha1="aa"),
            dict(ctx=8192, target_kv="v_fp8", mode="AR", order="A", gen_sha1="bb"),
            dict(ctx=8192, target_kv="v_fp8", mode="SpecDec", order="A", gen_sha1="cc")]
    te = M.cross_process_token_equal(rows)
    check("同档相等判 True", te.get((8192, "base_bf16", "A")) is True)
    check("同档不等判 False", te.get((8192, "v_fp8", "A")) is False)

    print("\n== 10. 报表措辞：subset 不得打印 12 conditions ==")
    summary = {(8192, "base_bf16", "AR"): M.summarize([fake_run(kv="base_bf16", mode="AR")]),
               (8192, "base_bf16", "SpecDec"): M.summarize([fake_run(kv="base_bf16", mode="SpecDec")]),
               (8192, "v_fp8", "AR"): M.summarize([fake_run(kv="v_fp8", mode="AR")]),
               (8192, "v_fp8", "SpecDec"): M.summarize([fake_run(kv="v_fp8", mode="SpecDec")])}
    for s in summary.values():
        s["oom"] = False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.print_report(summary, [8192], ("base_bf16", "v_fp8"))
    out = buf.getvalue()
    check("打印 '实际运行 4 conditions'", "实际运行 4 conditions" in out)
    check("未打印 formal 12-condition", "formal 12-condition" not in out)
    check("未打印 exploratory 12-condition", "exploratory 12-condition" not in out)
    check("明确声明本 subset 不外推", "本 subset 不做 per-family 外推" in out)
    check("外推假设与实测分开显示", "纯假设，不是测量值" in out)

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
