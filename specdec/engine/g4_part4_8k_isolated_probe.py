"""g4_part4_8k_isolated_probe.py — 8K 四条件的**独立进程**复验 + 离线报表。

为什么必须独立进程：8 GB 卡上 8K 双模型时分配器池会越过物理显存，池状态会显著改变同一格耗时
（历史实测同格 6.44s ↔ 10.87s）。每格从干净、确定的进程状态开始，格间才可比。

编排（`--run`，默认）
  顺序 A：base AR → v_fp8 AR → base Spec → v_fp8 Spec
  顺序 B：v_fp8 AR → base AR → v_fp8 Spec → base Spec
  每条件一个全新 Python 子进程：重载双模型（**AR 也保持双模型常驻**）、warm-up=1、reps=5、
  max_new=64、chunk=2048；退出即释放全部 CUDA 状态。

离线报表（`--report [run_id|latest]`，不碰 GPU）
  1. **within-run**：每条件 5 rep 的 median/min/max/spread/relative spread
  2. **between-order**：同一 (ctx,kv,mode) 跨顺序的 order_ratio 与 relative_order_difference
     （>20% → CHECK）；与 within-run 分开列出，**两种不确定性不得混为一谈**
  3. **AR vs SpecDec token 比较**：由父进程跨子进程按 input_token_hash/ctx/max_new/eos 校验后
     比较生成序列；缺对照时输出 `ar_equal=null, available=false`，**绝不填 False**
  4. **AR rel_spread 聚合**：只聚合有限值，全缺失则 `available=false`（不填 0/NaN）
  5. **显存口径审计**：peak_alloc / peak_resv / nvidia-smi used 三者来源与并列值
  6. 受保护历史文件 sha256 前后校验

判据（跑之前固定）
  - `v_fp8 AR 优势稳定`：两轮顺序都 v_fp8 > base；AR 各格 within-run rel ≤20%；
    between-order diff ≤20%；token/n_gen/stats 一致；温度/功耗/时钟无足以解释差异的漂移
  - 原 1.286×（历史 8K base Spec/AR）只有在 fresh process 中重新 >1 才可称复现

用法（交互式 GPU 终端；预计 25–40 分钟）
  python -m specdec.engine.g4_part4_8k_isolated_probe                # 跑 + 报表
  python -m specdec.engine.g4_part4_8k_isolated_probe --report latest # 只出报表
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import g4_part4_pilot_measure as M

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "part4"
PROTECTED = {
    "exploratory_run_1_fixed_order": OUTDIR / "part4_dc0bd5d87389.jsonl",
    "exploratory_run_2_reverse_order": OUTDIR / "part4_ba0c985bcd5d.jsonl",
}
CTX = 8192
REPS = 5
WARM = 1
MAXNEW = 64
CHUNK = 2048
REL_LIMIT = 20.0

PLANS = {
    "A": [("base_bf16", "AR"), ("v_fp8", "AR"), ("base_bf16", "SpecDec"), ("v_fp8", "SpecDec")],
    "B": [("v_fp8", "AR"), ("base_bf16", "AR"), ("v_fp8", "SpecDec"), ("base_bf16", "SpecDec")],
}
RUN1_BASE_SPEC_OVER_AR = 1.286      # 历史数字，仅用于"是否复现"对照

RUN_RE = re.compile(r"^part4_[0-9a-f]+_8k(?P<order>[AB])_(?P<stamp>\d{8}T\d{6}Z)\.jsonl$")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def hashes() -> dict:
    return {k: sha256_file(v) for k, v in PROTECTED.items()}


def collect(stamp: str | None = None):
    """收集 run/summary 行。stamp=None -> 全部 8k* 文件。"""
    runs, sums, files = [], [], []
    for f in sorted(OUTDIR.glob("part4_*_8k*.jsonl")):
        m = RUN_RE.match(f.name)
        if not m or (stamp and m.group("stamp") != stamp):
            continue
        files.append(f.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "run":
                r["_file"] = f.name
                runs.append(r)
            elif r.get("kind") == "summary":
                r["_file"] = f.name
                sums.append(r)
    return runs, sums, files


def by_condition(runs):
    out: dict = {}
    for r in runs:
        out.setdefault((r.get("order"), r.get("target_kv"), r.get("mode")), []).append(r)
    return out


def monitor(r, when, key):
    """取 nvidia-smi 快照字段为 float。旧版 harness 把 numinits 结果存成了字符串，这里做一次
    数值强转；强转失败（"[N/A]" 等）返回 None —— 绝不填 0。"""
    d = r.get(f"monitor_{when}") or {}
    if not d.get("available"):
        return None
    v = d.get(key)
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ 报表
def report_within_process(runs) -> dict:
    print("\n== [1] within-run 稳定性（每条件一个独立进程，5 rep）==")
    print("order kv         mode     n  dec_tok/s med[min..max]   rel%  dec_s med  pref_s med  "
          "n_gen同 hash同 stats同 finite")
    stats = {}
    for (order, kv, mode), rs in sorted(by_condition(runs).items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        d = M.rel_spread([r.get("decode_tok_s") for r in rs])
        ds = M.rel_spread([r.get("t_decode_s") for r in rs])
        ps = M.rel_spread([r.get("t_prefill_s") for r in rs])
        stats[(order, kv, mode)] = dict(decode=d, t_decode=ds, t_prefill=ps)
        same = (len({r.get("n_gen") for r in rs}) == 1,
                len({r.get("generated_token_hash") or r.get("gen_sha1") for r in rs}) == 1,
                len({tuple(r.get(k) for k in ("rounds", "proposed", "accepted",
                                              "target_block_calls", "draft_calls",
                                              "mismatch_rounds", "all_accept_rounds"))
                     for r in rs}) == 1,
                all(bool(r.get("prefill_logits_finite")) and bool(r.get("step_logits_finite", True))
                    for r in rs))
        print(f"{order:<5} {kv:<10} {mode:<8} {d['n']}  "
              f"{d['median']:.2f}[{d['min']:.2f}..{d['max']:.2f}]  "
              f"{(d['rel_spread_pct'] or 0):5.1f}  {ds['median']:8.2f}  {ps['median']:9.2f}  "
              f"{str(same[0]):<7} {str(same[1]):<6} {str(same[2]):<7} {same[3]}")
    return stats


def report_between_order(stats) -> dict:
    print("\n== [2] between-order 稳定性（同一条件的两个启动顺序）==")
    print("kv         mode     med_A(tok/s)  med_B(tok/s)  order_ratio  rel_order_diff%  flag")
    out = {}
    keys = sorted({(k[1], k[2]) for k in stats})
    for kv, mode in keys:
        a = stats.get(("A", kv, mode), {}).get("decode", {}).get("median")
        b = stats.get(("B", kv, mode), {}).get("decode", {}).get("median")
        st = M.order_stability({"A": a, "B": b}, REL_LIMIT)
        out[(kv, mode)] = st
        if not st.get("available"):
            print(f"{kv:<10} {mode:<8} {str(a):>12}  {str(b):>12}  "
                  f"{'n/a':>11}  {'n/a':>14}  unavailable({st.get('reason')})")
            continue
        print(f"{kv:<10} {mode:<8} {a:>12.2f}  {b:>12.2f}  {st['order_ratio']:>11.3f}  "
              f"{st['relative_order_difference_pct']:>14.1f}  {st['flag']}")
    return out


def report_token_comparison(runs):
    comp = M.compare_across_conditions(runs)
    print("\n== [3] AR vs SpecDec token 比较（跨子进程；校验 ctx/input-hash/max_new/eos）==")
    if not comp:
        print("  available=false（未找到可配对的 AR+SpecDec 条件）")
        return comp
    for (ctx, kv, order), c in comp.items():
        print(f"  ctx={ctx} {kv:<10} order={order}: AR {c['ar_len']} tok / SpecDec {c['spec_len']} tok"
              f"  ar_equal={c['ar_equal']}  available={c['ar_equal_available']}"
              f"  certified={c['certified']}")
        if c.get("first_diff_index") is not None:
            print(f"      first_diff_index={c['first_diff_index']} "
                  f"AR_token_id={c['ar_token_id']} SpecDec_token_id={c['spec_token_id']}")
        if c.get("preconditions"):
            bad = [k for k, v in c["preconditions"].items() if not v]
            print(f"      preconditions: {'全部满足' if not bad else '不满足=' + ','.join(bad)}")
        if c.get("certified_reason"):
            print(f"      note: {c['certified_reason']}")
        if c.get("ar_hash"):
            print(f"      hash: AR={c['ar_hash']} SpecDec={c['spec_hash']}")
    return comp


def report_ar_aggregation(runs):
    print("\n== [4] AR 的 rel_spread 聚合（只取有限值；全缺失 -> available=false）==")
    vals, flags = [], []
    for (order, kv, mode), rs in by_condition(runs).items():
        if mode != "AR":
            continue
        d = M.rel_spread([r.get("decode_tok_s") for r in rs])
        vals.append(d["rel_spread_pct"])
        flags.append(f"{order}/{kv}={d['rel_spread_pct']}")
    agg = M.aggregate_finite(vals)
    print(f"  各格值: {', '.join(flags) if flags else '(无 AR 条件)'}")
    if agg["available"]:
        print(f"  available=True  n={agg['n']}  (剔除非有限值 {agg['dropped_nonfinite']} 个)  "
              f"median={agg['median']:.2f}%  min={agg['min']:.2f}%  max={agg['max']:.2f}%")
        print(f"  => AR within-run 最大 rel_spread = {agg['max']:.2f}%  "
              f"({'通过 <=20%' if agg['max'] <= REL_LIMIT else 'CHECK >20%'})")
    else:
        print(f"  available=false（无有限 rel_spread 可聚合；剔除非有限值 "
              f"{agg['dropped_nonfinite']} 个）—— 不填 0/NaN")
    return agg


def fmt_or_na(agg: dict, spec: str = "{:.0f}") -> str:
    return spec.format(agg["median"]) if agg.get("available") else "n/a"


def report_memory_audit(runs):
    print("\n== [5] 显存口径审计 ==")
    print("  代码路径：peak_alloc/peak_resv 由 mem_read() 直接读 "
          "torch.cuda.max_memory_allocated()/max_memory_reserved()，在 run_once() 内 "
          "reset_peak_memory_stats() 之后单次读取；**无相加、无跨阶段累计、不区分 target/draft"
          "（同一进程同一分配器池）**。")
    print("  口径含义：peak_alloc = PyTorch 实际张量分配（主指标）；peak_resv = 分配器池峰值"
          "（含空闲块/虚拟预留，≠物理显存）；nvidia-smi used = 驱动观察的物理占用（辅助）。")
    print("  order kv         mode     peak_alloc  peak_resv  resv>=alloc  smi_used_MiB  "
          "smi_total_MiB  resv>8192MiB?")
    for (order, kv, mode), rs in sorted(by_condition(runs).items(),
                                        key=lambda x: (x[0][0], x[0][1], x[0][2])):
        pa = M.rel_spread([r.get("peak_alloc_GiB") for r in rs])["median"]
        pr = M.rel_spread([r.get("peak_resv_GiB") for r in rs])["median"]
        smi = M.aggregate_finite([monitor(r, "end", "mem_used_MiB") for r in rs])
        tot = M.aggregate_finite([monitor(r, "end", "mem_total_MiB") for r in rs])
        print(f"  {order:<5} {kv:<10} {mode:<8} {pa:>10.2f}  {pr:>9.2f}  {str(pr >= pa):<12}  "
              f"{fmt_or_na(smi):>12}  {fmt_or_na(tot):>13}  {pr * 1024 > 8192}")
    print("  !! peak_resv > 8 GiB **不得**解释为物理占用超过 8 GB：它是分配器池"
          "（含空闲块与虚拟预留）。物理口径以 nvidia-smi used 为准；主指标用 peak_alloc。")
    # 物理显存饱和检查：cuda_free=0 时任何新分配都可能触发 WDDM 换页
    free = M.aggregate_finite([(r.get("cuda_start") or {}).get("cuda_free_GiB") for r in runs])
    used = M.aggregate_finite([monitor(r, "start", "mem_used_MiB") for r in runs])
    tot = M.aggregate_finite([monitor(r, "start", "mem_total_MiB") for r in runs])
    if free["available"] and used["available"] and tot["available"]:
        pct = 100.0 * used["max"] / tot["median"]
        print(f"  物理显存现场：条件开始时 cuda_free min={free['min']:.2f} GiB；"
              f"nvidia-smi used max={used['max']:.0f} MiB / total {tot['median']:.0f} MiB "
              f"= {pct:.1f}%")
        if free["min"] <= 0.05 or pct >= 95:
            print("  => **物理显存已饱和（free≈0 / used≥95%）**：此时任何新增分配都可能触发 "
                  "WDDM 换页，8K 档的耗时被显存状态主导 ⇒ 该运行点必须按此解释，"
                  "不得当作纯计算加速结论。")
    else:
        print(f"  物理显存现场：available=false（free={free['available']}, "
              f"used={used['available']}, total={tot['available']}）")


def report_monitor(runs):
    print("\n== [6] 环境监控（缺失字段一律 available=false）==")
    r0 = runs[0] if runs else {}
    m0 = r0.get("monitor_start") or {}
    if m0.get("available"):
        print(f"  nvidia-smi: {m0.get('gpu_name')}  driver={m0.get('driver')}  "
              f"mem_total={m0.get('mem_total_MiB')} MiB")
    else:
        print(f"  nvidia-smi: available=false  reason={m0.get('error')}")
    print("  order kv         mode     temp°C start→end   power_W end  sm_MHz end  mem_MHz end")
    for (order, kv, mode), rs in sorted(by_condition(runs).items(),
                                        key=lambda x: (x[0][0], x[0][1], x[0][2])):
        agg = lambda key, when: M.aggregate_finite([monitor(r, when, key) for r in rs])  # noqa: E731
        t0, t1 = agg("temp_C", "start"), agg("temp_C", "end")
        pw, sm, mm = agg("power_W", "end"), agg("clock_sm_MHz", "end"), agg("clock_mem_MHz", "end")
        print(f"  {order:<5} {kv:<10} {mode:<8} "
              f"{fmt_or_na(t0):>6}→{fmt_or_na(t1):<6}  {fmt_or_na(pw, '{:.1f}'):>11}  "
              f"{fmt_or_na(sm):>10}  {fmt_or_na(mm):>11}")


def report_verdict(stats, order):
    print("\n== [7] 判定（判据在脚本头部预先固定）==")
    ar_med = {}
    for o in ("A", "B"):
        for kv in ("base_bf16", "v_fp8"):
            ar_med[(o, kv)] = stats.get((o, kv, "AR"), {}).get("decode", {}).get("median")
    for o in ("A", "B"):
        b, v = ar_med.get((o, "base_bf16")), ar_med.get((o, "v_fp8"))
        if b and v:
            print(f"  顺序 {o}: v_fp8/base AR = {v / b:.3f}×")
    st = order.get(("v_fp8", "AR"), {})
    if st.get("available"):
        print(f"  v_fp8 AR 跨顺序 relative_order_difference = "
              f"{st['relative_order_difference_pct']:.1f}% -> {st['flag']}")
        print(f"  => '8K v_fp8 AR 稳定优势' 判定："
              f"{'两顺序方向一致且跨顺序 diff<=20%' if st['relative_order_difference_pct'] <= REL_LIMIT else '跨顺序不稳定，不成立'}"
              f"（仍须结合 within-run rel 与温度/时钟漂移）")
    else:
        print(f"  v_fp8 AR 跨顺序稳定性 = unavailable（{st.get('reason')}）")
    for o in ("A", "B"):
        for kv in ("base_bf16", "v_fp8"):
            a = stats.get((o, kv, "AR"), {}).get("decode", {}).get("median")
            s = stats.get((o, kv, "SpecDec"), {}).get("decode", {}).get("median")
            if a and s:
                print(f"  顺序 {o} {kv:<10} SpecDec/AR decode = {s / a:.3f}× "
                      f"({'获利' if s > a else '未获利'})")
    print(f"  历史 {RUN1_BASE_SPEC_OVER_AR:.3f}×（8K base Spec/AR）是否复现："
          f"看上表 base_bf16 行，>1 才可称复现")


def report_accepted_histogram(runs):
    """接受长度直方图（每轮 m 的分布，跨 rep 汇总）。m 可取值 0..K。"""
    K = spec_dec_K()
    print("\n== [1b] 接受长度分布（SpecDec，跨 rep 汇总 m 直方图）==")
    any_rows = False
    for (order, kv, mode), rs in sorted(by_condition(runs).items(),
                                        key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if mode != "SpecDec":
            continue
        hist = [m for r in rs for m in (r.get("m_hist") or [])]
        if not hist:
            print(f"  order={order} {kv:<10} m_hist 缺失 -> available=false")
            continue
        any_rows = True
        cnt = {i: hist.count(i) for i in range(K + 1)}
        rounds = len(hist)
        print(f"  order={order} {kv:<10} m=0..{K}: "
              + " ".join(f"{i}:{cnt[i]}" for i in range(K + 1))
              + f"   总轮数={rounds}（{len(rs)} rep × {rounds // max(len(rs), 1)}）"
              + f"  sum(m)={sum(hist)}  全接受轮占比={100.0 * cnt[K] / max(rounds, 1):.1f}%")
    if not any_rows:
        print("  available=false（本 run 无 SpecDec 行）")


def spec_dec_K() -> int:
    return M.K


def full_report(runs, sums) -> None:
    stats = report_within_process(runs)
    report_accepted_histogram(runs)
    order = report_between_order(stats)
    report_token_comparison(runs)
    report_ar_aggregation(runs)
    report_memory_audit(runs)
    report_monitor(runs)
    report_verdict(stats, order)


def main() -> int:
    ap = argparse.ArgumentParser(description="8K 独立进程复验 + 离线报表")
    ap.add_argument("--report", nargs="?", const="latest", default=None,
                    help="只出报表，不跑 GPU。'latest' 或具体 stamp，例如 20260921T042125Z")
    args = ap.parse_args()

    h0 = hashes()
    print("[probe] 受保护历史文件 sha256（开始）:")
    for k, v in h0.items():
        print(f"  {k}: {v}  ({PROTECTED[k].name})", flush=True)

    if args.report is not None:
        stamp = None
        if args.report != "latest":
            stamp = args.report
        runs, sums, files = collect(stamp)
        if args.report == "latest" and not runs:
            print("[probe] 未找到任何 8k* run 文件")
            return 2
        print(f"[probe] report 模式：files={len(files)} runs={len(runs)} summaries={len(sums)}")
        for f in files:
            print(f"    {f}")
        full_report(runs, sums)
        h1 = hashes()
        print("\n== 受保护文件完整性 ==")
        for k in h0:
            print(f"  {k}: {'未改动' if h0[k] == h1[k] else '!! 已变化'}  {h1[k]}")
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    meta = []
    for o in ("A", "B"):
        rid = f"8k{o}_{stamp}"
        for kv, mode in PLANS[o]:
            env = os.environ.copy()
            env.update({"HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
                        "P4_CTX": str(CTX), "P4_KV": kv, "P4_MODE": mode,
                        "P4_REPS": str(REPS), "P4_WARM": str(WARM), "P4_MAXNEW": str(MAXNEW),
                        "P4_CHUNK": str(CHUNK), "P4_RUN_ID": rid, "P4_ORDER": o})
            for k in ("P4_ONLY", "P4_REVERSE"):
                env.pop(k, None)
            print(f"\n[probe] order={o} condition=({CTX},{kv},{mode}) run_id={rid}", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "-m", "specdec.engine.g4_part4_pilot_measure"],
                               env=env, cwd=str(ROOT))
            meta.append(dict(order=o, kv=kv, mode=mode, returncode=r.returncode,
                             wall_s=time.time() - t0))
            print(f"[probe] ({kv},{mode}) exit={r.returncode} wall={meta[-1]['wall_s']:.1f}s",
                  flush=True)

    runs, sums, files = collect(stamp)
    print(f"\n[probe] 本 run stamp={stamp} files={len(files)} runs={len(runs)}")
    full_report(runs, sums)

    h1 = hashes()
    print("\n== 受保护文件完整性 ==")
    for k in h0:
        print(f"  {k}: {'未改动' if h0[k] == h1[k] else '!! 已变化'}  {h1[k]}")
    print("\n[probe] 未运行的部分不得填预期值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
