"""g4_part4_8k_isolated_probe.py — 8K 四条件的**独立进程**复验（父进程只编排）。

为什么必须独立进程：本卡 8 GB，8K 双模型时 `peak_resv` 达 8.05–10.47 GiB（越过物理显存），
分配器池状态会显著改变同一格的耗时（实测同一格 base@8192 AR：6.44s ↔ 10.87s）。
⇒ 每格必须从干净、确定的进程/显存状态开始，格间才可比。

被复验的历史数字（**只读，不修改**）：
  output/part4/part4_dc0bd5d87389.jsonl   = exploratory_run_1_fixed_order
  output/part4/part4_ba0c985bcd5d.jsonl   = exploratory_run_2_reverse_order
本脚本在开始/结束各算一次这两个文件的 sha256，若中途被改动会明确报警。

编排
  顺序 A：base AR → v_fp8 AR → base Spec → v_fp8 Spec
  顺序 B：v_fp8 AR → base AR → v_fp8 Spec → base Spec
  每个条件 = 一个全新 Python 子进程：重新加载双模型（**AR 也保持双模型常驻**，避免改变显存
  状态）、warm-up=1、reps=5、max_new=64、chunk=2048，退出即释放全部 CUDA 状态。

判据（写在这里，跑之前就固定）
  - 每条件 rel_spread = spread/median；>20% → 该条件 CHECK（识别测量不稳定，不作显著性）
  - `v_fp8 AR 优势稳定`需同时满足：两轮顺序都 v_fp8 > base；四格 rel_spread ≤20%；
    token/n_gen/stats 一致；温度/功耗/时钟无足以解释差异的漂移（温度差 >5°C 即标记）
  - `order-sensitive` 若 |ratioA − ratioB| / max(ratioA, ratioB) > 20%
  - 原 1.286×（run 1 的 8K base Spec/AR）**只有在 fresh process 中重新 >1 才可称复现**

用法（交互式 GPU 终端；预计 25–40 分钟）
  python -m specdec.engine.g4_part4_8k_isolated_probe
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
RUN1_BASE_SPEC_OVER_AR = 1.286      # 历史数字（run 1），仅用于"是否复现"的对照


def sha256_file(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def hashes() -> dict:
    return {k: sha256_file(v) for k, v in PROTECTED.items()}


def run_order(order: str, run_id: str) -> list[dict]:
    """逐条件起子进程；返回每步的执行元信息（退出码/耗时）。"""
    meta = []
    for kv, mode in PLANS[order]:
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
            "P4_CTX": str(CTX), "P4_KV": kv, "P4_MODE": mode,
            "P4_REPS": str(REPS), "P4_WARM": str(WARM), "P4_MAXNEW": str(MAXNEW),
            "P4_CHUNK": str(CHUNK), "P4_RUN_ID": run_id, "P4_ORDER": order,
        })
        env.pop("P4_ONLY", None)
        env.pop("P4_REVERSE", None)
        print(f"\n[probe] order={order} condition=({CTX},{kv},{mode}) pid=child run_id={run_id}",
              flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "specdec.engine.g4_part4_pilot_measure"],
                           env=env, cwd=str(ROOT))
        dt = time.time() - t0
        meta.append(dict(order=order, kv=kv, mode=mode, returncode=r.returncode,
                         wall_s=dt, ts_start=t0, ts_end=time.time()))
        print(f"[probe] order={order} ({kv},{mode}) exit={r.returncode} wall={dt:.1f}s", flush=True)
    return meta


def collect(run_ids: list[str]) -> tuple[list[dict], list[dict]]:
    runs, sums = [], []
    for rid in run_ids:
        for f in sorted(OUTDIR.glob(f"part4_*_{rid}.jsonl")):
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
    return runs, sums


def rspread(vals) -> dict:
    vals = sorted(v for v in vals if isinstance(v, (int, float)) and v == v)
    if not vals:
        return dict(n=0)
    med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    sp = vals[-1] - vals[0]
    return dict(n=len(vals), median=med, min=vals[0], max=vals[-1], spread=sp,
                rel_pct=(100.0 * sp / abs(med)) if med else None)


def mon_field(row: dict, when: str, key: str):
    d = row.get(f"monitor_{when}") or {}
    return d.get(key) if d.get("available") else None


def main() -> int:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    h0 = hashes()
    print("[probe] 受保护的历史文件 sha256（开始）:")
    for k, v in h0.items():
        print(f"  {k}: {v}  ({PROTECTED[k].name})", flush=True)

    meta = []
    run_ids = []
    for order in ("A", "B"):
        rid = f"8k{order}_{stamp}"
        run_ids.append(rid)
        meta += run_order(order, rid)

    runs, sums = collect(run_ids)
    print(f"\n[probe] 收集到 run 行 {len(runs)} 条 / summary 行 {len(sums)} 条（{len(run_ids)} 个 run_id）")

    # ---------- 逐条件：5 rep 的 median/min/max/spread ----------
    print("\n== 逐条件统计（每个条件一个全新进程；reps=%d）==" % REPS)
    print("order kv         mode     n  dec_tok/s med[min..max]  rel%   dec_s med  pref_s med  "
          "peak_alloc  peak_resv  temp°C(start→end)  power_W  sm_MHz  n_gen_same hash_same stats_same finite")
    stats: dict = {}
    for order in ("A", "B"):
        for kv, mode in PLANS[order]:
            rs = [r for r in runs if r.get("order") == order and r.get("target_kv") == kv
                  and r.get("mode") == mode]
            if not rs:
                print(f"{order:<5} {kv:<10} {mode:<8} 未运行")
                continue
            d = rspread([r["decode_tok_s"] for r in rs])
            ds = rspread([r["t_decode_s"] for r in rs])
            ps = rspread([r["t_prefill_s"] for r in rs])
            pa = rspread([r["peak_alloc_GiB"] for r in rs])
            pr = rspread([r["peak_resv_GiB"] for r in rs])
            stats[(order, kv, mode)] = d
            temps = [(mon_field(r, "start", "temp_C"), mon_field(r, "end", "temp_C")) for r in rs]
            pw = [mon_field(r, "end", "power_W") for r in rs]
            clk = [mon_field(r, "end", "clock_sm_MHz") for r in rs]
            n_gen_same = len({r.get("n_gen") for r in rs}) == 1
            hash_same = len({r.get("gen_sha1") for r in rs}) == 1
            stats_same = len({tuple(r.get(k) for k in ("rounds", "proposed", "accepted",
                                                       "target_block_calls", "draft_calls",
                                                       "mismatch_rounds", "all_accept_rounds"))
                              for r in rs}) == 1
            finite = all(bool(r.get("prefill_logits_finite")) and bool(r.get("step_logits_finite", True))
                         for r in rs)
            t0 = next((t[0] for t in temps if t[0] is not None), None)
            t1 = next((t[1] for t in temps if t[1] is not None), None)
            print(f"{order:<5} {kv:<10} {mode:<8} {d['n']}  "
                  f"{d['median']:.2f}[{d['min']:.2f}..{d['max']:.2f}]  "
                  f"{(d['rel_pct'] or 0):5.1f}  {ds['median']:8.2f}  {ps['median']:9.2f}  "
                  f"{pa['median']:9.2f}  {pr['median']:9.2f}  "
                  f"{str(t0)+'→'+str(t1):<16}  {str(pw[-1]):>7}  {str(clk[-1]):>6}  "
                  f"{str(n_gen_same):<10} {str(hash_same):<9} {str(stats_same):<10} {finite}")

    # ---------- 监控字段可用性 ----------
    print("\n== 环境监控（缺失字段一律 available=false，不填 0）==")
    m0 = (runs[0].get("monitor_start") if runs else {}) or {}
    if m0.get("available"):
        print(f"  nvidia-smi: {m0.get('gpu_name')}  driver={m0.get('driver')}  "
              f"mem_total={m0.get('mem_total_MiB')} MiB")
    else:
        print(f"  nvidia-smi: available=false  reason={m0.get('error')}")
    c0 = (runs[0].get("cuda_start") if runs else {}) or {}
    print(f"  cuda snapshot available={c0.get('available')} "
          f"free={c0.get('cuda_free_GiB')} total={c0.get('cuda_total_GiB')}")
    print("  口径：peak_alloc=PyTorch 实际张量分配；peak_resv=分配器池/虚拟预留(≠物理显存)；"
          "nvidia-smi used=驱动观察值")

    # ---------- 比值与稳定性判定 ----------
    print("\n== v_fp8/base 比值与判定 ==")
    verdicts = {}
    for order in ("A", "B"):
        row = {}
        for mode in ("AR", "SpecDec"):
            b, v = stats.get((order, "base_bf16", mode)), stats.get((order, "v_fp8", mode))
            if b and v and b.get("median") and v.get("median"):
                row[mode] = v["median"] / b["median"]
        verdicts[order] = row
        if row:
            print(f"  顺序 {order}: AR v_fp8/base = "
                  f"{row.get('AR', float('nan')):.3f}×   SpecDec = "
                  f"{row.get('SpecDec', float('nan')):.3f}×")
    stable = False
    if all(verdicts.get(o, {}).get("AR") for o in ("A", "B")):
        a, b = verdicts["A"]["AR"], verdicts["B"]["AR"]
        rel = 100.0 * abs(a - b) / max(a, b)
        rels = [stats[k]["rel_pct"] for k in stats
                if k[1] == "AR" and stats[k].get("rel_pct") is not None]
        spread_ok = all(x <= REL_LIMIT for x in rels)
        order_sensitive = rel > REL_LIMIT
        print(f"  两顺序 AR 比值差异 {rel:.1f}% -> "
              f"{'order-sensitive(>20%)' if order_sensitive else '一致(<=20%)'}")
        print(f"  AR 各格 rel_spread 最大值 = {max(rels) if rels else float('nan'):.1f}% "
              f"-> {'<=20% 通过' if spread_ok else '>20% CHECK'}")
        stable = (a > 1 and b > 1 and spread_ok and not order_sensitive)
        print(f"  => '8K v_fp8 AR 存在稳定优势' = {'成立(仍需附机制推断措辞)' if stable else '不成立/未证实'}")
    else:
        print("  !! 缺少 AR 条件，无法判定")

    print("\n== SpecDec 在 8K 是否获利（同档 decode 与 e2e）==")
    for order in ("A", "B"):
        for kv in ("base_bf16", "v_fp8"):
            a, s = stats.get((order, kv, "AR")), stats.get((order, kv, "SpecDec"))
            if a and s and a.get("median") and s.get("median"):
                print(f"  顺序 {order} {kv:<10} decode Spec/AR = {s['median']/a['median']:.3f}×")

    print("\n== 原始 1.286×（run 1 的 8K base Spec/AR）是否复现 ==")
    for order in ("A", "B"):
        a, s = stats.get((order, "base_bf16", "AR")), stats.get((order, "base_bf16", "SpecDec"))
        if a and s and a.get("median"):
            r = s["median"] / a["median"]
            print(f"  顺序 {order}: {r:.3f}×（历史值 {RUN1_BASE_SPEC_OVER_AR:.3f}×；"
                  f"{'>=1 可称复现' if r > 1 else '<1 未复现'}）")

    # ---------- 保护校验 ----------
    h1 = hashes()
    print("\n== 受保护文件完整性 ==")
    for k in h0:
        ok = h0[k] == h1[k]
        print(f"  {k}: {'未改动 ✓' if ok else '!! 已变化'}  {h1[k]}")

    print("\n[probe] 结论仅在 fresh-process 数据下成立；两次启动顺序都在同一份 JSONL 里可审计。")
    print("[probe] 未运行的部分（数值）不得填预期值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
