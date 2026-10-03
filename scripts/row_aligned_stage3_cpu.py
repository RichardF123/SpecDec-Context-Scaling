"""row_aligned_stage3_cpu.py — **Stage 3** 的只读汇总（纯 CPU）：端到端计时 + 固定 block micro-latency。

## 作用
从已落盘产物独立复算（不重跑、不改产物）：
① `{RA3_TAG}_F13_{mode}_rep*.json`（每个 rep 一个独立进程；**rep1 作为 warm-up，不计入统计**）：
   每模式的 `wall_s`、`prefill_s`、`decode_s = wall - prefill`、`decode tok/s`、`e2e tok/s`、
   acceptance rate、rounds、target/draft calls、峰值显存、重复 spread（中位数/最小/最大）；
② `{MI_TAG}_*.json`（固定 block micro-latency）：verify OFF/ON 中位与比值、decode 单步 OFF/ON、
   OFF 两块漂移、P-state 是否合格、峰值；
③ `overhead`：`aligned_decode_over_original`、`aligned_verify_micro_over_original`；
④ `net_value`：`aligned_spec_vs_ar`（decode tok/s 比值）。

**不是**正式性能实验、**不**作 speedup 声明；不同 token 轨迹的 e2e 时间只作**观察**。
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "g5" / "attn"
TAG = "div8209_ra3"
MI_TAG = "div8209_ra3micro"
MODES = ("ar", "spec", "aligned")


def _runs(mode: str) -> list:
    out = []
    for f in sorted(glob.glob(str(OUT / f"{TAG}_F13_{mode}_rep*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("tool") == "row_aligned_stage2" and d.get("mode") == mode:
            d["_file"] = Path(f).name
            out.append(d)
    return out


def _med(xs):
    return float(statistics.median(xs)) if xs else None


def _mode_stats(mode: str) -> dict:
    rs = _runs(mode)
    warm, timed = [r for r in rs if r["rep"] == 1], [r for r in rs if r["rep"] != 1]
    def col(rows, key, sub=None):
        vals = []
        for r in rows:
            s = r.get("smoke") or {}
            v = (s.get(sub) or {}).get(key) if sub else s.get(key)
            if v is not None:
                vals.append(float(v))
        return vals
    wall = col(timed, "wall_s")
    pre = col(timed, "prefill_s")
    gen_len = [int((r.get("smoke") or {}).get("gen_len") or 0) for r in timed]
    dec = [w - p for w, p in zip(wall, pre)]
    acc = col(timed, "accepted", sub="specdec_stats")
    prop = col(timed, "proposed", sub="specdec_stats")
    tbc = col(timed, "target_block_calls", sub="specdec_stats")
    drc = col(timed, "draft_calls", sub="specdec_stats")
    rnd = col(timed, "rounds", sub="specdec_stats")
    peak_a = col(timed, "pt_peak_allocated_mib", sub="mem_timeline_last")
    peak_r = col(timed, "pt_peak_reserved_mib", sub="mem_timeline_last")
    return dict(mode=mode, n_warmup=len(warm), n_timed=len(timed),
                reps=[r["rep"] for r in timed],
                wall_s=wall, prefill_s=pre, decode_s=dec, gen_len=gen_len,
                wall_s_median=_med(wall), prefill_s_median=_med(pre), decode_s_median=_med(dec),
                wall_spread_s=(max(wall) - min(wall)) if wall else None,
                decode_tok_s_median=_med([g / d for g, d in zip(gen_len, dec) if d > 0]),
                e2e_tok_s_median=_med([g / w for g, w in zip(gen_len, wall) if w > 0]),
                accepted=acc, proposed=prop,
                acceptance_rate_median=_med([a / p for a, p in zip(acc, prop) if p > 0]),
                target_block_calls=tbc, draft_calls=drc, rounds=rnd,
                peak_alloc_mib_median=_med(peak_a), peak_res_mib_median=_med(peak_r),
                stop_reasons=[(r.get("smoke") or {}).get("stop_reason") for r in timed],
                gen_hash=[((r.get("smoke") or {}).get("end_state") or {}).get("gen_hash") for r in timed],
                candidate_counts=[(r.get("candidate") or {}).get("counts_match") for r in rs])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", default=None)
    a = ap.parse_args()
    mic = Path(a.micro) if a.micro else None
    if mic is None:
        fs = sorted(glob.glob(str(OUT / f"{MI_TAG}_*.json")), key=lambda p: Path(p).stat().st_mtime)
        mic = Path(fs[-1]) if fs else None
    rec: dict = dict(tool="row_aligned_stage3_cpu", tag=TAG, mi_tag=MI_TAG,
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     semantics="只读：从 Stage 3 的独立进程产物与固定 block micro-latency 产物汇总计时；"
                               "rep1 为 warm-up 不计入；**不作 speedup 声明**",
                     micro_artifact=(mic.name if mic else None))
    stats = {m: _mode_stats(m) for m in MODES}
    rec["per_mode"] = stats
    print("== Stage 3 端到端（F13，max_new=64；rep1 = warm-up 不计入）==")
    hdr = f"   {'mode':<8} {'n':>2} {'wall_s':>8} {'prefill_s':>10} {'decode_s':>9} {'dec tok/s':>10} {'e2e tok/s':>10} {'acc%':>6} {'rounds':>7} {'tgt_blk':>8} {'drf':>5} {'peakA':>7} {'peakR':>7}"
    print(hdr)
    for m in MODES:
        s = stats[m]
        f = lambda v, d=2: ("-" if v is None else f"{v:.{d}f}")
        print(f"   {m:<8} {s['n_timed']:>2} {f(s['wall_s_median']):>8} {f(s['prefill_s_median']):>10} "
              f"{f(s['decode_s_median']):>9} {f(s['decode_tok_s_median']):>10} {f(s['e2e_tok_s_median']):>10} "
              f"{f(None if s['acceptance_rate_median'] is None else s['acceptance_rate_median']*100,1):>6} "
              f"{f(_med(s['rounds']),1):>7} {f(_med(s['target_block_calls']),1):>8} "
              f"{f(_med(s['draft_calls']),1):>5} {f(s['peak_alloc_mib_median'],0):>7} "
              f"{f(s['peak_res_mib_median'],0):>7}")
        print(f"            stop={s['stop_reasons']} gen_hash={s['gen_hash']} "
              f"wall spread={f(s['wall_spread_s'],3)} s 候选计数一致={s['candidate_counts']}")
    overhead, net_value = {}, {}
    if stats["aligned"]["decode_s_median"] and stats["spec"]["decode_s_median"]:
        overhead["aligned_decode_over_original"] = float(stats["aligned"]["decode_s_median"]
                                                         / stats["spec"]["decode_s_median"])
    if stats["aligned"]["wall_s_median"] and stats["spec"]["wall_s_median"]:
        overhead["aligned_e2e_wall_over_original"] = float(stats["aligned"]["wall_s_median"]
                                                           / stats["spec"]["wall_s_median"])
    if stats["aligned"]["decode_tok_s_median"] and stats["ar"]["decode_tok_s_median"]:
        net_value["aligned_spec_vs_ar"] = float(stats["aligned"]["decode_tok_s_median"]
                                                / stats["ar"]["decode_tok_s_median"])
    if stats["spec"]["decode_tok_s_median"] and stats["ar"]["decode_tok_s_median"]:
        net_value["original_spec_vs_ar"] = float(stats["spec"]["decode_tok_s_median"]
                                                 / stats["ar"]["decode_tok_s_median"])
    micro = {}
    if mic and mic.exists():
        mj = json.loads(mic.read_text(encoding="utf-8"))
        micro = dict(artifact=mic.name, summary=mj.get("summary"), correctness=mj.get("correctness"),
                     pstate=mj.get("pstate"), timing_ms=mj.get("timing_ms"))
        if mj.get("summary", {}).get("verify_on_over_off") is not None:
            overhead["aligned_verify_micro_over_original"] = float(mj["summary"]["verify_on_over_off"])
            overhead["aligned_decode_step_micro_over_original"] = float(mj["summary"]["step_on_over_off"])
    rec["micro"] = micro
    rec["overhead"] = overhead
    rec["net_value"] = net_value
    rec["pstate_qualified"] = (micro.get("pstate") or {}).get("qualified") if micro else None
    print("\n== 固定 block micro-latency ==")
    if micro:
        s = micro["summary"]
        print(f"   verify（s=4）：OFF 中位 {s['verify_off_median_ms']:.2f} ms → ON 中位 "
              f"{s['verify_on_median_ms']:.2f} ms ⇒ **{s['verify_on_over_off']:.2f}×**"
              f"（OFF 两块漂移 {s['verify_off_block_spread_ms']:.2f} ms）")
        print(f"   decode 单步（s=1）：OFF {s['step_off_median_ms']:.2f} ms → ON "
              f"{s['step_on_median_ms']:.2f} ms ⇒ {s['step_on_over_off']:.2f}×"
              f"（单步 ON==OFF 逐位一致={micro['correctness']['decode_step_bitwise_equal']}）")
        print(f"   P-state qualified={rec['pstate_qualified']}；峰值 allocated "
              f"{s['mem']['peak_allocated_mib']:.0f} MiB / reserved {s['mem']['peak_reserved_mib']:.0f} MiB")
    print("\n== overhead / net_value ==")
    print(f"   overhead = {json.dumps(overhead, ensure_ascii=False)}")
    print(f"   net_value = {json.dumps(net_value, ensure_ascii=False)}")
    rec["note"] = ("不同 token 轨迹的 e2e 时间只作**观察**：原始 Spec 与对齐候选在 F13 上轨迹不同，"
                   "其 e2e 差**不能**解释为纯 kernel 开销；纯开销看固定 block micro-latency。")
    p = OUT / f"{TAG}_stage3_summary_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[ra3] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
