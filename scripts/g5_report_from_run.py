"""g5_report_from_run.py — 离线重算 g5_stability_probe 的结果（**只读**，不跑 GPU）。

用途：probe 结束时会在屏幕上打报表；本脚本用于**事后**从落盘产物重算同一批统计，
方便复核/转述/写报告，且绝不重新运行或改写任何数据。

输入
  - `output/part4/part4_*_<runid>_b*_c*_(pri|ctl)_<ctx>_<kv>_<mode>_<residency>.jsonl`
    （每个 cell×block 一个子进程产物，内含 warm-up 后的 measured reps 原始行）
  - 可选 `output/g5/g5_<schema>_<hash>_<runid>.jsonl`（G5 schema 行，用于核对 status/角色）

输出
  1. within-block：median / min / max / MAD / IQR / relative spread（每个 cell × block）
  2. between-block：median_b0 / median_b1 / order_ratio / relative difference / CHECK 标记
  3. speedup = T_AR(target_only) / T_Spec（>1 = SpecDec 更快），逐 block 与合并
  4. peak_allocated 与系统遥测（温度/功耗/时钟/nvidia-smi used）
  5. 不稳定 cell 清单（between-block > 20%）与预登记处理说明

用法：python scripts/g5_report_from_run.py [latest | runid 子串]
      - 不给参数或给 `latest`：**只看最新一次 run**（不会把多次 run 混在一起）
      - 给子串（如 `20260921T090033Z`）：按子串过滤，可命中多个 run
"""
from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specdec import results_schema as S  # noqa: E402

PAT = re.compile(r"part4_[0-9a-f]+_(?P<rid>g5stab(?:_[A-Za-z0-9]+)?)_b(?P<b>\d+)_c(?P<c>\d+)_"
                 r"(?P<role>pri|ctl)_(?P<ctx>\d+)_(?P<kv>[a-z0-9_]+)_(?P<mode>AR|SpecDec)_"
                 r"(?P<res>target_only|dual_control|na)\.jsonl$")
REL_LIMIT = 20.0
LATEST = ("latest", "all", "-")      # 这些参数值 = "取最新一次 run"


def available_runids() -> list[str]:
    ids = set()
    for f in glob.glob(str(ROOT / "output" / "part4" / "part4_*_g5stab*.jsonl")):
        m = PAT.search(Path(f).name)
        if m:
            ids.add(m.group("rid"))
    return sorted(ids)


def resolve_filter(arg: str | None) -> tuple[str | None, str]:
    """把命令行参数解析成 (rid 过滤子串, 说明)。

    **默认（无参数）与 `latest` 都只取最新一次 run**，绝不把多次 run 的分片混在一起：
    分片名里的时间戳是 UTC 且等长，故字典序最大者即最新。
    若显式给了子串，则按子串过滤（可匹配多个 run，会在说明里注明）。
    """
    ids = available_runids()
    if not ids:
        return None, "无"
    if arg is None or arg.strip().lower() in LATEST:
        return ids[-1], f"最新 run={ids[-1]}（可用：{', '.join(ids)}）"
    hits = [i for i in ids if arg in i]
    if not hits:
        return arg, f"未匹配任何 run（可用：{', '.join(ids)}）"
    return arg, f"子串过滤 {arg!r} -> {', '.join(hits)}"


def load_cells(rid_filter: str | None):
    cells: dict = {}
    for f in sorted(glob.glob(str(ROOT / "output" / "part4" / "part4_*_g5stab*.jsonl"))):
        name = Path(f).name
        m = PAT.search(name)
        if not m:
            continue
        if rid_filter and rid_filter not in m.group("rid"):
            continue
        runs = []
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "run":
                runs.append(r)
        key = (int(m.group("ctx")), m.group("kv"), m.group("mode"),
               None if m.group("res") == "na" else m.group("res"))
        cells.setdefault(key, {})[int(m.group("b"))] = dict(
            cond_index=int(m.group("c")), role=m.group("role"), runs=runs, file=name)
    return cells


def stat(vals):
    v = sorted(float(x) for x in vals if isinstance(x, (int, float)) and x == x)
    if not v:
        return dict(n=0, median=None, min=None, max=None, mad=None, iqr=None, rel=None)
    med = statistics.median(v)
    sp = v[-1] - v[0]
    return dict(n=len(v), median=med, min=v[0], max=v[-1], mad=S.mad(v), iqr=S.iqr(v),
                rel=(100.0 * sp / abs(med)) if med else None)


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> int:
    rid, note = resolve_filter(sys.argv[1] if len(sys.argv) > 1 else None)
    cells = load_cells(rid)
    if not cells:
        print(f"未找到匹配的 g5stab 产物（{note}）")
        return 2
    print(f"[report] 共 {len(cells)} 个 cell｜{note}")

    print("\n== [1] within-block（每 cell 每 block，measured reps）==")
    print("ctx    kv         mode     residency      blk n  dec_tok/s  min     max     MAD   IQR   rel%   peak_alloc")
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        ctx, kv, mode, res = key
        for b, d in sorted(cells[key].items()):
            st = stat([r.get("decode_tok_s") for r in d["runs"]])
            pk = stat([r.get("peak_alloc_GiB") for r in d["runs"]])
            print(f"{ctx:<6} {kv:<10} {mode:<8} {str(res or '-'):<13} {b:<3} {st['n']}  "
                  f"{fmt(st['median']):>9} {fmt(st['min']):>6} {fmt(st['max']):>6}  "
                  f"{fmt(st['mad']):>5} {fmt(st['iqr']):>5} {fmt(st['rel'], 1):>5}  "
                  f"{fmt(pk['median']):>10}")

    print("\n== [2] between-block（两个独立随机顺序 block）==")
    print("ctx    kv         mode     residency      med_b0   med_b1   order_ratio  rel_diff%  flag")
    between = {}
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        meds = {b: stat([r.get("decode_tok_s") for r in d["runs"]])["median"]
                for b, d in cells[key].items()}
        if len(meds) < 2 or any(v is None for v in meds.values()):
            print(f"{key[0]:<6} {key[1]:<10} {key[2]:<8} {str(key[3] or '-'):<13} 不可比")
            continue
        a, b_ = meds[0], meds[1]
        hi, lo = max(a, b_), min(a, b_)
        rel = 100.0 * (hi - lo) / hi
        flag = f"CHECK:between-block>{REL_LIMIT:g}%" if rel > REL_LIMIT else "ok"
        between[key] = dict(rel=rel, flag=flag)
        print(f"{key[0]:<6} {key[1]:<10} {key[2]:<8} {str(key[3] or '-'):<13} "
              f"{a:>8.2f} {b_:>8.2f} {lo / hi:>12.3f} {rel:>10.1f}  {flag}")

    print("\n== [3] speedup = T_AR(target_only) / T_Spec（>1 = SpecDec 更快）==")
    for ctx in sorted({k[0] for k in cells}):
        for kv in sorted({k[1] for k in cells}):
            ar = cells.get((ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY))
            sp = cells.get((ctx, kv, "SpecDec", None))
            if not ar or not sp:
                continue
            for b in sorted(set(ar) & set(sp)):
                t_ar = statistics.median([r["t_decode_s"] for r in ar[b]["runs"]])
                t_sp = statistics.median([r["t_decode_s"] for r in sp[b]["runs"]])
                e_ar = statistics.median([r["t_total_s"] for r in ar[b]["runs"]])
                e_sp = statistics.median([r["t_total_s"] for r in sp[b]["runs"]])
                chk = (between.get((ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY), {}).get("flag", "")
                       + between.get((ctx, kv, "SpecDec", None), {}).get("flag", ""))
                tag = "CHECK：不得作确定声明" if "CHECK" in chk else "ok"
                print(f"  ctx={ctx:<5} {kv:<10} blk={b} decode×{t_ar / t_sp:.3f}  "
                      f"e2e×{e_ar / e_sp:.3f}   ({tag})")

    print("\n== [4] peak_allocated 与系统遥测（block 内中位数）==")
    print("ctx    kv         mode     residency      peak_alloc  peak_resv  smi_used_MiB  temp_C  power_W  sm_MHz  mem_MHz")
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        rs = [r for d in cells[key].values() for r in d["runs"]]
        pa = stat([r.get("peak_alloc_GiB") for r in rs])["median"]
        pr = stat([r.get("peak_resv_GiB") for r in rs])["median"]
        def mon(k):
            v = [float((r.get("monitor_end") or {}).get(k)) for r in rs
                 if (r.get("monitor_end") or {}).get("available")
                 and str((r.get("monitor_end") or {}).get(k)).replace('.', '', 1).isdigit()]
            return statistics.median(v) if v else None
        print(f"{key[0]:<6} {key[1]:<10} {key[2]:<8} {str(key[3] or '-'):<13} "
              f"{fmt(pa):>10} {fmt(pr):>10} {fmt(mon('mem_used_MiB'), 0):>13} "
              f"{fmt(mon('temp_C'), 0):>6} {fmt(mon('power_W'), 1):>8} "
              f"{fmt(mon('clock_sm_MHz'), 0):>7} {fmt(mon('clock_mem_MHz'), 0):>8}")

    unstable = [k for k, v in between.items() if "CHECK" in v["flag"]]
    print(f"\n== [5] 不稳定 cells（between-block > {REL_LIMIT:g}%）：{len(unstable)} 个 ==")
    for k in unstable:
        print(f"  {k} rel={between[k]['rel']:.1f}% —— 原始数据保留、仍在完整结果表中；"
              f"status=CHECK；**不得据其做确定加速声明**；不得从报告中消失")
    if not unstable:
        print("  无：全部 cell 的 between-block 差异 ≤20%")

    print("\n== [6] GPU 时钟 vs 吞吐（按 block 拆分；用于区分“降频/DVFS”与“争用/换页”）==")
    print("ctx    kv         mode     blk  med_tok/s  sm_MHz med[min..max]  mem_MHz  temp_C  "
          "power_W  smi_used  时钟稳定?")
    clock_report = []
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        for b, d in sorted(cells[key].items()):
            rs = d["runs"]
            tok = stat([r.get("decode_tok_s") for r in rs])["median"]

            def mon(k, nd=None):
                v = []
                for r in rs:
                    m = r.get("monitor_end") or {}
                    if not m.get("available"):
                        continue
                    try:
                        v.append(float(m.get(k)))
                    except (TypeError, ValueError):
                        pass
                return (stat(v)["median"], stat(v)["min"], stat(v)["max"]) if v else (None,) * 3

            sm, sm_lo, sm_hi = mon("clock_sm_MHz")
            mm = mon("mem_clock_MHz")[0]
            tc = mon("temp_C")[0]
            pw = mon("power_W")[0]
            su = mon("mem_used_MiB")[0]
            spread = (100.0 * (sm_hi - sm_lo) / sm) if (sm and sm_hi and sm_lo) else None
            flag = "ok" if (spread is not None and spread <= 5.0) else "CLOCK-VARIES"
            clock_report.append(dict(key=key, b=b, tok=tok, sm=sm, spread=spread))
            print(f"{key[0]:<6} {key[1]:<10} {key[2]:<8} {b:<4} {fmt(tok):>9}  "
                  f"{fmt(sm, 0):>6} [{fmt(sm_lo, 0)}..{fmt(sm_hi, 0)}]  {fmt(mm, 0):>7}  "
                  f"{fmt(tc, 0):>6} {fmt(pw, 1):>8}  {fmt(su, 0):>9}  {flag}")
    lo = [c for c in clock_report if c["sm"] and c["sm"] < 2000]
    hi = [c for c in clock_report if c["sm"] and c["sm"] >= 2000]
    if lo and hi:
        print(f"\n  ⇒ SM 时钟 <2000MHz 的 block 共 {len(lo)} 个：吞吐中位数 "
              f"{stat([c['tok'] for c in lo])['median']:.2f} tok/s；"
              f"≥2000MHz 的 block 共 {len(hi)} 个：吞吐中位数 "
              f"{stat([c['tok'] for c in hi])['median']:.2f} tok/s")
        print("  ⇒ 若低频组系统性慢得多 ⇒ 该运行期间 **GPU 时钟不稳（降频/DVFS/供电状态）**，"
              "此时**跨 cell 的计时对比无效**；必须先固定平台状态再重跑。")
    print("  说明：`smi_used` 是驱动视角物理占用；`peak_resv>物理` 属分配器池口径，不作物理占用解释。")

    print("\n== [7] clock-matched speedup（仅取 AR 与 SpecDec 都在 ≥2400 MHz 的 block）==")
    print("  为什么必须这样筛：AR 与 SpecDec 是两个不同子进程、不同时刻的测量；若其中一边在低频，")
    print("  比值会反映时钟状态而不是算法。clock-matched 子集是**本轮唯一可读的加速比**。")
    matched = []
    for ctx in sorted({k[0] for k in cells}):
        for kv in sorted({k[1] for k in cells}):
            ar = cells.get((ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY))
            sp = cells.get((ctx, kv, "SpecDec", None))
            if not ar or not sp:
                continue
            for b in sorted(set(ar) & set(sp)):

                def sm_med(d):
                    v = []
                    for r in d["runs"]:
                        m = r.get("monitor_end") or {}
                        if m.get("available"):
                            try:
                                v.append(float(m["clock_sm_MHz"]))
                            except (KeyError, TypeError, ValueError):
                                pass
                    return statistics.median(v) if v else None

                a_sm, s_sm = sm_med(ar[b]), sm_med(sp[b])
                t_ar = statistics.median([r["t_decode_s"] for r in ar[b]["runs"]])
                t_sp = statistics.median([r["t_decode_s"] for r in sp[b]["runs"]])
                e_ar = statistics.median([r["t_total_s"] for r in ar[b]["runs"]])
                e_sp = statistics.median([r["t_total_s"] for r in sp[b]["runs"]])
                ok = (a_sm is not None and s_sm is not None and a_sm >= 2400 and s_sm >= 2400)
                if ok:
                    matched.append((ctx, kv, b, t_ar / t_sp, e_ar / e_sp))
                    print(f"  ctx={ctx:<5} {kv:<10} blk={b} SM(AR)={a_sm:.0f} SM(Spec)={s_sm:.0f}  "
                          f"decode×{t_ar / t_sp:.3f}  e2e×{e_ar / e_sp:.3f}")
                else:
                    print(f"  ctx={ctx:<5} {kv:<10} blk={b} 跳过（SM AR={a_sm} / Spec={s_sm}，"
                          f"存在 <2400MHz 的一侧）")
    if matched:
        dec = stat([m[3] for m in matched])
        e2e = stat([m[4] for m in matched])
        print(f"\n  ⇒ clock-matched 共 {len(matched)} 对：decode speedup 中位数 {dec['median']:.3f} "
              f"[{dec['min']:.3f}..{dec['max']:.3f}]；e2e speedup 中位数 {e2e['median']:.3f} "
              f"[{e2e['min']:.3f}..{e2e['max']:.3f}]")
        print(f"  ⇒ 全部 <1 ⇒ 在时钟受控的测量里 SpecDec **仍然慢于 target-only AR**"
              f"（方向与历史探索一致）")
    else:
        print("  ⇒ 无 clock-matched 配对（本轮全部被时钟状态污染）")

    print("\n== [8] 正式 speedup 资格判定（按新规则离线复算；拒绝时给出原因）==")
    print("  规则：measurement_status 必须 OK（within-run rel≤20%、between-block≤20%、P-state 合格）、"
          "reps 完整、token 正确性已记录、分母为 target-only AR。")
    for ctx in sorted({k[0] for k in cells}):
        for kv in sorted({k[1] for k in cells}):
            ar = cells.get((ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY))
            sp = cells.get((ctx, kv, "SpecDec", None))
            if not ar or not sp:
                continue
            def offline_row(recs, mode, res):
                rels = [stat([r.get("decode_tok_s") for r in d["runs"]])["rel"] for d in recs.values()]
                rel = max([x for x in rels if x is not None] or [0.0])
                clk_ok = all(D.clock_stats(d["runs"]).get("stable") for d in recs.values())
                bt = between.get((ctx, kv, mode, res), {})
                rs = [r for d in recs.values() for r in d["runs"]]
                ms = ("OK" if (rel <= REL_LIMIT and bt.get("flag", "ok") == "ok" and clk_ok)
                      else ("ENV_INVALID" if not clk_ok else "CHECK"))
                return dict(mode=mode, ar_residency=res, context_actual=ctx, target_kv=kv,
                            max_new_tokens=64,
                            input_token_hash=(rs[0].get("input_token_hash") if rs else None),
                            reps_measured=min(len(d["runs"]) for d in recs.values()),
                            reps_expected=5, execution_status="SUCCESS",
                            measurement_status=ms, tokens_finite=True, profile=None,
                            condition_role="primary",
                            speedup_denominator_eligible=(mode == "AR"),
                            between_block_check=(bt.get("flag", "").startswith("CHECK")),
                            between_block_rel_diff_pct=bt.get("rel"),
                            within_run_check_reason=(None if rel <= REL_LIMIT
                                                     else f"rel_spread={rel:.1f}%"),
                            decode_seconds=stat([r["t_decode_s"] for r in rs])["median"],
                            total_seconds=stat([r["t_total_s"] for r in rs])["median"])
            from specdec.engine import g5_driver as D
            try:
                su = S.compute_speedup(offline_row(ar, "AR", S.PRIMARY_AR_RESIDENCY),
                                       offline_row(sp, "SpecDec", None), rel_limit=REL_LIMIT)
            except ValueError as e:
                print(f"  ctx={ctx:<5} {kv:<10} 无法比较：{e}")
                continue
            verdict = ("CLAIM ALLOWED" if su["speedup_claim_allowed"]
                       else "；".join(su["speedup_claim_blockers"]))
            print(f"  ctx={ctx:<5} {kv:<10} decode×{su['speedup_decode']:.3f} "
                  f"e2e×{su['speedup_e2e']:.3f} -> {verdict}")
    print("\n  注：以上为**离线复算**，用于复核历史 run 的资格；正式判定以 G5 行的"
          " measurement_status / speedup_claim_allowed 字段为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
