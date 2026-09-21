"""g5_stability_probe.py — G5 计时稳定化探针（Task B）+ AR 基线口径分离。

条件定义（**必须分清，不许混算**）
  正式 primary factorial（**12 cells**，默认必须跑）：
      ctx {2048,4096,8192} × kv {base_bf16,v_fp8} × mode {AR-primary, SpecDec}
      AR-primary  = **target_only**：子进程只加载 target，从不加载 draft（正式 speedup 分母）
      SpecDec     = target + draft 双驻留（算法自身资源需求）
  可选诊断控制（**6 cells**，`P5_INCLUDE_DUAL_AR_CONTROL=1` 才跑，**不计入 factorial**）：
      ctx × kv × AR-dual-resident-control：加载双模型但 draft 不参与 AR prefill/decode，
      用于把"额外模型驻留/显存压力"从 AR 里隔离出来。

协议（跑之前固定）
  1. 每条件一个全新子进程；模型加载/tokenizer 不进计时；
  2. 每 block 条件顺序固定 seed 随机化（seed 记档可重放），至少 2 个独立 block；
  3. 每条件 warm-up=1 + measured reps=5；条件间 cooldown 不计时；
  4. block 开始前记录系统状态；计时区外才做 nvidia-smi / 文件写入 / 文本 decode；
  5. **不稳定 cell 的处理（预先规定）**：原始数据保留、仍出现在完整结果表中、必须报
     median/min/max/MAD/IQR；`status=CHECK`；**不允许**从该 cell 做确定加速声明，
     可从稳定 speedup 汇总中排除，但**不得从报告中消失**；
  6. speedup 方向统一：`speedup = T_AR / T_Spec`（>1 = SpecDec 更快）；
     `residency_penalty = T_AR_target_only / T_AR_dual_control`（>1 = 双驻留拖慢 AR）。

用法（交互式 GPU 终端）：python -m specdec.engine.g5_stability_probe
可选：P5_CTX / P5_BLOCKS=2 / P5_REPS=5 / P5_WARM=1 / P5_MAXNEW=64 / P5_COOLDOWN=10
      P5_SEED=20260921 / P5_KV / P5_MODE（子集过滤，仅诊断）/ P5_INCLUDE_DUAL_AR_CONTROL=1
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

from .. import results_schema as S
from . import g5_driver as D

CTX = [int(x) for x in os.environ.get("P5_CTX", "2048,4096,8192").split(",") if x.strip()]
KVS = [x for x in os.environ.get("P5_KV", "base_bf16,v_fp8").split(",") if x.strip()]
MODES = [x for x in os.environ.get("P5_MODE", "AR,SpecDec").split(",") if x.strip()]
BLOCKS = int(os.environ.get("P5_BLOCKS", "2"))
REPS = int(os.environ.get("P5_REPS", "5"))
WARM = int(os.environ.get("P5_WARM", "1"))
MAXNEW = int(os.environ.get("P5_MAXNEW", "64"))
CHUNK = int(os.environ.get("P5_CHUNK", "2048"))
COOLDOWN = float(os.environ.get("P5_COOLDOWN", "10"))
SEED = int(os.environ.get("P5_SEED", "20260921"))
INCLUDE_DUAL = S.env_flag("P5_INCLUDE_DUAL_AR_CONTROL", False)
K_PROTOCOL = 4
REL_LIMIT = 20.0


def med_iqr(vals):
    v = sorted(float(x) for x in vals if isinstance(x, (int, float)) and x == x)
    if not v:
        return dict(n=0, median=None, min=None, max=None, mad=None, iqr=None,
                    rel_spread_pct=None)
    med = statistics.median(v)
    sp = v[-1] - v[0]
    return dict(n=len(v), median=med, min=v[0], max=v[-1], mad=S.mad(v), iqr=S.iqr(v),
                rel_spread_pct=(100.0 * sp / abs(med)) if med else None)


def build_cells():
    """返回 (primary_cells, control_cells)。cell = (ctx, kv, mode, ar_residency|None)。"""
    primary = []
    for ctx, kv, mode in S.build_primary_matrix(CTX, KVS,
                                                [m for m in S.MODE_CHOICES if m in MODES]):
        role = S.resolve_condition_role(mode, S.PRIMARY_AR_RESIDENCY if mode == "AR" else None)
        primary.append((ctx, kv, mode, role["ar_residency"]))
    control = [(c, kv, "AR", S.DIAGNOSTIC_AR_RESIDENCY)
               for c, kv in S.build_dual_controls(CTX, KVS)] if INCLUDE_DUAL else []
    return primary, control


def cell_label(c) -> str:
    ctx, kv, mode, res = c
    return f"{ctx},{kv},{mode},{res or '-'}"


def main() -> int:
    run_id = S.default_run_id("g5stab")
    primary, control = build_cells()
    print(f"[g5stab] run_id={run_id} seed={SEED} blocks={BLOCKS} reps={REPS} warm={WARM} "
          f"k={K_PROTOCOL}(协议冻结)", flush=True)
    print(f"[g5stab] primary factorial = **{len(primary)} cells**"
          f"（AR=target_only + SpecDec=dual-resident）", flush=True)
    print(f"[g5stab] diagnostic control = **{len(control)} cells**"
          f"（AR=dual_control；{'启用' if INCLUDE_DUAL else '未启用'}）"
          f" —— **不计入 factorial**", flush=True)
    print(f"[g5stab] cooldown={COOLDOWN}s（不进计时）；每条件独立进程；"
          f"loading/tokenizer 不进计时", flush=True)
    D.G5_DIR.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha256(json.dumps(
        dict(ctx=CTX, kvs=KVS, modes=MODES, blocks=BLOCKS, reps=REPS, warm=WARM,
             maxnew=MAXNEW, chunk=CHUNK, seed=SEED, k=K_PROTOCOL, dual=INCLUDE_DUAL),
        sort_keys=True).encode()).hexdigest()[:12]
    out = S.g5_path(D.G5_DIR, cfg_hash, run_id)
    if out.exists():
        raise SystemExit(f"[拒绝覆盖] {out} 已存在")
    print(f"[g5stab] g5 jsonl = {out}", flush=True)

    all_cells = primary + control
    per_cell: dict = {}
    block_state = []
    for b in range(BLOCKS):
        order = D.randomized_order(all_cells, SEED + b)
        labels = [cell_label(c) for c in order]
        st = D.system_snapshot()
        block_state.append(dict(block=b, seed=SEED + b, monitor=st))
        print(f"\n[g5stab] ==== block {b}（seed={SEED + b}）系统状态："
              f"used={st.get('mem_used_MiB')}MiB temp={st.get('temp_C')}C "
              f"sm={st.get('clock_sm_MHz')}MHz available={st.get('available')}", flush=True)
        print(f"[g5stab] block {b} 执行顺序（随机化）：{labels}", flush=True)
        for ci, cell in enumerate(order):
            ctx, kv, mode, res = cell
            tag = "ctl" if cell in control else "pri"
            cid = f"g5stab_b{b}_c{ci}_{tag}_{ctx}_{kv}_{mode}_{res or 'na'}"
            info = D.spawn_condition(ctx=ctx, kv=kv, mode=mode, run_id=cid, reps=REPS,
                                     warm=WARM, maxnew=MAXNEW, chunk=CHUNK, k=K_PROTOCOL,
                                     order=f"b{b}", ar_residency=res)
            runs, sums, jpath = D.read_part4_rows(cid)
            dec = med_iqr([r.get("decode_tok_s") for r in runs])
            dec_s = med_iqr([r.get("t_decode_s") for r in runs])
            tot_s = med_iqr([r.get("t_total_s") for r in runs])
            peak = med_iqr([r.get("peak_alloc_GiB") for r in runs])
            resv = med_iqr([r.get("peak_resv_GiB") for r in runs])
            summ = (sums[0] if sums else {})
            status = "SKIP" if not runs else ("OK" if info["returncode"] == 0 else "CHECK")
            per_cell.setdefault((ctx, kv, mode, res), []).append(
                dict(block=b, condition_index=ci, info=info, decode=dec, dec_s=dec_s,
                     tot_s=tot_s, peak_alloc=peak, peak_resv=resv, summary=summ,
                     status=status, jsonl=str(jpath) if jpath else None, role=tag))
            meta = D.g5_meta(run_id=run_id, block_index=b, condition_index=ci,
                             seed=SEED + b, condition_order=labels,
                             extra=dict(k=K_PROTOCOL, warmup=WARM, chunk_size=CHUNK,
                                        context_requested=ctx, context_actual=ctx,
                                        target_model="Qwen/Qwen2.5-1.5B-Instruct",
                                        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                                        compute_dtype="bfloat16", mode=mode,
                                        ar_residency=res if mode == "AR" else None,
                                        target_kv=kv, draft_kv="base_bf16",
                                        max_new_tokens=MAXNEW, prompt_family=None,
                                        prompt_hash=(runs[0].get("input_token_hash")
                                                     if runs else None),
                                        status=status))
            src = dict(summ, **{
                "decode_tok_s": dec["median"], "t_decode_s": dec_s["median"],
                "t_total_s": tot_s["median"], "peak_alloc_GiB": peak["median"],
                "peak_resv_GiB": resv["median"], "mode": mode, "ctx": ctx, "target_kv": kv,
                "warm": WARM, "chunk": CHUNK, "k": K_PROTOCOL,
                "monitor_end": (runs[-1].get("monitor_end") if runs else None),
                "logits_finite": (all(bool(r.get("prefill_logits_finite")) for r in runs)
                                  if runs else None),
                "n_gen": (runs[-1].get("n_gen") if runs else None),
                "certified": (summ.get("certified") if summ else None)})
            if not runs:
                src["oom"] = True
            row = S.build_g5_row(src, meta)
            row["notes"] = (f"stability probe block={b} cond_idx={ci} role={tag}; "
                            f"k={K_PROTOCOL}(冻結); cooldown={COOLDOWN}s; "
                            f"原 {os.path.basename(str(jpath)) if jpath else 'no-jsonl'}")
            S.write_g5_run(out, row, append=True)
            print(f"  [{b}.{ci}|{tag}] ctx={ctx} {kv:<10} {mode:<8} res={str(res):<12} "
                  f"exit={info['returncode']} dec_tok/s={dec['median']} "
                  f"[{dec['min']}..{dec['max']}] rel={dec['rel_spread_pct']} "
                  f"peak={peak['median']} status={status} wall={info['wall_s']:.0f}s", flush=True)
            if COOLDOWN > 0:
                time.sleep(COOLDOWN)

    # ---------------- 报表 ----------------
    def table(cells, title):
        print(f"\n== {title} ==")
        print("ctx    kv         mode     residency      blk  n  dec_tok/s  min     max     "
              "MAD   IQR   rel%  peak_alloc  status")
        for c in cells:
            for e in sorted(per_cell.get(c, []), key=lambda x: x["block"]):
                d = e["decode"]
                f2 = lambda v: "n/a" if v is None else f"{v:.2f}"          # noqa: E731
                rel = d["rel_spread_pct"]
                rel_s = f"{rel:.1f}" if rel is not None else "n/a"
                print(f"{c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} {e['block']:<4} "
                      f"{d['n']}  {f2(d['median']):>9} {f2(d['min']):>6} {f2(d['max']):>6}  "
                      f"{f2(d['mad']):>5} {f2(d['iqr']):>5} {rel_s:>5}  "
                      f"{f2(e['peak_alloc']['median']):>10}  {e['status']}")
    table(primary, f"[B1a] primary factorial（{len(primary)} cells）within-run："
                   f"median/min/max/MAD/IQR")
    if control:
        table(control, f"[B1b] 诊断控制（{len(control)} cells，**非 factorial**）within-run")

    print("\n== [B2] between-block（两个独立随机顺序）差异：primary 与 control 分列 ==")
    print("group  ctx    kv         mode     residency      med_b0   med_b1   rel_diff%  flag")
    compare = {}
    for group, cells in (("primary", primary), ("control", control)):
        for c in cells:
            meds = {e["block"]: e["decode"]["median"] for e in per_cell.get(c, [])}
            if len(meds) < 2 or any(v is None for v in meds.values()):
                print(f"{group:<6} {c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} "
                      f"不可比（需 2 个 block 的有效中位数）")
                continue
            a, b = meds[0], meds[1]
            hi, lo = max(a, b), min(a, b)
            rel = 100.0 * (hi - lo) / hi
            flag = f"CHECK:between-block>{REL_LIMIT:g}%" if rel > REL_LIMIT else "ok"
            compare[c] = dict(rel=rel, flag=flag, ratio=lo / hi)
            print(f"{group:<6} {c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} "
                  f"{a:>8.2f} {b:>8.2f} {rel:>10.1f}  {flag}")

    print("\n== [B3] 系统状态（每 block 开始前）==")
    for bs in block_state:
        m = bs["monitor"]
        print(f"  block={bs['block']} seed={bs['seed']} available={m.get('available')} "
              f"used={m.get('mem_used_MiB')}MiB/{m.get('mem_total_MiB')}MiB "
              f"temp={m.get('temp_C')}C power={m.get('power_W')}W "
              f"sm={m.get('clock_sm_MHz')}MHz mem={m.get('clock_mem_MHz')}MHz")

    def med_of(cell, key):
        vals = [e[key]["median"] for e in per_cell.get(cell, [])
                if isinstance(e[key].get("median"), (int, float)) and e[key]["median"] == e[key]["median"]]
        return statistics.median(vals) if vals else None

    def row_like(cell, mode, res):
        return dict(mode=mode, ar_residency=res, context_actual=cell[0],
                    target_kv=cell[1], decode_seconds=med_of(cell, "dec_s"),
                    total_seconds=med_of(cell, "tot_s"),
                    speedup_claim_allowed=(bool(per_cell.get(cell))
                                           and all(e["status"] == "OK"
                                                   for e in per_cell.get(cell, []))),
                    speedup_denominator_eligible=(mode == "AR"
                                                  and res == S.PRIMARY_AR_RESIDENCY))

    print("\n== [B4] speedup = T_AR(target-only) / T_Spec（>1 = SpecDec 更快）==")
    for ctx in CTX:
        for kv in KVS:
            ar = (ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY)
            sp = (ctx, kv, "SpecDec", None)
            if not per_cell.get(ar) or not per_cell.get(sp):
                continue
            try:
                s = S.compute_speedup(row_like(ar, "AR", S.PRIMARY_AR_RESIDENCY),
                                      row_like(sp, "SpecDec", None))
            except ValueError as e:
                print(f"  ctx={ctx:<5} {kv:<10} 无法计算: {e}")
                continue
            claim = "可作确定声明" if s["speedup_claim_allowed"] else "CHECK：不得作确定声明"
            print(f"  ctx={ctx:<5} {kv:<10} decode×{s['speedup_decode']:.3f}  "
                  f"e2e×{s['speedup_e2e']:.3f}   ({claim})")

    if control:
        print("\n== [B5] residency_penalty = T_AR(target_only) / T_AR(dual_control)"
              "（>1 = 双驻留拖慢 AR）==")
        for ctx in CTX:
            for kv in KVS:
                a = (ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY)
                c = (ctx, kv, "AR", S.DIAGNOSTIC_AR_RESIDENCY)
                if not per_cell.get(a) or not per_cell.get(c):
                    continue
                p = S.residency_penalty(row_like(a, "AR", S.PRIMARY_AR_RESIDENCY),
                                        row_like(c, "AR", S.DIAGNOSTIC_AR_RESIDENCY))
                print(f"  ctx={ctx:<5} {kv:<10} penalty_decode×{p['penalty_decode']:.3f}  "
                      f"penalty_e2e×{p['penalty_e2e']:.3f}")

    unstable = [c for c, v in compare.items() if "CHECK" in v["flag"]]
    print(f"\n[g5stab] 不稳定 cells（between-block > {REL_LIMIT:g}%）= {len(unstable)} 个：")
    for c in unstable:
        print(f"    {cell_label(c)}  rel={compare[c]['rel']:.1f}%  "
              f"（原始数据保留、仍在完整结果表中；status=CHECK；**不得据其做确定加速声明**）")
    print("[g5stab] 若为空：全部 cell 的 between-block 差异 ≤20%，可进入稳定汇总")
    print(f"[g5stab] g5 jsonl 行数 = {len(S.read_g5_rows(out))} -> {out}")
    print("[g5stab] 计时边界：子进程内 CUDA 同步夹住 prefill/decode；cooldown/加载/文件写入不计时")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
