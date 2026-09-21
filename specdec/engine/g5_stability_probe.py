"""g5_stability_probe.py — G5 计时稳定化探针（Task B）：回答"波动来自哪里、随机化后是否稳定"。

条件（12 个）：ctx ∈ {2048, 4096, 8192} × kv ∈ {base_bf16, v_fp8} × mode ∈ {AR, SpecDec}
  k 用**协议冻结值 4**；max_new_tokens=64；每条件 warm-up=1 + measured reps=5。

协议（写死在这里，跑之前就固定）
  1. **每条件一个全新子进程**（复用已通过验证的测量核心；模型加载/tokenizer 不进计时）；
  2. **AR 也保持 target+draft 双模型常驻**，显存驻留条件与 SpecDec 一致；
  3. 每个 block 内的条件顺序用**固定 seed 随机化**（seed 记录在案，可重放）；
  4. 至少两个独立随机顺序 block（默认 2），block 开始前记录系统状态；
  5. 条件之间固定 cooldown（默认 10s，**不进计时**）；
  6. 计时区间内不做：`empty_cache` / `nvidia-smi` / 文件写入 / 文本 decode / 除同步外的诊断；
  7. 不稳定处理**预先规定**：`between-block 相对差 > 20%` ⇒ 该条件 `status=CHECK`，
     **不得进入正式主表**，也不删除数据；
  8. 计时/显存口径：`decode_seconds` 由 CUDA 同步边界夹住（子进程内）；`peak_allocated_gib` 为主
     指标，`peak_reserved_gib` 只是分配器池，物理口径看 `nvidia_smi_used_mib`。

汇报：每条件 median/min/max/MAD/IQR + within-run rep spread + between-block（顺序）差异；
结果同时写入 **G5 schema JSONL**（唯一命名，禁止覆盖）与屏幕。

用法（交互式 GPU 终端）：python -m specdec.engine.g5_stability_probe
可选：P5_CTX / P5_BLOCKS=2 / P5_REPS=5 / P5_WARM=1 / P5_MAXNEW=64 / P5_COOLDOWN=10
      P5_SEED=20260921 / P5_KV / P5_MODE（子集过滤，仅诊断用）
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


def main() -> int:
    run_id = S.default_run_id("g5stab")
    conds = [(c, kv, m) for c in CTX for kv in KVS for m in MODES]
    print(f"[g5stab] run_id={run_id} seed={SEED} blocks={BLOCKS} reps={REPS} warm={WARM} "
          f"k={K_PROTOCOL}(协议冻结) conditions={len(conds)}", flush=True)
    print(f"[g5stab] cooldown={COOLDOWN}s（不进计时）；每条件独立进程；AR 亦双模型常驻",
          flush=True)
    G5_DIR = D.G5_DIR
    G5_DIR.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha256(
        json.dumps(dict(ctx=CTX, kvs=KVS, modes=MODES, blocks=BLOCKS, reps=REPS,
                        warm=WARM, maxnew=MAXNEW, chunk=CHUNK, seed=SEED,
                        k=K_PROTOCOL), sort_keys=True).encode()).hexdigest()[:12]
    out = S.g5_path(G5_DIR, cfg_hash, run_id)
    if out.exists():
        raise SystemExit(f"[拒绝覆盖] {out} 已存在")
    print(f"[g5stab] g5 jsonl = {out}", flush=True)

    per_cond: dict = {}
    block_state = []
    for b in range(BLOCKS):
        order = D.randomized_order(conds, SEED + b)
        labels = [f"{c},{kv},{m}" for c, kv, m in order]
        st = D.system_snapshot()
        block_state.append(dict(block=b, seed=SEED + b, ts=time.time(),
                                monitor=st, nvidia_smi_available=st.get("available")))
        print(f"\n[g5stab] ==== block {b}（seed={SEED + b}）系统状态："
              f"used={st.get('mem_used_MiB')}MiB temp={st.get('temp_C')}C "
              f"sm={st.get('clock_sm_MHz')}MHz 可用={st.get('available')}", flush=True)
        print(f"[g5stab] block {b} 条件顺序：{labels}", flush=True)
        for ci, (ctx, kv, mode) in enumerate(order):
            cid = f"g5stab_b{b}_c{ci}_{ctx}_{kv}_{mode}"
            info = D.spawn_condition(ctx=ctx, kv=kv, mode=mode, run_id=cid, reps=REPS,
                                     warm=WARM, maxnew=MAXNEW, chunk=CHUNK, k=K_PROTOCOL,
                                     order=f"b{b}", quiet=False)
            runs, sums, jpath = D.read_part4_rows(cid)
            kvs_done = med_iqr([r.get("decode_tok_s") for r in runs])
            kvs_dec = med_iqr([r.get("t_decode_s") for r in runs])
            peak = med_iqr([r.get("peak_alloc_GiB") for r in runs])
            resv = med_iqr([r.get("peak_resv_GiB") for r in runs])
            summ = (sums[0] if sums else {})
            status = "OK" if (runs and info["returncode"] == 0) else "CHECK"
            if not runs:
                status = "SKIP"
            per_cond.setdefault((ctx, kv, mode), []).append(
                dict(block=b, condition_index=ci, info=info, decode=kvs_done,
                     dec_s=kvs_dec, peak_alloc=peak, peak_resv=resv, summary=summ,
                     status=status, jsonl=str(jpath) if jpath else None))
            meta = D.g5_meta(run_id=run_id, block_index=b, condition_index=ci,
                             seed=SEED + b, condition_order=labels,
                             extra=dict(k=K_PROTOCOL, warmup=WARM, rep_index=None,
                                        chunk_size=CHUNK, context_requested=ctx,
                                        target_model="Qwen/Qwen2.5-1.5B-Instruct",
                                        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                                        compute_dtype="bfloat16", mode=mode,
                                        target_kv=kv, draft_kv="base_bf16",
                                        max_new_tokens=MAXNEW, prompt_family=None,
                                        prompt_hash=(runs[0].get("input_token_hash")
                                                     if runs else None),
                                        status=status,
                                        condition_order=labels))
            row = S.build_g5_row(dict(summ, **{
                "decode_tok_s": kvs_done["median"], "t_decode_s": kvs_dec["median"],
                "peak_alloc_GiB": peak["median"], "peak_resv_GiB": resv["median"],
                "monitor_end": (runs[-1].get("monitor_end") if runs else None),
                "logits_finite": all(bool(r.get("prefill_logits_finite")) for r in runs) if runs else None,
                "ctx": ctx, "mode": mode, "target_kv": kv, "warm": WARM, "chunk": CHUNK,
                "n_gen": (runs[-1].get("n_gen") if runs else None),
                "certified": (summ.get("certified") if summ else None),
            }), meta)
            row["notes"] = (f"stability probe block={b} cond_idx={ci}; "
                            f"k={K_PROTOCOL}（协议冻结）; cooldown={COOLDOWN}s; "
                            f"原 {os.path.basename(str(jpath)) if jpath else 'no-jsonl'}")
            S.write_g5_run(out, row, append=True)
            print(f"  [{b}.{ci}] ctx={ctx} {kv:<10} {mode:<8} exit={info['returncode']} "
                  f"dec_tok/s={kvs_done['median']} [{kvs_done['min']}..{kvs_done['max']}] "
                  f"rel={kvs_done['rel_spread_pct']} peak={peak['median']} "
                  f"status={status} wall={info['wall_s']:.0f}s", flush=True)
            if COOLDOWN > 0:
                time.sleep(COOLDOWN)

    # ---------------- 报表 ----------------
    print("\n== [B1] within-run（每条件每 block，5 rep）≡ median / min / max / MAD / IQR ==")
    print("ctx    kv         mode     blk  n  dec_tok/s  rel%   MAD    IQR    peak_alloc  "
          "status")
    for (ctx, kv, mode), lst in sorted(per_cond.items()):
        for e in sorted(lst, key=lambda x: x["block"]):
            d = e["decode"]
            med = round(d["median"], 2) if d["median"] is not None else None
            rel = f"{d['rel_spread_pct']:.1f}" if d["rel_spread_pct"] is not None else "n/a"
            mad = round(d["mad"], 2) if d["mad"] is not None else None
            iqr = round(d["iqr"], 2) if d["iqr"] is not None else None
            pk = round(e["peak_alloc"]["median"], 2) if e["peak_alloc"]["median"] else None
            print(f"{ctx:<6} {kv:<10} {mode:<8} {e['block']:<4} {d['n']}  {str(med):>9}  "
                  f"{rel:>5}  {str(mad):>5}  {str(iqr):>5}  {str(pk):>10}  {e['status']}")

    print("\n== [B2] between-block（两个独立随机顺序）差异 ==")
    print("ctx    kv         mode     med_b0   med_b1   order_ratio  rel_diff%  flag")
    compare = {}
    for (ctx, kv, mode), lst in sorted(per_cond.items()):
        meds = {e["block"]: e["decode"]["median"] for e in lst}
        if len(meds) < 2 or any(v is None for v in meds.values()):
            print(f"{ctx:<6} {kv:<10} {mode:<8} 不可比（需要 2 个 block 的有效中位数）")
            continue
        a, b = meds[0], meds[1]
        lo, hi = min(a, b), max(a, b)
        rel = 100.0 * (hi - lo) / hi
        flag = f"CHECK:between-block>{REL_LIMIT:g}%" if rel > REL_LIMIT else "ok"
        compare[(ctx, kv, mode)] = dict(med_b0=a, med_b1=b, order_ratio=lo / hi,
                                        rel_diff_pct=rel, flag=flag)
        print(f"{ctx:<6} {kv:<10} {mode:<8} {a:>8.2f} {b:>8.2f} {lo / hi:>12.3f} "
              f"{rel:>10.1f}  {flag}")

    print("\n== [B3] 系统状态（block 开始前）==")
    for bs in block_state:
        m = bs["monitor"]
        print(f"  block={bs['block']} seed={bs['seed']} available={bs['nvidia_smi_available']} "
              f"used={m.get('mem_used_MiB')}MiB/{m.get('mem_total_MiB')}MiB "
              f"temp={m.get('temp_C')}C power={m.get('power_W')}W "
              f"sm={m.get('clock_sm_MHz')}MHz mem={m.get('clock_mem_MHz')}MHz")

    print("\n== [B4] 预算：SpecDec/AR 同档比值（仅揭示数量级；不作加速结论）==")
    for ctx in CTX:
        for kv in KVS:
            if "AR" not in MODES or "SpecDec" not in MODES:
                continue
            a = per_cond.get((ctx, kv, "AR"))
            s_ = per_cond.get((ctx, kv, "SpecDec"))
            if not a or not s_:
                continue
            ma = statistics.median([e["decode"]["median"] for e in a
                                    if e["decode"]["median"] is not None] or [float("nan")])
            ms = statistics.median([e["decode"]["median"] for e in s_
                                    if e["decode"]["median"] is not None] or [float("nan")])
            if ma == ma and ms == ms:
                print(f"  ctx={ctx:<5} {kv:<10} SpecDec/AR decode = {ms / ma:.3f}×")

    bad = [k for k, v in compare.items() if "CHECK" in v["flag"]]
    print(f"\n[g5stab] 不稳定条件（between-block > {REL_LIMIT:g}%）："
          f"{bad if bad else '无'}")
    print(f"[g5stab] 预登记处理：上述条件 status=CHECK，**不得进入正式主表**，数据保留")
    print(f"[g5stab] g5 jsonl 行数 = {len(S.read_g5_rows(out))} -> {out}")
    print("[g5stab] 计时边界：decode_seconds 由子进程内 CUDA 同步夹住；"
          "cooldown/加载/文件写入均不进入计时")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
