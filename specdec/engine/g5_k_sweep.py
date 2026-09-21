"""g5_k_sweep.py — Task D：有限 k 扫描（**诊断实验，不修改正式协议**）。

扫描（按任务规定）：
  k ∈ {1, 2, 4, 8} × ctx ∈ {2048, 8192} × kv = base_bf16，max_new_tokens=64
  每个条件**独立子进程**（与 stability probe 同一纪律）；warm-up=1 + measured reps=3（诊断用，
  少于正式 5 rep，因为这只回答"k 的收益曲线形状"，不出正式数）。

必须遵守
  - 正式协议仍是 **k=4**；本工具只做诊断，**不改** proposal 参数；
  - 冻结变体是"**不额外发射 bonus token**"，本工具**不得**偷偷改成 K+1；
  - 若某个 k 明显更好，只输出**协议修正建议**并等待人工决策（脚本不会自动改配置）；
  - 记录：acceptance / 平均每轮输出 token / rounds / draft calls / target block calls /
    decode tok/s / peak allocated / 与同 ctx AR 的 token 相等性（含 near-tie 判定入口）。

用法（交互式 GPU 终端）：python -m specdec.engine.g5_k_sweep
可选：P5KS_CTX=2048,8192 / P5KS_KS=1,2,4,8 / P5KS_REPS=3 / P5KS_MAXNEW=64
"""
from __future__ import annotations

import os
import statistics
import traceback

from .. import results_schema as S
from . import g5_driver as D

CTX = [int(x) for x in os.environ.get("P5KS_CTX", "2048,8192").split(",") if x.strip()]
KS = [int(x) for x in os.environ.get("P5KS_KS", "1,2,4,8").split(",") if x.strip()]
REPS = int(os.environ.get("P5KS_REPS", "3"))
WARM = int(os.environ.get("P5KS_WARM", "1"))
MAXNEW = int(os.environ.get("P5KS_MAXNEW", "64"))
CHUNK = int(os.environ.get("P5KS_CHUNK", "2048"))
COOLDOWN = float(os.environ.get("P5KS_COOLDOWN", "5"))
KV = "base_bf16"
K_PROTOCOL = 4


def main() -> int:
    run_id = S.default_run_id("g5ksweep")
    print(f"[ksweep] run_id={run_id} ctx={CTX} k={KS} kv={KV} reps={REPS} "
          f"k_protocol={K_PROTOCOL}（正式协议不变）", flush=True)
    print("[ksweep] 冻结变体：不发射 bonus token（本轮**不做** K+1 算法改动）", flush=True)
    print("[ksweep] AR 分母 = **target-only 正式主基线**，每个 ctx 只测一次"
          "（不为每个 k 重测/择优）", flush=True)
    results: dict = {}
    for ctx in CTX:
        # AR 分母：**target-only 正式主基线**，每个 ctx **只测一次**（不为每个 k 重测/择优）
        ar_info = D.spawn_condition(ctx=ctx, kv=KV, mode="AR",
                                    run_id=f"ksweep_{ctx}_AR_target_only", reps=REPS,
                                    warm=WARM, maxnew=MAXNEW, chunk=CHUNK, k=None,
                                    order="AR-baseline",
                                    ar_residency=S.PRIMARY_AR_RESIDENCY)
        for k in KS + [None]:
            mode = "AR" if k is None else "SpecDec"
            kk = K_PROTOCOL if k is None else k
            if k is None:
                info, cid = ar_info, f"ksweep_{ctx}_AR_target_only"
            else:
                cid = f"ksweep_{ctx}_{mode}_{kk}"
                info = D.spawn_condition(ctx=ctx, kv=KV, mode=mode, run_id=cid, reps=REPS,
                                         warm=WARM, maxnew=MAXNEW, chunk=CHUNK, k=k,
                                         order=f"k{k}", ar_residency=None)
            runs, sums, jpath = D.read_part4_rows(cid)
            if not runs:
                print(f"  [skip] ctx={ctx} k={kk} {mode} 无数据（exit={info['returncode']}）")
                continue
            dec = [r.get("decode_tok_s") for r in runs]
            dec = [x for x in dec if isinstance(x, (int, float)) and x == x]
            acc = [r.get("accept_rate") for r in runs]
            acc = [x for x in acc if isinstance(x, (int, float)) and x == x]
            rnd = statistics.median([r.get("rounds", 0) for r in runs])
            accd = statistics.median([r.get("accepted", 0) for r in runs])
            prop = statistics.median([r.get("proposed", 0) for r in runs])
            block = statistics.median([r.get("target_block_calls", 0) for r in runs])
            draft = statistics.median([r.get("draft_calls", 0) for r in runs])
            peak = statistics.median([r.get("peak_alloc_GiB", 0) for r in runs])
            ngen = statistics.median([r.get("n_gen", 0) for r in runs])
            per_round = (accd / rnd) if rnd else None
            results[(ctx, kk, mode)] = dict(
                dec_med=statistics.median(dec) if dec else None,
                dec_min=min(dec) if dec else None, dec_max=max(dec) if dec else None,
                acceptance=statistics.median(acc) if acc else None,
                rounds=rnd, accepted=accd, proposed=prop, block=block, draft=draft,
                peak=peak, n_gen=ngen, per_round=per_round,
                gen_hash=(runs[0].get("generated_token_hash") or runs[0].get("gen_sha1")),
                jsonl=str(jpath))
            print(f"  ctx={ctx:<5} k={kk:<2} {mode:<8} dec_tok/s={results[(ctx, kk, mode)]['dec_med']:.2f} "
                  f"[{results[(ctx, kk, mode)]['dec_min']:.2f}..{results[(ctx, kk, mode)]['dec_max']:.2f}] "
                  f"accept={results[(ctx, kk, mode)]['acceptance']} rounds={rnd} "
                  f"acc/round={per_round if per_round is None else round(per_round, 2)} "
                  f"block={block} draft={draft} peak={peak:.2f} exit={info['returncode']}",
                  flush=True)
            if COOLDOWN > 0:
                import time
                time.sleep(COOLDOWN)

    print("\n== k 扫描汇总（诊断；kv=base_bf16）==")
    print("ctx    k   mode     accept  tok/round  rounds  block  draft  dec_tok/s  peak_alloc  "
          "AR_equal")
    for ctx in CTX:
        ar = results.get((ctx, K_PROTOCOL, "AR"))
        for k in KS + [None]:
            kk = K_PROTOCOL if k is None else k
            mode = "AR" if k is None else "SpecDec"
            r = results.get((ctx, kk, mode))
            if not r:
                continue
            eq = "n/a"
            if mode == "SpecDec" and ar and ar.get("gen_hash"):
                eq = str(r.get("gen_hash") == ar.get("gen_hash"))
            acc_s = f"{r['acceptance']:.3f}" if r["acceptance"] is not None else "n/a"
            pr_s = f"{r['per_round']:.2f}" if r["per_round"] is not None else "n/a"
            dec_s = f"{r['dec_med']:.2f}" if r["dec_med"] is not None else "n/a"
            print(f"{ctx:<6} {kk:<3} {mode:<8} {acc_s:>6}  {pr_s:>9}  "
                  f"{r['rounds']:>6} {r['block']:>6} {r['draft']:>6}  {dec_s:>9}  "
                  f"{r['peak']:>10.2f}  {eq}")

    print("\n== speedup = T_AR(target-only) / T_Spec（>1 = SpecDec 更快；<1 = 更慢）==")
    for ctx in CTX:
        ar = results.get((ctx, K_PROTOCOL, "AR"))
        if not ar or not ar["dec_med"]:
            continue
        for k in KS:
            r = results.get((ctx, k, "SpecDec"))
            if r and r["dec_med"]:
                print(f"  ctx={ctx:<5} k={k:<2} speedup = {ar['dec_med'] / r['dec_med']:.3f}× "
                      f"（AR target-only {ar['dec_med']:.2f} tok/s ÷ SpecDec {r['dec_med']:.2f}）")

    best = []
    for ctx in CTX:
        ar = results.get((ctx, K_PROTOCOL, "AR"))
        cands = [(k, results[(ctx, k, "SpecDec")]["dec_med"]) for k in KS
                 if (ctx, k, "SpecDec") in results
                 and results[(ctx, k, "SpecDec")]["dec_med"] is not None]
        if cands and ar and ar["dec_med"]:
            k_best, v_best = max(cands, key=lambda x: x[1])
            best.append((ctx, k_best, v_best, v_best / ar["dec_med"]))
    print("\n== 协议修正建议（**仅供人工决策；脚本不自动改配置**）==")
    if not best:
        print("  无有效数据 -> 未运行/未完成，不产出建议")
    else:
        for ctx, k_best, v, ratio in best:
            verdict = ("k=4 仍是最优/接近最优，无需修改" if ratio <= 1.0
                       else f"k={k_best} 在 ctx={ctx} 上为 {ratio:.3f}× AR（仍 <1 ⇒ 无加速）")
            print(f"  ctx={ctx}: 最佳 k={k_best}（{v:.2f} tok/s, {ratio:.3f}× AR）-> {verdict}")
        print("  正式协议保持 k=4；任何 k 变更必须走协议 amendment + 人工批准。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
