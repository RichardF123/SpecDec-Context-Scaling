"""g5_stability_probe.py — G5 计时稳定化探针（Task B）+ AR 基线口径分离 + P-state 门禁。

条件定义（**必须分清，不许混算**）
  正式 primary factorial（**12 cells**，默认必须跑）：
      ctx {2048,4096,8192} × kv {base_bf16,v_fp8} × mode {AR-primary, SpecDec}
      AR-primary = **target_only**（子进程只加载 target，从不加载 draft；正式 speedup 分母）
      SpecDec    = target + draft 双驻留（算法必需）
  可选诊断控制（**6 cells**，`P5_INCLUDE_DUAL_AR_CONTROL=1`，**不计入 factorial**）：
      ctx × kv × AR-dual-resident-control

协议（跑之前固定）
  1. 每条件全新子进程；loading/tokenizer 不进计时；每 block 固定 seed 随机化，≥2 block；
  2. 每条件 warm-up=1 + measured reps=5；cooldown 不进计时；
  3. **P-state 门禁**：每个 block 开始前跑一次 active-load 预检（`gpu_pstate_preflight`）；
     不合格 ⇒ 该 block 标 `env_qualified=false` 并**跳过其条件**（不产生计时数据），
     除非显式 `P5_ALLOW_UNSTABLE_CLOCK=1`（则数据只能算探索性）；
  4. **状态分层**：`execution_status`(SUCCESS/ERROR/OOM) 与
     `measurement_status`(OK/CHECK/ENV_INVALID/NOT_MEASURED) 分开；
     **`exit=0` 不等于测量有效** —— 子进程 `spread_flag=CHECK` 必须把 measurement_status 降为 CHECK；
  5. **不稳定 cell**：保留原始数据、仍出现在完整结果表中、报 median/min/max/MAD/IQR、
     标 CHECK、不得作确定加速声明、不得从报告中消失；
  6. speedup 方向统一 `T_AR(target-only)/T_Spec`（>1 = SpecDec 更快）；
     只有当 `speedup_claim_allowed=true` 才写结论，否则打印具体 `NO CLAIM: 原因`。

用法（交互式 GPU 终端）：python -m specdec.engine.g5_stability_probe [--dry-run]
可选：P5_CTX / P5_KV / P5_MODE / P5_BLOCKS / P5_REPS / P5_WARM / P5_MAXNEW / P5_COOLDOWN
      P5_SEED / P5_INCLUDE_DUAL_AR_CONTROL / P5_REQUIRE_AC / P5_ALLOW_UNSTABLE_CLOCK
      P5_MIN_ACTIVE_SM_CLOCK_MHZ / P5_MIN_ACTIVE_MEM_CLOCK_MHZ / P5_PSTATE_PREFLIGHT_SECONDS
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
from . import gpu_pstate_preflight as PF

def _env_list(name: str, default: str) -> list[str]:
    """逗号分隔的环境变量列表；**逐项 strip**。

    为什么必须 strip：在 cmd 里写 `set P5_MODE=AR,SpecDec & set P5_BLOCKS=2` 时，
    `&` 前的空格会被并进变量值，得到 `'AR,SpecDec '` ⇒ 拆出 `' SpecDec'`。
    若不 strip，后面用 `m in MODES` 匹配会**静默丢掉整个 SpecDec**，
    于是"缩小版复验"实际只跑了 AR——**看起来成功、其实条件不全**，这是最危险的一类错误。
    """
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


CTX = [int(x) for x in _env_list("P5_CTX", "2048,4096,8192")]
KVS = _env_list("P5_KV", "base_bf16,v_fp8")
MODES = _env_list("P5_MODE", "AR,SpecDec")
BLOCKS = int(os.environ.get("P5_BLOCKS", "2"))
REPS = int(os.environ.get("P5_REPS", "5"))
WARM = int(os.environ.get("P5_WARM", "1"))
MAXNEW = int(os.environ.get("P5_MAXNEW", "64"))
CHUNK = int(os.environ.get("P5_CHUNK", "2048"))
COOLDOWN = float(os.environ.get("P5_COOLDOWN", "10"))
SEED = int(os.environ.get("P5_SEED", "20260921"))
INCLUDE_DUAL = S.env_flag("P5_INCLUDE_DUAL_AR_CONTROL", False)
REQUIRE_AC = S.env_flag("P5_REQUIRE_AC", True)
ALLOW_UNSTABLE = S.env_flag("P5_ALLOW_UNSTABLE_CLOCK", False)
K_PROTOCOL = 4


def validate_requested(ctypes: list[int] | None = None, kvs: list[str] | None = None,
                       modes: list[str] | None = None) -> None:
    """把**无法识别的**请求项变成硬错误，而不是静默少跑条件。

    `build_primary_matrix` 用的是 `m in MODES` 这种包含判断，未知 token 会被无声忽略；
    因此必须在入口处显式校验，避免"跑完了但条件不全"。
    """
    ctypes = CTX if ctypes is None else ctypes
    kvs = KVS if kvs is None else kvs
    modes = MODES if modes is None else modes
    bad = []
    # 空列表同样是危险状态：会静默跑出"0 个条件"的假成功
    for label, seq in (("P5_CTX", ctypes), ("P5_KV", kvs), ("P5_MODE", modes)):
        if not seq:
            bad.append(f"{label} 解析后为空（全空白或只写了逗号）⇒ 会跑出 0 个条件")
    for c in ctypes:
        if c not in S.CTX_CHOICES:
            bad.append(f"P5_CTX 含未知长度 {c}（允许 {S.CTX_CHOICES}）")
    for kv in kvs:
        if kv not in S.KV_CHOICES:
            bad.append(f"P5_KV 含未知格式 {kv!r}（允许 {S.KV_CHOICES}）")
    for m in modes:
        if m not in S.MODE_CHOICES:
            bad.append(f"P5_MODE 含未知模式 {m!r}（允许 {S.MODE_CHOICES}）")
    if bad:
        raise SystemExit("[g5stab] 环境变量校验失败（拒绝静默少跑条件）：\n  - "
                         + "\n  - ".join(bad))
REL_LIMIT = 20.0


def med_iqr(vals):
    v = sorted(float(x) for x in vals if isinstance(x, (int, float)) and x == x)
    if not v:
        return dict(n=0, median=None, min=None, max=None, mad=None, iqr=None, rel=None)
    med = statistics.median(v)
    sp = v[-1] - v[0]
    return dict(n=len(v), median=med, min=v[0], max=v[-1], mad=S.mad(v), iqr=S.iqr(v),
                rel=(100.0 * sp / abs(med)) if med else None)


def build_cells():
    """返回 (primary_cells, control_cells)。cell = (ctx, kv, mode, ar_residency|None)。"""
    primary = []
    for ctx, kv, mode in S.build_primary_matrix(CTX, KVS,
                                                [m for m in S.MODE_CHOICES if m in MODES]):
        role = S.resolve_condition_role(mode, S.PRIMARY_AR_RESIDENCY if mode == "AR" else None)
        primary.append((ctx, kv, mode, role["ar_residency"]))
    control = []
    if INCLUDE_DUAL:
        for ctx, kv, mode in S.build_dual_controls(CTX, KVS):
            role = S.resolve_condition_role(mode, S.DIAGNOSTIC_AR_RESIDENCY)
            control.append((ctx, kv, mode, role["ar_residency"]))
    return primary, control


def cell_label(c) -> str:
    ctx, kv, mode, res = c
    return f"{ctx},{kv},{mode},{res or '-'}"


def median_of(recs, key):
    vals = [r[key]["median"] for r in recs
            if isinstance(r.get(key), dict) and isinstance(r[key].get("median"), (int, float))
            and r[key]["median"] == r[key]["median"]]
    return statistics.median(vals) if vals else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="G5 稳定性探针（含 P-state 门禁）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印条件计划与随机化顺序，不启动子进程、不写 JSONL")
    args = ap.parse_args()

    run_id = S.default_run_id("g5stab")
    stamp = run_id.split("_", 1)[1] if "_" in run_id else run_id
    validate_requested()          # 拒绝未知条件名（防"跑完但条件不全"）
    primary, control = build_cells()
    print(f"[g5stab] run_id={run_id} seed={SEED} blocks={BLOCKS} reps={REPS} warm={WARM} "
          f"k={K_PROTOCOL}(协议冻结)", flush=True)
    print(f"[g5stab] primary factorial = **{len(primary)} cells**"
          f"（AR=target_only + SpecDec=dual-resident）", flush=True)
    print(f"[g5stab] diagnostic control = **{len(control)} cells**"
          f"（AR=dual_control；{'启用' if INCLUDE_DUAL else '未启用'}）—— **不计入 factorial**",
          flush=True)
    print(f"[g5stab] P-state 阈值(本机经验值, 可配置): SM>={D.MIN_SM:.0f}MHz "
          f"mem>={D.MIN_MEM:.0f}MHz | 预检 {PF.SECONDS}s", flush=True)

    st0 = D.platform_state()
    blockers = D.platform_gate(st0, require_ac=REQUIRE_AC)
    print(f"[g5stab] 平台状态：power_scheme={st0.get('power_scheme')} "
          f"on_ac={st0.get('on_ac')} sm_clock_now={st0.get('sm_clock_now')}", flush=True)
    if st0.get("warnings"):
        print(f"[g5stab] 平台状态告警：{st0['warnings']}", flush=True)
    if blockers and not ALLOW_UNSTABLE:
        for b in blockers:
            print(f"[g5stab] !! 阻断：{b}", flush=True)
        raise SystemExit("[g5stab] 平台状态不满足正式计时要求 -> 已中止（未启动任何条件）。\n"
                         "          处理：插电源、切高性能电源方案、确认 SM 时钟满频。\n"
                         "          若确要继续（数据只能算探索性）：P5_ALLOW_UNSTABLE_CLOCK=1")
    if blockers:
        print("[g5stab] P5_ALLOW_UNSTABLE_CLOCK=1 -> 继续，但本批数据只能标为探索性", flush=True)

    D.G5_DIR.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha256(json.dumps(
        dict(ctx=CTX, kvs=KVS, modes=MODES, blocks=BLOCKS, reps=REPS, warm=WARM,
             maxnew=MAXNEW, chunk=CHUNK, seed=SEED, k=K_PROTOCOL, dual=INCLUDE_DUAL,
             min_sm=D.MIN_SM, min_mem=D.MIN_MEM, preflight_s=PF.SECONDS),
        sort_keys=True).encode()).hexdigest()[:12]
    out = S.g5_path(D.G5_DIR, cfg_hash, run_id)
    all_cells = primary + control
    if args.dry_run:
        print(f"\n[g5stab] --dry-run：将执行 {len(primary)} primary + {len(control)} control "
              f"= {len(all_cells)} 次条件运行 × {BLOCKS} block"
              f"（**正式 factorial 仍为 {len(primary)} cells**；control 不计入）", flush=True)
        print(f"[g5stab] 输出文件（dry-run 不创建）= {out}", flush=True)
        for b in range(BLOCKS):
            order = D.randomized_order(all_cells, SEED + b)
            print(f"  block {b}（seed={SEED + b}）执行顺序：", flush=True)
            for i, c in enumerate(order):
                print(f"    {i:2d}. [{'ctl' if c in control else 'pri'}] {cell_label(c)}",
                      flush=True)
        print("[g5stab] dry-run 结束：未启动任何子进程、未写任何文件", flush=True)
        return 0
    if out.exists():
        raise SystemExit(f"[拒绝覆盖] {out} 已存在")
    print(f"[g5stab] g5 jsonl = {out}", flush=True)

    per_cell: dict = {}
    block_state = []
    for b in range(BLOCKS):
        order = D.randomized_order(all_cells, SEED + b)
        labels = [cell_label(c) for c in order]
        pf = PF.qualify()
        env_ok = bool(pf.get("available") and pf.get("qualified"))
        block_state.append(dict(block=b, seed=SEED + b, monitor=D.system_snapshot(),
                                pstate=pf, env_ok=env_ok))
        print(f"\n[g5stab] ==== block {b}（seed={SEED + b}）P-state 预检："
              f"available={pf.get('available')} qualified={pf.get('qualified')} "
              f"sm_med={pf.get('sm_median')} mem_med={pf.get('mem_median')} "
              f"power={pf.get('power_median')} reason={pf.get('reason')}", flush=True)
        if not env_ok:
            print(f"[g5stab] !! block {b} 环境不合格 -> {'跳过' if not ALLOW_UNSTABLE else '仍继续(探索性)'}",
                  flush=True)
            if not ALLOW_UNSTABLE:
                continue
        for ci, cell in enumerate(order):
            ctx, kv, mode, res = cell
            tag = "ctl" if cell in control else "pri"
            # **必须**把本次 run 的 UTC 时间戳放进子进程 run_id：否则两次相同配置的探针
            # 会产生**同名**分片文件，而测量核心在 P4_ONLY 模式下是**追加**写，
            # 于是第二次运行的数据会被追加进第一次的分片里（污染受保护数据 + 混两个环境）。
            cid = (f"g5stab_{stamp}_b{b}_c{ci}_{tag}_{ctx}_{kv}_{mode}_{res or 'na'}")
            prior = sorted(D.PART4_DIR.glob(f"part4_*_{cid}.jsonl"))
            if prior:
                raise SystemExit(f"[g5stab] 拒绝写入已存在的分片（防跨 run 追加污染）：{prior[0]}")
            info = D.spawn_condition(ctx=ctx, kv=kv, mode=mode, run_id=cid, reps=REPS,
                                     warm=WARM, maxnew=MAXNEW, chunk=CHUNK, k=K_PROTOCOL,
                                     order=f"b{b}", ar_residency=res)
            runs, sums, jpath = D.read_part4_rows(cid)
            summ = (sums[0] if sums else {})
            rec = dict(block=b, condition_index=ci, info=info, runs=runs, summary=summ,
                       role=tag, jsonl=str(jpath) if jpath else None,
                       decode=med_iqr([r.get("decode_tok_s") for r in runs]),
                       dec_s=med_iqr([r.get("t_decode_s") for r in runs]),
                       tot_s=med_iqr([r.get("t_total_s") for r in runs]),
                       peak=med_iqr([r.get("peak_alloc_GiB") for r in runs]),
                       resv=med_iqr([r.get("peak_resv_GiB") for r in runs]),
                       clk=D.clock_stats(runs), pstate=D.pstate_audit(runs))
            per_cell.setdefault((ctx, kv, mode, res), []).append(rec)
            print(f"  [{b}.{ci}|{tag}] ctx={ctx} {kv:<10} {mode:<8} res={str(res):<12} "
                  f"exit={info['returncode']} dec_tok/s={rec['decode']['median']} "
                  f"rel={rec['decode']['rel']} sm_med={rec['pstate'].get('sm_median')} "
                  f"pstate_ok={rec['pstate'].get('qualified')} "
                  f"clk_stable={rec['clk'].get('stable')} wall={info['wall_s']:.0f}s", flush=True)
            if COOLDOWN > 0:
                time.sleep(COOLDOWN)

    # ---------------- between-block（需要全部 block 完成后才能算）----------------
    between: dict = {}
    for cell, recs in per_cell.items():
        meds = {r["block"]: r["decode"]["median"] for r in recs}
        if len(meds) < 2 or any(v is None for v in meds.values()):
            continue
        hi, lo = max(meds.values()), min(meds.values())
        rel = 100.0 * (hi - lo) / hi
        between[cell] = dict(rel=rel, ratio=lo / hi,
                             flag=(f"CHECK:between-block>{REL_LIMIT:g}%" if rel > REL_LIMIT else "ok"))

    # ---------------- 写 G5 行（含状态分层 / P-state / between-block）----------------
    print(f"\n[g5stab] 写出 G5 行 -> {out}", flush=True)
    for cell, recs in per_cell.items():
        ctx, kv, mode, res = cell
        for rec in recs:
            b = rec["block"]
            blk = next((x for x in block_state if x["block"] == b), {})
            ps, clk = rec["pstate"], rec["clk"]
            bt = between.get(cell, {})
            meta = D.g5_meta(
                run_id=run_id, block_index=b, condition_index=rec["condition_index"],
                seed=SEED + b, condition_order=[cell_label(c) for c in all_cells],
                extra=dict(k=K_PROTOCOL, warmup=WARM, chunk_size=CHUNK,
                           context_requested=ctx, context_actual=ctx,
                           target_model="Qwen/Qwen2.5-1.5B-Instruct",
                           draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                           compute_dtype="bfloat16", mode=mode,
                           ar_residency=res if mode == "AR" else None,
                           target_kv=kv, draft_kv="base_bf16", max_new_tokens=MAXNEW,
                           prompt_family=None,
                           prompt_hash=(rec["runs"][0].get("input_token_hash")
                                        if rec["runs"] else None),
                           returncode=rec["info"]["returncode"],
                           reps_expected=REPS, reps_measured=len(rec["runs"]),
                           between_block_rel_diff_pct=bt.get("rel"),
                           between_block_check=(bt.get("flag", "").startswith("CHECK")
                                                if bt else None),
                           env_qualified=blk.get("env_ok"),
                           pstate_qualified=ps.get("qualified"),
                           pstate_reason=ps.get("reason"),
                           active_sm_clock_mhz=ps.get("sm_median"),
                           active_memory_clock_mhz=ps.get("mem_median"),
                           active_power_w=ps.get("power_median")))
            src = dict(rec["summary"], **{
                "decode_tok_s": rec["decode"]["median"], "t_decode_s": rec["dec_s"]["median"],
                "t_total_s": rec["tot_s"]["median"], "peak_alloc_GiB": rec["peak"]["median"],
                "peak_resv_GiB": rec["resv"]["median"], "mode": mode, "ctx": ctx,
                "target_kv": kv, "warm": WARM, "chunk": CHUNK, "k": K_PROTOCOL,
                "monitor_end": (rec["runs"][-1].get("monitor_end") if rec["runs"] else None),
                "logits_finite": (all(bool(r.get("prefill_logits_finite")) for r in rec["runs"])
                                  if rec["runs"] else None),
                "n_gen": (rec["runs"][-1].get("n_gen") if rec["runs"] else None),
                "certified": (rec["summary"].get("certified") if rec["summary"] else None)})
            if not rec["runs"]:
                src["oom"] = True
            row = S.build_g5_row(src, meta)
            row["notes"] = (f"stability probe block={b} cond_idx={rec['condition_index']} "
                            f"role={rec['role']}; k={K_PROTOCOL}(冻结); cooldown={COOLDOWN}s; "
                            f"clk_stable={clk.get('stable')}; "
                            f"原 {os.path.basename(rec['jsonl'] or 'no-jsonl')}")
            S.write_g5_run(out, row, append=True)

    # ---------------- 报表 ----------------
    def table(cells_list, title):
        print(f"\n== {title} ==")
        print("ctx    kv         mode     residency      blk n  dec_tok/s  min     max     MAD  "
              " IQR  rel%   sm_MHz pstate env  status")
        for c in cells_list:
            for rec in sorted(per_cell.get(c, []), key=lambda x: x["block"]):
                d = rec["decode"]
                f2 = lambda v: "n/a" if v is None else f"{v:.2f}"          # noqa: E731
                rel = f"{d['rel']:.1f}" if d["rel"] is not None else "n/a"
                st = S.derive_statuses(returncode=rec["info"]["returncode"], oom=not rec["runs"],
                                       spread_flag=rec["summary"].get("spread_flag"),
                                       pstate_qualified=rec["pstate"].get("qualified"))
                blk = next((x for x in block_state if x["block"] == rec["block"]), {})
                print(f"{c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} {rec['block']:<3} "
                      f"{d['n']}  {f2(d['median']):>9} {f2(d['min']):>6} {f2(d['max']):>6} "
                      f"{f2(d['mad']):>5} {f2(d['iqr']):>5} {rel:>5}  "
                      f"{f2(rec['pstate'].get('sm_median')):>6} "
                      f"{str(rec['pstate'].get('qualified')):<6} "
                      f"{str(blk.get('env_ok')):<5} {st['status']}")

    table(primary, f"[B1a] primary（{len(primary)} cells）within-block：median/min/max/MAD/IQR")
    if control:
        table(control, f"[B1b] 诊断控制（{len(control)} cells，非 factorial）")

    print("\n== [B2] between-block（两个独立随机顺序）==")
    print("group  ctx    kv         mode     residency      rel_diff%  flag")
    for group, cells_list in (("primary", primary), ("control", control)):
        for c in cells_list:
            bt = between.get(c)
            if not bt:
                print(f"{group:<6} {c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} "
                      f"不可比（需两个 block 的有效中位数）")
                continue
            print(f"{group:<6} {c[0]:<6} {c[1]:<10} {c[2]:<8} {str(c[3] or '-'):<13} "
                  f"{bt['rel']:>10.1f}  {bt['flag']}")

    print("\n== [B3] block 级环境资格（P-state 预检 + 全 cell P-state）==")
    for blk in block_state:
        pf = blk["pstate"]
        print(f"  block={blk['block']} seed={blk['seed']} env_ok={blk['env_ok']} "
              f"preflight(available={pf.get('available')}, qualified={pf.get('qualified')}, "
              f"sm_med={pf.get('sm_median')}, mem_med={pf.get('mem_median')}, "
              f"power={pf.get('power_median')}) reason={pf.get('reason')}")

    print("\n== [B4] speedup = T_AR(target-only) / T_Spec（>1 = SpecDec 更快）==")
    for ctx in CTX:
        for kv in KVS:
            ar_c = (ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY)
            sp_c = (ctx, kv, "SpecDec", None)
            if not per_cell.get(ar_c) or not per_cell.get(sp_c):
                continue
            bt_ar = between.get(ar_c, {}).get("flag", "")
            bt_sp = between.get(sp_c, {}).get("flag", "")
            ar_rec, sp_rec = per_cell[ar_c][0], per_cell[sp_c][0]
            base = dict(context_actual=ctx, target_kv=kv, max_new_tokens=MAXNEW,
                        input_token_hash=(ar_rec["runs"][0].get("input_token_hash")
                                          if ar_rec["runs"] else None),
                        reps_measured=min(len(ar_rec["runs"]), len(sp_rec["runs"])),
                        reps_expected=REPS, tokens_finite=True, profile=None)
            ar_row = dict(base, mode="AR", ar_residency=S.PRIMARY_AR_RESIDENCY,
                          condition_role="primary", speedup_denominator_eligible=True,
                          execution_status="SUCCESS",
                          measurement_status=("CHECK" if "CHECK" in bt_ar else "OK"),
                          between_block_check="CHECK" in bt_ar,
                          between_block_rel_diff_pct=between.get(ar_c, {}).get("rel"),
                          within_run_check_reason=ar_rec["summary"].get("spread_flag"),
                          decode_seconds=median_of([r for r in per_cell[ar_c]], "dec_s"),
                          total_seconds=median_of([r for r in per_cell[ar_c]], "tot_s"))
            sp_row = dict(base, mode="SpecDec", ar_residency=None, condition_role="primary",
                          speedup_denominator_eligible=False, execution_status="SUCCESS",
                          measurement_status=("CHECK" if "CHECK" in bt_sp else "OK"),
                          between_block_check="CHECK" in bt_sp,
                          between_block_rel_diff_pct=between.get(sp_c, {}).get("rel"),
                          within_run_check_reason=sp_rec["summary"].get("spread_flag"),
                          decode_seconds=median_of([r for r in per_cell[sp_c]], "dec_s"),
                          total_seconds=median_of([r for r in per_cell[sp_c]], "tot_s"))
            if ar_rec["pstate"].get("qualified") is False:
                ar_row["measurement_status"] = "ENV_INVALID"
            if sp_rec["pstate"].get("qualified") is False:
                sp_row["measurement_status"] = "ENV_INVALID"
            su = S.compute_speedup(ar_row, sp_row, rel_limit=REL_LIMIT)
            verdict = ("CLAIM ALLOWED（全部资格条件满足）" if su["speedup_claim_allowed"]
                       else "；".join(su["speedup_claim_blockers"]))
            print(f"  ctx={ctx:<5} {kv:<10} decode×{su['speedup_decode']:.3f} "
                  f"e2e×{su['speedup_e2e']:.3f}  -> {verdict}")

    print("\n== [B5] 不稳定 cell（原始数据保留、仍在完整结果表中、标 CHECK、不得作确定声明）==")
    n_bad = 0
    for cell, bt in between.items():
        if "CHECK" in bt["flag"]:
            n_bad += 1
            print(f"  {cell_label(cell)} rel={bt['rel']:.1f}%")
    for cell, recs in per_cell.items():
        for rec in recs:
            flag = rec["summary"].get("spread_flag")
            if flag and "CHECK" in str(flag):
                n_bad += 1
                print(f"  {cell_label(cell)} block{rec['block']} within-run {flag}")
            if rec["pstate"].get("qualified") is False:
                n_bad += 1
                print(f"  {cell_label(cell)} block{rec['block']} ENV_INVALID："
                      f"{rec['pstate'].get('reason')}")
    if n_bad == 0:
        print("  无：全部 cell 的 within-run / between-block / P-state 均通过")
    print(f"\n[g5stab] g5 jsonl 行数 = {len(S.read_g5_rows(out))} -> {out}")
    print("[g5stab] 计时边界：子进程内 CUDA 同步夹住 prefill/decode；cooldown/加载/预检均不计时")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
