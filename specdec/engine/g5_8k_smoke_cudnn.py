"""g5_8k_smoke_cudnn.py — **8K 单次 smoke**（条件性工具；只回答"能不能跑、峰值多少"）。

它**不是**吞吐测量，也**不是**稳定性验证。纪律（违反即数据作废）
  - **单次**运行，`prefill + 少量 decode`，不启动正式矩阵、不做长时间循环；
  - 上下文 / chunk / k / dtype 全部**冻结**，本工具只提供默认值，不在此处调参；
  - `allocated` **不是**全部物理显存占用；`reserved` 超过显存**也不单独**作为分页证据
    ⇒ 同时记录 PyTorch allocated/reserved/peak **与** 设备级 `nvidia-smi` 占用，两者分开陈述；
  - 计时字段仅供诊断（**含插桩**），**禁止**进入任何 speedup 表；
  - 发现结构错误 / 质量明显异常 / OOM ⇒ **保存结果并停止**，不升级测试；
  - 双模型都用**降峰值加载器**（`lowmem_load`，权重逐位校验）以提高一次跑成的概率；
    该选择写入结果，不与其他工具的加载路径混为一谈。

用法（交互式 GPU 终端）
  scripts\\run_g5_8k_smoke.cmd              （= 本文档默认配置）
环境变量
  SMOKE_CTX=8192 SMOKE_CHUNK=2048 SMOKE_K=4 SMOKE_MAXNEW=16 SMOKE_KV=base_bf16
  SMOKE_ARM=cudnn_prefill|eager   默认 cudnn_prefill（eager 在 8K 上历史上必 OOM，仅作对照）
  SMOKE_ALLOW_UNQUALIFIED=1       允许 commit/P-state 不合格仍跑（**数据只能算探索性**）
  SMOKE_DRY_RUN=1                 只打印计划，不加载、不占显存
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V
from . import g5_driver as D

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

CTX = int(os.environ.get("SMOKE_CTX", "8192"))
CHUNK = int(os.environ.get("SMOKE_CHUNK", "2048"))
K = int(os.environ.get("SMOKE_K", "4"))
MAXNEW = int(os.environ.get("SMOKE_MAXNEW", "16"))
KV = os.environ.get("SMOKE_KV", "base_bf16")
ARM = os.environ.get("SMOKE_ARM", "cudnn_prefill")
ALLOW_UNQUALIFIED = S.env_flag("SMOKE_ALLOW_UNQUALIFIED", False)
DRY_RUN = S.env_flag("SMOKE_DRY_RUN", False)


def mem(tag: str, rec: dict) -> None:
    """同时记录 **PyTorch 侧** 与 **设备侧**；两者含义不同，报告里必须分开写。"""
    free, total = torch.cuda.mem_get_info()
    smi = D.system_snapshot()
    blk = dict(tag=tag,
               pt_allocated_mib=torch.cuda.memory_allocated() / 2 ** 20,
               pt_reserved_mib=torch.cuda.memory_reserved() / 2 ** 20,
               pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
               pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
               cuda_free_mib=free / 2 ** 20, cuda_total_mib=total / 2 ** 20,
               smi_mem_used_MiB=smi.get("mem_used_MiB"),
               smi_mem_total_MiB=smi.get("mem_total_MiB"),
               smi_util_pct=smi.get("util_pct"),
               note="pt_* 是 PyTorch 记账（allocated≠全部物理占用；reserved 是缓存池上限）")
    rec.setdefault("mem_timeline", []).append(blk)
    print(f"[smoke] mem {tag:<14} pt_alloc={blk['pt_allocated_mib']:7.1f} "
          f"pt_reserved={blk['pt_reserved_mib']:7.1f} "
          f"pt_peak_res={blk['pt_peak_reserved_mib']:7.1f} "
          f"cuda_free={blk['cuda_free_mib']:7.1f} smi_used={blk['smi_mem_used_MiB']}", flush=True)


def _out_path(run_id: str) -> Path:
    return OUTDIR / f"smoke8k_{ARM}_{run_id}.json"


def _write(rec: dict, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _out_path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_8k_smoke_cudnn",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     config=dict(ctx=CTX, chunk=CHUNK, k=K, max_new=MAXNEW, kv=KV, arm=ARM,
                                 target_model=TARGET, draft_model=DRAFT,
                                 load_path="lowmem_load.load_ours_bf16 (双模型, 权重逐位校验)"),
                     scope="**单次 smoke**：只回答能不能跑 + 峰值多少。不是吞吐测量，不是稳定性验证。",
                     timing_policy="计时字段含插桩，仅供诊断，禁止进入 speedup 表")
    print(f"[smoke] run_id={run_id} arm={ARM} ctx={CTX} chunk={CHUNK} k={K} "
          f"max_new={MAXNEW} kv={KV}")
    if DRY_RUN:
        rec["stage"] = "dry_run"
        _write(rec, run_id)
        print(f"[smoke] DRY RUN（未加载、未占显存）-> {_out_path(run_id)}")
        return 0

    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=ok, why=why, system=P5V.system_memory(),
                               line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    rec["gpu_consumers_before"] = P5V.gpu_consumers()
    print(f"[smoke] commit: {why}")
    print(f"[smoke] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')} "
          f"mem={pf.get('mem_median')}")
    print(f"[smoke] GPU 上其它 C+G 进程={rec['gpu_consumers_before'].get('count')}")
    if (not ok or pf.get("qualified") is not True) and not ALLOW_UNQUALIFIED:
        rec["stage"] = "refused"
        rec["skipped"] = f"环境不合格：commit_ok={ok} pstate={pf.get('qualified')}"
        _write(rec, run_id)
        print(f"[smoke] !! 环境不合格，未加载：{rec['skipped']}")
        return 3
    rec["allow_unqualified"] = ALLOW_UNQUALIFIED

    from .lowmem_load import load_ours_bf16
    try:
        # **关闭所有诊断精度开关**：本 smoke 只回答"能否完成完整双模型流程"，
        # 不许把诊断用的精度干预混进来（它们只在 §9 的定位实验里启用）。
        rec["diag_precision"] = M.set_diag_preset("eager")
        rec["profiler_used"] = False              # 本工具不使用 profiler
        rec["default_strategy_at_start"] = M.attention_strategy()

        mem("before_load", rec)
        t0 = time.perf_counter()
        tgt, tok, tinfo = load_ours_bf16(TARGET)
        rec["stage_target_loaded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mem("after_target_load", rec)
        _write(rec, run_id)                       # 逐阶段落盘：target 加载完成
        drf, _t2, dinfo = load_ours_bf16(DRAFT)
        rec["load"] = dict(target=tinfo, draft=dinfo, load_s=time.perf_counter() - t0)
        rec["weights_bitmatch"] = bool(tinfo["weights_bitmatch"] and dinfo["weights_bitmatch"])
        rec["weights_checked"] = dict(target=tinfo["n_params_checked"],
                                      draft=dinfo["n_params_checked"])
        rec["stage"] = "loaded"
        rec["stage_draft_loaded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mem("after_load", rec)
        _write(rec, run_id)                       # 加载是历史崩溃高发点
        if not rec["weights_bitmatch"]:
            print("[smoke] !! 权重校验未通过 -> 停止（不产生后续数字）")
            rec["stage"] = "aborted_weight_mismatch"
            _write(rec, run_id)
            return 2

        tgt.set_kv_format(spec_dec._fmt(KV))
        # `set_kv_format` 收的是**引擎名**（'fp16'/'v_fp8'/'fp8'），不是实验名（'base_bf16'）
        drf.set_kv_format("fp16")   # draft head_dim=64 < GROUP_V=128 ⇒ 只能 base(bf16 K/V)
        rec["kv_format"] = dict(target=spec_dec._fmt(KV), draft="fp16")
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()

        full = P5V.load_ids(tok)
        ids = full[:, :CTX].contiguous()
        rec["input"] = dict(ctx=int(ids.shape[1]), ids_hash=P5V._gen_hash(
            [int(x) for x in ids[0].tolist()]),
            source="wikitext-2-raw test 前缀（与其它 G5 工具同源）",
            vocab_size=int(tgt.embed_tokens.weight.shape[0]),
            dtypes=dict(model=str(next(tgt.parameters()).dtype),
                        target_kv_payload=KV, draft_kv_payload="base_bf16"))
        print(f"[smoke] 输入 {ids.shape[1]} token; 开始分块 prefill (chunk={CHUNK}) ...", flush=True)

        # ---- 分块 prefill（显式标注阶段）+ 逐模型阶段状态/耗时/有限性 ----
        acc: dict = {}
        inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
        stage: dict = {"target_prefill_done": False, "draft_prefill_done": False}
        fin = {"target": None, "draft": None}

        def prefill(model, ids_):
            key = "target" if model is tgt else "draft"
            torch.cuda.synchronize()              # 计时边界同步：不让上一阶段残留引用污染峰值
            ts = time.perf_counter()
            with M.attention_phase("prefill"):
                lg, cache = inner(model, ids_)
            torch.cuda.synchronize()
            stage[f"{key}_prefill_s"] = time.perf_counter() - ts
            if lg is not None:
                fin[key] = bool(torch.isfinite(lg).all())
            mem(f"after_{key}_prefill", rec)
            stage[f"{key}_prefill_done"] = True
            rec["stage"] = f"{key}_prefill_done"
            _write(rec, run_id)                   # 逐阶段落盘
            return lg, cache                      # 必须原样返回：解码要用 lg 取 saved_next

        # ---- 阶段标注：**四个入口都要标**（只标 prefill 会让 propose/verify/repair 落进 'other'，
        #      那样"融合没有进入 verify"就只能靠 fused_by_phase 间接证明，无法给出分阶段 eager 计数）----
        orig_fns = dict(prefill=spec_dec._prefill, propose=spec_dec._propose,
                        verify=spec_dec._verify, step=spec_dec._step)

        def propose(*a, **k):
            with M.attention_phase("propose"):
                return orig_fns["propose"](*a, **k)

        def verify(*a, **k):
            with M.attention_phase("verify"):
                return orig_fns["verify"](*a, **k)

        def step(*a, **k):
            with M.attention_phase("repair"):
                return orig_fns["step"](*a, **k)

        spec_dec._prefill = prefill             # harness 侧标注阶段（不改 spec_dec.py）
        spec_dec._propose, spec_dec._verify, spec_dec._step = propose, verify, step
        rounds: list[dict] = []
        try:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=tok.eos_token_id,
                                              target_kv=KV, draft_kv="base_bf16",
                                              log=lambda d: rounds.append(dict(d)))
            torch.cuda.synchronize()
            wall = time.perf_counter() - t1
        finally:
            spec_dec._prefill = orig_fns["prefill"]
            spec_dec._propose, spec_dec._verify, spec_dec._step = (
                orig_fns["propose"], orig_fns["verify"], orig_fns["step"])

        mem("after_decode", rec)
        rec["stage_decode_done"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["stage_flags"] = dict(stage)
        rec["rounds"] = rounds
        # ---- 冻结的逐轮不变量（与 spec_dec 的契约一致）----
        inv = []
        for r in rounds:
            pre, kp, m = r["pre"], r["Kp"], r["m"]
            exp_tgt = pre + (m + 1 if m < kp else kp)
            inv.append(dict(round=r["round"], pre=pre, Kp=kp, m=m,
                            tgt_len=r["tgt_len"], drf_len=r["drf_len"],
                            tgt_len_ok=(r["tgt_len"] == exp_tgt),
                            drf_len_ok=(r["drf_len"] == exp_tgt),
                            lengths_equal=(r["tgt_len"] == r["drf_len"])))
        rec["round_invariants"] = dict(per_round=inv,
                                       all_ok=bool(inv) and all(x["tgt_len_ok"] and x["drf_len_ok"]
                                                                and x["lengths_equal"] for x in inv),
                                       rule="tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)")
        gen = [int(x) for x in out["gen"]]
        vocab = int(tgt.embed_tokens.weight.shape[0])
        attn_ok = None
        try:
            M.assert_strategy_used()
            attn_ok = "ok"
        except RuntimeError as e:
            attn_ok = f"RuntimeError: {e}"
        a_stats = M.attention_stats()
        # **确认新路径确实启用、且没有错误扩展到验证阶段**
        fused_by_phase = a_stats.get("fused_by_phase", {})
        eager_by_phase = a_stats.get("eager_by_phase", {})
        dispatch_ok = (sorted(k for k, v in fused_by_phase.items() if v > 0) == ["prefill"]
                       and all(fused_by_phase.get(p, 0) == 0
                               for p in ("propose", "verify", "repair", "decode"))
                       and eager_by_phase.get("verify", 0) > 0)
        rec["run"] = dict(gen=gen, gen_len=len(gen),
                          gen_ids_in_vocab=bool(all(0 <= g < vocab for g in gen)),
                          logits_finite=fin, prefill_s=acc.get("t"),
                          prefill_target_s=acc.get("target"), prefill_draft_s=acc.get("draft"),
                          stage_prefill_s=rec.get("stage_flags"),
                          wall_s=wall, specdec_stats=out["stats"],
                          attention_ok=attn_ok, attention_stats=a_stats,
                          dispatch_only_prefill=dispatch_ok,
                          s_max_by_phase=a_stats.get("s_max_by_phase"),
                          eager_reason_kinds=a_stats.get("eager_reason_kinds"),
                          note="计时含插桩，仅诊断；禁止进入任何 speedup 表")
        print(f"[smoke] 完成：gen={len(gen)} logits_finite={fin} "
              f"prefill={acc.get('t', 0):.2f}s wall={wall:.2f}s")
        print(f"[smoke] 阶段分流：fused_by_phase={fused_by_phase} "
              f"eager_by_phase={eager_by_phase} 仅prefill融合={dispatch_ok}")
        print(f"[smoke] 逐轮不变量 all_ok={rec['round_invariants']['all_ok']} "
              f"rounds={len(rounds)}")
        rec["stage"] = "complete"
        rec["verdict"] = dict(
            ran=True, weights_bitmatch=rec["weights_bitmatch"],
            logits_all_finite=all(v for v in fin.values() if v is not None),
            gen_in_vocab=rec["run"]["gen_ids_in_vocab"],
            attention_assert=attn_ok,
            dispatch_only_prefill=dispatch_ok,
            round_invariants_ok=rec["round_invariants"]["all_ok"],
            pt_peak_reserved_mib=rec["mem_timeline"][-1]["pt_peak_reserved_mib"],
            pt_peak_allocated_mib=rec["mem_timeline"][-1]["pt_peak_allocated_mib"],
            physical_limit_mib=rec["mem_timeline"][-1]["cuda_total_mib"],
            under_physical_line=bool(rec["mem_timeline"][-1]["pt_peak_reserved_mib"]
                                     <= rec["mem_timeline"][-1]["cuda_total_mib"]),
            claim="**首次跑通只代表『能跑』**：单一 run、无重复 ⇒ 不构成正确性/稳定性/性能结论；"
                  "allocated ≠ 整卡物理占用，reserved 超 8 GiB 不单独证明分页。")
        _write(rec, run_id)
        del tgt, drf
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as e:
        rec["stage"] = "oom"
        rec["oom"] = str(e)
        mem("on_oom", rec)
        _write(rec, run_id)
        print(f"[smoke] !! OOM，已落盘并停止（不升级测试）：{e}")
        return 2
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        mem("on_crash", rec)
        _write(rec, run_id)
        print(f"[smoke] !! 崩溃（已落盘）：\n{rec['traceback']}")
        return 2
    finally:
        M.set_attention_strategy("eager")
        M.reset_attention_stats()

    _write(rec, run_id)
    print(f"[smoke] -> {_out_path(run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
