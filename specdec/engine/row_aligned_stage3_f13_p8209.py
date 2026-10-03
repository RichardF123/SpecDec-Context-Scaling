"""row_aligned_stage3_f13_p8209.py — **Stage 3**：固定 block 的 target verify micro-latency（诊断性）。

## 为什么需要它
端到端时间受**token 轨迹**影响（原始 Spec 与对齐候选在 F13 上会走不同轨迹），
**不能**把两者的 wall time 差直接解释成 kernel 开销（本轮明令禁止）。因此这里测**同一件事**：
在**同一份 `H(8208)` 缓存**、**同一批候选 token**、**同一个模型实例**下，对 target block verify
（`spec_dec._verify`，一次 `s=4` 的 `_run`）分别测：
* **OFF**：生产路径（原样）；
* **ON** ：全层 row-aligned 候选（28 QK + 112 Linear 逐行 M=1）。

另外测 **decode 单步**（`spec_dec._step`，s=1）OFF/ON —— 期望两者**逐位一致**且耗时同量级
（s=1 时逐行 == 整块）。

## 纪律
不是正式性能实验、**不作 speedup 声明**；不做 profiler；不改生产默认路径；计时含插桩只作诊断；
原始张量/计时明细先落盘再摘要；旧产物不覆盖。
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import cache_clone as CC
from . import diag_checks as DC
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import row_aligned as RA
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("RA3_TAG", "div8209_ra3micro")
PRE = int(os.environ.get("RA3_PRE", "8208"))
K = int(os.environ.get("RA3_K", "4"))
FEED = [int(x) for x in os.environ.get("RA3_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("RA3_CANDS", "20642,659,576,4531").split(",")]
REPS = int(os.environ.get("RA3_REPS", "20"))
WARMUP = int(os.environ.get("RA3_WARMUP", "5"))
TARGET_ROPE = os.environ.get("RA3_TARGET_ROPE", "cpu_fp32").strip().lower()
CAP_H = os.environ.get("RA3_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
KV, ARM = "base_bf16", os.environ.get("RA3_ARM", "eager")


def _to_dev(snap, dev):
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _time_block(fn, reps: int) -> dict:
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)     # ms
    return dict(n=len(xs), median_ms=float(statistics.median(xs)), min_ms=float(min(xs)),
                max_ms=float(max(xs)), mean_ms=float(statistics.fmean(xs)),
                spread_ms=float(max(xs) - min(xs)), all_ms=[round(x, 3) for x in xs])


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="row_aligned_stage3_f13_p8209", family="F13", kv=KV, arm=ARM,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     question="在同一份 H(8208)、同一批候选下，target block verify 的"
                              "全层 row-aligned 候选相对生产路径的**固定 block micro-latency** 是多少？",
                     semantics="固定输入下 OFF/ON 交替分块计时（含 decode 单步 OFF/ON 与逐位一致性检查）；"
                                "**不是**正式性能实验、不作 speedup 声明、不用 profiler。",
                     config=dict(tag=TAG, pre=PRE, k=K, feed=FEED, cands=CANDS, reps=REPS,
                                 warmup=WARMUP, target_rope=TARGET_ROPE, cap_h=str(CAP_H),
                                 in_scope="固定 block 的 verify micro-latency；不改生产默认路径"))
    print(f"[ra3m] run_id={run_id} pre={PRE} reps={REPS} warmup={WARMUP}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        dev = next(tgt.parameters()).device
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["coverage"] = RA.coverage_check(tgt)
        rec["expected_counts"] = RA.expected_counts(tgt)
        rec["precision"] = dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))
        pf = P5V.PF.qualify()
        rec["pstate"] = pf
        rec["gpu_consumers_before"] = P5V.gpu_consumers()
        print(f"[ra3m] P-state qualified={pf.get('qualified')} sm={pf.get('sm_median')}", flush=True)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        cap = C._load_cap(Path(CAP_H))
        H, tinfo = C._truncate(cap["snapshot_cpu"], PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)),
                               source_len=tinfo["source_len"], pre=PRE,
                               is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H))
        DC.require(tinfo["is_pure_slice"] and C._cache_len(H) == PRE, "H 截断不符", str(tinfo))

        def _verify_once(cand: bool) -> tuple:
            cache = _to_dev(H, dev)
            if cand:
                with RA.row_aligned_verify(tgt):
                    with torch.inference_mode(), M.attention_phase("verify"):
                        return spec_dec._verify(tgt, cache, CANDS)
            with torch.inference_mode(), M.attention_phase("verify"):
                return spec_dec._verify(tgt, cache, CANDS)

        def _step_once(cand: bool) -> tuple:
            cache = _to_dev(H, dev)
            if cand:
                with RA.row_aligned_verify(tgt):
                    with torch.inference_mode(), M.attention_phase("decode"):
                        return spec_dec._step(tgt, FEED[0], cache)
            with torch.inference_mode(), M.attention_phase("decode"):
                return spec_dec._step(tgt, FEED[0], cache)

        # 正确性口径（**重要**）：
        #  * `decode` 单步（s=1）：逐行 == 整块 ⇒ ON 必须与 OFF **逐位一致**（硬门）；
        #  * `block verify`（s=4）：候选**就是要改变**这一次执行几何（把 M=4 搬到逐行 M=1）
        #    ⇒ `ON ≠ OFF` 是**预期行为**，只作**观测**记录（n_diff/max|Δ|），**不**作门。
        #    （首跑把这个写成了硬门 ⇒ 假失败；已修正并记录为缺陷。）
        rows_off, cache_off = _verify_once(False)
        rows_on, cache_on = _verify_once(True)
        d = (rows_off.float() - rows_on.float()).abs()
        verify_delta = dict(exact_equal=bool(torch.equal(rows_off, rows_on)),
                            n_diff=int((rows_off != rows_on).sum()), shape=[int(x) for x in rows_off.shape],
                            max_abs=float(d.max()), mean_abs=float(d.mean()),
                            cache_bitwise_equal=bool(all(torch.equal(cache_off[i][k], cache_on[i][k])
                                                         for i in range(len(cache_off)) for k in ("k", "v"))),
                            reading="候选模式**有意**改变 block verify 的执行几何（M=4 → 逐行 M=1）"
                                    "⇒ 这里不同是预期行为；面板目标是与 **s=1 的 AR 路径**一致，"
                                    "而那由 Stage 2 的 token equality 门判定。")
        step_off, c1 = _step_once(False)
        step_on, c2 = _step_once(True)
        same_step = bool(torch.equal(step_off, step_on) and all(torch.equal(c1[i][k], c2[i][k])
                                                               for i in range(len(c1)) for k in ("k", "v")))
        rec["correctness"] = dict(decode_step_bitwise_equal=same_step,
                                  verify_off_vs_on_observation=verify_delta,
                                  note="硬门只设在 s=1（decode 单步）等价上；verify 的 ON/OFF 差异是候选的"
                                       "**预期效果**，如实记录但不作门")
        print(f"[ra3m] 正确性：decode 单步 ON==OFF（硬门）={same_step}；"
              f"verify ON vs OFF 观测：n_diff={verify_delta['n_diff']} max|Δ|={verify_delta['max_abs']:.6g}"
              f"（**预期行为**）", flush=True)
        DC.require(same_step, "候选 ON/OFF 的 decode 单步不逐位一致（s=1 等价被破坏）",
                   str(rec["correctness"]))

        # 预热
        for _ in range(WARMUP):
            _verify_once(False)
            _verify_once(True)
        # ≥3 个计时块：OFF / ON / OFF（第三个用于检查漂移）
        blocks = dict(verify_off_1=_time_block(lambda: _verify_once(False), REPS),
                      verify_on=_time_block(lambda: _verify_once(True), REPS),
                      verify_off_2=_time_block(lambda: _verify_once(False), REPS),
                      step_off=_time_block(lambda: _step_once(False), REPS),
                      step_on=_time_block(lambda: _step_once(True), REPS))
        off_med = statistics.median([blocks["verify_off_1"]["median_ms"], blocks["verify_off_2"]["median_ms"]])
        rec["timing_ms"] = blocks
        rec["summary"] = dict(
            verify_off_median_ms=float(off_med), verify_on_median_ms=blocks["verify_on"]["median_ms"],
            verify_on_over_off=float(blocks["verify_on"]["median_ms"] / off_med),
            verify_off_block_spread_ms=float(abs(blocks["verify_off_1"]["median_ms"]
                                                 - blocks["verify_off_2"]["median_ms"])),
            step_off_median_ms=blocks["step_off"]["median_ms"], step_on_median_ms=blocks["step_on"]["median_ms"],
            step_on_over_off=float(blocks["step_on"]["median_ms"] / blocks["step_off"]["median_ms"]),
            mem=dict(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20),
            note="同一份 H、同一批候选、同一实例；OFF 取两个分块的中位数以抵消漂移；"
                 "**不作 speedup 声明**，只报告固定 block 的代价比")
        rec["stage"] = "complete"
        tp = OUTDIR / f"{TAG}_{run_id}.pt"
        torch.save(dict(rows_off=rows_off.detach().cpu(), rows_on=rows_on.detach().cpu(),
                        step_off=step_off.detach().cpu(), step_on=step_on.detach().cpu()), str(tp))
        rec["artifacts"] = dict(pt=str(tp))
        p = OUTDIR / f"{TAG}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        rec["artifacts"]["json"] = str(p)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        s = rec["summary"]
        print(f"[ra3m] verify：OFF 中位 {s['verify_off_median_ms']:.2f} ms、"
              f"ON 中位 {s['verify_on_median_ms']:.2f} ms ⇒ 比值 **{s['verify_on_over_off']:.2f}×**"
              f"（OFF 两块漂移 {s['verify_off_block_spread_ms']:.2f} ms）", flush=True)
        print(f"[ra3m] decode 单步：OFF {s['step_off_median_ms']:.2f} ms、ON {s['step_on_median_ms']:.2f} ms "
              f"⇒ {s['step_on_over_off']:.2f}×", flush=True)
        print(f"[ra3m] 峰值：allocated {s['mem']['peak_allocated_mib']:.0f} MiB、"
              f"reserved {s['mem']['peak_reserved_mib']:.0f} MiB", flush=True)
        print(f"[ra3m] RESULT_JSON={p}")
        return 0
    except Exception as e:                                            # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError,)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ra3m] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[ra3m] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            spec_dec.set_verify_wrap(None)
        except Exception:                                             # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()


if __name__ == "__main__":
    raise SystemExit(main())
