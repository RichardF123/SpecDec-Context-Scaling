"""correctness_protocol_margins.py — 决策位置的 **top-1/top-2/margin** 提取 + profiler 可用性检查。

## 为什么需要
协议要求每次运行记录"决策位置 top-1/top-2 与 margin"（用于判断分歧是否发生在**近并列**处）。
harness 的产物只存生成 token IDs 与统计量，不存决策行 ⇒ 这里**按需重建**：
对每个 family，用 **AR 的口径**（分块 prefill + 逐 token `M=1` 单步）走到
**首个分歧位置**（若与 AR 无分歧，则走到末尾位置），记录该位置的 top-1/top-2 与 margin，
以及 AR/Spec 在该位置各自选的 token。

## profiler 可用性
只**检查**本机是否已有 `nsys`/`ncu` 可执行文件（**不安装**）；并在进程内试一次 `torch.profiler`
CUDA activity 是否真的产出 kernel 事件。任一不可用 ⇒ 记 `PROFILER_UNAVAILABLE`，**不重试**、不阻塞。

**纪律**：不加载 draft（AR 口径只需 target）；不改 cache/mask/RoPE/权重；不做 FP32；
profiler 相关时间**不进入**任何性能结论；结果先落盘再摘要。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from . import sb_align as A
from . import spec_dec
from . import spec_cache
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("PM_TAG", "div8209_proto")
FAMILIES = [x for x in os.environ.get("PM_FAMILIES", "F01,F02,F08,F10").split(",") if x]
PROTO_TAG = os.environ.get("PM_RUNS_TAG", "div8209_proto")
CTX = int(os.environ.get("PM_CTX", "8192"))
CHUNK = int(os.environ.get("PM_CHUNK", "2048"))
MAXNEW = int(os.environ.get("PM_MAXNEW", "64"))
TARGET_ROPE = os.environ.get("PM_TARGET_ROPE", "cpu_fp32").strip().lower()
KV = "base_bf16"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _profiler_status() -> dict:
    """只**检查**，不安装：nsys/ncu 是否存在 + torch.profiler 是否真能给出 CUDA kernel 事件。"""
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    got_kernel = False
    detail = None
    try:
        a = torch.randn(1, 12, 1, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, 12, 8209, 128, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                torch.matmul(a, b.transpose(-2, -1))
        torch.cuda.synchronize()
        got_kernel = any(("CUDA" in str(getattr(e, "device_type", "")))
                         for e in prof.events())
        detail = dict(n_events=len(list(prof.events())),
                      n_cuda_events=sum(1 for e in prof.events()
                                        if "CUDA" in str(getattr(e, "device_type", ""))))
    except Exception as e:                                                # noqa: BLE001
        detail = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
    return dict(nsys=nsys, ncu=ncu, torch_profiler_cuda_kernel_events=bool(got_kernel),
                detail=detail,
                status=("PROFILER_AVAILABLE" if (nsys or ncu or got_kernel) else "PROFILER_UNAVAILABLE"))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="correctness_protocol_margins", tag=TAG,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), families=FAMILIES, runs_tag=PROTO_TAG,
                     protocol="post_shape_localization_v1",
                     semantics="按 AR 口径重建决策位置并读 top-1/top-2/margin；检查 profiler 可用性；不加载 draft")
    print(f"[pm] run_id={run_id} families={FAMILIES}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    TRS = None
    try:
        rec["profiler_status"] = _profiler_status()
        print(f"[pm] profiler: {rec['profiler_status']['status']} "
              f"(nsys={rec['profiler_status']['nsys']} ncu={rec['profiler_status']['ncu']} "
              f"torch_cuda_events={rec['profiler_status']['torch_profiler_cuda_kernel_events']})", flush=True)
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy("eager")
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        from . import family_input as FI
        out = {}
        for fam in FAMILIES:
            entry: dict = dict(family=fam)
            try:
                # 正确的签名：`load_family_ids(tok, family_id, *, manifest=None, ctx, verify_hash=True)`
                ids, prov = FI.load_family_ids(tok, fam, ctx=CTX, verify_hash=True)
            except Exception as e:                                        # noqa: BLE001
                ids, prov, entry["provenance_error"] = None, None, f"{type(e).__name__}: {str(e)[:300]}"
            if ids is None:
                out[fam] = dict(entry, error="family_input 输入不可用（校验失败或签名不符）")
                print(f"[pm] {fam}: 输入不可用 ⇒ 跳过（**不换样本**）：{entry['provenance_error']}", flush=True)
                continue
            dev = next(tgt.parameters()).device
            ids = ids.to(dev) if torch.is_tensor(ids) else torch.tensor([ids], device=dev)
            # 取该 family 的 AR/诊断生成结果（读 24-run 产物）
            def _gen(mode: str) -> list:
                import glob as _g
                fs = sorted(_g.glob(str(OUTDIR / f"{PROTO_TAG}_{fam}_{mode}_rep1_*.json")))
                if not fs:
                    return []
                d = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
                return list((d.get("smoke") or {}).get("gen") or [])
            ar_gen, diag_gen, prod_gen = _gen("ar"), _gen("aligned"), _gen("spec")
            cp = 0
            for x, y in zip(ar_gen, diag_gen or ar_gen):
                if x != y:
                    break
                cp += 1
            pos = cp if (diag_gen and cp < len(ar_gen)) else max(0, len(ar_gen) - 1)
            # 按 AR 口径重建：分块 prefill（与 AR 相同）+ 逐 token 单步走到 `pos`
            acc: dict = {}
            inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, cache = inner(tgt, ids)
            nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            for i in range(pos):
                tok_i = int(ar_gen[i])
                with torch.inference_mode(), M.attention_phase("decode"):
                    lg, cache = spec_dec._step(tgt, tok_i, cache)
                nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            row = lg.reshape(-1, lg.shape[-1])[-1].float()
            v, idx = torch.topk(row, 2)
            entry.update(
                provenance=prov, position=int(pos), prompt_len=int(ids.shape[1]),
                ar_top1=int(idx[0]), ar_top2=int(idx[1]),
                ar_top1_value=float(v[0]), ar_top2_value=float(v[1]), margin=float(v[0] - v[1]),
                argmax=int(row.argmax()), argmax_value=float(row.max()),
                tied_at_max=int((row == row.max()).sum()),
                ar_token=(ar_gen[pos] if pos < len(ar_gen) else None),
                diag_token=(diag_gen[pos] if pos < len(diag_gen) else None),
                prod_token=(prod_gen[pos] if pos < len(prod_gen) else None),
                cache_len_after=int(spec_cache.cache_length(cache)),
                note="按 **AR 口径**（分块 prefill + 逐 token M=1）重建决策位置；"
                     "该位置即 AR 与诊断路径的首个分歧（若无分歧则取末位）")
            out[fam] = entry
            print(f"[pm] {fam}: pos={pos} prompt_len={entry['prompt_len']} "
                  f"AR top1={entry['ar_top1']}({entry['ar_top1_value']:.4g}) "
                  f"top2={entry['ar_top2']}({entry['ar_top2_value']:.4g}) margin={entry['margin']:.4g} "
                  f"AR/DIAG token={entry['ar_token']}/{entry['diag_token']}", flush=True)
        rec["families"] = out
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_margins_{run_id}.json"
        rec["artifacts"] = dict(json=str(p))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[pm] RESULT_JSON={p}")
        return 0
    except Exception as e:                                                # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_margins_{run_id}.json"
        rec["artifacts"] = dict(json=str(p))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[pm] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[pm] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            spec_dec.set_verify_wrap(None)
        except Exception:                                                 # noqa: BLE001
            pass
        if TRS is not None:
            try:
                rec["target_rope_scope_restore"] = TRS.assert_restored()
            except Exception:                                             # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
