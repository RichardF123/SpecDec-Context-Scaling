"""g4_b3_dual_model_feasibility.py — B3: 双模型(1.5B target + 0.5B draft) 8GB 可行性（BF16 CUDA）

目的: 两模型同时常驻是否可行; base_bf16 与 v_fp8(target KV) 在 512/8192 context 下能否
prefill + short decode, 无 OOM, finite, 多次稳定。不做性能宣称。
约束(禁止): 权重量化/CPU offload/轮流加载/减层/换模型/改 8K/chunk 偷偷变。draft 一律 base BF16。
内存记录阶段(每条件): 双模型权重常驻基线 -> target prefill -> target decode -> draft prefill
-> draft decode; 记 torch allocated/reserved(max, 每 rep reset) 与 nvidia-smi used(尽力)。
计时: 每次前后 torch.cuda.synchronize(); model load 时间单列; 时间只作可行性参考。
OOM: try/finally 清理 cache + empty_cache, 模型保持常驻; 8K OOM 时记录并可额外尝试 chunk=1024
(如实标注 extra-diagnostic), 不再偷偷换条件。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.g4_b3_dual_model_feasibility
"""
from __future__ import annotations

import subprocess
import time

import torch

from .align_ours import build_from_config
from .g2_operational import make_text

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.bfloat16
CTXS = (512, 8192)
KV_MODES = ("base_bf16", "v_fp8")   # target KV; draft 恒为 base bf16
CHUNK_8K = 2048
DECODE_SMOKE = 5
REPS = 3


def load_ours(model_name: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True).to("cuda").eval()
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", DTYPE).eval()
    return ours


def smi_used_gib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip()) / 1024.0
    except Exception:
        return None


def chunky_prefill(ours, ids, chunk: int):
    cache_list = None
    last_logits = None
    S = ids.shape[1]
    with torch.inference_mode():
        for st in range(0, S, chunk):
            piece = ids[:, st:st + chunk]
            last = (st + chunk >= S)
            lg, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                       need_logits=last)
            if last:
                last_logits = lg
    torch.cuda.synchronize()
    return last_logits, cache_list


def decode_smoke(ours, last_logits, cache, n, eos):
    toks = []
    cur = last_logits
    with torch.inference_mode():
        for _ in range(n):
            t = int(cur[:, -1].argmax())
            if t == eos:
                break
            toks.append(t)
            cur, cache = ours._run(
                ours.embed_tokens(torch.tensor([[t]], device=ids_dev())), cache)
    torch.cuda.synchronize()
    return toks


def ids_dev():
    return torch.device("cuda")


def cache_finite(cache):
    ok = True
    for lay in cache:
        for key, t in lay.items():
            if key == "fmt":
                continue
            ok &= bool(torch.isfinite(t.to(torch.float32)).all().item())
    return ok


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    free0, total0 = torch.cuda.mem_get_info()
    print(f"start: cuda free={free0/2**30:.2f} GiB / {total0/2**30:.2f} GiB")

    t0 = time.perf_counter()
    print("loading target 1.5B (ours, bf16) ...", flush=True)
    tgt = load_ours(TARGET)
    free_t, _ = torch.cuda.mem_get_info()
    print(f"  target loaded ({(time.perf_counter()-t0):.1f}s), free={free_t/2**30:.2f} GiB")
    t1 = time.perf_counter()
    print("loading draft 0.5B (ours, bf16) ...", flush=True)
    drf = load_ours(DRAFT)
    load_s = time.perf_counter() - t0
    free_d, _ = torch.cuda.mem_get_info()
    print(f"  draft loaded ({load_s:.1f}s total), both-resident free={free_d/2**30:.2f} GiB, "
          f"smi used={smi_used_gib():.2f} GiB")

    verdicts = []
    for ctx in CTXS:
        ids = make_text(tok, ctx).to("cuda")
        for kv in KV_MODES:
            fmt = "v_fp8" if kv == "v_fp8" else "fp16"
            cond = f"ctx={ctx} target-KV={kv} draft=base_bf16"
            print(f"\n=== condition: {cond} ===")
            peaks_alloc, peaks_resv, first_toks, oom = [], [], [], None
            tries = [(CHUNK_8K if ctx > CHUNK_8K else ctx, "primary")]
            if ctx == 8192 and CHUNK_8K != 1024:
                pass  # 仅在 OOM 时追加 extra-diagnostic
            for chunk, tag in tries:
                for rep in range(REPS):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        tgt.set_kv_format(fmt)
                        ta = time.perf_counter()
                        t_lg, t_cache = chunky_prefill(tgt, ids, chunk)
                        tp = time.perf_counter() - ta
                        ttoks = decode_smoke(tgt, t_lg, t_cache, DECODE_SMOKE, eos)
                        d1 = time.perf_counter() - ta - tp
                        tfin = cache_finite(t_cache) and bool(
                            torch.isfinite(t_lg).all().item())
                        del t_lg, t_cache
                        torch.cuda.empty_cache()

                        db = time.perf_counter()
                        d_lg, d_cache = chunky_prefill(drf, ids, chunk)
                        dp = time.perf_counter() - db
                        dtoks = decode_smoke(drf, d_lg, d_cache, DECODE_SMOKE, eos)
                        d2 = time.perf_counter() - db - dp
                        dfin = cache_finite(d_cache) and bool(
                            torch.isfinite(d_lg).all().item())
                        del d_lg, d_cache
                        torch.cuda.empty_cache()
                        peaks_alloc.append(torch.cuda.max_memory_allocated() / 2**30)
                        peaks_resv.append(torch.cuda.max_memory_reserved() / 2**30)
                        first_toks.append((ttoks[0] if ttoks else None,
                                           dtoks[0] if dtoks else None))
                        print(f"  [{tag}] rep{rep+1}: tgt prefill {tp:.2f}s decode {d1:.2f}s | "
                              f"dft prefill {dp:.2f}s decode {d2:.2f}s | alloc {peaks_alloc[-1]:.2f} "
                              f"resv {peaks_resv[-1]:.2f} GiB | finite tgt/dft {tfin}/{dfin} | "
                              f"smoke {tok.decode(ttoks)[:14]!r} / {tok.decode(dtoks)[:14]!r}")
                    except torch.cuda.OutOfMemoryError:
                        oom = (chunk, rep + 1,
                               torch.cuda.max_memory_allocated() / 2**30,
                               torch.cuda.max_memory_reserved() / 2**30)
                        print(f"  [{tag}] rep{rep+1}: OOM at alloc "
                              f"{oom[2]:.2f} resv {oom[3]:.2f} GiB (during {cond})")
                        break
                if oom is not None:
                    if ctx == 8192 and chunk != 1024:
                        tries.append((1024, "extra-diagnostic-chunk1024"))
                        oom = None
                        continue
                    break
            stable = (len(first_toks) >= 2 and
                      all(f == first_toks[0] for f in first_toks))
            smi_now = smi_used_gib()
            if oom is not None:
                verdicts.append((cond, "OOM", oom))
            elif peaks_alloc and all(p < 7.0 for p in peaks_alloc) and stable:
                verdicts.append((cond, "PASS", peaks_alloc))
            elif peaks_alloc:
                verdicts.append((cond, "MIXED(check peaks/stability)", peaks_alloc))
            else:
                verdicts.append((cond, "FAIL/BLOCKER", None))
            print(f"  -> verdict: {verdicts[-1][1]} | alloc-peaks {[f'{p:.2f}' for p in peaks_alloc]} "
                  f"| stable={stable} | smi used now {smi_now if smi_now else float('nan'):.2f} GiB")

    del tgt, drf
    torch.cuda.empty_cache()
    print("\n=== B3 summary ===")
    for cond, v, extra in verdicts:
        print(f"  {cond:34s}: {v}  {extra}")
    print("note: times are feasibility-only, not performance; peaks are torch max_allocated/reserved "
          "with per-rep reset; not comparable to G2 fp16 single-model or G3 paired numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
