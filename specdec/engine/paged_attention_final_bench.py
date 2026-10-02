"""Final direct PagedAttention microbenchmark.

Measures:
- materialized: paged cache -> materialize (+ V dequant for V-FP8) -> CUDA attention
- direct: paged cache -> direct page-aware CUDA attention

Prefix/cache construction is outside timing.
This is an attention-path benchmark, not an H2O speed benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch

from . import paged_cache
from .model_self import dequantize_kv_fp8, quantize_kv_fp8, GROUP_V
from specdec.kernels import ops


def _pct(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    z = (len(ys) - 1) * p
    lo = int(z)
    hi = min(lo + 1, len(ys) - 1)
    a = z - lo
    return ys[lo] * (1 - a) + ys[hi] * a


def _build(fmt, context, page):
    layer = paged_cache.create_layer_cache(fmt, page_size=page)
    torch.manual_seed(123 + context + page)
    done = 0
    while done < context:
        n = min(64, context - done)
        k = torch.randn(
            1, 2, n, 128, device="cuda", dtype=torch.bfloat16)
        v16 = torch.randn_like(k)
        if fmt == "fp16":
            paged_cache.append_current(layer, k, v16)
        else:
            qv, scale = quantize_kv_fp8(v16, GROUP_V)
            paged_cache.append_current(layer, k, qv, scale)
        done += n
    return layer


def _materialized(q, layer):
    mat = paged_cache.materialize_layer(layer)
    if layer["fmt"] == "v_fp8":
        v = dequantize_kv_fp8(
            mat["v"], mat["v_scale"], torch.bfloat16)
    else:
        v = mat["v"]
    return ops.attention(q, mat["k"], v, backend="cuda")


def _direct(q, layer):
    return ops.paged_attention_block(
        q, layer, backend="cuda", validate=False)


def _measure(fn, *, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated()
    del out
    return elapsed * 1000 / iters, max(0, peak - base) / 2**20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--qlens", default="1,4")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--csv", default="paged_attention_final_bench.csv")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    qlens = [int(x) for x in args.qlens.split(",") if x]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print("Final PagedAttention direct-path benchmark")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    raw = []
    for fmt in ("fp16", "v_fp8"):
        for context in contexts:
            layer = _build(fmt, context, args.page_size)
            for qlen in qlens:
                torch.manual_seed(9000 + context + qlen)
                q = torch.randn(
                    1, 12, qlen, 128,
                    device="cuda", dtype=torch.bfloat16)

                a = _materialized(q, layer)
                b = _direct(q, layer)
                diff = float((a.float() - b.float()).abs().max())
                allowed = max(
                    2e-2,
                    0.02 * max(1.0, float(a.float().abs().max())),
                )
                assert diff <= allowed, (
                    f"parity failure fmt={fmt} ctx={context} q={qlen}: "
                    f"{diff} > {allowed}"
                )

                for rep in range(args.repeats):
                    order = (
                        ("materialized", "direct")
                        if rep % 2 == 0
                        else ("direct", "materialized")
                    )
                    for mode in order:
                        fn = (
                            (lambda: _materialized(q, layer))
                            if mode == "materialized"
                            else (lambda: _direct(q, layer))
                        )
                        ms, peak = _measure(
                            fn, warmup=args.warmup, iters=args.iters)
                        row = {
                            "fmt": fmt,
                            "context": context,
                            "qlen": qlen,
                            "page_size": args.page_size,
                            "repeat": rep,
                            "mode": mode,
                            "ms_per_call": ms,
                            "peak_delta_mib": peak,
                            "parity_max_abs": diff,
                        }
                        raw.append(row)
                        print(
                            f"fmt={fmt:<6} ctx={context:<5} q={qlen} "
                            f"r={rep} {mode:<12} "
                            f"{ms:.3f} ms peak={peak:.2f} MiB"
                        )

            del layer
            gc.collect()
            torch.cuda.empty_cache()

    groups = {}
    for r in raw:
        key = (r["fmt"], r["context"], r["qlen"], r["mode"])
        groups.setdefault(key, []).append(r)

    summary = []
    for key, rows in sorted(groups.items()):
        ms = [r["ms_per_call"] for r in rows]
        mem = [r["peak_delta_mib"] for r in rows]
        summary.append({
            "fmt": key[0],
            "context": key[1],
            "qlen": key[2],
            "mode": key[3],
            "repeats": len(rows),
            "latency_median_ms": statistics.median(ms),
            "latency_p10_ms": _pct(ms, 0.1),
            "latency_p90_ms": _pct(ms, 0.9),
            "peak_delta_mib_median": statistics.median(mem),
        })

    pairs = {}
    for r in summary:
        pairs.setdefault(
            (r["fmt"], r["context"], r["qlen"]), {}
        )[r["mode"]] = r

    print("\n=== direct relative to materialized ===")
    for key in sorted(pairs):
        p = pairs[key]
        if set(p) != {"direct", "materialized"}:
            continue
        ratio = (
            p["materialized"]["latency_median_ms"]
            / p["direct"]["latency_median_ms"]
        )
        saved = (
            p["materialized"]["peak_delta_mib_median"]
            - p["direct"]["peak_delta_mib_median"]
        )
        print(
            f"fmt={key[0]:<6} ctx={key[1]:<5} q={key[2]} "
            f"latency_ratio={ratio:.3f}x "
            f"peak_saved={saved:.2f} MiB"
        )

    out = Path(args.csv)
    raw_out = out.with_name(out.stem + "_raw" + out.suffix)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    with raw_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    print("\nSummary:", out.resolve())
    print("Raw:", raw_out.resolve())
    print("paged_attention_final_bench: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
