from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from . import paged_cache
from .model_self import quantize_kv_fp8, dequantize_kv_fp8
from specdec.kernels import ops


SHAPES = {
    "target": dict(H=12, KV=2, D=128),
    "draft": dict(H=14, KV=2, D=64),
}


def _timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(vals)


def _make_layer(fmt: str, KV: int, T: int, D: int):
    k = torch.randn(1, KV, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    layer = paged_cache.create_layer_cache(fmt, 16)
    if fmt == "fp16":
        paged_cache.append_current(layer, k, v)
        return layer, k, v
    qv, scale = quantize_kv_fp8(v, 128)
    paged_cache.append_current(layer, k, qv, scale)
    return layer, k, dequantize_kv_fp8(qv, scale, torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,2048,4096,8192")
    ap.add_argument("--qlens", default="1,2,3,4")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default="output/phase13_flash")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    qlens = [int(x) for x in args.qlens.split(",") if x]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    configs = [("target", "fp16"), ("target", "v_fp8"), ("draft", "fp16")]

    for shape_name, fmt in configs:
        sh = SHAPES[shape_name]
        H, KV, D = sh["H"], sh["KV"], sh["D"]
        for T in contexts:
            torch.manual_seed(9000 + T + D + (1 if fmt == "v_fp8" else 0))
            layer, k_contig, v_contig = _make_layer(fmt, KV, T, D)

            for S in qlens:
                q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
                ref = ops.paged_attention_block(q, layer, backend="cuda", validate=False)
                flash = ops.phase13_flash_paged_attention(q, layer, validate=False)
                diff = flash.float() - ref.float()
                max_abs = float(diff.abs().max().item())
                rel_l2 = float(
                    torch.linalg.vector_norm(diff).item()
                    / max(torch.linalg.vector_norm(ref.float()).item(), 1e-12)
                )

                functions = {
                    "paged_direct": lambda: ops.paged_attention_block(
                        q, layer, backend="cuda", validate=False
                    ),
                    "flash_paged": lambda: ops.phase13_flash_paged_attention(
                        q, layer, validate=False
                    ),
                    "contig_cuda_upper_bound": lambda: ops.attention(
                        q, k_contig, v_contig, backend="cuda"
                    ),
                }

                try:
                    import triton  # noqa: F401
                    functions["contig_triton_upper_bound"] = (
                        lambda: ops.attention(
                            q, k_contig, v_contig, backend="triton"
                        )
                    )
                except Exception:
                    pass

                for rep in range(args.repeats):
                    names = list(functions)
                    shift = rep % len(names)
                    names = names[shift:] + names[:shift]
                    for mode in names:
                        ms = _timed(functions[mode], args.warmup, args.iters)
                        rows.append({
                            "shape": shape_name, "fmt": fmt,
                            "H": H, "KV": KV, "D": D,
                            "context": T, "q_len": S, "repeat": rep,
                            "mode": mode, "median_ms": ms,
                            "flash_max_abs_vs_direct": max_abs,
                            "flash_rel_l2_vs_direct": rel_l2,
                        })
                        print(
                            f"[phase13] {shape_name}/{fmt} T={T} q={S} "
                            f"rep={rep} {mode}={ms:.4f} ms", flush=True
                        )

    fields = list(rows[0].keys())
    with (outdir / "phase13_flash_microbench_raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    keys = sorted(set(
        (r["shape"], r["fmt"], r["context"], r["q_len"]) for r in rows
    ))
    for shape, fmt, T, S in keys:
        g = [
            r for r in rows
            if r["shape"] == shape and r["fmt"] == fmt
            and r["context"] == T and r["q_len"] == S
        ]
        modes = sorted(set(r["mode"] for r in g))
        med = {
            m: statistics.median(
                r["median_ms"] for r in g if r["mode"] == m
            )
            for m in modes
        }
        item = {
            "shape": shape, "fmt": fmt, "context": T, "q_len": S,
            "paged_direct_ms": med["paged_direct"],
            "flash_paged_ms": med["flash_paged"],
            "flash_speedup_vs_direct_x":
                med["paged_direct"] / med["flash_paged"],
            "flash_max_abs_vs_direct":
                max(r["flash_max_abs_vs_direct"] for r in g),
            "flash_rel_l2_vs_direct":
                max(r["flash_rel_l2_vs_direct"] for r in g),
        }
        if "contig_cuda_upper_bound" in med:
            item["contig_cuda_ms"] = med["contig_cuda_upper_bound"]
        if "contig_triton_upper_bound" in med:
            item["contig_triton_ms"] = med["contig_triton_upper_bound"]
        summary.append(item)

    sf = sorted({k for r in summary for k in r})
    with (outdir / "phase13_flash_microbench_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=sf)
        w.writeheader()
        w.writerows(summary)

    focus = [
        r for r in summary
        if r["shape"] == "target" and r["fmt"] == "fp16"
        and r["q_len"] == 4 and r["context"] in (2048, 4096, 8192)
    ]
    decision = {
        "criterion":
            "BF16 target q=4 Flash-paged >=1.10x faster than current "
            "paged-direct on at least 2/3 of 2K,4K,8K",
        "focus": focus,
        "wins": sum(r["flash_speedup_vs_direct_x"] >= 1.10 for r in focus),
    }
    decision["microbench_win"] = decision["wins"] >= 2
    (outdir / "phase13_flash_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("\n=== Phase13 focus ===")
    for r in focus:
        print(
            f"T={r['context']} q=4 direct={r['paged_direct_ms']:.4f} ms "
            f"flash={r['flash_paged_ms']:.4f} ms "
            f"speedup={r['flash_speedup_vs_direct_x']:.3f}x"
        )
    gate_status = "PASS" if decision["microbench_win"] else "FAIL"
    print(
        f"\nPHASE13_FLASH_MICROBENCH: {gate_status}",
        f"(wins={decision['wins']}/3)"
    )
    return 0 if decision["microbench_win"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
