"""Pure official FlashAttention-2 operator microbenchmark."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from . import paged_cache
from specdec.kernels import ops


SHAPES = {
    "target": dict(H=12, KV=2, D=128),
    "draft": dict(H=14, KV=2, D=64),
}


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs)


def main():
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,2048,4096,8192")
    ap.add_argument("--qlens", default="1,2,3,4")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default="output/pure_flash2")
    a = ap.parse_args()

    contexts = [int(x) for x in a.contexts.split(",") if x]
    qlens = [int(x) for x in a.qlens.split(",") if x]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1302)
    rows = []

    for shape, sh in SHAPES.items():
        H, KV, D = sh["H"], sh["KV"], sh["D"]
        for T in contexts:
            for S in qlens:
                old = T - S
                if old < 0:
                    continue

                q = torch.randn(
                    1, H, S, D,
                    device="cuda", dtype=torch.bfloat16)
                k_old = torch.randn(
                    1, KV, old, D,
                    device="cuda", dtype=torch.bfloat16)
                v_old = torch.randn_like(k_old)
                k_new = torch.randn(
                    1, KV, S, D,
                    device="cuda", dtype=torch.bfloat16)
                v_new = torch.randn_like(k_new)
                k_all = torch.cat([k_old, k_new], dim=2)
                v_all = torch.cat([v_old, v_new], dim=2)

                qf = q.transpose(1, 2).contiguous()
                kf = k_all.transpose(1, 2).contiguous()
                vf = v_all.transpose(1, 2).contiguous()
                knf = k_new.transpose(1, 2).contiguous()
                vnf = v_new.transpose(1, 2).contiguous()

                # Pure preallocated contiguous FA2 cache.
                kc = torch.empty(
                    1, T + 32, KV, D,
                    device="cuda", dtype=torch.bfloat16)
                vc = torch.empty_like(kc)
                if old:
                    kc[:, :old].copy_(
                        k_old.transpose(1, 2).contiguous())
                    vc[:, :old].copy_(
                        v_old.transpose(1, 2).contiguous())
                lens = torch.tensor(
                    [old], device="cuda", dtype=torch.int32)

                # Current paged-direct comparison.
                layer = paged_cache.create_layer_cache("fp16", 16)
                paged_cache.append_current(layer, k_all, v_all)

                fns = {
                    "repo_contig_cuda": lambda: ops.attention(
                        q, k_all, v_all, backend="cuda"),
                    "repo_contig_triton": lambda: ops.attention(
                        q, k_all, v_all, backend="triton"),
                    "official_fa2_full": lambda: flash_attn_func(
                        qf, kf, vf,
                        dropout_p=0.0,
                        softmax_scale=D ** -0.5,
                        causal=True),
                    "official_fa2_kvcache": lambda: flash_attn_with_kvcache(
                        qf, kc, vc,
                        k=knf, v=vnf,
                        cache_seqlens=lens,
                        softmax_scale=D ** -0.5,
                        causal=True),
                    "current_paged_direct": lambda: ops.paged_attention_block(
                        q, layer, backend="cuda", validate=False),
                }

                for rep in range(a.repeats):
                    # Rotate order to reduce systematic thermal/order bias.
                    names = list(fns)
                    shift = rep % len(names)
                    names = names[shift:] + names[:shift]
                    for mode in names:
                        ms = timed(fns[mode], a.warmup, a.iters)
                        rows.append(dict(
                            shape=shape, H=H, KV=KV, D=D,
                            context=T, q_len=S, repeat=rep,
                            mode=mode, median_ms=ms))
                        print(
                            f"[pure-fa2] {shape} T={T} q={S} "
                            f"rep={rep} {mode}={ms:.4f} ms",
                            flush=True)

    fields = list(rows[0].keys())
    raw = out/"microbench_raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    groups = sorted(set(
        (r["shape"], r["context"], r["q_len"])
        for r in rows))
    for shape, T, S in groups:
        g = [
            r for r in rows
            if r["shape"] == shape
            and r["context"] == T
            and r["q_len"] == S
        ]
        meds = {}
        for mode in sorted(set(r["mode"] for r in g)):
            meds[mode] = statistics.median(
                r["median_ms"] for r in g if r["mode"] == mode)
        summary.append(dict(
            shape=shape,
            context=T,
            q_len=S,
            **{f"{k}_ms": v for k, v in meds.items()},
            fa2_kvcache_speedup_vs_paged=(
                meds["current_paged_direct"]
                / meds["official_fa2_kvcache"]),
            fa2_kvcache_speedup_vs_repo_contig_cuda=(
                meds["repo_contig_cuda"]
                / meds["official_fa2_kvcache"]),
        ))

    sf = list(summary[0].keys())
    with (out/"microbench_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=sf)
        w.writeheader()
        w.writerows(summary)

    focus = [
        r for r in summary
        if r["shape"] == "target"
        and r["q_len"] == 4
        and r["context"] in (2048, 4096, 8192)
    ]
    (out/"microbench_focus.json").write_text(
        json.dumps(focus, indent=2),
        encoding="utf-8")

    print("\n=== Pure FA2 target q=4 focus ===")
    for r in focus:
        print(
            f"T={r['context']} "
            f"paged={r['current_paged_direct_ms']:.4f} ms "
            f"FA2-kvcache={r['official_fa2_kvcache_ms']:.4f} ms "
            f"speedup={r['fa2_kvcache_speedup_vs_paged']:.3f}x")
    print("PURE_FLASH2_MICROBENCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
