"""PagedAttention v1d repeated decode benchmark.

This benchmark compares identical CUDA attention math under two paged_v0 paths:
- materialized_cuda: paged K/V -> contiguous materialization -> CUDA attention
- direct_cuda:       paged K/V -> direct page-aware CUDA attention

v1d adds:
- repeated measurements with alternating mode order;
- median/p10/p90 summaries;
- page-size sweeps;
- pointer-table build/hit/invalidation statistics;
- raw and summary CSV output.

Only BF16/base q_len=1 decode is benchmarked. Prefix construction is not timed.
Do not compare these numbers directly with historical runs from a different
Torch/CUDA environment.
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
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .model_self import (
    attention_phase,
    paged_attention_stats,
    reset_paged_attention_stats,
)

MODEL_MAP = {
    "small": MODEL_SMALL,
    "target": MODEL_TARGET,
}


def _parse_int_csv(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        n = int(item)
        if n <= 0:
            raise argparse.ArgumentTypeError("all values must be positive")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("at least one value is required")
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _build_prefix_cache(model, context: int, *, chunk: int, token_id: int):
    cache = None
    done = 0
    with torch.inference_mode(), attention_phase("prefill"):
        while done < context:
            n = min(chunk, context - done)
            ids = torch.full(
                (1, n), token_id, dtype=torch.long, device="cuda")
            _, cache = model._run(
                model.embed_tokens(ids), cache, need_logits=False)
            done += n
    return cache


def _decode_steps(model, cache, *, steps: int, token_id: int):
    with torch.inference_mode(), attention_phase("decode"):
        for _ in range(steps):
            ids = torch.tensor([[token_id]], dtype=torch.long, device="cuda")
            _, cache = model._run(
                model.embed_tokens(ids), cache, need_logits=False)
    return cache


def _run_one(
    model,
    *,
    mode: str,
    context: int,
    page_size: int,
    chunk: int,
    warmup: int,
    steps: int,
    token_id: int,
    repeat: int,
):
    model.set_kv_format("fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(mode)

    reset_paged_attention_stats()
    paged_cache.reset_pointer_table_stats()

    cache = _build_prefix_cache(
        model, context, chunk=chunk, token_id=token_id)

    if warmup:
        cache = _decode_steps(
            model, cache, steps=warmup, token_id=token_id)

    # Measure steady state.  Keep already-built pointer tables from warmup but
    # reset counters so timed-build frequency is visible separately.
    reset_paged_attention_stats()
    paged_cache.reset_pointer_table_stats()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()

    start = time.perf_counter()
    cache = _decode_steps(
        model, cache, steps=steps, token_id=token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated()
    end_alloc = torch.cuda.memory_allocated()
    route = paged_attention_stats()
    ptr = paged_cache.pointer_table_stats()

    result = {
        "mode": mode,
        "context": context,
        "page_size": page_size,
        "repeat": repeat,
        "warmup": warmup,
        "steps": steps,
        "ms_per_token": elapsed * 1000.0 / steps,
        "tokens_per_sec": steps / elapsed,
        "baseline_alloc_mib": baseline_alloc / 2**20,
        "end_alloc_mib": end_alloc / 2**20,
        "peak_alloc_mib": peak_alloc / 2**20,
        "peak_delta_mib": max(0, peak_alloc - baseline_alloc) / 2**20,
        "direct_cuda_calls": route["direct_cuda_calls"],
        "materialized_cuda_calls": route["materialized_cuda_calls"],
        "fallback_materialize_calls": route["fallback_materialize_calls"],
        "direct_decode_calls": route["direct_by_phase"].get("decode", 0),
        "materialized_decode_calls": route["materialized_by_phase"].get("decode", 0),
        "ptr_table_builds": ptr["builds"],
        "ptr_table_hits": ptr["hits"],
        "ptr_table_invalidations": ptr["invalidations"],
    }

    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _print_raw(model_label: str, row: dict) -> None:
    print(
        f"[{model_label}] r={row['repeat']} "
        f"mode={row['mode']:<17} "
        f"ctx={row['context']:<5} page={row['page_size']:<3} "
        f"lat={row['ms_per_token']:.3f} ms/token "
        f"tok/s={row['tokens_per_sec']:.2f} "
        f"peak_delta={row['peak_delta_mib']:.2f} MiB "
        f"ptr(build/hit/inv)={row['ptr_table_builds']}/"
        f"{row['ptr_table_hits']}/{row['ptr_table_invalidations']}"
    )


def _summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["context"], row["page_size"], row["mode"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (context, page_size, mode), group in sorted(grouped.items()):
        lat = [r["ms_per_token"] for r in group]
        tps = [r["tokens_per_sec"] for r in group]
        mem = [r["peak_delta_mib"] for r in group]
        builds = [r["ptr_table_builds"] for r in group]
        hits = [r["ptr_table_hits"] for r in group]
        invalidations = [r["ptr_table_invalidations"] for r in group]
        out.append({
            "context": context,
            "page_size": page_size,
            "mode": mode,
            "repeats": len(group),
            "latency_median_ms": statistics.median(lat),
            "latency_p10_ms": _percentile(lat, 0.10),
            "latency_p90_ms": _percentile(lat, 0.90),
            "tokens_per_sec_median": statistics.median(tps),
            "peak_delta_mib_median": statistics.median(mem),
            "ptr_table_builds_median": statistics.median(builds),
            "ptr_table_hits_median": statistics.median(hits),
            "ptr_table_invalidations_median": statistics.median(invalidations),
        })
    return out


def _print_summary(summary: list[dict]) -> None:
    print("\n=== repeated summary ===")
    for row in summary:
        print(
            f"ctx={row['context']:<5} page={row['page_size']:<3} "
            f"mode={row['mode']:<17} "
            f"lat_med={row['latency_median_ms']:.3f} ms "
            f"p10/p90={row['latency_p10_ms']:.3f}/{row['latency_p90_ms']:.3f} "
            f"peak_med={row['peak_delta_mib_median']:.2f} MiB "
            f"ptr_build_med={row['ptr_table_builds_median']:.1f} "
            f"ptr_hit_med={row['ptr_table_hits_median']:.1f}"
        )

    by_cfg: dict[tuple[int, int], dict[str, dict]] = {}
    for row in summary:
        by_cfg.setdefault((row["context"], row["page_size"]), {})[row["mode"]] = row

    print("\n=== direct_cuda relative to materialized_cuda (median) ===")
    for key in sorted(by_cfg):
        pair = by_cfg[key]
        if "materialized_cuda" not in pair or "direct_cuda" not in pair:
            continue
        base = pair["materialized_cuda"]
        direct = pair["direct_cuda"]
        ratio = base["latency_median_ms"] / direct["latency_median_ms"]
        saved = base["peak_delta_mib_median"] - direct["peak_delta_mib_median"]
        print(
            f"ctx={key[0]:<5} page={key[1]:<3} "
            f"latency_ratio={ratio:.3f}x (>1 direct faster), "
            f"peak_delta_saved={saved:.2f} MiB"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "target"), default="small")
    parser.add_argument(
        "--contexts", type=_parse_int_csv,
        default=_parse_int_csv("512,1024,2048"),
        help="comma-separated prefix lengths",
    )
    parser.add_argument(
        "--page-sizes", type=_parse_int_csv, default=None,
        help="comma-separated page sizes; overrides --page-size",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument(
        "--csv", type=str, default=None,
        help="summary CSV; a sibling *_raw.csv is also written",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.page_size <= 0 or args.chunk <= 0:
        raise SystemExit("page-size and chunk must be positive")
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("steps/repeats must be positive and warmup non-negative")

    page_sizes = args.page_sizes if args.page_sizes is not None else [args.page_size]
    model_name = MODEL_MAP[args.model]

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("model:", model_name)
    print(
        "benchmark:",
        f"contexts={args.contexts}",
        f"page_sizes={page_sizes}",
        f"chunk={args.chunk}",
        f"warmup={args.warmup}",
        f"steps={args.steps}",
        f"repeats={args.repeats}",
    )

    model, tok = load_ours(model_name)
    vocab_size = int(model.embed_tokens.weight.shape[0])
    if not (0 <= args.token_id < vocab_size):
        raise SystemExit(
            f"token-id {args.token_id} is outside vocab size {vocab_size}")

    rows = []
    for page_size in page_sizes:
        for context in args.contexts:
            for repeat in range(args.repeats):
                # Alternate order to reduce systematic thermal/order bias.
                modes = (
                    ("materialized_cuda", "direct_cuda")
                    if repeat % 2 == 0
                    else ("direct_cuda", "materialized_cuda")
                )
                for mode in modes:
                    row = _run_one(
                        model,
                        mode=mode,
                        context=context,
                        page_size=page_size,
                        chunk=args.chunk,
                        warmup=args.warmup,
                        steps=args.steps,
                        token_id=args.token_id,
                        repeat=repeat,
                    )
                    row["model"] = model_name
                    row["torch"] = torch.__version__
                    row["gpu"] = torch.cuda.get_device_name(0)
                    rows.append(row)
                    _print_raw(args.model, row)

    summary = _summaries(rows)
    _print_summary(summary)

    if args.csv:
        summary_path = Path(args.csv)
        raw_path = summary_path.with_name(
            summary_path.stem + "_raw" + summary_path.suffix)
        _write_csv(summary_path, summary)
        _write_csv(raw_path, rows)
        print(f"\nSummary CSV written: {summary_path.resolve()}")
        print(f"Raw CSV written: {raw_path.resolve()}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
