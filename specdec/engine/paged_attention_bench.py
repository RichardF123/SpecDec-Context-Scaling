"""PagedAttention v1c decode benchmark.

This is a performance probe, not a correctness gate.

It compares identical CUDA attention math under two paged_v0 paths:
- materialized_cuda:
    paged K/V -> full contiguous materialization -> existing CUDA attention
- direct_cuda:
    paged K/V -> direct page-aware CUDA attention

Only BF16/base cache and q_len=1 decode are benchmarked.

Examples:
    # quick
    python -m specdec.engine.paged_attention_bench \
      --model small --contexts 128,512,1024 --steps 16

    # fuller
    python -m specdec.engine.paged_attention_bench \
      --model small --contexts 256,512,1024,2048 --steps 32 \
      --page-size 16 --csv paged_attention_bench_small.csv

Notes:
- Reported latency is whole-model `_run(..., need_logits=False)` decode latency,
  so it includes all transformer layers but excludes the final LM-head matmul.
- Prefix construction is not timed.
- `peak_delta_mib` is additional peak allocated CUDA memory above the
  post-warmup baseline during timed decode steps.
- Do not compare these numbers directly with historical runs from a different
  Torch/CUDA environment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch

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


def _build_prefix_cache(model, context: int, *, chunk: int, token_id: int):
    cache = None
    done = 0
    with torch.inference_mode(), attention_phase("prefill"):
        while done < context:
            n = min(chunk, context - done)
            ids = torch.full(
                (1, n),
                token_id,
                dtype=torch.long,
                device="cuda",
            )
            _, cache = model._run(
                model.embed_tokens(ids),
                cache,
                need_logits=False,
            )
            done += n
    return cache


def _decode_steps(model, cache, *, steps: int, token_id: int):
    with torch.inference_mode(), attention_phase("decode"):
        for _ in range(steps):
            ids = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device="cuda",
            )
            _, cache = model._run(
                model.embed_tokens(ids),
                cache,
                need_logits=False,
            )
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
):
    model.set_kv_format("fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(mode)

    reset_paged_attention_stats()

    cache = _build_prefix_cache(
        model,
        context,
        chunk=chunk,
        token_id=token_id,
    )

    if warmup:
        cache = _decode_steps(
            model,
            cache,
            steps=warmup,
            token_id=token_id,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    baseline_alloc = torch.cuda.memory_allocated()

    start = time.perf_counter()
    cache = _decode_steps(
        model,
        cache,
        steps=steps,
        token_id=token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated()
    end_alloc = torch.cuda.memory_allocated()
    route = paged_attention_stats()

    result = {
        "mode": mode,
        "context": context,
        "page_size": page_size,
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
    }

    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _print_row(model_label: str, row: dict) -> None:
    print(
        f"[{model_label}] "
        f"mode={row['mode']:<17} "
        f"ctx={row['context']:<5} "
        f"page={row['page_size']:<3} "
        f"lat={row['ms_per_token']:.3f} ms/token "
        f"tok/s={row['tokens_per_sec']:.2f} "
        f"peak_delta={row['peak_delta_mib']:.2f} MiB "
        f"direct_decode_calls={row['direct_decode_calls']} "
        f"materialized_decode_calls={row['materialized_decode_calls']}"
    )


def _add_comparison(rows: list[dict]) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["context"], row["page_size"]), {})[row["mode"]] = row

    print("\n=== direct_cuda relative to materialized_cuda ===")
    for key in sorted(grouped):
        pair = grouped[key]
        if "materialized_cuda" not in pair or "direct_cuda" not in pair:
            continue
        base = pair["materialized_cuda"]
        direct = pair["direct_cuda"]
        speedup = base["ms_per_token"] / direct["ms_per_token"]
        mem_saved = base["peak_delta_mib"] - direct["peak_delta_mib"]
        print(
            f"ctx={key[0]:<5} page={key[1]:<3} "
            f"latency_ratio={speedup:.3f}x "
            f"(>1 means direct is faster), "
            f"peak_delta_saved={mem_saved:.2f} MiB"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("small", "target"),
        default="small",
    )
    parser.add_argument(
        "--contexts",
        type=_parse_int_csv,
        default=_parse_int_csv("128,512,1024"),
        help="comma-separated prefix lengths",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--token-id", type=int, default=42)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.page_size <= 0 or args.chunk <= 0:
        raise SystemExit("page-size and chunk must be positive")
    if args.steps <= 0 or args.warmup < 0:
        raise SystemExit("steps must be positive and warmup non-negative")

    model_name = MODEL_MAP[args.model]

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("model:", model_name)
    print(
        "benchmark:",
        f"contexts={args.contexts}",
        f"page_size={args.page_size}",
        f"chunk={args.chunk}",
        f"warmup={args.warmup}",
        f"steps={args.steps}",
    )

    model, tok = load_ours(model_name)
    vocab_size = int(model.embed_tokens.weight.shape[0])
    if not (0 <= args.token_id < vocab_size):
        raise SystemExit(
            f"token-id {args.token_id} is outside vocab size {vocab_size}"
        )

    rows = []
    for context in args.contexts:
        for mode in ("materialized_cuda", "direct_cuda"):
            row = _run_one(
                model,
                mode=mode,
                context=context,
                page_size=args.page_size,
                chunk=args.chunk,
                warmup=args.warmup,
                steps=args.steps,
                token_id=args.token_id,
            )
            row["model"] = model_name
            row["torch"] = torch.__version__
            row["gpu"] = torch.cuda.get_device_name(0)
            rows.append(row)
            _print_row(args.model, row)

    _add_comparison(rows)

    if args.csv:
        path = Path(args.csv)
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {path.resolve()}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
