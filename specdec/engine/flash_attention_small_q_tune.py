"""Bounded RTX-4060 tuning for G4 two-pass BF16 FlashAttention.

Every candidate must pass the eager numerical gate before it is timed.  The
script writes recommendations only; it never rewrites the production dispatch.
For each requested query length it first sweeps every BLOCK_M/BLOCK_N pair at
the validated 4-warp/2-stage setting, then refines warps/stages for the two
fastest passing tile pairs.  This covers decode q=1, G4 verification q=4,
nearby small-query shapes, and prefill without an unbounded Cartesian sweep.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from .flash_attention_g4_core_probe import compare, eager_reference
from .flash_attention_two_pass_triton import (
    FlashKernelConfig,
    KERNEL_REVISION,
    flash_attention_bf16_two_pass,
    validated_pr2_config,
)


Q_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
DTYPE = torch.bfloat16
QUERY_LENGTHS = (1, 2, 4, 8, 16, 64, 512)
KV_LENGTHS = {1: 8192, 2: 8192, 4: 8192, 8: 8192,
              16: 8192, 64: 8192, 512: 512}
WARMUP = int(os.environ.get("FLASH_TUNE_WARMUP", "3"))
REPETITIONS = int(os.environ.get("FLASH_TUNE_REPS", "9"))
MAX_ABS_GATE = 0.01
REL_L2_GATE = 0.001
OUTDIR = Path(os.environ.get(
    "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
))


def make_inputs(query_length: int, kv_length: int):
    generator = torch.Generator(device="cuda").manual_seed(
        9137 + query_length * 100_000 + kv_length
    )
    shapes = {
        "query": (1, Q_HEADS, query_length, HEAD_DIM),
        "key": (1, KV_HEADS, kv_length, HEAD_DIM),
    }
    query = torch.randn(
        shapes["query"], generator=generator, device="cuda", dtype=DTYPE,
    )
    key = torch.randn(
        shapes["key"], generator=generator, device="cuda", dtype=DTYPE,
    )
    value = torch.randn(
        shapes["key"], generator=generator, device="cuda", dtype=DTYPE,
    )
    return query, key, value


def latency_ms(function):
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPETITIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def evaluate(query, key, value, reference, config):
    try:
        candidate = flash_attention_bf16_two_pass(
            query, key, value, causal=True, config=config,
        )
        error = compare(reference, candidate)
        passed = bool(
            error["finite"]
            and error["max_abs"] <= MAX_ABS_GATE
            and error["relative_l2"] <= REL_L2_GATE
        )
        timing = latency_ms(lambda: flash_attention_bf16_two_pass(
            query, key, value, causal=True, config=config,
        )) if passed else None
        return {
            "config": asdict(config),
            "correctness": error,
            "passed": passed,
            "timing": timing,
            "error": None,
        }
    except Exception as error:
        torch.cuda.empty_cache()
        return {
            "config": asdict(config),
            "correctness": None,
            "passed": False,
            "timing": None,
            "error": f"{type(error).__name__}: {error}",
        }


def key(config):
    return (
        config.block_m, config.block_n,
        config.num_warps, config.num_stages,
    )


def tune_shape(query_length: int):
    kv_length = KV_LENGTHS[query_length]
    query, key_tensor, value = make_inputs(query_length, kv_length)
    reference = eager_reference(query, key_tensor, value, causal=True)
    coarse = [
        FlashKernelConfig(block_m, block_n, 4, 2)
        for block_m in (8, 16, 32, 64)
        for block_n in (32, 64, 128)
    ]
    records = [
        evaluate(query, key_tensor, value, reference, config)
        for config in coarse
    ]
    passing = sorted(
        (row for row in records if row["passed"]),
        key=lambda row: row["timing"]["median_ms"],
    )
    seen = {key(config) for config in coarse}
    refinement = []
    for row in passing[:2]:
        block_m = row["config"]["block_m"]
        block_n = row["config"]["block_n"]
        for num_warps in (2, 4, 8):
            for num_stages in (1, 2, 3):
                config = FlashKernelConfig(
                    block_m, block_n, num_warps, num_stages,
                )
                if key(config) not in seen:
                    seen.add(key(config))
                    refinement.append(config)
    records.extend(
        evaluate(query, key_tensor, value, reference, config)
        for config in refinement
    )
    passing = sorted(
        (row for row in records if row["passed"]),
        key=lambda row: row["timing"]["median_ms"],
    )
    baseline = evaluate(
        query, key_tensor, value, reference,
        validated_pr2_config(query_length),
    )
    winner = passing[0] if passing else None
    return {
        "query_length": query_length,
        "kv_length": kv_length,
        "baseline": baseline,
        "records": records,
        "winner": winner,
        "winner_speedup_vs_pr2": (
            baseline["timing"]["median_ms"] / winner["timing"]["median_ms"]
            if winner is not None and baseline["passed"] else None
        ),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if REPETITIONS < 3:
        raise ValueError("FLASH_TUNE_REPS must be at least 3")
    torch.manual_seed(0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        "Flash G4 small-query tuning: "
        f"gpu={torch.cuda.get_device_name(0)} revision={KERNEL_REVISION}",
        flush=True,
    )
    for query_length in QUERY_LENGTHS:
        row = tune_shape(query_length)
        rows.append(row)
        winner = row["winner"]
        if winner is None:
            print(f"q={query_length:3d}: no passing candidate", flush=True)
            continue
        print(
            f"q={query_length:3d} kv={row['kv_length']:4d} "
            f"winner={winner['config']} "
            f"median={winner['timing']['median_ms']:.4f}ms "
            f"vs_pr2={row['winner_speedup_vs_pr2']:.3f}x",
            flush=True,
        )
    all_pass = all(row["winner"] is not None for row in rows)
    config = {
        "kernel_revision": KERNEL_REVISION,
        "query_lengths": QUERY_LENGTHS,
        "kv_lengths": KV_LENGTHS,
        "warmup": WARMUP,
        "repetitions": REPETITIONS,
        "gates": {"max_abs": MAX_ABS_GATE, "relative_l2": REL_L2_GATE},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "triton": __import__("triton").__version__,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    result = {
        "status": "recommendation_only_production_dispatch_unchanged",
        "config_hash": config_hash,
        "config": config,
        "rows": rows,
        "decision": (
            "PASS_TUNING_CANDIDATES_AVAILABLE"
            if all_pass else "CHECK_TUNING_FAILURE"
        ),
    }
    output = OUTDIR / f"flash_attention_small_q_tune_{config_hash}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"decision={result['decision']}\nJSON={output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
