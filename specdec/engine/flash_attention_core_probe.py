"""Correctness and memory gate for the from-scratch Triton FlashAttention.

Run this before the whole-model trial.  It validates full causal prefill,
bottom-right cached/chunked prefill, and single-token decode against the custom
engine's eager attention semantics.  It then measures full-prefill attention-
core latency and temporary allocation at 2K/4K/8K.
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time
from typing import Callable

import torch

from .flash_attention_triton import KERNEL_REVISION, flash_attention_bf16
from .model_self import kv_repeat


BATCH = 1
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
CONTEXTS = (2048, 4096, 8192)
REPETITIONS = 5
DTYPE = torch.bfloat16
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_core_probe.json",
)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key_repeated = kv_repeat(key, repeats)
    value_repeated = kv_repeat(value, repeats)
    scores = torch.matmul(query, key_repeated.transpose(-2, -1)) / math.sqrt(
        query.shape[-1]
    )
    if causal:
        query_length = query.shape[2]
        kv_length = key.shape[2]
        query_start = kv_length - query_length
        query_positions = query_start + torch.arange(
            query_length, device=query.device,
        )
        key_positions = torch.arange(kv_length, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probability, value_repeated)


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference32 = reference.to(torch.float32)
    candidate32 = candidate.to(torch.float32)
    difference = candidate32 - reference32
    denominator = torch.linalg.vector_norm(reference32).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference) / denominator
        ).item(),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def tensors(query_length: int, kv_length: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        BATCH, QUERY_HEADS, query_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    ).contiguous()
    key = torch.randn(
        BATCH, KV_HEADS, kv_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    ).contiguous()
    value = torch.randn(
        BATCH, KV_HEADS, kv_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    ).contiguous()
    return query, key, value


def measure(function: Callable[[], torch.Tensor]) -> dict:
    try:
        for _ in range(2):
            output = function()
            del output
        torch.cuda.synchronize()

        latency = []
        for _ in range(REPETITIONS):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = function()
            torch.cuda.synchronize()
            latency.append((time.perf_counter() - started) * 1000.0)
            del output

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        floor = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = function()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del output
        return {
            "status": "ok",
            "median_latency_ms": statistics.median(latency),
            "latency_ms": latency,
            "peak_extra_mib": (peak - floor) / 2**20,
            "peak_total_mib": peak / 2**20,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        print(f"unsupported CUDA capability: {capability}")
        return 2

    print(
        "from-scratch BF16 GQA FlashAttention core probe: "
        f"gpu={torch.cuda.get_device_name(0)} capability={capability} "
        f"torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )
    print(
        "validates causal full prefill, cached chunk prefill, and decode; "
        "then measures full-prefill attention-core memory\n",
        flush=True,
    )

    correctness_cases = (
        ("full_prefill_127", 127, 127),
        ("full_prefill_512", 512, 512),
        ("cached_chunk_128_of_512", 128, 512),
        ("cached_chunk_256_of_2048", 256, 2048),
        ("decode_1_of_8192", 1, 8192),
    )
    correctness = []
    correctness_pass = True
    print("=== CORRECTNESS ===", flush=True)
    for case_index, (name, query_length, kv_length) in enumerate(correctness_cases):
        query, key, value = tensors(query_length, kv_length, 1000 + case_index)
        reference = eager_attention(query, key, value)
        candidate = flash_attention_bf16(query, key, value)
        torch.cuda.synchronize()
        metrics = error_metrics(reference, candidate)
        passed = (
            metrics["finite"]
            and metrics["max_abs"] <= 0.05
            and metrics["relative_l2"] <= 0.02
        )
        correctness_pass &= passed
        record = {
            "name": name,
            "query_length": query_length,
            "kv_length": kv_length,
            "metrics": metrics,
            "pass": passed,
        }
        correctness.append(record)
        print(
            f"{name:28s} max_abs={metrics['max_abs']:.6f} "
            f"rel_l2={metrics['relative_l2']:.6f} "
            f"finite={metrics['finite']} pass={passed}",
            flush=True,
        )
        del query, key, value, reference, candidate
        gc.collect()
        torch.cuda.empty_cache()

    benchmarks = []
    print("\n=== FULL-PREFILL ATTENTION CORE ===", flush=True)
    if correctness_pass:
        for context in CONTEXTS:
            query, key, value = tensors(context, context, 2000 + context)
            eager = measure(lambda: eager_attention(query, key, value))
            custom = measure(lambda: flash_attention_bf16(query, key, value))
            record = {"context": context, "eager": eager, "custom": custom}
            if eager["status"] == "ok" and custom["status"] == "ok":
                record["peak_allocated_reduction_pct"] = (
                    1.0 - custom["peak_extra_mib"] / eager["peak_extra_mib"]
                ) * 100.0
                record["speedup"] = (
                    eager["median_latency_ms"] / custom["median_latency_ms"]
                )
                print(
                    f"ctx={context:5d} eager={eager['median_latency_ms']:.2f}ms/"
                    f"{eager['peak_extra_mib']:.1f}MiB "
                    f"custom={custom['median_latency_ms']:.2f}ms/"
                    f"{custom['peak_extra_mib']:.1f}MiB "
                    f"memory_reduction={record['peak_allocated_reduction_pct']:+.1f}% "
                    f"speedup={record['speedup']:.2f}x",
                    flush=True,
                )
            else:
                print(
                    f"ctx={context:5d} eager={eager['status']} "
                    f"custom={custom['status']}",
                    flush=True,
                )
            benchmarks.append(record)
            del query, key, value
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print("SKIPPED: correctness gate failed", flush=True)

    custom_operational = all(
        row["custom"]["status"] == "ok" for row in benchmarks
    ) if benchmarks else False
    decision = (
        "PROCEED_TO_WHOLE_MODEL_TRIAL"
        if correctness_pass and custom_operational
        else "STOP_FIX_KERNEL"
    )
    result = {
        "status": "isolated_custom_flash_attention_core_probe",
        "kernel_revision": KERNEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "capability": capability,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": {
            "batch": BATCH,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "correctness_gates": {"max_abs": 0.05, "relative_l2": 0.02},
        "correctness": correctness,
        "benchmarks": benchmarks,
        "decision": decision,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"\ndecision={decision}\nJSON={OUTPUT}")
    return 0 if decision == "PROCEED_TO_WHOLE_MODEL_TRIAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
