"""Decode-only blockwise/on-the-fly quantized-KV attention probe.

This is deliberately isolated from the inference engine.  It compares the
current eager attention pattern (full V dequantization plus full GQA repeat)
with an online-softmax implementation that reads K and quantized V in token
blocks.  The probe answers one narrow question before any Triton/CUDA work:
does avoiding full-cache materialization remove enough temporary allocation to
justify a fused implementation?

It does not modify ``model_self.py``, does not change the frozen G3 policy, and
does not make an end-to-end model-speed claim.

Run on the target RTX 4060 after the existing KV-policy gates::

    python -m specdec.engine.kv_streaming_attention_probe
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import math
import os
import statistics
import time

import torch

from .model_self import (
    dequantize_kv_fp8,
    dequantize_kv_int4_packed,
    kv_repeat,
    quantize_kv_fp8,
    quantize_kv_int4_packed,
)


BATCH = 1
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
CONTEXTS = (2048, 4096, 8192)
BLOCK_SIZES = (128, 256, 512)
WARMUP = 2
REPETITIONS = 5
DTYPE = torch.bfloat16
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "14_streaming_attention_probe.json",
)


@dataclass(frozen=True)
class ValuePolicy:
    name: str
    fmt: str


POLICIES = (
    ValuePolicy("current_vfp8", "v_fp8"),
    ValuePolicy("optional_vint4_g32", "v_int4"),
)


def encode_value(value: torch.Tensor, fmt: str):
    if fmt == "v_fp8":
        return quantize_kv_fp8(value, group_size=128)
    if fmt == "v_int4":
        return quantize_kv_int4_packed(value, group_size=32)
    raise ValueError(fmt)


def decode_value(payload: torch.Tensor, scale: torch.Tensor, fmt: str):
    if fmt == "v_fp8":
        return dequantize_kv_fp8(payload, scale, DTYPE)
    if fmt == "v_int4":
        return dequantize_kv_int4_packed(payload, scale, DTYPE)
    raise ValueError(fmt)


def eager_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value_payload: torch.Tensor,
    value_scale: torch.Tensor,
    fmt: str,
) -> torch.Tensor:
    """Materialize full V and repeat both K/V to all query heads."""
    value = decode_value(value_payload, value_scale, fmt)
    repeats = QUERY_HEADS // KV_HEADS
    key_repeated = kv_repeat(key, repeats)
    value_repeated = kv_repeat(value, repeats)
    scores = torch.matmul(query, key_repeated.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(probability, value_repeated.to(torch.float32))
    return output.to(DTYPE)


def streaming_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value_payload: torch.Tensor,
    value_scale: torch.Tensor,
    fmt: str,
    block_size: int,
) -> torch.Tensor:
    """Online softmax over token blocks without full GQA K/V expansion."""
    batch, query_heads, query_tokens, head_dim = query.shape
    assert batch == BATCH and query_tokens == 1 and head_dim == HEAD_DIM
    assert query_heads == QUERY_HEADS and key.shape[1] == KV_HEADS
    assert QUERY_HEADS % KV_HEADS == 0

    repeats = QUERY_HEADS // KV_HEADS
    grouped_query = query.reshape(batch, KV_HEADS, repeats, 1, head_dim)
    state_shape = (batch, KV_HEADS, repeats, 1, 1)
    running_max = torch.full(
        state_shape, -float("inf"), device=query.device, dtype=torch.float32,
    )
    running_sum = torch.zeros(state_shape, device=query.device, dtype=torch.float32)
    accumulator = torch.zeros(
        (batch, KV_HEADS, repeats, 1, head_dim),
        device=query.device,
        dtype=torch.float32,
    )

    tokens = key.shape[2]
    for start in range(0, tokens, block_size):
        end = min(start + block_size, tokens)
        key_block = key[:, :, start:end]
        payload_block = value_payload[:, :, start:end]
        scale_block = value_scale[:, :, start:end]
        value_block = decode_value(payload_block, scale_block, fmt)

        scores = torch.matmul(
            grouped_query,
            key_block.unsqueeze(2).transpose(-2, -1),
        ).to(torch.float32) / math.sqrt(HEAD_DIM)
        block_max = scores.amax(dim=-1, keepdim=True)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        block_weight = torch.exp(scores - next_max)
        accumulator = accumulator * old_weight + torch.matmul(
            block_weight,
            value_block.unsqueeze(2).to(torch.float32),
        )
        running_sum = running_sum * old_weight + block_weight.sum(
            dim=-1, keepdim=True,
        )
        running_max = next_max

    output = (accumulator / running_sum).to(DTYPE)
    return output.reshape(batch, query_heads, 1, head_dim)


def benchmark(function):
    for _ in range(WARMUP):
        output = function()
        del output
    torch.cuda.synchronize()

    latency_ms = []
    peak_extra_mib = []
    peak_total_mib = []
    for _ in range(REPETITIONS):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        floor = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = function()
        torch.cuda.synchronize()
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        peak = torch.cuda.max_memory_allocated()
        peak_total_mib.append(peak / 2**20)
        peak_extra_mib.append((peak - floor) / 2**20)
        del output

    return {
        "median_latency_ms": statistics.median(latency_ms),
        "latency_ms": latency_ms,
        "median_peak_extra_mib": statistics.median(peak_extra_mib),
        "peak_extra_mib": peak_extra_mib,
        "median_peak_total_mib": statistics.median(peak_total_mib),
    }


def compare(reference: torch.Tensor, candidate: torch.Tensor):
    difference = candidate.to(torch.float32) - reference.to(torch.float32)
    denominator = torch.linalg.vector_norm(reference.to(torch.float32)).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / denominator).item(),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if QUERY_HEADS % KV_HEADS:
        raise RuntimeError("query heads must be divisible by KV heads")

    torch.manual_seed(0)
    records = []
    print(
        "streaming KV mechanism probe: decode-only, random Qwen-shaped tensors, "
        f"q_heads={QUERY_HEADS} kv_heads={KV_HEADS} d={HEAD_DIM} dtype={DTYPE}",
        flush=True,
    )
    print(
        "This measures attention-core temporary allocation, not whole-model peak VRAM.\n",
        flush=True,
    )

    for context in CONTEXTS:
        generator = torch.Generator(device="cuda").manual_seed(1000 + context)
        query = torch.randn(
            BATCH, QUERY_HEADS, 1, HEAD_DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        )
        key = torch.randn(
            BATCH, KV_HEADS, context, HEAD_DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        )

        for policy_index, policy in enumerate(POLICIES):
            value_generator = torch.Generator(device="cuda").manual_seed(2000 + context)
            value = torch.randn(
                BATCH, KV_HEADS, context, HEAD_DIM,
                generator=value_generator, device="cuda", dtype=DTYPE,
            )
            value_payload, value_scale = encode_value(value, policy.fmt)
            del value
            gc.collect()
            torch.cuda.empty_cache()

            eager = benchmark(
                lambda: eager_decode_attention(
                    query, key, value_payload, value_scale, policy.fmt,
                )
            )
            reference = eager_decode_attention(
                query, key, value_payload, value_scale, policy.fmt,
            )

            print(
                f"ctx={context:5d} policy={policy.name:20s} "
                f"eager={eager['median_latency_ms']:.3f}ms "
                f"extra={eager['median_peak_extra_mib']:.2f}MiB",
                flush=True,
            )
            for block_size in BLOCK_SIZES:
                streamed = benchmark(
                    lambda block_size=block_size: streaming_decode_attention(
                        query, key, value_payload, value_scale, policy.fmt, block_size,
                    )
                )
                candidate = streaming_decode_attention(
                    query, key, value_payload, value_scale, policy.fmt, block_size,
                )
                error = compare(reference, candidate)
                memory_reduction = (
                    1.0
                    - streamed["median_peak_extra_mib"]
                    / max(eager["median_peak_extra_mib"], 1e-12)
                ) * 100.0
                latency_ratio = (
                    streamed["median_latency_ms"] / eager["median_latency_ms"]
                )
                record = {
                    "context": context,
                    "policy": asdict(policy),
                    "block_size": block_size,
                    "eager": eager,
                    "streaming": streamed,
                    "correctness": error,
                    "temporary_memory_reduction_pct": memory_reduction,
                    "latency_ratio_vs_eager": latency_ratio,
                }
                records.append(record)
                print(
                    f"  block={block_size:3d} "
                    f"stream={streamed['median_latency_ms']:.3f}ms "
                    f"extra={streamed['median_peak_extra_mib']:.2f}MiB "
                    f"mem_reduction={memory_reduction:+.1f}% "
                    f"latency={latency_ratio:.2f}x "
                    f"max_abs={error['max_abs']:.6f} "
                    f"rel_l2={error['relative_l2']:.6f}",
                    flush=True,
                )
                del candidate

            del reference, value_payload, value_scale
            if policy_index + 1 < len(POLICIES):
                gc.collect()
                torch.cuda.empty_cache()
        del query, key

    result = {
        "status": "mechanism_probe_only_no_engine_change",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": {
            "batch": BATCH,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "query_tokens": 1,
        },
        "contexts": CONTEXTS,
        "block_sizes": BLOCK_SIZES,
        "warmup": WARMUP,
        "repetitions": REPETITIONS,
        "records": records,
        "interpretation_gate": {
            "correctness": "finite and numerically close to eager quantized-cache output",
            "memory": "substantial 8K temporary-allocation reduction is required",
            "latency": (
                "Python streaming latency is diagnostic; only a tolerable ratio "
                "justifies Triton work"
            ),
            "scope": "no end-to-end or speculative-decoding claim from this probe",
        },
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(f"\nJSON={OUTPUT}")
    print(
        "No automatic integration decision. Review 8K correctness, temporary "
        "memory, and latency together."
    )
    return 0 if all(record["correctness"]["finite"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
