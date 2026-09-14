"""Triton split-K decode-attention probe for the frozen V-FP8 KV policy.

This file is intentionally isolated from ``model_self.py`` and the production
inference path.  It tests the next mechanism suggested by
``kv_streaming_attention_probe``: fuse FP8 value dequantization and native-GQA
head mapping into block attention, then merge small per-block online-softmax
states.  No full V dequantization or 2-to-12 KV-head repeat is materialized.

The implementation is a two-kernel split-K prototype, not a production Flash
Attention replacement.  It measures decode attention only and cannot support
an end-to-end or speculative-decoding claim by itself.

Run on the target RTX 4060::

    python -m specdec.engine.kv_streaming_attention_triton_probe
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - target machine has Triton
    raise RuntimeError("Triton is required for this probe") from error

from .model_self import dequantize_kv_fp8, kv_repeat, quantize_kv_fp8


BATCH = 1
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
CONTEXTS = (2048, 4096, 8192)
BLOCK_SIZES = (64, 128, 256)
WARMUP = 10
REPETITIONS = 30
DTYPE = torch.bfloat16
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "15_streaming_attention_triton_probe.json",
)


@triton.jit
def _vfp8_attention_partials(
    query_ptr,
    key_ptr,
    value_ptr,
    value_scale_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_st: tl.constexpr,
    stride_sg: tl.constexpr,
    context,
    scale,
    N_BLOCKS: tl.constexpr,
    QUERY_HEADS_: tl.constexpr,
    KV_HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Produce one stable-softmax state per (batch, query head, token block)."""
    block_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // QUERY_HEADS_
    query_head_id = batch_head_id % QUERY_HEADS_
    queries_per_kv = QUERY_HEADS_ // KV_HEADS_
    kv_head_id = query_head_id // queries_per_kv

    token_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM_)
    token_mask = token_offsets < context

    query_offsets = (
        batch_id * stride_qb
        + query_head_id * stride_qh
        + 0 * stride_qt
        + dim_offsets * stride_qd
    )
    query = tl.load(query_ptr + query_offsets).to(tl.float32)

    key_offsets = (
        batch_id * stride_kb
        + kv_head_id * stride_kh
        + token_offsets[:, None] * stride_kt
        + dim_offsets[None, :] * stride_kd
    )
    key = tl.load(key_ptr + key_offsets, mask=token_mask[:, None], other=0.0)
    scores = tl.sum(key.to(tl.float32) * query[None, :], axis=1) * scale
    scores = tl.where(token_mask, scores, -float("inf"))

    local_max = tl.max(scores, axis=0)
    probability = tl.exp(scores - local_max)
    local_sum = tl.sum(probability, axis=0)

    value_offsets = (
        batch_id * stride_vb
        + kv_head_id * stride_vh
        + token_offsets[:, None] * stride_vt
        + dim_offsets[None, :] * stride_vd
    )
    value_payload = tl.load(
        value_ptr + value_offsets,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    value_scale_offsets = (
        batch_id * stride_sb
        + kv_head_id * stride_sh
        + token_offsets * stride_st
        + 0 * stride_sg
    )
    value_scale = tl.load(
        value_scale_ptr + value_scale_offsets,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    value = value_payload * value_scale[:, None]
    local_accumulator = tl.sum(probability[:, None] * value, axis=0)

    state_id = batch_head_id * N_BLOCKS + block_id
    tl.store(partial_max_ptr + state_id, local_max)
    tl.store(partial_sum_ptr + state_id, local_sum)
    tl.store(
        partial_acc_ptr + state_id * HEAD_DIM_ + dim_offsets,
        local_accumulator,
    )


@triton.jit
def _merge_attention_partials(
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    output_ptr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_od: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    QUERY_HEADS_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    BLOCK_STATES: tl.constexpr,
):
    """Merge token-block softmax states and write one decode output row."""
    batch_head_id = tl.program_id(0)
    batch_id = batch_head_id // QUERY_HEADS_
    query_head_id = batch_head_id % QUERY_HEADS_
    block_offsets = tl.arange(0, BLOCK_STATES)
    dim_offsets = tl.arange(0, HEAD_DIM_)
    block_mask = block_offsets < N_BLOCKS
    state_offsets = batch_head_id * N_BLOCKS + block_offsets

    local_max = tl.load(
        partial_max_ptr + state_offsets,
        mask=block_mask,
        other=-float("inf"),
    )
    global_max = tl.max(local_max, axis=0)
    rescale = tl.exp(local_max - global_max)
    local_sum = tl.load(
        partial_sum_ptr + state_offsets,
        mask=block_mask,
        other=0.0,
    )
    denominator = tl.sum(local_sum * rescale, axis=0)

    accumulator_offsets = (
        state_offsets[:, None] * HEAD_DIM_ + dim_offsets[None, :]
    )
    local_accumulator = tl.load(
        partial_acc_ptr + accumulator_offsets,
        mask=block_mask[:, None],
        other=0.0,
    )
    output = tl.sum(local_accumulator * rescale[:, None], axis=0) / denominator

    output_offsets = (
        batch_id * stride_ob
        + query_head_id * stride_oh
        + 0 * stride_ot
        + dim_offsets * stride_od
    )
    tl.store(output_ptr + output_offsets, output)


def eager_vfp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value_payload: torch.Tensor,
    value_scale: torch.Tensor,
) -> torch.Tensor:
    """Existing eager pattern: full V dequantization and full GQA repeat."""
    value = dequantize_kv_fp8(value_payload, value_scale, DTYPE)
    repeats = query.shape[1] // key.shape[1]
    key_repeated = kv_repeat(key, repeats)
    value_repeated = kv_repeat(value, repeats)
    scores = torch.matmul(query, key_repeated.transpose(-2, -1)) / math.sqrt(
        query.shape[-1]
    )
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(probability, value_repeated.to(torch.float32))
    return output.to(DTYPE)


def triton_vfp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value_payload: torch.Tensor,
    value_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Run split-K Triton attention without materializing dequantized/repeated KV."""
    batch, query_heads, query_tokens, head_dim = query.shape
    _, kv_heads, context, key_dim = key.shape
    if query_tokens != 1:
        raise ValueError("decode probe requires exactly one query token")
    if head_dim != HEAD_DIM or key_dim != HEAD_DIM:
        raise ValueError(f"this probe is specialized to head_dim={HEAD_DIM}")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if value_scale.shape != (batch, kv_heads, context, 1):
        raise ValueError("V-FP8 requires one scale per batch/KV-head/token")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"unsupported block_size={block_size}")
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value_payload", value_payload),
        ("value_scale", value_scale),
    ):
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA storage")

    number_of_blocks = triton.cdiv(context, block_size)
    block_states = triton.next_power_of_2(number_of_blocks)
    output = torch.empty_like(query)
    partial_max = torch.empty(
        batch, query_heads, number_of_blocks, device=query.device, dtype=torch.float32
    )
    partial_sum = torch.empty_like(partial_max)
    partial_accumulator = torch.empty(
        batch,
        query_heads,
        number_of_blocks,
        head_dim,
        device=query.device,
        dtype=torch.float32,
    )

    _vfp8_attention_partials[(number_of_blocks, batch * query_heads)](
        query,
        key,
        value_payload,
        value_scale,
        partial_max,
        partial_sum,
        partial_accumulator,
        *query.stride(),
        *key.stride(),
        *value_payload.stride(),
        *value_scale.stride(),
        context,
        1.0 / math.sqrt(head_dim),
        N_BLOCKS=number_of_blocks,
        QUERY_HEADS_=query_heads,
        KV_HEADS_=kv_heads,
        HEAD_DIM_=head_dim,
        BLOCK_N=block_size,
        num_warps=4,
        num_stages=2,
    )
    _merge_attention_partials[(batch * query_heads,)](
        partial_max,
        partial_sum,
        partial_accumulator,
        output,
        *output.stride(),
        N_BLOCKS=number_of_blocks,
        QUERY_HEADS_=query_heads,
        HEAD_DIM_=head_dim,
        BLOCK_STATES=block_states,
        num_warps=4,
        num_stages=2,
    )
    return output


def benchmark(function):
    for _ in range(WARMUP):
        output = function()
        del output
    torch.cuda.synchronize()

    latency_ms = []
    peak_extra_mib = []
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
        peak_extra_mib.append((peak - floor) / 2**20)
        del output

    return {
        "median_latency_ms": statistics.median(latency_ms),
        "latency_ms": latency_ms,
        "median_peak_extra_mib": statistics.median(peak_extra_mib),
        "peak_extra_mib": peak_extra_mib,
    }


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    reference32 = reference.to(torch.float32)
    difference = candidate.to(torch.float32) - reference32
    denominator = torch.linalg.vector_norm(reference32).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / denominator).item(),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def verdict(record: dict) -> str:
    error = record["correctness"]
    if (
        not error["finite"]
        or error["max_abs"] > 1e-3
        or error["relative_l2"] > 1e-2
    ):
        return "STOP_CORRECTNESS"
    if record["triton"]["median_peak_extra_mib"] > 10.0:
        return "STOP_MEMORY"
    ratio = record["latency_ratio_vs_eager"]
    if ratio <= 1.25:
        return "PASS_ENGINE_INTEGRATION_GATE"
    if ratio <= 1.50:
        return "BORDERLINE_ONE_OPTIMIZATION_PASS"
    return "STOP_LATENCY"


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        print(f"unsupported CUDA capability: {capability}")
        return 2

    torch.manual_seed(0)
    records = []
    print(
        "Triton V-FP8 GQA decode-attention probe: "
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={torch.cuda.get_device_name(0)} capability={capability}",
        flush=True,
    )
    print(
        "split-K partials fuse V dequantization and native-GQA mapping; "
        "attention-core only, no model integration\n",
        flush=True,
    )

    for context in CONTEXTS:
        generator = torch.Generator(device="cuda").manual_seed(3000 + context)
        query = torch.randn(
            BATCH,
            QUERY_HEADS,
            1,
            HEAD_DIM,
            generator=generator,
            device="cuda",
            dtype=DTYPE,
        ).contiguous()
        key = torch.randn(
            BATCH,
            KV_HEADS,
            context,
            HEAD_DIM,
            generator=generator,
            device="cuda",
            dtype=DTYPE,
        ).contiguous()
        value = torch.randn(
            BATCH,
            KV_HEADS,
            context,
            HEAD_DIM,
            generator=generator,
            device="cuda",
            dtype=DTYPE,
        )
        value_payload, value_scale = quantize_kv_fp8(value, group_size=HEAD_DIM)
        value_payload = value_payload.contiguous()
        value_scale = value_scale.contiguous()
        del value

        eager = benchmark(
            lambda: eager_vfp8_attention(query, key, value_payload, value_scale)
        )
        reference = eager_vfp8_attention(query, key, value_payload, value_scale)
        print(
            f"ctx={context:5d} eager={eager['median_latency_ms']:.3f}ms "
            f"extra={eager['median_peak_extra_mib']:.2f}MiB",
            flush=True,
        )

        for block_size in BLOCK_SIZES:
            candidate = triton_vfp8_attention(
                query, key, value_payload, value_scale, block_size
            )
            torch.cuda.synchronize()
            correctness = error_metrics(reference, candidate)
            del candidate
            measured = benchmark(
                lambda block_size=block_size: triton_vfp8_attention(
                    query, key, value_payload, value_scale, block_size
                )
            )
            latency_ratio = measured["median_latency_ms"] / max(
                eager["median_latency_ms"], 1e-12
            )
            memory_reduction = (
                1.0
                - measured["median_peak_extra_mib"]
                / max(eager["median_peak_extra_mib"], 1e-12)
            ) * 100.0
            record = {
                "context": context,
                "block_size": block_size,
                "eager": eager,
                "triton": measured,
                "correctness": correctness,
                "latency_ratio_vs_eager": latency_ratio,
                "temporary_memory_reduction_pct": memory_reduction,
            }
            record["verdict"] = verdict(record)
            records.append(record)
            print(
                f"  block={block_size:3d} "
                f"triton={measured['median_latency_ms']:.3f}ms "
                f"extra={measured['median_peak_extra_mib']:.2f}MiB "
                f"mem_reduction={memory_reduction:+.1f}% "
                f"latency={latency_ratio:.2f}x "
                f"max_abs={correctness['max_abs']:.6f} "
                f"rel_l2={correctness['relative_l2']:.6f} "
                f"verdict={record['verdict']}",
                flush=True,
            )

        del reference, query, key, value_payload, value_scale
        gc.collect()
        torch.cuda.empty_cache()

    eight_k = [record for record in records if record["context"] == 8192]
    passing = [
        record
        for record in eight_k
        if record["verdict"] == "PASS_ENGINE_INTEGRATION_GATE"
    ]
    best = min(passing, key=lambda row: row["triton"]["median_latency_ms"]) if passing else None
    result = {
        "status": "isolated_triton_probe_no_engine_change",
        "gpu": torch.cuda.get_device_name(0),
        "capability": capability,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
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
        "gates": {
            "max_abs": 1e-3,
            "relative_l2": 1e-2,
            "peak_extra_mib": 10.0,
            "pass_latency_ratio": 1.25,
            "borderline_latency_ratio": 1.50,
        },
        "records": records,
        "eight_k_decision": (
            {
                "verdict": "PROCEED_TO_BOUNDED_ENGINE_INTEGRATION",
                "block_size": best["block_size"],
            }
            if best is not None
            else {"verdict": "DO_NOT_INTEGRATE"}
        ),
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(f"\nJSON={OUTPUT}")
    print(
        "8K decision:",
        result["eight_k_decision"],
        "(probe only; whole-model validation is still required after any integration)",
    )
    return 0 if all(row["correctness"]["finite"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
