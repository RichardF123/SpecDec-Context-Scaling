"""Two-pass BF16 GQA FlashAttention backend.

The one-pass kernel keeps online-softmax state and never materializes the
attention matrix, but its block-local BF16 conversion does not match the
custom eager engine's ``softmax(..., dtype=float32).to(bfloat16)`` boundary.
This implementation deliberately recomputes QK in a second pass:

1. compute and store one global maximum and denominator per query row;
2. recompute QK, globally normalize each probability, cast that probability
   to BF16, and multiply by V.

The O(M*N) score/probability matrix is still never materialized.  The extra
storage is only two FP32 scalars per query row.  The engine keeps eager
attention as its default and imports this inference-only backend lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - validated on the GPU host
    raise RuntimeError("Triton is required for the two-pass backend") from error


SUPPORTED_HEAD_DIM = 128
KERNEL_REVISION = "v3_two_pass_global_probability_bf16"


@dataclass(frozen=True)
class FlashKernelConfig:
    """Explicit launch configuration used by the bounded tuning harness."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def validated_pr2_config(query_length: int) -> FlashKernelConfig:
    """Return the PR #2 launch policy retained as the production default."""
    return FlashKernelConfig(
        block_m=16 if query_length == 1 else 64,
        block_n=64,
        num_warps=4,
        num_stages=2,
    )


def _validate_config(config: FlashKernelConfig) -> None:
    if config.block_m not in (8, 16, 32, 64):
        raise ValueError(f"unsupported BLOCK_M={config.block_m}")
    if config.block_n not in (32, 64, 128):
        raise ValueError(f"unsupported BLOCK_N={config.block_n}")
    if config.num_warps not in (2, 4, 8):
        raise ValueError(f"unsupported num_warps={config.num_warps}")
    if config.num_stages not in (1, 2, 3):
        raise ValueError(f"unsupported num_stages={config.num_stages}")


@triton.jit
def _softmax_stats_bf16(
    query_ptr,
    key_ptr,
    row_max_ptr,
    row_sum_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_mb,
    stride_mh,
    stride_mt,
    query_length,
    kv_length,
    query_start,
    softmax_scale,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Pass 1: compute global FP32 softmax statistics for each query row."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_id = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < query_length
    query_offsets = (
        batch_id * stride_qb
        + query_head * stride_qh
        + rows[:, None] * stride_qt
        + dims[None, :] * stride_qd
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=row_mask[:, None],
        other=0.0,
    )

    running_max = tl.where(row_mask, -float("inf"), 0.0).to(tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    key_offsets_in_block = tl.arange(0, BLOCK_N)
    LOG2_E: tl.constexpr = 1.4426950408889634

    for key_start in tl.range(0, kv_length, BLOCK_N):
        key_rows = key_start + key_offsets_in_block
        key_mask = key_rows < kv_length
        key_offsets = (
            batch_id * stride_kb
            + kv_head * stride_kh
            + key_rows[:, None] * stride_kt
            + dims[None, :] * stride_kd
        )
        key = tl.load(
            key_ptr + key_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )

        # Match both BF16 materialization boundaries in the eager expression:
        # BF16 matmul output, followed by BF16 scalar division.
        scores = tl.dot(query, tl.trans(key)).to(tl.bfloat16)
        scores = (scores * softmax_scale).to(tl.bfloat16).to(tl.float32)
        allowed = row_mask[:, None] & key_mask[None, :]
        if CAUSAL:
            absolute_query_rows = query_start + rows
            allowed = allowed & (
                key_rows[None, :] <= absolute_query_rows[:, None]
            )
        scores = tl.where(allowed, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        block_exp = tl.exp2((scores - next_max[:, None]) * LOG2_E)
        block_exp = tl.where(allowed, block_exp, 0.0)
        old_weight = tl.exp2((running_max - next_max) * LOG2_E)
        running_sum = running_sum * old_weight + tl.sum(block_exp, axis=1)
        running_max = next_max

    stats_offsets = (
        batch_id * stride_mb
        + query_head * stride_mh
        + rows * stride_mt
    )
    tl.store(row_max_ptr + stats_offsets, running_max, mask=row_mask)
    tl.store(row_sum_ptr + stats_offsets, running_sum, mask=row_mask)


@triton.jit
def _globally_normalized_output_bf16(
    query_ptr,
    key_ptr,
    value_ptr,
    row_max_ptr,
    row_sum_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_mt,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    query_length,
    kv_length,
    query_start,
    softmax_scale,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Pass 2: cast globally normalized probabilities before P @ V."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_id = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < query_length
    query_offsets = (
        batch_id * stride_qb
        + query_head * stride_qh
        + rows[:, None] * stride_qt
        + dims[None, :] * stride_qd
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=row_mask[:, None],
        other=0.0,
    )

    stats_offsets = (
        batch_id * stride_mb
        + query_head * stride_mh
        + rows * stride_mt
    )
    global_max = tl.load(row_max_ptr + stats_offsets, mask=row_mask, other=0.0)
    global_sum = tl.load(row_sum_ptr + stats_offsets, mask=row_mask, other=1.0)

    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    key_offsets_in_block = tl.arange(0, BLOCK_N)
    LOG2_E: tl.constexpr = 1.4426950408889634

    for key_start in tl.range(0, kv_length, BLOCK_N):
        key_rows = key_start + key_offsets_in_block
        key_mask = key_rows < kv_length
        key_offsets = (
            batch_id * stride_kb
            + kv_head * stride_kh
            + key_rows[:, None] * stride_kt
            + dims[None, :] * stride_kd
        )
        key = tl.load(
            key_ptr + key_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key)).to(tl.bfloat16)
        scores = (scores * softmax_scale).to(tl.bfloat16).to(tl.float32)
        allowed = row_mask[:, None] & key_mask[None, :]
        if CAUSAL:
            absolute_query_rows = query_start + rows
            allowed = allowed & (
                key_rows[None, :] <= absolute_query_rows[:, None]
            )

        scores = tl.where(allowed, scores, -float("inf"))
        numerator = tl.exp2((scores - global_max[:, None]) * LOG2_E)
        probability = tl.where(
            allowed,
            numerator / global_sum[:, None],
            0.0,
        )
        value_offsets = (
            batch_id * stride_vb
            + kv_head * stride_vh
            + key_rows[:, None] * stride_vt
            + dims[None, :] * stride_vd
        )
        value = tl.load(
            value_ptr + value_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(probability.to(tl.bfloat16), value)

    output_offsets = (
        batch_id * stride_ob
        + query_head * stride_oh
        + rows[:, None] * stride_ot
        + dims[None, :] * stride_od
    )
    tl.store(output_ptr + output_offsets, accumulator, mask=row_mask[:, None])


def flash_attention_bf16_two_pass(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    config: FlashKernelConfig | None = None,
) -> torch.Tensor:
    """Run two-pass global-normalization BF16 GQA attention."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must all be rank-4 tensors")
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, kv_length, key_dim = key.shape
    if value.shape != key.shape:
        raise ValueError(f"value shape {value.shape} must equal key shape {key.shape}")
    if batch != key_batch:
        raise ValueError("query and KV batch sizes differ")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if head_dim != key_dim or head_dim != SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"this kernel requires head_dim={SUPPORTED_HEAD_DIM}; got {head_dim}"
        )
    if not 0 < query_length <= kv_length:
        raise ValueError((query_length, kv_length))
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    config = validated_pr2_config(query_length) if config is None else config
    _validate_config(config)
    block_m = config.block_m
    block_n = config.block_n
    row_max = torch.empty(
        (batch, query_heads, query_length),
        device=query.device,
        dtype=torch.float32,
    )
    row_sum = torch.empty_like(row_max)
    output = torch.empty_like(query)
    grid = (triton.cdiv(query_length, block_m), batch * query_heads)
    shared = (
        *query.stride(),
        *key.stride(),
        *row_max.stride(),
        query_length,
        kv_length,
        kv_length - query_length,
        1.0 / math.sqrt(head_dim),
    )
    constexpr = {
        "QUERY_HEADS": query_heads,
        "KV_HEADS": kv_heads,
        "HEAD_DIM": head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "CAUSAL": causal,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
    _softmax_stats_bf16[grid](
        query,
        key,
        row_max,
        row_sum,
        *shared,
        **constexpr,
    )
    _globally_normalized_output_bf16[grid](
        query,
        key,
        value,
        row_max,
        row_sum,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *row_max.stride(),
        *output.stride(),
        query_length,
        kv_length,
        kv_length - query_length,
        1.0 / math.sqrt(head_dim),
        **constexpr,
    )
    return output
