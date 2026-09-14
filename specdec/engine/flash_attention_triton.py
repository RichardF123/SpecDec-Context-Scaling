"""Inference-only BF16 FlashAttention forward for the custom Qwen engine.

This module is deliberately isolated from ``model_self.py`` until the numerical
and operational gates pass on the target RTX 4060.  Unlike the decode-only
V-FP8 split-K probe, this is a standard tiled FlashAttention forward pass:

* query rows and KV rows are tiled;
* the causal score matrix is never materialized;
* online softmax carries the running maximum, denominator, and output state;
* native GQA head mapping avoids physically repeating KV heads;
* both full prefill and cached/chunked prefill/decode are supported.

Only inference forward is in scope for this project; no backward kernel is
implemented.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - validated on the GPU host
    raise RuntimeError("Triton is required for the custom FlashAttention kernel") from error


SUPPORTED_HEAD_DIM = 128
KERNEL_REVISION = "v2_bf16_score_runtime_strides"


@triton.jit
def _flash_attention_bf16_fwd(
    query_ptr,
    key_ptr,
    value_ptr,
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
    """One program owns one query tile for one batch/query-head pair."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_id = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    queries_per_kv_head = QUERY_HEADS // KV_HEADS
    kv_head = query_head // queries_per_kv_head

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    row_mask = rows < query_length

    query_offsets = (
        batch_id * stride_qb
        + query_head * stride_qh
        + rows[:, None] * stride_qt
        + cols[None, :] * stride_qd
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=row_mask[:, None],
        other=0.0,
    )

    # Invalid padded rows use a finite state so they cannot create NaNs in the
    # online recurrence.  They are masked again at the output store.
    running_max = tl.where(row_mask, -float("inf"), 0.0).to(tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    key_cols = tl.arange(0, BLOCK_N)
    for key_start in tl.range(0, kv_length, BLOCK_N):
        key_rows = key_start + key_cols
        key_mask = key_rows < kv_length
        key_offsets = (
            batch_id * stride_kb
            + kv_head * stride_kh
            + key_rows[:, None] * stride_kt
            + cols[None, :] * stride_kd
        )
        key = tl.load(
            key_ptr + key_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(query, tl.trans(key)).to(tl.float32)
        scores *= softmax_scale
        # PyTorch BF16 matmul/division materializes BF16 scores before its
        # float32 softmax.  Keeping the dot product in FP32 here changes every
        # layer slightly; the difference compounds badly over the full model.
        # Reproduce the engine's score boundary before the online recurrence.
        scores = scores.to(tl.bfloat16).to(tl.float32)
        allowed = row_mask[:, None] & key_mask[None, :]
        if CAUSAL:
            absolute_query_rows = query_start + rows
            allowed = allowed & (
                key_rows[None, :] <= absolute_query_rows[:, None]
            )
        scores = tl.where(allowed, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        # exp2 is Triton's fast exponential.  LOG2_E converts natural-log
        # softmax exponents to base 2 without changing the mathematical result.
        LOG2_E: tl.constexpr = 1.4426950408889634
        probability = tl.exp2((scores - next_max[:, None]) * LOG2_E)
        probability = tl.where(allowed, probability, 0.0)
        old_weight = tl.exp2((running_max - next_max) * LOG2_E)

        value_offsets = (
            batch_id * stride_vb
            + kv_head * stride_vh
            + key_rows[:, None] * stride_vt
            + cols[None, :] * stride_vd
        )
        value = tl.load(
            value_ptr + value_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )

        accumulator = accumulator * old_weight[:, None]
        # Match the custom eager engine's BF16 probability x BF16 value matmul
        # while retaining FP32 accumulation and online-softmax state.
        accumulator += tl.dot(probability.to(tl.bfloat16), value)
        running_sum = running_sum * old_weight + tl.sum(probability, axis=1)
        running_max = next_max

    output = accumulator / running_sum[:, None]
    output_offsets = (
        batch_id * stride_ob
        + query_head * stride_oh
        + rows[:, None] * stride_ot
        + cols[None, :] * stride_od
    )
    tl.store(output_ptr + output_offsets, output, mask=row_mask[:, None])


def flash_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Compute GQA attention without materializing scores or repeated KV.

    Shapes are ``query=(B,Hq,M,D)`` and ``key/value=(B,Hkv,N,D)``.  For a
    cached chunk, the query tokens are assumed to be the final ``M`` positions
    of the ``N``-token KV sequence.  This bottom-right causal alignment is what
    the custom engine needs for both chunked prefill and single-token decode.
    """
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

    block_m = 16 if query_length == 1 else 64
    block_n = 64
    output = torch.empty_like(query)
    grid = (triton.cdiv(query_length, block_m), batch * query_heads)
    _flash_attention_bf16_fwd[grid](
        query,
        key,
        value,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        query_length,
        kv_length,
        kv_length - query_length,
        1.0 / math.sqrt(head_dim),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return output
