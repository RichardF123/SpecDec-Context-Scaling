"""Causal GQA attention in Triton.

Online softmax, fp32 dots (IEEE, not TF32), no score tensor.
Mask: key j is visible to query i when j <= i + (T - S).
"""

import triton
import triton.language as tl


@triton.jit
def _attn(q_ptr, k_ptr, v_ptr, o_ptr,
          stride_qb, stride_qh, stride_qs, stride_qd,
          stride_kb, stride_kh, stride_ks, stride_kd,
          stride_vb, stride_vh, stride_vs, stride_vd,
          stride_ob, stride_oh, stride_os, stride_od,
          H, KV, S, T, scale,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, BF16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    n_rep = H // KV
    pid_kv = pid_h // n_rep

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < S

    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = k_ptr + pid_b * stride_kb + pid_kv * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_kv * stride_vh

    for start_n in tl.range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=offs_n[None, :] < T, other=0.0).to(tl.float32)
        qk = tl.dot(q, k, input_precision="ieee") * scale
        visible = (offs_n[None, :] <= (offs_m[:, None] + (T - S))) & (offs_n[None, :] < T) & row_ok[:, None]
        qk = tl.where(visible, qk, -1.0e30)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    safe = l_i[:, None] > 0
    out = tl.where(safe, acc / tl.where(safe, l_i[:, None], 1.0), 0.0)
    o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    if BF16:
        tl.store(o_ptrs, out.to(tl.bfloat16), mask=row_ok[:, None])
    else:
        tl.store(o_ptrs, out, mask=row_ok[:, None])


def apply(q, k, v):
    import torch

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, S, D = q.shape
    KV, T = k.shape[1], k.shape[2]
    o = torch.empty_like(q)
    if S == 0 or T == 0:
        return o
    # fp32 tiles are wide at head_dim 128; keep the shared-memory footprint
    # under the 100 KB limit on consumer GPUs (including sm_89).
    if D >= 128:
        block_m, block_n = 16, 32
    else:
        block_m, block_n = 32, 64
    grid = (triton.cdiv(S, block_m), H, B)
    scale = D ** -0.5
    _attn[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, KV, S, T, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
        BF16=q.dtype == torch.bfloat16,
        num_stages=1,
        num_warps=4,
    )
    return o
