"""RoPE in Triton. rotate_half matches model_self.apply_rope."""

import triton
import triton.language as tl


@triton.jit
def _rope(x_ptr, cos_ptr, sin_ptr, y_ptr,
          stride_b, stride_h, stride_s, stride_d,
          heads, seq, head_dim, n, BLOCK: tl.constexpr, BF16: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    half = head_dim // 2
    d = offs % head_dim
    sd = head_dim * seq
    s = (offs // head_dim) % seq
    h = (offs // sd) % heads
    b = offs // (sd * heads)
    base = b * stride_b + h * stride_h + s * stride_s
    in_off = base + d * stride_d
    partner_d = tl.where(d < half, d + half, d - half)
    partner_off = base + partner_d * stride_d
    x = tl.load(x_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
    partner = tl.load(x_ptr + partner_off, mask=mask, other=0.0).to(tl.float32)
    partner = tl.where(d < half, -partner, partner)
    cs = s * head_dim + d
    cos = tl.load(cos_ptr + cs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs, mask=mask, other=0.0).to(tl.float32)
    y = x * cos + partner * sin
    if BF16:
        tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(y_ptr + offs, y, mask=mask)


def apply(x, cos, sin):
    import torch

    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n = x.numel()
    if n == 0:
        return y
    block = 256
    _rope[(triton.cdiv(n, block),)](
        x, cos, sin, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x.shape[1], x.shape[2], x.shape[3], n,
        BLOCK=block,
        BF16=x.dtype == torch.bfloat16,
    )
    return y
