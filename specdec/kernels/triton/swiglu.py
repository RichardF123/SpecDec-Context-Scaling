"""Fused silu(gate) * up in Triton."""

import triton
import triton.language as tl


@triton.jit
def _swiglu(gate_ptr, up_ptr, y_ptr, n, BLOCK: tl.constexpr, BF16: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g / (1.0 + tl.exp(-g))) * u
    if BF16:
        tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(y_ptr + offs, y, mask=mask)


def apply(gate, up):
    import torch

    gate = gate.contiguous()
    up = up.contiguous()
    y = torch.empty_like(gate)
    n = gate.numel()
    if n == 0:
        return y
    block = 1024
    _swiglu[(triton.cdiv(n, block),)](
        gate, up, y, n, BLOCK=block, BF16=gate.dtype == torch.bfloat16,
    )
    return y
