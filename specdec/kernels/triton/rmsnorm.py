"""RMSNorm in Triton. Same formula as the CUDA kernel and model_self.RMSNORM."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x_ptr, w_ptr, y_ptr, hidden, eps, BLOCK: tl.constexpr, BF16: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    inv = tl.rsqrt(acc / hidden + eps)
    for start in tl.range(0, hidden, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * inv
        if BF16:
            normed = normed.to(tl.bfloat16).to(tl.float32)
            y = (normed * w).to(tl.bfloat16)
        else:
            y = normed * w
        tl.store(y_ptr + row * hidden + offs, y, mask=mask)


def apply(x, weight, eps):
    import torch

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    if rows == 0:
        return y
    block = 1024
    _rmsnorm[(rows,)](
        x, weight, y, hidden, float(eps),
        BLOCK=block,
        BF16=x.dtype == torch.bfloat16,
    )
    return y
