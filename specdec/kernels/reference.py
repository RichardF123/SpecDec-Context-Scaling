"""Torch formulas copied from vendor specdec/engine/model_self.py.

These are the baseline the custom kernels are measured against. They are not
imported from the vendor module so an operator bench can run without patching
the engine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig = x.dtype
    x32 = x.float()
    var = torch.mean(x32 * x32, dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    return weight * normed.to(orig)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def kv_repeat(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


def _causal_mask(scores: torch.Tensor, s: int, t: int) -> torch.Tensor:
    # Unified engine mask. s == 1 leaves every key visible because
    # diagonal = (T - s) + 1 = T, and triu with that diagonal is empty.
    mask = torch.triu(torch.ones(s, t, device=scores.device, dtype=torch.bool), diagonal=(t - s) + 1)
    return scores.masked_fill(mask, float("-inf"))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Eager path: dtype matmul, fp32 softmax, dtype store. This is what the engine runs."""
    n_rep = q.shape[1] // k.shape[1]
    k_rep = kv_repeat(k, n_rep)
    v_rep = kv_repeat(v, n_rep)
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    s, t = q.shape[-2], k.shape[-2]
    scores = _causal_mask(scores, s, t)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v_rep)


def attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same mask, but QK and AV accumulate in fp32. Algorithm reference for the kernels."""
    n_rep = q.shape[1] // k.shape[1]
    k_rep = kv_repeat(k, n_rep).float()
    v_rep = kv_repeat(v, n_rep).float()
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k_rep.transpose(-2, -1)) * scale
    s, t = q.shape[-2], k.shape[-2]
    scores = _causal_mask(scores, s, t)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_rep)
    return out.to(q.dtype)
