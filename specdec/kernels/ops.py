"""Dispatch Torch, CUDA C++, and Triton implementations of the four kernels."""

from __future__ import annotations

from . import reference


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _cos_sin(x, cos, sin):
    seq, dim = x.shape[-2], x.shape[-1]

    def flat(t):
        t = t.to(dtype=x.dtype)
        while t.ndim > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        if tuple(t.shape) != (seq, dim):
            t = t.reshape(-1, seq, dim)[-1]
        return t.contiguous()

    return flat(cos), flat(sin)


def rmsnorm(x, weight, eps: float = 1e-6, backend: str = "cuda"):
    if backend == "torch":
        return reference.rmsnorm(x, weight, eps)
    if backend == "triton":
        from .triton import rmsnorm as tri
        return tri.apply(x, weight, eps)
    if backend == "cuda":
        from .build import load
        return load().rmsnorm(x.contiguous(), weight.contiguous(), float(eps))
    raise ValueError(backend)


def apply_rope(x, cos, sin, backend: str = "cuda"):
    cos, sin = _cos_sin(x, cos, sin)
    if backend == "torch":
        return reference.apply_rope(x, cos, sin)
    if backend == "triton":
        from .triton import rope as tri
        return tri.apply(x, cos, sin)
    if backend == "cuda":
        from .build import load
        return load().apply_rope(x, cos, sin)
    raise ValueError(backend)


def swiglu(gate, up, backend: str = "cuda"):
    if backend == "torch":
        return reference.swiglu(gate, up)
    if backend == "triton":
        from .triton import swiglu as tri
        return tri.apply(gate, up)
    if backend == "cuda":
        from .build import load
        return load().swiglu(gate.contiguous(), up.contiguous())
    raise ValueError(backend)


def attention(q, k, v, backend: str = "cuda"):
    if backend == "torch":
        return reference.attention(q, k, v)
    if backend == "triton":
        from .triton import attention as tri
        return tri.apply(q, k, v)
    if backend == "cuda":
        from .build import load
        return load().attention(q.contiguous(), k.contiguous(), v.contiguous())
    raise ValueError(backend)
