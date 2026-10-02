"""Dispatch Torch, CUDA C++, and Triton kernel implementations."""

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


def paged_attention_decode(
    q,
    layer,
    *,
    backend: str = "cuda",
    validate: bool = True,
):
    """Direct paged-KV q_len=1 CUDA attention.

    v1d keeps the numerical/kernel scope of v1c unchanged but reuses lazily
    cached device pointer tables.  `validate=False` is a fast-path option for
    model_self immediately after append_current(), which has already checked
    the cache; direct external callers should keep the default `True`.
    """
    import torch

    from specdec.engine import paged_cache

    if backend != "cuda":
        raise ValueError("paged_attention_decode v1d supports backend='cuda' only")

    if not paged_cache.is_paged_layer(layer):
        raise TypeError("paged_attention_decode requires a paged_v0 layer")

    if validate:
        paged_cache.check_consistent(layer)

    if layer["fmt"] != "fp16":
        raise ValueError("paged_attention_decode v1d supports fp16/base cache only")

    if not torch.is_tensor(q):
        raise TypeError("q must be a tensor")
    if not q.is_cuda:
        raise ValueError("q must be on CUDA")
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError(
            f"decode-only q shape must be [B,H,1,D], got {tuple(q.shape)}")

    length = paged_cache.logical_length(layer)
    page_size = int(layer["page_size"])
    if length <= 0:
        raise ValueError("paged cache must be non-empty")

    k_pages = layer["k_pages"]
    v_pages = layer["v_pages"]
    if not k_pages:
        raise ValueError("paged cache has no pages")

    first_k = k_pages[0]
    first_v = v_pages[0]
    if first_k.dtype != q.dtype or first_v.dtype != q.dtype:
        raise ValueError(
            f"q/K/V dtype mismatch: {q.dtype}, {first_k.dtype}, {first_v.dtype}")
    if first_k.device != q.device or first_v.device != q.device:
        raise ValueError("q and paged K/V must be on the same CUDA device")

    kv_heads = int(first_k.shape[1])
    k_ptrs, v_ptrs = paged_cache.cuda_pointer_tables(
        layer,
        validate=False,
    )

    from .build import load

    return load().paged_attention_decode(
        q.contiguous(),
        k_ptrs,
        v_ptrs,
        kv_heads,
        page_size,
        length,
    )
