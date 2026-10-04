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



def paged_attention_block(
    q,
    layer,
    *,
    backend: str = "cuda",
    validate: bool = True,
):
    '''Direct paged-KV CUDA attention for q_len=1..4.\n\n    FINAL_PAGED_V2_Q1_Q4_VFP8\n\n    Supported:\n    - fp16/base: K/V match q dtype (BF16 or FP32)\n    - v_fp8: K BF16, V float8_e4m3fn, V scale float32; q BF16\n    '''
    import torch
    from specdec.engine import paged_cache

    if backend != "cuda":
        raise ValueError("paged_attention_block supports backend='cuda' only")
    if not paged_cache.is_paged_layer(layer):
        raise TypeError("paged_attention_block requires a paged_v0 layer")
    if validate:
        paged_cache.check_consistent(layer)

    if not torch.is_tensor(q):
        raise TypeError("q must be a tensor")
    if not q.is_cuda:
        raise ValueError("q must be on CUDA")
    if q.ndim != 4 or not (1 <= int(q.shape[2]) <= 4):
        raise ValueError(
            f"q shape must be [B,H,S,D] with 1<=S<=4, got {tuple(q.shape)}")

    length = paged_cache.logical_length(layer)
    page_size = int(layer["page_size"])
    if length <= 0:
        raise ValueError("paged cache must be non-empty")
    if length < int(q.shape[2]):
        raise ValueError("paged cache length must be >= q_len")

    k_pages = layer["k_pages"]
    v_pages = layer["v_pages"]
    if not k_pages:
        raise ValueError("paged cache has no pages")

    first_k = k_pages[0]
    first_v = v_pages[0]
    if first_k.device != q.device or first_v.device != q.device:
        raise ValueError("q and paged K/V must be on the same CUDA device")
    if int(first_k.shape[0]) != int(q.shape[0]):
        raise ValueError("q and paged K/V batch sizes differ")
    if int(first_k.shape[-1]) != int(q.shape[-1]):
        raise ValueError("q and paged K head dimensions differ")

    kv_heads = int(first_k.shape[1])
    if int(q.shape[1]) % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")

    from .build import load

    if layer["fmt"] == "fp16":
        if first_k.dtype != q.dtype or first_v.dtype != q.dtype:
            raise ValueError(
                f"q/K/V dtype mismatch: {q.dtype}, {first_k.dtype}, {first_v.dtype}")
        k_ptrs, v_ptrs = paged_cache.cuda_pointer_tables(
            layer, validate=False)
        return load().paged_attention_block(
            q.contiguous(),
            k_ptrs,
            v_ptrs,
            kv_heads,
            page_size,
            length,
        )

    if layer["fmt"] == "v_fp8":
        if q.dtype != torch.bfloat16 or first_k.dtype != torch.bfloat16:
            raise ValueError("V-FP8 direct paged attention requires BF16 q/K")
        if first_v.dtype != torch.float8_e4m3fn:
            raise ValueError("V-FP8 direct paged attention requires E4M3FN V")
        first_scale = layer["v_scale_pages"][0]
        if first_scale.dtype != torch.float32:
            raise ValueError("V-FP8 direct paged attention requires FP32 scale")
        scale_groups = int(first_scale.shape[-1])
        if scale_groups <= 0 or int(q.shape[-1]) % scale_groups != 0:
            raise ValueError("invalid V-FP8 scale geometry")

        k_ptrs, v_ptrs, s_ptrs = paged_cache.cuda_pointer_tables_vfp8(
            layer, validate=False)
        return load().paged_attention_vfp8_block(
            q.contiguous(),
            k_ptrs,
            v_ptrs,
            s_ptrs,
            kv_heads,
            page_size,
            length,
            scale_groups,
        )

    raise ValueError(f"unsupported paged cache format {layer['fmt']!r}")


def phase12_paged_attention_h2o(q, layer, *, validate=True):
    """Experimental direct paged attention + H2O mass capture."""
    import torch
    from specdec.engine import paged_cache
    if validate:
        paged_cache.check_consistent(layer)
    if q.ndim != 4 or not (1 <= int(q.shape[2]) <= 4):
        raise ValueError("q must be [B,H,S,D], 1<=S<=4")
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("Phase12 H2O capture requires CUDA BF16 q")
    length=paged_cache.logical_length(layer)
    page=int(layer["page_size"])
    kv=int(layer["k_pages"][0].shape[1])
    from .build import load
    mod=load()
    if layer["fmt"]=="fp16":
        kp,vp=paged_cache.cuda_pointer_tables(layer,validate=False)
        pair=mod.phase12_paged_attention_h2o_bf16(
            q.contiguous(),kp,vp,kv,page,length)
    elif layer["fmt"]=="v_fp8":
        kp,vp,sp=paged_cache.cuda_pointer_tables_vfp8(layer,validate=False)
        groups=int(layer["v_scale_pages"][0].shape[-1])
        pair=mod.phase12_paged_attention_h2o_vfp8(
            q.contiguous(),kp,vp,sp,kv,page,length,groups)
    else:
        raise ValueError(f"unsupported fmt {layer['fmt']!r}")
    return pair[0],pair[1]


def phase12_paged_compact(src, dst, indices):
    """Copy selected resident payload directly page-to-page."""
    import torch
    from specdec.engine import paged_cache
    if src["fmt"] != dst["fmt"]:
        raise ValueError("src/dst fmt mismatch")
    if indices.ndim!=1 or indices.dtype!=torch.long or not indices.is_cuda:
        raise ValueError("indices must be CUDA rank-1 torch.long")
    first=src["k_pages"][0]
    if first.dtype != torch.bfloat16:
        raise ValueError("Phase12 compaction requires BF16 K")
    B,KV,_,D=map(int,first.shape)
    page=int(src["page_size"])
    from .build import load
    mod=load()
    if src["fmt"]=="fp16":
        sk,sv=paged_cache.cuda_pointer_tables(src,validate=False)
        dk,dv=paged_cache.cuda_pointer_tables(dst,validate=False)
        mod.phase12_paged_compact_bf16(
            sk,sv,dk,dv,indices.contiguous(),B,KV,D,page)
    elif src["fmt"]=="v_fp8":
        sk,sv,ss=paged_cache.cuda_pointer_tables_vfp8(src,validate=False)
        dk,dv,ds=paged_cache.cuda_pointer_tables_vfp8(dst,validate=False)
        groups=int(src["v_scale_pages"][0].shape[-1])
        mod.phase12_paged_compact_vfp8(
            sk,sv,ss,dk,dv,ds,indices.contiguous(),
            B,KV,D,groups,page)
    else:
        raise ValueError(f"unsupported fmt {src['fmt']!r}")
    return dst


def phase13_flash_paged_attention(q, layer, *, validate=True):
    """Page-native GQA-aware tiled online-softmax attention, q_len=1..4.

    This is an experimental Flash-style route.  It keeps K/V paged and shares
    each K/V tile across query heads that map to the same KV head.
    """
    import torch
    from specdec.engine import paged_cache

    if validate:
        paged_cache.check_consistent(layer)

    if not paged_cache.is_paged_layer(layer):
        raise TypeError("phase13 Flash-paged attention requires paged_v0")

    if q.ndim != 4 or not (1 <= int(q.shape[2]) <= 4):
        raise ValueError("q must be [B,H,S,D] with q_len 1..4")

    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("phase13 Flash-paged attention requires CUDA BF16 q")

    length = paged_cache.logical_length(layer)
    page_size = int(layer["page_size"])
    first_k = layer["k_pages"][0]
    kv_heads = int(first_k.shape[1])

    if int(q.shape[1]) % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")

    from .build import load
    mod = load()

    if layer["fmt"] == "fp16":
        if first_k.dtype != torch.bfloat16:
            raise ValueError("BF16 Flash-paged path requires BF16 K")
        if layer["v_pages"][0].dtype != torch.bfloat16:
            raise ValueError("BF16 Flash-paged path requires BF16 V")
        kp, vp = paged_cache.cuda_pointer_tables(layer, validate=False)
        return mod.phase13_flash_paged_bf16(
            q.contiguous(),
            kp,
            vp,
            kv_heads,
            page_size,
            length,
        )

    if layer["fmt"] == "v_fp8":
        if first_k.dtype != torch.bfloat16:
            raise ValueError("V-FP8 Flash-paged path requires BF16 K")
        if layer["v_pages"][0].dtype != torch.float8_e4m3fn:
            raise ValueError("V-FP8 Flash-paged path requires E4M3FN V")
        scale = layer["v_scale_pages"][0]
        if scale.dtype != torch.float32:
            raise ValueError("V-FP8 Flash-paged path requires FP32 scale")
        groups = int(scale.shape[-1])
        kp, vp, sp = paged_cache.cuda_pointer_tables_vfp8(
            layer, validate=False)
        return mod.phase13_flash_paged_vfp8(
            q.contiguous(),
            kp,
            vp,
            sp,
            kv_heads,
            page_size,
            length,
            groups,
        )

    raise ValueError(
        f"unsupported paged cache format {layer['fmt']!r}")
