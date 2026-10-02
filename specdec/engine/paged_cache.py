"""Paged KV cache v0 storage plus lazy CUDA pointer-table metadata.

The storage semantics are unchanged from paged_v0:
- token-aligned K/V payloads live in fixed-size page tensors;
- append never concatenates full history;
- trim returns a non-mutating prefix and clones a partial tail page for COW;
- materialize_layer recreates the legacy contiguous representation when needed.

PagedAttention v1d adds *metadata-only* pointer-table caching for the direct
CUDA decode kernel.  The page payload tensors remain exactly the same.  A
pointer table is built lazily the first time direct attention needs it and is
reused while the set of physical pages is unchanged.  It is invalidated only
when a new page is allocated.  trim_layer() creates a fresh logical cache, so
trimmed caches never inherit stale pointer metadata.
"""
from __future__ import annotations

import math
from typing import Any

import torch


BACKEND = "paged_v0"
DEFAULT_PAGE_SIZE = 16
SUPPORTED_FORMATS = ("fp16", "v_fp8")

_POINTER_TABLE_STATS = {
    "builds": 0,
    "hits": 0,
    "invalidations": 0,
}


def reset_pointer_table_stats() -> None:
    for key in _POINTER_TABLE_STATS:
        _POINTER_TABLE_STATS[key] = 0


def pointer_table_stats() -> dict:
    return dict(_POINTER_TABLE_STATS)


def is_paged_layer(layer: Any) -> bool:
    return isinstance(layer, dict) and layer.get("_cache_backend") == BACKEND


def _validate_page_size(page_size: int) -> int:
    if (not isinstance(page_size, int) or isinstance(page_size, bool)
            or page_size <= 0):
        raise ValueError(f"page_size must be a positive int, got {page_size!r}")
    return page_size


def _invalidate_pointer_tables(layer: dict) -> None:
    had = any(
        key in layer
        for key in ("_k_ptrs_cuda", "_v_ptrs_cuda", "_ptr_page_count")
    )
    layer.pop("_k_ptrs_cuda", None)
    layer.pop("_v_ptrs_cuda", None)
    layer.pop("_ptr_page_count", None)
    if had:
        _POINTER_TABLE_STATS["invalidations"] += 1


def create_layer_cache(fmt: str = "fp16", page_size: int = DEFAULT_PAGE_SIZE) -> dict:
    if fmt not in SUPPORTED_FORMATS:
        if fmt == "fp8":
            raise ValueError("paged_v0 does not support full fp8 K/V")
        raise ValueError(f"unsupported paged_v0 cache format {fmt!r}")
    page_size = _validate_page_size(page_size)
    layer = {
        "_cache_backend": BACKEND,
        "fmt": fmt,
        "page_size": page_size,
        "length": 0,
        "k_pages": [],
        "v_pages": [],
    }
    if fmt == "v_fp8":
        layer["v_scale_pages"] = []
    return layer


def logical_length(layer: dict) -> int:
    if not is_paged_layer(layer):
        raise TypeError("not a paged_v0 layer cache")
    length = layer.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise RuntimeError(f"invalid paged_v0 logical length {length!r}")
    return length


def _validate_current(layer: dict, k: torch.Tensor, v: torch.Tensor,
                      v_scale: torch.Tensor | None) -> int:
    check_consistent(layer)
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("k and v must be tensors")
    if k.dim() < 3 or v.dim() < 3:
        raise ValueError("k and v must have a token dimension at dim=2")
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
    if k.device != v.device:
        raise ValueError(f"k/v device mismatch: {k.device} vs {v.device}")
    n = int(k.shape[2])
    fmt = layer["fmt"]
    if fmt == "fp16":
        if v_scale is not None:
            raise ValueError("base/fp16 paged cache does not accept v_scale")
        if k.dtype != v.dtype or k.dtype not in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(
                f"base paged cache requires matching floating K/V, got {k.dtype}/{v.dtype}")
    else:
        if v_scale is None or not isinstance(v_scale, torch.Tensor):
            raise ValueError("v_fp8 paged cache requires a v_scale tensor")
        if k.dtype != torch.bfloat16:
            raise ValueError(f"v_fp8 K must be bfloat16, got {k.dtype}")
        if v.dtype != torch.float8_e4m3fn:
            raise ValueError(f"v_fp8 V must be float8_e4m3fn, got {v.dtype}")
        if v_scale.dtype != torch.float32:
            raise ValueError(f"v_fp8 v_scale must be float32, got {v_scale.dtype}")
        if v_scale.dim() < 3 or int(v_scale.shape[2]) != n:
            raise ValueError(
                f"v/v_scale token mismatch: {n} vs "
                f"{tuple(v_scale.shape) if v_scale.dim() >= 3 else 'missing dim=2'}")
        if tuple(v_scale.shape[:2]) != tuple(v.shape[:2]):
            raise ValueError("v/v_scale batch or KV-head dimensions differ")
        if v_scale.device != v.device:
            raise ValueError(f"v/v_scale device mismatch: {v.device} vs {v_scale.device}")
    return n


def _empty_page_like(current: torch.Tensor, page_size: int) -> torch.Tensor:
    shape = list(current.shape)
    shape[2] = page_size
    return torch.empty(shape, dtype=current.dtype, device=current.device)


def append_current(layer: dict, k: torch.Tensor, v: torch.Tensor,
                   v_scale: torch.Tensor | None = None) -> dict:
    """Append current-token payloads in place without concatenating history."""
    n = _validate_current(layer, k, v, v_scale)
    if n == 0:
        return layer
    page_size = layer["page_size"]
    offset = 0
    while offset < n:
        slot = layer["length"] % page_size
        if slot == 0:
            # Existing device pointer tables describe the old physical page set.
            # Invalidate before allocating the next page.  Within a page they stay
            # valid across appends and can be reused by every decode call.
            _invalidate_pointer_tables(layer)
            layer["k_pages"].append(_empty_page_like(k, page_size))
            layer["v_pages"].append(_empty_page_like(v, page_size))
            if layer["fmt"] == "v_fp8":
                layer["v_scale_pages"].append(_empty_page_like(v_scale, page_size))
        take = min(page_size - slot, n - offset)
        layer["k_pages"][-1].narrow(2, slot, take).copy_(k.narrow(2, offset, take))
        layer["v_pages"][-1].narrow(2, slot, take).copy_(v.narrow(2, offset, take))
        if layer["fmt"] == "v_fp8":
            layer["v_scale_pages"][-1].narrow(2, slot, take).copy_(
                v_scale.narrow(2, offset, take))
        layer["length"] += take
        offset += take
    check_consistent(layer)
    return layer


def _materialize_pages(pages: list[torch.Tensor], length: int,
                       page_size: int) -> torch.Tensor:
    if length <= 0 or not pages:
        raise ValueError("cannot materialize an empty paged_v0 layer cache")
    tail = length % page_size
    chunks = list(pages)
    if tail:
        chunks[-1] = chunks[-1].narrow(2, 0, tail)
    if len(chunks) == 1:
        return chunks[0].contiguous()
    return torch.cat(chunks, dim=2)


def materialize_layer(layer: dict) -> dict:
    """Return the legacy contiguous layer representation for attention."""
    check_consistent(layer)
    length = logical_length(layer)
    page_size = layer["page_size"]
    out = {
        "fmt": layer["fmt"],
        "k": _materialize_pages(layer["k_pages"], length, page_size),
        "v": _materialize_pages(layer["v_pages"], length, page_size),
    }
    if layer["fmt"] == "v_fp8":
        out["v_scale"] = _materialize_pages(
            layer["v_scale_pages"], length, page_size)
    return out


def materialize_k(layer: dict) -> torch.Tensor:
    """Materialize only K; used by the H2O v0 reference scorer."""
    check_consistent(layer)
    return _materialize_pages(
        layer["k_pages"], logical_length(layer), layer["page_size"])


def select_indices(layer: dict, indices: torch.Tensor) -> dict:
    """Return a fresh paged cache containing selected resident token positions.

    v0 uses a reference compaction path: materialize, gather, repack.
    """
    check_consistent(layer)
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("indices must be a rank-1 torch.long tensor")
    n = int(indices.numel())
    if n <= 0:
        raise ValueError("select_indices requires at least one retained token")

    cur = logical_length(layer)
    idx = indices.to(device=layer["k_pages"][0].device)
    if int(idx.min()) < 0 or int(idx.max()) >= cur:
        raise ValueError("select_indices contains an out-of-range resident index")
    if n > 1 and not bool(torch.all(idx[1:] > idx[:-1]).item()):
        raise ValueError("select_indices must be strictly increasing")

    raw = materialize_layer(layer)
    out = create_layer_cache(layer["fmt"], layer["page_size"])
    k = raw["k"].index_select(2, idx)
    v = raw["v"].index_select(2, idx)
    if layer["fmt"] == "v_fp8":
        scale = raw["v_scale"].index_select(2, idx)
        append_current(out, k, v, scale)
    else:
        append_current(out, k, v)
    check_consistent(out)
    return out


def payload_bytes(layer: dict) -> int:
    """Allocated bytes for paged K/V payload tensors, excluding metadata."""
    check_consistent(layer)
    total = 0
    for key in ("k_pages", "v_pages", "v_scale_pages"):
        for tensor in layer.get(key, []):
            total += tensor.numel() * tensor.element_size()
    return int(total)


def cuda_pointer_tables(layer: dict, *, validate: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached device arrays of raw K/V page addresses for direct CUDA.

    The returned tensors contain only int64 page addresses; no K/V payload is
    copied or stacked.  Tables are rebuilt only when the physical page count
    changes.  `validate=False` is intended for the model fast path immediately
    after append_current(), which has already run the full consistency check.
    """
    if validate:
        check_consistent(layer)
    elif not is_paged_layer(layer):
        raise TypeError("not a paged_v0 layer cache")

    if layer.get("fmt") != "fp16":
        raise ValueError("CUDA pointer tables v1d are only used for fp16/base cache")

    k_pages = layer.get("k_pages")
    v_pages = layer.get("v_pages")
    if not k_pages or not v_pages:
        raise ValueError("cannot build pointer tables for an empty paged cache")
    if len(k_pages) != len(v_pages):
        raise RuntimeError("paged K/V page count mismatch")

    device = k_pages[0].device
    if device.type != "cuda":
        raise ValueError("CUDA pointer tables require CUDA page tensors")

    count = len(k_pages)
    k_cached = layer.get("_k_ptrs_cuda")
    v_cached = layer.get("_v_ptrs_cuda")
    cached_count = layer.get("_ptr_page_count")

    cache_ok = (
        cached_count == count
        and isinstance(k_cached, torch.Tensor)
        and isinstance(v_cached, torch.Tensor)
        and k_cached.dtype == torch.int64
        and v_cached.dtype == torch.int64
        and k_cached.device == device
        and v_cached.device == device
        and k_cached.ndim == 1
        and v_cached.ndim == 1
        and k_cached.numel() == count
        and v_cached.numel() == count
    )
    if cache_ok:
        _POINTER_TABLE_STATS["hits"] += 1
        return k_cached, v_cached

    # Build only when first used or when a new physical page has appeared.
    k_ptrs = torch.tensor(
        [int(page.data_ptr()) for page in k_pages],
        dtype=torch.int64,
        device=device,
    )
    v_ptrs = torch.tensor(
        [int(page.data_ptr()) for page in v_pages],
        dtype=torch.int64,
        device=device,
    )
    layer["_k_ptrs_cuda"] = k_ptrs
    layer["_v_ptrs_cuda"] = v_ptrs
    layer["_ptr_page_count"] = count
    _POINTER_TABLE_STATS["builds"] += 1
    return k_ptrs, v_ptrs


def trim_layer(layer: dict, new_len: int) -> dict:
    """Return a non-mutating logical prefix with copy-on-write tail storage."""
    check_consistent(layer)
    cur = logical_length(layer)
    if not isinstance(new_len, int) or isinstance(new_len, bool):
        raise ValueError("new_len must be an int")
    if new_len <= 0 or new_len > cur:
        raise ValueError(f"new_len {new_len} out of range (current {cur})")

    page_size = layer["page_size"]
    keep_pages = math.ceil(new_len / page_size)
    out = create_layer_cache(layer["fmt"], page_size)
    out["length"] = new_len
    keys = ["k_pages", "v_pages"]
    if layer["fmt"] == "v_fp8":
        keys.append("v_scale_pages")
    partial_tail = new_len % page_size != 0
    for key in keys:
        retained = list(layer[key][:keep_pages])
        if partial_tail:
            # This page may be shared with the source and will be written by a
            # later append. Clone the whole fixed-capacity page before that can
            # happen; completed retained pages remain safely shared.
            retained[-1] = retained[-1].clone()
        out[key] = retained
    # `out` intentionally starts without any pointer-table metadata.  A partial
    # tail may have been cloned, so inheriting source addresses would be wrong.
    check_consistent(out)
    return out


truncate_layer = trim_layer


def layer_device(layer: dict) -> torch.device:
    check_consistent(layer)
    if not layer["k_pages"]:
        raise ValueError("empty paged_v0 layer has no device")
    return layer["k_pages"][0].device


def _check_pages(layer: dict, key: str, expected_count: int,
                 page_size: int) -> tuple[tuple[int, ...] | None, torch.dtype | None,
                                          torch.device | None]:
    pages = layer.get(key)
    if not isinstance(pages, list):
        raise RuntimeError(f"paged_v0 {key} must be a list")
    if len(pages) != expected_count:
        raise RuntimeError(
            f"paged_v0 {key} page count {len(pages)} != expected {expected_count}")
    shape = dtype = device = None
    for index, page in enumerate(pages):
        if not isinstance(page, torch.Tensor) or page.dim() < 3:
            raise RuntimeError(f"paged_v0 {key}[{index}] is not a valid tensor page")
        if int(page.shape[2]) != page_size:
            raise RuntimeError(
                f"paged_v0 {key}[{index}] token capacity {page.shape[2]} != {page_size}")
        if shape is None:
            shape, dtype, device = tuple(page.shape), page.dtype, page.device
        elif tuple(page.shape) != shape or page.dtype != dtype or page.device != device:
            raise RuntimeError(f"paged_v0 {key} pages differ in shape/dtype/device")
    return shape, dtype, device


def check_consistent(layer: dict) -> None:
    if not is_paged_layer(layer):
        raise RuntimeError("layer is not marked as paged_v0")
    fmt = layer.get("fmt")
    if fmt not in SUPPORTED_FORMATS:
        raise RuntimeError(f"invalid paged_v0 format {fmt!r}")
    try:
        page_size = _validate_page_size(layer.get("page_size"))
        length = logical_length(layer)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    expected_count = math.ceil(length / page_size) if length else 0
    k_shape, k_dtype, k_device = _check_pages(
        layer, "k_pages", expected_count, page_size)
    v_shape, v_dtype, v_device = _check_pages(
        layer, "v_pages", expected_count, page_size)
    if k_shape != v_shape or k_device != v_device:
        raise RuntimeError("paged_v0 K/V pages differ in shape or device")
    if fmt == "fp16" and "v_scale_pages" in layer:
        raise RuntimeError("paged_v0 base cache unexpectedly has v_scale_pages")
    if expected_count:
        if fmt == "fp16":
            if k_dtype != v_dtype or k_dtype not in (
                    torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise RuntimeError("paged_v0 base K/V dtype mismatch or unsupported dtype")
        else:
            s_shape, s_dtype, s_device = _check_pages(
                layer, "v_scale_pages", expected_count, page_size)
            if k_dtype != torch.bfloat16 or v_dtype != torch.float8_e4m3fn:
                raise RuntimeError("paged_v0 v_fp8 payload dtypes are malformed")
            if s_dtype != torch.float32 or s_device != v_device:
                raise RuntimeError("paged_v0 v_scale dtype/device is malformed")
            if s_shape is None or s_shape[:3] != v_shape[:3]:
                raise RuntimeError("paged_v0 V/v_scale token geometry is not aligned")
    elif fmt == "v_fp8" and layer.get("v_scale_pages") != []:
        raise RuntimeError("empty paged_v0 v_fp8 cache has malformed v_scale_pages")


__all__ = [
    "BACKEND", "DEFAULT_PAGE_SIZE", "SUPPORTED_FORMATS", "append_current",
    "check_consistent", "create_layer_cache", "cuda_pointer_tables",
    "is_paged_layer", "layer_device", "logical_length", "materialize_k",
    "materialize_layer", "payload_bytes", "pointer_table_stats",
    "reset_pointer_table_stats", "select_indices", "trim_layer", "truncate_layer",
]
