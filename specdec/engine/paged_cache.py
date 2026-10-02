"""Paged KV cache v0 storage.

This module deliberately stops at storage: callers append token-aligned K/V
payloads into fixed-size pages and materialize them back to the legacy
contiguous representation before attention.  It is not a page-aware attention
implementation.
"""
from __future__ import annotations

import math
from typing import Any

import torch


BACKEND = "paged_v0"
DEFAULT_PAGE_SIZE = 16
SUPPORTED_FORMATS = ("fp16", "v_fp8")


def is_paged_layer(layer: Any) -> bool:
    return isinstance(layer, dict) and layer.get("_cache_backend") == BACKEND


def _validate_page_size(page_size: int) -> int:
    if (not isinstance(page_size, int) or isinstance(page_size, bool)
            or page_size <= 0):
        raise ValueError(f"page_size must be a positive int, got {page_size!r}")
    return page_size


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
            # later append.  Clone the whole fixed-capacity page before that can
            # happen; completed retained pages remain safely shared.
            retained[-1] = retained[-1].clone()
        out[key] = retained
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
    "check_consistent", "create_layer_cache", "is_paged_layer", "layer_device",
    "logical_length", "materialize_layer", "trim_layer", "truncate_layer",
]
