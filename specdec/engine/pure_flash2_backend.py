"""Official FlashAttention-2 backend for SpecDec experiments.

This module intentionally does NOT use paged KV.

It installs a temporary runtime backend:
    preallocated contiguous BF16 KV
    + official flash_attn_with_kvcache
    + logical-length rollback

No repository core file is overwritten.  The monkeypatch is active only inside
pure_flash2_context().
"""
from __future__ import annotations

import contextlib
from typing import Iterable

import torch

from . import model_self, spec_cache


FLASH_CACHE_MARKER = "pure_flash2_prealloc"


def is_pure_flash_cache(layer) -> bool:
    return (
        isinstance(layer, dict)
        and layer.get("_cache_backend") == FLASH_CACHE_MARKER
    )


def _make_layer_cache(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    max_seqlen: int,
) -> dict:
    # incoming model layout: [B, KV, S, D]
    b, kv, _, d = map(int, k_new.shape)
    if max_seqlen <= 0:
        raise ValueError("max_seqlen must be positive")
    return {
        "_cache_backend": FLASH_CACHE_MARKER,
        "fmt": "fp16",
        "length": 0,
        "capacity": int(max_seqlen),
        # official FA2 cache layout: [B, T, KV, D]
        "k_cache": torch.empty(
            b, max_seqlen, kv, d,
            device=k_new.device, dtype=k_new.dtype),
        "v_cache": torch.empty(
            b, max_seqlen, kv, d,
            device=v_new.device, dtype=v_new.dtype),
    }


def _flash_forward(self, x, cos, sin, cache=None):
    """SelfAttention.forward replacement for enabled modules."""
    if not getattr(self, "_pure_flash2_enabled", False):
        return self._pure_flash2_original_forward(x, cos, sin, cache)

    if self.kv_format != "fp16":
        raise ValueError(
            "Pure official FlashAttention-2 backend intentionally supports "
            "BF16/FP16 contiguous KV only; use base_bf16 for this experiment.")
    if self.h2o_capture_scores:
        raise ValueError(
            "Pure FlashAttention-2 experiment does not combine H2O.")
    if not x.is_cuda:
        raise ValueError("Pure FlashAttention-2 backend requires CUDA.")

    try:
        from flash_attn import flash_attn_with_kvcache
    except Exception as exc:
        raise RuntimeError(
            "official flash-attn is not importable; install a compatible "
            "optional flash-attn package for this CUDA/PyTorch environment"
        ) from exc

    b, s, _ = x.shape
    q = self.q_proj(x).view(
        b, s, self.heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(
        b, s, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(
        b, s, self.kv_heads, self.head_dim).transpose(1, 2)

    q = model_self.apply_rope(q, cos, sin)
    k = model_self.apply_rope(k, cos, sin)

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"FA2 requires fp16/bf16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q/k/v dtype mismatch: {q.dtype}/{k.dtype}/{v.dtype}")

    max_seqlen = int(getattr(self, "_pure_flash2_max_seqlen", 0))
    if cache is None:
        cache = _make_layer_cache(k, v, max_seqlen)
    elif not is_pure_flash_cache(cache):
        raise TypeError(
            "Pure FlashAttention-2 received a non-Flash contiguous cache.")

    old_len = int(cache["length"])
    new_len = old_len + int(s)
    if new_len > int(cache["capacity"]):
        raise RuntimeError(
            f"pure FA2 KV capacity exceeded: {new_len} > "
            f"{cache['capacity']}")

    # FA2 layout [B, S, H, D] / [B, S, KV, D].
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    cache_seqlens = torch.full(
        (b,), old_len, dtype=torch.int32, device=q.device)

    out = flash_attn_with_kvcache(
        q_fa,
        cache["k_cache"],
        cache["v_cache"],
        k=k_fa,
        v=v_fa,
        cache_seqlens=cache_seqlens,
        softmax_scale=self.head_dim ** -0.5,
        causal=True,
    )

    cache["length"] = new_len
    out = out.reshape(b, s, self.heads * self.head_dim)
    return self.o_proj(out), cache


def _flash_cache_length(cache):
    if cache is None:
        return 0
    if not cache:
        raise RuntimeError("cache must contain at least one layer")
    if is_pure_flash_cache(cache[0]):
        return int(cache[0]["length"])
    return _ORIG_SPEC_CACHE_LENGTH(cache)


def _flash_cache_device(cache):
    if cache is None or not cache:
        raise ValueError("cache has no device")
    if is_pure_flash_cache(cache[0]):
        return cache[0]["k_cache"].device
    return _ORIG_SPEC_CACHE_DEVICE(cache)


def _flash_trim_cache(cache, new_len: int):
    if cache is None:
        if new_len == 0:
            return None
        raise ValueError("cannot trim None cache to positive length")
    if not cache or not is_pure_flash_cache(cache[0]):
        return _ORIG_SPEC_TRIM(cache, new_len)

    cur = _flash_cache_length(cache)
    if (
        not isinstance(new_len, int)
        or isinstance(new_len, bool)
        or new_len <= 0
        or new_len > cur
    ):
        raise ValueError(f"invalid new_len {new_len} for current {cur}")

    # Important: no K/V copy.  Keep the same preallocated buffers and only
    # move the logical cursor backwards.  The next FA2 append overwrites the
    # rejected speculative tail.
    out = []
    for layer in cache:
        if not is_pure_flash_cache(layer):
            raise RuntimeError("mixed cache backends are invalid")
        nl = dict(layer)
        nl["length"] = int(new_len)
        out.append(nl)
    return out


def _flash_check_consistent(cache):
    if cache is None:
        return
    if not cache or not is_pure_flash_cache(cache[0]):
        return _ORIG_SPEC_CHECK(cache)

    lens = []
    for i, layer in enumerate(cache):
        if not is_pure_flash_cache(layer):
            raise RuntimeError(f"mixed backend at layer {i}")
        n = int(layer["length"])
        cap = int(layer["capacity"])
        kc, vc = layer["k_cache"], layer["v_cache"]
        if n < 0 or n > cap:
            raise RuntimeError(
                f"invalid pure FA2 logical length {n}/{cap}")
        if kc.shape != vc.shape:
            raise RuntimeError(
                f"K/V cache shape mismatch at layer {i}")
        if int(kc.shape[1]) != cap:
            raise RuntimeError(
                f"capacity mismatch at layer {i}")
        if kc.device != vc.device or kc.dtype != vc.dtype:
            raise RuntimeError(
                f"K/V device or dtype mismatch at layer {i}")
        lens.append(n)
    if len(set(lens)) != 1:
        raise RuntimeError(
            f"cross-layer pure FA2 length mismatch: {lens}")


_ORIG_SPEC_CACHE_LENGTH = spec_cache.cache_length
_ORIG_SPEC_CACHE_DEVICE = spec_cache.cache_device
_ORIG_SPEC_TRIM = spec_cache.trim_cache
_ORIG_SPEC_CHECK = spec_cache.check_consistent


@contextlib.contextmanager
def pure_flash2_context(*models, max_seqlen: int):
    """Temporarily route enabled model attention through official FA2.

    The repository source is not edited.  On exit the monkeypatches and each
    model/layer execution configuration are restored to their exact entry
    state, including KV format, cache backend, page size, and paged-attention
    mode.
    """
    if max_seqlen <= 0:
        raise ValueError("max_seqlen must be positive")

    original_forward = model_self.SelfAttention.forward
    original_spec_cache = (
        spec_cache.cache_length,
        spec_cache.cache_device,
        spec_cache.trim_cache,
        spec_cache.check_consistent,
    )

    # Snapshot both model-level and layer-level execution state before forcing
    # the temporary contiguous BF16/FP16 FA2 configuration.  Layer snapshots
    # make restoration exact even if a caller had manually customized a layer.
    model_states = []
    seen_models = set()

    try:
        for model in models:
            marker = id(model)
            if marker in seen_models:
                continue
            seen_models.add(marker)

            layer_states = []
            for layer in model.layers:
                a = layer.self_attn
                runtime_attrs = {}
                for name in (
                    "_pure_flash2_enabled",
                    "_pure_flash2_max_seqlen",
                    "_pure_flash2_original_forward",
                ):
                    if hasattr(a, name):
                        runtime_attrs[name] = (True, getattr(a, name))
                    else:
                        runtime_attrs[name] = (False, None)

                layer_states.append({
                    "attn": a,
                    "kv_format": a.kv_format,
                    "cache_backend": a.cache_backend,
                    "cache_page_size": a.cache_page_size,
                    "paged_attention_mode": a.paged_attention_mode,
                    "runtime_attrs": runtime_attrs,
                })

            model_states.append({
                "model": model,
                "kv_format": model.kv_format,
                "cache_backend": model.cache_backend,
                "cache_page_size": model.cache_page_size,
                "paged_attention_mode": model.paged_attention_mode,
                "layers": layer_states,
            })

            model.set_kv_format("fp16")
            model.set_cache_backend("contiguous")
            model.set_paged_attention_mode("off")

            for state in layer_states:
                a = state["attn"]
                a._pure_flash2_enabled = True
                a._pure_flash2_max_seqlen = int(max_seqlen)
                a._pure_flash2_original_forward = original_forward.__get__(
                    a, type(a))

        model_self.SelfAttention.forward = _flash_forward
        spec_cache.cache_length = _flash_cache_length
        spec_cache.cache_device = _flash_cache_device
        spec_cache.trim_cache = _flash_trim_cache
        spec_cache.check_consistent = _flash_check_consistent

        yield
    finally:
        # Restore module/class monkeypatches to the exact functions present on
        # entry, then restore every model and layer field changed above.
        model_self.SelfAttention.forward = original_forward
        (
            spec_cache.cache_length,
            spec_cache.cache_device,
            spec_cache.trim_cache,
            spec_cache.check_consistent,
        ) = original_spec_cache

        for state in reversed(model_states):
            model = state["model"]
            model.kv_format = state["kv_format"]
            model.cache_backend = state["cache_backend"]
            model.cache_page_size = state["cache_page_size"]
            model.paged_attention_mode = state["paged_attention_mode"]

            for layer_state in state["layers"]:
                a = layer_state["attn"]
                a.kv_format = layer_state["kv_format"]
                a.cache_backend = layer_state["cache_backend"]
                a.cache_page_size = layer_state["cache_page_size"]
                a.paged_attention_mode = layer_state["paged_attention_mode"]

                for name, (existed, value) in layer_state[
                    "runtime_attrs"
                ].items():
                    if existed:
                        setattr(a, name, value)
                    else:
                        try:
                            delattr(a, name)
                        except AttributeError:
                            pass


def pure_flash_cache_payload_bytes(cache) -> int:
    if cache is None:
        return 0
    total = 0
    seen = set()
    for layer in cache:
        if not is_pure_flash_cache(layer):
            continue
        for key in ("k_cache", "v_cache"):
            t = layer[key]
            # shallow rollback cache objects can share the same backing tensor.
            ptr = (t.device.type, int(t.data_ptr()))
            if ptr not in seen:
                seen.add(ptr)
                total += t.numel() * t.element_size()
    return int(total)
