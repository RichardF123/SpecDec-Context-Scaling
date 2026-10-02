"""Final direct PagedAttention operator gate.

Covers:
- BF16 draft-like D64 and target-like D128
- q_len 1,2,3,4
- partial pages / page boundaries
- asymmetric target V-FP8 (K BF16, V FP8 + scale)
- direct page path vs materialized contiguous CUDA reference
"""
from __future__ import annotations

import torch

from . import paged_cache
from .model_self import dequantize_kv_fp8, quantize_kv_fp8, GROUP_V
from specdec.kernels import ops


def _append_fp16(layer, *, total, kv, d, dtype):
    done = 0
    while done < total:
        n = min(3, total - done)
        k = torch.randn(1, kv, n, d, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        paged_cache.append_current(layer, k, v)
        done += n


def _append_vfp8(layer, *, total, kv, d):
    done = 0
    while done < total:
        n = min(3, total - done)
        k = torch.randn(1, kv, n, d, device="cuda", dtype=torch.bfloat16)
        v16 = torch.randn_like(k)
        qv, scale = quantize_kv_fp8(v16, GROUP_V)
        paged_cache.append_current(layer, k, qv, scale)
        done += n


def _ref(q, layer):
    mat = paged_cache.materialize_layer(layer)
    if layer["fmt"] == "v_fp8":
        v = dequantize_kv_fp8(
            mat["v"], mat["v_scale"], torch.bfloat16)
    else:
        v = mat["v"]
    return ops.attention(q, mat["k"], v, backend="cuda")


def _case(*, label, fmt, h, kv, d, s, t, page):
    torch.manual_seed(10000 + d + s + t + page)
    dtype = torch.bfloat16
    q = torch.randn(1, h, s, d, device="cuda", dtype=dtype)
    layer = paged_cache.create_layer_cache(fmt, page_size=page)

    if fmt == "fp16":
        _append_fp16(layer, total=t, kv=kv, d=d, dtype=dtype)
    else:
        _append_vfp8(layer, total=t, kv=kv, d=d)

    ref = _ref(q, layer)
    got = ops.paged_attention_block(q, layer, backend="cuda")

    diff = (ref.float() - got.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    scale = max(1.0, float(ref.float().abs().max()))
    allowed = max(2e-2, 0.02 * scale)

    assert torch.isfinite(got).all().item()
    assert max_abs <= allowed, (
        f"{label}: max_abs={max_abs} > allowed={allowed}")

    print(
        f"[{label}] fmt={fmt:<6} S={s} T={t} page={page} D={d} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"allowed={allowed:.6g} -> PASS"
    )


def _vfp8_pointer_reuse():
    paged_cache.reset_pointer_table_stats()
    layer = paged_cache.create_layer_cache("v_fp8", page_size=4)
    _append_vfp8(layer, total=5, kv=2, d=128)

    a = paged_cache.cuda_pointer_tables_vfp8(layer)
    st1 = paged_cache.pointer_table_stats()
    b = paged_cache.cuda_pointer_tables_vfp8(layer)
    st2 = paged_cache.pointer_table_stats()

    assert st1["builds"] == 1
    assert st2["hits"] == 1
    assert all(x.data_ptr() == y.data_ptr() for x, y in zip(a, b))

    _append_vfp8(layer, total=1, kv=2, d=128)
    c = paged_cache.cuda_pointer_tables_vfp8(layer)
    st3 = paged_cache.pointer_table_stats()
    assert st3["hits"] == 2
    assert all(x.data_ptr() == y.data_ptr() for x, y in zip(a, c))

    _append_vfp8(layer, total=3, kv=2, d=128)
    d = paged_cache.cuda_pointer_tables_vfp8(layer)
    st4 = paged_cache.pointer_table_stats()
    assert st4["builds"] == 2
    assert st4["invalidations"] == 1
    assert d[2].numel() == len(layer["v_scale_pages"])

    print(
        "[vfp8-pointer-table] build/reuse/invalidate K/V/scale: PASS "
        f"{st4}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print("Final PagedAttention operator gate")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    for s in (1, 2, 3, 4):
        _case(
            label=f"bf16-draft-s{s}",
            fmt="fp16", h=14, kv=2, d=64,
            s=s, t=17 + s, page=4,
        )

    for s in (1, 2, 3, 4):
        _case(
            label=f"bf16-target-s{s}",
            fmt="fp16", h=12, kv=2, d=128,
            s=s, t=31 + s, page=16,
        )

    for s in (1, 2, 3, 4):
        _case(
            label=f"vfp8-target-s{s}",
            fmt="v_fp8", h=12, kv=2, d=128,
            s=s, t=31 + s, page=16,
        )

    _vfp8_pointer_reuse()

    layer = paged_cache.create_layer_cache("fp16", page_size=4)
    _append_fp16(layer, total=8, kv=2, d=64, dtype=torch.bfloat16)
    q = torch.randn(1, 14, 5, 64, device="cuda", dtype=torch.bfloat16)
    try:
        ops.paged_attention_block(q, layer)
        raise AssertionError("q_len=5 must be rejected")
    except ValueError:
        pass

    print("\npaged_attention_final_operator_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
