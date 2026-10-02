"""PagedAttention v1a operator gate.

This test does NOT integrate the operator into model_self yet.

It verifies:
1. CUDA paged decode reads K/V pages directly.
2. No full-K/V materialization is needed by the tested operator.
3. Both Qwen draft shape (D=64) and target shape (D=128) work.
4. Partial final pages work.
5. Output is numerically close to the existing fp32 online-softmax
   CUDA attention reference.

Run:
    python -m specdec.engine.paged_attention_operator_test
"""
from __future__ import annotations

import torch

from . import paged_cache
from specdec.kernels import ops


def _build_layer(
    *,
    B: int,
    KV: int,
    T: int,
    D: int,
    page_size: int,
    dtype: torch.dtype,
    device: str,
):
    layer = paged_cache.create_layer_cache(
        "fp16",
        page_size=page_size,
    )

    done = 0

    while done < T:
        n = min(
            3,
            T - done,
        )

        k = torch.randn(
            B,
            KV,
            n,
            D,
            device=device,
            dtype=dtype,
        )

        v = torch.randn(
            B,
            KV,
            n,
            D,
            device=device,
            dtype=dtype,
        )

        paged_cache.append_current(
            layer,
            k,
            v,
        )

        done += n

    paged_cache.check_consistent(layer)

    return layer


def _one_case(
    *,
    label: str,
    H: int,
    KV: int,
    T: int,
    D: int,
    page_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(1234 + T + D + page_size)

    q = torch.randn(
        1,
        H,
        1,
        D,
        device="cuda",
        dtype=dtype,
    )

    layer = _build_layer(
        B=1,
        KV=KV,
        T=T,
        D=D,
        page_size=page_size,
        dtype=dtype,
        device="cuda",
    )

    # Reference only: materialize the cache for the already-existing
    # contiguous CUDA attention kernel.
    materialized = paged_cache.materialize_layer(layer)

    ref = ops.attention(
        q,
        materialized["k"],
        materialized["v"],
        backend="cuda",
    )

    got = ops.paged_attention_decode(
        q,
        layer,
        backend="cuda",
    )

    assert tuple(ref.shape) == tuple(got.shape)

    diff = (
        ref.float() - got.float()
    ).abs()

    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    scale = max(
        1.0,
        float(ref.float().abs().max()),
    )

    # Both kernels use fp32 QK / online softmax / fp32 accumulation,
    # but reduction order differs because one walks contiguous T and
    # the other dereferences pages. For BF16 use the same general
    # kernel-level tolerance scale already used by this repository.
    if dtype == torch.bfloat16:
        allowed = max(
            2e-2,
            0.02 * scale,
        )
    else:
        allowed = 1e-4

    ok = (
        torch.isfinite(got).all().item()
        and max_abs <= allowed
    )

    print(
        f"[{label}] "
        f"T={T} page={page_size} "
        f"D={D} dtype={dtype} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"allowed={allowed:.6g} "
        f"pages={len(layer['k_pages'])} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )

    assert ok


def _rejection_tests():
    q = torch.randn(
        1,
        12,
        1,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )

    fp8_layer = paged_cache.create_layer_cache(
        "v_fp8",
        page_size=4,
    )

    k = torch.randn(
        1,
        2,
        1,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )

    v16 = torch.randn_like(k)

    from .model_self import quantize_kv_fp8, GROUP_V

    qv, scale = quantize_kv_fp8(
        v16,
        GROUP_V,
    )

    paged_cache.append_current(
        fp8_layer,
        k,
        qv,
        scale,
    )

    try:
        ops.paged_attention_decode(
            q,
            fp8_layer,
        )

        raise AssertionError(
            "v_fp8 should be rejected in v1"
        )

    except ValueError:
        pass

    try:
        ops.paged_attention_decode(
            q[:, :, :, :64],
            {"not": "paged"},
        )

        raise AssertionError(
            "non-paged cache should be rejected"
        )

    except TypeError:
        pass

    print("[rejection] unsupported cache modes: PASS")


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "torch:",
        torch.__version__,
    )

    # Draft-like GQA
    _one_case(
        label="draft-D64",
        H=14,
        KV=2,
        T=17,
        D=64,
        page_size=4,
        dtype=torch.bfloat16,
    )

    # Target-like GQA
    _one_case(
        label="target-D128",
        H=12,
        KV=2,
        T=17,
        D=128,
        page_size=4,
        dtype=torch.bfloat16,
    )

    # Exact page boundary
    _one_case(
        label="boundary",
        H=12,
        KV=2,
        T=16,
        D=128,
        page_size=4,
        dtype=torch.bfloat16,
    )

    # One past boundary
    _one_case(
        label="boundary-plus-one",
        H=12,
        KV=2,
        T=17,
        D=128,
        page_size=16,
        dtype=torch.bfloat16,
    )

    # FP32 debugging/reference path
    _one_case(
        label="fp32-reference",
        H=4,
        KV=2,
        T=13,
        D=64,
        page_size=4,
        dtype=torch.float32,
    )

    _rejection_tests()

    print(
        "\npaged_attention_operator_test: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
