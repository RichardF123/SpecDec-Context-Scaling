"""Structural gate for second-round native-head KV quantization candidates."""
from __future__ import annotations

import torch

from .model_self import (
    GROUP_V_INT4,
    K_CHANNEL_BLOCK,
    QwenLM,
    append_kv_int8_channel_cache,
    dequantize_kv_int4_packed,
    dequantize_kv_int8_channel_blocks,
    quantize_kv_int4_packed,
    quantize_kv_int8_channel_blocks,
    read_kv_int8_channel_cache,
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ok = True

    # Packed INT4: deterministic, two values/byte, scale per group, zero exact.
    x = torch.randn(1, 2, 7, 128, device=device, dtype=dtype)
    q1, s1 = quantize_kv_int4_packed(x, GROUP_V_INT4)
    q2, s2 = quantize_kv_int4_packed(x, GROUP_V_INT4)
    hat = dequantize_kv_int4_packed(q1, s1, dtype)
    int4_ok = (
        q1.dtype == torch.uint8
        and tuple(q1.shape) == (1, 2, 7, 64)
        and s1.dtype == torch.float32
        and tuple(s1.shape) == (1, 2, 7, 4)
        and torch.equal(q1, q2)
        and torch.equal(s1, s2)
        and bool(torch.isfinite(hat).all().item())
    )
    zeros = torch.zeros_like(x)
    qz, sz = quantize_kv_int4_packed(zeros, GROUP_V_INT4)
    zero_ok = bool(
        (dequantize_kv_int4_packed(qz, sz, dtype) == 0).all().item()
    )
    print(f"[1] packed V-INT4 shape/determinism/finite: {int4_ok}")
    print(f"[2] packed V-INT4 all-zero exact: {zero_ok}")
    ok &= bool(int4_ok and zero_ok)

    # Per-channel K: scale is shared across each 32-token block/channel.
    k = torch.randn(1, 2, 64, 128, device=device, dtype=dtype)
    qk, sk = quantize_kv_int8_channel_blocks(k, K_CHANNEL_BLOCK)
    khat = dequantize_kv_int8_channel_blocks(qk, sk, dtype, K_CHANNEL_BLOCK)
    pc_ok = (
        qk.dtype == torch.int8
        and tuple(qk.shape) == tuple(k.shape)
        and sk.dtype == torch.float32
        and tuple(sk.shape) == (1, 2, 2, 128)
        and bool(torch.isfinite(khat).all().item())
    )
    print(f"[3] blockwise per-channel K-INT8 structure: {pc_ok}")
    ok &= bool(pc_ok)

    # Residual behavior must remain < block and preserve total sequence length.
    first = k[:, :, :35]
    second = k[:, :, 35:]
    qc, sc, residual = append_kv_int8_channel_cache(None, None, None, first)
    first_q = qc.clone()
    qc, sc, residual = append_kv_int8_channel_cache(qc, sc, residual, second)
    restored = read_kv_int8_channel_cache(qc, sc, residual, dtype)
    residual_ok = (
        tuple(first_q.shape) == (1, 2, 32, 128)
        and tuple(qc.shape) == (1, 2, 64, 128)
        and tuple(sc.shape) == (1, 2, 2, 128)
        and residual.shape[2] == 0
        and restored.shape[2] == 64
        and torch.equal(qc[:, :, :32], first_q)
    )
    print(f"[4] completed-block append + BF16 residual semantics: {residual_ok}")
    ok &= bool(residual_ok)

    # End-to-end build and append for both new formats.
    model = QwenLM(
        vocab_size=64, hidden=512, heads=4, kv_heads=2, head_dim=128,
        intermediate=256, num_layers=1, attn_bias=True,
    ).to(device, dtype).eval()
    ids = torch.randint(0, 64, (1, 36), device=device)
    engine_ok = True
    with torch.inference_mode():
        model.set_kv_format("v_int4")
        _, cache_v = model._run(model.embed_tokens(ids[:, :35]), None, False)
        _, cache_v = model._run(model.embed_tokens(ids[:, 35:]), cache_v, False)
        cv = cache_v[0]
        engine_ok &= (
            cv["v"].dtype == torch.uint8
            and tuple(cv["v"].shape) == (1, 2, 36, 64)
            and tuple(cv["v_scale"].shape) == (1, 2, 36, 4)
            and cv["k"].dtype == dtype
            and cv["k"].shape[2] == 36
        )

        model.set_kv_format("k_pc_int8_v_fp8")
        _, cache_k = model._run(model.embed_tokens(ids[:, :35]), None, False)
        _, cache_k = model._run(model.embed_tokens(ids[:, 35:]), cache_k, False)
        ck = cache_k[0]
        engine_ok &= (
            ck["k"].dtype == torch.int8
            and tuple(ck["k"].shape) == (1, 2, 32, 128)
            and tuple(ck["k_scale"].shape) == (1, 2, 1, 128)
            and tuple(ck["k_residual"].shape) == (1, 2, 4, 128)
            and ck["v"].dtype == torch.float8_e4m3fn
            and ck["v"].shape[2] == 36
        )
    print(f"[5] engine build/append for both candidates: {engine_ok}")
    ok &= bool(engine_ok)

    print("\nKV policy V2:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
