"""Structural gate for recent-BF16 residual K cache."""
from __future__ import annotations

import torch

from .model_self import (
    K_CHANNEL_BLOCK,
    QwenLM,
    append_kv_int8_channel_cache,
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

    # 160 total tokens with residual target 128 -> 32 quantized + 128 BF16.
    x = torch.randn(1, 2, 160, 128, device=device, dtype=dtype)
    q, scale, residual = append_kv_int8_channel_cache(
        None, None, None, x[:, :, :35], K_CHANNEL_BLOCK, 128,
    )
    first_ok = q.shape[2] == 0 and residual.shape[2] == 35
    q, scale, residual = append_kv_int8_channel_cache(
        q, scale, residual, x[:, :, 35:], K_CHANNEL_BLOCK, 128,
    )
    restored = read_kv_int8_channel_cache(q, scale, residual, dtype)
    append_ok = (
        tuple(q.shape) == (1, 2, 32, 128)
        and tuple(scale.shape) == (1, 2, 1, 128)
        and tuple(residual.shape) == (1, 2, 128, 128)
        and restored.shape[2] == 160
    )
    print(f"[1] below-residual target stays BF16: {first_ok}")
    print(f"[2] completed old block + exact 128-token residual: {append_ok}")
    ok &= bool(first_ok and append_ok)

    # End-to-end append also checks RoPE start position counts q + residual.
    model = QwenLM(
        vocab_size=64, hidden=512, heads=4, kv_heads=2, head_dim=128,
        intermediate=256, num_layers=1, attn_bias=True,
    ).to(device, dtype).eval()
    model.set_kv_format("k_pc_int8_v_fp8")
    model.set_k_residual_length(128)
    ids = torch.randint(0, 64, (1, 160), device=device)
    with torch.inference_mode():
        _, cache = model._run(model.embed_tokens(ids[:, :159]), None, False)
        _, cache = model._run(model.embed_tokens(ids[:, 159:]), cache, False)
    layer = cache[0]
    engine_ok = (
        layer["k"].shape[2] == 32
        and layer["k_residual"].shape[2] == 128
        and layer["k_residual_length"] == 128
        and layer["v"].shape[2] == 160
    )
    print(f"[3] engine append/start-position with residual cache: {engine_ok}")
    ok &= bool(engine_ok)

    print("\nKV residual V1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
