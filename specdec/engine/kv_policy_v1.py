"""Structural gate for the experimental GQA-head x KV-quantization probe.

Checks only implementation invariants; it does not make a quality claim.
Run on CUDA before the pilot:
    python -m specdec.engine.kv_policy_v1
"""
from __future__ import annotations

import torch

from .model_self import (
    GROUP_K_INT8,
    GROUP_V,
    INT8_MAX,
    QwenLM,
    dequantize_kv_int8,
    quantize_kv_int8,
    reduce_kv_heads,
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ok = True

    # Helper: deterministic INT8, correct scale shape, zeros exact, max maps to 127.
    x = torch.randn(1, 2, 7, 128, device=device, dtype=dtype)
    q1, s1 = quantize_kv_int8(x, GROUP_K_INT8)
    q2, s2 = quantize_kv_int8(x, GROUP_K_INT8)
    hat = dequantize_kv_int8(q1, s1, dtype)
    helper_ok = (
        q1.dtype == torch.int8
        and s1.dtype == torch.float32
        and tuple(s1.shape) == (1, 2, 7, 1)
        and torch.equal(q1, q2)
        and torch.equal(s1, s2)
        and torch.isfinite(hat).all().item()
        and float(q1.abs().amax()) == INT8_MAX
    )
    z = torch.zeros_like(x)
    qz, sz = quantize_kv_int8(z, GROUP_K_INT8)
    zero_ok = bool((dequantize_kv_int8(qz, sz, dtype) == 0).all().item())
    print(f"[1] INT8 helper structure/determinism/max->127: {helper_ok}")
    print(f"[2] INT8 all-zero exact: {zero_ok}")
    ok &= bool(helper_ok and zero_ok)

    # Head reduction is a deterministic mean, separately configurable for K/V.
    r = reduce_kv_heads(x, 1)
    reduce_ok = tuple(r.shape) == (1, 1, 7, 128) and torch.equal(
        r, x.reshape(1, 1, 2, 7, 128).mean(dim=2)
    )
    print(f"[3] 2->1 head mean reduction exact: {reduce_ok}")
    ok &= bool(reduce_ok)

    # Tiny end-to-end cache build + append. head_dim=128 matches the frozen groups.
    model = QwenLM(
        vocab_size=64, hidden=512, heads=4, kv_heads=2, head_dim=128,
        intermediate=256, num_layers=1, attn_bias=True,
    ).to(device, dtype).eval()
    ids = torch.randint(0, 64, (1, 5), device=device)
    model.set_kv_format("k_int8_v_fp8")
    model.set_kv_head_policy(k_heads=1, v_heads=2)
    with torch.inference_mode():
        _, cache = model._run(model.embed_tokens(ids[:, :4]), None, need_logits=False)
        _, cache = model._run(model.embed_tokens(ids[:, 4:]), cache, need_logits=False)
    c = cache[0]
    engine_ok = (
        c.get("fmt") == "k_int8_v_fp8"
        and c["k"].dtype == torch.int8
        and c["v"].dtype == torch.float8_e4m3fn
        and tuple(c["k"].shape) == (1, 1, 5, 128)
        and tuple(c["v"].shape) == (1, 2, 5, 128)
        and tuple(c["k_scale"].shape) == (1, 1, 5, 1)
        and tuple(c["v_scale"].shape) == (1, 2, 5, 1)
    )
    print(f"[4] engine mixed head-count/cache append structure: {engine_ok}")
    ok &= bool(engine_ok)

    print("\nKV policy V1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
