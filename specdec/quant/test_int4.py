"""test_int4.py — L4 packed-int4 RTN correctness gate.

Checks:
  1) two 4-bit weights are really packed into each uint8;
  2) all target Transformer Linear modules are replaced;
  3) the unpack/dequant forward path is finite;
  4) the quantized engine still executes end-to-end.

This is an engineering gate. Final A6 quality is decided by ppl / held-out
evaluation on the 1.5B target model.

Run:
    python -m specdec.quant.test_int4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from specdec.engine.model_self import QwenLM
from .int4 import Int4Linear, quantize_model_int4


def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    model = QwenLM(
        vocab_size=5000,
        hidden=64,
        heads=8,
        kv_heads=2,
        head_dim=8,
        intermediate=256,
        num_layers=2,
        rope_theta=1_000_000.0,
    ).eval()

    ids = torch.randint(0, 5000, (1, 10))

    with torch.no_grad():
        float_logits = model(ids)

    replaced = quantize_model_int4(model, group_size=64)

    int4_modules = [
        m for m in model.modules()
        if isinstance(m, Int4Linear)
    ]

    remaining_linear = []
    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if isinstance(getattr(layer.self_attn, name), nn.Linear):
                remaining_linear.append(f"self_attn.{name}")

        for name in ("gate_proj", "up_proj", "down_proj"):
            if isinstance(getattr(layer.mlp, name), nn.Linear):
                remaining_linear.append(f"mlp.{name}")

    packed_uint8 = all(
        m.packed_weight.dtype == torch.uint8
        for m in int4_modules
    )

    packing_correct = all(
        m.packed_weight.numel() * 2
        == m.out_features * m.in_features
        for m in int4_modules
    )

    with torch.no_grad():
        int4_logits = model(ids)

    diff = (float_logits - int4_logits).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    finite = bool(torch.isfinite(int4_logits).all())

    expected = 2 * 7

    print(f"replaced modules: {replaced} (expected {expected})")
    print(f"Int4Linear count: {len(int4_modules)}")
    print(f"remaining target nn.Linear: {remaining_linear}")
    print(f"all packed_weight dtype uint8: {packed_uint8}")
    print(f"two int4 values per byte: {packing_correct}")
    print(f"logits max|Δ|: {max_diff:.3e}")
    print(f"logits mean|Δ|: {mean_diff:.3e}")
    print(f"quantized logits finite: {finite}")

    passed = (
        replaced == expected
        and len(int4_modules) == expected
        and not remaining_linear
        and packed_uint8
        and packing_correct
        and finite
    )

    print("GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
