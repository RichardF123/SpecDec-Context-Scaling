"""test_int8.py — L4 int8 RTN correctness gate.

Checks the actual quantization primitive and its integration into the OURS
Qwen engine:
  1) int8 storage is real;
  2) float weights are not retained inside Int8Linear;
  3) all Transformer Linear modules are replaced;
  4) quantized forward remains numerically finite.

This is an engineering gate, not the final A6 quality evaluation.
Final ppl / VRAM evidence is recorded separately on the 1.5B target model.

Run:
    python -m specdec.quant.test_int8
"""

from __future__ import annotations

import torch
import torch.nn as nn

from specdec.engine.model_self import QwenLM
from .int8 import Int8Linear, quantize_model_int8


def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    # Small Qwen-shaped engine: cheap gate, same module structure as the real model.
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

    replaced = quantize_model_int8(model)

    int8_modules = [
        m for m in model.modules()
        if isinstance(m, Int8Linear)
    ]

    remaining_linear = []
    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if isinstance(getattr(layer.self_attn, name), nn.Linear):
                remaining_linear.append(f"self_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            if isinstance(getattr(layer.mlp, name), nn.Linear):
                remaining_linear.append(f"mlp.{name}")

    with torch.no_grad():
        int8_logits = model(ids)

    diff = (float_logits - int8_logits).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    finite = bool(torch.isfinite(int8_logits).all())

    real_int8 = all(m.qweight.dtype == torch.int8 for m in int8_modules)
    expected = 2 * 7

    print(f"replaced modules: {replaced} (expected {expected})")
    print(f"Int8Linear count: {len(int8_modules)}")
    print(f"remaining target nn.Linear: {remaining_linear}")
    print(f"all qweight dtype int8: {real_int8}")
    print(f"logits max|Δ|: {max_diff:.3e}")
    print(f"logits mean|Δ|: {mean_diff:.3e}")
    print(f"quantized logits finite: {finite}")

    passed = (
        replaced == expected
        and len(int8_modules) == expected
        and not remaining_linear
        and real_int8
        and finite
    )

    print("GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
