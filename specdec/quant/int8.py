"""int8.py — RTN int8 weight-only quantization for the OURS engine.

Quantizes Transformer Linear weights with symmetric per-output-channel
round-to-nearest (RTN). Activations and compute stay in floating point.

The original float weight is not retained after conversion.

Typical use:
    model.load_state_dict(...)
    n = quantize_model_int8(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Inference-only Linear with int8 weight storage and float dequantization."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()

        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Int8Linear":
        weight = linear.weight.detach()

        # Symmetric RTN, one scale per output channel (one row of W).
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = max_abs / 127.0

        # A completely-zero row has no meaningful scale; use 1 so 0 / scale stays 0.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        qweight = torch.round(weight / scale)
        qweight = torch.clamp(qweight, -127, 127).to(torch.int8)

        bias = None if linear.bias is None else linear.bias.detach().clone()

        return cls(
            qweight=qweight,
            scale=scale,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L4 is weight-storage quantization. The first eager implementation
        # dequantizes to the activation dtype before the existing float matmul.
        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def quantize_model_int8(model: nn.Module) -> int:
    """Replace the 7 Transformer Linear projections in every layer.

    Returns the number of replaced modules. For Qwen2.5-0.5B (24 layers),
    the expected count is 24 * 7 = 168.
    """

    replaced = 0

    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, Int8Linear.from_float(linear))
            replaced += 1

        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            setattr(layer.mlp, name, Int8Linear.from_float(linear))
            replaced += 1

    return replaced
