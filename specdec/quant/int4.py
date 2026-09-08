"""int4.py — RTN int4 weight-only quantization for the OURS engine.

Uses symmetric group-wise round-to-nearest (RTN) quantization.
Two signed 4-bit values are packed into one uint8 so the storage reduction
is real, not merely an int8 tensor restricted to the int4 numerical range.

Activations and compute remain floating point; weights are unpacked and
dequantized on the eager forward path.

Typical use:
    model.load_state_dict(...)
    n = quantize_model_int4(model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int4Linear(nn.Module):
    """Inference-only Linear with packed int4 weight storage."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: torch.Tensor | None,
    ):
        super().__init__()

        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scale", scale)

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
    ) -> "Int4Linear":
        weight = linear.weight.detach()
        out_features, in_features = weight.shape

        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by "
                f"group_size={group_size}"
            )

        if group_size % 2 != 0:
            raise ValueError("group_size must be even for int4 packing")

        groups = in_features // group_size
        w = weight.reshape(out_features, groups, group_size)

        # Symmetric group-wise RTN: signed int4 range [-7, 7].
        max_abs = w.abs().amax(dim=2, keepdim=True)
        scale = max_abs / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q = torch.round(w / scale)
        q = torch.clamp(q, -7, 7).to(torch.int8)

        # Map [-7, 7] -> [1, 15], then pack two 4-bit values per uint8.
        uq = (q + 8).to(torch.uint8)
        low = uq[..., 0::2]
        high = uq[..., 1::2] << 4
        packed_weight = low | high

        bias = None if linear.bias is None else linear.bias.detach().clone()

        return cls(
            packed_weight=packed_weight,
            scale=scale,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bias=bias,
        )

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        low = self.packed_weight & 0x0F
        high = (self.packed_weight >> 4) & 0x0F

        # Interleave low/high nibbles back to the original group order.
        q = torch.stack((low, high), dim=-1)
        q = q.reshape(
            self.out_features,
            self.in_features // self.group_size,
            self.group_size,
        )

        q = q.to(torch.int16) - 8
        weight = q.to(dtype=dtype) * self.scale.to(dtype=dtype)

        return weight.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize(x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def quantize_model_int4(
    model: nn.Module,
    group_size: int = 64,
) -> int:
    """Replace the 7 Transformer Linear projections in every layer."""

    replaced = 0

    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear = getattr(layer.self_attn, name)
            setattr(
                layer.self_attn,
                name,
                Int4Linear.from_float(linear, group_size=group_size),
            )
            replaced += 1

        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            setattr(
                layer.mlp,
                name,
                Int4Linear.from_float(linear, group_size=group_size),
            )
            replaced += 1

    return replaced
