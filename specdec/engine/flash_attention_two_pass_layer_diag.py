"""Locate how two-pass FlashAttention error grows through Qwen layers.

For each layer this diagnostic records two complementary errors:

* local attention error: eager and Triton attention receive the same eager
  normalized hidden state, isolating the kernel at that layer;
* cumulative hidden error: eager and Triton model streams evolve separately,
  showing whether many small local differences accumulate smoothly or a
  particular layer introduces a discontinuity.

The production engine is unchanged and no decode benchmark is performed.
"""
from __future__ import annotations

import gc
import json
import os

import torch

from . import flash_attention_engine_trial as trial
from . import model_self as ms
from .flash_attention_two_pass_triton import (
    KERNEL_REVISION,
    flash_attention_bf16_two_pass,
)
from .kv_policy_pilot_v2 import MODEL, load_model


CONTEXT = 2048
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_two_pass_layer_diag.json",
)


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference32 = reference.to(torch.float32)
    candidate32 = candidate.to(torch.float32)
    difference = candidate32 - reference32
    denominator = torch.linalg.vector_norm(reference32).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference) / denominator
        ).item(),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    model.set_kv_head_policy(2, 2)
    model.set_kv_format("fp16")
    trial.flash_attention_bf16 = flash_attention_bf16_two_pass
    trial.install_trial_backend(model)

    token_ids = trial.make_token_ids(tokenizer, CONTEXT)
    eager_x = model.embed_tokens(token_ids)
    flash_x = eager_x.clone()
    cos, sin = ms.Build_Rope_cache(
        model.head_dim,
        seq_len=CONTEXT,
        theta=model.rope_theta,
        device=eager_x.device,
        dtype=eager_x.dtype,
        start_pos=0,
    )

    rows = []
    previous_cumulative = 0.0
    print(
        "two-pass layer-growth diagnostic: "
        f"gpu={torch.cuda.get_device_name(0)} model=1.5B ctx={CONTEXT} "
        f"revision={KERNEL_REVISION}",
        flush=True,
    )
    print(
        "local=identical layer input; cumulative=separately evolving streams\n",
        flush=True,
    )

    with torch.inference_mode():
        for index, layer in enumerate(model.layers):
            eager_input = eager_x
            flash_input = flash_x
            eager_norm = layer.input_layernorm(eager_input)
            flash_norm = layer.input_layernorm(flash_input)

            eager_attn, _ = layer.self_attn._flash_trial_original_forward(
                eager_norm, cos, sin, None,
            )
            # Same input as eager: isolates this layer's attention kernel.
            local_flash_attn, _ = trial._flash_trial_forward(
                layer.self_attn, eager_norm, cos, sin, None,
            )
            local_error = error(eager_attn, local_flash_attn)

            # Separately evolving FlashAttention stream.
            cumulative_flash_attn, _ = trial._flash_trial_forward(
                layer.self_attn, flash_norm, cos, sin, None,
            )
            eager_x = eager_input + eager_attn
            eager_x = eager_x + layer.mlp(layer.post_attention_layernorm(eager_x))
            flash_x = flash_input + cumulative_flash_attn
            flash_x = flash_x + layer.mlp(layer.post_attention_layernorm(flash_x))
            cumulative_error = error(eager_x, flash_x)
            current_cumulative = cumulative_error["relative_l2"]
            absolute_growth = current_cumulative - previous_cumulative
            growth_ratio = (
                current_cumulative / previous_cumulative
                if previous_cumulative > 0.0 else None
            )
            previous_cumulative = current_cumulative

            row = {
                "layer": index,
                "local_attention": local_error,
                "cumulative_hidden": cumulative_error,
                "cumulative_relative_l2_absolute_growth": absolute_growth,
                "cumulative_relative_l2_growth_ratio": growth_ratio,
            }
            rows.append(row)
            ratio_text = "n/a" if growth_ratio is None else f"{growth_ratio:.2f}x"
            print(
                f"layer={index:02d} "
                f"local_rel={local_error['relative_l2']:.7f} "
                f"cumulative_rel={current_cumulative:.7f} "
                f"growth={absolute_growth:+.7f} ratio={ratio_text}",
                flush=True,
            )
            del (
                eager_norm,
                flash_norm,
                eager_attn,
                local_flash_attn,
                cumulative_flash_attn,
            )

        eager_final = model.norm(eager_x)
        flash_final = model.norm(flash_x)
        hidden_final_error = error(eager_final, flash_final)
        eager_logits = eager_final[:, -1] @ model.embed_tokens.weight.T
        flash_logits = flash_final[:, -1] @ model.embed_tokens.weight.T
        logit_error = error(eager_logits, flash_logits)
        logit_error["argmax_same"] = bool(
            eager_logits.argmax().item() == flash_logits.argmax().item()
        )

    local_values = [row["local_attention"]["relative_l2"] for row in rows]
    growth_values = [
        row["cumulative_relative_l2_absolute_growth"] for row in rows
    ]
    # Layer 0 establishes the initial nonzero error baseline and therefore
    # cannot be classified as a mid-network discontinuity.
    diagnostic_layers = range(1, len(rows))
    largest_growth_layer = max(diagnostic_layers, key=growth_values.__getitem__)
    median_positive_growth = torch.tensor(
        [value for value in growth_values[1:] if value > 0.0],
        dtype=torch.float64,
    ).median().item()
    largest_growth = growth_values[largest_growth_layer]
    # A single layer is suspicious only if its absolute error increment is far
    # larger than the typical positive increment.  This is descriptive, not a
    # production correctness gate.
    sudden_jump = (
        median_positive_growth > 0.0
        and largest_growth > 8.0 * median_positive_growth
    )
    interpretation = (
        "SUSPECT_SINGLE_LAYER_OR_INTEGRATION_BUG"
        if sudden_jump else "SMOOTH_NUMERICAL_ACCUMULATION"
    )
    result = {
        "status": "isolated_two_pass_layer_growth_diagnostic",
        "kernel_revision": KERNEL_REVISION,
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "context": CONTEXT,
        "rows": rows,
        "maximum_local_attention_relative_l2": max(local_values),
        "median_positive_cumulative_growth": median_positive_growth,
        "largest_cumulative_growth_layer": largest_growth_layer,
        "largest_cumulative_growth": largest_growth,
        "final_hidden": hidden_final_error,
        "last_token_logits": logit_error,
        "interpretation": interpretation,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print("\n=== SUMMARY ===")
    print(f"max_local_attention_rel_l2={max(local_values):.8f}")
    print(f"largest_growth_layer={largest_growth_layer}")
    print(f"largest_growth={largest_growth:.8f}")
    print(f"median_positive_growth={median_positive_growth:.8f}")
    print(f"final_hidden_rel_l2={hidden_final_error['relative_l2']:.8f}")
    print(
        f"last_logit_rel_l2={logit_error['relative_l2']:.8f} "
        f"argmax_same={logit_error['argmax_same']}"
    )
    print(f"interpretation={interpretation}\nJSON={OUTPUT}")

    del token_ids, eager_x, flash_x, eager_final, flash_final, eager_logits, flash_logits
    gc.collect()
    torch.cuda.empty_cache()
    return 1 if sudden_jump else 0


if __name__ == "__main__":
    raise SystemExit(main())
