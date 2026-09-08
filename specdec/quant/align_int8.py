"""align_int8.py — L4 int8 integration gate on real Qwen weights.

Loads the real Qwen2.5-0.5B reference, builds the already-validated OURS
engine, copies the same weights, then converts only the Transformer Linear
weights to RTN int8.

Reports the numerical change and next-token agreement against the float OURS
engine. This is an integration gate; final A6 quality is decided by ppl /
held-out evaluation on the 1.5B target model.

Run:
    python -m specdec.quant.align_int8
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from specdec.engine.align_ours import PROMPTS, build_from_config
from .int8 import Int8Linear, quantize_model_int8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[1/5] loading HF weights: {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=cfg,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf.eval()

    print("[2/5] building validated OURS float engine")
    try:
        hf_q = hf.model.layers[0].self_attn.q_proj
        attn_bias = getattr(hf_q, "bias", None) is not None
    except Exception:
        attn_bias = False

    ours = build_from_config(cfg, attn_bias_force=attn_bias)

    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)

    res = ours.load_state_dict(sd, strict=False)
    print("  missing:", res.missing_keys)
    print("  unexpected:", res.unexpected_keys)

    if res.missing_keys:
        print("[FAIL] float engine is missing weights")
        return 1

    ours.eval()

    print("[3/5] recording float reference logits")
    references = []

    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids[:, : args.max_tokens]
            logits = ours(ids)[:, -1].clone()
            references.append((p, ids, logits))

    print("[4/5] converting Transformer Linear weights to int8")
    replaced = quantize_model_int8(ours)

    expected = len(ours.layers) * 7
    int8_modules = [m for m in ours.modules() if isinstance(m, Int8Linear)]

    remaining = []
    for i, layer in enumerate(ours.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if isinstance(getattr(layer.self_attn, name), nn.Linear):
                remaining.append(f"L{i}.self_attn.{name}")

        for name in ("gate_proj", "up_proj", "down_proj"):
            if isinstance(getattr(layer.mlp, name), nn.Linear):
                remaining.append(f"L{i}.mlp.{name}")

    real_int8 = all(m.qweight.dtype == torch.int8 for m in int8_modules)

    print(f"  replaced: {replaced} (expected {expected})")
    print(f"  Int8Linear count: {len(int8_modules)}")
    print(f"  remaining target nn.Linear: {remaining}")
    print(f"  all qweight dtype int8: {real_int8}")

    print("[5/5] comparing int8 vs float OURS on fixed prompts")

    worst = 0.0
    mean_sum = 0.0
    top1_matches = 0
    finite = True

    with torch.no_grad():
        for p, ids, float_logits in references:
            int8_logits = ours(ids)[:, -1]

            diff = (float_logits - int8_logits).abs()
            md = float(diff.max())
            mean = float(diff.mean())

            float_top1 = int(float_logits.argmax(dim=-1))
            int8_top1 = int(int8_logits.argmax(dim=-1))
            same = float_top1 == int8_top1

            worst = max(worst, md)
            mean_sum += mean
            top1_matches += int(same)
            finite = finite and bool(torch.isfinite(int8_logits).all())

            print(
                f"    max|Δ|={md:.3e}  "
                f"mean|Δ|={mean:.3e}  "
                f"top1 match={same}"
            )

    mean_avg = mean_sum / len(references)

    print()
    print(f"worst max|Δ|: {worst:.3e}")
    print(f"mean of mean|Δ|: {mean_avg:.3e}")
    print(f"top1 agreement: {top1_matches}/{len(references)}")
    print(f"all logits finite: {finite}")

    passed = (
        replaced == expected
        and len(int8_modules) == expected
        and not remaining
        and real_int8
        and finite
    )

    print("GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
