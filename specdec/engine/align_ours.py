"""align_ours.py — L1 alignment gate for the student's OWN engine (model_self.py).

Compares the student's from-scratch forward vs HuggingFace Qwen2 on the same
input ids: max |Δ logits| must be < 1e-3 (fp32, CPU, fixed seed).

Run:
    python -m specdec.engine.align_ours --model Qwen/Qwen2.5-0.5B-Instruct
If the HF download is slow:  set HF_ENDPOINT=https://hf-mirror.com  first.
"""
from __future__ import annotations

import argparse

import torch

from .model_self import QwenLM

PROMPTS = [
    "The capital of France is",
    "3 + 5 =",
    "Explain in one sentence: why is the sky blue?",
    "Qwen models are developed by",
    "def fibonacci(n):",
]


def build_from_config(cfg, attn_bias_force=None) -> QwenLM:
    d = cfg.to_dict()
    hidden = int(d["hidden_size"])
    heads = int(d["num_attention_heads"])
    kv = int(d.get("num_key_value_heads", heads))
    head_dim = int(d.get("head_dim", hidden // heads))
    attn_bias = (
        bool(attn_bias_force)
        if attn_bias_force is not None
        else bool(d.get("attention_bias", False))
    )
    # transformers folds rope_theta into rope_parameters; read it from there first.
    rope_theta = float(
        (d.get("rope_parameters") or {}).get("rope_theta", d.get("rope_theta", 10000.0))
    )
    return QwenLM(
        vocab_size=int(d["vocab_size"]),
        hidden=hidden,
        heads=heads,
        kv_heads=kv,
        head_dim=head_dim,
        intermediate=int(d["intermediate_size"]),
        num_layers=int(d["num_hidden_layers"]),
        rope_theta=rope_theta,
        rms_eps=float(d.get("rms_norm_eps", 1e-6)),
        attn_bias=attn_bias,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] loading HF reference: {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf.eval()

    print("[2/4] building YOUR engine from config and copying weights")
    # Detect attention bias from the HF model itself (config dict key is unreliable).
    try:
        hf_q = hf.model.layers[0].self_attn.q_proj
        attn_bias = getattr(hf_q, "bias", None) is not None
    except Exception:
        attn_bias = False
    print("  detected attention_bias =", attn_bias)
    ours = build_from_config(cfg, attn_bias_force=attn_bias)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    # tied embeddings: HF also stores lm_head.weight (== embed weight); our engine
    # always reads logits from the tied embedding, so this key is intentionally unused.
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    res = ours.load_state_dict(sd, strict=False)
    print("  missing:", res.missing_keys)
    print("  unexpected:", res.unexpected_keys)
    if res.missing_keys:
        print("[FAIL] your engine is missing weights — check attribute naming "
              "(embed_tokens/layers/norm/q_proj/gate_proj/input_layernorm/...).")
        return 1
    ours.eval()

    print("[3/4] running alignment on", len(PROMPTS), "prompts")
    worst = 0.0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids[:, : args.max_tokens]
            hf_logits = hf(input_ids=ids).logits
            our_logits = ours(ids)
            md = float((hf_logits - our_logits).abs().max())
            worst = max(worst, md)
            same = int(hf_logits[0, -1].argmax()) == int(our_logits[0, -1].argmax())
            print(f"    max|Δ|={md:.3e}  top1 match={same}")
    print(f"[4/4] GATE: worst max|Δ| = {worst:.3e}"
          f"  -> {'PASS (< 1e-3)' if worst < args.tol else 'FAIL'}")
    return 0 if worst < args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
