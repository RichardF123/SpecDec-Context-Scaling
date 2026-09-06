"""align_cache_real.py — cache==full check on REAL Qwen weights at several depths.

If cached-decode diverges from full-forward at some depth, we have a real bug
in the cache path (start_pos / mask / concat / GQA ordering).
Run:  python -m specdec.engine.align_cache_real
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config


def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    hf.eval()
    try:
        attn_bias = hf.model.layers[0].self_attn.q_proj.bias is not None
    except Exception:
        attn_bias = False
    ours = build_from_config(cfg, attn_bias_force=attn_bias)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours.eval()

    text = "The capital of France is Paris. It is the largest city in Europe and the third largest city in the"
    ids = tok(text, return_tensors="pt").input_ids  # (1, S)
    S = ids.shape[1]
    print(f"text tokens: {S}")

    # baseline: ours full vs hf full on the whole text
    with torch.no_grad():
        d_full = (ours(ids) - hf(ids).logits).abs().max().item()
    print(f"ours-full vs hf-full max|Δ|: {d_full:.3e}")

    for k in [5, 10, 15, 20, S - 1]:
        prefix = ids[:, : k + 1]          # full path: k+1 tokens
        with torch.no_grad():
            full = ours(prefix)                                    # (1,k+1,V)
            xk = ours.embed_tokens(ids[:, :k])
            _, caches = ours._run(xk, None)                        # prefill k tokens
            x1 = ours.embed_tokens(ids[:, k:k + 1])
            step_logits, _ = ours._run(x1, caches)                 # decode token k
        diff = (full[:, -1] - step_logits[:, -1]).abs().max().item()
        same = bool((full[:, -1].argmax(-1) == step_logits[:, -1].argmax(-1)).item())
        print(f"split k={k}:  cache-vs-full last-pos max|Δ| = {diff:.3e}  argmax same={same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
