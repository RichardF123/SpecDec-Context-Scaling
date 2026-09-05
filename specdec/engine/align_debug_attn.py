"""align_debug_attn.py — bisect INSIDE attention layer 0.

Checks in order:
  1. kv_repeat vs HF-canonical grouped expansion.
  2. attention probabilities (ours vs HF output_attentions).
  3. per-head error pattern.
  4. CAPTURE HF's rope cos/sin table and compare to ours (the decisive rope check).
Run:  python -m specdec.engine.align_debug_attn
"""
from __future__ import annotations

import torch

from .model_self import apply_rope, kv_repeat, Build_Rope_cache
from .align_ours import build_from_config, PROMPTS


def main() -> int:
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True,
    )
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

    ids = tok(PROMPTS[0], return_tensors="pt").input_ids[:, :16]
    att = ours.layers[0].self_attn
    b, s = ids.shape

    # ---- read HF's rope inv_freq table (no monkeypatching needed) ----
    hf_inv = None
    for name in ("rotary_emb", "rotary"):
        if hasattr(hf.model, name):
            emb = getattr(hf.model, name)
            if hasattr(emb, "inv_freq"):
                hf_inv = emb.inv_freq.detach().clone()
            break
    if hf_inv is not None:
        d = cfg.to_dict()
        print(f"cfg rope_theta={d.get('rope_theta', 10000.0)}  ours rope_theta={ours.rope_theta}")
        exponents = torch.arange(0, att.head_dim, 2, dtype=torch.float32) / att.head_dim
        our_inv = 1.0 / (ours.rope_theta ** exponents)
        print(f"HF inv_freq len={hf_inv.numel()}  ours len={our_inv.numel()}")
        print(f"inv_freq max|Δ| = {(hf_inv.to(our_inv.dtype) - our_inv).abs().max().item():.3e}")
        print("HF inv_freq[:4]:", hf_inv[:4].tolist())
        print("our inv_freq[:4]:", our_inv[:4].tolist())

    with torch.no_grad():
        x0 = ours.embed_tokens(ids)
        h = ours.layers[0].input_layernorm(x0)
        cos, sin = Build_Rope_cache(att.head_dim, s, ours.rope_theta, x0.device, x0.dtype)

        hf_out = hf(input_ids=ids, output_attentions=True)
        hf_probs = hf_out.attentions[0]

        # --- replicate our attention path ---
        q = att.q_proj(h).view(b, s, att.heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(h).view(b, s, att.kv_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(h).view(b, s, att.kv_heads, att.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        n_rep = att.heads // att.kv_heads
        k_rep = kv_repeat(k, n_rep)
        k_ref = k[:, :, None, :, :].expand(b, att.kv_heads, n_rep, s, att.head_dim).reshape(
            b, att.kv_heads * n_rep, s, att.head_dim)
        print("kv_repeat == canonical:", bool((k_rep == k_ref).all().item()))
        v = kv_repeat(v, n_rep)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (att.head_dim ** 0.5)
        mask = torch.triu(torch.ones(s, s, device=x0.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        our_probs = torch.softmax(scores, dim=-1)

    print("our probs shape:", tuple(our_probs.shape), " hf:", tuple(hf_probs.shape))
    pd = (our_probs - hf_probs).abs()
    print(f"attn probs  max|Δ| = {pd.max().item():.3e}   mean|Δ| = {pd.mean().item():.3e}")
    per_head = pd.squeeze(0).max(dim=-1).values.max(dim=-1).values
    print("per-head max|Δ|:", " ".join(f"{x:.3f}" for x in per_head.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
