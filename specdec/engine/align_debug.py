"""align_debug.py — per-layer diff: where do our logits diverge from HF?

Compares hidden states layer by layer so we can bisect the remaining mismatch
(rope? attention? mlp? compounding?). Test tooling only.
Run:  python -m specdec.engine.align_debug
"""
from __future__ import annotations

import torch

from .model_self import QwenLM
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

    ids = tok(PROMPTS[0], return_tensors="pt").input_ids[:, :32]

    # hooks capture our per-layer outputs
    our_layers = []
    handles = []
    for i, layer in enumerate(ours.layers):
        def hook(m, inp, out, i=i):
            our_layers.append((i, out.detach()))
        handles.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        # compare embedding output
        emb_ours = ours.embed_tokens(ids)
        emb_hf = hf.model.embed_tokens(ids)
        print(f"embedding  max|Δ| = {(emb_ours - emb_hf).abs().max().item():.3e}")

        hf_out = hf(input_ids=ids, output_hidden_states=True)
        _ = ours(ids)

    # HF hidden_states: [embed, after L0, after L1, ..., after L23]
    print("layer    max|Δ|        (ours vs HF hidden_states)")
    for i, (lidx, out) in enumerate(our_layers):
        hf_h = hf_out.hidden_states[i + 1]
        md = (out - hf_h).abs().max().item()
        print(f"L{i:2d}     {md:.3e}")
    for h in handles:
        h.remove()

    # final logits
    with torch.no_grad():
        ours_logits = ours(ids)
    md = (ours_logits - hf_out.logits).abs().max().item()
    print(f"logits  max|Δ| = {md:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
