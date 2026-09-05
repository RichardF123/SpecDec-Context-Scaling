"""align_debug_layer0.py — bisect INSIDE layer 0: norm? attention? mlp?

Compares our layer-0 submodule outputs against HF's layer-0 submodule outputs.
Run:  python -m specdec.engine.align_debug_layer0
"""
from __future__ import annotations

import torch

from .model_self import QwenLM
from .align_ours import build_from_config, PROMPTS

PARTS = ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]


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

    ours_cap, hf_cap = {}, {}
    handles = []

    def make(mod, tag, cap, take0=False):
        def hook(m, inp, out):
            o = out[0] if take0 and isinstance(out, tuple) else out
            cap.setdefault(tag, []).append(o.detach())
        handles.append(mod.register_forward_hook(hook))

    for tag in PARTS:
        # our layer 0 submodule
        make(getattr(ours.layers[0], tag), tag, ours_cap)
        # HF layer 0 submodule (self_attn returns a tuple in eager)
        make(getattr(hf.model.layers[0], tag), tag, hf_cap, take0=(tag == "self_attn"))

    with torch.no_grad():
        _ = ours(ids)
        _ = hf(input_ids=ids)

    for h in handles:
        h.remove()

    print("part                   max|Δ|")
    for tag in PARTS:
        a = ours_cap[tag][0]
        b = hf_cap[tag][0]
        if a.shape != b.shape:
            print(f"{tag:22s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        else:
            print(f"{tag:22s} {((a - b).abs().max()).item():.3e}")

    # also compare raw q/k projections (pre-head-split) via their own hooks
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        o_cap, h_cap = [], []
        h1 = ours.layers[0].self_attn.get_submodule(proj).register_forward_hook(
            lambda m, i, o, c=o_cap: c.append(o.detach()))
        h2 = hf.model.layers[0].self_attn.get_submodule(proj).register_forward_hook(
            lambda m, i, o, c=h_cap: c.append(o.detach()))
        with torch.no_grad():
            _ = ours(ids)
            _ = hf(input_ids=ids)
        h1.remove()
        h2.remove()
        md = (o_cap[0] - h_cap[0]).abs().max().item()
        print(f"{proj:22s} {md:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
