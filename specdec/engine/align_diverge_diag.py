"""align_diverge_diag.py — at the first divergence, compare top tokens from:
   (a) OUR full forward, (b) HF full forward, (c) our cached step, (d) what hf.generate picked.
Run:  python -m specdec.engine.align_diverge_diag
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config


def main() -> int:
    torch.manual_seed(42)
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

    p = "The capital of France is"
    ids = tok(p, return_tensors="pt").input_ids

    with torch.no_grad():
        ours_ids = ours.generate(ids, max_new_tokens=16, eos_id=tok.eos_token_id)
        hf_ids = hf.generate(input_ids=ids, max_new_tokens=16, do_sample=False, use_cache=False)

    o, h = ours_ids[0].tolist(), hf_ids[0].tolist()
    n = min(len(o), len(h))
    first = next(i for i in range(n) if o[i] != h[i])
    prefix = torch.tensor([o[:first]], dtype=torch.long)
    print(f"first diff at token #{first}: ours={tok.decode([o[first]])!r} hf={tok.decode([h[first]])!r}")
    print(f"prefix text: {tok.decode(prefix[0])!r}")

    with torch.no_grad():
        ours_full = ours(prefix)[0, -1]
        hf_full = hf(prefix).logits[0, -1]
        # cached step: prefill first-1 then step token
        xk = ours.embed_tokens(prefix[:, :-1])
        _, caches = ours._run(xk, None)
        x1 = ours.embed_tokens(prefix[:, -1:])
        ours_step, _ = ours._run(x1, caches)

    def top3(logits, name):
        v, idx = torch.topk(logits, 3)
        row = ", ".join(f"{tok.decode([i])!r}={x:.3f}" for x, i in zip(v.tolist(), idx.tolist()))
        print(f"  {name:22s}: {row}")

    top3(ours_full, "ours full forward")
    top3(hf_full, "hf full forward")
    top3(ours_step[0][:, -1], "our cached step")
    d1 = (ours_full - hf_full).abs().max().item()
    print(f"ours-full vs hf-full max|Δ| at prefix: {d1:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
