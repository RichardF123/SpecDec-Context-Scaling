"""align_generate.py — L2 generation alignment gate (near-tie aware).

Both engines greedy-generate on the same real Qwen2.5 weights.
- Full sequence equality => PASS.
- If sequences diverge, we check the FIRST diverging position: if the top-2
  logit margin there is tiny (near-tie), the flip is fp-noise and acceptable;
  otherwise the mismatch is a real error => FAIL.
Run:  python -m specdec.engine.align_generate
"""
from __future__ import annotations

import torch

from .model_self import QwenLM
from .align_ours import build_from_config

PROMPTS = [
    "The capital of France is",
    "3 + 5 = 8. And 7 times 6 is",
    "In one sentence, explain photosynthesis:",
    "def add(a, b):",
]
MAX_NEW = 16
TIE = 0.01  # logit margin below which a greedy flip counts as near-tie fp noise


def top2_gap(model, ids):
    """Full forward on ids; return (top1, top2, margin) for the last position."""
    with torch.no_grad():
        logits = model(ids)[0, -1]
    vals, _ = torch.topk(logits, 2)
    return vals[0].item(), vals[1].item(), (vals[0] - vals[1]).item()


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

    eos_id = tok.eos_token_id
    ok = True
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        # Reference: manual greedy loop over HF FULL forward (pure official math).
        # (hf.generate is NOT used: it applies its own logits pipeline that disagrees
        #  with the model's own full forward by >1.6 logits in places.)
        with torch.no_grad():
            ours_ids = ours.generate(ids, max_new_tokens=MAX_NEW, eos_id=eos_id)
            ref = ids
            for _ in range(MAX_NEW):
                next_id = hf(ref).logits[:, -1].argmax(dim=-1)        # (1,)
                if next_id.item() == eos_id:
                    break
                ref = torch.cat([ref, next_id.unsqueeze(-1)], dim=1)  # (1,S+1)
        o, h = ours_ids[0].tolist(), ref[0].tolist()
        n = min(len(o), len(h))
        exact = (o[:n] == h[:n] and len(o) == len(h))
        status = "PASS"
        if not exact:
            # find first differing token
            first = next(i for i in range(n) if o[i] != h[i])
            prefix = torch.tensor([o[:first]], dtype=torch.long)  # common prefix incl prompt
            top1, top2, margin = top2_gap(ours, prefix)
            near_tie = margin < TIE
            status = "NEAR-TIE(fp-noise)" if near_tie else "FAIL"
            if not near_tie:
                ok = False
            print(f"  first diff at token #{first}: ours={tok.decode([o[first]])!r} hf={tok.decode([h[first]])!r}")
            print(f"  top-2 margin at that point: {margin:.4f} (top1={top1:.2f} top2={top2:.2f}) -> {status}")
        else:
            print(f"  full sequence equal: True -> PASS")
        print(f"  ours: {tok.decode(o)!r}")
        print(f"  hf:   {tok.decode(h)!r}")
    print("GATE:", "PASS" if ok else "FAIL (a non-near-tie divergence found)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
