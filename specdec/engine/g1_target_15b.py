"""g1_target_15b.py — G1: Qwen2.5-1.5B-Instruct target correctness gate.

Per v7.1 §7 G1:
  1. OUR 1.5B full forward vs HF 1.5B full forward      max|Δ| < 1e-3
  2. OUR 1.5B cached decode vs OUR 1.5B full forward    max|Δ| < 1e-3 + argmax same
  3. OUR greedy generation vs HF full-recompute greedy  sequences equal (near-tie aware)
Runs fp32 CPU (deterministic, like the 0.5B gates). Heavier model (~1.5B), allow a few minutes.
Run:  python -m specdec.engine.g1_target_15b
"""
from __future__ import annotations

import torch

from .model_self import QwenLM
from .align_ours import build_from_config

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "The capital of France is",
    "3 + 5 = 8. And 7 times 6 is",
    "Explain photosynthesis in one sentence:",
    "def add(a, b):",
]
MAX_NEW = 12
TIE = 0.01


def main() -> int:
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[0/5] loading 1.5B reference: {MODEL}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True)
    hf.eval()
    try:
        attn_bias = hf.model.layers[0].self_attn.q_proj.bias is not None
    except Exception:
        attn_bias = False
    print(f"[1/5] config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv={getattr(cfg,'num_key_value_heads','?')} "
          f"vocab={cfg.vocab_size} attn_bias={attn_bias}", flush=True)

    ours = build_from_config(cfg, attn_bias_force=attn_bias)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours.eval()
    print("[2/5] weights copied", flush=True)

    # ---- Check 1: full forward vs HF full forward ----
    worst1 = 0.0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids[:, :48]
            d = (ours(ids) - hf(ids).logits).abs().max().item()
            worst1 = max(worst1, d)
    print(f"[3/5] G1.1 full-forward max|Δ| = {worst1:.3e} "
          f"({'PASS' if worst1 < 1e-3 else 'FAIL'})", flush=True)

    # ---- Check 2: cached decode vs full forward (ours) ----
    base = ("The quick brown fox jumps over the lazy dog and keeps running through "
            "fields and forests while thinking about artificial intelligence and how "
            "attention mechanisms work with keys queries and values.")
    long_text = base + " " + base
    ids = tok(long_text, return_tensors="pt").input_ids[:, :32]
    split = ids.size(1) - 1
    if split < 1:
        raise RuntimeError("G1.2 needs at least two tokens for cached-vs-full validation")
    with torch.no_grad():
        full = ours(ids)
        xk = ours.embed_tokens(ids[:, :split])
        _, caches = ours._run(xk, None)
        x1 = ours.embed_tokens(ids[:, split:])
        step_logits, _ = ours._run(x1, caches)
        d2 = (full[:, -1] - step_logits[:, -1]).abs().max().item()
        same2 = bool((full[:, -1].argmax(-1) == step_logits[:, -1].argmax(-1)).item())
    print(f"[4/5] G1.2 cached-vs-full max|Δ| = {d2:.3e}  argmax same={same2} "
          f"({'PASS' if d2 < 1e-3 and same2 else 'FAIL'})", flush=True)

    # ---- Check 3: greedy generation vs HF full-recompute greedy ----
    ok3 = True
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            ours_ids = ours.generate(ids, max_new_tokens=MAX_NEW, eos_id=tok.eos_token_id)
            ref = ids
            for _ in range(MAX_NEW):
                nxt = hf(ref).logits[:, -1].argmax(dim=-1)
                if nxt.item() == tok.eos_token_id:
                    break
                ref = torch.cat([ref, nxt.unsqueeze(-1)], dim=1)
            o, h = ours_ids[0].tolist(), ref[0].tolist()
            n = min(len(o), len(h))
            if o[:n] == h[:n] and len(o) == len(h):
                print(f"    gen equal: True  {tok.decode(o)[:60]!r}")
            else:
                first = next(i for i in range(n) if o[i] != h[i])
                prefix = torch.tensor([o[:first]], dtype=torch.long)
                vals, _ = torch.topk(ours(prefix)[0, -1], 2)
                margin = (vals[0] - vals[1]).item()
                near = margin < TIE
                print(f"    gen diff at #{first}: ours={tok.decode([o[first]])!r} "
                      f"hf={tok.decode([h[first]])!r} margin={margin:.4f} near-tie={near}")
                ok3 = ok3 and near
    print(f"[5/5] G1.3 generation equality: {'PASS' if ok3 else 'FAIL'}", flush=True)

    ok = worst1 < 1e-3 and d2 < 1e-3 and same2 and ok3
    print("G1 GATE:", "PASS — 1.5B target correctness" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
