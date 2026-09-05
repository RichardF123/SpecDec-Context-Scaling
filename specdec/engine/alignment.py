"""engine.alignment — A4 gate: our from-scratch forward vs HuggingFace Qwen2.

Run:
    python -m specdec.engine.alignment --model Qwen/Qwen2.5-0.5B-Instruct

Gate: max |Δ logits| < 1e-3 on every prompt (fp32, CPU, fixed seed).
If the HF checkpoint download is slow, set HF_ENDPOINT=https://hf-mirror.com first.
"""
from __future__ import annotations

import argparse
import torch

from .model import load_from_hf

PROMPTS = [
    "The capital of France is",
    "3 + 5 =",
    "Explain in one sentence: why is the sky blue?",
    "Qwen models are developed by",
    "def fibonacci(n):",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] loading HF model + config: {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf.eval()

    print("[2/4] building OUR engine and copying weights (no gradient)")
    with torch.no_grad():
        ours = load_from_hf(hf.state_dict(), cfg.to_dict())
    ours.eval()

    print("[3/4] running alignment on", len(PROMPTS), "prompts")
    worst = 0.0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids[:, : args.max_tokens]
            hf_logits = hf(input_ids=ids).logits
            our_logits = ours(ids)
            diff = (hf_logits - our_logits).abs()
            md = float(diff.max())
            worst = max(worst, md)
            top_hf = int(hf_logits[0, -1].argmax())
            top_ours = int(our_logits[0, -1].argmax())
            print(f"    max|Δ|={md:.3e}  top1 match={top_hf == top_ours}"
                  f"  pred={tok.decode(top_ours)!r}")
    print(f"[4/4] GATE: max|Δ| over all prompts = {worst:.3e}"
          f"  -> {'PASS (< 1e-3)' if worst < args.tol else 'FAIL'}")
    return 0 if worst < args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
