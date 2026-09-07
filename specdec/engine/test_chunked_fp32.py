"""test_chunked_fp32.py — isolate the 0.55 cache gap: is it fp16 rounding or a real bug?

Same chunked-vs-single comparison but in **fp32** on GPU. If fp32 caches match to
~1e-5, the fp16 gap is rounding (op-layout dependent) and behaviour-equality is the
right gate. If fp32 still differs a lot, chunked prefill has a logic bug.
Run:  python -m specdec.engine.test_chunked_fp32
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config

CHUNK = 512
N_TOKENS = 2048
DECODE_N = 6


def make_text(tok, n_tokens: int) -> torch.Tensor:
    base = ("The quick brown fox jumps over the lazy dog and keeps running through "
            "fields and forests while thinking about artificial intelligence and "
            "how attention mechanisms work with keys queries and values.")
    while True:
        ids = tok(base, return_tensors="pt").input_ids
        if ids.shape[1] >= n_tokens:
            return ids[:, :n_tokens]
        base = base + " " + base


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True).to("cuda").eval()

    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours = ours.to("cuda", torch.float32).eval()

    ids = make_text(tok, N_TOKENS).to("cuda")
    print(f"prompt tokens: {ids.shape[1]}  dtype=fp32", flush=True)

    with torch.inference_mode():
        logits1, cache1 = ours._run(ours.embed_tokens(ids), None, need_logits=True)

        cache2, logits2 = None, None
        S = ids.shape[1]
        for st in range(0, S, CHUNK):
            piece = ids[:, st:st + CHUNK]
            last = (st + CHUNK >= S)
            lg, cache2 = ours._run(ours.embed_tokens(piece), cache2, need_logits=last)
            if last:
                logits2 = lg

        # per-layer cache diffs
        print("layer  cache k max|Δ|   cache v max|Δ|")
        worst = 0.0
        for i, (c1, c2) in enumerate(zip(cache1, cache2)):
            dk = (c1["k"] - c2["k"]).abs().max().item()
            dv = (c1["v"] - c2["v"]).abs().max().item()
            worst = max(worst, dk, dv)
            if i < 6 or i >= len(cache1) - 2:
                print(f"L{i:<3d}   {dk:.3e}        {dv:.3e}")
        print(f"worst over all layers: {worst:.3e}")

        n1 = int(logits1[:, -1].argmax())
        n2 = int(logits2[:, -1].argmax())
        print(f"first token identical: {n1 == n2}")

        def decode_from(cache, prev_logits):
            gen = []
            cur_logits, cur_cache = prev_logits, cache
            for _ in range(DECODE_N):
                nxt = int(cur_logits[:, -1].argmax())
                gen.append(nxt)
                cur_logits, cur_cache = ours._run(
                    ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)), cur_cache)
            return gen

        g1 = decode_from(cache1, logits1)
        g2 = decode_from(cache2, logits2)
        print(f"decoded tokens equal ({DECODE_N}): {g1 == g2}")

    print("VERDICT:",
          "fp32 clean (~1e-5) -> fp16 gap is rounding; behaviour gate stands"
          if worst < 1e-3 else
          "fp32 ALSO differs -> real chunked-prefill bug, investigate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
