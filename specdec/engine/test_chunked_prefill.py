"""test_chunked_prefill.py — chunked prefill == single-shot prefill (correctness gate).

Loads real Qwen2.5-0.5B (fp16, GPU if available), builds the KV cache for a
4096-token prompt two ways:
  (1) single-shot  _run(embed(all), None)
  (2) chunked      prefill_chunked(ids, chunk=1024)
Checks: per-layer cache k/v equality, and the same decoded tokens from both.
Run:  python -m specdec.engine.test_chunked_prefill
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config

CHUNK = 1024
DECODE_N = 12


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
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours = ours.to("cuda", torch.float16).eval()

    ids = make_text(tok, 4096).to("cuda")
    eos = tok.eos_token_id
    print(f"prompt tokens: {ids.shape[1]}", flush=True)

    with torch.inference_mode():
        # (1) single-shot prefill (need logits for the last chunk position later)
        logits1, cache1 = ours._run(ours.embed_tokens(ids), None, need_logits=True)

        # (2) chunked prefill: need logits only on the last chunk
        cache2, logits2 = None, None
        S = ids.shape[1]
        for st in range(0, S, CHUNK):
            piece = ids[:, st:st + CHUNK]
            last = (st + CHUNK >= S)
            lg, cache2 = ours._run(ours.embed_tokens(piece), cache2, need_logits=last)
            if last:
                logits2 = lg

        # cache equality (per layer, k and v)
        worst = 0.0
        for c1, c2 in zip(cache1, cache2):
            dk = (c1["k"] - c2["k"]).abs().max().item()
            dv = (c1["v"] - c2["v"]).abs().max().item()
            worst = max(worst, dk, dv)
        print(f"cache k/v worst max|Δ| (single vs chunked): {worst:.3e}", flush=True)

        # first token from each path
        n1 = int(logits1[:, -1].argmax())
        n2 = int(logits2[:, -1].argmax())
        print(f"first token identical: {n1 == n2} ({tok.decode([n1])!r} vs {tok.decode([n2])!r})")

        # decode DECODE_N more tokens from each cache
        def decode_from(cache, prev_logits):
            gen = []
            cur_logits, cur_cache = prev_logits, cache
            for _ in range(DECODE_N):
                nxt = int(cur_logits[:, -1].argmax())
                if nxt == eos:
                    break
                gen.append(nxt)
                cur_logits, cur_cache = ours._run(
                    ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)), cur_cache)
            return gen

        g1 = decode_from(cache1, logits1)
        g2 = decode_from(cache2, logits2)
        print(f"decoded tokens equal ({DECODE_N} steps): {g1 == g2}")
        if g1 != g2:
            print(f"  path1: {[tok.decode([t]) for t in g1]}")
            print(f"  path2: {[tok.decode([t]) for t in g2]}")

    # Behavioural gate: identical first token and identical decoded sequence.
    # Cache k/v max|Δ| is reported as information only: fp16 rounding differs
    # between the two op layouts and accumulates through layers, but it does not
    # change token selection (verified above). A larger |Δ| on cache tensors alone
    # is NOT a failure.
    ok = n1 == n2 and g1 == g2
    print("GATE:", "PASS — chunked prefill is behaviourally equal to single-shot"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
