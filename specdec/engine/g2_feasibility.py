"""g2_feasibility.py — G2 minimal feasibility: 8K × 1.5B chunked prefill on GPU.

Checks (minimal operational gate):
  - 8K prompt + chunked prefill (chunk=2048) on Qwen2.5-1.5B fp16 CUDA: no OOM
  - record peak VRAM and prefill wall time
  - decode a few tokens from the built cache
  - run the whole thing twice; verify stable (same first/last tokens, similar time)
Correctness anchoring: chunked-prefill equivalence was validated at 4K (fp32) and
(fp16 behaviour) in earlier gates; this probe checks 8K operational stability.

Run (interactive GPU terminal):  python -m specdec.engine.g2_feasibility
"""
from __future__ import annotations

import time

import torch

from .align_ours import build_from_config

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CTX = 8192
CHUNK = 2048
DECODE_N = 6


def make_text(tok, n_tokens: int) -> torch.Tensor:
    base = ("The history of artificial intelligence and neural networks spans many "
            "decades of research into learning systems that process language text "
            "with attention mechanisms keys queries and values over long sequences "
            "of tokens in large transformer models trained on public corpora.")
    s = base
    while True:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] >= n_tokens:
            return ids[:, :n_tokens]
        s = s + " " + base


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours = ours.to("cuda", torch.float16).eval()

    ids = make_text(tok, CTX).to("cuda")
    eos = tok.eos_token_id
    print(f"context tokens: {ids.shape[1]}  chunk={CHUNK}  target=1.5B fp16", flush=True)

    results = []
    for run in range(2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        gen = []
        with torch.inference_mode():
            cache, last_logits = None, None
            S = ids.shape[1]
            for st in range(0, S, CHUNK):
                piece = ids[:, st:st + CHUNK]
                last = (st + CHUNK >= S)
                lg, cache = ours._run(ours.embed_tokens(piece), cache, need_logits=last)
                if last:
                    last_logits = lg
            torch.cuda.synchronize()
            pref = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30

            cur_lg = last_logits
            for _ in range(DECODE_N):
                nxt = int(cur_lg[:, -1].argmax())
                if nxt == eos:
                    break
                gen.append(nxt)
                cur_lg, cache = ours._run(
                    ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)), cache)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        text = tok.decode(gen)
        print(f"run{run+1}: prefill+decode {dt:.1f}s (prefill {pref:.1f}s) | "
              f"peak VRAM {peak:.2f} GiB | first token {tok.decode([gen[0]])!r} "
              f"| gen: {text[:40]!r}", flush=True)
        results.append((peak, dt, gen))

    stable = results[0][2] == results[1][2]
    peak_ok = results[0][0] < 7.5
    print(f"stable across runs: {stable}")
    print(f"peak VRAM < 7.5 GiB: {peak_ok}")
    print("G2 MINIMAL:", "PASS — 8K x 1.5B chunked prefill feasible"
          if (peak_ok and stable) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
