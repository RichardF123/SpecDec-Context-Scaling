"""bench_basic.py — first performance numbers (L2 milestone: decode tokens/s).

Measures (on the real Qwen2.5-0.5B weights):
  - prefill wall time for a prompt
  - decode tokens/s (per-token latency) via the KV-cache loop
Runs on CPU (fp32); if a CUDA GPU is visible it also runs fp16 on GPU.
NOTE: provisional numbers; a fair Spec-Bench-style protocol (warmup, reps,
fixed prompt length, seed) comes later in the benchmark phase.
Run:  python -m specdec.engine.bench_basic
"""
from __future__ import annotations

import time

import torch

from .align_ours import build_from_config

PROMPT = ("The history of artificial intelligence began in antiquity, with myths of "
          "artificial beings endowed with intelligence by craftsmen, and it continued "
          "through the twentieth century as the field of AI was formally founded.")
MAX_DECODE = 64


def timed(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(device, dtype, label):
    torch.manual_seed(0)
    print(f"\n== {label} ({device}, {dtype}) ==")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in
          AutoModelForCausalLM.from_pretrained(
              "Qwen/Qwen2.5-0.5B-Instruct", config=cfg, attn_implementation="eager",
              torch_dtype=torch.float32, trust_remote_code=True).state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    ours = ours.to(device=device, dtype=dtype).eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    S = ids.shape[1]

    with torch.no_grad():
        # warmup
        _ = ours.generate(ids, max_new_tokens=4, eos_id=tok.eos_token_id)

        def pref():
            with torch.no_grad():
                _ = ours(ids)
        pt = timed(pref)
        print(f"prefill: {pt*1000:.1f} ms for {S} tokens ({S/pt:.0f} tok/s)")

        def dec():
            with torch.no_grad():
                logits, caches = ours._run(ours.embed_tokens(ids), None)
                for _ in range(MAX_DECODE):
                    nid = logits[:, -1].argmax(dim=-1)
                    logits, caches = ours._run(ours.embed_tokens(nid.unsqueeze(-1)), caches)
        dt = timed(dec)
        print(f"decode: {dt*1000:.1f} ms for {MAX_DECODE} tokens "
              f"({MAX_DECODE/dt:.2f} tok/s, {dt/MAX_DECODE*1000:.2f} ms/token)")


def main() -> int:
    run("cpu", torch.float32, "CPU fp32")
    if torch.cuda.is_available():
        run("cuda", torch.float16, "GPU fp16 (RTX 4060)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
