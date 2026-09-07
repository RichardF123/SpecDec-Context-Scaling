"""probe_gpu.py — decode-baseline sanity + 8K VRAM probe (GPU fp16).

Answers:
  A) transformers eager decode tok/s on this GPU (reference level).
  B) our engine decode tok/s under the SAME conditions (inference_mode).
  C) does an 8K-token prefill OOM? (peak VRAM for ours and transformers)
Run:  python -m specdec.engine.probe_gpu
"""
from __future__ import annotations

import time

import torch

PROMPT = ("The history of artificial intelligence began in antiquity, with myths of "
          "artificial beings endowed with intelligence by craftsmen, and it continued "
          "through the twentieth century as the field of AI was formally founded.")
DECODE_N = 32
WARM = 2
import builtins


def log(*a):
    print(*a, flush=True)


def decode_tok_per_s(model, ids, max_new, eos_id=None, use_transformers=False):
    def step(seq):
        if use_transformers:
            out = model.generate(seq, max_new_tokens=1, do_sample=False, use_cache=True)
            return out
        with torch.inference_mode():
            logits, caches = model._run(model.embed_tokens(seq), None) \
                if seq.shape[1] == ids.shape[1] else (None, None)
        return None  # placeholder replaced below

    # warmup
    seq = ids
    with torch.inference_mode():
        if use_transformers:
            for _ in range(WARM):
                seq = model.generate(seq[:, :5], max_new_tokens=1, do_sample=False, use_cache=True)
        else:
            for _ in range(WARM):
                logits, caches = model._run(model.embed_tokens(ids[:, :5]), None)
                nid = logits[:, -1].argmax(dim=-1)
    # measure steady decode
    torch.cuda.synchronize()
    with torch.inference_mode():
        if use_transformers:
            t0 = time.perf_counter()
            for _ in range(max_new):
                ids = model.generate(ids, max_new_tokens=1, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        else:
            logits, caches = model._run(model.embed_tokens(ids), None)
            t0 = time.perf_counter()
            for _ in range(max_new):
                nid = logits[:, -1].argmax(dim=-1)
                logits, caches = model._run(model.embed_tokens(nid.unsqueeze(-1)), caches)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
    return max_new / dt


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from .align_ours import build_from_config

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

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    # A) transformers eager decode
    log("A) transformers decode...")
    t_hf = decode_tok_per_s(hf, ids, DECODE_N, use_transformers=True)
    log(f"A) transformers eager decode : {t_hf:.2f} tok/s")

    # B) our engine decode (same conditions, inference_mode)
    log("B) our engine decode...")
    torch.cuda.reset_peak_memory_stats()
    t_ours = decode_tok_per_s(ours, ids, DECODE_N)
    log(f"B) our engine decode         : {t_ours:.2f} tok/s")

    # C) 8K prefill VRAM (ours and transformers)
    log("C) 8K prefill (ours)...")
    long = torch.randint(5, 150000, (1, 8192)).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _ = ours(long)
    torch.cuda.synchronize()
    m_ours = torch.cuda.max_memory_allocated() / 2**30
    log(f"C) 8K prefill peak VRAM (ours) = {m_ours:.2f} GiB")
    log("C) 8K prefill (transformers)...")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _ = hf(long).logits
    torch.cuda.synchronize()
    m_hf = torch.cuda.max_memory_allocated() / 2**30
    log(f"C) 8K prefill peak VRAM (transformers) = {m_hf:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
