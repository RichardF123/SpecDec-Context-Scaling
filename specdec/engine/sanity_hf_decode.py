"""sanity_hf_decode.py — quick reference: transformers eager decode tok/s on GPU.
Run:  python -m specdec.engine.sanity_hf_decode
"""
import time
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = ("The history of artificial intelligence began in antiquity, with myths of "
          "artificial beings endowed with intelligence by craftsmen.")
N = 20


def main() -> int:
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", config=cfg, attn_implementation="eager",
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        for _ in range(2):
            _ = model.generate(ids[:, :5], max_new_tokens=1, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            ids = model.generate(ids, max_new_tokens=1, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    print(f"transformers eager decode (fp16, cuda): {N/dt:.2f} tok/s  ({dt/N*1000:.1f} ms/token)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
