"""fp8_vonly_8k_operational.py — 8K operational: base(BF16 KV) vs Asymmetric V-only FP8

条件: ctx=8192, chunk=2048(冻结协议), dtype=bf16, 每格式 3 轮稳定检查, decode 64 tok。
记录: prefill/decode 墙钟、torch 分配峰值/保留峰值、cache 逻辑字节
      (numel x element_size, 含 scale; K bf16 2B + V fp8 1B + v_scale 4B)、
      跨轮 token 稳定、finite。
理论对照(每 token/layer, Qwen2.5-1.5B kv=2 d=128):
  base:  1024 B | V-only: 512(K) + 256(V fp8) + 8(v_scale fp32, g=1) = 776 B (75.78%, 1.32x)
  28 层: 4096 tok: base 112 MiB, V-only ~84.9 MiB; 8192 tok: 224 vs ~169.75 MiB
注: 这只是 KV cache 逻辑字节/压缩比, 不代表整模型显存或 8GB 可行性(以 peak 实测为准)。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_vonly_8k_operational
"""
from __future__ import annotations

import time

import torch

from .align_ours import build_from_config
from .g2_operational import make_text

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
CTX = 8192
CHUNK = 2048
DECODE_N = 64
RUNS = 3


def load_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True).to("cuda").eval()
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", DTYPE).eval()
    return ours


def cache_logical_bytes(cache_list) -> int:
    tot = 0
    for lay in cache_list:
        for key, t in lay.items():
            if key == "fmt":
                continue
            tot += int(t.numel()) * t.element_size()
    return tot


def run_once(ours, ids, fmt: str, eos: int):
    ours.set_kv_format(fmt)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cache_list = None
    last_logits = None
    S = ids.shape[1]
    with torch.inference_mode():
        for st in range(0, S, CHUNK):
            piece = ids[:, st:st + CHUNK]
            last = (st + CHUNK >= S)
            lg, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                       need_logits=last)
            if last:
                last_logits = lg
    torch.cuda.synchronize()
    pref = time.perf_counter() - t0
    cache_bytes = cache_logical_bytes(cache_list)
    fin = True
    toks = []
    cur = last_logits
    with torch.inference_mode():
        for _ in range(DECODE_N):
            nxt = int(cur[:, -1].argmax())
            if nxt == eos:
                break
            toks.append(nxt)
            cur, cache_list = ours._run(
                ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)), cache_list)
    torch.cuda.synchronize()
    dec = time.perf_counter() - t0 - pref
    for lay in cache_list:
        for key, t in lay.items():
            if key != "fmt":
                fin &= bool(torch.isfinite(t.to(torch.float32)).all().item())
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_resv = torch.cuda.max_memory_reserved() / 2**30
    return dict(pref=pref, dec=dec, alloc=peak_alloc, resv=peak_resv,
                bytes=cache_bytes, toks=toks, fin=fin)


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos = tok.eos_token_id
    ids = make_text(tok, CTX).to("cuda")
    print(f"ctx={CTX} chunk={CHUNK} decode={DECODE_N} runs={RUNS} dtype={DTYPE} "
          f"model=1.5B", flush=True)

    ours = load_model()
    for fmt, label in (("fp16", "base/bf16 KV"), ("v_fp8", "V-only FP8 gs128")):
        print(f"\n=== {label} (fmt={fmt}) ===", flush=True)
        recs = []
        for r in range(RUNS):
            rec = run_once(ours, ids, fmt, eos)
            recs.append(rec)
            text = tok.decode(rec["toks"])
            print(f" run{r+1}: prefill {rec['pref']:.1f}s decode {rec['dec']:.1f}s | "
                  f"peak alloc {rec['alloc']:.2f} GiB resv {rec['resv']:.2f} GiB | "
                  f"cache logical {rec['bytes']/2**20:.2f} MiB | len {len(rec['toks'])} "
                  f"| tail {text[:36]!r} | finite {rec['fin']}", flush=True)
        stable = all(r["toks"] == recs[0]["toks"] for r in recs)
        cb = recs[0]["bytes"] / 2**20
        theo_bf16 = CTX * 28 * 1024 / 2**20      # 每 token/layer 1024B 的 bf16 理论基线
        theo_v = CTX * 28 * 776 / 2**20          # 每 token/layer 776B 的理论 V-only
        print(f" stable across {RUNS}: {stable}")
        print(f" cache logical (measured) = {cb:.2f} MiB | theoretical: "
              f"bf16 {theo_bf16:.2f} MiB, V-only {theo_v:.2f} MiB | "
              f"measured-vs-bf16 {cb/theo_bf16*100:.2f}% ({theo_bf16/cb:.3f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
