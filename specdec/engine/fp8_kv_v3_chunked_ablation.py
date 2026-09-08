"""fp8_kv_v3_chunked_ablation.py — G3 验证 V3: chunked fp8 消融（CUDA, 需引擎已实现 set_kv_format）

前置: 引擎已按 design_freeze_fp8_kv.md §4 + §4.1 实现 FP8 KV。未实现会抛 AttributeError。

流程: 同一 4K prompt, fp8 模式下分别以 chunk=1024 与 chunk=512 分块建缓存;
对比（同 G2-B 结构, 但格式为 fp8）:
  [1] 两缓存 payload 反量化后逐层 K/V max|Δ| + 层 0/mid/last 趋势（记录量级, fp8 舍入
      随深度累积; 不设数值阈值）
  [2] 各自续 decode 16 tok → 序列一致 = 行为等价（硬判据, 与 G2-B 同口径）
  [3] 结构/无 NaN 不变量
Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v3_chunked_ablation
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .model_self import dequantize_kv_fp8
from .g2_operational import make_text

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
PROMPT = 4096
DECODE_N = 16
CHUNKS = (1024, 512)


def load_model(fmt: str):
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
    ours.set_kv_format(fmt)
    return ours


def prefill(ours, ids, chunk: int):
    cache_list = None
    last_logits = None
    S = ids.shape[1]
    with torch.inference_mode():
        for st in range(0, S, chunk):
            piece = ids[:, st:st + chunk]
            last = (st + chunk >= S)
            lg, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                       need_logits=last)
            if last:
                last_logits = lg
    torch.cuda.synchronize()
    return last_logits, cache_list


def cache_kv_payload_diff(cache_a, cache_b) -> tuple[float, list[float]]:
    per_layer = []
    overall = 0.0
    for la, lb in zip(cache_a, cache_b):
        # 两缓存都是 fp8 payload + scale → 先反量化回运行 dtype 再比。
        ka = dequantize_kv_fp8(la["k"], la["k_scale"], DTYPE)
        kb = dequantize_kv_fp8(lb["k"], lb["k_scale"], DTYPE)
        va = dequantize_kv_fp8(la["v"], la["v_scale"], DTYPE)
        vb = dequantize_kv_fp8(lb["v"], lb["v_scale"], DTYPE)
        m = max((ka - kb).abs().max().item(), (va - vb).abs().max().item())
        per_layer.append(m)
        overall = max(overall, m)
    return overall, per_layer


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = make_text(tok, PROMPT).to("cuda")
    print(f"prompt={PROMPT} tok  chunks={CHUNKS}  decode={DECODE_N}  fp8 dtype={DTYPE}\n", flush=True)

    ours = load_model("fp8")
    eos = tok.eos_token_id

    caches, last_lgs = {}, {}
    for ch in CHUNKS:
        torch.cuda.empty_cache()
        lg, c = prefill(ours, ids, ch)
        caches[ch] = c
        last_lgs[ch] = lg
        print(f"chunk={ch}: cache built (fmt={c[0].get('fmt')}, "
              f"k dtype={c[0]['k'].dtype})", flush=True)

    # [0] 显存记账（B 版 scale 元数据; 以 chunk1024 fp8 cache 实测张量为准）
    c = caches[CHUNKS[0]]
    pl_n = sum(int(t.numel()) for lay in c for t in (lay["k"], lay["v"]))
    sc_n = sum(int(t.numel()) for lay in c for t in (lay["k_scale"], lay["v_scale"]))
    pl_b = pl_n * 1            # fp8 1 B/elem
    sc_b = sc_n * 4            # fp32 4 B/elem
    bf16_b = pl_n * 2          # 同形状 bf16 K/V
    tot = pl_b + sc_b
    print(f"[0] FP8 KV cache memory (T={c[0]['k'].shape[2]}, layers={len(c)}): "
          f"payload={pl_b/2**20:.2f} MiB + scales={sc_b/2**20:.2f} MiB "
          f"= {tot/2**20:.2f} MiB | vs BF16 {bf16_b/2**20:.2f} MiB "
          f"-> {tot/bf16_b*100:.1f}% ({bf16_b/tot:.2f}x)")
    print(f"    per token/layer: payload {pl_n//len(c)//c[0]['k'].shape[2]*1} B + "
          f"scales {sc_n//len(c)//c[0]['k'].shape[2]*4} B "
          f"= {(pl_n*1 + sc_n*4)//len(c)//c[0]['k'].shape[2]} B "
          f"(bf16 {pl_n//len(c)//c[0]['k'].shape[2]*2} B)")

    ok = True
    # [1] 漂移量级
    overall_d, per_layer_d = cache_kv_payload_diff(caches[CHUNKS[0]], caches[CHUNKS[1]])
    L = len(per_layer_d)
    trend = {i: per_layer_d[i] for i in [0, L // 2, L - 1]}
    print(f"[1] fp8 KV max|Δ| (chunk1024 vs chunk512, dequantized): overall={overall_d:.3e}")
    print(f"    per-layer @ 0/{L//2}/{L-1}: "
          f"{[f'{v:.3e}' for v in trend.values()]}")
    fin = all(torch.isfinite(t.to(torch.float32)).all().item()
              for c in caches.values()
              for t in (c[0]["k"], c[0]["v"], c[0]["k_scale"], c[0]["v_scale"]))
    print(f"    payload/scale finite: {fin}")
    ok &= fin

    # [2] 行为等价
    seqs = {}
    for ch in CHUNKS:
        cur, cc = last_lgs[ch], caches[ch]
        got = []
        with torch.inference_mode():
            for _ in range(DECODE_N):
                nxt = int(cur[:, -1].argmax())
                if nxt == eos:
                    break
                got.append(nxt)
                cur, cc = ours._run(
                    ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)), cc)
        seqs[ch] = got
        print(f"chunk={ch}: decode{DECODE_N} -> {tok.decode(got)[:50]!r} ({len(got)} tok)",
              flush=True)
    eq = seqs[CHUNKS[0]] == seqs[CHUNKS[1]]
    print(f"[2] decode sequences equal (fp8, chunk1024 vs chunk512): {eq}")
    ok &= eq

    print("\nV3:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
