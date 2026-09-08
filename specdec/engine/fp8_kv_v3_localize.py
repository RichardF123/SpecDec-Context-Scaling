"""fp8_kv_v3_localize.py — V3 分叉定位（D 之后仍 chunk 敏感时用）

目标: 分清 chunk 敏感来自"FP8 语义"还是"base bf16 内核/形状噪声", 并定位首个分叉 (layer, token)。
五条前向（同一 4096 prompt, bf16）:
  A = (fmt fp16, chunk 1024)    B = (fmt fp16, chunk 512)    → base bf16 chunk 漂移
  C = (fmt fp8 , chunk 1024)    D = (fmt fp8 , chunk 512)    → fp8 chunk 漂移
  E = (fmt fp8 , chunk 512) 重复                      → 确定性 (期望全 0)
对照:
  [1] per-layer KV max|Δ|: base(A,B) vs fp8(C,D) —— 若接近 → fp8 未新增 chunk 耦合
  [2] fp8-vs-base 同 chunk (C,A): 纯 FP8 量化 KV 误差（无 chunk 混淆）→ 判断 B 是否必要
  [3] 首个 max|Δ|>阈值(1e-1/1e-2) 的 (layer, 位置) + 该层最大差位置
  [4] 五路 decode16 序列与 base-1024(A) 的一致前缀
判读:
  - base 层0 漂移≈0 而 fp8 层0 漂移≈4.75 → fp8 耦合仍在（查实现）
  - base 层0 漂移也大 → 内核/形状级 bf16 噪声, fp8 只是继承它 → V3 判据须改为
    “fp8 额外耦合 = drift(C,D)−drift(A,B) 小” + 分歧模式与 base 一致
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_kv_v3_localize
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .g2_operational import make_text
from .model_self import dequantize_kv_fp8

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
PROMPT = 4096
DECODE_N = 16
CHUNKS = (1024, 512)
THRESH_LO, THRESH_HI = 1e-2, 1e-1


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


def run_once(ours, ids, fmt: str, chunk: int, eos: int):
    ours.set_kv_format(fmt)
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
    return cache_list, toks


def layerwise_kv_diff(cache_a, cache_b, dtype) -> list[float]:
    out = []
    for la, lb in zip(cache_a, cache_b):
        if la.get("fmt") == "fp8":
            ka = dequantize_kv_fp8(la["k"], la["k_scale"], torch.float32)
            va = dequantize_kv_fp8(la["v"], la["v_scale"], torch.float32)
        else:
            ka, va = la["k"].to(torch.float32), la["v"].to(torch.float32)
        if lb.get("fmt") == "fp8":
            kb = dequantize_kv_fp8(lb["k"], lb["k_scale"], torch.float32)
            vb = dequantize_kv_fp8(lb["v"], lb["v_scale"], torch.float32)
        else:
            kb, vb = lb["k"].to(torch.float32), lb["v"].to(torch.float32)
        out.append(max((ka - kb).abs().max().item(),
                       (va - vb).abs().max().item()))
    return out


def first_divergence(per_layer: list[float]) -> str:
    for i, v in enumerate(per_layer):
        if v > THRESH_LO:
            tag = "LO" if v > THRESH_HI else "hi" if v > THRESH_LO else ""
            return f"layer {i} (maxΔ={v:.3e})"
    return "none"


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = make_text(tok, PROMPT).to("cuda")

    ours = load_model()
    eos = tok.eos_token_id
    runs = {
        "A_base_1024": ("fp16", 1024),
        "B_base_512": ("fp16", 512),
        "C_fp8_1024": ("fp8", 1024),
        "D_fp8_512": ("fp8", 512),
        "E_fp8_512_rep": ("fp8", 512),
    }
    caches, seqs = {}, {}
    for name, (fmt, ch) in runs.items():
        torch.cuda.empty_cache()
        c, toks = run_once(ours, ids, fmt, ch, eos)
        caches[name] = c
        seqs[name] = toks
        print(f"{name}: cache built, decode16 -> "
              f"{tok.decode(toks)[:60]!r} ({len(toks)} tok)", flush=True)

    def show(label, pair):
        pl = layerwise_kv_diff(caches[pair[0]], caches[pair[1]], DTYPE)
        print(f"\n[{label}] per-layer KV max|Δ|:")
        print("   " + " ".join(f"{v:.2f}" for v in pl))
        print(f"    overall={max(pl):.3f}  first>{THRESH_LO}: {first_divergence(pl)}")

    show("BASE bf16 chunk drift (A1024 vs B512)", ("A_base_1024", "B_base_512"))
    show("FP8   chunk drift (C1024 vs D512)", ("C_fp8_1024", "D_fp8_512"))
    show("FP8-vs-BASE same chunk 1024 (C vs A) — pure FP8 KV error", ("C_fp8_1024", "A_base_1024"))
    show("determinism (D512 vs E512) — expect all 0", ("D_fp8_512", "E_fp8_512_rep"))

    base16 = seqs["A_base_1024"]
    print("\n[4] decode16 agreement prefix vs base-1024:")
    for name in runs:
        s = seqs[name]
        a = 0
        for x, y in zip(base16, s):
            if x == y:
                a += 1
            else:
                break
        print(f"    {name:14}: prefix {a}/{min(len(base16), len(s))}")

    plb = layerwise_kv_diff(caches["A_base_1024"], caches["B_base_512"], DTYPE)
    plf = layerwise_kv_diff(caches["C_fp8_1024"], caches["D_fp8_512"], DTYPE)
    extra = [f - b for f, b in zip(plf, plb)]
    print(f"\nfp8 额外 chunk 耦合 = drift_fp8 − drift_base: overall {max(extra):+.3f}  "
          f"(≈0 → D 达成, 残留为 base bf16 内核/形状噪声)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
