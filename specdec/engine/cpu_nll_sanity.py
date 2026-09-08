"""cpu_nll_sanity.py — HF sanity audit: 引擎 vs HF 在 CPU fp32 的 NLL/argmax（不碰 GPU）

目的: V4 的 GPU fp16 PPL≈1e7。本脚本在 CPU fp32（G1 已锚定该配置下 ours==HF,
max|Δ|≈3e-5）复算同一类指标:
  A) HF (transformers, fp32 CPU, eager) 对 wikitext-2 前 128 token 的 mean NLL / PPL / argmax
  B) 我们的引擎 (fp32 CPU) 同一切片: max|Δ| vs HF + NLL
判读:
  - A、B 都 ~16 nats / PPL~1e7 → 模型对裸 wikitext 退化（与精度/实现无关）→ V4 需模板化
  - A 正常(≈2-6 nats) 而 B≈A → 引擎数学无误; 则 GPU fp16 专属问题 → 需跑 gpu_fp16_align_probe
  - A 正常而 B 巨大 → 引擎前向 bug（CPU 也不该）

Run:  python -m specdec.engine.cpu_nll_sanity   （CPU, 无 GPU）
"""
from __future__ import annotations

import math

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import ensure_data

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 128


def nll_and_argmax(logits, ids):
    lg = logits[0]
    tgt = ids[1:]
    rows = tgt.shape[0]
    ls = torch.log_softmax(lg[:rows].to(torch.float32), dim=-1)
    nll = -ls[torch.arange(rows), tgt]
    am = lg[:8].argmax(dim=-1).tolist()
    return nll.mean().item(), math.exp(nll.mean().item()), am


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = ensure_data()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"][:N], dtype=torch.long)

    print("loading HF (fp32 CPU)...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float32, trust_remote_code=True).eval()
    with torch.inference_mode():
        lg_h = hf(ids.unsqueeze(0)).logits

    print("loading ours (fp32 CPU)...", flush=True)
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    with torch.inference_mode():
        lg_o = ours.forward(ids.unsqueeze(0))

    d = (lg_o - lg_h).abs()
    print(f"[A] ours-vs-HF max|Δ| (fp32 CPU) = {d.amax().item():.3e}  "
          f"(G1 锚定 ~3e-5)")

    nll_o, ppl_o, am8_o = nll_and_argmax(lg_o, ids)
    nll_h, ppl_h, am8_h = nll_and_argmax(lg_h, ids)
    print(f"[B] ours: mean NLL={nll_o:.4f}  PPL={ppl_o:.1f}")
    print(f"[C] HF  : mean NLL={nll_h:.4f}  PPL={ppl_h:.1f}")
    print("    pos  true / ours-argmax / HF-argmax")
    for i in range(8):
        print(f"      {i}: {tok.decode([ids[i+1].item()])[:14]!r:18} "
              f"{tok.decode([am8_o[i]])[:14]!r:18} {tok.decode([am8_h[i]])[:14]!r:18}")

    if ppl_h > 1e3 and ppl_o > 1e3:
        v = "MODEL DEGENERATES ON RAW WIKITEXT (precision-independent) → V4 需 chat 模板"
    elif abs(nll_o - nll_h) < 0.5:
        v = "ENGINE MATH OK on CPU fp32; HF sane → 疑 GPU fp16 专属 → 跑 gpu_fp16_align_probe"
    else:
        v = "ENGINE CPU BUG (ours huge, HF sane)"
    print(f"\nverdict: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
