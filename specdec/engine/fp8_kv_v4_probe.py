"""fp8_kv_v4_probe.py — V4 差分探针: 判定超高 PPL 是“instruct 模型裸文本退化”还是“引擎 NLL 路径 bug”

对同一个窗口(前 2048 tok 的 wikitext-2 test):
  A) 我们的引擎 fp16: 复用 v4 的 window_mean_nll
  B) HF transformers 一次全量前向: 同对齐公式算 mean NLL
若 A ≈ B ≈ 巨大 → 模型对裸文本退化(非引擎问题) → 换带 chat 模板的评测
若 A 巨大而 B 正常 → 引擎 NLL/前向路径有 bug
另打印前 12 个位置的 argmax token 文本, 直接看模型在吐什么。

Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v4_probe
"""
from __future__ import annotations

import math

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import WINDOW, ensure_data

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK = 512


def load_ours():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", torch.float16).eval()
    return ours


def ours_mean_nll_and_argmax(ours, ids_w):
    ours.set_kv_format("fp16")
    cache_list = None
    nll_sum = 0.0
    n = 0
    first_argmax = []
    S = ids_w.shape[0]
    with torch.inference_mode():
        for gs in range(0, S, CHUNK):
            piece = ids_w[gs:gs + CHUNK]
            lg, cache_list = ours._run(
                ours.embed_tokens(piece.unsqueeze(0)), cache_list, need_logits=True)
            tgt = ids_w[gs + 1: gs + 1 + CHUNK]
            rows = tgt.shape[0]
            ls = torch.log_softmax(lg[0, :rows].to(torch.float32), dim=-1)
            nll = -ls[torch.arange(rows, device=lg.device), tgt]
            nll_sum += nll.sum().item()
            n += rows
            if gs == 0:
                for r in range(min(12, rows)):
                    first_argmax.append(int(lg[0, r].argmax().item()))
    return nll_sum / n, first_argmax


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = ensure_data()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"][:WINDOW], dtype=torch.long)

    print("=== A: ours engine fp16 (chunked 512) ===", flush=True)
    ours = load_ours()
    nll_ours, am_ours = ours_mean_nll_and_argmax(ours, ids.cuda())
    print(f"mean NLL = {nll_ours:.4f}  ->  PPL = {math.exp(nll_ours):.1f}")
    print("argmax[0:12] =", tok.decode(am_ours)[:120])

    print("\n=== B: HF transformers one-shot full forward ===", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager",
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
    with torch.inference_mode():
        lg = hf(ids.unsqueeze(0).cuda()).logits  # (1, 2048, vocab)
        tgt = ids[1:]
        ls = torch.log_softmax(lg[0, :-1].to(torch.float32), dim=-1)
        nll = -ls[torch.arange(tgt.shape[0], device=lg.device), tgt]
        nll_hf = nll.mean().item()
    print(f"mean NLL = {nll_hf:.4f}  ->  PPL = {math.exp(nll_hf):.1f}")

    print("\n=== C: 真值 vs 预测 前 12 位 ===")
    for i in range(12):
        print(f"  pos{i:2d} true={tok.decode([ids[i+1].item()])!r} "
              f"argmax={tok.decode([am_ours[i]])!r}")

    verdict = ("INSTRUCT-ON-RAW DEGENERACY (both huge)"
               if math.exp(nll_ours) > 1e3 and math.exp(nll_hf) > 1e3
               else ("ENGINE BUG (A huge, B sane)"
                     if math.exp(nll_ours) > 1e3 else "BOTH SANE"))
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
