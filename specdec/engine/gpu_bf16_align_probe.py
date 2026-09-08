"""gpu_bf16_align_probe.py — bf16 对齐探针: OURS vs HF（CUDA bf16, 中等长度）

背景: 本机 CUDA fp16 对该 Qwen 路径 NaN 频发 → 质量/NLL 一律用 bf16。
本探针确认 OURS bf16 基线路径在中等长度上可信（不止短 prompt）:
  - 权重核对: load_state_dict(strict=True) → missing/unexpected 全空
  - finite、logits max|Δ|、argmax 一致率、mean NLL / PPL（ours vs HF, 同口径）
  - 短对照: "The capital of France is" → argmax 应为 " Paris"
顺序加载（一次只常驻一个模型, 防并行 OOM）; 数据 = wikitext-2 test 前 512 tok。

Run（交互式 GPU 终端, 单跑勿并行）:  python -m specdec.engine.gpu_bf16_align_probe
"""
from __future__ import annotations

import math

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import ensure_data

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
N = 512          # 中等长度


def load_model(name: str):
    """顺序加载单个模型; 'hf' | 'ours'. ours 用 strict=True 核对权重."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True).to("cuda").eval()
    if name == "hf":
        return hf, cfg
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    missing, unexpected = ours.load_state_dict(sd, strict=True)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", DTYPE).eval()
    print(f"weight check: missing={list(missing)} unexpected={list(unexpected)}")
    return ours, cfg


def nll(logits, ids):
    lg = logits[0]
    tgt = ids[1:]
    rows = tgt.shape[0]
    ls = torch.log_softmax(lg[:rows].to(torch.float32), dim=-1)
    nll = -ls[torch.arange(rows, device=lg.device), tgt]
    return nll.mean().item(), math.exp(nll.mean().item())


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    path = ensure_data()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"][:N], dtype=torch.long)
    ids_c = ids.unsqueeze(0).cuda()

    free, total = torch.cuda.mem_get_info()
    print(f"[0] cuda free={free/2**30:.2f} / {total/2**30:.2f} GiB", flush=True)

    # 短对照: 必须是 " Paris"
    ctl = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

    print("loading ours (bf16 cuda) ...", flush=True)
    ours, _ = load_model("ours")
    with torch.inference_mode():
        lg_o = ours.forward(ids_c)
        lg_ctl = ours.forward(ctl)
    torch.cuda.synchronize()
    lg_o = lg_o.detach().to("cpu", torch.float32)
    lg_ctl = lg_ctl.detach().to("cpu", torch.float32)
    del ours
    torch.cuda.empty_cache()

    print("loading HF (bf16 cuda) ...", flush=True)
    hf, _ = load_model("hf")
    with torch.inference_mode():
        lg_h = hf(ids_c).logits
        lg_h_ctl = hf(ctl).logits
    torch.cuda.synchronize()
    lg_h = lg_h.detach().to("cpu", torch.float32)
    lg_h_ctl = lg_h_ctl.detach().to("cpu", torch.float32)
    del hf
    torch.cuda.empty_cache()

    ctl_o = tok.decode([int(lg_ctl[0, -1].argmax())])
    ctl_h = tok.decode([int(lg_h_ctl[0, -1].argmax())])
    print(f"[1] control 'The capital of France is' -> ours={ctl_o!r}  HF={ctl_h!r}")

    d = (lg_o - lg_h).abs()
    fin = bool(torch.isfinite(lg_o).all().item() and torch.isfinite(lg_h).all().item())
    print(f"[2] logits finite (ours/HF): {fin}")
    print(f"[3] max|Δ| (ours vs HF, bf16 cuda, {N} tok) = {d.amax().item():.3f}")

    am_o = lg_o[0, :-1].argmax(dim=-1)
    am_h = lg_h[0, :-1].argmax(dim=-1)
    agree = (am_o == am_h).float().mean().item()
    print(f"[4] argmax agreement = {agree*100:.2f}%")

    nll_o, ppl_o = nll(lg_o, ids)
    nll_h, ppl_h = nll(lg_h, ids)
    print(f"[5] mean NLL ours={nll_o:.4f}  HF={nll_h:.4f}   (Δ={nll_o-nll_h:+.4f})")
    print(f"    PPL     ours={ppl_o:.2f}  HF={ppl_h:.2f}")

    ok = fin and agree > 0.95 and abs(nll_o - nll_h) < 0.3 and ctl_o == " Paris" and ctl_h == " Paris"
    print("\nverdict:", "BASE bf16 PATH TRUSTWORTHY" if ok
          else "CHECK — inspect above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
