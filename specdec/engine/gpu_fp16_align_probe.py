"""gpu_fp16_align_probe.py — 决定性对齐探针: OUR GPU fp16 vs HF GPU fp16 的 logits

背景: G1 对齐(3.10e-05)与生成对齐均在 fp32/CPU; GPU fp16 上从未直接对比过
      "我们的引擎"与 "HF" 的 logits。V4 的 A 段(ours)mean NLL≈16.3、argmax 全 '!'
      疑 GPU fp16 专属 bug —— 本探针一次性判定。

流程: wikitext-2 test 前 256 tokens → ours.forward(ids) 与 hf(ids).logits 对比:
  [1] max|Δ| (logits, fp32 尺度) —— <1e-2 量级 = 一致; 大 = bug
  [2] argmax 一致率 (rows 0..254)
  [3] 双方前 8 位 argmax decode + 真值 —— 看 HF 是否同样吐 '!'
  [4] 双方 mean NLL / PPL（同 V4 口径, 小切片防 OOM）

Run（交互式 GPU 终端）:  python -m specdec.engine.gpu_fp16_align_probe
"""
from __future__ import annotations

import math

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import ensure_data

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 256


def load_model(name: str):
    """按需加载单个模型; name='hf'|'ours'. 同一时刻只常驻一个模型(省显存)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if name == "hf":
        return AutoModelForCausalLM.from_pretrained(
            MODEL, config=cfg, attn_implementation="eager",
            torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
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


def mem_report() -> str:
    free, total = torch.cuda.mem_get_info()
    return f"cuda free={free/2**30:.2f} GiB / total={total/2**30:.2f} GiB"


def mean_nll_and_argmax(logits, ids):
    """logits (1,S,vocab) fp16; 返回 (mean_nll, ppl, argmax_ids[0:8])."""
    lg = logits[0]
    tgt = ids[1:]
    rows = tgt.shape[0]
    ls = torch.log_softmax(lg[:rows].to(torch.float32), dim=-1)
    nll = -ls[torch.arange(rows, device=lg.device), tgt]
    am = lg[:8].argmax(dim=-1).tolist()
    return nll.mean().item(), math.exp(nll.mean().item()), am


def main() -> int:
    torch.manual_seed(0)
    path = ensure_data()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"][:N], dtype=torch.long)
    ids_c = ids.unsqueeze(0).cuda()

    print(f"[0] {mem_report()}  —— 若 free < 3 GiB, 先查 nvidia-smi 里其它 GPU 占用进程",
          flush=True)

    # A) 只加载 ours, logits 落 CPU 后释放模型
    print("loading ours (fp16 cuda) ...", flush=True)
    ours = load_model("ours")
    with torch.inference_mode():
        lg_o = ours.forward(ids_c)
    torch.cuda.synchronize()
    lg_o = lg_o.detach().to("cpu", torch.float32)   # 落 CPU 再释放模型
    del ours
    torch.cuda.empty_cache()
    print(f"    ours done ({mem_report()})", flush=True)

    # B) 只加载 HF, logits 落 CPU
    print("loading HF (fp16 cuda) ...", flush=True)
    hf = load_model("hf")
    with torch.inference_mode():
        lg_h = hf(ids_c).logits
    torch.cuda.synchronize()
    lg_h = lg_h.detach().to("cpu", torch.float32)
    del hf
    torch.cuda.empty_cache()

    d = (lg_o - lg_h).abs()
    print(f"[1] logits max|Δ| (ours vs HF, fp16 cuda) = {d.amax().item():.3e}")

    am_o = lg_o[0, :-1].argmax(dim=-1)
    am_h = lg_h[0, :-1].argmax(dim=-1)
    agree = (am_o == am_h).float().mean().item()
    print(f"[2] argmax agreement (rows 0..{N-2}) = {agree*100:.2f}%")

    nll_o, ppl_o, am8_o = mean_nll_and_argmax(lg_o, ids)
    nll_h, ppl_h, am8_h = mean_nll_and_argmax(lg_h, ids)
    print(f"[3] ours: mean NLL={nll_o:.4f} PPL={ppl_o:.1f}")
    print(f"    HF  : mean NLL={nll_h:.4f} PPL={ppl_h:.1f}")
    print("    pos  true / ours-argmax / HF-argmax")
    for i in range(8):
        print(f"      {i}: {tok.decode([ids[i+1].item()])[:12]!r:16} "
              f"{tok.decode([am8_o[i]])[:12]!r:16} {tok.decode([am8_h[i]])[:12]!r:16}")

    same = d.amax().item() < 1e-2
    print("\nverdict:", "ENGINE OK — ours==HF on GPU fp16 → 高 PPL 是模型/数据问题(需模板化)"
          if same else
          "ENGINE BUG on GPU fp16 — ours != HF → 逐层定位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
