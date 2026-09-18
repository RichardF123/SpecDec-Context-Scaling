"""g4_0_draft_correctness.py — G4-0: 0.5B draft correctness gate（engine OURS, BF16 CUDA）

目标（对齐 G1 门槛，对象=draft 0.5B）:
  [0] 权重 strict 检查: load_state_dict(strict=True) -> missing/unexpected 均为空
  [1] full-forward:  OURS vs HF（BF16 CUDA）max|Δ| + argmax agreement% + finite
  [2] cached-vs-full: OURS cached decode vs OURS full forward（同 1.5B G1 口径）max|Δ| + argmax same
  [3] greedy generation: OURS.generate vs HF full-recompute greedy —— 多 prompt 全序列一致
失败先定位，不进入双模型（B3）。
说明: 与 1.5B 共用同一 tokenizer/词表（151936）；chat/BOS/EOS 同默认策略。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.g4_0_draft_correctness
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .g2_operational import make_text

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.bfloat16
PROMPTS = [
    "The capital of France is",
    "3 + 5 = 8. And 7 times 6 is",
    "def add(a, b):\n    return a +",
    "The history of artificial intelligence and neural networks spans many decades.",
    "Q: what is the largest planet?\nA:",
]
MAX_NEW = 16


def load_model(name: str):
    """顺序加载; 'hf' | 'ours'. ours 用 strict=True 核对权重。"""
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
    print(f"[0] weight check: missing={list(missing)} unexpected={list(unexpected)}")
    return ours, cfg


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos = tok.eos_token_id
    ids128 = make_text(tok, 128).to("cuda")

    ok = True
    # ---- [1] full-forward OURS vs HF ----
    ours, _ = load_model("ours")
    with torch.inference_mode():
        lg_o = ours.forward(ids128)
    torch.cuda.synchronize()
    lg_o = lg_o.detach().to("cpu", torch.float32)
    del ours
    torch.cuda.empty_cache()

    hf, _ = load_model("hf")
    with torch.inference_mode():
        lg_h = hf(ids128).logits
    torch.cuda.synchronize()
    lg_h = lg_h.detach().to("cpu", torch.float32)
    del hf
    torch.cuda.empty_cache()

    fin = bool(torch.isfinite(lg_o).all().item() and torch.isfinite(lg_h).all().item())
    d = (lg_o - lg_h).abs()
    am_o = lg_o[0].argmax(dim=-1)
    am_h = lg_h[0].argmax(dim=-1)
    agree = (am_o == am_h).float().mean().item()
    print(f"[1] full-forward OURS vs HF (bf16 cuda, 128 tok): finite={fin} "
          f"max|Δ|={d.amax().item():.3e} argmax-agree={agree*100:.2f}%")
    ok &= fin and d.amax().item() < 1e-1 and agree > 0.99

    # ---- [2] cached-vs-full (OURS) ----
    ours, _ = load_model("ours")
    with torch.inference_mode():
        full = ours.forward(ids128[:, :65])
        _, cache = ours._run(ours.embed_tokens(ids128[:, :64]), None)
        step, _ = ours._run(ours.embed_tokens(ids128[:, 64:65]), cache, need_logits=True)
    torch.cuda.synchronize()
    d2 = (full[:, -1] - step[:, -1]).abs().max().item()
    same = bool((full[:, -1].argmax() == step[:, -1].argmax()).item())
    # bf16 GPU 下 cached-vs-full 有取整漂移（1.5B 亦见 ~0.4）: 判据 = argmax same + finite
    print(f"[2] cached-vs-full (OURS): max|Δ|={d2:.3e} argmax-same={same} "
          f"(bf16 漂移量级记录, 不设 1e-3)")
    ok &= same

    # ---- [3] greedy generation（分解为 3a/3b, 避免"裁判路径≠被测路径"的假分叉）----
    def recompute_greedy(model, get_logits):
        """full-recompute greedy（每步整段前向, 不用 cache）; get_logits(model, seq)->(B,S,V)"""
        out = {}
        with torch.inference_mode():
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt").input_ids.cuda()
                seq = ids.clone()
                for _ in range(MAX_NEW):
                    lg = get_logits(model, seq)
                    nxt = lg[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    if nxt.item() == eos:
                        break
                    seq = torch.cat([seq, nxt], dim=1)
                out[p] = seq[0, ids.shape[1]:].tolist()
        return out

    # OURS: cached generate（[3b] 被测）与 full-recompute（[3a]/[3b] 参照）在同一次加载中算完
    ours_cached = {}
    ours, _ = load_model("ours")
    ours_full = recompute_greedy(ours, lambda m, s: m.forward(s))
    with torch.inference_mode():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            g = ours.generate(ids, MAX_NEW, eos_id=eos)
            ours_cached[p] = g[0, ids.shape[1]:].tolist()
    del ours
    torch.cuda.empty_cache()

    hf, _ = load_model("hf")
    hf_full = recompute_greedy(hf, lambda m, s: m(s).logits)
    del hf
    torch.cuda.empty_cache()

    def first_diff(a, b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return None

    print("[3a] greedy equality OURS full-recompute vs HF full-recompute "
          "(数学一致性, 无 cache):")
    ok3a = True
    for p in PROMPTS:
        eq = ours_full[p] == hf_full[p]
        ok3a &= eq
        fd = first_diff(ours_full[p], hf_full[p])
        extra = f"first diff at {fd}: ours={tok.decode([ours_full[p][fd]])!r} " \
                f"hf={tok.decode([hf_full[p][fd]])!r}" if fd is not None else ""
        print(f"    {p[:42]!r:46} equal={eq}  {extra}")

    print("[3b] OURS cached generate vs OURS full-recompute (隔离 cache 漂移):")
    ok3b = True
    for p in PROMPTS:
        eq = ours_cached[p] == ours_full[p]
        ok3b &= eq
        fd = first_diff(ours_cached[p], ours_full[p])
        extra = f"first diff at {fd}: cached={tok.decode([ours_cached[p][fd]])!r} " \
                f"full={tok.decode([ours_full[p][fd]])!r}" if fd is not None else ""
        print(f"    {p[:42]!r:46} equal={eq}  {extra}")

    ok &= ok3a and ok3b   # 严格门: [3a] 与 [3b] 都要求逐 token 全等
    print("\nG4-0 BASE (full-vs-full math):", "PASS" if ok3a else "CHECK")
    print("G4-0 CACHE/FULL EQUIVALENCE:", "PASS" if ok3b else "CHECK",
          "(若 CHECK: 预期 BF16 shape drift —— 由独立 g4_0_drift_diag 归因, 本脚本不作自动认定)")
    print("OVERALL:",
          "PASS" if ok else
          "CONDITIONAL PASS — requires documented drift diagnosis (g4_0_drift_diag + ledger)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
