"""g4_0_drift_diag.py — G4-0 cache 漂移归因诊断（0.5B draft, BF16 CUDA; 另含 FP32 控制）

目的：定位 cached-vs-full 分歧属于  ENGINE BUG / BF16 SHAPE DRIFT / NONDETERMINISM 中哪类
      （TARGET BLOCKER 由 g4_block_vs_serial.py 检查）。不修改引擎，不放松门槛。

四条生成路径（同一 tokenizer/ids/prompt/max_new=16, greedy argmax, batch=1, eager）:
  1. OURS full-recompute     2. OURS cached（engine _run 循环）
  3. HF full-recompute       4. HF cached（past_key_values 单步循环）
对 5 个 G4-0 prompts，每条路径重复 2 次（确定性检查）。

分歧定位：求四条路径生成序列的最长公共前缀长度 L；在共享前缀 prompt+L 处对每条路径
重算 next-logits（各按自己的 cache/full 语义），输出:
  argmax token/文本、top-1/top-2 logit、margin、路径两两 max|Δ|、finite。
FP32 控制：失败 prompt，OURS/HF × full/cached 四路径（短窗口, max_new=16）。

判读键：ours_cached vs hf_cached 是否一致（ENGINE BUG?）; hf_cached vs hf_full 是否同样分歧
      （BF16 shape drift?）; 同路径重复是否逐 token 确定（NONDETERMINISM?）。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.g4_0_drift_diag
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .g4_0_draft_correctness import PROMPTS as G40_PROMPTS

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 16
FAIL_P = "The history of artificial intelligence and neural networks spans many decades."
PROMPTS = G40_PROMPTS


def load_model(kind: str, dtype: torch.dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=dtype, trust_remote_code=True).to("cuda").eval()
    if kind == "hf":
        return hf, cfg, None
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", dtype).eval()
    return ours, cfg, None


def gen_ours_full(ours, ids, n, eos):
    seq = ids.clone()
    steps = []
    with torch.inference_mode():
        for _ in range(n):
            lg = ours.forward(seq)[0, -1]
            v, idx = lg.topk(2)
            tok = int(lg.argmax())
            steps.append((tok, int(idx[0]), float(v[0]), int(idx[1]), float(v[1]),
                          float(v[0] - v[1])))
            if tok == eos:
                break
            seq = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
    return steps


def gen_ours_cached(ours, ids, n, eos):
    steps = []
    with torch.inference_mode():
        lg, cache = ours._run(ours.embed_tokens(ids), None, need_logits=True)
        for _ in range(n):
            lg = lg[0, -1]
            v, idx = lg.topk(2)
            tok = int(lg.argmax())
            steps.append((tok, int(idx[0]), float(v[0]), int(idx[1]), float(v[1]),
                          float(v[0] - v[1])))
            if tok == eos:
                break
            lg, cache = ours._run(
                ours.embed_tokens(torch.tensor([[tok]], device=ids.device)),
                cache, need_logits=True)
    return steps


def gen_hf_full(hf, ids, n, eos):
    seq = ids.clone()
    steps = []
    with torch.inference_mode():
        for _ in range(n):
            lg = hf(seq).logits[0, -1]
            v, idx = lg.topk(2)
            tok = int(lg.argmax())
            steps.append((tok, int(idx[0]), float(v[0]), int(idx[1]), float(v[1]),
                          float(v[0] - v[1])))
            if tok == eos:
                break
            seq = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
    return steps


def gen_hf_cached(hf, ids, n, eos):
    steps = []
    with torch.inference_mode():
        out = hf(ids, use_cache=True)
        lg = out.logits
        pkv = out.past_key_values
        for _ in range(n):
            lg = lg[0, -1]
            v, idx = lg.topk(2)
            tok = int(lg.argmax())
            steps.append((tok, int(idx[0]), float(v[0]), int(idx[1]), float(v[1]),
                          float(v[0] - v[1])))
            if tok == eos:
                break
            out = hf(torch.tensor([[tok]], device=ids.device),
                     past_key_values=pkv, use_cache=True)
            lg = out.logits
            pkv = out.past_key_values
    return steps


def next_logits_aligned(kind, model, tok_ids, prompt_len):
    """在共享 prefix(tok_ids) 上按各路径语义给出 next-logits."""
    with torch.inference_mode():
        if kind == "ours_full":
            lg = model.forward(tok_ids)[0, -1]
        elif kind == "ours_cached":
            _, cache = model._run(model.embed_tokens(tok_ids[:, :-1]), None,
                                  need_logits=False)
            lg, _ = model._run(model.embed_tokens(tok_ids[:, -1:]), cache,
                               need_logits=True)
            lg = lg[0, -1]
        elif kind == "hf_full":
            lg = model(tok_ids).logits[0, -1]
        else:  # hf_cached
            out = model(tok_ids[:, :-1], use_cache=True)
            out = model(tok_ids[:, -1:], past_key_values=out.past_key_values,
                        use_cache=True)
            lg = out.logits[0, -1]
        return lg.to(torch.float32)


def toks_of(steps):
    return [s[0] for s in steps]


def common_prefix_len(*seqs):
    m = min(len(s) for s in seqs)
    c = 0
    while c < m and all(s[c] == seqs[0][c] for s in seqs):
        c += 1
    return c


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos = tok.eos_token_id

    print("== BF16 四条路径 (0.5B, max_new=16, x2 reps) ==")
    ours_b = load_model("ours", torch.bfloat16)[0]
    hf_b = load_model("hf", torch.bfloat16)[0]

    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]
        paths = {}
        for rep in (1, 2):
            paths[(1, rep)] = gen_ours_full(ours_b, ids, MAX_NEW, eos)
            paths[(2, rep)] = gen_ours_cached(ours_b, ids, MAX_NEW, eos)
            paths[(3, rep)] = gen_hf_full(hf_b, ids, MAX_NEW, eos)
            paths[(4, rep)] = gen_hf_cached(hf_b, ids, MAX_NEW, eos)
        det = all(toks_of(paths[(k, 1)]) == toks_of(paths[(k, 2)]) for k in (1, 2, 3, 4))
        L = common_prefix_len(*[toks_of(paths[(k, 1)]) for k in (1, 2, 3, 4)])
        print(f"\nprompt: {p[:50]!r}  (plen={plen})  common-prefix(L)={L}  "
              f"deterministic(rep1==rep2, 4 paths)={det}")
        names = {1: "ours_full", 2: "ours_cached", 3: "hf_full", 4: "hf_cached"}
        # 分歧区域: 围绕 min(L, 8) 附近, 打印每路径该步 (top1/top2/margin)
        j0 = max(0, min(L - 1, 6))
        jend = min(L + 2, min(len(paths[(k, 1)]) for k in (1, 2, 3, 4)))
        print("  step | " + " | ".join(f"{names[k]:11s} arg/top1/top2/margin" for k in (1, 2, 3, 4)))
        for j in range(j0, jend):
            row = []
            for k in (1, 2, 3, 4):
                st = paths[(k, 1)][j]
                row.append(f"{tok.decode([st[1]])[:6]!r}/{st[3]}={st[4]:.1f}/{st[5]:.2f}")
            print(f"  {j:4d} | " + " | ".join(f"{r:34s}" for r in row))
        # 对齐 next-logits 于共享前缀 L（若 L>=1）
        if L >= 1:
            pref = torch.cat([ids, torch.tensor([toks_of(paths[(1, 1)])[:L]],
                                                device="cuda")], dim=1)
            lgmap = {k: next_logits_aligned(names[k], ours_b if k <= 2 else hf_b,
                                            pref, plen) for k in (1, 2, 3, 4)}
            fin = all(bool(torch.isfinite(lg).all().item()) for lg in lgmap.values())
            def info(lg):
                v, idx = lg.topk(2)
                return int(idx[0]), tok.decode([int(idx[0])])[:12], float(v[0]), \
                    int(idx[1]), float(v[1]), float(v[0] - v[1])
            print(f"  aligned next-logits @prefix({plen}+{L}) finite={fin}:")
            for k in (1, 2, 3, 4):
                t1, d1, v1, t2, v2, m = info(lgmap[k])
                print(f"    {names[k]:11s}: argmax={t1} {d1!r} top1={v1:.3f} "
                      f"top2={t2}({v2:.3f}) margin={m:.3f}")
            dcs = (lgmap[2] - lgmap[4]).abs().max().item()
            dhf = (lgmap[4] - lgmap[3]).abs().max().item()
            dou = (lgmap[2] - lgmap[1]).abs().max().item()
            dfull = (lgmap[1] - lgmap[3]).abs().max().item()
            print(f"    max|Δ|: ours_cached-vs-hf_cached={dcs:.3e}  "
                  f"hf_cached-vs-hf_full={dhf:.3e}  ours_cached-vs-ours_full={dou:.3e}  "
                  f"ours_full-vs-hf_full={dfull:.3e}")

    del ours_b, hf_b
    torch.cuda.empty_cache()

    print("\n== FP32 控制（失败 prompt, max_new=16, 四路径一次） ==")
    ours_f = load_model("ours", torch.float32)[0]
    hf_f = load_model("hf", torch.float32)[0]
    ids = tok(FAIL_P, return_tensors="pt").input_ids.cuda()
    s1 = toks_of(gen_ours_full(ours_f, ids, MAX_NEW, eos))
    s2 = toks_of(gen_ours_cached(ours_f, ids, MAX_NEW, eos))
    s3 = toks_of(gen_hf_full(hf_f, ids, MAX_NEW, eos))
    s4 = toks_of(gen_hf_cached(hf_f, ids, MAX_NEW, eos))
    print(f"  fp32 ours_full == ours_cached: {s1 == s2}")
    print(f"  fp32 hf_full  == hf_cached : {s3 == s4}")
    print(f"  fp32 ours_full==hf_full    : {s1 == s3}   ours_cached==hf_cached: {s2 == s4}")
    if not (s1 == s2 == s3 == s4):
        for i, (a, b, c, d) in enumerate(zip(s1, s2, s3, s4)):
            if not (a == b == c == d):
                print(f"    first fp32 diff at {i}: ours_full={tok.decode([a])!r} "
                      f"ours_cached={tok.decode([b])!r} hf_full={tok.decode([c])!r} "
                      f"hf_cached={tok.decode([d])!r}")
                break
    print("\n分类依据（由数据判定, 非预设）: "
          "ours_cached==hf_cached 且 hf_cached 与 hf_full 同处分歧且 fp32 消失 -> BF16 SHAPE DRIFT; "
          "ours_cached!=hf_cached -> ENGINE BUG; 重复不同 -> NONDETERMINISM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
