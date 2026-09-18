"""g4_b6_tie_diag.py — B6 唯一失败 case 的 near-tie 归因（base_bf16 / natural-2）

问题: 同 prefix 下 spec=' convert' 而 ref=' use'（step7）。判定这是**数值可判定性**问题
还是实现 bug。判据不用任何人为阈值，只用“信号 vs 噪声”**在同一位置**的实测比较：
  signal = fp32 下该候选对的 logit 差(gap)
  noise  = |bf16 下同一对的 gap - fp32 下同一对的 gap|（形状/历史引入的差分误差）
  若 |signal| <= noise  =>  该位置在 bf16 下不可判定（与算法无关）

路径（同一 prefix = prompt + ref[:7]，取该位置 logits）:
  H  bf16 AR 历史: prefill(prompt) -> 逐 token 单步   (= generate 的真实形状历史)
  A  bf16 整段一次前向            B  bf16 缓存单步 (13+1)
  C  fp32 整段一次前向            D  fp32 缓存单步     E  fp32 AR 历史
另: 从 spec 的 log 回调里取出该位置所在轮次，打印 target 的**实际裁决** g[·] 与 draft 提议，
区分该 token 是“被接受”还是“target 替换”。

内存: fp32 阶段在 **CPU** 单模型跑（8GB 显存放不下 fp32 权重 6.2GB + 临时拷贝）。
Run（交互式终端, 单跑; 先 set HF_HUB_OFFLINE=1）: python -m specdec.engine.g4_b6_tie_diag
"""
from __future__ import annotations

import gc
import math
import traceback

import torch

from .g4_b4_cache_tests import _from_pretrained
from .spec_dec import spec_decode_greedy

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Photosynthesis is the process by which"
N = 16
K = 4


def load(model_name, dtype, device):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from .align_ours import build_from_config
    cfg = _from_pretrained(AutoConfig, model_name, trust_remote_code=True)
    hf = _from_pretrained(AutoModelForCausalLM, model_name, config=cfg,
                          attn_implementation="eager", torch_dtype=dtype,
                          trust_remote_code=True)
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf, sd
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    ours = ours.to(device, dtype).eval()
    tok = _from_pretrained(AutoTokenizer, model_name, trust_remote_code=True)
    return ours, tok


def at_prefix_full(model, prefix):
    with torch.inference_mode():
        return model.forward(prefix)[0, -1].to(torch.float32)


def at_prefix_cached(model, prefix):
    with torch.inference_mode():
        _, cache = model._run(model.embed_tokens(prefix[:, :-1]), None,
                              need_logits=False)
        lg, _ = model._run(model.embed_tokens(prefix[:, -1:]), cache, need_logits=True)
    return lg[0, -1].to(torch.float32)


def at_prefix_hist(model, prefix, split):
    """复刻 generate 的形状历史: prefill 前 split 个 -> 之后逐 token 单步。"""
    with torch.inference_mode():
        lg, cache = model._run(model.embed_tokens(prefix[:, :split]), None,
                               need_logits=True)
        for j in range(split, prefix.shape[1]):
            lg, cache = model._run(model.embed_tokens(prefix[:, j:j + 1]), cache,
                                   need_logits=True)
    return lg[0, -1].to(torch.float32)


def ulp_of(v: float, mant_bits: int) -> float:
    """v 在该精度下的 1 ULP(bf16 mant_bits=7, fp32=23)。"""
    if v == 0.0:
        return 0.0
    _, e = math.frexp(abs(v))
    return math.ldexp(1.0, e - 1 - mant_bits)


def report(tag, lg, tok, want_ids, mant_bits):
    am = int(lg.argmax())
    v1 = float(lg[am])
    tied = int((lg == lg.max()).sum())
    others = [w for w in sorted(set(want_ids)) if w != am]
    gap = min(v1 - float(lg[w]) for w in others) if others else None
    ulp = ulp_of(v1, mant_bits)
    flags = " ".join(f"{tok.decode([w])!r}@{float(lg[w]):.4f}" for w in sorted(set(want_ids)))
    gs = "n/a" if gap is None else f"{gap:+.4f}"
    print(f"  [{tag}] argmax={tok.decode([am])!r} ({v1:.4f})  tied_at_max={tied}  "
          f"gap_vs_other={gs}  1ulp={ulp:.2e} | {flags}", flush=True)
    return am, gap, ulp, (gap == 0.0)


def round_of(trace, prompt_len, idx):
    """找出 gen[idx] 由哪一轮产生: 返回 (轮次记录, 轮内偏移, 'accepted'|'replacement')。"""
    for d in trace:
        off = idx - (d["pre"] - prompt_len)
        if 0 <= off < len(d["emitted"]):
            return d, off, ("accepted" if off < d["m"] else "replacement")
    return None, None, None


def pair_gap(row, a, b):
    return float(row[a]) - float(row[b])


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok_t = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    ids = tok_t(PROMPT, return_tensors="pt").input_ids.cuda()
    split = int(ids.shape[1])
    eos = tok_t.eos_token_id

    print("[stage] bf16: rerun spec + reference, 找分歧点", flush=True)
    tgt, _ = load(TARGET, torch.bfloat16, "cuda")
    drf, _ = load(DRAFT, torch.bfloat16, "cuda")
    ref = tgt.generate(ids, N, eos_id=eos)[0, ids.shape[1]:].tolist()
    trace = []
    out = spec_decode_greedy(tgt, drf, ids, N, k=K, eos_id=eos,
                             target_kv="base_bf16", log=lambda d: trace.append(d))
    spec = out["gen"]
    d_idx = next((i for i, (a, b) in enumerate(zip(spec, ref)) if a != b), None)
    print(f"  spec={[tok_t.decode([t])[:8] for t in spec]}")
    print(f"  ref ={[tok_t.decode([t])[:8] for t in ref]}")
    print(f"  first diff index = {d_idx}", flush=True)
    if d_idx is None:
        print("  本轮未复现分歧(噪声敏感点); prefix 取 step7 的公共前缀")
        d_idx = 7
    prefix = torch.cat([ids, torch.tensor([ref[:d_idx]], device="cuda")], dim=1)
    want = sorted({spec[d_idx], ref[d_idx]})
    a_tok, b_tok = want[0], want[1]
    print(f"  prefix_len={prefix.shape[1]}  ar_split={split}  "
          f"want={[tok_t.decode([w]) for w in want]}", flush=True)

    d, off, kind = round_of(trace, split, d_idx)
    if d is not None:
        print(f"  [round] r={d['round']} Kp={d['Kp']} m={d['m']} pre={d['pre']} "
              f"off={off} kind={kind}")
        print(f"          cands ={[tok_t.decode([t])[:8] for t in d['cands']]}")
        print(f"          g     ={[tok_t.decode([t])[:8] for t in d['g']]}")
        print(f"          target 在该位置裁决 g[{off}]={tok_t.decode([d['g'][off]])!r} / "
              f"draft 提议 {tok_t.decode([d['cands'][off]])!r}", flush=True)

    lgA = at_prefix_full(tgt, prefix)
    lgB = at_prefix_cached(tgt, prefix)
    lgH = at_prefix_hist(tgt, prefix, split)
    rH = report("bf16 AR-history", lgH, tok_t, want, 7)
    rA = report("bf16 full      ", lgA, tok_t, want, 7)
    rB = report("bf16 cached    ", lgB, tok_t, want, 7)
    print(f"  bf16 full-vs-cached max|Δ| = {(lgA - lgB).abs().max().item():.3e}")
    print(f"  bf16 hist-vs-full   max|Δ| = {(lgH - lgA).abs().max().item():.3e}")
    print(f"  ref[{d_idx}]={tok_t.decode([ref[d_idx]])!r} "
          f"spec[{d_idx}]={tok_t.decode([spec[d_idx]])!r} "
          f"(bf16 AR-history 复现出 ref: {rH[0] == ref[d_idx]})", flush=True)
    bf16 = {"AR-history": rH, "full": rA, "cached": rB}
    bf16_am = [rH[0], rA[0], rB[0]]
    bf16_rows = {"AR-history": lgH.cpu(), "full": lgA.cpu(), "cached": lgB.cpu()}
    del tgt, drf, lgA, lgB, lgH
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[stage] fp32 (CPU, 单模型): full / cached / AR-history", flush=True)
    tgt32, _ = load(TARGET, torch.float32, "cpu")
    p32 = prefix.cpu()
    f_full = at_prefix_full(tgt32, p32)
    f_cach = at_prefix_cached(tgt32, p32)
    f_hist = at_prefix_hist(tgt32, p32, split)
    rC = report("fp32 full      ", f_full, tok_t, want, 23)
    rD = report("fp32 cached    ", f_cach, tok_t, want, 23)
    rE = report("fp32 AR-history", f_hist, tok_t, want, 23)
    del tgt32
    gc.collect()
    fp32_am = [rC[0], rD[0], rE[0]]
    fp32_rows = {"full": f_full, "cached": f_cach, "AR-history": f_hist}

    same_bf16 = len(set(bf16_am)) == 1
    same_fp32 = len(set(fp32_am)) == 1
    signal = rC[1] if rC[1] is not None else 0.0
    print("\n[signal vs noise] 同一候选对的 gap:")
    print(f"  fp32 (真值): {signal:+.4f}  -> argmax={tok_t.decode([rC[0]])!r}")
    noise = 0.0
    for nm, row in bf16_rows.items():
        gb = pair_gap(row, a_tok, b_tok)
        err = abs(gb - signal)
        noise = max(noise, err)
        print(f"  bf16 {nm:<10s} gap={gb:+.4f}  |gap_bf16-gap_fp32|={err:.4f}")
    for nm, row in fp32_rows.items():
        print(f"  fp32 {nm:<10s} gap={pair_gap(row, a_tok, b_tok):+.4f}  "
              f"argmax={tok_t.decode([int(row.argmax())])!r}")
    undecidable = abs(signal) <= noise
    print(f"  => |signal|={abs(signal):.4f} vs noise={noise:.4f}  "
          f"可判定={not undecidable}")

    print("\n判读:")
    print(f"  bf16 三路径同一 argmax: {same_bf16} "
          f"(精确并列={any(r[3] for r in bf16.values())})")
    print(f"  fp32 三路径同一 argmax: {same_fp32}")
    if not same_fp32:
        verdict = "NEEDS INSPECTION (fp32 自身不一致)"
    elif same_bf16 and bf16_am[0] == fp32_am[0]:
        verdict = "NO-SPLIT (bf16 与 fp32 同解; 本位置无争议)"
    else:
        tag = "TIE-COLLAPSE" if same_bf16 else "SHAPE-FLIP"
        if not undecidable:
            tag += "+NEEDS INSPECTION (signal>noise 却仍翻面)"
        verdict = (f"{tag} (bf16 无法分辨该位置: signal={abs(signal):.4f} <= "
                   f"noise={noise:.4f}; fp32 真值={tok_t.decode([fp32_am[0]])!r})")
    print(f"  verdict: {verdict}")
    print(f"  spec[{d_idx}]={tok_t.decode([spec[d_idx]])!r} 与 fp32 真值一致: "
          f"{spec[d_idx] == fp32_am[0]};  ref[{d_idx}]={tok_t.decode([ref[d_idx]])!r} "
          f"与 fp32 真值一致: {ref[d_idx] == fp32_am[0]}", flush=True)
    print(f"[b6-tie] bf16_argmax={bf16_am} fp32_argmax={fp32_am} signal={signal:+.4f} "
          f"noise={noise:.4f} undecidable={undecidable} "
          f"spec_is_fp32_correct={spec[d_idx] == fp32_am[0]} "
          f"verdict={verdict.split()[0]}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
