"""fp8_kv_v2_short_window.py — G3 验证 V2: fp8-vs-fp16 短窗行为（CUDA, 需引擎已实现 set_kv_format）

前置: 引擎已按 design_freeze_fp8_kv.md §4 + §4.1 实现 FP8 KV（model.set_kv_format("fp8")）。
     未实现会抛 AttributeError —— 先实现引擎, 再跑本 harness。

流程:
  - 同一 prompt (512 tok), chunked prefill (chunk 1024) 分别用 fp16 / fp8 建缓存;
  - 从各自末位置 logits 贪心 decode 32 tok, 逐 token 记录 logits 与 token;
对比/记录:
  [1] 结构检查: fp8 cache 的 "k"/"v" dtype == float8_e4m3fn, 有 k_scale/v_scale, fmt=="fp8"
  [2] prefill 末位置 logits: fp8 vs fp16 max|Δ|（整段历史 fp8 舍入对下一 token 分布的总效应）
  [3] decode 32 步: 首分歧位置 / 一致前缀长度; 步 0/1/5/10/31 的 logits max|Δ|
  [4] fp8 确定性: 同输入 decode 两次逐位一致
硬性判据（预注册）: 无 NaN/Inf; fp8 结构正确; fp8 确定性; 引擎两格式均跑完。
分歧数字只记录不设阈值（G3 门定义）; 若首分歧 < 4 且 logits 差异常大 → 另行归因。

Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v2_short_window
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .g2_operational import make_text

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
PROMPT = 512
CHUNK = 1024
DECODE_N = 32


def build(ids, fmt: str):
    """建缓存并贪心 decode, 返回 (tokens, per_step_logits[list of fp16 (1,vocab) cuda]) 与缓存."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
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

    eos = tok.eos_token_id
    cache_list = None
    S = ids.shape[1]
    last_logits = None
    with torch.inference_mode():
        for st in range(0, S, CHUNK):
            piece = ids[:, st:st + CHUNK]
            last = (st + CHUNK >= S)
            lg, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                       need_logits=last)
            if last:
                last_logits = lg
    torch.cuda.synchronize()

    toks: list[int] = []
    logs: list[torch.Tensor] = []
    cur = last_logits
    with torch.inference_mode():
        for _ in range(DECODE_N):
            logs.append(cur[:, -1].detach().clone())
            nxt = int(cur[:, -1].argmax())
            if nxt == eos:
                break
            toks.append(nxt)
            cur, cache_list = ours._run(
                ours.embed_tokens(torch.tensor([[nxt]], device=ids.device)),
                cache_list)
    torch.cuda.synchronize()
    return tok, toks, logs, cache_list


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = make_text(tok, PROMPT).to("cuda")
    print(f"prompt={PROMPT} tok  chunk={CHUNK}  decode={DECODE_N}  model=1.5B dtype={DTYPE}\n", flush=True)

    # fp16 一次; fp8 两次（确定性）
    print("running fp16 ...", flush=True)
    _, toks16, logs16, cache16 = build(ids, "fp16")
    print("running fp8 (pass A) ...", flush=True)
    _, toks8a, logs8a, cache8 = build(ids, "fp8")
    print("running fp8 (pass B) ...", flush=True)
    _, toks8b, logs8b, _ = build(ids, "fp8")

    ok = True
    # [1] 结构检查（正式非对称: K bf16 无 scale, V fp8 + v_scale (B,kv,S,g)）
    from .model_self import GROUP_SIZE
    c0 = cache8[0]
    d = c0["k"].shape[-1]
    g = d // GROUP_SIZE
    stru = (c0.get("fmt") == "fp8"
            and c0["k"].dtype == DTYPE                      # K: bf16（不量化）
            and "k_scale" not in c0
            and c0["v"].dtype == torch.float8_e4m3fn        # V: fp8
            and c0["v_scale"].shape[:3] == c0["v"].shape[:3]
            and c0["v_scale"].shape[-1] == g
            and c0["v_scale"].dtype == torch.float32)
    print(f"[1] fp8 structure ok (asymmetric V-only): {stru}  (head_dim={d} "
          f"v_groups={g}, v_scale={tuple(c0['v_scale'].shape)})")
    ok &= stru

    # [2] prefill 末位置 logits（各自第一次 decode 的 logits 即 prefill 末位置）
    d0 = (logs16[0].to(torch.float32) - logs8a[0].to(torch.float32)).abs().max().item()
    print(f"[2] prefill last-pos logits fp16-vs-fp8 max|Δ| = {d0:.3e}")
    fin = all(torch.isfinite(l).all().item() for l in logs8a) and \
          all(torch.isfinite(l).all().item() for l in logs16)
    print(f"    all logits finite: {fin}")
    ok &= fin

    # [3] decode 分歧
    agree = 0
    for a, b in zip(toks16, toks8a):
        if a == b:
            agree += 1
        else:
            break
    print(f"[3] tokens fp16: {tok.decode(toks16)[:50]!r}")
    print(f"    tokens fp8 : {tok.decode(toks8a)[:50]!r}")
    print(f"    agreeing prefix: {agree}/{min(len(toks16), len(toks8a))}  "
          f"(len16={len(toks16)} len8={len(toks8a)})")
    for k in (0, 1, 5, 10, min(31, len(logs16) - 1)):
        dk = (logs16[k].to(torch.float32) - logs8a[k].to(torch.float32)).abs().max().item()
        print(f"    step {k}: logits max|Δ| = {dk:.3e}")

    # [4] fp8 确定性
    det = toks8a == toks8b
    print(f"[4] fp8 deterministic (two passes identical): {det}")
    ok &= det

    print("\nV2:", "PASS" if ok else "CHECK",
          "| structure/deterministic/finite only; divergence recorded, not gated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
