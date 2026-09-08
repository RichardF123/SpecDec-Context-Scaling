"""fp8_vonly_v1_roundtrip.py — 正式 Asymmetric V-only FP8 (gs=128) 直接结构/往返测试

最小、明确的 v_fp8 直接验证(不重复实现算法;引擎即被测对象):
  [1] K: cache 中 BF16, 且与同 prompt 的 base(fp16 fmt, bf16 KV) 运行的 K 逐位相同
      -> "K 数值不被量化/不参与 FP8"的直接证据
  [2] V payload dtype = float8_e4m3fn; 无 k_scale; fmt == "v_fp8"
  [3] v_scale: dtype fp32, shape (B, kv_heads, tokens, 1)   [group=128 -> 1 group/head]
  [4] helper 级 gs128: 全零组往返严格 0; max 元素 -> |448|; 确定性
  [5] engine 级确定性: 同输入两次 v_fp8 运行, payload/scale 逐位一致 + decode 相同
  [6] D 一致性: attention 用反量化视图(经 V2 行为项覆盖; 此处校验 prefill 后
      cache["v"] 与 scale 维度拼接正确 -> tokens 沿 dim=2 增长)
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_vonly_v1_roundtrip
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config
from .model_self import E4M3_MAX, GROUP_V, dequantize_kv_fp8, quantize_kv_fp8

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
PROMPT = 96
CHUNK = 48


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


def prefill(ours, ids, fmt: str, chunk=CHUNK):
    ours.set_kv_format(fmt)
    cache_list = None
    with torch.inference_mode():
        for st in range(0, ids.shape[1], chunk):
            piece = ids[:, st:st + chunk]
            _, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                      need_logits=False)
    return cache_list


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("The history of artificial intelligence and neural networks spans many "
              "decades of research into learning systems that process language with "
              "attention over long sequences of tokens.", return_tensors="pt").input_ids
    ids = ids[:, :PROMPT].cuda()

    ours = load_model()
    ok = True

    cache_base = prefill(ours, ids, "fp16")          # base / bf16 KV
    cache_v1 = prefill(ours, ids, "v_fp8")           # 正式 V-only 第一遍
    cache_v2 = prefill(ours, ids, "v_fp8")           # 第二遍(确定性)

    c0 = cache_v1[0]
    cb = cache_base[0]
    # [1] K 数值不被量化: v_fp8 的 K 与 base 的 K 逐位相同
    k_untouched = bool((c0["k"] == cb["k"]).all().item())
    print(f"[1] K bf16 & identical to base-run K: {k_untouched}  "
          f"(k dtype={c0['k'].dtype}, base k dtype={cb['k'].dtype})")
    ok &= k_untouched and c0["k"].dtype == DTYPE

    # [2]/[3] 结构
    d = c0["k"].shape[-1]
    g = d // GROUP_V
    stru = (c0.get("fmt") == "v_fp8"
            and c0["v"].dtype == torch.float8_e4m3fn
            and "k_scale" not in c0
            and c0["v_scale"].shape[:3] == c0["v"].shape[:3]
            and c0["v_scale"].shape[-1] == g
            and c0["v_scale"].dtype == torch.float32)
    print(f"[2/3] v_fp8 structure ok: {stru}  (head_dim={d} v_groups={g}, "
          f"v_scale={tuple(c0['v_scale'].shape)})")
    ok &= stru

    # [4] helper 级 gs=GROUP_V: 全零精确 / max->448 / deterministic
    z = torch.zeros(1, 2, 16, 128, device="cuda", dtype=DTYPE)
    qz, sz = quantize_kv_fp8(z, GROUP_V)
    hz = dequantize_kv_fp8(qz, sz, DTYPE)
    zero_ok = bool(hz.abs().max().item() == 0.0 and torch.isfinite(hz).all().item())
    x = torch.randn(1, 2, 16, 128, device="cuda").abs() + 1e-3
    q1, s1 = quantize_kv_fp8(x.to(DTYPE), GROUP_V)
    q2, s2 = quantize_kv_fp8(x.to(DTYPE), GROUP_V)
    det_h = torch.equal(q1, q2) and torch.equal(s1, s2)
    # 每个非零组最大元素应映射到 |448|
    has_nz = (s1 > 0)
    map448 = bool((q1.to(torch.float32).abs()
                   .amax(dim=(-1, -2)).flatten()[has_nz.flatten()] == 448.0).all().item()
                  if has_nz.any().item() else True)
    print(f"[4] helper gs128: all-zero exact & finite: {zero_ok} | deterministic: {det_h} "
          f"| scale shape {tuple(s1.shape)} | nonzero groups max->448: {map448}")
    ok &= zero_ok and det_h and tuple(s1.shape) == (1, 2, 16, 1)

    # [5] engine 级确定性
    det_eng = all(
        torch.equal(cache_v1[i]["v"], cache_v2[i]["v"])
        and torch.equal(cache_v1[i]["v_scale"], cache_v2[i]["v_scale"])
        and torch.equal(cache_v1[i]["k"], cache_v2[i]["k"])
        for i in range(len(cache_v1)))
    print(f"[5] v_fp8 engine deterministic (two prefills bitwise): {det_eng}")
    ok &= det_eng

    # [6] 拼接语义: token 维 dim=2 正确增长
    T0 = cache_v1[0]["v"].shape[2]
    tok_ok = (T0 == PROMPT) and all(l["v"].shape[2] == T0 for l in cache_v1) and \
        all(l["v_scale"].shape[2] == T0 for l in cache_v1)
    print(f"[6] cache tokens along dim=2: v={cache_v1[0]['v'].shape[2]} "
          f"(expected {PROMPT}), v_scale={cache_v1[0]['v_scale'].shape[2]}: {tok_ok}")
    ok &= tok_ok

    print("\nV-only V1:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
