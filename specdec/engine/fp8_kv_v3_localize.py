"""fp8_kv_v3_localize.py — V3 分叉定位(修正版): prefill 与 decode 分离, 纯 prefill KV 比较

口径修正(v2):
  - prefill 4096 后立即 snapshot cache(浅拷贝 dict; decode 的 cat 只 rebind 键、
    不改旧张量 → 快照即纯 prefill 状态, 不含任何生成 token);
  - KV 比较只在快照上做; decode16 用原 cache(可被原地修改), 与比较隔离;
  - K 与 V 分开报告(不再只报二者 max); 每对组合给: 逐层 max、首个 >1e-2 的 layer
    及其首个越界/最大差异的 token position;
  - decode 分歧单独报告(含 同格式跨 chunk 前缀, 这才是 chunk 耦合的生成级表现);
  - 全 ASCII 输出(Windows GBK 安全)。

比较组合(纯 prefill 4096):
  P1 base bf16 chunk1024 vs base bf16 chunk512   -> base 自身 chunk 漂移
  P2 fp8   chunk1024 vs fp8   chunk512           -> fp8 chunk 漂移
  P3 fp8   chunk1024 vs base  chunk1024          -> 纯 FP8 量化 KV 误差(无 chunk 混淆)
  P4 fp8   chunk512 vs fp8   chunk512(重复)      -> 确定性(期望全 0)
判读: 关注 fp8 是否在 base drift 之外新增明显漂移, 以及该漂移是否改变生成;
      base bf16 不同 GEMM 形状存在数值漂移属预期, 不要求 bitwise 相同。

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
THR = 1e-2


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


def prefill(ours, ids, fmt: str, chunk: int):
    """分块 prefill 4096; 返回 (cache_list, last_logits). 不做任何 decode."""
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
    return cache_list, last_logits


def snapshot(cache_list):
    """浅拷贝每层 dict —— 之后 decode 原地改原 dict 的键, 不触快照中的旧张量."""
    return [{kk: vv for kk, vv in d.items()} for d in cache_list]


def decode(ours, last_logits, cache_list, eos: int, n: int):
    """贪心 decode n tok; 允许原地修改 cache_list(与快照隔离)."""
    toks = []
    cur = last_logits
    with torch.inference_mode():
        for _ in range(n):
            nxt = int(cur[:, -1].argmax())
            if nxt == eos:
                break
            toks.append(nxt)
            cur, cache_list = ours._run(
                ours.embed_tokens(torch.tensor([[nxt]], device=cur.device)),
                cache_list)
    return toks


def to_f32(kv_tensor):
    return kv_tensor.to(torch.float32)


def layer_tensor(entry, key):
    """返回第 entry 层 K 或 V 的 fp32 值(按需反量化)."""
    t = entry[key]
    if entry.get("fmt") == "fp8":
        return dequantize_kv_fp8(t, entry[key + "_scale"], torch.float32)
    return t.to(torch.float32)


def pair_stats(snap_a, snap_b):
    """逐层 K/V max diff + 首越界层定位. 返回 (dk_list, dv_list, info_str)."""
    L = len(snap_a)
    dk, dv = [], []
    lines = []
    first_k = first_v = None
    for i in range(L):
        dki = (layer_tensor(snap_a[i], "k") - layer_tensor(snap_b[i], "k")).abs()
        dvi = (layer_tensor(snap_a[i], "v") - layer_tensor(snap_b[i], "v")).abs()
        dk.append(dki.max().item())
        dv.append(dvi.max().item())
        if first_k is None and dki.max().item() > THR:
            first_k = _loc(dki, i)
        if first_v is None and dvi.max().item() > THR:
            first_v = _loc(dvi, i)
    if first_k:
        lines.append("K first-layer>1e-2: " + first_k)
    else:
        lines.append("K first-layer>1e-2: none")
    if first_v:
        lines.append("V first-layer>1e-2: " + first_v)
    else:
        lines.append("V first-layer>1e-2: none")
    return dk, dv, lines


def _loc(diff, layer):
    """diff (1,2,T,64); 返回首个>THR的 pos 与最大差 pos."""
    over = (diff > THR).nonzero()
    pos_first = int(over[0, 2].item()) if over.shape[0] else -1
    flat = diff.flatten()
    arg = int(flat.argmax().item())
    h = arg // (diff.shape[2] * diff.shape[3])
    rem = arg % (diff.shape[2] * diff.shape[3])
    pos_max = rem // diff.shape[3]
    return (f"layer {layer}: first>1e-2 at token {pos_first}; "
            f"max={flat.max().item():.3e} at (head {h}, token {pos_max})")


def fmt_row(vals):
    return " ".join(f"{v:.2f}" for v in vals)


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos = tok.eos_token_id
    ids = make_text(tok, PROMPT).to("cuda")

    ours = load_model()
    runs = {
        "A_base_1024": ("fp16", 1024),
        "B_base_512": ("fp16", 512),
        "C_fp8_1024": ("fp8", 1024),
        "D_fp8_512": ("fp8", 512),
        "E_fp8_512_rep": ("fp8", 512),
    }
    snaps, lg_map, seqs = {}, {}, {}
    for name, (fmt, ch) in runs.items():
        torch.cuda.empty_cache()
        cache, lg = prefill(ours, ids, fmt, ch)
        snaps[name] = snapshot(cache)
        lg_map[name] = lg
        seqs[name] = decode(ours, lg, cache, eos, DECODE_N)   # 改原 cache, 快照不受影响
        print(f"{name}: prefill done, decode16 -> "
              f"{tok.decode(seqs[name])[:56]!r} ({len(seqs[name])} tok)", flush=True)
        del cache
        torch.cuda.empty_cache()

    pairs = [
        ("P1 base chunk drift (A1024 vs B512)", "A_base_1024", "B_base_512"),
        ("P2 fp8   chunk drift (C1024 vs D512)", "C_fp8_1024", "D_fp8_512"),
        ("P3 fp8-vs-base same chunk 1024 (C vs A)", "C_fp8_1024", "A_base_1024"),
        ("P4 determinism (D512 vs E512)", "D_fp8_512", "E_fp8_512_rep"),
    ]
    for label, na, nb in pairs:
        dk, dv, info = pair_stats(snaps[na], snaps[nb])
        print(f"\n[{label}]")
        print(f"  K per-layer max: {fmt_row(dk)}")
        print(f"  V per-layer max: {fmt_row(dv)}")
        for line in info:
            print("  " + line)

    # fp8 相对 base 新增的 chunk coupling(逐层差)
    dk_b, dv_b, _ = pair_stats(snaps["A_base_1024"], snaps["B_base_512"])
    dk_f, dv_f, _ = pair_stats(snaps["C_fp8_1024"], snaps["D_fp8_512"])
    ex_k = max(f - b for f, b in zip(dk_f, dk_b))
    ex_v = max(f - b for f, b in zip(dv_f, dv_b))
    print(f"\nfp8 extra chunk coupling over base: K +{ex_k:.3f}  V +{ex_v:.3f} "
          f"(small => D unified semantics; residual = base bf16 GEMM-shape drift)")

    # decode 一致前缀(生成级 chunk 耦合 / 与 base 的差异)
    print("\n[decode16] agreeing prefix:")
    baseA = seqs["A_base_1024"]
    for name in ["A_base_1024", "B_base_512", "C_fp8_1024", "D_fp8_512", "E_fp8_512_rep"]:
        a = 0
        for x, y in zip(baseA, seqs[name]):
            if x == y:
                a += 1
            else:
                break
        print(f"    vs base-1024  {name:14}: {a}/{min(len(baseA), len(seqs[name]))}")
    for na, nb, lab in [("A_base_1024", "B_base_512", "base 1024 vs base 512"),
                        ("C_fp8_1024", "D_fp8_512", "fp8 1024 vs fp8 512"),
                        ("D_fp8_512", "E_fp8_512_rep", "fp8 512 vs fp8 512 rep")]:
        a = 0
        for x, y in zip(seqs[na], seqs[nb]):
            if x == y:
                a += 1
            else:
                break
        print(f"    {lab:28}: prefix {a}/{min(len(seqs[na]), len(seqs[nb]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
