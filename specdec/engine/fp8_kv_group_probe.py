"""fp8_kv_group_probe.py — FP8 KV 配置诊断探针（D 语义保留; 不覆盖正式脚本）

阶段:
  1) canary: gs=1(K/V) 往返必须与 bf16 逐位一致（group/broadcast 顺序 bug 探测器）;
             gs=128 pilot NLL 应与已知 A 方案吻合（回归哨兵）。
  2) 真实 K/V 逐层指标: 用 base bf16 prefill 的真实 cache K/V, 对每个 group size 离线
     quantize->dequant, 逐层报 MAE/RMSE/max/relL2/cosine(K、V 分开; 全层 + 0/14/27 明细)。
  3) attention 内部离线近似(层 0/14/27, 窗口末 128 行 query): scores/probs 误差(K 影响),
     attn-out relL2(V 影响)。
  4) 引擎 pilot NLL(4 窗 wikitext-2, chunk512, seed 固定): base 与
     (gk,gv) in {(128,128),(64,64),(32,32),(16,16),(1,1),(32,1),(1,32),(128,1),(1,128)}
     全部保留 D 语义(经 set_fp8_groups 显式传参)。
判读: mean NLL 为准; PPL 仅附列。K-only/V-only 用于不对称方案判断。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_kv_group_probe
"""
from __future__ import annotations

import math
import time

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import ensure_data
from .model_self import (Build_Rope_cache, E4M3_MAX, apply_rope, dequantize_kv_fp8,
                         kv_repeat, quantize_kv_fp8)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
WINDOW = 2048
CHUNK = 512
N_WIN = 4
HEAD_DIM = 128          # 1.5B: hidden1536/heads12
GS_LIST = (128, 64, 32, 16, 1)
ATN_LAYERS = (0, 14, 27)


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


def prefill_chunked(ours, ids, chunk=CHUNK, need_logits_last=True):
    """分块 prefill; 返回 (末块 logits, cache). hook 捕获的层输入以最后一次(末块)为准。"""
    cache_list = None
    last_logits = None
    S = ids.shape[1]
    with torch.inference_mode():
        for st in range(0, S, chunk):
            piece = ids[:, st:st + chunk]
            last = (st + chunk >= S)
            lg, cache_list = ours._run(ours.embed_tokens(piece), cache_list,
                                       need_logits=(need_logits_last and last))
            if last:
                last_logits = lg
    return last_logits, cache_list


def snapshot(cache_list):
    return [{kk: vv for kk, vv in d.items()} for d in cache_list]


def window_nll(ours, ids_w, fmt, gk, gv):
    """同窗口以 (fmt, K/V group) 运行, 返回 mean NLL."""
    ours.set_kv_format(fmt)
    if fmt == "fp8":
        # pilot 语义 = 对 K、V 都应用给定 group size（诊断用; 正式默认是 V-only）
        ours.set_fp8_matrices(k_on=True, v_on=True, k_group=gk, v_group=gv)
    cache_list = None
    nll_sum = 0.0
    n_tok = 0
    with torch.inference_mode():
        for gs in range(0, WINDOW, CHUNK):
            piece = ids_w[gs:gs + CHUNK]
            lg, cache_list = ours._run(ours.embed_tokens(piece.unsqueeze(0)),
                                       cache_list, need_logits=True)
            tgt = ids_w[gs + 1: gs + 1 + CHUNK]
            rows = tgt.shape[0]
            ls = torch.log_softmax(lg[0, :rows].to(torch.float32), dim=-1)
            nll_sum += -ls[torch.arange(rows, device=lg.device), tgt].sum().item()
            n_tok += rows
    return nll_sum / n_tok


def kv_metrics(base_bf16, gs):
    """base_bf16: dict entry {"k":(1,2,T,128) bf16,...}; 返回 (K metrics, V metrics)."""
    out = {}
    for key in ("k", "v"):
        x = base_bf16[key].to(torch.float32)
        q, s = quantize_kv_fp8(base_bf16[key], gs)
        hat = dequantize_kv_fp8(q, s, torch.float32)
        e = (hat - x)
        ae = e.abs()
        rel2 = e.norm() / x.norm()
        cos = torch.nn.functional.cosine_similarity(
            hat.flatten(1), x.flatten(1), dim=1).mean().item()
        out[key] = (ae.mean().item(), (e * e).mean().sqrt().item(),
                    ae.max().item(), rel2.item(), cos)
    return out["k"], out["v"]   # (K-metrics tuple, V-metrics tuple)


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(ensure_data(), encoding="utf-8") as f:
        text = f.read()
    ids_all = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    t0 = time.perf_counter()

    ours = load_model()

    # ---- base 运行: 窗0 prefill + 捕获层输入(最后一 chunk) ----
    w0 = ids_all[:WINDOW].cuda()
    caps = {}
    handles = []
    for li in ATN_LAYERS:
        cap = {"x": None}
        handles.append(ours.layers[li].register_forward_hook(
            lambda m, a, o, c=cap: c.__setitem__("x", a[0].detach())))
        caps[li] = cap
    lg_base, cache_base = prefill_chunked(ours, w0.unsqueeze(0))
    base_snap = snapshot(cache_base)
    for h in handles:
        h.remove()

    # ---- [1] canary: gs=1 无损 ----
    ok_lossless = True
    for key in ("k", "v"):
        q1, s1 = quantize_kv_fp8(base_snap[0][key], 1)
        h1 = dequantize_kv_fp8(q1, s1, DTYPE)
        exact = bool((h1 == base_snap[0][key]).all().item())
        print(f"[canary] gs=1 layer0 {key} lossless-exact: {exact}")
        ok_lossless &= exact
    print(f"canary summary: {ok_lossless}")
    if not ok_lossless:
        print("BUG: gs=1 not lossless -> group/reshape order problem; stop here")
        return 1

    # ---- [2] 真实 K/V 逐层指标 (层均 + 0/14/27) ----
    print(f"\n[2] real-KV offline metrics per group size (mean over 28 layers, "
          f"then layers 0/14/27)")
    kv_summary = {}
    for gs in GS_LIST:
        kk, vv = [], []
        for L in range(len(base_snap)):
            km, vm = kv_metrics(base_snap[L], gs)
            kk.append(km), vv.append(vm)
        def mean_of(seq, i):
            return sum(x[i] for x in seq) / len(seq)
        def pick(seq, i, li):
            return seq[li][i]
        kv_summary[gs] = (kk, vv)
        print(f" gs={gs:3d} K: relL2 {mean_of(kk,3):.4f} cos {mean_of(kk,4):.4f} "
              f"| L0 relL2 {pick(kk,3,0):.4f} L14 {pick(kk,3,14):.4f} "
              f"L27 {pick(kk,3,27):.4f} | V: relL2 {mean_of(vv,3):.4f} "
              f"cos {mean_of(vv,4):.4f} | L27 relL2 {pick(vv,3,27):.4f}")

    # ---- [3] attention 内部离线近似 (层 0/14/27, 末 chunk 128 query 行) ----
    print("\n[3] offline attention proxy (query rows 1920..2047 vs keys<=pos)")
    R0 = WINDOW - 128
    for li in ATN_LAYERS:
        lay = ours.layers[li]
        attn = lay.self_attn
        xcap = caps[li]["x"]                              # (1,512,1536) bf16
        xsub = lay.input_layernorm(xcap[:, -128:])        # (1,128,1536)
        q = attn.q_proj(xsub).view(1, 128, attn.heads, attn.head_dim).transpose(1, 2)
        cos, sin = Build_Rope_cache(attn.head_dim, 128, ours.rope_theta,
                                    xsub.device, DTYPE, start_pos=R0)
        q = apply_rope(q, cos, sin)                       # (1,12,128,128) bf16
        q = q.to(torch.float32)
        kb = base_snap[li]["k"].to(torch.float32)         # (1,2,2048,128)
        vb = base_snap[li]["v"].to(torch.float32)
        nrep = attn.heads // attn.kv_heads
        krep = kv_repeat(kb, nrep)                        # (1,12,2048,128)
        vrep = kv_repeat(vb, nrep)
        s_b = torch.matmul(q, krep.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        rows = torch.arange(128, device=q.device)
        mask = rows.view(128, 1) + R0 < torch.arange(2048, device=q.device).view(1, 2048)
        s_b = s_b.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        p_b = torch.softmax(s_b, dim=-1)
        out_b = torch.matmul(p_b, vrep)
        for gs in (128, 32):
            kpol = dequantize_kv_fp8(*quantize_kv_fp8(base_snap[li]["k"], gs),
                                     torch.float32)
            kpolr = kv_repeat(kpol, nrep)
            s_p = torch.matmul(q, kpolr.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            s_p = s_p.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            p_p = torch.softmax(s_p, dim=-1)
            ds = (s_p - s_b).abs()
            ds = ds.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0.0)   # 排除 masked(-inf) 位置
            dp = (p_p - p_b).abs()
            print(f" L{li:2d} gs={gs:3d}: score max/mean {ds.max().item():.2f}/"
                  f"{ds.mean().item():.4f} | prob L1-mean {dp.mean().item():.2e}")
        for gs in (128, 32):
            vpol = dequantize_kv_fp8(*quantize_kv_fp8(base_snap[li]["v"], gs),
                                     torch.float32)
            out_p = torch.matmul(p_b, kv_repeat(vpol, nrep))
            rel = (out_p - out_b).norm() / out_b.norm()
            print(f" L{li:2d} gs={gs:3d} (V only, base probs): attn-out relL2 {rel.item():.4f}")

    # ---- [4] 引擎 pilot NLL ----
    print("\n[4] engine pilot NLL (4 windows x 2048, chunk 512)")
    configs = [("fp16", None, None)] + [("fp8", g, g) for g in (128, 64, 32, 16, 1)] + \
              [("fp8", 32, 1), ("fp8", 1, 32), ("fp8", 128, 1), ("fp8", 1, 128)]
    rows_out = []
    for cfg in configs:
        fmt, gk, gv = cfg
        tot = 0.0
        n = 0
        for wi in range(N_WIN):
            w = ids_all[wi * WINDOW: (wi + 1) * WINDOW].cuda()
            nn = window_nll(ours, w, fmt, gk, gv)
            torch.cuda.empty_cache()
            tot += nn * (WINDOW - 1)
            n += (WINDOW - 1)
        m = tot / n
        # determinism: 窗0 重跑
        d0 = window_nll(ours, ids_all[:WINDOW].cuda(), fmt, gk, gv)
        torch.cuda.empty_cache()
        det = abs(d0 - window_nll(ours, ids_all[:WINDOW].cuda(), fmt, gk, gv)) < 1e-9
        tag = "base " if fmt == "fp16" else f"K{gk}/V{gv}"
        print(f" {tag:8s}: NLL {m:.4f}  PPL {math.exp(m):8.2f}  finite  det {det}")
        rows_out.append((tag, m, det))

    base_m = rows_out[0][1]
    print("\n== delta vs base ==")
    for tag, m, _ in rows_out[1:]:
        print(f" {tag:8s}: dNLL {m - base_m:+.4f}  dPPL% {(math.exp(m) / math.exp(base_m) - 1) * 100:+.1f}")
    print(f"total wall {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
