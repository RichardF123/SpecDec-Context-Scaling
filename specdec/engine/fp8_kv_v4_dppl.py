"""fp8_kv_v4_dppl.py — G3 正式质量门 V4: base(BF16 KV) vs Asymmetric V-only FP8(K BF16 + V FP8 gs128)

正式条件（2026-09-08 定案）:
  A: K BF16 + V BF16          -> set_kv_format("fp16")（计算/质量基线条目 = bf16）
  B: K BF16 + V FP8 E4M3 gs128 -> set_kv_format("v_fp8")（正式非对称方案）
口径（审计后显式化, 数学未变）:
  - 每窗 predicted tokens = WINDOW - 1（首 token 不作目标; 末 token 无下一目标）;
  - label shift: logits 行 r（位置 gs+r）对目标 ids_w[gs+r+1]; NLL: log_softmax(fp32) 取目标,
    reduction=sum 累加, ppl = exp(total_nll / total_predicted_tokens); 同窗/同 chunk/同 labels;
  - ΔNLL 为主判据, ΔPPL(absolute + relative %) 辅助; 无人工 PASS/FAIL 阈值;
  - full FP8 K/V 与 K-only 均为 pilot 级 negative ablation, 不在此跑 40 窗。

正式窗口与 pilot 不相交:
  - pilot（fp8_kv_group_probe, N_WIN=4）: token 偏移 0..3 x 2048
  - 本脚本正式确认集: token 偏移 40..(40+N_WINDOWS-1) x 2048, 非重叠, disjoint = True
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.fp8_kv_v4_dppl
"""
from __future__ import annotations

import math
import os
import time
import urllib.request
import zipfile

import torch

from .align_ours import build_from_config

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16     # CUDA fp16 produces NaNs for this Qwen path; bf16 is stable on RTX 4060.
WINDOW = 2048
CHUNK = 512            # base/v_fp8 同 chunk —— 只让 V 存储精度成为差异
LOSS_ROWS = 64         # loss 按位置小块算，避免整块 logits 转 fp32 造成 OOM
N_WINDOWS = 40         # >= 协议最低 32
WINDOW_OFFSET = 40     # 正式确认窗口: 40..79（与 pilot 0..3 不相交, 见模块 docstring）
DIAG_PPL = 1e3         # 诊断门（仅触发 debug 表, 不作质量判定）
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data")
RAW_DIR = os.path.join(DATA_DIR, "wikitext-2-raw")
TEST_RAW = os.path.join(RAW_DIR, "wiki.test.raw")
# 候选下载源（依次尝试）: llama.cpp CI 镜像 + hf-mirror 等价路径
ZIP_URLS = [
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
    "https://hf-mirror.com/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
]


def ensure_data() -> str:
    if os.path.exists(TEST_RAW):
        return TEST_RAW
    os.makedirs(RAW_DIR, exist_ok=True)
    last_err = None
    for url in ZIP_URLS:
        zip_path = os.path.join(DATA_DIR, os.path.basename(url).split("?")[0])
        try:
            print(f"downloading {url} ...", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extract("wikitext-2-raw/wiki.test.raw", DATA_DIR)
            print("ok", flush=True)
            return TEST_RAW
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed ({e}); trying next source", flush=True)
    raise RuntimeError(
        "all download sources failed; last error: " + repr(last_err) +
        "\nmanual fix: download wikitext-2-raw-v1.zip and extract "
        "wikitext-2-raw/wiki.test.raw into " + RAW_DIR)


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


def window_stats(ours, ids_w: torch.Tensor, fmt: str) -> tuple[float, int, bool]:
    """同窗 fixed ids、fmt KV 下跑分块 prefill（need_logits 每 chunk）,
    返回 (nll_sum, n_tok, logits_finite).  label shift 见模块 docstring."""
    ours.set_kv_format(fmt)
    cache_list = None
    nll_sum = 0.0
    n_tok = 0
    finite = True
    S = ids_w.shape[0]
    with torch.inference_mode():
        for gs in range(0, S, CHUNK):
            piece = ids_w[gs:gs + CHUNK]
            lg, cache_list = ours._run(
                ours.embed_tokens(piece.unsqueeze(0)), cache_list, need_logits=True)
            # lg: (1, s, vocab) fp16; finite 直接在 fp16 上查（不造 fp32 全量副本）
            finite &= bool(torch.isfinite(lg).all().item())
            # 目标 = 下一 token: 行 r(位置 gs+r) 对 ids_w[gs+r+1]; 窗口末尾无下一 token
            tgt = ids_w[gs + 1: gs + 1 + CHUNK]
            rows = tgt.shape[0]
            if rows <= 0:
                continue
            # 只按少量位置转 fp32/log_softmax，PPL 口径不变，峰值显存低很多。
            for rs in range(0, rows, LOSS_ROWS):
                re = min(rs + LOSS_ROWS, rows)
                ls = torch.log_softmax(lg[0, rs:re].to(torch.float32), dim=-1)
                idx = torch.arange(re - rs, device=lg.device)
                nll_sum += -ls[idx, tgt[rs:re]].sum().item()
                del ls, idx
            n_tok += rows
            del lg
    return nll_sum, n_tok, finite


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    path = ensure_data()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    n_avail = max(0, (ids.shape[0] - WINDOW_OFFSET * WINDOW) // WINDOW)
    n_win = min(N_WINDOWS, n_avail)
    offs = [WINDOW_OFFSET + wi for wi in range(n_win)]
    print(f"data: {path}\ntotal tokens={ids.shape[0]}  W={WINDOW} chunk={CHUNK} "
          f"model=1.5B dtype={DTYPE}", flush=True)
    print(f"pilot window offsets : 0..3 (fp8_kv_group_probe, N_WIN=4)")
    print(f"formal window offsets: {offs[0]}..{offs[-1]}  (n={n_win})")
    print(f"disjoint = {True if offs[0] > 3 else False}   [A=base/bf16 KV | "
          f"B=K bf16 + V FP8 E4M3 gs128 (v_fp8)]", flush=True)
    if n_win < 32:
        print(f"WARNING: windows={n_win} < protocol min 32", flush=True)

    ours = load_model()
    t0 = time.perf_counter()
    tot_b = tot_v = 0.0
    tot_tok = 0
    finite_all = True
    per = []        # (ppl_base_w, ppl_vonly_w)
    dbg = {}
    for wi in range(n_win):
        w = ids[(WINDOW_OFFSET + wi) * WINDOW: (WINDOW_OFFSET + wi + 1) * WINDOW].cuda()
        sb, nb, fb = window_stats(ours, w, "fp16")     # A: base / bf16 KV
        torch.cuda.empty_cache()
        sv, nv, fv = window_stats(ours, w, "v_fp8")    # B: V-only FP8 (formal)
        torch.cuda.empty_cache()
        assert nb == WINDOW - 1 and nv == WINDOW - 1, \
            f"window {WINDOW_OFFSET+wi}: predicted {nb}/{nv} != W-1"
        tot_b += sb
        tot_v += sv
        tot_tok += nb
        finite_all &= (fb and fv)
        pb, pv = math.exp(sb / nb), math.exp(sv / nv)
        per.append((pb, pv))
        if wi < 2:
            dbg[wi] = dict(sb=sb, sv=sv, n=nb, fb=fb, fv=fv,
                           first=int(w[0].item()), last=int(w[-1].item()))
        if (wi + 1) % 10 == 0:
            print(f"  window {WINDOW_OFFSET+wi+1}/{WINDOW_OFFSET+n_win} done "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    mean_b = tot_b / tot_tok
    mean_v = tot_v / tot_tok
    ppl_b = math.exp(mean_b)
    ppl_v = math.exp(mean_v)
    dppl = ppl_v - ppl_b
    rel = dppl / ppl_b * 100.0
    ds = torch.tensor([b - a for a, b in per])

    print("\n=== V4 report (formal: A=base/bf16 KV, B=V-only FP8 gs128) ===")
    print(f"dataset path      : {path}")
    print(f"total tokens      : {ids.shape[0]}")
    print(f"windows           : {n_win} (offsets {offs[0]}..{offs[-1]})")
    print(f"window length     : {WINDOW}")
    print(f"chunk size        : {CHUNK}")
    print(f"total predicted   : {tot_tok}")
    print(f"logits finite     : {finite_all}")
    print(f"mean NLL base     : {mean_b:.6f}")
    print(f"mean NLL V-only   : {mean_v:.6f}")
    print(f"dNLL (V-only - base): {mean_v - mean_b:+.6f}")
    print(f"PPL(base KV)      : {ppl_b:.4f}")
    print(f"PPL(V-only FP8)   : {ppl_v:.4f}")
    print(f"absolute dPPL     : {dppl:+.4f}")
    print(f"relative dPPL%    : {rel:+.3f}")
    print(f"per-window dPPL   : mean={ds.mean():+.4f}  min={ds.min():+.4f}  "
          f"max={ds.max():+.4f}")
    print(f"total wall        : {time.perf_counter()-t0:.0f}s")

    if ppl_b > DIAG_PPL:      # 诊断门: 只负责打印, 不作质量判定
        print("\n=== debug table (first 2 formal windows) ===")
        for wi in sorted(dbg):
            d = dbg[wi]
            print(f"window {WINDOW_OFFSET+wi}: tokens={WINDOW} predicted={d['n']}  "
                  f"nll_sum_base={d['sb']:.4f} nll_sum_vonly={d['sv']:.4f}  "
                  f"mean_base={d['sb']/d['n']:.4f} ppl_base={math.exp(d['sb']/d['n']):.2f}  "
                  f"finite_base={d['fb']} finite_vonly={d['fv']}")
            print(f"           first label={d['first']} ({tok.decode([d['first']])[:20]!r})  "
                  f"last label={d['last']} ({tok.decode([d['last']])[:20]!r})")

    print("\nV4: REPORT (no artificial pass/fail; quality judged on dNLL per protocol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
