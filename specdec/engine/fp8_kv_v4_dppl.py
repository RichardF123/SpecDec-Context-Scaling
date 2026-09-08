"""fp8_kv_v4_dppl.py — G3 验证 V4: FP8 KV ΔPPL 质量协议（v7.1 §4.6 + freeze §3）

口径（2026-09-08 审计后显式化, 数学未变 —— 审计结论: 原实现已满足全部 7 点）:
  - 每窗 predicted tokens = WINDOW - 1（首 token 不作目标; 末 token 无下一目标）;
  - label shift: logits 行 r（位置 gs+r）对目标 ids_w[gs+1+r] —— 等价全局
    logits[:, :-1] vs ids[:, 1:];
  - NLL: log_softmax(fp32) 取目标, reduction=sum 逐 chunk 累加, 最后统一
    ppl = exp(total_nll / total_predicted_tokens);  两格式同窗/同 chunk/同 labels;
  - ΔPPL = PPL(fp8) − PPL(fp16); relative ΔPPL% = ΔPPL / PPL(fp16) × 100;
  - 无人工 PASS/FAIL 阈值（freeze 规定）; 若 pooled PPL(fp16) > 1e3（诊断门,
    非质量门）自动打印前 1–2 窗 debug 表。

协议参数:
  - 数据集 Wikitext-2 test (raw); 非重叠固定窗口 W=2048; chunk=512（两格式相同）;
  - windows >= 32（协议最低）。

Run（交互式 GPU 终端）:  python -m specdec.engine.fp8_kv_v4_dppl
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
WINDOW = 2048
CHUNK = 512            # fp16/fp8 同 chunk —— 只让 KV 存储精度成为差异
N_WINDOWS = 40         # >= 协议最低 32
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
            # lg: (1, s, vocab)
            lg32 = lg[0].to(torch.float32)
            finite &= bool(torch.isfinite(lg32).all().item())
            # 目标 = 下一 token: 行 r(位置 gs+r) 对 ids_w[gs+r+1]; 窗口末尾无下一 token
            tgt = ids_w[gs + 1: gs + 1 + CHUNK]
            rows = tgt.shape[0]
            if rows <= 0:
                continue
            ls = torch.log_softmax(lg32[:rows], dim=-1)
            nll_sum += -ls[torch.arange(rows, device=lg.device), tgt].sum().item()
            n_tok += rows
    return nll_sum, n_tok, finite


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    path = ensure_data()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    n_win = min(N_WINDOWS, ids.shape[0] // WINDOW)
    print(f"data: {path}\ntotal tokens={ids.shape[0]}  windows={n_win}  "
          f"W={WINDOW} chunk={CHUNK}  model=1.5B", flush=True)
    if n_win < 32:
        print(f"WARNING: windows={n_win} < protocol min 32", flush=True)

    ours = load_model()
    t0 = time.perf_counter()
    tot16 = tot8 = 0.0
    tot_tok = 0
    finite_all = True
    per = []        # (ppl16_w, ppl8_w)
    dbg = {}        # wi -> 窗口级 debug 信息
    for wi in range(n_win):
        w = ids[wi * WINDOW: (wi + 1) * WINDOW].cuda()
        s16, n16, f16 = window_stats(ours, w, "fp16")
        torch.cuda.empty_cache()
        s8, n8, f8 = window_stats(ours, w, "fp8")
        torch.cuda.empty_cache()
        assert n16 == WINDOW - 1 and n8 == WINDOW - 1, \
            f"window {wi}: predicted tokens {n16}/{n8} != W-1"   # 审计点 4
        tot16 += s16
        tot8 += s8
        tot_tok += n16
        finite_all &= (f16 and f8)
        p16w, p8w = math.exp(s16 / n16), math.exp(s8 / n8)
        per.append((p16w, p8w))
        if wi < 2:
            dbg[wi] = dict(s16=s16, s8=s8, n=n16, f16=f16, f8=f8,
                           first=int(w[0].item()), last=int(w[-1].item()))
        if (wi + 1) % 10 == 0:
            print(f"  window {wi+1}/{n_win} done ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    mean16 = tot16 / tot_tok
    mean8 = tot8 / tot_tok
    ppl16 = math.exp(mean16)
    ppl8 = math.exp(mean8)
    dppl = ppl8 - ppl16
    rel = dppl / ppl16 * 100.0
    ds = torch.tensor([b - a for a, b in per])

    print("\n=== V4 report ===")
    print(f"dataset path      : {path}")
    print(f"total tokens      : {ids.shape[0]}")
    print(f"windows           : {n_win}")
    print(f"window length     : {WINDOW}")
    print(f"chunk size        : {CHUNK}")
    print(f"total predicted   : {tot_tok}")
    print(f"logits finite     : {finite_all}")
    print(f"mean NLL fp16 KV  : {mean16:.6f}")
    print(f"mean NLL fp8 KV   : {mean8:.6f}")
    print(f"PPL(fp16 KV)      : {ppl16:.4f}")
    print(f"PPL(fp8 KV)       : {ppl8:.4f}")
    print(f"ΔPPL              : {dppl:+.4f}")
    print(f"relative ΔPPL%    : {rel:+.3f}")
    print(f"per-window ΔPPL   : mean={ds.mean():+.4f}  min={ds.min():+.4f}  "
          f"max={ds.max():+.4f}")
    print(f"total wall        : {time.perf_counter()-t0:.0f}s")

    if ppl16 > DIAG_PPL:      # 诊断门: 只负责打印, 不作质量判定
        print("\n=== debug table (first 1-2 windows) ===")
        for wi in sorted(dbg):
            d = dbg[wi]
            print(f"window {wi}: tokens={WINDOW}  predicted={d['n']}  "
                  f"nll_sum16={d['s16']:.4f}  nll_sum8={d['s8']:.4f}  "
                  f"mean16={d['s16']/d['n']:.4f}  ppl16={math.exp(d['s16']/d['n']):.2f}  "
                  f"finite16={d['f16']} finite8={d['f8']}")
            print(f"           first label={d['first']} ({tok.decode([d['first']])[:20]!r})  "
                  f"last label={d['last']} ({tok.decode([d['last']])[:20]!r})")

    print("\nV4: REPORT (no artificial pass/fail; numbers recorded per protocol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
