"""fp8_kv_v4_dppl.py — G3 验证 V4: FP8 KV ΔPPL 质量协议（v7.1 §4.6 + freeze §3）

协议:
  - 数据集 Wikitext-2 **test**（raw）; 非重叠固定窗口, 窗长 W=2048;
  - 同一窗口分别以 fp16-KV / fp8-KV 运行（chunk 同为 512, 只差 KV 存储精度）;
  - 每窗 PPL = exp(mean NLL over 窗内可预测位置); 跨窗 pooled mean → absolute PPL;
  - 报告: PPL(fp16)、PPL(fp8)、ΔPPL = ppl8 − ppl16、relative ΔPPL% ; **无人工阈值**（freeze 规定）;
  - 硬性: windows >= 32（协议最低）; 无 NaN; 两格式同窗同 chunk。

数据: 自动下载 wikitext-2-raw-v1.zip 到 specdec/data/（若失败, 手动下载后解压出
      wiki.test.raw 放到 data/wikitext-2-raw/ 再跑）。

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
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data")
RAW_DIR = os.path.join(DATA_DIR, "wikitext-2-raw")
TEST_RAW = os.path.join(RAW_DIR, "wiki.test.raw")
ZIP_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"


def ensure_data() -> str:
    if os.path.exists(TEST_RAW):
        return TEST_RAW
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "wikitext-2-raw-v1.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {ZIP_URL} ...", flush=True)
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("wikitext-2-raw/wiki.test.raw", DATA_DIR)
    return TEST_RAW


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


def window_mean_nll(ours, ids_w: torch.Tensor, fmt: str) -> float:
    """同一窗口(fixed ids)以 fmt KV 运行, 返回 mean NLL（nats/token, pooled over 窗内位置）."""
    ours.set_kv_format(fmt)
    cache_list = None
    nll_sum = 0.0
    n_tok = 0
    S = ids_w.shape[0]
    with torch.inference_mode():
        for gs in range(0, S, CHUNK):
            piece = ids_w[gs:gs + CHUNK]
            lg, cache_list = ours._run(
                ours.embed_tokens(piece.unsqueeze(0)), cache_list, need_logits=True)
            # lg: (1, s, vocab); 目标: ids_w[gs+1 : gs+s+1]（窗口末尾无下一 token → 少一行）
            tgt = ids_w[gs + 1: gs + 1 + CHUNK]
            rows = tgt.shape[0]
            if rows <= 0:
                continue
            ls = torch.log_softmax(lg[0, :rows].to(torch.float32), dim=-1)
            nll = -ls[torch.arange(rows, device=lg.device), tgt].sum().item()
            nll_sum += nll
            n_tok += rows
    return nll_sum / n_tok


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
    ppl16_sum_nll = 0.0
    ppl8_sum_nll = 0.0
    n_tot = 0
    deltas = []
    for wi in range(n_win):
        w = ids[wi * WINDOW: (wi + 1) * WINDOW].cuda()
        nll16 = window_mean_nll(ours, w, "fp16")
        torch.cuda.empty_cache()
        nll8 = window_mean_nll(ours, w, "fp8")
        torch.cuda.empty_cache()
        ppl16_sum_nll += nll16 * (WINDOW - 1)
        ppl8_sum_nll += nll8 * (WINDOW - 1)
        n_tot += (WINDOW - 1)
        deltas.append(math.exp(nll8) - math.exp(nll16))
        if (wi + 1) % 10 == 0:
            print(f"  window {wi+1}/{n_win} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    ppl16 = math.exp(ppl16_sum_nll / n_tot)
    ppl8 = math.exp(ppl8_sum_nll / n_tot)
    dppl = ppl8 - ppl16
    rel = dppl / ppl16 * 100.0
    ds = torch.tensor(deltas)
    print("\n=== V4 report ===")
    print(f"PPL(fp16 KV) = {ppl16:.4f}")
    print(f"PPL(fp8 KV)  = {ppl8:.4f}")
    print(f"ΔPPL = {dppl:+.4f}   (relative {rel:+.3f}%)")
    print(f"per-window ΔPPL: mean={ds.mean():+.4f}  min={ds.min():+.4f}  "
          f"max={ds.max():+.4f}")
    print(f"total wall {time.perf_counter()-t0:.0f}s")
    print("\nV4: REPORT (no artificial pass/fail; numbers recorded per protocol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
