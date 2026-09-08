"""g2_operational.py — 正式 G2 门: 8K Operational Feasibility（v7.1 §7 + P11）

对照 v7.1 §7 的 G2 定义逐项落地:
  [x] 1.5B + 8K + chunked prefill 不 OOM
  [x] peak VRAM 记录（prefill 后 / 全程两笔）
  [x] 规定 decode 长度 = 128 tok（冻结生成协议 §4.2 定长；遇 eos 停但记录长度）
  [x] 连续多次稳定 = 3 runs 序列逐 token 一致
  [x] correctness sanity = chunk 消融（同一 8K 输入, chunk=2048 vs chunk=512 建缓存:
      逐层 K/V max|Δ| 记录量级 + 各自续 decode 16 tok 行为等价）
  [x] 温度/时钟记录（P11: decode 期间采样 nvidia-smi GPU 温度 + 每 run 时间戳/时长）

判据（预注册, 不事后改）:
  - 稳定性: 3 runs 生成序列全同 → PASS；有分叉 → 打印 run 号, CHECK
  - 显存:   peak VRAM < 7.5 GiB → PASS
  - sanity: chunk512/2048 两缓存的 16-tok decode 序列一致 → PASS；
            数值 max|Δ| 只记录量级（fp16 8K 有逐层累积取整漂移, 不硬卡 1e-3）
  - 温度:   只记录 min/max（P11 要求记录, 不设人工阈值）

Run（交互式 GPU 终端）:  python -m specdec.engine.g2_operational
"""
from __future__ import annotations

import subprocess
import time

import torch

from .align_ours import build_from_config

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CTX = 8192
CHUNK = 2048                # 主运行 prefill 分块（与 g2_feasibility 一致）
DECODE_N = 128              # 冻结生成协议定长（§4.2, max_new_tokens）
RUNS = 3                    # 连续多次稳定（P9 最低 3）
TEMP_EVERY = 16             # 每 decode 多少 token 采样一次 GPU 温度
SANITY_N = 16               # sanity 续 decode 长度
SANITY_CHUNKS = (2048, 512)  # chunk 消融的两个分块


def sample_gpu_temp() -> float | None:
    """nvidia-smi 温度采样; 失败返回 None（不中断）."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip())
    except Exception:
        return None


def make_text(tok, n_tokens: int) -> torch.Tensor:
    base = ("The history of artificial intelligence and neural networks spans many "
            "decades of research into learning systems that process language text "
            "with attention mechanisms keys queries and values over long sequences "
            "of tokens in large transformer models trained on public corpora.")
    s = base
    while True:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] >= n_tokens:
            return ids[:, :n_tokens]
        s = s + " " + base


def chunked_prefill(ours, ids, chunk: int, need_logits_last: bool):
    """分块 prefill; need_logits_last=True 时返回末块 logits 与完整缓存."""
    cache_list = None
    last_logits = None
    S = ids.shape[1]
    with torch.inference_mode():
        for st in range(0, S, chunk):
            piece = ids[:, st:st + chunk]
            last = (st + chunk >= S)
            lg, cache_list = ours._run(
                ours.embed_tokens(piece), cache_list,
                need_logits=(need_logits_last and last))
            if need_logits_last and last:
                last_logits = lg
    return last_logits, cache_list


def decode_from(ours, last_logits, cache, n: int, device, eos, on_every=None):
    """贪心续 decode ≤n tok（遇 eos 停）。on_every(step)->None 用于温度采样.
    返回 (tokens, decode_seconds)."""
    tokens: list[int] = []
    cur = last_logits
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(n):
            nxt = int(cur[:, -1].argmax())
            if nxt == eos:
                break
            tokens.append(nxt)
            if on_every is not None:
                on_every(step)
            cur, cache = ours._run(
                ours.embed_tokens(torch.tensor([[nxt]], device=device)), cache)
    torch.cuda.synchronize()
    return tokens, time.perf_counter() - t0


def kv_max_delta(cache_a: list[dict], cache_b: list[dict]) -> tuple[float, list[float]]:
    """逐层 K/V max|Δ|; 返回 (总体 max, 每层 max)."""
    per_layer: list[float] = []
    overall = 0.0
    for la, lb in zip(cache_a, cache_b):
        m = max((la["k"] - lb["k"]).abs().max().item(),
                (la["v"] - lb["v"]).abs().max().item())
        per_layer.append(m)
        overall = max(overall, m)
    return overall, per_layer


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=False)
    # 释放 HF 模型 —— 峰值显存按"单模型常驻"（实验真实条件）计量
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", torch.float16).eval()

    ids = make_text(tok, CTX).to("cuda")
    eos = tok.eos_token_id
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  context={CTX}  chunk={CHUNK}  decode={DECODE_N}  runs={RUNS}  "
          f"target=1.5B fp16", flush=True)

    # ============ A. 运行稳定性（3 × 128-tok decode, 温度/时间/显存采样） ============
    print("\n=== A. operational runs ===", flush=True)
    run_records: list[list[int]] = []
    temps: list[float] = []
    peaks_total = []

    for run in range(RUNS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        start_wall = time.strftime("%H:%M:%S")

        # 单次分块 prefill（拿末块 logits + 缓存）→ 记录 prefill 时间与随后峰值
        last_logits, cache = chunked_prefill(ours, ids, CHUNK, need_logits_last=True)
        torch.cuda.synchronize()
        pref = time.perf_counter() - t_start
        peak_prefill = torch.cuda.max_memory_allocated() / 2**30

        # 续 decode 128 tok（协议 §4.2: 遇 eos 停, 记录生成长度; 期间采样温度）
        temps_now: list[float] = []

        def on_every(step: int) -> None:
            if step % TEMP_EVERY == 0:
                t = sample_gpu_temp()
                if t is not None:
                    temps_now.append(t)

        tokens, dec_t = decode_from(ours, last_logits, cache, DECODE_N,
                                    ids.device, eos, on_every=on_every)
        temps.extend(temps_now)
        peak_total = torch.cuda.max_memory_allocated() / 2**30
        peaks_total.append(peak_total)
        dt = time.perf_counter() - t_start

        text = tok.decode(tokens)
        first = tok.decode([tokens[0]]) if tokens else "(empty)"
        tail = tok.decode(tokens[-20:]) if tokens else "(empty)"
        print(f"run{run+1}: start={start_wall} prefill={pref:.1f}s decode={dec_t:.1f}s "
              f"total={dt:.1f}s | len={len(tokens)} first={first!r} tail={tail[:60]!r} "
              f"| peak(prefill)={peak_prefill:.2f} peak(total)={peak_total:.2f} GiB",
              flush=True)
        run_records.append(tokens)
        # 关键: 释放本 run 的 0.62 GiB last_logits 与缓存引用 —— 否则它们跨
        # empty_cache/reset 存活, 污染下一 run 的峰值测量（曾致 6.83 -> 7.62 伪影）
        del last_logits, cache

    stable = all(r == run_records[0] for r in run_records)
    print(f"stable across {RUNS} runs: {stable}")
    print(f"peak(total) per run (GiB): {[f'{p:.2f}' for p in peaks_total]}")
    print(f"temp samples: n={len(temps)} min={min(temps) if temps else '-'} "
          f"max={max(temps) if temps else '-'} °C")

    # ============ B. correctness sanity（chunk 消融） ============
    print("\n=== B. chunk-ablation sanity ===", flush=True)
    caches = {}
    for ch in SANITY_CHUNKS:
        torch.cuda.empty_cache()
        lg_last, c = chunked_prefill(ours, ids, ch, need_logits_last=True)
        caches[ch] = (lg_last, c)
        print(f"chunk={ch}: cache built", flush=True)

    overall_d, per_layer_d = kv_max_delta(
        caches[SANITY_CHUNKS[0]][1], caches[SANITY_CHUNKS[1]][1])
    layers = len(per_layer_d)
    trend = {i: per_layer_d[i] for i in [0, layers // 2, layers - 1]}
    print(f"KV max|Δ| (chunk2048 vs chunk512): overall={overall_d:.3e}")
    print(f"  per-layer max|Δ| @ layers 0/{layers//2}/{layers-1}: "
          f"{[f'{v:.3e}' for v in trend.values()]}")

    seqs = {}
    for ch in SANITY_CHUNKS:
        lg, cc = caches[ch]
        got, _ = decode_from(ours, lg, cc, SANITY_N, ids.device, eos)
        seqs[ch] = got
        print(f"chunk={ch}: decode{SANITY_N} -> {tok.decode(got)[:60]!r} "
              f"({len(got)} tok)", flush=True)

    sanity_ok = seqs[SANITY_CHUNKS[0]] == seqs[SANITY_CHUNKS[1]]
    print(f"sanity decode sequences equal (chunk2048 vs chunk512): {sanity_ok}")

    # ============ verdict ============
    peaks_ok = all(p < 7.5 for p in peaks_total)
    print(f"\nG2 FULL: {'PASS' if (stable and sanity_ok and peaks_ok) else 'CHECK'}"
          f" | stable={stable} | sanity_ok={sanity_ok} | peak<7.5GiB={peaks_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
