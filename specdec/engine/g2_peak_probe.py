"""g2_peak_probe.py — G2 峰值显存方差探针（定性 6.83 vs 7.62 GiB 跳变）

背景: g2_operational 3 次同代码 8K prefill, peak 分别为 6.83 / 7.62 / 7.63 GiB。
本探针连跑 5 次同一 8K chunked prefill, 记录:
  - torch 分配峰值 (max_memory_allocated)   —— 纯 torch 活内存高水位
  - torch 保留峰值 (max_memory_reserved)    —— 分配器从驱动圈走的池
  - nvidia-smi memory.used                   —— 驱动级真实占用(含上下文/其它进程)
判读: 全 ~7.6 GiB → 6.83 为冷启动低估, 稳态峰值=7.6;  出现交替 → 分配器方差,
正式基准峰值列必须以 reps 中位数记（P9）。

Run（交互式 GPU 终端）:  python -m specdec.engine.g2_peak_probe
"""
from __future__ import annotations

import subprocess
import time

import torch

from .align_ours import build_from_config
from .g2_operational import CTX, CHUNK, MODEL, chunked_prefill, make_text

REPS = 5


def smi_memory_used_gib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip()) / 1024.0  # MiB -> GiB
    except Exception:
        return None


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
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", torch.float16).eval()

    ids = make_text(tok, CTX).to("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  ctx={CTX} chunk={CHUNK} reps={REPS}\n",
          flush=True)

    rows = []
    for rep in range(REPS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        smi_before = smi_memory_used_gib()
        t0 = time.perf_counter()
        chunked_prefill(ours, ids, CHUNK, need_logits_last=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        alloc = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        smi_after = smi_memory_used_gib()
        print(f"rep{rep+1}: prefill={dt:.1f}s | alloc_peak={alloc:.2f} GiB | "
              f"reserved_peak={reserved:.2f} GiB | smi used: {smi_before:.2f} -> "
              f"{smi_after if smi_after else float('nan'):.2f} GiB", flush=True)
        rows.append((alloc, reserved))

    allocs = [r[0] for r in rows]
    resvs = [r[1] for r in rows]
    spread = max(allocs) - min(allocs)
    print(f"\nalloc_peak: min={min(allocs):.2f} max={max(allocs):.2f} "
          f"spread={spread:.2f} GiB")
    print(f"reserved_peak: min={min(resvs):.2f} max={max(resvs):.2f} "
          f"spread={max(resvs)-min(resvs):.2f} GiB")
    if spread < 0.1:
        print("VERDICT: prefill alloc_peak is deterministic (", f"{min(allocs):.2f}",
              "GiB across all reps); any higher peak seen elsewhere is a script-state "
              "artifact (previous-run tensors held across reset), not a real footprint")
    else:
        print("VERDICT: allocator variance confirmed (spread", f"{spread:.2f} GiB)",
              ")— record peak as median over reps in formal runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
