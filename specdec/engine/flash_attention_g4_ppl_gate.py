"""Frozen-window PPL gate for the integrated G4 hybrid backend.

The kernel and its block size were selected without consulting these data.
This script reuses the project's pre-existing frozen WikiText-2 windows 40..79
to compare BF16 eager attention with BF16 two-pass Triton attention.  It does
not modify the model, weights, KV representation, or production engine.
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time

import torch

from .flash_attention_two_pass_triton import KERNEL_REVISION
from .fp8_kv_v4_dppl import ensure_data
from .g4_b4_cache_tests import load_ours


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FLASH_BACKEND = "triton_bf16_g4_hybrid"


DTYPE = torch.bfloat16
WINDOW = 2048
CHUNK = 512
LOSS_ROWS = 64
WINDOW_OFFSET = 40
N_WINDOWS = 40
# Pre-registered numerical-equivalence margin: approximately 0.1% PPL.
DNLL_EQUIVALENCE_MARGIN = 0.001
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_g4_hybrid_ppl_gate.json",
)


def window_nll(model, ids: torch.Tensor, backend: str) -> tuple[float, int, bool]:
    model.set_attention_backend(
        "eager" if backend == "eager_bf16" else FLASH_BACKEND
    )
    model.set_kv_format("fp16")
    cache = None
    nll_sum = 0.0
    token_count = 0
    finite = True
    with torch.inference_mode():
        for start in range(0, WINDOW, CHUNK):
            piece = ids[start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece.unsqueeze(0)),
                cache,
                need_logits=True,
            )
            finite &= bool(torch.isfinite(logits).all().item())
            targets = ids[start + 1:start + 1 + CHUNK]
            rows = int(targets.shape[0])
            for row_start in range(0, rows, LOSS_ROWS):
                row_end = min(row_start + LOSS_ROWS, rows)
                log_probability = torch.log_softmax(
                    logits[0, row_start:row_end].to(torch.float32),
                    dim=-1,
                )
                indices = torch.arange(
                    row_end - row_start,
                    device=logits.device,
                )
                nll_sum += -log_probability[
                    indices,
                    targets[row_start:row_end],
                ].sum().item()
                del log_probability, indices
            token_count += rows
            del logits
    del cache
    return nll_sum, token_count, finite


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.model_max_length = 10**9
    with open(ensure_data(), encoding="utf-8") as stream:
        all_ids = torch.tensor(
            tokenizer(stream.read())["input_ids"],
            dtype=torch.long,
        )
    required = (WINDOW_OFFSET + N_WINDOWS) * WINDOW
    if all_ids.numel() < required:
        raise RuntimeError(f"dataset has {all_ids.numel()} tokens, need {required}")

    model, _ = load_ours(MODEL)

    print(
        "G4 hybrid FlashAttention frozen PPL gate: "
        f"gpu={torch.cuda.get_device_name(0)} model=1.5B "
        f"windows={WINDOW_OFFSET}..{WINDOW_OFFSET + N_WINDOWS - 1} "
        f"W={WINDOW} chunk={CHUNK} revision={KERNEL_REVISION}",
        flush=True,
    )
    print(
        "paired BF16 eager vs BF16 G4 hybrid; "
        "kernel was selected without these windows\n",
        flush=True,
    )

    records = []
    total = {
        "eager_bf16": {"nll": 0.0, "tokens": 0, "finite": True},
        "flash_bf16": {"nll": 0.0, "tokens": 0, "finite": True},
    }
    started = time.perf_counter()
    for relative_index in range(N_WINDOWS):
        window_index = WINDOW_OFFSET + relative_index
        offset = window_index * WINDOW
        ids = all_ids[offset:offset + WINDOW].cuda()
        # Alternate order to avoid associating either backend with systematic
        # temperature or clock drift, even though latency is not evaluated.
        order = (
            ("eager_bf16", "flash_bf16")
            if relative_index % 2 == 0
            else ("flash_bf16", "eager_bf16")
        )
        values = {}
        for backend in order:
            nll_sum, tokens, finite = window_nll(model, ids, backend)
            values[backend] = {
                "nll_sum": nll_sum,
                "tokens": tokens,
                "mean_nll": nll_sum / tokens,
                "finite": finite,
            }
            total[backend]["nll"] += nll_sum
            total[backend]["tokens"] += tokens
            total[backend]["finite"] &= finite
            gc.collect()
            torch.cuda.empty_cache()
        delta = values["flash_bf16"]["mean_nll"] - values["eager_bf16"]["mean_nll"]
        record = {
            "window": window_index,
            "order": order,
            "eager_bf16": values["eager_bf16"],
            "flash_bf16": values["flash_bf16"],
            "delta_nll": delta,
        }
        records.append(record)
        print(
            f"window={window_index:02d} "
            f"eager={values['eager_bf16']['mean_nll']:.6f} "
            f"flash={values['flash_bf16']['mean_nll']:.6f} "
            f"dNLL={delta:+.7f}",
            flush=True,
        )
        del ids

    summaries = {}
    for backend in ("eager_bf16", "flash_bf16"):
        mean_nll = total[backend]["nll"] / total[backend]["tokens"]
        summaries[backend] = {
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll),
            "tokens": total[backend]["tokens"],
            "finite": total[backend]["finite"],
        }
    deltas = [record["delta_nll"] for record in records]
    paired_mean = statistics.fmean(deltas)
    paired_se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    two_sided_95_bound = abs(paired_mean) + 1.96 * paired_se
    relative_ppl_pct = (math.exp(paired_mean) - 1.0) * 100.0
    max_abs_window_delta = max(abs(value) for value in deltas)
    passed = (
        summaries["eager_bf16"]["finite"]
        and summaries["flash_bf16"]["finite"]
        and two_sided_95_bound <= DNLL_EQUIVALENCE_MARGIN
    )
    decision = (
        "PASS_FLASHATTENTION_PPL_EQUIVALENCE"
        if passed else "FAIL_FLASHATTENTION_PPL_EQUIVALENCE"
    )
    comparison = {
        "paired_mean_delta_nll": paired_mean,
        "paired_se": paired_se,
        "two_sided_95_equivalence_bound": two_sided_95_bound,
        "delta_nll_margin": DNLL_EQUIVALENCE_MARGIN,
        "relative_ppl_pct": relative_ppl_pct,
        "max_abs_window_delta_nll": max_abs_window_delta,
    }
    result = {
        "status": "latest_main_g4_hybrid_flashattention_ppl_gate",
        "kernel_revision": KERNEL_REVISION,
        "flash_backend": FLASH_BACKEND,
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": str(DTYPE),
        "window": WINDOW,
        "chunk": CHUNK,
        "window_offset": WINDOW_OFFSET,
        "n_windows": N_WINDOWS,
        "summaries": summaries,
        "comparison": comparison,
        "records": records,
        "decision": decision,
        "elapsed_s": time.perf_counter() - started,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print("\n=== FROZEN PPL SUMMARY ===")
    for backend in ("eager_bf16", "flash_bf16"):
        row = summaries[backend]
        print(
            f"{backend:12s} NLL={row['mean_nll']:.6f} "
            f"PPL={row['ppl']:.4f} finite={row['finite']}"
        )
    print(
        f"paired dNLL={paired_mean:+.7f} SE={paired_se:.7f} "
        f"95% bound={two_sided_95_bound:.7f} "
        f"margin={DNLL_EQUIVALENCE_MARGIN:.7f}"
    )
    print(
        f"relative dPPL={relative_ppl_pct:+.4f}% "
        f"max_abs_window_dNLL={max_abs_window_delta:.7f}"
    )
    print(
        f"decision={decision}\nelapsed={result['elapsed_s']:.0f}s\nJSON={OUTPUT}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
