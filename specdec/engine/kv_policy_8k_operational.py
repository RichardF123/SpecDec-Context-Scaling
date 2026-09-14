"""8K operational comparison for the confirmed KV policy candidate.

Measures BF16 KV, current V-FP8, and selected blockwise/per-channel K-INT8 +
V-FP8 on the same RTX 4060.  Three rotated-order rounds reduce systematic warm
or clock-order bias.  This is an eager implementation measurement, not a fused-
kernel speed claim.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time

import torch

from .g2_operational import make_text
from .kv_policy_pilot_v2 import MODEL, load_model

DTYPE = torch.bfloat16
CTX = 8192
CHUNK = 2048
DECODE_N = 64
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "08_operational_8k.json",
)

POLICIES = {
    "base_bf16": "fp16",
    "current_vfp8": "v_fp8",
    "selected_kpcint8_b32_vfp8": "k_pc_int8_v_fp8",
}
ORDERS = (
    ("base_bf16", "current_vfp8", "selected_kpcint8_b32_vfp8"),
    ("current_vfp8", "selected_kpcint8_b32_vfp8", "base_bf16"),
    ("selected_kpcint8_b32_vfp8", "base_bf16", "current_vfp8"),
)
EXPECTED_PREFILL_MIB = {
    "base_bf16": 224.00,
    "current_vfp8": 169.75,
    "selected_kpcint8_b32_vfp8": 120.75,
}


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def run_once(model, ids, fmt: str, eos: int):
    model.set_kv_head_policy(2, 2)
    model.set_kv_format(fmt)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache = None
    last_logits = None

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, CTX, CHUNK):
            piece = ids[:, start:start + CHUNK]
            is_last = start + CHUNK >= CTX
            logits, cache = model._run(
                model.embed_tokens(piece), cache, need_logits=is_last,
            )
            if is_last:
                last_logits = logits
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - started
    prefill_cache_bytes = cache_logical_bytes(cache)

    tokens = []
    current_logits = last_logits
    decode_started = time.perf_counter()
    finite = bool(torch.isfinite(current_logits).all().item())
    with torch.inference_mode():
        for _ in range(DECODE_N):
            next_id = int(current_logits[:, -1].argmax())
            if next_id == eos:
                break
            tokens.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=ids.device)
            current_logits, cache = model._run(
                model.embed_tokens(next_tensor), cache, need_logits=True,
            )
            finite &= bool(torch.isfinite(current_logits).all().item())
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - decode_started

    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tokens": len(tokens),
        "tokens": tokens,
        "finite": finite,
        "prefill_cache_mib": prefill_cache_bytes / 2**20,
        "final_cache_mib": cache_logical_bytes(cache) / 2**20,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def median(records, key):
    return statistics.median(record[key] for record in records)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = make_text(tokenizer, CTX).to("cuda")
    model = load_model()
    by_policy = {name: [] for name in POLICIES}
    all_runs = []
    print(
        f"8K operational: model=1.5B ctx={CTX} chunk={CHUNK} "
        f"decode={DECODE_N} dtype={DTYPE}", flush=True,
    )
    print("three rotated-order rounds; native KV heads; eager attention\n", flush=True)
    experiment_started = time.perf_counter()

    for round_index, order in enumerate(ORDERS, start=1):
        print(f"=== round {round_index} order={','.join(order)} ===", flush=True)
        for name in order:
            record = run_once(model, ids, POLICIES[name], tokenizer.eos_token_id)
            record["policy"] = name
            record["round"] = round_index
            by_policy[name].append(record)
            all_runs.append(record)
            print(
                f"{name:30s} prefill={record['prefill_s']:.2f}s "
                f"decode={record['decode_s']:.2f}s/{record['decode_tokens']}tok "
                f"peak_alloc={record['peak_alloc_gib']:.3f}GiB "
                f"peak_reserved={record['peak_reserved_gib']:.3f}GiB "
                f"cache8k={record['prefill_cache_mib']:.2f}MiB "
                f"finite={record['finite']}", flush=True,
            )

    summary = {}
    for name, records in by_policy.items():
        stable = all(record["tokens"] == records[0]["tokens"] for record in records)
        measured = records[0]["prefill_cache_mib"]
        if abs(measured - EXPECTED_PREFILL_MIB[name]) > 0.01:
            raise RuntimeError(
                f"8K cache-byte mismatch for {name}: {measured:.4f} vs "
                f"{EXPECTED_PREFILL_MIB[name]:.4f} MiB"
            )
        summary[name] = {
            "median_prefill_s": median(records, "prefill_s"),
            "median_decode_s": median(records, "decode_s"),
            "median_peak_alloc_gib": median(records, "peak_alloc_gib"),
            "median_peak_reserved_gib": median(records, "peak_reserved_gib"),
            "prefill_cache_mib": measured,
            "final_cache_mib": records[0]["final_cache_mib"],
            "stable_tokens_across_rounds": stable,
            "finite_all": all(record["finite"] for record in records),
            "decode_tokens": [record["decode_tokens"] for record in records],
        }

    base = summary["base_bf16"]
    current = summary["current_vfp8"]
    selected = summary["selected_kpcint8_b32_vfp8"]
    comparison = {
        "selected_cache_mib_saved_vs_current": (
            current["prefill_cache_mib"] - selected["prefill_cache_mib"]
        ),
        "selected_cache_pct_saved_vs_current": (
            1.0 - selected["prefill_cache_mib"] / current["prefill_cache_mib"]
        ) * 100.0,
        "selected_peak_alloc_mib_delta_vs_current": (
            selected["median_peak_alloc_gib"] - current["median_peak_alloc_gib"]
        ) * 1024.0,
        "selected_peak_reserved_mib_delta_vs_current": (
            selected["median_peak_reserved_gib"] - current["median_peak_reserved_gib"]
        ) * 1024.0,
        "selected_prefill_pct_delta_vs_current": (
            selected["median_prefill_s"] / current["median_prefill_s"] - 1.0
        ) * 100.0,
        "selected_decode_pct_delta_vs_current": (
            selected["median_decode_s"] / current["median_decode_s"] - 1.0
        ) * 100.0,
        "selected_peak_alloc_mib_delta_vs_base": (
            selected["median_peak_alloc_gib"] - base["median_peak_alloc_gib"]
        ) * 1024.0,
    }
    elapsed = time.perf_counter() - experiment_started
    result = {
        "status": "operational_evidence_no_automatic_verdict",
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ctx": CTX,
        "chunk": CHUNK,
        "decode_n": DECODE_N,
        "elapsed_s": elapsed,
        "summary": summary,
        "comparison": comparison,
        "runs": all_runs,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print("\n=== 8K OPERATIONAL SUMMARY (medians of 3 rotated rounds) ===")
    for name in POLICIES:
        record = summary[name]
        print(
            f"{name:30s} prefill={record['median_prefill_s']:.2f}s "
            f"decode={record['median_decode_s']:.2f}s "
            f"peak_alloc={record['median_peak_alloc_gib']:.3f}GiB "
            f"peak_reserved={record['median_peak_reserved_gib']:.3f}GiB "
            f"cache8k={record['prefill_cache_mib']:.2f}MiB "
            f"stable={record['stable_tokens_across_rounds']} "
            f"finite={record['finite_all']}",
        )
    print(
        "selected vs current: "
        f"cache={comparison['selected_cache_mib_saved_vs_current']:.2f}MiB lower "
        f"({comparison['selected_cache_pct_saved_vs_current']:.1f}%) | "
        f"peak_alloc_delta={comparison['selected_peak_alloc_mib_delta_vs_current']:+.1f}MiB | "
        f"peak_reserved_delta={comparison['selected_peak_reserved_mib_delta_vs_current']:+.1f}MiB | "
        f"prefill_delta={comparison['selected_prefill_pct_delta_vs_current']:+.1f}% | "
        f"decode_delta={comparison['selected_decode_pct_delta_vs_current']:+.1f}%",
    )
    print(f"elapsed={elapsed:.0f}s\nJSON={OUTPUT}")
    print("Interpret peak VRAM and latency separately from logical KV compression.")
    all_ok = all(
        record["stable_tokens_across_rounds"] and record["finite_all"]
        for record in summary.values()
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
