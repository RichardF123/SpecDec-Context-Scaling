"""Fresh confirmation for K-PC-INT8 block32 + 128-token BF16 residual + V-FP8.

The residual size was selected only on pilot offsets 0..3.  Offsets 40..79 were
already used by the r0 candidate, so this script uses untouched offsets 80..119.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

import torch

from .fp8_kv_v4_dppl import ensure_data
from .kv_policy_pilot_v2 import MODEL, WINDOW, load_model
from .kv_policy_pilot_v3 import Policy, window_stats

N_WINDOWS = 40
WINDOW_OFFSET = 80
POLICIES = (
    Policy("base_bf16", "fp16", 0),
    Policy("current_vfp8", "v_fp8", 0),
    Policy("selected_kpcint8_b32_r128_vfp8", "k_pc_int8_v_fp8", 128),
)
EXPECTED_BYTES_PER_TOKEN_LAYER = {
    "base_bf16": 1024.0,
    "current_vfp8": 776.0,
    "selected_kpcint8_b32_r128_vfp8": 566.0,
}
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "11_confirm_residual_v3.json",
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.model_max_length = 10**9
    with open(ensure_data(), encoding="utf-8") as stream:
        all_ids = torch.tensor(tokenizer(stream.read())["input_ids"], dtype=torch.long)
    needed = (WINDOW_OFFSET + N_WINDOWS) * WINDOW
    if all_ids.numel() < needed:
        raise RuntimeError(f"dataset has {all_ids.numel()} tokens, need {needed}")

    print(
        f"FRESH confirmation: windows={WINDOW_OFFSET}.."
        f"{WINDOW_OFFSET + N_WINDOWS - 1}, n={N_WINDOWS}, W={WINDOW}",
        flush=True,
    )
    print(
        "selected before confirmation: K-PC-INT8 block32 + BF16 residual128 "
        "+ V-FP8, native heads\n",
    )
    model = load_model()
    totals = {
        policy.name: {"nll_sum": 0.0, "tokens": 0, "finite": True}
        for policy in POLICIES
    }
    per_window = []
    logical = {}
    started = time.perf_counter()

    for index in range(N_WINDOWS):
        offset = WINDOW_OFFSET + index
        ids_w = all_ids[offset * WINDOW:(offset + 1) * WINDOW].cuda()
        row = {"window_offset": offset}
        for policy in POLICIES:
            nll_sum, tokens, finite, cache_bytes = window_stats(model, ids_w, policy)
            nll = nll_sum / tokens
            row[policy.name] = {"nll": nll, "ppl": math.exp(nll), "finite": finite}
            totals[policy.name]["nll_sum"] += nll_sum
            totals[policy.name]["tokens"] += tokens
            totals[policy.name]["finite"] &= finite
            logical[policy.name] = cache_bytes / (WINDOW * len(model.layers))
            torch.cuda.empty_cache()
        row["current_dNLL"] = row["current_vfp8"]["nll"] - row["base_bf16"]["nll"]
        row["selected_dNLL"] = (
            row["selected_kpcint8_b32_r128_vfp8"]["nll"]
            - row["base_bf16"]["nll"]
        )
        per_window.append(row)
        print(
            f"window={offset:03d} base={row['base_bf16']['nll']:.6f} "
            f"current_dNLL={row['current_dNLL']:+.6f} "
            f"selected_dNLL={row['selected_dNLL']:+.6f}",
            flush=True,
        )
        del ids_w

    aggregate = {}
    base_nll = totals["base_bf16"]["nll_sum"] / totals["base_bf16"]["tokens"]
    base_ppl = math.exp(base_nll)
    for policy in POLICIES:
        name = policy.name
        mean_nll = totals[name]["nll_sum"] / totals[name]["tokens"]
        ppl = math.exp(mean_nll)
        aggregate[name] = {
            "nll": mean_nll,
            "ppl": ppl,
            "dNLL_vs_base": mean_nll - base_nll,
            "relative_dPPL_vs_base_pct": (ppl / base_ppl - 1.0) * 100.0,
            "bytes_per_token_layer": logical[name],
            "compression_vs_base": logical["base_bf16"] / logical[name],
            "finite": totals[name]["finite"],
            "tokens": totals[name]["tokens"],
        }
        expected = EXPECTED_BYTES_PER_TOKEN_LAYER[name]
        if logical[name] != expected:
            raise RuntimeError(
                f"logical-byte mismatch for {name}: {logical[name]} != {expected}"
            )

    current_deltas = [row["current_dNLL"] for row in per_window]
    selected_deltas = [row["selected_dNLL"] for row in per_window]
    aggregate["current_vfp8"]["paired_dNLL_se"] = (
        statistics.stdev(current_deltas) / math.sqrt(N_WINDOWS)
    )
    aggregate["selected_kpcint8_b32_r128_vfp8"]["paired_dNLL_se"] = (
        statistics.stdev(selected_deltas) / math.sqrt(N_WINDOWS)
    )
    current = aggregate["current_vfp8"]
    selected = aggregate["selected_kpcint8_b32_r128_vfp8"]
    selected["relative_dPPL_vs_current_pct"] = (
        selected["ppl"] / current["ppl"] - 1.0
    ) * 100.0
    selected["bytes_vs_current"] = (
        selected["bytes_per_token_layer"] / current["bytes_per_token_layer"]
    )

    elapsed = time.perf_counter() - started
    result = {
        "status": "fresh_confirmation_evidence_no_automatic_verdict",
        "selection_source": "pilot windows 0..3",
        "excluded_seen_windows": [40, 79],
        "confirmation_windows": [WINDOW_OFFSET, WINDOW_OFFSET + N_WINDOWS - 1],
        "model": MODEL,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_s": elapsed,
        "aggregate": aggregate,
        "per_window": per_window,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print("\n=== FRESH RESIDUAL CONFIRMATION SUMMARY ===")
    for policy in POLICIES:
        record = aggregate[policy.name]
        se = record.get("paired_dNLL_se", 0.0)
        print(
            f"{policy.name:38s} NLL={record['nll']:.6f} "
            f"PPL={record['ppl']:.4f} dNLL={record['dNLL_vs_base']:+.6f} "
            f"rel_dPPL={record['relative_dPPL_vs_base_pct']:+.3f}% "
            f"paired_SE={se:.6f} compression={record['compression_vs_base']:.3f}x",
        )
    print(
        "selected vs current: "
        f"rel_dPPL={selected['relative_dPPL_vs_current_pct']:+.3f}% | "
        f"logical bytes={selected['bytes_vs_current']:.3f}x "
        f"({(1.0 - selected['bytes_vs_current']) * 100.0:.1f}% lower)",
    )
    print(f"elapsed={elapsed:.0f}s\nJSON={OUTPUT}")
    print("No automatic verdict; run 8K operational only after review.")
    return 0 if all(totals[p.name]["finite"] for p in POLICIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
