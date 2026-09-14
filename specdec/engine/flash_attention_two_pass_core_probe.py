"""Cheap numerical gate for the two-pass FlashAttention diagnostic."""
from __future__ import annotations

import gc
import json
import os

import torch

from .flash_attention_core_probe import eager_attention, error_metrics, tensors
from .flash_attention_triton import flash_attention_bf16
from .flash_attention_two_pass_triton import (
    KERNEL_REVISION,
    flash_attention_bf16_two_pass,
)


OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_two_pass_core_probe.json",
)
CASES = (
    ("full_prefill_127", 127, 127),
    ("full_prefill_512", 512, 512),
    ("cached_chunk_128_of_512", 128, 512),
    ("cached_chunk_256_of_2048", 256, 2048),
    ("decode_1_of_8192", 1, 8192),
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    rows = []
    improved_all = True
    strict_all = True
    print(
        "two-pass global-softmax semantics probe: "
        f"gpu={torch.cuda.get_device_name(0)} revision={KERNEL_REVISION}",
        flush=True,
    )
    print(
        "Compares eager, current one-pass, and two-pass; no model weights are used.\n",
        flush=True,
    )
    for index, (name, query_length, kv_length) in enumerate(CASES):
        query, key, value = tensors(query_length, kv_length, 3000 + index)
        with torch.inference_mode():
            eager = eager_attention(query, key, value)
            one_pass = flash_attention_bf16(query, key, value)
            two_pass = flash_attention_bf16_two_pass(query, key, value)
        torch.cuda.synchronize()
        one_error = error_metrics(eager, one_pass)
        two_error = error_metrics(eager, two_pass)
        ratio = two_error["relative_l2"] / max(one_error["relative_l2"], 1e-12)
        improved = two_error["relative_l2"] < one_error["relative_l2"]
        strict = two_error["relative_l2"] <= 0.002
        improved_all &= improved
        strict_all &= strict and two_error["finite"]
        rows.append(
            {
                "name": name,
                "query_length": query_length,
                "kv_length": kv_length,
                "one_pass": one_error,
                "two_pass": two_error,
                "two_pass_rel_l2_ratio": ratio,
                "improved": improved,
                "strict_gate": strict,
            }
        )
        print(
            f"{name:28s} one_rel_l2={one_error['relative_l2']:.6f} "
            f"two_rel_l2={two_error['relative_l2']:.6f} "
            f"ratio={ratio:.3f} two_max_abs={two_error['max_abs']:.6f} "
            f"improved={improved} strict={strict}",
            flush=True,
        )
        del query, key, value, eager, one_pass, two_pass
        gc.collect()
        torch.cuda.empty_cache()

    if strict_all:
        decision = "PROCEED_TO_2K_WHOLE_MODEL_DIAGNOSTIC"
    elif improved_all:
        decision = "REVIEW_IMPROVEMENT_BEFORE_MODEL_RUN"
    else:
        decision = "STOP_TWO_PASS_DOES_NOT_EXPLAIN_DRIFT"
    result = {
        "status": "isolated_two_pass_global_softmax_semantics_probe",
        "kernel_revision": KERNEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "strict_relative_l2_gate": 0.002,
        "cases": rows,
        "decision": decision,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"\ndecision={decision}\nJSON={OUTPUT}")
    return 0 if strict_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
