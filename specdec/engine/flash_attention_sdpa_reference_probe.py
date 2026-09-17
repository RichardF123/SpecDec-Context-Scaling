"""Sanity-check the official PyTorch Flash SDPA reference used by G4 isolation.

The phase-isolation diagnostic found that forced PyTorch Flash SDPA also fails
strict end-to-end SpecDec trace identity.  Before interpreting that result, this
small probe verifies that the reference wrapper itself implements native GQA
and lower-right cached causal alignment correctly.

This is a semantic sanity gate, not an eager-bitwise-equivalence gate and not a
performance benchmark.  No model weights or corpus are loaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .flash_attention_g4_core_probe import (
    DTYPE,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    compare,
    eager_reference,
)
from .flash_attention_g4_phase_isolation import _sdpa_flash


CASES = (
    ("noncached_q1", 1, 1),
    ("noncached_q2", 2, 2),
    ("noncached_q4", 4, 4),
    ("noncached_q8", 8, 8),
    ("cached_q1", 1, 38),
    ("cached_q2", 2, 39),
    ("cached_q4", 4, 41),
    ("cached_q8", 8, 45),
    ("cached_q4_long", 4, 2052),
    ("decode_q1_8k", 1, 8192),
)

# These bounds detect a wrong mask, head mapping, or reference dispatch.  They
# deliberately do not demand the custom two-pass kernel's tighter 1e-3 gate:
# a standard fused SDPA kernel uses a different legal floating-point reduction.
MAX_ABS_SANITY = 0.05
REL_L2_SANITY = 0.02


def make_inputs(query_length: int, kv_length: int):
    generator = torch.Generator(device="cuda").manual_seed(
        20260918 + query_length * 100_000 + kv_length
    )
    query = torch.randn(
        1, Q_HEADS, query_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    key = torch.randn(
        1, KV_HEADS, kv_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    value = torch.randn(
        1, KV_HEADS, kv_length, HEAD_DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    return query, key, value


def run_case(name: str, query_length: int, kv_length: int):
    query, key, value = make_inputs(query_length, kv_length)
    reference = eager_reference(query, key, value, causal=True)
    candidate = _sdpa_flash(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        causal=True,
    )
    error = compare(reference, candidate)
    return {
        "name": name,
        "query_length": query_length,
        "kv_length": kv_length,
        "query_start": kv_length - query_length,
        "cached": kv_length != query_length,
        "error": error,
        "passed_sanity": bool(
            error["finite"]
            and error["max_abs"] <= MAX_ABS_SANITY
            and error["relative_l2"] <= REL_L2_SANITY
        ),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    records = []
    unavailable = []
    print(
        "PyTorch Flash SDPA GQA/cached-causal reference probe: "
        f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}",
        flush=True,
    )
    for name, query_length, kv_length in CASES:
        try:
            row = run_case(name, query_length, kv_length)
        except (RuntimeError, NotImplementedError) as error:
            row = {
                "name": name,
                "query_length": query_length,
                "kv_length": kv_length,
                "query_start": kv_length - query_length,
                "cached": kv_length != query_length,
                "available": False,
                "error_message": f"{type(error).__name__}: {error}",
                "passed_sanity": False,
            }
            unavailable.append(name)
            print(f"{name:20s} unavailable: {row['error_message']}", flush=True)
        else:
            row["available"] = True
            error = row["error"]
            print(
                f"{name:20s} q={query_length:2d} kv={kv_length:5d} "
                f"start={kv_length - query_length:5d} "
                f"max_abs={error['max_abs']:.6f} "
                f"rel_l2={error['relative_l2']:.7f} "
                f"pass={row['passed_sanity']}",
                flush=True,
            )
        records.append(row)

    if unavailable:
        decision = "CHECK_SDPA_FLASH_REFERENCE_UNAVAILABLE"
    elif all(row["passed_sanity"] for row in records):
        decision = "PASS_SDPA_FLASH_REFERENCE_SEMANTICS"
    else:
        decision = "CHECK_SDPA_FLASH_REFERENCE_SEMANTICS"

    result = {
        "status": "pytorch_flash_sdpa_gqa_cached_causal_reference",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "query_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE),
        },
        "sanity_gates": {
            "max_abs": MAX_ABS_SANITY,
            "relative_l2": REL_L2_SANITY,
        },
        "records": records,
        "unavailable_cases": unavailable,
        "decision": decision,
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_sdpa_reference_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"decision={decision}\nJSON={output}")
    return 0 if decision == "PASS_SDPA_FLASH_REFERENCE_SEMANTICS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
