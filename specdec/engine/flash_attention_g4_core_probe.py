"""Core correctness gate for cached multi-token G4 FlashAttention.

This probe isolates the attention semantics that speculative verification needs:

* non-cached causal forward for q_len in {1, 2, 4, 8};
* cached causal forward where KV length = prefix + q_len;
* the production q_len=4 verification shape;
* a long-cache q_len=1 decode case.

It compares the Triton backend against the exact eager expression used by
model_self.py. No model weights or WikiText windows are used.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from .flash_attention_two_pass_triton import (
    KERNEL_REVISION,
    flash_attention_bf16_two_pass,
)


Q_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
DTYPE = torch.bfloat16
QUERY_LENGTHS = (1, 2, 4, 8)
SHORT_PREFIX = 37
LONG_CONTEXT = 8192
MAX_ABS_GATE = 0.01
REL_L2_GATE = 0.001


def eager_reference(query, key, value, *, causal: bool):
    repeats = query.shape[1] // key.shape[1]
    key_rep = (
        key.unsqueeze(2)
        .expand(key.shape[0], key.shape[1], repeats, key.shape[2], key.shape[3])
        .reshape(key.shape[0], query.shape[1], key.shape[2], key.shape[3])
    )
    value_rep = (
        value.unsqueeze(2)
        .expand(value.shape[0], value.shape[1], repeats, value.shape[2], value.shape[3])
        .reshape(value.shape[0], query.shape[1], value.shape[2], value.shape[3])
    )
    scores = torch.matmul(query, key_rep.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    if causal:
        query_start = key.shape[2] - query.shape[2]
        query_pos = query_start + torch.arange(query.shape[2], device=query.device)
        key_pos = torch.arange(key.shape[2], device=query.device)
        allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        scores = scores.masked_fill(~allowed.view(1, 1, *allowed.shape), float("-inf"))
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(DTYPE)
    return torch.matmul(probability, value_rep)


def compare(reference, candidate):
    diff = candidate.float() - reference.float()
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denominator),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def run_case(name: str, query_length: int, kv_length: int):
    generator = torch.Generator(device="cuda").manual_seed(
        20260917 + query_length * 100_000 + kv_length
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
    reference = eager_reference(query, key, value, causal=True)
    candidate = flash_attention_bf16_two_pass(
        query.contiguous(), key.contiguous(), value.contiguous(), causal=True,
    )
    error = compare(reference, candidate)
    passed = bool(
        error["finite"]
        and error["max_abs"] <= MAX_ABS_GATE
        and error["relative_l2"] <= REL_L2_GATE
    )
    return {
        "name": name,
        "query_length": query_length,
        "kv_length": kv_length,
        "query_start": kv_length - query_length,
        "cached": kv_length != query_length,
        "error": error,
        "passed": passed,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if torch.cuda.get_device_capability(0) < (8, 0):
        print("BF16-capable CUDA GPU required")
        return 2

    records = []
    for query_length in QUERY_LENGTHS:
        records.append(run_case(
            f"noncached_q{query_length}", query_length, query_length,
        ))
        records.append(run_case(
            f"cached_q{query_length}", query_length, SHORT_PREFIX + query_length,
        ))
    records.append(run_case("cached_q4_long", 4, 2048 + 4))
    records.append(run_case("decode_q1_8k", 1, LONG_CONTEXT))

    all_pass = all(row["passed"] for row in records)
    result = {
        "status": "g4_cached_multi_token_core_gate",
        "kernel_revision": KERNEL_REVISION,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "query_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": str(DTYPE),
        },
        "gates": {
            "max_abs": MAX_ABS_GATE,
            "relative_l2": REL_L2_GATE,
        },
        "records": records,
        "decision": "PASS_G4_CAUSAL_ALIGNMENT" if all_pass else "CHECK_G4_CAUSAL_ALIGNMENT",
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_core_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "G4 FlashAttention core probe: "
        f"gpu={result['gpu']} revision={KERNEL_REVISION}"
    )
    for row in records:
        error = row["error"]
        print(
            f"{row['name']:20s} q={row['query_length']:2d} "
            f"kv={row['kv_length']:5d} start={row['query_start']:5d} "
            f"max_abs={error['max_abs']:.6f} "
            f"rel_l2={error['relative_l2']:.7f} pass={row['passed']}"
        )
    print(f"decision={result['decision']}\nJSON={output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
