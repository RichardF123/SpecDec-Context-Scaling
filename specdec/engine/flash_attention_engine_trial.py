"""Whole-model gate for the from-scratch BF16 Triton FlashAttention.

The production engine remains unchanged.  This script installs a runtime-only
attention method, compares it with the existing eager path at 2K/4K/8K, and
records correctness, chunked-prefill/decode latency, peak allocation, and cache
bytes.  Run only after ``flash_attention_core_probe`` passes.
"""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time
import types

import torch

from . import model_self as ms
from .flash_attention_triton import KERNEL_REVISION, flash_attention_bf16
from .kv_policy_pilot_v2 import MODEL, load_model


DTYPE = torch.bfloat16
CONTEXTS = (2048, 4096, 8192)
CHUNK = 2048
DECODE_TOKENS = 64
ORDERS = (
    ("eager_bf16", "flash_bf16"),
    ("flash_bf16", "eager_bf16"),
    ("flash_bf16", "eager_bf16"),
    ("eager_bf16", "flash_bf16"),
)
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_engine_trial.json",
)


def make_token_ids(tokenizer, length: int) -> torch.Tensor:
    text = (
        "The history of artificial intelligence and neural networks spans many "
        "decades of research into learning systems that process language with "
        "attention mechanisms, keys, queries, and values over long sequences."
    )
    base = tokenizer(text, add_special_tokens=False).input_ids
    if not base:
        raise RuntimeError("tokenizer returned an empty prompt")
    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    repeats = math.ceil((length - len(prefix)) / len(base))
    values = (prefix + base * repeats)[:length]
    return torch.tensor([values], dtype=torch.long, device="cuda")


def _flash_trial_forward(self, x: torch.Tensor, cos, sin, cache=None):
    """Native-head BF16 prefill/decode using the isolated Triton kernel."""
    fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
    eligible = (
        fmt == "fp16"
        and x.is_cuda
        and x.dtype == DTYPE
        and self.head_dim == 128
        and self.heads == 12
        and self.kv_heads == 2
        and self.cache_k_heads == self.kv_heads
        and self.cache_v_heads == self.kv_heads
    )
    if not eligible:
        return self._flash_trial_original_forward(x, cos, sin, cache)

    batch, sequence, _ = x.shape
    query = self.q_proj(x).view(
        batch, sequence, self.heads, self.head_dim,
    ).transpose(1, 2)
    key = self.k_proj(x).view(
        batch, sequence, self.kv_heads, self.head_dim,
    ).transpose(1, 2)
    value = self.v_proj(x).view(
        batch, sequence, self.kv_heads, self.head_dim,
    ).transpose(1, 2)
    query = ms.apply_rope(query, cos, sin).contiguous()
    key = ms.apply_rope(key, cos, sin).contiguous()
    value = value.contiguous()

    if cache is None:
        key_all = key
        value_all = value
    else:
        key_all = torch.cat([cache["k"], key], dim=2).contiguous()
        value_all = torch.cat([cache["v"], value], dim=2).contiguous()

    attention_output = flash_attention_bf16(
        query,
        key_all,
        value_all,
        causal=True,
    )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch, sequence, self.heads * self.head_dim,
    )
    new_cache = {"k": key_all, "v": value_all}
    return self.o_proj(attention_output), new_cache


def install_trial_backend(model) -> None:
    for layer in model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_flash_trial_original_forward"):
            attention._flash_trial_original_forward = attention.forward


def select_backend(model, backend: str) -> None:
    if backend not in ("eager_bf16", "flash_bf16"):
        raise ValueError(backend)
    for layer in model.layers:
        attention = layer.self_attn
        if backend == "flash_bf16":
            attention.forward = types.MethodType(_flash_trial_forward, attention)
        else:
            attention.forward = attention._flash_trial_original_forward


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def run_once(model, token_ids: torch.Tensor, backend: str, capture: bool) -> dict:
    select_backend(model, backend)
    model.set_kv_head_policy(2, 2)
    model.set_kv_format("fp16")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    cache = None
    last_logits = None
    prefill_logits = None
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, token_ids.shape[1], CHUNK):
            piece = token_ids[:, start:start + CHUNK]
            is_last = start + CHUNK >= token_ids.shape[1]
            logits, cache = model._run(
                model.embed_tokens(piece), cache, need_logits=is_last,
            )
            if is_last:
                last_logits = logits
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - prefill_started
    prefill_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    prefill_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    prefill_cache_mib = cache_logical_bytes(cache) / 2**20

    current_logits = last_logits[:, -1:].contiguous()
    if capture:
        prefill_logits = current_logits[:, -1].to(torch.float32).cpu()
    del last_logits
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    decode_floor_gib = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    generated = []
    captured_logits = []
    finite = bool(torch.isfinite(current_logits).all().item())
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(DECODE_TOKENS):
            next_id = int(current_logits[:, -1].argmax())
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=token_ids.device)
            current_logits, cache = model._run(
                model.embed_tokens(next_tensor), cache, need_logits=True,
            )
            finite &= bool(torch.isfinite(current_logits).all().item())
            if capture:
                captured_logits.append(
                    current_logits[:, -1].to(torch.float32).cpu()
                )
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - decode_started
    decode_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    decode_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30

    record = {
        "backend": backend,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "milliseconds_per_token": 1000.0 * decode_s / len(generated),
        "decode_tokens": len(generated),
        "tokens": generated,
        "finite": finite,
        "prefill_cache_mib": prefill_cache_mib,
        "final_cache_mib": cache_logical_bytes(cache) / 2**20,
        "prefill_peak_allocated_gib": prefill_peak_allocated_gib,
        "prefill_peak_reserved_gib": prefill_peak_reserved_gib,
        "decode_floor_gib": decode_floor_gib,
        "decode_peak_allocated_gib": decode_peak_allocated_gib,
        "decode_peak_reserved_gib": decode_peak_reserved_gib,
        "decode_peak_extra_mib": (
            decode_peak_allocated_gib - decode_floor_gib
        ) * 1024.0,
        "_prefill_logits": prefill_logits,
        "_captured_logits": captured_logits,
    }
    del cache, current_logits
    return record


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference) / denominator
        ).item(),
        "argmax_same": bool(reference.argmax().item() == candidate.argmax().item()),
    }


def first_difference(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def compare_captured(reference: dict, candidate: dict) -> dict:
    token_diff = first_difference(reference["tokens"], candidate["tokens"])
    comparable_steps = (
        min(len(reference["_captured_logits"]), len(candidate["_captured_logits"]))
        if token_diff is None else min(
            token_diff + 1,
            len(reference["_captured_logits"]),
            len(candidate["_captured_logits"]),
        )
    )
    per_step = [
        tensor_error(
            reference["_captured_logits"][index],
            candidate["_captured_logits"][index],
        )
        for index in range(comparable_steps)
    ]
    return {
        "prefill_last_logits": tensor_error(
            reference["_prefill_logits"], candidate["_prefill_logits"],
        ),
        "tokens_identical": token_diff is None,
        "first_token_difference": token_diff,
        "comparable_decode_logit_steps": comparable_steps,
        "worst_decode_max_abs_before_divergence": max(
            (row["max_abs"] for row in per_step), default=0.0,
        ),
        "worst_decode_relative_l2_before_divergence": max(
            (row["relative_l2"] for row in per_step), default=0.0,
        ),
        "all_decode_argmax_same_before_divergence": all(
            row["argmax_same"] for row in per_step
        ),
        "per_step_before_divergence": per_step,
    }


def median(records: list[dict], key: str) -> float:
    return statistics.median(record[key] for record in records)


def strip_private(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    install_trial_backend(model)
    # Compile both prefill and decode shapes before measured rounds.
    warm_ids = make_token_ids(tokenizer, 128)
    warm = run_once(model, warm_ids, "flash_bf16", capture=False)
    del warm, warm_ids
    gc.collect()
    torch.cuda.empty_cache()

    all_runs = []
    context_results = {}
    overall_pass = True
    print(
        "whole-model from-scratch FlashAttention trial: "
        f"gpu={torch.cuda.get_device_name(0)} model=1.5B "
        f"contexts={CONTEXTS} chunk={CHUNK} decode={DECODE_TOKENS} "
        f"dtype={DTYPE}",
        flush=True,
    )
    print(
        "runtime-only integration; production model_self.py is unchanged\n",
        flush=True,
    )

    for context in CONTEXTS:
        token_ids = make_token_ids(tokenizer, context)
        by_backend = {"eager_bf16": [], "flash_bf16": []}
        captured = {}
        print(f"=== context={context} ===", flush=True)
        for round_index, order in enumerate(ORDERS, start=1):
            print(f"round={round_index} order={','.join(order)}", flush=True)
            for backend in order:
                capture = backend not in captured
                record = run_once(model, token_ids, backend, capture)
                if capture:
                    captured[backend] = record
                record["context"] = context
                record["round"] = round_index
                by_backend[backend].append(record)
                all_runs.append(strip_private(record))
                print(
                    f"  {backend:12s} prefill={record['prefill_s']:.2f}s "
                    f"decode={record['decode_s']:.2f}s "
                    f"prefill_peak={record['prefill_peak_allocated_gib']:.3f}GiB "
                    f"decode_peak={record['decode_peak_allocated_gib']:.3f}GiB "
                    f"finite={record['finite']}",
                    flush=True,
                )

        correctness = compare_captured(
            captured["eager_bf16"], captured["flash_bf16"],
        )
        summary = {}
        for backend, records in by_backend.items():
            summary[backend] = {
                "median_prefill_s": median(records, "prefill_s"),
                "median_decode_s": median(records, "decode_s"),
                "median_milliseconds_per_token": median(
                    records, "milliseconds_per_token",
                ),
                "median_prefill_peak_allocated_gib": median(
                    records, "prefill_peak_allocated_gib",
                ),
                "median_prefill_peak_reserved_gib": median(
                    records, "prefill_peak_reserved_gib",
                ),
                "median_decode_peak_allocated_gib": median(
                    records, "decode_peak_allocated_gib",
                ),
                "median_decode_peak_reserved_gib": median(
                    records, "decode_peak_reserved_gib",
                ),
                "median_decode_peak_extra_mib": median(
                    records, "decode_peak_extra_mib",
                ),
                "prefill_cache_mib": records[0]["prefill_cache_mib"],
                "final_cache_mib": records[0]["final_cache_mib"],
                "finite_all": all(record["finite"] for record in records),
                "stable_tokens": all(
                    record["tokens"] == records[0]["tokens"] for record in records
                ),
            }
        eager = summary["eager_bf16"]
        flash = summary["flash_bf16"]
        comparison = {
            "prefill_speedup": (
                eager["median_prefill_s"] / flash["median_prefill_s"]
            ),
            "decode_speedup": (
                eager["median_decode_s"] / flash["median_decode_s"]
            ),
            "prefill_peak_allocated_mib_delta": (
                flash["median_prefill_peak_allocated_gib"]
                - eager["median_prefill_peak_allocated_gib"]
            ) * 1024.0,
            "decode_peak_allocated_mib_delta": (
                flash["median_decode_peak_allocated_gib"]
                - eager["median_decode_peak_allocated_gib"]
            ) * 1024.0,
            "decode_peak_extra_mib_delta": (
                flash["median_decode_peak_extra_mib"]
                - eager["median_decode_peak_extra_mib"]
            ),
        }
        context_pass = (
            eager["finite_all"]
            and flash["finite_all"]
            and eager["stable_tokens"]
            and flash["stable_tokens"]
            and correctness["prefill_last_logits"]["argmax_same"]
            and correctness["prefill_last_logits"]["relative_l2"] <= 0.02
            and correctness["tokens_identical"]
        )
        overall_pass &= context_pass
        context_results[str(context)] = {
            "summary": summary,
            "correctness": correctness,
            "comparison": comparison,
            "pass": context_pass,
        }
        print(
            f"summary ctx={context}: prefill_speedup={comparison['prefill_speedup']:.3f}x "
            f"decode_speedup={comparison['decode_speedup']:.3f}x "
            f"prefill_peak_delta={comparison['prefill_peak_allocated_mib_delta']:+.1f}MiB "
            f"decode_peak_delta={comparison['decode_peak_allocated_mib_delta']:+.1f}MiB "
            f"tokens_equal={correctness['tokens_identical']} pass={context_pass}\n",
            flush=True,
        )
        del token_ids
        gc.collect()
        torch.cuda.empty_cache()

    decision = (
        "PASS_PRODUCTION_INTEGRATION_CANDIDATE"
        if overall_pass else "DO_NOT_MODIFY_PRODUCTION_ENGINE"
    )
    result = {
        "status": "runtime_only_custom_flash_attention_trial",
        "kernel_revision": KERNEL_REVISION,
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": str(DTYPE),
        "contexts": CONTEXTS,
        "chunk": CHUNK,
        "decode_tokens": DECODE_TOKENS,
        "orders": ORDERS,
        "correctness_gates": {
            "prefill_last_logit_relative_l2": 0.02,
            "prefill_argmax_same": True,
            "generated_tokens_identical": True,
        },
        "context_results": context_results,
        "decision": decision,
        "runs": all_runs,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"decision={decision}\nJSON={OUTPUT}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
