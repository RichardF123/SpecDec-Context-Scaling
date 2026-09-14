"""Isolated 8K whole-model trial for Triton V-FP8 decode attention.

The production ``model_self.py`` is not modified.  This harness installs a
runtime-only forward method on each attention module.  Prefill and every format
except native-head ``v_fp8`` continue to use the original eager method; only
single-token cached decode is redirected to the validated Triton block-128
kernel.

The experiment compares eager and Triton decode in four balanced-order rounds
on Qwen2.5-1.5B.  It reports generated-token identity, per-step logits drift,
decode latency, decode-only peak allocation, and cache bytes.  Results from this
file remain experimental until reviewed; importing it changes nothing.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time
import types

import torch

from . import model_self as ms
from .g2_operational import make_text
from .kv_policy_pilot_v2 import MODEL, load_model
from .kv_streaming_attention_triton_probe import triton_vfp8_attention


DTYPE = torch.bfloat16
CONTEXT = 8192
CHUNK = 2048
DECODE_TOKENS = 64
BLOCK_SIZE = 128
ROUNDS = (
    ("eager_vfp8", "triton_vfp8"),
    ("triton_vfp8", "eager_vfp8"),
    ("triton_vfp8", "eager_vfp8"),
    ("eager_vfp8", "triton_vfp8"),
)
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "16_triton_vfp8_engine_trial.json",
)


def _triton_trial_forward(self, x: torch.Tensor, cos, sin, cache=None):
    """Specialize only native-head, cached, single-token V-FP8 decode."""
    fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
    eligible = (
        cache is not None
        and x.shape[1] == 1
        and fmt == "v_fp8"
        and x.is_cuda
        and x.dtype == DTYPE
        and self.head_dim == 128
        and self.heads == 12
        and self.kv_heads == 2
        and self.cache_k_heads == self.kv_heads
        and self.cache_v_heads == self.kv_heads
        and cache["k"].dtype == DTYPE
        and cache["v"].dtype == torch.float8_e4m3fn
        and "v_scale" in cache
    )
    if not eligible:
        return self._triton_trial_original_forward(x, cos, sin, cache)

    batch, sequence, _ = x.shape
    query = self.q_proj(x).view(
        batch, sequence, self.heads, self.head_dim
    ).transpose(1, 2)
    key = self.k_proj(x).view(
        batch, sequence, self.kv_heads, self.head_dim
    ).transpose(1, 2)
    value = self.v_proj(x).view(
        batch, sequence, self.kv_heads, self.head_dim
    ).transpose(1, 2)
    query = ms.apply_rope(query, cos, sin).contiguous()
    key = ms.apply_rope(key, cos, sin)

    current_value, current_scale = ms.quantize_kv_fp8(value, ms.GROUP_V)
    key_cache = torch.cat([cache["k"], key], dim=2).contiguous()
    value_cache = torch.cat([cache["v"], current_value], dim=2).contiguous()
    scale_cache = torch.cat(
        [cache["v_scale"], current_scale], dim=2
    ).contiguous()

    attention_output = triton_vfp8_attention(
        query,
        key_cache,
        value_cache,
        scale_cache,
        BLOCK_SIZE,
    )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch, sequence, self.heads * self.head_dim
    )
    new_cache = {
        "fmt": "v_fp8",
        "k": key_cache,
        "v": value_cache,
        "v_scale": scale_cache,
    }
    return self.o_proj(attention_output), new_cache


def install_trial_backend(model) -> None:
    """Install runtime dispatch while retaining each module's original method."""
    for layer in model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_triton_trial_original_forward"):
            attention._triton_trial_original_forward = attention.forward
        attention.forward = types.MethodType(_triton_trial_forward, attention)


def select_backend(model, backend: str) -> None:
    if backend not in ("eager_vfp8", "triton_vfp8"):
        raise ValueError(backend)
    for layer in model.layers:
        attention = layer.self_attn
        if backend == "triton_vfp8":
            attention.forward = types.MethodType(_triton_trial_forward, attention)
        else:
            attention.forward = attention._triton_trial_original_forward


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def run_once(model, token_ids, eos_token_id: int, backend: str, capture_logits: bool):
    select_backend(model, backend)
    model.set_kv_head_policy(2, 2)
    model.set_kv_format("v_fp8")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache = None
    last_logits = None
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, CONTEXT, CHUNK):
            piece = token_ids[:, start:start + CHUNK]
            is_last = start + CHUNK >= CONTEXT
            logits, cache = model._run(
                model.embed_tokens(piece), cache, need_logits=is_last
            )
            if is_last:
                last_logits = logits
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    prefill_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    prefill_cache_mib = cache_logical_bytes(cache) / 2**20

    current_logits = last_logits[:, -1:].contiguous()
    del last_logits
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    decode_floor_gib = torch.cuda.memory_allocated() / 2**30

    tokens = []
    captured_logits = []
    finite = bool(torch.isfinite(current_logits).all().item())
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(DECODE_TOKENS):
            next_id = int(current_logits[:, -1].argmax())
            if next_id == eos_token_id:
                break
            tokens.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=token_ids.device)
            current_logits, cache = model._run(
                model.embed_tokens(next_tensor), cache, need_logits=True
            )
            finite &= bool(torch.isfinite(current_logits).all().item())
            if capture_logits:
                captured_logits.append(
                    current_logits[:, -1].to(torch.float32).cpu()
                )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    decode_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    decode_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30

    return {
        "backend": backend,
        "prefill_s": prefill_seconds,
        "decode_s": decode_seconds,
        "decode_tokens": len(tokens),
        "milliseconds_per_token": (
            1000.0 * decode_seconds / max(len(tokens), 1)
        ),
        "tokens": tokens,
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
        "_captured_logits": captured_logits,
    }


def compare_logits(reference, candidate):
    if len(reference) != len(candidate):
        return {
            "steps_compared": min(len(reference), len(candidate)),
            "same_length": False,
            "all_step_argmax_same": False,
        }
    per_step = []
    for reference_step, candidate_step in zip(reference, candidate):
        difference = candidate_step - reference_step
        denominator = torch.linalg.vector_norm(reference_step).clamp_min(1e-12)
        per_step.append(
            {
                "max_abs": difference.abs().max().item(),
                "mean_abs": difference.abs().mean().item(),
                "relative_l2": (
                    torch.linalg.vector_norm(difference) / denominator
                ).item(),
                "argmax_same": bool(
                    reference_step.argmax().item() == candidate_step.argmax().item()
                ),
            }
        )
    return {
        "steps_compared": len(per_step),
        "same_length": True,
        "all_step_argmax_same": all(row["argmax_same"] for row in per_step),
        "worst_max_abs": max(
            (row["max_abs"] for row in per_step), default=0.0
        ),
        "worst_relative_l2": max(
            (row["relative_l2"] for row in per_step), default=0.0
        ),
        "median_relative_l2": statistics.median(
            [row["relative_l2"] for row in per_step]
        ) if per_step else 0.0,
        "per_step": per_step,
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
    token_ids = make_text(tokenizer, CONTEXT).to("cuda")
    model = load_model()
    install_trial_backend(model)

    by_backend = {"eager_vfp8": [], "triton_vfp8": []}
    captured = {}
    serialized_runs = []
    print(
        f"isolated whole-model Triton trial: model=1.5B ctx={CONTEXT} "
        f"chunk={CHUNK} decode={DECODE_TOKENS} block={BLOCK_SIZE} "
        f"dtype={DTYPE}",
        flush=True,
    )
    print(
        "four balanced-order rounds; production model_self.py is unchanged\n",
        flush=True,
    )
    experiment_started = time.perf_counter()

    for round_index, order in enumerate(ROUNDS, start=1):
        print(f"=== round {round_index} order={','.join(order)} ===", flush=True)
        for backend in order:
            capture_logits = backend not in captured
            record = run_once(
                model,
                token_ids,
                tokenizer.eos_token_id,
                backend,
                capture_logits,
            )
            if capture_logits:
                captured[backend] = record.pop("_captured_logits")
            else:
                record.pop("_captured_logits")
            record["round"] = round_index
            by_backend[backend].append(record)
            serialized_runs.append(record)
            print(
                f"{backend:16s} prefill={record['prefill_s']:.2f}s "
                f"decode={record['decode_s']:.2f}s/{record['decode_tokens']}tok "
                f"ms/tok={record['milliseconds_per_token']:.2f} "
                f"decode_peak={record['decode_peak_allocated_gib']:.3f}GiB "
                f"decode_extra={record['decode_peak_extra_mib']:.1f}MiB "
                f"finite={record['finite']}",
                flush=True,
            )

    all_records = serialized_runs
    reference_tokens = by_backend["eager_vfp8"][0]["tokens"]
    tokens_identical = all(
        record["tokens"] == reference_tokens for record in all_records
    )
    logits = compare_logits(
        captured["eager_vfp8"], captured["triton_vfp8"]
    )
    summary = {}
    for backend, records in by_backend.items():
        summary[backend] = {
            "median_prefill_s": median(records, "prefill_s"),
            "median_decode_s": median(records, "decode_s"),
            "median_milliseconds_per_token": median(
                records, "milliseconds_per_token"
            ),
            "median_decode_peak_allocated_gib": median(
                records, "decode_peak_allocated_gib"
            ),
            "median_decode_peak_reserved_gib": median(
                records, "decode_peak_reserved_gib"
            ),
            "median_decode_peak_extra_mib": median(
                records, "decode_peak_extra_mib"
            ),
            "prefill_cache_mib": records[0]["prefill_cache_mib"],
            "final_cache_mib": records[0]["final_cache_mib"],
            "stable_tokens_across_rounds": all(
                row["tokens"] == records[0]["tokens"] for row in records
            ),
            "finite_all": all(row["finite"] for row in records),
        }

    eager = summary["eager_vfp8"]
    triton_result = summary["triton_vfp8"]
    comparison = {
        "tokens_identical_all_runs_and_backends": tokens_identical,
        "decode_speedup": (
            eager["median_decode_s"] / triton_result["median_decode_s"]
        ),
        "decode_latency_pct_change": (
            triton_result["median_decode_s"] / eager["median_decode_s"] - 1.0
        ) * 100.0,
        "decode_peak_allocated_mib_delta": (
            triton_result["median_decode_peak_allocated_gib"]
            - eager["median_decode_peak_allocated_gib"]
        ) * 1024.0,
        "decode_peak_extra_mib_delta": (
            triton_result["median_decode_peak_extra_mib"]
            - eager["median_decode_peak_extra_mib"]
        ),
        "logits": logits,
    }
    pass_gate = (
        tokens_identical
        and logits.get("all_step_argmax_same", False)
        and eager["finite_all"]
        and triton_result["finite_all"]
        and comparison["decode_speedup"] >= 1.05
        and comparison["decode_peak_extra_mib_delta"] < 0.0
    )
    result = {
        "status": "isolated_runtime_monkeypatch_no_production_change",
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "context": CONTEXT,
        "chunk": CHUNK,
        "decode_tokens": DECODE_TOKENS,
        "block_size": BLOCK_SIZE,
        "orders": ROUNDS,
        "elapsed_s": time.perf_counter() - experiment_started,
        "summary": summary,
        "comparison": comparison,
        "decision": (
            "PASS_FORMAL_INTEGRATION_CANDIDATE"
            if pass_gate
            else "DO_NOT_MODIFY_PRODUCTION_ENGINE"
        ),
        "runs": serialized_runs,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print("\n=== WHOLE-MODEL SUMMARY ===")
    for backend in ("eager_vfp8", "triton_vfp8"):
        row = summary[backend]
        print(
            f"{backend:16s} decode={row['median_decode_s']:.2f}s "
            f"ms/tok={row['median_milliseconds_per_token']:.2f} "
            f"decode_peak={row['median_decode_peak_allocated_gib']:.3f}GiB "
            f"decode_extra={row['median_decode_peak_extra_mib']:.1f}MiB "
            f"stable={row['stable_tokens_across_rounds']} "
            f"finite={row['finite_all']}"
        )
    print(
        f"triton vs eager: speedup={comparison['decode_speedup']:.3f}x "
        f"latency_delta={comparison['decode_latency_pct_change']:+.1f}% "
        f"peak_alloc_delta={comparison['decode_peak_allocated_mib_delta']:+.1f}MiB "
        f"peak_extra_delta={comparison['decode_peak_extra_mib_delta']:+.1f}MiB"
    )
    print(
        f"tokens_identical={tokens_identical} "
        f"logit_argmax_same={logits.get('all_step_argmax_same')} "
        f"worst_logit_rel_l2={logits.get('worst_relative_l2')}"
    )
    print(f"decision={result['decision']}\nJSON={OUTPUT}")
    return 0 if tokens_identical and eager["finite_all"] and triton_result["finite_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
