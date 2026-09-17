"""Isolate the source of custom-Flash q_len=1 versus q_len=4 drift.

This target-only diagnostic runs identical forced token prefixes through
serial cached q_len=1 and block cached q_len=4 paths.  It varies the prefill
and online attention routes explicitly:

* CC: custom serial, custom block (current experimental backend);
* CE: custom serial, eager block;
* EC: eager serial, custom block;
* XE: custom prefill, eager serial, eager block;
* EE: eager prefill, eager serial, eager block (BF16 reference behavior);
* CS: custom serial, custom block API implemented as four full-model q_len=1
  calls (diagnostic upper bound, not a performance implementation).

No official FlashAttention implementation is used.  Production dispatch,
``spec_dec.py``, and the cache contract are not modified.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

import torch

from . import spec_cache
from .g4_b4_cache_tests import load_ours
from .g4_part4_pilot_measure import load_text


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT = int(os.environ.get("FLASH_OPERATOR_CTX", "512"))
TOKENS = int(os.environ.get("FLASH_OPERATOR_TOKENS", "16"))
CHUNK = int(os.environ.get("FLASH_OPERATOR_CHUNK", "512"))
K = 4

ROUTES = (
    {
        "name": "CC_custom_prefill_custom_serial_custom_block",
        "prefill_backend": "triton_bf16_two_pass",
        "serial_backend": "triton_bf16_two_pass",
        "block_backend": "triton_bf16_two_pass",
        "serialize_block": False,
    },
    {
        "name": "CE_custom_prefill_custom_serial_eager_block",
        "prefill_backend": "triton_bf16_two_pass",
        "serial_backend": "triton_bf16_two_pass",
        "block_backend": "eager",
        "serialize_block": False,
    },
    {
        "name": "EC_custom_prefill_eager_serial_custom_block",
        "prefill_backend": "triton_bf16_two_pass",
        "serial_backend": "eager",
        "block_backend": "triton_bf16_two_pass",
        "serialize_block": False,
    },
    {
        "name": "XE_custom_prefill_eager_serial_eager_block",
        "prefill_backend": "triton_bf16_two_pass",
        "serial_backend": "eager",
        "block_backend": "eager",
        "serialize_block": False,
    },
    {
        "name": "EE_eager_prefill_eager_serial_eager_block",
        "prefill_backend": "eager",
        "serial_backend": "eager",
        "block_backend": "eager",
        "serialize_block": False,
    },
    {
        "name": "CS_custom_prefill_custom_serial_serialized_custom_block",
        "prefill_backend": "triton_bf16_two_pass",
        "serial_backend": "triton_bf16_two_pass",
        "block_backend": "triton_bf16_two_pass",
        "serialize_block": True,
    },
)


def synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def clone_cache(cache):
    return [
        {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in layer.items()
        }
        for layer in cache
    ]


def chunked_prefill(model, ids):
    cache = None
    logits = None
    with torch.inference_mode():
        for start in range(0, ids.shape[1], CHUNK):
            piece = ids[:, start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece),
                cache,
                need_logits=(start + CHUNK >= ids.shape[1]),
            )
            spec_cache.check_consistent(cache)
    return logits[:, -1:], cache


def compare_rows(serial, block):
    serial32 = serial.float()
    block32 = block.float()
    difference = block32 - serial32
    denominator = torch.linalg.vector_norm(serial32).clamp_min(1e-12)
    serial_top2 = torch.topk(serial32.flatten(), k=2).values
    block_top2 = torch.topk(block32.flatten(), k=2).values
    return {
        "serial_argmax": int(serial.argmax()),
        "block_argmax": int(block.argmax()),
        "argmax_same": bool(serial.argmax() == block.argmax()),
        "serial_margin": float(serial_top2[0] - serial_top2[1]),
        "block_margin": float(block_top2[0] - block_top2[1]),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / denominator
        ),
        "finite": bool(
            torch.isfinite(serial32).all() and torch.isfinite(block32).all()
        ),
    }


def serialized_run(model, embeddings, cache):
    rows = []
    current = cache
    for index in range(embeddings.shape[1]):
        logits, current = model._run(
            embeddings[:, index:index + 1], current, need_logits=True,
        )
        spec_cache.check_consistent(current)
        rows.append(logits)
    return torch.cat(rows, dim=1), current


def run_route(model, base_logits, base_cache, route):
    serial_logits = base_logits.clone()
    block_logits = base_logits.clone()
    serial_cache = clone_cache(base_cache)
    block_cache = clone_cache(base_cache)
    generated = 0
    records = []
    started = synchronize()

    with torch.inference_mode():
        while generated < TOKENS:
            width = min(K, TOKENS - generated)
            pre = spec_cache.cache_length(serial_cache)
            before = compare_rows(serial_logits[0, -1], block_logits[0, -1])

            candidates = []
            serial_rows = []
            model.set_attention_backend(route["serial_backend"])
            for _ in range(width):
                token = int(serial_logits[0, -1].argmax())
                candidates.append(token)
                token_tensor = torch.tensor(
                    [[token]], device=serial_logits.device,
                )
                serial_logits, serial_cache = model._run(
                    model.embed_tokens(token_tensor),
                    serial_cache,
                    need_logits=True,
                )
                spec_cache.check_consistent(serial_cache)
                serial_rows.append(serial_logits[0, -1].clone())

            candidate_tensor = torch.tensor(
                [candidates], device=serial_logits.device,
            )
            embeddings = model.embed_tokens(candidate_tensor)
            model.set_attention_backend(route["block_backend"])
            if route["serialize_block"]:
                block_output, block_cache = serialized_run(
                    model, embeddings, block_cache,
                )
            else:
                block_output, block_cache = model._run(
                    embeddings, block_cache, need_logits=True,
                )
                spec_cache.check_consistent(block_cache)

            row_checks = [
                compare_rows(serial_rows[index], block_output[0, index])
                for index in range(width)
            ]
            expected_length = pre + width
            serial_length = spec_cache.cache_length(serial_cache)
            block_length = spec_cache.cache_length(block_cache)
            passed = bool(
                before["argmax_same"]
                and before["finite"]
                and all(row["argmax_same"] and row["finite"] for row in row_checks)
                and serial_length == expected_length
                and block_length == expected_length
            )
            records.append({
                "round": len(records),
                "pre": pre,
                "width": width,
                "candidates": candidates,
                "before": before,
                "rows": row_checks,
                "serial_cache_length": serial_length,
                "block_cache_length": block_length,
                "expected_cache_length": expected_length,
                "passed": passed,
            })
            block_logits = block_output[:, -1:]
            generated += width

    elapsed_s = synchronize() - started
    checks = [record["before"] for record in records]
    checks += [row for record in records for row in record["rows"]]
    first_failure = None
    for record in records:
        if not record["before"]["argmax_same"]:
            first_failure = {
                "round": record["round"],
                "position": "before_block",
                **record["before"],
            }
            break
        for position, row in enumerate(record["rows"]):
            if not row["argmax_same"]:
                first_failure = {
                    "round": record["round"],
                    "position": position,
                    **row,
                }
                break
        if first_failure is not None:
            break

    return {
        **route,
        "tokens_tested": generated,
        "rounds": records,
        "first_failure": first_failure,
        "worst_max_abs": max(
            (row["max_abs"] for row in checks), default=0.0,
        ),
        "worst_relative_l2": max(
            (row["relative_l2"] for row in checks), default=0.0,
        ),
        "elapsed_s": elapsed_s,
        "passed": bool(records and all(row["passed"] for row in records)),
    }


def classify(results):
    by_name = {row["name"]: row for row in results}
    cc = by_name[
        "CC_custom_prefill_custom_serial_custom_block"
    ]["passed"]
    ce = by_name[
        "CE_custom_prefill_custom_serial_eager_block"
    ]["passed"]
    ec = by_name[
        "EC_custom_prefill_eager_serial_custom_block"
    ]["passed"]
    xe = by_name[
        "XE_custom_prefill_eager_serial_eager_block"
    ]["passed"]
    ee = by_name[
        "EE_eager_prefill_eager_serial_eager_block"
    ]["passed"]
    cs = by_name[
        "CS_custom_prefill_custom_serial_serialized_custom_block"
    ]["passed"]
    if not cs:
        return "CHECK_SERIALIZED_CONTROL_OR_HARNESS"
    if cc:
        return "CUSTOM_BLOCK_SEMANTICS_PASS"
    if ce:
        return "CUSTOM_Q4_ATTENTION_SUSPECT"
    if ec:
        return "CUSTOM_Q1_ATTENTION_SUSPECT"
    if not ee:
        return "BF16_BLOCK_SHAPE_DRIFT_PRESENT_IN_EAGER_REFERENCE"
    if not xe:
        return "CUSTOM_PREFILL_AMPLIFIES_BLOCK_SHAPE_DRIFT"
    return "CUSTOM_ATTENTION_INTERACTION_SUSPECT"


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if TOKENS < K:
        raise ValueError("FLASH_OPERATOR_TOKENS must be at least 4")

    torch.manual_seed(0)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")

    print(
        "custom Triton G4 operator isolation: "
        f"gpu={torch.cuda.get_device_name(0)} ctx={CONTEXT} "
        f"tokens={TOKENS} chunk={CHUNK}",
        flush=True,
    )
    model, tokenizer = load_ours(MODEL)
    ids = tokenizer(
        text[:400_000], return_tensors="pt",
    ).input_ids[:, :CONTEXT].cuda()
    if ids.shape[1] != CONTEXT:
        raise RuntimeError(f"corpus only provided {ids.shape[1]} tokens")

    model.set_kv_format("fp16")
    base_states = {}
    for backend in ("triton_bf16_two_pass", "eager"):
        model.set_attention_backend(backend)
        base_logits, base_cache = chunked_prefill(model, ids)
        if spec_cache.cache_length(base_cache) != CONTEXT:
            raise RuntimeError(f"{backend} prefill cache length mismatch")
        base_states[backend] = (base_logits, base_cache)

    results = []
    for route in ROUTES:
        torch.cuda.empty_cache()
        base_logits, base_cache = base_states[route["prefill_backend"]]
        row = run_route(model, base_logits, base_cache, route)
        results.append(row)
        failure = row["first_failure"]
        if failure is None:
            failure_text = "none"
        else:
            failure_text = (
                f"round={failure['round']} pos={failure['position']} "
                f"serial={failure['serial_argmax']} "
                f"block={failure['block_argmax']} "
                f"serial_margin={failure['serial_margin']:.6f} "
                f"block_margin={failure['block_margin']:.6f}"
            )
        print(
            f"{route['name']:42s} pass={row['passed']} "
            f"first_failure={failure_text} "
            f"worst_rel={row['worst_relative_l2']:.6f} "
            f"max_abs={row['worst_max_abs']:.6f} "
            f"elapsed={row['elapsed_s']:.3f}s",
            flush=True,
        )

    decision = classify(results)
    result = {
        "status": "custom_triton_g4_operator_source_isolation",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL,
        "context": CONTEXT,
        "tokens": TOKENS,
        "chunk": CHUNK,
        "text_source": source,
        "production_dispatch_unchanged": True,
        "interpretation_note": (
            "BF16 block-vs-serial exact-token drift is behavior evidence, not "
            "by itself a mainline implementation failure; classify with the "
            "repository's tie/noise and high-precision correctness protocol."
        ),
        "routes": results,
        "decision": decision,
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_operator_isolation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"decision={decision}\nJSON={output}", flush=True)

    del model, ids, base_states, base_logits, base_cache
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if decision != "CHECK_SERIALIZED_CONTROL_OR_HARNESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
