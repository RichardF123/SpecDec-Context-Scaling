"""Screen custom-Triton small-query dispatch for G4 block semantics.

The authoritative G4 correctness contract compares speculative decoding with
target-only cached AR from the same engine/backend.  This bounded diagnostic
isolates the attention shape that distinguishes those paths: serial q_len=1
updates versus cached q_len=4 target verification.  It does not use an
official attention implementation, does not change production dispatch, and
does not modify ``spec_dec.py`` or the cache contract.

For every candidate launch policy, the script advances two target streams on
the same forced token prefix:

* serial stream: four cached q_len=1 calls;
* block stream: one cached q_len=4 call.

It compares the decision before each block, every same-context output row,
cache lengths, finiteness, and accumulated behavior across several rounds.
Any surviving policy must still pass the full same-backend G4 gate before it
can be considered for production dispatch.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time

import torch

from . import flash_attention_two_pass_triton as flash_impl
from . import spec_cache
from .g4_b4_cache_tests import load_ours
from .g4_part4_pilot_measure import load_text


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT = int(os.environ.get("FLASH_BLOCK_SERIAL_CTX", "512"))
TOKENS = int(os.environ.get("FLASH_BLOCK_SERIAL_TOKENS", "16"))
CHUNK = int(os.environ.get("FLASH_BLOCK_SERIAL_CHUNK", "512"))
K = 4


CANDIDATES = (
    {
        "name": "current_bm16_64_bn64",
        "q1_block_m": 16,
        "q4_block_m": 64,
        "block_n": 64,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm8_bn64",
        "q1_block_m": 8,
        "q4_block_m": 8,
        "block_n": 64,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm16_bn64",
        "q1_block_m": 16,
        "q4_block_m": 16,
        "block_n": 64,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm32_bn64",
        "q1_block_m": 32,
        "q4_block_m": 32,
        "block_n": 64,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm64_bn64",
        "q1_block_m": 64,
        "q4_block_m": 64,
        "block_n": 64,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm16_bn32",
        "q1_block_m": 16,
        "q4_block_m": 16,
        "block_n": 32,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm16_bn128",
        "q1_block_m": 16,
        "q4_block_m": 16,
        "block_n": 128,
        "accurate_exp": False,
    },
    {
        "name": "unified_bm16_bn64_accurate",
        "q1_block_m": 16,
        "q4_block_m": 16,
        "block_n": 64,
        "accurate_exp": True,
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


@contextmanager
def candidate_dispatch(candidate):
    original = flash_impl.flash_attention_bf16_two_pass

    def configured(query, key, value, *, causal=True, config=None):
        del config
        query_length = int(query.shape[2])
        if query_length == 1:
            block_m = candidate["q1_block_m"]
        elif query_length <= K:
            block_m = candidate["q4_block_m"]
        else:
            return original(query, key, value, causal=causal)
        selected = flash_impl.FlashKernelConfig(
            block_m=block_m,
            block_n=candidate["block_n"],
            num_warps=4,
            num_stages=2,
            accurate_exp=candidate["accurate_exp"],
        )
        return original(
            query, key, value, causal=causal, config=selected,
        )

    flash_impl.flash_attention_bf16_two_pass = configured
    try:
        yield
    finally:
        flash_impl.flash_attention_bf16_two_pass = original


def compare_rows(serial, block):
    difference = block.float() - serial.float()
    denominator = torch.linalg.vector_norm(serial.float()).clamp_min(1e-12)
    return {
        "serial_argmax": int(serial.argmax()),
        "block_argmax": int(block.argmax()),
        "argmax_same": bool(serial.argmax() == block.argmax()),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / denominator
        ),
        "finite": bool(
            torch.isfinite(serial).all() and torch.isfinite(block).all()
        ),
    }


def run_candidate(model, base_logits, base_cache, candidate):
    serial_logits = base_logits.clone()
    block_logits = base_logits.clone()
    serial_cache = clone_cache(base_cache)
    block_cache = clone_cache(base_cache)
    generated = 0
    records = []
    started = synchronize()

    with candidate_dispatch(candidate), torch.inference_mode():
        while generated < TOKENS:
            width = min(K, TOKENS - generated)
            round_record = {
                "round": len(records),
                "pre": spec_cache.cache_length(serial_cache),
                "width": width,
            }
            before = compare_rows(serial_logits[0, -1], block_logits[0, -1])
            candidates = []
            serial_rows = []
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
            block_output, block_cache = model._run(
                model.embed_tokens(candidate_tensor),
                block_cache,
                need_logits=True,
            )
            spec_cache.check_consistent(block_cache)
            row_checks = [
                compare_rows(serial_rows[index], block_output[0, index])
                for index in range(width)
            ]
            serial_length = spec_cache.cache_length(serial_cache)
            block_length = spec_cache.cache_length(block_cache)
            expected_length = round_record["pre"] + width
            round_record.update({
                "candidates": candidates,
                "before": before,
                "rows": row_checks,
                "serial_cache_length": serial_length,
                "block_cache_length": block_length,
                "expected_cache_length": expected_length,
                "cache_lengths_ok": (
                    serial_length == expected_length
                    and block_length == expected_length
                ),
                "passed": bool(
                    before["argmax_same"]
                    and before["finite"]
                    and all(row["argmax_same"] and row["finite"] for row in row_checks)
                    and serial_length == expected_length
                    and block_length == expected_length
                ),
            })
            records.append(round_record)
            block_logits = block_output[:, -1:]
            generated += width

    elapsed_s = synchronize() - started
    row_checks = [row for record in records for row in record["rows"]]
    all_checks = [record["before"] for record in records] + row_checks
    first_failure = None
    for record in records:
        if not record["before"]["argmax_same"]:
            first_failure = {
                "round": record["round"],
                "position": "before_block",
                **record["before"],
            }
            break
        for index, row in enumerate(record["rows"]):
            if not row["argmax_same"]:
                first_failure = {
                    "round": record["round"],
                    "position": index,
                    **row,
                }
                break
        if first_failure is not None:
            break

    passed = bool(records and all(record["passed"] for record in records))
    return {
        **candidate,
        "tokens_tested": generated,
        "rounds": records,
        "first_failure": first_failure,
        "worst_max_abs": max(
            (row["max_abs"] for row in all_checks), default=0.0,
        ),
        "worst_relative_l2": max(
            (row["relative_l2"] for row in all_checks), default=0.0,
        ),
        "elapsed_s": elapsed_s,
        "passed_screen": passed,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if TOKENS < K:
        raise ValueError("FLASH_BLOCK_SERIAL_TOKENS must be at least 4")

    torch.manual_seed(0)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")

    print(
        "custom Triton G4 block-vs-serial screen: "
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
    model.set_attention_backend("triton_bf16_two_pass")
    base_logits, base_cache = chunked_prefill(model, ids)
    base_length = spec_cache.cache_length(base_cache)
    if base_length != CONTEXT:
        raise RuntimeError(f"prefill cache {base_length}, expected {CONTEXT}")

    results = []
    for candidate in CANDIDATES:
        torch.cuda.empty_cache()
        row = run_candidate(model, base_logits, base_cache, candidate)
        results.append(row)
        failure = row["first_failure"]
        if failure is None:
            failure_text = "none"
        else:
            failure_text = (
                f"round={failure['round']} pos={failure['position']} "
                f"serial={failure['serial_argmax']} "
                f"block={failure['block_argmax']}"
            )
        print(
            f"{candidate['name']:28s} pass={row['passed_screen']} "
            f"first_failure={failure_text} "
            f"worst_rel={row['worst_relative_l2']:.6f} "
            f"max_abs={row['worst_max_abs']:.6f} "
            f"elapsed={row['elapsed_s']:.3f}s",
            flush=True,
        )

    passing = [row for row in results if row["passed_screen"]]
    passing.sort(key=lambda row: (row["worst_relative_l2"], row["elapsed_s"]))
    selected = passing[0]["name"] if passing else None
    decision = (
        "PROCEED_TO_SAME_BACKEND_G4_GATE"
        if selected is not None
        else "STOP_NO_BLOCK_SERIAL_CANDIDATE"
    )
    result = {
        "status": "custom_triton_g4_block_serial_dispatch_screen",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL,
        "context": CONTEXT,
        "tokens": TOKENS,
        "chunk": CHUNK,
        "text_source": source,
        "production_dispatch_unchanged": True,
        "candidates": results,
        "passing_candidates": [row["name"] for row in passing],
        "selected_candidate": selected,
        "decision": decision,
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_block_serial_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"decision={decision}\nselected_candidate={selected}\nJSON={output}",
        flush=True,
    )

    del model, ids, base_logits, base_cache
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
