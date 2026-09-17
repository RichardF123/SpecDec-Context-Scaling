"""Bounded whole-model numerical screen for the G4 Flash backend.

This is deliberately narrower than the compatibility gate.  It answers one
question after a real-prompt greedy divergence: can a larger key tile and/or
CUDA libdevice exponentiation remove the divergence on the *same* token
prefix?  Production dispatch is not changed by this diagnostic.

The screen runs one 1.5B target model, one real WikiText prompt, and only a
few generated tokens.  Every candidate is checked both free-running and with
the eager token stream teacher-forced so post-divergence logits are never
mistaken for a kernel error.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time

import torch

from .g4_b4_cache_tests import load_ours
from .g4_part4_pilot_measure import load_text
from . import flash_attention_two_pass_triton as flash_impl


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT = int(os.environ.get("FLASH_NUMERIC_CTX", "2048"))
MAX_NEW = int(os.environ.get("FLASH_NUMERIC_MAXNEW", "4"))
CHUNK = int(os.environ.get("FLASH_NUMERIC_CHUNK", "512"))


CANDIDATES = (
    {
        "name": "fast_bn32",
        "prefill_block_n": 32,
        "decode_block_n": 32,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": False,
    },
    {
        "name": "current_fast_bn64",
        "prefill_block_n": 64,
        "decode_block_n": 64,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": False,
    },
    {
        "name": "fast_bn128",
        "prefill_block_n": 128,
        "decode_block_n": 128,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": False,
    },
    {
        "name": "fast_prefill128_decode64",
        "prefill_block_n": 128,
        "decode_block_n": 64,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": False,
    },
    {
        "name": "fast_prefill64_decode128",
        "prefill_block_n": 64,
        "decode_block_n": 128,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": False,
    },
    {
        "name": "accurate_prefill_bn64",
        "prefill_block_n": 64,
        "decode_block_n": 64,
        "prefill_accurate_exp": True,
        "decode_accurate_exp": False,
    },
    {
        "name": "accurate_decode_bn64",
        "prefill_block_n": 64,
        "decode_block_n": 64,
        "prefill_accurate_exp": False,
        "decode_accurate_exp": True,
    },
    {
        "name": "accurate_all_bn32",
        "prefill_block_n": 32,
        "decode_block_n": 32,
        "prefill_accurate_exp": True,
        "decode_accurate_exp": True,
    },
    {
        "name": "accurate_all_bn64",
        "prefill_block_n": 64,
        "decode_block_n": 64,
        "prefill_accurate_exp": True,
        "decode_accurate_exp": True,
    },
    {
        "name": "accurate_all_bn128",
        "prefill_block_n": 128,
        "decode_block_n": 128,
        "prefill_accurate_exp": True,
        "decode_accurate_exp": True,
    },
)


def synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


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
    return logits, cache


def run_stream(model, ids, backend, forced_tokens=None):
    model.set_kv_format("fp16")
    model.set_attention_backend(backend)
    started = synchronize()
    logits, cache = chunked_prefill(model, ids)
    rows = []
    predictions = []
    with torch.inference_mode():
        for index in range(MAX_NEW):
            row = logits[:, -1].float().cpu()
            rows.append(row)
            predicted = int(row.argmax())
            predictions.append(predicted)
            if index == MAX_NEW - 1:
                break
            token = (
                predicted if forced_tokens is None
                else int(forced_tokens[index])
            )
            token_tensor = torch.tensor([[token]], device=ids.device)
            logits, cache = model._run(
                model.embed_tokens(token_tensor), cache, need_logits=True,
            )
    elapsed_s = synchronize() - started
    return {
        "predictions": predictions,
        "rows": rows,
        "elapsed_s": elapsed_s,
        "finite": all(bool(torch.isfinite(row).all()) for row in rows),
    }


def compare_rows(reference, candidate):
    records = []
    for index, (eager, flash) in enumerate(zip(reference, candidate)):
        difference = flash - eager
        denominator = torch.linalg.vector_norm(eager).clamp_min(1e-12)
        eager_top2 = torch.topk(eager.flatten(), k=2)
        flash_top2 = torch.topk(flash.flatten(), k=2)
        records.append({
            "step": index,
            "eager_argmax": int(eager.argmax()),
            "flash_argmax": int(flash.argmax()),
            "argmax_same": bool(eager.argmax() == flash.argmax()),
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference) / denominator
            ),
            "eager_top1_margin": float(eager_top2.values[0] - eager_top2.values[1]),
            "flash_top1_margin": float(flash_top2.values[0] - flash_top2.values[1]),
        })
    first_difference = next(
        (row["step"] for row in records if not row["argmax_same"]), None,
    )
    return {
        "records": records,
        "first_argmax_difference": first_difference,
        "all_argmax_same": first_difference is None,
        "worst_relative_l2": max(
            (row["relative_l2"] for row in records), default=0.0,
        ),
        "worst_max_abs": max(
            (row["max_abs"] for row in records), default=0.0,
        ),
    }


@contextmanager
def candidate_dispatch(candidate):
    original = flash_impl.flash_attention_bf16_two_pass

    def configured(query, key, value, *, causal=True, config=None):
        is_decode = query.shape[2] == 1
        selected = flash_impl.FlashKernelConfig(
            block_m=16 if is_decode else 64,
            block_n=(
                candidate["decode_block_n"] if is_decode
                else candidate["prefill_block_n"]
            ),
            num_warps=4,
            num_stages=2,
            accurate_exp=(
                candidate["decode_accurate_exp"] if is_decode
                else candidate["prefill_accurate_exp"]
            ),
        )
        return original(
            query, key, value, causal=causal, config=selected,
        )

    flash_impl.flash_attention_bf16_two_pass = configured
    try:
        yield
    finally:
        flash_impl.flash_attention_bf16_two_pass = original


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")

    print(
        "Flash G4 numerical screen: "
        f"gpu={torch.cuda.get_device_name(0)} ctx={CONTEXT} "
        f"maxnew={MAX_NEW} chunk={CHUNK}",
        flush=True,
    )
    model, tokenizer = load_ours(MODEL)
    ids = tokenizer(text[:400_000], return_tensors="pt").input_ids[:, :CONTEXT].cuda()
    if ids.shape[1] != CONTEXT:
        raise RuntimeError(f"corpus only provided {ids.shape[1]} tokens")

    eager = run_stream(model, ids, "eager")
    eager_tokens = eager["predictions"]
    print(f"eager tokens={eager_tokens} elapsed={eager['elapsed_s']:.3f}s")

    results = []
    for candidate in CANDIDATES:
        torch.cuda.empty_cache()
        with candidate_dispatch(candidate):
            free = run_stream(model, ids, "triton_bf16_two_pass")
            forced = run_stream(
                model, ids, "triton_bf16_two_pass",
                forced_tokens=eager_tokens,
            )
        comparison = compare_rows(eager["rows"], forced["rows"])
        free_same = free["predictions"] == eager_tokens
        passed = bool(
            free_same and comparison["all_argmax_same"]
            and free["finite"] and forced["finite"]
        )
        record = {
            **candidate,
            "free_predictions": free["predictions"],
            "free_tokens_identical": free_same,
            "free_elapsed_s": free["elapsed_s"],
            "forced_elapsed_s": forced["elapsed_s"],
            "teacher_forced_comparison": comparison,
            "passed_screen": passed,
        }
        results.append(record)
        first = comparison["first_argmax_difference"]
        if first is None:
            first_text = "none"
        else:
            row = comparison["records"][first]
            first_text = (
                f"{first} eager={row['eager_argmax']} flash={row['flash_argmax']} "
                f"margin={row['eager_top1_margin']:.6f}"
            )
        print(
            f"{candidate['name']:25s} tokens_same={free_same} "
            f"first_diff={first_text} "
            f"worst_rel={comparison['worst_relative_l2']:.6f} "
            f"max_abs={comparison['worst_max_abs']:.6f} "
            f"elapsed={free['elapsed_s']:.3f}s pass={passed}",
            flush=True,
        )

    passing = [row["name"] for row in results if row["passed_screen"]]
    decision = (
        "PROCEED_WITH_FRESH_COMPATIBILITY_GATE"
        if passing else "STOP_NO_NUMERIC_CANDIDATE"
    )
    result = {
        "status": "flash_g4_whole_model_numerical_screen",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL,
        "context": CONTEXT,
        "max_new_tokens": MAX_NEW,
        "chunk": CHUNK,
        "text_source": source,
        "eager_tokens": eager_tokens,
        "candidates": results,
        "passing_candidates": passing,
        "decision": decision,
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_numerical_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"decision={decision}\nJSON={output}")

    del model, ids, eager
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
