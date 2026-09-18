"""Supporting FlashAttention ablation for the latest G4 workflow.

This is deliberately outside the frozen 3x2x2 G4 matrix.  It compares only
BF16 eager and the opt-in BF16 two-pass Triton backend for AR and SpecDec while
both target and draft models are resident.  Measurement discipline mirrors
``g4_part4_pilot_measure.py``: real nested WikiText prefixes, chunked 8K
prefill, discarded warm-up, synchronized timing, >=3 repetitions, median and
range, allocation/reservation peaks, OOM-safe cleanup, and hashed JSONL output.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from . import spec_cache, spec_dec
from .flash_attention_two_pass_triton import KERNEL_REVISION
from .g4_b4_cache_tests import _from_pretrained, load_ours
from .g4_part4_pilot_measure import DRAFT, TARGET, load_text


MAIN_FRESH_START = "d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897"
PR2_REFERENCE = "b1904b5d867dd6eb7262f733d1818c940145c819"
CONTEXTS = tuple(int(value) for value in os.environ.get(
    "FLASH_G4_CTX", "512,2048,8192",
).split(",") if value.strip())
REPETITIONS = int(os.environ.get("FLASH_G4_REPS", "3"))
WARMUP = int(os.environ.get("FLASH_G4_WARM", "1"))
MAX_NEW = int(os.environ.get("FLASH_G4_MAXNEW", "64"))
CHUNK = int(os.environ.get("FLASH_G4_CHUNK", "2048"))
BACKENDS = ("eager", "triton_bf16_two_pass")
MODES = ("AR", "SpecDec")
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(os.environ.get(
    "SPECDEC_FLASH_G4_OUTDIR", str(ROOT / "output" / "flash_g4"),
))


def sync_time() -> float:
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
            spec_cache.check_consistent(cache)
    return logits, cache


def run_ar(target, ids, eos_id):
    started = sync_time()
    logits, cache = chunked_prefill(target, ids)
    prefill_s = sync_time() - started
    generated = []
    target_calls = 0
    started = sync_time()
    with torch.inference_mode():
        for index in range(MAX_NEW):
            token = int(logits[:, -1].argmax())
            generated.append(token)
            if token == eos_id or index == MAX_NEW - 1:
                break
            logits, cache = spec_dec._step(target, token, cache)
            target_calls += 1
            spec_cache.check_consistent(cache)
    decode_s = sync_time() - started
    return generated, prefill_s, decode_s, {
        "rounds": 0,
        "proposed": 0,
        "accepted": 0,
        "mismatch_rounds": 0,
        "all_accept_rounds": 0,
        "rollbacks": 0,
        "target_block_calls": 0,
        "draft_calls": 0,
        "target_decode_calls": target_calls,
        "m_hist": [],
        "final_cache_length": spec_cache.cache_length(cache),
    }


def run_specdec(target, draft, ids, eos_id):
    prefill = {"seconds": 0.0}
    original_prefill = spec_dec._prefill

    def timed_prefill(model, prompt):
        started = sync_time()
        logits, cache = chunked_prefill(model, prompt)
        prefill["seconds"] += sync_time() - started
        return logits, cache

    trace = []
    spec_dec._prefill = timed_prefill
    try:
        started = sync_time()
        result = spec_dec.spec_decode_greedy(
            target,
            draft,
            ids,
            MAX_NEW,
            k=spec_dec.K,
            eos_id=eos_id,
            target_kv="base_bf16",
            draft_kv="base_bf16",
            log=lambda row: trace.append(row),
        )
        total_s = sync_time() - started
    finally:
        spec_dec._prefill = original_prefill
    stats = dict(result["stats"])
    stats.update({
        "target_decode_calls": 0,
        "m_hist": [row["m"] for row in trace],
        "final_cache_length": trace[-1]["tgt_len"] if trace else ids.shape[1],
        "cache_invariants": all(
            row["tgt_len"] == row["drf_len"]
            == row["pre"] + (row["m"] + 1 if row["m"] < row["Kp"] else row["Kp"])
            for row in trace
        ),
    })
    return result["gen"], prefill["seconds"], total_s - prefill["seconds"], stats


def run_once(target, draft, ids, eos_id, backend, mode):
    target.set_kv_format("fp16")
    draft.set_kv_format("fp16")
    target.set_attention_backend(backend)
    draft.set_attention_backend("eager")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weights_resident = torch.cuda.memory_allocated() / 2**30
    if mode == "AR":
        generated, prefill_s, decode_s, stats = run_ar(target, ids, eos_id)
    else:
        generated, prefill_s, decode_s, stats = run_specdec(
            target, draft, ids, eos_id,
        )
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    total_s = prefill_s + decode_s
    row = {
        "ctx": int(ids.shape[1]),
        "mode": mode,
        "backend": backend,
        "generated": generated,
        "n_generated": len(generated),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tok_s": len(generated) / decode_s,
        "e2e_tok_s": len(generated) / total_s,
        "weights_resident_gib": weights_resident,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "peak_allocated_delta_gib": peak_allocated - weights_resident,
        **stats,
    }
    row["accept_rate"] = (
        row["accepted"] / row["proposed"] if row["proposed"] else None
    )
    return row


def summarize(records):
    numeric = (
        "prefill_s", "decode_s", "total_s", "decode_tok_s", "e2e_tok_s",
        "weights_resident_gib", "peak_allocated_gib", "peak_reserved_gib",
        "peak_allocated_delta_gib", "rounds", "proposed", "accepted",
        "mismatch_rounds", "all_accept_rounds", "rollbacks",
        "target_block_calls", "draft_calls", "target_decode_calls",
    )
    result = {
        "ctx": records[0]["ctx"],
        "mode": records[0]["mode"],
        "backend": records[0]["backend"],
        "repetitions": len(records),
        "stable_generated": all(
            row["generated"] == records[0]["generated"] for row in records
        ),
        "generated": records[0]["generated"],
        "m_hist": [value for row in records for value in row["m_hist"]],
    }
    for key in numeric:
        values = [float(row[key]) for row in records]
        result[key] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    rates = [row["accept_rate"] for row in records if row["accept_rate"] is not None]
    result["accept_rate"] = (
        {
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        } if rates else None
    )
    result["cache_invariants"] = all(
        row.get("cache_invariants", True) for row in records
    )
    return result


def environment():
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
        "gpu_total_memory_gib": properties.total_memory / 2**30,
        "triton": __import__("triton").__version__,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if REPETITIONS < 3:
        raise ValueError("FLASH_G4_REPS must be at least 3")
    torch.manual_seed(0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")
    from transformers import AutoTokenizer

    tokenizer = _from_pretrained(
        AutoTokenizer, TARGET, trust_remote_code=True,
    )
    tokenizer.model_max_length = 10**9
    all_ids = tokenizer(text[:400_000], return_tensors="pt").input_ids
    if all_ids.shape[1] < max(CONTEXTS):
        raise RuntimeError(
            f"corpus has {all_ids.shape[1]} tokens, need {max(CONTEXTS)}"
        )
    print("[load] target 1.5B", flush=True)
    target, _ = load_ours(TARGET)
    print("[load] draft 0.5B", flush=True)
    draft, _ = load_ours(DRAFT)
    env = environment()
    config = {
        "kind": "supporting_flash_g4_ablation",
        "main_fresh_start": MAIN_FRESH_START,
        "pr2_reference": PR2_REFERENCE,
        "kernel_revision": KERNEL_REVISION,
        "contexts": CONTEXTS,
        "modes": MODES,
        "backends": BACKENDS,
        "repetitions": REPETITIONS,
        "warmup": WARMUP,
        "max_new": MAX_NEW,
        "chunk": CHUNK,
        "k": spec_dec.K,
        "target": TARGET,
        "draft": DRAFT,
        "text_source": source,
        "nested_prefix": True,
        "formal_g4_matrix_changed": False,
        **env,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    output = OUTDIR / f"flash_g4_ablation_{config_hash}.jsonl"
    summaries = {}
    failed = False
    with output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"config_hash": config_hash, **config}) + "\n")
        for context in CONTEXTS:
            ids = all_ids[:, :context].cuda()
            for mode in MODES:
                print(f"\n=== ctx={context} mode={mode} ===", flush=True)
                for backend in BACKENDS:
                    for _ in range(WARMUP):
                        try:
                            run_once(target, draft, ids, tokenizer.eos_token_id, backend, mode)
                        except torch.cuda.OutOfMemoryError:
                            gc.collect()
                            torch.cuda.empty_cache()
                            break
                records = {backend: [] for backend in BACKENDS}
                oom = {backend: False for backend in BACKENDS}
                for repetition in range(REPETITIONS):
                    order = BACKENDS if repetition % 2 == 0 else tuple(reversed(BACKENDS))
                    for backend in order:
                        if oom[backend]:
                            continue
                        try:
                            row = run_once(
                                target, draft, ids, tokenizer.eos_token_id,
                                backend, mode,
                            )
                        except torch.cuda.OutOfMemoryError:
                            oom[backend] = True
                            failed = True
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(f"  {backend}: OOM", flush=True)
                            continue
                        row["repetition"] = repetition
                        records[backend].append(row)
                        serializable = {
                            key: value for key, value in row.items()
                            if key != "generated"
                        }
                        stream.write(json.dumps({
                            "kind": "run", "config_hash": config_hash,
                            **serializable,
                        }) + "\n")
                        stream.flush()
                        print(
                            f"  rep={repetition} {backend:24s} "
                            f"prefill={row['prefill_s']:.3f}s "
                            f"decode={row['decode_s']:.3f}s "
                            f"peak={row['peak_allocated_gib']:.3f}GiB",
                            flush=True,
                        )
                pair = {}
                for backend in BACKENDS:
                    if records[backend]:
                        pair[backend] = summarize(records[backend])
                    else:
                        pair[backend] = {"oom": True}
                if all(records[backend] for backend in BACKENDS):
                    eager = pair["eager"]
                    flash = pair["triton_bf16_two_pass"]
                    output_same = eager["generated"] == flash["generated"]
                    comparison = {
                        "output_same": output_same,
                        "prefill_speedup": (
                            eager["prefill_s"]["median"]
                            / flash["prefill_s"]["median"]
                        ),
                        "decode_speedup": (
                            eager["decode_s"]["median"]
                            / flash["decode_s"]["median"]
                        ),
                        "total_speedup": (
                            eager["total_s"]["median"]
                            / flash["total_s"]["median"]
                        ),
                        "peak_allocated_mib_delta": (
                            flash["peak_allocated_gib"]["median"]
                            - eager["peak_allocated_gib"]["median"]
                        ) * 1024,
                        "peak_reserved_mib_delta": (
                            flash["peak_reserved_gib"]["median"]
                            - eager["peak_reserved_gib"]["median"]
                        ) * 1024,
                    }
                    failed |= not output_same
                else:
                    comparison = {"output_same": False, "oom": True}
                summary = {
                    "kind": "summary",
                    "config_hash": config_hash,
                    "ctx": context,
                    "mode": mode,
                    "backends": pair,
                    "comparison": comparison,
                }
                summaries[f"{context}:{mode}"] = summary
                stream.write(json.dumps(summary) + "\n")
                stream.flush()
                print(f"  comparison={comparison}", flush=True)
            del ids
            gc.collect()
            torch.cuda.empty_cache()
    final = {
        "kind": "final",
        "config_hash": config_hash,
        "decision": (
            "PASS_FLASH_G4_SUPPORTING_ABLATION"
            if not failed else "CHECK_FLASH_G4_SUPPORTING_ABLATION"
        ),
        "output": str(output),
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(final) + "\n")
    print(f"\n{final['decision']}\nJSONL={output}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
