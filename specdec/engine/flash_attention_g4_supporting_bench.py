"""G4-disciplined supporting ablation for the explicit Flash/eager hybrid.

This does not alter the frozen G4 3x2x2 matrix.  It measures the supporting
systems question ``eager_bf16`` versus the explicit hybrid at nested real-text
contexts for target-only AR and dual-model SpecDec.

Two forms of evidence are kept separate:

* one natural free-running validation per context/backend records tokens,
  acceptance statistics, traces, cache invariants, and whether workloads
  happen to match across floating-point paths;
* timing uses controlled replay of the eager reference tokens/SpecDec trace so
  both backends execute identical model-call shapes, token inputs, verification
  lengths, trims, and corrective steps.  This makes latency and memory deltas
  comparable even when natural BF16 greedy paths split at a near-tie.

Timing follows ``g4_part4_pilot_measure.py``: CUDA synchronization, discarded
warm-up, >=3 repetitions, medians plus min/max, allocated/reserved peaks,
weights-resident baseline, OOM-safe cleanup, real WikiText nested prefixes,
chunked 8K prefill, configuration hash, and reproducible JSON output.
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
from .g4_b4_cache_tests import _from_pretrained, load_ours
from .g4_part4_pilot_measure import load_text


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXTS = tuple(
    int(value) for value in os.environ.get(
        "FLASH_G4_BENCH_CTX", "512,2048,8192"
    ).split(",") if value.strip()
)
REPS = int(os.environ.get("FLASH_G4_BENCH_REPS", "3"))
WARMUP = int(os.environ.get("FLASH_G4_BENCH_WARMUP", "1"))
MAX_NEW = int(os.environ.get("FLASH_G4_BENCH_MAXNEW", "64"))
CHUNK = int(os.environ.get("FLASH_G4_BENCH_CHUNK", "512"))
K = spec_dec.K
BACKENDS = ("eager_bf16", "hybrid_bf16")
MODES = ("AR", "SpecDec")
OUTPUT_DIR = Path(os.environ.get(
    "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4_supporting",
))


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def set_target_backend(model, backend):
    model.set_kv_format("fp16")
    if backend == "eager_bf16":
        model.set_attention_backend("eager")
    elif backend == "hybrid_bf16":
        model.set_attention_backend("triton_bf16_g4_hybrid")
    else:
        raise ValueError(f"unknown backend: {backend}")


def chunked_prefill(model, ids):
    cache = None
    logits = None
    with torch.inference_mode():
        for start in range(0, ids.shape[1], CHUNK):
            piece = ids[:, start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece), cache,
                need_logits=(start + CHUNK >= ids.shape[1]),
            )
            spec_cache.check_consistent(cache)
    return logits, cache


def natural_ar(target, ids, eos_id, backend):
    set_target_backend(target, backend)
    logits, cache = chunked_prefill(target, ids)
    generated = []
    finite = bool(torch.isfinite(logits).all())
    with torch.inference_mode():
        for index in range(MAX_NEW):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos_id or index == MAX_NEW - 1:
                break
            logits, cache = spec_dec._step(target, token, cache)
            finite &= bool(torch.isfinite(logits).all())
            spec_cache.check_consistent(cache)
    return {
        "tokens": generated,
        "finite": finite,
        "final_cache_length": spec_cache.cache_length(cache),
    }


def natural_specdec(target, draft, ids, eos_id, backend):
    set_target_backend(target, backend)
    draft.set_kv_format("fp16")
    draft.set_attention_backend("eager")
    original_prefill = spec_dec._prefill

    def measured_prefill(model, prompt):
        return chunked_prefill(model, prompt)

    trace = []
    spec_dec._prefill = measured_prefill
    try:
        output = spec_dec.spec_decode_greedy(
            target, draft, ids, MAX_NEW, k=K, eos_id=eos_id,
            target_kv="base_bf16", draft_kv="base_bf16",
            log=lambda row: trace.append(row),
        )
    finally:
        spec_dec._prefill = original_prefill
    cache_ok = bool(trace) and all(
        row["tgt_len"] == row["drf_len"]
        and row["tgt_len"] == (
            row["pre"] + row["Kp"]
            if row["m"] == row["Kp"]
            else row["pre"] + row["m"] + 1
        )
        for row in trace
    )
    return {
        "tokens": output["gen"],
        "stats": output["stats"],
        "trace": trace,
        "cache_ok": cache_ok,
    }


def trace_signature(trace):
    keys = ("Kp", "m", "pre", "cands", "g", "emitted")
    return [tuple(json.dumps(row[key], sort_keys=True) for key in keys)
            for row in trace]


def validate_natural(target, draft, ids, eos_id):
    records = {}
    for backend in BACKENDS:
        ar = natural_ar(target, ids, eos_id, backend)
        spec = natural_specdec(target, draft, ids, eos_id, backend)
        records[backend] = {
            "ar": ar,
            "specdec": spec,
            "same_backend_specdec_equals_ar": (
                spec["tokens"] == ar["tokens"]
            ),
        }
        gc.collect()
        torch.cuda.empty_cache()
    eager = records["eager_bf16"]
    hybrid = records["hybrid_bf16"]
    return {
        "backends": records,
        "cross_backend": {
            "ar_tokens_equal": eager["ar"]["tokens"] == hybrid["ar"]["tokens"],
            "specdec_tokens_equal": (
                eager["specdec"]["tokens"]
                == hybrid["specdec"]["tokens"]
            ),
            "specdec_stats_equal": (
                eager["specdec"]["stats"]
                == hybrid["specdec"]["stats"]
            ),
            "specdec_trace_equal": (
                trace_signature(eager["specdec"]["trace"])
                == trace_signature(hybrid["specdec"]["trace"])
            ),
        },
        "reference": {
            "ar_tokens": eager["ar"]["tokens"],
            "specdec_tokens": eager["specdec"]["tokens"],
            "specdec_stats": eager["specdec"]["stats"],
            "specdec_trace": eager["specdec"]["trace"],
        },
    }


def replay_ar(target, ids, tokens, backend):
    set_target_backend(target, backend)
    prefill_start = sync_time()
    _, cache = chunked_prefill(target, ids)
    prefill_s = sync_time() - prefill_start
    decode_start = sync_time()
    with torch.inference_mode():
        for token in tokens[:-1]:
            _, cache = spec_dec._step(target, token, cache)
            spec_cache.check_consistent(cache)
    decode_s = sync_time() - decode_start
    final_length = spec_cache.cache_length(cache)
    del cache
    return prefill_s, decode_s, final_length


def replay_specdec(target, draft, ids, trace, backend):
    set_target_backend(target, backend)
    draft.set_kv_format("fp16")
    draft.set_attention_backend("eager")
    prefill_start = sync_time()
    _, target_cache = chunked_prefill(target, ids)
    _, draft_cache = chunked_prefill(draft, ids)
    prefill_s = sync_time() - prefill_start
    decode_start = sync_time()
    with torch.inference_mode():
        for row in trace:
            pre = spec_cache.cache_length(target_cache)
            if pre != row["pre"]:
                raise RuntimeError(
                    f"replay pre mismatch: got {pre}, expected {row['pre']}"
                )
            candidates = row["cands"]
            for token in candidates:
                _, draft_cache = spec_dec._step(
                    draft, token, draft_cache,
                )
            _, target_block = spec_dec._verify(
                target, target_cache, candidates,
            )
            accepted = row["m"]
            if accepted < row["Kp"]:
                replacement = row["emitted"][-1]
                target_cache = spec_cache.trim_cache(
                    target_block, pre + accepted,
                )
                _, target_cache = spec_dec._step(
                    target, replacement, target_cache,
                )
                draft_cache = spec_cache.trim_cache(
                    draft_cache, pre + accepted,
                )
                _, draft_cache = spec_dec._step(
                    draft, replacement, draft_cache,
                )
            else:
                target_cache = target_block
                draft_cache = spec_cache.trim_cache(
                    draft_cache, pre + row["Kp"],
                )
            spec_cache.check_consistent(target_cache)
            spec_cache.check_consistent(draft_cache)
    decode_s = sync_time() - decode_start
    final_target = spec_cache.cache_length(target_cache)
    final_draft = spec_cache.cache_length(draft_cache)
    del target_cache, draft_cache
    return prefill_s, decode_s, final_target, final_draft


def run_controlled(target, draft, ids, reference, backend, mode):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weights_resident = torch.cuda.memory_allocated() / 2**30
    if mode == "AR":
        tokens = reference["ar_tokens"]
        prefill_s, decode_s, target_length = replay_ar(
            target, ids, tokens, backend,
        )
        draft_length = None
        stats = {
            "rounds": 0, "proposed": 0, "accepted": 0,
            "target_block_calls": 0, "draft_calls": 0,
            "mismatch_rounds": 0, "all_accept_rounds": 0,
        }
    else:
        tokens = reference["specdec_tokens"]
        stats = reference["specdec_stats"]
        prefill_s, decode_s, target_length, draft_length = replay_specdec(
            target, draft, ids, reference["specdec_trace"], backend,
        )
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    total_s = prefill_s + decode_s
    row = {
        "backend": backend, "mode": mode, "ctx": int(ids.shape[1]),
        "n_gen": len(tokens), "prefill_s": prefill_s,
        "decode_s": decode_s, "total_s": total_s,
        "decode_tok_s": len(tokens) / decode_s,
        "e2e_tok_s": len(tokens) / total_s,
        "weights_resident_gib": weights_resident,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "memory_delta_allocated_gib": peak_allocated - weights_resident,
        "final_target_cache_length": target_length,
        "final_draft_cache_length": draft_length,
        "controlled_replay": True,
        "trace_source": "eager_bf16 natural reference",
    }
    row.update({key: stats.get(key, 0) for key in (
        "rounds", "proposed", "accepted", "target_block_calls",
        "draft_calls", "mismatch_rounds", "all_accept_rounds",
    )})
    row["accept_rate"] = (
        row["accepted"] / row["proposed"]
        if row["proposed"] else None
    )
    return row


def summarize(rows):
    numeric = (
        "prefill_s", "decode_s", "total_s", "decode_tok_s",
        "e2e_tok_s", "weights_resident_gib", "peak_allocated_gib",
        "peak_reserved_gib", "memory_delta_allocated_gib",
    )
    summary = {
        key: value for key, value in rows[0].items()
        if key not in numeric
    }
    for key in numeric:
        values = [row[key] for row in rows]
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    summary["repetitions"] = len(rows)
    return summary


def comparison(eager, hybrid):
    def speedup(key):
        return eager[f"{key}_median"] / hybrid[f"{key}_median"]

    return {
        "prefill_speedup": speedup("prefill_s"),
        "decode_speedup": speedup("decode_s"),
        "total_speedup": speedup("total_s"),
        "peak_allocated_mib_delta": (
            hybrid["peak_allocated_gib_median"]
            - eager["peak_allocated_gib_median"]
        ) * 1024,
        "peak_reserved_mib_delta": (
            hybrid["peak_reserved_gib_median"]
            - eager["peak_reserved_gib_median"]
        ) * 1024,
        "memory_delta_allocated_mib_delta": (
            hybrid["memory_delta_allocated_gib_median"]
            - eager["memory_delta_allocated_gib_median"]
        ) * 1024,
    }


def environment():
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton_version, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_gib": properties.total_memory / 2**30,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if REPS < 3:
        raise ValueError("FLASH_G4_BENCH_REPS must be at least 3")
    torch.manual_seed(0)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")
    from transformers import AutoTokenizer
    tokenizer = _from_pretrained(
        AutoTokenizer, TARGET, trust_remote_code=True,
    )
    all_ids = tokenizer(
        text[:400_000], return_tensors="pt",
    ).input_ids
    if all_ids.shape[1] < max(CONTEXTS):
        raise RuntimeError(
            f"corpus has {all_ids.shape[1]} tokens, need {max(CONTEXTS)}"
        )
    eos_id = tokenizer.eos_token_id
    print("[load] target 1.5B", flush=True)
    target, _ = load_ours(TARGET)
    print("[load] draft 0.5B", flush=True)
    draft, _ = load_ours(DRAFT)

    env = environment()
    config = {
        "contexts": CONTEXTS, "repetitions": REPS, "warmup": WARMUP,
        "max_new_tokens": MAX_NEW, "chunk": CHUNK, "k": K,
        "target": TARGET, "draft": DRAFT, "dtype": "torch.bfloat16",
        "kv_format": "fp16", "text_source": source,
        "hybrid_backend": "triton_bf16_g4_hybrid",
        "nested_real_text_prefixes": True,
        "formal_g4_matrix_modified": False,
        "measurement": "controlled replay of eager natural reference",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Flash G4 supporting ablation: "
        f"gpu={env['gpu']} ctx={CONTEXTS} reps={REPS} warmup={WARMUP} "
        f"maxnew={MAX_NEW} chunk={CHUNK} config={config_hash}",
        flush=True,
    )

    validations = {}
    summaries = {}
    comparisons = {}
    all_runs = []
    with torch.inference_mode():
        for context in CONTEXTS:
            ids = all_ids[:, :context].cuda()
            print(f"\n=== ctx={context} natural validation ===", flush=True)
            validation = validate_natural(target, draft, ids, eos_id)
            validations[str(context)] = validation
            cross = validation["cross_backend"]
            for backend in BACKENDS:
                same = validation["backends"][backend][
                    "same_backend_specdec_equals_ar"
                ]
                print(f"{backend:12s} SpecDec==AR: {same}", flush=True)
            print(
                "cross backend: "
                f"AR_tokens={cross['ar_tokens_equal']} "
                f"Spec_tokens={cross['specdec_tokens_equal']} "
                f"stats={cross['specdec_stats_equal']} "
                f"trace={cross['specdec_trace_equal']}", flush=True,
            )
            reference = validation["reference"]

            for mode in MODES:
                print(
                    f"=== ctx={context} mode={mode} controlled replay ===",
                    flush=True,
                )
                for _ in range(WARMUP):
                    for backend in BACKENDS:
                        try:
                            run_controlled(
                                target, draft, ids, reference, backend, mode,
                            )
                        except torch.cuda.OutOfMemoryError:
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(f"warmup OOM backend={backend}", flush=True)

                runs_by_backend = {backend: [] for backend in BACKENDS}
                oom = {backend: False for backend in BACKENDS}
                for repetition in range(REPS):
                    order = (
                        BACKENDS if repetition % 2 == 0
                        else tuple(reversed(BACKENDS))
                    )
                    for backend in order:
                        try:
                            row = run_controlled(
                                target, draft, ids, reference, backend, mode,
                            )
                            row["repetition"] = repetition
                            row["order"] = list(order)
                            runs_by_backend[backend].append(row)
                            all_runs.append(row)
                            print(
                                f"rep={repetition} {backend:12s} "
                                f"prefill={row['prefill_s']:.3f}s "
                                f"decode={row['decode_s']:.3f}s "
                                f"peak={row['peak_allocated_gib']:.3f}GiB",
                                flush=True,
                            )
                        except torch.cuda.OutOfMemoryError:
                            oom[backend] = True
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(
                                f"rep={repetition} {backend} OOM",
                                flush=True,
                            )
                key = f"{context}:{mode}"
                summaries[key] = {}
                for backend in BACKENDS:
                    rows = runs_by_backend[backend]
                    if rows:
                        summaries[key][backend] = summarize(rows)
                        summaries[key][backend]["oom"] = oom[backend]
                    else:
                        summaries[key][backend] = {
                            "oom": True, "repetitions": 0,
                        }
                if all(runs_by_backend.values()):
                    comparisons[key] = comparison(
                        summaries[key]["eager_bf16"],
                        summaries[key]["hybrid_bf16"],
                    )
                    row = comparisons[key]
                    print(
                        f"delta: prefill={row['prefill_speedup']:.3f}x "
                        f"decode={row['decode_speedup']:.3f}x "
                        f"total={row['total_speedup']:.3f}x "
                        f"peak_alloc={row['peak_allocated_mib_delta']:+.1f}MiB",
                        flush=True,
                    )
            del ids
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "status": "flash_g4_supporting_ablation_controlled_replay",
        "config_hash": config_hash, "config": config, "environment": env,
        "runtime_policy": {
            "prefill": "custom_triton_two_pass",
            "q_len_1": "custom_triton_two_pass",
            "cached_q_len_2_to_4": "repository_eager_expression",
            "official_flash_used": False,
            "model_selector": "triton_bf16_g4_hybrid",
            "runtime_monkeypatch": False,
            "default_backend": "eager",
        },
        "natural_validation": validations,
        "summaries": summaries, "comparisons": comparisons,
        "runs": all_runs,
        "interpretation": (
            "Performance deltas use controlled replay only. Natural outputs "
            "and traces are correctness/sensitivity evidence and are not "
            "used as comparable timing when workloads differ."
        ),
    }
    output = OUTPUT_DIR / f"flash_g4_supporting_{config_hash}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== CONTROLLED-REPLAY SUMMARY ===")
    for context in CONTEXTS:
        for mode in MODES:
            key = f"{context}:{mode}"
            if key not in comparisons:
                print(f"ctx={context} mode={mode}: OOM/CHECK")
                continue
            row = comparisons[key]
            print(
                f"ctx={context:<4} mode={mode:<7} "
                f"prefill={row['prefill_speedup']:.3f}x "
                f"decode={row['decode_speedup']:.3f}x "
                f"total={row['total_speedup']:.3f}x "
                f"peak_alloc={row['peak_allocated_mib_delta']:+.1f}MiB "
                f"peak_resv={row['peak_reserved_mib_delta']:+.1f}MiB"
            )
    print(f"JSON={output}")
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
