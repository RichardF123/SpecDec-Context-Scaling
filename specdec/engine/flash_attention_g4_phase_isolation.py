"""Isolate where Flash numerical drift enters the latest G4 workflow.

This diagnostic is intentionally run after the all-Flash strict compatibility
gate has failed.  It compares four custom-kernel routes against eager:

    EE: eager prefill, eager online attention
    FE: Flash prefill, eager AR decode / SpecDec verification
    EF: eager prefill, Flash AR decode / SpecDec verification
    FF: Flash prefill, Flash online attention

It also tries PyTorch's *forced* FLASH_ATTENTION SDPA backend (SS) as a
reference.  The SDPA route is diagnostic only: it does not add another
production selector and it never silently falls back to the math backend.

The test uses real WikiText prefixes, teacher-forced AR logits, free-running
greedy tokens, latest ``spec_decode_greedy``, per-round traces, and cache
invariants.  It is not a performance benchmark.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from . import flash_attention_two_pass_triton as flash_impl
from . import spec_cache, spec_dec
from .g4_b4_cache_tests import load_ours
from .g4_part4_pilot_measure import load_text


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXTS = tuple(
    int(value)
    for value in os.environ.get("FLASH_PHASE_CTX", "512,2048").split(",")
    if value.strip()
)
MAX_NEW = int(os.environ.get("FLASH_PHASE_MAXNEW", "32"))
CHUNK = int(os.environ.get("FLASH_PHASE_CHUNK", "512"))
K = spec_dec.K

# F = the retained from-scratch two-pass Triton implementation.
PHASES = (
    {"name": "EE", "prefill": "eager", "online": "eager", "impl": "custom"},
    {"name": "FE", "prefill": "flash", "online": "eager", "impl": "custom"},
    {"name": "EF", "prefill": "eager", "online": "flash", "impl": "custom"},
    {"name": "FF", "prefill": "flash", "online": "flash", "impl": "custom"},
    {"name": "SS", "prefill": "flash", "online": "flash", "impl": "sdpa_flash"},
)


def synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def select_backend(model, backend: str) -> None:
    if backend == "eager":
        model.set_attention_backend("eager")
    elif backend == "flash":
        model.set_attention_backend("triton_bf16_two_pass")
    else:
        raise ValueError(f"unknown phase backend: {backend}")


def _sdpa_flash(query, key, value, *, causal=True, config=None):
    """Call only PyTorch's CUDA Flash SDPA kernel with GQA preserved."""
    del config
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q_len = int(query.shape[-2])
    kv_len = int(key.shape[-2])
    attention_mask = None
    is_causal = False
    if causal:
        if q_len == kv_len:
            is_causal = True
        else:
            # Cached multi-token verification needs lower-right causal
            # alignment: query row r has absolute position kv_len-q_len+r.
            try:
                from torch.nn.attention.bias import causal_lower_right
                attention_mask = causal_lower_right(q_len, kv_len)
            except ImportError:
                # Older builds can still express the same alignment as a
                # boolean mask.  FLASH_ATTENTION remains forced below; if the
                # installed kernel cannot consume it, the diagnostic records
                # SDPA as unavailable instead of falling back silently.
                q_position = (
                    torch.arange(q_len, device=query.device) + kv_len - q_len
                )
                k_position = torch.arange(kv_len, device=query.device)
                attention_mask = k_position[None, :] <= q_position[:, None]

    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True,
        )


@contextmanager
def implementation_scope(implementation: str):
    original = flash_impl.flash_attention_bf16_two_pass
    if implementation == "sdpa_flash":
        flash_impl.flash_attention_bf16_two_pass = _sdpa_flash
    elif implementation != "custom":
        raise ValueError(f"unknown implementation: {implementation}")
    try:
        yield
    finally:
        flash_impl.flash_attention_bf16_two_pass = original


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


def compare_logits(reference, candidate):
    records = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        difference = right - left
        denominator = torch.linalg.vector_norm(left).clamp_min(1e-12)
        top2 = torch.topk(left.flatten(), k=2).values
        records.append({
            "step": index,
            "eager_argmax": int(left.argmax()),
            "candidate_argmax": int(right.argmax()),
            "argmax_same": bool(left.argmax() == right.argmax()),
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference) / denominator
            ),
            "eager_top1_margin": float(top2[0] - top2[1]),
        })
    first_difference = next(
        (row["step"] for row in records if not row["argmax_same"]), None,
    )
    return {
        "same_length": len(reference) == len(candidate),
        "all_argmax_same": first_difference is None,
        "first_argmax_difference": first_difference,
        "worst_max_abs": max((row["max_abs"] for row in records), default=0.0),
        "worst_relative_l2": max(
            (row["relative_l2"] for row in records), default=0.0,
        ),
        "per_step": records,
    }


def run_ar(model, ids, eos_id, phase, forced_tokens=None):
    model.set_kv_format("fp16")
    select_backend(model, phase["prefill"])
    started = synchronize()
    logits, cache = chunked_prefill(model, ids)
    prefill_s = synchronize() - started
    select_backend(model, phase["online"])

    predictions = []
    decision_logits = []
    finite = bool(torch.isfinite(logits).all())
    started = synchronize()
    with torch.inference_mode():
        for index in range(MAX_NEW):
            row = logits[:, -1].float().cpu()
            decision_logits.append(row)
            predicted = int(row.argmax())
            predictions.append(predicted)
            if index == MAX_NEW - 1 or predicted == eos_id:
                break
            token = (
                predicted if forced_tokens is None
                else int(forced_tokens[index])
            )
            token_tensor = torch.tensor([[token]], device=ids.device)
            logits, cache = model._run(
                model.embed_tokens(token_tensor), cache, need_logits=True,
            )
            spec_cache.check_consistent(cache)
            finite &= bool(torch.isfinite(logits).all())
    decode_s = synchronize() - started
    return {
        "predictions": predictions,
        "decision_logits": decision_logits,
        "finite": finite,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "final_cache_length": spec_cache.cache_length(cache),
    }


def run_specdec(target, draft, ids, eos_id, phase):
    target.set_kv_format("fp16")
    draft.set_kv_format("fp16")
    draft.set_attention_backend("eager")
    target_id = id(target)
    target_q_lengths = []
    finite = True
    cache_consistent = True
    trace = []

    original_prefill = spec_dec._prefill
    original_verify = spec_dec._verify
    original_step = spec_dec._step

    def phase_prefill(model, prompt):
        nonlocal finite, cache_consistent
        if id(model) == target_id:
            select_backend(model, phase["prefill"])
        else:
            model.set_attention_backend("eager")
        logits, cache = chunked_prefill(model, prompt)
        finite &= bool(torch.isfinite(logits).all())
        try:
            spec_cache.check_consistent(cache)
        except RuntimeError:
            cache_consistent = False
            raise
        if id(model) == target_id:
            select_backend(model, phase["online"])
        return logits, cache

    def phase_verify(model, cache, candidates):
        nonlocal finite, cache_consistent
        if id(model) == target_id:
            select_backend(model, phase["online"])
            target_q_lengths.append(len(candidates))
        rows, next_cache = original_verify(model, cache, candidates)
        finite &= bool(torch.isfinite(rows).all())
        try:
            spec_cache.check_consistent(next_cache)
        except RuntimeError:
            cache_consistent = False
            raise
        return rows, next_cache

    def phase_step(model, token, cache):
        nonlocal finite, cache_consistent
        if id(model) == target_id:
            select_backend(model, phase["online"])
        else:
            model.set_attention_backend("eager")
        logits, next_cache = original_step(model, token, cache)
        finite &= bool(torch.isfinite(logits).all())
        try:
            spec_cache.check_consistent(next_cache)
        except RuntimeError:
            cache_consistent = False
            raise
        return logits, next_cache

    spec_dec._prefill = phase_prefill
    spec_dec._verify = phase_verify
    spec_dec._step = phase_step
    try:
        started = synchronize()
        output = spec_dec.spec_decode_greedy(
            target,
            draft,
            ids,
            MAX_NEW,
            k=K,
            eos_id=eos_id,
            target_kv="base_bf16",
            draft_kv="base_bf16",
            log=lambda row: trace.append(row),
        )
        elapsed_s = synchronize() - started
    finally:
        spec_dec._prefill = original_prefill
        spec_dec._verify = original_verify
        spec_dec._step = original_step

    round_cache_ok = all(
        row["tgt_len"] == row["pre"] + (
            row["Kp"] if row["m"] == row["Kp"] else row["m"] + 1
        )
        and row["drf_len"] == row["tgt_len"]
        for row in trace
    )
    return {
        "tokens": output["gen"],
        "stats": output["stats"],
        "trace": trace,
        "elapsed_s": elapsed_s,
        "finite": finite,
        "cache_consistent": cache_consistent and round_cache_ok,
        "target_q_lengths": target_q_lengths,
    }


def compare_specdec(reference, candidate):
    stat_keys = (
        "rounds", "proposed", "accepted", "mismatch_rounds",
        "all_accept_rounds", "rollbacks", "target_block_calls", "draft_calls",
    )
    trace_keys = (
        "Kp", "m", "pre", "tgt_len", "drf_len", "cands", "g", "emitted",
    )
    stats_equal = all(
        reference["stats"].get(key) == candidate["stats"].get(key)
        for key in stat_keys
    )
    trace_equal = (
        len(reference["trace"]) == len(candidate["trace"])
        and all(
            all(left.get(key) == right.get(key) for key in trace_keys)
            for left, right in zip(reference["trace"], candidate["trace"])
        )
    )
    return {
        "tokens_identical": reference["tokens"] == candidate["tokens"],
        "stats_equal": stats_equal,
        "trace_equal": trace_equal,
        "finite": candidate["finite"],
        "cache_consistent": candidate["cache_consistent"],
        "passed": bool(
            reference["tokens"] == candidate["tokens"]
            and stats_equal
            and trace_equal
            and candidate["finite"]
            and candidate["cache_consistent"]
        ),
    }


def light_ar(record):
    return {
        key: value for key, value in record.items()
        if key != "decision_logits"
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")

    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained

    tokenizer = _from_pretrained(
        AutoTokenizer, TARGET, trust_remote_code=True,
    )
    all_ids = tokenizer(text[:400_000], return_tensors="pt").input_ids
    if all_ids.shape[1] < max(CONTEXTS):
        raise RuntimeError(
            f"corpus has {all_ids.shape[1]} tokens, need {max(CONTEXTS)}"
        )

    print(
        "Flash G4 phase isolation: "
        f"gpu={torch.cuda.get_device_name(0)} ctx={CONTEXTS} "
        f"maxnew={MAX_NEW} chunk={CHUNK}",
        flush=True,
    )
    print("[load] target 1.5B", flush=True)
    target, _ = load_ours(TARGET)
    print("[load] draft 0.5B", flush=True)
    draft, _ = load_ours(DRAFT)
    eos_id = tokenizer.eos_token_id

    results = {}
    phase_passes = {phase["name"]: [] for phase in PHASES}
    for context in CONTEXTS:
        ids = all_ids[:, :context].cuda()
        print(f"\n=== context={context} ===", flush=True)
        context_rows = {}
        reference_ar = None
        reference_spec = None
        reference_tokens = None

        for phase in PHASES:
            name = phase["name"]
            try:
                with implementation_scope(phase["impl"]):
                    free_ar = run_ar(target, ids, eos_id, phase)
                    if name == "EE":
                        reference_tokens = free_ar["predictions"]
                        forced_ar = free_ar
                    else:
                        forced_ar = run_ar(
                            target, ids, eos_id, phase,
                            forced_tokens=reference_tokens,
                        )
                    spec_record = run_specdec(
                        target, draft, ids, eos_id, phase,
                    )
            except (RuntimeError, NotImplementedError) as error:
                message = f"{type(error).__name__}: {error}"
                context_rows[name] = {
                    "phase": phase,
                    "available": False,
                    "error": message,
                    "passed": False,
                }
                phase_passes[name].append(False)
                print(f"{name}: unavailable ({message})", flush=True)
                torch.cuda.empty_cache()
                continue

            if name == "EE":
                reference_ar = free_ar
                reference_spec = spec_record
                ar_comparison = compare_logits(
                    reference_ar["decision_logits"],
                    reference_ar["decision_logits"],
                )
                spec_comparison = compare_specdec(
                    reference_spec, reference_spec,
                )
            else:
                ar_comparison = compare_logits(
                    reference_ar["decision_logits"],
                    forced_ar["decision_logits"],
                )
                spec_comparison = compare_specdec(
                    reference_spec, spec_record,
                )

            ar_pass = bool(
                free_ar["predictions"] == reference_tokens
                and free_ar["finite"]
                and forced_ar["finite"]
                and ar_comparison["same_length"]
                and ar_comparison["all_argmax_same"]
            )
            passed = ar_pass and spec_comparison["passed"]
            phase_passes[name].append(passed)
            context_rows[name] = {
                "phase": phase,
                "available": True,
                "free_ar": light_ar(free_ar),
                "teacher_forced_ar": light_ar(forced_ar),
                "ar_comparison": ar_comparison,
                "ar_passed": ar_pass,
                "specdec": spec_record,
                "specdec_comparison": spec_comparison,
                "passed": passed,
            }
            first = ar_comparison["first_argmax_difference"]
            print(
                f"{name}: AR={ar_pass} SpecDec={spec_comparison['passed']} "
                f"tokens={free_ar['predictions'] == reference_tokens} "
                f"first_argmax_diff={first} "
                f"worst_rel={ar_comparison['worst_relative_l2']:.6f} "
                f"spec_trace={spec_comparison['trace_equal']}",
                flush=True,
            )

        results[str(context)] = context_rows
        del ids
        gc.collect()
        torch.cuda.empty_cache()

    strict_pass = {
        name: len(values) == len(CONTEXTS) and all(values)
        for name, values in phase_passes.items()
    }
    if strict_pass.get("FE"):
        decision = "PROCEED_WITH_FLASH_PREFILL_EAGER_ONLINE_HYBRID"
    elif strict_pass.get("EF"):
        decision = "PROCEED_WITH_EAGER_PREFILL_FLASH_ONLINE_HYBRID"
    elif strict_pass.get("SS"):
        decision = "CUSTOM_CHECK_OFFICIAL_SDPA_FLASH_PASS"
    else:
        decision = "NO_STRICT_PHASE_CANDIDATE"

    payload = {
        "status": "flash_g4_phase_isolation",
        "main_fresh_start_sha": "d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "target": TARGET,
        "draft": DRAFT,
        "contexts": CONTEXTS,
        "max_new_tokens": MAX_NEW,
        "chunk": CHUNK,
        "text_source": source,
        "phase_strict_pass": strict_pass,
        "context_results": results,
        "decision": decision,
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_phase_isolation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\ndecision={decision}\nJSON={output}")

    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    # The diagnostic itself completed even when no candidate passes.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
