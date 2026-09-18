"""Attribute the first hybrid SpecDec-vs-AR BF16 decision split.

This is a diagnostic only.  It keeps latest-main ``spec_dec.py`` and
``spec_cache.py`` unchanged, reproduces the explicit hybrid policy, captures
the actual BF16 decision rows used by target-only AR and SpecDec, and then
evaluates the same common prefix with the repository's CPU/FP32 full, cached,
and AR-history references.

The verdict follows the repository's B6 evidence discipline: exact BF16
behavior is always reported, while a split is classified through measured
decision signal versus measured path noise.  It does not relax or replace any
mainline correctness gate.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

from . import flash_attention_two_pass_triton as flash_impl
from . import spec_cache, spec_dec
from .flash_attention_g4_core_probe import eager_reference
from .g4_b4_cache_tests import _from_pretrained, load_ours
from .g4_b6_fp32_correctness import load_cpu
from .g4_b6_tie_diag import at_prefix_cached, at_prefix_full, at_prefix_hist
from .g4_part4_pilot_measure import load_text


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT = int(os.environ.get("FLASH_HYBRID_ATTR_CTX", "2048"))
MAX_NEW = int(os.environ.get("FLASH_HYBRID_ATTR_MAXNEW", "32"))
CHUNK = int(os.environ.get("FLASH_HYBRID_ATTR_CHUNK", "512"))
K = spec_dec.K


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


def run_ar(model, ids, eos_id):
    model.set_kv_format("fp16")
    model.set_attention_backend("triton_bf16_two_pass")
    logits, cache = chunked_prefill(model, ids)
    tokens = []
    decisions = []
    with torch.inference_mode():
        for index in range(MAX_NEW):
            decisions.append(logits[0, -1].float().cpu())
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            if token == eos_id or index == MAX_NEW - 1:
                break
            piece = torch.tensor([[token]], device=ids.device)
            logits, cache = model._run(
                model.embed_tokens(piece), cache, need_logits=True,
            )
            spec_cache.check_consistent(cache)
    return {"tokens": tokens, "decision_logits": decisions}


def run_specdec_with_rows(target, draft, ids, eos_id):
    """Mirror latest-main G4 while retaining each emitted decision row."""
    target.set_kv_format("fp16")
    draft.set_kv_format("fp16")
    target.set_attention_backend("triton_bf16_two_pass")
    draft.set_attention_backend("eager")
    tgt_lg, tgt_cache = chunked_prefill(target, ids)
    drf_lg, drf_cache = chunked_prefill(draft, ids)
    saved_next = tgt_lg[0, -1]
    drf_lg = drf_lg[:, -1:]
    generated = []
    decision_rows = []
    trace = []

    while len(generated) < MAX_NEW:
        k_eff = min(K, MAX_NEW - len(generated))
        pre = spec_cache.cache_length(tgt_cache)
        cands, drf_cache, prop_lg = spec_dec._propose(
            draft, drf_cache, drf_lg, k_eff, eos_id,
        )
        kp = len(cands)
        if kp == 0:
            break
        rows, tgt_blk = spec_dec._verify(target, tgt_cache, cands)
        judges = [saved_next] + [rows[j] for j in range(kp - 1)]
        choices = [int(row.argmax()) for row in judges]
        accepted = 0
        while accepted < kp and choices[accepted] == cands[accepted]:
            accepted += 1

        if accepted < kp:
            replacement = choices[accepted]
            emitted = cands[:accepted] + [replacement]
            used_rows = judges[:accepted + 1]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + accepted)
            next_logits, tgt_cache = spec_dec._step(
                target, replacement, tgt_cache,
            )
            saved_next = next_logits[0, -1]
            drf_cache = spec_cache.trim_cache(drf_cache, pre + accepted)
            drf_lg, drf_cache = spec_dec._step(
                draft, replacement, drf_cache,
            )
        else:
            emitted = list(cands)
            used_rows = judges
            tgt_cache = tgt_blk
            drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
            saved_next = rows[kp - 1]
            drf_lg = prop_lg

        decision_rows.extend(row.float().cpu() for row in used_rows)
        generated.extend(emitted)
        trace.append({
            "round": len(trace), "pre": pre, "Kp": kp, "m": accepted,
            "cands": cands, "g": choices, "emitted": emitted,
            "tgt_len": spec_cache.cache_length(tgt_cache),
            "drf_len": spec_cache.cache_length(drf_cache),
        })
        spec_cache.check_consistent(tgt_cache)
        spec_cache.check_consistent(drf_cache)
        if emitted and emitted[-1] == eos_id:
            break

    return generated, decision_rows, trace


def row_summary(row, a, b):
    values, indices = torch.topk(row, k=2)
    return {
        "argmax": int(row.argmax()),
        "top1": int(indices[0]),
        "top2": int(indices[1]),
        "top1_margin": float(values[0] - values[1]),
        "gap_spec_minus_ar": float(row[a] - row[b]),
        "finite": bool(torch.isfinite(row).all()),
    }


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def trace_location(trace, generated_index):
    start = 0
    for row in trace:
        stop = start + len(row["emitted"])
        if start <= generated_index < stop:
            offset = generated_index - start
            return {
                "round": row["round"],
                "offset": offset,
                "kind": "accepted" if offset < row["m"] else "replacement",
                "round_record": row,
            }
        start = stop
    return None


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError("real WikiText input is required")

    from transformers import AutoTokenizer
    tokenizer = _from_pretrained(
        AutoTokenizer, TARGET, trust_remote_code=True,
    )
    all_ids = tokenizer(text[:400_000], return_tensors="pt").input_ids
    ids = all_ids[:, :CONTEXT].cuda()
    if ids.shape[1] != CONTEXT:
        raise RuntimeError(f"corpus only provided {ids.shape[1]} tokens")
    eos_id = tokenizer.eos_token_id

    print(
        "hybrid first-split attribution: "
        f"gpu={torch.cuda.get_device_name(0)} ctx={CONTEXT} "
        f"maxnew={MAX_NEW} chunk={CHUNK}", flush=True,
    )
    target, _ = load_ours(TARGET)
    draft, _ = load_ours(DRAFT)
    original = flash_impl.flash_attention_bf16_two_pass

    def explicit_hybrid(query, key, value, *, causal=True, config=None):
        if 2 <= int(query.shape[2]) <= K:
            return eager_reference(query, key, value, causal=causal)
        return original(query, key, value, causal=causal, config=config)

    flash_impl.flash_attention_bf16_two_pass = explicit_hybrid
    try:
        ar = run_ar(target, ids, eos_id)
        spec_tokens, spec_rows, trace = run_specdec_with_rows(
            target, draft, ids, eos_id,
        )
    finally:
        flash_impl.flash_attention_bf16_two_pass = original

    ar_tokens = ar["tokens"]
    index = first_difference(spec_tokens, ar_tokens)
    base = {
        "status": "flash_g4_hybrid_first_split_attribution",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "context": CONTEXT,
        "max_new_tokens": MAX_NEW, "chunk": CHUNK, "text_source": source,
        "ar_tokens": ar_tokens, "specdec_tokens": spec_tokens,
        "first_difference": index, "trace": trace,
        "production_dispatch_unchanged": True,
    }
    if index is None:
        base["decision"] = "NO_DIVERGENCE"
        return finish(base, 0)
    if index >= len(ar["decision_logits"]) or index >= len(spec_rows):
        base["decision"] = "CHECK_LENGTH_ONLY_DIVERGENCE"
        return finish(base, 1)

    spec_token = spec_tokens[index]
    ar_token = ar_tokens[index]
    ar_row = ar["decision_logits"][index]
    spec_row = spec_rows[index]
    difference = spec_row - ar_row
    denominator = torch.linalg.vector_norm(ar_row).clamp_min(1e-12)
    bf16 = {
        "hybrid_ar_history": row_summary(
            ar_row, spec_token, ar_token,
        ),
        "hybrid_specdec_actual": row_summary(
            spec_row, spec_token, ar_token,
        ),
        "actual_row_comparison": {
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference) / denominator
            ),
            "argmax_same": bool(spec_row.argmax() == ar_row.argmax()),
        },
    }
    common_prefix = torch.cat([
        ids.cpu(), torch.tensor([ar_tokens[:index]], dtype=ids.dtype),
    ], dim=1)
    split = CONTEXT

    del target, draft, ids, all_ids, ar, spec_rows
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"first difference={index} spec={spec_token} ar={ar_token}\n"
        f"BF16 AR gap={bf16['hybrid_ar_history']['gap_spec_minus_ar']:+.6f} "
        f"SpecDec gap={bf16['hybrid_specdec_actual']['gap_spec_minus_ar']:+.6f}",
        flush=True,
    )
    print("[stage] loading target FP32 on CPU", flush=True)
    target32, _ = load_cpu(TARGET)
    fp32_rows = {
        "full": at_prefix_full(target32, common_prefix),
        "cached": at_prefix_cached(target32, common_prefix),
        "ar_history": at_prefix_hist(target32, common_prefix, split),
    }
    fp32 = {
        name: row_summary(row, spec_token, ar_token)
        for name, row in fp32_rows.items()
    }
    signal = fp32["full"]["gap_spec_minus_ar"]
    observed_gaps = [
        bf16["hybrid_ar_history"]["gap_spec_minus_ar"],
        bf16["hybrid_specdec_actual"]["gap_spec_minus_ar"],
    ]
    noise = max(abs(gap - signal) for gap in observed_gaps)
    fp32_argmax = {row["argmax"] for row in fp32.values()}
    fp32_consistent = len(fp32_argmax) == 1
    undecidable = abs(signal) <= noise
    if not fp32_consistent:
        decision = "CHECK_FP32_PATH_INSTABILITY"
    elif undecidable:
        decision = "BF16_PATH_UNDECIDABLE"
    elif next(iter(fp32_argmax)) == spec_token:
        decision = "HYBRID_AR_PATH_SUSPECT_SPECDEC_MATCHES_FP32"
    elif next(iter(fp32_argmax)) == ar_token:
        decision = "HYBRID_IMPLEMENTATION_SUSPECT"
    else:
        decision = "CHECK_FP32_SELECTS_THIRD_TOKEN"

    base.update({
        "spec_token": spec_token, "ar_token": ar_token,
        "trace_location": trace_location(trace, index),
        "bf16": bf16, "fp32": fp32,
        "signal_abs": abs(signal), "observed_noise": noise,
        "signal_le_noise": undecidable,
        "fp32_paths_same_argmax": fp32_consistent,
        "decision": decision,
    })
    print(
        f"FP32 signal={signal:+.6f} observed_noise={noise:.6f} "
        f"signal_le_noise={undecidable} fp32_consistent={fp32_consistent}\n"
        f"decision={decision}", flush=True,
    )
    del target32, fp32_rows, common_prefix
    gc.collect()
    return finish(base, 0 if decision in {
        "NO_DIVERGENCE", "BF16_PATH_UNDECIDABLE",
    } else 1)


def finish(result, code):
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_hybrid_attribution.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"JSON={output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
