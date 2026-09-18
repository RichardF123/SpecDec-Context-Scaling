"""Decode bottleneck attribution for the 8K Flash/G4 supporting path.

Runtime-only instrumentation; production engine and cache contracts are not
modified.  CUDA events around existing modules attribute target/draft layer,
projection, normalization, MLP, and attention-block time.  The diagnostic also
computes exact bytes written by the current full-cache ``torch.cat`` append
contract for every replayed model call.

The workload is controlled replay of an eager natural reference, matching the
supporting ablation.  This keeps token inputs, q lengths, verification rounds,
rollbacks, and corrective calls identical across eager and hybrid backends.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import statistics

import torch

from . import flash_attention_g4_supporting_bench as supporting
from . import model_self as model_impl
from . import spec_cache, spec_dec
from .flash_attention_g4_supporting_bench import (
    BACKENDS,
    chunked_prefill,
    hybrid_scope,
    set_target_backend,
    validate_natural,
)
from .g4_b4_cache_tests import _from_pretrained, load_ours
from .g4_part4_pilot_measure import load_text


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT = int(os.environ.get("FLASH_G4_PROFILE_CTX", "8192"))
MAX_NEW = int(os.environ.get("FLASH_G4_PROFILE_MAXNEW", "64"))
CHUNK = int(os.environ.get("FLASH_G4_PROFILE_CHUNK", "512"))
REPS = int(os.environ.get("FLASH_G4_PROFILE_REPS", "3"))
OUTPUT_DIR = Path(os.environ.get(
    "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4_profile",
))


class CudaRecorder:
    def __init__(self):
        self.enabled = False
        self.events = defaultdict(list)

    def pair(self, category):
        if not self.enabled:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[category].append((start, end))
        return end

    def totals(self):
        torch.cuda.synchronize()
        return {
            category: sum(start.elapsed_time(end) for start, end in pairs)
            for category, pairs in self.events.items()
        }


def add_hook(recorder, module, category, handles):
    stack = []

    def before(_module, _inputs):
        stack.append(recorder.pair(category))

    def after(_module, _inputs, _output):
        end = stack.pop()
        if end is not None:
            end.record()

    handles.append(module.register_forward_pre_hook(before))
    handles.append(module.register_forward_hook(after))


def instrument_model(recorder, model, prefix, handles):
    for layer in model.layers:
        add_hook(recorder, layer, f"{prefix}.decoder_layers", handles)
        add_hook(recorder, layer.self_attn, f"{prefix}.self_attention", handles)
        add_hook(recorder, layer.self_attn.q_proj, f"{prefix}.q_projection", handles)
        add_hook(recorder, layer.self_attn.k_proj, f"{prefix}.k_projection", handles)
        add_hook(recorder, layer.self_attn.v_proj, f"{prefix}.v_projection", handles)
        add_hook(recorder, layer.self_attn.o_proj, f"{prefix}.o_projection", handles)
        add_hook(recorder, layer.mlp, f"{prefix}.mlp", handles)
        add_hook(
            recorder, layer.input_layernorm,
            f"{prefix}.layernorm", handles,
        )
        add_hook(
            recorder, layer.post_attention_layernorm,
            f"{prefix}.layernorm", handles,
        )
    add_hook(recorder, model.norm, f"{prefix}.final_norm", handles)


@contextmanager
def model_run_events(recorder, target, draft):
    handles = []
    instrument_model(recorder, target, "target", handles)
    instrument_model(recorder, draft, "draft", handles)
    target_run = target._run
    draft_run = draft._run
    apply_rope = model_impl.apply_rope
    build_rope = model_impl.Build_Rope_cache
    active = {"model": None}

    def wrap(original, category, model_name):
        def measured(*args, **kwargs):
            end = recorder.pair(category)
            previous = active["model"]
            active["model"] = model_name
            try:
                result = original(*args, **kwargs)
                if end is not None:
                    end.record()
                return result
            finally:
                active["model"] = previous
        return measured

    def measured_apply_rope(*args, **kwargs):
        category = f"{active['model']}.rope_apply"
        end = recorder.pair(category)
        result = apply_rope(*args, **kwargs)
        if end is not None:
            end.record()
        return result

    def measured_build_rope(*args, **kwargs):
        category = f"{active['model']}.rope_build"
        end = recorder.pair(category)
        result = build_rope(*args, **kwargs)
        if end is not None:
            end.record()
        return result

    target._run = wrap(target_run, "target.run_total", "target")
    draft._run = wrap(draft_run, "draft.run_total", "draft")
    model_impl.apply_rope = measured_apply_rope
    model_impl.Build_Rope_cache = measured_build_rope
    try:
        yield
    finally:
        target._run = target_run
        draft._run = draft_run
        model_impl.apply_rope = apply_rope
        model_impl.Build_Rope_cache = build_rope
        for handle in handles:
            handle.remove()


def model_shape(model):
    attention = model.layers[0].self_attn
    return {
        "layers": len(model.layers),
        "kv_heads": attention.kv_heads,
        "head_dim": attention.head_dim,
        "element_bytes": 2,
    }


def append_bytes(shape, old_length, query_length):
    """Bytes written by K/V cat outputs across all layers for one call."""
    return (
        shape["layers"] * 2 * shape["kv_heads"] * shape["head_dim"]
        * (old_length + query_length) * shape["element_bytes"]
    )


def profile_ar(recorder, target, ids, tokens, backend, target_shape):
    set_target_backend(target, backend)
    _, cache = chunked_prefill(target, ids)
    cache_bytes = 0
    recorder.enabled = True
    total_end = recorder.pair("workload.decode_total")
    with torch.inference_mode():
        for token in tokens[:-1]:
            old_length = spec_cache.cache_length(cache)
            cache_bytes += append_bytes(target_shape, old_length, 1)
            _, cache = spec_dec._step(target, token, cache)
    total_end.record()
    recorder.enabled = False
    final_length = spec_cache.cache_length(cache)
    del cache
    return {
        "target_cache_cat_output_bytes": cache_bytes,
        "draft_cache_cat_output_bytes": 0,
        "final_target_cache_length": final_length,
        "final_draft_cache_length": None,
    }


def profile_specdec(
    recorder, target, draft, ids, trace, backend,
    target_shape, draft_shape,
):
    set_target_backend(target, backend)
    draft.set_kv_format("fp16")
    draft.set_attention_backend("eager")
    _, target_cache = chunked_prefill(target, ids)
    _, draft_cache = chunked_prefill(draft, ids)
    target_bytes = 0
    draft_bytes = 0
    recorder.enabled = True
    total_end = recorder.pair("workload.decode_total")
    with torch.inference_mode():
        for row in trace:
            pre = spec_cache.cache_length(target_cache)
            for token in row["cands"]:
                old_length = spec_cache.cache_length(draft_cache)
                draft_bytes += append_bytes(draft_shape, old_length, 1)
                _, draft_cache = spec_dec._step(
                    draft, token, draft_cache,
                )
            target_bytes += append_bytes(
                target_shape, pre, row["Kp"],
            )
            _, target_block = spec_dec._verify(
                target, target_cache, row["cands"],
            )
            if row["m"] < row["Kp"]:
                replacement = row["emitted"][-1]
                target_cache = spec_cache.trim_cache(
                    target_block, pre + row["m"],
                )
                target_bytes += append_bytes(
                    target_shape, pre + row["m"], 1,
                )
                _, target_cache = spec_dec._step(
                    target, replacement, target_cache,
                )
                draft_cache = spec_cache.trim_cache(
                    draft_cache, pre + row["m"],
                )
                draft_bytes += append_bytes(
                    draft_shape, pre + row["m"], 1,
                )
                _, draft_cache = spec_dec._step(
                    draft, replacement, draft_cache,
                )
            else:
                target_cache = target_block
                draft_cache = spec_cache.trim_cache(
                    draft_cache, pre + row["Kp"],
                )
    total_end.record()
    recorder.enabled = False
    target_length = spec_cache.cache_length(target_cache)
    draft_length = spec_cache.cache_length(draft_cache)
    del target_cache, draft_cache
    return {
        "target_cache_cat_output_bytes": target_bytes,
        "draft_cache_cat_output_bytes": draft_bytes,
        "final_target_cache_length": target_length,
        "final_draft_cache_length": draft_length,
    }


def derive(timings):
    result = dict(timings)
    for prefix in ("target", "draft"):
        qkv = sum(result.get(f"{prefix}.{name}_projection", 0.0)
                  for name in ("q", "k", "v"))
        result[f"{prefix}.qkv_projection"] = qkv
        run_total = result.get(f"{prefix}.run_total", 0.0)
        layers = result.get(f"{prefix}.decoder_layers", 0.0)
        final_norm = result.get(f"{prefix}.final_norm", 0.0)
        rope_build = result.get(f"{prefix}.rope_build", 0.0)
        result[f"{prefix}.logits_and_run_residual"] = max(
            0.0, run_total - layers - final_norm - rope_build,
        )
        self_attention = result.get(f"{prefix}.self_attention", 0.0)
        output = result.get(f"{prefix}.o_projection", 0.0)
        rope_apply = result.get(f"{prefix}.rope_apply", 0.0)
        result[f"{prefix}.attention_kv_residual"] = max(
            0.0, self_attention - qkv - output - rope_apply,
        )
    return result


def one_profile(target, draft, ids, reference, backend, mode):
    recorder = CudaRecorder()
    with model_run_events(recorder, target, draft):
        if mode == "AR":
            metadata = profile_ar(
                recorder, target, ids, reference["ar_tokens"], backend,
                model_shape(target),
            )
        else:
            metadata = profile_specdec(
                recorder, target, draft, ids,
                reference["specdec_trace"], backend,
                model_shape(target), model_shape(draft),
            )
    timings = derive(recorder.totals())
    total_ms = timings["workload.decode_total"]
    timings["workload.embedding_bookkeeping_residual"] = max(
        0.0,
        total_ms
        - timings.get("target.run_total", 0.0)
        - timings.get("draft.run_total", 0.0),
    )
    metadata.update({
        "backend": backend, "mode": mode,
        "decode_total_ms": total_ms,
        "timings_ms": timings,
        "target_cache_cat_output_gib": (
            metadata["target_cache_cat_output_bytes"] / 2**30
        ),
        "draft_cache_cat_output_gib": (
            metadata["draft_cache_cat_output_bytes"] / 2**30
        ),
    })
    return metadata


def median_profiles(rows):
    timing_keys = sorted(set().union(*(
        row["timings_ms"].keys() for row in rows
    )))
    return {
        "decode_total_ms_median": statistics.median(
            row["decode_total_ms"] for row in rows
        ),
        "timings_ms_median": {
            key: statistics.median(
                row["timings_ms"].get(key, 0.0) for row in rows
            ) for key in timing_keys
        },
        "target_cache_cat_output_gib": rows[0][
            "target_cache_cat_output_gib"
        ],
        "draft_cache_cat_output_gib": rows[0][
            "draft_cache_cat_output_gib"
        ],
        "repetitions": len(rows),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    if REPS < 3:
        raise ValueError("FLASH_G4_PROFILE_REPS must be at least 3")
    torch.manual_seed(0)
    supporting.MAX_NEW = MAX_NEW
    supporting.CHUNK = CHUNK
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
    print("[load] target 1.5B", flush=True)
    target, _ = load_ours(TARGET)
    print("[load] draft 0.5B", flush=True)
    draft, _ = load_ours(DRAFT)
    eos_id = tokenizer.eos_token_id
    print(
        "Flash G4 bottleneck profile: "
        f"gpu={torch.cuda.get_device_name(0)} ctx={CONTEXT} "
        f"maxnew={MAX_NEW} reps={REPS}", flush=True,
    )

    with hybrid_scope():
        validation = validate_natural(target, draft, ids, eos_id)
        reference = validation["reference"]
        # Unrecorded warm-up covers Triton compilation and hook registration.
        for backend in BACKENDS:
            for mode in ("AR", "SpecDec"):
                one_profile(target, draft, ids, reference, backend, mode)
                gc.collect()
                torch.cuda.empty_cache()

        raw = {}
        summaries = {}
        for mode in ("AR", "SpecDec"):
            for backend in BACKENDS:
                key = f"{mode}:{backend}"
                rows = []
                for repetition in range(REPS):
                    row = one_profile(
                        target, draft, ids, reference, backend, mode,
                    )
                    row["repetition"] = repetition
                    rows.append(row)
                    print(
                        f"{key:24s} rep={repetition} "
                        f"decode={row['decode_total_ms']:.2f}ms "
                        f"target_cat={row['target_cache_cat_output_gib']:.2f}GiB "
                        f"draft_cat={row['draft_cache_cat_output_gib']:.2f}GiB",
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                raw[key] = rows
                summaries[key] = median_profiles(rows)

    result = {
        "status": "flash_g4_8k_decode_bottleneck_profile",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "context": CONTEXT,
        "max_new_tokens": MAX_NEW, "chunk": CHUNK, "repetitions": REPS,
        "text_source": source,
        "natural_validation": validation,
        "summaries": summaries, "runs": raw,
        "cache_copy_note": (
            "cat bytes are exact K/V output bytes written by the current "
            "full-cache append contract; approximate read+write traffic is "
            "about twice this value, excluding allocator effects."
        ),
        "production_engine_modified": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "flash_g4_bottleneck_profile.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== BOTTLENECK SUMMARY (median CUDA-event ms) ===")
    for mode in ("AR", "SpecDec"):
        for backend in BACKENDS:
            key = f"{mode}:{backend}"
            row = summaries[key]
            timing = row["timings_ms_median"]
            rope_ms = (
                timing.get("target.rope_apply", 0.0)
                + timing.get("target.rope_build", 0.0)
            )
            bookkeeping_ms = timing.get(
                "workload.embedding_bookkeeping_residual", 0.0,
            )
            cat_gib = (
                row["target_cache_cat_output_gib"]
                + row["draft_cache_cat_output_gib"]
            )
            print(
                f"{key:24s} total={row['decode_total_ms_median']:.2f} "
                f"target_qkv={timing.get('target.qkv_projection', 0.0):.2f} "
                f"target_rope={rope_ms:.2f} "
                f"target_attn+kv={timing.get('target.attention_kv_residual', 0.0):.2f} "
                f"target_o={timing.get('target.o_projection', 0.0):.2f} "
                f"target_mlp={timing.get('target.mlp', 0.0):.2f} "
                f"target_logits+res={timing.get('target.logits_and_run_residual', 0.0):.2f} "
                f"draft_layers={timing.get('draft.decoder_layers', 0.0):.2f} "
                f"bookkeeping+embed={bookkeeping_ms:.2f} "
                f"cat_write={cat_gib:.2f}GiB"
            )
    print(f"JSON={output}")
    del target, draft, ids, all_ids
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
