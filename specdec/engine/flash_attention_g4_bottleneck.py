"""Isolated 8K AR/SpecDec bottleneck diagnostic for eager versus Flash.

The script instruments the latest engine at runtime; production files and the
G4 cache contract are unchanged.  CUDA events measure projections, RoPE,
BF16 KV ``torch.cat`` movement, attention, MLP, normalization, logits, and full
model calls.  The wall-time residual is reported as host/bookkeeping plus any
unlabelled GPU work, so it is diagnostic rather than a profiler-perfect causal
decomposition.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time
import types

import torch

from . import model_self as ms
from .flash_attention_g4_ablation import (
    CHUNK, DRAFT, MAX_NEW, TARGET, run_ar, run_specdec,
)
from .g4_b4_cache_tests import _from_pretrained, load_ours
from .g4_part4_pilot_measure import load_text
from . import flash_attention_two_pass_triton as flash_module


CONTEXT = int(os.environ.get("FLASH_PROFILE_CTX", "8192"))
OUTDIR = Path(os.environ.get(
    "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
))


class Recorder:
    def __init__(self):
        self.enabled = False
        self.current_role = None
        self.events = defaultdict(list)

    def start(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, label, start):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events[label].append((start, end))

    def call(self, label, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = self.start()
        result = function(*args, **kwargs)
        self.stop(label, start)
        return result

    def reset(self):
        self.events.clear()

    def totals_ms(self):
        torch.cuda.synchronize()
        return {
            label: sum(start.elapsed_time(end) for start, end in pairs)
            for label, pairs in self.events.items()
        }


def add_module_timer(module, label, recorder, handles, starts):
    def before(_module, _args):
        if recorder.enabled:
            starts[id(_module)] = recorder.start()

    def after(_module, _args, _output):
        if recorder.enabled:
            recorder.stop(label, starts.pop(id(_module)))

    handles.append(module.register_forward_pre_hook(before))
    handles.append(module.register_forward_hook(after))


def install_module_timers(model, role, recorder, handles, starts):
    add_module_timer(model.embed_tokens, f"{role}.embedding", recorder, handles, starts)
    add_module_timer(model.norm, f"{role}.normalization", recorder, handles, starts)
    for layer in model.layers:
        attention = layer.self_attn
        add_module_timer(
            attention, f"{role}.self_attention_total", recorder, handles, starts,
        )
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            add_module_timer(
                projection, f"{role}.qkv_projection", recorder, handles, starts,
            )
        add_module_timer(
            attention.o_proj, f"{role}.output_projection", recorder, handles, starts,
        )
        add_module_timer(layer.mlp, f"{role}.mlp", recorder, handles, starts)
        add_module_timer(
            layer.input_layernorm,
            f"{role}.normalization", recorder, handles, starts,
        )
        add_module_timer(
            layer.post_attention_layernorm,
            f"{role}.normalization", recorder, handles, starts,
        )


def install_profiled_run(model, role, recorder):
    def profiled_run(self, x, cache_list=None, need_logits=True):
        start = recorder.start() if recorder.enabled else None
        previous_role = recorder.current_role
        recorder.current_role = role
        try:
            start_pos = 0
            if cache_list is not None and cache_list[0] is not None:
                start_pos = cache_list[0]["k"].shape[2]
            cos, sin = ms.Build_Rope_cache(
                self.head_dim,
                seq_len=x.size(1),
                theta=self.rope_theta,
                device=x.device,
                dtype=x.dtype,
                start_pos=start_pos,
            )
            new_cache_list = []
            for index, layer in enumerate(self.layers):
                cache = cache_list[index] if cache_list is not None else None
                x, new_cache = layer(x, cos, sin, cache)
                new_cache_list.append(new_cache)
            x = self.norm(x)
            if need_logits:
                logits_start = recorder.start() if recorder.enabled else None
                logits = x @ self.embed_tokens.weight.T
                if recorder.enabled:
                    recorder.stop(f"{role}.logits", logits_start)
                return logits, new_cache_list
            return None, new_cache_list
        finally:
            recorder.current_role = previous_role
            if recorder.enabled:
                recorder.stop(f"{role}.model_run", start)

    model._run = types.MethodType(profiled_run, model)


def install_function_timers(recorder):
    originals = {
        "apply_rope": ms.apply_rope,
        "cat": torch.cat,
        "flash": flash_module.flash_attention_bf16_two_pass,
    }

    def profiled_rope(*args, **kwargs):
        role = recorder.current_role or "unknown"
        return recorder.call(
            f"{role}.rope", originals["apply_rope"], *args, **kwargs,
        )

    def profiled_cat(tensors, dim=0, *args, **kwargs):
        role = recorder.current_role or "unknown"
        is_kv_append = bool(
            dim == 2
            and tensors
            and all(isinstance(value, torch.Tensor) and value.ndim == 4
                    for value in tensors)
        )
        if is_kv_append:
            return recorder.call(
                f"{role}.kv_append_cat",
                originals["cat"], tensors, dim, *args, **kwargs,
            )
        return originals["cat"](tensors, dim, *args, **kwargs)

    def profiled_flash(*args, **kwargs):
        role = recorder.current_role or "unknown"
        return recorder.call(
            f"{role}.flash_kernel", originals["flash"], *args, **kwargs,
        )

    ms.apply_rope = profiled_rope
    torch.cat = profiled_cat
    flash_module.flash_attention_bf16_two_pass = profiled_flash
    return originals


def restore_function_timers(originals):
    ms.apply_rope = originals["apply_rope"]
    torch.cat = originals["cat"]
    flash_module.flash_attention_bf16_two_pass = originals["flash"]


def value(totals, suffix):
    return sum(milliseconds for label, milliseconds in totals.items()
               if label.endswith(suffix))


def exclusive_breakdown(totals, wall_ms):
    model_run = value(totals, ".model_run")
    embedding = value(totals, ".embedding")
    qkv = value(totals, ".qkv_projection")
    rope = value(totals, ".rope")
    kv_append = value(totals, ".kv_append_cat")
    output_projection = value(totals, ".output_projection")
    self_attention = value(totals, ".self_attention_total")
    attention_math = max(
        0.0,
        self_attention - qkv - rope - kv_append - output_projection,
    )
    mlp = value(totals, ".mlp")
    normalization = value(totals, ".normalization")
    logits = value(totals, ".logits")
    labelled = (
        embedding + qkv + rope + kv_append + output_projection
        + attention_math + mlp + normalization + logits
    )
    return {
        "wall_ms": wall_ms,
        "model_run_gpu_ms": model_run,
        "embedding_gpu_ms": embedding,
        "qkv_projection_gpu_ms": qkv,
        "rope_gpu_ms": rope,
        "kv_append_cat_gpu_ms": kv_append,
        "attention_math_gpu_ms": attention_math,
        "flash_kernel_gpu_ms_inside_attention": value(totals, ".flash_kernel"),
        "output_projection_gpu_ms": output_projection,
        "mlp_gpu_ms": mlp,
        "normalization_gpu_ms": normalization,
        "logits_gpu_ms": logits,
        "other_or_overlap_gpu_ms": max(0.0, model_run + embedding - labelled),
        "host_bookkeeping_and_unlabelled_ms": max(
            0.0, wall_ms - model_run - embedding,
        ),
        "raw_event_totals_ms": totals,
    }


def run_profile(target, draft, ids, eos_id, backend, mode, recorder):
    target.set_kv_format("fp16")
    draft.set_kv_format("fp16")
    target.set_attention_backend(backend)
    draft.set_attention_backend("eager")
    recorder.enabled = False
    if mode == "AR":
        run_ar(target, ids, eos_id)
    else:
        run_specdec(target, draft, ids, eos_id)
    gc.collect()
    torch.cuda.empty_cache()
    recorder.reset()
    recorder.enabled = True
    started = time.perf_counter()
    if mode == "AR":
        generated, prefill_s, decode_s, stats = run_ar(target, ids, eos_id)
    else:
        generated, prefill_s, decode_s, stats = run_specdec(
            target, draft, ids, eos_id,
        )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    recorder.enabled = False
    totals = recorder.totals_ms()
    return {
        "backend": backend,
        "mode": mode,
        "ctx": int(ids.shape[1]),
        "n_generated": len(generated),
        "generated": generated,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "stats": stats,
        "breakdown": exclusive_breakdown(totals, wall_ms),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
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
    if all_ids.shape[1] < CONTEXT:
        raise RuntimeError(f"corpus has {all_ids.shape[1]} tokens, need {CONTEXT}")
    target, _ = load_ours(TARGET)
    draft, _ = load_ours(DRAFT)
    ids = all_ids[:, :CONTEXT].cuda()
    recorder = Recorder()
    handles = []
    starts = {}
    install_module_timers(target, "target", recorder, handles, starts)
    install_module_timers(draft, "draft", recorder, handles, starts)
    install_profiled_run(target, "target", recorder)
    install_profiled_run(draft, "draft", recorder)
    originals = install_function_timers(recorder)
    records = []
    try:
        for mode in ("AR", "SpecDec"):
            for backend in ("eager", "triton_bf16_two_pass"):
                print(f"[profile] ctx={CONTEXT} mode={mode} backend={backend}", flush=True)
                row = run_profile(
                    target, draft, ids, tokenizer.eos_token_id,
                    backend, mode, recorder,
                )
                records.append(row)
                b = row["breakdown"]
                print(
                    f"  wall={b['wall_ms']:.1f}ms "
                    f"qkv={b['qkv_projection_gpu_ms']:.1f} "
                    f"rope={b['rope_gpu_ms']:.1f} "
                    f"kv_cat={b['kv_append_cat_gpu_ms']:.1f} "
                    f"attention={b['attention_math_gpu_ms']:.1f} "
                    f"out_proj={b['output_projection_gpu_ms']:.1f} "
                    f"logits={b['logits_gpu_ms']:.1f}",
                    flush=True,
                )
    finally:
        recorder.enabled = False
        restore_function_timers(originals)
        for handle in handles:
            handle.remove()
    comparisons = {}
    for mode in ("AR", "SpecDec"):
        eager = next(row for row in records if row["mode"] == mode and row["backend"] == "eager")
        flash = next(row for row in records if row["mode"] == mode and row["backend"] == "triton_bf16_two_pass")
        comparisons[mode] = {
            "generated_equal": eager["generated"] == flash["generated"],
            "eager_wall_ms": eager["breakdown"]["wall_ms"],
            "flash_wall_ms": flash["breakdown"]["wall_ms"],
            "wall_speedup": (
                eager["breakdown"]["wall_ms"] / flash["breakdown"]["wall_ms"]
            ),
            "eager_kv_append_ms": eager["breakdown"]["kv_append_cat_gpu_ms"],
            "flash_kv_append_ms": flash["breakdown"]["kv_append_cat_gpu_ms"],
            "eager_attention_ms": eager["breakdown"]["attention_math_gpu_ms"],
            "flash_attention_ms": flash["breakdown"]["attention_math_gpu_ms"],
        }
    result = {
        "status": "runtime_only_bottleneck_diagnostic",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "context": CONTEXT,
        "chunk": CHUNK,
        "max_new": MAX_NEW,
        "text_source": source,
        "records": records,
        "comparisons": comparisons,
        "caveat": (
            "CUDA-event categories are diagnostic; host residual includes Python, "
            "embedding gaps, synchronization, and any unlabelled work."
        ),
    }
    output = OUTDIR / "flash_attention_g4_bottleneck.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"JSON={output}")
    return 0 if all(row["generated_equal"] for row in comparisons.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
