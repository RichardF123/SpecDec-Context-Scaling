"""Autoregressive H2O v0 reference integration.

Separate from SpecDec. Uses absolute processed-token positions for RoPE while
resident KV may be much shorter after eviction.
"""
from __future__ import annotations

import torch
from . import paged_cache, spec_cache
from .h2o_policy import (
    H2OConfig, H2OStats, cache_payload_bytes, init_layer_state,
    update_after_decode,
)
from .model_self import attention_phase


def h2o_generate_greedy(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    hh_size: int,
    recent_size: int,
    page_size: int = 16,
    eos_id: int | None = None,
    attention_mode: str = "direct_cuda",
) -> dict:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("H2O v0 requires prompt_ids shape [1, T]")
    if prompt_ids.shape[1] <= 0:
        raise ValueError("prompt must be non-empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if attention_mode not in ("direct_cuda", "materialized_cuda"):
        raise ValueError("attention_mode must be direct_cuda or materialized_cuda")

    config = H2OConfig(hh_size=hh_size, recent_size=recent_size)
    model.set_kv_format("fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(attention_mode)
    model.set_h2o_score_capture(False)

    prompt_len = int(prompt_ids.shape[1])
    with torch.inference_mode(), attention_phase("prefill"):
        logits, cache = model._run(
            model.embed_tokens(prompt_ids), None, need_logits=True,
            start_pos_override=0,
        )
    spec_cache.check_consistent(cache)

    states = [
        init_layer_state(prompt_len, prompt_ids.device)
        for _ in model.layers
    ]
    generated = prompt_ids.clone()
    seen = prompt_len
    prune_events = 0
    tokens_evicted = 0
    max_resident = spec_cache.cache_length(cache)

    model.set_h2o_score_capture(True)
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                token = logits[:, -1].argmax(dim=-1)
                if eos_id is not None and bool((token == eos_id).all()):
                    break
                generated = torch.cat([generated, token[:, None]], dim=1)

                with attention_phase("decode"):
                    logits, cache = model._run(
                        model.embed_tokens(token[:, None]),
                        cache,
                        need_logits=True,
                        start_pos_override=seen,
                    )
                absolute_position = seen
                seen += 1

                masses = model.pop_h2o_scores()
                if len(masses) != len(cache):
                    raise RuntimeError("H2O score/cache layer count mismatch")

                new_cache = []
                step_pruned = False
                for layer_index, (layer, state, mass) in enumerate(
                        zip(cache, states, masses)):
                    before = paged_cache.logical_length(layer)
                    updated, state, keep = update_after_decode(
                        layer, state, mass, absolute_position, config)
                    states[layer_index] = state
                    new_cache.append(updated)
                    if keep is not None:
                        step_pruned = True
                        tokens_evicted += before - int(keep.numel())

                cache = new_cache
                if step_pruned:
                    prune_events += 1
                spec_cache.check_consistent(cache)
                resident = spec_cache.cache_length(cache)
                max_resident = max(max_resident, resident)
                if resident > config.budget:
                    raise RuntimeError(
                        f"H2O resident cache {resident} exceeds budget {config.budget}")
    finally:
        model.set_h2o_score_capture(False)

    final_resident = spec_cache.cache_length(cache)
    stats = H2OStats(
        prune_events=prune_events,
        tokens_evicted=tokens_evicted,
        max_resident=max_resident,
        final_resident=final_resident,
        seen_tokens=seen,
        per_layer_evictions=[state.evictions for state in states],
    )
    return {
        "ids": generated,
        "cache": cache,
        "states": states,
        "stats": stats,
        "payload_bytes": cache_payload_bytes(cache),
        "budget": config.budget,
        "hh_size": config.hh_size,
        "recent_size": config.recent_size,
    }
