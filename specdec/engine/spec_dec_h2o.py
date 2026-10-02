"""Speculative decoding + H2O committed-history integration (v2).

Key invariant
-------------
H2O state is mutated ONLY after a speculative round has resolved.

During a round:
- draft proposal score rows are temporary;
- target verification score rows are temporary;
- no H2O pruning occurs;
- resident lengths therefore remain stable except for the speculative tail;
- rollback can safely trim by resident-prefix length.

After the accept/mismatch decision:
- only accepted proposal/verification rows are committed;
- a replacement repair row is committed on mismatch;
- rejected speculative rows are discarded;
- H2O pruning runs once at the round boundary.

Absolute RoPE positions use `seen_tokens`, never resident KV length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import paged_cache, spec_cache
from .h2o_policy import (
    H2OConfig,
    LayerH2OState,
    accumulate_committed,
    init_layer_state,
    prune_committed,
    validate_committed_alignment,
)
from .model_self import attention_phase


K = 4
FMT = {"base_bf16": "fp16", "v_fp8": "v_fp8"}


@dataclass
class SpecH2OStats:
    prune_rounds: int = 0
    target_layer_token_evictions: int = 0
    draft_layer_token_evictions: int = 0

    # Captured speculative evidence.
    proposal_rows_captured: int = 0
    verify_rows_captured: int = 0
    repair_rows_captured: int = 0

    # Evidence that was actually allowed to update committed H2O state.
    committed_proposal_rows: int = 0
    committed_verify_rows: int = 0
    committed_repair_rows: int = 0

    # Explicit proof that rejected speculative evidence was discarded.
    discarded_proposal_rows: int = 0
    discarded_verify_rows: int = 0

    max_target_resident: int = 0
    max_draft_resident: int = 0
    final_target_resident: int = 0
    final_draft_resident: int = 0
    seen_tokens: int = 0

    target_per_layer_evictions: list[int] = field(default_factory=list)
    draft_per_layer_evictions: list[int] = field(default_factory=list)


def _configure_paged(
    model,
    *,
    kv_name: str,
    page_size: int,
    attention_mode: str,
    allow_v_fp8: bool,
) -> None:
    if attention_mode not in ("direct_cuda", "materialized_cuda", "off"):
        raise ValueError(
            "attention_mode must be direct_cuda, materialized_cuda, or off")
    if kv_name not in FMT:
        raise ValueError(f"unknown KV format {kv_name!r}")
    if kv_name == "v_fp8" and not allow_v_fp8:
        raise ValueError("v_fp8 is not supported for the draft model in H2O v3")

    model.set_kv_format(FMT[kv_name])
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(attention_mode)
    model.set_h2o_score_capture(False)


def _prefill(model, ids):
    with torch.inference_mode(), attention_phase("prefill"):
        logits, cache = model._run(
            model.embed_tokens(ids),
            None,
            need_logits=True,
            start_pos_override=0,
        )
    return logits, cache


def _proposal_step(
    draft,
    token: int,
    cache,
    *,
    absolute_position: int,
):
    with torch.inference_mode(), attention_phase("propose"):
        logits, cache = draft._run(
            draft.embed_tokens(torch.tensor(
                [[token]],
                device=spec_cache.cache_device(cache),
            )),
            cache,
            need_logits=True,
            start_pos_override=absolute_position,
        )
    masses = draft.pop_h2o_scores()
    return logits, cache, masses


def _propose(
    draft,
    cache,
    logits,
    k_eff: int,
    eos_id: int | None,
    *,
    start_position: int,
):
    candidates: list[int] = []
    masses_by_token: list[list[torch.Tensor]] = []

    for j in range(k_eff):
        token = int(
            logits.reshape(-1, logits.shape[-1])[-1].argmax())
        candidates.append(token)

        logits, cache, masses = _proposal_step(
            draft,
            token,
            cache,
            absolute_position=start_position + j,
        )
        masses_by_token.append(masses)

        if eos_id is not None and token == eos_id:
            break

    return candidates, cache, logits, masses_by_token


def _verify(
    target,
    cache,
    candidates: list[int],
    *,
    start_position: int,
):
    with torch.inference_mode(), attention_phase("verify"):
        logits, cache_blk = target._run(
            target.embed_tokens(torch.tensor(
                [candidates],
                device=spec_cache.cache_device(cache),
            )),
            cache,
            need_logits=True,
            start_pos_override=start_position,
        )
    masses = target.pop_h2o_scores()
    kp = len(candidates)
    return logits[0, -kp:], cache_blk, masses


def _repair(
    model,
    token: int,
    cache,
    *,
    absolute_position: int,
):
    with torch.inference_mode(), attention_phase("repair"):
        logits, cache = model._run(
            model.embed_tokens(torch.tensor(
                [[token]],
                device=spec_cache.cache_device(cache),
            )),
            cache,
            need_logits=True,
            start_pos_override=absolute_position,
        )
    masses = model.pop_h2o_scores()
    return logits[:, -1:], cache, masses


def _mass_row(
    captured: torch.Tensor,
    *,
    query_index: int,
    expected_length: int,
) -> torch.Tensor:
    """Extract one causal row and trim unreachable speculative suffix."""
    if captured.ndim == 1:
        if query_index != 0:
            raise RuntimeError(
                "rank-1 H2O capture only has query_index=0")
        row = captured
    elif captured.ndim == 2:
        if query_index < 0 or query_index >= captured.shape[0]:
            raise RuntimeError("H2O query index out of range")
        row = captured[query_index]
    else:
        raise RuntimeError(
            f"unexpected H2O capture rank {captured.ndim}")

    if expected_length <= 0 or expected_length > int(row.numel()):
        raise RuntimeError(
            f"invalid committed H2O row length {expected_length} "
            f"for captured length {row.numel()}"
        )

    row = row[:expected_length].contiguous()
    if not bool(torch.isfinite(row).all().item()):
        raise RuntimeError("committed H2O mass contains non-finite values")
    return row


def _commit_proposal_prefix(
    states: list[LayerH2OState],
    masses_by_token: list[list[torch.Tensor]],
    *,
    count: int,
    base_resident: int,
    start_position: int,
) -> None:
    if count < 0 or count > len(masses_by_token):
        raise ValueError("invalid committed proposal prefix length")

    for i in range(count):
        layer_masses = masses_by_token[i]
        if len(layer_masses) != len(states):
            raise RuntimeError(
                "proposal H2O layer count mismatch")
        expected = base_resident + i + 1
        for li, state in enumerate(states):
            row = _mass_row(
                layer_masses[li],
                query_index=0,
                expected_length=expected,
            )
            states[li] = accumulate_committed(
                state,
                row,
                start_position + i,
            )


def _commit_verify_prefix(
    states: list[LayerH2OState],
    layer_captures: list[torch.Tensor],
    *,
    count: int,
    base_resident: int,
    start_position: int,
) -> None:
    if len(layer_captures) != len(states):
        raise RuntimeError(
            "verify H2O layer count mismatch")

    for i in range(count):
        expected = base_resident + i + 1
        for li, state in enumerate(states):
            row = _mass_row(
                layer_captures[li],
                query_index=i,
                expected_length=expected,
            )
            states[li] = accumulate_committed(
                state,
                row,
                start_position + i,
            )


def _commit_repair(
    states: list[LayerH2OState],
    layer_masses: list[torch.Tensor],
    *,
    expected_length: int,
    absolute_position: int,
) -> None:
    if len(layer_masses) != len(states):
        raise RuntimeError(
            "repair H2O layer count mismatch")

    for li, state in enumerate(states):
        row = _mass_row(
            layer_masses[li],
            query_index=0,
            expected_length=expected_length,
        )
        states[li] = accumulate_committed(
            state,
            row,
            absolute_position,
        )


def _prune_model(
    cache,
    states: list[LayerH2OState],
    config: H2OConfig,
) -> tuple[list[dict], list[LayerH2OState], int, bool]:
    if len(cache) != len(states):
        raise RuntimeError("H2O cache/state layer count mismatch")

    new_cache = []
    total_evicted = 0
    any_pruned = False

    for index, (layer, state) in enumerate(zip(cache, states)):
        before = paged_cache.logical_length(layer)
        updated, state, keep = prune_committed(
            layer,
            state,
            config,
        )
        states[index] = state
        new_cache.append(updated)

        if keep is not None:
            any_pruned = True
            total_evicted += before - int(keep.numel())

    return new_cache, states, total_evicted, any_pruned


def spec_decode_greedy_h2o(
    target,
    draft,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    hh_size: int,
    recent_size: int,
    k: int = K,
    eos_id: int | None = None,
    page_size: int = 16,
    attention_mode: str = "direct_cuda",
    target_kv: str = "base_bf16",
    draft_kv: str = "base_bf16",
    log=None,
    return_caches: bool = True,
) -> dict:
    """Greedy SpecDec with H2O pruning only at committed round boundaries.

    v3 scope:
    - target supports BF16/base or V-FP8;
    - draft remains BF16/base;
    - paged_v0 cache;
    - batch size 1;
    - H2O score capture is reference/materialized;
    - target verify itself remains the existing materialized attention path.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(
            "H2O/SpecDec v2 requires prompt_ids shape [1,T]")
    if int(prompt_ids.shape[1]) <= 0:
        raise ValueError("prompt must be non-empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if k <= 0:
        raise ValueError("k must be positive")

    config = H2OConfig(
        hh_size=hh_size,
        recent_size=recent_size,
    )

    _configure_paged(
        target,
        kv_name=target_kv,
        page_size=page_size,
        attention_mode=attention_mode,
        allow_v_fp8=True,
    )
    _configure_paged(
        draft,
        kv_name=draft_kv,
        page_size=page_size,
        attention_mode=attention_mode,
        allow_v_fp8=False,
    )

    target_logits, target_cache = _prefill(
        target, prompt_ids)
    draft_logits, draft_cache = _prefill(
        draft, prompt_ids)

    saved_next = target_logits[:, -1:]
    draft_logits = draft_logits[:, -1:]

    prompt_len = int(prompt_ids.shape[1])
    seen = prompt_len

    target_states = [
        init_layer_state(
            prompt_len,
            prompt_ids.device,
        )
        for _ in target.layers
    ]
    draft_states = [
        init_layer_state(
            prompt_len,
            prompt_ids.device,
        )
        for _ in draft.layers
    ]

    validate_committed_alignment(
        target_cache,
        target_states,
        seen_tokens=seen,
    )
    validate_committed_alignment(
        draft_cache,
        draft_states,
        seen_tokens=seen,
    )

    gen: list[int] = []
    st = dict(
        rounds=0,
        proposed=0,
        accepted=0,
        target_block_calls=0,
        draft_calls=0,
        mismatch_rounds=0,
        all_accept_rounds=0,
        rollbacks=0,
    )
    hst = SpecH2OStats(
        max_target_resident=spec_cache.cache_length(
            target_cache),
        max_draft_resident=spec_cache.cache_length(
            draft_cache),
        seen_tokens=seen,
    )

    target.set_h2o_score_capture(True)
    draft.set_h2o_score_capture(True)

    try:
        while len(gen) < max_new_tokens:
            if seen != prompt_len + len(gen):
                raise RuntimeError(
                    "logical seen-token counter drifted from committed output")

            validate_committed_alignment(
                target_cache,
                target_states,
                seen_tokens=seen,
            )
            validate_committed_alignment(
                draft_cache,
                draft_states,
                seen_tokens=seen,
            )

            k_eff = min(
                k,
                max_new_tokens - len(gen),
            )

            pre_target = spec_cache.cache_length(
                target_cache)
            pre_draft = spec_cache.cache_length(
                draft_cache)

            candidates, draft_spec_cache, prop_logits, proposal_masses = _propose(
                draft,
                draft_cache,
                draft_logits,
                k_eff,
                eos_id,
                start_position=seen,
            )
            kp = len(candidates)
            st["draft_calls"] += kp
            hst.proposal_rows_captured += kp

            if kp == 0:
                break

            rows, target_spec_cache, verify_masses = _verify(
                target,
                target_cache,
                candidates,
                start_position=seen,
            )
            st["target_block_calls"] += 1
            st["proposed"] += kp
            hst.verify_rows_captured += kp

            greedy = [
                int(x.argmax())
                for x in (
                    [saved_next]
                    + [rows[j] for j in range(kp - 1)]
                )
            ]

            m = 0
            while m < kp and greedy[m] == candidates[m]:
                m += 1

            # Only accepted proposal/verify rows may update committed H2O state.
            _commit_proposal_prefix(
                draft_states,
                proposal_masses,
                count=m,
                base_resident=pre_draft,
                start_position=seen,
            )
            _commit_verify_prefix(
                target_states,
                verify_masses,
                count=m,
                base_resident=pre_target,
                start_position=seen,
            )
            hst.committed_proposal_rows += m
            hst.committed_verify_rows += m
            hst.discarded_proposal_rows += kp - m
            hst.discarded_verify_rows += kp - m

            if m < kp:
                replacement = greedy[m]
                emitted = candidates[:m] + [replacement]

                target_cache = spec_cache.trim_cache(
                    target_spec_cache,
                    pre_target + m,
                )
                draft_cache = spec_cache.trim_cache(
                    draft_spec_cache,
                    pre_draft + m,
                )

                repair_position = seen + m

                saved_next, target_cache, target_repair_mass = _repair(
                    target,
                    replacement,
                    target_cache,
                    absolute_position=repair_position,
                )
                draft_logits, draft_cache, draft_repair_mass = _repair(
                    draft,
                    replacement,
                    draft_cache,
                    absolute_position=repair_position,
                )
                hst.repair_rows_captured += 2

                _commit_repair(
                    target_states,
                    target_repair_mass,
                    expected_length=pre_target + m + 1,
                    absolute_position=repair_position,
                )
                _commit_repair(
                    draft_states,
                    draft_repair_mass,
                    expected_length=pre_draft + m + 1,
                    absolute_position=repair_position,
                )
                hst.committed_repair_rows += 2

                st["mismatch_rounds"] += 1
                st["rollbacks"] += 1

            else:
                emitted = list(candidates)
                target_cache = target_spec_cache
                draft_cache = draft_spec_cache
                saved_next = rows[kp - 1]
                draft_logits = prop_logits
                st["all_accept_rounds"] += 1

            # At this point caches + H2O states describe only committed tokens.
            # Pruning is finally allowed.
            target_cache, target_states, target_evicted, target_pruned = (
                _prune_model(
                    target_cache,
                    target_states,
                    config,
                )
            )
            draft_cache, draft_states, draft_evicted, draft_pruned = (
                _prune_model(
                    draft_cache,
                    draft_states,
                    config,
                )
            )

            if target_pruned or draft_pruned:
                hst.prune_rounds += 1
            hst.target_layer_token_evictions += target_evicted
            hst.draft_layer_token_evictions += draft_evicted

            gen += emitted
            seen += len(emitted)

            st["accepted"] += m
            st["rounds"] += 1

            validate_committed_alignment(
                target_cache,
                target_states,
                seen_tokens=seen,
            )
            validate_committed_alignment(
                draft_cache,
                draft_states,
                seen_tokens=seen,
            )

            target_resident = spec_cache.cache_length(
                target_cache)
            draft_resident = spec_cache.cache_length(
                draft_cache)

            hst.max_target_resident = max(
                hst.max_target_resident,
                target_resident,
            )
            hst.max_draft_resident = max(
                hst.max_draft_resident,
                draft_resident,
            )

            if target_resident > config.budget:
                raise RuntimeError(
                    f"target resident {target_resident} "
                    f"> H2O budget {config.budget}"
                )
            if draft_resident > config.budget:
                raise RuntimeError(
                    f"draft resident {draft_resident} "
                    f"> H2O budget {config.budget}"
                )

            if log is not None:
                log(dict(
                    round=st["rounds"] - 1,
                    Kp=kp,
                    m=m,
                    seen=seen,
                    pre_target_resident=pre_target,
                    pre_draft_resident=pre_draft,
                    target_resident=target_resident,
                    draft_resident=draft_resident,
                    candidates=candidates,
                    greedy=greedy,
                    emitted=emitted,
                    pruned=bool(
                        target_pruned or draft_pruned),
                ))

            if emitted and eos_id is not None:
                if emitted[-1] == eos_id:
                    break

    finally:
        target.set_h2o_score_capture(False)
        draft.set_h2o_score_capture(False)

    hst.final_target_resident = spec_cache.cache_length(
        target_cache)
    hst.final_draft_resident = spec_cache.cache_length(
        draft_cache)
    hst.seen_tokens = seen
    hst.target_per_layer_evictions = [
        state.evictions for state in target_states
    ]
    hst.draft_per_layer_evictions = [
        state.evictions for state in draft_states
    ]

    # Accounting invariants: rejected speculative rows were captured but never
    # admitted to committed H2O state.
    if hst.committed_proposal_rows != st["accepted"]:
        raise RuntimeError(
            "draft committed proposal-row accounting mismatch")
    if hst.committed_verify_rows != st["accepted"]:
        raise RuntimeError(
            "target committed verify-row accounting mismatch")
    if hst.discarded_proposal_rows != st["proposed"] - st["accepted"]:
        raise RuntimeError(
            "discarded proposal-row accounting mismatch")
    if hst.discarded_verify_rows != st["proposed"] - st["accepted"]:
        raise RuntimeError(
            "discarded verify-row accounting mismatch")
    if hst.committed_repair_rows != 2 * st["mismatch_rounds"]:
        raise RuntimeError(
            "repair-row accounting mismatch")
    if len(gen) != st["accepted"] + st["mismatch_rounds"]:
        raise RuntimeError(
            "committed token accounting mismatch")

    result = {
        "ids": list(prompt_ids[0]) + gen,
        "gen": gen,
        "stats": st,
        "h2o_stats": hst,
        "budget": config.budget,
        "hh_size": config.hh_size,
        "recent_size": config.recent_size,
        "target_kv": target_kv,
        "draft_kv": draft_kv,
        "seen_tokens": seen,
        "target_states": target_states,
        "draft_states": draft_states,
    }

    if return_caches:
        result["target_cache"] = target_cache
        result["draft_cache"] = draft_cache

    return result
