"""H2O-inspired global heavy-hitter retention policy for paged KV.

v0 is a correctness/controller prototype. It keeps accumulated heavy hitters
plus a recent window, but uses one global token set per layer because this
engine's paged GQA storage has one shared token axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from . import paged_cache


@dataclass(frozen=True)
class H2OConfig:
    hh_size: int
    recent_size: int

    def __post_init__(self) -> None:
        for name, value in (("hh_size", self.hh_size),
                            ("recent_size", self.recent_size)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ValueError(f"{name} must be a non-negative int")
        if self.hh_size + self.recent_size <= 0:
            raise ValueError("H2O cache budget must be positive")

    @property
    def budget(self) -> int:
        return self.hh_size + self.recent_size


@dataclass
class LayerH2OState:
    scores: torch.Tensor
    positions: torch.Tensor
    evictions: int = 0
    tokens_evicted: int = 0


@dataclass
class H2OStats:
    prune_events: int = 0
    tokens_evicted: int = 0
    max_resident: int = 0
    final_resident: int = 0
    seen_tokens: int = 0
    per_layer_evictions: list[int] = field(default_factory=list)


def init_layer_state(prompt_len: int, device: torch.device) -> LayerH2OState:
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    return LayerH2OState(
        scores=torch.zeros(prompt_len, device=device, dtype=torch.float32),
        positions=torch.arange(prompt_len, device=device, dtype=torch.long),
    )


def keep_indices(scores: torch.Tensor, config: H2OConfig) -> torch.Tensor:
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
        raise ValueError("scores must be a rank-1 tensor")
    n = int(scores.numel())
    if n <= config.budget:
        return torch.arange(n, device=scores.device, dtype=torch.long)

    recent = min(config.recent_size, n)
    old_end = n - recent
    hh = min(config.hh_size, old_end)

    pieces = []
    if hh:
        heavy = torch.topk(
            scores[:old_end], k=hh, largest=True, sorted=False).indices
        pieces.append(heavy.sort().values)
    if recent:
        pieces.append(torch.arange(
            old_end, n, device=scores.device, dtype=torch.long))
    if not pieces:
        raise RuntimeError("H2O policy selected no tokens")
    return torch.cat(pieces).sort().values


def update_after_decode(
    layer: dict,
    state: LayerH2OState,
    attention_mass: torch.Tensor,
    absolute_position: int,
    config: H2OConfig,
) -> tuple[dict, LayerH2OState, torch.Tensor | None]:
    paged_cache.check_consistent(layer)
    if layer["fmt"] != "fp16":
        raise ValueError("H2O v0 supports fp16/base paged cache only")
    if not isinstance(attention_mass, torch.Tensor) or attention_mass.ndim != 1:
        raise ValueError("attention_mass must be a rank-1 tensor")
    attention_mass = attention_mass.float()

    prior = int(state.scores.numel())
    resident = paged_cache.logical_length(layer)
    if resident != prior + 1:
        raise RuntimeError(
            f"H2O state/cache mismatch: resident={resident}, expected={prior + 1}")
    if int(attention_mass.numel()) != resident:
        raise RuntimeError(
            f"H2O attention length {attention_mass.numel()} != resident {resident}")
    if int(state.positions.numel()) != prior:
        raise RuntimeError("H2O state positions/scores length mismatch")
    if state.positions.numel() and absolute_position <= int(state.positions[-1]):
        raise ValueError("absolute_position must increase monotonically")

    scores = torch.cat([
        state.scores,
        torch.zeros(1, device=state.scores.device, dtype=torch.float32),
    ]) + attention_mass.to(state.scores.device)
    positions = torch.cat([
        state.positions,
        torch.tensor([absolute_position],
                     device=state.positions.device, dtype=torch.long),
    ])

    keep = keep_indices(scores, config)
    if int(keep.numel()) == resident:
        state.scores = scores
        state.positions = positions
        return layer, state, None

    new_layer = paged_cache.select_indices(layer, keep)
    evicted = resident - int(keep.numel())
    state.scores = scores.index_select(0, keep)
    state.positions = positions.index_select(0, keep)
    state.evictions += 1
    state.tokens_evicted += evicted

    if state.positions.numel() > 1:
        if not bool(torch.all(state.positions[1:] > state.positions[:-1]).item()):
            raise RuntimeError("H2O retained absolute positions are not increasing")
    return new_layer, state, keep


def cache_payload_bytes(cache_list: list[dict]) -> int:
    return sum(paged_cache.payload_bytes(layer) for layer in cache_list)


def clone_layer_state(state: LayerH2OState) -> LayerH2OState:
    """Deep-copy mutable H2O state tensors/counters."""
    return LayerH2OState(
        scores=state.scores.clone(),
        positions=state.positions.clone(),
        evictions=int(state.evictions),
        tokens_evicted=int(state.tokens_evicted),
    )


def accumulate_committed(
    state: LayerH2OState,
    attention_mass: torch.Tensor,
    absolute_position: int,
) -> LayerH2OState:
    """Commit one token's attention mass without pruning yet.

    This is the primitive used by H2O/SpecDec v2.  Speculative proposal /
    verification rows are captured separately.  Only rows corresponding to
    tokens that actually commit are passed here.

    The caller may append several committed tokens from one speculative round
    and then call :func:`prune_committed` once at the round boundary.
    """
    if not isinstance(attention_mass, torch.Tensor) or attention_mass.ndim != 1:
        raise ValueError("attention_mass must be a rank-1 tensor")
    if (not isinstance(absolute_position, int)
            or isinstance(absolute_position, bool)
            or absolute_position < 0):
        raise ValueError("absolute_position must be a non-negative int")

    prior = int(state.scores.numel())
    if int(state.positions.numel()) != prior:
        raise RuntimeError("H2O state positions/scores length mismatch")
    if int(attention_mass.numel()) != prior + 1:
        raise RuntimeError(
            f"H2O committed attention length {attention_mass.numel()} "
            f"!= expected {prior + 1}"
        )
    if state.positions.numel():
        if absolute_position <= int(state.positions[-1]):
            raise ValueError(
                "absolute_position must increase monotonically")

    device = state.scores.device
    mass = attention_mass.to(
        device=device,
        dtype=torch.float32,
    )

    state.scores = torch.cat([
        state.scores,
        torch.zeros(
            1,
            device=device,
            dtype=torch.float32,
        ),
    ]) + mass

    state.positions = torch.cat([
        state.positions,
        torch.tensor(
            [absolute_position],
            device=state.positions.device,
            dtype=torch.long,
        ),
    ])

    return state


def prune_committed(
    layer: dict,
    state: LayerH2OState,
    config: H2OConfig,
) -> tuple[dict, LayerH2OState, torch.Tensor | None]:
    """Prune an already-committed layer/state pair at a safe boundary.

    No speculative token may call this function.  The resident cache and H2O
    state must already describe exactly the same committed token set.
    """
    paged_cache.check_consistent(layer)
    if layer["fmt"] not in ("fp16", "v_fp8"):
        raise ValueError(
            "H2O committed pruning supports fp16/base or v_fp8 paged cache only")

    resident = paged_cache.logical_length(layer)
    if resident != int(state.scores.numel()):
        raise RuntimeError(
            f"H2O committed state/cache mismatch: resident={resident}, "
            f"scores={state.scores.numel()}"
        )
    if resident != int(state.positions.numel()):
        raise RuntimeError(
            "H2O committed positions/cache length mismatch")

    keep = keep_indices(state.scores, config)
    if int(keep.numel()) == resident:
        return layer, state, None

    out = paged_cache.select_indices(layer, keep)
    evicted = resident - int(keep.numel())

    state.scores = state.scores.index_select(0, keep)
    state.positions = state.positions.index_select(0, keep)
    state.evictions += 1
    state.tokens_evicted += evicted

    if state.positions.numel() > 1:
        if not bool(torch.all(
                state.positions[1:] > state.positions[:-1]).item()):
            raise RuntimeError(
                "H2O retained absolute positions are not increasing")

    return out, state, keep


def validate_committed_alignment(
    cache_list: list[dict],
    states: list[LayerH2OState],
    *,
    seen_tokens: int | None = None,
) -> None:
    """Validate resident cache/state alignment across decoder layers."""
    if len(cache_list) != len(states):
        raise RuntimeError(
            "H2O cache/state layer count mismatch")

    for index, (layer, state) in enumerate(zip(cache_list, states)):
        paged_cache.check_consistent(layer)
        resident = paged_cache.logical_length(layer)

        if resident != int(state.scores.numel()):
            raise RuntimeError(
                f"layer {index}: resident/scores mismatch "
                f"{resident} != {state.scores.numel()}"
            )
        if resident != int(state.positions.numel()):
            raise RuntimeError(
                f"layer {index}: resident/positions mismatch "
                f"{resident} != {state.positions.numel()}"
            )

        if state.positions.numel() > 1:
            if not bool(torch.all(
                    state.positions[1:] > state.positions[:-1]).item()):
                raise RuntimeError(
                    f"layer {index}: absolute positions not increasing")

        if seen_tokens is not None and state.positions.numel():
            if int(state.positions[-1]) >= seen_tokens:
                raise RuntimeError(
                    f"layer {index}: retained position "
                    f"{int(state.positions[-1])} >= seen_tokens {seen_tokens}"
                )
