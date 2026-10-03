"""PagedAttention final SpecDec direct-route parity gate.

Scope:
- paged_v0 storage
- direct page-aware CUDA for BF16 and asymmetric V-FP8, q_len=1..4
- phases: decode / propose / verify / repair / other are eligible
- prefill remains on the existing materialized path

Two parity cases are exercised:
1. canonical: target=v_fp8, draft=base_bf16
   -> direct path covers target verification and eligible draft calls.
2. all_bf16: target=base_bf16, draft=base_bf16
   -> direct path covers eligible target and draft calls.

The comparison baseline is materialized_cuda, so attention math is identical;
the intended variable is only paged K/V addressing.

Run:
    python -m specdec.engine.paged_attention_specdec_test
"""
from __future__ import annotations

import gc

import torch

from . import paged_cache, spec_cache, spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .model_self import (
    paged_attention_stats,
    reset_paged_attention_stats,
)

PROMPT = "The capital of France is"


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu().clone()


def _raw_snapshot(cache) -> list[dict]:
    spec_cache.check_consistent(cache)
    out = []
    for layer in cache:
        current = (
            paged_cache.materialize_layer(layer)
            if paged_cache.is_paged_layer(layer)
            else layer
        )
        out.append({
            key: _cpu_tensor(value)
            for key, value in current.items()
            if isinstance(value, torch.Tensor)
        })
    return out


def _assert_tensor_dicts_equal(label: str, left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right), f"{label}: layer count differs"
    for index, (a, b) in enumerate(zip(left, right)):
        assert a.keys() == b.keys(), (
            f"{label}: layer {index} keys {a.keys()} != {b.keys()}"
        )
        for key in a:
            assert a[key].dtype == b[key].dtype, (
                f"{label}: layer {index} {key} dtype differs: "
                f"{a[key].dtype} != {b[key].dtype}"
            )
            assert torch.equal(a[key], b[key]), (
                f"{label}: layer {index} {key} differs"
            )


def _ids_as_ints(values) -> list[int]:
    return [int(x) for x in values]


def _run_case(
    target,
    draft,
    ids,
    *,
    mode: str,
    page_size: int,
    max_new_tokens: int,
    eos_id: int | None,
    target_kv: str,
    draft_kv: str,
):
    target.set_cache_backend("paged_v0", page_size=page_size)
    draft.set_cache_backend("paged_v0", page_size=page_size)
    target.set_paged_attention_mode(mode)
    draft.set_paged_attention_mode(mode)

    reset_paged_attention_stats()

    result = spec_dec.spec_decode_greedy(
        target,
        draft,
        ids,
        max_new_tokens,
        k=4,
        eos_id=eos_id,
        target_kv=target_kv,
        draft_kv=draft_kv,
        return_caches=True,
    )

    route_stats = paged_attention_stats()

    target_cache = result.pop("target_cache")
    draft_cache = result.pop("draft_cache")

    target_len = spec_cache.cache_length(target_cache)
    draft_len = spec_cache.cache_length(draft_cache)
    target_raw = _raw_snapshot(target_cache)
    draft_raw = _raw_snapshot(draft_cache)

    del target_cache, draft_cache

    return {
        "ids": _ids_as_ints(result["ids"]),
        "gen": list(result["gen"]),
        "stats": dict(result["stats"]),
        "route_stats": route_stats,
        "target_len": target_len,
        "draft_len": draft_len,
        "target_raw": target_raw,
        "draft_raw": draft_raw,
    }


def _assert_route_contract(label: str, direct: dict) -> None:
    route = direct["route_stats"]
    dphase = route["direct_by_phase"]
    fphase = route["fallback_by_phase"]

    assert route["direct_cuda_calls"] > 0, (
        f"{label}: direct_cuda was never exercised"
    )
    assert dphase.get("propose", 0) > 0, (
        f"{label}: propose did not use direct paged attention"
    )
    # FINAL_PAGED_V2_Q1_Q4_VFP8: verify q_len=1..4 is now direct for BF16 and V-FP8.
    assert dphase.get("verify", 0) > 0, (
        f"{label}: verify did not use final direct paged attention"
    )
    assert dphase.get("prefill", 0) == 0, (
        f"{label}: prefill must remain outside the direct paged path"
    )
    assert fphase.get("verify", 0) == 0, (
        f"{label}: verify unexpectedly fell back to materialization"
    )
    assert fphase.get("prefill", 0) > 0, (
        f"{label}: prefill materialized/reference path was not observed"
    )

    if direct["stats"]["mismatch_rounds"] > 0:
        assert dphase.get("repair", 0) > 0, (
            f"{label}: mismatch occurred but repair did not exercise direct path"
        )


def _compare(label: str, baseline: dict, direct: dict) -> None:
    assert baseline["ids"] == direct["ids"], (
        f"{label}: generated IDs differ"
    )
    assert baseline["gen"] == direct["gen"], (
        f"{label}: generated suffix differs"
    )
    assert baseline["stats"] == direct["stats"], (
        f"{label}: SpecDec stats differ:\n"
        f"baseline={baseline['stats']}\n"
        f"direct={direct['stats']}"
    )
    assert baseline["target_len"] == direct["target_len"], (
        f"{label}: target cache lengths differ"
    )
    assert baseline["draft_len"] == direct["draft_len"], (
        f"{label}: draft cache lengths differ"
    )
    _assert_tensor_dicts_equal(
        f"{label} target cache",
        baseline["target_raw"],
        direct["target_raw"],
    )
    _assert_tensor_dicts_equal(
        f"{label} draft cache",
        baseline["draft_raw"],
        direct["draft_raw"],
    )
    _assert_route_contract(label, direct)

    b_route = baseline["route_stats"]
    assert b_route["direct_cuda_calls"] == 0, (
        f"{label}: materialized baseline unexpectedly used direct CUDA"
    )
    assert b_route["materialized_by_phase"].get("propose", 0) > 0, (
        f"{label}: materialized baseline did not exercise q_len=1 CUDA reference"
    )

    summary = {
        key: direct["stats"][key]
        for key in (
            "rounds",
            "proposed",
            "accepted",
            "mismatch_rounds",
            "rollbacks",
        )
    }
    print(
        f"[{label}] materialized_cuda == direct_cuda: PASS "
        f"(target_len={direct['target_len']}, "
        f"draft_len={direct['draft_len']}, "
        f"specdec={summary}, "
        f"direct_by_phase={direct['route_stats']['direct_by_phase']}, "
        f"fallback_by_phase={direct['route_stats']['fallback_by_phase']})"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    torch.manual_seed(0)

    print(f"\n[load] target {MODEL_TARGET}")
    target, tok = load_ours(MODEL_TARGET)

    print(f"\n[load] draft {MODEL_SMALL}")
    draft, _ = load_ours(MODEL_SMALL)

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    assert ids.shape[1] > 1

    cases = (
        ("canonical-vfp8-target", "v_fp8", "base_bf16"),
        ("all-bf16", "base_bf16", "base_bf16"),
    )

    for label, target_kv, draft_kv in cases:
        print(
            f"\n[case] {label}: "
            f"target={target_kv}, draft={draft_kv}"
        )
        baseline = _run_case(
            target,
            draft,
            ids,
            mode="materialized_cuda",
            page_size=4,
            max_new_tokens=12,
            eos_id=tok.eos_token_id,
            target_kv=target_kv,
            draft_kv=draft_kv,
        )
        direct = _run_case(
            target,
            draft,
            ids,
            mode="direct_cuda",
            page_size=4,
            max_new_tokens=12,
            eos_id=tok.eos_token_id,
            target_kv=target_kv,
            draft_kv=draft_kv,
        )
        _compare(label, baseline, direct)

        del baseline, direct
        gc.collect()
        torch.cuda.empty_cache()

    del target, draft, tok, ids
    gc.collect()
    torch.cuda.empty_cache()

    print("\npaged_attention_specdec_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
