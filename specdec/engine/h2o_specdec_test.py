"""H2O v2 SpecDec committed-history correctness gate."""
from __future__ import annotations

import gc

import torch

from . import paged_cache, spec_cache, spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .h2o_policy import validate_committed_alignment
from .spec_dec_h2o import spec_decode_greedy_h2o


PROMPT = "The capital of France is"


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu().clone()


def _raw_snapshot(cache) -> list[dict]:
    out = []
    for layer in cache:
        current = (
            paged_cache.materialize_layer(layer)
            if paged_cache.is_paged_layer(layer)
            else layer
        )
        out.append({
            key: _cpu(value)
            for key, value in current.items()
            if isinstance(value, torch.Tensor)
        })
    return out


def _assert_cache_equal(label: str, left, right) -> None:
    a = _raw_snapshot(left)
    b = _raw_snapshot(right)
    assert len(a) == len(b), f"{label}: layer count differs"

    for li, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys(), (
            f"{label}: layer {li} keys differ")
        for key in x:
            assert x[key].dtype == y[key].dtype, (
                f"{label}: layer {li} {key} dtype differs")
            assert torch.equal(x[key], y[key]), (
                f"{label}: layer {li} {key} tensor differs")


def _load_pair():
    print(f"[load] target {MODEL_TARGET}", flush=True)
    target, tok = load_ours(MODEL_TARGET)
    print(f"[load] draft {MODEL_SMALL}", flush=True)
    draft, _ = load_ours(MODEL_SMALL)
    return target, draft, tok


def _configure_pair(target, draft, *, page_size: int) -> None:
    for model in (target, draft):
        model.set_kv_format("fp16")
        model.set_cache_backend(
            "paged_v0",
            page_size=page_size,
        )
        model.set_paged_attention_mode(
            "direct_cuda")


def _no_prune_parity(target, draft, tok) -> None:
    ids = tok(
        PROMPT,
        return_tensors="pt",
    ).input_ids.cuda()

    page_size = 4
    max_new = 12

    _configure_pair(
        target, draft, page_size=page_size)

    baseline = spec_dec.spec_decode_greedy(
        target,
        draft,
        ids,
        max_new,
        k=4,
        eos_id=tok.eos_token_id,
        target_kv="base_bf16",
        draft_kv="base_bf16",
        return_caches=True,
    )

    # Budget exceeds prompt + generated sequence, therefore H2O must never
    # prune. Score capture/state bookkeeping must be semantically invisible.
    h2o = spec_decode_greedy_h2o(
        target,
        draft,
        ids,
        max_new,
        hh_size=1024,
        recent_size=0,
        k=4,
        eos_id=tok.eos_token_id,
        page_size=page_size,
        attention_mode="direct_cuda",
        return_caches=True,
    )

    assert baseline["ids"] == h2o["ids"], (
        "no-prune generated IDs differ")
    assert baseline["gen"] == h2o["gen"], (
        "no-prune generated suffix differs")
    assert baseline["stats"] == h2o["stats"], (
        "no-prune SpecDec stats differ")

    _assert_cache_equal(
        "target no-prune cache",
        baseline["target_cache"],
        h2o["target_cache"],
    )
    _assert_cache_equal(
        "draft no-prune cache",
        baseline["draft_cache"],
        h2o["draft_cache"],
    )

    hs = h2o["h2o_stats"]
    assert hs.prune_rounds == 0
    assert hs.target_layer_token_evictions == 0
    assert hs.draft_layer_token_evictions == 0

    print(
        "[no-prune] standard SpecDec == H2O capture/state integration: PASS "
        f"(rounds={h2o['stats']['rounds']}, "
        f"proposed={h2o['stats']['proposed']}, "
        f"accepted={h2o['stats']['accepted']}, "
        f"mismatch={h2o['stats']['mismatch_rounds']})"
    )


def _active_prune(target, draft, tok) -> None:
    # Long enough prompt to force real resident/logical separation.
    base = (
        "Speculative decoding verifies draft tokens with a target model. "
        "Only committed history may be retained or evicted. "
    )
    text = base
    while True:
        ids = tok(
            text,
            return_tensors="pt",
        ).input_ids
        if ids.shape[1] >= 64:
            ids = ids[:, :64].cuda()
            break
        text += base

    page_size = 8
    budget = 32
    max_new = 16

    _configure_pair(
        target, draft, page_size=page_size)

    logs = []
    result = spec_decode_greedy_h2o(
        target,
        draft,
        ids,
        max_new,
        hh_size=budget // 2,
        recent_size=budget - budget // 2,
        k=4,
        eos_id=None,
        page_size=page_size,
        attention_mode="direct_cuda",
        log=logs.append,
        return_caches=True,
    )

    hs = result["h2o_stats"]
    st = result["stats"]

    assert hs.prune_rounds > 0
    assert result["seen_tokens"] == (
        int(ids.shape[1]) + len(result["gen"])
    )
    assert hs.final_target_resident <= budget
    assert hs.final_draft_resident <= budget
    assert hs.final_target_resident < result["seen_tokens"]
    assert hs.final_draft_resident < result["seen_tokens"]

    validate_committed_alignment(
        result["target_cache"],
        result["target_states"],
        seen_tokens=result["seen_tokens"],
    )
    validate_committed_alignment(
        result["draft_cache"],
        result["draft_states"],
        seen_tokens=result["seen_tokens"],
    )

    # The accounting is the executable proof of the state boundary:
    # rejected speculative score rows were observed, but never committed.
    assert hs.committed_proposal_rows == st["accepted"]
    assert hs.committed_verify_rows == st["accepted"]
    assert hs.discarded_proposal_rows == st["proposed"] - st["accepted"]
    assert hs.discarded_verify_rows == st["proposed"] - st["accepted"]
    assert hs.committed_repair_rows == 2 * st["mismatch_rounds"]

    # Every post-prune round must continue using absolute logical positions;
    # resident cache length is intentionally smaller than `seen`.
    assert logs
    separated = [
        row for row in logs
        if row["target_resident"] < row["seen"]
        and row["draft_resident"] < row["seen"]
    ]
    assert separated, (
        "test never reached logical_seen > resident_length")

    print(
        "[active-prune] committed-history isolation + budget + "
        "absolute-position separation: PASS "
        f"(seen={result['seen_tokens']}, "
        f"target_resident={hs.final_target_resident}, "
        f"draft_resident={hs.final_draft_resident}, "
        f"prune_rounds={hs.prune_rounds}, "
        f"proposed={st['proposed']}, accepted={st['accepted']}, "
        f"discarded={hs.discarded_proposal_rows}, "
        f"mismatch={st['mismatch_rounds']})"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v2 SpecDec committed-history gate")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    torch.manual_seed(0)
    target, draft, tok = _load_pair()

    _no_prune_parity(
        target, draft, tok)
    _active_prune(
        target, draft, tok)

    print("\nh2o_specdec_test: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
