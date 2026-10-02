"""H2O v3 canonical V-FP8 target + BF16 draft correctness gate."""
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
        raw = paged_cache.materialize_layer(layer)
        out.append({
            key: _cpu(value)
            for key, value in raw.items()
            if isinstance(value, torch.Tensor)
        })
    return out


def _assert_cache_equal(label: str, left, right) -> None:
    a = _raw_snapshot(left)
    b = _raw_snapshot(right)
    assert len(a) == len(b), f"{label}: layer count differs"
    for li, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys(), f"{label}: layer {li} keys differ"
        for key in x:
            assert x[key].dtype == y[key].dtype, (
                f"{label}: layer {li} {key} dtype differs")
            assert torch.equal(x[key], y[key]), (
                f"{label}: layer {li} {key} tensor differs")


def _configure(target, draft, page_size: int) -> None:
    target.set_kv_format("v_fp8")
    target.set_cache_backend("paged_v0", page_size=page_size)
    target.set_paged_attention_mode("direct_cuda")

    draft.set_kv_format("fp16")
    draft.set_cache_backend("paged_v0", page_size=page_size)
    draft.set_paged_attention_mode("direct_cuda")


def _no_prune_canonical(target, draft, tok) -> None:
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    page_size = 4
    max_new = 12

    _configure(target, draft, page_size)

    baseline = spec_dec.spec_decode_greedy(
        target,
        draft,
        ids,
        max_new,
        k=4,
        eos_id=tok.eos_token_id,
        target_kv="v_fp8",
        draft_kv="base_bf16",
        return_caches=True,
    )

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
        target_kv="v_fp8",
        draft_kv="base_bf16",
        return_caches=True,
    )

    assert baseline["ids"] == h2o["ids"]
    assert baseline["gen"] == h2o["gen"]
    assert baseline["stats"] == h2o["stats"]

    _assert_cache_equal(
        "canonical target no-prune",
        baseline["target_cache"],
        h2o["target_cache"],
    )
    _assert_cache_equal(
        "canonical draft no-prune",
        baseline["draft_cache"],
        h2o["draft_cache"],
    )

    assert all(layer["fmt"] == "v_fp8" for layer in h2o["target_cache"])
    assert all(layer["fmt"] == "fp16" for layer in h2o["draft_cache"])
    assert h2o["h2o_stats"].prune_rounds == 0

    print(
        "[canonical/no-prune] V-FP8 target + BF16 draft exact parity: PASS "
        f"(rounds={h2o['stats']['rounds']}, "
        f"proposed={h2o['stats']['proposed']}, "
        f"accepted={h2o['stats']['accepted']})"
    )


def _active_prune_canonical(target, draft, tok) -> None:
    base = (
        "Speculative decoding commits only verified history. "
        "V-FP8 compresses values while H2O controls which committed tokens "
        "remain resident in the KV cache. "
    )
    text = base
    while True:
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] >= 64:
            ids = ids[:, :64].cuda()
            break
        text += base

    page_size = 8
    budget = 32
    max_new = 16

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
        target_kv="v_fp8",
        draft_kv="base_bf16",
        return_caches=True,
    )

    hs = result["h2o_stats"]
    st = result["stats"]

    assert hs.prune_rounds > 0
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

    # V-FP8 target must retain V scale tensors exactly in token alignment.
    for layer in result["target_cache"]:
        paged_cache.check_consistent(layer)
        assert layer["fmt"] == "v_fp8"
        raw = paged_cache.materialize_layer(layer)
        assert raw["v"].shape[:3] == raw["v_scale"].shape[:3]

    assert hs.committed_proposal_rows == st["accepted"]
    assert hs.committed_verify_rows == st["accepted"]
    assert hs.discarded_proposal_rows == st["proposed"] - st["accepted"]
    assert hs.discarded_verify_rows == st["proposed"] - st["accepted"]
    assert hs.committed_repair_rows == 2 * st["mismatch_rounds"]

    target_payload = sum(
        paged_cache.payload_bytes(layer)
        for layer in result["target_cache"]
    )
    draft_payload = sum(
        paged_cache.payload_bytes(layer)
        for layer in result["draft_cache"]
    )

    print(
        "[canonical/active-prune] V-FP8 target + H2O committed pruning: PASS "
        f"(seen={result['seen_tokens']}, "
        f"resident(T/D)={hs.final_target_resident}/"
        f"{hs.final_draft_resident}, "
        f"prune_rounds={hs.prune_rounds}, "
        f"mismatch={st['mismatch_rounds']}, "
        f"target_payload={target_payload/2**20:.2f} MiB, "
        f"draft_payload={draft_payload/2**20:.2f} MiB)"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v3 canonical V-FP8 + SpecDec gate")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    torch.manual_seed(0)

    print(f"[load] target {MODEL_TARGET}")
    target, tok = load_ours(MODEL_TARGET)
    print(f"[load] draft {MODEL_SMALL}")
    draft, _ = load_ours(MODEL_SMALL)

    _no_prune_canonical(target, draft, tok)
    _active_prune_canonical(target, draft, tok)

    print("\nh2o_v3_combo_test: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
