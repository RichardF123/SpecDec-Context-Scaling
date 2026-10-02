"""H2O v0 correctness gates."""
from __future__ import annotations

import gc
import torch
from . import paged_cache
from .g4_b4_cache_tests import MODEL_SMALL, load_ours
from .h2o_generate import h2o_generate_greedy
from .h2o_policy import H2OConfig, keep_indices


def _policy_gate() -> None:
    scores = torch.tensor(
        [0.1, 9.0, 0.2, 8.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        dtype=torch.float32,
    )
    keep = keep_indices(scores, H2OConfig(hh_size=2, recent_size=3))
    expected = torch.tensor([1, 3, 7, 8, 9], dtype=torch.long)
    assert torch.equal(keep.cpu(), expected)
    print("[policy] heavy=2 + recent=3 selection: PASS")


def _cache_selection_gate() -> None:
    layer = paged_cache.create_layer_cache("fp16", page_size=4)
    k = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
    v = (100 + torch.arange(10, dtype=torch.float32)).view(1, 1, 10, 1)
    paged_cache.append_current(layer, k, v)
    idx = torch.tensor([1, 3, 7, 8, 9], dtype=torch.long)
    selected = paged_cache.select_indices(layer, idx)
    raw = paged_cache.materialize_layer(selected)
    assert torch.equal(raw["k"].reshape(-1), torch.tensor([1., 3., 7., 8., 9.]))
    assert torch.equal(raw["v"].reshape(-1), torch.tensor([101., 103., 107., 108., 109.]))
    assert paged_cache.logical_length(selected) == 5
    print("[cache] select_indices exact K/V retention: PASS")


def _real_model_gate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-model H2O gate")
    torch.manual_seed(0)
    model, tok = load_ours(MODEL_SMALL)
    text = (
        "Paged attention stores key value states in fixed pages while "
        "heavy hitter retention keeps important and recent context. "
    ) * 8
    ids = tok(text, return_tensors="pt").input_ids[:, :48].cuda()
    assert ids.shape[1] >= 24

    ref = h2o_generate_greedy(
        model, ids, 8, hh_size=256, recent_size=0, page_size=16,
        eos_id=None, attention_mode="direct_cuda")
    again = h2o_generate_greedy(
        model, ids, 8, hh_size=256, recent_size=0, page_size=16,
        eos_id=None, attention_mode="direct_cuda")
    assert torch.equal(ref["ids"], again["ids"])
    assert ref["stats"].prune_events == 0
    assert again["stats"].prune_events == 0
    print(
        f"[real/no-evict] deterministic parity: PASS "
        f"(seen={ref['stats'].seen_tokens}, resident={ref['stats'].final_resident})"
    )

    bounded = h2o_generate_greedy(
        model, ids, 12, hh_size=8, recent_size=8, page_size=16,
        eos_id=None, attention_mode="direct_cuda")
    st = bounded["stats"]
    assert st.prune_events > 0
    assert st.final_resident <= 16
    assert st.seen_tokens > st.final_resident
    assert all(state.positions.numel() == st.final_resident
               for state in bounded["states"])
    assert all(
        bool(torch.all(state.positions[1:] > state.positions[:-1]).item())
        for state in bounded["states"]
        if state.positions.numel() > 1
    )
    latest = st.seen_tokens - 1
    for state in bounded["states"]:
        assert int(state.positions[-1]) == latest

    print(
        f"[real/evict] H2O budget enforcement: PASS "
        f"(budget=16 seen={st.seen_tokens} resident={st.final_resident} "
        f"prune_events={st.prune_events} payload={bounded['payload_bytes']/2**20:.2f} MiB)"
    )

    del model, tok, ref, again, bounded
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    print("H2O v0: global-token GQA adaptation; correctness/controller prototype")
    _policy_gate()
    _cache_selection_gate()
    _real_model_gate()
    print("\nh2o_v0_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
