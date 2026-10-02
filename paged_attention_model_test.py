"""PagedAttention v1b real-model integration gate.

This gate compares two paths with identical CUDA attention math:

1. materialized_cuda:
   paged KV -> materialize contiguous K/V -> existing CUDA attention

2. direct_cuda:
   paged KV -> page-aware CUDA attention directly

Only the K/V addressing method differs.  The default engine behavior remains
paged_attention_mode="off".

Run:
    python -m specdec.engine.paged_attention_model_test
"""
from __future__ import annotations

import gc

import torch

from . import paged_cache, spec_cache
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours


PROMPT = "The capital of France is"


def _cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().contiguous().cpu().clone()


def _cache_snapshot(cache) -> list[dict]:
    out = []
    for layer in cache:
        mat = paged_cache.materialize_layer(layer)
        out.append({
            key: _cpu(value)
            for key, value in mat.items()
            if isinstance(value, torch.Tensor)
        })
    return out


def _run(model, tok, *, mode: str, page_size: int, max_new_tokens: int) -> dict:
    model.set_kv_format("fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(mode)

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    assert ids.shape[1] > 1, "tokenizer produced an empty/degenerate prompt"

    generated = ids.clone()
    logits_trace = []

    # Prompt prefill intentionally uses the existing paged_v0 path in both modes.
    with torch.inference_mode():
        logits, cache = model._run(model.embed_tokens(ids), None, need_logits=True)
    logits_trace.append(_cpu(logits))

    # Positive proof for the direct path: after prefill, any attempt to
    # materialize full-history K/V during decode fails immediately.
    original_materialize = paged_cache.materialize_layer
    if mode == "direct_cuda":
        def forbidden_materialize(*args, **kwargs):
            raise AssertionError(
                "direct_cuda decode unexpectedly materialized full K/V")
        paged_cache.materialize_layer = forbidden_materialize

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                token = logits[:, -1].argmax(dim=-1)
                generated = torch.cat([generated, token[:, None]], dim=1)
                logits, cache = model._run(
                    model.embed_tokens(token[:, None]),
                    cache,
                    need_logits=True,
                )
                logits_trace.append(_cpu(logits))
    finally:
        paged_cache.materialize_layer = original_materialize

    spec_cache.check_consistent(cache)
    snapshot = _cache_snapshot(cache)

    return {
        "ids": generated.cpu(),
        "logits": logits_trace,
        "cache": snapshot,
        "length": spec_cache.cache_length(cache),
    }


def _assert_tensor_dicts_equal(label: str, left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right), f"{label}: layer count differs"
    for layer_index, (a, b) in enumerate(zip(left, right)):
        assert a.keys() == b.keys(), (
            f"{label}: layer {layer_index} keys differ: {a.keys()} != {b.keys()}")
        for key in a:
            assert a[key].dtype == b[key].dtype, (
                f"{label}: layer {layer_index} {key} dtype differs: "
                f"{a[key].dtype} != {b[key].dtype}")
            assert torch.equal(a[key], b[key]), (
                f"{label}: layer {layer_index} {key} differs")


def _model_gate(model_name: str, *, page_size: int = 4,
                max_new_tokens: int = 8) -> None:
    print(f"\n[load] {model_name}")
    model, tok = load_ours(model_name)

    reference = _run(
        model,
        tok,
        mode="materialized_cuda",
        page_size=page_size,
        max_new_tokens=max_new_tokens,
    )
    direct = _run(
        model,
        tok,
        mode="direct_cuda",
        page_size=page_size,
        max_new_tokens=max_new_tokens,
    )

    assert reference["length"] == direct["length"], "cache lengths differ"
    assert torch.equal(reference["ids"], direct["ids"]), "generated IDs differ"
    assert len(reference["logits"]) == len(direct["logits"]), "logit call counts differ"

    for index, (a, b) in enumerate(zip(reference["logits"], direct["logits"])):
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            raise AssertionError(
                f"logits differ at call {index}: max_abs={float(diff.max())}")

    _assert_tensor_dicts_equal("cache", reference["cache"], direct["cache"])

    print(
        f"[{model_name}] materialized_cuda == direct_cuda: PASS "
        f"(cache_len={direct['length']}, generated={max_new_tokens})"
    )

    del model, tok, reference, direct
    gc.collect()
    torch.cuda.empty_cache()


def _mode_api_gate() -> None:
    from .model_self import QwenLM

    model = QwenLM(
        vocab_size=32,
        hidden=32,
        heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate=64,
        num_layers=1,
    )
    assert model.paged_attention_mode == "off"
    for mode in ("off", "materialized_cuda", "direct_cuda"):
        model.set_paged_attention_mode(mode)
        assert model.paged_attention_mode == mode
        assert model.layers[0].self_attn.paged_attention_mode == mode
    try:
        model.set_paged_attention_mode("bad")
        raise AssertionError("invalid paged attention mode should fail")
    except ValueError:
        pass
    print("[mode API] default/off + opt-in modes: PASS")


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    _mode_api_gate()

    # Draft shape: 14 Q heads / 2 KV heads / D=64.
    _model_gate(MODEL_SMALL, page_size=4, max_new_tokens=8)

    # Target shape: 12 Q heads / 2 KV heads / D=128.
    _model_gate(MODEL_TARGET, page_size=4, max_new_tokens=8)

    print("\npaged_attention_model_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
