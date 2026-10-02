"""GPU parity gates for the opt-in paged KV cache v0 backend.

This is a correctness harness, not a performance benchmark.  It requires CUDA
and the Qwen 0.5B/1.5B checkpoints used by the existing cache tests.

Run:
  python -m specdec.engine.paged_cache_parity --mode all --max-new-tokens 8
"""
from __future__ import annotations

import argparse
import gc

import torch

from . import paged_cache, spec_cache, spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .model_self import dequantize_kv_fp8


PROMPT = "The capital of France is"
STAT_KEYS = ("rounds", "proposed", "accepted", "mismatch_rounds", "rollbacks")


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu().clone()


def _raw_snapshot(cache) -> list[dict]:
    out = []
    for layer in cache:
        current = (paged_cache.materialize_layer(layer)
                   if paged_cache.is_paged_layer(layer) else layer)
        out.append({key: _cpu_tensor(value) for key, value in current.items()
                    if isinstance(value, torch.Tensor)})
    return out


def _attention_snapshot(raw: list[dict], fmt: str) -> list[dict]:
    out = []
    for layer in raw:
        if fmt == "v_fp8":
            out.append({"k": layer["k"],
                        "v": dequantize_kv_fp8(
                            layer["v"], layer["v_scale"], torch.bfloat16)})
        else:
            out.append({"k": layer["k"], "v": layer["v"]})
    return out


def _assert_tensor_dicts_equal(label: str, left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right), f"{label}: layer count differs"
    for index, (a, b) in enumerate(zip(left, right)):
        assert a.keys() == b.keys(), f"{label}: layer {index} keys {a.keys()} != {b.keys()}"
        for key in a:
            assert a[key].dtype == b[key].dtype, (
                f"{label}: layer {index} {key} dtype {a[key].dtype} != {b[key].dtype}")
            assert torch.equal(a[key], b[key]), f"{label}: layer {index} {key} differs"


def _greedy_run(model, ids, *, fmt: str, backend: str, page_size: int,
                max_new_tokens: int, eos_id: int | None):
    model.set_kv_format(fmt)
    model.set_cache_backend(backend, page_size=page_size)
    generated = ids.clone()
    logits_trace = []
    with torch.inference_mode():
        logits, cache = model._run(model.embed_tokens(ids), None, need_logits=True)
        logits_trace.append(_cpu_tensor(logits))
        for _ in range(max_new_tokens):
            token = logits[:, -1].argmax(dim=-1)
            if eos_id is not None and bool((token == eos_id).all()):
                break
            generated = torch.cat([generated, token[:, None]], dim=1)
            logits, cache = model._run(model.embed_tokens(token[:, None]), cache,
                                       need_logits=True)
            logits_trace.append(_cpu_tensor(logits))
    spec_cache.check_consistent(cache)
    return {
        "ids": generated.cpu(),
        "length": spec_cache.cache_length(cache),
        "raw": _raw_snapshot(cache),
        "logits": logits_trace,
    }


def _run_model_gate(model_name: str, fmt: str, page_size: int,
                    max_new_tokens: int) -> None:
    model, tok = load_ours(model_name)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    contiguous = _greedy_run(
        model, ids, fmt=fmt, backend="contiguous", page_size=page_size,
        max_new_tokens=max_new_tokens, eos_id=tok.eos_token_id)
    paged = _greedy_run(
        model, ids, fmt=fmt, backend="paged_v0", page_size=page_size,
        max_new_tokens=max_new_tokens, eos_id=tok.eos_token_id)

    assert contiguous["length"] == paged["length"], "cache lengths differ"
    assert torch.equal(contiguous["ids"], paged["ids"]), "generated IDs differ"
    assert len(contiguous["logits"]) == len(paged["logits"]), "logit call counts differ"
    for index, (a, b) in enumerate(zip(contiguous["logits"], paged["logits"])):
        assert torch.equal(a, b), f"logits differ at call {index}"
    _assert_tensor_dicts_equal("raw cache", contiguous["raw"], paged["raw"])
    _assert_tensor_dicts_equal(
        "attention materialization",
        _attention_snapshot(contiguous["raw"], fmt),
        _attention_snapshot(paged["raw"], fmt))
    print(f"[{model_name} {fmt}] contiguous == paged_v0: PASS "
          f"(cache_len={paged['length']}, generated={paged['ids'].shape[1] - ids.shape[1]})")
    del model, tok, ids, contiguous, paged
    gc.collect()
    torch.cuda.empty_cache()


def _specdec_gate(page_size: int, max_new_tokens: int) -> None:
    target, tok = load_ours(MODEL_TARGET)
    draft, _ = load_ours(MODEL_SMALL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    def run(backend: str):
        target.set_cache_backend(backend, page_size=page_size)
        draft.set_cache_backend(backend, page_size=page_size)
        return spec_dec.spec_decode_greedy(
            target, draft, ids, max_new_tokens, k=4, eos_id=tok.eos_token_id,
            target_kv="v_fp8", draft_kv="base_bf16")

    contiguous = run("contiguous")
    paged = run("paged_v0")
    assert [int(x) for x in contiguous["ids"]] == [int(x) for x in paged["ids"]], (
        "SpecDec generated IDs differ")
    for key in STAT_KEYS:
        assert contiguous["stats"][key] == paged["stats"][key], (
            f"SpecDec {key} differs: {contiguous['stats'][key]} != {paged['stats'][key]}")
    summary = {key: paged["stats"][key] for key in STAT_KEYS}
    print(f"[SpecDec k=4] contiguous == paged_v0: PASS ({summary})")
    del target, draft, tok, ids, contiguous, paged
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "base", "v_fp8", "specdec"),
                        default="all")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for paged_cache_parity")
    torch.manual_seed(0)
    if args.mode in ("all", "base"):
        _run_model_gate(MODEL_SMALL, "fp16", args.page_size, args.max_new_tokens)
    if args.mode in ("all", "v_fp8"):
        _run_model_gate(MODEL_TARGET, "v_fp8", args.page_size, args.max_new_tokens)
    if args.mode in ("all", "specdec"):
        _specdec_gate(args.page_size, args.max_new_tokens)
    print("paged_cache_parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
