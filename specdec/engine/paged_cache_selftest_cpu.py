"""CPU correctness gates for paged KV cache v0.

Run: python -m specdec.engine.paged_cache_selftest_cpu
"""
from __future__ import annotations

import copy

import torch

from . import paged_cache as pc
from . import spec_cache


BOUNDARIES = (15, 16, 17, 31, 32, 33)


def _payload(fmt: str, start: int, n: int):
    token = torch.arange(start, start + n, dtype=torch.float32).view(1, 1, n, 1)
    head = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1)
    dim = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    base = token * 32 + head * 8 + dim
    k = base.to(torch.bfloat16)
    if fmt == "fp16":
        return k, (base + 128).to(torch.bfloat16), None
    v = ((base.remainder(16) - 8) / 2).to(torch.float8_e4m3fn)
    scale = token + head * 100 + 1
    return k, v, scale


def _expected(fmt: str, n: int):
    k, v, scale = _payload(fmt, 0, n)
    out = {"fmt": fmt, "k": k, "v": v}
    if scale is not None:
        out["v_scale"] = scale
    return out


def _assert_materialized(layer: dict, expected: dict) -> None:
    got = pc.materialize_layer(layer)
    assert got["fmt"] == expected["fmt"]
    assert torch.equal(got["k"], expected["k"])
    assert torch.equal(got["v"], expected["v"])
    assert got["k"].shape[2] == got["v"].shape[2] == pc.logical_length(layer)
    if expected["fmt"] == "v_fp8":
        assert torch.equal(got["v_scale"], expected["v_scale"])
        assert got["v"].shape[2] == got["v_scale"].shape[2]
        assert got["v"].dtype == torch.float8_e4m3fn
        assert got["v_scale"].dtype == torch.float32


def _build(fmt: str, n: int, chunks=(7, 8, 1, 16, 1)) -> dict:
    layer = pc.create_layer_cache(fmt, page_size=16)
    done = 0
    for width in chunks:
        if done == n:
            break
        width = min(width, n - done)
        k, v, scale = _payload(fmt, done, width)
        pc.append_current(layer, k, v, scale)
        done += width
    if done < n:
        k, v, scale = _payload(fmt, done, n - done)
        pc.append_current(layer, k, v, scale)
    assert pc.logical_length(layer) == n
    pc.check_consistent(layer)
    return layer


def test_boundaries(fmt: str) -> None:
    for n in BOUNDARIES:
        layer = _build(fmt, n)
        assert len(layer["k_pages"]) == (n + 15) // 16
        _assert_materialized(layer, _expected(fmt, n))


def test_trim_and_cow(fmt: str) -> None:
    source = _build(fmt, 33)
    source_before = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in pc.materialize_layer(source).items()}
    for n in BOUNDARIES:
        trimmed = pc.trim_layer(source, n)
        _assert_materialized(trimmed, _expected(fmt, n))
        assert pc.logical_length(source) == 33

    partial = pc.trim_layer(source, 17)
    assert partial["k_pages"][0].data_ptr() == source["k_pages"][0].data_ptr()
    assert partial["k_pages"][1].data_ptr() != source["k_pages"][1].data_ptr()
    k, v, scale = _payload(fmt, 17, 9)
    pc.append_current(partial, k, v, scale)
    _assert_materialized(partial, _expected(fmt, 26))
    source_after = pc.materialize_layer(source)
    for key in source_before:
        if isinstance(source_before[key], torch.Tensor):
            assert torch.equal(source_before[key], source_after[key]), key

    exact = pc.trim_layer(source, 32)
    assert all(exact["k_pages"][i].data_ptr() == source["k_pages"][i].data_ptr()
               for i in range(2))
    k, v, scale = _payload(fmt, 32, 1)
    pc.append_current(exact, k, v, scale)
    assert exact["k_pages"][2].data_ptr() != source["k_pages"][2].data_ptr()
    _assert_materialized(exact, _expected(fmt, 33))

    for bad in (0, -1, 34, True, 1.5):
        try:
            pc.trim_layer(source, bad)
            raise AssertionError(f"trim {bad!r} should fail")
        except ValueError:
            pass


def test_spec_cache_api(fmt: str) -> None:
    cache = [_build(fmt, 33), _build(fmt, 33)]
    assert spec_cache.cache_length(cache) == 33
    assert spec_cache.cache_device(cache) == torch.device("cpu")
    spec_cache.check_consistent(cache)
    trimmed = spec_cache.trim_cache(cache, 17)
    assert spec_cache.cache_length(trimmed) == 17
    assert spec_cache.cache_length(cache) == 33
    k, v, scale = _payload(fmt, 17, 1)
    for layer in trimmed:
        pc.append_current(layer, k, v, scale)
    assert spec_cache.cache_length(trimmed) == 18
    assert spec_cache.cache_length(cache) == 33


def test_malformed_detection() -> None:
    valid = _build("fp16", 17)
    cases = []

    bad = copy.deepcopy(valid)
    bad["length"] = 33
    cases.append(bad)

    bad = copy.deepcopy(valid)
    bad["k_pages"][0] = bad["k_pages"][0][:, :, :-1]
    cases.append(bad)

    bad = copy.deepcopy(valid)
    bad["v_pages"][0] = bad["v_pages"][0].to(torch.float16)
    cases.append(bad)

    bad = _build("v_fp8", 17)
    bad["v_scale_pages"].pop()
    cases.append(bad)

    for index, malformed in enumerate(cases):
        try:
            pc.check_consistent(malformed)
            raise AssertionError(f"malformed case {index} was accepted")
        except RuntimeError:
            pass

    cross = [_build("fp16", 16), _build("fp16", 17)]
    try:
        spec_cache.check_consistent(cross)
        raise AssertionError("cross-layer mismatch was accepted")
    except RuntimeError:
        pass


def test_full_fp8_rejected() -> None:
    try:
        pc.create_layer_cache("fp8")
        raise AssertionError("paged_v0 + full fp8 should be rejected")
    except ValueError:
        pass
    layer = pc.create_layer_cache("fp16", page_size=7)
    k, v, _ = _payload("fp16", 0, 17)
    pc.append_current(layer, k, v)
    assert len(layer["k_pages"]) == 3
    _assert_materialized(layer, _expected("fp16", 17))
    for bad in (0, -1, True):
        try:
            pc.create_layer_cache("fp16", page_size=bad)
            raise AssertionError(f"page_size {bad!r} should fail")
        except ValueError:
            pass


def _run_model_chunks(model, ids, backend: str, fmt: str):
    model.set_kv_format(fmt)
    model.set_cache_backend(backend, page_size=16)
    cache = None
    logits = []
    start = 0
    for width in (15, 1, 1, 14, 1, 1):
        if start == ids.shape[1]:
            break
        width = min(width, ids.shape[1] - start)
        with torch.inference_mode():
            lg, cache = model._run(model.embed_tokens(ids[:, start:start + width]), cache)
        logits.append(lg.clone())
        start += width
    return logits, cache


def _assert_model_cache_equal(contiguous, paged) -> None:
    assert spec_cache.cache_length(contiguous) == spec_cache.cache_length(paged)
    for old, new in zip(contiguous, paged):
        mat = pc.materialize_layer(new)
        for key, value in old.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, mat[key]), key


def test_model_and_specdec_integration() -> None:
    from . import spec_dec
    from .model_self import QwenLM

    torch.manual_seed(7)
    base = QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8,
                  intermediate=64, num_layers=2)
    assert base.cache_backend == "contiguous"
    ids = torch.randint(0, 64, (1, 33))
    a_logits, a_cache = _run_model_chunks(base, ids, "contiguous", "fp16")
    b_logits, b_cache = _run_model_chunks(base, ids, "paged_v0", "fp16")
    assert all(torch.equal(a, b) for a, b in zip(a_logits, b_logits))
    _assert_model_cache_equal(a_cache, b_cache)

    torch.manual_seed(11)
    target = QwenLM(vocab_size=64, hidden=128, heads=1, kv_heads=1, head_dim=128,
                    intermediate=128, num_layers=1).to(torch.bfloat16)
    ids_short = torch.randint(0, 64, (1, 17))
    a_logits, a_cache = _run_model_chunks(target, ids_short, "contiguous", "v_fp8")
    b_logits, b_cache = _run_model_chunks(target, ids_short, "paged_v0", "v_fp8")
    assert all(torch.equal(a, b) for a, b in zip(a_logits, b_logits))
    _assert_model_cache_equal(a_cache, b_cache)

    torch.manual_seed(13)
    draft = QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8,
                   intermediate=64, num_layers=1).to(torch.bfloat16)
    prompt = torch.randint(0, 64, (1, 6))

    def run(backend: str):
        target.set_cache_backend(backend, page_size=16)
        draft.set_cache_backend(backend, page_size=16)
        return spec_dec.spec_decode_greedy(
            target, draft, prompt, 6, k=4, eos_id=None,
            target_kv="v_fp8", draft_kv="base_bf16")

    contiguous = run("contiguous")
    paged = run("paged_v0")
    assert [int(x) for x in contiguous["ids"]] == [int(x) for x in paged["ids"]]
    for key in ("rounds", "proposed", "accepted", "mismatch_rounds", "rollbacks"):
        assert contiguous["stats"][key] == paged["stats"][key], key

    target.set_cache_backend("contiguous")
    target.set_kv_format("fp8")
    try:
        target.set_cache_backend("paged_v0")
        raise AssertionError("model accepted paged_v0 + full fp8")
    except ValueError:
        pass


def test_specdec_m_0_through_4() -> None:
    from . import spec_dec

    class PositionStub:
        def __init__(self, overrides=None):
            self.overrides = overrides or {}
            self.backend = "contiguous"

        def set_kv_format(self, fmt):
            assert fmt == "fp16"

        def set_cache_backend(self, backend, page_size=16):
            self.backend = backend
            self.page_size = page_size

        def embed_tokens(self, ids):
            return ids.to(torch.float32)

        def _run(self, x, cache_list=None, need_logits=True):
            start = spec_cache.cache_length(cache_list)
            n = int(x.shape[1])
            logits = torch.zeros(1, n, 128)
            for j in range(n):
                pos = start + j
                logits[0, j, self.overrides.get(pos, pos % 100)] = 10
            current = torch.zeros(1, 1, n, 2, dtype=torch.bfloat16)
            if cache_list is None:
                if self.backend == "paged_v0":
                    cache_list = [pc.create_layer_cache("fp16", self.page_size)
                                  for _ in range(2)]
                    for layer in cache_list:
                        pc.append_current(layer, current, current)
                else:
                    cache_list = [{"k": current.clone(), "v": current.clone()}
                                  for _ in range(2)]
            elif self.backend == "paged_v0":
                for layer in cache_list:
                    pc.append_current(layer, current, current)
            else:
                for layer in cache_list:
                    layer["k"] = torch.cat([layer["k"], current], dim=2)
                    layer["v"] = torch.cat([layer["v"], current], dim=2)
            return logits if need_logits else None, cache_list

    prompt = torch.zeros(1, 6, dtype=torch.long)
    for expected_m in range(5):
        override = {} if expected_m == 4 else {5 + expected_m: 120}

        def run(backend):
            target, draft = PositionStub(override), PositionStub()
            target.set_cache_backend(backend)
            draft.set_cache_backend(backend)
            trace = []
            out = spec_dec.spec_decode_greedy(
                target, draft, prompt, 4, k=4, eos_id=None,
                log=lambda event: trace.append(event))
            return out, trace

        contiguous, trace_a = run("contiguous")
        paged, trace_b = run("paged_v0")
        assert trace_a[0]["m"] == trace_b[0]["m"] == expected_m
        assert [int(x) for x in contiguous["ids"]] == [int(x) for x in paged["ids"]]
        for key in ("rounds", "proposed", "accepted", "mismatch_rounds", "rollbacks"):
            assert contiguous["stats"][key] == paged["stats"][key], (expected_m, key)


def main() -> int:
    for fmt in ("fp16", "v_fp8"):
        test_boundaries(fmt)
        test_trim_and_cow(fmt)
        test_spec_cache_api(fmt)
        print(f"[{fmt}] boundary/append/trim/COW/materialization: PASS")
    test_malformed_detection()
    test_full_fp8_rejected()
    print("[consistency/rejection] malformed states and full fp8: PASS")
    test_model_and_specdec_integration()
    print("[model/SpecDec] base, v_fp8, rollback parity: PASS")
    test_specdec_m_0_through_4()
    print("[SpecDec rollback] m=0..4 contiguous/paged parity: PASS")
    print("paged_cache_selftest_cpu: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
