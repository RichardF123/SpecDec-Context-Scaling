"""PagedAttention v1d CUDA pointer-table cache gate.

Verifies that:
1. first direct use builds K/V page-address tables;
2. repeated direct use reuses the same table tensors;
3. appends within the current page do not rebuild the tables;
4. allocating a new physical page invalidates/rebuilds them;
5. trim/COW produces a fresh cache with fresh pointer metadata;
6. direct CUDA attention remains numerically equal to the materialized CUDA
   reference after trim and append.

Run:
    python -m specdec.engine.paged_pointer_table_test
"""
from __future__ import annotations

import torch

from . import paged_cache
from specdec.kernels import ops


def _rand(shape, *, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _append(layer, n: int, *, kv: int = 2, d: int = 64):
    k = _rand((1, kv, n, d))
    v = _rand((1, kv, n, d))
    paged_cache.append_current(layer, k, v)


def _stats(label: str):
    st = paged_cache.pointer_table_stats()
    print(
        f"[{label}] builds={st['builds']} hits={st['hits']} "
        f"invalidations={st['invalidations']}"
    )
    return st


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    torch.manual_seed(123)
    paged_cache.reset_pointer_table_stats()

    layer = paged_cache.create_layer_cache("fp16", page_size=4)
    _append(layer, 5)  # two pages, partial tail

    k1, v1 = paged_cache.cuda_pointer_tables(layer)
    st = _stats("first-build")
    assert st == {"builds": 1, "hits": 0, "invalidations": 0}
    assert k1.numel() == 2 and v1.numel() == 2

    k2, v2 = paged_cache.cuda_pointer_tables(layer)
    st = _stats("reuse")
    assert st["builds"] == 1 and st["hits"] == 1
    assert k1.data_ptr() == k2.data_ptr()
    assert v1.data_ptr() == v2.data_ptr()

    # length 5 -> 6 remains within the second physical page.
    _append(layer, 1)
    k3, v3 = paged_cache.cuda_pointer_tables(layer)
    st = _stats("same-page-append")
    assert st["builds"] == 1 and st["hits"] == 2
    assert st["invalidations"] == 0
    assert k1.data_ptr() == k3.data_ptr()
    assert v1.data_ptr() == v3.data_ptr()

    # length 6 + 3 crosses from page 2 into a newly allocated page 3.
    _append(layer, 3)
    k4, v4 = paged_cache.cuda_pointer_tables(layer)
    st = _stats("new-page")
    assert st["builds"] == 2
    assert st["invalidations"] == 1
    assert k4.numel() == 3 and v4.numel() == 3
    assert k4.data_ptr() != k1.data_ptr()
    assert v4.data_ptr() != v1.data_ptr()

    # Trim to a partial tail: the tail page is cloned for COW, therefore a
    # pointer table from the source must never be inherited.
    trimmed = paged_cache.trim_layer(layer, 7)
    assert "_k_ptrs_cuda" not in trimmed
    assert "_v_ptrs_cuda" not in trimmed
    tk, tv = paged_cache.cuda_pointer_tables(trimmed)
    st = _stats("trim-rebuild")
    assert st["builds"] == 3
    assert tk.numel() == 2 and tv.numel() == 2

    # Append after trim stays COW-safe and the cached table remains valid until
    # a new physical page is required.
    _append(trimmed, 1)
    tk2, tv2 = paged_cache.cuda_pointer_tables(trimmed)
    assert tk.data_ptr() == tk2.data_ptr()
    assert tv.data_ptr() == tv2.data_ptr()

    # Operator parity through the fast validate=False path used by model_self.
    q = _rand((1, 14, 1, 64))
    materialized = paged_cache.materialize_layer(trimmed)
    ref = ops.attention(
        q, materialized["k"], materialized["v"], backend="cuda")
    got = ops.paged_attention_decode(
        q, trimmed, backend="cuda", validate=False)
    assert torch.equal(ref, got), (
        "direct paged attention differs after trim/append; "
        f"max_abs={float((ref.float() - got.float()).abs().max())}"
    )

    paged_cache.check_consistent(layer)
    paged_cache.check_consistent(trimmed)

    print("[operator-after-trim] materialized_cuda == direct_cuda: PASS")
    print("\npaged_pointer_table_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
