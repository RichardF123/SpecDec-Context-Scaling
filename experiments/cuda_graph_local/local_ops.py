"""CUDA Graph 局部 A/B 的 capture/replay/对照/计时工具（固定历史长度口径）。"""
from __future__ import annotations

import torch

from specdec.engine import model_self, spec_cache


def clone_cache(cache):
    return [{k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in cache]


def cache_view(cache):
    """浅 dict 副本：cat 只改写副本的 dict，初始张量不变。"""
    return [dict(layer) for layer in cache]


def capture(build_fn, warmup: int = 3, pool=None):
    """build_fn 在 capture 内执行一次；返回 (graph, outputs, memory_delta)。"""
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            build_fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    alloc0 = torch.cuda.memory_allocated()
    res0 = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        outs = build_fn()
    torch.cuda.synchronize()
    alloc1 = torch.cuda.memory_allocated()
    res1 = torch.cuda.memory_reserved()
    return graph, outs, dict(pool_alloc_delta=alloc1 - alloc0,
                             pool_reserved_delta=res1 - res0)


def time_trials(fn, n: int, trials: int = 5, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / n)
    samples.sort()
    return dict(n_per_trial=n, trials=trials, samples_ms=samples,
                median_ms=samples[len(samples) // 2], min_ms=samples[0], max_ms=samples[-1])


def compare_logits(a, b) -> dict:
    shape_ok = tuple(a.shape) == tuple(b.shape)
    bitwise = shape_ok and torch.equal(a, b)
    if shape_ok:
        maxdiff = 0.0 if bitwise else float((a.float() - b.float()).abs().max().item())
    else:
        maxdiff = None
    argmax_equal = shape_ok and bool(torch.equal(a.argmax(-1), b.argmax(-1)))
    return dict(shape_ok=shape_ok, bitwise=bitwise, max_abs_diff=maxdiff, argmax_equal=argmax_equal)


def compare_cache(a, b) -> dict:
    rows = []
    ok = len(a) == len(b)
    for i, (la, lb) in enumerate(zip(a, b)):
        for key in ("k", "v"):
            ta, tb = la[key], lb[key]
            same_shape = tuple(ta.shape) == tuple(tb.shape)
            same_dtype = ta.dtype == tb.dtype
            bitwise = same_shape and same_dtype and torch.equal(ta, tb)
            md = 0.0 if bitwise else (float((ta.float() - tb.float()).abs().max().item())
                                      if same_shape else None)
            rows.append(dict(layer=i, key=key, bitwise=bitwise, max_abs_diff=md,
                             dtype=str(ta.dtype), shape=list(ta.shape)))
            ok = ok and bitwise
    finite = all(bool(torch.isfinite(layer[key]).all()) for layer in a for key in ("k", "v"))
    return dict(bitwise=ok, layers=len(a), finite=finite, rows=rows)


def cache_consistent(cache) -> bool:
    try:
        spec_cache.check_consistent(cache)
        return True
    except Exception:
        return False


def set_phase(name: str):
    return model_self.attention_phase(name)
