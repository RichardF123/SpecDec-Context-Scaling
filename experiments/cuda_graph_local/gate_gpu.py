"""CUDA Graph 局部 A/B 正确性门（每个组件：eager vs graph 的 logits/cache/输入新鲜度/确定性）。"""
from __future__ import annotations

import torch

from . import local_ops as L


def gate_component(comp: dict, graph, outs, alt_tokens: list) -> dict:
    static = comp["static_tok"]
    tok_tensor = torch.tensor([comp["tokens"]], dtype=torch.long, device=torch.device("cuda"))
    alt_tensor = torch.tensor([alt_tokens], dtype=torch.long, device=torch.device("cuda"))

    # 1) eager / graph 的 logits 与 cache（同一 token、同一初始 cache 内容）
    static.copy_(tok_tensor)
    torch.cuda.synchronize()
    lg_e, cache_e = comp["eager"]()
    graph.replay()
    torch.cuda.synchronize()
    lg_g, cache_g = outs
    logits_cmp = L.compare_logits(lg_e, lg_g)
    cache_cmp = L.compare_cache(cache_e, cache_g)
    finite = bool(torch.isfinite(lg_g).all()) and bool(torch.isfinite(lg_e).all())

    # 2) determinism：同输入连续两次 replay
    graph.replay()
    torch.cuda.synchronize()
    lg_a = lg_g.clone()
    ca = L.clone_cache(cache_g)
    graph.replay()
    torch.cuda.synchronize()
    lg_b = lg_g.clone()
    cb = L.clone_cache(cache_g)
    det_lg = torch.equal(lg_a, lg_b)
    det_cache = len(ca) == len(cb) and all(
        torch.equal(x["k"], y["k"]) and torch.equal(x["v"], y["v"]) for x, y in zip(ca, cb))

    # 3) input-update：换 token 后 replay 必须使用新 token（与 eager 一致，且不同于旧输出）
    stale_ok = True
    new_logits_cmp = None
    static.copy_(alt_tensor)
    graph.replay()
    torch.cuda.synchronize()
    lg_new = lg_g.clone()
    lg_e2, _ = comp["eager"]()
    new_logits_cmp = L.compare_logits(lg_e2, lg_new)
    changed = not torch.equal(lg_new, lg_a)
    stale_ok = changed and new_logits_cmp["argmax_equal"]
    static.copy_(tok_tensor)   # 恢复

    ok = (logits_cmp["shape_ok"] and logits_cmp["argmax_equal"]
          and cache_cmp["bitwise"] and cache_cmp["finite"]
          and L.cache_consistent(cache_g) and L.cache_consistent(cache_e)
          and finite and det_lg and det_cache and stale_ok)
    return dict(ok=bool(ok), logits=logits_cmp, cache_bitwise=cache_cmp["bitwise"],
                cache_layers=cache_cmp["layers"], cache_finite=cache_cmp["finite"],
                cache_consistent=L.cache_consistent(cache_g), finite=finite,
                deterministic_logits=bool(det_lg), deterministic_cache=bool(det_cache),
                input_update_changed=bool(changed), input_update_ok=bool(stale_ok),
                input_update_logits=new_logits_cmp,
                cache_max_abs_diff=max([r["max_abs_diff"] or 0.0 for r in cache_cmp["rows"]]))
