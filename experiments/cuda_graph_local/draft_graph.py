"""Draft proposal 局部 A/B：A1 单步 s=1、A2 4 步 candidate chain、A3 bonus catch-up + 4 步 chain。

所有路径：真实 0.5B draft、真实 BF16 KV cache、固定历史长度 T、真实轨迹 token；
eager 与 graph 使用完全相同的输入 token 与 cache 起始状态（graph 输入 cache 为独立 clone）。
"""
from __future__ import annotations

import torch

from specdec.engine import model_self

from . import local_ops as L


def _embed(model, tok_tensor):
    return model.embed_tokens(tok_tensor)


def build_draft_step(model, init_cache, token: int) -> dict:
    """A1：单个 draft s=1 step（输入 token 固定，静态张量可 copy_ 更新）。"""
    static_tok = torch.tensor([[token]], dtype=torch.long, device=torch.device("cuda"))

    def eager():
        with model_self.attention_phase("propose"):
            lg, cache = model._run(_embed(model, static_tok), L.cache_view(init_cache),
                                   need_logits=True)
        return lg, cache

    def capture_fn():
        with model_self.attention_phase("propose"):
            return model._run(_embed(model, static_tok), L.cache_view(init_cache),
                              need_logits=True)

    return dict(name="A1", kind="draft", tokens=[token], static_tok=static_tok,
                eager=eager, capture_fn=capture_fn, init_cache=init_cache)


def build_draft_chain(model, init_cache, tokens: list) -> dict:
    """A2/A3：连续 len(tokens) 个 s=1 step（链内 cache 逐级推进；token 输入静态张量）。"""
    static_toks = torch.tensor([tokens], dtype=torch.long, device=torch.device("cuda"))

    def run_chain(cache):
        lg, c = None, cache
        for i in range(static_toks.shape[1]):
            lg, c = model._run(_embed(model, static_toks[:, i:i + 1]), c, need_logits=True)
        return lg, c

    def eager():
        with model_self.attention_phase("propose"):
            return run_chain(L.cache_view(init_cache))

    def capture_fn():
        with model_self.attention_phase("propose"):
            return run_chain(L.cache_view(init_cache))

    return dict(name="A3" if len(tokens) == 5 else "A2", kind="draft", tokens=list(tokens),
                static_tok=static_toks, eager=eager, capture_fn=capture_fn,
                init_cache=init_cache, n_steps=len(tokens))
