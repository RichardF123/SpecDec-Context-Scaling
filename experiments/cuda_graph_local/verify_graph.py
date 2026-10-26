"""Target verification 局部 A/B：B1 M=4 block verify、B2 M=5 block verify。"""
from __future__ import annotations

import torch

from specdec.engine import model_self

from . import local_ops as L


def build_verify(model, init_cache, tokens: list) -> dict:
    name = "B1" if len(tokens) == 4 else "B2"
    static_toks = torch.tensor([tokens], dtype=torch.long, device=torch.device("cuda"))

    def eager():
        with model_self.attention_phase("verify"):
            lg, cache = model._run(model.embed_tokens(static_toks), L.cache_view(init_cache),
                                   need_logits=True)
        return lg, cache

    def capture_fn():
        with model_self.attention_phase("verify"):
            return model._run(model.embed_tokens(static_toks), L.cache_view(init_cache),
                              need_logits=True)

    return dict(name=name, kind="target", tokens=list(tokens), static_tok=static_toks,
                eager=eager, capture_fn=capture_fn, init_cache=init_cache, M=len(tokens))
