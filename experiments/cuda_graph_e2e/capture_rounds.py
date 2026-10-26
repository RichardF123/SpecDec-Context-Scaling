"""为 13 轮各 capture 一个 draft 链 graph 和一个 target verify graph（共享 pool，全部存活）。"""
from __future__ import annotations

import contextlib

import torch

from experiments.cuda_graph_local import rope_tables as _RT

from specdec.engine import model_self

from . import design

_QUIET = contextlib.nullcontext()


def build_chain(draft, state, io, r: int) -> dict:
    L = design.DRAFT_START[r]
    steps = design.DRAFT_STEPS[r]

    def fn():
        with model_self.attention_phase("propose"):
            cache = state.views(L)
            base = L
            if r == 0:
                lg = io.draft_logits0
                for i in range(4):
                    t = lg.reshape(-1, lg.shape[-1])[-1].argmax()
                    io.cand[0, i].copy_(t)
                    io.gen[0, design.GEN_CAND_START[r] + i].copy_(t)
                    lg, cache = draft._run(draft.embed_tokens(t.view(1, 1)), cache,
                                           need_logits=True)
                    state.write_slice(cache, base + i, 1)
            else:
                bonus = io.bonus[0]
                lg, cache = draft._run(draft.embed_tokens(bonus.view(1, 1)), cache,
                                       need_logits=True)
                state.write_slice(cache, L, 1)
                base = L + 1
                for i in range(4):
                    t = lg.reshape(-1, lg.shape[-1])[-1].argmax()
                    io.cand[0, i].copy_(t)
                    io.gen[0, design.GEN_CAND_START[r] + i].copy_(t)
                    lg, cache = draft._run(draft.embed_tokens(t.view(1, 1)), cache,
                                           need_logits=True)
                    state.write_slice(cache, base + i, 1)
            return None   # 不保留 lg/cache 引用：cache 已 copy_ 回 big buffer，保留会占显存

    return dict(name=f"chain{r}", fn=fn, kind="chain", round=r, steps=steps)


def build_verify(target, state, io, r: int) -> dict:
    L = design.VERIFY_START[r]
    s = design.VERIFY_S[r]

    def fn():
        with model_self.attention_phase("verify"):
            cache = state.views(L)
            toks = io.verify_tok4 if s == 4 else io.verify_tok5
            lg, cache = target._run(target.embed_tokens(toks), cache, need_logits=True)
            state.write_slice(cache, L, s)
            bonus = lg[0, -1].argmax()
            io.bonus.copy_(bonus)
            if r < design.ROUNDS - 1:
                io.gen[0, design.BONUS_POS[r]].copy_(bonus)
            return None   # 同上：不保留输出 cache

    return dict(name=f"verify{r}", fn=fn, kind="verify", round=r, M=s)


def assemble_verify_tokens(io, r: int) -> None:
    """GPU-only 拼装 verify 输入（图外 copy_；成本在 benchmark 中单独测量，不在这里同步）。"""
    if r == 0:
        io.verify_tok4.copy_(io.cand)
    else:
        io.verify_tok5[0, 0].copy_(io.bonus[0])
        io.verify_tok5[0, 1:5].copy_(io.cand)


def capture_all(models: dict, state_d, state_t, io, shared_pool: bool = False,
                rope_tables: dict | None = None) -> dict:
    """capture 26 个 graph；返回 graphs 列表与 capture 总时间。"""
    import time
    draft = models["draft"]
    target = models["target"]
    specs = []
    for r in range(design.ROUNDS):
        specs.append(build_chain(draft, state_d, io, r))
        specs.append(build_verify(target, state_t, io, r))

    def ctx_for(spec):
        if spec["kind"] == "verify" and rope_tables is not None:
            return _RT.patched_rope(rope_tables)
        return contextlib.nullcontext()

    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    for spec in specs:   # 统一 warm-up（pool 分配、cuBLAS workspace）
        with torch.cuda.stream(stream):
            with ctx_for(spec):
                spec["fn"]()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    graphs = []
    for spec in specs:
        g = torch.cuda.CUDAGraph()
        with ctx_for(spec):
            with torch.cuda.graph(g, pool=pool if pool is not None else torch.cuda.graph_pool_handle()):
                outs = spec["fn"]()
        graphs.append(dict(name=spec["name"], kind=spec["kind"], round=spec["round"],
                           graph=g, outs=outs))
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - t_start
    return dict(graphs=graphs, capture_s=capture_s, shared_pool=shared_pool)


def replay_pipeline(graphs: list, io, r) -> None:
    """固定 13 轮顺序 replay（Python 只做调用；token/cache 由 GPU 产生）。"""
    by_round = {}
    for item in graphs:
        by_round.setdefault(item["round"], {})[item["kind"]] = item["graph"]
    for r in range(design.ROUNDS):
        by_round[r]["chain"].replay()
        assemble_verify_tokens(io, r)
        by_round[r]["verify"].replay()
