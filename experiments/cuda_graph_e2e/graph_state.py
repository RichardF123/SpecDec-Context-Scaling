"""静态大容量 KV buffer 与静态输入/输出张量（端到端 Graph 的数据面）。"""
from __future__ import annotations

import torch

from . import design


def _kv_shape(model):
    layer = model.layers[0].self_attn
    return int(layer.kv_heads), int(layer.head_dim)


class BigBuffers:
    def __init__(self, model, capacity: int = design.MAX_CAPACITY):
        self.model = model
        kv_heads, head_dim = _kv_shape(model)
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.k = []
        self.v = []
        for _ in model.layers:
            self.k.append(torch.empty(1, kv_heads, capacity, head_dim, dtype=torch.bfloat16,
                                      device=torch.device("cuda")))
            self.v.append(torch.empty(1, kv_heads, capacity, head_dim, dtype=torch.bfloat16,
                                      device=torch.device("cuda")))

    def load_from_cache(self, cache):
        """把真实 prefill cache 的 [0:len] 写入 big buffer。"""
        n = int(cache[0]["k"].shape[2])
        for i, layer in enumerate(cache):
            self.k[i][:, :, :n].copy_(layer["k"])
            self.v[i][:, :, :n].copy_(layer["v"])

    def views(self, length: int):
        return [dict(k=self.k[i][:, :, :length], v=self.v[i][:, :, :length])
                for i in range(len(self.k))]

    def write_slice(self, cache, start: int, s: int):
        """把一次 _run 新增的 [start:start+s] 段写回 buffer（图内 copy_）。"""
        for i, layer in enumerate(cache):
            self.k[i][:, :, start:start + s].copy_(layer["k"][:, :, start:start + s])
            self.v[i][:, :, start:start + s].copy_(layer["v"][:, :, start:start + s])


class StaticIO:
    def __init__(self):
        dev = torch.device("cuda")
        # 用 0 初始化（合法 token id）：capture/warm-up 期间输入值任意但必须是合法索引
        self.cand = torch.zeros(1, design.K, dtype=torch.long, device=dev)
        self.bonus = torch.zeros(1, dtype=torch.long, device=dev)
        self.gen = torch.zeros(1, design.MAX_NEW, dtype=torch.long, device=dev)
        self.verify_tok4 = torch.zeros(1, 4, dtype=torch.long, device=dev)
        self.verify_tok5 = torch.zeros(1, 5, dtype=torch.long, device=dev)
        self.draft_logits0 = None   # round0 链的初始 logits（1,1,V），由 prefill 填充

    def load_draft_logits0(self, drf_lg):
        self.draft_logits0 = drf_lg[:, -1:].clone()

    def snapshot(self):
        return dict(cand=self.cand.clone(), bonus=self.bonus.clone(), gen=self.gen.clone())

    def restore(self, snap):
        self.cand.copy_(snap["cand"])
        self.bonus.copy_(snap["bonus"])
        self.gen.copy_(snap["gen"])
