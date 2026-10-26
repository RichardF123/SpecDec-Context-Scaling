"""给 target verify 组件预生成 RoPE 表：把 CPU-FP32 inv_freq 的搬运移到 capture 之外。

原因：`target_rope_scope.scoped_build` 在每次 `_run` 里做 `inv_cpu.to(device)`（H2D 阻塞拷贝），
CUDA Graph capture 期间不允许 ---- 这是实测到的 graph-safety 阻塞点。
本模块在 capture 前用同一公式（`build_with_cpu_invfreq`，同一 inv_freq）预生成 (seq_len, start_pos)
对应的 cos/sin，并把 `model_self.Build_Rope_cache` 临时替换为"返回预生成表"的闭包；
数值与 scoped_build 完全一致（同函数、同 inv_freq），只是去掉了 capture 期间的 H2D。
"""
from __future__ import annotations

import contextlib

import torch

import specdec.engine.model_self as M


def build_rope_tables(target, seq_len_list, start_pos: int, dtype) -> dict:
    """按 (seq_len, start_pos) 预生成 target 的 cos/sin（CPU-FP32 inv_freq 公式）。"""
    from specdec.engine import target_rope_scope
    head_dim = int(target.head_dim)
    theta = float(target.rope_theta)
    inv_cpu = target_rope_scope.make_cpu_inv_freq(head_dim, theta)
    device = torch.device("cuda")
    tables = {}
    for s in seq_len_list:
        cos, sin = target_rope_scope.build_with_cpu_invfreq(
            head_dim, s, theta, device, dtype, start_pos=start_pos, inv_cpu=inv_cpu)
        tables[(s, start_pos)] = (cos, sin)
    return tables


@contextlib.contextmanager
def patched_rope(tables: dict):
    original = M.Build_Rope_cache

    def builder(head_dim, seq_len, theta, device, dtype, start_pos=0):
        key = (seq_len, start_pos)
        if key not in tables:
            raise RuntimeError(f"预生成 RoPE 表缺少 key={key}（capture 期间禁止 H2D）")
        cos, sin = tables[key]
        return cos, sin

    M.Build_Rope_cache = builder
    try:
        yield builder
    finally:
        M.Build_Rope_cache = original
