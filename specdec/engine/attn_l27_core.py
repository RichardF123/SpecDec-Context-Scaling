"""attn_l27_core.py — L27 attention 核心的**功能式**实现（生产表达式逐字对应）。

为什么存在
  为在「RoPE 后、attention 核心之前」这个边界上**固定输入**、只改调用形状（查询行数）或 K/V 长度，
  需要把生产 `SelfAttention.forward` 里的核心几步单独取出来。生产的这几步**不是独立函数**，故此处
  逐字复刻其表达式（含 dtype 与 dtype 转换顺序），并**不**修改生产文件。

与生产代码的对应（`specdec/engine/model_self.py`）
  :408-409  `k_rep = kv_repeat(k_all, 6)` / `v_rep = kv_repeat(v_all, 6)`
  :414      `scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)`
  :415-423  `p = T - s`；`cache is None → triu(1)`；`s == 1 → 不施加 mask`；否则 `triu(diag=p+1)`
  :424      `attn = torch.softmax(scores, dim=-1, dtype=torch.float32)`
  :429-430  `attn = attn.to(q.dtype)`；`out = torch.matmul(attn, v_rep)`

**本模块刻意接受显式 mask**（而不是复刻 `s==1` 的捷径）：G1 的单查询必须使用与块行 1 **逐元素相同**的
可见性约束，否则会看见未来 key。调用方负责在结果里标注「这偏离了生产的 `s==1` 分支」。

本模块**不做任何断言以外的张量修改**，也不写文件；所有检查由调用方用 `diag_checks` 的 `require` 完成。
"""
from __future__ import annotations

import math

import torch

from . import model_self as M

HEAD_DIM = 128
N_HEADS = 12
N_KV_HEADS = 2
N_REP = N_HEADS // N_KV_HEADS


def block_mask(s: int, T: int, device=None) -> torch.Tensor:
    """**保留**形式（True = 可参与），与生产 eager 路径同义。

    第 i 个 query 只能访问 `key_position ≤ (T - s) + i`（`model_self.py:246` 的 `_kv_keep_mask`）。
    """
    return M._kv_keep_mask(s, T, device if device is not None else torch.device("cpu"))


def scores_from(q: torch.Tensor, k_all: torch.Tensor) -> torch.Tensor:
    """`scores = matmul(q, k_repᵀ) / sqrt(head_dim)`，与生产 :414 一致（含除法后仍为 bf16）。"""
    _check_qk(q, k_all)
    k_rep = M.kv_repeat(k_all, q.shape[1] // k_all.shape[1])
    return torch.matmul(q, k_rep.transpose(-2, -1)) / (HEAD_DIM ** 0.5)


def probs_from(scores: torch.Tensor, keep: torch.Tensor, q_dtype: torch.dtype) -> torch.Tensor:
    """`masked_fill(-inf) → softmax(fp32) → 回转 q.dtype`，与生产 :415-429 一致。

    `keep` 形状必须能广播到 `scores` 的 `(s,T)` 末尾两维，且 dtype 为 bool。
    """
    if keep.dtype != torch.bool:
        raise TypeError(f"keep 必须是 bool，收到 {keep.dtype}")
    if tuple(keep.shape[-2:]) != tuple(scores.shape[-2:]):
        raise ValueError(f"keep 尾两维 {tuple(keep.shape[-2:])} 与 scores {tuple(scores.shape[-2:])} 不一致")
    masked = scores.masked_fill(~keep, float("-inf"))
    attn32 = torch.softmax(masked, dim=-1, dtype=torch.float32)
    return attn32.to(q_dtype)


def av_from(probs_bf16: torch.Tensor, v_all: torch.Tensor) -> torch.Tensor:
    """`out = matmul(attn, v_rep)`，与生产 :430 一致。"""
    v_rep = M.kv_repeat(v_all, probs_bf16.shape[1] // v_all.shape[1])
    out = torch.matmul(probs_bf16, v_rep)
    return out.to(probs_bf16.dtype)


def core_forward(q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor,
                 keep: torch.Tensor) -> dict:
    """一次功能式核心：返回 scores / 概率(bf16) / 核心输出（转置展平前的 `(b,heads,s,hd)`）。

    ⚠️ 只做一次「scores→概率」的 dtype 往返（与生产一致）；不重排、不转 dtype、不搬设备。
    """
    sc = scores_from(q, k_all)
    pr = probs_from(sc, keep, q.dtype)
    av = av_from(pr, v_all)
    return dict(scores=sc, probs=pr, av=av)


def flatten_core(av: torch.Tensor) -> torch.Tensor:
    """与生产 :432 一致：`transpose(1,2).contiguous().view(b,s,heads*hd)`。"""
    b, h, s, hd = av.shape
    return av.transpose(1, 2).contiguous().view(b, s, h * hd)


def truncate_kv(k_all: torch.Tensor, v_all: torch.Tensor, T_new: int) -> tuple[torch.Tensor, torch.Tensor]:
    """**纯切片**截断：只允许从尾部删除（`T_new <= T`），返回的是视图（不复制数值）。"""
    if T_new > k_all.shape[2]:
        raise ValueError(f"T_new={T_new} 大于现有长度 {k_all.shape[2]}")
    return k_all[:, :, :T_new, :], v_all[:, :, :T_new, :]


def visible_count(keep_row: torch.Tensor) -> int:
    """一行的可见 key 数（True 的个数）。"""
    return int(keep_row.reshape(-1).sum())


def last_visible_key(keep_row: torch.Tensor) -> int:
    """一行最后一个可见 key 的**绝对位置**（无可见项返回 -1）。"""
    idx = torch.nonzero(keep_row.reshape(-1)).reshape(-1)
    return int(idx[-1]) if idx.numel() else -1


def _check_qk(q: torch.Tensor, k_all: torch.Tensor) -> None:
    if q.dim() != 4 or k_all.dim() != 4:
        raise ValueError(f"q/k_all 必须是 4 维，收到 {tuple(q.shape)} / {tuple(k_all.shape)}")
    if q.shape[1] % k_all.shape[1] != 0:
        raise ValueError(f"heads {q.shape[1]} 不是 kv_heads {k_all.shape[1]} 的整数倍")
    if q.shape[3] != k_all.shape[3]:
        raise ValueError(f"head_dim 不一致 {q.shape[3]} vs {k_all.shape[3]}")
    if q.dtype != k_all.dtype:
        raise ValueError(f"dtype 不一致 {q.dtype} vs {k_all.dtype}")
