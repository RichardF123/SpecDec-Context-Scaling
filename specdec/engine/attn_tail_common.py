"""attn_tail_common.py — A/B 末端传播实验的**共享助手**（CPU 可测；不含任何模型调用）。

设计要点（对应任务 §三/§四）
  * `as_row3` 只接受 `(1,1,1536)` 或可无损 reshape 的 `(1536,)`，且**必须是 bf16**；
  * `bf16_add_row` 强制**两个输入都是 bf16、形状完全相同**，并在 **bf16** 下做加法，
    **拒绝**任何「FP32 求和后再回转」的替代路径（该路径会给出不同结果，见 CPU 自测）；
  * `check_residual_identity` 断言 `layer_in + attn_out == post_attn_residual`：
    残差必须取 **L27 层输入**，若误用 **attention 之后的残差**（重复加 attention）该检查必然失败；
  * 所有比较都**先断言 shape**，禁止广播；有限性对完整张量实算。
"""
from __future__ import annotations

import torch

from . import diag_checks as DC

HIDDEN = 1536
ROW3 = (1, 1, HIDDEN)


def as_row3(t, what: str, *, device=None) -> torch.Tensor:
    """把 `(1,1,1536)` 或 `(1536,)` 归一为 `(1,1,1536)`；只允许 **bf16**，且必须 finite。"""
    DC.require(torch.is_tensor(t), f"{what}: 必须是张量", f"收到 {type(t).__name__}")
    DC.require(str(t.dtype) == "torch.bfloat16",
               f"{what}: 必须是 bf16（禁止用 FP32 值替代真实 bf16 张量）", str(t.dtype))
    if tuple(t.shape) == ROW3:
        out = t
    elif tuple(t.shape) == (HIDDEN,):
        out = t.reshape(1, 1, HIDDEN)
    else:
        raise DC.CheckError(f"{what}: 形状必须是 {ROW3} 或 ({HIDDEN},)，收到 {tuple(t.shape)}")
    DC.require(DC.finite_all(t), f"{what}: 含非有限元素")
    if device is not None:
        out = out.to(device)
    return out


def require_bf16_exact(t, what: str) -> None:
    """确认 fp32 记录值确实**无损**等于某个 bf16 真值（否则不能当作 bf16 复用）。"""
    DC.require(torch.equal(t.float(), t.float().to(torch.bfloat16).float()),
               f"{what}: fp32→bf16 往返不精确 ⇒ 不是 bf16 真值的精确记录")


def bf16_add_row(a: torch.Tensor, b: torch.Tensor, what: str) -> torch.Tensor:
    """**生产语义**的 bf16 残差加法：两个输入必须已是 bf16 且同形，结果仍是 bf16。"""
    for t, nm in ((a, "a"), (b, "b")):
        DC.require(torch.is_tensor(t), f"{what}: {nm} 必须是张量")
        DC.require(str(t.dtype) == "torch.bfloat16",
                   f"{what}: {nm} 必须是 bf16（**禁止** FP32 求和后再回转替代真实 bf16 加法）",
                   str(t.dtype))
        DC.require(tuple(t.shape) == ROW3, f"{what}: {nm} 形状必须是 {ROW3}", str(tuple(t.shape)))
        DC.require(DC.finite_all(t), f"{what}: {nm} 含非有限元素")
    DC.require(a.device == b.device, f"{what}: 两侧设备不一致", f"{a.device} vs {b.device}")
    out = a + b
    DC.require(str(out.dtype) == "torch.bfloat16", f"{what}: 结果必须仍是 bf16", str(out.dtype))
    DC.require(DC.finite_all(out), f"{what}: 结果含非有限元素")
    return out


def check_residual_identity(layer_in: torch.Tensor, attn_out: torch.Tensor,
                            post_attn_residual: torch.Tensor, what: str) -> dict:
    """**残差选择**的硬检查：`layer_in + attn_out` 必须逐元素等于 attention **之后**的残差。

    * 成立 ⇒ 说明 `layer_in` 取的确实是 **L27 层输入**（而不是 attention 之后的残差）；
    * 若有人误把「attention 之后的残差」当残差再与 `attn_out` 相加（重复加 attention），
      本检查会失败并阻止实验继续。
    """
    got = bf16_add_row(layer_in, attn_out, f"{what}: 残差恒等式左侧")
    DC.assert_same_shape(got, post_attn_residual, f"{what}: 残差恒等式")
    DC.require(str(post_attn_residual.dtype) == "torch.bfloat16",
               f"{what}: 参考残差必须是 bf16", str(post_attn_residual.dtype))
    DC.require(DC.finite_all(post_attn_residual), f"{what}: 参考残差含非有限元素")
    eq = bool(torch.equal(got, post_attn_residual))
    DC.require(eq, f"{what}: `layer_in + attn_out` 未逐元素等于 attention 之后的残差 ⇒ "
                    f"残差来源选错（可能重复加了 attention）")
    return dict(identity_holds=True, shape=list(got.shape), dtype=str(got.dtype))


def stage_cmp(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    """逐阶段比较：先断言 shape（禁止广播），再逐元素与绝对误差统计，并实算有限性。"""
    return DC.cmp_tensor(a, b, name)
