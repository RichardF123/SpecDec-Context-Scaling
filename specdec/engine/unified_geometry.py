"""unified_geometry.py — **统一执行几何（UCG）**最小原型的共享逻辑。

设计见 `docs/unified_exec_geometry_design_2026-10-01.md`（**§12.5 为修订说明**）。核心：
  target 的**每一次**前向都使用 `s = K_BLOCK(=4)` 的块形状：
    * AR 每步喂 `[t, pad, pad, pad]` ⇒ 取 **row 0** ⇒ `trim_cache(blk, pre+1)`；
    * Spec 的 verify 本来就是 s=4；repair 走 `_step_padded`（pad 成 4 行）；
    * 尾轮 `k_eff < 4` 时把候选 pad 到 4，只取前 `k_eff` 行的 logits。

⚠️ **口径限制（2026-10-01 修订，勿再过度表述）**：
  * 本模块**只统一** `s` 与 `T-pre`（以及块 mask 的对角线偏移）；**没有**证明"同一预测位置的**完整**
    计算几何一致"（查询行数 / `T` / 批次形状带来的 kernel 启发式都可能各自不同）；
  * F13 实测：**两臂仍分歧**（首次分歧 index 47）⇒ **固定行数不足以保证一致，具体致因尚未唯一定位**；
  * **不要**把本模块表述为"结构上必然修复"或"块验证与严格一致不相容"；后者已**撤回**；
  * 模块内"几何一致"只指 `require_geometry_uniform` 检查的那三个量，**不是**完整计算等价。

本模块只做**纯逻辑**与**断言**；调用生产 `spec_dec`/`spec_cache` 的部分在运行脚本里。
**不改生产默认行为**：`step_padded` / `verify_padded` 只在诊断入口显式使用。
"""
from __future__ import annotations

import torch

KB = 4          # 统一块形状（= proposal 冻结的 k=4）
PAD_ID = 0      # 与答案无关的固定 pad id


class URError(Exception):
    pass


def require(cond, what: str, detail=""):
    if not cond:
        raise URError(f"{what} 未满足。{detail}".strip())
    return True


def pad_to_block(tokens: list, *, block: int = KB, pad_id: int = PAD_ID) -> tuple:
    """把 `tokens`(1..block) pad 到 `block` 个；返回 `(padded, n_real)`。"""
    require(1 <= len(tokens) <= block, "待 pad 的 token 数必须在 1..block", str(len(tokens)))
    n = len(tokens)
    return list(tokens) + [pad_id] * (block - n), n


def ar_step_plan(pre: int, token: int, *, block: int = KB) -> dict:
    """AR 单步的统一几何计划：喂 block 行、取 row 0、提交 1 个位置。"""
    padded, n_real = pad_to_block([token], block=block)
    return dict(kind="ar_step", pre=pre, fed=padded, n_real=n_real, take_row=0,
                commit=1, s=block, T=pre + block,
                mask_diagonal=(pre + block) - block + 1,       # triu(diag=T-s+1) = pre+1
                visible_last_key=pre,                          # row 0 只能看见 key ≤ pre
                trim_to=pre + 1)


def verify_plan(pre: int, cands: list, *, block: int = KB) -> dict:
    """Spec verify 的统一几何计划：pad 候选到 block、只取前 n_real 行、提交 m 个位置（由生产决定）。"""
    require(1 <= len(cands) <= block, "候选数必须在 1..block", str(len(cands)))
    padded, n_real = pad_to_block(cands, block=block)
    return dict(kind="verify", pre=pre, fed=padded, n_real=n_real,
                take_rows=list(range(n_real)), s=block, T=pre + block,
                mask_diagonal=(pre + block) - block + 1,
                row_visible_last_key=[pre + j for j in range(n_real)],
                trim_to=None)          # 由生产按 pre+m 决定


def repair_plan(pre: int, token: int, *, block: int = KB) -> dict:
    """Spec repair 的统一几何计划（与 AR 单步同形，保证几何一致）。"""
    p = ar_step_plan(pre, token, block=block)
    p["kind"] = "repair"
    return p


def tail_plan(pre: int, cands: list, k_eff: int, *, block: int = KB) -> dict:
    """尾轮：`k_eff < block` 时补足候选（pad 到 block），但只取前 `k_eff` 行。"""
    require(1 <= k_eff <= block, "k_eff 必须在 1..block", str(k_eff))
    require(len(cands) == k_eff, "候选数应与 k_eff 一致", f"{len(cands)} vs {k_eff}")
    p = verify_plan(pre, list(cands) + [PAD_ID] * (block - k_eff), block=block)
    p["kind"] = "tail_verify"
    p["take_rows"] = list(range(k_eff))
    p["n_real"] = k_eff
    return p


def require_geometry_uniform(plans: list, *, block: int = KB) -> dict:
    """断言所有计划（AR 步 / verify / repair / 尾轮）的**几何完全一致**。"""
    require(plans, "至少需要一个计划")
    shapes = {(p["s"], p["T"] - p["pre"], p["mask_diagonal"] - p["pre"]) for p in plans}
    require(len(shapes) == 1, "计划之间的几何不一致（s / T-pre / mask 对角线偏移）", str(shapes))
    s, t_off, d_off = next(iter(shapes))
    require(s == block and t_off == block and d_off == 1,
            "几何应为 (s=block, T=pre+block, diag=pre+1)", f"{(s, t_off, d_off)}")
    return dict(s=s, T_minus_pre=t_off, mask_diagonal_minus_pre=d_off, n_plans=len(plans))


def require_pad_rows_not_visible(plan: dict) -> None:
    """pad 行（位置 pre+n_real .. pre+block-1）不得被**参与决策的行**看见。"""
    take = plan["take_rows"]
    vis = plan["row_visible_last_key"]
    vis_list = vis if isinstance(vis, list) else [vis] * len(take)
    for j, v in zip(take, vis_list):
        require(v <= plan["pre"] + j, "行的可见范围超过了它自己的位置", f"row {j} vis={v}")
        require(v < plan["pre"] + plan["n_real"],
                "参与决策的行看见了 pad 行（违反因果/掩码）",
                f"row {j} vis={v} n_real={plan['n_real']}")


def require_commit_only_real(plan: dict, committed: int) -> None:
    """提交数不得超过真实行数（pad 行的 KV 不得进入有效 cache）。"""
    require(0 <= committed <= plan["n_real"], "提交数超过真实行数（pad 行被提交）",
            f"committed={committed} n_real={plan['n_real']}")


# ---------------- 与生产对接的两个包装（诊断入口专用） ----------------

def step_padded(model, cache, token: int, *, call, phase: str = "decode", block: int = KB):
    """`_step` 的统一几何版本：pad 成 block 行 → 取 row 0 → trim 回 `pre+1`。

    返回值与生产 `spec_dec._step` 兼容：`(lg (1,1,V), cache)`。

    ⚠️ `call` **必填**，且必须是 **verify 形态**（签名 `(model, cache, token_list) -> (rows, blk)`）
    的底层函数 —— 通常传 `spec_dec._verify` 的**原始**（hooks 安装之前）引用：
      * 不填会在"模块属性已被 hooks 替换"时自我递归（真踩过）；
      * 传 hooks 包装后的版本会让相位标签被内层覆盖（repair 被记成 verify），记账就不准了。
    """
    from . import model_self as M
    from . import spec_cache
    pre = int(spec_cache.cache_length(cache))
    plan = ar_step_plan(pre, token, block=block)
    with torch.inference_mode(), M.attention_phase(phase):
        rows, blk = call(model, cache, plan["fed"])
    require(int(rows.shape[0]) == block, "块调用返回行数应为 block", str(list(rows.shape)))
    lg = rows[plan["take_row"]].reshape(1, 1, -1)
    require(bool(torch.isfinite(lg.float()).all()), "padded 单步 logits 非有限")
    new_cache = spec_cache.trim_cache(blk, plan["trim_to"])
    require(int(spec_cache.cache_length(new_cache)) == plan["trim_to"],
            "trim 后 cache 长度不等于 pre+commit", str(int(spec_cache.cache_length(new_cache))))
    return lg, new_cache


def verify_padded(model, cache, cands: list, *, call, phase: str = "verify", block: int = KB):
    """`_verify` 的统一几何版本：pad 候选到 block，只回传前 `len(cands)` 行。

    `call` 必填（同 `step_padded`）：传 `spec_dec._verify` 的**原始**引用。
    """
    from . import model_self as M
    from . import spec_cache
    pre = int(spec_cache.cache_length(cache))
    plan = verify_plan(pre, list(cands), block=block)
    with torch.inference_mode(), M.attention_phase(phase):
        rows, blk = call(model, cache, plan["fed"])
    require(int(rows.shape[0]) == block, "块调用返回行数应为 block", str(list(rows.shape)))
    out = rows[:plan["n_real"]].contiguous()
    require(bool(torch.isfinite(out.float()).all()), "padded verify 返回行含非有限值")
    return out, blk
