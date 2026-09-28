"""diag_fields.py — 诊断**字段语义**的单一来源（纯函数，可在 CPU 上测，不加载大模型）。

为什么单独成模块：任务要求把"topk 返回顺序"与"实际解码 argmax"分开、把并列情况写全、
并把 ULP 改成**逐元素按真实相邻可表示值**计算。这些规则必须只有一份实现，
否则工具与测试会各说各话。

字段口径（**严格**）
  * `argmax_id`   = `torch.argmax` 真正返回的下标（= 解码实际会选的 token）。
                    ⚠️ **不得**把 `topk(...)[0]` 的顺序称作"实际解码 top1"：
                    并列时 `topk` 与 `argmax` 的下标选择可以不同。
  * `max_score`   = 该行的最大值（数值本身，与下标无关）。
  * `n_max_tied`  = **等于最大值**的元素个数（并列数）。
  * `tied_max_tokens` = 所有取到最大值的 token 下标（完整集合，不只是前 2 个）。
  * `topk`        = 前 k 个 (token, score)，**仅作参考列表**。
  * `watch_scores`= 关注 token 的**原始分数**。
  * `finite`      = 整行是否有限。
误差口径
  * 优先报**原始绝对误差** `abs_err`；
  * 需要 ULP 时**逐元素**计算：`ulp_i` = 该元素在 bf16 下的相邻可表示值间距，
    再取 `(err_i / ulp_i)` 的最大值/中位数。**不得**用单一幅度（如 |x|≈20）的 ULP 归一化整行。
  * 参考：bf16 有 7 位尾数 ⇒ [16,32) 区间的间距是 2^(4-7) = **0.125**（不是 0.0625）。
"""
from __future__ import annotations

import torch

BF16 = torch.bfloat16


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """逐元素 bf16 间距：取相邻可表示值（上/下）距离的**较小者**。"""
    xb = x.to(BF16)
    up = torch.nextafter(xb, torch.full_like(xb, float("inf"))).float()
    dn = torch.nextafter(xb, torch.full_like(xb, float("-inf"))).float()
    xf = xb.float()
    ulp = torch.minimum((up - xf).abs(), (xf - dn).abs())
    return torch.where(ulp > 0, ulp, torch.full_like(ulp, float("nan")))


def row_fields(row: torch.Tensor, *, watch: list[int], k: int = 5) -> dict:
    """把一行 logits 变成**口径明确**的字段字典。`row` 形状 (V,) 或 (1,V)。"""
    r = row.reshape(-1).float()
    amax = int(torch.argmax(r))
    max_score = float(r[amax])
    tied = torch.nonzero(r == r[amax]).reshape(-1).tolist()
    tk = torch.topk(r, k)
    return dict(
        argmax_id=amax,                       # 解码实际会选的 token
        max_score=max_score,
        n_max_tied=len(tied),
        tied_max_tokens=[int(t) for t in tied],
        topk=[dict(token=int(t), score=float(s)) for t, s in zip(tk.indices, tk.values)],
        topk_note="topk 只是参考列表；并列时其顺序不等于解码 argmax",
        watch_scores={str(t): float(r[t]) for t in watch},
        finite=bool(torch.isfinite(r).all()),
        absmax=float(r.abs().max()),
    )


def compare_rows(a: torch.Tensor, b: torch.Tensor) -> dict:
    """两行的差异：**原始绝对误差**为主，ULP 逐元素计算。"""
    x, y = a.reshape(-1).float(), b.reshape(-1).float()
    d = (x - y).abs()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    out = dict(abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
               abs_err_p99=float(torch.quantile(d, 0.99)) if d.numel() else None,
               finite=finite, argmax_same=bool(int(torch.argmax(x)) == int(torch.argmax(y))))
    if finite:
        ulp = bf16_ulp(x)
        # ⚠️ **近零元素的 ULP 不是有意义的尺度**：x_i≈0 时相邻间距也≈0，
        # 比值会被放大到 1e6 量级而无信息量（首跑实测 ulp_err_max=1.35e6）。
        # ⇒ 主口径用**原始绝对误差**；ULP 只在 |x| ≥ ULP_FLOOR 的元素上统计，并单独给出：
        #   ① 绝对误差最大那个元素处的 ULP 比；② 受限行内最大/中位 ULP 比。
        ULP_FLOOR = 1.0
        ok = torch.isfinite(ulp) & (x.abs() >= ULP_FLOOR)
        i_max = int(torch.argmax(d))
        out.update(n_elements=int(d.numel()), n_ulp_valid=int(ok.sum()),
                   ulp_floor=ULP_FLOOR,
                   ulp_basis=f"逐元素分母 + 仅在 |x|>={ULP_FLOOR} 上统计（近零元素排除，"
                             f"否则比值无信息量）",
                   abs_err_at_max_element=dict(index=i_max, ref=float(x[i_max]),
                                               cand=float(y[i_max]), abs_err=float(d[i_max])))
        if bool(ok.any()):
            ratio = d[ok] / ulp[ok]
            out.update(ulp_err_max=float(ratio.max()), ulp_err_median=float(ratio.median()))
        else:
            out.update(ulp_err_max=None, ulp_err_median=None)
        u = ulp[i_max]
        out["ulp_err_at_max_element"] = (float(d[i_max] / u) if torch.isfinite(u) and float(u) > 0
                                         else None)
        # 归一化到"该行最大幅度处的 ULP"仅作**辅助**参考，且明确标注它不能代表整行
        amax = float(x.abs().max())
        out["ulp_at_row_max"] = float(bf16_ulp(torch.tensor([amax]))[0]) if amax > 0 else None
        out["abs_err_over_ulp_at_row_max"] = (float(d.max() / out["ulp_at_row_max"])
                                             if out["ulp_at_row_max"] else None)
        out["normalization_caveat"] = ("单一行内不同幅度元素的 ULP 不同；"
                                       "`abs_err_over_ulp_at_row_max` 只用该行最大幅度处的间距归一化，"
                                       "**不得**当作整行的 ULP 误差")
    else:
        out.update(ulp_err_max=None, ulp_err_median=None, ulp_err_at_max_element=None)
    return out


def position_labels(prompt_len: int, gen_index: int, cache_len_before: int, kp: int) -> dict:
    """把"输入位置 / 所预测的 token 位置 / 生成序列下标"三件事分开写清楚。

    * `cache_len_before` = 喂入本块之前 cache 的长度（也是本块第一行的**输入位置**）
    * 本块第 j 行的 logits 位于**输入位置** `cache_len_before + j`，
      预测的是**位置** `cache_len_before + j + 1` 上的 token
    * `gen_index` = 该 token 在**生成序列**里的 0-based 下标；
      由于 prompt_len 个 token 占据位置 0..prompt_len-1，
      位置 p 上的生成 token 的下标 = p − prompt_len
      ⇒ 由 `gen_index` 反推出的输入位置应为 `prompt_len + gen_index − 1`
    """
    return dict(
        gen_index=gen_index,
        input_position=cache_len_before,
        predicted_token_position=cache_len_before + 1,
        block_rows_positions=list(range(cache_len_before, cache_len_before + kp)),
        expected_input_position_from_gen_index=prompt_len + gen_index - 1,
        consistent=(cache_len_before == prompt_len + gen_index - 1),
        note="三套编号不可混用：输入位置 8222 ↔ 预测 token 位置 8223 ↔ 生成序列 index 31",
    )
