"""spec_verify_index_selftest_cpu.py — `_verify` 的**索引/位置/mask/cache 语义**哨兵（CPU，tiny 模型）。

任务背景：8K 同后端 AR/SpecDec 在第 32 个生成 token 分歧。要判断这是"实现/索引错误"还是"数值路径差异"，
先得把**位置关系**钉死。本哨兵**不加载大模型**，用 tiny 随机权重的真引擎 + 真 `spec_dec._verify` 验证：

  P  = prompt 长度（cache 长度）
  ① `saved_next` = prefill 最后一个位置的 logits（位置 P−1）⇒ 它回答的是 **cands[0]**；
  ② `rows[j]`  = block 验证里第 j 行的 logits，位于**位置 P+j** ⇒ 它回答的是 **cands[j+1]**；
     ⇒ 即"`rows[j-1]` 对应第 j 个候选"（用户要求核对的那条）。
  ③ 因此 `rows[j]` 必须与"在长度为 P+j 的 cache 上单步输入 cands[j] 得到的 logits"是**同一位置**；
  ④ block 内的**因果性**：把**未来**候选换成别的（**长度不变**）时，`rows[0..j-1]` 不得改变
     —— 用同长度替换检查泄漏，而不是用不同长度差值；
  ⑤ `_verify` **是否会就地修改传入的 cache**（决定快照该怎么做）；
  ⑥ mask 的偏移量：block 第 j 行只能看到 ≤ P+j。

用法：python -m specdec.engine.spec_verify_index_selftest_cpu
"""
from __future__ import annotations

import torch

from . import model_self as M
from . import spec_cache
from . import spec_dec

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def tiny(seed: int = 0, layers: int = 2, heads: int = 4, kv_heads: int = 2, head_dim: int = 8):
    torch.manual_seed(seed)
    return M.QwenLM(vocab_size=64, hidden=heads * head_dim, heads=heads, kv_heads=kv_heads,
                    head_dim=head_dim, intermediate=2 * heads * head_dim, num_layers=layers)


def clone_cache(cache):
    """**逐层 dict 浅拷贝**：引擎把新张量重新绑定到键上、从不就地改写旧张量
    （本哨兵第 ⑤ 项会验证这一点），所以浅拷贝即可隔离两条分支。"""
    return [dict(d) for d in cache]


def lens(cache) -> list[int]:
    return [int(d["k"].shape[2]) for d in cache]


def prefill(model, ids):
    with torch.inference_mode():
        lg, cache = model._run(model.embed_tokens(ids), None, need_logits=True)
    return lg, cache


def step(model, tok: int, cache):
    with torch.inference_mode():
        lg, c2 = model._run(model.embed_tokens(torch.tensor([[tok]])), cache, need_logits=True)
    return lg, c2


def main() -> int:
    torch.set_grad_enabled(False)
    P, KP, V = 10, 4, 64
    tgt = tiny(0)
    ids = torch.randint(0, V, (1, P))
    cands = [int(x) for x in torch.tensor([3, 17, 42, 8])]
    lg_pre, cache0 = prefill(tgt, ids)

    print("== ①/② 位置关系：saved_next↔cands[0]，rows[j]↔cands[j+1] ==")
    check("prefill cache 长度 == P", lens(cache0) == [P] * len(cache0), str(lens(cache0)))
    saved_next = lg_pre[:, -1:]
    check("saved_next 是 prefill 最后一行（位置 P−1）",
          bool(torch.equal(saved_next, lg_pre[:, P - 1:P])))
    check("saved_next 回答的是 cands[0]（它预测位置 P 的 token）",
          saved_next.shape == (1, 1, V), str(tuple(saved_next.shape)))

    cache_for_verify = clone_cache(cache0)
    rows, cache_blk = spec_dec._verify(tgt, cache_for_verify, cands)
    check("_verify 返回 rows 形状 == (kp, V)", tuple(rows.shape) == (KP, V), str(tuple(rows.shape)))
    check("_verify 后 block cache 长度 == P + kp",
          lens(cache_blk) == [P + KP] * len(cache_blk), str(lens(cache_blk)))

    # rows[j] 应当与"cache 长度 P+j、单步输入 cands[j]"的 logits 是同一位置
    worst = 0.0
    for j in range(KP):
        c = clone_cache(cache0)
        for k in range(j):                       # 先把 cands[0..j-1] 喂进去
            _, c = step(tgt, cands[k], c)
        assert lens(c) == [P + j] * len(c), (j, lens(c))
        lg_j, _ = step(tgt, cands[j], c)         # 单步输入 cands[j] ⇒ 位置 P+j
        d = float((lg_j[0, -1].float() - rows[j].float()).abs().max())
        worst = max(worst, d)
        check(f"rows[{j}] == 位置 P+{j} 的单步 logits（仅形状不同）", d < 1e-4,
              f"max|Δ|={d:.3e}")
    print(f"  [info] 单步 vs block 的最大差（fp32 CPU，同位置不同形状）：{worst:.3e}")

    print("\n== ④ block 内因果性：同长度替换**未来**候选不得改变前面的行 ==")
    c1 = clone_cache(cache0)
    r1, _ = spec_dec._verify(tgt, c1, cands)
    cands2 = [cands[0], cands[1], 61, 62]        # 只改 index 2,3（未来），长度不变
    c2 = clone_cache(cache0)
    r2, _ = spec_dec._verify(tgt, c2, cands2)
    d01 = max(float((r1[0].float() - r2[0].float()).abs().max()),
              float((r1[1].float() - r2[1].float()).abs().max()))
    d23 = max(float((r1[2].float() - r2[2].float()).abs().max()),
              float((r1[3].float() - r2[3].float()).abs().max()))
    check("替换未来候选后 rows[0],rows[1] **逐位不变**（无因果泄漏）", d01 == 0.0, f"max|Δ|={d01:.3e}")
    check("而 rows[2],rows[3] 确实改变（说明替换真的生效了，不是空过）", d23 > 0.0,
          f"max|Δ|={d23:.3e}")

    print("\n== ⑤ 快照隔离：`_verify` 是否会就地修改传入的 cache ==")
    c3 = clone_cache(cache0)
    before = [int(d["k"].data_ptr()) for d in c3]
    before_len = lens(c3)
    spec_dec._verify(tgt, c3, cands)
    after = [int(d["k"].data_ptr()) for d in c3]
    after_len = lens(c3)
    print(f"  [info] 传入 dict 的 k 指针前={before[:2]} 后={after[:2]}；长度前={before_len[:2]} 后={after_len[:2]}")
    check("**传入的 cache 字典会被就地改写** ⇒ 两条分支必须各自 clone_cache()",
          after_len != before_len,
          "若此处 FAIL 说明引擎转为非就地，快照策略可简化")
    c4 = clone_cache(cache0)
    l0 = lens(c4)
    spec_dec._verify(tgt, c4, cands)
    check("clone 出的副本被修改，但**原始 cache0 不受影响**（浅拷贝足够）",
          lens(cache0) == [P] * len(cache0), f"cache0={lens(cache0)} 修改前={l0}")

    print("\n== ⑥ mask 偏移：block 第 j 行只能看到 ≤ P+j ==")
    # 用"只改位置 P+j+1 处的 token"验证：该改动不得影响第 j 行
    for j in range(KP - 1):
        a = clone_cache(cache0)
        ra, _ = spec_dec._verify(tgt, a, cands)
        c2v = list(cands)
        c2v[j + 1] = (cands[j + 1] + 7) % V
        b = clone_cache(cache0)
        rb, _ = spec_dec._verify(tgt, b, c2v)
        d = float((ra[j].float() - rb[j].float()).abs().max())
        check(f"改动 cands[{j + 1}] 不影响 rows[{j}]（因果边界正确）", d == 0.0, f"max|Δ|={d:.3e}")

    print(f"\nCPU 哨兵：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
