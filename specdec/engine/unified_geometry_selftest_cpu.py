"""unified_geometry_selftest_cpu.py — UCG 原型的 CPU / 短输入测试（**GPU 之前必须过**）。

覆盖设计文档 §9：
  1. **padding 不变性**（关键）：同一 cache + 同一 row-0 token，**不同 pad 内容**下 row 0 的 logits
     必须**逐位相同**。用 `model_self` 的**微型模型**在 CPU 上实测全链路（q/k/v → RoPE → cache →
     mask → softmax → AV → o_proj → 后续层 → final norm → LM head），不是"应该相同"的断言。
  2. **trim 正确性**：padded 调用 + trim 后，长度与已提交位置的 K/V 与"非 padded 单步"一致。
  3. **尾轮 / 接受 / 拒绝 / rollback / EOS 的计划表**：`k_eff<4`、提交数、pad 行不提交。
  4. **有限性 / 位置 / 长度 / 快照未修改**。

运行：`python -m specdec.engine.unified_geometry_selftest_cpu`
"""
from __future__ import annotations

import torch

from . import model_self as M
from . import spec_dec as _SD
from . import spec_cache
from . import unified_geometry as U

HID, HEADS, KVH, HD, INTER, LAYERS, VOCAB = 32, 4, 2, 8, 64, 2, 128
DT = torch.bfloat16


def _tiny():
    torch.manual_seed(0)
    m = M.QwenLM(VOCAB, HID, HEADS, KVH, HD, INTER, LAYERS, rope_theta=10000.0)
    return m.to(DT).eval()


def _run(model, ids, cache=None):
    with torch.inference_mode():
        return model._run(model.embed_tokens(ids), cache, need_logits=True)


def main() -> int:
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    def raises(fn):
        try:
            fn()
            return False
        except U.URError:
            return True

    print("== 1. 计划表：几何必须完全一致 ==")
    plans = [U.ar_step_plan(100, 7), U.repair_plan(100, 7), U.verify_plan(100, [1, 2, 3, 4]),
             U.tail_plan(100, [5, 6], k_eff=2)]
    g = U.require_geometry_uniform(plans)
    check("★ AR 步 / repair / verify / 尾轮 的 (s, T-pre, mask 对角线) 完全一致",
          g["s"] == 4 and g["T_minus_pre"] == 4 and g["mask_diagonal_minus_pre"] == 1, g)
    check("尾轮 k_eff=2 时喂入 4 个（pad 2 个）、只取前 2 行",
          plans[3]["fed"] == [5, 6, 0, 0] and plans[3]["take_rows"] == [0, 1]
          and plans[3]["n_real"] == 2)
    check("AR 步 row 0 可见 key ≤ pre（看不到 pad 行）", plans[0]["visible_last_key"] == 100)
    U.require_pad_rows_not_visible(plans[3])
    check("★ pad 行不被参与决策的行看见（因果检查通过）", True)
    check("★ 提交数超过真实行数被拦截", raises(lambda: U.require_commit_only_real(plans[3], 3)))
    U.require_commit_only_real(plans[3], 2)
    check("提交 2（= 真实行数）通过", True)

    print("== 2. padding 不变性（微型模型实测，全链路）==")
    m = _tiny()
    ids = torch.randint(0, VOCAB, (1, 6))
    lg0, cache = _run(m, ids)
    check("prefill 后 cache 长度 = 6", int(spec_cache.cache_length(cache)) == 6)
    tok = int(lg0[0, -1].argmax())
    outs = {}
    for tag, pads in (("zeros", [0, 0, 0]), ("ones", [1, 1, 1]),
                      ("repeat", [tok, tok, tok]), ("rand", [11, 23, 97])):
        c = [{k: v.clone() for k, v in d.items()} for d in cache]
        rows, blk = U.verify_padded(m, c, [tok] + pads, call=_SD._verify)
        outs[tag] = rows[0].detach().clone()
    base = outs["zeros"]
    same = {t: bool(torch.equal(base, v)) for t, v in outs.items()}
    check("★ 不同 pad 内容下 row 0 的 logits **逐位相同**（证明 pad 不影响决策行）",
          all(same.values()), same)
    # row 0 与非 padded 单步（s=1）应当**不同**（正是要消除的差异），如实打印
    c1 = [{k: v.clone() for k, v in d.items()} for d in cache]
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_s1, _ = m._run(m.embed_tokens(torch.tensor([[tok]])), c1, need_logits=True)
    lg_s1_row = lg_s1[0, -1].detach().float()
    d = (base.detach().float() - lg_s1_row).abs().max()
    print(f"    （诊断信息）s=4 的 row 0 与 s=1 单步的最大绝对差 = {float(d):.6g}"
          f" —— 这正是 UCG 要消除的形状差异，不是本测试的失败条件")
    check("★ padding 不变性**逐位**成立（本候选的前提）", all(same.values()))

    print("== 3. trim 正确性：padded+trim 与单步提交的 K/V 一致 ==")
    for name, maker in (("AR 步", lambda mm, cc, t: U.step_padded(mm, cc, t, call=_SD._verify,
                                                                 phase="decode")),
                        ("repair", lambda mm, cc, t: U.step_padded(mm, cc, t, call=_SD._verify,
                                                                   phase="repair"))):
        c4 = [{k: v.clone() for k, v in d.items()} for d in cache]
        c1 = [{k: v.clone() for k, v in d.items()} for d in cache]
        lg4, b4 = maker(m, c4, tok)
        with torch.inference_mode(), M.attention_phase("decode"):
            lg1, b1 = _run(m, torch.tensor([[tok]]), c1)
        check(f"{name}: trim 后长度 = pre+1", int(spec_cache.cache_length(b4)) == 7,
              int(spec_cache.cache_length(b4)))
        kv_eq = all(torch.equal(b4[i][k], b1[i][k]) for i in range(LAYERS) for k in ("k", "v"))
        check(f"{name}: 提交位置(0..6)的 K/V 与单步**逐位相同**（pad 行的 KV 未进入有效 cache）",
              kv_eq)
        check(f"{name}: row 0 logits 形状为 (1,1,V)", tuple(lg4.shape) == (1, 1, VOCAB),
              tuple(lg4.shape))
        check(f"{name}: 有限", bool(torch.isfinite(lg4.float()).all()))

    print("== 4. verify 包装：pad 候选后只回传真实行 ==")
    c = [{k: v.clone() for k, v in d.items()} for d in cache]
    rows2, blk2 = U.verify_padded(m, c, [tok, 3], call=_SD._verify)
    check("回传行数 = len(cands) = 2", tuple(rows2.shape) == (2, VOCAB), tuple(rows2.shape))
    check("blk 长度为 pre+4（生产随后按 pre+m trim）", int(spec_cache.cache_length(blk2)) == 10,
          int(spec_cache.cache_length(blk2)))
    trimmed = spec_cache.trim_cache(blk2, 6 + 1)
    check("trim 到 pre+m=7 后长度正确", int(spec_cache.cache_length(trimmed)) == 7)

    print("== 5. 计划表：接受 / 拒绝 / rollback / EOS / 尾轮 ==")
    for cands, g, emitted, m_exp in ([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], 4],
                                     [[1, 2, 3, 4], [1, 9, 3, 4], [1, 9], 1],
                                     [[1, 2, 3, 4], [7, 2, 3, 4], [7], 0]):
        from . import decision_rows as DR
        al = DR.align_decision(pre=100, kp=4, cands=cands, g=g, emitted=emitted)
        check(f"cands={cands} g={g} ⇒ m={al['m']}（期望 {m_exp}）", al["m"] == m_exp)
    tp = U.tail_plan(100, [5], k_eff=1)
    U.require_commit_only_real(tp, 1)
    check("尾轮 k_eff=1：喂 4 个、只取 row 0、提交 ≤1", tp["fed"] == [5, 0, 0, 0]
          and tp["take_rows"] == [0])
    check("★ 尾轮几何与 AR 步/verify 一致（不因 k_eff 改变 s）",
          (tp["s"], tp["T"] - tp["pre"]) == (4, 4))

    print("== 6. 快照与输入未被意外修改 ==")
    snap = [{k: v.clone() for k, v in d.items()} for d in cache]
    c = [{k: v.clone() for k, v in d.items()} for d in cache]
    U.step_padded(m, c, tok, call=_SD._verify)
    check("★ 源快照内容未被修改（padded 调用只改传入的 cache）",
          all(torch.equal(cache[i][k], snap[i][k]) for i in range(LAYERS) for k in ("k", "v")))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
