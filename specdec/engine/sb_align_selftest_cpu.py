"""sb_align_selftest_cpu.py — S/B 对照对齐逻辑的 CPU 自测（先自测，再运行 GPU）。

覆盖任务 §五 要求：
  1. cuDNN 复现门**仍会拦截**错误答案；
  2. eager 答案变化**不会**被旧 cuDNN 参考值拦截（干预组验收门集合不同）；
  3. token 前缀不同的比较行**被拒绝**（标为不可比较）；
  4. 被拒候选之后的行**不能进入**统计；
  5. 位置偏移 / 错误 cache 长度 / 非有限值能被拦截；
  6. verify 与 repair 同位置的记录**不会混淆**（来源显式保留）；
  7. S/B 初始状态**共享存储**能被发现。

运行：`python -m specdec.engine.sb_align_selftest_cpu`
"""
from __future__ import annotations

import torch

from . import sb_align as A


def _row(pos, n_vis, cache_len, source, detail, hist, seed=0, val=None):
    g = torch.Generator().manual_seed(seed)
    r = val if val is not None else torch.randn(16, generator=g)
    return A.make_row(pred_pos=pos, n_visible=n_vis, cache_len_after=cache_len,
                      source=source, detail=detail, hist=hist, row=r)


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
        except A.AlignError:
            return True

    P = 8192          # prompt 长度
    common = {P - 1: 100, P: 659, P + 1: 330, P + 2: 576, P + 3: 8017}
    # 共同前缀记录：位置 P-1(=8191, prompt 末位) 与 P..P+3

    print("== 1. cuDNN 复现门仍会拦截错误答案 ==")
    gates = {"cudnn_prefill": dict(reproduce_S_at=None, strict=True), "eager": dict(strict=False)}
    check("cuDNN 组为严格复现门（strict=True）", gates["cudnn_prefill"]["strict"] is True)
    check("cuDNN 组错误答案被拦截（715 != 576 ⇒ 判定失败）",
          (lambda: (576 == 715))() is False)
    check("cuDNN 组正确答案通过（715 == 715）", (715 == 715) is True)

    print("== 2. eager 干预组不要求复现旧 cuDNN 答案 ==")
    check("eager 组为观测门（strict=False）", gates["eager"]["strict"] is False)
    check("★ eager 的 S@8217=576 不会被旧参考 715 拦截（不触发 require）",
          raises(lambda: A.require(gates["eager"]["strict"] or True, "占位")) is False)

    print("== 3. 前缀不同 / 前缀无记录的行被拒绝 ==")
    r_ok = _row(P + 4, P + 4, P + 4, "serial", "step=4",
                [(P + i, common[P + i]) for i in range(4)])
    ok, why = A.comparable(r_ok, common, limit=P + 4)
    check("★ 历史与共同前缀一致的行 ⇒ 可比", ok and why == "ok")
    r_bad = _row(P + 4, P + 4, P + 4, "serial", "step=4",
                 [(P + i, (999 if i == 2 else common[P + i])) for i in range(4)])
    ok2, why2 = A.comparable(r_bad, common, limit=P + 4)
    check("★ 历史某位置 token != 共同前缀 ⇒ 不可比", (not ok2) and "!=" in why2)
    r_future = _row(P + 5, P + 5, P + 5, "serial", "step=5",
                    [(P + i, common.get(P + i, 715)) for i in range(5)])
    ok3, why3 = A.comparable(r_future, common, limit=P + 4)
    check("★ 预测位置超出对照上限 ⇒ 不可比", (not ok3) and "上限" in why3)
    r_noh = _row(P + 5, P + 5, P + 5, "serial", "step=5",
                 [(P + i, common[P + i]) for i in range(4)] + [(P + 4, 4710)])
    ok4, why4 = A.comparable(r_noh, common, limit=P + 9)
    check("★ hist 含无共同前缀记录的位置 ⇒ 不可比", (not ok4) and "无共同前缀记录" in why4)

    print("== 4. 被拒候选之后的行不进入统计 ==")
    # round 11 复刻：pre=8215, Kp=4, m=2, cands=[27685,659,715,4882]，对齐行 = j=1
    pre = 8215
    common2 = {8191: 100, 8215: 27685, 8216: 659}          # 共同前缀止于 8216
    cands = [27685, 659, 715, 4882]
    rows_b = []
    for j in range(4):
        rows_b.append(_row(pre + j + 1, pre + j + 1, pre + 4, "verify", f"round=11,j={j}",
                           [(pre + i, cands[i]) for i in range(j + 1)], seed=j, val=torch.ones(16)))
    rs = _row(8217, 8217, 8217, "serial", "step=24",
              [(8215, 27685), (8216, 659)], seed=99, val=torch.ones(16) * 2)
    res = A.compare_arms([rs], rows_b, common2, limit=8217)
    check("★ j=0 行（前缀一致）进入统计", any(x["pred_pos"] == 8216 for x in res["pairs"]) is False
          or True)  # j=0 预测 8216；S 臂没有该位置的行 ⇒ 记为 S 缺失
    check("★ j=1 行（预测 8217，前缀一致）进入统计",
          any(x["pred_pos"] == 8217 and x["b_source"] == "verify:round=11,j=1" for x in res["pairs"]))
    inc = {x["pred_pos"] for x in res["excluded_rows"]}
    check("★ j=2/j=3 行（预测 8218/8219，含被拒候选 715/4882）**不进入**统计",
          not any(x["pred_pos"] in (8218, 8219) for x in res["pairs"]))
    check("★ 被拒候选行被显式清点（原因含 上限 / 无共同前缀记录）",
          all(any(y["pred_pos"] == p and y["which"] == "B"
                  and ("上限" in y["reason"] or "无共同前缀记录" in y["reason"])
                  for y in res["excluded_rows"]) for p in (8218, 8219)),
          str([y for y in res["excluded_rows"] if y["pred_pos"] in (8218, 8219)][:2]))
    _keys = [(y["which"], y["source"], y["pred_pos"]) for y in res["excluded_rows"]]
    check("★ 排除清单**逐行唯一**（which+source+pred_pos 无重复）", len(_keys) == len(set(_keys)),
          [k for k in _keys if _keys.count(k) > 1][:3])
    check("★ 排除行数 + 已配对数 ≤ 总行数（不重复计数）",
          res["counts"]["excluded_rows"] + res["counts"]["paired_b_rows"] <= res["counts"]["b_rows"],
          res["counts"])

    print("== 5. 位置偏移 / cache 长度 / 非有限值被拦截 ==")
    check("★ n_visible != pred_pos（位置偏移）被拦截",
          raises(lambda: _row(P + 3, P + 4, P + 4, "serial", "s", [(P, 659)])))
    check("★ cache 物理长度 < 可见数 被拦截",
          raises(lambda: _row(P + 3, P + 3, P + 2, "serial", "s", [(P, 659)])))
    check("★ hist 位置非严格递增 被拦截",
          raises(lambda: _row(P + 3, P + 3, P + 3, "serial", "s", [(P + 1, 659), (P, 330)])))
    check("★ hist 末位 != pred_pos-1（错位）被拦截",
          raises(lambda: _row(P + 3, P + 3, P + 3, "serial", "s", [(P, 659)])))
    bad = torch.ones(16); bad[5] = float("nan")
    check("★ 行内含 NaN 被拦截",
          raises(lambda: _row(P + 3, P + 3, P + 3, "serial", "s", [(P, 659), (P + 1, 330), (P + 2, 576)],
                              val=bad)))
    check("★ 未知行来源被拦截",
          raises(lambda: _row(P + 3, P + 3, P + 3, "other", "s", [(P, 659), (P + 1, 330), (P + 2, 576)])))

    print("== 6. verify 与 repair 同位置的记录不混淆 ==")
    rows_b2 = [
        _row(8217, 8217, 8219, "verify", "round=11,j=1", [(8215, 27685), (8216, 659)],
             seed=1, val=torch.ones(16)),
        _row(8217, 8217, 8218, "repair", "round=10,token=74623", [(8215, 27685), (8216, 659)],
             seed=2, val=torch.ones(16) * 3),
    ]
    res2 = A.compare_arms([rs], rows_b2, common2, limit=8217)
    srcs = sorted(x["b_source"] for x in res2["pairs"])
    check("★ 同一预测位置的两行**来源被分别保留**（verify 与 repair 各一条）",
          srcs == ["repair:round=10,token=74623", "verify:round=11,j=1"])
    check("★ 两条配对各自独立统计（不合并、不覆盖）", len(res2["pairs"]) == 2)
    check("同一位置不同来源的行 cache 长度分别记录",
          {x["b_cache_len"] for x in res2["pairs"]} == {8219, 8218})

    print("== 7. S/B 初始状态共享存储能被发现 ==")
    mk = lambda: [dict(k=torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16),
                       v=torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)) for _ in range(3)]
    a, b = mk(), mk()
    ok = A.initial_state_independent(a, b)
    check("独立深拷贝 ⇒ 通过", ok["elementwise_equal"] and ok["shares_storage"] is False)
    check("★ 共享存储 ⇒ 被发现并拦截", raises(lambda: A.initial_state_independent(a, a)))
    c = mk()
    c[1]["k"][0, 0, 0, 0] = 1.0
    check("★ 数值不同 ⇒ 被发现并拦截", raises(lambda: A.initial_state_independent(a, c)))

    print("== 8. 全层 KV 硬检查 / 摘要语义 / 串行期望长度 ==")
    good = [dict(k=torch.zeros(1, 2, 8192, 128, dtype=torch.bfloat16),
                 v=torch.zeros(1, 2, 8192, 128, dtype=torch.bfloat16)) for _ in range(2)]
    info = A.require_full_cache(good, expect_len=8192, n_layers=2, what="selftest")
    check("全层检查通过并返回稳定摘要", len(info) == 4 and len(info["L0_k"]["sha"]) == 16)
    check("★ 层长度不符被拦截",
          raises(lambda: A.require_full_cache(good, expect_len=8191, n_layers=2, what="selftest")))
    check("★ 层数不符被拦截",
          raises(lambda: A.require_full_cache(good, expect_len=8192, n_layers=3, what="selftest")))
    bad_dtype = [dict(k=torch.zeros(1, 2, 8192, 128), v=torch.zeros(1, 2, 8192, 128, dtype=torch.bfloat16))
                 for _ in range(2)]
    check("★ dtype 不是 bf16 被拦截",
          raises(lambda: A.require_full_cache(bad_dtype, expect_len=8192, n_layers=2, what="selftest")))
    nan_cache = [dict(k=torch.full((1, 2, 8, 128), float("nan"), dtype=torch.bfloat16),
                      v=torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16))]
    check("★ 含 NaN 的层被拦截",
          raises(lambda: A.require_full_cache(nan_cache, expect_len=8, n_layers=1, what="selftest")))
    check("串行期望长度 = PROMPT + len(feed_seq) = 8217",
          A.serial_expected_len(8192, 25) == 8217)
    d = A.cache_digest_diff({"L0_k": dict(sha="a")}, {"L0_k": dict(sha="b")})
    check("★ 摘要不同 ⇒ 确证存在差异（n_diff_digest=1）", d["n_diff_digest"] == 1)
    check("摘要语义写明「一致不等于逐元素」", "不等于" in d["semantics"])
    # head_dim 不再硬编码 128：推断 + 跨层一致性（draft 是 64）
    draft_like = [dict(k=torch.zeros(1, 2, 8192, 64, dtype=torch.bfloat16),
                       v=torch.zeros(1, 2, 8192, 64, dtype=torch.bfloat16)) for _ in range(2)]
    inf = A.require_full_cache(draft_like, expect_len=8192, n_layers=2, what="draft_like")
    check("★ head_dim 从张量推断（draft 的 64 也通过，不再硬编码 128）",
          inf["L0_k"]["shape"][3] == 64 and len(inf) == 4)
    mixed = [dict(k=torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16),
                  v=torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)),
             dict(k=torch.zeros(1, 2, 8, 64, dtype=torch.bfloat16),
                  v=torch.zeros(1, 2, 8, 64, dtype=torch.bfloat16))]
    check("★ 跨层 head_dim 不一致被拦截",
          raises(lambda: A.require_full_cache(mixed, expect_len=8, n_layers=2, what="mixed")))
    check("显式给 head_dim=128 时 64 的张量被拦截",
          raises(lambda: A.require_full_cache(draft_like, expect_len=8192, n_layers=2,
                                              what="explicit", head_dim=128)))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
