"""attn_phase_selftest_cpu.py — prefill-only cuDNN attention 分支的**纯 CPU** 自测。

不加载真实权重、不碰 GPU：用 tiny `QwenLM`（随机权重、CPU）跑**真实的 `spec_dec.spec_decode_greedy`**，
并把融合 kernel 替换成**记录用的 stub** —— 这样能证明"**调用时机**"，而不是只测纯函数。

覆盖（对应任务的验证顺序第 1 步）
  1. 掩码等价：`_kv_keep_mask` 与 eager 的 `masked_fill` 语义在 全部 情形一致
     （方阵 / s==1 / **非方阵缓存偏移**）；
  2. 阶段分流裁决 `decide_attention_path` 的所有组合；
  3. **真实调用序列**上证明：fused **只**发生在 `prefill`；`propose/verify/repair` 全 eager
     —— 尤其覆盖"候选验证实际长度 1/2/3/4"；
  4. 短块策略：prefill 里 s==1 的末块 → eager **且记录原因**（不静默）；
  5. `assert_strategy_used()`：策略已切但没有任何 prefill 标注 → **必须抛错**（防静默不生效）；
  6. 传给 SDPA 的 kwargs：**不含 `is_causal`**、且 k/v **保持 kv_heads 未展开**（GQA 交给 enable_gqa）；
  7. 默认 `eager` 时融合函数**从不被调用**（行为与历史一致）。

用法：python -m specdec.engine.attn_phase_selftest_cpu
"""
from __future__ import annotations

import traceback

import torch

from . import model_self as M
from . import spec_dec

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def tiny_model(seed: int = 0, heads: int = 4, kv_heads: int = 2, head_dim: int = 8):
    torch.manual_seed(seed)
    return M.QwenLM(vocab_size=64, hidden=heads * head_dim, heads=heads, kv_heads=kv_heads,
                    head_dim=head_dim, intermediate=2 * heads * head_dim, num_layers=2)


class FusedRecorder:
    """替换 `_fused_prefill_attention`：只记录调用与形状，不真的算融合 kernel。"""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, q, k, v, head_dim):
        self.calls.append(dict(phase=M.current_attention_phase(),
                               s=int(q.shape[2]), T=int(k.shape[2]),
                               q_heads=int(q.shape[1]), kv_heads=int(k.shape[1]),
                               head_dim=int(head_dim)))
        # 用与 eager 等价的数学产出同形状结果，保证后续 o_proj 能继续跑
        return _eager_like(q, k, v, head_dim)


def _eager_like(q, k, v, head_dim):
    """stub 的替代实现：用与 eager 等价的数学（在 CPU 上也能算）产出同形状张量。"""
    kk = M.kv_repeat(k, q.shape[1] // k.shape[1])
    vv = M.kv_repeat(v, q.shape[1] // k.shape[1])
    s, T = q.shape[2], k.shape[2]
    scores = torch.matmul(q, kk.transpose(-2, -1)) / (head_dim ** 0.5)
    drop = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool), diagonal=(T - s) + 1)
    scores = scores.masked_fill(drop, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, vv)


def label_phases():
    """把 spec_dec 的四个入口包上**显式阶段标注**（harness 侧，不改 spec_dec 本身）。

    **为什么 `_prefill` 也必须包**：`spec_dec._prefill` 是**直接**调 `model._run` 的
    （`spec_dec.py:17`），它**不经过** `model_self.chunked_prefill`；不包标签时 phase 落在
    `'other'` ⇒ `decide_attention_path` 判定"非 prefill" ⇒ **静默不融合**。
    这次自测第一轮就真的踩到了这个坑（fused=0），`assert_strategy_used()` 正是它的报警器。
    真实测量工具里 prefill 也由 harness 包装（`timed_chunked_prefill` / 各 probe 的 prefill 包装），
    所以在这里包标签 = 复刻真实调用方式。
    """
    orig = dict(prefill=spec_dec._prefill, propose=spec_dec._propose,
                verify=spec_dec._verify, step=spec_dec._step)

    def prefill(*a, **k):
        with M.attention_phase("prefill"):
            return orig["prefill"](*a, **k)

    def propose(*a, **k):
        with M.attention_phase("propose"):
            return orig["propose"](*a, **k)

    def verify(*a, **k):
        with M.attention_phase("verify"):
            return orig["verify"](*a, **k)

    def step(*a, **k):
        with M.attention_phase("repair"):
            return orig["step"](*a, **k)

    spec_dec._prefill, spec_dec._propose, spec_dec._verify, spec_dec._step = (
        prefill, propose, verify, step)
    return orig


def restore_phases(orig):
    spec_dec._prefill, spec_dec._propose, spec_dec._verify, spec_dec._step = (
        orig["prefill"], orig["propose"], orig["verify"], orig["step"])


def run_specdec(strategy: str, prompt_len: int = 20, max_new: int = 8,
                tgt_seed: int = 0, drf_seed: int = 1):
    """跑一次真实投机解码（CPU、tiny 模型），返回 (out, stats, fused_calls)。"""
    tgt, drf = tiny_model(tgt_seed), tiny_model(drf_seed)
    ids = torch.randint(0, 64, (1, prompt_len))
    M.set_attention_strategy(strategy)
    M.reset_attention_stats()
    rec = FusedRecorder()
    real_fused, M._fused_prefill_attention = M._fused_prefill_attention, rec
    orig = label_phases()
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, max_new, k=4, eos_id=None,
                                         target_kv="base_bf16")
    finally:
        restore_phases(orig)
        M._fused_prefill_attention = real_fused
    return out, M.attention_stats(), rec.calls


def main() -> int:
    real_fused = M._fused_prefill_attention

    print("== 1. 掩码等价：`_kv_keep_mask` vs eager 的 masked_fill 语义 ==")
    for (s, T) in ((8, 8), (5, 32), (1, 32), (1, 8192), (2048, 8192), (3, 7)):
        keep = M._kv_keep_mask(s, T, torch.device("cpu"))
        drop = ~keep
        exp = torch.zeros(s, T, dtype=torch.bool)
        for i in range(s):
            exp[i] = torch.arange(T) > (T - s) + i          # j <= (T-s)+i 才允许
        check(f"① 掩码 s={s:<5} T={T:<5} 与 eager 语义逐元素一致", bool((drop == exp).all()))
    check("① s==1 时全部允许（j ≤ T−1）",
          bool(M._kv_keep_mask(1, 8192, torch.device("cpu")).all()))
    check("① 方阵时等于标准因果",
          bool((~M._kv_keep_mask(6, 6, torch.device("cpu"))
                == torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)).all()))

    print("\n== 2. 阶段分流裁决（纯函数）==")
    check("② 策略=eager ⇒ 永远 eager（连 prefill 也是）",
          all(M.decide_attention_path(p, s, "eager")[0] == "eager"
              for p in ("prefill", "verify", "decode", "propose", "repair", "other")
              for s in (1, 4, 2048)))
    check("② 策略=cudnn + phase=prefill + s≥2 ⇒ fused",
          M.decide_attention_path("prefill", 2048, "cudnn_prefill") == ("fused", None))
    for ph in ("verify", "decode", "propose", "repair", "other"):
        for s in (1, 2, 4, 2048):
            got, why = M.decide_attention_path(ph, s, "cudnn_prefill")
            if not (got == "eager" and why and "!= 'prefill'" in why):
                check(f"② {ph}/s={s} 必须 eager", False, f"{got}/{why}")
                break
        else:
            check(f"② phase={ph:<8} 任意 s（含多 token）⇒ 恒 eager", True)
    check("② prefill 里的 s==1 末块 ⇒ eager 且有原因",
          M.decide_attention_path("prefill", 1, "cudnn_prefill")[0] == "eager"
          and "seq_len=1" in M.decide_attention_path("prefill", 1, "cudnn_prefill")[1])

    print("\n== 3. 真实调用序列上的阶段分流（tiny 模型，CPU，stub 掉融合 kernel）==")
    out_s, st_s, calls_s = run_specdec("cudnn_prefill")
    print(f"  [info] spec_dec 轮次统计：{out_s['stats']}")
    print(f"  [info] by_phase={st_s['by_phase']} s_max={st_s['s_max_by_phase']}")
    fused_phases = sorted(k for k, v in st_s["fused_by_phase"].items() if v > 0)
    check("③ fused **只**发生在 prefill（其余阶段 fused 计数恒为 0）",
          fused_phases == ["prefill"], str(st_s["fused_by_phase"]))
    check("③ 至少一次 prefill 融合调用", len(calls_s) >= 1, f"n={len(calls_s)}")
    check("③ 每个 fused 调用的 phase 都是 'prefill'",
          bool(calls_s) and all(c["phase"] == "prefill" for c in calls_s),
          str(sorted({c["phase"] for c in calls_s})))
    check("③ 送进融合 kernel 的 s 全部 ≥2（s=1 从不进 kernel）",
          bool(calls_s) and all(c["s"] >= M.PREFILL_FUSED_MIN_S for c in calls_s),
          str(sorted({c["s"] for c in calls_s})))
    check("③ 融合调用覆盖 target+draft 两侧 prefill（次数为 2 的倍数）",
          len(calls_s) % 2 == 0, f"n={len(calls_s)}")
    check("③ 真实序列里 verify 确实出现多 token（否则「未误用」是空证明）",
          st_s["s_max_by_phase"].get("verify", 0) > 1,
          f"verify max_s={st_s['s_max_by_phase'].get('verify')}")
    check("③ 真实序列里 propose 恒为 s==1",
          st_s["s_max_by_phase"].get("propose", 0) == 1,
          f"propose max_s={st_s['s_max_by_phase'].get('propose')}")
    for ph in ("propose", "verify", "repair"):
        n_e = st_s["eager_by_phase"].get(ph, 0)
        n_f = st_s["fused_by_phase"].get(ph, 0)
        check(f"③ phase={ph:<8} 若被触发则必然全 eager（本次 eager={n_e} fused={n_f}）",
              n_f == 0, f"eager={n_e} fused={n_f}")

    print("\n== 3b. 逐阶段确定性证明：验证长度 1/2/3/4 与 decode/repair 全 eager ==")
    tgt_b = tiny_model(0)
    ids_b = torch.randint(0, 64, (1, 20))
    M.set_attention_strategy("cudnn_prefill")
    orig_b = label_phases()
    try:
        # 历史缓存的构建**不标注阶段**（落在 'other' ⇒ eager）：3b 只需要一段 KV 历史，
        # 而 CPU 上没有可用的融合后端，误标 prefill 会（正确地）直接抛 RuntimeError。
        _, base_cache = tgt_b._run(tgt_b.embed_tokens(ids_b), None, need_logits=True)
        nl = len(base_cache)
        for kp in (1, 2, 3, 4):
            M.reset_attention_stats()
            # 浅拷贝每层 dict：eager 路径会**就地写回** cache["k"]/["v"]（model_self.py:329），
            # 但张量本身从不被就地改写 ⇒ 浅拷贝即可得到互不干扰的历史缓存
            spec_dec._verify(tgt_b, [dict(d) for d in base_cache],
                             [int(x) for x in torch.randint(0, 64, (kp,))])
            spec_dec._step(tgt_b, 7, [dict(d) for d in base_cache])
            tgt_b._run(tgt_b.embed_tokens(torch.tensor([[5]])),
                       [dict(d) for d in base_cache], need_logits=True)   # 未标注 → 'other'
            with M.attention_phase("decode"):
                tgt_b._run(tgt_b.embed_tokens(torch.tensor([[5]])),
                           [dict(d) for d in base_cache], need_logits=True)
            st_b = M.attention_stats()
            check(f"③b kp={kp}：verify 的 s=={kp} 且**全 eager**（fused=0）",
                  st_b["s_max_by_phase"].get("verify", 0) == kp
                  and st_b["fused_by_phase"].get("verify", 0) == 0
                  and st_b["eager_by_phase"].get("verify", 0) == nl,
                  f"max_s={st_b['s_max_by_phase'].get('verify')} "
                  f"eager={st_b['eager_by_phase'].get('verify')} "
                  f"fused={st_b['fused_by_phase'].get('verify', 0)}")
            check(f"③b kp={kp}：repair/decode/未标注 全 eager，全场 fused=0",
                  st_b["fused_by_phase"].get("repair", 0) == 0
                  and st_b["eager_by_phase"].get("repair", 0) == nl
                  and st_b["fused_by_phase"].get("decode", 0) == 0
                  and st_b["eager_by_phase"].get("decode", 0) == nl
                  and st_b["fused_by_phase"].get("other", 0) == 0
                  and st_b["eager_by_phase"].get("other", 0) == nl
                  and st_b["calls"]["fused"] == 0,
                  f"repair={st_b['eager_by_phase'].get('repair')} "
                  f"decode={st_b['eager_by_phase'].get('decode')} "
                  f"other={st_b['eager_by_phase'].get('other')} "
                  f"总 fused={st_b['calls']['fused']}")
    finally:
        restore_phases(orig_b)

    print("\n== 4. 传给 kernel 的 kwargs / 形状（GQA 未展开、不用 is_causal）==")
    captured = {}
    real_sdpa, M._sdpa = M._sdpa, None
    try:
        def spy(q, k, v, **kw):
            captured.update(kwargs=dict(kw), q_heads=int(q.shape[1]), kv_heads=int(k.shape[1]),
                            s=int(q.shape[2]), T=int(k.shape[2]))
            return _eager_like(q, k, v, q.shape[-1])

        M._sdpa = spy
        q = torch.randn(1, 12, 2048, 128)
        k = torch.randn(1, 2, 8192, 128)
        v = torch.randn(1, 2, 8192, 128)
        M._fused_prefill_attention(q, k, v, 128)
    finally:
        M._sdpa = real_sdpa
    kw = captured.get("kwargs", {})
    check("④ 未使用 is_causal（非方阵不等价，必须用 attn_mask）", "is_causal" not in kw,
          str(sorted(kw)))
    check("④ 使用 bool attn_mask 且形状为 (s,T)",
          kw.get("attn_mask") is not None and tuple(kw["attn_mask"].shape) == (2048, 8192),
          str(None if kw.get("attn_mask") is None else tuple(kw["attn_mask"].shape)))
    check("④ enable_gqa=True（GQA 交给 kernel，不做 kv_repeat 展开）", kw.get("enable_gqa") is True)
    check("④ k/v 保持 kv_heads 未展开（2，而非 12）",
          captured.get("kv_heads") == 2 and captured.get("q_heads") == 12,
          f"q={captured.get('q_heads')} kv={captured.get('kv_heads')}")
    check("④ scale = head_dim^-0.5", abs(kw.get("scale", 0) - 128 ** -0.5) < 1e-12)

    print("\n== 5. 防静默失效 / 短块策略 / 默认行为 ==")
    M.set_attention_strategy("cudnn_prefill")
    M.reset_attention_stats()
    with M.attention_phase("prefill"):          # 什么都不调用
        pass
    try:
        M.assert_strategy_used()
        check("⑤ 策略已切但零 fused ⇒ 必须抛错（防静默不生效）", False)
    except RuntimeError as e:
        check("⑤ 策略已切但零 fused ⇒ 必须抛错（防静默不生效）", "静默不生效" in str(e))
    out_e, st_e, calls_e = run_specdec("eager")
    check("⑤ 策略=eager ⇒ 融合函数一次都没被调用（历史行为不变）",
          not calls_e and st_e["calls"]["fused"] == 0 and st_e["calls"]["eager"] > 0,
          f"fused={st_e['calls']['fused']} eager={st_e['calls']['eager']}")
    # 人为制造 prefill 末块 s==1：prompt 长度 = chunk+1
    M.set_attention_strategy("cudnn_prefill")
    M.reset_attention_stats()
    tgt = tiny_model(0)
    ids = torch.randint(0, 64, (1, 9))
    rec = FusedRecorder()
    real2, M._fused_prefill_attention = M._fused_prefill_attention, rec
    try:
        M.chunked_prefill(tgt, ids, chunk=8)     # 两块：8 + 1
    finally:
        M._fused_prefill_attention = real2
    st_short = M.attention_stats()
    check("⑤ 末块 s=1 ⇒ 回退 eager 且**记录原因**（不静默）",
          {c["s"] for c in rec.calls} == {8} and st_short["fallback_count"] >= 1
          and any("seq_len=1" in (f["reason"] or "") for f in st_short["fallbacks"]),
          f"fused_s={sorted({c['s'] for c in rec.calls})} "
          f"n_fused={len(rec.calls)} fallbacks={st_short['fallback_count']}")
    check("⑤ 回退记录里带 stage 形状与 policy",
          all({"phase", "s", "T", "reason", "policy", "heads", "kv_heads", "head_dim"}
              <= set(f) for f in st_short["fallbacks"]))

    M.set_attention_strategy("eager")
    M.reset_attention_stats()
    M._fused_prefill_attention = real_fused

    print(f"\nCPU 自测：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
