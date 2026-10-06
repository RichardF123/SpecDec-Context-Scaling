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

    # ---- 6. `assert_strategy_used()` 与既有分派规则的一致性（2026-10-05 修正的**反例回归**）----
    # 修正范围：**只**修断言与"既有分派规则"的矛盾；不改分派、阈值、精度、数值路径。
    # 判定要求：调用长度 / 回退理由 / 计数**三者一致**才算合法回退；
    #           "合法回退"≠"融合已通过"（必须显式给出 fused_path_covered=False）。
    print("\n== 6. 策略断言与分派规则一致性（short 合法回退 vs 真矛盾）==")

    def _fake_stats(**kw) -> dict:
        """直接构造统计（不跑模型），逐条覆盖 assert 的判定分支。"""
        st = M._fresh_stats()
        st.update({k: v for k, v in kw.items()})
        st["strategy"] = "cudnn_prefill"
        return st

    def _assert_with(st: dict) -> tuple:
        """把给定统计装进 `_ATTENTION_STATS` 后调用断言，返回 (是否抛, 消息/返回值)。"""
        saved = M._stats()
        M.set_attention_strategy("cudnn_prefill")
        try:
            M._ATTENTION_STATS = st                                     # noqa: SLF001（测试内直填）
            try:
                return False, M.assert_strategy_used()
            except RuntimeError as e:
                return True, str(e)
        finally:
            M._ATTENTION_STATS = saved
            M.set_attention_strategy("eager")

    def _prefill_only_stats(*, s: int, eager_calls: int, short_reasons: int, fused: int,
                            other_kinds: dict | None = None, extra_fused_phase=None) -> dict:
        st = M._fresh_stats()
        st["calls"] = dict(eager=eager_calls, fused=fused)
        st["by_phase"] = {"prefill": eager_calls + fused}
        st["eager_by_phase"] = {"prefill": eager_calls} if eager_calls else {}
        st["fused_by_phase"] = ({"prefill": fused} if fused else {})
        st["s_max_by_phase"] = {"prefill": s} if (eager_calls + fused) else {}
        st["fused_s_max_by_phase"] = ({"prefill": s} if fused else {})
        st["eager_reason_kinds"] = dict(other_kinds or {})
        if short_reasons:
            st["eager_reason_kinds"]["short_block_seq_len_lt_min"] = short_reasons
        st["fallback_count"] = eager_calls
        st["phases_seen"] = ["prefill"]
        if extra_fused_phase:
            st["fused_by_phase"].update(extra_fused_phase)
            st["calls"]["fused"] += sum(extra_fused_phase.values())
        return st

    # (1) 全部 prefill 调用都短 + 理由计数一致 ⇒ **合法回退，不抛**，且明确"融合路径未覆盖"
    raised, res = _assert_with(_prefill_only_stats(s=1, eager_calls=2, short_reasons=2, fused=0))
    check("⑥a 全部 prefill s<min 且理由/计数一致 ⇒ 不抛（合法回退）", raised is False, res if raised else "")
    check("⑥b 合法回退必须显式标注 fused_path_covered=False（≠ 融合已通过）",
          (not raised) and res.get("strategy_check", {}).get("verdict") == "legal_short_block_fallback"
          and res["strategy_check"].get("fused_path_covered") is False,
          "" if raised else res.get("strategy_check", {}).get("verdict"))

    # (2) 存在满足融合条件的 prefill 调用却零融合 ⇒ 仍抛
    raised, msg = _assert_with(_prefill_only_stats(s=8, eager_calls=1, short_reasons=0, fused=0))
    check("⑥c 有满足融合条件的 prefill（s≥min）却零融合 ⇒ 必须抛",
          raised and "满足融合条件" in msg, msg)

    # (3) 混合长短：短块记了理由，但长块没融合 ⇒ 也抛（不得凭"存在 short 理由"整体豁免）
    raised, msg = _assert_with(_prefill_only_stats(s=8, eager_calls=3, short_reasons=1, fused=0))
    check("⑥d 混合长短且长块未融合 ⇒ 必须抛（不得整体豁免）",
          raised and "满足融合条件" in msg, msg)

    # (4) 理由缺失 / 计数不一致 ⇒ 抛（存在未解释的回退）
    raised, msg = _assert_with(_prefill_only_stats(s=1, eager_calls=2, short_reasons=0, fused=0))
    check("⑥e 短块回退但**无理由记录** ⇒ 必须抛",
          raised and "未解释的回退" in msg, msg)
    raised, msg = _assert_with(_prefill_only_stats(s=1, eager_calls=3, short_reasons=1, fused=0))
    check("⑥f 短块回退理由计数与 prefill eager 调用数**不一致** ⇒ 必须抛",
          raised and "未解释的回退" in msg, msg)
    raised, msg = _assert_with(_prefill_only_stats(
        s=1, eager_calls=2, short_reasons=2, fused=0, other_kinds={"other": 1}))
    check("⑥g 存在 kind=other 的未解释回退（计数不一致）⇒ 必须抛",
          raised and "未解释的回退" in msg, msg)

    # (5) 非 prefill 阶段融合 ⇒ 仍抛
    raised, msg = _assert_with(_prefill_only_stats(s=8, eager_calls=0, short_reasons=0, fused=1,
                                                   extra_fused_phase={"verify": 1}))
    check("⑥h 非 prefill 阶段出现融合 ⇒ 必须抛",
          raised and "非 prefill" in msg, msg)

    # (6) 零 prefill 调用（忘了标注阶段）⇒ 仍抛（旧报警意图保留）
    st_none = M._fresh_stats()
    st_none["phases_seen"] = ["prefill"]
    raised, msg = _assert_with(st_none)
    check("⑥i prefill 阶段零调用（忘标注）⇒ 必须抛", raised and "静默不生效" in msg, msg)

    # (7) 真的融合 ⇒ 返回 ok_with_fusion 且 fused_path_covered=True
    raised, res = _assert_with(_prefill_only_stats(s=8, eager_calls=1, short_reasons=0, fused=2))
    check("⑥j 真的发生 prefill 融合 ⇒ 不抛且 fused_path_covered=True",
          (not raised) and res["strategy_check"]["verdict"] == "ok_with_fusion"
          and res["strategy_check"]["fused_path_covered"] is True, "" if raised else res)

    # (8) 端到端（真跑一次短输入）：断言不抛，且统计里能读出"融合未覆盖"
    M.set_attention_strategy("cudnn_prefill")
    M.reset_attention_stats()
    tgt2 = tiny_model(0)
    real3, M._fused_prefill_attention = M._fused_prefill_attention, FusedRecorder()
    try:
        M.chunked_prefill(tgt2, torch.randint(0, 64, (1, 1)), chunk=8)     # 单块 s=1
        st_e2e = M.assert_strategy_used()
    finally:
        M._fused_prefill_attention = real3
        M.set_attention_strategy("eager")
        M.reset_attention_stats()
    check("⑥k 端到端 1-token prompt：断言不抛、融合未发生且理由/计数一致",
          st_e2e["strategy_check"]["verdict"] == "legal_short_block_fallback"
          and st_e2e["calls"]["fused"] == 0
          and st_e2e["eager_reason_kinds"].get("short_block_seq_len_lt_min")
          == st_e2e["eager_by_phase"].get("prefill")
          and st_e2e["eager_by_phase"].get("prefill", 0) > 0,
          st_e2e.get("strategy_check"))

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
