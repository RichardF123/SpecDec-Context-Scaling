from __future__ import annotations

import contextlib

import torch

from . import spec_cache

K = 4
FMT = {"base_bf16": "fp16", "v_fp8": "v_fp8"}

# ---------------------------------------------------------------------------
# **默认关闭**的 verify 包装点（全层 row-aligned verify 候选模式的唯一挂载处）
#
# 为什么放在这里：候选模式**只允许**作用于 SpecDec 的 **target block verify**（`_verify` 那一次
# s=Kp 的 `_run`），不得影响 AR/decode（`_step`）与 prefill。把挂载点放在 `_verify` 的调用处
# 是"作用域最小"的做法：AR 路径根本不经过这里。
#
# 默认 `None` ⇒ `spec_decode_greedy` 走**原样的一行**（`rows, tgt_blk = _verify(...)`），
# 生产路径逐位不变（由 Stage 0/1 的开关对照与既有 F13 记录的复现共同证明）。
# ---------------------------------------------------------------------------
_VERIFY_WRAP = None


def set_verify_wrap(fn) -> dict:
    """安装 verify 包装（`fn(target)` 返回上下文管理器）；传 `None` 卸载。默认不安装。"""
    global _VERIFY_WRAP
    prev = _VERIFY_WRAP
    _VERIFY_WRAP = fn
    return dict(installed=bool(fn is not None), had_previous=bool(prev is not None))


def verify_wrap_state() -> dict:
    return dict(active=bool(_VERIFY_WRAP is not None),
                note="默认 `None` ⇒ target block verify 走生产原样路径")


def _fmt(kv):
    return FMT[kv]


def _prefill(model, ids):
    with torch.inference_mode():
        lg, cache = model._run(model.embed_tokens(ids), None, need_logits=True)
    return lg, cache


def _propose(draft, cache, lg, k_eff, eos_id):
    cands = []
    with torch.inference_mode():
        for _ in range(k_eff):
            t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            cands.append(t)
            lg, cache = draft._run(
                draft.embed_tokens(torch.tensor([[t]], device=lg.device)),
                cache, need_logits=True)
            if t == eos_id:
                break
    return cands, cache, lg


def _verify(target, cache, cands):
    with torch.inference_mode():
        lg, cache_blk = target._run(
            target.embed_tokens(
                torch.tensor([cands], device=cache[0]["k"].device)),
            cache, need_logits=True)
    kp = len(cands)
    return lg[0, -kp:], cache_blk


def _step(model, token, cache):
    with torch.inference_mode():
        lg, cache = model._run(
            model.embed_tokens(
                torch.tensor([[token]], device=cache[0]["k"].device)),
            cache, need_logits=True)
    return lg[:, -1:], cache


def spec_decode_greedy(target, draft, prompt_ids, max_new_tokens, *, k=K,
                       eos_id=None, target_kv="base_bf16", draft_kv="base_bf16",
                       log=None):
    target.set_kv_format(_fmt(target_kv))
    draft.set_kv_format(_fmt(draft_kv))

    tgt_lg, tgt_cache = _prefill(target, prompt_ids)
    drf_lg, drf_cache = _prefill(draft, prompt_ids)
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]

    gen = []
    st = dict(rounds=0, proposed=0, accepted=0, target_block_calls=0, draft_calls=0,
              mismatch_rounds=0, all_accept_rounds=0, rollbacks=0)

    while len(gen) < max_new_tokens:
        k_eff = min(k, max_new_tokens - len(gen))
        pre = spec_cache.cache_length(tgt_cache)

        cands, drf_cache, prop_lg = _propose(draft, drf_cache, drf_lg, k_eff, eos_id)
        kp = len(cands)
        st["draft_calls"] += kp
        if kp == 0:
            break

        # 默认（`_VERIFY_WRAP is None`）：**原样**一行，生产路径逐位不变。
        # 安装包装时：候选模式**只**包住这一次 block verify（AR/decode 不经过这里）。
        if _VERIFY_WRAP is None:
            rows, tgt_blk = _verify(target, tgt_cache, cands)
        else:
            with _VERIFY_WRAP(target):
                rows, tgt_blk = _verify(target, tgt_cache, cands)
        st["target_block_calls"] += 1
        st["proposed"] += kp

        g = [int(x.argmax()) for x in [saved_next] + [rows[j] for j in range(kp - 1)]]
        m = 0
        while m < kp and g[m] == cands[m]:
            m += 1

        if m < kp:
            r = g[m]
            emitted = cands[:m] + [r]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + m)
            saved_next, tgt_cache = _step(target, r, tgt_cache)
            drf_cache = spec_cache.trim_cache(drf_cache, pre + m)
            drf_lg, drf_cache = _step(draft, r, drf_cache)
            st["mismatch_rounds"] += 1
            st["rollbacks"] += 1
        else:
            emitted = list(cands)
            tgt_cache = tgt_blk
            drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
            saved_next = rows[kp - 1]
            drf_lg = prop_lg
            st["all_accept_rounds"] += 1

        gen += emitted
        st["accepted"] += m
        st["rounds"] += 1
        if log is not None:
            log(dict(round=st["rounds"] - 1, Kp=kp, m=m, pre=pre,
                     tgt_len=spec_cache.cache_length(tgt_cache),
                     drf_len=spec_cache.cache_length(drf_cache),
                     cands=cands, g=g, emitted=emitted))
        if emitted and emitted[-1] == eos_id:
            break

    return {"ids": list(prompt_ids[0]) + gen, "gen": gen, "stats": st}
