from __future__ import annotations

import torch

from . import spec_cache

K = 4
FMT = {"base_bf16": "fp16", "v_fp8": "v_fp8"}


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
                torch.tensor([cands], device=spec_cache.cache_device(cache))),
            cache, need_logits=True)
    kp = len(cands)
    return lg[0, -kp:], cache_blk


def _step(model, token, cache):
    with torch.inference_mode():
        lg, cache = model._run(
            model.embed_tokens(
                torch.tensor([[token]], device=spec_cache.cache_device(cache))),
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
