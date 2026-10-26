"""relaxed_specdec — margin-aware relaxed verification（机制诊断用）。

默认行为不变：`specdec/engine/spec_dec.py` 的 strict 路径未被修改。
本模块提供：

    decode(target, draft, prompt_ids, max_new_tokens, *, delta, ...)

接受规则（每个位置 j，按顺序；longest-accepted-prefix 语义）：
    candidate   = candidates[j]
    top1        = argmax(target_logits[j])
    gap         = target_logits[j, top1] - target_logits[j, candidate]
    accept      = (candidate == top1)                    若 delta == 0（严格）
                 或 (candidate == top1) or (gap <= delta) 若 delta > 0（宽松）
    一旦某位置拒绝：停止，发出 target_top1，rollback 并 repair；后续候选不得再接受。

严格基线 = Phase O strict executor（`execution_optimization.fast_spec_dec`）：
`delta == 0` 时**直接委托**给该 executor，保证 ids/stats/cache/stop 逐位一致；
所有预热/计时边界与其相同。relaxed 路径复用同一套 proposal/verify/repair 函数，
只替换 acceptance 判定（gap 判定在 timed 区只计算 top1_logit、candidate_logit、gap）。
"""
from __future__ import annotations

import time

import torch

from specdec.engine import spec_cache

from experiments.execution_optimization import fast_spec_dec as F


class RelaxedDecodeError(RuntimeError):
    pass


def accept_mask(g_t: torch.Tensor, cands_t: torch.Tensor, gap: torch.Tensor,
                delta: float) -> torch.Tensor:
    """位置级接受掩码（严格 top1 或 gap<=delta）。g_t/top1 与 cands_t 同形状 (kp,)。"""
    eq = (g_t == cands_t)
    return eq if delta == 0.0 else (eq | (gap <= torch.as_tensor(delta, dtype=gap.dtype,
                                                                 device=gap.device)))


def prefix_length(acc: torch.Tensor) -> torch.Tensor:
    """首个 False 之前 True 的个数（GPU 标量；longest accepted prefix）。"""
    return acc.to(torch.int64).cumprod(0).sum()


def _control_trace(round_index, kp, m, pre, cands, g, emitted, tgt_len, drf_len,
                   strict_count, relaxed_count, regret_sum):
    return dict(round=round_index, Kp=kp, m=m, pre=pre, cands=list(cands), g=list(g),
                emitted=list(emitted), tgt_len=tgt_len, drf_len=drf_len,
                strictly_matched=strict_count, relaxed_only=relaxed_count,
                regret_sum=regret_sum)


def decode(target, draft, prompt_ids, max_new_tokens, *, k=4, delta=0.0, eos_id=None,
           target_kv="base_bf16", draft_kv="base_bf16", log=None, timer=None,
           finite_check=False, capture_cache=False, measure_timing=False,
           prefill_chunk=2048):
    """relaxed SpecDec 解码。delta=0 委托 Phase O strict executor。"""
    if delta < 0:
        raise ValueError("delta 必须 >= 0")
    if delta == 0.0:
        out = F.spec_decode_greedy_optimized(
            target, draft, prompt_ids, max_new_tokens, k=k, eos_id=eos_id,
            target_kv=target_kv, draft_kv=draft_kv, log=log, timer=timer,
            finite_check=finite_check, capture_cache=capture_cache,
            measure_timing=measure_timing, prefill_chunk=prefill_chunk)
        out["delta"] = 0.0
        return out
    return _decode_relaxed(target, draft, prompt_ids, max_new_tokens, k=k, delta=delta,
                           eos_id=eos_id, target_kv=target_kv, draft_kv=draft_kv,
                           log=log, timer=timer, finite_check=finite_check,
                           capture_cache=capture_cache, measure_timing=measure_timing,
                           prefill_chunk=prefill_chunk)


def _decode_relaxed(target, draft, prompt_ids, max_new_tokens, *, k, delta, eos_id,
                    target_kv, draft_kv, log, timer, finite_check, capture_cache,
                    measure_timing, prefill_chunk):
    if timer is None:
        timer = F.StageTimer(enabled=False)
    ledger = F.SyncLedger()
    target.set_kv_format(F._fmt(target_kv))
    draft.set_kv_format(F._fmt(draft_kv))

    if measure_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with F._phase("prefill"):
            tgt_lg, tgt_cache = F._prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
        torch.cuda.synchronize()
        prefill_target_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with F._phase("prefill"):
            drf_lg, drf_cache = F._prefill_chunked(draft, prompt_ids, chunk=prefill_chunk)
        torch.cuda.synchronize()
        prefill_draft_s = time.perf_counter() - t0
        prefill_s = prefill_target_s + prefill_draft_s
        t_decode = time.perf_counter()
    else:
        with F._phase("prefill"):
            with timer.span("prefill_target", ledger):
                tgt_lg, tgt_cache = F._prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
            with timer.span("prefill_draft", ledger):
                drf_lg, drf_cache = F._prefill_chunked(draft, prompt_ids, chunk=prefill_chunk)
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]

    gen: list[int] = []
    st = dict(rounds=0, proposed=0, accepted=0, strictly_matched=0,
              relaxed_only_accepted=0, sum_target_logit_regret=0.0,
              max_target_logit_regret=0.0, mismatch_rounds=0, all_accept_rounds=0,
              target_block_calls=0, draft_calls=0, rollbacks=0)
    control = []

    while len(gen) < max_new_tokens:
        k_eff = min(k, max_new_tokens - len(gen))
        pre = spec_cache.cache_length(tgt_cache)
        ledger.rounds += 1

        with F._phase("propose"):
            with timer.span("propose", ledger):
                cands_t, drf_cache, prop_lg = F._propose_gpu(
                    draft, drf_cache, drf_lg, k_eff, ledger, finite_check)
        if eos_id is None:
            kp = k_eff
        else:
            hit = (cands_t == eos_id).nonzero()
            kp = int(hit[0].item()) + 1 if hit.numel() else k_eff
            ledger.sync += 1
        st["draft_calls"] += kp
        if kp == 0:
            break
        cands_t_kp = cands_t[:kp]

        with F._phase("verify"):
            with timer.span("verify", ledger):
                rows, tgt_blk = F._verify_tensor(target, tgt_cache, cands_t_kp)
                F._maybe_finite(rows, finite_check)
        st["target_block_calls"] += 1
        st["proposed"] += kp

        # 判定用 logits：g[0] 来自 saved_next，g[j>=1] 来自 rows[j-1]（与 strict executor 一致）
        rowsj = torch.cat([saved_next.reshape(1, -1), rows[:kp - 1]], dim=0)  # (kp, V)
        g_t = rowsj.argmax(dim=-1)
        top1_logit = rowsj.gather(1, g_t.view(-1, 1)).squeeze(1)
        cand_logit = rowsj.gather(1, cands_t_kp.view(-1, 1)).squeeze(1)
        gap = top1_logit - cand_logit
        acc = accept_mask(g_t, cands_t_kp, gap, delta)
        m_t = prefix_length(acc)
        idx = torch.arange(kp, device=rowsj.device)
        valid = idx < m_t
        eq = (g_t == cands_t_kp)
        strict_cnt_t = (eq & valid).sum()
        relaxed_cnt_t = m_t - strict_cnt_t
        regret = gap * (~eq & valid)
        regret_sum_t = regret.sum()
        regret_max_t = regret.max() if kp > 0 else torch.zeros((), device=rowsj.device)

        vals = torch.cat([cands_t_kp, g_t, m_t.view(1), strict_cnt_t.view(1),
                          relaxed_cnt_t.view(1), regret_sum_t.view(1),
                          regret_max_t.view(1)]).tolist()
        ledger.sync += 1
        cands = [int(v) for v in vals[:kp]]
        g = [int(v) for v in vals[kp:2 * kp]]
        m = int(vals[2 * kp])
        strict_cnt = int(vals[2 * kp + 1])
        relaxed_cnt = int(vals[2 * kp + 2])
        regret_sum = float(vals[2 * kp + 3])
        regret_max = float(vals[2 * kp + 4])

        if m < kp:
            r = g[m]
            emitted = cands[:m] + [r]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + m)
            tok = g_t[m]
            with F._phase("repair"):
                with timer.span("repair", ledger):
                    saved_next, tgt_cache = F._step_gpu(target, tok, tgt_cache)
                    drf_cache = spec_cache.trim_cache(drf_cache, pre + m)
                    drf_lg, drf_cache = F._step_gpu(draft, tok, drf_cache)
                    F._maybe_finite(saved_next, finite_check)
                    F._maybe_finite(drf_lg, finite_check)
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
        st["strictly_matched"] += strict_cnt
        st["relaxed_only_accepted"] += relaxed_cnt
        st["sum_target_logit_regret"] += regret_sum
        st["max_target_logit_regret"] = max(st["max_target_logit_regret"], regret_max)
        st["rounds"] += 1
        row = _control_trace(st["rounds"] - 1, kp, m, pre, cands, g, emitted,
                             spec_cache.cache_length(tgt_cache),
                             spec_cache.cache_length(drf_cache),
                             strict_cnt, relaxed_cnt, regret_sum)
        control.append(row)
        if log is not None:
            log(row)
        if emitted and emitted[-1] == eos_id:
            break

    if measure_timing:
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode
    result = dict(ids=list(prompt_ids[0]) + gen, gen=gen, stats=st,
                  ledger=ledger.as_dict(), control=control, delta=float(delta))
    if measure_timing:
        result["timing"] = dict(prefill_s=prefill_s, decode_s=decode_s,
                                total_s=prefill_s + decode_s,
                                prefill_components=[dict(role="target", seconds=prefill_target_s),
                                                    dict(role="draft", seconds=prefill_draft_s)])
    if capture_cache:
        info = dict(tgt_len=spec_cache.cache_length(tgt_cache),
                    drf_len=spec_cache.cache_length(drf_cache))
        for name, cache in (("tgt", tgt_cache), ("drf", drf_cache)):
            try:
                spec_cache.check_consistent(cache)
                info[name + "_consistent"] = True
            except Exception as exc:  # noqa: BLE001
                info[name + "_consistent"] = False
                info[name + "_error"] = str(exc)
        result["final_cache"] = info
    return result
