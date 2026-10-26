"""bonus_accept_oracle — Bonus-Aware Perfect-Accept Latency Floor（机制诊断）。

两种路径（同一 token 语义，仅"target 何时消费 bonus"不同）：

1. mode="catchup"（explicit catch-up，**仅正确性参照，不进计时**）：
   每轮全接受 → bonus = argmax(rows[-1]) → emit candidates + bonus
   → target._step(bonus)、draft._step(bonus) 立即消费 → 两侧 cache 对齐；
   下一轮 verify 输入只含 candidates。
2. mode="carry"（pending-bonus carry-forward，**计时主路径**）：
   每轮全接受 → bonus = argmax(rows[-1]) → emit candidates + bonus → **不立即消费**；
   pending bonus 保留给下一轮：
     - 下一轮 draft 先做一次 catch-up `_step(bonus)`（单独计数），再 propose k 个 candidate；
     - 下一轮 target block 输入 = [pending_bonus] + candidates（首轮无 pending，L=kp；之后 L=kp+1），
       首个 row 让 target cache 纳入 bonus，最后一行产生新 bonus；
   全程 target 只做 block call（**无额外 target 单步**），timed path 的
   `target_extra_single_step_calls == 0`。

EOS/预算契约沿用现有 SpecDec 语义（见 docs/bonus_accept_design_audit_2026-10-26.md）：
  * candidate 中出现 EOS：接受后停止，不发 bonus；
  * bonus 本身是 EOS：发出后停止，不再开始下一轮（carry 下目标 cache 比 committed 少 1，属设计状态）；
  * 最后一轮 remaining ≤ k 时只生成 remaining 个 candidate，不发 bonus（长度严格 = max_new）。
"""
from __future__ import annotations

import time

import torch

from specdec.engine import spec_cache

from experiments.execution_optimization import fast_spec_dec as F

EXPERIMENT_CLASS = "mechanism_diagnostic"
ACCEPT_MODE = "all_accept_bonus_oracle"


def decode(target, draft, prompt_ids, max_new_tokens, *, k=4, eos_id=None,
           mode="carry", target_kv="base_bf16", draft_kv="base_bf16",
           log=None, timer=None, finite_check=False, capture_cache=False,
           measure_timing=False, prefill_chunk=2048, diagnostic_top2=False):
    if mode not in ("carry", "catchup"):
        raise ValueError(f"未知 mode: {mode!r}")
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
            tgt_lg, tgt_cache = F._prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
            drf_lg, drf_cache = F._prefill_chunked(draft, prompt_ids, chunk=prefill_chunk)
    drf_lg = drf_lg[:, -1:]
    ctx_len = int(prompt_ids.shape[1])

    gen: list[int] = []
    committed = 0
    pending = None                    # 0-d GPU tensor（carry 模式）
    pending_included = 0
    st = dict(rounds=0, candidate_tokens=0, bonus_tokens=0, candidate_calls=0,
              bonus_catchup_calls=0, draft_calls=0, target_block_calls=0,
              target_block_input_tokens=0, target_extra_single_step_calls=0,
              draft_extra_single_step_calls=0, proposed=0, accepted=0,
              mismatch_rounds=0, repair_count=0, rollback_count=0,
              verify_first_round_len=None, verify_later_round_len=None)
    control = []
    eos_hit = False
    pending_at_stop = 0

    while len(gen) < max_new_tokens:
        remaining = max_new_tokens - len(gen)
        k_eff = min(k, remaining)
        pre_tgt = spec_cache.cache_length(tgt_cache)
        pre_drf = spec_cache.cache_length(drf_cache)
        had_pending = pending is not None
        ledger.rounds += 1

        if had_pending:
            with F._phase("propose"):
                with timer.span("draft_bonus_catchup", ledger):
                    drf_lg, drf_cache = F._step_gpu(draft, pending, drf_cache)
                    F._maybe_finite(drf_lg, finite_check)
            st["bonus_catchup_calls"] += 1
            st["draft_calls"] += 1

        with F._phase("propose"):
            with timer.span("propose", ledger):
                cands_t, drf_cache, prop_lg = F._propose_gpu(
                    draft, drf_cache, drf_lg, k_eff, ledger, finite_check)
        st["candidate_calls"] += k_eff
        st["draft_calls"] += k_eff

        if eos_id is None:
            kp = k_eff
        else:
            hit = (cands_t == eos_id).nonzero()
            kp = int(hit[0].item()) + 1 if hit.numel() else k_eff
            ledger.sync += 1
        if kp == 0:
            break
        cands_t_kp = cands_t[:kp]
        st["candidate_tokens"] += kp
        st["proposed"] += kp

        if pending is not None:
            verify_tokens = torch.cat([pending.view(1), cands_t_kp])
            pending_included = 1
        else:
            verify_tokens = cands_t_kp
            pending_included = 0
        L = int(verify_tokens.numel())
        with F._phase("verify"):
            with timer.span("verify", ledger):
                rows, tgt_blk = F._verify_tensor(target, tgt_cache, verify_tokens)
                F._maybe_finite(rows, finite_check)
        st["target_block_calls"] += 1
        st["target_block_input_tokens"] += L
        if st["verify_first_round_len"] is None:
            st["verify_first_round_len"] = L
        else:
            st["verify_later_round_len"] = L

        emitted_cands = cands_t_kp.tolist()
        ledger.sync += 1
        gen += emitted_cands
        committed += kp
        st["accepted"] += kp
        tgt_cache = tgt_blk

        budget_after = max_new_tokens - len(gen)
        eos_in_cands = bool(eos_id is not None and emitted_cands[-1] == eos_id)
        bonus = None
        stop_now = False
        top2_gap = None
        if diagnostic_top2:
            _v = torch.topk(rows[-1], 2).values
            top2_gap = float((_v[0] - _v[1]).item())
        if budget_after > 0 and not eos_in_cands:
            bonus_t = rows[-1].argmax()
            bonus = int(bonus_t)
            ledger.sync += 1
            gen.append(bonus)
            committed += 1
            st["bonus_tokens"] += 1
            if mode == "carry":
                pending = bonus_t
            else:
                with F._phase("repair"):
                    with timer.span("target_bonus_catchup", ledger):
                        _, tgt_cache = F._step_gpu(target, bonus_t, tgt_cache)
                with F._phase("repair"):
                    with timer.span("draft_bonus_catchup_catchup", ledger):
                        drf_lg, drf_cache = F._step_gpu(draft, bonus_t, drf_cache)
                        F._maybe_finite(drf_lg, finite_check)
                st["target_extra_single_step_calls"] += 1
                st["draft_extra_single_step_calls"] += 1
                st["draft_calls"] += 1
                pending = None
            if eos_id is not None and bonus == eos_id:
                stop_now = True
                eos_hit = True
        else:
            pending = None

        # draft cache trim：应恰好为 pre_drf + (pending 追赶?) + kp（+ catchup 模式已消费的 bonus）
        bonus_consumed = 1 if (mode == "catchup" and bonus is not None) else 0
        expected_drf = pre_drf + (1 if had_pending else 0) + kp + bonus_consumed
        cur_drf = spec_cache.cache_length(drf_cache)
        if cur_drf > expected_drf:
            drf_cache = spec_cache.trim_cache(drf_cache, expected_drf)
        elif cur_drf < expected_drf:
            raise RuntimeError(f"draft cache 长度异常: cur={cur_drf} expected={expected_drf}")

        pending_new = 1 if pending is not None else 0
        tgt_len = spec_cache.cache_length(tgt_cache)
        drf_len = spec_cache.cache_length(drf_cache)
        if tgt_len != drf_len or tgt_len + pending_new != ctx_len + committed:
            raise RuntimeError(
                f"cache 不变量失败: tgt={tgt_len} drf={drf_len} committed={committed} ctx={ctx_len} "
                f"pending={pending_new} (round {st['rounds']})")

        st["rounds"] += 1
        row = dict(round=st["rounds"] - 1, kp=kp, pending_included=pending_included,
                   verify_input_length=L, candidate_start_row=(1 if pending_included else 0),
                   next_bonus_row=L - 1, emitted_candidates=list(emitted_cands),
                   emitted_bonus=bonus, emitted_count=len(emitted_cands) + (1 if bonus is not None else 0),
                   remaining_before=remaining, remaining_after=max_new_tokens - len(gen),
                   committed=committed, pending_new=pending_new,
                   pre_target_len=pre_tgt, pre_draft_len=pre_drf,
                   target_len=tgt_len, draft_len=drf_len, eos_end=bool(stop_now),
                   last_row_top2_gap=top2_gap)
        control.append(row)
        if log is not None:
            log(row)
        if stop_now:
            break
        if emitted_cands and emitted_cands[-1] == eos_id:
            break
        if budget_after <= 0:
            break
    pending_at_stop = 1 if pending is not None else 0

    if measure_timing:
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode
    result = dict(ids=list(prompt_ids[0]) + gen, gen=gen,
                  stats=st, ledger=ledger.as_dict(), control=control,
                  mode=mode, pending_at_stop=pending_at_stop, committed=committed,
                  experiment_class=EXPERIMENT_CLASS, accept_mode=ACCEPT_MODE,
                  quality_claim_allowed=False, formal_result_allowed=False,
                  bonus_contract=("bonus=argmax(last verify row); carry-forward keeps one pending bonus; "
                                  "next target block consumes it as first input token"))
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
