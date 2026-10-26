"""all_accept_oracle — Perfect-Accept Latency Floor 诊断模式（机制诊断，不是正式配置）。

语义（严格按任务定义）：
  * draft 仍然真实提出最多 k=4 个 candidate（proposal 全走）；
  * target 仍然真实执行一次 block verification（rows 全走）；
  * 对所有有效 candidate 位置**无条件接受**（m = kp）；
  * 走现有 m == kp 分支：emitted = candidates（当前契约不额外发出 bonus token），
    target cache = block cache，draft cache trim 到 pre+kp，saved_next = rows[kp-1]，drf_lg = prop_lg；
  * candidate 中出现 EOS 时沿用现有 EOS 契约（proposal 在 EOS 处截断、本轮 emitted 以 EOS 结尾即停止）；
  * max_new 截断沿用现有契约（最后一轮 k_eff < k）；
  * 不跳过 proposal、不跳过 verification、不伪造计时、不删除生产路径中的 cache 操作。

明确标记：experiment_class = "mechanism_diagnostic"；accept_mode = "all_accept_oracle"；
quality_claim_allowed = False；formal_result_allowed = False。
"""
from __future__ import annotations

import time

import torch

from specdec.engine import spec_cache

from experiments.execution_optimization import fast_spec_dec as F

EXPERIMENT_CLASS = "mechanism_diagnostic"
ACCEPT_MODE = "all_accept_oracle"


def decode(target, draft, prompt_ids, max_new_tokens, *, k=4, eos_id=None,
           target_kv="base_bf16", draft_kv="base_bf16", log=None, timer=None,
           finite_check=False, capture_cache=False, measure_timing=False,
           prefill_chunk=2048):
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
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]

    gen: list[int] = []
    st = dict(rounds=0, proposed=0, accepted=0, candidate_count=0, bonus_token_count=0,
              mismatch_rounds=0, repair_count=0, rollback_count=0, all_accept_rounds=0,
              target_block_calls=0, draft_calls=0)
    control = []
    eos_hit = False

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
        st["candidate_count"] += kp

        # 无条件接受全部候选（m = kp）
        cands = cands_t_kp.tolist()
        ledger.sync += 1
        emitted = list(cands)
        tgt_cache = tgt_blk
        drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
        saved_next = rows[kp - 1]
        drf_lg = prop_lg
        st["all_accept_rounds"] += 1
        st["accepted"] += kp
        if eos_id is not None and emitted[-1] == eos_id:
            eos_hit = True

        gen += emitted
        st["rounds"] += 1
        row = dict(round=st["rounds"] - 1, Kp=kp, m=kp, pre=pre, cands=list(cands),
                   emitted=list(emitted), tgt_len=spec_cache.cache_length(tgt_cache),
                   drf_len=spec_cache.cache_length(drf_cache),
                   round_tokens_emitted=len(emitted), eos_end=bool(eos_hit))
        control.append(row)
        if log is not None:
            log(row)
        if eos_hit:
            break

    if measure_timing:
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode
    result = dict(ids=list(prompt_ids[0]) + gen, gen=gen, stats=st,
                  ledger=ledger.as_dict(), control=control,
                  experiment_class=EXPERIMENT_CLASS, accept_mode=ACCEPT_MODE,
                  quality_claim_allowed=False, formal_result_allowed=False,
                  bonus_contract="m==kp emits candidates only (no extra bonus token)")
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
