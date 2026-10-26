"""Draft-Free Perfect-Accept Bonus Oracle（机制诊断，不可部署）。

语义：计时区间内**保留** target block verification、target KV/cache 推进、bonus emission、
round accounting 与 stop 控制；**删除**一切 draft forward / candidate 生成 / catch-up。
候选 token 来自计时区间外冻结的 64-token 序列（由 target AR greedy 生成、记录 sha1），
按 bonus-aware 轮次规则切块：round r 的 candidates = frozen[5r : 5r+4]，期望 bonus = frozen[5r+4]。
强制接受（all-accept），因此 `draft_forward_calls` 必须为 0。

这不是可部署 SpecDec、不是质量实验、不进入正式 confirmatory 区间。
"""
from __future__ import annotations

import time

import torch

from specdec.engine import spec_cache

from experiments.execution_optimization import fast_spec_dec as F


def candidate_layout(frozen_ids: list, k: int = 4, max_new: int = 64) -> list:
    """按 bonus-aware 规则把冻结序列切成 {round: {'cands': [...], 'expected_bonus': int|None}}。

    round r 提交 4 个 candidate + 1 个 bonus（最后一轮只有 candidates）；位置 = 5r .. 5r+3，
    bonus 位置 = 5r+4（存在时）。
    """
    out = []
    pos = 0
    r = 0
    while pos < max_new:
        k_eff = min(k, max_new - pos)
        cands = [int(x) for x in frozen_ids[pos:pos + k_eff]]
        expect_bonus = None
        if pos + k_eff < max_new:
            expect_bonus = int(frozen_ids[pos + k_eff])
        out.append(dict(round=r, cands=cands, expected_bonus=expect_bonus, start=pos))
        pos += k_eff + (1 if expect_bonus is not None else 0)
        r += 1
        if k_eff < k:
            break
    return out


def freeze_candidates(target, prompt_ids, max_new_tokens: int = 64):
    """用 target AR greedy 生成冻结候选序列（计时之外调用）；返回 ids 与 sha1。"""
    import hashlib
    gen: list[int] = []
    with torch.inference_mode():
        lg, cache = F._prefill_chunked(target, prompt_ids, chunk=2048)
        cur = lg[:, -1:]
        vocab = int(cur.shape[-1])
        for _ in range(max_new_tokens):
            tok = cur.reshape(-1, vocab)[-1].argmax()
            gen.append(int(tok))
            lg, cache = F._step_gpu(target, tok, cache)
            cur = lg
    sha1 = hashlib.sha1(bytes(str(gen), "utf-8")).hexdigest()
    return dict(gen=gen, sha1=sha1)


class DraftCallGuard:
    """计时区间内统计（并拦截）draft forward 调用；非零即硬失败。"""

    def __init__(self, draft):
        self.draft = draft
        self.calls = 0
        self._orig = None

    def __enter__(self):
        self._orig = self.draft._run

        def wrapper(*args, **kwargs):
            self.calls += 1
            raise RuntimeError("计时区间内检测到 draft forward 调用（Draft-Free oracle 硬失败）")

        self.draft._run = wrapper
        return self

    def __exit__(self, *exc):
        self.draft._run = self._orig
        return False


def decode_draft_free(target, draft, prompt_ids, frozen_ids, *, k=4, eos_id=None,
                      target_kv="base_bf16", timer=None, finite_check=False,
                      capture_cache=False, measure_timing=False, prefill_chunk=2048):
    """Draft-Free oracle 解码：target 真实 verify + bonus，draft 不参与计时区间。"""
    target.set_kv_format(F._fmt(target_kv))
    if measure_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with F._phase("prefill"):
            tgt_lg, tgt_cache = F._prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        t_decode = time.perf_counter()
        del tgt_lg
    else:
        with torch.inference_mode():
            tgt_lg, tgt_cache = F._prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
        del tgt_lg

    gen: list[int] = []
    committed = 0
    pending = None
    st = dict(rounds=0, candidate_tokens=0, bonus_tokens=0, target_block_calls=0,
              target_block_input_tokens=0, draft_forward_calls=0, mismatch_rounds=0,
              rollback_count=0, repair_count=0, proposed=0, accepted=0,
              bonus_mismatch_rounds=0)
    control = []
    stop_now = False
    with torch.inference_mode():
        while len(gen) < len(frozen_ids):
            remaining = len(frozen_ids) - len(gen)
            k_eff = min(k, remaining)
            cands = [int(x) for x in frozen_ids[committed:committed + k_eff]]
            pending_included = pending is not None
            toks = ([int(pending)] if pending_included else []) + list(cands)
            tok_t = torch.tensor([toks], dtype=torch.long, device=torch.device("cuda"))
            with F._phase("verify"):
                lg, tgt_blk = target._run(target.embed_tokens(tok_t), tgt_cache, need_logits=True)
                F._maybe_finite(lg, finite_check)
            st["target_block_calls"] += 1
            st["target_block_input_tokens"] += len(toks)
            st["candidate_tokens"] += len(cands)
            st["proposed"] += len(cands)
            st["accepted"] += len(cands)
            gen.extend(cands)
            committed += len(cands)
            tgt_cache = tgt_blk
            bonus = None
            expected = (frozen_ids[committed] if committed < len(frozen_ids) else None)
            if committed < len(frozen_ids):
                # forced-accept 契约：bonus 也强制为冻结序列的下一 token（保持轨迹 = AR），
                # 同时记录 target argmax 与冻结 token 的不一致数（bf16 近并列的诊断量）
                real_bonus = int(lg[0, -1].argmax())
                bonus = int(expected)
                mismatch = None
                if real_bonus != expected:
                    st["bonus_mismatch_rounds"] += 1
                    top2 = torch.topk(lg[0, -1].float(), 2).values
                    mismatch = dict(round=st["rounds"], real=real_bonus, expected=expected,
                                    top1_minus_top2=float((top2[0] - top2[1]).item()))
                    st.setdefault("mismatch_details", []).append(mismatch)
                gen.append(bonus)
                committed += 1
                st["bonus_tokens"] += 1
            pending = bonus if bonus is not None else None
            st["rounds"] += 1
            control.append(dict(round=st["rounds"] - 1, kp=len(cands),
                                pending_included=bool(pending_included),
                                verify_input_length=len(toks), cands=list(cands),
                                bonus=bonus, expected_bonus=expected,
                            bonus_argmax_match=(mismatch is None if bonus is not None else None),
                                emitted_count=len(cands) + (0 if bonus is None else 1),
                                committed=committed,
                                target_len=spec_cache.cache_length(tgt_cache)))
            if bonus is not None and eos_id is not None and bonus == eos_id:
                stop_now = True
                break
            if cands and eos_id is not None and cands[-1] == eos_id:
                stop_now = True
                break
            if committed >= len(frozen_ids):
                break

    if measure_timing:
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode
    result = dict(ids=list(prompt_ids[0]) + gen, gen=gen, stats=st, control=control,
                  stop_reason=("eos" if stop_now else "max_new_reached"),
                  pending_at_stop=(0 if pending is None else 1))
    if measure_timing:
        result["timing"] = dict(prefill_s=prefill_s, decode_s=decode_s,
                                total_s=prefill_s + decode_s)
    if capture_cache:
        info = dict(tgt_len=spec_cache.cache_length(tgt_cache))
        try:
            spec_cache.check_consistent(tgt_cache)
            info["tgt_consistent"] = True
        except Exception:  # noqa: BLE001
            info["tgt_consistent"] = False
        result["final_cache"] = info
    return result


def freeze_via_ar(target, tokenizer, ids, job):
    """用正式 AR runner（W.run_ar_once）生成冻结候选（同 greedy/EOS 语义），返回 ids 与 sha1。"""
    import hashlib
    from experiments.formal import worker as W
    r = W.run_ar_once(target, tokenizer, ids, job)
    gen = [int(x) for x in r["generated_ids"]]
    return dict(gen=gen, sha1=hashlib.sha1(bytes(str(gen), "utf-8")).hexdigest(),
                ar_stop=r["stop_reason"])
