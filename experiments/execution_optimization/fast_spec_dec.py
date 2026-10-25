"""execution_optimization.fast_spec_dec — 优化执行器（显示语义：与 spec_dec.spec_decode_greedy 逐位等价）。

原理（每项对应 Phase O 的 O2.x）：
O2.1 同步移除
  * `_propose` 的每个候选 token 不再 `int(argmax)`（隐式 .item 同步），argmax 结果作为 0 维 CUDA
    张量直接送入 embedding（`f.embedding` 接受 device 上索引），不再 H2D token 拷贝；
  * acceptance 由 Python 逐 token（kp 次 int 同步）改为 GPU 上一次 `argmax` + `cumprod` 求首个不匹配；
  * repair 的 token 直接取自 GPU 张量（不再 `torch.tensor([[token]])` H2D）；
  * 每轮只保留 2 次显式同步：EOS 截断（`nonzero`）与轮末一次 `.tolist()` 取控制值。
O2.2 预分配/复用
  * 候选缓冲 `cands` 每轮按 k 预分配并 `copy_` 写入；
  * 控制值（cands/g/m/kp）一次性从 GPU 拉回，避免逐 token 往返。
O2.3 向量化 acceptance（见上）。

语义保证（与 reference 的唯一差异仅在于"谁在 GPU 上算"，不改变任何数值或控制转移）：
  * proposal 顺序、draft `_run` 次数、EOS 截断规则、verify 的 s 与候选集合、acceptance/rollback/repair
    的判定值、emitted token、cache trim 长度、stats 计数全部与 reference 一致；
  * verification 仍走 `target._run`（block，s=kp），repair 仍走 `model._run`（s=1），无自研算子。
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from specdec.engine import spec_cache, spec_dec

from . import config


class SyncLedger:
    """记录 host 侧同步/回传操作的显式计数（用于 before/after 对照）。"""

    def __init__(self):
        self.sync = 0          # .item()/int()/nonzero 等强制 GPU→CPU 同步
        self.h2d = 0           # host→device token 拷贝
        self.rounds = 0
        self.stage_s = dict(propose=0.0, verify=0.0, repair=0.0)

    def add_stage(self, name, seconds):
        self.stage_s[name] = self.stage_s.get(name, 0.0) + float(seconds)

    def as_dict(self):
        return dict(sync=self.sync, h2d=self.h2d, rounds=self.rounds,
                    sync_per_round=(self.sync / self.rounds) if self.rounds else None,
                    stage_s=dict(self.stage_s))


def _fmt(kv):
    return spec_dec.FMT[kv]


def _prefill_chunked(model, ids, chunk: int = 2048):
    """与 experiments.formal.worker.test_chunked_prefill 完全同构的分块 prefill。

    **必须分块**：正式基线（run_ar_once / run_spec_once / 诊断）都用 chunk=2048 的逐块
    `model._run` 得到 prefill logits 与 cache；一次性 s=ctx 前向在 4K/8K 上与分块路径存在
    bf16 数值差异（attention kernel 形状不同），会让后续 token 轨迹漂移。这里逐字复刻：
    每块 `attention_phase('prefill')` 包裹的 `model._run`，仅末块取 logits。
    """
    from specdec.engine import model_self
    n = ids.shape[1]
    logits, cache = None, None
    with torch.inference_mode():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            piece = ids[:, start:end]
            with model_self.attention_phase("prefill"):
                logits, cache = model._run(model.embed_tokens(piece), cache,
                                           need_logits=(end == n))
    next_logits = logits[:, -1, :].clone()
    return next_logits.unsqueeze(1), cache


def _argmax_last(lg: torch.Tensor) -> torch.Tensor:
    """末位置 argmax，返回 0 维 long 张量（留在 GPU）。"""
    return lg.reshape(-1, lg.shape[-1])[-1].argmax()


def _propose_gpu(draft, cache, lg, k_eff: int, ledger: SyncLedger, finite_check: bool = False):
    """O2.1/O2.2：proposal 循环全部留在 GPU（无 .item、无 H2D）。"""
    device = lg.device
    cands = torch.empty(k_eff, dtype=torch.long, device=device)
    with torch.inference_mode():
        for i in range(k_eff):
            t = _argmax_last(lg)
            cands[i].copy_(t)                       # GPU→GPU，无同步
            ids = t.view(1, 1)
            lg, cache = draft._run(
                draft.embed_tokens(ids), cache, need_logits=True)
            _maybe_finite(lg, finite_check)
    return cands, cache, lg


def _maybe_finite(lg, enabled: bool):
    """仅诊断（probe）用：显式检查 logits 有限性；默认关闭，不进计时。"""
    if enabled and lg is not None:
        if not bool(torch.isfinite(lg).all()):
            raise RuntimeError("非有限 logits（propose/verify 之后）")


def _propose_reference(draft, cache, lg, k_eff: int, eos_id, ledger: SyncLedger):
    """reference 语义的 proposal（逐 token int()，用作逐项消融对照）。"""
    cands, cache, lg = spec_dec._propose(draft, cache, lg, k_eff, eos_id)
    ledger.sync += len(cands) * 2       # int(argmax) + H2D 各 kp 次（保守计 2kp 次同步类操作）
    return cands, cache, lg


def _verify_tensor(target, cache, cands_t: torch.Tensor):
    """与 spec_dec._verify（block 分支）逐字同构，但候选以 GPU 张量传入（省一次 H2D）。"""
    with torch.inference_mode():
        lg, cache_blk = target._run(
            target.embed_tokens(cands_t.view(1, -1)), cache, need_logits=True)
    kp = cands_t.numel()
    return lg[0, -kp:], cache_blk


def _verify_reference(target, cache, cands):
    return spec_dec._verify(target, cache, cands)


def _step_gpu(model, token, cache):
    """与 spec_dec._step 同构，但 token 是 GPU 0 维张量（无 H2D）。"""
    with torch.inference_mode():
        lg, cache = model._run(
            model.embed_tokens(token.view(1, 1)), cache, need_logits=True)
    return lg[:, -1:], cache


def _step_reference(model, token, cache):
    return spec_dec._step(model, token, cache)


@contextmanager
def _phase(name):
    from specdec.engine import model_self
    with model_self.attention_phase(name):
        yield


class StageTimer:
    """CUDA Event 阶段计时（与 profile_specdec_cost_2k.phase_scope 同口径）。"""

    def __init__(self, enabled=True):
        self.enabled = enabled

    @contextmanager
    def span(self, name, ledger: SyncLedger):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            ledger.add_stage(name, time.perf_counter() - start)


def spec_decode_greedy_optimized(target, draft, prompt_ids, max_new_tokens, *, k=4,
                                 eos_id=None, target_kv="base_bf16", draft_kv="base_bf16",
                                 log=None, flags=None, timer=None, finite_check=False,
                                 capture_cache=False, measure_timing=False,
                                 prefill_chunk=2048):
    """优化执行器。返回 dict 与 reference 兼容（ids/gen/stats/ledger/control）。"""
    if flags is None:
        flags = config.optimization_flags()
    if timer is None:
        timer = StageTimer(enabled=False)
    ledger = SyncLedger()
    fn_propose = _propose_gpu if flags["proposal_gpu"] else _propose_reference
    fn_verify = _verify_tensor
    fn_step = _step_gpu if flags["gpu_tokens"] else _step_reference

    target.set_kv_format(_fmt(target_kv))
    draft.set_kv_format(_fmt(draft_kv))

    if measure_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _phase("prefill"):
            tgt_lg, tgt_cache = _prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
        torch.cuda.synchronize()
        prefill_target_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with _phase("prefill"):
            drf_lg, drf_cache = _prefill_chunked(draft, prompt_ids, chunk=prefill_chunk)
        torch.cuda.synchronize()
        prefill_draft_s = time.perf_counter() - t0
        prefill_s = prefill_target_s + prefill_draft_s
        t_decode = time.perf_counter()
    else:
        with _phase("prefill"):
            with timer.span("prefill_target", ledger):
                tgt_lg, tgt_cache = _prefill_chunked(target, prompt_ids, chunk=prefill_chunk)
            with timer.span("prefill_draft", ledger):
                drf_lg, drf_cache = _prefill_chunked(draft, prompt_ids, chunk=prefill_chunk)
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]

    gen = []
    st = dict(rounds=0, proposed=0, accepted=0, target_block_calls=0, draft_calls=0,
              mismatch_rounds=0, all_accept_rounds=0, rollbacks=0)
    control = []

    while len(gen) < max_new_tokens:
        k_eff = min(k, max_new_tokens - len(gen))
        pre = spec_cache.cache_length(tgt_cache)
        ledger.rounds += 1

        with _phase("propose"):
            with timer.span("propose", ledger):
                if flags["proposal_gpu"]:
                    cands_t, drf_cache, prop_lg = _propose_gpu(draft, drf_cache, drf_lg, k_eff,
                                                               ledger, finite_check)
                else:
                    cands, drf_cache, prop_lg = _propose_reference(draft, drf_cache, drf_lg, k_eff, eos_id, ledger)
                    cands_t = torch.tensor(cands, dtype=torch.long, device=tgt_lg.device)
                    ledger.h2d += 1

        # EOS 截断：reference 在 proposal 循环内 break，候选数 kp = 首个 EOS 位置 + 1
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
        with _phase("verify"):
            with timer.span("verify", ledger):
                rows, tgt_blk = _verify_tensor(target, tgt_cache, cands_t_kp)
                _maybe_finite(rows, finite_check)
        st["target_block_calls"] += 1
        st["proposed"] += kp

        # 向量化 acceptance：g = argmax([saved_next] + rows[:kp-1])，m = 首个不匹配前缀长度
        g_t = torch.cat([saved_next.reshape(1, -1), rows[:kp - 1]], dim=0).argmax(dim=-1)  # (kp,)
        eq = (g_t == cands_t_kp)
        m_t = eq.to(torch.int64).cumprod(0).sum()                          # GPU int64 标量

        # 一次同步取回全部控制值（候选 / 目标判词 / m）
        vals = torch.cat([cands_t_kp, g_t, m_t.view(1)]).tolist()
        ledger.sync += 1
        cands = [int(v) for v in vals[:kp]]
        g = [int(v) for v in vals[kp:2 * kp]]
        m = int(vals[-1])

        if m < kp:
            r = g[m]
            emitted = cands[:m] + [r]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + m)
            tok = g_t[m] if flags["gpu_tokens"] else r
            with _phase("repair"):
                with timer.span("repair", ledger):
                    saved_next, tgt_cache = (_step_gpu(target, tok, tgt_cache)
                                             if flags["gpu_tokens"]
                                             else _step_reference(target, r, tgt_cache))
                    drf_cache = spec_cache.trim_cache(drf_cache, pre + m)
                    drf_lg, drf_cache = (_step_gpu(draft, tok, drf_cache)
                                         if flags["gpu_tokens"]
                                         else _step_reference(draft, r, drf_cache))
                    _maybe_finite(saved_next, finite_check)
                    _maybe_finite(drf_lg, finite_check)
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
        control.append(dict(round=st["rounds"] - 1, Kp=kp, m=m, pre=pre,
                            cands=list(cands), g=list(g), emitted=list(emitted),
                            tgt_len=spec_cache.cache_length(tgt_cache),
                            drf_len=spec_cache.cache_length(drf_cache)))
        if log is not None:
            log(control[-1])
        if emitted and emitted[-1] == eos_id:
            break

    if measure_timing:
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode
    result = dict(ids=list(prompt_ids[0]) + gen, gen=gen, stats=st,
                  ledger=ledger.as_dict(), control=control)
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
