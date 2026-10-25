"""CPU 单元测试：优化执行器 vs reference 执行器的**逐位/逐控制轨迹等价性**。

用轻量 stub 模型（不加载真模型、不触 GPU）覆盖：
  * proposal 正常链（kp=4）、末轮 k_eff<4；
  * acceptance 的 m=0 / m=1 / m=kp 分支；
  * draft 在候选块内出现 EOS 的截断分支；
  * verification 的 rows 语义（s=kp 与逐步 s=1 等价）；
  * 显式同步计数（优化执行器每轮 ≤ 2 次 .item/.tolist；reference 每轮 ≥ 2·kp 次）。
"""
from __future__ import annotations

import itertools

import torch

from specdec.engine import spec_dec
from experiments.execution_optimization import fast_spec_dec as F


class StubLM(torch.nn.Module):
    """极简因果 stub：logits_j = (E[token_j] + P[pos_j]) @ W^T，cache 只存原始 embedding 行。

    P[pos] 是与位置相关的随机向量，让"同一 token 在不同位置"的 argmax 不同，
    从而真模型那样的多轮、部分接受、失配位置都能在 CPU 上被覆盖。
    """

    def __init__(self, name, vocab=48, hidden=8, seed=0, noise=0.0, base=None, max_pos=64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(vocab, hidden, generator=g)
        self.E = self.E / self.E.norm(dim=1, keepdim=True)
        self.P = 0.6 * torch.randn(max_pos, hidden, generator=torch.Generator().manual_seed(seed + 1))
        if base is not None:
            noise_gen = torch.Generator().manual_seed(seed + 7777)
            self.W = base.W + noise * torch.randn(vocab, hidden, generator=noise_gen) / (hidden ** 0.5)
        else:
            self.W = torch.randn(vocab, hidden, generator=g) / (hidden ** 0.5)
        self.kv_format = "fp16"
        self.name = name
        self.head_dim = hidden
        self.rope_theta = 10000.0

    def set_kv_format(self, fmt):
        self.kv_format = fmt

    def embed_tokens(self, ids):
        return torch.nn.functional.embedding(ids, self.E)

    def _run(self, x, cache=None, need_logits=True):
        assert x.dim() == 3
        b, s, _ = x.shape
        T0 = cache[0]["k"].shape[2] if cache else 0
        pos = torch.arange(T0, T0 + s, dtype=torch.long)
        h = x + self.P[pos].unsqueeze(0)
        logits = h @ self.W.T                                  # (1, s, V)
        k_new = x[:, :, :1].transpose(1, 2).unsqueeze(-1)      # (1, 1, s, 1)
        v_new = k_new.clone()
        if cache is None:
            new_cache = [{"k": k_new, "v": v_new}]
        else:
            cache[0]["k"] = torch.cat([cache[0]["k"], k_new], dim=2)
            cache[0]["v"] = torch.cat([cache[0]["v"], v_new], dim=2)
            new_cache = cache
        return (logits if need_logits else None), new_cache


class SyncCounter:
    """统计 .item()/.tolist()/.cpu()/int() 的调用次数（CPU stub 上等价于同步类操作）。"""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._orig = (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu, torch.Tensor.__int__)
        counter = self

        def item(self_):
            counter.n += 1
            return counter._orig[0](self_)

        def tolist(self_):
            counter.n += 1
            return counter._orig[1](self_)

        def cpu(self_, *a, **k):
            counter.n += 1
            return counter._orig[2](self_, *a, **k)

        def __int__(self_):
            counter.n += 1
            return counter._orig[3](self_)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.Tensor.cpu = cpu
        torch.Tensor.__int__ = __int__
        return self

    def __exit__(self, *exc):
        (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu, torch.Tensor.__int__) = self._orig
        return False


def make_case(noise, seed, vocab=48, max_new=12, eos_id=None):
    tgt = StubLM("target", vocab=vocab, seed=seed)
    drf = StubLM("draft", vocab=vocab, seed=seed, base=tgt, noise=noise)
    prompt = torch.randint(0, vocab, (1, 6), generator=torch.Generator().manual_seed(seed + 999))
    return tgt, drf, prompt, max_new, eos_id


def run_reference(tgt, drf, prompt, max_new, eos_id, k=4):
    trace = []

    def log(row):
        trace.append(row)

    out = spec_dec.spec_decode_greedy(tgt, drf, prompt, max_new, k=k, eos_id=eos_id,
                                      log=log)
    return out, trace


def run_fast(tgt, drf, prompt, max_new, eos_id, flags=None, k=4):
    return F.spec_decode_greedy_optimized(
        tgt, drf, prompt, max_new, k=k, eos_id=eos_id, flags=flags)


def _assert_same_case(name, noise, seed, max_new=12, eos_id=None):
    tgt, drf, prompt, mn, eos = make_case(noise, seed, max_new=max_new, eos_id=eos_id)
    ref, ref_trace = run_reference(tgt, drf, prompt, mn, eos)
    fast = run_fast(tgt, drf, prompt, mn, eos)
    assert fast["gen"] == ref["gen"], f"{name}: emitted tokens differ"
    assert fast["stats"] == ref["stats"], f"{name}: stats differ {fast['stats']} vs {ref['stats']}"
    assert len(fast["control"]) == len(ref_trace), f"{name}: round count differs"
    for a, b in zip(fast["control"], ref_trace):
        for key in ("round", "Kp", "m", "pre", "tgt_len", "drf_len"):
            assert a[key] == b[key], f"{name}: {key} mismatch at round {a['round']}: {a[key]} vs {b[key]}"
        assert a["cands"] == b["cands"], f"{name}: cands mismatch round {a['round']}"
        assert a["g"] == b["g"], f"{name}: g mismatch round {a['round']}"
        assert a["emitted"] == b["emitted"], f"{name}: emitted mismatch round {a['round']}"
    return fast, ref, ref_trace


def test_numerical_equivalence_various_noise():
    for i, noise in enumerate((0.0, 0.05, 0.2, 0.5, 1.0)):
        _assert_same_case(f"noise={noise}", noise, seed=7 + i)


def test_tail_keff_lt_k():
    # max_new=6 -> 2 轮；最后一轮 k_eff=2
    fast, ref, trace = _assert_same_case("tail", 0.2, seed=21, max_new=6)
    assert any(row["Kp"] < 4 for row in trace), trace


def test_eos_inside_block():
    # 找到一个"第 2 轮候选块第 3 个 token"，把它当成 EOS：proposal 循环应在中途 break，
    # 候选数 kp 从 4 截断为 3，后续 verify/accept 语义必须与 reference 完全一致。
    tgt = StubLM("target", seed=3)
    drf = StubLM("draft", seed=3, base=tgt, noise=0.2)
    prompt = torch.randint(0, 48, (1, 5), generator=torch.Generator().manual_seed(5))
    ref0, trace0 = run_reference(tgt, drf, prompt, 12, None)
    full = [row for row in trace0 if row["Kp"] == 4]
    assert full, "stub 没产生 kp=4 的轮次"
    eos = full[1]["cands"][2]
    fast, ref, trace = _assert_same_case("eos", 0.2, seed=3, max_new=12, eos_id=eos)
    assert any(row["Kp"] < 4 for row in trace), [row["Kp"] for row in trace]
    assert fast["gen"][-1] == eos or len(fast["gen"]) >= 12


def test_all_accept_and_mismatch_branches_present():
    noise_cases = []
    for i in range(12):
        tgt, drf, prompt, mn, eos = make_case(0.3 * i, seed=100 + i)
        ref, trace = run_reference(tgt, drf, prompt, mn, eos)
        noise_cases.append(trace)
    ms = [row["m"] for trace in noise_cases for row in trace]
    assert any(m == 0 for m in ms), "no m=0 case produced"
    assert any(0 < m < 4 for m in ms), "no 0<m<kp case produced"
    assert any(m == row["Kp"] for trace in noise_cases for row in trace
               for m in [row["m"]]), "no all-accept case produced"


def test_flags_off_equivalence():
    # 全部开关关闭 => 走 reference 的 proposal/repair 语义，也必须等价
    flags = dict(proposal_gpu=False, vector_accept=False, gpu_tokens=False)
    tgt, drf, prompt, mn, eos = make_case(0.4, seed=31)
    ref, ref_trace = run_reference(tgt, drf, prompt, mn, eos)
    fast = run_fast(tgt, drf, prompt, mn, eos, flags=flags)
    assert fast["gen"] == ref["gen"]
    assert fast["stats"] == ref["stats"]
    for a, b in zip(fast["control"], ref_trace):
        assert a["emitted"] == b["emitted"] and a["m"] == b["m"]


def test_sync_count_budget():
    tgt, drf, prompt, mn, eos = make_case(0.2, seed=11)
    with SyncCounter() as ref_counter:
        ref, ref_trace = run_reference(tgt, drf, prompt, mn, eos)
    with SyncCounter() as fast_counter:
        fast = run_fast(tgt, drf, prompt, mn, eos)
    rounds = len(ref_trace)
    assert rounds > 2
    assert fast["gen"] == ref["gen"]
    # 优化执行器：每轮 1 次 EOS 检查（.item）+ 1 次 .tolist = 2 次；reference ≥ 2*kp 次 int()（经 item）
    assert fast_counter.n <= 2 * rounds + 2, (fast_counter.n, rounds)
    assert ref_counter.n >= 6 * rounds, (ref_counter.n, rounds)
    assert fast["ledger"]["sync"] <= 2 * rounds, fast["ledger"]
    # 每轮控制轨迹自洽：cache 长度 = pre + emitted
    for row in fast["control"]:
        assert row["tgt_len"] == row["pre"] + len(row["emitted"])
        assert row["drf_len"] == row["pre"] + len(row["emitted"])


def test_no_h2d_in_fast_path():
    # 优化路径不允许 host->device token 拷贝：ledger.h2d == 0
    tgt, drf, prompt, mn, eos = make_case(0.2, seed=12)
    fast = run_fast(tgt, drf, prompt, mn, eos)
    assert fast["ledger"]["h2d"] == 0, fast["ledger"]


if __name__ == "__main__":
    import sys
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
