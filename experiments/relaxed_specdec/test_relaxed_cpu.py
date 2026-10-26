"""relaxed_specdec 的 CPU 单元测试（轻量 stub，不加载真模型、不触 GPU）。

覆盖：
  * 纯张量 acceptance：longest-prefix、m=0 / m=kp / 首个拒绝发出 top1、后续候选不得接受；
  * relaxed 解码循环 vs Phase O strict executor：delta=0 委托一致性、tiny-delta 等价、
    宽松解码的 cache/控制轨迹不变量、regret 只计 relaxed-only、接受随 delta 非下降；
  * 无额外同步（relaxed 每轮 2 次 .item/.tolist 预算）。
"""
from __future__ import annotations

import sys

import torch

from specdec.engine import spec_cache, spec_dec
from experiments.execution_optimization import fast_spec_dec as F
from experiments.relaxed_specdec import relaxed_spec as R

torch.set_grad_enabled(False)


class StubLM(torch.nn.Module):
    """与 Phase O CPU 测试同构的因果 stub（logits_j = (E[token_j] + P[pos_j]) @ W^T）。"""

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
        logits = h @ self.W.T
        k_new = x[:, :, :1].transpose(1, 2).unsqueeze(-1)
        v_new = k_new.clone()
        if cache is None:
            new_cache = [{"k": k_new, "v": v_new}]
        else:
            cache[0]["k"] = torch.cat([cache[0]["k"], k_new], dim=2)
            cache[0]["v"] = torch.cat([cache[0]["v"], v_new], dim=2)
            new_cache = cache
        return (logits if need_logits else None), new_cache


def make_case(noise, seed, max_new=12, vocab=48):
    tgt = StubLM("target", vocab=vocab, seed=seed)
    drf = StubLM("draft", vocab=vocab, seed=seed, base=tgt, noise=noise)
    prompt = torch.randint(0, vocab, (1, 6), generator=torch.Generator().manual_seed(seed + 999))
    return tgt, drf, prompt, max_new


def run_relaxed(tgt, drf, prompt, max_new, delta, eos=None, k=4):
    return R.decode(tgt, drf, prompt, max_new, k=k, delta=delta, eos_id=eos,
                    capture_cache=True)


def test_accept_mask_prefix_semantics():
    g = torch.tensor([5, 7, 9, 11])
    cands = torch.tensor([5, 3, 4, 11])
    gap = torch.tensor([0.0, 0.05, 0.2, 0.0])
    acc = R.accept_mask(g, cands, gap, delta=0.1)
    assert acc.tolist() == [True, True, False, True], acc
    assert int(R.prefix_length(acc)) == 2
    acc0 = R.accept_mask(g, cands, gap, delta=0.0)
    assert acc0.tolist() == [True, False, False, True], acc0  # strict: only exact top1
    assert int(R.prefix_length(acc0)) == 1
    # 首个拒绝之后即使后面的位置满足条件也不得被接受
    assert int(R.prefix_length(torch.tensor([False, True, True]))) == 0
    assert int(R.prefix_length(torch.tensor([True, True, True]))) == 3


def test_delta0_delegates_to_strict_executor():
    tgt, drf, prompt, max_new = make_case(0.3, seed=31)
    strict = F.spec_decode_greedy_optimized(tgt, drf, prompt, max_new, k=4,
                                            capture_cache=True)
    out = run_relaxed(tgt, drf, prompt, max_new, delta=0.0)
    assert out["delta"] == 0.0
    assert out["gen"] == strict["gen"]
    assert out["stats"] == strict["stats"]
    assert out["control"] == strict["control"]
    assert out["final_cache"] == strict["final_cache"]


def test_tiny_delta_matches_strict_trajectory():
    for noise in (0.0, 0.2, 0.4):
        tgt, drf, prompt, max_new = make_case(noise, seed=11)
        strict = F.spec_decode_greedy_optimized(tgt, drf, prompt, max_new, k=4)
        out = run_relaxed(tgt, drf, prompt, max_new, delta=1e-9)
        assert out["gen"] == strict["gen"], noise
        assert out["stats"]["rounds"] == strict["stats"]["rounds"]
        assert out["stats"]["accepted"] == strict["stats"]["accepted"]


def test_relaxed_longest_prefix_and_counts():
    for noise in (0.2, 0.35, 0.5):
        tgt, drf, prompt, max_new = make_case(noise, seed=41)
        out = run_relaxed(tgt, drf, prompt, max_new, delta=0.5)
        st = out["stats"]
        assert st["strictly_matched"] + st["relaxed_only_accepted"] == st["accepted"]
        assert st["sum_target_logit_regret"] >= 0.0
        assert st["max_target_logit_regret"] <= 0.5 + 1e-5
        # 逐轮控制轨迹：cache 长度 = pre + emitted；首个拒绝位置发出 top1；后续候选未被接受
        for row in out["control"]:
            kp, m = row["Kp"], row["m"]
            assert row["tgt_len"] == row["pre"] + len(row["emitted"])
            assert row["drf_len"] == row["pre"] + len(row["emitted"])
            if m < kp:
                assert row["emitted"][:m] == row["cands"][:m]
                assert row["emitted"][m] == row["g"][m]      # 替换 token = target top1
                assert row["emitted"][m] != row["cands"][m]
            else:
                assert row["emitted"] == row["cands"]
            assert row["strictly_matched"] + row["relaxed_only"] == m


def test_acceptance_monotone_in_delta():
    rates = {}
    for delta in (0.0, 0.1, 0.3, 0.5, 1.0):
        tot = prop = 0
        for noise in (0.2, 0.4, 0.6):
            tgt, drf, prompt, max_new = make_case(noise, seed=51)
            out = run_relaxed(tgt, drf, prompt, max_new, delta=delta)
            tot += out["stats"]["accepted"]
            prop += out["stats"]["proposed"]
        rates[delta] = tot / prop
    ordered = [rates[d] for d in (0.0, 0.1, 0.3, 0.5, 1.0)]
    assert all(b >= a - 1e-12 for a, b in zip(ordered, ordered[1:])), rates


def test_cache_and_lengths():
    for noise in (0.0, 0.3):
        tgt, drf, prompt, max_new = make_case(noise, seed=61)
        out = run_relaxed(tgt, drf, prompt, max_new, delta=0.3)
        fc = out["final_cache"]
        expected = prompt.shape[1] + len(out["gen"])
        assert fc["tgt_len"] == expected and fc["drf_len"] == expected, fc
        assert fc["tgt_consistent"] and fc["drf_consistent"]


def test_eos_inside_block_relaxed():
    tgt = StubLM("target", seed=3)
    drf = StubLM("draft", seed=3, base=tgt, noise=0.2)
    prompt = torch.randint(0, 48, (1, 5), generator=torch.Generator().manual_seed(5))
    ref0 = R.decode(tgt, drf, prompt, 12, delta=0.0)
    full = [row for row in ref0["control"] if row["Kp"] == 4]
    assert full
    eos = full[1]["cands"][2]
    out = run_relaxed(tgt, drf, prompt, 12, delta=0.3, eos=eos)
    assert any(row["Kp"] < 4 for row in out["control"]), [r["Kp"] for r in out["control"]]


def test_no_extra_sync_in_relaxed_timed_path():
    tgt, drf, prompt, max_new = make_case(0.3, seed=71)
    out = run_relaxed(tgt, drf, prompt, max_new, delta=0.3)
    rounds = len(out["control"])
    assert rounds > 2
    assert out["ledger"]["sync"] <= 2 * rounds, out["ledger"]
    assert out["ledger"]["h2d"] <= 1, out["ledger"]  # relaxed 路径无 host token 拷贝


if __name__ == "__main__":
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
