"""bonus_accept_oracle 的 CPU 单元测试（stub 模型，无 GPU）。

覆盖任务 Gate 1 清单：
 1. 第一轮无 pending bonus；
 2. 全接受时输出 candidates + bonus；
 3. 下一轮 draft 先消费 pending bonus；
 4. 下一轮 target block 输入以 pending bonus 开头；
 5. pending bonus 不被重复输出；
 6. row 位置映射正确；
 7. 64 token、k=4 时理论 13 轮；
 8. 最后一轮 remaining=4 时不输出 bonus；
 9. candidate EOS；
10. bonus EOS；
11. max_new 截断；
12. target cache 长度正确；
13. draft cache 长度正确；
14. 最终 cache 与已提交历史一致；
15. explicit catch-up 与 carry-forward 生成 token/控制轨迹一致；
16. 默认 strict / relaxed / 无 bonus all-accept 行为完全不变。
"""
from __future__ import annotations

import sys

import torch

from experiments.execution_optimization import fast_spec_dec as F
from experiments.relaxed_specdec import all_accept_oracle as A0
from experiments.relaxed_specdec import bonus_accept_oracle as B
from experiments.relaxed_specdec import relaxed_spec as R


class StubLM(torch.nn.Module):
    def __init__(self, name, vocab=48, hidden=8, seed=0, noise=0.0, base=None, max_pos=128):
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


def make_case(noise=0.2, seed=31, max_new=64, vocab=48):
    tgt = StubLM("target", vocab=vocab, seed=seed)
    drf = StubLM("draft", vocab=vocab, seed=seed, base=tgt, noise=noise)
    prompt = torch.randint(0, vocab, (1, 6), generator=torch.Generator().manual_seed(seed + 999))
    return tgt, drf, prompt, max_new


def run(tgt, drf, prompt, max_new, *, mode="carry", eos=None, k=4):
    return B.decode(tgt, drf, prompt, max_new, k=k, eos_id=eos, mode=mode,
                    capture_cache=True)


def test_first_round_no_pending():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 15)
    first = out["control"][0]
    assert first["pending_included"] == 0
    assert first["candidate_start_row"] == 0
    assert first["verify_input_length"] == first["kp"]


def test_emit_candidates_plus_bonus():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 15)
    for row in out["control"]:
        assert len(row["emitted_candidates"]) == row["kp"]
        assert row["emitted_bonus"] is not None or row["remaining_after"] == 0
        assert row["emitted_count"] == len(row["emitted_candidates"]) + (1 if row["emitted_bonus"] is not None else 0)
    st = out["stats"]
    assert st["candidate_tokens"] + st["bonus_tokens"] == len(out["gen"])


def test_draft_catchup_next_round():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 15)
    st = out["stats"]
    # 每个"下一轮"对上一轮 bonus 做一次 catch-up；结束时的 pending bonus 不再有下一轮
    assert st["bonus_catchup_calls"] == st["bonus_tokens"] - out["pending_at_stop"]
    assert st["bonus_catchup_calls"] == max(0, st["rounds"] - 1)


def test_target_block_starts_with_pending():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 33)
    second = out["control"][1]
    assert second["pending_included"] == 1
    assert second["candidate_start_row"] == 1
    assert second["verify_input_length"] == second["kp"] + 1
    assert second["next_bonus_row"] == second["verify_input_length"] - 1


def test_pending_not_duplicated_and_row_mapping():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 64)
    st = out["stats"]
    assert len(out["gen"]) == 64
    assert st["candidate_tokens"] == 52
    assert st["bonus_tokens"] == 12
    assert st["rounds"] == 13
    assert out["committed"] == 64
    for row in out["control"]:
        assert row["candidate_start_row"] == row["pending_included"]
        assert row["next_bonus_row"] == row["verify_input_length"] - 1
        # bonus 只出现一次：每轮最多 1 个 bonus，且不是 candidate 的复制
        if row["emitted_bonus"] is not None:
            assert len(row["emitted_candidates"]) + 1 == row["emitted_count"]


def test_rounds_13_and_tail_no_bonus():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 64)
    last = out["control"][-1]
    assert out["stats"]["rounds"] == 13
    assert last["emitted_bonus"] is None
    assert last["remaining_before"] == 4
    assert last["remaining_after"] == 0
    assert last["kp"] == 4


def test_candidate_eos():
    tgt, drf, prompt, max_new = make_case()
    ref = run(tgt, drf, prompt, 15)
    cands = ref["control"][0]["emitted_candidates"]
    eos = cands[-1]
    out = run(tgt, drf, prompt, 15, eos=eos)
    first = out["control"][0]
    assert first["kp"] == 4
    assert first["emitted_candidates"][-1] == eos
    assert first["emitted_bonus"] is None
    assert len(out["control"]) == 1
    assert out["gen"][-1] == eos
    assert out["stats"]["bonus_tokens"] == 0


def test_bonus_eos():
    tgt, drf, prompt, max_new = make_case()
    ref = run(tgt, drf, prompt, 15)
    bonus = ref["control"][0]["emitted_bonus"]
    out = run(tgt, drf, prompt, 15, eos=bonus)
    first = out["control"][0]
    assert first["emitted_bonus"] == bonus
    assert len(out["control"]) == 1
    assert out["gen"][-1] == bonus
    assert out["pending_at_stop"] == 1   # carry 模式下两侧 cache 都比 committed 少 1（pending 状态）
    fc = out["final_cache"]
    assert fc["tgt_len"] == fc["drf_len"] == prompt.shape[1] + len(out["gen"]) - 1


def test_max_new_truncation():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 7)
    assert len(out["gen"]) == 7
    assert out["stats"]["rounds"] == 2
    assert out["control"][0]["emitted_count"] == 5
    assert out["control"][1]["emitted_count"] == 2
    assert out["control"][1]["emitted_bonus"] is None
    assert out["control"][-1]["remaining_after"] == 0


def test_cache_lengths_per_round():
    tgt, drf, prompt, max_new = make_case()
    out = run(tgt, drf, prompt, 64)
    ctx = prompt.shape[1]
    for row in out["control"]:
        committed = row["committed"]
        pending_new = row["pending_new"]
        assert row["target_len"] == row["draft_len"] == ctx + committed - pending_new
        assert row["target_len"] + pending_new == ctx + committed
    fc = out["final_cache"]
    assert fc["tgt_len"] == fc["drf_len"] == ctx + len(out["gen"]) - out["pending_at_stop"]
    assert fc["tgt_consistent"] and fc["drf_consistent"]


def test_catchup_equals_carry():
    for noise in (0.0, 0.2, 0.5):
        tgt, drf, prompt, max_new = make_case(noise=noise, seed=41)
        carry = run(tgt, drf, prompt, 64, mode="carry")
        catch = run(tgt, drf, prompt, 64, mode="catchup")
        assert carry["gen"] == catch["gen"], noise
        assert carry["stats"]["rounds"] == catch["stats"]["rounds"], noise
        assert carry["stats"]["bonus_tokens"] == catch["stats"]["bonus_tokens"], noise
        assert carry["stats"]["candidate_tokens"] == catch["stats"]["candidate_tokens"], noise
        assert catch["stats"]["target_extra_single_step_calls"] == catch["stats"]["bonus_tokens"]
        assert carry["stats"]["target_extra_single_step_calls"] == 0
        for a, b in zip(carry["control"], catch["control"]):
            assert a["emitted_candidates"] == b["emitted_candidates"]
            assert a["emitted_bonus"] == b["emitted_bonus"]
            assert a["kp"] == b["kp"]


def test_defaults_unchanged():
    tgt, drf, prompt, max_new = make_case(noise=0.25, seed=27)
    strict = F.spec_decode_greedy_optimized(tgt, drf, prompt, max_new, k=4)
    relaxed0 = R.decode(tgt, drf, prompt, max_new, delta=0.0)
    assert relaxed0["gen"] == strict["gen"]
    assert relaxed0["stats"] == strict["stats"]
    no_bonus = A0.decode(tgt, drf, prompt, 64, k=4, capture_cache=True)
    assert no_bonus["stats"]["bonus_token_count"] == 0
    assert no_bonus["stats"]["rounds"] == 16
    strict2 = F.spec_decode_greedy_optimized(tgt, drf, prompt, max_new, k=4)
    assert strict2["gen"] == strict["gen"]
    bonus = run(tgt, drf, prompt, 64)
    assert bonus["accept_mode"] == "all_accept_bonus_oracle"
    assert bonus["quality_claim_allowed"] is False
    assert bonus["formal_result_allowed"] is False


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
    print(f"BONUS_ACCEPT_CPU: PASS {passed} / {passed + failed}")
    raise SystemExit(0 if failed == 0 else 1)
