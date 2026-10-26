"""CPU 测试：Draft-Free oracle 的候选切块、64-token 预算与 draft 调用护栏。"""
from __future__ import annotations

import sys

from experiments.mechanism.draft_free_oracle import DraftCallGuard, candidate_layout


def test_layout_64_tokens():
    frozen = list(range(64))
    layout = candidate_layout(frozen, k=4, max_new=64)
    assert len(layout) == 13
    assert layout[0]["cands"] == [0, 1, 2, 3] and layout[0]["expected_bonus"] == 4
    assert layout[0]["start"] == 0
    assert layout[1]["cands"] == [5, 6, 7, 8] and layout[1]["expected_bonus"] == 9
    assert layout[-1]["cands"] == [60, 61, 62, 63]
    assert layout[-1]["expected_bonus"] is None
    assert sum(len(x["cands"]) for x in layout) == 52
    assert sum(1 for x in layout if x["expected_bonus"] is not None) == 12


def test_layout_positions_contiguous():
    frozen = list(range(64))
    layout = candidate_layout(frozen, k=4, max_new=64)
    committed = 0
    for row in layout:
        assert row["start"] == committed
        committed += len(row["cands"]) + (1 if row["expected_bonus"] is not None else 0)
    assert committed == 64


def test_total_budget_respected():
    frozen = list(range(64))
    layout = candidate_layout(frozen, k=4, max_new=64)
    total = 0
    for row in layout:
        total += len(row["cands"]) + (1 if row["expected_bonus"] is not None else 0)
        assert total <= 64
    assert total == 64


def test_guard_negative():
    class FakeDraft:
        def _run(self, x, cache=None, need_logits=True):
            return None, cache

    fake = FakeDraft()
    guard = DraftCallGuard(fake)
    with guard:
        try:
            fake._run(None)
        except RuntimeError as exc:
            assert "draft forward" in str(exc)
            raised = True
        else:
            raised = False
    assert raised and guard.calls == 1
    # 退出后恢复原方法
    assert fake._run.__name__ == "_run"


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
    print(f"DRAFT_FREE_CPU: PASS {passed} / {passed + failed}")
    raise SystemExit(0 if failed == 0 else 1)
