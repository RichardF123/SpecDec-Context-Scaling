"""CPU 测试：2K/64-token carry-forward 轮次表、摊销与 break-even 公式。"""
from __future__ import annotations

import sys

from experiments.cuda_graph_e2e import design


def test_schedule():
    s = design.check_schedule()
    assert s["rounds"] == 13 and s["candidates"] == 52 and s["bonus"] == 12 and s["total"] == 64
    assert s["draft_starts"][0] == 2048
    assert s["draft_starts"][-1] == 2107
    assert s["capacity"] == 2116


def test_verify_lengths():
    assert design.VERIFY_S[0] == 4 and all(x == 5 for x in design.VERIFY_S[1:])
    assert design.DRAFT_STEPS[0] == 4 and all(x == 5 for x in design.DRAFT_STEPS[1:])


def test_gen_positions():
    # candidates 占 5r..5r+3；bonus 占 5r+4（最后一轮无 bonus）
    assert design.GEN_CAND_START[0] == 0 and design.GEN_CAND_START[-1] == 60
    assert design.BONUS_POS == [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    assert max(design.BONUS_POS) < design.MAX_NEW


def test_amortized():
    t = design.amortized(10.0, 2.5, 10)
    assert abs(t - (10.0 + 25.0) / 10) < 1e-9
    # capture 越大，N 越大越接近 replay
    assert design.amortized(10.0, 2.5, 100) < design.amortized(10.0, 2.5, 10) < design.amortized(10.0, 2.5, 1)


def test_break_even():
    n = design.break_even_requests(6.0, 2.0, 2.374)
    assert n is not None and n > 1
    assert design.amortized(6.0, 2.0, n) <= 2.374 + 1e-9
    # 若 replay 本身就慢于 AR，任何 N 都不可能 break-even
    assert design.break_even_requests(6.0, 2.5, 2.374) is None


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
    print(f"E2E_CPU: PASS {passed} / {passed + failed}")
    raise SystemExit(0 if failed == 0 else 1)
