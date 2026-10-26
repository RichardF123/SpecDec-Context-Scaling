"""2K bonus-aware 端到端 Graph 的冻结范围与轮次表（F09/2K/64-token，13 轮）。"""
from __future__ import annotations

CTX = 2048
K = 4
MAX_NEW = 64
ROUNDS = 13

# 每轮 draft 链起点（draft cache 长度）与步数、target verify 起点与 s
DRAFT_START = [CTX] + [CTX + 5 * r - 1 for r in range(1, ROUNDS)]
DRAFT_STEPS = [4] + [5] * (ROUNDS - 1)
VERIFY_START = [CTX] + [CTX + 5 * r - 1 for r in range(1, ROUNDS)]
VERIFY_S = [4] + [5] * (ROUNDS - 1)
# 每轮提交的 gen 区间：candidates 4 个（round r 起点 = 5r），bonus 1 个（除最后一轮）
GEN_CAND_START = [5 * r for r in range(ROUNDS)]
BONUS_POS = [5 * r + 4 for r in range(ROUNDS - 1)]
MAX_CAPACITY = CTX + MAX_NEW + K   # 2116

EXPERIMENT_CLASS = "end_to_end_graph_feasibility_diagnostic"
SCOPE = "F09/2K/bonus-all-accept/64-token"


def check_schedule():
    """校验轮次表与 bonus carry-forward 契约一致。"""
    assert len(DRAFT_START) == len(DRAFT_STEPS) == len(VERIFY_START) == len(VERIFY_S) == ROUNDS
    assert DRAFT_STEPS[0] == 4 and all(s == 5 for s in DRAFT_STEPS[1:])
    assert VERIFY_S[0] == 4 and all(s == 5 for s in VERIFY_S[1:])
    assert DRAFT_START[0] == VERIFY_START[0] == CTX
    emitted = 0
    for r in range(ROUNDS):
        cand = 4
        bonus = 0 if r == ROUNDS - 1 else 1
        emitted += cand + bonus
        if r + 1 < ROUNDS:
            assert DRAFT_START[r + 1] == CTX + emitted - 1, (r, DRAFT_START[r + 1], emitted)
            assert VERIFY_START[r + 1] == CTX + emitted - 1
    assert emitted == MAX_NEW
    assert sum(4 for _ in range(ROUNDS)) == 52
    assert sum(0 if r == ROUNDS - 1 else 1 for r in range(ROUNDS)) == 12
    return dict(rounds=ROUNDS, candidates=52, bonus=12, total=MAX_NEW,
                draft_starts=DRAFT_START, verify_starts=VERIFY_START,
                capacity=MAX_CAPACITY)


def amortized(capture_s: float, replay_s: float, n: int) -> float:
    return (capture_s + n * replay_s) / n


def break_even_requests(capture_s: float, replay_s: float, ar_s: float):
    """求最小 N，使 T_avg(N) <= AR（Graph 请求平均延迟不劣于 AR）。"""
    if replay_s >= ar_s:
        return None
    n = 1
    while n < 10**9:
        if amortized(capture_s, replay_s, n) <= ar_s:
            return n
        n *= 2
    return None
