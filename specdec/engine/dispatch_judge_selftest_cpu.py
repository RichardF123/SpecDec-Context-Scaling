"""dispatch_judge_selftest_cpu.py — 按后端阶段分流判定的 CPU 自测。

关键：**不手工填布尔值**，而是用引擎**自己的**记录路径生成 `attention_stats()`：
  1. `M.set_attention_strategy(arm)`；
  2. 对一组 `(phase, s)` 依次调用 `M.decide_attention_path(phase, s)`（**生产裁决函数**）；
  3. 用 `M._record_attention_call(...)`（**生产记录函数**）把结果写进真实统计；
  4. 再让 `judge_dispatch` 基于 `M.attention_stats()` 判定。
⇒ 自测覆盖的是**真实 dispatch 判定逻辑**，而不是断言一个手工构造的字典。

覆盖场景（对应任务 §二）：
  合法 eager；合法 cuDNN（含全融合 prefill）；未执行 prefill；错误阶段融合；
  后端没有实际启用（cudnn 策略却零融合）；已定义回退（s < PREFILL_FUSED_MIN_S）；
  以及「旧检查会误判」的对照（全融合 cuDNN 在旧口径下会被判失败）。

运行：`python -m specdec.engine.dispatch_judge_selftest_cpu`
"""
from __future__ import annotations

from . import model_self as M
from .dispatch_judge import judge_dispatch


def _stats_from(arm: str, calls: list) -> dict:
    """用**生产函数**驱动一轮记录，返回真实 `attention_stats()`。"""
    M.set_attention_strategy(arm)
    M.reset_attention_stats()
    for phase, s in calls:
        T = s if phase in ("prefill", "other") else max(s, 4)
        path, reason = M.decide_attention_path(phase, s)
        M._record_attention_call(path, phase, s, T, reason, 12, 2, 128)
    return M.attention_stats()


def _legacy_check(stats: dict) -> bool:
    """**旧口径**（本轮要修的那个）：无论后端都要求 eager prefill > 0 且融合只在 prefill。"""
    fused = stats.get("fused_by_phase") or {}
    eager = stats.get("eager_by_phase") or {}
    return (sorted(k for k, v in fused.items() if v > 0) == ["prefill"]
            and all(fused.get(p, 0) == 0 for p in ("propose", "verify", "repair", "decode"))
            and eager.get("prefill", 0) > 0)


def main() -> int:
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    AR = [("prefill", 2048)] * 4 + [("decode", 1)] * 8
    SP = [("prefill", 2048)] * 4 + [("propose", 1)] * 6 + [("verify", 4)] * 3 + [("repair", 1)] * 2

    print("== 1. 合法 eager ==")
    st = _stats_from("eager", AR)
    j = judge_dispatch("eager", "ar", st)
    check("合法 eager/AR 通过", j["ok"], j["failures"])
    check("记录里 eager prefill 被真实计入", st["eager_by_phase"].get("prefill", 0) == 4, st["eager_by_phase"])
    check("★ eager 下融合严格为 0", not any(v > 0 for v in st["fused_by_phase"].values()))
    st2 = _stats_from("eager", SP)
    j2 = judge_dispatch("eager", "spec", st2)
    check("合法 eager/Spec 通过（propose+verify+repair 都走过）", j2["ok"], j2["failures"])

    print("== 2. 合法 cuDNN（全融合 prefill）与**旧检查的误判** ==")
    stc = _stats_from("cudnn_prefill", AR)
    jc = judge_dispatch("cudnn_prefill", "ar", stc)
    check("★ 全融合 cuDNN prefill 通过新判定", jc["ok"], jc["failures"])
    check("★ 期望路径来自生产契约 = fused", jc["expected_prefill_path"] == "fused",
          jc["expected_prefill_path"])
    check("融合只出现在 prefill", sorted(k for k, v in stc["fused_by_phase"].items() if v > 0) == ["prefill"],
          stc["fused_by_phase"])
    check("★ 旧口径会把这个**正常**运行判为失败（证明本轮修的是真问题）",
          _legacy_check(stc) is False, stc["eager_by_phase"])
    stc2 = _stats_from("cudnn_prefill", SP)
    check("合法 cuDNN/Spec 通过", judge_dispatch("cudnn_prefill", "spec", stc2)["ok"])

    print("== 3. prefill 从未执行 ==")
    st3 = _stats_from("eager", [("decode", 1)] * 8)
    j3 = judge_dispatch("eager", "ar", st3)
    check("★ prefill 未执行 ⇒ 失败", not j3["ok"] and any("从未执行" in f for f in j3["failures"]),
          j3["failures"])

    print("== 4. 非 prefill 阶段出现融合 ==")
    st4 = _stats_from("eager", AR)
    st4["fused_by_phase"] = dict(st4["fused_by_phase"], verify=3)   # 人为注入"错误阶段融合"
    st4["by_phase"] = dict(st4["by_phase"], verify=3)
    j4 = judge_dispatch("eager", "spec", st4)
    check("★ verify 阶段融合 ⇒ 失败", not j4["ok"] and any("非 prefill" in f for f in j4["failures"]),
          j4["failures"])
    st4b = _stats_from("cudnn_prefill", AR)
    st4b["fused_by_phase"] = dict(st4b["fused_by_phase"], decode=5)
    j4b = judge_dispatch("cudnn_prefill", "ar", st4b)
    check("★ cuDNN 臂下 decode 融合也 ⇒ 失败",
          not j4b["ok"] and any("非 prefill" in f for f in j4b["failures"]), j4b["failures"])

    print("== 5. 后端没有实际启用（策略=cudnn 但零融合；且 s 达到融合门槛）==")
    st5 = _stats_from("cudnn_prefill", [("decode", 1)] * 4)
    st5["by_phase"] = dict(st5["by_phase"], prefill=4)             # 声称跑过 prefill
    st5["eager_by_phase"] = dict(st5["eager_by_phase"], prefill=4)  # 但实际是 eager
    st5["s_max_by_phase"] = dict(st5["s_max_by_phase"], prefill=2048)
    j5 = judge_dispatch("cudnn_prefill", "ar", st5)
    check("★ 策略=cudnn_prefill 却零融合（s=2048 应融合）⇒ 失败，且原因写明未实际启用",
          not j5["ok"] and any("未实际启用" in f for f in j5["failures"]), j5["failures"])

    print("== 6. 已定义回退（s < PREFILL_FUSED_MIN_S）不判失败但记录依据 ==")
    st6 = _stats_from("cudnn_prefill", [("prefill", 1)] * 2 + [("decode", 1)] * 3)
    j6 = judge_dispatch("cudnn_prefill", "ar", st6)
    check("★ s=1 的 prefill 回退到 eager ⇒ 通过", j6["ok"], j6["failures"])
    check("★ 且记录回退依据与期望路径", j6["expected_prefill_path"] == "eager"
          and bool(j6["fallback_reason"]), j6)
    check("门槛常量来自引擎", M.PREFILL_FUSED_MIN_S == 2, M.PREFILL_FUSED_MIN_S)

    print("== 7. 策略名与实际不符 / 未知 arm / 未知 mode ==")
    st7 = _stats_from("eager", AR)
    check("★ arm=eager 但记录策略是 cudnn_prefill ⇒ 失败",
          not judge_dispatch("eager", "ar", dict(st7, strategy="cudnn_prefill"))["ok"])
    check("★ 未知 arm ⇒ 失败", not judge_dispatch("weird", "ar", st7)["ok"])
    check("★ 未知 mode ⇒ 失败", not judge_dispatch("eager", "weird", st7)["ok"])

    print("== 8. 模式要求：AR 必须有 decode、Spec 必须有 verify/propose ==")
    st8 = _stats_from("eager", [("prefill", 2048)])
    j8 = judge_dispatch("eager", "ar", st8)
    check("★ AR 但 decode 为 0 ⇒ 失败", not j8["ok"] and any("decode" in f for f in j8["failures"]),
          j8["failures"])
    j8b = judge_dispatch("eager", "spec", st8)
    check("★ Spec 但 verify/propose 为 0 ⇒ 失败",
          not j8b["ok"] and any("verify" in f for f in j8b["failures"]), j8b["failures"])

    print("== 9. 判定不改动引擎状态 ==")
    before = M.attention_strategy()
    st9 = _stats_from("cudnn_prefill", AR)
    judge_dispatch("cudnn_prefill", "ar", st9)
    check("judge_dispatch 是纯判定（不改变策略/不写统计）",
          M.attention_strategy() == "cudnn_prefill" and M.attention_stats()["calls"]["fused"] == 4)

    M.set_attention_strategy("eager")   # 还原默认，避免影响同进程后续测试
    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
