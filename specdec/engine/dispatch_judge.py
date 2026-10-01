"""dispatch_judge.py — 阶段分流（attention backend）的**按后端**判定（纯函数，CPU 可测）。

为什么需要
  旧检查 `dispatch_only_prefill` 无论后端都要求 `eager_by_phase['prefill'] > 0`，
  于是**正常的全融合 cuDNN prefill** 会被误判（真发生过）。
  这里改成**按后端**判定，并且**不另造规则**：
  「prefill 是否**应当**融合」由引擎自己的 `model_self.decide_attention_path('prefill', s_max)`
  裁决 —— 也就是说判定依据是**生产策略契约本身**，不是本文件重新发明的一套阈值。

判定（`stats` = `model_self.attention_stats()` 的返回）
  公共：`by_phase['prefill'] == 0` ⇒ 失败（**prefill 从未执行**）。
        任何 `phase != 'prefill'` 的融合调用 ⇒ 失败（契约：仅 prefill 允许融合）。
  eager 臂：`eager_by_phase['prefill'] > 0` 且**融合调用严格为 0**；
            另按模式要求 `decode`（AR）或 `verify`（Spec）有 eager 调用。
  cudnn_prefill 臂：用 `decide_attention_path('prefill', s_max_by_phase['prefill'])` 求**期望路径**：
        * 期望 `fused` ⇒ 必须 `fused_by_phase['prefill'] > 0`（否则 = **后端未实际启用**/静默不生效）；
        * 期望 `eager`（已定义的回退，例如 `s < PREFILL_FUSED_MIN_S`）⇒ 记入 `fallback_reason`，
          **不**判失败（这是既有策略允许的合法回退）。
"""
from __future__ import annotations

from . import model_self as M

ALLOWED_FUSED_PHASES = ("prefill",)
NON_PREFILL_PHASES = ("decode", "propose", "verify", "repair")


def judge_dispatch(arm: str, mode: str, stats: dict) -> dict:
    """返回 `dict(ok, failures, expected_prefill_path, fallback_reason, detail, counts)`。"""
    failures: list = []
    by_phase = dict(stats.get("by_phase") or {})
    fused_by_phase = dict(stats.get("fused_by_phase") or {})
    eager_by_phase = dict(stats.get("eager_by_phase") or {})
    s_max = dict(stats.get("s_max_by_phase") or {})
    strategy = stats.get("strategy")
    counts = dict(calls=stats.get("calls") or {}, by_phase=by_phase,
                  fused_by_phase=fused_by_phase, eager_by_phase=eager_by_phase,
                  s_max_by_phase=s_max, strategy=strategy)

    # 1) prefill 必须真的执行过
    if int(by_phase.get("prefill", 0)) <= 0:
        failures.append("prefill 从未执行（by_phase['prefill'] == 0）")

    # 2) 融合只允许出现在 prefill（契约：decide_attention_path 对非 prefill 一律 eager）
    bad_fused = {p: v for p, v in fused_by_phase.items()
                 if v > 0 and p not in ALLOWED_FUSED_PHASES}
    if bad_fused:
        failures.append(f"非 prefill 阶段出现融合调用：{bad_fused}")

    # 3) 策略名必须与实际生效的后端一致
    if arm == "eager":
        if strategy not in (None, "eager"):
            failures.append(f"arm=eager 但记录里的策略是 {strategy!r}")
    elif arm == "cudnn_prefill":
        if strategy not in (None, "cudnn_prefill"):
            failures.append(f"arm=cudnn_prefill 但记录里的策略是 {strategy!r}")
    else:
        failures.append(f"未知 arm {arm!r}")

    # 4) 按 arm 判定 prefill 路径
    expected_path, fallback_reason = (None, None)
    if arm == "eager":
        if int(eager_by_phase.get("prefill", 0)) <= 0:
            failures.append("arm=eager 但 prefill 没有走 eager（未实际执行 eager prefill）")
        if any(v > 0 for v in fused_by_phase.values()):
            failures.append(f"arm=eager 但存在融合调用："
                            f"{ {p: v for p, v in fused_by_phase.items() if v > 0} }")
    elif arm == "cudnn_prefill":
        s_pref = int(s_max.get("prefill", 0))
        expected_path, fallback_reason = M.decide_attention_path("prefill", s_pref)
        if expected_path == "fused":
            if int(fused_by_phase.get("prefill", 0)) <= 0:
                failures.append("arm=cudnn_prefill 且按策略 prefill 应融合，但 fused_by_phase['prefill']==0"
                                "（后端未实际启用/静默不生效）")
        else:
            # 合法回退：不判失败，但要留下**依据**
            fallback_reason = fallback_reason or "策略对该 s 判定为 eager（见 decide_attention_path）"
            if int(eager_by_phase.get("prefill", 0)) <= 0:
                failures.append("期望回退到 eager prefill，但 eager_by_phase['prefill']==0")

    # 5) 按模式要求后续阶段确实走过（防"没跑就报通过"）
    if mode == "ar":
        if int(eager_by_phase.get("decode", 0)) <= 0:
            failures.append("mode=ar 但 eager_by_phase['decode']==0（AR 解码未执行过 eager 调用）")
    elif mode == "spec":
        if int(eager_by_phase.get("verify", 0)) <= 0:
            failures.append("mode=spec 但 eager_by_phase['verify']==0（verify 未执行过 eager 调用）")
        if int(eager_by_phase.get("propose", 0)) <= 0:
            failures.append("mode=spec 但 eager_by_phase['propose']==0（draft propose 未执行）")
    else:
        failures.append(f"未知 mode {mode!r}")

    return dict(ok=not failures, failures=failures, expected_prefill_path=expected_path,
                fallback_reason=fallback_reason, counts=counts,
                semantics="按后端判定；期望路径来自生产 `decide_attention_path` 契约；"
                          "非 prefill 阶段融合一律失败；prefill 未执行一律失败")
