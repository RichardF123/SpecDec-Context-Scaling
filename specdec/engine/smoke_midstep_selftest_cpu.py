"""smoke_midstep_selftest_cpu.py — **`_propose` 中间步**有限性检查的轻量集成测试（**不加载大模型**）。

要证明的事（任务要求）
  * `spec_dec._propose` 的循环里，每次 `draft._run` 产出的 logits 会被**下一次迭代的 argmax** 使用；
    ⇒ 只在"入口/出口"检查**不足以**覆盖它；
  * 若第 2 次 forward 产出 NaN、而后续 forward 会恢复有限，检查必须在该 NaN
    **进入下一次 argmax 之前**停止，并且**不再继续 forward**；
  * 同时覆盖：Inf、正常多步、`k_eff` 缩短、EOS 提前结束。

做法：用**真实**的 `spec_dec._propose` 与**真实**的 `DecisionHooks`（与 8K smoke 同一套代码路径），
只把 draft 模型换成一个按脚本产出 logits 的 stub —— 不复制投机算法、不改生产语义。

用法：python -m specdec.engine.smoke_midstep_selftest_cpu
"""
from __future__ import annotations

import torch

from . import model_self as M
from . import spec_dec
from .smoke_checks import CheckRegistry, FinitenessError, verify_decision_coverage
from .smoke_hooks import DecisionHooks

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


V = 8


def _lg(logits, nan: bool = False, inf: bool = False):
    t = torch.tensor([[float(x) for x in logits]], dtype=torch.float32)
    if nan:
        t[0, 0] = float("nan")
    if inf:
        t[0, 1] = float("inf")
    return t


class StubDraft:
    """按脚本产出 logits 的假 draft：第 i 次 `_run` 返回 `script[i]`。

    `script` 只描述"每次 forward 产出的 logits"；`_propose` 的算法本身是**真实的**。
    """

    def __init__(self, script: list):
        self.script = script
        self.calls = 0
        self.argmax_inputs: list = []

    def embed_tokens(self, ids):
        return ids.float()                                  # 占位，_run 不使用

    def _run(self, x, cache=None, need_logits=True):
        i = self.calls
        self.calls += 1
        if i >= len(self.script):
            return _lg([0.0] * V), cache                     # 脚本用尽后给有限值
        return self.script[i], cache


def _install(draft, reg):
    """装一套**真实** hooks：`_run` hook 装在该 stub 的类上（生产用 `QwenLM`），
    因此测试跑的是与 8K smoke **同一套 hook 代码**，且完全不碰生产类。"""
    hooks = DecisionHooks(tgt=None, drf=draft, registry=reg,
                          prefill_impl=lambda m, i: (None, None),
                          run_owner=type(draft))
    hooks.install()
    return hooks


def _run_propose(script, *, k_eff=4, eos_id=None, incoming=None, use_hooks=True):
    """跑一次**真实** `spec_dec._propose`；返回 (cands, draft, reg, error)。"""
    draft = StubDraft(script)
    reg = CheckRegistry()
    hooks = _install(draft, reg) if use_hooks else None
    lg = incoming if incoming is not None else _lg([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    err = None
    cands = None
    try:
        cands, cache, lg_out = spec_dec._propose(draft, None, lg, k_eff, eos_id)
    except FinitenessError as e:
        err = e
    finally:
        if hooks is not None:
            hooks.remove()
    return cands, draft, reg, err


ARG = [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]     # argmax = 1


def main() -> int:
    # 记下"生产函数"的原始对象，用于 ⑨ 的**身份**核对（只比名字会因同名而失真）
    orig = dict(prefill=spec_dec._prefill, propose=spec_dec._propose, verify=spec_dec._verify,
                step=spec_dec._step, run=M.QwenLM._run, stub_run=StubDraft._run)

    print("== ① 正常多步（k_eff=4，全部有限）==")
    cands, draft, reg, err = _run_propose([_lg(ARG)] * 4)
    check("① 无异常", err is None, str(err))
    check("① 4 次 forward 全部发生", draft.calls == 4, f"calls={draft.calls}")
    check("① 4 个候选", cands is not None and len(cands) == 4, str(cands))
    mid = reg.calls[("propose", "draft")]["role_counts"].get("mid_lg", 0)
    check("① 中间步检查次数 == forward 次数（每步都检查）", mid == 4, f"mid_lg={mid}")
    cov = verify_decision_coverage(reg, draft_calls_expected=4, rounds_expected=1)
    check("① 决策步覆盖核对通过", cov["ok"] is True, cov["detail"])
    steps = reg.calls[("propose", "draft")]["steps"]
    mids = [s for s in steps if s["role"] == "mid_lg"]
    check("① 中间步步号连续 1..4 且带轮次（轮次 0 = 正在执行的第一轮）",
          [s["step"] for s in mids] == [1, 2, 3, 4] and all(s["round"] == 0 for s in mids),
          str([(s["round"], s["step"], s["role"]) for s in steps]))

    print("\n== ② 中间步 NaN + 后续会恢复有限：必须在进入下一次 argmax 前停止 ==")
    script = [_lg(ARG), _lg(ARG, nan=True), _lg(ARG), _lg(ARG)]   # 第 2 次 forward 出 NaN
    cands, draft, reg, err = _run_propose(script)
    check("② 抛出 FinitenessError", isinstance(err, FinitenessError), str(err)[:60])
    check("② **只发生了 2 次 forward**（未继续生成后再报告）", draft.calls == 2,
          f"calls={draft.calls}")
    check("② 失败记录含 round/step 与 NaN 计数",
          reg.first_bad is not None and reg.first_bad["step"] == 2 and reg.first_bad["n_nan"] == 1,
          str(reg.first_bad and {k: reg.first_bad[k] for k in ("round", "step", "role", "n_nan")}))
    check("② 失败位置标为 propose/draft/mid_lg",
          reg.first_bad["phase"] == "propose" and reg.first_bad["model"] == "draft"
          and reg.first_bad["role"] == "mid_lg")
    check("② 报错文本说明「在 argmax/接受判断之前」检出",
          "argmax/接受判断**之前**" in str(err))

    print("\n== ③ 对照：**不装 hooks** 时同一份脚本不会被拦住（证明入口/出口检查不够）==")
    cands2, draft2, reg2, err2 = _run_propose(script, use_hooks=False)
    check("③ 无 hook ⇒ 不报错（NaN 静默通过）", err2 is None, str(err2))
    check("③ 且 4 次 forward 全部发生（NaN 已被用于 argmax）", draft2.calls == 4,
          f"calls={draft2.calls}")
    check("③ 候选仍被产出 ⇒ 说明缺口真实存在",
          cands2 is not None and len(cands2) == 4, str(cands2))

    print("\n== ④ 入口/出口检查单独存在的覆盖缺口（同一脚本，只做入口+出口检查）==")
    reg3 = CheckRegistry()
    # 手动模拟「只有入口+出口检查」：入口 lg 有限、出口（最后一次）logits 有限
    reg3.check_finite(_lg(ARG), phase="propose", model="draft", role="input_lg")
    reg3.check_finite(script[-1], phase="propose", model="draft", role="return_lg")
    mid3 = reg3.calls[("propose", "draft")]["role_counts"].get("mid_lg", 0)
    cov3 = verify_decision_coverage(reg3, draft_calls_expected=4, rounds_expected=1)
    check("④ 入口+出口检查：mid_lg=0 ⇒ 覆盖核对**判不通过**", mid3 == 0 and cov3["ok"] is False,
          cov3["detail"])

    print("\n== ⑤ Inf 中间步 ==")
    cands, draft, reg, err = _run_propose([_lg(ARG), _lg(ARG, inf=True), _lg(ARG), _lg(ARG)])
    check("⑤ Inf ⇒ 抛错且只 2 次 forward", isinstance(err, FinitenessError) and draft.calls == 2,
          f"calls={draft.calls}")
    check("⑤ 记录含 Inf 计数", reg.first_bad is not None and reg.first_bad["n_inf"] == 1)

    print("\n== ⑥ 入口本身非有限 ==")
    cands, draft, reg, err = _run_propose([_lg(ARG)] * 4, incoming=_lg(ARG, nan=True))
    check("⑥ 入口 NaN ⇒ 抛错", isinstance(err, FinitenessError), str(err)[:60])
    check("⑥ 入口 NaN 时**不发生** forward（在第一次 argmax 前就停）", draft.calls == 0,
          f"calls={draft.calls}")
    check("⑥ 记录 role=input_lg", reg.first_bad is not None and reg.first_bad["role"] == "input_lg")

    print("\n== ⑦ k_eff 缩短 ==")
    cands, draft, reg, err = _run_propose([_lg(ARG)] * 2, k_eff=2)
    check("⑦ k_eff=2 ⇒ 2 次 forward / 2 个候选", draft.calls == 2 and len(cands) == 2,
          f"calls={draft.calls} cands={cands}")
    cov = verify_decision_coverage(reg, draft_calls_expected=2, rounds_expected=1)
    check("⑦ 覆盖核对按实际步数通过", cov["ok"] is True, cov["detail"])
    cov_bad = verify_decision_coverage(reg, draft_calls_expected=4, rounds_expected=1)
    check("⑦ 但若上报的 draft 前向数与实际不符 ⇒ 判不通过", cov_bad["ok"] is False, cov_bad["detail"])

    print("\n== ⑧ EOS 提前结束 ==")
    # 第 1 次 argmax 得到 eos（让 eos 出现在可 argmax 的位置），循环在 1 次 forward 后 break
    eos_lg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]      # argmax = 7
    cands, draft, reg, err = _run_propose([_lg(ARG)] * 4, k_eff=4, eos_id=7, incoming=_lg(eos_lg))
    check("⑧ EOS 触发 ⇒ 只 1 个候选、1 次 forward", err is None and cands == [7] and draft.calls == 1,
          f"cands={cands} calls={draft.calls}")
    mid = reg.calls[("propose", "draft")]["role_counts"].get("mid_lg", 0)
    check("⑧ 该次 forward 仍被检查（不因提前结束而漏检）", mid == 1, f"mid_lg={mid}")

    print("\n== ⑨ 收尾：hook 必须干净卸载（不改生产语义）==")
    _d = StubDraft([_lg(ARG)])
    _h = _install(_d, CheckRegistry())
    _stub_changed = StubDraft._run is not orig["stub_run"]
    _prod_untouched = M.QwenLM._run is orig["run"]
    _h.remove()
    check("⑨ 安装后确实替换了 stub 的 _run（否则覆盖证明无意义）", _stub_changed)
    check("⑨ 安装期间**完全不碰生产类** QwenLM._run", _prod_untouched)
    check("⑨ 卸载后按**对象身份**恢复（不是只比名字）",
          spec_dec._prefill is orig["prefill"] and spec_dec._propose is orig["propose"]
          and spec_dec._verify is orig["verify"] and spec_dec._step is orig["step"]
          and StubDraft._run is orig["stub_run"] and M.QwenLM._run is orig["run"])

    print(f"\nCPU 集成测试：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
