"""serial_oracle_stage0.py — **Stage 0**：串行 target oracle 的纯 CPU 状态机测试（13 例 + 2 负对照）。

## 方法（为什么这样做）
本阶段要验证的是**控制逻辑**（accept / replacement / rollback / emitted / EOS / 截断 / cache 提交），
不是数值算子，因此：
* 用**脚本化 stub** 提供完全可控的 logits（target / draft 各一份脚本），**复用真实**的
  `spec_dec.spec_decode_greedy` / `_propose` / `_verify` / `_verify_serial` / `_step`；
* stub 的 cache 是真实的 `[{k,v},...]`（token 维精确增长）⇒ 逐轮核对 cache 长度与回滚；
* stub 记录**每一次**调用的 `seq_len` ⇒ 核对 serial 下 generation 调用全为 1、block 下出现 `s == Kp`；
* stub 的 logits 按"已喂入 token 序列"索引 ⇒ m=0 / m=1 / m=Kp-1 / m=Kp / EOS / 预算截断可**精确构造**。

## 脚本索引约定（由 `spec_dec` 语义推导，不是猜的）
设 `prompt_len = P`，某一路（target 或 draft）脚本第 `i` 行的 argmax 记为 `a[i]`：
* prefill 消费第 `0..P-1` 行 ⇒ `saved_next`（target）/ `drf_lg`（draft）的 argmax = `a[P-1]`；
* `cands[j] = a_draft[P-1+j]`（`_propose` 先取 argmax 再喂入该 token，故消费 `P..P+keff-1` 行）；
* `g[j] = a_target[P-1+j]`（`g[0]` 来自 `saved_next`，`g[j≥1]` 来自 block 的 `lg[0,-kp:][j-1]`
  即第 `P+j-1` 行）；
* 因此 `m` = 两条脚本 argmax 序列在 `P-1` 处对齐后的最长公共前缀长度**（与 verifier 模式无关）**；
* block 与 serial 消费的**行数与顺序完全相同**（各 `kp` 行）⇒ 二者控制流必须逐轮一致（用例 8）。

**不加载任何模型、不使用 GPU。** 任一检查失败即非零退出（Stage 0 门）。
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch

from . import serial_oracle as SO
from . import spec_dec

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
V = 32                      # 词表（stub）
L = 2                       # 层数（stub cache）
D = 4                       # head_dim（stub）
SENT = 11                   # "永不 argmax" 的哑 token（必须在词表内）

GEN_KEYS = ("Kp", "m", "pre", "tgt_len", "drf_len")


class StubModel:
    """脚本化模型：脚本第 `i` 行 = 第 `i` 次喂入的 token 所对应的 logits 行（长度 V）。

    * 一次 `_run(s)` 返回 `script[pos:pos+s]`，复现"一次调用拿多行"的 block 语义；
    * 脚本用尽后返回 `onehot(SENT)`（不会与脚本内的值意外相等）；
    * cache = `[{k:(1,2,T,D), v:...}] * L`，`T` 每次精确增加 `s`；
    * `self.calls` 记录每次调用的 `s`（调用几何）。
    """
    def __init__(self, script: list, *, name: str = "stub"):
        self.script = list(script)
        self.name = name
        self.calls: list = []
        self.pos = 0

    def set_kv_format(self, fmt):        # noqa: D401
        self.fmt = fmt

    def embed_tokens(self, ids):
        return ids

    def _run(self, x, cache, *, need_logits: bool = True, **kw):
        s = int(x.shape[1])
        self.calls.append(s)
        rows = []
        for j in range(s):
            idx = self.pos + j
            rows.append(self.script[idx] if idx < len(self.script) else _onehot(SENT))
        self.pos += s
        if cache is None:
            cache = [dict(k=torch.zeros(1, 2, 0, D), v=torch.zeros(1, 2, 0, D)) for _ in range(L)]
        out = [dict(k=torch.cat([lay["k"], torch.zeros(1, 2, s, D)], dim=2),
                    v=torch.cat([lay["v"], torch.zeros(1, 2, s, D)], dim=2)) for lay in cache]
        return torch.stack(rows, dim=0).unsqueeze(0), out          # (1, s, V), cache


def _onehot(i: int) -> torch.Tensor:
    i = int(i)
    if not (0 <= i < V):
        raise ValueError(f"stub token id 必须在 [0,{V}) 内，收到 {i}")
    t = torch.full((V,), -10.0)
    t[i] = 10.0
    return t


def _seq(seq: list, prompt_len: int = 2, filler: int = 0) -> list:
    """构造脚本：前 `prompt_len-1` 行为 filler（唯一作用是让 `a[P-1]` 对上 `seq[0]`）。"""
    return [_onehot(filler)] * (prompt_len - 1) + [_onehot(v) for v in seq]


def _case(name: str, *, prompt_len: int, tgt_seq: list, drf_seq: list, max_new: int, k: int,
          eos_id=None, mode: str = "serial", audit: bool = True, set_mode: bool = True,
          wrap=None) -> dict:
    """跑一个受控 round 序列，返回逐轮 trace 与调用几何。"""
    tgt = StubModel(_seq(tgt_seq, prompt_len), name="target")
    drf = StubModel(_seq(drf_seq, prompt_len), name="draft")
    ids = torch.zeros(1, prompt_len, dtype=torch.long)
    rounds: list = []

    def _log(d):
        rounds.append(dict(d))

    spec_dec.clear_serial_trace()
    prev = spec_dec.set_target_verify_mode(mode) if set_mode else None
    if wrap is not None:
        spec_dec.set_verify_wrap(wrap)
    aud = aud_d = None
    try:
        with contextlib.ExitStack() as st:
            if audit:
                aud = st.enter_context(SO.target_call_audit(tgt, prompt_len=prompt_len))
            aud_d = st.enter_context(SO.draft_call_audit(drf))
            out = spec_dec.spec_decode_greedy(tgt, drf, ids, max_new, k=k, eos_id=eos_id, log=_log)
    finally:
        spec_dec.set_verify_wrap(None)
        spec_dec.set_target_verify_mode("block")
    return dict(name=name, mode=mode, prompt_len=prompt_len, k=k, max_new=max_new, eos_id=eos_id,
                prev_mode=(prev or {}).get("previous"), gen=list(out["gen"]), stats=out["stats"],
                rounds=rounds, serial_trace=spec_dec.serial_trace(),
                target_calls=list(tgt.calls), draft_calls=list(drf.calls),
                audit=(aud.summary() if aud else None),
                audit_draft=(aud_d.summary() if aud_d else None))


def _case_raise(name: str, **kw) -> dict:
    """跑一个**预期抛异常**的用例；无论结果如何都恢复模式/包装。"""
    try:
        c = _case(name, **kw)
        return dict(name=name, raised=None, case=c)
    except Exception as ex:                                            # noqa: BLE001
        return dict(name=name, raised=type(ex).__name__, msg=str(ex)[:300], case=None)
    finally:
        spec_dec.set_verify_wrap(None)
        spec_dec.set_target_verify_mode("block")


def _rt(d: dict) -> tuple:
    """一轮的"控制流指纹"（与 verifier 模式无关的量）。"""
    return tuple(d[k] for k in GEN_KEYS) + (tuple(d["emitted"]), tuple(d["g"]), tuple(d["cands"]))


def _ck(name: str, cond: bool, extra=None) -> tuple:
    return (name, bool(cond), extra)


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    res: list = []
    cases: list = []
    checks: list = []

    # 用例 1：m=0（第一个候选即被拒 ⇒ emitted = [replacement]）
    c = _case("m0", prompt_len=2, tgt_seq=[1, 9, 9, 9, 9], drf_seq=[5, 9, 9, 9, 9],
              max_new=1, k=4)
    cases.append(c)
    r0 = c["rounds"][0]
    checks += [_ck("c01_m0_gen_is_replacement", c["gen"] == [1], c["gen"]),
               _ck("c01_m0_m_is_0", r0["m"] == 0, r0["m"]),
               _ck("c01_m0_gen_calls_all_s1", all(s == 1 for s in c["target_calls"][1:]),
                   c["target_calls"]),
               _ck("c01_m0_audit_all_m1", c["audit"]["all_generation_m1"], c["audit"])]

    # 用例 2：m=1
    c = _case("m1", prompt_len=2, tgt_seq=[3, 8, 8, 8, 8, 8, 8], drf_seq=[3, 7, 7, 7, 7, 7, 7],
              max_new=2, k=4)
    cases.append(c)
    checks += [_ck("c02_m1_m_is_1", c["rounds"][0]["m"] == 1, c["rounds"][0]["m"]),
               _ck("c02_m1_gen", c["gen"] == [3, 8], c["gen"])]

    # 用例 3：m=Kp-1
    c = _case("m_kp_minus_1", prompt_len=2, tgt_seq=[3, 4, 5, 6, 9, 9, 9, 9, 9, 9],
              drf_seq=[3, 4, 5, 7, 9, 9, 9, 9, 9, 9], max_new=4, k=4)
    cases.append(c)
    checks += [_ck("c03_m_is_Kp_minus_1", c["rounds"][0]["m"] == 3, c["rounds"][0]["m"]),
               _ck("c03_gen", c["gen"] == [3, 4, 5, 6], c["gen"])]

    # 用例 4：m=Kp（全接受；bonus 行留给下一轮，不并入本轮 emitted）
    c = _case("m_kp", prompt_len=2, tgt_seq=[3, 4, 5, 6, 7, 7, 7, 7, 7, 7],
              drf_seq=[3, 4, 5, 6, SENT, SENT, SENT, SENT, SENT, SENT], max_new=5, k=4)
    cases.append(c)
    r0 = c["rounds"][0]
    checks += [_ck("c04_full_accept_m_eq_Kp", r0["m"] == 4, r0["m"]),
               _ck("c04_emitted_is_all_cands", r0["emitted"] == r0["cands"],
                   (r0["emitted"], r0["cands"])),
               _ck("c04_gen_calls_all_s1", all(s == 1 for s in c["target_calls"][1:]),
                   c["target_calls"]),
               _ck("c04_round2_uses_bonus_next_round",
                   len(c["rounds"]) >= 2 and c["rounds"][1]["m"] == 0 and c["gen"] == [3, 4, 5, 6, 7],
                   dict(rounds=len(c["rounds"]), gen=c["gen"]))]

    # 用例 5：EOS 出现在 replacement
    c = _case("eos_at_replacement", prompt_len=2, tgt_seq=[3, 2, 2, 2, 2, 2, 2, 2],
              drf_seq=[3, 7, 7, 7, 7, 7, 7, 7], max_new=8, k=4, eos_id=2)
    cases.append(c)
    r0 = c["rounds"][0]
    checks += [_ck("c05_stop_at_eos_replacement", bool(c["gen"]) and c["gen"][-1] == 2, c["gen"]),
               # 精确语义：EOS 只能通过 **replacement** 进入（m == 1 < Kp），且被接受的前缀里没有 EOS。
               # 首版误期望 m == 0（错）：`g[0] = a_t[P-1] = 3` 与 `cands[0] = a_d[P-1] = 3` 相等是脚本约定
               # 决定的，EOS 落在 `g[1]`（replacement 位）。**修正的是断言，不是被测代码。**
               _ck("c05_eos_entered_via_replacement_path",
                   r0["m"] == 1 and r0["m"] < r0["Kp"] and r0["emitted"][-1] == 2
                   and r0["emitted"][:-1] == r0["cands"][:r0["m"]]
                   and 2 not in r0["cands"][:r0["m"]], r0)]

    # 用例 6：EOS 出现在 accepted candidate ⇒ draft 提前 break（Kp < k）且立即停止
    c = _case("eos_at_candidate", prompt_len=2, tgt_seq=[3, 2, 2, 2, 2, 2, 2, 2],
              drf_seq=[3, 2, 2, 2, 2, 2, 2, 2], max_new=8, k=4, eos_id=2)
    cases.append(c)
    checks += [_ck("c06_kp_lt_k_when_draft_hits_eos", c["rounds"][0]["Kp"] < 4,
                   c["rounds"][0]["Kp"]),
               _ck("c06_stop_at_eos_candidate", bool(c["gen"]) and c["gen"][-1] == 2, c["gen"])]

    # 用例 7：预算截断（max_new 剩余 < k ⇒ Kp = 剩余）
    c = _case("budget_truncation", prompt_len=2, tgt_seq=[3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7],
              drf_seq=[3, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9], max_new=2, k=4)
    cases.append(c)
    checks += [_ck("c07_Kp_eq_remaining_budget", c["rounds"][0]["Kp"] == 2, c["rounds"][0]["Kp"]),
               _ck("c07_budget_respected", len(c["gen"]) == 2, c["gen"])]

    # 用例 8：同一脚本下 **block 与 serial 全循环逐轮等价**（核心等价性在 stub 层的直接检验）
    kw = dict(prompt_len=2, tgt_seq=[3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7],
              drf_seq=[3, 4, 5, 6, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT],
              max_new=6, k=4)
    cB = _case("same_script_block", mode="block", audit=False, **kw)
    cS = _case("same_script_serial", mode="serial", audit=True, **kw)
    cases += [cB, cS]
    flat = lambda c: [r[k] for r in c["rounds"] for k in GEN_KEYS]              # noqa: E731
    checks += [_ck("c08_block_serial_same_gen", cB["gen"] == cS["gen"],
                   (cB["gen"], cS["gen"])),
               _ck("c08_block_serial_same_round_control_flow",
                   [_rt(d) for d in cB["rounds"]] == [_rt(d) for d in cS["rounds"]],
                   [(_rt(a), _rt(b)) for a, b in zip(cB["rounds"], cS["rounds"]) if _rt(a) != _rt(b)]),
               _ck("c08_block_serial_same_stats", cB["stats"] == cS["stats"],
                   (cB["stats"], cS["stats"])),
               _ck("c08_serial_rows_reproduce_block_rows",
                   all(t["kp"] == d["Kp"]
                       and list(t["step_argmax"])[:max(0, d["Kp"] - 1)] == list(d["g"][1:])
                       for t, d in zip(cS["serial_trace"], cB["rounds"])),
                   dict(serial_trace=cS["serial_trace"],
                        block_g=[d["g"] for d in cB["rounds"]])),
               _ck("c08_bonus_row_chain_full_accept",
                   all(cS["serial_trace"][k]["step_argmax"][-1] == cB["rounds"][k + 1]["g"][0]
                       for k in range(len(cB["rounds"]) - 1) if cB["rounds"][k]["m"] == cB["rounds"][k]["Kp"]),
                   dict(trace=[t["step_argmax"] for t in cS["serial_trace"]],
                        g0=[d["g"][0] for d in cB["rounds"]])),
               _ck("c08_block_mode_had_s_eq_Kp", 4 in cB["target_calls"][1:], cB["target_calls"])]

    # 用例 9/10/11：cache 提交与回滚（被拒候选不落 cache；cache 长度恒等于 pre+m+1 / pre+Kp）
    c = _case("cache_rollback", prompt_len=2, tgt_seq=[3, 4, 9, 9, 9, 9, 9, 9, 9],
              drf_seq=[3, 7, 7, 7, 7, 7, 7, 7, 7], max_new=3, k=4)
    cases.append(c)
    r0 = c["rounds"][0]
    checks += [_ck("c09_rejected_candidate_not_persisted",
                   r0["tgt_len"] == r0["pre"] + r0["m"] + 1,
                   dict(tgt_len=r0["tgt_len"], pre=r0["pre"], m=r0["m"])),
               _ck("c10_draft_rolled_back_to_pre_plus_m",
                   r0["drf_len"] == r0["pre"] + r0["m"] + 1,
                   dict(drf_len=r0["drf_len"], pre=r0["pre"], m=r0["m"])),
               _ck("c11_cache_len_eq_pre_plus_emitted_each_round",
                   all(d["tgt_len"] == d["pre"] + (d["m"] + 1 if d["m"] < d["Kp"] else d["Kp"])
                       for d in c["rounds"]), c["rounds"]),
               _ck("c11b_cache_len_eq_prompt_plus_cum_emitted",
                   all(d["tgt_len"] == c["prompt_len"]
                       + sum(len(x["emitted"]) for x in c["rounds"][:i + 1])
                       for i, d in enumerate(c["rounds"])),
                   [d["tgt_len"] for d in c["rounds"]])]

    # 用例 12：异常退出后 `_run` 与模式必须恢复
    t12 = StubModel(_seq([1, 1, 1, 1]), name="t12")
    orig12 = t12._run
    raised = None
    try:
        with SO.target_call_audit(t12, prompt_len=1):
            raise RuntimeError("注入异常（验证 finally 恢复）")
    except RuntimeError as ex:
        raised = str(ex)
    # ⚠️ `_run` 是**类属性**（bound method）：每次属性访问都产生**新的** method 对象，
    #    因此 `t12._run is orig12` **恒为 False**（首版断言错，非代码未恢复）。正确核对 = `__func__` + `__self__`。
    restored12 = (getattr(t12._run, "__func__", None) is getattr(orig12, "__func__", None)
                  and getattr(t12._run, "__self__", None) is t12)
    checks += [_ck("c12_run_restored_after_exception", restored12,
                   dict(is_audit=isinstance(t12._run, SO._Audit), restored=str(t12._run))),
               _ck("c12b_audit_no_longer_installed", not isinstance(t12._run, SO._Audit),
                   str(t12._run)),
               _ck("c12_mode_restored_after_exception", spec_dec.target_verify_mode() == "block"),
               _ck("c12c_exception_propagated", raised is not None, raised)]

    # 用例 13：serial 关闭 ⇒ block 路径（出现 s == Kp 的 generation 调用）
    c = _case("block_mode_unchanged", prompt_len=2, tgt_seq=[3, 4, 5, 6, 7, 7, 7, 7, 7],
              drf_seq=[3, 4, 5, 6, 9, 9, 9, 9, 9], max_new=5, k=4, mode="block", audit=False)
    cases.append(c)
    checks += [_ck("c13_block_mode_has_s_eq_Kp_call", 4 in c["target_calls"][1:], c["target_calls"]),
               _ck("c13_mode_back_to_block", spec_dec.target_verify_mode() == "block"),
               _ck("c13_serial_trace_empty_in_block_mode", c["serial_trace"] == [],
                   c["serial_trace"])]

    # 用例 14（**负对照**）：调用几何审计必须能**失败** —— block 模式 + 审计 ⇒ CallGeometryError
    r14 = _case_raise("audit_rejects_block_mode", prompt_len=2, tgt_seq=[3, 4, 5, 6, 7, 7, 7, 7],
                      drf_seq=[3, 4, 5, 6, 9, 9, 9, 9], max_new=5, k=4, mode="block", audit=True)
    res.append(r14)
    checks += [_ck("c14_audit_rejects_block_mode_with_CallGeometryError",
                   r14["raised"] == "CallGeometryError", r14)]

    # 用例 15（**负对照**）：serial 与 row-aligned wrapper **互斥**
    r15a = _case_raise("serial_plus_wrap_install_order_a", prompt_len=2,
                       tgt_seq=[3, 9, 9], drf_seq=[3, 9, 9], max_new=1, k=2, mode="serial",
                       wrap=lambda m: contextlib.nullcontext())
    r15b_prev = spec_dec.set_target_verify_mode("serial")      # 先 serial（此时无 wrapper ⇒ 允许）
    r15b = _case_raise("serial_plus_wrap_install_order_b", prompt_len=2,
                       tgt_seq=[3, 9, 9], drf_seq=[3, 9, 9], max_new=1, k=2, mode="serial",
                       set_mode=False, wrap=lambda m: contextlib.nullcontext())
    r15b["mode_at_install"] = r15b_prev
    res += [r15a, r15b]
    checks += [_ck("c15a_wrap_then_serial_rejected", r15a["raised"] == "VerifyModeError", r15a),
               _ck("c15b_serial_then_wrap_rejected_in_loop",
                   r15b["raised"] == "VerifyModeError", r15b)]

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== Stage 0：串行 target oracle 状态机测试（纯 CPU、无模型、无 GPU）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}"
              + (f"  {str(extra)[:240]}" if (extra is not None and not ok) else ""))
    for c in cases:
        print(f"  [case] {c['name']:<18} mode={c['mode']:<6} gen={c['gen']} rounds={len(c['rounds'])}"
              f" tgt_calls={c['target_calls']}")
    print(f"\nStage 0：PASS {npass} / FAIL {nfail}")

    rec = dict(tool="serial_oracle_stage0", run_id=run_id,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               semantics=("纯 CPU 状态机测试：脚本化 stub（可控 logits/真实 cache 结构）"
                          "+ 真实 spec_dec 控制逻辑（accept/replacement/rollback/EOS/截断/cache）；"
                          "不加载模型、不使用 GPU；含 2 个负对照（审计器必须能失败、模式互斥必须报错）"),
               design_doc="docs/serial_target_oracle_design_2026-10-03.md",
               script_index_convention=("a[P-1]=saved_next/drf_lg；cands[j]=a_draft[P-1+j]；"
                                        "g[j]=a_target[P-1+j]；block 与 serial 消费同一批 kp 行"),
               diagnostic_only=True, gpu_used=False, model_loaded=False,
               first_run_fail_ref="output/g5/attn/serial_oracle_stage0_20261003T134138Z.json",
               assertion_defects_fixed=[
                   dict(check="c05_m_is_0_then_eos", kind="test_expectation_wrong",
                        detail=("脚本索引约定决定 g[0]=a_t[P-1] 与 cands[0]=a_d[P-1] 相等 ⇒ m=1，"
                                "EOS 落在 g[1]（replacement 位）；原断言 m==0 是错的。"
                                "改为精确核对 EOS 经 replacement 路径进入（m<Kp 且被接受前缀无 EOS）。")),
                   dict(check="c12_run_restored_after_exception", kind="invalid_identity_check",
                        detail=("`_run` 为类属性（bound method），每次访问都是新对象 ⇒ `t12._run is orig12` "
                                "恒 False，与是否恢复无关；改为核对 `__func__`+`__self__` 并加 "
                                "`not isinstance(_run, _Audit)`。")),
               ],
               npass=npass, nfail=nfail, decision=("STAGE0_PASS" if nfail == 0 else "STAGE0_FAIL"),
               checks=[dict(name=n, ok=bool(o), extra=(None if o else e)) for n, o, e in checks],
               equal_script_pair=dict(block_gen=cB["gen"], serial_gen=cS["gen"],
                                      block_rounds=cB["rounds"], serial_rounds=cS["rounds"]),
               cases=[dict(name=c["name"], mode=c["mode"], prompt_len=c["prompt_len"], k=c["k"],
                           max_new=c["max_new"], eos_id=c["eos_id"], gen=c["gen"], stats=c["stats"],
                           rounds=c["rounds"], serial_trace=c["serial_trace"],
                           target_calls=c["target_calls"], draft_calls=c["draft_calls"],
                           audit=c["audit"], audit_draft=c["audit_draft"]) for c in cases],
               expected_raise_cases=res)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"serial_oracle_stage0_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[stage0] RESULT_JSON={p}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
