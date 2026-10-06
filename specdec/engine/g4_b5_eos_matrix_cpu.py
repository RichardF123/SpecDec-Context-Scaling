"""g4_b5_eos_matrix_cpu.py — EOS 分支的**矩阵化 CPU 覆盖**（纯 CPU；不加载模型、不使用 GPU）。

## 与本仓库既有工具的关系（**复用，不重复造工具**）
* `g4_b5_eos_branch_tests.py`（**已存在**，2026-09 修 EOS cache 越界时写的）覆盖 4 条分支，
  并带"同一组 case 在**旧实现**下必须抛 ValueError"的非空过保证；
* 本模块**复用**它的 `StubModel`（位置策略 stub）、`_propose_old`（旧实现对照）与 `invariants`，
  在其上补 **6 条此前未覆盖的分支**，并把断言从"gen 相等 + 长度不变式"扩展到
  **停止原因 / EOS 唯一性与位置 / EOS 之后不再生成 / cache 提交合同 / 候选不含 EOS 之后 token /
  分支见证（确实走进了目标分支，而不是数字凑巧对上）/ 统计自洽**。

## 证据边界（必须如实声明）
* 本模块用**位置策略 stub**替换 `target`/`draft` 的 logits ⇒ 证明的是**控制逻辑与 cache 账**，
  **不是**真实模型行为，**不是**模型质量或数值证据；
* stub 的 `tok_at(pos)` 只依赖**绝对位置**（与输入 token 值无关）⇒ 对"控制流/长度/cache"足够，
  对"数值传播"**不适用**；
* ⇒ **CPU EOS 通过 ≠ 真实模型 EOS 通过**：冻结判据 `docs/design_freeze_g4_specdec.md:83` 的 B6 要求
  真实模型上的 **exact token ids equality + 全 finite + 多次确定性一致**。

## 覆盖矩阵（`branch` = 被测**实现分支**；证据 = `spec_dec.py` 行号）
| id | 覆盖分支 | spec_dec 位置 |
| --- | --- | --- |
| `A_all_accept_eos_last` | 全接受轮、末候选 = EOS | `_propose` 155-156；244-250；260-261 |
| `B_all_accept_eos_kp3` | 同上、kp=3 | 同上 |
| `C_budget1_first_eos` | 预算 = 1 且首个候选 = EOS | 205 `k_eff=min(k,budget)`；260 |
| `D_midround_eos_rejected` | EOS 候选**被拒**（`m<kp`、替换 token ≠ EOS） | 235-243（§5 错配分支） |
| **`E_reject_replacement_is_eos`** | **拒绝后 target 替换 token = EOS** | 236 `r=g[m]`；237；260 |
| **`F_eos_at_budget_boundary`** | **EOS 恰落在预算边界**（`len(gen)==max_new` 且末 token=EOS） | 204；205；260-261 |
| **`G_budget_cut_before_eos`** | 预算在 EOS 之前截断 ⇒ 停止原因仍是 `max_new_reached`（对照） | 205 |
| **`H_rollback_then_eos_next_round`** | 回滚/repair 之后**下一轮**以 EOS 收尾 | 235-243；238-241；260 |
| **`I_carried_bonus_is_eos`** | 全接受轮留下的"额外 token"（`saved_next`）在下一轮以 EOS 兑现 | 248 `saved_next=rows[kp-1]` |
| **`J_first_candidate_eos_rejected`** | 首个候选 = EOS 但被拒（`m == 0`）⇒ EOS 候选 K/V 写入 draft cache 后按 §5 裁剪 | 155-156；238/240 |

**实现里没有独立的全接受"额外 token 路径"**：全接受轮的 `saved_next = rows[kp-1]`（248 行）只是**留到下一轮**
用的 logits 行，当轮**不**输出 ⇒ 需求 3 的等价路径由 `I_carried_bonus_is_eos` 覆盖（EOS 由 `saved_next` 在下一轮兑现）。

冻结判据相关条文：L27-30（提议遇 EOS 停提议；**提议结束时 draft cache 长度恒为 L+k'**，EOS 检查必须在推进之后）、
L42-44（EOS 被接受/输出即终止；**EOS 候选照常写入 draft cache**；若 `m<k'` 走 §5 错配分支）。

Run: python -m specdec.engine.g4_b5_eos_matrix_cpu
"""
from __future__ import annotations

import torch

from . import spec_dec
from .g4_b5_eos_branch_tests import EOS, StubModel, _propose_old, invariants

VOCAB = 32          # 位置策略里出现的 token id 必须 < VOCAB（首版用 8/9 越界 ⇒ 已修）
K = 4

# 每条 case：id / 分支说明 / prompt_len / draft_policy / target_policy / max_new /
#            期望 gen / 期望 stop / 期望轮数 / 见证断言 / 非空过对照
# 见证断言键：m, m_is_0, m_lt_kp, all_accept, cands_last_eos, emitted, emitted_last_eos, g0_eos
# 非空过对照：old_fails（旧 `_propose` 必抛）/ old_same（新旧一致）/ mut_no_terminate（需靠变异体证明非空过）
MATRIX = [
    dict(id="A_all_accept_eos_last", branch="全接受轮、末候选=EOS", P=2,
         draft={1: 5, 2: EOS}, target={1: 5, 2: EOS}, max_new=8,
         gen=[5, EOS], stop="eos", rounds=1,
         witness=[(0, dict(all_accept=True, cands_last_eos=True, emitted_last_eos=True))],
         control="old_fails"),
    dict(id="B_all_accept_eos_kp3", branch="全接受轮、kp=3、末候选=EOS", P=2,
         draft={1: 5, 2: 7, 3: EOS}, target={1: 5, 2: 7, 3: EOS}, max_new=8,
         gen=[5, 7, EOS], stop="eos", rounds=1,
         witness=[(0, dict(all_accept=True, cands_last_eos=True, emitted_last_eos=True))],
         control="old_fails"),
    dict(id="C_budget1_first_eos", branch="预算=1 且首个候选=EOS", P=2,
         draft={1: EOS}, target={1: EOS}, max_new=1,
         gen=[EOS], stop="eos", rounds=1,
         witness=[(0, dict(all_accept=True, cands_last_eos=True, emitted_last_eos=True))],
         control="old_fails"),
    dict(id="D_midround_eos_rejected", branch="EOS 候选被拒（替换 token 非 EOS）", P=2,
         draft={1: 5, 2: EOS}, target={1: 5, 2: 6}, max_new=4,
         gen=[5, 6, 0, 0], stop="max_new_reached", rounds=2,
         witness=[(0, dict(m=1, m_lt_kp=True, cands_last_eos=True))],
         control="old_same"),
    dict(id="E_reject_replacement_is_eos", branch="★拒绝后 target 替换 token=EOS", P=2,
         draft={1: 5, 2: 6}, target={1: 5, 2: EOS}, max_new=8,
         gen=[5, EOS], stop="eos", rounds=1,
         witness=[(0, dict(m=1, m_lt_kp=True, emitted_last_eos=True))],
         control="mut_no_terminate"),
    dict(id="F_eos_at_budget_boundary", branch="★EOS 恰落在预算边界", P=2,
         draft={1: 5, 2: 7, 3: EOS}, target={1: 5, 2: 7, 3: EOS}, max_new=3,
         gen=[5, 7, EOS], stop="eos", rounds=1,
         witness=[(0, dict(all_accept=True, emitted_last_eos=True))],
         control="old_fails"),
    dict(id="G_budget_cut_before_eos", branch="★预算在 EOS 之前截断（对照）", P=2,
         draft={1: 5, 2: 7, 3: EOS}, target={1: 5, 2: 7, 3: EOS}, max_new=2,
         gen=[5, 7], stop="max_new_reached", rounds=1,
         witness=[(0, dict(Kp=2, all_accept=True))],
         control="old_same"),
    dict(id="H_rollback_then_eos_next_round", branch="★回滚/repair 后下一轮以 EOS 收尾", P=2,
         draft={1: 5, 2: 6, 3: EOS}, target={1: 5, 2: 9, 3: EOS}, max_new=8,
         gen=[5, 9, EOS], stop="eos", rounds=2,
         witness=[(0, dict(m=1, m_lt_kp=True)), (1, dict(all_accept=True, emitted_last_eos=True))],
         control="old_fails", stats_min=dict(rollbacks=1, mismatch_rounds=1)),
    dict(id="I_carried_bonus_is_eos", branch="★全接受轮的额外 token 在下一轮以 EOS 兑现", P=2,
         draft={1: 5, 2: 7, 3: 8, 4: 0, 5: EOS}, target={1: 5, 2: 7, 3: 8, 4: 0, 5: EOS},
         max_new=6, gen=[5, 7, 8, 0, EOS], stop="eos", rounds=2,
         witness=[(0, dict(all_accept=True, Kp=4)), (1, dict(g0_eos=True, emitted=[EOS]))],
         control="old_fails"),
    dict(id="J_first_candidate_eos_rejected", branch="★首个候选=EOS 但被拒（m=0）", P=2,
         draft={1: EOS, 2: EOS}, target={1: 4, 2: 4}, max_new=4,
         gen=[4, 4, 0, 0], stop="max_new_reached", rounds=3,
         witness=[(0, dict(m=0, m_is_0=True, cands_last_eos=True, emitted=[4]))],
         control="old_same"),
]


def stop_reason_of(gen: list, max_new: int, eos_id: int) -> str:
    """**harness 的停止规则**（`g5_8k_smoke_cudnn.py:713-718` 逐字口径）：EOS 优先于预算。"""
    if gen and eos_id is not None and gen[-1] == eos_id:
        return "eos"
    if len(gen) >= max_new:
        return "max_new_reached"
    return "other_short"


def _witness_failures(case: dict, trace: list) -> list:
    """**分支见证**：证明真的走进了目标分支（而不是"数字碰巧对上"）。"""
    bad = []
    for idx, want in case.get("witness") or []:
        if idx >= len(trace):
            bad.append(f"{case['id']}: 缺第 {idx} 轮（只有 {len(trace)} 轮）")
            continue
        d = trace[idx]
        for k, v in want.items():
            if k == "all_accept":
                got = (d["m"] == d["Kp"])
            elif k == "m_is_0":
                got = (d["m"] == 0)
            elif k == "m_lt_kp":
                got = (d["m"] < d["Kp"])
            elif k == "cands_last_eos":
                got = bool(d["cands"]) and d["cands"][-1] == EOS
            elif k == "emitted_last_eos":
                got = bool(d["emitted"]) and d["emitted"][-1] == EOS
            elif k == "g0_eos":
                got = bool(d["g"]) and d["g"][0] == EOS
            else:
                got = (d.get(k) == v)
            if got != True:                                             # noqa: E712
                bad.append(f"{case['id']}: round{idx} 见证 {k} 不成立（实得 {d.get(k)!r}）")
    return bad


def check_case(case: dict, out, trace) -> list:
    """**结果断言**（不是"没有异常"）。返回失败原因列表（空 = 通过）。"""
    cid = case["id"]
    if out is None:
        return [f"{cid}: 未得到结果（异常）"]
    gen, st = out["gen"], out["stats"]
    n = case["max_new"]
    bad: list = []
    # ① 序列
    if gen != case["gen"]:
        bad.append(f"{cid}: gen={gen} ≠ 期望 {case['gen']}")
    # ② EOS 唯一性 / 位置 / 之后不再生成
    if case["stop"] == "eos":
        if gen.count(EOS) != 1:
            bad.append(f"{cid}: EOS 出现 {gen.count(EOS)} 次（必须恰好 1 次）")
        if not gen or gen[-1] != EOS:
            bad.append(f"{cid}: 末 token 不是 EOS（{gen}）")
    elif EOS in gen:
        bad.append(f"{cid}: 不该出现 EOS 却出现（{gen}）")
    # ③ 停止原因（EOS 优先于预算）+ gen_len ↔ EOS 位置 ↔ max_new 自洽
    sr = stop_reason_of(gen, n, EOS)
    if sr != case["stop"]:
        bad.append(f"{cid}: stop_reason={sr} ≠ 期望 {case['stop']}")
    if len(gen) > n:
        bad.append(f"{cid}: len(gen)={len(gen)} > max_new={n}")
    if sr == "eos" and len(gen) != gen.index(EOS) + 1:
        bad.append(f"{cid}: EOS 位置 {gen.index(EOS)} 与 gen_len={len(gen)} 不自洽")
    # ④ 轮次：EOS 之后不得再有轮次；候选/输出里 EOS 必须是最后一个
    if len(trace) != case["rounds"]:
        bad.append(f"{cid}: 轮数 {len(trace)} ≠ 期望 {case['rounds']}")
    for d in trace:
        if EOS in d["cands"] and d["cands"][-1] != EOS:
            bad.append(f"{cid}: round{d['round']} 候选在 EOS 之后还有 token {d['cands']}")
        if EOS in d["emitted"] and d["emitted"][-1] != EOS:
            bad.append(f"{cid}: round{d['round']} 输出在 EOS 之后还有 token {d['emitted']}")
    bad += [f"{cid}: {x}" for x in invariants(trace)]
    # ⑤ cache 提交合同（末轮点名 EOS 被写入）
    if trace:
        d = trace[-1]
        want = d["pre"] + (d["m"] + 1 if d["m"] < d["Kp"] else d["Kp"])
        if not (d["tgt_len"] == d["drf_len"] == want):
            bad.append(f"{cid}: 末轮 cache tgt={d['tgt_len']} drf={d['drf_len']} 期望 {want}")
        if case["stop"] == "eos" and d["emitted"] and d["emitted"][-1] == EOS \
                and d["tgt_len"] != want:
            bad.append(f"{cid}: EOS 未按合同提交（cache {d['tgt_len']} ≠ {want}）")
    # ⑥ 统计自洽 + 见证
    if st["draft_calls"] != st["proposed"]:
        bad.append(f"{cid}: draft_calls={st['draft_calls']} ≠ proposed={st['proposed']}")
    if st["rounds"] != len(trace):
        bad.append(f"{cid}: stats.rounds={st['rounds']} ≠ trace={len(trace)}")
    if st["rollbacks"] != st["mismatch_rounds"]:
        bad.append(f"{cid}: rollbacks={st['rollbacks']} ≠ mismatch_rounds={st['mismatch_rounds']}")
    if st["accepted"] > st["proposed"]:
        bad.append(f"{cid}: accepted={st['accepted']} > proposed={st['proposed']}")
    for k, vmin in (case.get("stats_min") or {}).items():
        if st.get(k, 0) < vmin:
            bad.append(f"{cid}: stats[{k}]={st.get(k)} < 期望下界 {vmin}")
    bad += _witness_failures(case, trace)
    return bad


# --------------------------------------------------------------------------- 变异体
def _loop_variant(target, draft, prompt_ids, max_new_tokens, *, k=K, eos_id=None, mutate=None):
    """`spec_decode_greedy` 的**逐行复制**，只允许 `mutate` 注入一处缺陷。

    **等价性自证**：`mutate=None`（默认返回原 `emitted`）时，必须在所有矩阵 case 上与真实实现
    给出完全相同的 `gen` / `stats` / 轮数（`main()` 断言）⇒ 变异体 = "已证等价的副本 + 单点改动"。
    """
    if spec_dec._VERIFY_WRAP is not None or spec_dec._VERIFY_MODE != "block":
        raise RuntimeError("变异体只在默认 verify 模式（block、无 wrap）下使用")
    mut = mutate or (lambda **kw: kw["emitted"])
    with torch.inference_mode():
        tgt_lg, tgt_cache = spec_dec._prefill(target, prompt_ids)
        drf_lg, drf_cache = spec_dec._prefill(draft, prompt_ids)
        saved_next = tgt_lg[:, -1:]
        drf_lg = drf_lg[:, -1:]
        gen: list = []
        st = dict(rounds=0, proposed=0, accepted=0, target_block_calls=0, draft_calls=0,
                  mismatch_rounds=0, all_accept_rounds=0, rollbacks=0)
        trace: list = []
        while len(gen) < max_new_tokens:
            k_eff = min(k, max_new_tokens - len(gen))
            pre = spec_dec.spec_cache.cache_length(tgt_cache)
            cands, drf_cache, prop_lg = spec_dec._propose(draft, drf_cache, drf_lg, k_eff, eos_id)
            kp = len(cands)
            st["draft_calls"] += kp
            rows, tgt_blk = spec_dec._verify(target, tgt_cache, cands)
            st["target_block_calls"] += 1
            st["proposed"] += kp
            g = [int(x.argmax()) for x in [saved_next] + [rows[j] for j in range(kp - 1)]]
            m = 0
            while m < kp and g[m] == cands[m]:
                m += 1
            if m < kp:
                r = g[m]
                emitted = cands[:m] + [r]
                tgt_cache = spec_dec.spec_cache.trim_cache(tgt_blk, pre + m)
                saved_next, tgt_cache = spec_dec._step(target, r, tgt_cache)
                drf_cache = spec_dec.spec_cache.trim_cache(drf_cache, pre + m)
                drf_lg, drf_cache = spec_dec._step(draft, r, drf_cache)
                st["mismatch_rounds"] += 1
                st["rollbacks"] += 1
            else:
                emitted = list(cands)
                tgt_cache = tgt_blk
                drf_cache = spec_dec.spec_cache.trim_cache(drf_cache, pre + kp)
                saved_next = rows[kp - 1]
                drf_lg = prop_lg
                st["all_accept_rounds"] += 1
            emitted = mut(emitted=list(emitted), cands=list(cands), gen=list(gen), eos_id=eos_id,
                          m=m, kp=kp)
            gen += emitted
            st["accepted"] += m
            st["rounds"] += 1
            trace.append(dict(round=st["rounds"] - 1, Kp=kp, m=m, pre=pre,
                              tgt_len=spec_dec.spec_cache.cache_length(tgt_cache),
                              drf_len=spec_dec.spec_cache.cache_length(drf_cache),
                              cands=cands, g=g, emitted=emitted))
            if emitted and emitted[-1] == eos_id and not getattr(mut, "no_break", False):
                break
        return {"ids": list(prompt_ids[0]) + gen, "gen": gen, "stats": st}, trace


def _cases_model(case: dict, *, vocab: int = VOCAB):
    """返回 **(target, draft)** —— ⚠️ 顺序必须与 `run_case` 的解包一致。

    首版写成 (draft, target) 而 `run_case` 按 (target, draft) 解包 ⇒ **两侧 policy 相同**的 case
    看不出问题，只有 draft≠target 的 4 条（D/E/H/J）暴露"draft 用 target 的 policy"。
    该缺陷**只在自测夹具里**，不涉及引擎；探针 `pip-tmp/eos_probe2.py` 逐 case 打印 policy 与
    `cands/g` 才定位到（对照 3 个数字对不上的现象是抓不出来的）。
    """
    return StubModel(case["target"], vocab=vocab), StubModel(case["draft"], vocab=vocab)


def run_case(case: dict, *, propose=None, vocab: int = VOCAB):
    """真实 `spec_dec.spec_decode_greedy` + 位置策略 stub ⇒ (out, err, trace)。"""
    tgt, drf = _cases_model(case, vocab=vocab)
    ids = torch.zeros(1, case["P"], dtype=torch.long)
    trace: list = []
    saved = spec_dec._propose
    spec_dec._propose = propose or saved
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, case["max_new"], k=K, eos_id=EOS,
                                          log=lambda d: trace.append(d))
        return out, None, trace
    except Exception as e:                                             # noqa: BLE001
        return None, e, trace
    finally:
        spec_dec._propose = saved


def main() -> int:
    npass = nfail = 0

    def ck(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    print("== 1. EOS 覆盖矩阵（真实 `spec_dec.spec_decode_greedy` + 位置策略 stub）==")
    results = {}
    for case in MATRIX:
        out, err, trace = run_case(case)
        if err is not None:
            ck(f"{case['id']} 无异常", False, f"{type(err).__name__}: {err}")
            continue
        bad = check_case(case, out, trace)
        results[case["id"]] = (out, trace)
        rounds_desc = " | ".join(
            f"r{d['round']}:pre={d['pre']},kp={d['Kp']},m={d['m']},tgt={d['tgt_len']},"
            f"drf={d['drf_len']}" for d in trace)
        print(f"  [{case['id']:32s}] gen={out['gen']} stop={stop_reason_of(out['gen'], case['max_new'], EOS)}"
              f"  {rounds_desc}")
        ck(f"{case['id']} 结果+见证断言全过", not bad, bad[:3])

    print("\n== 2. 非空过保证（每条 case 都必须有能证明它非空的对照）==")
    for case in MATRIX:
        cid, ctl = case["id"], case["control"]
        out_new, err_new, _t = run_case(case)
        if ctl == "old_fails":
            _o, err_old, _to = run_case(case, propose=_propose_old)
            ck(f"{cid} 旧 `_propose` 在该路径上失败（ValueError）", isinstance(err_old, ValueError),
               f"old -> {type(err_old).__name__}: {err_old}" if err_old else "old -> 无异常")
        elif ctl == "old_same":
            out_old, err_old, _to = run_case(case, propose=_propose_old)
            ck(f"{cid} 非 EOS 收尾路径上新旧一致",
               err_old is None and out_old is not None and out_old["gen"] == out_new["gen"],
               f"old gen={None if out_old is None else out_old['gen']}")
        elif ctl == "mut_no_terminate":
            tgt, drf = _cases_model(case)

            def _no_break(**kw):
                return kw["emitted"]
            _no_break.no_break = True
            out_m, tr_m = _loop_variant(tgt, drf, torch.zeros(1, case["P"], dtype=torch.long),
                                        case["max_new"], k=K, eos_id=EOS, mutate=_no_break)
            bad = check_case(case, out_m, tr_m)
            ck(f"{cid} 关闭 EOS 终止后**被同一个检查器抓出**（非空过）", bool(bad), bad[:2])
            ck(f"{cid} 抓出原因指向停止/末 token/EOS 次数",
               any(("stop_reason" in b) or ("末 token" in b) or ("EOS 出现" in b) or ("轮数" in b)
                   for b in bad), bad[:2])
        else:
            ck(f"{cid} 声明了非空过对照", False, f"未知对照 {ctl!r}")

    print("\n== 3. 变异体（=已证等价的副本 + 单点缺陷）必须被同一个检查器抓出 ==")
    equiv_bad = []
    for case in MATRIX:
        tgt, drf = _cases_model(case)
        out_v, tr_v = _loop_variant(tgt, drf, torch.zeros(1, case["P"], dtype=torch.long),
                                    case["max_new"], k=K, eos_id=EOS)
        out_r, err_r, tr_r = run_case(case)
        if err_r is not None or out_v["gen"] != out_r["gen"] or out_v["stats"] != out_r["stats"] \
                or len(tr_v) != len(tr_r):
            equiv_bad.append(case["id"])
    ck("副本（mutate 默认）与真实实现逐位等价（gen/stats/轮数全同）", not equiv_bad, equiv_bad)

    caseA = MATRIX[0]
    # 缺陷①：EOS 后不终止
    tgt, drf = _cases_model(caseA)

    def _no_break2(**kw):
        return kw["emitted"]
    _no_break2.no_break = True
    out_m, tr_m = _loop_variant(tgt, drf, torch.zeros(1, caseA["P"], dtype=torch.long),
                                caseA["max_new"], k=K, eos_id=EOS, mutate=_no_break2)
    bad_m = check_case(caseA, out_m, tr_m)
    ck("缺陷①EOS 后不终止 ⇒ 被抓出", bool(bad_m), bad_m[:2])

    # 缺陷②：EOS 之后多带一个 token
    tgtA, drfA = _cases_model(caseA)

    def _emit_extra(**kw):
        return list(kw["emitted"]) + ([kw["emitted"][-1]] if kw["eos_id"] in kw["emitted"] else [])
    out_m2, tr_m2 = _loop_variant(tgtA, drfA,
                                  torch.zeros(1, caseA["P"], dtype=torch.long),
                                  caseA["max_new"], k=K, eos_id=EOS, mutate=_emit_extra)
    bad_m2 = check_case(caseA, out_m2, tr_m2)
    ck("缺陷②EOS 之后多带 token ⇒ 被抓出", bool(bad_m2), bad_m2[:2])

    # 缺陷③：全接受轮把 cache 裁短一个 token（EOS 候选未按合同提交）
    orig_trim = spec_dec.spec_cache.trim_cache

    def _short_trim(cache, new_len):
        return orig_trim(cache, max(1, new_len - 1))
    spec_dec.spec_cache.trim_cache = _short_trim
    try:
        out_m3, err_m3, tr_m3 = run_case(caseA)
    finally:
        spec_dec.spec_cache.trim_cache = orig_trim
    bad_m3 = check_case(caseA, out_m3, tr_m3) if err_m3 is None else [f"异常：{err_m3}"]
    ck("缺陷③cache 被裁短（EOS 未按合同提交）⇒ 被抓出", bool(bad_m3), bad_m3[:2])

    # 缺陷④：候选里在 EOS 之后塞 token（模拟"停提议"失效）
    tgtA2, drfA2 = _cases_model(caseA)

    def _cands_after_eos(**kw):
        return kw["emitted"]
    out_m4, tr_m4 = _loop_variant(tgtA2, drfA2,
                                  torch.zeros(1, caseA["P"], dtype=torch.long),
                                  caseA["max_new"], k=K, eos_id=EOS, mutate=_cands_after_eos)
    tr_rot = [dict(tr_m4[0], cands=[5, EOS, 6])]
    ck("缺陷④候选在 EOS 之后还有 token ⇒ 被抓出（检查器自测）",
       bool(check_case(caseA, out_m4, tr_rot)), None)

    # 检查器自测：手工喂入错误停止 / 错误 cache
    good_out, good_tr = results[caseA["id"]]
    rot = dict(good_out, gen=list(good_out["gen"]) + [0])
    ck("自测：EOS 之后补 token ⇒ 检查器报错", bool(check_case(caseA, rot, good_tr)))
    rot2 = dict(good_out, gen=list(good_out["gen"])[:-1])
    ck("自测：丢掉 EOS ⇒ 检查器报错", bool(check_case(caseA, rot2, good_tr)))
    rot3 = [dict(good_tr[0], drf_len=int(good_tr[0]["drf_len"]) - 1)]
    ck("自测：cache 长度不符 ⇒ 检查器报错", bool(check_case(caseA, good_out, rot3)))

    print(f"\nCPU EOS 矩阵：PASS {npass} / FAIL {nfail}"
          f"（case={len(MATRIX)}；stub fixture ⇒ **非真实模型证据**）")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
