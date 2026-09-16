"""g4_b5_eos_branch_tests.py — EOS 分支的**纯逻辑**回归测试（无需模型/GPU/网络, CPU 毫秒级）

背景（修前）: `_propose` 把 `if t == eos_id: break` 放在推进 cache **之前**，于是当 draft
提议的最后一个候选是 EOS 时，draft cache 长度只有 `pre+kp-1`。若 target 把该轮全部确认
(`m == kp`，即这轮真的以 EOS 结束)，全接受分支的 `trim_cache(drf_cache, pre+kp)` 就越界
→ 按 `spec_cache` 契约抛 `ValueError`。触发条件是"生成正常以 EOS 收尾"的常规路径。

修法: EOS 检查挪到推进之后 ⇒ `_propose` 出口恒定 `len(cache) == pre + kp`（唯一不变量）。
副产物: `st["draft_calls"] += kp` 与实际 draft 前向次数**精确相等**（此前 EOS 轮会少算 1）。

被测对象: **控制流与 cache 长度账**，不是数值 ⇒ 用"位置策略 stub 模型"即可精确复现。
非空过保证: 同一组 case 在**旧实现**下必须抛 ValueError，否则说明本测试没覆盖到该分支。
Run: python -m specdec.engine.g4_b5_eos_branch_tests
"""
from __future__ import annotations

import traceback

import torch

from . import spec_dec

EOS = 3


class StubModel:
    """位置策略 stub: 位置 p 处的 greedy 输出 = policy.get(p, default)。
    只实现 spec_dec 真正用到的接口: set_kv_format / embed_tokens / _run。"""

    def __init__(self, policy: dict[int, int], default: int = 0, vocab: int = 8,
                 layers: int = 2, dim: int = 2):
        self.policy = dict(policy)
        self.default = default
        self.vocab = vocab
        self.layers = layers
        self.dim = dim
        self.kv_format = None

    def set_kv_format(self, fmt):
        self.kv_format = fmt

    def tok_at(self, pos: int) -> int:
        return int(self.policy.get(pos, self.default))

    def embed_tokens(self, ids):
        return ids.to(torch.float32)

    def _run(self, x, cache_list=None, need_logits=True):
        s = int(x.shape[1])
        start = 0 if (cache_list is None or cache_list[0] is None) else \
            int(cache_list[0]["k"].shape[2])
        lg = None
        if need_logits:
            lg = torch.zeros(1, s, self.vocab)
            for j in range(s):
                lg[0, j, self.tok_at(start + j)] = 10.0
        cache = [{"k": torch.zeros(1, 1, start + s, self.dim),
                  "v": torch.zeros(1, 1, start + s, self.dim)}
                 for _ in range(self.layers)]
        return lg, cache


def _propose_old(draft, cache, lg, k_eff, eos_id):
    """修前实现（仅用于对照: 必须在 EOS 全接受 case 上抛 ValueError）。"""
    cands = []
    with torch.inference_mode():
        for _ in range(k_eff):
            t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            cands.append(t)
            if t == eos_id:
                break
            lg, cache = draft._run(
                draft.embed_tokens(torch.tensor([[t]], device=lg.device)),
                cache, need_logits=True)
    return cands, cache, lg


# (name, prompt_len, draft_policy, draft_default, target_policy, target_default,
#  max_new_tokens, expect_gen, old_expect)
CASES = [
    ("eos_last_of_kp_all_accept", 2, {1: 5, 2: EOS}, 0, {1: 5, 2: EOS}, 0,
     8, [5, EOS], "raises"),
    ("eos_first_budget1", 2, {1: EOS}, 0, {1: EOS}, 0,
     1, [EOS], "raises"),
    ("eos_third_of_four_all_accept", 2, {1: 5, 2: 7, 3: EOS}, 0,
     {1: 5, 2: 7, 3: EOS}, 0, 8, [5, 7, EOS], "raises"),
    ("eos_midround_then_mismatch", 2, {1: 5, 2: EOS}, 0, {1: 5, 2: 6}, 0,
     4, [5, 6, 0, 0], "same"),
]


def run_case(case, propose=None):
    (name, s_len, dpol, ddef, tpol, tdef, n, expect_gen, _old) = case
    tgt = StubModel(tpol, tdef)
    drf = StubModel(dpol, ddef)
    ids = torch.zeros(1, s_len, dtype=torch.long)
    trace = []
    saved = spec_dec._propose
    spec_dec._propose = propose or saved
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, n, k=4, eos_id=EOS,
                                         log=lambda d: trace.append(d))
        return out, None, trace
    except Exception as e:  # noqa: BLE001
        return None, e, trace
    finally:
        spec_dec._propose = saved


def invariants(trace):
    bad = []
    for d in trace:
        want = d["pre"] + (d["m"] + 1 if d["m"] < d["Kp"] else d["Kp"])
        if not (d["tgt_len"] == d["drf_len"] == want):
            bad.append(f"round{d['round']}: tgt={d['tgt_len']} drf={d['drf_len']} "
                       f"want={want} (m={d['m']}/{d['Kp']})")
    return bad


def main() -> int:
    ok = True
    print("== 修后实现 ==", flush=True)
    for case in CASES:
        name, expect_gen, old_expect = case[0], case[7], case[8]
        out, err, trace = run_case(case)
        if err is not None:
            print(f"  [{name:30s}] ERROR {type(err).__name__}: {err}")
            ok = False
            continue
        gen = out["gen"]
        st = out["stats"]
        bad = invariants(trace)
        good = (gen == expect_gen) and not bad
        ok &= good
        r0 = trace[0] if trace else {}
        print(f"  [{name:30s}] gen={gen} (expect {expect_gen}) "
              f"rounds={st['rounds']} Kp={r0.get('Kp')} m={r0.get('m')} "
              f"acc={st['accepted']}/{st['proposed']} "
              f"allacc={st['all_accept_rounds']} mis={st['mismatch_rounds']} "
              f"draft_calls={st['draft_calls']} invariants={'OK' if not bad else bad}")
        if st["draft_calls"] != st["proposed"]:
            print(f"      NOTE: draft_calls({st['draft_calls']}) != proposed"
                  f"({st['proposed']}) -> EOS 轮记账不一致")
            ok = False
    print("\n== 旧实现对照（非空过保证）==", flush=True)
    for case in CASES:
        name, old_expect = case[0], case[8]
        out_new, _e, _t = run_case(case)
        out_old, err_old, trace_old = run_case(case, propose=_propose_old)
        if old_expect == "raises":
            hit = err_old is not None and isinstance(err_old, ValueError)
            ok &= hit
            print(f"  [{name:30s}] old -> "
                  f"{'ValueError: ' + str(err_old) if err_old else 'NO ERROR'}"
                  f"  {'[OK] 已复现' if hit else '[FAIL] 未覆盖到该分支(vacuity!)'}")
        else:
            same = (err_old is None and out_old is not None and
                    out_new is not None and out_old["gen"] == out_new["gen"])
            ok &= same
            print(f"  [{name:30s}] old -> "
                  f"{'ValueError: ' + str(err_old) if err_old else 'gen=' + str(out_old['gen'])}"
                  f"  {'[OK] 两版一致(非崩溃路径)' if same else '[FAIL] 行为不一致'}")

    print(f"\nB5-EOS-branch: {'PASS' if ok else 'CHECK'} "
          f"(cases={len(CASES)}, CPU stub, 无模型无 GPU)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
