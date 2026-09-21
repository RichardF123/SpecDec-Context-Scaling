"""g4_specdec_index_tests.py — SpecDec **logits 索引语义**的确定性哨兵测试（CPU，无模型）。

为什么要有它：`spec_decode_greedy` 的候选判定在语义上必须满足
    candidate i==0 用 prefix_next_logits
    candidate i>=1 用 block_logits[i-1]
    block_logits[kp-1] 只用于"全部接受后"的下一轮 prefix（本设计不产出 bonus）
错位一格（用 block_logits[i] 判 candidate i）在真实模型上是"偶发且难分辨"的错误，
但在**可控哨兵**上会立刻暴露：让 target 每个位置的 argmax 等于该位置的下标，
draft 提议"正确判定行的下标"，于是
    correct 映射 → 全部接受
    buggy   映射（block_logits[i]）→ 第 0 个候选就被拒
T1 用同一份捕获数据同时演示这两种映射，证明测试**有区分力**（不是空过）。

覆盖（对应任务 §四 1–4）
  T1 哨兵：判定行必须是 [pre-1, pre, pre+1, pre+2]（= prefix + rows[0..2]），且 all-accept
  T2 mismatch 分别出现在 i=0,1,2,K-1：accepted=m、replacement 来自对应判定行、emitted 正确、
     cache 已提交长度 = pre+m+1
  T3 全部接受：emitted == K 个候选（不产出 bonus），且被携带的下一轮 prefix 行 = block 最后一行
  T4 EOS 与尾块：candidate 是 EOS / replacement 是 EOS / 预算 < K 时不越界
用法：python -m specdec.engine.g4_specdec_index_tests
"""
from __future__ import annotations

import traceback

import torch

from . import spec_dec

VOCAB = 8192
EOS = 3
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class TargetStub:
    """argmax(位置 p) = p（哨兵）；可选把某个位置的 argmax 换成别的 token 以制造 mismatch。"""

    def __init__(self, bad_pos: int | None = None, bad_tok: int = 900,
                 bad_pos2: int | None = None, bad_tok2: int = 901):
        self.bad_pos, self.bad_tok = bad_pos, bad_tok
        self.bad_pos2, self.bad_tok2 = bad_pos2, bad_tok2

    def set_kv_format(self, fmt):
        pass

    def embed_tokens(self, ids):
        return ids.float()

    def _run(self, x, cache_list=None, need_logits=True):
        start = 0 if (cache_list is None or cache_list[0] is None) \
            else int(cache_list[0]["k"].shape[2])
        s = int(x.shape[1])
        lg = torch.full((1, s, VOCAB), -1e4)
        for j in range(s):
            pos = start + j
            tok = pos
            if self.bad_pos is not None and pos == self.bad_pos:
                tok = self.bad_tok
            if self.bad_pos2 is not None and pos == self.bad_pos2:
                tok = self.bad_tok2
            lg[0, j, int(tok)] = 10.0
        cache = [{"k": torch.zeros(1, 1, start + s, 2), "v": torch.zeros(1, 1, start + s, 2)}]
        return (lg if need_logits else None), cache


class DraftStub:
    """提议"正确判定行的下标"：argmax = (该次 _run 之后的 cache 长度 - 1)。
    这样 candidate i 恰好等于 target 在位置 pre+i-1 的 argmax ⇒ 正确映射下全部接受。
    eos_at_call 让第 n 次 _run 提议 EOS。"""

    def __init__(self, eos_at_call: int | None = None):
        self.calls = 0
        self.eos_at_call = eos_at_call

    def set_kv_format(self, fmt):
        pass

    def embed_tokens(self, ids):
        return ids.float()

    def _run(self, x, cache_list=None, need_logits=True):
        start = 0 if (cache_list is None or cache_list[0] is None) \
            else int(cache_list[0]["k"].shape[2])
        s = int(x.shape[1])
        self.calls += 1
        tok = EOS if self.calls == self.eos_at_call else (start + s - 1)
        lg = torch.full((1, s, VOCAB), -1e4)
        lg[0, -1, int(tok)] = 10.0
        cache = [{"k": torch.zeros(1, 1, start + s, 2), "v": torch.zeros(1, 1, start + s, 2)}]
        return (lg if need_logits else None), cache


def run_case(prompt_len=6, k=4, maxnew=16, bad_pos=None, bad_tok=900,
             bad_pos2=None, bad_tok2=901, eos_at_call=None):
    """跑一次 spec_decode_greedy（哨兵 stub），并捕获每轮实际使用的判定行。"""
    tgt = TargetStub(bad_pos, bad_tok, bad_pos2, bad_tok2)
    drf = DraftStub(eos_at_call)
    ids = torch.zeros(1, prompt_len, dtype=torch.long)
    events: list[dict] = []
    orig_pre, orig_ver, orig_step = spec_dec._prefill, spec_dec._verify, spec_dec._step

    def prefill(model, ids_):
        lg, c = orig_pre(model, ids_)
        if model is tgt:
            events.append(dict(kind="pre", lg=lg[:, -1:]))
        return lg, c

    def verify(target, cache, cands):
        rows, blk = orig_ver(target, cache, cands)
        events.append(dict(kind="ver", cands=list(cands), rows=rows, kp=len(cands)))
        return rows, blk

    def step(model, token, cache):
        lg, c = orig_step(model, token, cache)
        if model is tgt:
            events.append(dict(kind="step", tok=int(token), lg=lg[:, -1:]))
        return lg, c

    spec_dec._prefill, spec_dec._verify, spec_dec._step = prefill, verify, step
    trace: list[dict] = []
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, maxnew, k=k, eos_id=EOS,
                                          target_kv="base_bf16",
                                          log=lambda d: trace.append(d))
    finally:
        spec_dec._prefill, spec_dec._verify, spec_dec._step = orig_pre, orig_ver, orig_step
    return out, trace, events


def judges_from_events(events):
    """按实现的实际映射重建每轮的判定行。

    关键：**全接受路径不调用 `_step`**，下一轮的 `saved_next` 直接取本轮的
    `rows[kp-1]`（block 最后一行）；错配路径才由随后的 `_step` 覆盖。重建必须与之一致。
    """
    rounds, last = [], None
    for e in events:
        if e["kind"] in ("pre", "step"):
            last = e["lg"][0, -1]
        elif e["kind"] == "ver":
            rounds.append(dict(saved=last, rows=e["rows"], cands=e["cands"], kp=e["kp"]))
            last = e["rows"][e["kp"] - 1]      # 全接受：下一轮 prefix 行 = 本块最后一行
    return rounds


def main() -> int:
    S = 6

    print("== T1 哨兵：判定行必须是 [pre-1, pre, pre+1, pre+2]，且 all-accept ==")
    out, trace, events = run_case(prompt_len=S, maxnew=8)
    r0 = judges_from_events(events)[0]
    used = [int(r0["saved"].argmax())] + [int(r0["rows"][j].argmax()) for j in range(r0["kp"] - 1)]
    expect = [S - 1 + i for i in range(r0["kp"])]
    check("判定行 = prefix_next + block_logits[0..kp-2]", used == expect, f"{used} vs {expect}")
    check("draft 候选 = 各判定行 argmax", r0["cands"] == expect, f"{r0['cands']}")
    check("T1 all-accept: m == kp", trace[0]["m"] == r0["kp"], f"m={trace[0]['m']}")
    check("T1 emitted == cands（K 个）", trace[0]["emitted"] == r0["cands"],
          f"{trace[0]['emitted']}")
    buggy = [int(r0["rows"][i].argmax()) for i in range(r0["kp"])]
    check("T1 有区分力：buggy 映射（block_logits[i]）判不出全部接受",
          buggy[:r0["kp"] - 1] != r0["cands"][:r0["kp"] - 1],
          f"buggy={buggy} cands={r0['cands']}")
    check("T1 buggy 的 g[0] 与候选 0 不等（会立刻 mismatch）",
          buggy[0] != r0["cands"][0], f"buggy[0]={buggy[0]} cand0={r0['cands'][0]}")

    print("\n== T2 mismatch 分别出现在 i=0,1,2,K-1 ==")
    for i in range(4):
        bad_pos = S + i - 1
        out, trace, events = run_case(prompt_len=S, maxnew=8, bad_pos=bad_pos, bad_tok=900)
        t0 = trace[0]
        r0 = judges_from_events(events)[0]
        judge = r0["saved"] if i == 0 else r0["rows"][i - 1]
        check(f"T2 i={i}: m=={i}", t0["m"] == i, f"m={t0['m']}")
        check(f"T2 i={i}: replacement=900 且来自该候选的判定行",
              t0["g"][i] == 900 and int(judge.argmax()) == 900, f"g={t0['g']}")
        check(f"T2 i={i}: emitted = cands[:m]+[900]",
              t0["emitted"] == t0["cands"][:i] + [900], f"{t0['emitted']}")
        want = t0["pre"] + i + 1
        check(f"T2 i={i}: cache 已提交长度 = pre+m+1 = {want}",
              t0["tgt_len"] == want and t0["drf_len"] == want,
              f"tgt={t0['tgt_len']} drf={t0['drf_len']}")

    print("\n== T3 全部接受：K 个候选、不产出 bonus、下一轮 prefix 行 = block 最后一行 ==")
    out, trace, events = run_case(prompt_len=S, maxnew=8)
    rounds = judges_from_events(events)
    t0 = trace[0]
    check("T3 emitted 长度 == Kp（无 bonus）", len(t0["emitted"]) == t0["Kp"],
          f"{len(t0['emitted'])} vs Kp={t0['Kp']}")
    check("T3 第二轮 prefix 行 == 第一轮 block 最后一行",
          int(rounds[1]["saved"].argmax())
          == int(rounds[0]["rows"][rounds[0]["kp"] - 1].argmax()))
    check("T3 第二轮候选 0 由该行判定",
          rounds[1]["cands"][0] == int(rounds[1]["saved"].argmax()))
    check("T3 target block calls == rounds",
          out["stats"]["target_block_calls"] == out["stats"]["rounds"])
    check("T3 恒等式 gen_len == accepted + mismatch_rounds",
          len(out["gen"]) == out["stats"]["accepted"] + out["stats"]["mismatch_rounds"],
          f"{len(out['gen'])} vs {out['stats']['accepted']}+{out['stats']['mismatch_rounds']}")

    print("\n== T4 EOS 与尾块 ==")
    out, trace, events = run_case(prompt_len=S, maxnew=8, eos_at_call=1)
    check("T4 draft 提议 EOS → Kp==1（遇 EOS 立即停提议）",
          trace[0]["Kp"] == 1 and trace[0]["cands"] == [EOS],
          f"Kp={trace[0]['Kp']} cands={trace[0]['cands']}")
    check("T4 该 EOS 未被 target 接受 → 走 replacement，生成中不写入 EOS",
          out["gen"][0] != EOS and trace[0]["m"] == 0,
          f"gen[:2]={out['gen'][:2]} m={trace[0]['m']}")
    out, trace, events = run_case(prompt_len=S, maxnew=8, bad_pos=S, bad_tok=EOS,
                                  bad_pos2=S - 1, bad_tok2=EOS)
    check("T4 candidate0=EOS 且 target 也认为 EOS → 接受并终止 gen=[EOS]",
          out["gen"] == [EOS], f"{out['gen']}")
    want = trace[0]["pre"] + (trace[0]["m"] + 1 if trace[0]["m"] < trace[0]["Kp"]
                              else trace[0]["Kp"])
    check("T4 EOS 轮 cache 长度 = pre+(m+1 或 Kp)",
          trace[0]["tgt_len"] == want and trace[0]["drf_len"] == want,
          f"tgt={trace[0]['tgt_len']} drf={trace[0]['drf_len']} want={want} "
          f"(m={trace[0]['m']} Kp={trace[0]['Kp']})")
    out, trace, events = run_case(prompt_len=S, maxnew=8, bad_pos=S + 1, bad_tok=EOS)
    check("T4 replacement = EOS → 生成即终止（不越过预算）",
          out["gen"][-1] == EOS and len(out["gen"]) <= 3, f"{out['gen']}")
    out, trace, events = run_case(prompt_len=S, maxnew=2)
    check("T4 预算<K：k_eff=2 且不越界",
          len(out["gen"]) == 2 and all(t["Kp"] <= 2 for t in trace),
          f"gen={out['gen']} Kps={[t['Kp'] for t in trace]}")
    out, trace, events = run_case(prompt_len=S, maxnew=5)
    check("T4 非 K 倍数预算（5）：总长恰为 5", len(out["gen"]) == 5, f"{len(out['gen'])}")
    out, trace, events = run_case(prompt_len=S, maxnew=8, bad_pos=S + 3, bad_tok=EOS)
    check("T4 全接受轮不产出 bonus（即使 block 最后一行为 EOS）",
          len(trace[0]["emitted"]) == trace[0]["Kp"] and EOS not in trace[0]["emitted"],
          f"emitted={trace[0]['emitted']}")

    print(f"\nSpecDec index 测试：PASS {len(PASS)} / FAIL {len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
