"""g4_b5_smoke.py — B5: greedy SpecDec 主循环「结构不变量」自检（不判正确性）

前置：用户实现的 `specdec/engine/spec_dec.py` 暴露
    spec_decode_greedy(target, draft, prompt_ids, max_new_tokens, *, k=4, eos_id,
                       target_kv, draft_kv, log) -> {"ids","gen","stats"}
本脚本只查（正确性由 g4_b6_correctness.py 判）:
  [1] 每轮 cache 长度回滚正确: m<Kp -> tgt=pre+m+1, drf=pre+m;m==Kp -> tgt=pre+Kp, drf=pre+Kp
  [2] stats 字段齐全且自洽(proposed>=accepted, rounds>=all_accept+mismatch, ...)
  [3] 确定性: 同输入两次 gen 完全相同
  [4] 两档 target_kv 都能跑(base_bf16 / v_fp8)
Run（交互式 GPU 终端, 单跑; 双模型 8GB 已验证可行）: python -m specdec.engine.g4_b5_smoke
"""
from __future__ import annotations

import gc
import traceback

import torch

from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
K = 4
PROMPT = "The history of artificial intelligence and neural networks spans many decades."


def check_rounds(trace) -> bool:
    ok = True
    for d in trace:
        pre, m, Kp = d["pre"], d["m"], d["Kp"]
        exp_tgt = pre + (m + 1 if m < Kp else Kp)
        exp_drf = pre + m
        good = (d["tgt_len"] == exp_tgt and d["drf_len"] == exp_drf)
        ok &= good
        print(f"  round{d['round']:2d}: Kp={Kp} m={m} pre={pre} "
              f"tgt {d['tgt_len']}(exp {exp_tgt}) drf {d['drf_len']}(exp {exp_drf}) "
              f"{'ok' if good else 'MISMATCH'}")
    return ok


def main() -> int:
    from .spec_dec import spec_decode_greedy
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    print("[stage] loading target 1.5B ...", flush=True)
    tgt, tok = load_ours(TARGET)
    print("[stage] loading draft 0.5B ...", flush=True)
    drf, _ = load_ours(DRAFT)
    eos = tok.eos_token_id
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    ok = True
    for kv in ("base_bf16", "v_fp8"):
        print(f"\n== target_kv={kv} ==", flush=True)
        trace = []
        out = spec_decode_greedy(tgt, drf, ids, 16, k=K, eos_id=eos,
                                 target_kv=kv, log=lambda d: trace.append(d))
        ok &= check_rounds(trace)
        st = out["stats"]
        need = ("rounds", "proposed", "accepted", "target_block_calls", "draft_calls",
                "mismatch_rounds", "all_accept_rounds", "rollbacks")
        have = all(k2 in st for k2 in need)
        sane = (st.get("proposed", 0) >= st.get("accepted", 0)
                and st.get("rounds", 0) >= st.get("mismatch_rounds", 0) + st.get("all_accept_rounds", 0))
        print(f"  stats: {st}")
        print(f"  fields-complete={have} sane={sane} gen_len={len(out['gen'])}")
        ok &= have and sane

        out2 = spec_decode_greedy(tgt, drf, ids, 16, k=K, eos_id=eos, target_kv=kv)
        det = out["gen"] == out2["gen"]
        print(f"  determinism: {det}")
        ok &= det

    del tgt, drf
    gc.collect()
    torch.cuda.empty_cache()
    print("\nB5 (structure):", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
