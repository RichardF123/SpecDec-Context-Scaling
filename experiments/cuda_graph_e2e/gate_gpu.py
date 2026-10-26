"""端到端 Graph 与 eager bonus oracle 的逐项对齐门（路线 A）。"""
from __future__ import annotations

import torch

from . import design
from . import capture_rounds as CR


def _cmp_tensor(a, b):
    same_shape = tuple(a.shape) == tuple(b.shape)
    same_dtype = a.dtype == b.dtype
    bitwise = same_shape and same_dtype and torch.equal(a, b)
    md = None
    if same_shape:
        md = float((a.float() - b.float()).abs().max().item())
    return dict(shape_ok=same_shape, dtype_ok=same_dtype, bitwise=bitwise, max_abs_diff=md)


def run_gate(graphs, io, state_d, state_t, ref: dict) -> dict:
    CR.replay_pipeline(graphs, io, None)
    torch.cuda.synchronize()
    gen1 = io.gen.clone()
    tok_equal = torch.equal(gen1, torch.tensor([ref["gen"]], dtype=torch.long,
                                               device=torch.device("cuda")))
    # 逐轮 candidate / bonus 与参考 control 对齐（从 gen 切片反查）
    per_round = []
    ok_rounds = True
    for r in range(design.ROUNDS):
        start = design.GEN_CAND_START[r]
        cands = [int(x) for x in gen1[0, start:start + 4].tolist()]
        ctrl = ref["control"][r]
        ref_cands = [int(x) for x in ctrl["emitted_candidates"]]
        per_round.append(dict(round=r, cands_match=cands == ref_cands,
                              cands=cands, ref_cands=ref_cands))
        ok_rounds = ok_rounds and cands == ref_cands
        if r < design.ROUNDS - 1:
            bpos = design.BONUS_POS[r]
            b = int(gen1[0, bpos].item())
            ok_rounds = ok_rounds and b == int(ctrl["emitted_bonus"])
    # cache 数值对齐（末轮 big buffer vs eager 最终 cache）
    cache_rows = []
    cache_ok = True
    ref_cache = ref["final_cache"]
    for i in range(len(ref_cache)):
        a_k = state_t.k[i][:, :, :design.MAX_CAPACITY]
        a_v = state_t.v[i][:, :, :design.MAX_CAPACITY]
        row_k = _cmp_tensor(a_k, ref_cache[i]["k"])
        row_v = _cmp_tensor(a_v, ref_cache[i]["v"])
        cache_rows.append(dict(layer=i, k=row_k, v=row_v))
        cache_ok = cache_ok and row_k["bitwise"] and row_v["bitwise"]
    # 确定性：再 replay 一次
    CR.replay_pipeline(graphs, io, None)
    torch.cuda.synchronize()
    gen2 = io.gen.clone()
    deterministic = torch.equal(gen1, gen2)

    # verify 输入长度与 round 语义
    lens = [design.VERIFY_S[r] for r in range(design.ROUNDS)]
    finite = all(bool(torch.isfinite(state_t.k[i]).all()) and bool(torch.isfinite(state_t.v[i]).all())
                 for i in range(len(state_t.k)))
    ok = (tok_equal and ok_rounds and cache_ok and deterministic and finite
          and lens[0] == 4 and all(x == 5 for x in lens[1:]))
    return dict(ok=bool(ok), tokens_equal=bool(tok_equal), rounds=design.ROUNDS,
                per_round=per_round, cache_bitwise=bool(cache_ok),
                cache_rows=cache_rows, deterministic=bool(deterministic),
                finite=bool(finite), verify_lens=lens,
                max_cache_diff=max([max(r["k"]["max_abs_diff"] or 0.0,
                                        r["v"]["max_abs_diff"] or 0.0) for r in cache_rows]))
