"""fixtures：从真实 bonus oracle 轨迹读取 token，并构建真实 prefill cache 快照。"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BONUS_RUN = ROOT / "output" / "bonus_accept_oracle" / "run"


def load_tokens(family: str, ctx: int) -> dict:
    path = BONUS_RUN / f"{family}_{ctx}_bonus" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    control = result["control"]
    r0, r1 = control[0], control[1]
    bonus0 = r0["emitted_bonus"]
    cand0 = r0["emitted_candidates"]
    cand1 = r1["emitted_candidates"]
    if bonus0 is None:
        raise RuntimeError("fixture 期望 round0 有 bonus")
    return dict(
        family=family, context=ctx,
        input_sha1=result["input"]["token_sha1"],
        a1=[int(cand0[0])],
        a2=[int(t) for t in cand0],
        a3=[int(bonus0)] + [int(t) for t in cand1],
        b1=[int(t) for t in cand0],
        b2=[int(bonus0)] + [int(t) for t in cand1],
        bonus0=int(bonus0),
    )


def prefill_caches(target, draft, ids):
    """用与正式口径相同的分块 prefill 生成 (target_cache, draft_cache) 快照。"""
    from experiments.formal.worker import test_chunked_prefill
    prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
    tgt_lg, tgt_cache = test_chunked_prefill(target, prompt, chunk=2048, diagnostic=False)
    drf_lg, drf_cache = test_chunked_prefill(draft, prompt, chunk=2048, diagnostic=False)
    return prompt, (tgt_lg, tgt_cache), (drf_lg, drf_cache)
