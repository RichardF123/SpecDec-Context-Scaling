"""Correctness gate for official FlashAttention-2.

Tests both:
- flash_attn_func (pure contiguous full attention)
- flash_attn_with_kvcache (preallocated contiguous incremental cache)
against the repository contiguous CUDA attention reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from specdec.kernels import ops


SHAPES = {
    "draft": dict(H=14, KV=2, D=64),
    "target": dict(H=12, KV=2, D=128),
}


def _err(a, b):
    af, bf = a.float(), b.float()
    diff = af - bf
    return (
        float(diff.abs().max().item()),
        float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(bf).item(), 1e-12)
        ),
    )


def main():
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(1301)

    rows = []
    ok = True
    for name, sh in SHAPES.items():
        H, KV, D = sh["H"], sh["KV"], sh["D"]
        for T in (16, 17, 63, 128, 512):
            for S in (1, 2, 3, 4):
                if S > T:
                    continue
                old = T - S

                q = torch.randn(
                    1, H, S, D, device="cuda",
                    dtype=torch.bfloat16)
                k_old = torch.randn(
                    1, KV, old, D, device="cuda",
                    dtype=torch.bfloat16)
                v_old = torch.randn_like(k_old)
                k_new = torch.randn(
                    1, KV, S, D, device="cuda",
                    dtype=torch.bfloat16)
                v_new = torch.randn_like(k_new)

                k_all = torch.cat([k_old, k_new], dim=2)
                v_all = torch.cat([v_old, v_new], dim=2)

                ref = ops.attention(q, k_all, v_all, backend="cuda")
                ref_fa = ref.transpose(1, 2).contiguous()

                qf = q.transpose(1, 2).contiguous()
                kf = k_all.transpose(1, 2).contiguous()
                vf = v_all.transpose(1, 2).contiguous()

                full = flash_attn_func(
                    qf, kf, vf,
                    dropout_p=0.0,
                    softmax_scale=D ** -0.5,
                    causal=True)

                # Preallocated contiguous cache. Add a little tail capacity so
                # the test proves logical length != physical capacity.
                cap = T + 11
                kc = torch.empty(
                    1, cap, KV, D,
                    device="cuda", dtype=torch.bfloat16)
                vc = torch.empty_like(kc)
                if old:
                    kc[:, :old].copy_(
                        k_old.transpose(1, 2).contiguous())
                    vc[:, :old].copy_(
                        v_old.transpose(1, 2).contiguous())

                lens = torch.tensor(
                    [old], device="cuda", dtype=torch.int32)
                cached = flash_attn_with_kvcache(
                    qf, kc, vc,
                    k=k_new.transpose(1, 2).contiguous(),
                    v=v_new.transpose(1, 2).contiguous(),
                    cache_seqlens=lens,
                    softmax_scale=D ** -0.5,
                    causal=True)

                full_abs, full_rel = _err(full, ref_fa)
                cache_abs, cache_rel = _err(cached, ref_fa)
                append_ok = (
                    torch.equal(
                        kc[:, old:T],
                        k_new.transpose(1, 2))
                    and torch.equal(
                        vc[:, old:T],
                        v_new.transpose(1, 2))
                )
                passed = (
                    full_abs <= 0.03
                    and full_rel <= 0.01
                    and cache_abs <= 0.03
                    and cache_rel <= 0.01
                    and append_ok
                )
                row = dict(
                    shape=name, H=H, KV=KV, D=D,
                    T=T, q_len=S,
                    full_max_abs=full_abs,
                    full_rel_l2=full_rel,
                    kvcache_max_abs=cache_abs,
                    kvcache_rel_l2=cache_rel,
                    append_exact=append_ok,
                    pass_=passed,
                )
                rows.append(row)
                ok &= passed
                print(
                    f"[FA2 gate] {name} T={T} q={S} "
                    f"full_abs={full_abs:.6g} "
                    f"cache_abs={cache_abs:.6g} "
                    f"append={int(append_ok)} "
                    f"{'PASS' if passed else 'FAIL'}",
                    flush=True)

    out = Path("output/pure_flash2")
    out.mkdir(parents=True, exist_ok=True)
    (out/"operator_gate.json").write_text(
        json.dumps(
            {"pass": ok, "cases": rows},
            indent=2),
        encoding="utf-8")

    print("PURE_FLASH2_OPERATOR_GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
