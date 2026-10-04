from __future__ import annotations

import json
from pathlib import Path

import torch

from . import paged_cache
from .model_self import quantize_kv_fp8
from specdec.kernels import ops


def _case(*, fmt: str, H: int, KV: int, D: int, T: int, S: int) -> dict:
    torch.manual_seed(1000 + T * 10 + S + D)

    q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, KV, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    layer = paged_cache.create_layer_cache(fmt, 16)
    if fmt == "fp16":
        paged_cache.append_current(layer, k, v)
    else:
        qv, scale = quantize_kv_fp8(v, 128)
        paged_cache.append_current(layer, k, qv, scale)

    ref = ops.paged_attention_block(q, layer, backend="cuda", validate=True)
    got = ops.phase13_flash_paged_attention(q, layer, validate=True)

    diff = got.float() - ref.float()
    max_abs = float(diff.abs().max().item())
    rel_l2 = float(
        torch.linalg.vector_norm(diff).item()
        / max(torch.linalg.vector_norm(ref.float()).item(), 1e-12)
    )
    exact = bool(torch.equal(got, ref))
    ok = exact or (max_abs <= 7.8125e-3 and rel_l2 <= 1e-3)

    return {
        "fmt": fmt, "H": H, "KV": KV, "D": D, "T": T, "S": S,
        "exact": exact, "max_abs": max_abs, "rel_l2": rel_l2, "pass": ok,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    rows = []
    for shape in (
        dict(name="draft", H=14, KV=2, D=64),
        dict(name="target", H=12, KV=2, D=128),
    ):
        for T in (16, 17, 63, 128, 512):
            for S in (1, 2, 3, 4):
                r = _case(
                    fmt="fp16", H=shape["H"], KV=shape["KV"],
                    D=shape["D"], T=T, S=S,
                )
                r["shape"] = shape["name"]
                rows.append(r)
                print("[BF16]", r, flush=True)

    for T in (16, 17, 63, 128, 512):
        for S in (1, 2, 3, 4):
            r = _case(fmt="v_fp8", H=12, KV=2, D=128, T=T, S=S)
            r["shape"] = "target"
            rows.append(r)
            print("[V-FP8]", r, flush=True)

    ok = all(r["pass"] for r in rows)
    out = Path("output/phase13_flash")
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase13_operator_gate.json").write_text(
        json.dumps({"pass": ok, "cases": rows}, indent=2), encoding="utf-8"
    )
    print("\nPHASE13_FLASH_OPERATOR_GATE:", "PASS" if ok else "CHECK")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
