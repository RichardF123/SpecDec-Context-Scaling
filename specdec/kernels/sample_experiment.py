"""One operator case: formula check, a short timing row, and a pack install.

No model weights. From the SpecDec repository root:

    python -m specdec.kernels.sample_experiment

From the experiment checkout:

    python -m kernels.sample_experiment
"""

from __future__ import annotations

import statistics
import sys

from . import ops, patch
from . import reference as ref


FP32_ATOL = 1e-4
BF16_ATOL = 2e-2


def _cap_8gb() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    cap = 8 * (1024 ** 3)
    fraction = min(1.0, cap / props.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        f"gpu={props.name} total_mib={props.total_memory / (1024 ** 2):.0f} "
        f"allocator_fraction={fraction:.3f}",
        flush=True,
    )


def _max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def _ok(err: float, limit: float, scale: float) -> bool:
    rel = err / max(scale, 1e-6)
    return err == err and (err <= limit or rel <= 0.02)


def _check(name: str, err: float, limit: float, scale: float, rows: list) -> None:
    passed = _ok(err, limit, scale)
    rows.append(passed)
    rel = err / max(scale, 1e-6)
    print(
        f"  {'OK' if passed else 'FAIL'} {name}: max_abs={err:.3e} rel={rel:.3e} limit={limit:.3e}",
        flush=True,
    )


def _backends() -> list[str]:
    names = ["cuda"]
    if ops.triton_available():
        names.append("triton")
    else:
        print("triton: not installed, skipping those rows", flush=True)
    return names


def _formulas(rows: list) -> None:
    import torch

    print("== formulas ==", flush=True)
    x = torch.randn(4, 128, 1536, device="cuda", dtype=torch.float32)
    weight = torch.randn(1536, device="cuda", dtype=torch.float32)
    rms_ref = ref.rmsnorm(x, weight)

    rope_x = torch.randn(1, 32, 12, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    seq, dim = rope_x.shape[-2], rope_x.shape[-1]
    cos = torch.randn(1, 1, seq, dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    rope_ref = ref.apply_rope(rope_x, cos, sin)

    gate = torch.randn(2, 32, 8960, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    swi_ref = ref.swiglu(gate, up)

    q32 = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float32)
    k32 = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float32)
    v32 = torch.randn_like(k32)
    attn32_ref = ref.attention(q32, k32, v32)

    q = torch.randn(1, 12, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    attn_ref = ref.attention(q, k, v)

    for backend in _backends():
        got = ops.rmsnorm(x, weight, backend=backend)
        _check(f"rmsnorm/fp32/h1536/{backend}", _max_abs(got, rms_ref), FP32_ATOL, float(rms_ref.abs().max()), rows)

        got = ops.apply_rope(rope_x, cos, sin, backend=backend)
        _check(f"rope/bf16/h128/{backend}", _max_abs(got, rope_ref), BF16_ATOL, float(rope_ref.float().abs().max()), rows)

        got = ops.swiglu(gate, up, backend=backend)
        _check(f"swiglu/bf16/i8960/{backend}", _max_abs(got, swi_ref), BF16_ATOL, float(swi_ref.float().abs().max()), rows)

        got = ops.attention(q32, k32, v32, backend=backend)
        _check(f"attn/fp32/s64/{backend}", _max_abs(got, attn32_ref), FP32_ATOL, float(attn32_ref.abs().max()), rows)

        got = ops.attention(q, k, v, backend=backend)
        _check(f"attn/bf16/decode/{backend}", _max_abs(got, attn_ref), BF16_ATOL, float(attn_ref.float().abs().max()), rows)


def _time_decode() -> None:
    import torch

    print("== time decode attention S=1 T=128 ==", flush=True)
    q = torch.randn(1, 12, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for backend in ("torch", "cuda"):
        def run():
            return ops.attention(q, k, v, backend=backend)

        for _ in range(5):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del out
        samples.sort()
        print(f"  {backend}: median_ms={statistics.median(samples):.4f} min_ms={samples[0]:.4f} max_ms={samples[-1]:.4f}", flush=True)


def _pack() -> None:
    print("== pack ==", flush=True)
    name = patch.install("all")
    active = patch.current_pack()
    patch.restore()
    restored = patch.current_pack()
    print(f"  install={name} active={active} restored={restored}", flush=True)
    if active != "all" or restored != "torch":
        raise SystemExit("pack install/restore failed")


def main() -> int:
    _cap_8gb()
    rows: list[bool] = []
    _formulas(rows)
    _time_decode()
    _pack()
    if not rows or not all(rows):
        print("SAMPLE FAIL", flush=True)
        return 1
    print("SAMPLE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
