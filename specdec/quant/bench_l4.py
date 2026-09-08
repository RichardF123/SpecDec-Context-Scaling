"""bench_l4.py — A6 L4 memory / decode evidence on the 1.5B target.

Measures one weight mode per process:
  - fp16 baseline
  - RTN int8
  - packed group-wise RTN int4

Reports:
  - actual model storage bytes (parameters + persistent buffers)
  - peak CUDA memory during prefill + decode
  - decode tokens/s and ms/token

This is the system-cost half of the A6 gate. Quality (ppl / held-out set)
is evaluated separately.

Run on the RTX 4060 8GB machine:
    python -m specdec.quant.bench_l4 --mode fp16
    python -m specdec.quant.bench_l4 --mode int8
    python -m specdec.quant.bench_l4 --mode int4
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from datetime import datetime, timezone

import torch

from specdec.engine.align_ours import build_from_config
from specdec.engine.bench_basic import PROMPT
from .int8 import quantize_model_int8
from .int4 import quantize_model_int4


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_DECODE = 64
WARMUP_DECODE = 4
REPS = 3


def storage_bytes(model: torch.nn.Module) -> int:
    """Actual persistent model storage: parameters + registered buffers."""
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )
    return param_bytes + buffer_bytes


def build_model(model_id: str, mode: str, group_size: int):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Reference weights stay on CPU while the OURS model is constructed.
    hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf.eval()

    try:
        hf_q = hf.model.layers[0].self_attn.q_proj
        attn_bias = getattr(hf_q, "bias", None) is not None
    except Exception:
        attn_bias = False

    ours = build_from_config(
        cfg,
        attn_bias_force=attn_bias,
    )

    sd = {
        k.removeprefix("model."): v
        for k, v in hf.state_dict().items()
    }

    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)

    res = ours.load_state_dict(sd, strict=False)

    print("  missing:", res.missing_keys)
    print("  unexpected:", res.unexpected_keys)

    if res.missing_keys:
        raise RuntimeError("OURS engine is missing model weights")

    # HF is only the weight source. Do not keep a second model resident.
    del sd
    del hf
    gc.collect()

    replaced = 0

    if mode == "int8":
        replaced = quantize_model_int8(ours)

    elif mode == "int4":
        replaced = quantize_model_int4(
            ours,
            group_size=group_size,
        )

    expected = len(ours.layers) * 7

    if mode != "fp16":
        print(f"  replaced: {replaced} (expected {expected})")
        if replaced != expected:
            raise RuntimeError(
                f"quantized {replaced} modules, expected {expected}"
            )

    # Compute remains fp16 for all three modes.
    # Integer qweight / packed_weight buffers remain integer under Module.to().
    ours = ours.to(
        device="cuda",
        dtype=torch.float16,
    ).eval()

    return ours


def decode_once(model, ids, n_tokens: int) -> tuple[float, int]:
    """Prefill first, then time decode only."""
    with torch.inference_mode():
        logits, caches = model._run(
            model.embed_tokens(ids),
            None,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_tokens):
            next_id = logits[:, -1].argmax(dim=-1)
            logits, caches = model._run(
                model.embed_tokens(next_id.unsqueeze(1)),
                caches,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated()
    return elapsed, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fp16", "int8", "int4"),
        required=True,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] A6 formal benchmark requires the RTX 4060 CUDA machine")
        return 1

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    from transformers import AutoTokenizer

    print("== L4 A6 system evidence ==")
    print("date_utc:", datetime.now(timezone.utc).isoformat())
    print("model:", args.model)
    print("mode:", args.mode)
    print("seed: 0")
    print("torch:", torch.__version__)
    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "protocol:",
        f"batch=1, decode={MAX_DECODE}, warmup={WARMUP_DECODE}, reps={REPS}",
    )

    print("[1/3] loading OURS engine + weights")
    model = build_model(
        args.model,
        args.mode,
        args.group_size,
    )

    model_bytes = storage_bytes(model)

    print(
        f"model storage: "
        f"{model_bytes / 2**20:.1f} MiB "
        f"({model_bytes / 2**30:.3f} GiB)"
    )

    print("[2/3] preparing fixed prompt")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    ids = tok(
        PROMPT,
        return_tensors="pt",
    ).input_ids.to("cuda")

    print(f"prompt tokens: {ids.shape[1]}")

    print("[3/3] warmup + decode measurement")

    # Short warmup; its allocations are not part of the measured peak.
    with torch.inference_mode():
        logits, caches = model._run(
            model.embed_tokens(ids),
            None,
        )

        for _ in range(WARMUP_DECODE):
            next_id = logits[:, -1].argmax(dim=-1)
            logits, caches = model._run(
                model.embed_tokens(next_id.unsqueeze(1)),
                caches,
            )

    del logits
    del caches
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    times = []
    peaks = []

    for rep in range(REPS):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        elapsed, peak = decode_once(
            model,
            ids,
            MAX_DECODE,
        )

        times.append(elapsed)
        peaks.append(peak)

        print(
            f"  rep {rep + 1}: "
            f"{MAX_DECODE / elapsed:.2f} tok/s, "
            f"{elapsed / MAX_DECODE * 1000:.2f} ms/token, "
            f"peak={peak / 2**30:.3f} GiB"
        )

    best = min(times)
    median = statistics.median(times)
    peak = max(peaks)

    print()
    print("== SUMMARY ==")
    print(f"mode: {args.mode}")
    print(
        f"model storage: "
        f"{model_bytes / 2**20:.1f} MiB "
        f"({model_bytes / 2**30:.3f} GiB)"
    )
    print(
        f"decode best: "
        f"{MAX_DECODE / best:.2f} tok/s "
        f"({best / MAX_DECODE * 1000:.2f} ms/token)"
    )
    print(
        f"decode median: "
        f"{MAX_DECODE / median:.2f} tok/s "
        f"({median / MAX_DECODE * 1000:.2f} ms/token)"
    )
    print(f"peak VRAM: {peak / 2**30:.3f} GiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
