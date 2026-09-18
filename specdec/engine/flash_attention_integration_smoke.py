"""Smoke test for the production opt-in FlashAttention selector.

This is intentionally smaller than the frozen validation suite.  It verifies
that ``QwenLM.set_attention_backend`` reaches the same two-pass kernel already
validated by the core, whole-model, and PPL experiments.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time

import torch

from .flash_attention_two_pass_triton import KERNEL_REVISION
from .kv_policy_pilot_v2 import MODEL, load_model


CONTEXT = 2048
CHUNK = 512
DECODE_TOKENS = 16
OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
    "flash_attention_integration_smoke.json",
)


def make_ids(tokenizer) -> torch.Tensor:
    text = (
        "FlashAttention evaluates query, key, and value blocks with an online "
        "softmax while avoiding a materialized quadratic attention matrix."
    )
    base = tokenizer(text, add_special_tokens=False).input_ids
    repeats = math.ceil(CONTEXT / len(base))
    return torch.tensor(
        [(base * repeats)[:CONTEXT]],
        dtype=torch.long,
        device="cuda",
    )


def run(model, ids: torch.Tensor, backend: str) -> dict:
    model.set_attention_backend(backend)
    model.set_kv_head_policy(2, 2)
    model.set_kv_format("fp16")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    cache = None
    logits = None
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, CONTEXT, CHUNK):
            piece = ids[:, start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece),
                cache,
                need_logits=start + CHUNK >= CONTEXT,
            )
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - started
    prefill_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    last_logits = logits[:, -1].to(torch.float32).cpu()

    generated = []
    finite = bool(torch.isfinite(logits).all().item())
    with torch.inference_mode():
        for _ in range(DECODE_TOKENS):
            next_id = int(logits[:, -1].argmax())
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=ids.device)
            logits, cache = model._run(
                model.embed_tokens(next_tensor),
                cache,
                need_logits=True,
            )
            finite &= bool(torch.isfinite(logits).all().item())
    return {
        "backend": backend,
        "prefill_s": prefill_s,
        "prefill_peak_allocated_gib": prefill_peak_gib,
        "tokens": generated,
        "finite": finite,
        "_last_logits": last_logits,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = load_model()
    ids = make_ids(tokenizer)
    eager = run(model, ids, "eager")
    flash = run(model, ids, "triton_bf16_two_pass")

    difference = flash["_last_logits"] - eager["_last_logits"]
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(eager["_last_logits"]).clamp_min(1e-12)
    ).item()
    argmax_same = bool(
        eager["_last_logits"].argmax().item()
        == flash["_last_logits"].argmax().item()
    )
    tokens_identical = eager["tokens"] == flash["tokens"]
    passed = eager["finite"] and flash["finite"] and argmax_same and tokens_identical
    result = {
        "status": "production_selector_smoke",
        "kernel_revision": KERNEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "context": CONTEXT,
        "chunk": CHUNK,
        "decode_tokens": DECODE_TOKENS,
        "eager": {key: value for key, value in eager.items() if not key.startswith("_")},
        "flash": {key: value for key, value in flash.items() if not key.startswith("_")},
        "last_logit_relative_l2": relative_l2,
        "last_logit_argmax_same": argmax_same,
        "tokens_identical": tokens_identical,
        "decision": "PASS_INTEGRATION_SMOKE" if passed else "FAIL_INTEGRATION_SMOKE",
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(
        "production-selector smoke: "
        f"revision={KERNEL_REVISION} gpu={result['gpu']} ctx={CONTEXT} "
        f"chunk={CHUNK} decode={DECODE_TOKENS}"
    )
    print(
        f"eager prefill={eager['prefill_s']:.3f}s "
        f"peak={eager['prefill_peak_allocated_gib']:.3f}GiB"
    )
    print(
        f"flash prefill={flash['prefill_s']:.3f}s "
        f"peak={flash['prefill_peak_allocated_gib']:.3f}GiB"
    )
    print(
        f"last_logit_rel_l2={relative_l2:.7f} argmax_same={argmax_same} "
        f"tokens_identical={tokens_identical}"
    )
    print(f"decision={result['decision']}\nJSON={OUTPUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
