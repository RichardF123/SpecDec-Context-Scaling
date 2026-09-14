"""2K-only whole-model diagnostic for the two-pass FlashAttention kernel.

This wraps the existing runtime-only engine trial.  It intentionally limits
the run to 2K and 16 decode tokens; a full 2K/4K/8K benchmark is justified only
if the prefill-logit error collapses here.
"""
from __future__ import annotations

import os

from . import flash_attention_engine_trial as trial
from .flash_attention_two_pass_triton import (
    KERNEL_REVISION,
    flash_attention_bf16_two_pass,
)


def main() -> int:
    trial.flash_attention_bf16 = flash_attention_bf16_two_pass
    trial.KERNEL_REVISION = KERNEL_REVISION
    trial.CONTEXTS = (2048,)
    trial.DECODE_TOKENS = 16
    trial.ORDERS = (
        ("eager_bf16", "flash_bf16"),
        ("flash_bf16", "eager_bf16"),
    )
    trial.OUTPUT = os.path.join(
        os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
        "flash_attention_two_pass_engine_diagnostic.json",
    )
    print(
        "TWO-PASS 2K DIAGNOSTIC ONLY: full benchmark remains blocked until "
        "this numerical gate passes.",
        flush=True,
    )
    return trial.main()


if __name__ == "__main__":
    raise SystemExit(main())
