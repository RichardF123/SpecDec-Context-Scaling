"""Full isolated 2K/4K/8K operational trial for two-pass FlashAttention.

This reuses the balanced-order whole-model harness while selecting the
two-pass globally normalized Triton kernel at runtime.  Production
``model_self.py`` remains unchanged.  The existing strict eager-equivalence
gate is intentionally retained; quality evaluation is a separate later gate.
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
    trial.CONTEXTS = (2048, 4096, 8192)
    trial.CHUNK = 2048
    trial.DECODE_TOKENS = 64
    trial.ORDERS = (
        ("eager_bf16", "flash_bf16"),
        ("flash_bf16", "eager_bf16"),
        ("flash_bf16", "eager_bf16"),
        ("eager_bf16", "flash_bf16"),
    )
    trial.OUTPUT = os.path.join(
        os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/flash_attention"),
        "flash_attention_two_pass_full_trial.json",
    )
    print(
        "ISOLATED TWO-PASS FULL TRIAL: production model_self.py is unchanged; "
        "strict eager-equivalence gate remains active.",
        flush=True,
    )
    return trial.main()


if __name__ == "__main__":
    raise SystemExit(main())
