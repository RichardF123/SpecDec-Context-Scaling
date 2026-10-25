"""execution_optimization.config — 显式开关与冻结项声明。

用法：
    SPECDEC_EXECUTION_MODE=reference（默认） / optimized
    SPECDEC_OPT_PROPOSAL_GPU=0|1            默认 1（仅 optimized 模式生效）
    SPECDEC_OPT_VECTOR_ACCEPT=0|1           默认 1
    SPECDEC_OPT_GPU_TOKENS=0|1              默认 1（repair 用 GPU token，去掉 H2D）
"""
from __future__ import annotations

import os

MODES = ("reference", "optimized")

# 冻结配置（审计用途；执行器不得覆盖这些值）
FROZEN = dict(
    target="Qwen/Qwen2.5-1.5B-Instruct",
    draft="Qwen/Qwen2.5-0.5B-Instruct",
    compute_dtype="bfloat16",
    target_kv="base_bf16",
    draft_kv="base_bf16",
    attention="cudnn_prefill(prefill only)",
    k=4,
    prefill_chunk=2048,
    max_new=64,
    decode="greedy argmax, EOS-aware",
    contexts=(2048, 4096, 8192),
    dev_families=("F09", "F03", "F11", "F04"),
)


def execution_mode() -> str:
    mode = os.environ.get("SPECDEC_EXECUTION_MODE", "reference").strip().lower()
    if mode not in MODES:
        raise ValueError(f"SPECDEC_EXECUTION_MODE 非法: {mode!r}")
    return mode


def _flag(name: str, default: int = 1) -> bool:
    return os.environ.get(name, str(default)).strip() not in ("0", "false", "False", "no")


def optimization_flags() -> dict:
    """返回优化执行器的开关组合（只在 optimized 模式下调用有意义）。"""
    return dict(
        proposal_gpu=_flag("SPECDEC_OPT_PROPOSAL_GPU"),
        vector_accept=_flag("SPECDEC_OPT_VECTOR_ACCEPT"),
        gpu_tokens=_flag("SPECDEC_OPT_GPU_TOKENS"),
    )


def describe() -> dict:
    return dict(execution_mode=execution_mode(), flags=optimization_flags(), frozen=FROZEN)
