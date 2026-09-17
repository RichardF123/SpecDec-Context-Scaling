"""Runtime-only compatibility gate for an explicit small-block hybrid.

Evidence from ``flash_attention_g4_operator_isolation.py`` shows that the
custom q_len=4 Triton attention is the source of the tested serial/block
decision split, while this explicit route passed the bounded screen:

* prompt prefill (q_len > 4): custom two-pass Triton;
* cached AR decode (q_len == 1): custom two-pass Triton;
* cached SpecDec verification (2 <= q_len <= 4): repository eager expression.

This is not a silent production fallback and is not an official FlashAttention
implementation.  It is a runtime-only candidate used to decide whether a
correct integration boundary exists before any selector or production code is
changed.  The unchanged same-backend G4 diagnostic supplies cache, rollback,
determinism, and output evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import flash_attention_g4_specdec_compat as compat
from . import flash_attention_two_pass_triton as flash_impl
from .flash_attention_g4_core_probe import eager_reference


POLICY = {
    "name": "custom_prefill_q1_eager_q2_to_q4",
    "prefill": "custom_triton_two_pass",
    "q_len_1": "custom_triton_two_pass",
    "q_len_2_to_4": "repository_eager_expression",
    "official_flash_used": False,
    "production_selector_modified": False,
}


def main() -> int:
    original = flash_impl.flash_attention_bf16_two_pass

    def explicit_hybrid(query, key, value, *, causal=True, config=None):
        query_length = int(query.shape[2])
        if 2 <= query_length <= compat.K:
            return eager_reference(query, key, value, causal=causal)
        return original(
            query, key, value, causal=causal, config=config,
        )

    flash_impl.flash_attention_bf16_two_pass = explicit_hybrid
    try:
        return_code = compat.main()
    finally:
        flash_impl.flash_attention_bf16_two_pass = original

    output_dir = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    ))
    source = output_dir / "flash_attention_g4_specdec_compat.json"
    if not source.exists():
        raise RuntimeError(f"compatibility JSON missing: {source}")
    result = json.loads(source.read_text(encoding="utf-8"))
    result["status"] = "latest_main_flash_g4_explicit_hybrid_compatibility"
    result["runtime_policy"] = POLICY
    result["underlying_compatibility_decision"] = result["decision"]
    result["decision"] = (
        "PASS_FLASH_G4_EXPLICIT_HYBRID_COMPATIBILITY"
        if return_code == 0
        else "CHECK_FLASH_G4_EXPLICIT_HYBRID_COMPATIBILITY"
    )
    output = output_dir / "flash_attention_g4_hybrid_compat.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"hybrid_decision={result['decision']}\n"
        f"hybrid_JSON={output}",
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
