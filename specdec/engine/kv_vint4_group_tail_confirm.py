"""Fresh confirmation for the single screened V-INT4 group-64 candidate.

The candidate was selected on WikiText-2 windows 36..39.  This script compares
BF16, current V-FP8, V-INT4 group 32, and V-INT4 group 64 on the remaining
previously unread complete windows 120..145.  It does not modify production
``model_self.py`` and does not run an 8K operational benchmark.

Because group 64 saves only 3.5 MiB of logical 8K KV versus group 32, promotion
uses a strict pre-registered non-inferiority rule.  The one-sided 95% paired
upper confidence bound for extra dNLL versus group 32 must not exceed 0.002,
and aggregate relative PPL versus current V-FP8 must not exceed 1.0%.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

import torch

from . import model_self as ms
from .fp8_kv_v4_dppl import ensure_data
from .kv_policy_pilot_v2 import MODEL, WINDOW, load_model
from .kv_vint4_group_tail_screen import (
    EXPECTED_BYTES_PER_TOKEN_LAYER,
    Policy,
    quantizer_sanity,
    window_stats,
)


WINDOW_OFFSET = 120
N_WINDOWS = 26
NONINFERIORITY_DNLL_MARGIN = 0.002
MAX_REL_DPPL_VS_CURRENT_PCT = 1.0
# t(25), one-sided 95%.  Hard-coded to avoid adding a scipy dependency.
T_ONE_SIDED_95_DF25 = 1.708

POLICIES = (
    Policy("base_bf16", "fp16"),
    Policy("current_vfp8", "v_fp8"),
    Policy("vint4_g32", "v_int4", 32),
    Policy("vint4_g64", "v_int4", 64),
)

OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "18_vint4_group_tail_confirm.json",
)


def paired_se(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    original_group_size = ms.GROUP_V_INT4
    try:
        sanity = [
            row for row in quantizer_sanity()
            if row["group_size"] in (32, 64)
        ]
        if not all(row["passed"] for row in sanity):
            print("confirmation_decision=STOP_QUANTIZER_SANITY_FAILED")
            return 1

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        tokenizer.model_max_length = 10**9
        with open(ensure_data(), encoding="utf-8") as stream:
            all_ids = torch.tensor(
                tokenizer(stream.read())["input_ids"], dtype=torch.long
            )
        needed = (WINDOW_OFFSET + N_WINDOWS) * WINDOW
        if all_ids.numel() < needed:
            raise RuntimeError(
                f"dataset has {all_ids.numel()} tokens, need {needed}"
            )

        print(
            f"V-INT4 g64 fresh confirmation: model=1.5B "
            f"windows={WINDOW_OFFSET}..{WINDOW_OFFSET + N_WINDOWS - 1} "
            f"n={N_WINDOWS} W={WINDOW}",
            flush=True,
        )
        print(
            "g64 was selected on windows 36..39; this run uses only fresh "
            "windows 120..145\n",
            flush=True,
        )
        model = load_model()
        totals = {
            policy.name: {"nll_sum": 0.0, "tokens": 0, "finite": True}
            for policy in POLICIES
        }
        logical = {}
        per_window = []
        started = time.perf_counter()

        for index in range(N_WINDOWS):
            window_offset = WINDOW_OFFSET + index
            ids_w = all_ids[
                window_offset * WINDOW:(window_offset + 1) * WINDOW
            ].cuda()
            row = {"window_offset": window_offset}
            for policy in POLICIES:
                nll_sum, tokens, finite, cache_bytes = window_stats(
                    model, ids_w, policy
                )
                nll = nll_sum / tokens
                row[policy.name] = {
                    "nll": nll,
                    "ppl": math.exp(nll),
                    "finite": finite,
                }
                totals[policy.name]["nll_sum"] += nll_sum
                totals[policy.name]["tokens"] += tokens
                totals[policy.name]["finite"] &= finite
                logical[policy.name] = (
                    cache_bytes / (WINDOW * len(model.layers))
                )
                torch.cuda.empty_cache()

            base_nll = row["base_bf16"]["nll"]
            row["current_dNLL"] = row["current_vfp8"]["nll"] - base_nll
            row["g32_dNLL"] = row["vint4_g32"]["nll"] - base_nll
            row["g64_dNLL"] = row["vint4_g64"]["nll"] - base_nll
            row["g64_minus_g32_dNLL"] = (
                row["vint4_g64"]["nll"] - row["vint4_g32"]["nll"]
            )
            per_window.append(row)
            print(
                f"window={window_offset:03d} base={base_nll:.6f} "
                f"current_dNLL={row['current_dNLL']:+.6f} "
                f"g32_dNLL={row['g32_dNLL']:+.6f} "
                f"g64_dNLL={row['g64_dNLL']:+.6f} "
                f"g64-g32={row['g64_minus_g32_dNLL']:+.6f}",
                flush=True,
            )
            del ids_w

        aggregate = {}
        for policy in POLICIES:
            name = policy.name
            mean_nll = totals[name]["nll_sum"] / totals[name]["tokens"]
            aggregate[name] = {
                "nll": mean_nll,
                "ppl": math.exp(mean_nll),
                "finite": totals[name]["finite"],
                "tokens": totals[name]["tokens"],
                "bytes_per_token_layer": logical[name],
                "cache_mib_at_8k_28_layers": (
                    logical[name] * 8192 * 28 / 2**20
                ),
            }
            expected = EXPECTED_BYTES_PER_TOKEN_LAYER[name]
            if logical[name] != expected:
                raise RuntimeError(
                    f"logical-byte mismatch for {name}: "
                    f"{logical[name]} != {expected}"
                )

        base = aggregate["base_bf16"]
        current = aggregate["current_vfp8"]
        for record in aggregate.values():
            record["dNLL_vs_base"] = record["nll"] - base["nll"]
            record["relative_dPPL_vs_base_pct"] = (
                record["ppl"] / base["ppl"] - 1.0
            ) * 100.0
            record["relative_dPPL_vs_current_pct"] = (
                record["ppl"] / current["ppl"] - 1.0
            ) * 100.0

        paired = {}
        for name, key in (
            ("current_vs_base", "current_dNLL"),
            ("g32_vs_base", "g32_dNLL"),
            ("g64_vs_base", "g64_dNLL"),
            ("g64_vs_g32", "g64_minus_g32_dNLL"),
        ):
            values = [row[key] for row in per_window]
            mean = statistics.mean(values)
            se = paired_se(values)
            paired[name] = {
                "mean_dNLL": mean,
                "se": se,
                "one_sided_95_upper": mean + T_ONE_SIDED_95_DF25 * se,
            }

        g64 = aggregate["vint4_g64"]
        g64_vs_g32 = paired["g64_vs_g32"]
        all_finite = all(
            totals[policy.name]["finite"] for policy in POLICIES
        )
        passes_quality = bool(
            all_finite
            and g64["relative_dPPL_vs_current_pct"]
            <= MAX_REL_DPPL_VS_CURRENT_PCT
            and g64_vs_g32["one_sided_95_upper"]
            <= NONINFERIORITY_DNLL_MARGIN
        )
        decision = (
            "CONFIRM_G64_AND_RUN_ONE_OPERATIONAL_TEST"
            if passes_quality
            else "STOP_KEEP_G32"
        )

        elapsed = time.perf_counter() - started
        result = {
            "status": "fresh_supporting_confirmation",
            "selection_source": "screening windows 36..39",
            "confirmation_windows": [
                WINDOW_OFFSET, WINDOW_OFFSET + N_WINDOWS - 1
            ],
            "confirmation_window_count": N_WINDOWS,
            "reason_for_26_windows": (
                "all remaining complete, non-overlapping WikiText-2 windows "
                "after prior experiments"
            ),
            "model": MODEL,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "guardrails": {
                "max_relative_dPPL_vs_current_pct": (
                    MAX_REL_DPPL_VS_CURRENT_PCT
                ),
                "g64_vs_g32_dNLL_noninferiority_margin": (
                    NONINFERIORITY_DNLL_MARGIN
                ),
                "one_sided_confidence": 0.95,
                "t_critical_df25": T_ONE_SIDED_95_DF25,
            },
            "sanity": sanity,
            "aggregate": aggregate,
            "paired": paired,
            "per_window": per_window,
            "passes_quality_gate": passes_quality,
            "decision": decision,
            "elapsed_s": elapsed,
        }
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)

        print("\n=== V-INT4 G64 FRESH CONFIRMATION SUMMARY ===")
        for policy in POLICIES:
            record = aggregate[policy.name]
            print(
                f"{policy.name:16s} NLL={record['nll']:.6f} "
                f"PPL={record['ppl']:.4f} "
                f"dNLL_base={record['dNLL_vs_base']:+.6f} "
                f"rel_dPPL_current="
                f"{record['relative_dPPL_vs_current_pct']:+.3f}% "
                f"B/tok/layer={record['bytes_per_token_layer']:.1f} "
                f"8K_cache={record['cache_mib_at_8k_28_layers']:.2f}MiB"
            )
        print(
            "g64 vs g32 paired: "
            f"mean_dNLL={g64_vs_g32['mean_dNLL']:+.6f} "
            f"SE={g64_vs_g32['se']:.6f} "
            f"one_sided_95_upper="
            f"{g64_vs_g32['one_sided_95_upper']:+.6f} "
            f"margin={NONINFERIORITY_DNLL_MARGIN:.6f}"
        )
        print(f"confirmation_decision={decision}")
        print(f"elapsed={elapsed:.0f}s\nJSON={OUTPUT}")
        if passes_quality:
            print("Quality gate passed. Run no operational test until review.")
        else:
            print("Quality gate failed. Keep g32 and close the group-size tail.")
        return 0 if all_finite else 1
    finally:
        ms.GROUP_V_INT4 = original_group_size


if __name__ == "__main__":
    raise SystemExit(main())
