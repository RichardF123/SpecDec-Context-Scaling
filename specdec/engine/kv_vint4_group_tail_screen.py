"""Bounded V-INT4 scale-granularity tail screening.

This is an isolated, supporting experiment.  It compares group sizes 32, 64,
and 128 on four previously unused WikiText-2 windows (36..39).  The production
``model_self.py`` is not modified: before each fresh-cache run, this harness
temporarily changes the module-level V-INT4 group size used by the existing
``v_int4`` path.

The script never reads the reserved follow-up windows 120..145 and never runs an
8K operational benchmark.  It may nominate at most one group size for a later
fresh confirmation, using the pre-registered screening guardrails below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import time

import torch

from . import model_self as ms
from .fp8_kv_v4_dppl import ensure_data
from .kv_policy_pilot_v2 import MODEL, WINDOW, CHUNK, LOSS_ROWS, load_model


DTYPE = torch.bfloat16
WINDOW_OFFSET = 36
N_WINDOWS = 4
CONFIRMATION_WINDOWS_RESERVED = (120, 145)
GROUP_SIZES = (32, 64, 128)

# Screening is deliberately a little looser than the eventual confirmation
# rule (1.0% versus current and +0.002 dNLL versus g32).  Four windows are used
# only to reject clearly bad choices, not to make the final scientific claim.
SCREEN_MAX_REL_DPPL_VS_CURRENT_PCT = 1.25
SCREEN_MAX_EXTRA_DNLL_VS_G32 = 0.003
QUALITY_EQUIVALENCE_DNLL = 0.0005

OUTPUT = os.path.join(
    os.environ.get("SPECDEC_RUN_LOG_DIR", "run_logs/gqa_kv"),
    "17_vint4_group_tail_screen.json",
)


@dataclass(frozen=True)
class Policy:
    name: str
    fmt: str
    int4_group_size: int | None = None


POLICIES = (
    Policy("base_bf16", "fp16"),
    Policy("current_vfp8", "v_fp8"),
    Policy("vint4_g32", "v_int4", 32),
    Policy("vint4_g64", "v_int4", 64),
    Policy("vint4_g128", "v_int4", 128),
)

EXPECTED_BYTES_PER_TOKEN_LAYER = {
    "base_bf16": 1024.0,
    "current_vfp8": 776.0,
    "vint4_g32": 672.0,
    "vint4_g64": 656.0,
    "vint4_g128": 648.0,
}


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def select_policy(model, policy: Policy) -> None:
    """Select a policy before starting a fresh cache."""
    if policy.int4_group_size is not None:
        if model.head_dim % policy.int4_group_size:
            raise RuntimeError(
                f"head_dim={model.head_dim} is not divisible by "
                f"group={policy.int4_group_size}"
            )
        ms.GROUP_V_INT4 = policy.int4_group_size
    model.set_kv_head_policy(2, 2)
    model.set_kv_format(policy.fmt)


def quantizer_sanity() -> list[dict]:
    """Check shape, determinism, finiteness, and zero behavior for every group."""
    generator = torch.Generator(device="cuda").manual_seed(20260914)
    sample = torch.randn(
        (1, 2, 3, 128), device="cuda", dtype=DTYPE, generator=generator
    )
    zeros = torch.zeros_like(sample)
    records = []
    for group_size in GROUP_SIZES:
        packed_1, scale_1 = ms.quantize_kv_int4_packed(sample, group_size)
        packed_2, scale_2 = ms.quantize_kv_int4_packed(sample, group_size)
        restored = ms.dequantize_kv_int4_packed(packed_1, scale_1, DTYPE)
        zero_q, zero_s = ms.quantize_kv_int4_packed(zeros, group_size)
        zero_hat = ms.dequantize_kv_int4_packed(zero_q, zero_s, DTYPE)
        expected_groups = 128 // group_size
        passed = bool(
            packed_1.dtype == torch.uint8
            and packed_1.shape[-1] == 64
            and scale_1.shape[-1] == expected_groups
            and torch.equal(packed_1, packed_2)
            and torch.equal(scale_1, scale_2)
            and torch.isfinite(restored).all().item()
            and (zero_hat == 0).all().item()
        )
        record = {
            "group_size": group_size,
            "scale_groups_per_head": expected_groups,
            "packed_head_bytes": 64,
            "passed": passed,
        }
        records.append(record)
        print(
            f"sanity g{group_size}: groups/head={expected_groups} "
            f"packed/head=64B deterministic+finite+zero_exact={passed}",
            flush=True,
        )
    return records


def window_stats(model, ids_w: torch.Tensor, policy: Policy):
    select_policy(model, policy)
    cache = None
    nll_sum = 0.0
    n_tokens = 0
    finite = True
    with torch.inference_mode():
        for start in range(0, WINDOW, CHUNK):
            piece = ids_w[start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece.unsqueeze(0)), cache, need_logits=True
            )
            finite &= bool(torch.isfinite(logits).all().item())
            targets = ids_w[start + 1:start + 1 + CHUNK]
            rows = targets.shape[0]
            for row_start in range(0, rows, LOSS_ROWS):
                row_end = min(row_start + LOSS_ROWS, rows)
                log_probs = torch.log_softmax(
                    logits[0, row_start:row_end].to(torch.float32), dim=-1
                )
                indices = torch.arange(
                    row_end - row_start, device=logits.device
                )
                nll_sum += -log_probs[
                    indices, targets[row_start:row_end]
                ].sum().item()
                del log_probs, indices
            n_tokens += rows
            del logits
    return nll_sum, n_tokens, finite, cache_logical_bytes(cache)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    original_group_size = ms.GROUP_V_INT4
    try:
        sanity = quantizer_sanity()
        if not all(row["passed"] for row in sanity):
            print("screening_decision=STOP_QUANTIZER_SANITY_FAILED")
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

        model = load_model()
        print(
            f"V-INT4 group tail screening: model=1.5B "
            f"windows={WINDOW_OFFSET}..{WINDOW_OFFSET + N_WINDOWS - 1} "
            f"W={WINDOW} chunk={CHUNK} dtype={DTYPE}",
            flush=True,
        )
        print(
            "supporting screen only; confirmation windows 120..145 remain unread\n",
            flush=True,
        )

        totals = {
            policy.name: {"nll_sum": 0.0, "tokens": 0, "finite": True}
            for policy in POLICIES
        }
        logical = {}
        per_window = []
        started = time.perf_counter()

        # Window-major order makes every policy comparison use exactly the same
        # data while each call starts from a fresh cache.
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
            per_window.append(row)
            print(
                f"window={window_offset:02d} "
                f"base={row['base_bf16']['nll']:.6f} "
                f"current_dNLL="
                f"{row['current_vfp8']['nll'] - row['base_bf16']['nll']:+.6f} "
                f"g32_dNLL="
                f"{row['vint4_g32']['nll'] - row['base_bf16']['nll']:+.6f} "
                f"g64_dNLL="
                f"{row['vint4_g64']['nll'] - row['base_bf16']['nll']:+.6f} "
                f"g128_dNLL="
                f"{row['vint4_g128']['nll'] - row['base_bf16']['nll']:+.6f}",
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
        g32 = aggregate["vint4_g32"]
        for record in aggregate.values():
            record["dNLL_vs_base"] = record["nll"] - base["nll"]
            record["relative_dPPL_vs_base_pct"] = (
                record["ppl"] / base["ppl"] - 1.0
            ) * 100.0
            record["relative_dPPL_vs_current_pct"] = (
                record["ppl"] / current["ppl"] - 1.0
            ) * 100.0
            record["dNLL_vs_g32"] = record["nll"] - g32["nll"]
            record["bytes_vs_g32"] = (
                record["bytes_per_token_layer"]
                / g32["bytes_per_token_layer"]
            )

        eligible = []
        for name in ("vint4_g64", "vint4_g128"):
            record = aggregate[name]
            record["passes_screening_guardrails"] = bool(
                record["finite"]
                and record["relative_dPPL_vs_current_pct"]
                <= SCREEN_MAX_REL_DPPL_VS_CURRENT_PCT
                and record["dNLL_vs_g32"] <= SCREEN_MAX_EXTRA_DNLL_VS_G32
            )
            if record["passes_screening_guardrails"]:
                eligible.append(name)

        selected = None
        if eligible:
            best_nll = min(aggregate[name]["nll"] for name in eligible)
            quality_band = [
                name for name in eligible
                if aggregate[name]["nll"] <= best_nll + QUALITY_EQUIVALENCE_DNLL
            ]
            # Within the pre-registered quality-equivalence band, prefer the
            # smaller representation.  Otherwise, quality wins.
            selected = min(
                quality_band,
                key=lambda name: aggregate[name]["bytes_per_token_layer"],
            )

        elapsed = time.perf_counter() - started
        result = {
            "status": "bounded_supporting_screen_no_confirmation_data_read",
            "model": MODEL,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "screening_windows": [
                WINDOW_OFFSET, WINDOW_OFFSET + N_WINDOWS - 1
            ],
            "reserved_confirmation_windows": list(
                CONFIRMATION_WINDOWS_RESERVED
            ),
            "guardrails": {
                "max_relative_dPPL_vs_current_pct": (
                    SCREEN_MAX_REL_DPPL_VS_CURRENT_PCT
                ),
                "max_extra_dNLL_vs_g32": SCREEN_MAX_EXTRA_DNLL_VS_G32,
                "quality_equivalence_dNLL": QUALITY_EQUIVALENCE_DNLL,
            },
            "sanity": sanity,
            "policies": [asdict(policy) for policy in POLICIES],
            "aggregate": aggregate,
            "per_window": per_window,
            "selected_for_fresh_confirmation": selected,
            "elapsed_s": elapsed,
        }
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)

        print("\n=== V-INT4 GROUP TAIL SCREENING SUMMARY ===")
        for policy in POLICIES:
            record = aggregate[policy.name]
            gate = record.get("passes_screening_guardrails")
            gate_text = "n/a" if gate is None else str(gate)
            print(
                f"{policy.name:16s} NLL={record['nll']:.6f} "
                f"PPL={record['ppl']:.4f} "
                f"dNLL_base={record['dNLL_vs_base']:+.6f} "
                f"rel_dPPL_current="
                f"{record['relative_dPPL_vs_current_pct']:+.3f}% "
                f"dNLL_g32={record['dNLL_vs_g32']:+.6f} "
                f"B/tok/layer={record['bytes_per_token_layer']:.1f} "
                f"8K_cache={record['cache_mib_at_8k_28_layers']:.2f}MiB "
                f"gate={gate_text}",
            )
        if selected is None:
            decision = "STOP_KEEP_G32"
            print("screening_decision=STOP_KEEP_G32")
        else:
            decision = f"PROCEED_WITH_{selected.upper()}"
            print(
                f"screening_decision={decision}; confirmation windows "
                "120..145 remain unread"
            )
        print(f"elapsed={elapsed:.0f}s\nJSON={OUTPUT}")
        print(
            "Do not run an operational benchmark yet. Review this screen first."
        )
        return 0
    finally:
        ms.GROUP_V_INT4 = original_group_size


if __name__ == "__main__":
    raise SystemExit(main())
