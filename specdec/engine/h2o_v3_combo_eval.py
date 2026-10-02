"""Combined KV optimization evaluation: V-FP8 target + H2O + SpecDec.

This measures quality and resident payload, not speed.  H2O v3 still uses
reference score capture and reference compaction.
"""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from . import paged_cache, spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .spec_dec_h2o import spec_decode_greedy_h2o


PROMPTS = {
    "systems": (
        "A KV cache optimization should reduce memory while preserving the "
        "state required by speculative decoding and rollback. "
    ),
    "reasoning": (
        "A reliable experiment changes one mechanism at a time, compares "
        "against a fixed baseline, and records both successes and failures. "
    ),
    "code": (
        "def commit(state, accepted): state.extend(accepted); "
        "return state  # rejected speculative tokens are discarded. "
    ),
}


def _prompt(tok, text: str, context: int) -> torch.Tensor:
    value = text
    while True:
        ids = tok(value, return_tensors="pt").input_ids
        if ids.shape[1] >= context:
            return ids[:, :context].cuda()
        value += text


def _agreement(a: list[int], b: list[int], prompt_len: int):
    x = [int(v) for v in a[prompt_len:]]
    y = [int(v) for v in b[prompt_len:]]
    n = min(len(x), len(y))
    if n == 0:
        return 1.0, 0, True

    agree = sum(int(x[i] == y[i]) for i in range(n)) / n
    prefix = 0
    while prefix < n and x[prefix] == y[prefix]:
        prefix += 1
    exact = len(x) == len(y) and prefix == len(x)
    return agree, prefix, exact


def _payload(cache) -> int:
    return sum(paged_cache.payload_bytes(layer) for layer in cache)


def _configure(target, draft, *, target_fmt: str, page_size: int):
    target.set_kv_format(target_fmt)
    target.set_cache_backend("paged_v0", page_size=page_size)
    target.set_paged_attention_mode("direct_cuda")

    draft.set_kv_format("fp16")
    draft.set_cache_backend("paged_v0", page_size=page_size)
    draft.set_paged_attention_mode("direct_cuda")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="128,256")
    parser.add_argument("--ratios", default="0.25,0.5")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--csv", default="h2o_v3_combo.csv")
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    ratios = [float(x) for x in args.ratios.split(",") if x]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v3 combined KV optimization evaluation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print(
        f"contexts={contexts} ratios={ratios} "
        f"steps={args.steps} page={args.page_size}"
    )

    torch.manual_seed(0)

    print(f"[load] target {MODEL_TARGET}")
    target, tok = load_ours(MODEL_TARGET)
    print(f"[load] draft {MODEL_SMALL}")
    draft, _ = load_ours(MODEL_SMALL)

    rows = []

    for family, text in PROMPTS.items():
        for context in contexts:
            ids = _prompt(tok, text, context)
            prompt_len = int(ids.shape[1])

            # Full BF16 baseline.
            _configure(
                target, draft,
                target_fmt="fp16",
                page_size=args.page_size,
            )
            bf16 = spec_dec.spec_decode_greedy(
                target,
                draft,
                ids,
                args.steps,
                k=4,
                eos_id=None,
                target_kv="base_bf16",
                draft_kv="base_bf16",
                return_caches=True,
            )
            bf16_payload = (
                _payload(bf16["target_cache"])
                + _payload(bf16["draft_cache"])
            )

            # Quantization-only baseline.
            _configure(
                target, draft,
                target_fmt="v_fp8",
                page_size=args.page_size,
            )
            vfp8 = spec_dec.spec_decode_greedy(
                target,
                draft,
                ids,
                args.steps,
                k=4,
                eos_id=None,
                target_kv="v_fp8",
                draft_kv="base_bf16",
                return_caches=True,
            )
            vfp8_payload = (
                _payload(vfp8["target_cache"])
                + _payload(vfp8["draft_cache"])
            )

            q_agree, q_prefix, q_exact = _agreement(
                bf16["ids"], vfp8["ids"], prompt_len)

            print(
                f"[{family:<9}] ctx={context:<4} quant-only "
                f"agree_vs_bf16={q_agree:.3f} "
                f"prefix={q_prefix:<2}/{args.steps} "
                f"exact={int(q_exact)} "
                f"payload_ratio={vfp8_payload/bf16_payload:.3f}"
            )

            for ratio in ratios:
                budget = max(2, int(round(context * ratio)))
                result = spec_decode_greedy_h2o(
                    target,
                    draft,
                    ids,
                    args.steps,
                    hh_size=budget // 2,
                    recent_size=budget - budget // 2,
                    k=4,
                    eos_id=None,
                    page_size=args.page_size,
                    attention_mode="direct_cuda",
                    target_kv="v_fp8",
                    draft_kv="base_bf16",
                    return_caches=True,
                )

                combo_payload = (
                    _payload(result["target_cache"])
                    + _payload(result["draft_cache"])
                )

                agree_v, prefix_v, exact_v = _agreement(
                    vfp8["ids"], result["ids"], prompt_len)
                agree_b, prefix_b, exact_b = _agreement(
                    bf16["ids"], result["ids"], prompt_len)

                hs = result["h2o_stats"]

                row = {
                    "family": family,
                    "context": context,
                    "ratio": ratio,
                    "budget": budget,
                    "steps": args.steps,
                    "quant_agree_vs_bf16": q_agree,
                    "quant_prefix_vs_bf16": q_prefix,
                    "quant_exact_vs_bf16": int(q_exact),
                    "combo_agree_vs_vfp8": agree_v,
                    "combo_prefix_vs_vfp8": prefix_v,
                    "combo_exact_vs_vfp8": int(exact_v),
                    "combo_agree_vs_bf16": agree_b,
                    "combo_prefix_vs_bf16": prefix_b,
                    "combo_exact_vs_bf16": int(exact_b),
                    "bf16_payload_mib": bf16_payload / 2**20,
                    "vfp8_payload_mib": vfp8_payload / 2**20,
                    "combo_payload_mib": combo_payload / 2**20,
                    "vfp8_payload_ratio_vs_bf16":
                        vfp8_payload / bf16_payload,
                    "combo_payload_ratio_vs_bf16":
                        combo_payload / bf16_payload,
                    "combo_payload_ratio_vs_vfp8":
                        combo_payload / vfp8_payload,
                    "target_resident": hs.final_target_resident,
                    "draft_resident": hs.final_draft_resident,
                    "seen_tokens": result["seen_tokens"],
                    "prune_rounds": hs.prune_rounds,
                    "mismatch_rounds":
                        result["stats"]["mismatch_rounds"],
                }
                rows.append(row)

                print(
                    f"[{family:<9}] ctx={context:<4} ratio={ratio:.2f} "
                    f"combo agree(vfp8/bf16)="
                    f"{agree_v:.3f}/{agree_b:.3f} "
                    f"prefix={prefix_v}/{prefix_b} "
                    f"payload_ratio(bf16/vfp8)="
                    f"{combo_payload/bf16_payload:.3f}/"
                    f"{combo_payload/vfp8_payload:.3f} "
                    f"resident={hs.final_target_resident}/"
                    f"{hs.final_draft_resident}"
                )

    out = Path(args.csv).resolve()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV written:", out)
    print("h2o_v3_combo_eval: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
