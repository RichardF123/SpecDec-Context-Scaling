"""Paired BF16-H2O vs V-FP8-H2O interaction audit.

This audit deliberately uses the SAME prompt, context, budget, model weights,
SpecDec settings, and H2O policy for both arms.

It answers:
1. Does V-FP8 alone change the no-H2O SpecDec output?
2. Do BF16-H2O and V-FP8-H2O choose different heavy-hitter / keep sets?
3. If they do, does the retention difference appear before the first generated
   token divergence?

The trace hook is observation-only; it never changes H2O selection.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .spec_dec_h2o import spec_decode_greedy_h2o


PROMPTS = {
    # Keep these EXACTLY aligned with h2o_v3_combo_eval.py.
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


def _ints(values) -> list[int]:
    return [int(v) for v in values]


def _generated(result: dict, prompt_len: int) -> list[int]:
    return _ints(result["ids"][prompt_len:])


def _first_token_diff(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _jaccard(a, b) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / len(aa | bb)


def _max_abs_delta(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b):
        return None
    if not a:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _event_key(e: dict):
    return (e["role"], int(e["round"]), int(e["layer"]))


def _first_trace_difference(
    left: list[dict],
    right: list[dict],
    *,
    field: str,
    role: str = "target",
):
    lm = {_event_key(e): e for e in left if e["role"] == role}
    rm = {_event_key(e): e for e in right if e["role"] == role}
    keys = sorted(set(lm) & set(rm), key=lambda x: (x[1], x[2]))

    for key in keys:
        a, b = lm[key], rm[key]
        # Once resident absolute positions have already diverged, this is no
        # longer the first causal decision point; earlier trace events should
        # explain how we got there.
        if a["pre_positions"] != b["pre_positions"]:
            return {
                "key": key,
                "reason": "pre_positions_already_different",
                "left": a,
                "right": b,
            }

        if a[field] != b[field]:
            return {
                "key": key,
                "reason": f"{field}_different",
                "left": a,
                "right": b,
            }
    return None


def _first_score_delta(
    left: list[dict],
    right: list[dict],
    *,
    role: str = "target",
):
    lm = {_event_key(e): e for e in left if e["role"] == role}
    rm = {_event_key(e): e for e in right if e["role"] == role}
    keys = sorted(set(lm) & set(rm), key=lambda x: (x[1], x[2]))

    for key in keys:
        a, b = lm[key], rm[key]
        if a["pre_positions"] != b["pre_positions"]:
            return None
        delta = _max_abs_delta(a["pre_scores"], b["pre_scores"])
        if delta is not None and delta > 0.0:
            return {
                "key": key,
                "max_abs_score_delta": delta,
                "left": a,
                "right": b,
            }
    return None


def _configure(target, draft, *, target_fmt: str, page_size: int) -> None:
    target.set_kv_format(target_fmt)
    target.set_cache_backend("paged_v0", page_size=page_size)
    target.set_paged_attention_mode("direct_cuda")

    draft.set_kv_format("fp16")
    draft.set_cache_backend("paged_v0", page_size=page_size)
    draft.set_paged_attention_mode("direct_cuda")


def _plain_specdec(
    target,
    draft,
    ids,
    *,
    target_kv: str,
    page_size: int,
    steps: int,
):
    _configure(
        target,
        draft,
        target_fmt="v_fp8" if target_kv == "v_fp8" else "fp16",
        page_size=page_size,
    )
    return spec_dec.spec_decode_greedy(
        target,
        draft,
        ids,
        steps,
        k=4,
        eos_id=None,
        target_kv=target_kv,
        draft_kv="base_bf16",
    )


def _h2o_run(
    target,
    draft,
    ids,
    *,
    target_kv: str,
    budget: int,
    page_size: int,
    steps: int,
):
    trace = []
    result = spec_decode_greedy_h2o(
        target,
        draft,
        ids,
        steps,
        hh_size=budget // 2,
        recent_size=budget - budget // 2,
        k=4,
        eos_id=None,
        page_size=page_size,
        attention_mode="direct_cuda",
        target_kv=target_kv,
        draft_kv="base_bf16",
        trace_prune=trace.append,
        return_caches=False,
    )
    return result, trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="128,256")
    parser.add_argument("--ratios", default="0.25,0.5")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--csv", default="h2o_v3a_interaction_summary.csv")
    parser.add_argument("--json", default="h2o_v3a_interaction_detail.json")
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    ratios = [float(x) for x in args.ratios.split(",") if x]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v3a paired BF16/V-FP8 interaction audit")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print(
        f"contexts={contexts} ratios={ratios} "
        f"steps={args.steps} page={args.page_size}"
    )
    print("IMPORTANT: both H2O arms use the exact same prompt and budget.")

    torch.manual_seed(0)

    print(f"[load] target {MODEL_TARGET}")
    target, tok = load_ours(MODEL_TARGET)
    print(f"[load] draft {MODEL_SMALL}")
    draft, _ = load_ours(MODEL_SMALL)

    rows = []
    details = []

    for family, text in PROMPTS.items():
        for context in contexts:
            ids = _prompt(tok, text, context)
            prompt_len = int(ids.shape[1])

            # Quantization-only paired control, same exact input.
            plain_bf16 = _plain_specdec(
                target, draft, ids,
                target_kv="base_bf16",
                page_size=args.page_size,
                steps=args.steps,
            )
            plain_vfp8 = _plain_specdec(
                target, draft, ids,
                target_kv="v_fp8",
                page_size=args.page_size,
                steps=args.steps,
            )
            plain_b = _generated(plain_bf16, prompt_len)
            plain_v = _generated(plain_vfp8, prompt_len)
            quant_first_diff = _first_token_diff(plain_b, plain_v)
            quant_exact = quant_first_diff is None

            for ratio in ratios:
                budget = max(2, int(round(context * ratio)))

                bf16, trace_b = _h2o_run(
                    target, draft, ids,
                    target_kv="base_bf16",
                    budget=budget,
                    page_size=args.page_size,
                    steps=args.steps,
                )
                vfp8, trace_v = _h2o_run(
                    target, draft, ids,
                    target_kv="v_fp8",
                    budget=budget,
                    page_size=args.page_size,
                    steps=args.steps,
                )

                gen_b = _generated(bf16, prompt_len)
                gen_v = _generated(vfp8, prompt_len)
                output_diff = _first_token_diff(gen_b, gen_v)

                score_diff = _first_score_delta(trace_b, trace_v)
                rank_diff = _first_trace_difference(
                    trace_b, trace_v,
                    field="heavy_ranked_positions",
                )
                keep_diff = _first_trace_difference(
                    trace_b, trace_v,
                    field="keep_positions",
                )

                keep_round = keep_layer = keep_generated_after = None
                keep_jaccard = None
                if keep_diff is not None:
                    _, keep_round, keep_layer = keep_diff["key"]
                    a = keep_diff["left"]
                    b = keep_diff["right"]
                    keep_generated_after = int(a["generated_after"])
                    keep_jaccard = _jaccard(
                        a["keep_positions"],
                        b["keep_positions"],
                    )

                rank_round = rank_layer = None
                rank_jaccard = None
                if rank_diff is not None:
                    _, rank_round, rank_layer = rank_diff["key"]
                    rank_jaccard = _jaccard(
                        rank_diff["left"]["heavy_ranked_positions"],
                        rank_diff["right"]["heavy_ranked_positions"],
                    )

                score_round = score_layer = None
                score_max_abs = None
                if score_diff is not None:
                    _, score_round, score_layer = score_diff["key"]
                    score_max_abs = float(score_diff["max_abs_score_delta"])

                retention_before_output = None
                if output_diff is not None and keep_generated_after is not None:
                    # Pruning after N committed generated tokens can affect token
                    # index N and later (0-based generated suffix indexing).
                    retention_before_output = (
                        keep_generated_after <= output_diff
                    )

                if not quant_exact:
                    classification = "quantization_output_confounded"
                elif output_diff is None and keep_diff is None:
                    classification = "no_observed_interaction"
                elif output_diff is None and keep_diff is not None:
                    classification = "retention_changed_output_same"
                elif keep_diff is not None and retention_before_output:
                    classification = "retention_diverged_before_output"
                elif output_diff is not None and keep_diff is None:
                    classification = "output_diverged_without_observed_keep_diff"
                else:
                    classification = "ordering_not_established"

                row = {
                    "family": family,
                    "context": context,
                    "ratio": ratio,
                    "budget": budget,
                    "steps": args.steps,
                    "quant_only_exact": int(quant_exact),
                    "quant_only_first_output_diff": quant_first_diff,
                    "bf16_h2o_rounds": bf16["stats"]["rounds"],
                    "vfp8_h2o_rounds": vfp8["stats"]["rounds"],
                    "bf16_h2o_accepted": bf16["stats"]["accepted"],
                    "vfp8_h2o_accepted": vfp8["stats"]["accepted"],
                    "first_h2o_output_diff": output_diff,
                    "first_score_diff_round": score_round,
                    "first_score_diff_layer": score_layer,
                    "first_score_max_abs_delta": score_max_abs,
                    "first_heavy_rank_diff_round": rank_round,
                    "first_heavy_rank_diff_layer": rank_layer,
                    "first_heavy_set_jaccard": rank_jaccard,
                    "first_keep_diff_round": keep_round,
                    "first_keep_diff_layer": keep_layer,
                    "first_keep_set_jaccard": keep_jaccard,
                    "keep_generated_after": keep_generated_after,
                    "retention_before_output":
                        None if retention_before_output is None
                        else int(retention_before_output),
                    "classification": classification,
                }
                rows.append(row)

                details.append({
                    "summary": row,
                    "plain_bf16_generated": plain_b,
                    "plain_vfp8_generated": plain_v,
                    "bf16_h2o_generated": gen_b,
                    "vfp8_h2o_generated": gen_v,
                    "first_score_difference": score_diff,
                    "first_heavy_ranking_difference": rank_diff,
                    "first_keep_difference": keep_diff,
                })

                print(
                    f"[{family:<9}] ctx={context:<4} ratio={ratio:.2f} "
                    f"quant_exact={int(quant_exact)} "
                    f"h2o_out_diff={output_diff} "
                    f"score_diff=({score_round},{score_layer}) "
                    f"rank_diff=({rank_round},{rank_layer}) "
                    f"keep_diff=({keep_round},{keep_layer}) "
                    f"class={classification}"
                )

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(args.json).resolve()
    json_path.write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    print("\n=== audit interpretation counts ===")
    counts = {}
    for row in rows:
        counts[row["classification"]] = counts.get(
            row["classification"], 0) + 1
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")

    print("\nSummary CSV written:", csv_path)
    print("Detailed JSON written:", json_path)
    print("h2o_v3a_interaction_audit: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
