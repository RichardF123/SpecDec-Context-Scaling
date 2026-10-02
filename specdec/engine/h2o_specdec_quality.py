"""H2O v2 SpecDec quality/budget smoke evaluation.

This is not a proof of losslessness.  It measures how approximate H2O
retention changes SpecDec output under active pruning.
"""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .spec_dec_h2o import spec_decode_greedy_h2o


PROMPTS = {
    "systems": (
        "Speculative decoding reduces target-model work by drafting tokens "
        "and verifying them in blocks. KV cache optimizations must preserve "
        "the state transitions required for correct rollback. "
    ),
    "reasoning": (
        "A controlled experiment should keep the environment fixed, compare "
        "one mechanism at a time, and report both successful and failed cases. "
    ),
    "code": (
        "def commit(state, accepted): state.extend(accepted); "
        "return state  # rejected speculative tokens must not persist. "
    ),
}


def _prompt(tok, text: str, context: int) -> torch.Tensor:
    value = text
    while True:
        ids = tok(
            value,
            return_tensors="pt",
        ).input_ids
        if ids.shape[1] >= context:
            return ids[:, :context].cuda()
        value += text


def _agreement(
    baseline_ids: list[int],
    candidate_ids: list[int],
    prompt_len: int,
):
    x = [int(v) for v in baseline_ids[prompt_len:]]
    y = [int(v) for v in candidate_ids[prompt_len:]]
    n = min(len(x), len(y))

    if n == 0:
        return 1.0, 0, True

    agreement = (
        sum(int(x[i] == y[i]) for i in range(n))
        / n
    )

    prefix = 0
    while prefix < n and x[prefix] == y[prefix]:
        prefix += 1

    exact = (
        len(x) == len(y)
        and prefix == len(x)
    )
    return agreement, prefix, exact


def _configure(model, page_size: int):
    model.set_kv_format("fp16")
    model.set_cache_backend(
        "paged_v0",
        page_size=page_size,
    )
    model.set_paged_attention_mode(
        "direct_cuda")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        default="128,256",
    )
    parser.add_argument(
        "--ratios",
        default="0.25,0.5",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--csv",
        default="h2o_v2_specdec_quality.csv",
    )
    args = parser.parse_args()

    contexts = [
        int(x)
        for x in args.contexts.split(",")
        if x
    ]
    ratios = [
        float(x)
        for x in args.ratios.split(",")
        if x
    ]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v2 SpecDec quality smoke")
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
            ids = _prompt(
                tok,
                text,
                context,
            )
            prompt_len = int(ids.shape[1])

            _configure(
                target, args.page_size)
            _configure(
                draft, args.page_size)

            baseline = spec_dec.spec_decode_greedy(
                target,
                draft,
                ids,
                args.steps,
                k=4,
                eos_id=None,
                target_kv="base_bf16",
                draft_kv="base_bf16",
            )

            for ratio in ratios:
                budget = max(
                    2,
                    int(round(context * ratio)),
                )
                hh = budget // 2
                recent = budget - hh

                result = spec_decode_greedy_h2o(
                    target,
                    draft,
                    ids,
                    args.steps,
                    hh_size=hh,
                    recent_size=recent,
                    k=4,
                    eos_id=None,
                    page_size=args.page_size,
                    attention_mode="direct_cuda",
                    return_caches=False,
                )

                agreement, prefix, exact = _agreement(
                    baseline["ids"],
                    result["ids"],
                    prompt_len,
                )

                hs = result["h2o_stats"]
                st = result["stats"]

                row = {
                    "family": family,
                    "context": context,
                    "ratio": ratio,
                    "budget": budget,
                    "steps": args.steps,
                    "token_agreement": agreement,
                    "common_prefix": prefix,
                    "exact_match": int(exact),
                    "baseline_rounds":
                        baseline["stats"]["rounds"],
                    "h2o_rounds": st["rounds"],
                    "baseline_accepted":
                        baseline["stats"]["accepted"],
                    "h2o_accepted": st["accepted"],
                    "h2o_proposed": st["proposed"],
                    "h2o_mismatch_rounds":
                        st["mismatch_rounds"],
                    "prune_rounds":
                        hs.prune_rounds,
                    "target_resident":
                        hs.final_target_resident,
                    "draft_resident":
                        hs.final_draft_resident,
                    "seen_tokens":
                        result["seen_tokens"],
                    "discarded_speculative_rows":
                        hs.discarded_proposal_rows
                        + hs.discarded_verify_rows,
                }
                rows.append(row)

                print(
                    f"[{family:<9}] ctx={context:<4} "
                    f"ratio={ratio:.2f} budget={budget:<4} "
                    f"agree={agreement:.3f} "
                    f"prefix={prefix:<2}/{args.steps} "
                    f"exact={int(exact)} "
                    f"resident(T/D)="
                    f"{hs.final_target_resident}/"
                    f"{hs.final_draft_resident} "
                    f"seen={result['seen_tokens']} "
                    f"prune_rounds={hs.prune_rounds}"
                )

    path = Path(args.csv).resolve()
    with path.open(
            "w",
            newline="",
            encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV written:", path)
    print("h2o_specdec_quality: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
