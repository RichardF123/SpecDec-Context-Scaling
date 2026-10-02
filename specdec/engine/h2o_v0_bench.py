"""H2O v0 retention/quality smoke benchmark.

No speed claim: v0 uses reference score capture and reference compaction.
"""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import torch
from .g4_b4_cache_tests import MODEL_SMALL, load_ours
from .h2o_generate import h2o_generate_greedy


BASE_TEXT = (
    "Large language model inference keeps key and value states from previous "
    "tokens. Paged storage controls allocation while selective retention can "
    "bound the resident context. This sentence is repeated only to create a "
    "deterministic long prompt for an engineering smoke test. "
)


def _prompt(tok, context: int) -> torch.Tensor:
    text = BASE_TEXT
    while True:
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] >= context:
            return ids[:, :context].cuda()
        text += BASE_TEXT


def _agreement(a: torch.Tensor, b: torch.Tensor, prompt_len: int) -> tuple[float, int]:
    x = a[0, prompt_len:].tolist()
    y = b[0, prompt_len:].tolist()
    n = min(len(x), len(y))
    if n == 0:
        return 1.0, 0
    match = sum(int(x[i] == y[i]) for i in range(n)) / n
    prefix = 0
    while prefix < n and x[prefix] == y[prefix]:
        prefix += 1
    return match, prefix


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="256,512")
    p.add_argument("--budget-ratios", default="0.25,0.5")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--csv", default="h2o_v0_bench.csv")
    args = p.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    ratios = [float(x) for x in args.budget_ratios.split(",") if x]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v0 benchmark: retention/quality smoke only; NO speed claim")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    model, tok = load_ours(MODEL_SMALL)
    rows = []

    for context in contexts:
        ids = _prompt(tok, context)
        baseline = h2o_generate_greedy(
            model, ids, args.steps,
            hh_size=context + args.steps + 32, recent_size=0,
            page_size=args.page_size, eos_id=None,
            attention_mode="direct_cuda")
        baseline_bytes = baseline["payload_bytes"]

        for ratio in ratios:
            budget = max(2, int(round(context * ratio)))
            hh = budget // 2
            recent = budget - hh
            result = h2o_generate_greedy(
                model, ids, args.steps,
                hh_size=hh, recent_size=recent,
                page_size=args.page_size, eos_id=None,
                attention_mode="direct_cuda")
            agree, prefix = _agreement(
                baseline["ids"], result["ids"], int(ids.shape[1]))
            row = {
                "context": context,
                "ratio": ratio,
                "budget": budget,
                "hh_size": hh,
                "recent_size": recent,
                "steps": args.steps,
                "seen_tokens": result["stats"].seen_tokens,
                "resident_tokens": result["stats"].final_resident,
                "prune_events": result["stats"].prune_events,
                "tokens_evicted": result["stats"].tokens_evicted,
                "payload_mib": result["payload_bytes"] / 2**20,
                "baseline_payload_mib": baseline_bytes / 2**20,
                "payload_ratio": result["payload_bytes"] / baseline_bytes,
                "token_agreement": agree,
                "common_prefix": prefix,
            }
            rows.append(row)
            print(
                f"ctx={context:<4} ratio={ratio:.2f} budget={budget:<4} "
                f"resident={row['resident_tokens']:<4} "
                f"payload={row['payload_mib']:.2f} MiB "
                f"payload_ratio={row['payload_ratio']:.3f} "
                f"agreement={agree:.3f} prefix={prefix}"
            )

    out = Path(args.csv).resolve()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("CSV written:", out)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
