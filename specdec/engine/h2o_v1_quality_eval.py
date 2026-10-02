"""H2O v1 quality/retention evaluation.

Purpose
-------
H2O v0 established controller/correctness behavior.  This v1 harness asks the
next question before SpecDec integration:

    At the same resident-token budget, does heavy-hitter + recent retention
    preserve greedy generation better than a recent-only cache?

The harness also includes a strict no-eviction parity gate showing that H2O
score capture itself does not change model/cache behavior when nothing is
pruned.

This remains a quality/retention evaluation.  It is NOT a latency benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch

from . import paged_cache, spec_cache
from .g4_b4_cache_tests import MODEL_SMALL, load_ours
from .h2o_generate import h2o_generate_greedy
from .model_self import attention_phase


PROMPT_FAMILIES = {
    "systems": (
        "A transformer inference engine stores key and value states for prior "
        "tokens. Paged allocation avoids repeatedly copying the full cache. "
        "Selective retention can then reduce the number of resident tokens, "
        "but it must preserve information needed by future attention. "
    ),
    "reasoning": (
        "When comparing two engineering methods, keep the environment fixed, "
        "change one mechanism at a time, record the output, and check whether "
        "the observed difference matches the proposed explanation. "
    ),
    "code": (
        "def update_cache(state, token): state.append(token); "
        "assert len(state) > 0; return state  # deterministic test sequence. "
    ),
}


def _prompt(tok, base_text: str, context: int) -> torch.Tensor:
    if context <= 0:
        raise ValueError("context must be positive")
    text = base_text
    while True:
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] >= context:
            return ids[:, :context].cuda()
        text += base_text


def _cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().contiguous().cpu().clone()


def _raw_cache(cache) -> list[dict]:
    out = []
    for layer in cache:
        raw = paged_cache.materialize_layer(layer)
        out.append({
            key: _cpu(value)
            for key, value in raw.items()
            if isinstance(value, torch.Tensor)
        })
    return out


def _assert_cache_equal(left, right) -> None:
    a = _raw_cache(left)
    b = _raw_cache(right)
    assert len(a) == len(b)
    for li, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys(), f"layer {li}: cache keys differ"
        for key in x:
            assert x[key].dtype == y[key].dtype, (
                f"layer {li} {key}: dtype differs")
            assert torch.equal(x[key], y[key]), (
                f"layer {li} {key}: tensor differs")


def _plain_greedy(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    page_size: int,
) -> dict:
    """Paged/direct greedy generation with H2O score capture disabled."""
    model.set_kv_format("fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode("direct_cuda")
    model.set_h2o_score_capture(False)

    prompt_len = int(prompt_ids.shape[1])
    with torch.inference_mode(), attention_phase("prefill"):
        logits, cache = model._run(
            model.embed_tokens(prompt_ids),
            None,
            need_logits=True,
            start_pos_override=0,
        )

    generated = prompt_ids.clone()
    seen = prompt_len

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat([generated, token[:, None]], dim=1)
            with attention_phase("decode"):
                logits, cache = model._run(
                    model.embed_tokens(token[:, None]),
                    cache,
                    need_logits=True,
                    start_pos_override=seen,
                )
            seen += 1

    spec_cache.check_consistent(cache)
    return {
        "ids": generated,
        "cache": cache,
        "seen_tokens": seen,
        "resident_tokens": spec_cache.cache_length(cache),
        "payload_bytes": sum(
            paged_cache.payload_bytes(layer) for layer in cache),
    }


def _agreement(
    baseline_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    prompt_len: int,
) -> tuple[float, int, bool, list[int], list[int]]:
    x = baseline_ids[0, prompt_len:].tolist()
    y = candidate_ids[0, prompt_len:].tolist()
    n = min(len(x), len(y))
    if n == 0:
        return 1.0, 0, True, x, y

    agree = sum(int(x[i] == y[i]) for i in range(n)) / n
    prefix = 0
    while prefix < n and x[prefix] == y[prefix]:
        prefix += 1

    exact = (len(x) == len(y) and prefix == len(x))
    return agree, prefix, exact, x, y


def _no_evict_parity(model, tok, *, page_size: int, steps: int) -> None:
    ids = _prompt(tok, PROMPT_FAMILIES["systems"], 96)
    baseline = _plain_greedy(
        model, ids, steps, page_size=page_size)

    # Guaranteed to exceed every possible resident length in this gate.
    h2o = h2o_generate_greedy(
        model,
        ids,
        steps,
        hh_size=int(ids.shape[1]) + steps + 64,
        recent_size=0,
        page_size=page_size,
        eos_id=None,
        attention_mode="direct_cuda",
    )

    assert h2o["stats"].prune_events == 0
    assert torch.equal(baseline["ids"], h2o["ids"]), (
        "H2O score capture changed generated IDs without eviction")
    assert baseline["seen_tokens"] == h2o["stats"].seen_tokens
    assert baseline["resident_tokens"] == h2o["stats"].final_resident
    assert baseline["payload_bytes"] == h2o["payload_bytes"]
    _assert_cache_equal(baseline["cache"], h2o["cache"])

    print(
        "[no-evict parity] plain direct_cuda == H2O capture/no-prune: PASS "
        f"(seen={baseline['seen_tokens']}, "
        f"resident={baseline['resident_tokens']})"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="256,512")
    parser.add_argument("--budget-ratios", default="0.25,0.5")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--csv", default="h2o_v1_quality_raw.csv")
    parser.add_argument("--summary-csv", default="h2o_v1_quality_summary.csv")
    parser.add_argument("--divergences-json",
                        default="h2o_v1_divergences.json")
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    ratios = [float(x) for x in args.budget_ratios.split(",") if x]
    if not contexts or not ratios:
        raise SystemExit("contexts and budget-ratios must be non-empty")
    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("H2O v1 quality evaluation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print(
        f"contexts={contexts} ratios={ratios} "
        f"steps={args.steps} page_size={args.page_size}"
    )
    print("methods: h2o_balanced vs recent_only at identical token budgets")

    torch.manual_seed(0)
    model, tok = load_ours(MODEL_SMALL)

    _no_evict_parity(
        model, tok, page_size=args.page_size, steps=min(args.steps, 8))

    rows: list[dict] = []
    divergences: list[dict] = []

    for family, base_text in PROMPT_FAMILIES.items():
        for context in contexts:
            ids = _prompt(tok, base_text, context)
            prompt_len = int(ids.shape[1])

            baseline = _plain_greedy(
                model,
                ids,
                args.steps,
                page_size=args.page_size,
            )
            baseline_bytes = baseline["payload_bytes"]

            for ratio in ratios:
                budget = max(2, int(round(context * ratio)))

                methods = (
                    ("h2o_balanced", budget // 2, budget - budget // 2),
                    ("recent_only", 0, budget),
                )

                for method, hh_size, recent_size in methods:
                    result = h2o_generate_greedy(
                        model,
                        ids,
                        args.steps,
                        hh_size=hh_size,
                        recent_size=recent_size,
                        page_size=args.page_size,
                        eos_id=None,
                        attention_mode="direct_cuda",
                    )

                    agree, prefix, exact, baseline_new, candidate_new = _agreement(
                        baseline["ids"], result["ids"], prompt_len)

                    payload_ratio = (
                        result["payload_bytes"] / baseline_bytes
                        if baseline_bytes else float("nan")
                    )

                    row = {
                        "family": family,
                        "context": context,
                        "ratio": ratio,
                        "method": method,
                        "budget": budget,
                        "hh_size": hh_size,
                        "recent_size": recent_size,
                        "steps": args.steps,
                        "seen_tokens": result["stats"].seen_tokens,
                        "resident_tokens": result["stats"].final_resident,
                        "prune_events": result["stats"].prune_events,
                        "tokens_evicted_layer_sum":
                            result["stats"].tokens_evicted,
                        "payload_mib": result["payload_bytes"] / 2**20,
                        "baseline_payload_mib": baseline_bytes / 2**20,
                        "payload_ratio": payload_ratio,
                        "token_agreement": agree,
                        "common_prefix": prefix,
                        "exact_match": int(exact),
                    }
                    rows.append(row)

                    print(
                        f"[{family:<9}] ctx={context:<4} ratio={ratio:.2f} "
                        f"{method:<12} budget={budget:<4} "
                        f"resident={row['resident_tokens']:<4} "
                        f"payload_ratio={payload_ratio:.3f} "
                        f"agree={agree:.3f} prefix={prefix:<2} "
                        f"exact={int(exact)}"
                    )

                    if not exact:
                        divergences.append({
                            "family": family,
                            "context": context,
                            "ratio": ratio,
                            "method": method,
                            "budget": budget,
                            "baseline_generated_ids": baseline_new,
                            "candidate_generated_ids": candidate_new,
                            "common_prefix": prefix,
                            "token_agreement": agree,
                        })

    raw_path = Path(args.csv).resolve()
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["context"], row["ratio"], row["method"])
        ].append(row)

    summary_rows = []
    print("\n=== aggregate across prompt families ===")
    for key in sorted(grouped):
        context, ratio, method = key
        group = grouped[key]
        summary = {
            "context": context,
            "ratio": ratio,
            "method": method,
            "n_prompts": len(group),
            "mean_token_agreement": _mean(
                [float(r["token_agreement"]) for r in group]),
            "mean_common_prefix": _mean(
                [float(r["common_prefix"]) for r in group]),
            "exact_match_rate": _mean(
                [float(r["exact_match"]) for r in group]),
            "mean_payload_ratio": _mean(
                [float(r["payload_ratio"]) for r in group]),
            "mean_prune_events": _mean(
                [float(r["prune_events"]) for r in group]),
        }
        summary_rows.append(summary)
        print(
            f"ctx={context:<4} ratio={ratio:.2f} {method:<12} "
            f"agreement={summary['mean_token_agreement']:.3f} "
            f"prefix={summary['mean_common_prefix']:.2f}/{args.steps} "
            f"exact_rate={summary['exact_match_rate']:.3f} "
            f"payload_ratio={summary['mean_payload_ratio']:.3f}"
        )

    summary_path = Path(args.summary_csv).resolve()
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    divergence_path = Path(args.divergences_json).resolve()
    divergence_path.write_text(
        json.dumps(divergences, indent=2),
        encoding="utf-8",
    )

    print("\nRaw CSV written:", raw_path)
    print("Summary CSV written:", summary_path)
    print("Divergences JSON written:", divergence_path)
    print("\nh2o_v1_quality_eval: PASS")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
