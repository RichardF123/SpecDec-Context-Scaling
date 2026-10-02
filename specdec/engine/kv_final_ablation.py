"""Final KV optimization ablation for the SpecDec project.

This is an evaluation-only harness.  It does not modify any core algorithm.

Configurations
--------------
A  BF16 contiguous baseline
B  + paged KV storage (reference/materialized attention path)
C  + direct page-aware q_len=1 attention
D  + V-FP8 target cache
E  BF16 + H2O 50% resident-token budget
F  V-FP8 + H2O 50% resident-token budget
G  V-FP8 + H2O 25% resident-token budget (aggressive stress setting)

Quality is always compared against configuration A on the exact same input.
Payload is the actually allocated target+draft KV payload at the end of the
run.  H2O latency is intentionally NOT reported because the current H2O score
capture/compaction path is a reference implementation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean

import torch

from . import paged_cache, spec_cache, spec_dec
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


CONFIGS = (
    {
        "id": "A",
        "name": "BF16 contiguous baseline",
        "kind": "standard",
        "backend": "contiguous",
        "attention_mode": "off",
        "target_kv": "base_bf16",
        "h2o_ratio": None,
    },
    {
        "id": "B",
        "name": "+ Paged KV storage",
        "kind": "standard",
        "backend": "paged_v0",
        "attention_mode": "off",
        "target_kv": "base_bf16",
        "h2o_ratio": None,
    },
    {
        "id": "C",
        "name": "+ Direct PagedAttention",
        "kind": "standard",
        "backend": "paged_v0",
        "attention_mode": "direct_cuda",
        "target_kv": "base_bf16",
        "h2o_ratio": None,
    },
    {
        "id": "D",
        "name": "+ V-FP8 target",
        "kind": "standard",
        "backend": "paged_v0",
        "attention_mode": "direct_cuda",
        "target_kv": "v_fp8",
        "h2o_ratio": None,
    },
    {
        "id": "E",
        "name": "BF16 + H2O 50%",
        "kind": "h2o",
        "backend": "paged_v0",
        "attention_mode": "direct_cuda",
        "target_kv": "base_bf16",
        "h2o_ratio": 0.50,
    },
    {
        "id": "F",
        "name": "V-FP8 + H2O 50%",
        "kind": "h2o",
        "backend": "paged_v0",
        "attention_mode": "direct_cuda",
        "target_kv": "v_fp8",
        "h2o_ratio": 0.50,
    },
    {
        "id": "G",
        "name": "V-FP8 + H2O 25% (stress)",
        "kind": "h2o",
        "backend": "paged_v0",
        "attention_mode": "direct_cuda",
        "target_kv": "v_fp8",
        "h2o_ratio": 0.25,
    },
)


def _prompt(tok, text: str, context: int) -> torch.Tensor:
    value = text
    while True:
        ids = tok(value, return_tensors="pt").input_ids
        if ids.shape[1] >= context:
            return ids[:, :context].cuda()
        value += text


def _as_int_list(values) -> list[int]:
    return [int(v) for v in values]


def _generated(result: dict, prompt_len: int) -> list[int]:
    return _as_int_list(result["ids"][prompt_len:])


def _agreement(
    baseline: list[int],
    candidate: list[int],
) -> tuple[float, int, bool, int | None]:
    n = min(len(baseline), len(candidate))
    if n == 0:
        exact = len(baseline) == len(candidate)
        return 1.0, 0, exact, None if exact else 0

    agree = sum(
        int(baseline[i] == candidate[i])
        for i in range(n)
    ) / n

    prefix = 0
    while prefix < n and baseline[prefix] == candidate[prefix]:
        prefix += 1

    if prefix < n:
        first_diff = prefix
    elif len(baseline) != len(candidate):
        first_diff = n
    else:
        first_diff = None

    exact = first_diff is None
    return agree, prefix, exact, first_diff


def _layer_payload_bytes(layer: dict) -> int:
    if paged_cache.is_paged_layer(layer):
        return paged_cache.payload_bytes(layer)

    total = 0
    for key in ("k", "v", "v_scale"):
        value = layer.get(key)
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return int(total)


def _cache_payload_bytes(cache) -> int:
    return sum(_layer_payload_bytes(layer) for layer in cache)


def _cache_resident(cache) -> int:
    return int(spec_cache.cache_length(cache))


def _configure_standard(
    target,
    draft,
    *,
    backend: str,
    attention_mode: str,
    target_kv: str,
    page_size: int,
) -> None:
    target.set_kv_format(
        "v_fp8" if target_kv == "v_fp8" else "fp16")
    target.set_cache_backend(
        backend,
        page_size=page_size,
    )
    target.set_paged_attention_mode(attention_mode)
    target.set_h2o_score_capture(False)

    draft.set_kv_format("fp16")
    draft.set_cache_backend(
        backend,
        page_size=page_size,
    )
    draft.set_paged_attention_mode(attention_mode)
    draft.set_h2o_score_capture(False)


def _run_standard(
    target,
    draft,
    ids,
    cfg: dict,
    *,
    steps: int,
    page_size: int,
):
    _configure_standard(
        target,
        draft,
        backend=cfg["backend"],
        attention_mode=cfg["attention_mode"],
        target_kv=cfg["target_kv"],
        page_size=page_size,
    )

    return spec_dec.spec_decode_greedy(
        target,
        draft,
        ids,
        steps,
        k=4,
        eos_id=None,
        target_kv=cfg["target_kv"],
        draft_kv="base_bf16",
        return_caches=True,
    )


def _run_h2o(
    target,
    draft,
    ids,
    cfg: dict,
    *,
    context: int,
    steps: int,
    page_size: int,
):
    budget = max(
        2,
        int(round(context * float(cfg["h2o_ratio"]))),
    )

    return spec_decode_greedy_h2o(
        target,
        draft,
        ids,
        steps,
        hh_size=budget // 2,
        recent_size=budget - budget // 2,
        k=4,
        eos_id=None,
        page_size=page_size,
        attention_mode=cfg["attention_mode"],
        target_kv=cfg["target_kv"],
        draft_kv="base_bf16",
        return_caches=True,
    )


def _stats_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(type(value))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:
        return "nan"
    return f"{x:.{digits}f}"


def _write_markdown(
    path: Path,
    rows: list[dict],
    summary_rows: list[dict],
    *,
    contexts: list[int],
    steps: int,
    page_size: int,
) -> None:
    by_config = {
        row["config_id"]: row
        for row in summary_rows
    }

    lines = [
        "# Final KV Optimization Ablation",
        "",
        "## Setup",
        "",
        f"- Target: `{MODEL_TARGET}`",
        f"- Draft: `{MODEL_SMALL}`",
        f"- Contexts: `{contexts}`",
        f"- Prompt families: `{list(PROMPTS)}`",
        f"- Generated tokens per case: `{steps}`",
        f"- Page size: `{page_size}`",
        "- SpecDec `k=4`",
        "- Quality metrics are relative to configuration A on the exact same input.",
        "- Payload is final allocated target+draft KV payload.",
        "- H2O latency is intentionally excluded: current H2O score capture and compaction are reference implementations.",
        "",
        "## Overall quality / payload summary",
        "",
        "| ID | Configuration | Exact rate | Token agreement | Common prefix | Payload ratio vs A | Mean resident/seen |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for cfg in CONFIGS:
        s = by_config[cfg["id"]]
        lines.append(
            "| {id} | {name} | {exact:.3f} | {agree:.3f} | "
            "{prefix:.2f}/{steps} | {payload:.3f} | {resident:.3f} |".format(
                id=cfg["id"],
                name=cfg["name"],
                exact=s["exact_match_rate"],
                agree=s["mean_token_agreement"],
                prefix=s["mean_common_prefix"],
                steps=steps,
                payload=s["mean_payload_ratio_vs_A"],
                resident=s["mean_resident_ratio"],
            )
        )

    lines += [
        "",
        "## Memory by context",
        "",
        "| Config | Context | Mean payload MiB | Ratio vs A | Target resident | Draft resident | Seen tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["config_id"], row["context"])].append(row)

    for cfg in CONFIGS:
        for context in contexts:
            group = grouped[(cfg["id"], context)]
            lines.append(
                "| {id} | {ctx} | {mib:.3f} | {ratio:.3f} | "
                "{tr:.1f} | {dr:.1f} | {seen:.1f} |".format(
                    id=cfg["id"],
                    ctx=context,
                    mib=mean(float(r["payload_mib"]) for r in group),
                    ratio=mean(float(r["payload_ratio_vs_A"]) for r in group),
                    tr=mean(float(r["target_resident"]) for r in group),
                    dr=mean(float(r["draft_resident"]) for r in group),
                    seen=mean(float(r["seen_tokens"]) for r in group),
                )
            )

    lines += [
        "",
        "## SpecDec behavior",
        "",
        "| ID | Mean proposed | Mean accepted | Acceptance/proposal | Mean mismatch rounds | Mean rollbacks |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for cfg in CONFIGS:
        group = [r for r in rows if r["config_id"] == cfg["id"]]
        proposed = mean(float(r["proposed"]) for r in group)
        accepted = mean(float(r["accepted"]) for r in group)
        lines.append(
            "| {id} | {proposed:.2f} | {accepted:.2f} | {rate:.3f} | "
            "{mismatch:.2f} | {rollback:.2f} |".format(
                id=cfg["id"],
                proposed=proposed,
                accepted=accepted,
                rate=_safe_div(accepted, proposed),
                mismatch=mean(float(r["mismatch_rounds"]) for r in group),
                rollback=mean(float(r["rollbacks"]) for r in group),
            )
        )

    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- A→B evaluates paged storage semantics/allocated payload, not a speed claim.",
        "- Direct PagedAttention latency should be supported by the dedicated repeated v1d benchmark (`materialized_cuda` vs `direct_cuda`), not inferred from this table.",
        "- D isolates the established V-FP8 target cache on these inputs.",
        "- E/F use the practical 50% H2O operating point.",
        "- G is an aggressive 25% stress setting; output changes are expected and must be reported rather than treated as a correctness failure.",
        "- V-FP8 × H2O interaction evidence belongs to the separate paired v3a interaction audit.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="128,256")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--raw-csv",
        default="kv_final_ablation_raw.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="kv_final_ablation_summary.csv",
    )
    parser.add_argument(
        "--report",
        default="kv_final_ablation_report.md",
    )
    parser.add_argument(
        "--manifest",
        default="kv_final_ablation_manifest.json",
    )
    args = parser.parse_args()

    contexts = [
        int(x) for x in args.contexts.split(",") if x
    ]
    if not contexts:
        raise SystemExit("contexts must be non-empty")
    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print("Final KV optimization ablation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print(
        f"contexts={contexts} steps={args.steps} "
        f"page_size={args.page_size}"
    )
    print("configs:", ", ".join(
        f"{c['id']}={c['name']}" for c in CONFIGS))

    torch.manual_seed(0)

    print(f"[load] target {MODEL_TARGET}")
    target, tok = load_ours(MODEL_TARGET)
    print(f"[load] draft {MODEL_SMALL}")
    draft, _ = load_ours(MODEL_SMALL)

    rows: list[dict] = []

    for family, text in PROMPTS.items():
        for context in contexts:
            ids = _prompt(tok, text, context)
            prompt_len = int(ids.shape[1])

            # Run A first because every other configuration is compared with
            # this exact-input reference.
            cfg_a = CONFIGS[0]
            baseline = _run_standard(
                target,
                draft,
                ids,
                cfg_a,
                steps=args.steps,
                page_size=args.page_size,
            )
            baseline_gen = _generated(
                baseline,
                prompt_len,
            )
            baseline_payload = (
                _cache_payload_bytes(
                    baseline["target_cache"])
                + _cache_payload_bytes(
                    baseline["draft_cache"])
            )

            configs_to_run = [
                (cfg_a, baseline)
            ]

            for cfg in CONFIGS[1:]:
                if cfg["kind"] == "standard":
                    result = _run_standard(
                        target,
                        draft,
                        ids,
                        cfg,
                        steps=args.steps,
                        page_size=args.page_size,
                    )
                else:
                    result = _run_h2o(
                        target,
                        draft,
                        ids,
                        cfg,
                        context=context,
                        steps=args.steps,
                        page_size=args.page_size,
                    )
                configs_to_run.append((cfg, result))

            for cfg, result in configs_to_run:
                candidate = _generated(
                    result,
                    prompt_len,
                )
                agree, prefix, exact, first_diff = _agreement(
                    baseline_gen,
                    candidate,
                )

                target_cache = result["target_cache"]
                draft_cache = result["draft_cache"]

                payload = (
                    _cache_payload_bytes(target_cache)
                    + _cache_payload_bytes(draft_cache)
                )

                target_resident = _cache_resident(
                    target_cache)
                draft_resident = _cache_resident(
                    draft_cache)

                stats = _stats_dict(result["stats"])
                seen = int(
                    result.get(
                        "seen_tokens",
                        prompt_len + len(candidate),
                    )
                )

                h2o_stats = result.get("h2o_stats")
                if h2o_stats is not None:
                    hs = _stats_dict(h2o_stats)
                    prune_rounds = int(
                        hs["prune_rounds"])
                    discarded_rows = int(
                        hs["discarded_proposal_rows"]
                        + hs["discarded_verify_rows"])
                else:
                    prune_rounds = 0
                    discarded_rows = 0

                row = {
                    "family": family,
                    "context": context,
                    "config_id": cfg["id"],
                    "config_name": cfg["name"],
                    "steps": args.steps,
                    "page_size": args.page_size,
                    "target_kv": cfg["target_kv"],
                    "backend": cfg["backend"],
                    "attention_mode":
                        cfg["attention_mode"],
                    "h2o_ratio":
                        "" if cfg["h2o_ratio"] is None
                        else cfg["h2o_ratio"],
                    "token_agreement_vs_A": agree,
                    "common_prefix_vs_A": prefix,
                    "exact_match_vs_A": int(exact),
                    "first_output_diff_vs_A":
                        "" if first_diff is None
                        else first_diff,
                    "payload_mib": payload / 2**20,
                    "payload_ratio_vs_A":
                        payload / baseline_payload,
                    "target_resident":
                        target_resident,
                    "draft_resident":
                        draft_resident,
                    "seen_tokens": seen,
                    "resident_ratio":
                        (target_resident + draft_resident)
                        / (2 * seen),
                    "rounds":
                        int(stats["rounds"]),
                    "proposed":
                        int(stats["proposed"]),
                    "accepted":
                        int(stats["accepted"]),
                    "mismatch_rounds":
                        int(stats["mismatch_rounds"]),
                    "rollbacks":
                        int(stats["rollbacks"]),
                    "prune_rounds":
                        prune_rounds,
                    "discarded_speculative_rows":
                        discarded_rows,
                }
                rows.append(row)

                print(
                    f"[{family:<9}] ctx={context:<4} "
                    f"{cfg['id']} {cfg['name']:<28} "
                    f"exact={int(exact)} "
                    f"agree={agree:.3f} "
                    f"prefix={prefix:<2}/{args.steps} "
                    f"payload={payload/2**20:.2f} MiB "
                    f"ratio={payload/baseline_payload:.3f} "
                    f"resident={target_resident}/{draft_resident}"
                )

    raw_path = Path(args.raw_csv).resolve()
    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for cfg in CONFIGS:
        group = [
            r for r in rows
            if r["config_id"] == cfg["id"]
        ]
        summary_rows.append({
            "config_id": cfg["id"],
            "config_name": cfg["name"],
            "n_cases": len(group),
            "exact_match_rate": mean(
                float(r["exact_match_vs_A"])
                for r in group
            ),
            "mean_token_agreement": mean(
                float(r["token_agreement_vs_A"])
                for r in group
            ),
            "mean_common_prefix": mean(
                float(r["common_prefix_vs_A"])
                for r in group
            ),
            "mean_payload_ratio_vs_A": mean(
                float(r["payload_ratio_vs_A"])
                for r in group
            ),
            "mean_resident_ratio": mean(
                float(r["resident_ratio"])
                for r in group
            ),
            "mean_proposed": mean(
                float(r["proposed"])
                for r in group
            ),
            "mean_accepted": mean(
                float(r["accepted"])
                for r in group
            ),
            "mean_mismatch_rounds": mean(
                float(r["mismatch_rounds"])
                for r in group
            ),
            "mean_rollbacks": mean(
                float(r["rollbacks"])
                for r in group
            ),
            "mean_prune_rounds": mean(
                float(r["prune_rounds"])
                for r in group
            ),
        })

    summary_path = Path(args.summary_csv).resolve()
    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = Path(args.report).resolve()
    _write_markdown(
        report_path,
        rows,
        summary_rows,
        contexts=contexts,
        steps=args.steps,
        page_size=args.page_size,
    )

    manifest = {
        "purpose": (
            "Final quality/memory/SpecDec-behavior ablation. "
            "H2O latency intentionally excluded."
        ),
        "target_model": MODEL_TARGET,
        "draft_model": MODEL_SMALL,
        "contexts": contexts,
        "prompt_families": list(PROMPTS),
        "steps": args.steps,
        "page_size": args.page_size,
        "specdec_k": 4,
        "configs": list(CONFIGS),
        "claim_boundaries": [
            "Do not infer H2O speedup from this experiment.",
            "Use the dedicated repeated v1d paired benchmark for PagedAttention latency.",
            "Treat H2O 25% as an aggressive stress regime, not a lossless operating point.",
            "Use the separate v3a paired audit for V-FP8 × H2O interaction claims.",
        ],
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== final summary ===")
    for row in summary_rows:
        print(
            f"{row['config_id']} {row['config_name']:<28} "
            f"exact={row['exact_match_rate']:.3f} "
            f"agree={row['mean_token_agreement']:.3f} "
            f"prefix={row['mean_common_prefix']:.2f}/{args.steps} "
            f"payload_ratio={row['mean_payload_ratio_vs_A']:.3f} "
            f"resident_ratio={row['mean_resident_ratio']:.3f}"
        )

    print("\nRaw CSV:", raw_path)
    print("Summary CSV:", summary_path)
    print("Markdown report:", report_path)
    print("Manifest:", manifest_path)
    print("kv_final_ablation: PASS")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
