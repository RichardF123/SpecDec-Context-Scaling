from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .kv_final_ablation import PROMPTS, _prompt
from .model_self import (
    attention_phase,
    paged_attention_stats,
    reset_paged_attention_stats,
)


def _chunked_prefill(model, ids, chunk: int):
    cache = None
    last_logits = None
    n = int(ids.shape[1])

    with torch.inference_mode(), attention_phase("prefill"):
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            logits, cache = model._run(
                model.embed_tokens(ids[:, start:end]),
                cache,
                need_logits=(end == n),
                start_pos_override=start,
            )
            if end == n:
                last_logits = logits

    if last_logits is None or cache is None:
        raise RuntimeError("chunked prefill failed")
    return last_logits, cache


def _hash_gen(gen) -> str:
    return hashlib.sha256(
        ",".join(str(int(x)) for x in gen).encode("utf-8")
    ).hexdigest()


def _run_one(
    target,
    draft,
    ids,
    *,
    mode: str,
    target_kv: str,
    steps: int,
    chunk: int,
    page_size: int,
):
    target.set_cache_backend("paged_v0", page_size=page_size)
    draft.set_cache_backend("paged_v0", page_size=page_size)
    target.set_paged_attention_mode(mode)
    draft.set_paged_attention_mode(mode)
    target.set_h2o_score_capture(False)
    draft.set_h2o_score_capture(False)

    original = spec_dec._prefill
    prefill_s = 0.0

    def timed_prefill(model, prompt_ids):
        nonlocal prefill_s
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _chunked_prefill(model, prompt_ids, chunk)
        torch.cuda.synchronize()
        prefill_s += time.perf_counter() - t0
        return out

    spec_dec._prefill = timed_prefill
    try:
        reset_paged_attention_stats()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = spec_dec.spec_decode_greedy(
            target,
            draft,
            ids,
            steps,
            k=4,
            eos_id=None,
            target_kv=target_kv,
            draft_kv="base_bf16",
            return_caches=False,
        )

        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        route = paged_attention_stats()
    finally:
        spec_dec._prefill = original

    decode_s = max(0.0, total_s - prefill_s)

    return {
        "mode": mode,
        "target_kv": target_kv,
        "total_s": total_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": steps / decode_s if decode_s > 0 else None,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "gen_sha256": _hash_gen(result["gen"]),
        "stats_json": json.dumps(result["stats"], sort_keys=True),
        "route_json": json.dumps(route, sort_keys=True),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="2048,4096,8192")
    ap.add_argument("--target-kvs", default="base_bf16,v_fp8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--outdir", default="output/phase13_flash")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    kvs = [x for x in args.target_kvs.split(",") if x]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target, tok = load_ours(MODEL_TARGET)
    draft, _ = load_ours(MODEL_SMALL)

    rows = []

    for kv in kvs:
        for ctx in contexts:
            ids = _prompt(tok, PROMPTS["systems"], ctx)

            for rep in range(args.repeats):
                order = (
                    ("direct_cuda", "flash_paged_cuda")
                    if rep % 2 == 0
                    else ("flash_paged_cuda", "direct_cuda")
                )
                pair = {}

                for mode in order:
                    print(
                        f"[specdec-flash] kv={kv} ctx={ctx} "
                        f"rep={rep} mode={mode}",
                        flush=True,
                    )

                    row = _run_one(
                        target,
                        draft,
                        ids,
                        mode=mode,
                        target_kv=kv,
                        steps=args.steps,
                        chunk=args.chunk,
                        page_size=args.page_size,
                    )
                    row.update({
                        "context": ctx,
                        "repeat": rep,
                    })
                    rows.append(row)
                    pair[mode] = row

                direct = pair["direct_cuda"]
                flash = pair["flash_paged_cuda"]

                flash["exact_vs_direct"] = int(
                    flash["gen_sha256"] == direct["gen_sha256"])
                flash["stats_equal_vs_direct"] = int(
                    flash["stats_json"] == direct["stats_json"])
                flash["decode_speedup_vs_direct_x"] = (
                    direct["decode_s"] / flash["decode_s"])
                flash["total_speedup_vs_direct_x"] = (
                    direct["total_s"] / flash["total_s"])

                print(
                    f"  decode speedup="
                    f"{flash['decode_speedup_vs_direct_x']:.3f}x "
                    f"exact={flash['exact_vs_direct']} "
                    f"stats={flash['stats_equal_vs_direct']} "
                    f"peak={flash['peak_alloc_gib']:.3f} GiB",
                    flush=True,
                )

    fields = sorted({k for r in rows for k in r})
    with (outdir / "phase13_flash_specdec_raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for kv in kvs:
        for ctx in contexts:
            direct = [
                r for r in rows
                if r["target_kv"] == kv
                and r["context"] == ctx
                and r["mode"] == "direct_cuda"
            ]
            flash = [
                r for r in rows
                if r["target_kv"] == kv
                and r["context"] == ctx
                and r["mode"] == "flash_paged_cuda"
            ]

            direct_dec = statistics.median(
                r["decode_s"] for r in direct)
            flash_dec = statistics.median(
                r["decode_s"] for r in flash)

            summary.append({
                "target_kv": kv,
                "context": ctx,
                "direct_decode_s": direct_dec,
                "flash_decode_s": flash_dec,
                "flash_decode_speedup_vs_direct_x":
                    direct_dec / flash_dec,
                "direct_decode_tok_s":
                    args.steps / direct_dec,
                "flash_decode_tok_s":
                    args.steps / flash_dec,
                "flash_peak_alloc_gib":
                    statistics.median(
                        r["peak_alloc_gib"] for r in flash),
                "exact_rate":
                    sum(int(r.get("exact_vs_direct", 0))
                        for r in flash) / len(flash),
                "stats_equal_rate":
                    sum(int(r.get("stats_equal_vs_direct", 0))
                        for r in flash) / len(flash),
            })

    sf = list(summary[0].keys())
    with (outdir / "phase13_flash_specdec_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=sf)
        w.writeheader()
        w.writerows(summary)

    (outdir / "phase13_flash_specdec_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Phase13 real SpecDec summary ===")
    for r in summary:
        print(
            f"ctx={r['context']:<5} kv={r['target_kv']:<10} "
            f"speedup={r['flash_decode_speedup_vs_direct_x']:.3f}x "
            f"direct={r['direct_decode_tok_s']:.3f} tok/s "
            f"flash={r['flash_decode_tok_s']:.3f} tok/s "
            f"exact={r['exact_rate']:.3f}"
        )

    ok = all(
        r["exact_rate"] == 1.0
        and r["stats_equal_rate"] == 1.0
        for r in summary
    )

    print(
        "\nPHASE13_FLASH_SPECDEC:",
        "PASS" if ok else "CHECK",
    )

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
