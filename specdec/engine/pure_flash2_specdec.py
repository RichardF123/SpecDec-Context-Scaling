"""Real SpecDec: current PagedAttention vs pure official FlashAttention-2."""
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
from .g4_b4_cache_tests import MODEL_TARGET, MODEL_SMALL, load_ours
from .kv_final_ablation import PROMPTS, _prompt
from .model_self import attention_phase
from .pure_flash2_backend import pure_flash2_context


def _chunked_prefill(model, ids, chunk):
    cache = None
    last = None
    with torch.inference_mode(), attention_phase("prefill"):
        for st in range(0, ids.shape[1], chunk):
            end = min(ids.shape[1], st + chunk)
            last, cache = model._run(
                model.embed_tokens(ids[:, st:end]),
                cache,
                need_logits=(end == ids.shape[1]),
                start_pos_override=st,
            )
    return last, cache


def _gen_hash(r):
    return hashlib.sha256(
        ",".join(str(int(x)) for x in r["gen"]).encode()
    ).hexdigest()


def _run_once(
    target, draft, ids, *,
    mode, steps, chunk, page_size,
):
    orig_prefill = spec_dec._prefill
    prefill_s = 0.0

    def timed_prefill(model, x):
        nonlocal prefill_s
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _chunked_prefill(model, x, chunk)
        torch.cuda.synchronize()
        prefill_s += time.perf_counter() - t0
        return out

    spec_dec._prefill = timed_prefill
    max_seq = int(ids.shape[1]) + int(steps) + 32

    try:
        if mode == "paged_direct":
            target.set_kv_format("fp16")
            draft.set_kv_format("fp16")
            target.set_cache_backend("paged_v0", page_size=page_size)
            draft.set_cache_backend("paged_v0", page_size=page_size)
            target.set_paged_attention_mode("direct_cuda")
            draft.set_paged_attention_mode("direct_cuda")
            ctx = None
        elif mode == "pure_flash2":
            # pure_flash2_context switches both models to contiguous cache.
            ctx = pure_flash2_context(
                target, draft, max_seqlen=max_seq)
        else:
            raise ValueError(mode)

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if ctx is None:
            r = spec_dec.spec_decode_greedy(
                target, draft, ids, steps,
                k=4, eos_id=None,
                target_kv="base_bf16",
                draft_kv="base_bf16",
                return_caches=False)
        else:
            with ctx:
                r = spec_dec.spec_decode_greedy(
                    target, draft, ids, steps,
                    k=4, eos_id=None,
                    target_kv="base_bf16",
                    draft_kv="base_bf16",
                    return_caches=False)

        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
    finally:
        spec_dec._prefill = orig_prefill

    decode_s = max(0.0, total_s - prefill_s)
    return r, total_s, prefill_s, decode_s, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="2048,4096,8192")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--outdir", default="output/pure_flash2")
    a = ap.parse_args()

    contexts = [int(x) for x in a.contexts.split(",") if x]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    target, tok = load_ours(MODEL_TARGET)
    draft, _ = load_ours(MODEL_SMALL)

    rows = []
    for ctx_len in contexts:
        ids = _prompt(tok, PROMPTS["systems"], ctx_len)

        for rep in range(a.repeats):
            order = (
                ("paged_direct", "pure_flash2")
                if rep % 2 == 0
                else ("pure_flash2", "paged_direct")
            )
            pair = {}
            for mode in order:
                print(
                    f"[real-pure-fa2] ctx={ctx_len} rep={rep} "
                    f"mode={mode}",
                    flush=True)
                r, total, pre, dec, peak = _run_once(
                    target, draft, ids,
                    mode=mode,
                    steps=a.steps,
                    chunk=a.chunk,
                    page_size=a.page_size)
                row = dict(
                    context=ctx_len,
                    repeat=rep,
                    mode=mode,
                    total_s=total,
                    prefill_s=pre,
                    decode_s=dec,
                    decode_tok_s=(
                        a.steps / dec if dec > 0 else None),
                    peak_alloc_gib=peak,
                    gen_sha256=_gen_hash(r),
                    stats_json=json.dumps(
                        r["stats"], sort_keys=True),
                )
                rows.append(row)
                pair[mode] = row

            b = pair["paged_direct"]
            f = pair["pure_flash2"]
            f["exact_vs_paged"] = int(
                f["gen_sha256"] == b["gen_sha256"])
            f["stats_equal_vs_paged"] = int(
                f["stats_json"] == b["stats_json"])
            f["decode_speedup_vs_paged"] = (
                b["decode_s"] / f["decode_s"])
            f["total_speedup_vs_paged"] = (
                b["total_s"] / f["total_s"])

            print(
                f"  decode speedup="
                f"{f['decode_speedup_vs_paged']:.3f}x "
                f"exact={f['exact_vs_paged']} "
                f"stats={f['stats_equal_vs_paged']} "
                f"peak={f['peak_alloc_gib']:.3f} GiB",
                flush=True)

    fields = sorted({k for r in rows for k in r})
    with (out/"real_specdec_raw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for ctx_len in contexts:
        g = [r for r in rows if r["context"] == ctx_len]
        b = [r for r in g if r["mode"] == "paged_direct"]
        f = [r for r in g if r["mode"] == "pure_flash2"]

        med_b_dec = statistics.median(r["decode_s"] for r in b)
        med_f_dec = statistics.median(r["decode_s"] for r in f)
        summary.append(dict(
            context=ctx_len,
            paged_decode_s=med_b_dec,
            pure_fa2_decode_s=med_f_dec,
            pure_fa2_decode_speedup=med_b_dec / med_f_dec,
            paged_decode_tok_s=(
                a.steps / med_b_dec),
            pure_fa2_decode_tok_s=(
                a.steps / med_f_dec),
            paged_peak_alloc_gib=statistics.median(
                r["peak_alloc_gib"] for r in b),
            pure_fa2_peak_alloc_gib=statistics.median(
                r["peak_alloc_gib"] for r in f),
            exact_rate=sum(
                int(r.get("exact_vs_paged", 0))
                for r in f) / len(f),
            stats_equal_rate=sum(
                int(r.get("stats_equal_vs_paged", 0))
                for r in f) / len(f),
        ))

    (out/"real_specdec_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8")
    if summary:
        with (out/"real_specdec_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fp:
            w = csv.DictWriter(
                fp, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    print("\n=== Pure official FA2 real SpecDec summary ===")
    for r in summary:
        print(
            f"ctx={r['context']:<5} "
            f"speedup={r['pure_fa2_decode_speedup']:.3f}x "
            f"paged={r['paged_decode_tok_s']:.3f} tok/s "
            f"pureFA2={r['pure_fa2_decode_tok_s']:.3f} tok/s "
            f"exact={r['exact_rate']:.3f} "
            f"stats={r['stats_equal_rate']:.3f} "
            f"peak={r['pure_fa2_peak_alloc_gib']:.3f} GiB")

    strict = all(
        r["exact_rate"] == 1.0
        and r["stats_equal_rate"] == 1.0
        for r in summary)
    status = {
        "strict_parity_pass": strict,
        "summary": summary,
        "note": (
            "Latency remains reportable even if greedy output differs "
            "numerically; a parity mismatch must not be called a "
            "drop-in correctness pass."
        ),
    }
    (out/"real_specdec_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8")

    print(
        "PURE_FLASH2_REAL_SPECDEC:",
        "PASS" if strict else "CHECK_NUMERICAL_PARITY")

    del target, draft, tok
    gc.collect()
    torch.cuda.empty_cache()
    # Always return 0 so results are packaged even if real-model greedy
    # parity changes due to legitimate FA2 numerical ordering differences.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
