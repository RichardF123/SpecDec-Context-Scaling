"""Operational latency benchmark for the current H2O + SpecDec path.

Measures the CURRENT reference implementation, including score capture and
committed-cache compaction. This is not a fused/optimized H2O kernel benchmark.

Each (context, target KV) runs in a fresh worker process. Models are loaded once
per worker; standard and H2O runs then alternate order across repeats.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch

from . import kv_final_ablation as K
from . import spec_dec
from . import spec_dec_h2o as H
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours


def _hash_gen(result):
    gen = result.get("gen", [])
    return hashlib.sha256(",".join(str(int(x)) for x in gen).encode()).hexdigest()


def _stats_sig(result):
    return json.dumps(result.get("stats", {}), sort_keys=True, default=str)


def _timed_call(fn, prefill_module, *args, **kwargs):
    original = prefill_module._prefill
    prefill_s = 0.0

    def timed_prefill(*a, **kw):
        nonlocal prefill_s
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original(*a, **kw)
        torch.cuda.synchronize()
        prefill_s += time.perf_counter() - t0
        return out

    prefill_module._prefill = timed_prefill
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    finally:
        prefill_module._prefill = original

    decode_s = max(0.0, total_s - prefill_s)
    return result, total_s, prefill_s, decode_s, peak_alloc, peak_reserved


def _run_standard(target, draft, ids, target_kv, steps, page_size):
    cfg = dict(
        backend="paged_v0",
        attention_mode="direct_cuda",
        target_kv=target_kv,
    )
    return _timed_call(
        K._run_standard, spec_dec,
        target, draft, ids, cfg,
        steps=steps, page_size=page_size,
    )


def _run_h2o(target, draft, ids, target_kv, ratio, context, steps, page_size):
    cfg = dict(
        backend="paged_v0",
        attention_mode="direct_cuda",
        target_kv=target_kv,
        h2o_ratio=ratio,
    )
    return _timed_call(
        K._run_h2o, H,
        target, draft, ids, cfg,
        context=context, steps=steps, page_size=page_size,
    )


def _measure(target, draft, ids, *, kind, target_kv, ratio, context, steps, page_size):
    if kind == "standard":
        result, total_s, prefill_s, decode_s, pa, pr = _run_standard(
            target, draft, ids, target_kv, steps, page_size)
    else:
        result, total_s, prefill_s, decode_s, pa, pr = _run_h2o(
            target, draft, ids, target_kv, ratio, context, steps, page_size)

    row = dict(
        kind=kind,
        ratio="" if kind == "standard" else ratio,
        status="PASS",
        total_s=total_s,
        prefill_s=prefill_s,
        decode_s=decode_s,
        decode_tok_s=steps / decode_s if decode_s > 0 else None,
        peak_alloc_gib=pa,
        peak_reserved_gib=pr,
        gen_sha256=_hash_gen(result),
        stats_signature=_stats_sig(result),
    )
    st = result.get("stats", {})
    for k in ("rounds", "proposed", "accepted", "mismatch_rounds", "rollbacks"):
        row[k] = st.get(k)

    if kind == "h2o":
        hs = result["h2o_stats"]
        row.update(
            prune_rounds=hs.prune_rounds,
            seen_tokens=hs.seen_tokens,
            target_resident=hs.final_target_resident,
            draft_resident=hs.final_draft_resident,
            resident_ratio=(
                (hs.final_target_resident + hs.final_draft_resident)
                / (2 * hs.seen_tokens)
            ),
        )
    else:
        row.update(
            prune_rounds=0,
            seen_tokens=context + steps,
            target_resident=context + steps,
            draft_resident=context + steps,
            resident_ratio=1.0,
        )
    return row


def worker(args):
    out = Path(args.worker_out)
    rows = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        torch.manual_seed(0)
        print(f"[worker] ctx={args.context} kv={args.target_kv}", flush=True)
        target, tok = load_ours(MODEL_TARGET)
        draft, _ = load_ours(MODEL_SMALL)

        families = [x for x in args.prompt_families.split(",") if x]
        ratios = [float(x) for x in args.ratios.split(",") if x]

        for family in families:
            if family not in K.PROMPTS:
                raise ValueError(f"unknown prompt family {family}")
            ids = K._prompt(tok, K.PROMPTS[family], args.context)

            for ratio in ratios:
                for rep in range(args.repeats):
                    order = ("standard", "h2o") if rep % 2 == 0 else ("h2o", "standard")
                    pair = {}
                    for kind in order:
                        print(
                            f"[run] ctx={args.context} kv={args.target_kv} "
                            f"family={family} ratio={ratio:.2f} rep={rep} kind={kind}",
                            flush=True,
                        )
                        try:
                            r = _measure(
                                target, draft, ids,
                                kind=kind, target_kv=args.target_kv,
                                ratio=ratio, context=args.context,
                                steps=args.steps, page_size=args.page_size,
                            )
                        except torch.cuda.OutOfMemoryError as e:
                            r = dict(kind=kind, ratio=ratio if kind == "h2o" else "",
                                     status="OOM", error=str(e))
                            torch.cuda.empty_cache()
                        except Exception as e:
                            r = dict(kind=kind, ratio=ratio if kind == "h2o" else "",
                                     status="ERROR",
                                     error=f"{type(e).__name__}: {e}",
                                     traceback=traceback.format_exc())

                        r.update(
                            context=args.context,
                            target_kv=args.target_kv,
                            family=family,
                            repeat=rep,
                        )
                        rows.append(r)
                        pair[kind] = r

                    b, h = pair.get("standard"), pair.get("h2o")
                    if b and h and b.get("status") == h.get("status") == "PASS":
                        h["exact_vs_standard"] = int(b["gen_sha256"] == h["gen_sha256"])
                        h["stats_equal_vs_standard"] = int(
                            b["stats_signature"] == h["stats_signature"])
                        h["decode_overhead_x"] = h["decode_s"] / b["decode_s"]
                        h["decode_speed_ratio"] = b["decode_s"] / h["decode_s"]
                        h["total_overhead_x"] = h["total_s"] / b["total_s"]
                        print(
                            f"  pair exact={h['exact_vs_standard']} "
                            f"stats_equal={h['stats_equal_vs_standard']} "
                            f"H2O/std_decode={h['decode_overhead_x']:.3f}x "
                            f"resident={h['resident_ratio']:.3f}",
                            flush=True,
                        )

        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        del target, draft, tok
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except Exception as e:
        out.write_text(json.dumps({
            "worker_status": "ERROR",
            "context": args.context,
            "target_kv": args.target_kv,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "rows": rows,
        }, indent=2), encoding="utf-8")
        traceback.print_exc()
        return 2


def _median(vals):
    xs = [float(v) for v in vals if v not in (None, "")]
    return statistics.median(xs) if xs else None


def orchestrator(args):
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    kvs = [x for x in args.target_kvs.split(",") if x]
    all_rows = []

    for ctx in contexts:
        for kv in kvs:
            jf = outdir / f"worker_ctx{ctx}_{kv}.json"
            cmd = [
                sys.executable, "-m", "specdec.engine.h2o_latency_final",
                "--worker",
                "--context", str(ctx),
                "--target-kv", kv,
                "--steps", str(args.steps),
                "--page-size", str(args.page_size),
                "--repeats", str(args.repeats),
                "--ratios", args.ratios,
                "--prompt-families", args.prompt_families,
                "--worker-out", str(jf),
            ]
            print("\n[launch]", " ".join(cmd), flush=True)
            p = subprocess.run(cmd, check=False)
            if not jf.exists():
                all_rows.append(dict(
                    context=ctx, target_kv=kv, status="PROCESS_FAIL",
                    error=f"child returncode={p.returncode}"))
                continue
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_rows.extend(data)
            else:
                all_rows.extend(data.get("rows", []))
                all_rows.append(dict(
                    context=ctx, target_kv=kv,
                    status=data.get("worker_status", "ERROR"),
                    error=data.get("error", "")))

    fields = sorted({k for r in all_rows for k in r if k != "traceback"})
    raw_csv = outdir / "h2o_latency_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    hrows = [
        r for r in all_rows
        if r.get("kind") == "h2o"
        and r.get("status") == "PASS"
        and r.get("decode_overhead_x") is not None
    ]
    ratios = [float(x) for x in args.ratios.split(",") if x]
    summary = []

    for ctx in contexts:
        for kv in kvs:
            for ratio in ratios:
                g = [
                    r for r in hrows
                    if int(r["context"]) == ctx
                    and r["target_kv"] == kv
                    and abs(float(r["ratio"]) - ratio) < 1e-9
                ]
                if not g:
                    summary.append(dict(
                        context=ctx, target_kv=kv, ratio=ratio,
                        n_pairs=0, status="NO_VALID_PAIRS"))
                    continue
                summary.append(dict(
                    context=ctx,
                    target_kv=kv,
                    ratio=ratio,
                    n_pairs=len(g),
                    status="PASS",
                    median_h2o_decode_overhead_x=
                        _median([r["decode_overhead_x"] for r in g]),
                    median_h2o_decode_speed_ratio=
                        _median([r["decode_speed_ratio"] for r in g]),
                    median_h2o_total_overhead_x=
                        _median([r["total_overhead_x"] for r in g]),
                    median_h2o_decode_tok_s=
                        _median([r["decode_tok_s"] for r in g]),
                    median_h2o_peak_alloc_gib=
                        _median([r["peak_alloc_gib"] for r in g]),
                    median_h2o_resident_ratio=
                        _median([r["resident_ratio"] for r in g]),
                    exact_output_rate=
                        sum(int(r.get("exact_vs_standard", 0)) for r in g) / len(g),
                    stats_equal_rate=
                        sum(int(r.get("stats_equal_vs_standard", 0)) for r in g) / len(g),
                ))

    summary_csv = outdir / "h2o_latency_summary.csv"
    sf = sorted({k for r in summary for k in r})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sf)
        w.writeheader()
        w.writerows(summary)

    report = outdir / "h2o_latency_report.md"
    lines = [
        "# H2O Operational Latency Benchmark", "",
        "Measures the current reference H2O score-capture + compaction path.",
        "",
        "`H2O/standard decode > 1` means H2O is slower.", "",
        "| Context | Target KV | Budget | Pairs | H2O/Std decode | Std/H2O speed | "
        "H2O tok/s | Peak alloc GiB | Resident/seen | Exact output | Stats equal |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["status"] != "PASS":
            lines.append(
                f"| {r['context']} | {r['target_kv']} | {r['ratio']:.2f} | 0 | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {r['context']} | {r['target_kv']} | {r['ratio']:.2f} | "
                f"{r['n_pairs']} | {r['median_h2o_decode_overhead_x']:.3f}x | "
                f"{r['median_h2o_decode_speed_ratio']:.3f}x | "
                f"{r['median_h2o_decode_tok_s']:.3f} | "
                f"{r['median_h2o_peak_alloc_gib']:.3f} | "
                f"{r['median_h2o_resident_ratio']:.3f} | "
                f"{r['exact_output_rate']:.3f} | {r['stats_equal_rate']:.3f} |")
    lines += [
        "", "## Boundary", "",
        "- This is an operational benchmark of the current Python/reference H2O path.",
        "- It is not a fused/optimized H2O kernel benchmark.",
        "- A slowdown here measures the present score-capture + cache-compaction implementation.",
        "- If outputs or SpecDec stats differ, latency ratios are operational rather than path-identical.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== H2O latency summary ===")
    for r in summary:
        if r["status"] == "PASS":
            print(
                f"ctx={r['context']:<5} kv={r['target_kv']:<10} ratio={r['ratio']:.2f} "
                f"pairs={r['n_pairs']:<2} H2O/std={r['median_h2o_decode_overhead_x']:.3f}x "
                f"speed={r['median_h2o_decode_speed_ratio']:.3f}x "
                f"resident={r['median_h2o_resident_ratio']:.3f}")
        else:
            print(
                f"ctx={r['context']:<5} kv={r['target_kv']:<10} ratio={r['ratio']:.2f} "
                "NO_VALID_PAIRS")
    print("\nRaw:", raw_csv)
    print("Summary:", summary_csv)
    print("Report:", report)
    print("\nh2o_latency_final: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--context", type=int)
    ap.add_argument("--target-kv", choices=("base_bf16", "v_fp8"))
    ap.add_argument("--worker-out", default="h2o_worker.json")

    ap.add_argument("--contexts", default="512,2048,4096")
    ap.add_argument("--target-kvs", default="base_bf16,v_fp8")
    ap.add_argument("--prompt-families", default="systems,reasoning,code")
    ap.add_argument("--ratios", default="0.5")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default="output/h2o_latency_final")
    args = ap.parse_args()

    if args.worker:
        if args.context is None or args.target_kv is None:
            raise SystemExit("--worker requires --context and --target-kv")
        return worker(args)
    return orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
