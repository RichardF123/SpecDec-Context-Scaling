"""Fresh-process 4K/8K validation for the final direct PagedAttention path."""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch

from . import spec_cache, spec_dec
from .g4_b4_cache_tests import MODEL_SMALL, MODEL_TARGET, load_ours
from .model_self import attention_phase, paged_attention_stats, reset_paged_attention_stats


def _make_ids(tok, context: int) -> torch.Tensor:
    text = (
        "This technical note evaluates long-context transformer inference, "
        "paged key value caching, speculative decoding, numerical parity, "
        "memory pressure, and reproducible systems measurements. "
    )
    base = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if int(base.numel()) == 0:
        raise RuntimeError("tokenizer produced no tokens")
    reps = (context + int(base.numel()) - 1) // int(base.numel())
    return base.repeat(reps)[:context].unsqueeze(0).contiguous().cuda()


def _configure(model, kv: str, mode: str, page_size: int) -> None:
    model.set_kv_format("v_fp8" if kv == "v_fp8" else "fp16")
    model.set_cache_backend("paged_v0", page_size=page_size)
    model.set_paged_attention_mode(mode)
    model.set_h2o_score_capture(False)


def _chunked_prefill(model, ids: torch.Tensor, chunk: int):
    cache = None
    last_logits = None
    n = int(ids.shape[1])
    with torch.inference_mode(), attention_phase("prefill"):
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            piece = ids[:, start:end]
            need_logits = end == n
            logits, cache = model._run(
                model.embed_tokens(piece), cache, need_logits=need_logits
            )
            if need_logits:
                last_logits = logits
    if last_logits is None or cache is None:
        raise RuntimeError("prefill failed")
    if spec_cache.cache_length(cache) != n:
        raise RuntimeError("prefill cache length mismatch")
    return last_logits, cache


def _spec_from_prefill(target, draft, tgt_lg, drf_lg, tgt_cache, drf_cache, steps: int, k: int):
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]
    gen = []
    st = dict(rounds=0, proposed=0, accepted=0, target_block_calls=0, draft_calls=0,
              mismatch_rounds=0, all_accept_rounds=0, rollbacks=0)

    while len(gen) < steps:
        k_eff = min(k, steps - len(gen))
        pre = spec_cache.cache_length(tgt_cache)

        cands, drf_cache, prop_lg = spec_dec._propose(
            draft, drf_cache, drf_lg, k_eff, None
        )
        kp = len(cands)
        st["draft_calls"] += kp
        if kp == 0:
            break

        rows, tgt_blk = spec_dec._verify(target, tgt_cache, cands)
        st["target_block_calls"] += 1
        st["proposed"] += kp

        greedy = [int(x.argmax()) for x in [saved_next] + [rows[j] for j in range(kp - 1)]]
        m = 0
        while m < kp and greedy[m] == cands[m]:
            m += 1

        if m < kp:
            replacement = greedy[m]
            emitted = cands[:m] + [replacement]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + m)
            saved_next, tgt_cache = spec_dec._step(target, replacement, tgt_cache)
            drf_cache = spec_cache.trim_cache(drf_cache, pre + m)
            drf_lg, drf_cache = spec_dec._step(draft, replacement, drf_cache)
            st["mismatch_rounds"] += 1
            st["rollbacks"] += 1
        else:
            emitted = list(cands)
            tgt_cache = tgt_blk
            drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
            saved_next = rows[kp - 1]
            drf_lg = prop_lg
            st["all_accept_rounds"] += 1

        gen += emitted
        st["accepted"] += m
        st["rounds"] += 1

        expected = pre + len(emitted)
        if spec_cache.cache_length(tgt_cache) != expected:
            raise RuntimeError("target cache invariant failed")
        if spec_cache.cache_length(drf_cache) != expected:
            raise RuntimeError("draft cache invariant failed")

    return dict(gen=gen, stats=st, target_cache=tgt_cache, draft_cache=drf_cache)


def _hash_ids(ids):
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def worker(args) -> int:
    row = {
        "context": args.context,
        "target_kv": args.target_kv,
        "mode": args.mode,
        "repeat": args.repeat,
        "status": "UNKNOWN",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        torch.manual_seed(0)
        torch.cuda.empty_cache()
        row["gpu"] = torch.cuda.get_device_name(0)
        row["torch"] = torch.__version__

        target, tok = load_ours(MODEL_TARGET)
        draft, _ = load_ours(MODEL_SMALL)

        _configure(target, args.target_kv, args.mode, args.page_size)
        _configure(draft, "base_bf16", args.mode, args.page_size)

        ids = _make_ids(tok, args.context)
        reset_paged_attention_stats()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        tp0 = time.perf_counter()
        tgt_lg, tgt_cache = _chunked_prefill(target, ids, args.chunk)
        drf_lg, drf_cache = _chunked_prefill(draft, ids, args.chunk)
        torch.cuda.synchronize()
        tp1 = time.perf_counter()

        td0 = time.perf_counter()
        out = _spec_from_prefill(
            target, draft, tgt_lg, drf_lg, tgt_cache, drf_cache,
            args.steps, args.k
        )
        torch.cuda.synchronize()
        td1 = time.perf_counter()

        route = paged_attention_stats()
        final_t = spec_cache.cache_length(out["target_cache"])
        final_d = spec_cache.cache_length(out["draft_cache"])
        expected = args.context + len(out["gen"])
        if final_t != expected or final_d != expected:
            raise RuntimeError("final cache length mismatch")

        if args.mode == "direct_cuda":
            direct = route["direct_by_phase"]
            fallback = route["fallback_by_phase"]
            if direct.get("propose", 0) <= 0 or direct.get("verify", 0) <= 0:
                raise RuntimeError("direct propose/verify route not exercised")
            if fallback.get("verify", 0) != 0:
                raise RuntimeError("verify unexpectedly materialized")

        row.update({
            "status": "PASS",
            "prefill_s": tp1 - tp0,
            "decode_s": td1 - td0,
            "total_s": td1 - t0,
            "decode_tok_s": len(out["gen"]) / (td1 - td0),
            "n_gen": len(out["gen"]),
            "gen_sha256": _hash_ids(out["gen"]),
            "stats": out["stats"],
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "direct_by_phase": route["direct_by_phase"],
            "fallback_by_phase": route["fallback_by_phase"],
        })

        del target, draft, tok, ids, tgt_lg, drf_lg, out
        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        row["status"] = "OOM"
        row["error"] = str(e)
        row["traceback"] = traceback.format_exc()
        try:
            row["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
            row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        except Exception:
            pass
        torch.cuda.empty_cache()
    except Exception as e:
        row["status"] = "ERROR"
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()

    Path(args.json_out).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"[worker-result] ctx={args.context} kv={args.target_kv} "
        f"mode={args.mode} rep={args.repeat} status={row['status']} "
        f"peak={row.get('peak_alloc_gib')} prefill={row.get('prefill_s')} "
        f"decode={row.get('decode_s')}",
        flush=True,
    )
    return 0


def _median(xs):
    vals = [float(x) for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def orchestrator(args) -> int:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    kvs = [x.strip() for x in args.target_kvs.split(",") if x.strip()]
    runs = []

    def launch(ctx, kv, mode, rep):
        out = outdir / f"worker_ctx{ctx}_{kv}_{mode}_r{rep}.json"
        cmd = [
            sys.executable, "-m", "specdec.engine.paged_attention_long_context",
            "--worker", "--context", str(ctx), "--target-kv", kv,
            "--mode", mode, "--repeat", str(rep), "--steps", str(args.steps),
            "--chunk", str(args.chunk), "--page-size", str(args.page_size),
            "--k", str(args.k), "--json-out", str(out)
        ]
        print("\n[launch]", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, check=False)
        if out.exists():
            row = json.loads(out.read_text(encoding="utf-8"))
            row["child_returncode"] = p.returncode
        else:
            row = dict(context=ctx, target_kv=kv, mode=mode, repeat=rep,
                       status="PROCESS_FAIL", error=f"returncode={p.returncode}")
        runs.append(row)

    for ctx in contexts:
        for kv in kvs:
            for rep in range(args.direct_repeats):
                launch(ctx, kv, "direct_cuda", rep)
            if args.materialized_reference:
                launch(ctx, kv, "materialized_cuda", 0)

    raw = []
    for r in runs:
        st = r.get("stats") or {}
        dbp = r.get("direct_by_phase") or {}
        fbp = r.get("fallback_by_phase") or {}
        raw.append({
            "context": r.get("context"), "target_kv": r.get("target_kv"),
            "mode": r.get("mode"), "repeat": r.get("repeat"),
            "status": r.get("status"), "prefill_s": r.get("prefill_s"),
            "decode_s": r.get("decode_s"), "decode_tok_s": r.get("decode_tok_s"),
            "peak_alloc_gib": r.get("peak_alloc_gib"),
            "peak_reserved_gib": r.get("peak_reserved_gib"),
            "n_gen": r.get("n_gen"), "gen_sha256": r.get("gen_sha256"),
            "rounds": st.get("rounds"), "proposed": st.get("proposed"),
            "accepted": st.get("accepted"),
            "mismatch_rounds": st.get("mismatch_rounds"),
            "rollbacks": st.get("rollbacks"),
            "direct_propose": dbp.get("propose", 0),
            "direct_verify": dbp.get("verify", 0),
            "direct_repair": dbp.get("repair", 0),
            "fallback_verify": fbp.get("verify", 0),
            "error": r.get("error", ""),
        })

    raw_csv = outdir / "paged_attention_4k8k_real_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    summary = []
    overall = True
    for ctx in contexts:
        for kv in kvs:
            direct = [r for r in runs if r.get("context") == ctx and
                      r.get("target_kv") == kv and r.get("mode") == "direct_cuda"]
            mat = next((r for r in runs if r.get("context") == ctx and
                        r.get("target_kv") == kv and r.get("mode") == "materialized_cuda"), None)
            passed = [r for r in direct if r.get("status") == "PASS"]
            hashes = {r.get("gen_sha256") for r in passed if r.get("gen_sha256")}
            statsigs = {json.dumps(r.get("stats"), sort_keys=True) for r in passed}
            route_ok = bool(passed) and all(
                (r.get("direct_by_phase") or {}).get("propose", 0) > 0 and
                (r.get("direct_by_phase") or {}).get("verify", 0) > 0 and
                (r.get("fallback_by_phase") or {}).get("verify", 0) == 0
                for r in passed
            )
            stable = (
                len(passed) == args.direct_repeats and
                len(hashes) == 1 and len(statsigs) == 1 and route_ok
            )
            parity = ""
            if mat and mat.get("status") == "PASS" and passed:
                parity = int(
                    mat.get("gen_sha256") == passed[0].get("gen_sha256") and
                    mat.get("stats") == passed[0].get("stats")
                )
                overall &= bool(parity)
            overall &= stable
            summary.append({
                "context": ctx,
                "target_kv": kv,
                "direct_runs": f"{len(passed)}/{args.direct_repeats}",
                "stable": int(stable),
                "deterministic_output": int(len(hashes) == 1 and bool(hashes)),
                "deterministic_stats": int(len(statsigs) == 1 and bool(statsigs)),
                "route_ok": int(route_ok),
                "prefill_median_s": _median([r.get("prefill_s") for r in passed]),
                "decode_median_s": _median([r.get("decode_s") for r in passed]),
                "decode_tok_s_median": _median([r.get("decode_tok_s") for r in passed]),
                "peak_alloc_median_gib": _median([r.get("peak_alloc_gib") for r in passed]),
                "peak_reserved_median_gib": _median([r.get("peak_reserved_gib") for r in passed]),
                "materialized_status": mat.get("status") if mat else "NOT_RUN",
                "direct_vs_materialized_exact": parity,
            })

    summary_csv = outdir / "paged_attention_4k8k_real_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    report = outdir / "paged_attention_4k8k_real_report.md"
    lines = [
        "# Final PagedAttention 4K/8K real dual-model validation", "",
        "| Context | Target KV | Direct runs | Stable | Prefill med s | Decode med s | "
        "Decode tok/s | Peak alloc GiB | Materialized | Direct=materialized |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|"
    ]
    for r in summary:
        def fm(x): return "n/a" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {r['context']} | {r['target_kv']} | {r['direct_runs']} | "
            f"{'PASS' if r['stable'] else 'FAIL'} | {fm(r['prefill_median_s'])} | "
            f"{fm(r['decode_median_s'])} | {fm(r['decode_tok_s_median'])} | "
            f"{fm(r['peak_alloc_median_gib'])} | {r['materialized_status']} | "
            f"{r['direct_vs_materialized_exact'] if r['direct_vs_materialized_exact'] != '' else 'n/a'} |"
        )
    lines += ["", "Stable=PASS requires 3/3 fresh direct runs, deterministic output/stats, "
              "direct propose+verify routing, and zero verify fallback.",
              "H2O is disabled here; this isolates the PagedAttention 4K/8K question."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 4K/8K real dual-model summary ===")
    for r in summary:
        print(
            f"ctx={r['context']:<5} kv={r['target_kv']:<10} "
            f"direct={r['direct_runs']} stable={bool(r['stable'])} "
            f"peak={r['peak_alloc_median_gib']} GiB "
            f"tok/s={r['decode_tok_s_median']} "
            f"materialized={r['materialized_status']} "
            f"parity={r['direct_vs_materialized_exact']}"
        )
    print("\nRaw:", raw_csv)
    print("Summary:", summary_csv)
    print("Report:", report)
    print("\npaged_attention_4k8k_real:", "PASS" if overall else "CHECK")
    return 0 if overall else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--context", type=int)
    ap.add_argument("--target-kv", choices=("base_bf16", "v_fp8"), default="base_bf16")
    ap.add_argument("--mode", choices=("direct_cuda", "materialized_cuda"), default="direct_cuda")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--json-out", default="worker.json")
    ap.add_argument("--contexts", default="4096,8192")
    ap.add_argument("--target-kvs", default="base_bf16,v_fp8")
    ap.add_argument("--direct-repeats", type=int, default=3)
    ap.add_argument("--materialized-reference", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--outdir", default="output/paged_attention_4k8k_final")
    args = ap.parse_args()
    if args.worker:
        if args.context is None:
            raise SystemExit("--worker requires --context")
        return worker(args)
    return orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
