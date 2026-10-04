from __future__ import annotations
import argparse, csv, gc, hashlib, json, statistics, time
from pathlib import Path
import torch

from . import spec_dec
from . import spec_dec_h2o as H
from .g4_b4_cache_tests import MODEL_TARGET, MODEL_SMALL, load_ours
from .kv_final_ablation import PROMPTS, _prompt, _configure_standard
from .model_self import attention_phase


def chunked_prefill(model, ids, chunk):
    cache = None; last = None
    with torch.inference_mode(), attention_phase("prefill"):
        for st in range(0, ids.shape[1], chunk):
            end = min(ids.shape[1], st + chunk)
            last, cache = model._run(
                model.embed_tokens(ids[:, st:end]), cache,
                need_logits=(end == ids.shape[1]),
                start_pos_override=st)
    return last, cache


def timed(module, fn, chunk):
    orig = module._prefill
    pre = 0.0
    def pf(model, ids):
        nonlocal pre
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = chunked_prefill(model, ids, chunk)
        torch.cuda.synchronize(); pre += time.perf_counter() - t0
        return out
    module._prefill = pf
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
    finally:
        module._prefill = orig
    return result, total, pre, max(0.0, total - pre), peak


def gen_hash(gen):
    return hashlib.sha256(",".join(map(str, map(int, gen))).encode()).hexdigest()


def run_standard(target, draft, ids, kv, steps, page, chunk):
    def call():
        _configure_standard(
            target, draft, backend="paged_v0",
            attention_mode="direct_cuda", target_kv=kv, page_size=page)
        return spec_dec.spec_decode_greedy(
            target, draft, ids, steps, k=4, eos_id=None,
            target_kv=kv, draft_kv="base_bf16", return_caches=False)
    return timed(spec_dec, call, chunk)


def run_h2o(target, draft, ids, kv, steps, page, chunk, score, compact):
    budget = max(2, int(round(ids.shape[1] * 0.5)))
    def call():
        return H.spec_decode_greedy_h2o(
            target, draft, ids, steps,
            hh_size=budget // 2, recent_size=budget - budget // 2,
            k=4, eos_id=None, page_size=page,
            attention_mode="direct_cuda", target_kv=kv,
            draft_kv="base_bf16", return_caches=False,
            score_capture_mode=score, compaction_mode=compact)
    return timed(H, call, chunk)


def med(xs):
    return statistics.median([float(x) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,2048,4096,8192")
    ap.add_argument("--target-kvs", default="base_bf16,v_fp8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--outdir", default="output/phase12_h2o_opt")
    a = ap.parse_args()

    contexts = [int(x) for x in a.contexts.split(",")]
    kvs = a.target_kvs.split(",")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    target, tok = load_ours(MODEL_TARGET)
    draft, _ = load_ours(MODEL_SMALL)
    base_modes = [
        ("standard", None, None),
        ("h2o_reference", "reference", "reference"),
        ("h2o_optimized", "paged_cuda", "page_native"),
    ]
    rows = []

    for kv in kvs:
        for ctx in contexts:
            ids = _prompt(tok, PROMPTS["systems"], ctx)
            # Decompose the two hotspots at representative mid/long contexts.
            modes = list(base_modes)
            if ctx in (2048, 8192):
                modes += [
                    ("h2o_score_only", "paged_cuda", "reference"),
                    ("h2o_compact_only", "reference", "page_native"),
                ]
            for rep in range(a.repeats):
                shift = rep % len(modes)
                order = modes[shift:] + modes[:shift]
                pair = {}
                for name, score, compact in order:
                    print(f"[H2O-opt] kv={kv} ctx={ctx} rep={rep} mode={name}", flush=True)
                    if name == "standard":
                        r,total,pre,dec,peak = run_standard(
                            target,draft,ids,kv,a.steps,a.page_size,a.chunk)
                    else:
                        r,total,pre,dec,peak = run_h2o(
                            target,draft,ids,kv,a.steps,a.page_size,a.chunk,
                            score,compact)
                    row = dict(
                        target_kv=kv, context=ctx, repeat=rep, mode=name,
                        total_s=total, prefill_s=pre, decode_s=dec,
                        decode_tok_s=a.steps/dec if dec else None,
                        peak_alloc_gib=peak, gen_sha256=gen_hash(r["gen"]),
                        stats_json=json.dumps(r["stats"], sort_keys=True))
                    if name != "standard":
                        hs = r["h2o_stats"]
                        row.update(
                            resident_ratio=(hs.final_target_resident + hs.final_draft_resident)/(2*hs.seen_tokens),
                            prune_rounds=hs.prune_rounds)
                    rows.append(row); pair[name] = row
                std=pair["standard"]; ref=pair["h2o_reference"]; opt=pair["h2o_optimized"]
                opt["exact_vs_ref"] = int(opt["gen_sha256"] == ref["gen_sha256"])
                opt["stats_equal_vs_ref"] = int(opt["stats_json"] == ref["stats_json"])
                opt["speedup_vs_ref"] = ref["decode_s"] / opt["decode_s"]
                opt["overhead_vs_standard"] = opt["decode_s"] / std["decode_s"]
                ref["overhead_vs_standard"] = ref["decode_s"] / std["decode_s"]
                print(
                    f"  ref/std={ref['overhead_vs_standard']:.3f}x "
                    f"opt/std={opt['overhead_vs_standard']:.3f}x "
                    f"ref/opt={opt['speedup_vs_ref']:.3f}x "
                    f"exact={opt['exact_vs_ref']} stats={opt['stats_equal_vs_ref']}",
                    flush=True)

    fields = sorted({k for r in rows for k in r})
    with (out/"h2o_execution_opt_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary=[]
    for kv in kvs:
        for ctx in contexts:
            g=[r for r in rows if r["target_kv"]==kv and r["context"]==ctx]
            std=[r for r in g if r["mode"]=="standard"]
            ref=[r for r in g if r["mode"]=="h2o_reference"]
            opt=[r for r in g if r["mode"]=="h2o_optimized"]
            std_s, ref_s, opt_s = med([r["decode_s"] for r in std]), med([r["decode_s"] for r in ref]), med([r["decode_s"] for r in opt])
            score_only=[r for r in g if r["mode"]=="h2o_score_only"]
            compact_only=[r for r in g if r["mode"]=="h2o_compact_only"]
            summary.append(dict(
                target_kv=kv, context=ctx,
                standard_decode_s=std_s, reference_decode_s=ref_s, optimized_decode_s=opt_s,
                reference_overhead_x=ref_s/std_s,
                optimized_overhead_x=opt_s/std_s,
                optimized_speedup_vs_reference_x=ref_s/opt_s,
                score_only_overhead_x=(med([r["decode_s"] for r in score_only])/std_s if score_only else ""),
                compact_only_overhead_x=(med([r["decode_s"] for r in compact_only])/std_s if compact_only else ""),
                optimized_resident_ratio=med([r["resident_ratio"] for r in opt]),
                exact_vs_reference_rate=sum(int(r.get("exact_vs_ref",0)) for r in opt)/len(opt),
                stats_equal_vs_reference_rate=sum(int(r.get("stats_equal_vs_ref",0)) for r in opt)/len(opt)))

    with (out/"h2o_execution_opt_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    (out/"h2o_execution_opt_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\n=== H2O execution optimization summary ===")
    for r in summary:
        print(f"ctx={r['context']:<5} kv={r['target_kv']:<10} ref/std={r['reference_overhead_x']:.3f}x opt/std={r['optimized_overhead_x']:.3f}x ref/opt={r['optimized_speedup_vs_reference_x']:.3f}x resident={r['optimized_resident_ratio']:.3f}")
    ok=all(r["exact_vs_reference_rate"]==1.0 and r["stats_equal_vs_reference_rate"]==1.0 for r in summary)
    print("H2O_EXECUTION_OPT_BENCH:", "PASS" if ok else "CHECK")
    del target,draft,tok; gc.collect(); torch.cuda.empty_cache()
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
