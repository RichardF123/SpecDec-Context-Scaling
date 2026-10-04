from __future__ import annotations
import argparse, csv, json, statistics, time
from pathlib import Path
import torch
from . import paged_cache
from specdec.kernels import ops

SHAPES={
    "target": dict(H=12,KV=2,D=128),
    "draft": dict(H=14,KV=2,D=64),
}

def timed(fn,warmup,iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    vals=[]
    for _ in range(iters):
        torch.cuda.synchronize(); t=time.perf_counter()
        fn(); torch.cuda.synchronize()
        vals.append((time.perf_counter()-t)*1000)
    return statistics.median(vals)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--contexts",default="512,2048,4096,8192")
    ap.add_argument("--qlens",default="1,2,3,4")
    ap.add_argument("--warmup",type=int,default=4)
    ap.add_argument("--iters",type=int,default=20)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--outdir",default="output/phase12_flash")
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    contexts=[int(x) for x in a.contexts.split(",")]
    qlens=[int(x) for x in a.qlens.split(",")]
    rows=[]; torch.manual_seed(11)

    import triton
    print("triton:",triton.__version__)
    for shape_name,sh in SHAPES.items():
        H,KV,D=sh["H"],sh["KV"],sh["D"]
        for T in contexts:
            k=torch.randn(1,KV,T,D,device="cuda",dtype=torch.bfloat16)
            v=torch.randn_like(k)
            layer=paged_cache.create_layer_cache("fp16",16)
            paged_cache.append_current(layer,k,v)
            for S in qlens:
                q=torch.randn(1,H,S,D,device="cuda",dtype=torch.bfloat16)
                ref=ops.attention(q,k,v,backend="cuda")
                tri=ops.attention(q,k,v,backend="triton")
                pag=ops.paged_attention_block(q,layer,backend="cuda",validate=False)
                err_tri=float((tri.float()-ref.float()).abs().max().item())
                err_pag=float((pag.float()-ref.float()).abs().max().item())
                def mat_cuda():
                    raw=paged_cache.materialize_layer(layer)
                    return ops.attention(q,raw["k"],raw["v"],backend="cuda")
                def mat_tri():
                    raw=paged_cache.materialize_layer(layer)
                    return ops.attention(q,raw["k"],raw["v"],backend="triton")
                fns={
                    "cuda_contig_kernel":lambda:ops.attention(q,k,v,backend="cuda"),
                    "triton_contig_kernel":lambda:ops.attention(q,k,v,backend="triton"),
                    "paged_direct":lambda:ops.paged_attention_block(q,layer,backend="cuda",validate=False),
                    "paged_materialize_cuda_e2e":mat_cuda,
                    "paged_materialize_triton_e2e":mat_tri,
                }
                for rep in range(a.repeats):
                    for mode,fn in fns.items():
                        ms=timed(fn,a.warmup,a.iters)
                        rows.append(dict(
                            shape=shape_name,H=H,KV=KV,D=D,
                            context=T,q_len=S,repeat=rep,mode=mode,
                            median_ms=ms,
                            max_abs_vs_cuda=(err_tri if "triton" in mode else err_pag if mode=="paged_direct" else 0.0)))
                        print(f"[flash] {shape_name} T={T} q={S} {mode} {ms:.4f} ms",flush=True)

    with (out/"flash_tiled_microbench_raw.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary=[]
    keys=sorted(set((r["shape"],r["context"],r["q_len"]) for r in rows))
    for shape,T,S in keys:
        g=[r for r in rows if r["shape"]==shape and r["context"]==T and r["q_len"]==S]
        vals={}
        for mode in sorted(set(r["mode"] for r in g)):
            vals[mode]=statistics.median([r["median_ms"] for r in g if r["mode"]==mode])
        summary.append(dict(
            shape=shape,context=T,q_len=S,
            **{m+"_ms":v for m,v in vals.items()},
            triton_contig_speedup_vs_paged=vals["paged_direct"]/vals["triton_contig_kernel"],
            triton_e2e_speedup_vs_paged=vals["paged_direct"]/vals["paged_materialize_triton_e2e"],
            triton_max_abs=max(r["max_abs_vs_cuda"] for r in g if "triton" in r["mode"]),
        ))
    with (out/"flash_tiled_microbench_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    q4=[r for r in summary if r["shape"]=="target" and r["q_len"]==4 and r["context"] in (2048,4096,8192)]
    wins=[r for r in q4 if r["triton_e2e_speedup_vs_paged"]>=1.10 and r["triton_max_abs"]<=0.02]
    decision=dict(
        criterion="target q=4: materialize+Triton E2E >=1.10x faster than current paged direct on at least 2/3 of 2K,4K,8K; max_abs<=0.02",
        q4=q4,wins=len(wins),run_specdec_probe=len(wins)>=2)
    (out/"flash_decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    print("\nFLASH_DECISION:",json.dumps(decision,indent=2))
    print("FLASH_TILED_MICROBENCH: PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
