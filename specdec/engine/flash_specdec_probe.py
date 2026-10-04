from __future__ import annotations
import csv,json,statistics,time,gc
from pathlib import Path
import torch
from . import spec_dec
from .g4_b4_cache_tests import MODEL_TARGET,MODEL_SMALL,load_ours
from .kv_final_ablation import PROMPTS,_prompt
from .model_self import attention_phase


def chunked_prefill(model,ids,chunk=512):
    cache=None;lg=None
    with torch.inference_mode(),attention_phase("prefill"):
        for st in range(0,ids.shape[1],chunk):
            end=min(ids.shape[1],st+chunk)
            lg,cache=model._run(
                model.embed_tokens(ids[:,st:end]),cache,
                need_logits=end==ids.shape[1],start_pos_override=st)
    return lg,cache


def run(target,draft,ids,mode):
    target.set_cache_backend("paged_v0",page_size=16)
    draft.set_cache_backend("paged_v0",page_size=16)
    target.set_paged_attention_mode(mode)
    draft.set_paged_attention_mode(mode)
    orig=spec_dec._prefill
    spec_dec._prefill=lambda m,x:chunked_prefill(m,x,512)
    try:
        torch.cuda.synchronize();t=time.perf_counter()
        r=spec_dec.spec_decode_greedy(
            target,draft,ids,16,k=4,eos_id=None,
            target_kv="base_bf16",draft_kv="base_bf16")
        torch.cuda.synchronize();dt=time.perf_counter()-t
    finally:
        spec_dec._prefill=orig
    return r,dt


def main():
    out=Path("output/phase12_flash");out.mkdir(parents=True,exist_ok=True)
    decision=json.loads((out/"flash_decision.json").read_text())
    if not decision.get("run_specdec_probe"):
        (out/"flash_specdec_probe_SKIPPED.txt").write_text(
            "Microbenchmark gate did not pass; real SpecDec Triton probe intentionally skipped.\n")
        print("FLASH_SPECDEC_PROBE: SKIPPED_BY_GATE")
        return 0

    target,tok=load_ours(MODEL_TARGET); draft,_=load_ours(MODEL_SMALL)
    rows=[]
    for ctx in (2048,4096,8192):
        ids=_prompt(tok,PROMPTS["systems"],ctx)
        for rep in range(3):
            order=("direct_cuda","materialized_triton") if rep%2==0 else ("materialized_triton","direct_cuda")
            pair={}
            for mode in order:
                r,dt=run(target,draft,ids,mode)
                row=dict(context=ctx,repeat=rep,mode=mode,seconds=dt,
                         gen=list(map(int,r["gen"])),stats=json.dumps(r["stats"],sort_keys=True))
                rows.append(row);pair[mode]=row
            tri=pair["materialized_triton"]; direct=pair["direct_cuda"]
            tri["exact_vs_direct"]=int(tri["gen"]==direct["gen"])
            tri["stats_equal_vs_direct"]=int(tri["stats"]==direct["stats"])
            tri["speedup_vs_direct"]=direct["seconds"]/tri["seconds"]
            print(f"[probe] ctx={ctx} speedup={tri['speedup_vs_direct']:.3f}x exact={tri['exact_vs_direct']}",flush=True)

    safe=[{k:v for k,v in r.items() if k!="gen"} for r in rows]
    fields=sorted({k for r in safe for k in r})
    with (out/"flash_specdec_probe_raw.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(safe)
    summary=[]
    for ctx in (2048,4096,8192):
        tri=[r for r in rows if r["context"]==ctx and r["mode"]=="materialized_triton"]
        direct=[r for r in rows if r["context"]==ctx and r["mode"]=="direct_cuda"]
        summary.append(dict(
            context=ctx,
            direct_s=statistics.median([r["seconds"] for r in direct]),
            triton_s=statistics.median([r["seconds"] for r in tri]),
            speedup=statistics.median([r.get("speedup_vs_direct",0) for r in tri]),
            exact_rate=sum(r.get("exact_vs_direct",0) for r in tri)/len(tri),
            stats_equal_rate=sum(r.get("stats_equal_vs_direct",0) for r in tri)/len(tri)))
    (out/"flash_specdec_probe_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    ok=all(r["exact_rate"]==1 and r["stats_equal_rate"]==1 for r in summary)
    print("FLASH_SPECDEC_PROBE:","PASS" if ok else "CHECK")
    del target,draft,tok;gc.collect();torch.cuda.empty_cache()
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())
