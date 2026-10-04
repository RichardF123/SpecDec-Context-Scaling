from __future__ import annotations
import json, gc
from pathlib import Path
import torch

from . import paged_cache
from .model_self import quantize_kv_fp8
from .g4_b4_cache_tests import MODEL_TARGET, MODEL_SMALL, load_ours
from .kv_final_ablation import PROMPTS, _prompt
from .spec_dec_h2o import spec_decode_greedy_h2o


def same_materialized(a, b):
    ra, rb = paged_cache.materialize_layer(a), paged_cache.materialize_layer(b)
    keys = ["k", "v"] + (["v_scale"] if a["fmt"] == "v_fp8" else [])
    return all(torch.equal(ra[k], rb[k]) for k in keys)


def compaction_gate():
    torch.manual_seed(7)
    out = {}
    for fmt, d in (("fp16", 64), ("v_fp8", 128)):
        t, kv, page = 65, 2, 16
        k = torch.randn(1, kv, t, d, device="cuda", dtype=torch.bfloat16)
        v16 = torch.randn_like(k)
        layer = paged_cache.create_layer_cache(fmt, page)
        if fmt == "fp16":
            paged_cache.append_current(layer, k, v16)
        else:
            qv, sc = quantize_kv_fp8(v16, 128)
            paged_cache.append_current(layer, k, qv, sc)
        keep = torch.tensor(
            [0,1,3,4,7,9,15,16,17,23,31,32,33,40,48,55,63,64],
            device="cuda", dtype=torch.long)
        ref = paged_cache.select_indices(layer, keep)
        opt = paged_cache.select_indices_page_native(layer, keep)
        out[fmt] = same_materialized(ref, opt)
    return out


def state_positions(result):
    return (
        [tuple(int(x) for x in st.positions.detach().cpu().tolist())
         for st in result["target_states"]],
        [tuple(int(x) for x in st.positions.detach().cpu().tolist())
         for st in result["draft_states"]],
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    comp = compaction_gate()
    print("[gate] compaction", comp)
    if not all(comp.values()):
        raise SystemExit("H2O_OPT_GATE_FAIL: compaction payload mismatch")

    target, tok = load_ours(MODEL_TARGET)
    draft, _ = load_ours(MODEL_SMALL)
    rows = []
    ok = True
    for family in ("systems", "code"):
        ids = _prompt(tok, PROMPTS[family], 256)
        common = dict(
            max_new_tokens=16, hh_size=64, recent_size=64,
            k=4, eos_id=None, page_size=16,
            attention_mode="direct_cuda",
            target_kv="base_bf16", draft_kv="base_bf16",
            return_caches=False,
        )
        ref = spec_decode_greedy_h2o(
            target, draft, ids,
            score_capture_mode="reference",
            compaction_mode="reference", **common)
        opt = spec_decode_greedy_h2o(
            target, draft, ids,
            score_capture_mode="paged_cuda",
            compaction_mode="page_native", **common)
        exact = list(ref["gen"]) == list(opt["gen"])
        stats = ref["stats"] == opt["stats"]
        positions = state_positions(ref) == state_positions(opt)
        row = dict(
            family=family,
            output_exact=exact,
            stats_equal=stats,
            retained_positions_equal=positions,
            ref_prune_rounds=ref["h2o_stats"].prune_rounds,
            opt_prune_rounds=opt["h2o_stats"].prune_rounds,
        )
        rows.append(row)
        print("[gate]", row)
        ok &= exact and stats and positions

    out = Path("output/phase12_h2o_opt")
    out.mkdir(parents=True, exist_ok=True)
    (out/"h2o_opt_gate.json").write_text(
        json.dumps({"compaction": comp, "cases": rows, "pass": ok}, indent=2),
        encoding="utf-8")
    del target, draft, tok
    gc.collect(); torch.cuda.empty_cache()
    print("H2O_OPT_GATE:", "PASS" if ok else "CHECK")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
