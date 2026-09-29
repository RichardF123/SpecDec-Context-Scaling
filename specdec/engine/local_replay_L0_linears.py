"""local_replay_L0_linears.py — §六 局部同输入实验：把 L0.q_proj / L0.o_proj 单独拿出来重放。

目的（只选有证据的 1–2 个模块）
  逐层结果显示：L0 的 `q_proj` 与 `o_proj` 在**对齐行输入逐元素相同**的情况下输出仍不同。
  Linear 是**逐行独立**的（`Y[row] = X[row] @ W`，行间不做归约），所以"对齐行输入相同"即"该行输入相同"。
  本工具把各臂的**完整输入张量**（块 (1,4,H) / 串行 (1,1,H)）抓下来，用**同一份权重**在隔离环境里调用同一模块：
    - 同输入仍不同 ⇒ 支持"该模块的计算路径对 query 长度敏感"；
    - 同输入相同   ⇒ 原观察差异可能来自上游，不能归因该模块。

边界：只跑这一案例；只读抓取 + 隔离重放；不改权重/精度/形状语义；无性能结论。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
SRC_TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
ROUND = 11
CAND0, FED = 27685, 659
TARGETS = ("q_proj", "o_proj", "k_proj", "v_proj")     # 有证据的两个 + 两个对照
LAYER = 0
ALIGN_BLOCK_ROW = 1


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cmp(a, b, name):
    d = (a - b).abs()
    flat = int(d.reshape(-1).argmax())
    return dict(name=name, shape=[int(x) for x in a.shape], dtype=str(a.dtype),
                elementwise_equal=bool(torch.equal(a, b)), abs_err_max=float(d.max()),
                abs_err_mean=float(d.mean()),
                max_err_index=[int(x) for x in torch.unravel_index(torch.tensor(flat), d.shape)],
                a_absmax=float(a.abs().max()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="local_replay_L0_linears", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               scope="L0 Linear 的隔离同输入重放（§六）；只读抓取；不改权重/精度；无性能结论。")
    print(f"[loc] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[loc] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_localreplay_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

    rec["diag_precision"] = M.set_diag_preset("eager")
    from .lowmem_load import load_ours_bf16
    tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
    tgt.set_kv_format(spec_dec._fmt("base_bf16"))
    M.set_attention_strategy("cudnn_prefill")
    snap = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"), map_location="cpu")[f"round{ROUND}_snap_pre"]

    def to_gpu(c):
        return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]

    grabbed: dict = {}

    def _grab(arm: str):
        """只读抓取 L0 目标模块的完整输入张量。"""
        layer = tgt.layers[LAYER]
        mods = {t: (layer.self_attn if t in ("q_proj", "k_proj", "v_proj", "o_proj")
                    else layer) .__getattr__(t) for t in TARGETS}
        hs = []
        for name, mod in mods.items():
            def mk(nm):
                def h(m, inp, out):
                    if isinstance(inp[0], torch.Tensor):
                        grabbed[f"{arm}:{nm}"] = inp[0].detach().clone()
                return h
            hs.append(mod.register_forward_hook(mk(name)))
        try:
            with torch.inference_mode():
                if arm == "block":
                    with M.attention_phase("verify"):
                        spec_dec._verify(tgt, to_gpu(snap), [CAND0, FED, 715, 4882])
                else:
                    w = to_gpu(snap)
                    with M.attention_phase("repair"):
                        spec_dec._step(tgt, CAND0, w)
                    with M.attention_phase("repair"):
                        spec_dec._step(tgt, FED, w)
        finally:
            for h in hs:
                h.remove()

    _grab("block")
    _grab("single")

    layer = tgt.layers[LAYER]
    res = {}
    with torch.inference_mode():
        for t in TARGETS:
            mod = (layer.self_attn if t in ("q_proj", "k_proj", "v_proj", "o_proj")
                   else layer).__getattr__(t)
            xa, xb = grabbed.get(f"block:{t}"), grabbed.get(f"single:{t}")
            entry = dict(block_input_shape=[int(v) for v in xa.shape],
                         single_input_shape=[int(v) for v in xb.shape]) if xa is not None else {}
            if xa is not None and xb is not None:
                ya, yb = mod(xa), mod(xb)
                ra = ya[0, ALIGN_BLOCK_ROW]          # 块路径对齐到输入位置 8216 的那一行
                rb = yb[0, 0]                        # 串行最后一步唯一的行（同位置）
                entry["aligned_output_cmp"] = _cmp(ra.float(), rb.float(), f"L{LAYER}.{t}.aligned_row")
            res[t] = entry
    rec["results"] = res
    rec["note"] = ("Linear 逐行独立 ⇒ 对齐行输入相同即该行输入相同；"
                   "同输入不同输出支持'该模块计算路径对 query 长度敏感'，"
                   "同输入相同则不能把原差异归因到该模块。")
    rec["stage"] = "complete"
    p = OUTDIR / f"{SRC_TAG}_localreplay_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"== L{LAYER} Linear 隔离同输入重放 ==")
    for t, e in res.items():
        c = e.get("aligned_output_cmp")
        if not c:
            print(f"  {t:8s}: 未抓到输入")
            continue
        print(f"  {t:8s} in {e['block_input_shape']} vs {e['single_input_shape']} | "
              f"aligned out equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']:.6g} "
              f"idx={c['max_err_index']} absmax={c['a_absmax']}")
    print(f"[loc] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
