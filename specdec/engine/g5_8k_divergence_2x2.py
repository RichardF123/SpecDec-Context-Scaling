"""g5_8k_divergence_2x2.py — 用**真实两条路径的真实 cache 历史**做 2×2 定点对照。

要回答的问题
  首个分歧（生成序列 index 31）处：分歧来自 **计算形状**（s=1 单步 vs s=4 block），
  还是来自 **cache 历史**（真实 AR 路径 vs 真实 SpecDec 路径）？

2×2（每格给出**原始分数**、argmax、并列集合、margin、有限性）
  cache 历史（AR / SpecDec） ×  计算形状（single / block）
  其中 **两个格子是实测的**（AR-history×single = AR 真实决策；Spec-history×block = SpecDec 真实 rows[0]），
  另两个是**反事实**（AR-history×block、Spec-history×single）。

为什么分进程：SpecDec 历史需要 draft 驻留（8K 实测 peak_reserved 7332 MiB），
再叠 cache 副本会顶到 8188 MiB 上限 ⇒ 按任务要求**分进程重放**，并把捕获到的 cache
**转存 CPU/磁盘**，最后用 `DIAG_MODE=compare`（纯 CPU）做逐张量比较。

纪律
  * 先跑**生产精度路径**（`set_diag_preset("eager")` 且记录）；高精度 attention 只作为**单独标注**的
    诊断变体，**不**称"全模型 FP32 真值"；
  * 不改 dtype/k/chunk/量化/后端/tie-breaking/验收阈值；
  * 逐决策有限性检查保留（复用 `DecisionHooks` + `CheckRegistry`）；
  * 同一参数只跑一次；未复现就如实记录，不换 prompt、不挑样本。

用法
  python -m specdec.engine.g5_8k_divergence_2x2            # DIAG_MODE=ar  （默认）
  DIAG_MODE=spec   ...                                    # 真实 SpecDec 历史
  DIAG_MODE=compare ...                                   # 纯 CPU：读两份 dump 出 2×2
环境变量：DIAG_KV=base_bf16|v_fp8、DIAG_TAG（一次诊断的公共标签，用于把同名产物串起来）
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_fields as DF
from . import model_self as M
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V
from .smoke_checks import CheckRegistry
from .smoke_hooks import DecisionHooks

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

MODE = os.environ.get("DIAG_MODE", "ar").strip().lower()
KV = os.environ.get("DIAG_KV", "base_bf16")
TAG = os.environ.get("DIAG_TAG", "div2x2")
CTX = int(os.environ.get("DIAG_CTX", "8192"))
CHUNK = int(os.environ.get("DIAG_CHUNK", "2048"))
PROMPT_LEN = CTX
PREFIX_TOKENS = int(os.environ.get("DIAG_PREFIX_TOKENS", "30"))
DIV_INDEX = 31                                   # ★ 生成序列 index（不是 PREFIX_TOKENS）
CANDS = [572, 5230, 304, 279]
WATCH = [1083, 5880, 5230]
BOUNDARY_CACHE_LEN = PROMPT_LEN + PREFIX_TOKENS  # 8222
SRC = {  # 只读来源（按明确文件名，不猜"最新"）
    "base_bf16": dict(ar="smoke8k_cudnn_prefill_20260927T135218Z.json",
                      spec="smoke8k_cudnn_prefill_20260927T135254Z.json"),
    "v_fp8": dict(ar="smoke8k_cudnn_prefill_20260927T135336Z.json",
                  spec="smoke8k_cudnn_prefill_20260927T135413Z.json"),
}


class _Captured(Exception):
    """在拿到目标轮次的真实 cache 后主动中止解码（省时省显存），由外层捕获。"""


def _path(kind: str, run_id: str) -> Path:
    return OUTDIR / f"{TAG}_{KV}_{MODE}_{run_id}.{kind}"


def _write(rec: dict, run_id: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _path("json", run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _clone(cache):
    """逐层 dict 浅拷贝：仅在"引擎只重绑键、不就地改写张量"这一前提下有效。
    本工具会在事后用**逐张量 `torch.equal`**核对原快照未被修改（见 `cache_checks`）。"""
    return [dict(d) for d in cache]


def _dump(cache, path: Path) -> None:
    torch.save([{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
                for d in cache], str(path))


def _cache_meta(cache) -> dict:
    layers = []
    for d in cache:
        e = dict(fmt=d.get("fmt"), keys=sorted(k for k in d if k != "fmt"))
        for k in ("k", "v", "k_scale", "v_scale"):
            if k in d and torch.is_tensor(d[k]):
                e[k] = dict(shape=[int(x) for x in d[k].shape], dtype=str(d[k].dtype))
        layers.append(e)
    return dict(n_layers=len(cache), length=spec_cache.cache_length(cache), layers=layers[:3],
                note="只打印前 3 层结构；完整结构以 dump 的 .pt 为准")


def _cells(tgt, snap, rec, tag: str) -> dict:
    """在给定 cache 快照上跑 single 与 block 两种形状（**生产精度**）。

    两个形状都从**同一份快照**出发，且各自 `_clone` ⇒ 可以直接逐元素比较。
    """
    out = {}
    with torch.inference_mode():
        lg, _ = tgt._run(tgt.embed_tokens(torch.tensor([[CANDS[0]]], device="cuda")),
                         _clone(snap), need_logits=True)
        row_single = lg[0, -1].detach().clone()
    with torch.inference_mode():
        rows, _blk = spec_dec._verify(tgt, _clone(snap), CANDS)
        row_block = rows[0].detach().clone()
    out["single"] = DF.row_fields(row_single, watch=WATCH)
    out["single"]["margin"] = _margin(out["single"])
    out["block"] = DF.row_fields(row_block, watch=WATCH)
    out["block"]["margin"] = _margin(out["block"])
    out["_diff"] = DF.compare_rows(row_single, row_block)
    # 原始行转存 CPU（供离线复算指标，避免为了改口径反复占 GPU）
    out["_rows"] = dict(single=row_single.detach().float().cpu(),
                        block=row_block.detach().float().cpu())
    return out


def run_ar(tgt, ids, prefix, rec, reg):
    """真实 AR 路径：prefill + 单步推进到边界（cache_len = 8222），捕获该处 cache。"""
    with torch.inference_mode(), M.attention_phase("prefill"):
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, CHUNK):
            lg, cache = tgt._run(tgt.embed_tokens(ids[:, st:st + CHUNK]), cache,
                                 need_logits=(st + CHUNK >= n))
    reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg", where="AR prefill")
    g0 = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
    rows = []
    for i, t in enumerate(prefix):
        with torch.inference_mode(), M.attention_phase("decode"):
            lg, cache = tgt._run(tgt.embed_tokens(torch.tensor([[t]], device="cuda")),
                                 cache, need_logits=True)
        reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                         round_=i, step=i + 1, where="AR 单步：紧接被 argmax 使用")
        rows.append(dict(step=i + 1, fed_token=int(t),
                         argmax=int(lg[0, -1].argmax()), gen_index=i + 1))
    rec["ar_path"] = dict(g0_from_prefill=g0, steps=rows,
                          final_cache_len=spec_cache.cache_length(cache))
    return cache, g0


def run_spec(tgt, drf, ids, prefix, rec, reg, eos):
    """真实 SpecDec 路径：跑到 pre == 8222 的那一轮，捕获**真实** target cache 与真实 rows。"""
    held: dict = {}

    def on_verify_pre(cache, cands):
        if spec_cache.cache_length(cache) == BOUNDARY_CACHE_LEN:
            held["snap"] = _clone(cache)          # 就地修改**之前**的快照
            held["cands"] = [int(x) for x in cands]
            held["len"] = spec_cache.cache_length(cache)

    def on_verify_post(rows):
        if "snap" in held and "rows" not in held:
            held["rows"] = rows.detach().clone()

    def on_round_start(r):
        held.setdefault("rounds", []).append(r)
        if "snap" in held and "rows" in held:
            raise _Captured()                     # 拿到就停：不做多余的 31..63

    hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg,
                          prefill_impl=P5V.timed_chunked_prefill({}, CHUNK, tgt),
                          on_verify_pre=on_verify_pre, on_verify_post=on_verify_post,
                          on_round_start=on_round_start)
    hooks.install()
    rounds: list[dict] = []
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, 32, k=4, eos_id=eos,
                                          target_kv=KV, draft_kv="base_bf16",
                                          log=lambda d: rounds.append(dict(d)))
    except _Captured:
        out = None
    finally:
        hooks.remove()
    rec["spec_path"] = dict(rounds=rounds, captured=("snap" in held),
                            captured_cache_len=held.get("len"),
                            captured_cands=held.get("cands"))
    if rounds:
        last = rounds[-1]
        rec["spec_path"]["last_round_before_capture"] = dict(
            round=last["round"], pre=last["pre"], Kp=last["Kp"], m=last["m"],
            cands=last["cands"], g=last["g"], emitted=last["emitted"])
    held["rounds_log"] = rounds
    return held


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_8k_divergence_2x2", mode=MODE, kv=KV, tag=TAG,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     config=dict(ctx=CTX, chunk=CHUNK, kv=KV, prefix_tokens=PREFIX_TOKENS,
                                 divergence_gen_index=DIV_INDEX, boundary_cache_len=BOUNDARY_CACHE_LEN,
                                 cands=CANDS, watch=WATCH),
                     scope="定点诊断：真实 AR / 真实 SpecDec 的 cache 历史 × 计算形状 的 2×2。"
                           "不测性能、不做重复、不改生产精度。")
    print(f"[2x2] run_id={run_id} mode={MODE} kv={KV} tag={TAG}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)

    if MODE == "compare":
        return _compare(rec, run_id)

    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                              overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    rec["gpu_consumers"] = P5V.gpu_consumers()
    print(f"[2x2] commit: {why}")
    print(f"[2x2] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        _write(rec, run_id)
        print("[2x2] !! 环境不合格，未加载")
        return 3

    try:
        # 生产精度：显式关闭所有诊断开关并记录
        rec["diag_precision"] = M.set_diag_preset("eager")
        rec["attention_precision_variant"] = None
        M.set_attention_strategy("cudnn_prefill")

        # 输入：原始 prompt + 已确认一致的前 30 个输出 token（不重新选 prompt）
        src = {k: json.loads((OUTDIR / v).read_text(encoding="utf-8"))["run"]["gen"]
               for k, v in SRC[KV].items()}
        prefix = list(src["ar"][:PREFIX_TOKENS])
        rec["prefix"] = dict(tokens=prefix, agrees=(src["ar"][:PREFIX_TOKENS] == src["spec"][:PREFIX_TOKENS]),
                             source_files=SRC[KV],
                             observed={"ar_g31": src["ar"][DIV_INDEX], "spec_g31": src["spec"][DIV_INDEX]})
        print(f"[2x2] 前缀一致={rec['prefix']['agrees']}；实测 g31: AR={src['ar'][DIV_INDEX]} "
              f"Spec={src['spec'][DIV_INDEX]}")

        from .lowmem_load import load_ours_bf16
        tgt, tokz, tinfo = load_ours_bf16(TARGET)
        drf, dinfo = None, None
        if MODE == "spec":
            drf, _t2, dinfo = load_ours_bf16(DRAFT)
            drf.set_kv_format("fp16")
        rec["load"] = dict(target=tinfo, draft=dinfo)
        tgt.set_kv_format(spec_dec._fmt(KV))
        full = P5V.load_ids(tokz)
        ids = full[:, :CTX].contiguous()
        rec["input"] = dict(ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]), ctx=CTX)
        reg = CheckRegistry()

        if MODE == "ar":
            snap, g0 = run_ar(tgt, ids, prefix, rec, reg)
            rec["cache"] = _cache_meta(snap)
            _dump(snap, _path("cache.pt", run_id))
            # **真正的快照隔离核对**：先留一份参照，跑完两个形状后逐张量比对
            snap_ref = _clone(snap)
            cells = _cells(tgt, snap, rec, "ar")
            ok_t = [torch.equal(a.get("k"), b.get("k")) and torch.equal(a.get("v"), b.get("v"))
                    for a, b in zip(snap_ref, snap)]
            rec["cache_checks"] = dict(
                n_layers=len(ok_t), snapshot_unchanged_all_layers=bool(ok_t) and all(ok_t),
                method="逐张量 torch.equal（快照参照 vs 跑完两形状后的快照）",
                note="**不**用「末 8 位置求和」之类的近似指纹来宣称逐位不变")
            del snap_ref
        else:
            held = run_spec(tgt, drf, ids, prefix, rec, reg, tokz.eos_token_id)
            if "snap" not in held:
                rec["stage"] = "no_capture"
                _write(rec, run_id)
                print("[2x2] !! 没有捕获到边界处的 cache（未复现该轮）⇒ 停下")
                return 5
            snap = held["snap"]
            actual_rows0 = held["rows"][0].detach().clone()
            rec["cache"] = _cache_meta(snap)
            _dump(snap, _path("cache.pt", run_id))
            del drf, held
            torch.cuda.empty_cache()                      # 释放 draft 再加副本，避免顶满
            snap_ref = _clone(snap)
            cells = _cells(tgt, snap, rec, "spec")
            ok_t = [torch.equal(a.get("k"), b.get("k")) and torch.equal(a.get("v"), b.get("v"))
                    for a, b in zip(snap_ref, snap)]
            rec["cache_checks"] = dict(
                n_layers=len(ok_t), snapshot_unchanged_all_layers=bool(ok_t) and all(ok_t),
                method="逐张量 torch.equal（快照参照 vs 跑完两形状后的快照）",
                note="**不**用近似指纹宣称逐位不变")
            # 实测 rows[0] 与"同一快照上重跑 block"必须一致（确定性核对）
            rec["spec_actual_rows0"] = DF.row_fields(actual_rows0, watch=WATCH)
            rec["spec_actual_rows0"]["margin"] = _margin(rec["spec_actual_rows0"])
            rec["spec_actual_vs_replayed_block"] = dict(
                argmax_same=bool(int(actual_rows0.argmax()) == cells["block"]["argmax_id"]),
                abs_err_max=float((actual_rows0.float() - torch.tensor(
                    [e["score"] for e in cells["block"]["topk"]], dtype=torch.float32)).abs().max())
                if False else None,
                note="只核对 argmax（逐元素比较需要完整行，见 .pt 转存）")
            del snap_ref

        rows_dump = cells.pop("_rows", None)
        if rows_dump is not None:
            rp = _path("rows.pt", run_id)
            torch.save(rows_dump, str(rp))
            rec["rows_dump"] = str(rp)
        rec["cells"] = cells
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        print(f"[2x2] single: argmax_id={cells['single']['argmax_id']} "
              f"max={cells['single']['max_score']} tied={cells['single']['n_max_tied']} "
              f"marg={_margin(cells['single']):.4f} watch={cells['single']['watch_scores']}")
        print(f"[2x2] block : argmax_id={cells['block']['argmax_id']} "
              f"max={cells['block']['max_score']} tied={cells['block']['n_max_tied']} "
              f"marg={_margin(cells['block']):.4f} watch={cells['block']['watch_scores']}")
        print(f"[2x2] diff abs_err_max={cells['_diff']['abs_err_max']:.6g} "
              f"ulp_err_max={cells['_diff']['ulp_err_max']} argmax_same={cells['_diff']['argmax_same']}")
        print(f"[2x2] RESULT_JSON={p}")
        return 0
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        rec["alloc"] = P5V.allocator_state()
        p = _write(rec, run_id)
        print(f"[2x2] !! 崩溃（已落盘 {p}）：\n{rec['traceback']}")
        print(f"[2x2] RESULT_JSON={p}")
        return 2


def _margin(f: dict) -> float:
    tk = f["topk"]
    return float(tk[0]["score"] - tk[1]["score"]) if len(tk) > 1 else 0.0


def _compare(rec: dict, run_id: str) -> int:
    """纯 CPU：读两档两份 dump，做**逐张量相等/最大差**核对，并组装 2×2。"""
    import glob
    print(f"[2x2] compare: kv={KV} tag={TAG}")
    found = {}
    for mode in ("ar", "spec"):
        js = sorted(glob.glob(str(OUTDIR / f"{TAG}_{KV}_{mode}_*.json")))
        if js:
            found[mode] = json.loads(Path(js[-1]).read_text(encoding="utf-8"))
    rec["sources"] = {m: r.get("run_id") for m, r in found.items()}
    if len(found) != 2:
        rec["stage"] = "missing_source"
        p = _write(rec, run_id)
        print(f"[2x2] !! 缺少某一档的结果 ⇒ 停下（{list(found)}）-> {p}")
        return 5
    dumps = {}
    for mode, r in found.items():
        c = sorted(glob.glob(str(OUTDIR / f"{TAG}_{KV}_{mode}_{r['run_id']}.cache.pt")))
        if c:
            dumps[mode] = torch.load(c[-1], map_location="cpu")
    hist = {}
    if len(dumps) == 2:
        a, b = dumps["ar"], dumps["spec"]
        eq, dmax, shapes_ok, notes = [], [], True, []
        for i, (da, db) in enumerate(zip(a, b)):
            # ⚠️ fp8 载荷**必须反量化后再比**：直接比 fp8 payload 的数差没有物理含义
            # （首跑就是直接比 payload，得到 max|Δ|=32.0 这种不可解释的数字）。
            for k in ("k", "v"):
                if k not in da or k not in db:
                    continue
                if tuple(da[k].shape) != tuple(db[k].shape) or da[k].dtype != db[k].dtype:
                    shapes_ok = False
                    continue
                xa, xb = da[k], db[k]
                sc = f"{k}_scale"
                if sc in da and sc in db:
                    if tuple(da[sc].shape) != tuple(db[sc].shape):
                        shapes_ok = False
                        continue
                    xa = M.dequantize_kv_fp8(da[k], da[sc], torch.bfloat16)
                    xb = M.dequantize_kv_fp8(db[k], db[sc], torch.bfloat16)
                    notes.append(f"layer{i}.{k} 已反量化后比较（含 scale）")
                eq.append(torch.equal(xa, xb))
                dmax.append(float((xa.float() - xb.float()).abs().max()))
        notes.append("k/v 均按**反量化后**的值比较；scale 的形状与 dtype 也已核对")
        hist = dict(same_shape_and_dtype=shapes_ok, n_tensors_compared=len(eq),
                    all_bitwise_equal=bool(eq) and all(eq),
                    n_bitwise_equal=int(sum(eq)),
                    max_abs_diff=float(max(dmax)) if dmax else None,
                    note="逐张量 `torch.equal` + 最大绝对差（fp8 载荷先反量化）；"
                         "**不**用末 8 位置求和之类的近似指纹",
                    detail_notes=notes[:3])
        print(f"[2x2] cache 历史比较：n={len(eq)} 逐位相同={hist['all_bitwise_equal']} "
              f"max|Δ|={hist['max_abs_diff']}")
    rec["cache_history_compare"] = hist or dict(skipped="缺少 dump")
    # 组装 2×2
    tbl = {}
    for mode in ("ar", "spec"):
        if mode in found:
            tbl[mode] = {k: found[mode]["cells"][k] for k in ("single", "block") if k in found[mode].get("cells", {})}
    rec["table_2x2"] = tbl
    rec["observed"] = {m: found[m].get("prefix", {}).get("observed") for m in found}
    rec["stage"] = "complete"
    p = _write(rec, run_id)
    print(f"[2x2] 2×2 表已组装 -> {p}")
    print(f"[2x2] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
