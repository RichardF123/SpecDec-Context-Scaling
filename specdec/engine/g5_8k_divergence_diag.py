"""g5_8k_divergence_diag.py — **定点**诊断 8K 同后端 AR/SpecDec 的首个分歧（阶段 1：同 cache、不同形状）。

已知（来自 4 份原始 JSON）
  base_bf16：AR 的 g31 = 1083；SpecDec round 8 的 `rows[0]` argmax = 5880
  v_fp8    ：AR 的 g31 = 5880；SpecDec round 8 的 `rows[0]` argmax = 1083   （两者**交换**）
  round 8：`pre=8222`、`cands=[572,5230,304,279]`、`m=1` ⇒ 输出 `[572, g[1]]`，`g[1]=argmax(rows[0])`
  位置关系（已由 `spec_verify_index_selftest_cpu` 16/16 钉死）：
    `saved_next` = 位置 P−1 的 logits ⇒ 回答 `cands[0]`；`rows[j]` = 位置 P+j ⇒ 回答 `cands[j+1]`
    ⇒ 本轮的决策位置 = **8222**（= 8191 + 31），两档都在 8222 处、且位置 8222 上的 token 都是 572

本工具做什么（**只做定点正确性诊断**，不测性能、不跑重复）
  1. 用**原始输入 token**（wikitext 前缀 + 已确认一致的前 30 个输出 token）重建到 `cache_len == 8222`；
  2. 打印"位置 8221 的决策"作为复现检查（应为 572，与两档一致）；
  3. 对**同一份 cache 快照**分别做：
       ① 单步输入 `cands[0]=572`（= AR 路径的形状，s=1）      → 决策行 A
       ② block 输入 `[572,5230,304,279]`（= SpecDec 形状，s=4）→ `rows[0]` = 决策行 B
     两者**必须预测同一位置、看到相同有效前缀**；
  4. 比较两行的 top1/top2/margin、1083/5880/5230 的分数、max|Δ|、有限性；
  5. **快照隔离核对**：对比前后指纹，证明两条分支没有互相污染；
  6. （可选、明确标注）把**注意力路径**提到更高精度（`DIAG_QK_FP32` / `DIAG_PROB_FP32`）后重算这两行。
     ⚠️ 这**只**提高 attention 的精度，**不是**"整个模型 FP32 真值"，不得如此表述。

纪律：不改 dtype/k/chunk/量化/后端；不采用未批准判据；耗时仅诊断；OOM/非有限即落盘停止。
用法：python -m specdec.engine.g5_8k_divergence_diag      （DIAG_KV=base_bf16 默认）
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET = P5V.TARGET

KV = os.environ.get("DIAG_KV", "base_bf16")
CTX = int(os.environ.get("DIAG_CTX", "8192"))
CHUNK = int(os.environ.get("DIAG_CHUNK", "2048"))
PREFIX_TOKENS = int(os.environ.get("DIAG_PREFIX_TOKENS", "30"))   # 已确认一致的输出 token 数
DIV_INDEX = PREFIX_TOKENS                                        # 首个分歧的 0-based 下标（= 31）
CANDS = [572, 5230, 304, 279]
WATCH = [1083, 5880, 5230]
# 记录来源（不重新选 prompt，也不猜产物路径）
SRC_AR = OUTDIR / "smoke8k_cudnn_prefill_20260927T135218Z.json"
SRC_SPEC = OUTDIR / "smoke8k_cudnn_prefill_20260927T135254Z.json"
SRC_AR_V = OUTDIR / "smoke8k_cudnn_prefill_20260927T135336Z.json"
SRC_SPEC_V = OUTDIR / "smoke8k_cudnn_prefill_20260927T135413Z.json"


def _fp(cache) -> dict:
    """cache 指纹（便宜且敏感）：逐层长度 + 末 8 个位置的 k/v 和。"""
    out = {}
    for i, d in enumerate(cache):
        for name in ("k", "v"):
            if name in d and hasattr(d[name], "shape"):
                t = d[name]
                out[f"{i}.{name}"] = [list(t.shape), float(t[:, :, -8:].float().sum())]
    return out


def _top2(row) -> dict:
    v = torch.topk(row.float(), 2)
    return dict(top1=int(v.indices[0]), top2=int(v.indices[1]),
                top1_score=float(v.values[0]), top2_score=float(v.values[1]),
                margin=float(v.values[0] - v.values[1]))


def _scores(row, toks) -> dict:
    r = row.float()
    return {str(t): float(r[t]) for t in toks}


def _row_stats(row) -> dict:
    return dict(finite=bool(torch.isfinite(row).all()), absmax=float(row.float().abs().max()),
                **_top2(row), watch_scores=_scores(row, WATCH))


def _write(rec, run_id) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"divdiag_{KV}_{run_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _step(model, tok: int, cache):
    with torch.inference_mode():
        lg, c2 = model._run(model.embed_tokens(torch.tensor([[tok]], device="cuda")),
                            cache, need_logits=True)
    return lg, c2


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_8k_divergence_diag",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), kv=KV, stage="start",
                     config=dict(ctx=CTX, chunk=CHUNK, kv=KV, prefix_tokens=PREFIX_TOKENS,
                                 divergence_index=DIV_INDEX, cands=CANDS, watch=WATCH),
                     scope="**定点正确性诊断**：只定位首个分歧；不是性能测试、不是重复稳定性。")
    print(f"[divdiag] run_id={run_id} kv={KV} ctx={CTX} chunk={CHUNK} "
          f"prefix={PREFIX_TOKENS} cands={CANDS}")
    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    rec["gpu_consumers"] = P5V.gpu_consumers()
    print(f"[divdiag] commit: {why}")
    print(f"[divdiag] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        _write(rec, run_id)
        print("[divdiag] !! 环境不合格，未加载")
        return 3

    try:
        # ---- 输入：原始 token ids（prompt 前缀 + 已确认一致的前 30 个输出 token）----
        src = {}
        for tag, p in (("ar_base", SRC_AR), ("spec_base", SRC_SPEC),
                       ("ar_vfp8", SRC_AR_V), ("spec_vfp8", SRC_SPEC_V)):
            if p.exists():
                src[tag] = json.loads(p.read_text(encoding="utf-8"))["run"]["gen"]
        rec["source_gens"] = {k: v[:PREFIX_TOKENS + 2] for k, v in src.items()}
        agree = all(v[:PREFIX_TOKENS] == src["ar_base"][:PREFIX_TOKENS] for v in src.values())
        rec["prefix_agrees_across_all4"] = bool(agree)
        print(f"[divdiag] 4 份来源的前 {PREFIX_TOKENS} 个 token 完全一致 = {agree}")
        if not agree:
            rec["stage"] = "aborted_prefix_disagrees"
            _write(rec, run_id)
            print("[divdiag] !! 前 30 token 不一致 ⇒ 停下（不强行制造）")
            return 5
        prefix = list(src["ar_base"][:PREFIX_TOKENS])

        from .lowmem_load import load_ours_bf16
        tgt, tokz, tinfo = load_ours_bf16(TARGET)
        rec["load"] = tinfo
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy("cudnn_prefill")
        rec["kv_format"] = dict(target=spec_dec._fmt(KV))
        full = P5V.load_ids(tokz)
        ids = full[:, :CTX].contiguous()
        rec["input"] = dict(ctx=int(ids.shape[1]), ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]),
                            prefix_hash=P5V._gen_hash(prefix))
        print(f"[divdiag] prompt {ids.shape[1]} token, ids_hash={rec['input']['ids_hash']}")

        # ---- 分块 prefill（阶段显式标注）----
        with torch.inference_mode(), M.attention_phase("prefill"):
            cache, lg, n = None, None, ids.shape[1]
            for st in range(0, n, CHUNK):
                lg, cache = tgt._run(tgt.embed_tokens(ids[:, st:st + CHUNK]), cache,
                                     need_logits=(st + CHUNK >= n))
        clen = spec_cache.cache_length(cache)
        rec["after_prefill"] = dict(cache_len=clen, finite=bool(torch.isfinite(lg).all()),
                                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
        print(f"[divdiag] prefill 完成 cache_len={clen} finite={rec['after_prefill']['finite']}")

        # ---- 复现检查：位置 8191 的决策应当是 g0（两档一致）----
        g0 = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        rec["g0_from_prefill"] = g0
        print(f"[divdiag] 位置 {clen - 1} 的决策 = {g0}（应为 {prefix[0]}）")

        # ---- 逐 token 推进到 cache_len == CTX + PREFIX_TOKENS（位置 8222 之前）----
        last_row = None
        for t in prefix:
            lg, cache = _step(tgt, t, cache)
            last_row = lg[0, -1].detach()
        clen = spec_cache.cache_length(cache)
        dec_pos = clen - 1                      # = 8191 + PREFIX_TOKENS = 8222
        g_next_obs = int(last_row.argmax())
        rec["repro"] = dict(cache_len=clen, decision_position=dec_pos,
                            observed_next_token=g_next_obs,
                            expected_next_token=src["ar_base"][PREFIX_TOKENS],
                            reproduced=(g_next_obs == src["ar_base"][PREFIX_TOKENS]))
        print(f"[divdiag] 推进到 cache_len={clen}；位置 {dec_pos} 的决策 = {g_next_obs}"
              f"（AR 实测 g{DIV_INDEX} = {src['ar_base'][DIV_INDEX]}）"
              f" ⇒ 复现={rec['repro']['reproduced']}")

        # ---- 快照（逐层 dict 浅拷贝即可：引擎只重绑键、不就地改写张量；哨兵已验证）----
        snap = [dict(d) for d in cache]
        fp_before = _fp(snap)

        # ---- 分支 ①：单步 cands[0]（s=1，AR 形状）----
        c1 = [dict(d) for d in snap]
        lg_single, _ = _step(tgt, CANDS[0], c1)
        rowA = lg_single[0, -1].detach()        # 位置 dec_pos
        # ---- 分支 ②：block 验证（s=4，SpecDec 形状）→ rows[0] ----
        with torch.inference_mode():
            c2 = [dict(d) for d in snap]
            rows, cache_blk = spec_dec._verify(tgt, c2, CANDS)
        rowB = rows[0].detach()                 # 同一位置 dec_pos
        fp_after = _fp(snap)

        # 位置口径：喂入 cands[0] 后，该 forward 的 logits 位于**位置 clen**（= 8222）；
        # 而 `lg[0,-1]` 正是这一行。`repro.decision_position` 记的是**上一个**决策（产出 572 的那个，
        # 位置 8221）—— 两者必须分开写，不能混。
        rec["same_cache_two_shapes"] = dict(
            position=clen, position_note="喂入 cands[0] 后 logits 所在位置 = 8222；"
                                  "block 的 4 行占据 8222..8225（由 block_cache_len_after=8226 推得）",
            cache_len_before=clen,
            rowA_role="single-step cands[0] (s=1)", rowB_role="block verify rows[0] (s=4)",
            rowA=_row_stats(rowA), rowB=_row_stats(rowB),
            max_abs_diff=float((rowA.float() - rowB.float()).abs().max()),
            argmax_same=bool(int(rowA.argmax()) == int(rowB.argmax())),
            snapshot_unchanged=(fp_before == fp_after),
            block_cache_len_after=spec_cache.cache_length(cache_blk))
        s = rec["same_cache_two_shapes"]
        print(f"[divdiag] ① 单步(s=1) : top1={s['rowA']['top1']} marg={s['rowA']['margin']:.4f} "
              f"scores={s['rowA']['watch_scores']}")
        print(f"[divdiag] ② block(s=4): top1={s['rowB']['top1']} marg={s['rowB']['margin']:.4f} "
              f"scores={s['rowB']['watch_scores']}")
        print(f"[divdiag] 两行 max|Δ|={s['max_abs_diff']:.3e} argmax 相同={s['argmax_same']} "
              f"快照未被污染={s['snapshot_unchanged']}")

        # ---- （可选、明确标注）把**注意力路径**提到更高精度后重算这两行 ----
        if S.env_flag("DIAG_FP32_ATTN", True):
            res = {}
            for tag, preset in (("qk32", "qk32"), ("fp32_attn", "fp32")):
                M.set_diag_preset(preset)
                try:
                    c = [dict(d) for d in snap]
                    l1, _ = _step(tgt, CANDS[0], c)
                    with torch.inference_mode():
                        r, _ = spec_dec._verify(tgt, [dict(d) for d in snap], CANDS)
                    res[tag] = dict(preset=M.diag_precision(),
                                    single_top1=int(l1[0, -1].argmax()),
                                    block_top1=int(r[0].argmax()),
                                    single_scores=_scores(l1[0, -1].detach(), WATCH),
                                    block_scores=_scores(r[0].detach(), WATCH))
                    print(f"[divdiag] {tag}: single_top1={res[tag]['single_top1']} "
                          f"block_top1={res[tag]['block_top1']}")
                finally:
                    M.set_diag_preset("eager")
            rec["attention_precision_raised"] = dict(
                variants=res,
                caveat="**只**提高 attention 的 QK/概率精度，**不是**整个模型的 FP32 真值；"
                       "用于看「决策是否对该精度敏感」，不构成真值对照")
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        print(f"[divdiag] RESULT_JSON={p}")
        return 0
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        rec["alloc"] = P5V.allocator_state()
        p = _write(rec, run_id)
        print(f"[divdiag] !! 崩溃（已落盘 {p}）：\n{rec['traceback']}")
        print(f"[divdiag] RESULT_JSON={p}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
