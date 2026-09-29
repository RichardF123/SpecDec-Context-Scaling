"""g5_f06_index25_forensics.py — 定位 F06 / base_bf16 首次 AR–Spec 分歧（index 25）的具体来源。

固定案例
  family **F06**、target KV **base_bf16**、ctx 8192 / chunk 2048 / k 4 / **max_new 64**、
  greedy、`cudnn_prefill`（仅 prefill 融合）、**QK32/prob32 关闭**。
  上一轮首个分歧：**生成 index 25**，AR token=**715**，Spec token=**576**。

纪律（对应任务逐条）
  * **不缩短 max_new**：跑真实 64-token 语义，靠**观察钩子**在目标轮之后暂停
    （`_Captured`），与"减少预算提前结束"完全不同；
  * 快照一律 `diag_cache_cmp.deep_clone`（逐张量独立副本）并**立即转 CPU**，避免插桩 OOM；
  * 只保存**必要层/行**：只对目标窗口内的轮次做 cache 快照，不做全量转存；
  * 决策行必须来自**真正决定该 token 的那次调用**：若来自 `saved_next`，就回到**产生它的那次调用**
    （上一轮的 block 行或 repair `_step`），**不得**用后来重算的行替代；
  * 同历史比较只在**同一份 cache、同一预测位置、同一可见范围**上比 单步 vs 块。

模式
  PROBE_MODE=ar      真实 AR（自由生成）到目标位置，捕获决策行 + 决策前 cache
  PROBE_MODE=spec    真实 SpecDec（max_new=64）到决定 index 25 的那一轮，捕获证据后暂停
  PROBE_MODE=compare 纯 CPU：比较两侧真实决策行、同历史单步-块、cache 历史
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import diag_fields as DF
from . import family_input as FI
from . import model_self as M
from . import spec_cache
from . import spec_dec
from . import status_fields as SF
from .smoke_checks import CheckRegistry
from .smoke_hooks import DecisionHooks
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

MODE = os.environ.get("PROBE_MODE", "spec").strip().lower()
FAMILY = os.environ.get("PROBE_FAMILY", "F06")
KV = os.environ.get("PROBE_KV", "base_bf16")
TAG = os.environ.get("PROBE_TAG", "f06i25")
CTX = int(os.environ.get("PROBE_CTX", "8192"))
CHUNK = int(os.environ.get("PROBE_CHUNK", "2048"))
MAXNEW = int(os.environ.get("PROBE_MAXNEW", "64"))
K = int(os.environ.get("PROBE_K", "4"))
ARM = os.environ.get("PROBE_ARM", "cudnn_prefill")
DIV_INDEX = int(os.environ.get("PROBE_DIV_INDEX", "25"))
EXPECT_AR = int(os.environ.get("PROBE_EXPECT_AR", "715"))
EXPECT_SPEC = int(os.environ.get("PROBE_EXPECT_SPEC", "576"))
# 决定 index i 的那一行的**输入位置** = prompt_len + i - 1；其 `_run` 之前 cache 长度同值
BOUNDARY = CTX + DIV_INDEX - 1                    # 8216
WIN_LO, WIN_HI = BOUNDARY - 3, BOUNDARY + 3       # 只对窗口内的轮次做快照（省显存）

assert KV == "base_bf16", "本轮只定位 base_bf16"
assert MAXNEW == 64 and K == 4, "不得缩短 max_new / 改 k"


class _Captured(Exception):
    pass


def _p(kind: str, run_id: str) -> Path:
    return OUTDIR / f"{TAG}_{FAMILY}_{MODE}_{run_id}.{kind}"


def _write(rec: dict, run_id: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _p("json", run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _row(t: torch.Tensor) -> dict:
    r = t.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=[EXPECT_AR, EXPECT_SPEC])
    t2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(t2[0] - t2[1])
    f["tie_at_top"] = bool(float(t2[0]) == float(t2[1]))
    return f


def _cpu(cache):
    """独立 CPU 副本（深拷贝逐张量），用于落盘与事后比较。"""
    if cache is None:
        return None
    return [{k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in d.items()}
            for d in cache]


def _meta(cache, where: str) -> dict:
    if cache is None:
        return dict(where=where, none=True)
    layers = []
    for i, d in enumerate(cache):
        e = dict(layer=i, fmt=d.get("fmt"), tensor_keys=CC.tensor_keys(d))
        for k in CC.tensor_keys(d):
            e[k] = dict(dtype=str(d[k].dtype), shape=[int(x) for x in d[k].shape])
        layers.append(e)
    return dict(where=where, length=spec_cache.cache_length(cache), n_layers=len(cache),
                layers=layers,
                all_layers_same_len=len({int(d["k"].shape[2]) for d in cache}) == 1,
                note="**全层**记录；长度一致性由 all_layers_same_len 显式给出")


def _mask_evidence(s: int, T: int) -> dict:
    """当前这次前向的 mask 形状与**有效可见范围**（用引擎自己的 `_kv_keep_mask` 口径）。"""
    keep = M._kv_keep_mask(s, T, torch.device("cpu"))
    vis = [int(keep[i].sum()) for i in range(min(s, 8))]
    return dict(s=s, T=T, p=T - s,
                mask_shape=[int(x) for x in keep.shape],
                mask_dtype=str(keep.dtype),
                row_visible_counts_head=vis,
                row0_last_visible_key=int(T - s) if s > 0 else None,
                rule="query i 只能看见 key_position <= (T - s) + i（与 eager 路径同义）",
                eager_triu_diagonal=(1 if s == 1 else (T - s) + 1),
                s1_no_mask_applied=(s == 1))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        run_id=run_id, tool="g5_f06_index25_forensics", mode=MODE, family=FAMILY, kv=KV,
        tag=TAG, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=S.git_meta(ROOT),
        config=dict(ctx=CTX, chunk=CHUNK, k=K, max_new=MAXNEW, greedy=True,
                    attention_strategy=ARM, target_kv=KV, draft_kv="base_bf16",
                    div_index=DIV_INDEX, boundary_cache_len=BOUNDARY,
                    expect=dict(ar=EXPECT_AR, spec=EXPECT_SPEC),
                    stop_policy="观察钩子暂停（**不**缩短 max_new）"),
        scope="只定位 F06/base_bf16 首个分歧 index 25 的来源；不改精度/算法/门槛，不做性能结论。")
    print(f"[f25] run_id={run_id} mode={MODE} family={FAMILY} kv={KV}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "compare":
        return _compare(rec, run_id)

    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    print(f"[f25] commit: {why}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        _write(rec, run_id)
        return 3

    hooks = None
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")     # QK32/prob32 关闭
        from .lowmem_load import load_ours_bf16
        tgt, tok, tinfo = load_ours_bf16(P5V.TARGET)
        drf = None
        if MODE == "spec":
            drf, _t2, _di = load_ours_bf16(P5V.DRAFT)
            drf.set_kv_format("fp16")
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        prompt_len = int(ids.shape[1])
        rec["input"] = dict(ctx=prompt_len, family=fam_meta, eos=FI.eos_config(tok),
                            ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]))
        assert prompt_len + DIV_INDEX - 1 == BOUNDARY, "位置自洽检查失败"
        print(f"[f25] family={FAMILY} hash={fam_meta['token_sha1']} boundary={BOUNDARY}")

        reg = CheckRegistry()
        torch.cuda.reset_peak_memory_stats()
        inner = P5V.timed_chunked_prefill({}, CHUNK, tgt)
        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner)
        hooks.install()

        held: dict = {}
        _ov, _os = spec_dec._verify, spec_dec._step
        rounds: list = []

        if MODE == "spec":
            def _verify_spy(model, cache, cands, *a, **k):
                pre = spec_cache.cache_length(cache)
                inwin = WIN_LO <= pre <= WIN_HI
                snap = CC.deep_clone(cache) if inwin else None
                rows, blk = _ov(model, cache, cands, *a, **k)
                if inwin:
                    # ⚠️ 张量**不能**放进 JSON 记录：`json.dumps(default=str)` 会把它变成字符串，
                    #    精确数值就丢了（首跑就是这样丢掉 SpecDec 决策行的）。
                    #    张量一律留在 held 里，最后用 torch.save 落盘；JSON 只放摘要。
                    held.setdefault("rounds", []).append(dict(
                        round=reg.round, pre=pre, Kp=len(cands),
                        cands=[int(x) for x in cands],
                        rows_t=rows.detach().float().cpu(),
                        snap_pre_cpu=_cpu(snap), blk_cpu=_cpu(blk),
                        mask=_mask_evidence(len(cands), pre + len(cands))))
                if "decided" in held:
                    raise _Captured()                     # 目标轮之后的第一轮就停
                return rows, blk

            def _step_spy(model, token, cache, *a, **k):
                len_before = spec_cache.cache_length(cache)
                lg, c = _os(model, token, cache, *a, **k)
                if model is tgt and "decided" not in held:
                    held.setdefault("repairs", []).append(dict(
                        token=int(token), cache_len_before=len_before,
                        cache_len_after=spec_cache.cache_length(c),
                        row=lg[0, -1].detach().float().cpu()))
                return lg, c

            spec_dec._verify, spec_dec._step = _verify_spy, _step_spy
            try:
                spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K,
                                            eos_id=tok.eos_token_id,
                                            target_kv=KV, draft_kv="base_bf16",
                                            log=lambda d: _on_round(rounds, held, d))
            except _Captured:
                pass
            finally:
                spec_dec._verify, spec_dec._step = _ov, _os
                hooks.remove()
                hooks = None
            rec["rounds"] = rounds
            wr = held.get("rounds", [])
            rec["window_rounds"] = [dict(round=w["round"], pre=w["pre"], Kp=w["Kp"],
                                         cands=w["cands"], mask=w["mask"],
                                         rows_summary=[_row(r) for r in w["rows_t"]],
                                         snap_pre_meta=_meta(w["snap_pre_cpu"], "snap_pre"),
                                         blk_meta=_meta(w["blk_cpu"], "blk"))
                                    for w in wr]
            pt = _p("cache.pt", run_id)
            torch.save({f"round{w['round']}_snap_pre": w["snap_pre_cpu"] for w in wr} |
                       {f"round{w['round']}_blk": w["blk_cpu"] for w in wr}, str(pt))
            rec["cache_dump"] = str(pt)
            pt2 = _p("rows.pt", run_id)
            torch.save({f"round{w['round']}_rows": w["rows_t"] for w in wr}, str(pt2))
            rec["rows_dump"] = str(pt2)
            rec["repairs"] = [dict(token=r["token"], cache_len_before=r["cache_len_before"],
                                   cache_len_after=r["cache_len_after"], row=_row(r["row"]))
                              for r in held.get("repairs", [])]
            dec = _resolve_decision(rec, held)
            rec["decision"] = dec
        else:
            # ---- 真实 AR：自由生成到 BOUNDARY，再走**自然**的一步 ----
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, cache = inner(tgt, ids)
            reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg",
                             where="AR prefill")
            nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            gen = [nxt]
            fed = []
            while spec_cache.cache_length(cache) < BOUNDARY:
                with torch.inference_mode(), M.attention_phase("decode"):
                    lg, cache = tgt._run(
                        tgt.embed_tokens(torch.tensor([[nxt]], device=ids.device)),
                        cache, need_logits=True)
                reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                                 round_=len(gen), step=len(gen), where="AR 单步")
                fed.append(nxt)
                nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
                gen.append(nxt)
            clen = spec_cache.cache_length(cache)
            snap = CC.deep_clone(cache)
            meta = _meta(snap, "ar_cache_before_deciding_step")
            tok_fed = nxt                       # gen[BOUNDARY-prompt_len] = gen[idx]
            with torch.inference_mode(), M.attention_phase("decode"):
                lg, _c = tgt._run(
                    tgt.embed_tokens(torch.tensor([[tok_fed]], device=ids.device)),
                    cache, need_logits=True)
            reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                             round_=len(gen), step=len(gen) + 1,
                             where="AR 真正决定 index 25 的那一步")
            real = lg[0, -1].detach().clone()
            rec["ar"] = dict(cache_len_before=clen, expected_len=BOUNDARY,
                             fed_token=tok_fed, fed_token_position=clen,
                             predicted_position=clen + 1, gen_index=clen - prompt_len + 1,
                             gen_head=gen[:DIV_INDEX + 1],
                             real_row=_row(real),
                             real_argmax=int(real.argmax()),
                             matches_expected=(int(real.argmax()) == EXPECT_AR),
                             mask=_mask_evidence(1, clen + 1))
            rec["cache_meta"] = meta
            # 同入口重放（证明重放等于真实行）
            work = CC.deep_clone(snap)
            with torch.inference_mode(), M.attention_phase("repair"):
                lg2, _n = spec_dec._step(tgt, tok_fed, work)
            rep = lg2[0, -1].detach().clone()
            cmpr = DF.compare_rows(real, rep)
            rec["replay"] = dict(entry="spec_dec._step(model, token, cache)",
                                 elementwise_equal=cmpr["elementwise_equal"],
                                 abs_err_max=cmpr["abs_err_max"], argmax_same=cmpr["argmax_same"],
                                 replay_row=_row(rep))
            ref_cpu = _cpu(snap)
            cp = _p("cache.pt", run_id)
            torch.save(ref_cpu, str(cp))
            rec["cache_dump"] = str(cp)
            same = _cpu(snap)
            rec["snapshot_integrity"] = CC.integrity_check(ref_cpu, same)
            rp = _p("rows.pt", run_id)
            torch.save(dict(real=real.detach().float().cpu(), replay=rep.detach().float().cpu()),
                       str(rp))
            rec["rows_dump"] = str(rp)
            hooks.remove()
            hooks = None
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        print(f"[f25] RESULT_JSON={p}")
        return 0
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        try:
            if hooks is not None:
                hooks.remove()
        except Exception:                                        # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        p = _write(rec, run_id)
        print(f"[f25] !! 崩溃（已落盘 {p}）：\n{rec['traceback'][-900:]}")
        return 2


def _on_round(rounds: list, held: dict, d: dict):
    """记录轮次；一旦本轮已覆盖 DIV_INDEX，就标记 decided（下一轮 verify 入口暂停）。"""
    rounds.append(dict(d))
    so_far = sum(len(r["emitted"]) for r in rounds)
    s0 = rounds[-1]["pre"] - CTX
    if so_far > DIV_INDEX and s0 <= DIV_INDEX:
        held["decided"] = dict(round=d["round"], s0=s0)


def _resolve_decision(rec: dict, held: dict) -> dict:
    """确定**真正决定** index 25 的那一行来自哪次调用（block 行 / prefill / saved_next → 回到其来源）。"""
    rounds = rec["rounds"]
    pos, cur = 0, None
    for r in rounds:
        n = len(r["emitted"])
        if pos <= DIV_INDEX < pos + n:
            cur = (r, DIV_INDEX - pos)
            break
        pos += n
    if cur is None:
        return dict(error="DIV_INDEX 不在任何轮次的 emitted 覆盖范围内")
    r, slot = cur
    out = dict(deciding_round=r["round"], deciding_slot=slot, pre=r["pre"], Kp=r["Kp"], m=r["m"],
               cands=r["cands"], g=r["g"], emitted=r["emitted"],
               emitted_token=int(r["emitted"][slot]))
    prev = [x for x in rounds if x["round"] == r["round"] - 1]
    if slot >= 1:
        out.update(source_kind="block_row",
                   source_detail=f"本轮 verify 的 rows[{slot - 1}]",
                   predicted_position=r["pre"] + slot,
                   input_position=r["pre"] + slot - 1,
                   note="rows[j] 的 logits 位于输入位置 pre+j，预测 pre+j+1")
    else:
        p = prev[0] if prev else None
        if p is None:
            out.update(source_kind="prefill_saved_next", source_detail="首轮 ⇒ prefill 结果",
                       predicted_position=r["pre"], input_position=r["pre"] - 1)
        elif p["m"] == p["Kp"]:
            out.update(source_kind="saved_next_from_prev_block_all_accept",
                       source_detail=f"上一轮(round={p['round']}) 全接受 ⇒ saved_next = 其 rows[Kp-1]",
                       predicted_position=p["pre"] + p["Kp"], input_position=p["pre"] + p["Kp"] - 1)
        else:
            out.update(source_kind="saved_next_from_prev_repair",
                       source_detail=f"上一轮(round={p['round']}) 有否决(m={p['m']}) ⇒ saved_next = 其 repair 的 _step",
                       predicted_position=p["pre"] + p["m"] + 1, input_position=p["pre"] + p["m"])
    return out


def _compare(rec: dict, run_id: str) -> int:
    import glob
    found = {}
    for m in ("ar", "spec"):
        js = sorted(glob.glob(str(OUTDIR / f"{TAG}_{FAMILY}_{m}_*.json")))
        if js:
            found[m] = json.loads(Path(js[-1]).read_text(encoding="utf-8"))
    rec["sources"] = {m: r.get("run_id") for m, r in found.items()}
    if len(found) != 2 or any(r.get("stage") != "complete" for r in found.values()):
        rec["stage"] = "missing_source"
        rec["source_stages"] = {m: r.get("stage") for m, r in found.items()}
        p = _write(rec, run_id)
        print(f"[f25] !! 来源缺失/未完成 ⇒ 停下 -> {p}")
        return 5
    a, b = found["ar"], found["spec"]
    ra = torch.load(sorted(glob.glob(str(OUTDIR / f"{TAG}_{FAMILY}_ar_*.rows.pt")))[-1],
                    map_location="cpu")
    ca = torch.load(sorted(glob.glob(str(OUTDIR / f"{TAG}_{FAMILY}_ar_*.cache.pt")))[-1],
                    map_location="cpu")
    rowcmp = dict(DF.compare_rows(ra["real"], ra["replay"]))
    out = dict(
        reproduction=dict(
            ar_expected=EXPECT_AR, ar_actual=a["ar"]["real_argmax"],
            ar_match=a["ar"]["real_argmax"] == EXPECT_AR,
            spec_expected=EXPECT_SPEC, spec_actual=b["decision"]["emitted_token"],
            spec_match=b["decision"]["emitted_token"] == EXPECT_SPEC,
            ar_common_prefix=(a["ar"]["gen_head"][:DIV_INDEX]
                              if len(a["ar"]["gen_head"]) >= DIV_INDEX else None),
        ),
        ar_replay_vs_real=dict(**rowcmp, replay_row=a["replay"]["replay_row"],
                               note="必须 elementwise_equal=True 才能用重放做同历史对照"),
        spec_decision=b["decision"],
        ar_decision=dict(source_kind="ar_decode_single_step",
                         input_position=a["ar"]["fed_token_position"],
                         predicted_position=a["ar"]["predicted_position"],
                         fed_token=a["ar"]["fed_token"], real_row=a["ar"]["real_row"]),
        mask_evidence=dict(ar=a["ar"]["mask"], spec_window=[w["mask"] for w in b["window_rounds"]]),
        cache_len=dict(ar=a["ar"]["cache_len_before"], ar_expected=a["ar"]["expected_len"],
                       spec_window=[w["pre"] for w in b["window_rounds"]]),
        snapshot_integrity=dict(ar=a.get("snapshot_integrity")),
        stage="complete")
    hist = CC.compare_caches(ca, b["window_rounds"][-1]["snap_pre_cpu"] if b["window_rounds"]
                             else None)
    out["cache_history_compare"] = hist
    p = _write(rec, run_id)
    print(f"[f25] AR argmax={out['reproduction']['ar_actual']} (expect {EXPECT_AR}) "
          f"match={out['reproduction']['ar_match']}")
    print(f"[f25] Spec emitted={out['reproduction']['spec_actual']} (expect {EXPECT_SPEC}) "
          f"match={out['reproduction']['spec_match']}")
    print(f"[f25] Spec 决策来源: {b['decision'].get('source_kind')} @ {b['decision'].get('source_detail')}")
    print(f"[f25] AR 重放==真实: {rowcmp['elementwise_equal']} max|Δ|={rowcmp['abs_err_max']}")
    print(f"[f25] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
