"""prefill_sb_replay_f06.py — 初始状态 × 后续计算路径的 2×2 对照（F06/index25，base_bf16）。

    PREFSB_ARM=cudnn_prefill（生产） | eager（干预组）
    同一脚本、同一代码路径；唯一差别是 prefill backend。

实验语义（反事实重放）
  * 固定 F06 / base_bf16 / ctx=8192 / chunk=2048 / k=4；使用**首次分歧前的共同 token 前缀**；
  * **不加载 draft、不自由生成**：B 臂吃**记录下来的候选与 repair 日程**，
    因此它是「**固定日程的反事实重放**」，**不是** eager 自由运行的真实 Spec；
  * 两臂从**同一 eager（或同一 cuDNN）prefill 状态**的独立深拷贝出发；
  * 对照**止于预测位置 8217**：不喂入分歧 token，不越过上限继续归因。

验收门（**分组**）
  * cuDNN 生产重放：保留**原始复现门**（S@8217 == 715；B 第 11 轮 rows[1] == 576；12 轮 g 全等）。
  * eager 干预组：**不要求**复现旧 cuDNN 的 token 或 g；但仍必须通过
    有限性 / 结构 / 位置 / 输入来源 / 初始状态独立性 / 确定性 / 全层 KV 硬检查。
    与旧记录的差异**完整落盘**（观测结果，不是失败）。

对齐与可比性由 `sb_align` 提供：每行保留**来源**（prefill/verify/repair/serial）、
可见 token 历史与 cache 长度；前缀与共同前缀不一致的行（含**被拒候选之后**的行）
标为**不可比较**并**不进入**误差统计。
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import model_self as M
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import family_input as FI
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
CTX, CHUNK, K = 8192, 2048, 4
PROMPT = 8192
LIMIT = 8217                     # 共同前缀对照上限（预测位置）
OLD_AR_S, OLD_SPEC_B = 715, 576  # 旧 cuDNN 生产记录（仅 cuDNN 组当门；eager 组只作观测）
ARM = os.environ.get("PREFSB_ARM", "cudnn_prefill")
STRICT = (ARM == "cudnn_prefill")


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cache_len(c) -> int:
    return int(spec_cache.cache_length(c))


def _row_fields(row_t) -> dict:
    r = row_t.detach().float().reshape(-1)
    t2 = torch.topk(r, 2).values
    am = int(torch.argmax(r))
    tied = [int(x) for x in torch.nonzero(r == r.amax()).reshape(-1)]
    return dict(argmax_id=am, max_score=float(r[am]), n_max_tied=len(tied), tied_max_tokens=tied,
                margin=float(t2[0] - t2[1]),
                watch={str(t): float(r[t]) for t in (OLD_AR_S, OLD_SPEC_B, 4710)},
                finite=bool(torch.isfinite(r).all()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="prefill_sb_replay_f06", arm=ARM, strict_gates=STRICT,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", ctx=CTX, chunk=CHUNK, k=K,
                           arm=ARM, prompt_len=PROMPT, limit_pred_pos=LIMIT,
                           diag_precision="qk_fp32=False, prob_fp32=False"),
               semantics=dict(kind="固定日程反事实重放",
                              note="B 臂使用**记录下来的**候选与 accept/repair 日程；"
                                   "**不是** eager 自由运行的真实 Spec 轨迹；两臂喂同一串共同前缀 token；"
                                   "对照止于预测位置 8217"),
               scope="2×2 的 S/B 对照（同一初始状态、同一 token 前缀）；不加载 draft、不自由生成；"
                     "不改生产实现/精度/KV/门槛；无性能结论。")
    print(f"[sb] run_id={run_id} arm={ARM} strict={STRICT}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[sb] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_prefillsb_{ARM}_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        A.require(rec["diag_precision"]["qk_fp32"] is False and
                  rec["diag_precision"]["prob_fp32"] is False, "精度开关必须关闭")

        spec = json.load(open(_newest(f"{TAG}_{FAMILY}_spec_*.json"), encoding="utf-8"))
        ar = json.load(open(_newest(f"{TAG}_{FAMILY}_ar_*.json"), encoding="utf-8"))
        rounds = spec["rounds"]
        emitted = [t for r in rounds for t in r["emitted"]]
        A.require(len(rounds) == 12, "记录轮数应为 12", str(len(rounds)))
        A.require(len(emitted) == 26, "记录里 emitted 应为 26 个 token（index 0..25）", str(len(emitted)))
        feed_seq = emitted[:25]
        A.require(feed_seq == [int(x) for x in ar["ar"]["gen_head"]],
                  "前 25 个 emitted 必须等于 AR 的 gen_head", f"{feed_seq} vs {ar['ar']['gen_head']}")
        common = {PROMPT + i: feed_seq[i] for i in range(len(feed_seq))}     # 8192..8216
        rec["trajectory"] = dict(n_rounds=len(rounds), feed_seq=feed_seq,
                                 divergent_26th=emitted[25], common_positions=sorted(common),
                                 limit_pred_pos=LIMIT,
                                 pre_list=[r["pre"] for r in rounds],
                                 m_list=[r["m"] for r in rounds],
                                 cands=[list(r["cands"]) for r in rounds])

        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy(ARM)
        A.require(M.attention_strategy() == ARM, "attention 策略设置失败", str(M.attention_strategy()))
        M.reset_attention_stats()
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        A.require(int(ids.shape[1]) == PROMPT, "prompt 长度必须 8192", str(int(ids.shape[1])))
        rec["input"] = dict(family=fam_meta, ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]))
        inner = P5V.timed_chunked_prefill({}, CHUNK, tgt)

        # ---------------- 1) 规范初始状态（两次独立建立 ⇒ 确定性；全层硬检查） ----------------
        def build(tag_):
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, c = inner(tgt, ids)
            A.require(_cache_len(c) == PROMPT, f"{tag_}: prefill 后 cache 长度必须 8192",
                      str(_cache_len(c)))
            info = A.require_full_cache(c, expect_len=PROMPT, n_layers=28, what=f"{tag_} 全层 KV")
            return lg, c, info

        lg0, P0, fp0 = build("P0")
        lg0b, P0b, fp0b = build("P0'")
        A.require(bool(torch.equal(lg0.detach(), lg0b.detach())), "两次建立的 next logits 不逐位相同")
        same_dig = [k for k in fp0 if fp0[k]["sha"] == fp0b[k]["sha"]]
        A.require(len(same_dig) == len(fp0), "两次建立的全层 KV 稳定摘要不一致",
                  str([k for k in fp0 if fp0[k]["sha"] != fp0b[k]["sha"]][:6]))
        first = _row_fields(lg0[0, -1])
        A.require(first["argmax_id"] == feed_seq[0], "prefill 首 token 必须等于记录的首 token",
                  str(first["argmax_id"]))
        rec["initial_state"] = dict(cache_len=PROMPT, n_layers=len(P0), first_token=first,
                                    layer_digests=fp0,
                                    determinism=dict(logits_bitwise_equal=True,
                                                     n_layer_digest_equal=len(same_dig),
                                                     n_layer_digest_total=len(fp0)))
        print(f"[sb] 初始状态 ✔ cache={PROMPT} 首 token={first['argmax_id']}；"
              f"两次建立 logits 逐位相同、全层摘要 {len(same_dig)}/{len(fp0)} 一致")

        # ---------------- 2) 两臂（同一份 P0 的独立深拷贝） ----------------
        t0 = time.perf_counter()
        cacheS = CC.deep_clone(P0)
        cacheB = CC.deep_clone(P0)
        rec["initial_state_independence"] = A.initial_state_independent(cacheS, cacheB)
        print("[sb] 起点独立性与逐元素相同 ✔")

        # ---- S 臂：teacher forcing 逐 token（止于 LIMIT）----
        s_rows, saved_next, s_trace = [], lg0[0, -1].detach().clone(), []
        s_rows.append(A.make_row(pred_pos=PROMPT, n_visible=PROMPT, cache_len_after=PROMPT,
                                 source="prefill", detail="shared", hist=[],
                                 row=lg0[0, -1]))
        for i, tk_ in enumerate(feed_seq):
            in_pos = PROMPT + i
            pred_pos = in_pos + 1
            if pred_pos > LIMIT:
                break
            with torch.inference_mode(), M.attention_phase("decode"):
                lg, cacheS = spec_dec._step(tgt, tk_, cacheS)
            want = A.serial_expected_len(PROMPT, i + 1)
            A.require(_cache_len(cacheS) == want,
                      f"S 臂 step={i} cache 长度应为 {want}", str(_cache_len(cacheS)))
            A.require_full_cache(cacheS, expect_len=want, n_layers=28, what=f"S step={i} 全层 KV")
            hist = [(PROMPT + j, feed_seq[j]) for j in range(i + 1)]
            row = A.make_row(pred_pos=pred_pos, n_visible=pred_pos, cache_len_after=want,
                             source="serial", detail=f"step={i}", hist=hist, row=lg[0, -1])
            s_rows.append(row)
            s_trace.append(dict(step=i, fed_token=int(tk_), input_pos=in_pos, pred_pos=pred_pos,
                                cache_len=want, **_row_fields(lg[0, -1])))
        rec["S"] = dict(n_rows=len(s_rows), n_feed=len(s_trace), expected_final_len=A.serial_expected_len(PROMPT, len(s_trace)),
                        final_cache_len=_cache_len(cacheS),
                        trace=s_trace, row_inventory=[
                            dict(pred_pos=r["pred_pos"], source=r["source"], detail=r["detail"],
                                 n_visible=r["n_visible"], cache_len_after=r["cache_len_after"],
                                 argmax=r["argmax"], hist_len=len(r["hist"]), row_sha=r["row_sha"])
                            for r in s_rows])
        A.require(_cache_len(cacheS) == rec["S"]["expected_final_len"],
                  "S 臂最终 cache 长度与期望不符",
                  f"{_cache_len(cacheS)} vs {rec['S']['expected_final_len']}")

        # ---- B 臂：按记录的候选与 accept/repair 日程重放（止于 LIMIT）----
        b_rows, b_trace, g_mismatch = [], [], []
        for r in rounds:
            r0, pre, kp, m = int(r["round"]), int(r["pre"]), int(r["Kp"]), int(r["m"])
            cands = [int(x) for x in r["cands"]]
            A.require(_cache_len(cacheB) == pre, f"B 臂第 {r0} 轮起点 cache 长度不符",
                      f"{_cache_len(cacheB)} vs {pre}")
            A.require_full_cache(cacheB, expect_len=pre, n_layers=28, what=f"B round{r0} 起点")
            with torch.inference_mode(), M.attention_phase("verify"):
                rows, blk = spec_dec._verify(tgt, cacheB, cands)
            A.require(rows.dim() == 2 and int(rows.shape[0]) == kp, "rows 形状异常", str(list(rows.shape)))
            g_mine = [int(saved_next.reshape(-1).argmax())] + \
                     [int(rows[j].reshape(-1).argmax()) for j in range(kp - 1)]
            if g_mine != [int(x) for x in r["g"]]:
                g_mismatch.append(dict(round=r0, g_mine=g_mine, g_recorded=[int(x) for x in r["g"]]))
            prefix_hist = [(PROMPT + j, feed_seq[j]) for j in range(pre - PROMPT)]
            for j in range(kp):
                hist = prefix_hist + [(pre + i, cands[i]) for i in range(j + 1)]
                b_rows.append(A.make_row(pred_pos=pre + j + 1, n_visible=pre + j + 1,
                                         cache_len_after=pre + kp, source="verify",
                                         detail=f"round={r0},j={j}", hist=hist, row=rows[j]))
            cacheB = spec_cache.trim_cache(blk, pre + m)
            b_trace.append(dict(round=r0, pre=pre, Kp=kp, m=m, cands=cands, g_mine=g_mine,
                                g_recorded=[int(x) for x in r["g"]],
                                cache_len_after=_cache_len(cacheB),
                                rows_argmax=[int(rows[j].reshape(-1).argmax()) for j in range(kp)]))
            if m < kp:
                r_tok = int(r["emitted"][-1])
                if pre + m + 1 > LIMIT:
                    b_trace[-1]["repair_skipped"] = f"pred_pos {pre + m + 1} > 上限 {LIMIT}（不喂分歧 token）"
                    break
                with torch.inference_mode(), M.attention_phase("repair"):
                    saved_next, cacheB = spec_dec._step(tgt, r_tok, cacheB)
                want = pre + m + 1
                A.require(_cache_len(cacheB) == want, f"B 臂 repair 后长度应为 {want}",
                          str(_cache_len(cacheB)))
                hist = prefix_hist + [(pre + i, cands[i]) for i in range(m)] + [(pre + m, r_tok)]
                b_rows.append(A.make_row(pred_pos=want, n_visible=want, cache_len_after=want,
                                         source="repair", detail=f"round={r0}", hist=hist,
                                         row=saved_next[0, -1]))
            else:
                saved_next = rows[kp - 1].detach().clone()
        rec["B"] = dict(n_rows=len(b_rows), n_rounds=len(b_trace), g_mismatch=g_mismatch,
                        final_cache_len=_cache_len(cacheB), trace=b_trace,
                        row_inventory=[
                            dict(pred_pos=r["pred_pos"], source=r["source"], detail=r["detail"],
                                 n_visible=r["n_visible"], cache_len_after=r["cache_len_after"],
                                 argmax=r["argmax"], hist_len=len(r["hist"]), row_sha=r["row_sha"])
                            for r in b_rows],
                        note="repair 日程与接受长度**取自旧记录**（固定日程干预），"
                             "不是 eager 自身验证得到的结果")

        # ---------------- 3) 分组验收门 ----------------
        s_at = {r["pred_pos"]: r for r in s_rows}
        b_at: dict = {}
        for r in b_rows:
            b_at.setdefault(r["pred_pos"], []).append(r)
        obs = dict(
            S_at_8217=dict(argmax=s_at[LIMIT]["argmax"] if LIMIT in s_at else None,
                           old_cudnn=OLD_AR_S),
            B_round11_rows1=dict(argmax=(b_at[LIMIT][0]["argmax"] if LIMIT in b_at else None),
                                 old_cudnn=OLD_SPEC_B),
            n_rounds_with_g_mismatch=len(g_mismatch))
        rec["observed_vs_old_cudnn"] = obs
        if STRICT:
            A.require(obs["S_at_8217"]["argmax"] == OLD_AR_S,
                      f"cuDNN 复现门失败：S @8217 应 == {OLD_AR_S}", str(obs["S_at_8217"]))
            A.require(obs["B_round11_rows1"]["argmax"] == OLD_SPEC_B,
                      f"cuDNN 复现门失败：B round11 rows[1] 应 == {OLD_SPEC_B}",
                      str(obs["B_round11_rows1"]))
            A.require(not g_mismatch, "cuDNN 复现门失败：某轮 g 与记录不符", str(g_mismatch[:2]))
            print(f"[sb] cuDNN 复现门 ✔ S@8217={OLD_AR_S}、B round11 rows[1]={OLD_SPEC_B}、12 轮 g 全等")
        else:
            print(f"[sb] eager 干预组（观测门）：S@8217={obs['S_at_8217']['argmax']}"
                  f"（旧 cuDNN={OLD_AR_S}）、B rows[1]={obs['B_round11_rows1']['argmax']}"
                  f"（旧 cuDNN={OLD_SPEC_B}）、g 不一致轮数={len(g_mismatch)}"
                  f" ⇒ 作为**观测结果**落盘，不作失败")

        # ---------------- 4) 对齐对照（保留来源；不可比者不入统计） ----------------
        cmp_res = A.compare_arms(s_rows, b_rows, common, limit=LIMIT)
        rec["compare"] = dict(
            limit=LIMIT, n_pairs=len(cmp_res["pairs"]), pairs=cmp_res["pairs"],
            incomparable=cmp_res["incomparable"],
            first_nonzero_logits_diff=cmp_res["first_nonzero"],
            first_argmax_divergence=cmp_res["first_argmax"],
            n_b_rows=len(b_rows), n_s_rows=len(s_rows),
            note="按预测位置配对并**保留行来源**；前缀与共同前缀不一致（含被拒候选之后）的行"
                 "标为不可比较且不入误差统计；不越过预测位置 8217")
        t2 = {}
        for tag_, rr in (("S", s_at.get(LIMIT)), ("B", (b_at.get(LIMIT) or [None])[0])):
            if rr is not None:
                t2[tag_] = dict(source=rr["source"], detail=rr["detail"],
                                fields=_row_fields(rr["row"]))
        rec["at_8217"] = t2
        att = M.attention_stats()
        rec["attention_stats"] = dict(strategy=att["strategy"], calls=att["calls"],
                                      fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att["s_max_by_phase"], arm=ARM)
        A.require(sorted(k for k, v in att["fused_by_phase"].items() if v > 0) ==
                  (["prefill"] if STRICT else []),
                  f"融合路径出现位置与 arm={ARM} 不符", str(att["fused_by_phase"]))
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_prefillsb_{ARM}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{TAG}_prefillsb_{ARM}_{run_id}.pt"
        torch.save(dict(
            prefill_row=s_rows[0]["row"],
            S_rows={str(r["pred_pos"]): r["row"] for r in s_rows},
            B_rows={f"{r['pred_pos']}_{r['source']}_{r['detail']}": r["row"] for r in b_rows},
            L27_K_P0=P0[27]["k"].detach().float().cpu(), L27_V_P0=P0[27]["v"].detach().float().cpu()),
            str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp), wall_s=time.perf_counter() - t0)
        print("== 对齐对照（保留来源） ==")
        for x in cmp_res["pairs"]:
            print(f"  pred {x['pred_pos']:5d} S[{x['s_source']}]={x['s_argmax']:6d} "
                  f"B[{x['b_source']}]={x['b_argmax']:6d} same={x['same_argmax']} "
                  f"max|Δ|={x['max_abs']:.6g} n_diff={x['n_diff']}")
        print(f"  配对 {len(cmp_res['pairs'])} 条；不可比/缺失 {len(cmp_res['incomparable'])} 条：")
        for x in cmp_res["incomparable"]:
            print(f"    pred {x['pred_pos']:5d} which={x['which']} reason={x['reason'][:70]}")
        print(f"  首次非零 logits 差异 @ {cmp_res['first_nonzero']}；"
              f"首次 argmax 分歧 @ {cmp_res['first_argmax']}")
        print(f"[sb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, A.AlignError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_prefillsb_{ARM}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[sb] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sb] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
