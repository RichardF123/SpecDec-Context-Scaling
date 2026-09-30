"""prefill_sb_replay_f06.py — 初始状态 × 后续计算方式的**最小直接对照**（F06/index25，base_bf16）。

回答三个假设：
  H1 配置不一致     → 由 `docs/ar_spec_config_audit_2026-09-30.md` 的代码+产物审计回答（本工具只做**结构检查**复核）；
  H2 初始状态不一致 → 同一次分块 prefill 建两次规范初始状态 P0 / P0'，**逐元素**比较（确定性 + 可复现性）；
  H3 相同初始状态下 S（串行单步）与 B（分块 verify + repair）是否分歧
                     → 从**同一份** P0 深拷贝出 S/B 两臂，teacher forcing 喂**同一串**已记录 token。

关键控制
  * **不加载 draft**：B 臂直接吃真实运行记录下来的 `cands`（verify 的输入）与 `emitted`（repair 的 token），
    因此复现的是**真实调用边界与 repair 操作**，不是「连续四 token」简化版。
  * 两臂喂入的 token 序列完全相同（teacher forcing），禁止自由生成后再比不同前缀。
  * 初始状态（完整 target KV + cache 长度 + next logits）由**同一实现**建立，深拷贝给 S/B，且不共享可变存储。
  * 复现门：S 臂在输入位置 8216 的 argmax 必须 == 真实 AR 的 **715**；
            B 臂第 11 轮 `rows[1]` 的 argmax 必须 == 真实 Spec 的 **576**；
            每轮 `g` 必须逐元素等于记录值。任一不满足 ⇒ 停止，不进入因果分析。
  * 只保存必要快照；不做性能声明；不改生产实现/精度/KV/门槛。
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
from . import diag_checks as DC
from . import model_self as M
from . import spec_cache
from . import spec_dec
from . import family_input as FI
from . import g5_8k_spec_variance_probe as P5V
from . import diag_cache_cmp as CC

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
CTX, CHUNK, K = 8192, 2048, 4
PROMPT = 8192
AR_AT_8216, SPEC_ROUND11_ROW1 = 715, 576
DECIDING_INPUT_POS = 8216
# 2×2 的 prefill backend 轴：PREFSB_ARM=cudnn_prefill（默认，生产）| eager（对照）
ARM = os.environ.get("PREFSB_ARM", "cudnn_prefill")


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _row_of(lg_row: torch.Tensor) -> torch.Tensor:
    """把 (V,) 或 (1,V)/(1,1,V) 归一为 (V,) fp32 CPU。"""
    return lg_row.detach().float().reshape(-1).cpu()


def _brief(lg_row) -> dict:
    r = _row_of(lg_row)
    t2 = torch.topk(r, 2).values
    am = int(torch.argmax(r))
    tied = [int(x) for x in torch.nonzero(r == r[am]).reshape(-1)]
    return dict(argmax_id=am, max_score=float(r[am]), n_max_tied=len(tied),
                margin=float(t2[0] - t2[1]), finite=bool(torch.isfinite(r).all()))


def _cache_len(c) -> int:
    return int(spec_cache.cache_length(c))


def _stable_sha(t: torch.Tensor) -> str:
    """**跨进程可比**的内容哈希（不用进程内 `hash()`，它带随机种子、跨进程不可比）。"""
    import hashlib
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _cache_fingerprint(c) -> dict:
    """全层 K/V 的逐层摘要（不落盘全量）：shape/dtype/finite/稳定内容 hash/sum。"""
    out = {}
    for i, d in enumerate(c):
        for kk in ("k", "v"):
            t = d[kk]
            out[f"L{i}_{kk}"] = dict(shape=[int(x) for x in t.shape], dtype=str(t.dtype),
                                     finite=bool(torch.isfinite(t).all()),
                                     sha=_stable_sha(t),
                                     sum=float(t.detach().float().sum()))
    return out


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="prefill_sb_replay_f06",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", ctx=CTX, chunk=CHUNK, k=K,
                           arm=ARM, diag_precision="qk_fp32=False, prob_fp32=False",
                           prompt_len=PROMPT, deciding_input_pos=DECIDING_INPUT_POS),
               scope="H2/H3 最小直接对照：同一规范初始状态下的串行 S vs 分块 B；不加载 draft；不跑生成；"
                     "不改生产实现/精度/KV/门槛；无性能结论。")
    print(f"[sb] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[sb] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_prefillsb_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        DC.require(rec["diag_precision"]["qk_fp32"] is False and
                   rec["diag_precision"]["prob_fp32"] is False, "精度开关必须关闭")

        spec_p = _newest(f"{TAG}_{FAMILY}_spec_*.json")
        ar_p = _newest(f"{TAG}_{FAMILY}_ar_*.json")
        spec = json.load(open(spec_p, encoding="utf-8"))
        ar = json.load(open(ar_p, encoding="utf-8"))
        rounds = spec["rounds"]
        DC.require(len(rounds) == 12, "记录轮数应为 12", str(len(rounds)))
        DC.require(all(r["Kp"] == K for r in rounds), "记录里 Kp 不全是 4")
        emitted = [t for r in rounds for t in r["emitted"]]
        rec["trajectory"] = dict(
            n_rounds=len(rounds), n_emitted=len(emitted), emitted=emitted,
            pre_list=[r["pre"] for r in rounds], m_list=[r["m"] for r in rounds],
            source=str(spec_p),
            note="teacher forcing 的 token 序列 = 真实运行记录的 `emitted` 串的**共同前缀**"
                 "（S/B 两臂完全相同）；第 26 个 token 是 Spec 臂在 index 25 的**分歧输出本身**，不喂给 S 臂")
        # 记录里 12 轮共 emit 26 个 token = index 0..25；共同前缀 = 前 25 个（index 0..24）
        DC.require(len(emitted) == 26, "记录里 emitted 应为 26 个 token（index 0..25）", str(len(emitted)))
        feed_seq = emitted[:25]
        DC.require(feed_seq == [int(x) for x in ar["ar"]["gen_head"]],
                   "前 25 个 emitted 必须等于 AR 的 gen_head（共同前缀）",
                   f"{feed_seq} vs {ar['ar']['gen_head']}")
        rec["trajectory"]["feed_seq_len"] = len(feed_seq)
        rec["trajectory"]["divergent_26th"] = emitted[25]

        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy(ARM)
        DC.require(M.attention_strategy() == ARM, "attention 策略设置失败", str(M.attention_strategy()))
        M.reset_attention_stats()
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        DC.require(int(ids.shape[1]) == PROMPT, "prompt 长度必须 8192", str(int(ids.shape[1])))
        rec["input"] = dict(family=fam_meta, ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]))
        inner = P5V.timed_chunked_prefill({}, CHUNK, tgt)

        # ---------------- 1) 规范初始状态 P0 / P0'（同一实现，两次独立建立） ----------------
        def build_state(tag_):
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, c = inner(tgt, ids)
            DC.require(_cache_len(c) == PROMPT, f"{tag_}: prefill 后 cache 长度必须 8192",
                       str(_cache_len(c)))
            DC.require(DC.finite_all(lg), f"{tag_}: prefill logits 非有限")
            return lg, c

        lg0, P0 = build_state("P0")
        first0 = _brief(lg0[0, -1])
        DC.require(first0["argmax_id"] == emitted[0],
                   "prefill 首 token 必须等于记录的首 token", str(first0))
        lg0b, P0b = build_state("P0'")
        first0b = _brief(lg0b[0, -1])
        same_logits = bool(torch.equal(lg0.detach(), lg0b.detach()))
        fp0, fp0b = _cache_fingerprint(P0), _cache_fingerprint(P0b)
        layer_eq = {k: bool(fp0[k]["sha"] == fp0b[k]["sha"] and fp0[k]["sum"] == fp0b[k]["sum"])
                    for k in fp0}
        DC.require(all(layer_eq.values()), "两次独立建立初始状态的逐层 K/V 摘要不一致",
                   str([k for k, v in layer_eq.items() if not v]))
        DC.require(same_logits, "两次独立建立初始状态的 next logits 不逐元素相同")
        rec["initial_state"] = dict(
            P0=dict(cache_len=PROMPT, layers=len(P0), first_token=first0,
                    per_layer=fp0, k_dtype=str(P0[0]["k"].dtype), k_shape=list(P0[0]["k"].shape)),
            determinism=dict(two_builds_logits_bitwise_equal=True, per_layer_hashes_equal=True,
                             n_layers_checked=len(layer_eq)))
        print(f"[sb] 规范初始状态 ✔ cache={PROMPT} 首 token={first0['argmax_id']}；"
              f"二次独立建立逐元素一致（logits + 28 层 K/V 摘要）")

        # ---------------- 2) 两臂：同一份 P0 的深拷贝 ----------------
        cacheS = CC.deep_clone(P0)
        cacheB = CC.deep_clone(P0)
        DC.require(_cache_len(cacheS) == PROMPT and _cache_len(cacheB) == PROMPT, "深拷贝后长度错")
        # 不共享可变存储（逐层 tensor 不同对象、不同数据指针）
        shared = [i for i, (a, b) in enumerate(zip(cacheS, cacheB))
                  if a["k"].data_ptr() == b["k"].data_ptr() or a["v"].data_ptr() == b["v"].data_ptr()]
        DC.require(not shared, "S/B 的 cache 共享可变存储", str(shared))
        rec["arms"] = dict(S="串行单步（phase=decode，逐 token）", B="分块 verify(4) + repair(1)（真实记录边界）",
                           shared_storage_layers=shared)

        # ---- S 臂：teacher forcing 逐 token ----
        S_rows: dict[int, torch.Tensor] = {}
        saved_next = lg0[0, -1].detach().clone()
        S_rows[PROMPT] = saved_next.detach().float().cpu()          # prefill 预测位置 8192
        s_trace = []
        for i, tk_ in enumerate(feed_seq):
            in_pos = PROMPT + i
            with torch.inference_mode(), M.attention_phase("decode"):
                lg, cacheS = spec_dec._step(tgt, tk_, cacheS)
            row = lg[0, -1].detach().clone()
            DC.require(_cache_len(cacheS) == in_pos + 1, "S 臂 cache 长度与位置不符",
                       f"{_cache_len(cacheS)} vs {in_pos + 1}")
            pred_pos = in_pos + 1
            S_rows[pred_pos] = row.detach().float().cpu()
            s_trace.append(dict(step=i, fed_token=tk_, input_pos=in_pos, pred_pos=pred_pos,
                                cache_len=_cache_len(cacheS), **_brief(row)))
        s_final = s_trace[-1]
        rec["S"] = dict(trace=s_trace, n_steps=len(s_trace))
        DC.require(s_final["input_pos"] == DECIDING_INPUT_POS,
                   "S 臂最后一步的输入位置必须是 8216", str(s_final["input_pos"]))
        DC.require(s_final["argmax_id"] == AR_AT_8216,
                   f"S 臂复现门失败：输入位置 8216 的 argmax 应 == 真实 AR 的 {AR_AT_8216}",
                   str(s_final["argmax_id"]))
        print(f"[sb] S 臂复现门 ✔ 输入位置 8216 argmax={s_final['argmax_id']}（== 真实 AR）")

        # ---- B 臂：按记录的真实调用边界重放（verify 吃记录的 cands，repair 吃记录的 emitted） ----
        B_rows: dict[int, torch.Tensor] = {}
        b_trace = []
        saved_nextB = lg0[0, -1].detach().clone()
        B_rows[PROMPT] = saved_nextB.detach().float().cpu()
        for r in rounds:
            pre, kp = int(r["pre"]), int(r["Kp"])
            DC.require(_cache_len(cacheB) == pre, f"B 臂第 {r['round']} 轮起点 cache 长度不符",
                       f"{_cache_len(cacheB)} vs {pre}")
            with torch.inference_mode(), M.attention_phase("verify"):
                rows, blk = spec_dec._verify(tgt, cacheB, list(r["cands"]))
            DC.require(rows.dim() == 2 and int(rows.shape[0]) == kp,
                       "rows 形状异常（应为 (Kp,V)）", str(list(rows.shape)))
            g_mine = [int(saved_nextB.reshape(-1).argmax())] + \
                     [int(rows[j].reshape(-1).argmax()) for j in range(kp - 1)]
            DC.require(g_mine == [int(x) for x in r["g"]],
                       f"B 臂第 {r['round']} 轮 g 与记录不符", f"{g_mine} vs {r['g']}")
            for j in range(kp):
                B_rows[pre + j + 1] = rows[j].detach().float().cpu()
            m = int(r["m"])
            cacheB = spec_cache.trim_cache(blk, pre + m)
            if m < kp:
                r_tok = int(r["emitted"][-1])
                with torch.inference_mode(), M.attention_phase("repair"):
                    saved_nextB, cacheB = spec_dec._step(tgt, r_tok, cacheB)
                B_rows[pre + m + 1] = saved_nextB[0, -1].detach().float().cpu()
            else:
                saved_nextB = rows[kp - 1].detach().clone()
            b_trace.append(dict(round=int(r["round"]), pre=pre, Kp=kp, m=m,
                                cands=[int(x) for x in r["cands"]], g_mine=g_mine,
                                g_recorded=[int(x) for x in r["g"]], cache_len_after=_cache_len(cacheB),
                                rows_argmax=[int(rows[j].reshape(-1).argmax()) for j in range(kp)]))
        rec["B"] = dict(trace=b_trace, n_rounds=len(b_trace), n_rows=len(B_rows))
        r11 = [t for t in b_trace if t["round"] == 11][0]
        DC.require(r11["pre"] == 8215, "第 11 轮 pre 应为 8215", str(r11["pre"]))
        DC.require(r11["rows_argmax"][1] == SPEC_ROUND11_ROW1,
                   f"B 臂复现门失败：第 11 轮 rows[1] 应 == 真实 Spec 的 {SPEC_ROUND11_ROW1}",
                   str(r11["rows_argmax"]))
        print(f"[sb] B 臂复现门 ✔ 第 11 轮 rows[1] argmax={r11['rows_argmax'][1]}（== 真实 Spec），"
              f"且 12 轮 g 全部逐元素等于记录值")

        # ---------------- 3) 同预测位置对照 ----------------
        common = sorted(set(S_rows) & set(B_rows))
        DC.require(len(common) > 0, "两臂没有可同位置比较的预测位置")
        per_pos = []
        for p in common:
            a, b = S_rows[p], B_rows[p]
            DC.assert_same_shape(a, b, f"pos{p}")
            d = (a - b).abs()
            per_pos.append(dict(pred_pos=p, s_argmax=int(torch.argmax(a)), b_argmax=int(torch.argmax(b)),
                                same_argmax=bool(int(torch.argmax(a)) == int(torch.argmax(b))),
                                elementwise_equal=bool(torch.equal(a, b)),
                                max_abs=float(d.max()), mean_abs=float(d.mean()),
                                n_diff=int((d > 0).sum())))
        first_nonzero = next((x["pred_pos"] for x in per_pos if not x["elementwise_equal"]), None)
        first_argmax = next((x["pred_pos"] for x in per_pos if not x["same_argmax"]), None)
        rec["same_position"] = dict(
            common_positions=common, n_common=len(common),
            first_nonzero_logits_diff_pos=first_nonzero,
            first_argmax_divergence_pos=first_argmax,
            per_pos=per_pos,
            note="**首次非零 logits 差异**与**首次 argmax 分歧**分开记录；"
                 "位置 p 的 logits 预测 token p")
        print(f"[sb] 同位置对照：共同位置 {len(common)} 个；"
              f"首次非零 logits 差异 @ {first_nonzero}；首次 argmax 分歧 @ {first_argmax}")
        for x in per_pos:
            if x["pred_pos"] in (8193, 8194, 8197, 8200, 8201, 8205, 8216, 8217, 8218):
                print(f"   pos {x['pred_pos']:5d} S={x['s_argmax']:6d} B={x['b_argmax']:6d} "
                      f"same={x['same_argmax']} max|Δ|={x['max_abs']:.6g} n_diff={x['n_diff']}")

        # ---------------- 4) 结构检查 + 落盘 ----------------
        rec["structure"] = dict(
            cacheS_len=_cache_len(cacheS), cacheB_len=_cache_len(cacheB),
            expected_S=PROMPT + len(emitted),
            note="两臂喂入的 token 序列完全相同（teacher forcing）；B 臂按记录的 verify/repair 边界执行")
        att = M.attention_stats()
        rec["attention_stats"] = dict(strategy=att["strategy"], calls=att["calls"],
                                      fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att["s_max_by_phase"])
        DC.require(sorted(k for k, v in att["fused_by_phase"].items() if v > 0) ==
                   (["prefill"] if ARM == "cudnn_prefill" else []),
                   f"融合路径出现位置与 arm={ARM} 不符", str(att["fused_by_phase"]))
        rec["attention_stats"]["arm"] = ARM
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_prefillsb_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{TAG}_prefillsb_{run_id}.pt"
        torch.save(dict(prefill_row=S_rows[PROMPT],
                        S_rows={str(k): v for k, v in S_rows.items()},
                        B_rows={str(k): v for k, v in B_rows.items()},
                        L27_K_P0=P0[27]["k"].detach().float().cpu(),
                        L27_V_P0=P0[27]["v"].detach().float().cpu()),
                   str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        print(f"[sb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_prefillsb_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[sb] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sb] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
