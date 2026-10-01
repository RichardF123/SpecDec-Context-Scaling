"""f13_i50_capture.py — F13 index50 首次分歧的**真实路径捕获**（`F13_I50_MODE=ar|spec`）。

只回答：① 原始决策行是否取对、可见 token 前缀是否相同；② 固定本轮起点后 S/B 是否仍不同；
③ 历史差异与本轮执行差异各自的证据。

捕获（**新运行**；旧产物没有决策 logits，不能从 token ID 反推 margin）
  * 共同 token 前缀（位置 8192..8241，共 50 个）；
  * AR：`cache 长度 8240` 的**深拷贝（CPU）** + 喂入 [220,16] 得到的**预测 8241 / 8242 的真实行**；
  * Spec：round19 进入 `_verify` **之前**的快照 + `saved_next`（经只读包装 `_step` 取得）+
    `rows[0..3]` + 轮次元数据；
  * 决策行由 `decision_rows` 硬检查：`rows[1] ↔ 位置 8242`、可见最后 key = 8241（**不含**未来候选）。

配置：F13 / 8192 / chunk2048 / k4 / BF16 / base_bf16 / **全 eager** / qk32=prob32=False / greedy / 原 EOS 规则。
`max_new` 仍配 64；捕获到目标决策行后**主动结束** ⇒ 标 `diagnostic_capture_stop`，**不**冒充完整生成通过。
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
from . import decision_rows as DR
from . import model_self as M
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import family_input as FI
from . import g5_8k_spec_variance_probe as P5V
from .smoke_checks import CheckRegistry
from .smoke_hooks import DecisionHooks

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f13i50_20261001"
FAMILY = "F13"
CTX, CHUNK, K, MAXNEW = 8192, 2048, 4, 64
KV, ARM = "base_bf16", "eager"
PRE = 8240
FEED = [220, 16]
CANDS = [220, 16, 16, 1154]
G_REC = [220, 16, 20, 11]
EMITTED_REC = [220, 16, 20]
TARGET_POS = 8242
EXPECT_AR, EXPECT_SP = 17, 20
MODE = os.environ.get("F13_I50_MODE", "ar").strip().lower()


class _Captured(Exception):
    pass


def _cache_len(c):
    return int(spec_cache.cache_length(c))


def _row_summary(r) -> dict:
    x = r.detach().float().reshape(-1)
    t5 = torch.topk(x, 5)
    t2 = torch.topk(x, 2).values
    return dict(argmax=int(torch.argmax(x)), max_score=float(x.max()), margin=float(t2[0] - t2[1]),
                top5=[dict(token=int(i), logit=float(v)) for i, v in zip(t5.indices, t5.values)],
                watch={str(t): float(x[t]) for t in (16, 17, 20)},
                finite=bool(torch.isfinite(x).all()), sha=A.stable_sha(r),
                n_elem=int(x.numel()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="f13_i50_capture", mode=MODE, family=FAMILY, kv=KV,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(ctx=CTX, chunk=CHUNK, k=K, max_new=MAXNEW, arm=ARM, kv=KV,
                           diag_precision="qk_fp32=False, prob_fp32=False",
                           pre=PRE, feed=FEED, cands=CANDS, target_pos=TARGET_POS,
                           eos_policy="沿用冻结 EOS 规则（记录 eos_token_id 与实际停止原因）"),
               semantics="**新捕获运行**（不是旧数据离线补算）；真实自由生成到目标决策行即主动结束，"
                         "标 diagnostic_capture_stop；不改生产计算/精度/tie-breaking。")
    print(f"[i50] run_id={run_id} mode={MODE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[i50] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_capture_{MODE}_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    hooks = None
    orig_step = spec_dec._step
    step_patched = False
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        drf = None
        if MODE == "spec":
            drf, _t, _i = load_ours_bf16(P5V.DRAFT)
            drf.set_kv_format(spec_dec._fmt(KV))
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        A.require(int(ids.shape[1]) == CTX, "prompt 长度必须 8192", str(int(ids.shape[1])))
        A.require(fam_meta["token_sha1"] == "9a278731d744a859", "F13 token_sha1 不符",
                  str(fam_meta["token_sha1"]))
        rec["input"] = dict(family=fam_meta,
                            ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]),
                            eos_id=int(tok.eos_token_id))
        rec["eos_config"] = dict(eos_id=int(tok.eos_token_id), source="tokenizer")
        inner = P5V.timed_chunked_prefill({}, CHUNK, tgt)
        reg = CheckRegistry()
        held: dict = {"acc": [], "saved_next_next": None}

        def on_prefill_post(key, lg, cache):
            # Spec 的 `saved_next` 初值 = prefill 末位 logits（与 spec_dec 一致）
            if key == "target" and held["saved_next_next"] is None:
                held["saved_next_next"] = lg.reshape(-1, lg.shape[-1])[-1].detach().clone()

        def on_verify_pre(cache, cands):
            # 每轮都记录（用于**逐轮累积共同前缀**，与 spec_dec 的规则同源）
            held["cur_cands"] = [int(x) for x in cands]
            if _cache_len(cache) == PRE and "snapshot" not in held:
                held["snapshot"] = DR.snapshot(cache)
                held["snap_len"] = PRE
                held["snap_digest"] = A.require_full_cache(
                    held["snapshot"], expect_len=PRE, n_layers=len(held["snapshot"]),
                    what="Spec 快照（_verify 之前）")
                held["prefix_len_at_pre"] = len(held["acc"])
                A.require(held["saved_next_next"] is not None, "未能取得 round19 的 saved_next")
                held["saved_next_t"] = held["saved_next_next"]
                held["saved_next"] = _row_summary(held["saved_next_t"])
                held["armed"] = True

        def on_verify_post(rows):
            if "rows_t" in held:
                return
            cands = held.get("cur_cands")
            A.require(cands is not None, "on_verify_post 缺少本轮 cands")
            kp = len(cands)
            g = [int(held["saved_next_next"].reshape(-1).argmax())] + \
                [int(rows[j].reshape(-1).argmax()) for j in range(kp - 1)]
            m = DR.deciding_slot(cands, g)
            emitted = list(cands[:m]) + ([g[m]] if m < kp else [])
            held["acc"] = held["acc"] + emitted
            # 复刻 spec_dec 的 saved_next 更新规则
            held["saved_next_next"] = (rows[kp - 1].detach().clone() if m == kp else None)
            if held.get("armed") and "rows_t" not in held:
                A.require(len(held["acc"]) - len(emitted) == PRE - CTX,
                          "round19 之前的累计 token 数应为 48",
                          str(len(held["acc"]) - len(emitted)))
                held["rows_t"] = rows.detach().clone()
                held["rows"] = [_row_summary(rows[j]) for j in range(kp)]
                held["cands_raw"], held["g"], held["m"], held["emitted"] = cands, g, m, emitted
                held["round_meta"] = dict(pre=PRE, Kp=kp, m=m, cands=cands, g=g, emitted=emitted)
                held["prefix"] = list(held["acc"][:PRE - CTX]) + list(cands[:m])
                A.require(len(held["prefix"]) == TARGET_POS - CTX,
                          "前缀长度应为 50", str(len(held["prefix"])))
                raise _Captured()

        def step_recorder(model, token, cache, *a, **k):
            lg, c = orig_step(model, token, cache, *a, **k)
            if model is tgt:
                # 与 spec_dec 一致：repair 之后 `saved_next` 就是它的返回行
                held["saved_next_next"] = lg[0, -1].detach().clone()
            return lg, c

        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner,
                              on_verify_pre=on_verify_pre, on_verify_post=on_verify_post,
                              on_prefill_post=on_prefill_post)
        hooks.install()
        spec_dec._step = step_recorder        # 只读包装：返回原函数结果
        step_patched = True
        try:
            if MODE == "ar":
                try:
                    _run_ar(tgt, ids, inner, held)
                except _Captured:
                    pass          # 捕获完成 ⇒ 主动结束（diagnostic_capture_stop）
            else:
                try:
                    spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K,
                                                eos_id=tok.eos_token_id, target_kv=KV,
                                                draft_kv="base_bf16")
                except _Captured:
                    pass
        finally:
            spec_dec._step = orig_step
            step_patched = False
            hooks.remove()
            hooks = None

        # ---------------- 对齐与复现门 ----------------
        pref = held["prefix"]
        A.require(len(pref) == TARGET_POS - CTX, "共同前缀长度应为 50", str(len(pref)))
        A.require(_cache_len_cpu(held["snapshot"]) == PRE, "快照长度必须 8240",
                  str(_cache_len_cpu(held["snapshot"])))
        DR.require_unmodified(held["snapshot"], held["snapshot"], "快照自检")  # 结构自洽
        rec["capture"] = dict(prefix_len=len(pref), prefix_tokens=pref, snap_len=PRE,
                              snapshot_layer_digest=held["snap_digest"],
                              snapshot_is_cpu=all(t.device.type == "cpu"
                                                  for d in held["snapshot"] for t in d.values()))
        if MODE == "ar":
            al = DR.align_decision(pre=PRE, kp=K, cands=CANDS, g=G_REC, emitted=EMITTED_REC)
            DR.require_deciding_row_matches(al, expect_source="rows[1]", expect_row_index=1,
                                            expect_pred_pos=TARGET_POS)
            DR.require_aligned_row_is_causal(al)
            A.require(held["gen"][PRE - CTX] == FEED[0], "AR 在位置 8240 的 token 应为 220",
                      str(held["gen"][PRE - CTX]))
            # 位置 8241/8242 的 token 在本捕获里由**记录**决定（喂入 16），因此与旧记录交叉核对
            old_gen, _ = _old_record("ar")
            A.require(list(old_gen[:TARGET_POS - CTX]) == list(pref),
                      "AR token 前缀与旧记录不一致",
                      f"{list(old_gen[:TARGET_POS - CTX])[:6]}... vs {list(pref)[:6]}...")
            A.require(int(old_gen[TARGET_POS - CTX]) == EXPECT_AR,
                      "旧记录里 AR 在位置 8242 的 token 应为 17", str(old_gen[TARGET_POS - CTX]))
            A.require(FEED == [int(old_gen[PRE - CTX]), int(old_gen[PRE - CTX + 1])],
                      "喂入的 [220,16] 必须等于旧记录在位置 8240/8241 的 token",
                      f"{FEED} vs {[old_gen[PRE - CTX], old_gen[PRE - CTX + 1]]}")
            A.require(held["row_8242"]["argmax"] == EXPECT_AR,
                      "AR 复现门失败：位置 8242 应为 17", str(held["row_8242"]["argmax"]))
            rec["align"] = al
            rec["ar"] = dict(row_8241=held["row_8241"], row_8242=held["row_8242"],
                             prefix_end_token=FEED[-1],
                             stop_reason="diagnostic_capture_stop")
            print(f"[i50] AR 复现门 ✔ 位置 8242 argmax={held['row_8242']['argmax']}（期望 17）；"
                  f"位置 8240 由 AR 自决={held['gen'][PRE - CTX]}、8241 按记录喂入={FEED[1]}")
        else:
            al = DR.align_decision(pre=PRE, kp=len(held["cands_raw"]), cands=held["cands_raw"],
                                   g=held["g"], emitted=held["emitted"])
            DR.require_deciding_row_matches(al, expect_source="rows[1]", expect_row_index=1,
                                            expect_pred_pos=TARGET_POS)
            DR.require_aligned_row_is_causal(al)
            A.require(al["cands"] == CANDS, "round19 cands 与旧记录不符", str(al["cands"]))
            A.require(al["g"] == G_REC, "round19 g 与旧记录不符", str(al["g"]))
            A.require(al["m"] == 2, "round19 m 与旧记录不符", str(al["m"]))
            old_emitted, old_rec = _old_record("spec")
            old_rd = _old_round_at_pre(old_rec, PRE)
            A.require(old_rd is not None, "旧记录里找不到 pre=8240 的轮次")
            A.require(old_rd["cands"] == al["cands"] and old_rd["g"] == al["g"]
                      and old_rd["m"] == al["m"],
                      "旧记录 round19 的 cands/g/m 与本轮捕获不一致",
                      f"{old_rd} vs {al['cands']}/{al['g']}/{al['m']}")
            old_prefix = list(old_emitted[:TARGET_POS - CTX])
            A.require(old_prefix == list(pref), "Spec token 前缀与旧记录不一致")
            A.require(int(old_emitted[TARGET_POS - CTX]) == EXPECT_SP,
                      "旧记录里 Spec 在位置 8242 的 token 应为 20",
                      str(old_emitted[TARGET_POS - CTX]))
            rec["old_record_cross_check"] = dict(
                old_round19=old_rd, old_token_at_target=int(old_emitted[TARGET_POS - CTX]),
                old_ar_token_at_target=int(_old_record("ar")[0][TARGET_POS - CTX]))
            A.require(held["rows"][1]["argmax"] == EXPECT_SP,
                      "Spec 复现门失败：rows[1] @8242 应为 20", str(held["rows"][1]["argmax"]))
            rec["align"] = al
            rec["spec"] = dict(saved_next=held["saved_next"], rows=held["rows"],
                               round_meta=held["round_meta"],
                               stop_reason="diagnostic_capture_stop")
            print(f"[i50] Spec 复现门 ✔ rows[1] @8242 argmax={held['rows'][1]['argmax']}（期望 20）；"
                  f"pre/cands/g/m 与旧记录一致")

        att = M.attention_stats()
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att["s_max_by_phase"])
        A.require(not any(v > 0 for v in att["fused_by_phase"].values()),
                  "本配置要求零融合调用", str(att["fused_by_phase"]))
        tp = OUTDIR / f"{TAG}_capture_{MODE}_{run_id}.pt"
        torch.save(dict(snapshot_cpu=held["snapshot"], snapshot_len=PRE,
                        row_8241=held.get("row_8241_t"), row_8242=held.get("row_8242_t"),
                        saved_next=held.get("saved_next_t"), rows=held.get("rows_t"),
                        prefix_tokens=pref), str(tp))
        rec["stage"] = "complete"
        rec["artifacts"] = dict(json=str(OUTDIR / f"{TAG}_capture_{MODE}_{run_id}.json"), pt=str(tp))
        p = OUTDIR / f"{TAG}_capture_{MODE}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[i50] 快照层数={len(held['snapshot'])} 长度={PRE}；fused_by_phase={att['fused_by_phase']}")
        print(f"[i50] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        if step_patched:
            spec_dec._step = orig_step
        if hooks is not None:
            try:
                hooks.remove()
            except Exception:                                     # noqa: BLE001
                pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DR.AlignError, A.AlignError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_capture_{MODE}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[i50] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[i50] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


def _cache_len_cpu(snap) -> int:
    return int(snap[0]["k"].shape[2])


def _old_record(mode: str):
    """取最近一次 F13 eager 完整生成记录（用于**与旧记录交叉核对**，不是复用其结果）。"""
    cands = []
    for p in sorted(glob.glob(str(OUTDIR / "smoke8k_eager_*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        fam = ((r.get("input") or {}).get("family") or {}).get("family_id")
        if (r.get("config") or {}).get("mode") == mode and fam == FAMILY:
            cands.append((p, r))
    A.require(cands, f"找不到 F13/{mode} 的旧记录（无法做前缀交叉核对）")
    r = cands[-1][1]
    if mode == "ar":
        return ((r.get("run") or {}).get("gen") or []), r
    return [x for rd in (r.get("rounds") or []) for x in rd["emitted"]], r


def _old_round_at_pre(rec_old: dict, pre: int):
    for rd in (rec_old.get("rounds") or []):
        if int(rd["pre"]) == pre:
            return dict(pre=int(rd["pre"]), Kp=int(rd["Kp"]), m=int(rd["m"]),
                        cands=[int(x) for x in rd["cands"]], g=[int(x) for x in rd["g"]],
                        emitted=[int(x) for x in rd["emitted"]])
    return None


def _run_ar(tgt, ids, inner, held):
    """真实 AR：分块 prefill + 逐步解码；到位置 8240 前捕获快照与决策行。"""
    with torch.inference_mode(), M.attention_phase("prefill"):
        lg, cache = inner(tgt, ids)
    A.require(_cache_len(cache) == CTX, "AR prefill 后 cache 必须 8192", str(_cache_len(cache)))
    gen = [int(lg.reshape(-1, lg.shape[-1])[-1].argmax())]
    while True:
        if _cache_len(cache) == PRE:
            held["gen"] = list(gen)
            A.require(len(gen) == PRE - CTX + 1, "AR 在 cache=8240 时应已确定 index 0..48",
                      f"len(gen)={len(gen)}")
            held["snapshot"] = DR.snapshot(cache)
            held["snap_len"] = PRE
            held["snap_digest"] = A.require_full_cache(held["snapshot"], expect_len=PRE,
                                                      n_layers=len(held["snapshot"]),
                                                      what="AR 快照")
            held["prefix"] = list(gen[:PRE - CTX]) + list(FEED)
            first = True
            for tk in FEED:
                with torch.inference_mode(), M.attention_phase("decode"):
                    lg, cache = spec_dec._step(tgt, tk, cache)
                if first:
                    held["row_8241_t"] = lg[0, -1].detach().clone()
                    held["row_8241"] = _row_summary(held["row_8241_t"])
                    first = False
                else:
                    held["row_8242_t"] = lg[0, -1].detach().clone()
                    held["row_8242"] = _row_summary(held["row_8242_t"])
            raise _Captured()
        A.require(_cache_len(cache) < PRE + 8, "AR 未在预期步数内到达 cache=8240",
                  f"cache={_cache_len(cache)}")
        with torch.inference_mode(), M.attention_phase("decode"):
            lg, cache = spec_dec._step(tgt, gen[-1], cache)
        gen.append(int(lg[0, -1].argmax()))


def _spec_prefix_from_rounds(emitted: list, upto: int) -> list:
    return list(emitted[:upto])


if __name__ == "__main__":
    raise SystemExit(main())
