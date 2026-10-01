"""f13_i50_cells.py — F13 index50 的**同前缀四格局部对照**（不自由生成）。

在同一 cache 长度 8240 处，按 **历史来源 × 执行方式** 组成四格：
                    串行 S              分块 B
  真实 AR 历史 H_AR   H_AR-S              H_AR-B
  真实 Spec 历史 H_SP H_SP-S              H_SP-B

固定：相同权重/dtype/backend/位置规则；`max_new` 不参与（局部对照不生成）。
  * S：依次喂入 `[220,16]`（= 已接受候选），取**预测 8242** 的那一行；
  * B：喂入已记录候选 `[220,16,16,1154]`，取 `rows[1]`（同样预测 8242，可见 ≤8241）；
  * 每格从**独立深拷贝**开始（校验不共享可变存储）；源快照事后校验未被修改。

控制：**H_AR-S 必须逐元素复现捕获的真实 AR 决策行**；**H_SP-B 必须逐元素复现捕获的真实 Spec rows[1]**。
（另两格没有对应的真实捕获行 —— 那两种组合在真实运行里没有发生，本工具**不**为其编造参考。）
"""
from __future__ import annotations

import glob
import json
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
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f13i50_20261001"
KV, ARM = "base_bf16", "eager"
PRE, K, CHUNK = 8240, 4, 2048
FEED = [220, 16]
CANDS = [220, 16, 16, 1154]
TARGET_POS = 8242
WATCH = (16, 17, 20)


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cache_len(c):
    return int(spec_cache.cache_length(c))


def _summary(r, tag) -> dict:
    x = r.detach().float().reshape(-1)
    t5, t2 = torch.topk(x, 5), torch.topk(x, 2).values
    return dict(cell=tag, argmax=int(torch.argmax(x)), max_score=float(x.max()),
                margin=float(t2[0] - t2[1]),
                top5=[dict(token=int(i), logit=float(v)) for i, v in zip(t5.indices, t5.values)],
                watch={str(t): float(x[t]) for t in WATCH},
                finite=bool(torch.isfinite(x).all()), sha=A.stable_sha(r), n_elem=int(x.numel()))


def _cmp(a, b, name) -> dict:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    A.require(x.numel() == y.numel(), f"{name}: 长度不一致")
    d = (x - y).abs()
    return dict(name=name, elementwise_equal=bool(torch.equal(x, y)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()),
                a_argmax=int(torch.argmax(x)), b_argmax=int(torch.argmax(y)),
                same_argmax=bool(int(torch.argmax(x)) == int(torch.argmax(y))))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="f13_i50_cells", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(pre=PRE, k=K, feed=FEED, cands=CANDS, target_pos=TARGET_POS,
                           diag_precision="qk_fp32=False, prob_fp32=False"),
               semantics="同前缀四格局部对照（历史来源 × 执行方式）；**不做**自由生成；"
                         "四格各自的输入前缀相同（到 index49 为止），但**分块/串行各自新增 KV 后数值历史可能不同**。")
    print(f"[cells] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[cells] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_cells_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    return _run(rec, run_id)


def _run(rec, run_id) -> int:
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        cap_ar = torch.load(_newest(f"{TAG}_capture_ar_*.pt"), map_location="cpu")
        cap_sp = torch.load(_newest(f"{TAG}_capture_spec_*.pt"), map_location="cpu")
        rec["sources"] = dict(capture_ar=str(_newest(f"{TAG}_capture_ar_*.pt")),
                              capture_spec=str(_newest(f"{TAG}_capture_spec_*.pt")),
                              captured_AR_row_sha=A.stable_sha(cap_ar["row_8242"]),
                              captured_SP_rows1_sha=A.stable_sha(cap_sp["rows"][1]))
        A.require(list(cap_ar["prefix_tokens"]) == list(cap_sp["prefix_tokens"]),
                  "两次捕获的 token 前缀不一致",
                  f"{list(cap_ar['prefix_tokens'])[:6]} vs {list(cap_sp['prefix_tokens'])[:6]}")
        A.require(len(cap_ar["prefix_tokens"]) == TARGET_POS - 8192, "前缀长度应为 50")
        rec["prefix"] = dict(n=len(cap_ar["prefix_tokens"]), tokens=list(cap_ar["prefix_tokens"]),
                             note="两轮捕获的共同前缀逐 token 相同（到 index49 为止）")
        snap_ar, snap_sp = cap_ar["snapshot_cpu"], cap_sp["snapshot_cpu"]
        A.require(_cache_len(snap_ar) == PRE and _cache_len(snap_sp) == PRE,
                  "两份快照长度必须都是 8240")
        # 两份历史 KV 的逐层差异（只作描述，不据此定位根因层）
        dig_ar = A.require_full_cache(snap_ar, expect_len=PRE, n_layers=len(snap_ar), what="H_AR")
        dig_sp = A.require_full_cache(snap_sp, expect_len=PRE, n_layers=len(snap_sp), what="H_SP")
        rec["history_kv"] = dict(compare=A.cache_digest_diff(dig_ar, dig_sp),
                                 note="只报逐层摘要差异，**不**据此自动定位根因层")
        # 两份快照的**日志行**（saved_next / row_8241）也一并记录（描述性）
        probes = dict(
            ar_row8241_vs_sp_saved_next=_cmp(cap_ar["row_8241"], cap_sp["saved_next"],
                                             "AR_row8241_vs_SP_saved_next"))
        rec["probe_rows"] = probes

        def run_cell(tag, snap, mode):
            cache = [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()}
                     for d in snap]
            DR.require_distinct_storage(cache, snap, f"{tag}: 与源快照")
            if mode == "S":
                rows = []
                for tk in FEED:
                    with torch.inference_mode(), M.attention_phase("decode"):
                        lg, cache = spec_dec._step(tgt, tk, cache)
                    rows.append(lg[0, -1].detach().clone())
                A.require(_cache_len(cache) == PRE + len(FEED),
                          f"{tag}: 串行后 cache 应为 {PRE + len(FEED)}", str(_cache_len(cache)))
                row = rows[-1]
            else:
                with torch.inference_mode(), M.attention_phase("verify"):
                    rr, blk = spec_dec._verify(tgt, cache, list(CANDS))
                A.require(int(rr.shape[0]) == K, f"{tag}: rows 形状异常", str(list(rr.shape)))
                row = rr[1].detach().clone()
            tab = DR.slot_table(PRE, K)
            A.require(tab[2]["predicted_pos"] == TARGET_POS, "槽位表位置错误")
            A.require(tab[2]["visible_last_key"] == TARGET_POS - 1,
                      "决策行可见范围应为 target-1", str(tab[2]["visible_last_key"]))
            return row, _summary(row, tag)

        res, summaries = {}, {}
        for tag, snap, mode in (("H_AR-S", snap_ar, "S"), ("H_AR-B", snap_ar, "B"),
                                ("H_SP-S", snap_sp, "S"), ("H_SP-B", snap_sp, "B")):
            row, summ = run_cell(tag, snap, mode)
            res[tag] = row.cpu()
            summaries[tag] = summ
        rec["cells"] = summaries

        # ---- 控制：只对"真实运行里发生过的组合"要求逐元素复现 ----
        rec["controls"] = dict(
            H_AR_S_vs_captured_AR_row=_cmp(res["H_AR-S"], cap_ar["row_8242"], "H_AR-S_vs_captured_AR"),
            H_SP_B_vs_captured_SP_rows1=_cmp(res["H_SP-B"], cap_sp["rows"][1], "H_SP-B_vs_captured_SP"),
            note="只有这两格有对应的**真实捕获行**；H_AR-B / H_SP-S 是干预组合，"
                 "真实运行里没有发生 ⇒ 不设参考、不编造参考。")
        A.require(rec["controls"]["H_AR_S_vs_captured_AR_row"]["elementwise_equal"],
                  "控制失败：H_AR-S 未逐元素复现真实 AR 决策行",
                  str(rec["controls"]["H_AR_S_vs_captured_AR_row"]))
        A.require(rec["controls"]["H_SP_B_vs_captured_SP_rows1"]["elementwise_equal"],
                  "控制失败：H_SP-B 未逐元素复现真实 Spec rows[1]",
                  str(rec["controls"]["H_SP_B_vs_captured_SP_rows1"]))
        print("[cells] 控制 ✔ H_AR-S == 真实 AR 行、H_SP-B == 真实 Spec rows[1]（均逐元素）")

        # ---- 四格两两比较 ----
        rec["compare"] = dict(
            same_history_S_vs_B_AR=_cmp(res["H_AR-S"], res["H_AR-B"], "H_AR_S_vs_B"),
            same_history_S_vs_B_SP=_cmp(res["H_SP-S"], res["H_SP-B"], "H_SP_S_vs_B"),
            same_exec_S_AR_vs_SP=_cmp(res["H_AR-S"], res["H_SP-S"], "S_AR_vs_SP"),
            same_exec_B_AR_vs_SP=_cmp(res["H_AR-B"], res["H_SP-B"], "B_AR_vs_SP"))
        rec["verdict"] = dict(
            execution_effect_AR=rec["compare"]["same_history_S_vs_B_AR"]["same_argmax"],
            execution_effect_SP=rec["compare"]["same_history_S_vs_B_SP"]["same_argmax"],
            history_effect_S=rec["compare"]["same_exec_S_AR_vs_SP"]["same_argmax"],
            history_effect_B=rec["compare"]["same_exec_B_AR_vs_SP"]["same_argmax"],
            rule="同一历史下 S/B 不同 ⇒ 支持「本轮执行方式足以产生差异」；"
                 "相同执行方式下两历史不同 ⇒ 支持「历史状态影响决定」；"
                 "两者并存或依赖历史时如实报告交互，**不**强行选唯一原因。"
                 "小 margin 只是敏感性证据，**不是**数值误差豁免；**不**凭 max_abs 判实现错误。")
        att = M.attention_stats()
        A.require(not any(v > 0 for v in att["fused_by_phase"].values()),
                  "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_cells_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{TAG}_cells_{run_id}.pt"
        torch.save({k: v for k, v in res.items()}, str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        print("== 四格 ==")
        for tag in ("H_AR-S", "H_AR-B", "H_SP-S", "H_SP-B"):
            s = summaries[tag]
            print(f"  {tag:8s} argmax={s['argmax']:6d} margin={s['margin']:.6g} "
                  f"16/17/20={s['watch']['16']}/{s['watch']['17']}/{s['watch']['20']} "
                  f"top5={[x['token'] for x in s['top5']]}")
        print("== 对照 ==")
        for k, v in rec["compare"].items():
            print(f"  {k:26s} same_argmax={v['same_argmax']} max|Δ|={v['max_abs']:.6g} "
                  f"mean={v['mean_abs']:.6g} n_diff={v['n_diff']} ({v['a_argmax']} vs {v['b_argmax']})")
        print(f"  history_kv: {json.dumps(rec['history_kv']['compare'], ensure_ascii=False)[:200]}")
        print(f"[cells] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DR.AlignError, A.AlignError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_cells_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[cells] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[cells] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
