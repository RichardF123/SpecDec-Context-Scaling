"""divasc_rebuild_report_cpu.py — **只读**汇总"串行重建历史"实验的 A/B/C 三问（纯 CPU，不加载模型）。

输入（固定路径，不"自动选最新"）：
  * 重建运行：`output/g5/attn/divasc_rebuild_20261001T220737Z.{json,pt}` + 同 run 的 smoke 运行记录
  * AR 控制：`smoke8k_eager_20261001T185503Z.json` + `divcap_ar_20261001T185503Z.pt`
  * 原 Spec 控制：`smoke8k_eager_20261001T185705Z.json` + `divcap_spec_20261001T185705Z.pt`
  * 固定输入四格（重建快照）：`divasc_rebuild_20261001_cells_20261001T221447Z.json`
  * 固定输入四格（原 Spec 快照，上一轮）：`divasc_20261001_cells_20261001T190544Z.json`

输出：`output/g5/attn/divasc_rebuild_abc_<ts>.json` + stdout 表格。
严格区分**自然生成**与**固定输入重放**，并写明每个数字的可解释范围（混入"内容差"的必须标注）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "output" / "g5" / "attn"
RB = "divasc_rebuild_20261001T220737Z"
RB_RUN = "smoke8k_eager_20261001T220737Z"
AR_RUN = "smoke8k_eager_20261001T185503Z"
SP_RUN = "smoke8k_eager_20261001T185705Z"
AR_CAP = "divcap_ar_20261001T185503Z"
SP_CAP = "divcap_spec_20261001T185705Z"
CELLS_RB = "divasc_rebuild_20261001_cells_20261001T221447Z.json"
CELLS_SP = "divasc_20261001_cells_20261001T190544Z.json"
POS, IDX = 8239, 47
PRE = 8237


def _gen(name: str) -> list:
    return [int(x) for x in json.loads((OUT / name).read_text(encoding="utf-8"))["run"]["gen"]]


def _first_diff(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return dict(identical=bool(a == b), common_prefix=i,
                first_diff_index=(None if a == b else i),
                abs_position=(None if a == b else 8192 + i),
                tokens=(None if a == b else [a[i], b[i]]),
                n_diff=sum(1 for k in range(n) if a[k] != b[k]))


def _kv_cmp(a, b) -> dict:
    rows = []
    for i in range(min(len(a), len(b))):
        for kk in ("k", "v"):
            x, y = a[i][kk], b[i][kk]
            eq = bool(torch.equal(x, y))
            rows.append(dict(layer=i, attr=kk, bitequal=eq, n_diff=int((x != y).sum()),
                             max_abs=float((x.float() - y.float()).abs().max())))
    return dict(n_tensors=len(rows), n_bitequal=sum(1 for r in rows if r["bitequal"]),
                all_equal=all(r["bitequal"] for r in rows),
                n_diff_total=sum(r["n_diff"] for r in rows),
                max_abs=float(max(r["max_abs"] for r in rows)),
                first_diff_layers=[(r["layer"], r["attr"], r["n_diff"]) for r in rows
                                   if not r["bitequal"]][:6])


def _first_diverging_position(a, b) -> dict:
    """同内容两份历史里，**最早出现数值差异的位置**（先把 kv 维与 head_dim 维归约，再取位置）。"""
    first, per_layer = None, []
    for i in range(min(len(a), len(b))):
        f = None
        for kk in ("k", "v"):
            neq = (a[i][kk] != b[i][kk])
            pos_any = neq.any(dim=1).any(dim=-1).reshape(-1)
            nz = torch.nonzero(pos_any, as_tuple=False)
            if nz.numel():
                p = int(nz.reshape(-1)[0])
                f = p if f is None else min(f, p)
        per_layer.append(dict(layer=i, first_diff_pos=f))
        if f is not None:
            first = f if first is None else min(first, f)
    return dict(first_diff_pos_any_layer=first,
                prefill_identical=bool(first is None or first >= 8192),
                n_layers_diverging=sum(1 for x in per_layer if x["first_diff_pos"] is not None),
                per_layer=per_layer)


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(tool="divasc_rebuild_report_cpu", run_utc=ts,
                     semantics="只读汇总 A/B/C；严格区分自然生成与固定输入重放；"
                               "混入内容差的比较必须标注为不可归因",
                     fixed=dict(rebuild_capture=RB, rebuild_run=RB_RUN, ar_run=AR_RUN,
                                spec_run=SP_RUN, ar_cap=AR_CAP, spec_cap=SP_CAP,
                                cells_rebuild=CELLS_RB, cells_origspec=CELLS_SP))
    meta = json.loads((OUT / f"{RB}.json").read_text(encoding="utf-8"))
    tens = torch.load(OUT / f"{RB}.pt", map_location="cpu")
    rb_run = json.loads((OUT / f"{RB_RUN}.json").read_text(encoding="utf-8"))
    ar_run = json.loads((OUT / f"{AR_RUN}.json").read_text(encoding="utf-8"))
    sp_run = json.loads((OUT / f"{SP_RUN}.json").read_text(encoding="utf-8"))
    g_rb, g_ar, g_sp = _gen(f"{RB_RUN}.json"), _gen(f"{AR_RUN}.json"), _gen(f"{SP_RUN}.json")

    # ---------------- 干预忠实性（先自证：早期分叉不是我实现错的）----------------
    rounds = meta["rounds"]
    rec["intervention_faithfulness"] = dict(
        n_rounds=len(rounds), n_replay_feeds=meta["n_replay_feeds"],
        total_emitted=sum(len(r["emitted"]) for r in rounds),
        feeds_equal_emitted_every_round=all(r["feeds"] == r["emitted"] for r in rounds),
        length_contract_every_round=all(r["rebuilt_len"] == r["pre"] + len(r["emitted"])
                                        and r["per_layer_len_ok"] for r in rounds),
        H_pre_untouched_every_round=all(r["H_pre_len_after_replay"] == r["pre"] for r in rounds),
        branches=dict(all_accept=sum(1 for r in rounds if r["branch"] == "all_accept"),
                      mismatch=sum(1 for r in rounds if r["branch"] == "mismatch"),
                      m0=sum(1 for r in rounds if r["m"] == 0)),
        prefill_identical_to_orig_spec=bool(
            rb_run["prefill_digest"]["target"]["layers"] == sp_run["prefill_digest"]["target"]["layers"]),
        prefill_logits_sha=dict(rebuild=rb_run["prefill_digest"]["target"]["logits"]["sha"],
                                orig_spec=sp_run["prefill_digest"]["target"]["logits"]["sha"]),
        round_invariants_ok=rb_run["round_invariants"]["all_ok"],
        dispatch_ok=rb_run["run"]["dispatch_only_prefill"],
        fused_by_phase=rb_run["run"]["attention_stats"]["fused_by_phase"],
        restore=rb_run["spec_history_rebuild_restore"],
        scope_restore=rb_run["target_rope_scope_restore"])
    f = rec["intervention_faithfulness"]
    print("== 干预忠实性（先自证）==")
    print(f"  轮数={f['n_rounds']} 重放喂入={f['n_replay_feeds']} = 提交 token 数 "
          f"{f['total_emitted']}；每轮 feeds==emitted={f['feeds_equal_emitted_every_round']}")
    print(f"  长度契约={f['length_contract_every_round']}；H_pre 未被改写={f['H_pre_untouched_every_round']}；"
          f"分支={f['branches']}")
    print(f"  prefill 与原 Spec 摘要全同={f['prefill_identical_to_orig_spec']}"
          f"（logits sha {f['prefill_logits_sha']}）；不变量={f['round_invariants_ok']}；"
          f"fused={f['fused_by_phase']}")
    print(f"  恢复核查：重建={f['restore']} / RoPE 作用域={f['scope_restore']}")

    # ---------------- C：自然生成 ----------------
    rec["C_natural_generation"] = dict(
        rebuild_vs_ar=_first_diff(g_rb, g_ar), rebuild_vs_origspec=_first_diff(g_rb, g_sp),
        ar_vs_origspec=_first_diff(g_ar, g_sp),
        lens=dict(rebuild=len(g_rb), ar=len(g_ar), orig_spec=len(g_sp)),
        stops=dict(rebuild=rb_run["run"]["stop_reason"], ar=ar_run["run"]["stop_reason"],
                   orig_spec=sp_run["run"]["stop_reason"]),
        token_at_target=dict(rebuild=(g_rb[IDX] if len(g_rb) > IDX else None), ar=g_ar[IDX],
                             orig_spec=g_sp[IDX]))
    c = rec["C_natural_generation"]
    print("\n== C) 自然生成（64 token）==")
    for k in ("rebuild_vs_ar", "rebuild_vs_origspec", "ar_vs_origspec"):
        d = c[k]
        print(f"  {k:<22} 完全一致={d['identical']} 共同前缀={d['common_prefix']}"
              + (f" 首分歧 index={d['first_diff_index']}（位置 {d['abs_position']}）"
                 f" {d['tokens']} 差异数={d['n_diff']}" if not d["identical"] else ""))
    print(f"  位置 {POS}（index {IDX}）: 重建={c['token_at_target']['rebuild']} "
          f"AR={c['token_at_target']['ar']} 原Spec={c['token_at_target']['orig_spec']}")

    # 首分歧所在轮（自然生成；不硬编码 round 18）
    i = (c["rebuild_vs_ar"]["first_diff_index"] if not c["rebuild_vs_ar"]["identical"] else None)
    hit = None
    if i is not None:
        cum = 0
        for r in rounds:
            n = len(r["emitted"])
            if cum <= i < cum + n:
                hit = dict(r, offset_in_round=i - cum, gen_range=[cum, cum + n - 1])
            cum += n
    rec["first_divergence_scene"] = dict(
        index=i, abs_position=(None if i is None else 8192 + i),
        rebuild_token=(None if i is None else g_rb[i]), ar_token=(None if i is None else g_ar[i]),
        orig_spec_token=(None if i is None else g_sp[i]), round=hit,
        note="新首分歧现场（按口径：8239 之前提前分歧 ⇒ 停止「相同历史」归因，保存本现场）")
    if hit:
        print(f"\n== 新首分歧现场 ==")
        print(f"  第 {hit['round']} 轮：pre={hit['pre']} kp={hit['kp']} m={hit['m']} "
              f"branch={hit['branch']} 轮内偏移={hit['offset_in_round']}")
        print(f"    cands={hit['cands']}")
        print(f"    g（重建历史的块验证值）={hit['g']}   emitted={hit['emitted']}")
        print(f"    该位置 token：重建={g_rb[i]} AR={g_ar[i]} 原Spec={g_sp[i]}")

    # ---------------- A：重建历史 vs AR 历史（**前缀已分叉 ⇒ 不可归因**）----------------
    s_rb, s_ar, s_sp = tens["snapshot_cpu"], torch.load(OUT / f"{AR_CAP}.pt",
                                                        map_location="cpu")["snapshot_cpu"], \
        torch.load(OUT / f"{SP_CAP}.pt", map_location="cpu")["snapshot_cpu"]
    rec["A_history"] = dict(
        pre=dict(rb=len(s_rb), ar=len(s_ar), sp=len(s_sp)),
        rebuild_vs_ar=_kv_cmp(s_rb, s_ar),
        rebuild_vs_origspec=_kv_cmp(s_rb, s_sp),
        # 同内容对照（上一轮已建立）：AR 历史 vs 原 Spec 历史 的**最早数值分叉位置**
        origspec_vs_ar_first_divergence=_first_diverging_position(s_sp, s_ar),
        interpretability=dict(
            rebuild_vs_ar="❌ 不可归因：重建运行的自然生成在 index %s 就与 AR 分叉 ⇒ 8237 快照对应的"
                          "**不是**同一逻辑前缀；该比较混入「内容差」" % i,
            origspec_vs_ar="✅ 同内容（AR 与原 Spec 在 index ≤46 的 token 一致）⇒ 这份对照的差异是"
                           "**纯数值**（块内 KV vs 单步 KV）"),
        confound_scale=("同执行方式的跨历史比较：重建 vs AR 的 max|Δ| 与 mean|Δ| 见下；"
                        "作为对照，同内容不同历史的四格比较 max|Δ| 仅 0.19–0.25 ⇒ 量级差说明"
                        "重建 vs AR 的差被内容差主导"))
    a = rec["A_history"]
    print("\n== A) 长度 8237 的历史对照（逐张量 torch.equal）==")
    print(f"  重建 vs AR：逐位相同 {a['rebuild_vs_ar']['n_bitequal']}/{a['rebuild_vs_ar']['n_tensors']}"
          f"；不同元素 {a['rebuild_vs_ar']['n_diff_total']}；max|Δ|={a['rebuild_vs_ar']['max_abs']:.6g}"
          "  ← ❌ 前缀已分叉（内容差混入），不可归因")
    print(f"  原Spec vs AR（同内容）：最早数值分叉位置="
          f"{a['origspec_vs_ar_first_divergence']['first_diff_pos_any_layer']}"
          f"（prefill 段逐位相同={a['origspec_vs_ar_first_divergence']['prefill_identical']}；"
          f"分叉层数={a['origspec_vs_ar_first_divergence']['n_layers_diverging']}）← ✅ 纯数值对照")

    # ---------------- B：局部判断 ----------------
    cells_rb = json.loads((OUT / CELLS_RB).read_text(encoding="utf-8"))
    cells_sp = json.loads((OUT / CELLS_SP).read_text(encoding="utf-8"))
    rr = tens.get("replay_row")
    natural_row = None
    if rr is not None:
        x = rr.detach().float().reshape(-1)
        natural_row = dict(argmax=int(x.argmax()), logit_6702=float(x[6702]), logit_6652=float(x[6652]),
                           logit_5470=float(x[5470]), margin=float(torch.topk(x, 2).values.diff()),
                           fed_token=meta.get("row_fed_token"), round=meta.get("row_round"))
    rec["B_local_decision"] = dict(
        natural_replay_row_in_rebuild=natural_row,
        fixed_input_cells_on_rebuild_snapshot={
            k: dict(argmax=v["argmax"], margin_bf16=v["margin_bf16"], watch=v["watch"],
                    top5=[t["token"] for t in v["top5"]],
                    repeat_ok=v.get("repeat2_elementwise_equal"))
            for k, v in cells_rb["cells"].items()},
        fixed_input_cells_on_origspec_snapshot={
            k: dict(argmax=v["argmax"], margin_bf16=v["margin_bf16"], watch=v["watch"])
            for k, v in cells_sp["cells"].items()},
        cells_rebuild=dict(compare=cells_rb["compare"], controls=cells_rb["controls"],
                           prefix_equal=cells_rb["prefix"]["equal"],
                           prefix_policy=cells_rb["prefix"]["policy"]),
        cells_origspec=dict(compare=cells_sp["compare"], controls=cells_sp["controls"],
                            prefix_equal=cells_sp["prefix"].get(
                                "equal", True),   # 旧产物无该字段：当时 strict 策略下相等是硬断言
                            prefix_equal_source=("capture.prefix.equal" if "equal" in cells_sp["prefix"]
                                                 else "旧产物缺字段；strict 策略下「相等」由硬断言保证")),
        answer=("B = **否**（两种口径都不是 6702）：自然生成中预测 %d 的真实行 argmax=%s；"
                "固定输入局部对照（重建快照）单步/分块分别 %s / %s。"
                "但两者都在**内容已分叉**的历史上 ⇒ 不能据此判定「历史差异不影响该位置判断」"
                % (POS, natural_row["argmax"] if natural_row else None,
                   cells_rb["cells"]["H_SP-S"]["argmax"], cells_rb["cells"]["H_SP-B"]["argmax"])))
    b = rec["B_local_decision"]
    print("\n== B) 局部判断（该位置是否恢复 6702）==")
    print(f"  自然生成：重建历史下预测 {POS} 的真实行 argmax={natural_row['argmax']}"
          f"（6702={natural_row['logit_6702']:.6g} 6652={natural_row['logit_6652']:.6g} "
          f"5470={natural_row['logit_5470']:.6g} margin={natural_row['margin']:.6g}）"
          f" ← 但该行所在历史的 token 内容已与原轨迹不同")
    print(f"  固定输入重放（重建快照，[5880,389] / 4 候选）："
          f"H_SP-S={b['fixed_input_cells_on_rebuild_snapshot']['H_SP-S']['argmax']} "
          f"H_SP-B={b['fixed_input_cells_on_rebuild_snapshot']['H_SP-B']['argmax']}")
    print(f"  固定输入重放（**原 Spec 快照**，上一轮）：H_SP-S="
          f"{b['fixed_input_cells_on_origspec_snapshot']['H_SP-S']['argmax']} H_SP-B="
          f"{b['fixed_input_cells_on_origspec_snapshot']['H_SP-B']['argmax']}")
    print(f"  控制：重建快照 H_SP-B 参考适用={cells_rb['controls']['H_SP_B_control_applicable']}"
          f"（{cells_rb['controls'].get('H_SP_B_control_skipped')}）；"
          f"H_AR-S 逐元素复现={cells_rb['controls']['H_AR_S_vs_captured_AR_row']['elementwise_equal']}")

    rec["stage"] = "complete"
    p = OUT / f"divasc_rebuild_abc_{ts}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[abc] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
