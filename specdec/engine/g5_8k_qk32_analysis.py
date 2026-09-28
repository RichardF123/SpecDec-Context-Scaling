"""g5_8k_qk32_analysis.py — 读 8 份结果，产出"控制组 vs 干预组"表与分歧证据（**纯 CPU**）。

判读纪律（写进产物，避免事后放宽）
  * 控制组必须先复现已知分歧；**未复现**就标 `reproduction_problem`，不得把干预组的改善归功于 QK FP32；
  * 组内比较 AR vs SpecDec 只用**共同前缀**；**不**比较分歧之后不同前缀上的 logits；
  * 分歧推迟 = 部分改善，**不是 PASS**；
  * 不以 margin 小 / 文本接近 / 结构检查通过 来豁免 token 一致性。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"


def _load(tag: str, group: str, kv: str, mode: str) -> dict | None:
    js = sorted(glob.glob(str(OUTDIR / f"{tag}_{group}_{kv}_{mode}_*.json")))
    js = [p for p in js if not p.endswith("compare.json")]
    if not js:
        return None
    return json.loads(Path(js[-1]).read_text(encoding="utf-8"))


def first_divergence(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return dict(diverged=True, index=i, a=a[i], b=b[i],
                        agreeing_prefix_len=i,
                        prefix_identical=(a[:i] == b[:i]))
    if len(a) != len(b):
        return dict(diverged=True, index=n, a=(a[n] if n < len(a) else None),
                    b=(b[n] if n < len(b) else None), agreeing_prefix_len=n,
                    prefix_identical=(a[:n] == b[:n]),
                    reason="长度不同（较短序列已完全一致）")
    return dict(diverged=False, index=None, a=None, b=None, agreeing_prefix_len=n,
                prefix_identical=True)


def decision_row_for(rec: dict, gen_index: int) -> dict | None:
    """取"决定生成序列 index=gen_index 的那个 token"的真实决策行（**同一共同前缀**下）。"""
    dr = rec.get("decision_rows") or {}
    if rec["mode"] == "ar":
        for r in dr.get("ar_rows", []):
            if r["gen_index"] == gen_index:
                return dict(source="ar_decode_row", input_position=r["input_position"],
                            gen_index=gen_index, argmax_id=r["argmax_id"],
                            max_score=r["max_score"], n_max_tied=r["n_max_tied"],
                            tied_max_tokens=r["tied_max_tokens"],
                            margin_top1_top2=r["margin_top1_top2"], tie_at_top=r["tie_at_top"])
        return None
    # SpecDec：saved_next 决定 emitted[0]；rows[j] 决定其后第 j+1 个
    for vr in dr.get("verify_rows", []):
        base_gi = vr["pre"] - dr["prompt_len"] + 1
        for j, row in enumerate(vr["rows"]):
            if base_gi + j == gen_index:
                return dict(source=f"verify_rows[round={vr['round']}][{j}]",
                            input_position=vr["pre"] + j, gen_index=gen_index,
                            argmax_id=row["argmax_id"], max_score=row["max_score"],
                            n_max_tied=row["n_max_tied"],
                            tied_max_tokens=row["tied_max_tokens"],
                            margin_top1_top2=row["margin_top1_top2"],
                            tie_at_top=row["tie_at_top"])
    return None


def _pack(row: dict, source: str, input_position: int, gen_index: int) -> dict:
    return dict(source=source, input_position=input_position, gen_index=gen_index,
                argmax_id=row["argmax_id"], max_score=row["max_score"],
                n_max_tied=row["n_max_tied"], tied_max_tokens=row["tied_max_tokens"],
                margin_top1_top2=row["margin_top1_top2"], tie_at_top=row["tie_at_top"])


def semantics_check(rec: dict) -> dict:
    """核对 `_verify` 的位置语义：对每轮、每个 j>=1，`argmax(rows[j-1])` 必须等于记录的 `g[j]`。

    这是"决策行归属"正确性的前提；不通过说明位置映射错了，后续归因全部作废。
    """
    if rec["mode"] != "spec":
        return dict(applicable=False)
    dr = rec.get("decision_rows") or {}
    bypre = {v["pre"]: v for v in dr.get("verify_rows", [])}
    checked = bad = 0
    examples = []
    for r in rec["rounds"]:
        v = bypre.get(r["pre"])
        if not v:
            continue
        for j in range(1, r["Kp"]):
            checked += 1
            am = v["rows"][j - 1]["argmax_id"]
            if am != r["g"][j]:
                bad += 1
                if len(examples) < 3:
                    examples.append(dict(round=r["round"], pre=r["pre"], j=j,
                                         argmax_rows=am, g_j=r["g"][j]))
    return dict(applicable=True, checked=checked, mismatch=bad, ok=(bad == 0),
                examples=examples,
                rule="rows[j-1] 决定 g[j]（生成序列 s0+j），s0 = pre - prompt_len")


def deciding_row_for(rec: dict, gen_index: int) -> dict:
    """**真正决定** `gen_index` 那个 token 的决策行。

    ⚠️ 与"该位置存在一行"不同：某轮一旦被否决(`m<Kp`)，它的 `rows[m..]` 虽然算过却**被丢弃**，
    并不决定任何已输出 token。因此必须先用 `rounds[].emitted` 定位"哪一轮的哪个槽"产出了该 token，
    再按槽位取行。槽位 0 的行来自 `saved_next`（不在 verify_rows 里），按下述规则**重建**：
        saved_next(round k) = 上一轮 m==Kp 时的 rows[Kp-1]；否则上一轮的 repair 行。
    """
    dr = rec.get("decision_rows") or {}
    pl = dr["prompt_len"]
    if rec["mode"] == "ar":
        for r in dr.get("ar_rows", []):
            if r["gen_index"] == gen_index:
                return _pack(r, "ar_decode_row(s=1, eager)", r["input_position"], gen_index)
        return None

    bypre = {v["pre"]: v for v in dr.get("verify_rows", [])}
    repairs = dr.get("repair_lg", [])
    rounds = rec["rounds"]

    saved: dict = {}
    mis = 0
    for k, r in enumerate(rounds):
        if k == 0:
            saved[k] = dict(row=None, how="首轮 saved_next = target prefill 行（本工具未捕获）",
                            input_position=r["pre"] - 1)
        else:
            p = rounds[k - 1]
            if p["m"] == p["Kp"]:
                row = bypre[p["pre"]]["rows"][p["Kp"] - 1]
                saved[k] = dict(row=row,
                                how=f"saved_next = 上一轮(round={p['round']}, 全接受) rows[Kp-1]",
                                input_position=p["pre"] + p["Kp"] - 1)
            else:
                row = repairs[mis - 1] if mis - 1 < len(repairs) else None
                saved[k] = dict(row=row,
                                how=f"saved_next = 上一轮(round={p['round']}, 有否决) repair 行",
                                input_position=p["pre"] + p["m"])
        if r["m"] < r["Kp"]:
            mis += 1

    pos = 0
    for k, r in enumerate(rounds):
        n = len(r["emitted"])
        if pos <= gen_index < pos + n:
            slot = gen_index - pos
            out = dict(deciding_round=r["round"], deciding_slot=slot,
                       round_Kp=r["Kp"], round_m=r["m"], round_pre=r["pre"],
                       emitted_len=n, emitted=r["emitted"])
            if slot == 0:
                s = saved[k]
                if s.get("row") is None:
                    out.update(unavailable=s["how"], input_position=s["input_position"],
                               note="saved_next 不可用 ⇒ **不得**用别的行替代")
                    return out
                out.update(_pack(s["row"], f"saved_next[{s['how']}]",
                                 s["input_position"], gen_index))
            else:
                row = bypre[r["pre"]]["rows"][slot - 1]
                out.update(_pack(row, f"verify_rows[round={r['round']}][{slot - 1}]",
                                 r["pre"] + slot - 1, gen_index))
            out["same_input_position_as_other_side"] = True
            return out
        pos += n
    return dict(unavailable="未在 emitted 覆盖范围内")


def _orig_gen(kv: str, mode: str):
    """原失败实验（max_new=64）的生成序列，用于核对控制组是否**原样复现**。"""
    src = {"base_bf16": dict(ar="smoke8k_cudnn_prefill_20260927T135218Z.json",
                             spec="smoke8k_cudnn_prefill_20260927T135254Z.json"),
           "v_fp8": dict(ar="smoke8k_cudnn_prefill_20260927T135336Z.json",
                         spec="smoke8k_cudnn_prefill_20260927T135413Z.json")}[kv][mode]
    p = OUTDIR / src
    if not p.exists():
        return None, None
    j = json.loads(p.read_text(encoding="utf-8"))
    return [int(x) for x in j["run"]["gen"]], src


def main() -> int:
    tag = __import__("os").environ.get("QK_TAG", "qk32")
    recs = {}
    for g in ("control", "qk32"):
        for kv in ("base_bf16", "v_fp8"):
            for m in ("ar", "spec"):
                recs[(g, kv, m)] = _load(tag, g, kv, m)

    out: dict = dict(tag=tag, tool="g5_8k_qk32_analysis",
                     available={f"{g}/{kv}/{m}": (r is not None)
                                for (g, kv, m), r in recs.items()})
    missing = [k for k, v in out["available"].items() if not v]
    if missing:
        out["stage"] = "missing_runs"
        out["missing"] = missing
        p = OUTDIR / f"{tag}_analysis_partial.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[ana] 缺少运行：{missing} -> {p}")
        return 5

    for (g, kv, m), r in recs.items():
        if r.get("stage") != "complete":
            out["stage"] = "bad_run"
            out["bad"] = dict(group=g, kv=kv, mode=m, stage=r.get("stage"),
                              failure=r.get("failure"))
            p = OUTDIR / f"{tag}_analysis_failed.json"
            p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[ana] {g}/{kv}/{m} stage={r.get('stage')} ⇒ 停下（不产出成功结论）")
            return 5

    table = {}
    per_run = {}
    for g in ("control", "qk32"):
        for kv in ("base_bf16", "v_fp8"):
            ar = recs[(g, kv, "ar")]
            sp = recs[(g, kv, "spec")]
            ga, gs = [int(x) for x in ar["run"]["gen"]], [int(x) for x in sp["run"]["gen"]]
            dv = first_divergence(ga, gs)
            rows = {}
            for key, idx in (("divergence_position", dv["index"]),
                             ("known_index_31", 31)):
                if idx is None:
                    continue
                rows[key] = dict(
                    index=idx,
                    ar_gen=(ga[idx] if idx < len(ga) else None),
                    spec_gen=(gs[idx] if idx < len(gs) else None),
                    ar_row=deciding_row_for(ar, idx),
                    spec_row=deciding_row_for(sp, idx),
                    same_input_position=(
                        (deciding_row_for(ar, idx) or {}).get("input_position")
                        == (deciding_row_for(sp, idx) or {}).get("input_position")),
                    comparable=(idx <= dv["index"]),
                    note="只在**共同前缀**(= 首个分歧位置)上比较决策行；"
                         "两侧 input_position 必须相同才可比（本表给出 same_input_position）；"
                         "分歧之后的 logits 前缀不同，**不可**用于归因",
                )
            per_run_key = f"{g}/{kv}"
            table[per_run_key] = dict(
                ar_gen=ga, spec_gen=gs,
                sequences_equal=bool(ga == gs),
                agreeing_prefix_len=dv["agreeing_prefix_len"],
                divergence=dv,
                ar_len=len(ga), spec_len=len(gs),
                ar_stop=ar["run"]["stop_reason"], spec_stop=sp["run"]["stop_reason"],
                decision_rows=rows,
                verify_position_semantics=semantics_check(sp),
            )
    for g in ("control", "qk32"):
        for kv in ("base_bf16", "v_fp8"):
            for m in ("ar", "spec"):
                r0 = recs[(g, kv, m)]
                per_run[f"{g}/{kv}/{m}"] = dict(
                    run_id=r0["run_id"],
                    gen_len=r0["run"]["gen_len"], stop_reason=r0["run"]["stop_reason"],
                    peak_allocated_mib=r0["run"]["peak_allocated_mib"],
                    peak_reserved_mib=r0["run"]["peak_reserved_mib"],
                    prefill_logits_finite=r0["run"]["prefill_logits_finite"],
                    checks_passed=r0["checks"].get("passed"),
                    qk_scope=r0.get("qk_scope_report"),
                    target_qk_fp32=r0["precision_scope_effective"]["target_qk_fp32"],
                    prefill_affected=r0["precision_scope_effective"]["prefill_affected"],
                    ids_hash=r0["input"]["ids_hash"],
                )
    out["per_kv_ar_vs_spec"] = table
    out["per_run"] = per_run

    # 控制组是否复现已知分歧
    repro = {}
    for kv in ("base_bf16", "v_fp8"):
        c = table[f"control/{kv}"]
        repro[kv] = dict(
            diverged=c["divergence"]["diverged"],
            index=c["divergence"]["index"],
            ar_token=(c["divergence"]["a"]), spec_token=(c["divergence"]["b"]),
            known_expectation=("base_bf16: AR=1083 / Spec=5880（index 31）" if kv == "base_bf16"
                               else "v_fp8: AR=5880 / Spec=1083（index 31）"),
            matches_known=(c["divergence"]["diverged"] and c["divergence"]["index"] == 31),
        )
    out["control_reproduction"] = repro
    out["reproduction_problem"] = (not all(v["matches_known"] for v in repro.values()))

    # 干预是否改变了参考生成（各组 AR 之间、Spec 之间）
    changed = {}
    for kv in ("base_bf16", "v_fp8"):
        ca, ka = table[f"control/{kv}"]["ar_gen"], table[f"qk32/{kv}"]["ar_gen"]
        cs, ks = table[f"control/{kv}"]["spec_gen"], table[f"qk32/{kv}"]["spec_gen"]
        changed[kv] = dict(
            ar_equal=bool(ca == ka),
            ar_first_diff=first_divergence(ca, ka),
            spec_equal=bool(cs == ks),
            spec_first_diff=first_divergence(cs, ks),
        )
    out["intervention_changed_reference"] = changed

    # 控制组是否**原样复现**原失败实验（不是只"也分歧了"）
    orig = {}
    for kv in ("base_bf16", "v_fp8"):
        for m in ("ar", "spec"):
            og, src = _orig_gen(kv, m)
            cg = [int(x) for x in recs[("control", kv, m)]["run"]["gen"]]
            kg = [int(x) for x in recs[("qk32", kv, m)]["run"]["gen"]]
            orig[f"{kv}/{m}"] = dict(
                source_file=src, original_available=og is not None,
                control_equals_original=(og == cg) if og is not None else None,
                qk32_equals_original=(og == kg) if og is not None else None,
                control_first_diff=(first_divergence(og, cg) if og is not None else None),
                qk32_first_diff=(first_divergence(og, kg) if og is not None else None),
            )
    out["original_reproduction"] = orig
    out["control_reproduces_original"] = all(
        v["control_equals_original"] is True for v in orig.values())

    # 判读
    verdict = {}
    for kv in ("base_bf16", "v_fp8"):
        k = table[f"qk32/{kv}"]
        c = table[f"control/{kv}"]
        if k["sequences_equal"]:
            v = "INTERVENTION_FULLY_CONSISTENT_ON_THESE_CASES"
        elif (k["agreeing_prefix_len"] > c["agreeing_prefix_len"]):
            v = "DIVERGENCE_POSTPONED_PARTIAL_NOT_PASS"
        elif k["agreeing_prefix_len"] == c["agreeing_prefix_len"]:
            v = "UNCHANGED_STILL_INCONSISTENT"
        else:
            v = "WORSE"
        verdict[kv] = dict(
            verdict=v,
            control_prefix=c["agreeing_prefix_len"],
            qk32_prefix=k["agreeing_prefix_len"],
            control_diverged=c["divergence"]["diverged"],
            qk32_diverged=k["divergence"]["diverged"],
        )
    out["verdict"] = verdict
    out["verdict_rules"] = dict(
        full_consistency="只说明这两个测试用例通过；下一步才是扩展正确性回归，**不直接放行正式矩阵**",
        postponed="部分改善，**不是 PASS**",
        unchanged="报告首个分歧证据并停止；**不自动扩展**全模型 FP32 / 换 backend / 堆叠精度开关",
        forbidden=["以 margin 小豁免 token 一致性", "以文本意思接近豁免",
                   "以结构检查通过豁免", "把「新精度下对齐」写成「原配置原样通过」"],
        timing="本轮时间只作诊断记录，不做加速/减速声明",
    )
    p = OUTDIR / f"{tag}_analysis.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"== {tag} ==")
    for kv in ("base_bf16", "v_fp8"):
        c, k = table[f"control/{kv}"], table[f"qk32/{kv}"]
        print(f"[{kv}] control: equal={c['sequences_equal']} prefix={c['agreeing_prefix_len']} "
              f"div={c['divergence']['index']} ({c['divergence']['a']} vs {c['divergence']['b']})")
        print(f"[{kv}] qk32   : equal={k['sequences_equal']} prefix={k['agreeing_prefix_len']} "
              f"div={k['divergence']['index']} ({k['divergence']['a']} vs {k['divergence']['b']})")
        print(f"[{kv}] verdict: {verdict[kv]['verdict']}")
        print(f"[{kv}] AR changed by intervention: {not changed[kv]['ar_equal']}; "
              f"Spec changed: {not changed[kv]['spec_equal']}")
    print(f"[ana] reproduction_problem={out['reproduction_problem']}")
    print(f"[ana] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
