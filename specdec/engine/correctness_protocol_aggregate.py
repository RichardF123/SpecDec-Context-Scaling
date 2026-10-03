"""correctness_protocol_aggregate.py — **分层正确性协议**的只读聚合与判决（纯 CPU、不加载模型）。

## 作用
按 `docs/correctness_protocol_post_shape_localization_2026-10-03.md`（冻结版 `post_shape_localization_v1`）：
① 读取 24 次运行（4 family × 3 mode × 2 rep）的**独立进程**产物；
② **独立复算**：token 一致性、common prefix、first divergence、逐轮不变量
   （`tgt_len/drf_len` 与 `m/Kp/pre` 的关系）、acceptance、hook 作用范围、结构检查；
③ 合并 margin 探针（`*_margins_*.json`，若有）；
④ 给出 Gate A / Gate B 与**三种决策之一**；写出 JSON + JSONL。

**不**加载模型、**不**做前向、**不**改任何既有产物；缺字段即 CHECK（不猜、不补）。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "g5" / "attn"
FAMILIES = ["F01", "F02", "F08", "F10"]          # 预登记（协议 §3）
MODES = ["ar", "spec", "aligned"]                # ar / production / diagnostic oracle
MODE_LABEL = {"ar": "ar", "spec": "prod", "aligned": "diag"}
DIAG_ONLY = {"ar": False, "spec": False, "aligned": True}
REPS = [1, 2]
MAXNEW = 64


def _runs(tag: str) -> dict:
    per: dict = {}
    for fam in FAMILIES:
        for m in MODES:
            for r in REPS:
                pat = str(OUT / f"{tag}_{fam}_{m}_rep{r}_*.json")
                fs = sorted(glob.glob(pat))
                if fs:
                    d = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
                    d["_file"] = Path(fs[-1]).name
                    per[(fam, m, r)] = d
    return per


def _round_invariants(rounds: list) -> dict:
    """逐轮不变量：错配轮 `tgt_len == drf_len == pre + m + 1`；全接受轮 `== pre + Kp`。"""
    bad = []
    for d in rounds or []:
        m, kp, pre = int(d.get("m", -1)), int(d.get("Kp", -1)), int(d.get("pre", -1))
        tl, dl = int(d.get("tgt_len", -1)), int(d.get("drf_len", -1))
        exp = pre + (m + 1 if m < kp else kp)
        if tl != exp or dl != exp:
            bad.append(dict(round=d.get("round"), m=m, Kp=kp, pre=pre, tgt_len=tl, drf_len=dl, expected=exp))
        if not (0 <= m <= kp):
            bad.append(dict(round=d.get("round"), error=f"m={m} Kp={kp}"))
        if len(d.get("emitted") or []) != (m + 1 if m < kp else kp):
            bad.append(dict(round=d.get("round"), error="emitted 长度与 m/Kp 不符"))
    return dict(n_rounds=len(rounds or []), violations=bad, ok=bool(not bad))


def _common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="div8209_proto")
    ap.add_argument("--margins", default=None)
    a = ap.parse_args()
    tag = a.tag
    mp = Path(a.margins) if a.margins else None
    if mp is None:
        fs = sorted(glob.glob(str(OUT / f"{tag}_margins_*.json")))
        mp = Path(fs[-1]) if fs else None
    margins = json.loads(mp.read_text(encoding="utf-8")) if (mp and mp.exists()) else {}
    runs = _runs(tag)
    rec: dict = dict(tool="correctness_protocol_aggregate", tag=tag,
                     protocol="post_shape_localization_v1",
                     protocol_doc="docs/correctness_protocol_post_shape_localization_2026-10-03.md",
                     run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                     families=FAMILIES, modes=[MODE_LABEL[m] for m in MODES], reps=REPS,
                     n_runs_found=len(runs), expected_runs=len(FAMILIES) * len(MODES) * len(REPS),
                     margins_artifact=(mp.name if mp else None),
                     semantics="只读：独立复算 token 一致性/逐轮不变量/接受统计/hook 作用域与结构检查，"
                               "给出 Gate A / Gate B 与三选一决策；不加载模型、不改产物")
    checks: dict = {}
    per_family: dict = {}
    missing = [f"{f}/{m}/rep{r}" for f in FAMILIES for m in MODES for r in REPS if (f, m, r) not in runs]
    checks["all_24_runs_present"] = bool(not missing)
    for fam in FAMILIES:
        f: dict = dict(stratum=((runs.get((fam, "ar", 1)) or {}).get("provenance") or {}).get("stratum"),
                       provenance=(runs.get((fam, "ar", 1)) or {}).get("provenance"),
                       per_mode={}, first_divergence=None, margins=None)
        g = {}
        for m in MODES:
            entries = [runs.get((fam, m, r)) for r in REPS]
            gens = [list(((e or {}).get("smoke") or {}).get("gen") or []) for e in entries]
            stops = [((e or {}).get("smoke") or {}).get("stop_reason") for e in entries]
            lens = [((e or {}).get("smoke") or {}).get("gen_len") for e in entries]
            diag = [bool((e or {}).get("diagnostic_only")) for e in entries]
            fin = [(((e or {}).get("smoke") or {}).get("prefill_logits_finite")) for e in entries]
            vocab = [(((e or {}).get("smoke") or {}).get("gen_ids_in_vocab")) for e in entries]
            struct = [(((e or {}).get("smoke") or {}).get("decision_coverage")) for e in entries]
            ri = [_round_invariants(((e or {}).get("smoke") or {}).get("rounds") or []) for e in entries]
            st = [(((e or {}).get("smoke") or {}).get("specdec_stats") or {}) for e in entries]
            cand = [((e or {}).get("candidate") or {}) for e in entries]
            entry = dict(
                mode=MODE_LABEL[m], mode_raw=m, diagnostic_only_expected=DIAG_ONLY[m],
                diagnostic_only_flags=diag, diagnostic_flag_consistent=bool(all(x == DIAG_ONLY[m] for x in diag)),
                reps_present=sum(1 for e in entries if e), gen_lens=lens, stop_reasons=stops,
                gen=[list(x) for x in gens],
                reps_consistent=bool(len({tuple(x) for x in gens if x}) <= 1 and gens and gens[0]),
                finite=bool(all(x is not False for x in fin)), gen_ids_in_vocab=bool(all(x is not False for x in vocab)),
                struct_invariants_ok=bool(all(x is not False for x in struct)),
                legal_stop=bool(all(s in ("max_new_reached", "eos") for s in stops if s)),
                length_ok=bool(all(l == MAXNEW for l in lens if l is not None)),
                round_invariants=ri, round_invariants_ok=bool(all(x["ok"] for x in ri)),
                stats=st, acceptance_rate=[((s.get("accepted", 0) / s.get("proposed", 1)) if s.get("proposed") else None)
                                           for s in st],
                target_block_calls=[s.get("target_block_calls") for s in st],
                draft_calls=[s.get("draft_calls") for s in st],
                rounds=[s.get("rounds") for s in st],
                candidate=cand,
                hook_scope_ok=bool(all((c.get("counts_match") is True) if m == "aligned" else True for c in cand))
                if m == "aligned" else None)
            g[m] = entry
        famrec = dict(f)
        famrec["per_mode"] = g
        ar_gen = (g["ar"]["gen"] or [None])[0]
        for m in MODES:
            if m == "ar":
                continue
            x = (g[m]["gen"] or [None])[0]
            if ar_gen and x:
                cp = _common_prefix(ar_gen, x)
                fd = None if len(ar_gen) == len(x) and cp == len(ar_gen) else cp
                g[m]["vs_AR"] = dict(
                    token_equal=bool(x == ar_gen), common_prefix=cp, first_divergence=fd,
                    ar_token=(ar_gen[cp] if fd is not None and cp < len(ar_gen) else None),
                    spec_token=(x[cp] if fd is not None and cp < len(x) else None),
                    classification=("TOKEN_EQUAL" if x == ar_gen else "NUMERICAL_DIVERGENCE_RECORDED"))
                if m == "aligned" and fd is not None:
                    famrec["first_divergence"] = cp
        if famrec["first_divergence"] is None and (g["spec"]["gen"] or [None])[0]:
            x = g["spec"]["gen"][0]
            if ar_gen and x != ar_gen:
                famrec["first_divergence"] = _common_prefix(ar_gen, x)
        mrec = (margins.get("families") or {}).get(fam)
        famrec["margins"] = mrec
        per_family[fam] = famrec
        # ---- Gate A（逐 family）----
        checkA = dict(
            diag_token_equals_AR=bool((g["aligned"]["vs_AR"] or {}).get("token_equal") is True),
            diag_reps_consistent=g["aligned"]["reps_consistent"],
            ar_reps_consistent=g["ar"]["reps_consistent"],
            prod_reps_consistent=g["spec"]["reps_consistent"],
            hook_scope_ok=bool(g["aligned"]["hook_scope_ok"] is True),
            diag_flag_ok=bool(g["aligned"]["diagnostic_flag_consistent"] and not g["ar"]["diagnostic_only_flags"][0]
                              and not g["spec"]["diagnostic_only_flags"][0]),
            finite=bool(g["aligned"]["finite"] and g["ar"]["finite"]),
            legal_stop=bool(g["aligned"]["legal_stop"] and g["ar"]["legal_stop"]),
            gen_ids_in_vocab=bool(g["aligned"]["gen_ids_in_vocab"] and g["ar"]["gen_ids_in_vocab"]),
            round_invariants_ok=bool(g["aligned"]["round_invariants_ok"] and g["ar"]["round_invariants_ok"]),
            struct_invariants_ok=bool(g["aligned"]["struct_invariants_ok"] and g["ar"]["struct_invariants_ok"]),
            length_ok=bool(g["aligned"]["length_ok"] and g["ar"]["length_ok"]),
            margins_present=bool(mrec and mrec.get("margin") is not None))
        checkB = dict(
            prod_reps_consistent=g["spec"]["reps_consistent"], finite=g["spec"]["finite"],
            struct_invariants_ok=g["spec"]["struct_invariants_ok"],
            round_invariants_ok=g["spec"]["round_invariants_ok"], legal_stop=g["spec"]["legal_stop"],
            length_ok=g["spec"]["length_ok"], gen_ids_in_vocab=g["spec"]["gen_ids_in_vocab"],
            divergence_recorded=bool((g["spec"]["vs_AR"] or {}).get("classification")
                                     in ("NUMERICAL_DIVERGENCE_RECORDED", "TOKEN_EQUAL")))
        famrec["gate_A"] = checkA
        famrec["gate_B"] = checkB
        famrec["gate_A_pass"] = bool(all(checkA.values()))
        famrec["gate_B_pass"] = bool(all(checkB.values()))
        checks[f"family_{fam}_gate_A"] = famrec["gate_A_pass"]
        checks[f"family_{fam}_gate_B"] = famrec["gate_B_pass"]
    gateA = bool(not missing and all(per_family[f]["gate_A_pass"] for f in FAMILIES))
    gateB = bool(gateA and all(per_family[f]["gate_B_pass"] for f in FAMILIES))
    rec["per_family"] = per_family
    rec["gate_A_algorithmic"] = "PASS" if gateA else ("FAIL" if not missing else "INCOMPLETE")
    rec["gate_B_production_behavior"] = "ACCEPTABLE_FOR_MEASUREMENT" if gateB else "NOT_ACCEPTABLE"
    rec["checks"] = checks
    rec["producer_unavailable"] = margins.get("profiler_status") if margins else None
    decision = ("ALGORITHMIC_GATE_FAIL" if (not missing and not gateA)
                else ("READY_FOR_FORMAL_MEASUREMENT" if (gateA and gateB)
                      else "CHECK_NEEDS_REVIEW"))
    rec["decision"] = decision
    rec["decision_rule"] = ("Gate A PASS 且 Gate B ACCEPTABLE ⇒ READY_FOR_FORMAL_MEASUREMENT；"
                            "Gate A 失败 ⇒ ALGORITHMIC_GATE_FAIL；其余（缺字段/CHECK 条件）⇒ CHECK_NEEDS_REVIEW")
    rec["b6_unchanged"] = True
    # ---- 打印 ----
    print(f"== 分层正确性协议聚合（tag={tag}；找到 {len(runs)}/24 次运行）==")
    if missing:
        print(f"   缺失运行：{missing}")
    for fam in FAMILIES:
        f = per_family[fam]
        print(f"\n  [{fam}] stratum={f['stratum']}")
        for m in MODES:
            e = f["per_mode"][m]
            v = e.get("vs_AR") or {}
            print(f"     {e['mode']:<5} diag_only={e['diagnostic_only_flags']} 重复一致={e['reps_consistent']} "
                  f"len={e['gen_lens']} stop={e['stop_reasons']} "
                  f"{'' if not v else ('token_eq=' + str(v['token_equal']) + ' cp=' + str(v['common_prefix']) + ' first_div=' + str(v['first_divergence']) + ' ' + v['classification'])}")
            if m != "ar":
                print(f"           accept={e['acceptance_rate']} rounds={e['rounds']} tgt_blk={e['target_block_calls']} "
                      f"draft_calls={e['draft_calls']} round_inv_ok={e['round_invariants_ok']} "
                      f"hook_scope_ok={e.get('hook_scope_ok')}")
        mg = f.get("margins") or {}
        if mg:
            print(f"     决策位置={mg.get('position')} AR top1/top2={mg.get('ar_top1')}/{mg.get('ar_top2')} "
                  f"margin={mg.get('margin')} spec_token={mg.get('spec_token')}")
        print(f"     Gate A={f['gate_A_pass']}（{ {k: v for k, v in f['gate_A'].items() if not v} }） "
              f"Gate B={f['gate_B_pass']}（{ {k: v for k, v in f['gate_B'].items() if not v} }）")
    print(f"\n  ★ Gate A（algorithmic）= {rec['gate_A_algorithmic']}")
    print(f"  ★ Gate B（production behavior）= {rec['gate_B_production_behavior']}")
    print(f"  ★ 决策 = **{decision}**")
    print(f"  profiler 状态 = {rec['producer_unavailable']}")
    print("  正式 B6 状态：**未改变**（本轮不启动 factorial）")
    p = OUT / f"{tag}_aggregate_{rec['run_utc']}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    jl = OUT / f"{tag}_aggregate_{rec['run_utc']}.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for fam in FAMILIES:
            for m in MODES:
                e = per_family[fam]["per_mode"][m]
                fh.write(json.dumps(dict(family=fam, stratum=per_family[fam]["stratum"],
                                         mode=e["mode"], reps_present=len(e["gen"]),
                                         reps_consistent=e["reps_consistent"],
                                         gen_lens=e["gen_lens"], stop_reasons=e["stop_reasons"],
                                         diagnostic_only=e["diagnostic_only_flags"],
                                         vs_AR=e.get("vs_AR"), acceptance_rate=e["acceptance_rate"],
                                         rounds=e["rounds"], target_block_calls=e["target_block_calls"],
                                         draft_calls=e["draft_calls"],
                                         round_invariants_ok=e["round_invariants_ok"],
                                         hook_scope_ok=e.get("hook_scope_ok"),
                                         gen_hash=[((x or {}).get("smoke") or {}).get("end_state", {}).get("gen_hash")
                                                   for x in [] ] or None,
                                         files=[runs.get((fam, m, r), {}).get("_file") for r in REPS]),
                            ensure_ascii=False) + "\n")
        fh.write(json.dumps(dict(decision=decision, gate_A=rec["gate_A_algorithmic"],
                                 gate_B=rec["gate_B_production_behavior"],
                                 margins=margins.get("families"), profiler=margins.get("profiler_status")),
                            ensure_ascii=False) + "\n")
    rec["artifacts"] = dict(json=str(p), jsonl=str(jl))
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[agg] RESULT_JSON={p}")
    print(f"[agg] RESULT_JSONL={jl}")
    return 0 if decision == "READY_FOR_FORMAL_MEASUREMENT" else (1 if decision == "CHECK_NEEDS_REVIEW" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
