"""b6_branch_coverage_cpu.py — **B6 分支覆盖 + 验收规则**的只读 CPU 检查器（2026-10-05，C-3）。

## 为什么需要
`docs/design_freeze_g4_specdec.md:83` 要求 B6 覆盖 10 类 prompt 形态与若干分支；
**自然语言 prompt 不保证触发某个分支** ⇒ 覆盖必须由**实际 trace**证明。
Stage 2b 之前的分支结论是用临时脚本统计的；本模块把它固化为**可复跑、可判定**的检查。

## 判据（**只读产物**；不加载模型、不使用 GPU、不写数值产物）
对每个 (case, mode, rep) 产物：
  1. **必需字段**齐全 ⇒ 否则 `CASE_INCOMPLETE`（缺证据不得 PASS）；
  2. **配置符合性**（冻结候选配置）⇒ 违规即 `CASE_FAIL`；
  3. **分支覆盖**：逐分支判定 `COVERED` / `UNCOVERED`；**未触发即"未覆盖"，不得算通过**；
  4. **serial 门**（严格）：serial 的**完整 token IDs** 必须与**同一 case 的 AR** 逐 token 相同；
  5. **独立重复一致**：同一 mode 的两次重复必须逐 token 相同；
  6. **production 分歧**：只**记录**（首分歧 / 共同前缀 / 两侧长度 / 停止原因），**不**判控制逻辑失败、
     **不**自动归为无害浮点误差、**不**声明同输出无损加速。
  7. **后端/融合**：`requested_backend`、实际 `strategy`、`calls`、`fused_by_phase` 必须落盘；
     **禁止在非 prefill 阶段融合**；prefill 是否融合**按实现的分派规则**（用例可声明 `True/False/None`），
     不得用"一律 fused>0"误判合理的短输入分派。

## 与 Stage 2b 工具的关系（**重要**）
`stage_check` / `stage2b_verdict` 的校验以 **manifest family（`F\\d{2}`）**为单位；
字面 prompt 用例不是 family ⇒ 那些工具**不适用**（会判"family 形状非法"）。
本模块是字面用例的**下游校验**，避免"只新增入口、遗漏下游校验"。

## 用法
```powershell
python -X utf8 -m specdec.engine.b6_branch_coverage_cpu --tag div8209_b6 `
  --declared output/g5/attn/b6_declared_branches.json --expect-backend CUDNN
```
退出码：`0` 全部 `CASE_PASS` · `1` 存在 INCOMPLETE/FAIL · `3` 参数非法（不写产物）。
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
MODES = ("ar", "prod", "serial")
REPS = (1, 2)
# 非 prefill 阶段**禁止融合**（阶段名取自 harness 的 `by_phase`/`fused_by_phase`）
NON_PREFILL_PHASES = ("propose", "verify", "repair", "decode", "generate")
# 冻结候选配置中**与用例无关**的部分（逐 run 必须相等）
FROZEN_CONFIG = dict(k=4, chunk=2048, kv="base_bf16", model_dtype="torch.bfloat16",
                     target_kv_payload="base_bf16", draft_kv_payload="base_bf16")
# 必需字段（点号路径；`<mode>` 处按模式取值）
REQUIRED_PATHS = (
    "case", "mode", "rep", "tag", "run_id",
    "smoke.input.source_kind", "smoke.input.ctx", "smoke.input.ids_hash",
    "smoke.gen", "smoke.stop_reason", "smoke.specdec_stats",
    "smoke.cache_structure_ok", "smoke.cache_structure_n_checks",
    "smoke.round_invariants", "smoke.prefill_logits_finite",
    "smoke.attention_stats.strategy", "smoke.attention_stats.requested_backend",
    "smoke.attention_stats.calls", "smoke.attention_stats.fused_by_phase",
    "diag_precision_before_run", "diag_precision_after_run", "diag_precision_stable",
    "accounting.matches", "self_checks_all_ok", "verify_mode_after_run",
)


def _dig(rec: dict, path: str):
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_artifact(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def find_runs(tag: str, case: str, outdir: Path = OUTDIR) -> dict:
    """按 (mode, rep) 收集某 case 的产物（**排除** aggregate/verdict/margins/selftest）。"""
    out: dict = {}
    for p in sorted(glob.glob(str(Path(outdir) / f"{tag}_{case}_*_rep*_*.json"))):
        name = Path(p).name
        if any(x in name for x in ("aggregate", "verdict", "margins", "coverage", "selftest")):
            continue
        try:
            rec = load_artifact(Path(p))
        except Exception:                                             # noqa: BLE001
            continue
        if rec.get("tool") != "serial_oracle_gpu" or rec.get("case") != case:
            continue
        mode, rep = rec.get("mode"), rec.get("rep")
        if mode in MODES and isinstance(rep, int):
            out[(mode, rep)] = rec
    return out


def missing_required_fields(rec: dict) -> list:
    """缺字段判定。

    ⚠️ 语义区分（自测 G11b 抓到过一次误判）：**空容器不等于缺失** ——
    `fused_by_phase={}` 是**合法且有意义**的取值（"没有任何阶段发生融合"，例如 `arm=eager`
    或短输入按分派规则回退），真实 Stage 2b 产物就是这样。只有 `None` / 空字符串算"没有记录"。
    `smoke.gen` 例外：**空列表**意味着没有生成，不能算证据。
    """
    miss = []
    for path in REQUIRED_PATHS:
        v = _dig(rec, path)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss.append(path)
    gen = _dig(rec, "smoke.gen")
    if isinstance(gen, list) and not gen:
        miss.append("smoke.gen（为空 ⇒ 无生成证据）")
    ctx = _dig(rec, "smoke.input.ctx")
    if ctx is not None and not isinstance(ctx, int):
        miss.append("smoke.input.ctx（非整数）")
    return miss


def config_violations(rec: dict, *, expect_max_new: int | None = None) -> list:
    bad = []
    sc = rec.get("smoke_config") or {}
    inp = (_dig(rec, "smoke.input") or {})
    dt = inp.get("dtypes") or {}
    if sc.get("k") != FROZEN_CONFIG["k"]:
        bad.append(f"k={sc.get('k')}≠{FROZEN_CONFIG['k']}")
    if sc.get("chunk") != FROZEN_CONFIG["chunk"]:
        bad.append(f"chunk={sc.get('chunk')}≠{FROZEN_CONFIG['chunk']}")
    if sc.get("kv") != FROZEN_CONFIG["kv"]:
        bad.append(f"kv={sc.get('kv')}≠{FROZEN_CONFIG['kv']}")
    if sc.get("struct_check") is not True:
        bad.append("struct_check≠True")
    if dt.get("model") != FROZEN_CONFIG["model_dtype"]:
        bad.append(f"model_dtype={dt.get('model')}≠{FROZEN_CONFIG['model_dtype']}")
    if dt.get("target_kv_payload") != FROZEN_CONFIG["target_kv_payload"]:
        bad.append(f"target_kv_payload={dt.get('target_kv_payload')}")
    if dt.get("draft_kv_payload") not in (FROZEN_CONFIG["draft_kv_payload"], None):
        bad.append(f"draft_kv_payload={dt.get('draft_kv_payload')}")
    if expect_max_new is not None and sc.get("max_new") != expect_max_new:
        bad.append(f"max_new={sc.get('max_new')}≠用例声明 {expect_max_new}")
    dp = rec.get("diag_precision_after_run")
    # ⚠️ **缺字段**由 `missing_required_fields()` 归为"证据不足"（CASE_INCOMPLETE）；
    #    这里只在**字段存在**时判"值不对"（CASE_FAIL）——两者语义不同，不得混为一谈。
    if isinstance(dp, dict):
        for key in ("qk_fp32", "prob_fp32"):
            if dp.get(key) is not False:
                bad.append(f"diag.{key}={dp.get(key)!r}≠False")
    elif dp is not None:
        bad.append(f"diag_precision_after_run 类型非法：{type(dp).__name__}")
    # ⚠️ 只在**显式为 False**（即两处记录确实不同）时判违规；字段**缺失**由
    #    `missing_required_fields()` 归为"证据不足"（CASE_INCOMPLETE），不得混为"配置违规"。
    if rec.get("diag_precision_stable") is False:
        bad.append("diag_precision 运行前后不一致（配置漂移）")
    # 非 prefill 阶段禁止融合
    att = (_dig(rec, "smoke.attention_stats") or {})
    fused = att.get("fused_by_phase") or {}
    bad_fused = {ph: n for ph, n in fused.items() if ph in NON_PREFILL_PHASES and n}
    if bad_fused:
        bad.append(f"非 prefill 阶段出现融合调用：{bad_fused}（**禁止**）")
    return bad


def backend_report(rec: dict, *, expect_backend: str | None, expect_prefill_fused) -> dict:
    """后端/融合的阶段感知判定（**不**用"一律 fused>0"）。"""
    att = (_dig(rec, "smoke.attention_stats") or {})
    req = str(att.get("requested_backend") or "")
    fused = att.get("fused_by_phase") or {}
    eager = att.get("eager_by_phase") or {}
    problems = []
    if expect_backend and expect_backend.upper() not in req.upper():
        problems.append(f"requested_backend={req!r} 未包含 {expect_backend!r}")
    if not att.get("strategy"):
        problems.append("未记录实际 attention strategy")
    if expect_prefill_fused is True and not (fused.get("prefill") or 0) > 0:
        problems.append("用例声明**应触发** prefill 融合，但 fused_by_phase['prefill']=0")
    if expect_prefill_fused is False and (fused.get("prefill") or 0) > 0:
        problems.append("用例声明**不应**融合，但出现了 prefill 融合")
    # expect_prefill_fused is None ⇒ **按分派规则**，不做硬性要求（只记录）
    return dict(requested_backend=req, strategy=att.get("strategy"), calls=att.get("calls"),
                fused_by_phase=fused, eager_by_phase=eager,
                expect_prefill_fused=expect_prefill_fused, problems=problems,
                note=("expect_prefill_fused=None ⇒ 按实现的分派规则，不作硬性要求（短输入可能合理走 eager）"
                      if expect_prefill_fused is None else ""))


def branch_observations(rec: dict) -> dict:
    """从**实际 trace**（`smoke.rounds` + 停止原因 + 长度）统计分支触发次数。"""
    rounds = list((_dig(rec, "smoke.rounds") or []))
    gen = list((_dig(rec, "smoke.gen") or []))
    k = int((rec.get("smoke_config") or {}).get("k") or 4)
    stop = _dig(rec, "smoke.stop_reason")
    obs = dict(all_accept=0, first_reject=0, mid_reject=0, rollback=0, repair=0,
               trim_any=0, trim_last_round=0, eos_stop=0, eos_round=0,
               len_not_multiple_of_k=0, rounds=len(rounds), gen_len=len(gen))
    for i, rr in enumerate(rounds):
        m, kp = int(rr.get("m") or 0), int(rr.get("Kp") or 0)
        if rr.get("all_accept") or (kp and m == kp):
            obs["all_accept"] += 1
        if m == 0:
            obs["first_reject"] += 1
        elif 0 < m < kp:
            obs["mid_reject"] += 1
        if rr.get("rollback"):
            obs["rollback"] += 1
        if rr.get("repair"):
            obs["repair"] += 1
        if kp and kp < k:
            obs["trim_any"] += 1
            if i == len(rounds) - 1:
                obs["trim_last_round"] += 1
        # EOS：本轮候选里出现过 EOS（`cands`/`g` 含 eos id 由调用方给出；这里只按停止原因与 emitted 判定）
    if stop == "eos":
        obs["eos_stop"] = 1
    if gen and len(gen) % k != 0:
        obs["len_not_multiple_of_k"] = 1
    return obs


def pair_divergence(gen_a: list, gen_b: list) -> dict:
    """**显式的按模式比较**（不跨读）：返回 a→b 的首分歧与共同前缀。"""
    cp = 0
    for x, y in zip(gen_a, gen_b):
        if x != y:
            break
        cp += 1
    first = cp if cp < min(len(gen_a), len(gen_b)) else None
    return dict(first_divergence=first, common_prefix=cp, len_a=len(gen_a), len_b=len(gen_b))


def evaluate_case(case_id: str, declared: dict, runs: dict, *, expect_backend: str | None = None,
                  expect_config: dict | None = None) -> dict:
    """对单个 case 给出：必需字段 / 配置 / 分支覆盖 / serial 门 / 重复一致 / production 分歧。"""
    expect_config = expect_config or {}
    expect_max_new = expect_config.get("max_new")
    res: dict = dict(case_id=case_id, declared=dict(declared), per_mode={},
                     branches={}, reasons=[], verdict=None)
    required_modes = list(expect_config.get("modes") or MODES)
    # ---- 1) 逐 run：字段 + 配置 + 后端 ----
    field_problems, config_problems, backend_problems = [], [], []
    for mode in required_modes:
        for rep in REPS:
            rec = runs.get((mode, rep))
            if rec is None:
                field_problems.append(f"{mode}/rep{rep}: 产物缺失")
                continue
            miss = missing_required_fields(rec)
            if miss:
                field_problems.append(f"{mode}/rep{rep}: 缺字段 {miss}")
            cfgbad = config_violations(rec, expect_max_new=expect_max_new)
            if cfgbad:
                config_problems.append(f"{mode}/rep{rep}: {cfgbad}")
            bexp = None if mode == "ar" else declared.get("expect_prefill_fused") \
                if mode == "prod" else declared.get("expect_prefill_fused")
            br = backend_report(rec, expect_backend=expect_backend,
                                expect_prefill_fused=(declared.get("expect_prefill_fused")
                                                      if mode in ("prod", "serial") else False))
            if br["problems"]:
                backend_problems.append(f"{mode}/rep{rep}: {br['problems']}")
            res["per_mode"].setdefault(mode, {})[f"rep{rep}"] = br
    # ---- 2) 分支覆盖（以 prod/serial 的 trace 为证；AR 无投机轮）----
    obs_total = {}
    for mode in ("prod", "serial"):
        for rep in REPS:
            rec = runs.get((mode, rep))
            if rec is None:
                continue
            for kk, vv in branch_observations(rec).items():
                if isinstance(vv, int) and kk != "rounds" and kk != "gen_len":
                    obs_total[kk] = obs_total.get(kk, 0) + vv
    for branch in (declared.get("branches") or []):
        n = int(obs_total.get(branch, 0))
        res["branches"][branch] = dict(observed=n, verdict="COVERED" if n > 0 else "UNCOVERED")
    # ---- 3) serial 门（严格）+ 重复一致 ----
    gate = dict(serial_vs_ar={}, production_vs_ar={}, repeats={})
    for rep in REPS:
        ar, se, pr = runs.get(("ar", rep)), runs.get(("serial", rep)), runs.get(("prod", rep))
        if ar and se:
            d = pair_divergence(list(_dig(se, "smoke.gen") or []), list(_dig(ar, "smoke.gen") or []))
            gate["serial_vs_ar"][f"rep{rep}"] = d
            if d["first_divergence"] is not None:
                res["reasons"].append(f"serial/rep{rep} 与 AR 不一致（首分歧 {d['first_divergence']}）")
        if ar and pr:
            gate["production_vs_ar"][f"rep{rep}"] = pair_divergence(
                list(_dig(pr, "smoke.gen") or []), list(_dig(ar, "smoke.gen") or []))
    for mode in required_modes:
        g1 = list(_dig(runs.get((mode, 1)) or {}, "smoke.gen") or [])
        g2 = list(_dig(runs.get((mode, 2)) or {}, "smoke.gen") or [])
        same = bool(g1 and g2 and g1 == g2)
        gate["repeats"][mode] = same
        if not same:
            res["reasons"].append(f"{mode} 两次独立重复不一致")
    res["gate"] = gate
    # ---- 4) 判定 ----
    if field_problems:
        res["reasons"].append("**缺字段**：" + "；".join(field_problems[:4]))
    if config_problems:
        res["reasons"].append("**配置违规**：" + "；".join(config_problems[:4]))
    if backend_problems:
        res["reasons"].append("**后端/融合违规**：" + "；".join(backend_problems[:4]))
    uncovered = [b for b, v in res["branches"].items() if v["verdict"] == "UNCOVERED"]
    if uncovered:
        res["reasons"].append(f"**未覆盖分支**（未触发即不得算通过）：{uncovered}")
    hard_fail = bool(config_problems or backend_problems
                     or any(d["first_divergence"] is not None for d in gate["serial_vs_ar"].values())
                     or not all(gate["repeats"].values()))
    if hard_fail:
        res["verdict"] = "CASE_FAIL"
    elif field_problems or uncovered:
        res["verdict"] = "CASE_INCOMPLETE"
    else:
        res["verdict"] = "CASE_PASS"
    res["production_divergence_note"] = (
        "production 与 AR 的差异**只作发散轨迹记录**：不判控制逻辑失败、不自动归为无害浮点误差、"
        "不声明同输出无损加速")
    res["branch_observations_total"] = obs_total
    return res


def _case_spec(declared: dict, case_id: str) -> dict:
    spec = dict(declared.get(case_id) or {})
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description="B6 分支覆盖 + 验收规则（只读 CPU）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--declared", default=None,
                    help="用例声明 JSON：{cases:{case_id:{branches,max_new,expect_prefill_fused}},"
                         "set_requirements:{branches,types}}；**省略则用内建单一来源** `b6_cases.py`")
    ap.add_argument("--cases", default=None, help="逗号分隔；默认取声明的全部 case")
    ap.add_argument("--expect-backend", default=None, help="如 CUDNN（大小写不敏感子串匹配）")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--list-cases", action="store_true", help="只打印内建用例表（不评估）")
    a = ap.parse_args()
    from . import b6_cases as BC
    if a.list_cases:
        for c in BC.B6_CASES:
            print(f"  {c['case_id']:10s} kind={c['kind']:26s} max_new={c['max_new']:2d} "
                  f"branches={list(c['branches'])} expect_prefill_fused={c['expect_prefill_fused']}")
        return 0
    if a.declared:
        dpath = Path(a.declared)
        if not dpath.exists():
            print(f"[b6cov] 声明文件不存在：{dpath} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
            return 3
        try:
            declared = json.loads(dpath.read_text(encoding="utf-8-sig"))
        except Exception as e:                                        # noqa: BLE001
            print(f"[b6cov] 声明文件不可解析：{e} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
            return 3
    else:
        declared = BC.declared_document()
        dpath = None
    if not isinstance(declared, dict) or not declared:
        print("[b6cov] 声明为空 ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    # 兼容两种形态：{cases:{...}, set_requirements:{...}} 或直接 {case_id: {...}}
    if isinstance(declared.get("cases"), dict):
        case_map = declared["cases"]
        set_req = declared.get("set_requirements") or {}
    else:
        case_map, set_req = declared, {}
    if not case_map:
        print("[b6cov] 声明里没有任何 case ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    cases = ([x.strip() for x in a.cases.split(",") if x.strip()] if a.cases else sorted(case_map))
    missing_decl = [c for c in cases if c not in case_map]
    if missing_decl:
        print(f"[b6cov] 用例未在声明中出现：{missing_decl} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    out = dict(tool="b6_branch_coverage_cpu", tag=a.tag, cases=cases,
               expect_backend=a.expect_backend,
               declared_file=(str(dpath) if dpath else "内建 b6_cases.py（单一来源）"),
               set_requirements=set_req,
               run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
               cases_result={})
    print(f"== B6 分支覆盖检查 == tag={a.tag} cases={cases} expect_backend={a.expect_backend}")
    for c in cases:
        spec = dict(case_map.get(c) or {})
        runs = find_runs(a.tag, c, Path(a.outdir))
        r = evaluate_case(c, spec, runs, expect_backend=a.expect_backend,
                          expect_config=dict(max_new=spec.get("max_new"),
                                             modes=spec.get("modes") or list(MODES)))
        out["cases_result"][c] = r
        print(f"  [{c}] verdict={r['verdict']}  分支={ {b: v['verdict'] for b, v in r['branches'].items()} }")
        for x in r["reasons"]:
            print(f"      · {x}")
        pv = r["gate"]["production_vs_ar"]
        print(f"      production vs AR（**只记录**）："
              f"{ {k: v['first_divergence'] for k, v in pv.items()} }  "
              f"serial vs AR：{ {k: v['first_divergence'] for k, v in r['gate']['serial_vs_ar'].items()} }")
    # ---- 集合级要求：跨全部用例合计的分支/类型覆盖 ----
    set_branches: dict = {}
    types_seen = set()
    for c, r in out["cases_result"].items():
        for b, n in (r.get("branch_observations_total") or {}).items():
            set_branches[b] = set_branches.get(b, 0) + int(n)
        if case_map.get(c, {}).get("kind"):
            types_seen.add(case_map[c]["kind"])
    set_branch_verdict = {b: ("COVERED" if set_branches.get(b, 0) > 0 else "UNCOVERED")
                          for b in (set_req.get("branches") or [])}
    set_type_verdict = {t: ("COVERED" if t in types_seen else "UNCOVERED")
                        for t in (set_req.get("types") or [])}
    out["set_result"] = dict(branches=set_branch_verdict, types=set_type_verdict,
                             observed=dict(set_branches),
                             note="集合级要求未满足 ⇒ 整体不得判 PASS（未触发即未覆盖）")
    if set_branch_verdict or set_type_verdict:
        print(f"  集合级：分支={set_branch_verdict}")
        print(f"  集合级：类型={set_type_verdict}")
    verdicts = [r["verdict"] for r in out["cases_result"].values()]
    n_pass = verdicts.count("CASE_PASS")
    set_ok = all(v == "COVERED" for v in set_branch_verdict.values()) and \
        all(v == "COVERED" for v in set_type_verdict.values())
    out["summary"] = dict(n_cases=len(verdicts), n_pass=n_pass,
                          n_incomplete=verdicts.count("CASE_INCOMPLETE"),
                          n_fail=verdicts.count("CASE_FAIL"),
                          set_requirements_ok=bool(set_ok),
                          all_pass=bool(n_pass == len(verdicts) and verdicts and set_ok))
    print(f"== 汇总：CASE_PASS={out['summary']['n_pass']}/{len(verdicts)} "
          f"INCOMPLETE={out['summary']['n_incomplete']} FAIL={out['summary']['n_fail']} "
          f"集合级要求={out['summary']['set_requirements_ok']}")
    if not a.no_write:
        p = Path(a.outdir) / f"{a.tag}_b6_branch_coverage_{out['run_utc']}.json"
        out["artifact_json"] = str(p)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[b6cov] RESULT_JSON={p}")
    return 0 if out["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
