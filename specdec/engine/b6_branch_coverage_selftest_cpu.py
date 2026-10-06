"""b6_branch_coverage_selftest_cpu.py — C-1/C-2/C-3 新增能力的 **CPU 自测**（纯 CPU；不加载模型、不使用 GPU）。

## rev.2：新增"验收器漏洞"回归（每个漏洞都先复现、再修复、再断言**最终 verdict**）

| 组 | 用例 | 期望 |
| --- | --- | --- |
| **G1** | 输入计划：都不设⇒legacy；`SMOKE_FAMILY`⇒manifest；`SMOKE_PROMPT`⇒literal；两者同设⇒显式报错；`prompt_sha1` 稳定 | 成立（未设参数时行为不变） |
| **G2** | 短输入 `ctx=5` 如实记录、不被判违规 | 不违规 |
| **G3** | EOS 停止 + 非 k 整数倍 + 末轮裁剪 | 三分支 `COVERED` |
| **G4** | 声明分支未触发 | `UNCOVERED` ⇒ `CASE_INCOMPLETE`（非 PASS） |
| **G5** | 缺字段 / `k=3` / `qk_fp32=True` / 非 prefill 融合 | 缺字段⇒INCOMPLETE；其余⇒FAIL |
| **G6** | serial 与 AR 同长度 token 差异 | `CASE_FAIL` |
| **G7** | 分歧字段不串读（含误导字段被忽略；反向情形） | 各自独立 |
| **G8** | production 分歧只记录、不污染 serial 判决、无"无损"表述 | 成立 |
| **G9** | CLI 声明文件缺失 | exit 3 且不写产物 |
| **G10** | 用例表与旧 `g4_b6_correctness.py:CASES` 逐字一致 + 类型/分支声明齐备 | 成立 |
| **G11** | 集合级分支要求未满足 ⇒ 非 PASS；齐备 ⇒ exit 0 | 成立 |
| **G12 序列一致性** | serial 为 AR 严格前缀 / AR 为 serial 严格前缀 / 同长度差异 | 三者均 `CASE_FAIL`（长度不同也算不一致） |
| **G13 硬检查验值** | `self_checks_all_ok` / `cache_structure_ok` / `prefill_logits_finite` / `round_invariants.all_ok` / `accounting.matches` / `checks_passed` / `gen_ids_in_vocab` / 杀开关 / `stage` **逐个 False** | 逐个 `CASE_FAIL` |
| **G14 硬检查缺失/非法类型** | 上述字段逐个**缺失**、逐个**类型非法** | 缺失⇒`INCOMPLETE`；非法类型⇒非 PASS |
| **G15 自洽性** | `stop=max_new_reached` 却短输出 / `gen_len` 不符 / `stop_reason` 非法 / `stop=eos` 但末 token≠eos | 逐个 `CASE_FAIL`（沿用冻结停止规则） |
| **G16 AR 融合 prefill** | `arm=cudnn_prefill` 且 AR 与 Spec **都**触发 prefill 融合 | **不**被误拒（策略一致） |
| **G17 短输入分派** | `expect=None`：无融合**且无**回退理由 ⇒ `INCOMPLETE`；无融合**但有**记录的回退理由 ⇒ 可 PASS | 成立（`None` ≠ 不检查策略） |
| **G18 跨模式/跨重复匹配** | input hash / 实际长度 / `k`·`chunk`·`max_new` 不一致；跨 rep 不一致；AR 误加载 draft | 逐个 `CASE_FAIL` |
| **G19 产物选择** | 同一 `(mode,rep)` 重复产物 / 非法 `rep` / 身份字段不符 | 显式报告且非 PASS（**不**静默择一） |
| **G20 完整有效夹具** | 全字段合法、硬检查全通过、跨模式/rep 一致、分支齐备 | `CASE_PASS` |
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import b6_branch_coverage_cpu as C
from . import family_input as FI

TAG = "div8209_b6"
RUNID = "20261006T000000Z"
EOS = 212


def _round(rnd: int, kp: int, m: int, *, all_accept=None, rollback=False, repair=False,
           emitted=None, pre=8) -> dict:
    return dict(round=rnd, Kp=kp, m=m, pre=pre, tgt_len=pre + (m + 1 if m < kp else kp),
                drf_len=pre + (m + 1 if m < kp else kp),
                cands=list(range(100 + rnd * 10, 100 + rnd * 10 + kp)),
                g=list(range(100 + rnd * 10, 100 + rnd * 10 + kp)),
                emitted=list(emitted if emitted is not None else range(100 + rnd * 10, 100 + rnd * 10 + m)),
                rollback=rollback, repair=repair, correction=None,
                all_accept=bool(all_accept if all_accept is not None else (m == kp)),
                tgt_len_after=pre + (m + 1 if m < kp else kp),
                drf_len_after=pre + (m + 1 if m < kp else kp))


def _mk(case: str, mode: str, rep: int, *, gen: list, rounds: list, stop: str = "max_new_reached",
        ctx: int = 12, k: int = 4, max_new: int = 16, arm: str = "cudnn_prefill",
        strategy: str | None = None, fused_by_phase=None, fallbacks=None, reasons=None,
        qk32: bool = False, prob32: bool = False, drop=(), overrides: dict | None = None,
        ids_hash: str = "deadbeefcafe0001", eos_id: int | None = None, eos_seen=None,
        draft: str | None = None, stage: str = "complete", self_ok=True, cache_ok=True,
        finite=True, inv_ok=True, checks_passed=True, accounting_matches=True,
        gen_len=None, uncovered=None, source_kind: str = "literal_prompt",
        case_in_record: str | None = None, tag_in_record: str | None = None,
        mode_in_record: str | None = None) -> dict:
    """合成一份**默认完全合法**的产物；各用例只改需要出问题的那一项。"""
    eos_id = EOS if eos_id is None else eos_id
    if eos_seen is None:
        eos_seen = (stop == "eos")
    if strategy is None:
        strategy = ("eager" if arm == "eager" else "cudnn_prefill")
    if fused_by_phase is None:
        fused_by_phase = {} if arm == "eager" else {"prefill": 2}
    prefill_fused = int(fused_by_phase.get("prefill") or 0)
    draft_kv = draft if draft is not None else ("not_loaded" if mode == "ar" else "fp16")
    # ---- `smoke.checks` 逐条明细（**按真实产物结构**合成；rev.5 起验收器要逐条核对
    #      `required/executed/ok`，因此夹具必须与真实产物同构，否则自测只会测到"夹具不像真的"）----
    _uncovered = list(uncovered or [])
    _entries = [
        dict(name="finiteness:prefill/target", required=True, executed=True, ok=bool(finite),
             detail="无条件必需（该分支必然执行）"),
        dict(name="finiteness:prefill/draft", required=True, executed=True, ok=bool(finite),
             detail="无条件必需（该分支必然执行）"),
        dict(name="dispatch_only_prefill", required=True, executed=True, ok=True, detail="仅 prefill 融合"),
        dict(name="attention_assert", required=True, executed=True, ok=True, detail="策略断言=ok"),
        dict(name="weights_bitmatch", required=True, executed=True, ok=True, detail="逐张量相等"),
        dict(name="round_invariants", required=True, executed=True, ok=bool(inv_ok), detail="逐轮不变量"),
        dict(name="accounting_identity", required=True, executed=True, ok=bool(accounting_matches),
             detail="调用计数恒等式"),
    ] + [dict(name=nm, required=False, executed=False, ok=None, detail="该分支本次未执行（非必需）")
         for nm in _uncovered]
    _n_req = sum(1 for e in _entries if e["required"] is True)
    _failures = ([] if checks_passed else
                 [dict(name="（合成夹具）未通过项", detail="fixture: checks_passed=False")])
    rec = dict(
        tool="serial_oracle_gpu", tag=(tag_in_record or TAG), run_id=RUNID, case=(case_in_record or case),
        mode=(mode_in_record or mode), rep=rep, stage=stage, model_run_happened=True,
        env_qualified=True, blocked_by=None, diagnostic_only=True,
        smoke_config=dict(ctx=ctx, chunk=2048, k=k, max_new=max_new, kv="base_bf16", arm=arm,
                          mode=("ar" if mode == "ar" else "spec"), family=case,
                          target_rope="cpu_fp32", bf16_red_red="1", struct_check=True),
        diag_precision_before_run=dict(qk_fp32=qk32, prob_fp32=prob32),
        diag_precision_after_run=dict(qk_fp32=qk32, prob_fp32=prob32),
        diag_precision_stable=True,
        accounting=dict(mode=mode, matches=accounting_matches, target_generation_calls_measured=10,
                        expected=10),
        self_checks_all_ok=self_ok, verify_mode_after_run="block",
        verify_mode_at_run_end=("serial" if mode == "serial" else "block"),
        smoke=dict(gen=list(gen), gen_len=(len(gen) if gen_len is None else gen_len),
                   stop_reason=stop, gen_ids_in_vocab=True, prefill_logits_finite=finite,
                   checks_passed=checks_passed, n_rounds=len(rounds), rounds=rounds,
                   decision_coverage=True,
                   checks=dict(passed=checks_passed, n_required=_n_req, n_checks=len(_entries),
                               uncovered=list(_uncovered), failures=_failures,
                               not_applicable=["finiteness:propose/target", "finiteness:verify/draft"],
                               checks=_entries,
                               rule="只有 passed=True 才允许 exit 0；执行结束 ≠ 验证通过",
                               finiteness_coverage=dict(n_checked=148, first_bad=None,
                                                        by_phase_model={"prefill/target": dict(ok=True),
                                                                        "verify/target": dict(ok=True)})),
                   specdec_stats=dict(rounds=len(rounds), proposed=8, accepted=4,
                                      target_block_calls=len(rounds), draft_calls=8, rollbacks=1,
                                      mismatch_rounds=1, all_accept_rounds=1),
                   round_invariants=dict(all_ok=inv_ok, mode=("ar" if mode == "ar" else "spec"),
                                         per_round=[]),
                   cache_structure_ok=cache_ok, cache_structure_n_checks=40,
                   end_state=dict(stop_reason=stop, gen_len=(len(gen) if gen_len is None else gen_len),
                                  max_new=max_new, eos_id=eos_id, eos_seen=eos_seen, prompt_len=ctx),
                   attention_stats=dict(strategy=strategy, requested_backend="CUDNN_ATTENTION",
                                        calls=dict(eager=100, fused=prefill_fused),
                                        by_phase=dict(prefill=2 + prefill_fused, decode=10),
                                        fused_by_phase=dict(fused_by_phase),
                                        eager_by_phase=dict(prefill=2, decode=10),
                                        s_max_by_phase=dict(prefill=2048, decode=1),
                                        fused_s_max_by_phase=({"prefill": 2048} if prefill_fused else {}),
                                        fallbacks=list(fallbacks or []),
                                        fallback_count=len(fallbacks or []),
                                        eager_reason_kinds=dict(reasons or {}), errors=[]),
                   kv_format=dict(target="fp16", draft=draft_kv),
                   input=dict(ctx=ctx, ids_hash=ids_hash, source_kind=source_kind,
                              prompt_id="p1", prompt_sha1="aabbccdd00112233", input_tokens=ctx,
                              eos_config=dict(eos_id=eos_id),
                              dtypes=dict(model="torch.bfloat16", target_kv_payload="base_bf16",
                                          draft_kv_payload="base_bf16"))))
    for path in drop:
        cur = rec
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else {}
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    for path, val in (overrides or {}).items():
        cur = rec
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return rec


def _write(outdir: Path, recs: list, *, names: dict | None = None) -> None:
    for i, r in enumerate(recs):
        name = (names or {}).get(i) or (f"{r['tag']}_{r['case']}_{r['mode']}_rep{r['rep']}_"
                                        f"{r.get('run_id', RUNID)}.json")
        (outdir / name).write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")


def _trio(case: str, *, ar_gen: list, prod_gen: list, serial_gen: list, **kw) -> list:
    """三模式 × 两重复的合成产物；AR 无投机轮（`rounds=[]`），Spec 侧用同一份 rounds。"""
    rounds = kw.pop("rounds", [_round(0, 4, 4)])
    out = []
    for mode, g in (("ar", ar_gen), ("prod", prod_gen), ("serial", serial_gen)):
        for rep in (1, 2):
            out.append(_mk(case, mode, rep, gen=g, rounds=([] if mode == "ar" else list(rounds)),
                           **kw))
    return out


DECL = dict(natural_ok=dict(branches=["all_accept", "trim_last_round"], max_new=16,
                            expect_prefill_fused=True),
            eos_case=dict(branches=["eos_stop", "len_not_multiple_of_k"], max_new=16,
                          expect_prefill_fused=None),
            strict=dict(branches=["all_accept"], max_new=16, expect_prefill_fused=None))
AR16 = list(range(200, 216))


def main() -> int:
    checks: list = []

    def ck(name: str, ok: bool, extra=None) -> None:
        checks.append((name, bool(ok), extra))

    tmp = Path(tempfile.mkdtemp(prefix="b6cov_"))
    counter = {"n": 0}

    def ev(case: str, spec: dict, recs: list, *, expect_backend="CUDNN", max_new=16,
           names: dict | None = None) -> dict:
        """写盘 → `find_runs` → `evaluate_case`（断言最终 verdict 用）。"""
        counter["n"] += 1
        d = tmp / f"d{counter['n']}"
        d.mkdir()
        _write(d, [r for r in recs if r is not None], names=names)
        runs, probs = C.find_runs(TAG, case, d)
        return C.evaluate_case(case, spec, runs, expect_backend=expect_backend,
                               expect_config=dict(max_new=max_new), run_problems=probs)

    def rc_of(outdir: Path, declared: dict, cases: str | None = None) -> int:
        buf = io.StringIO()
        old = sys.argv
        dp = outdir / "_decl.json"
        dp.write_text(json.dumps(declared, ensure_ascii=False), encoding="utf-8")
        args = ["b6_branch_coverage_cpu", "--tag", TAG, "--declared", str(dp), "--outdir", str(outdir)]
        if cases:
            args += ["--cases", cases]
        try:
            sys.argv = args
            with contextlib.redirect_stdout(buf):
                return C.main()
        finally:
            sys.argv = old

    try:
        # ---------------- G1：输入计划 ----------------
        keys = ("SMOKE_PROMPT", "SMOKE_PROMPT_ID", "SMOKE_FAMILY")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            ck("G1_都不设⇒legacy（默认行为不变）", FI.resolve_input_plan()["mode"] == "legacy", None)
            os.environ["SMOKE_FAMILY"] = "F09"
            ck("G1_设FAMILY⇒manifest", FI.resolve_input_plan()["mode"] == "manifest", None)
            os.environ.pop("SMOKE_FAMILY")
            os.environ["SMOKE_PROMPT"] = "Hello"
            p1 = FI.resolve_input_plan()
            ck("G1_设PROMPT⇒literal_prompt", p1["mode"] == "literal_prompt", None)
            os.environ["SMOKE_PROMPT_ID"] = "natural-1"
            ck("G1_prompt_id被记录", FI.resolve_input_plan().get("prompt_id") == "natural-1", None)
            ck("G1_同一文本prompt_sha1稳定",
               FI.resolve_input_plan()["prompt_sha1"] == p1["prompt_sha1"], None)
            os.environ["SMOKE_PROMPT"] = "Hello!"
            ck("G1_不同文本sha1不同", FI.resolve_input_plan()["prompt_sha1"] != p1["prompt_sha1"], None)
            os.environ["SMOKE_FAMILY"] = "F09"
            err = None
            try:
                FI.resolve_input_plan()
            except Exception as e:                                    # noqa: BLE001
                err = type(e).__name__
            ck("G1_PROMPT与FAMILY同时设⇒显式报错", err == "FamilyInputError", err)
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

        # ---------------- G20：完整有效夹具 ⇒ CASE_PASS ----------------
        r_ok = ev("natural_ok", DECL["natural_ok"],
                  _trio("natural_ok", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        rounds=[_round(0, 4, 4), _round(1, 3, 2)]))
        ck("G20_完整有效夹具⇒CASE_PASS", r_ok["verdict"] == "CASE_PASS",
           (r_ok["verdict"], r_ok["reasons"][:3]))
        ck("G20_硬检查全 ok",
           all(v["state"] == "ok" for v in r_ok["hard_checks"].values()),
           {k: v for k, v in r_ok["hard_checks"].items() if v["state"] != "ok"})

        # ---------------- G2/G3：短输入 + EOS/裁剪分支 ----------------
        eos_gen = AR16[:13]
        rounds_eos = [_round(0, 4, 4), _round(1, 4, 4), _round(2, 3, 3), _round(3, 1, 1)]
        eos_recs = _trio("eos_case", ar_gen=eos_gen, prod_gen=list(eos_gen), serial_gen=list(eos_gen),
                         stop="eos", rounds=rounds_eos, ctx=5, eos_id=EOS)
        r_eos = ev("eos_case", DECL["eos_case"], eos_recs)
        ck("G2_短输入ctx=5如实记录且不违规", r_eos["verdict"] == "CASE_PASS",
           (r_eos["verdict"], r_eos["reasons"][:2]))
        ck("G3_EOS/非k倍数/末轮裁剪均COVERED",
           all(v["verdict"] == "COVERED" for v in r_eos["branches"].values()), r_eos["branches"])
        ck("G3b_短输入prompt_id/prompt_sha1/input_tokens如实记录",
           all(C._dig(rec, "smoke.input.input_tokens") == 5
               and C._dig(rec, "smoke.input.prompt_id")
               and C._dig(rec, "smoke.input.prompt_sha1") for rec in eos_recs), None)

        # ---------------- G4：未触发分支 ⇒ INCOMPLETE ----------------
        r4 = ev("eos_case", DECL["eos_case"],
                _trio("eos_case", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16)))
        ck("G4_未触发EOS记UNCOVERED", r4["branches"]["eos_stop"]["verdict"] == "UNCOVERED", None)
        ck("G4_未覆盖⇒CASE_INCOMPLETE（非PASS）", r4["verdict"] == "CASE_INCOMPLETE", r4["verdict"])
        ck("G4b_理由点名未覆盖分支", any("未覆盖分支" in x for x in r4["reasons"]), r4["reasons"][:2])

        # ---------------- G5：缺字段 / 配置违规 ----------------
        recs5 = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        for r in recs5:
            if (r["mode"], r["rep"]) == ("prod", 1):
                r.pop("diag_precision_after_run", None)
        r5 = ev("strict", DECL["strict"], recs5)
        ck("G5a_缺diag_precision⇒INCOMPLETE", r5["verdict"] == "CASE_INCOMPLETE", r5["verdict"])
        r5b = ev("strict", DECL["strict"],
                 _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16), k=3))
        ck("G5b_k=3⇒CASE_FAIL", r5b["verdict"] == "CASE_FAIL", r5b["verdict"])
        r5c = ev("strict", DECL["strict"],
                 _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16), qk32=True))
        ck("G5c_qk_fp32=True⇒CASE_FAIL", r5c["verdict"] == "CASE_FAIL", r5c["verdict"])
        r5d = ev("strict", DECL["strict"],
                 _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                       fused_by_phase={"verify": 3}))
        ck("G5d_非prefill融合⇒CASE_FAIL", r5d["verdict"] == "CASE_FAIL", r5d["verdict"])

        # ---------------- G6：serial 门（同长度 token 差异）----------------
        bad5 = list(AR16); bad5[5] = 999
        r6 = ev("strict", DECL["strict"],
                _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=bad5))
        ck("G6_serial同长度差异⇒CASE_FAIL", r6["verdict"] == "CASE_FAIL"
           and r6["gate"]["serial_vs_ar"]["rep1"]["first_token_diff"] == 5, r6["verdict"])

        # ---------------- G7/G8：分歧字段不串读；production 只记录 ----------------
        pr_div = list(AR16); pr_div[7] = 777
        r7recs = _trio("strict", ar_gen=AR16, prod_gen=pr_div, serial_gen=list(AR16))
        r7 = ev("strict", DECL["strict"], r7recs)
        ck("G7a_serial==AR⇒严格相等", r7["gate"]["serial_vs_ar"]["rep1"]["equal"] is True, None)
        ck("G7b_production分歧独立记录（TOKEN_DIFF@7）",
           r7["gate"]["production_vs_ar"]["rep1"]["first_token_diff"] == 7, r7["gate"]["production_vs_ar"])
        ck("G7c_production分歧不污染serial判决", r7["verdict"] == "CASE_PASS", r7["verdict"])
        ck("G8_production只记录且无'无损'表述",
           "不声明同输出无损加速" in r7["production_divergence_note"]
           and not any(("无损" in x and "不声明" not in x) for x in r7["reasons"]), None)
        # 误导字段：在 serial 产物里塞 production_divergence，必须被忽略（判据由 gen 现算）
        r7recs2 = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        r7recs2[4]["production_divergence"] = dict(first_divergence=999)
        r7b = ev("strict", DECL["strict"], r7recs2)
        ck("G7d_误导性字段被忽略（由gen现算）", r7b["verdict"] == "CASE_PASS", r7b["verdict"])
        # （保留既有断言）反向：production==AR 而 serial 分歧 ⇒ 两个字段各自独立，不串读
        se_div = list(AR16); se_div[9] = 555
        r7c = ev("strict", DECL["strict"],
                 _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=se_div))
        ck("G7e_反向：production==AR而serial分歧⇒字段各自独立",
           r7c["gate"]["production_vs_ar"]["rep1"]["equal"] is True
           and r7c["gate"]["serial_vs_ar"]["rep1"]["first_token_diff"] == 9
           and r7c["verdict"] == "CASE_FAIL",
           (r7c["verdict"], r7c["gate"]["serial_vs_ar"]["rep1"], r7c["gate"]["production_vs_ar"]["rep1"]))

        # ---------------- G9：CLI 非法参数 ----------------
        dcli = tmp / "cli"; dcli.mkdir()
        buf = io.StringIO()
        old = sys.argv
        try:
            sys.argv = ["b6_branch_coverage_cpu", "--tag", TAG, "--declared", str(tmp / "nope.json"),
                        "--outdir", str(dcli)]
            with contextlib.redirect_stdout(buf):
                rc_bad = C.main()
        finally:
            sys.argv = old
        ck("G9_声明文件缺失⇒exit3且不写产物", rc_bad == 3 and not list(dcli.glob("*.json")), rc_bad)

        # ---------------- G10：用例表 ----------------
        from . import b6_cases as BC
        legacy = sorted(BC.legacy_cases())
        mine = sorted((c["case_id"], c["text"], c["max_new"]) for c in BC.B6_CASES)
        ck("G10_用例表与g4_b6_correctness.CASES逐字一致", mine == legacy,
           [x for x in mine if x not in legacy][:2])
        ck("G10_类型齐备",
           {"natural_language", "math", "code", "short_input", "long_input",
            "length_not_multiple_of_k", "length_lt_k", "eos"} <= {c["kind"] for c in BC.B6_CASES}, None)
        ck("G10_非k倍数用例声明裁剪与长度分支",
           all(set(c["branches"]) >= {"trim_last_round", "len_not_multiple_of_k"}
               for c in BC.B6_CASES if c["case_id"] in ("len13", "len5", "len2")), None)
        ck("G10_EOS用例声明eos_stop", any("eos_stop" in c["branches"] for c in BC.B6_CASES), None)

        # ---------------- G11：集合级要求 ----------------
        decl11 = dict(cases=dict(strict=dict(kind="natural_language", max_new=16,
                                             branches=["all_accept"], expect_prefill_fused=None)),
                      set_requirements=dict(branches=["all_accept", "first_reject"],
                                            types=["natural_language"]))
        d11 = tmp / "g11"; d11.mkdir()
        _write(d11, _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16)))
        ck("G11_集合级分支未覆盖⇒非PASS(exit1)", rc_of(d11, decl11) == 1, None)
        d11c = tmp / "g11c"; d11c.mkdir()
        _write(d11c, _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                           rounds=[_round(0, 4, 4), _round(1, 4, 0), _round(2, 3, 1)]))
        ck("G11b_集合级要求齐备⇒exit0", rc_of(d11c, decl11) == 0, None)

        # ---------------- G12：完整序列一致性（长度不同也算不一致）----------------
        r12a = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=AR16[:8]))
        ck("G12a_serial是AR严格前缀⇒CASE_FAIL", r12a["verdict"] == "CASE_FAIL"
           and r12a["gate"]["serial_vs_ar"]["rep1"]["kind"] == "PREFIX_LENGTH_MISMATCH",
           (r12a["verdict"], r12a["gate"]["serial_vs_ar"]["rep1"]))
        r12b = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16[:8], prod_gen=list(AR16[:8]), serial_gen=list(AR16)))
        ck("G12b_AR是serial严格前缀⇒CASE_FAIL", r12b["verdict"] == "CASE_FAIL",
           (r12b["verdict"], r12b["gate"]["serial_vs_ar"]["rep1"]))
        r12c = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=AR16))
        ck("G12c_完全相等⇒不因长度误判", r12c["verdict"] == "CASE_PASS", r12c["verdict"])
        # ⚠️ production 侧变短必须**自洽**，否则先触发 stop/length 自洽性失败（测不到"只记录"）：
        #    prod = AR 的前 8 个 token 且以 eos 结束（末位 == eos_id=207）⇒ 严格前缀、长度不同。
        recs12d = _trio("strict", ar_gen=AR16, prod_gen=AR16[:8], serial_gen=list(AR16), eos_id=207)
        for rec in recs12d:
            if rec["mode"] == "prod":
                rec["smoke"]["stop_reason"] = "eos"
                rec["smoke"]["end_state"]["stop_reason"] = "eos"
                rec["smoke"]["end_state"]["eos_seen"] = True
        r12d = ev("strict", DECL["strict"], recs12d)
        ck("G12d_production长度差异如实记录为严格前缀",
           r12d["gate"]["production_vs_ar"]["rep1"]["kind"] == "PREFIX_LENGTH_MISMATCH",
           r12d["gate"]["production_vs_ar"]["rep1"])
        ck("G12e_production长度差异不判控制逻辑失败（serial 仍严格相等⇒PASS）",
           r12d["verdict"] == "CASE_PASS", (r12d["verdict"], r12d["reasons"][:2]))

        # ---------------- G13/G14：硬检查逐个验值 / 缺失 / 非法类型 ----------------
        PATHS = {"self_checks_all_ok": "self_checks_all_ok",
                 "cache_structure_ok": "smoke.cache_structure_ok",
                 "prefill_logits_finite": "smoke.prefill_logits_finite",
                 "round_invariants.all_ok": "smoke.round_invariants.all_ok",
                 "accounting.matches": "accounting.matches",
                 "checks_passed": "smoke.checks_passed",
                 "gen_ids_in_vocab": "smoke.gen_ids_in_vocab",
                 "kill_switch": "verify_mode_after_run",
                 "stage": "stage"}

        def _set(rec: dict, path: str, value) -> None:
            cur = rec
            parts = path.split(".")
            for pp in parts[:-1]:
                cur = cur[pp]
            cur[parts[-1]] = value

        def _del(rec: dict, path: str) -> None:
            cur = rec
            parts = path.split(".")
            for pp in parts[:-1]:
                cur = cur[pp]
            cur.pop(parts[-1], None)

        def _base_trio() -> list:
            return _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))

        for knob, path in PATHS.items():
            # 字符串型期望值（stage / kill_switch）用**非法取值**测"明确失败"；
            # bool 塞进字符串槽属"类型非法"（非 PASS），由下面的 G14b 覆盖。
            wrong = {"stage": "crashed", "kill_switch": "serial"}.get(knob, False)
            recs_f = _base_trio(); _set(recs_f[2], path, wrong)
            r_f = ev("strict", DECL["strict"], recs_f)
            ck(f"G13_{knob}=非法取值⇒CASE_FAIL", r_f["verdict"] == "CASE_FAIL",
               (r_f["verdict"], r_f["reasons"][:1]))
            recs_m = _base_trio(); _del(recs_m[2], path)
            r_m = ev("strict", DECL["strict"], recs_m)
            ck(f"G14_{knob}=缺失⇒CASE_INCOMPLETE", r_m["verdict"] == "CASE_INCOMPLETE",
               (r_m["verdict"], r_m["reasons"][:1]))
            recs_t = _base_trio(); _set(recs_t[2], path, "yes")
            r_t = ev("strict", DECL["strict"], recs_t)
            ck(f"G14b_{knob}=类型非法⇒非PASS", r_t["verdict"] != "CASE_PASS",
               (r_t["verdict"], r_t["reasons"][:1]))

        # ---------------- G15：自洽性 ----------------
        r15a = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16[:8]), serial_gen=list(AR16),
                        stop="max_new_reached"))
        ck("G15a_max_new_reached却短输出⇒CASE_FAIL", r15a["verdict"] == "CASE_FAIL", r15a["verdict"])
        recs15b = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs15b[2]["smoke"]["gen_len"] = 99
        ck("G15b_gen_len与实际长度不符⇒CASE_FAIL",
           ev("strict", DECL["strict"], recs15b)["verdict"] == "CASE_FAIL", None)
        r15c = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        stop="other_short"))
        ck("G15c_非法stop_reason⇒CASE_FAIL", r15c["verdict"] == "CASE_FAIL", r15c["verdict"])
        r15d = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16[:13], prod_gen=list(AR16[:13]), serial_gen=list(AR16[:13]),
                        stop="eos", eos_id=999, eos_seen=True))
        ck("G15d_stop=eos但末token≠eos_id⇒CASE_FAIL", r15d["verdict"] == "CASE_FAIL", r15d["verdict"])

        # ---------------- G16：AR 合理融合 prefill 不被误拒 ----------------
        r16 = ev("strict", DECL["strict"],
                 _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                       arm="cudnn_prefill", fused_by_phase={"prefill": 4}))
        ck("G16a_AR与Spec都融合⇒不被误拒（策略一致）",
           r16["per_mode"]["ar"]["rep1"]["problems"] == [] and r16["verdict"] == "CASE_PASS",
           (r16["per_mode"]["ar"]["rep1"]["problems"], r16["verdict"]))
        r16b = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        arm="eager", fused_by_phase={}))
        ck("G16b_arm=eager且融合0⇒合法（不判失败）", r16b["verdict"] == "CASE_PASS", r16b["verdict"])
        r16c = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        arm="eager", fused_by_phase={"prefill": 2}))
        ck("G16c_arm=eager却出现融合⇒CASE_FAIL", r16c["verdict"] == "CASE_FAIL", r16c["verdict"])
        ck("G16d_证据边界被显式记录（target/draft 不可分）",
           "不区分 target/draft" in r16["per_mode"]["prod"]["rep1"]["evidence_boundary"], None)

        # ---------------- G17：短输入分派（expect=None）----------------
        r17a = ev("eos_case", DECL["eos_case"],
                  _trio("eos_case", ar_gen=AR16[:13], prod_gen=list(AR16[:13]),
                        serial_gen=list(AR16[:13]), stop="eos", eos_id=EOS, ctx=5,
                        rounds=[_round(0, 4, 4), _round(1, 3, 1), _round(2, 2, 2)],
                        fused_by_phase={}))
        ck("G17a_expect=None且无融合无理由⇒INCOMPLETE（不得当通过）",
           r17a["verdict"] == "CASE_INCOMPLETE", (r17a["verdict"], r17a["reasons"][:2]))
        r17b = ev("eos_case", DECL["eos_case"],
                  _trio("eos_case", ar_gen=AR16[:13], prod_gen=list(AR16[:13]),
                        serial_gen=list(AR16[:13]), stop="eos", eos_id=EOS, ctx=5,
                        rounds=[_round(0, 4, 4), _round(1, 3, 1), _round(2, 2, 2)],
                        fused_by_phase={}, fallbacks=[dict(phase="prefill", why="s<min")],
                        reasons={"s_below_min": 1}))
        ck("G17b_expect=None但**有记录的回退理由**⇒PASS", r17b["verdict"] == "CASE_PASS",
           (r17b["verdict"], r17b["reasons"][:2]))

        # ---------------- G18：跨模式 / 跨 rep 匹配 ----------------
        recs18 = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs18[2]["smoke"]["input"]["ids_hash"] = "ffffffffffffffff"
        ck("G18a_跨模式input hash不一致⇒CASE_FAIL",
           ev("strict", DECL["strict"], recs18)["verdict"] == "CASE_FAIL", None)
        recs18b = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs18b[4]["smoke"]["input"]["ctx"] = 99
        recs18b[4]["smoke"]["input"]["input_tokens"] = 99
        ck("G18b_跨rep实际输入长度不一致⇒CASE_FAIL",
           ev("strict", DECL["strict"], recs18b)["verdict"] == "CASE_FAIL", None)
        recs18c = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs18c[2]["smoke_config"]["chunk"] = 1024
        ck("G18c_跨模式chunk不一致⇒CASE_FAIL（且配置违规）",
           ev("strict", DECL["strict"], recs18c)["verdict"] == "CASE_FAIL", None)
        recs18d = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs18d[0]["smoke"]["kv_format"]["draft"] = "fp16"      # AR 误加载 draft
        ck("G18d_AR误加载draft⇒CASE_FAIL（基线不纯）",
           ev("strict", DECL["strict"], recs18d)["verdict"] == "CASE_FAIL", None)
        recs18e = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        recs18e[1]["smoke"]["input"]["ids_hash"] = "1111111111111111"   # ar rep2 输入不同
        ck("G18e_同模式跨rep输入不一致⇒CASE_FAIL",
           ev("strict", DECL["strict"], recs18e)["verdict"] == "CASE_FAIL", None)

        # ---------------- G19：产物选择（重复/非法 rep/身份不符）----------------
        recs19 = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        dup = _mk("strict", "prod", 1, gen=list(AR16), rounds=[_round(0, 4, 4)])
        dup["run_id"] = "20261006T000001Z"
        recs19.append(dup)
        r19 = ev("strict", DECL["strict"], recs19)
        ck("G19a_重复产物⇒显式报告且非PASS",
           r19["verdict"] == "CASE_FAIL"
           and any("重复产物" in x for x in r19["duplicate_or_identity_problems"]),
           (r19["verdict"], r19["duplicate_or_identity_problems"][:2]))
        recs19b = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        bad_rep = _mk("strict", "prod", 1, gen=list(AR16), rounds=[_round(0, 4, 4)])
        bad_rep["rep"] = "bad"
        bad_rep["run_id"] = "20261006T000002Z"
        recs19b.append(bad_rep)
        r19b = ev("strict", DECL["strict"], recs19b)
        ck("G19b_非法rep⇒显式报告且非PASS",
           r19b["verdict"] == "CASE_FAIL"
           and any("非法 rep" in x for x in r19b["duplicate_or_identity_problems"]),
           (r19b["verdict"], r19b["duplicate_or_identity_problems"][:2]))
        recs19c = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16))
        foreign = _mk("strict", "prod", 2, gen=list(AR16), rounds=[_round(0, 4, 4)],
                      case_in_record="other-case")
        foreign["run_id"] = "20261006T000003Z"
        recs19c.append(foreign)
        # ⚠️ 文件名必须**匹配请求的 case**（否则 `find_runs` 的 glob 根本看不到它 ⇒ 测不到身份核对）；
        #    这里显式指定文件名，模拟"文件名叫 strict、记录内 case 是 other-case"的错配。
        names19c = {len(recs19c) - 1: f"{TAG}_strict_prod_rep2_20261006T000003Z.json"}
        r19c = ev("strict", DECL["strict"], recs19c, names=names19c)
        ck("G19c_身份字段不符⇒显式报告且非PASS",
           r19c["verdict"] == "CASE_FAIL"
           and any("case=" in x for x in r19c["duplicate_or_identity_problems"]),
           (r19c["verdict"], r19c["duplicate_or_identity_problems"][:2]))
        # ---------------- G21：诊断精度验收（rev.3 修漏洞）----------------
        # 旧行为（漏洞）：before/after/stable 全为 "invalid" ⇒ 三条检查路径都不覆盖 ⇒ CASE_PASS。
        # 现在：逐条验值 + 自行核对前后一致，**不信任** stable 字段，且不做 bool() 强制转换。
        def _dp_case(mut) -> dict:
            recs = _base_trio()
            for r in recs:
                mut(r)
            return ev("strict", DECL["strict"], recs)

        def _set_dp(rec, **kw):
            b, a, s = (rec["diag_precision_before_run"], rec["diag_precision_after_run"],
                       rec["diag_precision_stable"])
            for k, v in kw.items():
                if k in ("before", "after", "stable"):
                    pass
            return None

        # (1) 三个字段分别为字符串 / 列表 / None（用户报告的最小复现）
        r21 = _dp_case(lambda r: (r.__setitem__("diag_precision_before_run", "invalid"),
                                 r.__setitem__("diag_precision_after_run", "invalid"),
                                 r.__setitem__("diag_precision_stable", "invalid")))
        ck("G21a_三字段全为字符串⇒非PASS（用户报告的最小复现）", r21["verdict"] in
           ("CASE_INCOMPLETE", "CASE_FAIL"), r21["verdict"])
        ck("G21a_原因为诊断精度类型非法",
           any("诊断精度类型非法" in x for x in r21["reasons"]), r21["reasons"][:2])
        for label, val in (("字符串", "x"), ("列表", [1]), ("None", None)):
            r = _dp_case(lambda rec, v=val: rec.__setitem__("diag_precision_before_run", v))
            ck(f"G21b_before={label}⇒非PASS", r["verdict"] != "CASE_PASS",
               (r["verdict"], r["reasons"][:1]))
            r = _dp_case(lambda rec, v=val: rec.__setitem__("diag_precision_after_run", v))
            ck(f"G21c_after={label}⇒非PASS", r["verdict"] != "CASE_PASS",
               (r["verdict"], r["reasons"][:1]))
        # (2) 缺任一开关键
        r = _dp_case(lambda rec: rec.__setitem__("diag_precision_before_run", dict(prob_fp32=False)))
        ck("G21d_缺before.qk_fp32⇒INCOMPLETE（证据不足，不升级为FAIL）",
           r["verdict"] == "CASE_INCOMPLETE"
           and any("诊断精度证据不足" in x for x in r["reasons"]), (r["verdict"], r["reasons"][:1]))
        r = _dp_case(lambda rec: rec.__setitem__("diag_precision_after_run", dict(qk_fp32=False)))
        ck("G21e_缺after.prob_fp32⇒INCOMPLETE", r["verdict"] == "CASE_INCOMPLETE",
           (r["verdict"], r["reasons"][:1]))
        # (3) 开关为 0 / 1 / "False" / None / True
        for label, val, want in (("0", 0, "CASE_FAIL"), ("1", 1, "CASE_FAIL"),
                                 ("'False'", "False", "CASE_FAIL"), ("None", None, "CASE_INCOMPLETE"),
                                 ("True", True, "CASE_FAIL")):
            r = _dp_case(lambda rec, v=val: rec.__setitem__(
                "diag_precision_before_run", dict(qk_fp32=v, prob_fp32=False)))
            ck(f"G21f_before.qk_fp32={label}⇒{want}", r["verdict"] == want,
               (r["verdict"], r["reasons"][:1]))
            r = _dp_case(lambda rec, v=val: rec.__setitem__(
                "diag_precision_after_run", dict(qk_fp32=False, prob_fp32=v)))
            ck(f"G21g_after.prob_fp32={label}⇒{want}", r["verdict"] == want, r["verdict"])
        # (4) stable 为字符串 / 整数 / None / False
        for label, val, want in (("字符串", "true", "CASE_INCOMPLETE"), ("整数", 1, "CASE_INCOMPLETE"),
                                 ("None", None, "CASE_INCOMPLETE"), ("False", False, "CASE_FAIL")):
            r = _dp_case(lambda rec, v=val: rec.__setitem__("diag_precision_stable", v))
            ck(f"G21h_stable={label}⇒{want}", r["verdict"] == want,
               (r["verdict"], r["reasons"][:1]))
        # (5) 前后不同但 stable=True ⇒ 必须由**自行核对**抓出（不信任 stable）
        r = _dp_case(lambda rec: (rec.__setitem__("diag_precision_before_run",
                                                  dict(qk_fp32=False, prob_fp32=False)),
                                  rec.__setitem__("diag_precision_stable", True)))
        ck("G21i_前后不同但stable=True⇒未出现（前后相同）", r["verdict"] == "CASE_PASS", r["verdict"])
        r = _dp_case(lambda rec: (rec.__setitem__("diag_precision_after_run",
                                                  dict(qk_fp32=True, prob_fp32=False)),
                                  rec.__setitem__("diag_precision_stable", True)))
        ck("G21j_前后不同且stable=True⇒CASE_FAIL（自行核对抓出）", r["verdict"] == "CASE_FAIL",
           (r["verdict"], r["reasons"][:1]))
        ck("G21j_理由含自行核对前后不同",
           any("before_after_一致（自行核对）" in x for x in r["reasons"]), r["reasons"][:2])
        # (6) 前后合法一致且 stable=True ⇒ PASS（对照组）
        r = _dp_case(lambda rec: None)
        ck("G21k_前后合法一致且stable=True⇒CASE_PASS", r["verdict"] == "CASE_PASS", r["verdict"])
        # (7) 记录里可以有**其它合法字段**（不限制只能两个键），且它们不参与比较
        r = _dp_case(lambda rec: rec.__setitem__(
            "diag_precision_before_run",
            dict(qk_fp32=False, prob_fp32=False, note="extra", nested=dict(a=1), other=7)))
        ck("G21l_允许其它合法字段⇒仍PASS（不擅自限制键集）", r["verdict"] == "CASE_PASS",
           (r["verdict"], r["reasons"][:1]))
        ck("G21m_比较范围已明确声明（仅两个开关）",
           any("比较范围仅" in v["detail"] or "仅比较" in v["detail"]
               for k, v in (r.get("diag_precision") or {}).items() if "一致" in k), None)
        # (8) 旧产物缺新增字段：只记证据不足，不改历史判决
        runs_hist, probs_hist = C.find_runs("div8209_p2", "F09")
        if runs_hist and any(runs_hist.values()):
            rh = C.evaluate_case("F09", dict(branches=[], max_new=64, expect_prefill_fused=None),
                                 runs_hist, expect_backend="CUDNN",
                                 expect_config=dict(max_new=64), run_problems=probs_hist)
            ck("G21n_旧产物缺新增字段⇒只记证据不足（不改历史判决）",
               rh["verdict"] == "CASE_INCOMPLETE"
               and any("诊断精度证据不足" in x for x in rh["reasons"]), rh["verdict"])
            ck("G21o_旧产物序列一致性未被本修复影响（serial==AR）",
               all(v["equal"] for v in rh["gate"]["serial_vs_ar"].values()),
               rh["gate"]["serial_vs_ar"])
        else:
            ck("G21n_旧产物缺新增字段⇒只记证据不足（不改历史判决）", False, "无历史产物可测")
            ck("G21o_旧产物序列一致性未被本修复影响（serial==AR）", False, "无历史产物可测")

        # ---------------- G22：`smoke.checks` 覆盖规则的**分类修正**回归（rev.5，2026-10-05）----------------
        # 修正的事实依据（本地实测、可复核）：`smoke_checks.evaluate` **只在** `required=False` 且未执行时
        # 才写 `uncovered`；必需但缺失写 `failures` 且 `passed=False`。
        # ⇒ `uncovered` 非空**本身不是**失败（此前误判成失败：natural-2 prod 实测 triggered）。
        # 新规则更细：failures 必须空 + uncovered 每条必须 required=False/executed=False + 必需条目须已执行且通过。
        def _c22(mut) -> dict:
            """构造一份**完全合法**的产物再按 `mut(rec)` 变形，返回 CHK 判决（三模式 × 两 rep）。"""
            recs = _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                         rounds=[_round(0, 4, 4), _round(1, 3, 2)])
            for r in recs:
                mut(r)
            return ev("strict", DECL["strict"], recs)

        def _set_unc(rec, names):
            ch = rec["smoke"]["checks"]
            ch["uncovered"] = list(names)
            ch["checks"] = [e for e in ch["checks"] if e["name"] not in names]
            ch["checks"] += [dict(name=nm, required=False, executed=False, ok=None,
                                  detail="该分支本次未执行（非必需）") for nm in names]

        # (a) uncovered 非空但全为"非必需且未执行" ⇒ **不再**判失败；仍如实记录覆盖缺口
        r22a = _c22(lambda rec: _set_unc(rec, ["finiteness:repair/draft", "finiteness:repair/target"]))
        ck("G22a_uncovered 非空但均为非必需未执行⇒**不判失败**（rev.5 修分类漏洞）",
           r22a["verdict"] == "CASE_PASS", (r22a["verdict"], r22a["reasons"][:2]))
        ck("G22a_覆盖缺口被如实记为只读覆盖记录",
           any("未执行" in n for ns in (r22a.get("coverage_notes") or {}).values()
               for n in (ns.get("rep1") or [])), r22a.get("coverage_notes"))
        # (a2) 缺 `checks.checks[]` 逐条明细 ⇒ **证据不足**（不是失败）——防止"无法核对却当通过"
        def _drop_entries(rec):
            rec["smoke"]["checks"].pop("checks", None)
        r22a2 = _c22(_drop_entries)
        ck("G22a2_缺逐条明细⇒INCOMPLETE（不得当通过）", r22a2["verdict"] == "CASE_INCOMPLETE",
           (r22a2["verdict"], r22a2["reasons"][:1]))
        # (b) uncovered 里出现**必需**条目 ⇒ FAIL（防止把必需检查塞进 uncovered 混过去）
        def _unc_required(rec):
            _set_unc(rec, ["finiteness:repair/draft"])
            for e in rec["smoke"]["checks"]["checks"]:
                if e["name"] == "finiteness:repair/draft":
                    e["required"] = True
        r22b = _c22(_unc_required)
        ck("G22b_uncovered 含必需条目⇒CASE_FAIL（与 harness 契约矛盾）",
           r22b["verdict"] == "CASE_FAIL", (r22b["verdict"], r22b["reasons"][:1]))
        # (c) uncovered 里出现**已执行**条目 ⇒ FAIL
        def _unc_executed(rec):
            _set_unc(rec, ["finiteness:repair/draft"])
            for e in rec["smoke"]["checks"]["checks"]:
                if e["name"] == "finiteness:repair/draft":
                    e["executed"] = True
        r22c = _c22(_unc_executed)
        ck("G22c_uncovered 含已执行条目⇒CASE_FAIL", r22c["verdict"] == "CASE_FAIL",
           (r22c["verdict"], r22c["reasons"][:1]))
        # (d) uncovered 里有名字在逐条明细里**不存在** ⇒ FAIL（无逐条明细可核对）
        r22d = _c22(lambda rec: rec["smoke"]["checks"]["uncovered"].append("finiteness:不存在/x"))
        ck("G22d_uncovered 出现无明细的名字⇒CASE_FAIL", r22d["verdict"] == "CASE_FAIL",
           (r22d["verdict"], r22d["reasons"][:1]))
        # (e) harness failures 非空 ⇒ FAIL
        r22e = _c22(lambda rec: rec["smoke"]["checks"].__setitem__(
            "failures", [dict(name="finiteness:verify/target", detail="预期检查缺失")]))
        ck("G22e_harness failures 非空⇒CASE_FAIL", r22e["verdict"] == "CASE_FAIL",
           (r22e["verdict"], r22e["reasons"][:1]))
        # (f) 必需条目未执行 / 未通过 ⇒ FAIL（即便 passed 字段被伪造成 True）
        def _req_notexec(rec):
            for e in rec["smoke"]["checks"]["checks"]:
                if e["name"] == "dispatch_only_prefill":
                    e["executed"] = False
        r22f = _c22(_req_notexec)
        ck("G22f_必需条目未执行（即使 passed=True）⇒CASE_FAIL", r22f["verdict"] == "CASE_FAIL",
           (r22f["verdict"], r22f["reasons"][:1]))

        def _req_notok(rec):
            for e in rec["smoke"]["checks"]["checks"]:
                if e["name"] == "finiteness:prefill/target":
                    e["ok"] = False
        r22g = _c22(_req_notok)
        ck("G22g_必需条目 ok=False（即使 passed=True）⇒CASE_FAIL", r22g["verdict"] == "CASE_FAIL",
           (r22g["verdict"], r22g["reasons"][:1]))
        # (h) 缺 `checks.uncovered` 字段 ⇒ INCOMPLETE
        r22h = _c22(lambda rec: rec["smoke"]["checks"].pop("uncovered", None))
        ck("G22h_缺 checks.uncovered⇒INCOMPLETE（证据不足）", r22h["verdict"] == "CASE_INCOMPLETE",
           (r22h["verdict"], r22h["reasons"][:1]))
        # (i) 融合路径覆盖记录：真的融合 ⇒ 记为"已实际发生"；短块合法回退 ⇒ 记为"未覆盖"
        r22i = _c22(lambda rec: None)
        ck("G22i_真融合⇒覆盖记录写明'已实际发生'",
           any("已实际发生" in n for ns in (r22i.get("coverage_notes") or {}).values()
               for n in (ns.get("rep1") or [])), r22i.get("coverage_notes"))
        r22j = ev("strict", DECL["strict"],
                  _trio("strict", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        rounds=[_round(0, 4, 4), _round(1, 3, 2)], fused_by_phase={},
                        fallbacks=[dict(phase="prefill", s=1,
                                        reason="s=1 < PREFILL_FUSED_MIN_S=2（cuDNN 不支持 seq_len=1）",
                                        kind="short_block_seq_len_lt_min")],
                        reasons={"short_block_seq_len_lt_min": 1}))
        ck("G22j_短块合法回退⇒覆盖记录写明'融合路径未被覆盖'",
           any("融合路径未被覆盖" in n for ns in (r22j.get("coverage_notes") or {}).values()
               for n in (ns.get("rep1") or [])), r22j.get("coverage_notes"))
        ck("G22j_短块回退在有记录理由时不算失败", r22j["verdict"] == "CASE_PASS",
           (r22j["verdict"], r22j["reasons"][:1]))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== B6 工具链自测（纯 CPU；合成产物在临时目录，未触碰真实产物）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}" + (f"  {str(extra)[:200]}" if not ok else ""))
    print(f"\nB6 工具链自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
