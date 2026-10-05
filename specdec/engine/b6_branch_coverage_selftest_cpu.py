"""b6_branch_coverage_selftest_cpu.py — C-1/C-2/C-3 新增能力的 **CPU 自测**（纯 CPU；不加载模型、不使用 GPU）。

## 覆盖（对应用户要求的最小回归集）
| 组 | 用例 | 期望 |
| --- | --- | --- |
| **G1 固定输入与长度可复现** | `resolve_input_plan()`：都不设 ⇒ `legacy`；只设 `SMOKE_FAMILY` ⇒ `manifest`；只设 `SMOKE_PROMPT` ⇒ `literal_prompt`；两者都设 ⇒ **显式报错**；同一文本 `prompt_sha1` 稳定、不同文本不同 | 全部成立（**未设参数时既有行为不变**） |
| **G2 短输入不写成 8192** | 合成产物 `input.ctx=5`（字面短 prompt） | 不被判配置违规；`input_tokens` 如实记录 |
| **G3 EOS 与长度裁剪** | 合成产物 `stop_reason="eos"`、`gen_len=13`（非 k 倍数）、末轮 `Kp<k` | `eos_stop`/`len_not_multiple_of_k`/`trim_last_round` 均 **COVERED** |
| **G4 未触发分支 = 未覆盖（不得 PASS）** | 声明 `eos_stop` 但 `stop_reason="max_new_reached"` | 该分支 **UNCOVERED** ⇒ `CASE_INCOMPLETE`（**不是** PASS） |
| **G5 缺字段/错误配置不得 PASS** | ① 删掉 `diag_precision_*`；② `k=3`；③ `diag.qk_fp32=True`；④ `fused_by_phase` 在 `verify` 阶段 >0 | ① `CASE_INCOMPLETE`；②③④ `CASE_FAIL` |
| **G6 serial 门** | serial 与 AR 第 5 位不同 | `serial_vs_ar` 首分歧=5 ⇒ `CASE_FAIL`（token mismatch 即停） |
| **G7 分歧字段不串读** | ① serial==AR 但 production 分歧；② production==AR 但 serial 分歧；③ 在 serial 产物里塞入**误导性的** `production_divergence` 字段 | 两种情形的 `serial_vs_ar`/`production_vs_ar` **各自独立**；误导字段被**忽略**（判据由 `gen` 现算） |
| **G8 production 只记录** | 存在 production 分歧 | 不产生"控制逻辑失败"字样，且带"只作发散轨迹记录"说明 |
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from . import b6_branch_coverage_cpu as C
from . import family_input as FI

TAG = "div8209_b6"
RUNID = "20261006T000000Z"


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
        ctx: int = 12, k: int = 4, max_new: int = 16, qk32: bool = False, prob32: bool = False,
        fused_by_phase=None, drop=(), case_in_record: str | None = None, extra: dict | None = None,
        source_kind: str = "literal_prompt") -> dict:
    rec = dict(tool="serial_oracle_gpu", tag=TAG, run_id=RUNID, case=case_in_record or case,
               mode=mode, rep=rep, stage="complete", model_run_happened=True, env_qualified=True,
               blocked_by=None, diagnostic_only=True,
               smoke_config=dict(ctx=ctx, chunk=2048, k=k, max_new=max_new, kv="base_bf16",
                                 arm="cudnn_prefill", mode=("ar" if mode == "ar" else "spec"),
                                 family=case, target_rope="cpu_fp32", bf16_red_red="1",
                                 struct_check=True),
               diag_precision_before_run=dict(qk_fp32=qk32, prob_fp32=prob32),
               diag_precision_after_run=dict(qk_fp32=qk32, prob_fp32=prob32),
               diag_precision_stable=True,
               accounting=dict(mode=mode, matches=True, target_generation_calls_measured=10,
                               expected=10),
               self_checks_all_ok=True, verify_mode_after_run="block",
               verify_mode_at_run_end=("serial" if mode == "serial" else "block"),
               smoke=dict(gen=list(gen), gen_len=len(gen), stop_reason=stop,
                          gen_ids_in_vocab=True, prefill_logits_finite=True,
                          checks_passed=True, n_rounds=len(rounds), rounds=rounds,
                          specdec_stats=dict(rounds=len(rounds), proposed=8, accepted=4,
                                             target_block_calls=len(rounds), draft_calls=8,
                                             rollbacks=1, mismatch_rounds=1, all_accept_rounds=1),
                          round_invariants=dict(all_ok=True, mode=mode, per_round=[]),
                          cache_structure_ok=True, cache_structure_n_checks=40,
                          attention_stats=dict(strategy="eager", requested_backend="CUDNN_ATTENTION",
                                               calls=dict(eager=100, fused=(fused_by_phase or {}).get("prefill", 0)),
                                               by_phase=dict(prefill=2),
                                               fused_by_phase=dict(fused_by_phase or {}),
                                               eager_by_phase=dict(prefill=2, verify=10)),
                          input=dict(ctx=ctx, ids_hash="deadbeefcafe0001", source_kind=source_kind,
                                     prompt_id="p1", prompt_sha1="aabbccdd00112233",
                                     input_tokens=ctx,
                                     dtypes=dict(model="torch.bfloat16",
                                                 target_kv_payload="base_bf16",
                                                 draft_kv_payload="base_bf16"))))
    for path in drop:
        cur = rec
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else {}
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    if extra:
        rec.update(extra)
    return rec


def _write(outdir: Path, recs: list) -> None:
    for r in recs:
        (outdir / f"{r['tag']}_{r['case']}_{r['mode']}_rep{r['rep']}_{r.get('run_id', RUNID)}.json"
         ).write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")


def _trio(case: str, *, ar_gen: list, prod_gen: list, serial_gen: list, stop: str = "max_new_reached",
          rounds=None, **kw) -> list:
    rounds = rounds if rounds is not None else [_round(0, 4, 4)]
    out = []
    for mode, g in (("ar", ar_gen), ("prod", prod_gen), ("serial", serial_gen)):
        for rep in (1, 2):
            out.append(_mk(case, mode, rep, gen=g, rounds=([] if mode == "ar" else rounds),
                           stop=stop, **kw))
    return out


DECL = dict(natural_ok=dict(branches=["all_accept", "trim_last_round"], max_new=16,
                            expect_prefill_fused=True),
            eos_case=dict(branches=["eos_stop", "len_not_multiple_of_k"], max_new=16,
                          expect_prefill_fused=None),
            strict=dict(branches=["all_accept"], max_new=16, expect_prefill_fused=None))


def main() -> int:
    checks: list = []

    def ck(name: str, ok: bool, extra=None) -> None:
        checks.append((name, bool(ok), extra))

    # ---------------- G1：输入计划（含"未设参数行为不变"）----------------
    keys = ("SMOKE_PROMPT", "SMOKE_PROMPT_ID", "SMOKE_FAMILY")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        p_legacy = FI.resolve_input_plan()
        ck("G1_都不设⇒legacy（默认行为不变）", p_legacy["mode"] == "legacy", p_legacy)
        os.environ["SMOKE_FAMILY"] = "F09"
        ck("G1_设FAMILY⇒manifest", FI.resolve_input_plan()["mode"] == "manifest", None)
        os.environ.pop("SMOKE_FAMILY")
        os.environ["SMOKE_PROMPT"] = "Hello"
        p1 = FI.resolve_input_plan()
        ck("G1_设PROMPT⇒literal_prompt", p1["mode"] == "literal_prompt", p1)
        os.environ["SMOKE_PROMPT_ID"] = "natural-1"
        p2 = FI.resolve_input_plan()
        ck("G1_prompt_id被记录", p2.get("prompt_id") == "natural-1", p2.get("prompt_id"))
        ck("G1_同一文本prompt_sha1稳定", p1["prompt_sha1"] == p2["prompt_sha1"], p1["prompt_sha1"])
        os.environ["SMOKE_PROMPT"] = "Hello!"
        ck("G1_不同文本sha1不同", FI.resolve_input_plan()["prompt_sha1"] != p1["prompt_sha1"], None)
        os.environ["SMOKE_FAMILY"] = "F09"
        err = None
        try:
            FI.resolve_input_plan()
        except Exception as e:                                        # noqa: BLE001
            err = type(e).__name__
        ck("G1_PROMPT与FAMILY同时设⇒显式报错", err == "FamilyInputError", err)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    tmp = Path(tempfile.mkdtemp(prefix="b6cov_"))
    try:
        # 公共片段：AR 与 production 相同的 16 token 序列
        ar = list(range(200, 216))
        pr_same = list(ar)
        se_same = list(ar)

        # ---------------- G2 + G3：短输入长度如实 + 分支覆盖（含 EOS/裁剪）----------------
        d1 = tmp / "g2g3"; d1.mkdir()
        eos_gen = ar[:13]                     # 13 非 4 的倍数；stop=eos
        rounds_eos = [_round(0, 4, 4),
                      _round(1, 4, 4),
                      _round(2, 3, 3),        # 末轮 Kp<k ⇒ trim_last_round
                      _round(3, 1, 1)]
        _write(d1, _trio("natural_ok", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same,
                         rounds=[_round(0, 4, 4), _round(1, 4, 4), _round(2, 3, 2)]))
        _write(d1, _trio("eos_case", ar_gen=eos_gen, prod_gen=list(eos_gen), serial_gen=list(eos_gen),
                         stop="eos", rounds=rounds_eos, ctx=5))
        r_ok = C.evaluate_case("natural_ok", DECL["natural_ok"], C.find_runs(TAG, "natural_ok", d1),
                               expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G2_短输入ctx=12不被判配置违规",
           not any("max_new" in x or "ctx" in x for x in r_ok["reasons"]), r_ok["reasons"])
        ck("G3_全接受与末轮裁剪COVERED",
           r_ok["branches"]["all_accept"]["verdict"] == "COVERED"
           and r_ok["branches"]["trim_last_round"]["verdict"] == "COVERED", r_ok["branches"])
        r_eos = C.evaluate_case("eos_case", DECL["eos_case"], C.find_runs(TAG, "eos_case", d1),
                                expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G3_EOS停止被记录为COVERED", r_eos["branches"]["eos_stop"]["verdict"] == "COVERED",
           r_eos["branches"])
        ck("G3_非k整数倍长度被记录为COVERED",
           r_eos["branches"]["len_not_multiple_of_k"]["verdict"] == "COVERED", r_eos["branches"])
        ck("G3_短输入prompt_sha1/input_tokens如实记录",
           C._dig(C.find_runs(TAG, "eos_case", d1)[("prod", 1)], "smoke.input.input_tokens") == 5, None)

        # ---------------- G4：未触发分支 ⇒ 未覆盖（不得 PASS）----------------
        d4 = tmp / "g4"; d4.mkdir()
        _write(d4, _trio("eos_case", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same,
                         stop="max_new_reached", rounds=[_round(0, 4, 4)]))
        r4 = C.evaluate_case("eos_case", DECL["eos_case"], C.find_runs(TAG, "eos_case", d4),
                             expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G4_未触发EOS记UNCOVERED", r4["branches"]["eos_stop"]["verdict"] == "UNCOVERED", r4["branches"])
        ck("G4_未覆盖⇒CASE_INCOMPLETE（不是PASS）", r4["verdict"] == "CASE_INCOMPLETE", r4["verdict"])
        ck("G4_理由点名未覆盖分支", any("未覆盖分支" in x for x in r4["reasons"]), r4["reasons"])

        # ---------------- G5：缺字段 / 错误配置不得 PASS ----------------
        d5 = tmp / "g5"; d5.mkdir()
        recs5 = _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same)
        for r in recs5:
            if (r["mode"], r["rep"]) == ("prod", 1):
                r.pop("diag_precision_after_run", None)
        _write(d5, recs5)
        r5 = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d5),
                             expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G5a_缺diag_precision⇒INCOMPLETE且非PASS",
           r5["verdict"] == "CASE_INCOMPLETE" and any("缺字段" in x for x in r5["reasons"]), r5["verdict"])

        d5b = tmp / "g5b"; d5b.mkdir()
        _write(d5b, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same, k=3))
        r5b = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d5b),
                              expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G5b_k=3⇒CASE_FAIL", r5b["verdict"] == "CASE_FAIL"
           and any("k=" in x for x in r5b["reasons"]), r5b["reasons"])

        d5c = tmp / "g5c"; d5c.mkdir()
        _write(d5c, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same, qk32=True))
        r5c = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d5c),
                              expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G5c_qk_fp32=True⇒CASE_FAIL", r5c["verdict"] == "CASE_FAIL", r5c["reasons"])

        d5d = tmp / "g5d"; d5d.mkdir()
        _write(d5d, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same,
                          fused_by_phase={"verify": 3}))
        r5d = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d5d),
                              expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G5d_非prefill融合⇒CASE_FAIL（禁止）", r5d["verdict"] == "CASE_FAIL"
           and any("禁止" in x for x in r5d["reasons"]), r5d["reasons"])

        # ---------------- G6：serial 门（token mismatch ⇒ FAIL）----------------
        d6 = tmp / "g6"; d6.mkdir()
        bad_serial = list(ar); bad_serial[5] = 999
        _write(d6, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=bad_serial))
        r6 = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d6),
                             expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G6_serial与AR不一致⇒CASE_FAIL",
           r6["verdict"] == "CASE_FAIL"
           and r6["gate"]["serial_vs_ar"]["rep1"]["first_divergence"] == 5, r6["gate"]["serial_vs_ar"])

        # ---------------- G7：分歧字段不串读 ----------------
        d7 = tmp / "g7"; d7.mkdir()
        pr_div = list(ar); pr_div[7] = 777
        _write(d7, _trio("strict", ar_gen=ar, prod_gen=pr_div, serial_gen=se_same))
        # 误导字段：在 serial 产物里塞入假的 production_divergence
        r7rec = C.find_runs(TAG, "strict", d7)[("serial", 1)]
        r7rec["production_divergence"] = dict(first_divergence=999)
        (d7 / f"{TAG}_strict_serial_rep1_{RUNID}.json").write_text(
            json.dumps(r7rec, ensure_ascii=False, default=str), encoding="utf-8")
        r7 = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d7),
                             expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G7a_serial==AR⇒serial首分歧None",
           r7["gate"]["serial_vs_ar"]["rep1"]["first_divergence"] is None, r7["gate"]["serial_vs_ar"])
        ck("G7b_production分歧被单独记录=7",
           r7["gate"]["production_vs_ar"]["rep1"]["first_divergence"] == 7, r7["gate"]["production_vs_ar"])
        ck("G7c_误导性production_divergence字段被忽略（由gen现算）",
           r7["gate"]["production_vs_ar"]["rep1"]["first_divergence"] != 999, None)

        d7b = tmp / "g7b"; d7b.mkdir()
        se_div = list(ar); se_div[9] = 555
        _write(d7b, _trio("strict", ar_gen=ar, prod_gen=list(ar), serial_gen=se_div))
        r7b = C.evaluate_case("strict", DECL["strict"], C.find_runs(TAG, "strict", d7b),
                              expect_backend="CUDNN", expect_config=dict(max_new=16))
        ck("G7d_反向：production==AR而serial分歧⇒两字段各自独立",
           r7b["gate"]["production_vs_ar"]["rep1"]["first_divergence"] is None
           and r7b["gate"]["serial_vs_ar"]["rep1"]["first_divergence"] == 9,
           (r7b["gate"]["production_vs_ar"], r7b["gate"]["serial_vs_ar"]))

        # ---------------- G8：production 只记录 ----------------
        ck("G8_production分歧不被判控制逻辑失败",
           r7["verdict"] in ("CASE_PASS", "CASE_INCOMPLETE")
           and "只作发散轨迹记录" in r7["production_divergence_note"], r7["verdict"])
        ck("G8_无'同输出无损'类表述",
           not any(("无损" in x and "不声明" not in x) for x in r7["reasons"])
           and ("不声明同输出无损加速" in r7["production_divergence_note"]), r7["reasons"][:2])

        # ---------------- G9：CLI 参数非法 ⇒ exit 3 且不写产物 ----------------
        import sys
        import contextlib
        import io
        dcli = tmp / "cli"; dcli.mkdir()
        buf = io.StringIO()
        old = sys.argv
        try:
            sys.argv = ["b6_branch_coverage_cpu", "--tag", TAG, "--declared", str(tmp / "nope.json"),
                        "--outdir", str(dcli)]
            with contextlib.redirect_stdout(buf):
                rc = C.main()
        finally:
            sys.argv = old
        ck("G9_声明文件缺失⇒exit3且不写产物",
           rc == 3 and not list(dcli.glob("*.json")), (rc, [p.name for p in dcli.glob("*.json")]))
        # ---------------- G10：用例表与旧 B6 工具**逐字一致**（单一来源 + 不得擅自改输入）----------------
        from . import b6_cases as BC
        legacy = BC.legacy_cases()
        mine = [(c["case_id"], c["text"], c["max_new"]) for c in BC.B6_CASES]
        ck("G10_用例表与g4_b6_correctness.CASES逐字一致",
           sorted(mine) == sorted(legacy), dict(mine=len(mine), legacy=len(legacy),
                                                diff=[x for x in mine if x not in legacy][:2]))
        ck("G10_用例表含全部要求的类型",
           {"natural_language", "math", "code", "short_input", "long_input",
            "length_not_multiple_of_k", "length_lt_k", "eos"} <= {c["kind"] for c in BC.B6_CASES},
           sorted({c["kind"] for c in BC.B6_CASES}))
        ck("G10_非k倍数用例声明了裁剪与长度分支",
           all(set(c["branches"]) >= {"trim_last_round", "len_not_multiple_of_k"}
               for c in BC.B6_CASES if c["case_id"] in ("len13", "len5", "len2")),
           [(c["case_id"], list(c["branches"])) for c in BC.B6_CASES])
        ck("G10_EOS用例声明了eos_stop（未触发须记未覆盖）",
           any(c["case_id"] == "eos-prone" and "eos_stop" in c["branches"] for c in BC.B6_CASES), None)

        # ---------------- G11：集合级要求未满足 ⇒ 整体不得 PASS ----------------
        d11 = tmp / "g11"; d11.mkdir()
        _write(d11, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same,
                          rounds=[_round(0, 4, 4)]))     # 只有 all_accept，无 first_reject 等
        d11c = tmp / "g11c"; d11c.mkdir()
        # ⚠️ 夹具必须**真的**触发集合级要求的分支：`first_reject` 需要 m==0 的轮
        #    （首版漏了这一轮 ⇒ G11b 误报失败；属**测试夹具缺陷**，不是工具缺陷）。
        _write(d11c, _trio("strict", ar_gen=ar, prod_gen=pr_same, serial_gen=se_same,
                           rounds=[_round(0, 4, 4), _round(1, 4, 0), _round(2, 3, 1)]))
        decl11 = dict(cases=dict(strict=dict(kind="natural_language", max_new=16,
                                             branches=["all_accept"], expect_prefill_fused=None)),
                      set_requirements=dict(branches=["all_accept", "first_reject"],
                                            types=["natural_language"]))
        import sys as _sys
        import contextlib as _cl
        import io as _io
        dp = tmp / "decl11.json"
        dp.write_text(json.dumps(decl11, ensure_ascii=False), encoding="utf-8")

        def _run_cli(outdir: Path):
            buf = _io.StringIO()
            old = _sys.argv
            try:
                _sys.argv = ["b6_branch_coverage_cpu", "--tag", TAG, "--declared", str(dp),
                             "--outdir", str(outdir)]
                with _cl.redirect_stdout(buf):
                    rc = C.main()
            finally:
                _sys.argv = old
            return rc, buf.getvalue()

        rc11, _out11 = _run_cli(d11)
        ck("G11_集合级分支未覆盖⇒非PASS(exit1)", rc11 == 1, rc11)
        rc11c, _out11c = _run_cli(d11c)
        ck("G11b_集合级分支齐备⇒可PASS(exit0)", rc11c == 0,
           dict(rc=rc11c, out=_out11c[-600:]))
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
