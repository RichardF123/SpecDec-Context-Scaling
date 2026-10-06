"""b6_orchestrate_selftest_cpu.py — **编排级**判定的纯 CPU 自测（不加载模型、不使用 GPU）。

被测对象：`b6_orchestrate_cpu.evaluate_group()` / `progress()` / `main()`。
夹具直接复用 `b6_branch_coverage_selftest_cpu` 的合成产物构造器（`_mk`/`_trio`/`_round`/`_write`），
避免"两份夹具各自漂移"。

| 组 | 场景 | 期望 |
| --- | --- | --- |
| O1 | 3 模式 × 2 rep 全合法且一致 | `CONTINUE`，无停止类别，CHK 判 `CASE_PASS` |
| O2 | 子进程退出码非 0 | `STOP/CHILD_RC_NONZERO` |
| O3 | 退出码 0 但产物硬检查失败（`self_checks_all_ok=False`） | `STOP/HARD_CHECK_FAIL`（**退出码 0 不等于通过**） |
| O4 | 只有 rep1（缺 rep2） | `STOP/ARTIFACT_MISSING` |
| O5 | 同一 `(mode,rep)` 两份产物 | `STOP/ARTIFACT_DUPLICATE`（不静默择一） |
| O6 | 同模式两次重复**完整序列不相等** | `STOP/REPEAT_MISMATCH` |
| O7 | serial 与 AR **完整序列不相等** | `STOP/SERIAL_VS_AR_MISMATCH` |
| O8 | **只有** production 与 AR 不同（serial==AR） | `CONTINUE` + 记录 `PRODUCTION_DIVERGENCE`；CHK 仍 `CASE_PASS`；无"无损"表述 |
| O9 | 声明分支未触发 | `CONTINUE` + `BRANCH_UNCOVERED`；CHK 判 `CASE_INCOMPLETE`（**不得** PASS） |
| O10 | 环境门拒绝（`blocked_by`） | `STOP/ENV_BLOCKED` |
| O11 | 臂漂移（跑成 `arm=eager`） | `STOP/ARM_MISMATCH`（Stage 2b 遗迹正是靠这条拦住） |
| O12 | 要求证明 prefill 融合但 `fused=0` | 要求时 `STOP/FUSED_PREFILL_REQUIRED_BUT_ABSENT`；不要求时 `CONTINUE` |
| O13 | 非 prefill 阶段出现融合 | `STOP/BACKEND_FUSION_PROBLEM` |
| O14 | 必需字段缺失 / 类型非法 | `STOP/FIELD_MISSING` / `STOP/TYPE_ILLEGAL` |
| O15 | 诊断精度非法取值 | `STOP/DIAG_PRECISION_FAIL` |
| O16 | 进度清单（6 槽中 3 槽完成） | 已完成/未完成**照实**列出，重复产物标注 |
| O17 | CLI 退出码与 `--no-write` | STOP⇒1、CONTINUE⇒0、参数非法⇒3；`--no-write` 不落盘 |
| O18 | **对偶断言**：CHK 硬失败必须成为编排停止类别 | O3/O6/O7 同时满足 `CASE_FAIL` 且 `STOP` |
| O19 | `expect=None` 无融合且无回退理由 | `CONTINUE` + `DISPATCH_RULE_EVIDENCE_BOUNDARY`；CHK `CASE_INCOMPLETE` |
| O20 | 只记录项不得被写成通过 | 结果 JSON 中不得出现"无损/安全/通过"式表述 |
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import b6_branch_coverage_cpu as CHK
from . import b6_orchestrate_cpu as O
from .b6_branch_coverage_selftest_cpu import (AR16, DECL, TAG, _mk, _round, _trio, _write)


def _spec(**kw) -> dict:
    base = dict(kind="natural_language", max_new=16, branches=["all_accept", "trim_last_round"],
                expect_prefill_fused=True)
    base.update(kw)
    return base


def _spec_none(**kw) -> dict:
    """`expect_prefill_fused=None` 的声明（短输入/按分派规则）。"""
    return _spec(expect_prefill_fused=None, branches=[], **kw)


ROUNDS_OK = [_round(0, 4, 4), _round(1, 3, 2)]


def main() -> int:
    checks: list = []

    def ck(name: str, ok: bool, extra=None) -> None:
        checks.append((name, bool(ok), extra))

    tmp = Path(tempfile.mkdtemp(prefix="b6orch_"))
    counter = {"n": 0}

    def group(case: str, recs: list, spec: dict, *, modes=("ar", "prod", "serial"), reps=(1, 2),
              names=None, child_rc=None, require_fused_prefill=False, expect_arm=O.DEFAULT_ARM,
              extra_recs=None) -> dict:
        """写盘 → `evaluate_group`。"""
        counter["n"] += 1
        d = tmp / f"g{counter['n']}"
        d.mkdir()
        _write(d, [r for r in recs if r is not None], names=names)
        for r in (extra_recs or []):
            _write(d, [r])
        return O.evaluate_group(TAG, case, list(modes), list(reps), spec, outdir=d,
                                expect_backend="CUDNN", expect_arm=expect_arm,
                                require_fused_prefill=require_fused_prefill, child_rc=child_rc)

    def chk_case(case: str, recs: list, spec: dict, *, modes=("ar", "prod", "serial"),
                 names=None, extra_recs=None) -> str:
        """同一批产物交给 CHK 判定（用于对偶断言）。"""
        counter["n"] += 1
        d = tmp / f"g{counter['n']}"
        d.mkdir()
        _write(d, [r for r in recs if r is not None], names=names)
        for r in (extra_recs or []):
            _write(d, [r])
        runs, probs = CHK.find_runs(TAG, case, d)
        return CHK.evaluate_case(case, spec, runs, expect_backend="CUDNN",
                                 expect_config=dict(max_new=spec.get("max_new"),
                                                    modes=list(modes)),
                                 run_problems=probs)["verdict"]

    try:
        # ---------------- O1：合法整组 ⇒ CONTINUE 且用例 PASS ----------------
        recs_ok = _trio("o1", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        rounds=ROUNDS_OK)
        g1 = group("o1", recs_ok, _spec())
        ck("O1_合法整组⇒CONTINUE且无停止类别",
           g1["decision"] == "CONTINUE" and not g1["stop_categories"],
           (g1["decision"], g1["stop_categories"], g1["case_reasons"][:2]))
        ck("O1_用例判定=CASE_PASS", g1["case_verdict"] == "CASE_PASS", g1["case_reasons"][:3])

        # ---------------- O2：子进程退出码非 0 ⇒ STOP ----------------
        g2 = group("o1", recs_ok, _spec(), child_rc=1)
        ck("O2_退出码非0⇒STOP/CHILD_RC_NONZERO",
           g2["decision"] == "STOP" and g2["stop_categories"] == ["CHILD_RC_NONZERO"],
           (g2["decision"], g2["stop_categories"]))

        # ---------------- O3：退出码 0 但硬检查失败 ⇒ STOP ----------------
        recs_bad = _trio("o3", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                         rounds=ROUNDS_OK, self_ok=False)
        g3 = group("o3", recs_bad, _spec())
        ck("O3_退出码0但硬检查失败⇒STOP/HARD_CHECK_FAIL",
           g3["decision"] == "STOP" and "HARD_CHECK_FAIL" in g3["stop_categories"],
           (g3["decision"], g3["stop_categories"], g3["stop_reasons"][:2]))
        ck("O3对偶_CHK判CASE_FAIL",
           chk_case("o3", recs_bad, _spec()) == "CASE_FAIL", None)

        # ---------------- O4：缺 rep2 ⇒ STOP ----------------
        recs_half = [r for r in recs_ok if r["rep"] == 1]
        g4 = group("o4", recs_half, _spec())
        ck("O4_缺产物⇒STOP/ARTIFACT_MISSING",
           g4["decision"] == "STOP" and "ARTIFACT_MISSING" in g4["stop_categories"],
           (g4["decision"], g4["stop_categories"], g4["stop_reasons"][:2]))

        # ---------------- O5：重复产物 ⇒ STOP（不静默择一）----------------
        # ⚠️ 必须用**同 case** 的一套产物：写盘文件名含 case，用别的 case 的产物会变成"缺产物"（首版即如此）。
        recs5 = _trio("o5", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                      rounds=ROUNDS_OK)
        dup = dict(recs5[0])
        dup["run_id"] = "20261006T111111Z"
        g5 = group("o5", recs5, _spec(), extra_recs=[dup])
        ck("O5_重复产物⇒STOP/ARTIFACT_DUPLICATE",
           g5["decision"] == "STOP" and "ARTIFACT_DUPLICATE" in g5["stop_categories"],
           (g5["decision"], g5["stop_categories"], g5["stop_reasons"][:2]))

        # ---------------- O6：同模式跨 rep 序列不等 ⇒ STOP ----------------
        recs_r = _trio("o6", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                       rounds=ROUNDS_OK)
        for r in recs_r:
            if r["mode"] == "prod" and r["rep"] == 2:
                r["smoke"]["gen"] = [999] + list(AR16[1:])
                r["smoke"]["gen_len"] = len(AR16)
        g6 = group("o6", recs_r, _spec())
        ck("O6_重复不一致⇒STOP/REPEAT_MISMATCH",
           g6["decision"] == "STOP" and "REPEAT_MISMATCH" in g6["stop_categories"],
           (g6["decision"], g6["stop_categories"], g6["stop_reasons"][:2]))
        ck("O6对偶_CHK判CASE_FAIL", chk_case("o6", recs_r, _spec()) == "CASE_FAIL", None)

        # ---------------- O7：serial 与 AR 不等 ⇒ STOP ----------------
        ser_bad = list(AR16)
        ser_bad[3] = 777
        recs_s = _trio("o7", ar_gen=AR16, prod_gen=list(AR16), serial_gen=ser_bad,
                       rounds=ROUNDS_OK)
        g7 = group("o7", recs_s, _spec())
        ck("O7_serial≠AR⇒STOP/SERIAL_VS_AR_MISMATCH",
           g7["decision"] == "STOP" and "SERIAL_VS_AR_MISMATCH" in g7["stop_categories"],
           (g7["decision"], g7["stop_categories"], g7["stop_reasons"][:2]))
        ck("O7对偶_CHK判CASE_FAIL", chk_case("o7", recs_s, _spec()) == "CASE_FAIL", None)

        # ---------------- O8：**只有** production 不同 ⇒ CONTINUE + 只记录 ----------------
        prod_diff = list(AR16)
        prod_diff[5] = 4242
        recs_p = _trio("o8", ar_gen=AR16, prod_gen=prod_diff, serial_gen=list(AR16),
                       rounds=ROUNDS_OK)
        g8 = group("o8", recs_p, _spec())
        kinds8 = {f["kind"] for f in g8["findings"]}
        ck("O8_production分歧⇒CONTINUE且列为只记录项",
           g8["decision"] == "CONTINUE" and "PRODUCTION_DIVERGENCE" in kinds8,
           (g8["decision"], sorted(kinds8), g8["stop_categories"]))
        ck("O8_production分歧不使用例判FAIL（协议：只记录）",
           g8["case_verdict"] == "CASE_PASS" and chk_case("o8", recs_p, _spec()) == "CASE_PASS",
           (g8["case_verdict"],))
        blob8 = json.dumps(g8, ensure_ascii=False)
        ck("O8_不得出现“无损/安全/通过”式表述",
           not any(w in blob8 for w in ("无损", "安全无损", "同输出", "已验证通过")), None)

        # ---------------- O9：分支未触发 ⇒ CONTINUE + 覆盖缺口（非 PASS）----------------
        g9 = group("o9", _trio("o9", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16)),
                   _spec(branches=["eos_stop", "len_not_multiple_of_k"]))
        kinds9 = {f["kind"] for f in g9["findings"]}
        ck("O9_覆盖缺口⇒CONTINUE+BRANCH_UNCOVERED",
           g9["decision"] == "CONTINUE" and "BRANCH_UNCOVERED" in kinds9,
           (g9["decision"], sorted(kinds9), g9["stop_categories"]))
        ck("O9_覆盖缺口⇒用例不得判PASS（INCOMPLETE）",
           g9["case_verdict"] == "CASE_INCOMPLETE", g9["case_reasons"][:3])

        # ---------------- O10：环境门拒绝 ⇒ STOP ----------------
        g10 = group("o10", _trio("o10", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK, overrides={"blocked_by": "environment"}),
                    _spec())
        ck("O10_环境门拒绝⇒STOP/ENV_BLOCKED",
           g10["decision"] == "STOP" and "ENV_BLOCKED" in g10["stop_categories"],
           (g10["decision"], g10["stop_categories"]))

        # ---------------- O11：臂漂移 ⇒ STOP ----------------
        g11 = group("o11", _trio("o11", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK, arm="eager"), _spec_none())
        ck("O11_跑成eager⇒STOP/ARM_MISMATCH（不得当候选臂证据）",
           g11["decision"] == "STOP" and "ARM_MISMATCH" in g11["stop_categories"],
           (g11["decision"], g11["stop_categories"]))

        # ---------------- O12：要求证明 prefill 融合 ----------------
        no_fuse = _trio("o12", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                        rounds=ROUNDS_OK, fused_by_phase={},
                        fallbacks=[dict(phase="decode", s=1, reason="phase='decode' != 'prefill'",
                                        kind="phase_not_prefill")],
                        reasons={"phase_not_prefill": 30})
        g12a = group("o12", no_fuse, _spec_none(), require_fused_prefill=True)
        ck("O12a_要求融合但fused=0⇒STOP/FUSED_PREFILL_REQUIRED_BUT_ABSENT",
           g12a["decision"] == "STOP"
           and "FUSED_PREFILL_REQUIRED_BUT_ABSENT" in g12a["stop_categories"],
           (g12a["decision"], g12a["stop_categories"]))
        g12b = group("o12", no_fuse, _spec_none(), require_fused_prefill=False)
        ck("O12b_不要求融合时同产物⇒CONTINUE（回退理由已记录）",
           g12b["decision"] == "CONTINUE", (g12b["decision"], g12b["stop_categories"]))

        # ---------------- O13：非 prefill 融合 ⇒ STOP ----------------
        g13 = group("o13", _trio("o13", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK, fused_by_phase={"prefill": 2, "verify": 1}),
                    _spec())
        ck("O13_非prefill融合⇒STOP/BACKEND_FUSION_PROBLEM",
           g13["decision"] == "STOP" and "BACKEND_FUSION_PROBLEM" in g13["stop_categories"],
           (g13["decision"], g13["stop_categories"], g13["stop_reasons"][:2]))

        # ---------------- O14：缺字段 / 类型非法 ⇒ STOP ----------------
        g14a = group("o14", _trio("o14", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                  rounds=ROUNDS_OK, drop=("smoke.cache_structure_n_checks",)),
                     _spec())
        ck("O14a_缺字段⇒STOP/FIELD_MISSING",
           g14a["decision"] == "STOP" and "FIELD_MISSING" in g14a["stop_categories"],
           (g14a["decision"], g14a["stop_categories"], g14a["stop_reasons"][:1]))
        g14b = group("o14b", _trio("o14b", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                   rounds=ROUNDS_OK,
                                   overrides={"smoke.cache_structure_n_checks": "40"}), _spec())
        ck("O14b_类型非法⇒STOP/TYPE_ILLEGAL",
           g14b["decision"] == "STOP" and "TYPE_ILLEGAL" in g14b["stop_categories"],
           (g14b["decision"], g14b["stop_categories"], g14b["stop_reasons"][:1]))

        # ---------------- O15：诊断精度非法 ⇒ STOP ----------------
        g15 = group("o15", _trio("o15", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK,
                                 overrides={"diag_precision_before_run.qk_fp32": "invalid"}), _spec())
        ck("O15_诊断精度非法⇒STOP/DIAG_PRECISION_FAIL",
           g15["decision"] == "STOP" and "DIAG_PRECISION_FAIL" in g15["stop_categories"],
           (g15["decision"], g15["stop_categories"], g15["stop_reasons"][:1]))

        # ---------------- O16：进度清单（3/6 完成）----------------
        counter["n"] += 1
        dp = tmp / f"g{counter['n']}"
        dp.mkdir()
        _write(dp, [r for r in _trio("o16", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                     rounds=ROUNDS_OK) if not (r["mode"] == "serial")])
        pr = O.progress(TAG, dict(cases=["o16"], modes=["ar", "prod", "serial"], reps=[1, 2]),
                        outdir=dp)
        ck("O16_进度清单如实（完成4/未完成2）",
           pr["n_planned"] == 6 and pr["n_completed"] == 4 and pr["n_pending"] == 2
           and set(pr["pending"]) == {"o16/serial/rep1", "o16/serial/rep2"},
           (pr["n_completed"], pr["pending"]))
        ck("O16_进度产物只用只读口径", pr["note"].startswith("只读进度"), None)

        # ---------------- O17：CLI 退出码 / --no-write ----------------
        def cli(args: list) -> int:
            buf = io.StringIO()
            old = sys.argv
            try:
                sys.argv = ["b6_orchestrate_cpu"] + args
                with contextlib.redirect_stdout(buf):
                    return O.main()
            finally:
                sys.argv = old

        counter["n"] += 1
        dc = tmp / f"g{counter['n']}"
        dc.mkdir()
        _write(dc, _trio("o17", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                         rounds=ROUNDS_OK))
        # 合成 case 不在内建用例表里 ⇒ 必须给 `--declared`（与真实批次的用法一致）
        decl17 = dc / "_decl.json"
        decl17.write_text(json.dumps(dict(cases={"o17": _spec()}), ensure_ascii=False),
                          encoding="utf-8")
        before = sorted(p.name for p in dc.iterdir())
        base = ["--tag", TAG, "--outdir", str(dc), "--declared", str(decl17), "--no-write"]
        rc_ok = cli(base + ["--case", "o17"])
        after = sorted(p.name for p in dc.iterdir())
        ck("O17a_CONTINUE⇒退出码0", rc_ok == 0, rc_ok)
        ck("O17b_--no-write不落盘", before == after, (before, after))
        rc_stop = cli(base + ["--case", "o17", "--child-rc", "1"])
        ck("O17c_STOP⇒退出码1", rc_stop == 1, rc_stop)
        rc_bad = cli(["--tag", TAG, "--case", "o17", "--outdir", str(dc), "--modes", "nope",
                      "--no-write"])
        ck("O17d_参数非法⇒退出码3", rc_bad == 3, rc_bad)
        rc_bad2 = cli(base + ["--case", "不存在"])
        ck("O17e_未知用例⇒退出码3", rc_bad2 == 3, rc_bad2)

        # ---------------- O19：expect=None 无融合且无回退理由 ⇒ 证据边界（非硬失败）----------------
        g19 = group("o19", _trio("o19", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK, fused_by_phase={}), _spec_none())
        kinds19 = {f["kind"] for f in g19["findings"]}
        ck("O19_分派规则证据边界⇒CONTINUE+记录（非硬失败）",
           g19["decision"] == "CONTINUE" and "DISPATCH_RULE_EVIDENCE_BOUNDARY" in kinds19,
           (g19["decision"], sorted(kinds19)))
        ck("O19_证据边界⇒用例不得判PASS",
           g19["case_verdict"] == "CASE_INCOMPLETE", g19["case_reasons"][:3])

        # ---------------- O20：停止类别集合是白名单（防拼写漂移）----------------
        seen = set()
        for g in (g2, g3, g4, g5, g6, g7, g10, g11, g12a, g13, g14a, g14b, g15):
            seen |= set(g["stop_categories"])
        ck("O20_停止类别全部在预登记白名单内",
           seen <= set(O.STOP_CATEGORIES), sorted(seen - set(O.STOP_CATEGORIES)))
        ck("O20_只记录类型全部在预登记白名单内",
           {f["kind"] for f in (g8["findings"] + g9["findings"] + g19["findings"])}
           <= set(O.RECORD_ONLY_KINDS), None)

        # ---------------- O21：rev.5 覆盖规则修正后，编排判定不得把"非必需未执行"当失败 ----------------
        # `uncovered` 非空但逐条为 required=False/executed=False ⇒ 编排必须 CONTINUE（只记录）。
        def _unc_nonreq(rec: dict) -> None:
            ch = rec["smoke"]["checks"]
            ch["uncovered"] = ["finiteness:repair/draft", "finiteness:repair/target"]
            ch["checks"] = [e for e in ch["checks"] if e["name"] not in ch["uncovered"]]
            ch["checks"] += [dict(name=nm, required=False, executed=False, ok=None,
                                  detail="该分支本次未执行（非必需）") for nm in ch["uncovered"]]

        recs21 = _trio("o21", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                       rounds=ROUNDS_OK)
        for r in recs21:
            _unc_nonreq(r)
        g21 = group("o21", recs21, _spec())
        ck("O21_非必需未执行的检查⇒CONTINUE（不再误判为失败）",
           g21["decision"] == "CONTINUE" and g21["case_verdict"] == "CASE_PASS",
           (g21["decision"], g21["case_verdict"], g21["stop_categories"]))
        ck("O21_覆盖缺口被如实记录（含'未执行'字样）",
           any("未执行" in n for n in g21["notes"]), g21["notes"][:2])

        # ---------------- O22：短块合法回退 ⇒ CONTINUE，且明确记录"融合路径未被覆盖" ----------------
        g22 = group("o22", _trio("o22", ar_gen=AR16, prod_gen=list(AR16), serial_gen=list(AR16),
                                 rounds=ROUNDS_OK, fused_by_phase={},
                                 fallbacks=[dict(phase="prefill", s=1,
                                                 reason="s=1 < PREFILL_FUSED_MIN_S=2（cuDNN 不支持 seq_len=1）",
                                                 kind="short_block_seq_len_lt_min")],
                                 reasons={"short_block_seq_len_lt_min": 1}), _spec_none())
        ck("O22_短块合法回退⇒CONTINUE（不算硬失败）", g22["decision"] == "CONTINUE",
           (g22["decision"], g22["stop_categories"]))
        ck("O22_必须明确记录'融合路径未被覆盖'",
           any("融合路径未被覆盖" in n for n in g22["notes"]), g22["notes"][:3])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, extra in checks:
        if not ok:
            print(f"  [FAIL] {name} {extra}")
    print(f"b6_orchestrate_selftest_cpu: {npass}/{len(checks)} 通过")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
