"""stage_check_selftest_cpu.py — 逐阶段核对器的 **CPU 自测**（纯 CPU；不加载模型、不使用 GPU）。

## 目的
反例（本轮修复）：旧 `stage_check` 只用 `len(mrows) < reps` 判断重复是否齐备 ⇒ 下面三种**都能返回 OK**：
  * 两个 `rep1`（份数 = 2，长度检查看不见）；
  * `rep1 + rep3`（份数 = 2，且 rep3 不在预登记集合里）；
  * 同一份产物被计入两次（文件名相同）。

## 用例
| 用例 | 构造 | 期望 |
| --- | --- | --- |
| N1 | 齐全：每模式 `rep1+rep2` 各一份，其余硬门全过 | **OK（exit 0）** |
| N2 | 把 `rep2` 改名成第二份 `rep1`（份数仍 = 2） | STOP(1)，含"重复编号"与"缺 rep2" |
| N3 | 把 `rep2` 改成 `rep3` | STOP(1)，含"缺 rep2"与"预登记之外的编号 [3]" |
| N4 | 只写 `rep1`（`rep2` 未跑） | STOP(1)，含"缺 rep2" |
| N5 | 加载器把同一份产物返回两次（模拟重复计入） | STOP(1)，含"同一产物被重复计入" |
| N6 | 直接单测 `_rep_coverage`（含 `rep` 字段缺失/非法） | 各自报出对应问题 |
| N7 | 某模式 0 份产物 | 仍为 **PENDING(2)**（"未跑"不等于"编号错"，语义不能被改坏） |
| N8 | `rep='bad'`（非法值） | **不崩溃**、判 **STOP(1)** 并点名非法 `rep`（旧实现在 `int('bad')` 处抛 `ValueError`） |
| V1–V5 | 用**同一批夹具**跑 `stage2b_verdict.evaluate()` | 既有判决**不受影响**：齐全 ⇒ PASS；N2/N3/N4/N8 四种 ⇒ `CHECK_NEEDS_REVIEW` |

夹具复用 `stage2b_verdict_selftest_cpu` 的合成产物构造器（临时目录；**绝不触碰真实产物**）。
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import stage2b_verdict as SV
from . import stage2b_verdict_selftest_cpu as FIX
from . import stage_check as SC

FAM = "F09"


def _run_main(argv: list) -> tuple:
    """以给定 argv 跑 `stage_check.main()`，返回 (exit_code, stdout)。"""
    old = sys.argv
    buf = io.StringIO()
    try:
        sys.argv = ["stage_check"] + list(argv)
        with contextlib.redirect_stdout(buf):
            rc = SC.main()
    finally:
        sys.argv = old
    return rc, buf.getvalue()


def _argv(outdir: Path, *, modes: str = "ar,prod,serial", family: str = FAM) -> list:
    return ["--tag", FIX.TAG_GOOD, "--family", family, "--modes", modes,
            "--reps", str(FIX.REPS), "--outdir", str(outdir)]


def _recs(*, drop_rep2_for=None, rename_rep2_to=None) -> list:
    """取一份齐全夹具，并按用例需要改造 F09 的 AR 重复编号。

    `drop_rep2_for`：`(family, mode)` 的 rep2 产物**不写**（模拟"还没跑"）。
    `rename_rep2_to`：`(family, mode)` 的 rep2 记录**改成**给定编号
    （1 ⇒ 两个 rep1；3 ⇒ rep1+rep3）。编号用**新 run_id** ⇒ 不会覆盖第一份产物。
    """
    recs = FIX._full_set(FIX.TAG_GOOD)
    out = []
    for r in recs:
        key = (r.get("case"), r.get("mode"))
        if drop_rep2_for and key == tuple(drop_rep2_for[:2]) and int(r.get("rep") or 0) == 2:
            continue
        if rename_rep2_to and key == tuple(rename_rep2_to[:2]) and int(r.get("rep") or 0) == 2:
            r = {**r, "rep": int(rename_rep2_to[2]), "run_id": "20261004T000009Z"}
        out.append(r)
    return out


def main() -> int:
    checks: list = []

    def ck(name: str, ok: bool, extra=None) -> None:
        checks.append((name, bool(ok), extra))

    tmp = Path(tempfile.mkdtemp(prefix="scsel_"))
    try:
        # ---------- N1：齐全（每模式 rep1+rep2）⇒ OK ----------
        d1 = tmp / "n1"; d1.mkdir()
        FIX._write(d1, FIX._full_set(FIX.TAG_GOOD))
        rc1, out1 = _run_main(_argv(d1))
        ck("N1_齐全时返回OK", rc1 == 0 and "结论 = OK" in out1, (rc1, out1.strip()[-300:]))

        # ---------- N2：两个 rep1（份数仍 = 2）⇒ 不得 OK ----------
        d2 = tmp / "n2"; d2.mkdir()
        FIX._write(d2, _recs(rename_rep2_to=(FAM, "ar", 1)))
        rc2, out2 = _run_main(_argv(d2))
        ck("N2_两个rep1不得OK", rc2 != 0 and "结论 = OK" not in out2, (rc2,))
        ck("N2_报重复编号", "重复编号" in out2 and "rep1 有 2 份产物" in out2, out2[-400:])
        ck("N2_同时报缺rep2", "缺 rep2" in out2, out2[-400:])

        # ---------- N3：rep1 + rep3 ⇒ 不得 OK ----------
        d3 = tmp / "n3"; d3.mkdir()
        FIX._write(d3, _recs(rename_rep2_to=(FAM, "ar", 3)))
        rc3, out3 = _run_main(_argv(d3))
        ck("N3_rep1加rep3不得OK", rc3 != 0 and "结论 = OK" not in out3, (rc3,))
        ck("N3_报缺rep2", "缺 rep2" in out3, out3[-400:])
        ck("N3_报越界编号", "预登记之外的编号 [3]" in out3, out3[-400:])

        # ---------- N4：只有 rep1（rep2 未跑）⇒ 不得 OK ----------
        d4 = tmp / "n4"; d4.mkdir()
        FIX._write(d4, _recs(drop_rep2_for=(FAM, "ar")))
        rc4, out4 = _run_main(_argv(d4))
        ck("N4_缺rep2不得OK", rc4 != 0 and "结论 = OK" not in out4, (rc4,))
        ck("N4_报缺rep2", "缺 rep2" in out4, out4[-400:])

        # ---------- N5：加载器把同一份产物返回两次 ⇒ 不得 OK ----------
        d5 = tmp / "n5"; d5.mkdir()
        FIX._write(d5, FIX._full_set(FIX.TAG_GOOD))
        orig = SV.load_runs
        try:
            def _dup(tag, outdir):                       # noqa: ANN001
                rows = list(orig(tag, outdir))
                # ⚠️ 必须复制**被核对 family** 的那一条：首版复制了排序后的第一条
                #    （F03），而核对的是 F09 ⇒ 重复行被 family 过滤掉，用例形同虚设（自测抓到）。
                mine = [r for r in rows if r.get("case") == FAM]
                rows.append(mine[0] if mine else rows[0])
                return rows
            SV.load_runs = _dup
            rc5, out5 = _run_main(_argv(d5))
        finally:
            SV.load_runs = orig
        ck("N5_同一产物重复计入不得OK", rc5 != 0 and "结论 = OK" not in out5, (rc5,))
        ck("N5_报同一产物被重复计入", "同一产物被重复计入" in out5, out5[-400:])

        # ---------- N6：`_rep_coverage` 单测（含 rep 字段缺失/非法）----------
        clean = [dict(rep=1, file="a.json"), dict(rep=2, file="b.json")]
        ck("N6_齐全时无问题", SC._rep_coverage(clean, reps=2) == [], SC._rep_coverage(clean, reps=2))
        two1 = [dict(rep=1, file="a.json"), dict(rep=1, file="b.json")]
        p_two1 = SC._rep_coverage(two1, reps=2)
        ck("N6_两个rep1报重复编号", any("重复编号" in x for x in p_two1), p_two1)
        ck("N6_两个rep1同时报缺rep2", any("缺 rep2" in x for x in p_two1), p_two1)
        p_13 = SC._rep_coverage([dict(rep=1, file="a.json"), dict(rep=3, file="c.json")], reps=2)
        ck("N6_rep1加rep3报缺号与越界",
           any("缺 rep2" in x for x in p_13) and any("预登记之外的编号 [3]" in x for x in p_13), p_13)
        p_bad = SC._rep_coverage([dict(rep=None, file="a.json"), dict(rep=2, file="b.json")], reps=2)
        ck("N6_rep缺失或非法被报出", any("非法或缺失" in x for x in p_bad), p_bad)
        p_same = SC._rep_coverage([dict(rep=1, file="a.json")] * 2, reps=2)
        ck("N6_同一文件名两次被报出", any("同一产物被重复计入" in x for x in p_same), p_same)

        # ---------- N8：`rep='bad'`（非法值）⇒ 明确 STOP 原因，**不得抛 ValueError** ----------
        # 旧实现：`_rep_coverage` 内第二次 `int(raw)` 与 `main()` 的排序键都会崩。
        recs_bad = FIX._full_set(FIX.TAG_GOOD)
        for r in recs_bad:
            if (r.get("case"), r.get("mode"), int(r.get("rep") or 0)) == (FAM, "ar", 2):
                r["rep"] = "bad"                      # 非法编号；文件名会变成 ..._repbad_...
        d8 = tmp / "n8"; d8.mkdir()
        FIX._write(d8, recs_bad)
        crash8 = None
        try:
            rc8, out8 = _run_main(_argv(d8, modes="ar"))
        except Exception as e:                        # noqa: BLE001
            crash8, rc8, out8 = f"{type(e).__name__}: {e}", None, ""
        ck("N8_非法rep不崩溃", crash8 is None, crash8)
        ck("N8_非法rep判STOP而非OK", rc8 == 1 and "结论 = STOP" in out8, (rc8, out8[-260:]))
        ck("N8_理由点名非法rep", "非法或缺失" in out8 and "'bad'" in out8, out8[-400:])
        p_bad_str = SC._rep_coverage([dict(rep="bad", file="a.json")], reps=2)
        ck("N8_单测非法rep可解析出问题（不抛异常）",
           any("非法或缺失" in x for x in p_bad_str), p_bad_str)
        ck("N8_单测同时报缺号",
           any("缺 rep1" in x for x in p_bad_str) and any("缺 rep2" in x for x in p_bad_str), p_bad_str)

        # ---------- N9：`_as_rep` 的**边界**——只接受 ASCII 数字组成的整数表示 ----------
        # 旧版用 `isdigit()` 当门：`'²'.isdigit()` 为真但 `int('²')` 抛 ValueError ⇒ 解析器自崩；
        # 且 `'٣'`/`'１'` 会被放行并解析成 3/1 ⇒ 编号口径不唯一。
        acc = {"1": 1, "-1": -1, "01": 1, " 2 ": 2, "10": 10}
        rej = {"²": None, "³": None, "½": None, "٣": None, "１": None, "一": None,
               "+1": None, "1.0": None, "": None, "  ": None, "1 2": None, "0x2": None}
        bad_acc = {k: SV._as_rep(k) for k, want in acc.items() if SV._as_rep(k) != want}
        ck("N9_ASCII数字被正确接受", not bad_acc, bad_acc)
        raised = {}
        for k in rej:
            try:
                got = SV._as_rep(k)
            except Exception as e:                                    # noqa: BLE001
                got = f"RAISED:{type(e).__name__}"
            if got is not None:
                raised[k] = got
        ck("N9_非ASCII数字/非法串返回None且不抛异常", not raised, raised)
        typed = {repr(v): SV._as_rep(v) for v in (None, True, False, 1.5, [], {}, 2)}
        ck("N9_类型边界（bool/float/None/容器）",
           typed["True"] is None and typed["False"] is None and typed["None"] is None
           and typed["1.5"] is None and typed["[]"] is None and typed["{}"] is None
           and typed["2"] == 2, typed)

        # ---------- N10：上标数字 `rep='²'` 端到端 ⇒ STOP，且**不崩溃** ----------
        recs_sup = FIX._full_set(FIX.TAG_GOOD)
        for r in recs_sup:
            if (r.get("case"), r.get("mode"), int(r.get("rep") or 0)) == (FAM, "ar", 2):
                r["rep"] = "²"
        d10 = tmp / "n10"; d10.mkdir()
        FIX._write(d10, recs_sup)
        crash10 = None
        try:
            rc10, out10 = _run_main(_argv(d10, modes="ar"))
        except Exception as e:                                        # noqa: BLE001
            crash10, rc10, out10 = f"{type(e).__name__}: {e}", None, ""
        ck("N10_上标数字rep不崩溃", crash10 is None, crash10)
        ck("N10_上标数字rep判STOP", rc10 == 1 and "结论 = STOP" in out10, (rc10, out10[-240:]))
        ck("N10_理由点名该非法值", "非法或缺失" in out10 and "²" in out10, out10[-360:])

        # ---------- N7：某模式 0 份产物 ⇒ 仍是 PENDING(2) ----------
        d7 = tmp / "n7"; d7.mkdir()
        FIX._write(d7, [r for r in FIX._full_set(FIX.TAG_GOOD) if r.get("mode") != "serial"])
        rc7, out7 = _run_main(_argv(d7))
        ck("N7_未跑的阶段仍为PENDING", rc7 == 2 and "结论 = PENDING" in out7, (rc7, out7[-260:]))

        # ---------- V1–V4：同一批夹具下 `stage2b_verdict` 的既有判决不受影响 ----------
        sV1 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d1)
        ck("V1_齐审判决仍为PASS", sV1["decision"] == "SERIAL_ORACLE_PASS", sV1["decision"])
        sV2 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d2)
        ck("V2_两个rep1仍被判重复编号冲突",
           sV2["decision"] == "CHECK_NEEDS_REVIEW"
           and any("重复编号冲突" in x for x in sV2["needs_review"]), sV2["needs_review"][:3])
        sV3 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d3)
        ck("V3_rep1加rep3仍被判缺运行与越界三元组",
           sV3["decision"] == "CHECK_NEEDS_REVIEW"
           and any("缺运行" in x for x in sV3["needs_review"])
           and any("非预期三元组" in x for x in sV3["needs_review"]), sV3["needs_review"][:4])
        sV4 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d4)
        ck("V4_缺rep2仍被判缺运行",
           sV4["decision"] == "CHECK_NEEDS_REVIEW"
           and any("缺运行" in x for x in sV4["needs_review"]), sV4["needs_review"][:3])
        sV5 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d8)
        ck("V5_非法rep不崩溃且判需复核",
           sV5["decision"] == "CHECK_NEEDS_REVIEW"
           and any("rep 字段非法" in x for x in sV5["needs_review"]), sV5["needs_review"][:3])
        sV6 = SV.evaluate(FIX.TAG_GOOD, list(FIX.NEW), list(FIX.MODES), FIX.REPS, outdir=d10)
        ck("V6_上标数字rep不崩溃且判需复核",
           sV6["decision"] == "CHECK_NEEDS_REVIEW"
           and any("rep 字段非法" in x for x in sV6["needs_review"]), sV6["needs_review"][:3])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== 逐阶段核对器自测（纯 CPU；合成产物在临时目录，未触碰真实产物）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}" + (f"  {str(extra)[:200]}" if not ok else ""))
    print(f"\n逐阶段核对器自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
