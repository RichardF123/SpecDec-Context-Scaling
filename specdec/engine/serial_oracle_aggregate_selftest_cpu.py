"""serial_oracle_aggregate_selftest_cpu.py — **聚合器**的针对性自测（纯 CPU、只读、不加载模型）。

## 为什么需要它（本轮的教训）
本轮在**同一类**缺陷上栽了三次：
* **D7**：有限性计数读 `checks.n_checked`（不存在）⇒ 真实路径是 `checks.finiteness_coverage.n_checked`
  ⇒ 硬检查被误判为失败；
* **D9**：D6 硬检查读 `r["verify_mode_at_run_end"]` / `r["verify_mode_after_run"]`，而 `_row`
  只暴露 `verify_mode` / `verify_mode_after` ⇒ 恒为 None ⇒ D6 被误判为失败（假阴性）。
* （**D8** 属判定范围，不属字段名。）

⇒ 必须有一条**自动化的"字段名契约"检查**：硬检查里引用的每个 `r.<字段>` 都必须在 `_row` 的输出里存在。
本自测就是这条检查，外加对 D6/D7 两个**具体**字段的显式断言（回归）。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import serial_oracle_aggregate as AG

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

# 由聚合器在 `_row` **之后**附加的内部键（不是原始产物字段）
INTERNAL_KEYS = {"_rounds", "_trace", "_cross", "_first_div_vs_ar", "_ar_stop", "_file"}


def _ck(name, cond, extra=None):
    return (name, bool(cond), extra)


def _referenced_row_fields(src: str) -> set:
    """扫描源码里 `r.get("X")` / `r["X"]` 形式的字段引用。"""
    pats = re.findall(r"""\br\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""", src)
    pats += re.findall(r"""\br\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]""", src)
    return set(pats)


def main() -> int:
    checks: list = []
    src = (ROOT / "specdec" / "engine" / "serial_oracle_aggregate.py").read_text(encoding="utf-8")

    # 找一个**真实的** serial 运行产物。⚠️ 必须区分"post-fix"（有 `verify_mode_at_run_end`）
    # 与"pre-fix"（F13 旧产物，没有该键）—— 首版把两者混为一谈，于是 D6 检查对 pre-fix 记录给出
    # 正确的 False 时，测试误报失败（**测试自身的设计缺陷**，已修正）。
    all_recs = []
    for f in sorted(OUTDIR.glob("div8209_so_F*_serial_rep*_*.json"), key=lambda p: p.stat().st_mtime):
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if d.get("model_run_happened"):
            all_recs.append((f, d))
    post = [(f, d) for f, d in all_recs if d.get("verify_mode_at_run_end") is not None]
    pre = [(f, d) for f, d in all_recs if d.get("verify_mode_at_run_end") is None]
    real, rec = (post[-1] if post else (all_recs[-1] if all_recs else (None, None)))
    row = AG._row(rec) if rec else {}
    keys = set(row.keys())

    checks += [_ck("s00_real_artifact_available", real is not None,
                   "没有可用的真实 serial 产物 ⇒ 字段契约检查无法评估（不假装通过）"),
               _ck("s00b_post_fix_artifact_available", bool(post),
                   dict(n_post=len(post), n_pre=len(pre),
                        note="post-fix = 记录里有 verify_mode_at_run_end（D6 修正之后）"))]

    # 1) 硬检查源码里引用的字段必须都在 `_row` 输出里（字段名契约）
    hard_src = "\n".join([src[src.find("def _serial_hard_checks("):src.find("def _stage1_rounds_fields(")],
                          src[src.find("def _serial_hard_checks2("):src.find("def aggregate_final(")]])
    refs = _referenced_row_fields(hard_src) - INTERNAL_KEYS
    missing = sorted(refs - keys)
    checks += [_ck("s01_hard_check_field_names_exist_in_row", not missing,
                   dict(missing=missing, n_refs=len(refs))),
               _ck("s02_scan_actually_found_references", len(refs) >= 8, len(refs))]

    # 2) D6 回归：字段必须存在且能正确判定（本轮 8 次新运行的实测值）
    checks += [_ck("s03_d6_fields_exposed",
                   "verify_mode_at_run_end" in keys and "verify_mode_after_run" in keys,
                   dict(has_at_end=("verify_mode_at_run_end" in keys),
                        has_after=("verify_mode_after_run" in keys)))]
    if rec is not None:
        h = AG._serial_hard_checks2(row, require_d6=True)
        d6 = h.get("18_d6_mode_restored_after_cleanup") or {}
        # post-fix 记录：该项必须读出**真实值**（不能是 None）且 ok=True
        checks += [_ck("s04_d6_check_reads_real_values_not_none",
                       d6.get("ok") is not None and "None" not in str(d6.get("detail"))[:60]
                       and d6.get("ok") is True,
                       d6),
                   _ck("s05_d6_ok_matches_raw_record",
                       bool(d6.get("ok")) == bool(row.get("verify_mode_after_run") == "block"
                                                  and row.get("verify_mode_at_run_end") is not None),
                       dict(ok=d6.get("ok"), after=row.get("verify_mode_after_run"),
                            at_end=row.get("verify_mode_at_run_end")))]
    # pre-fix 记录（F13）：require_d6=True 时必须 **False**（如实），require_d6=False 时必须 **na**
    if pre:
        rpre = AG._row(pre[-1][1])
        d_strict = (AG._serial_hard_checks2(rpre, require_d6=True)
                    .get("18_d6_mode_restored_after_cleanup") or {})
        d_na = (AG._serial_hard_checks2(rpre, require_d6=False)
                .get("18_d6_mode_restored_after_cleanup") or {})
        checks += [_ck("s04b_pre_fix_record_yields_false_when_required",
                       d_strict.get("ok") is False, dict(file=pre[-1][0].name, d=d_strict)),
                   _ck("s04c_pre_fix_record_is_na_for_f13", d_na.get("ok") is None
                       and d_na.get("na") is True, d_na)]

    # 3) D7 回归：有限性计数必须从真实路径读出且 >0（若有该字段）
    checks += [_ck("s06_d7_finiteness_count_path",
                   (row.get("finiteness_n_checked") or 0) > 0,
                   dict(finiteness_n_checked=row.get("finiteness_n_checked"),
                        note="真实路径 checks.finiteness_coverage.n_checked"))]

    # 4) 16 项硬检查在真实记录上都应产出 ok 非 None（除 na 项）
    h16 = AG._serial_hard_checks(row)
    none_items = [k for k, v in h16.items() if v["ok"] is None and not v.get("na")]
    checks += [_ck("s07_no_unexpectedly_unevaluable_hard_check", not none_items, none_items),
               _ck("s08_hard_check_count_is_16", len(h16) == 16, sorted(h16))]

    # 5) F13 的 D6 项必须是 **na**（带理由），不是 False、也不是静默跳过
    f13 = [f for f in sorted(OUTDIR.glob("div8209_so_F13_serial_rep*_*.json"))
           if json.loads(f.read_text(encoding="utf-8-sig")).get("model_run_happened")]
    if f13:
        r13 = AG._row(json.loads(f13[-1].read_text(encoding="utf-8-sig")))
        h13 = AG._serial_hard_checks2(r13, require_d6=False)
        d = h13.get("18_d6_mode_restored_after_cleanup") or {}
        checks += [_ck("s09_f13_d6_is_na_with_reason",
                       d.get("ok") is None and d.get("na") is True and len(str(d.get("detail"))) > 20, d)]
    else:
        checks += [_ck("s09_f13_d6_is_na_with_reason", False, "没有 F13 serial 产物")]

    # 6) 上一轮 AR 参考可独立读出（F01..F10）且与登记锚点一致
    for fam in ("F01", "F02", "F08", "F10"):
        p = AG._prior_ar(fam)
        checks += [_ck(f"s10_{fam}_prior_ar_available_and_consistent",
                       bool(p.get("available") and p.get("tokens_identical") is True
                            and p.get("agrees_with_registered_anchor") is True),
                       {k: p[k] for k in ("available", "n_files", "hashes", "tokens_identical",
                                          "agrees_with_registered_anchor")})]
    p13 = AG._prior_ar("F13")
    checks += [_ck("s11_f13_prior_ar_uses_registered_anchor_not_false",
                   p13.get("available") is False
                   and p13.get("agrees_with_registered_anchor") is None,
                   {k: p13[k] for k in ("available", "source", "agrees_with_registered_anchor")})]

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== 聚合器自测（纯 CPU、只读）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}"
              + (f"  {str(extra)[:240]}" if (extra is not None and not ok) else ""))
    print(f"\n聚合器自测：PASS {npass} / FAIL {nfail}")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec_out = dict(tool="serial_oracle_aggregate_selftest_cpu", run_id=run_id,
                   timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   diagnostic_only=True, gpu_used=False, model_loaded=False,
                   semantics=("校验聚合器的**字段名契约**（硬检查引用的字段必须存在于 `_row` 输出），"
                              "并对 D6/D7 两个已发生的字段映射缺陷做显式回归；"
                              "使用**真实**运行产物（没有产物时如实报 FAIL，不假装通过）"),
                   real_artifact=(real.name if real else None),
                   row_keys=sorted(keys), npass=npass, nfail=nfail,
                   decision=("AGG_SELFTEST_PASS" if nfail == 0 else "AGG_SELFTEST_FAIL"),
                   checks=[dict(name=n, ok=bool(o), extra=(None if o else e)) for n, o, e in checks])
    p = OUTDIR / f"serial_oracle_aggregate_selftest_{run_id}.json"
    p.write_text(json.dumps(rec_out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[agg-selftest] RESULT_JSON={p}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
