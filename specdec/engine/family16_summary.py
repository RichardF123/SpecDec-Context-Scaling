"""family16_summary.py — 16 格清单的**汇总**（纯 CPU；只读 run JSON 与索引）。

设计要点（本轮要求）
  * **缺失运行不能生成"完整成功"汇总**：只要有一格缺记录，`diagnostic_completeness`
    就**不是** `complete`，且 `complete_success_claimable=False`；
  * 五类状态**分开**输出（execution / structural / repeatability / ar_spec_equality / formal）；
  * **没有**任何名为 pass/success 的总体字段；正式判据一律 `not_established`
    （单次/单模式/单 family 诊断结构上不可能评估冻结的 B6 门）；
  * 环境拒绝与算法失败**分开计数**，但两者都计入"停止推进"。

索引格式（由编排脚本写出；也可手工构造用于离线复核）
```json
{"tag": "...", "cells": [
  {"n":1,"family":"F06","kv":"base_bf16","mode":"ar","rep":1,
   "label":"F06_base_ar_r1","json":"<path or null>","rc":0}, ...]}
```
"""
from __future__ import annotations

import json
from pathlib import Path

from . import family_input as FI
from . import status_fields as SF


def _load(path) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        # `utf-8-sig`：编排脚本用 PowerShell 5.1 的 `Set-Content -Encoding UTF8` 写索引，
        # 那会带 BOM ⇒ 纯 `utf-8` 会 JSONDecodeError（首版就是这么炸的）。
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:                                            # noqa: BLE001
        return None


def _index_by_key(cells: list) -> dict:
    out = {}
    for c in cells:
        out[(c.get("family"), c.get("kv"), c.get("mode"), int(c.get("rep", 0)))] = c
    return out


def summarize(index: dict, *, expected_cells: list | None = None,
               on_divergence: str = "record") -> dict:
    """把索引 + 各 run JSON 汇总成**分开状态**的报告。**不产生"总体 PASS"。**"""
    exp = expected_cells if expected_cells is not None else FI.family_cells()
    given = _index_by_key(index.get("cells") or [])
    exp_keys = [(c["family"], c["kv"], c["mode"], c["rep"]) for c in exp]

    missing, extra = [], []
    recs: dict = {}
    cell_rows = []
    for c in exp:
        k = (c["family"], c["kv"], c["mode"], c["rep"])
        g = given.get(k)
        if g is None:
            missing.append(dict(cell=c, reason="索引里没有这一格"))
            cell_rows.append(dict(cell=c, present=False,
                                  execution_status=SF.execution_status(None)))
            recs[k] = None
            continue
        rec = _load(g.get("json"))
        recs[k] = rec
        if rec is None:
            missing.append(dict(cell=c, reason=f"索引指向的记录缺失/不可读：{g.get('json')}"))
        cell_rows.append(dict(
            cell=dict(c, **{kk: g.get(kk) for kk in ("label", "json", "rc", "run_id")
                            if g.get(kk) is not None}),
            present=rec is not None,
            execution_status=SF.execution_status(rec),
            structural_status=SF.structural_status(rec),
            failure_class=SF.failure_class(rec),
            gen_len=((rec or {}).get("run") or {}).get("gen_len"),
            stop_reason=((rec or {}).get("run") or {}).get("stop_reason"),
            input_source=((rec or {}).get("input") or {}).get("source_kind"),
            family_id=((rec or {}).get("input") or {}).get("family_id"),
        ))
    for k in given:
        if k not in exp_keys:
            extra.append(dict(key=list(k)))

    # ---- 逐 (family, kv) 的 AR/Spec 相等性；逐 (family, kv, mode) 的重复一致性 ----
    ar_spec, repeats = [], []
    for fam in sorted({c["family"] for c in exp}):
        for kv in sorted({c["kv"] for c in exp}):
            for rep in sorted({c["rep"] for c in exp}):
                a = recs.get((fam, kv, "ar", rep))
                s = recs.get((fam, kv, "spec", rep))
                r = SF.ar_spec_equality(a, s, on_divergence=on_divergence)
                r.update(family=fam, kv=kv, rep=rep,
                         pair_present=bool(a is not None and s is not None))
                ar_spec.append(r)
    for fam in sorted({c["family"] for c in exp}):
        for kv in sorted({c["kv"] for c in exp}):
            for mode in sorted({c["mode"] for c in exp}):
                r = SF.repeatability_status(recs.get((fam, kv, mode, 1)),
                                            recs.get((fam, kv, mode, 2)))
                r.update(family=fam, kv=kv, mode=mode)
                repeats.append(r)

    exec_counts: dict = {}
    struct_counts: dict = {}
    for row in cell_rows:
        st = row["execution_status"]["status"]
        exec_counts[st] = exec_counts.get(st, 0) + 1
        if row["present"]:
            ss = row["structural_status"]["status"]
            struct_counts[ss] = struct_counts.get(ss, 0) + 1

    hard_fail_kinds = sorted({row["execution_status"]["status"] for row in cell_rows
                              if row["execution_status"]["status"] in
                              (SF.EXEC_ENV_REFUSED, SF.EXEC_CRASHED, SF.EXEC_ALGORITHM_FAILED)})
    stops = sorted({row["cell"]["label"] for row in cell_rows
                    if row["execution_status"].get("stops_advance")})
    rep_diff = [r for r in repeats if r["status"] == "different"]
    struct_fail = [row["cell"]["label"] for row in cell_rows
                   if row["present"] and row["structural_status"]["status"] == "fail"]

    all_present = not missing and not extra
    all_exec_complete = all(r["execution_status"]["status"] == SF.EXEC_COMPLETE
                            for r in cell_rows)
    all_struct_ok = all((not r["present"]) or r["structural_status"]["status"] == "ok"
                        for r in cell_rows)
    all_repeat_identical = bool(repeats) and all(r["status"] == "identical" for r in repeats)
    all_pairs_assessable = all(r["pair_present"] for r in ar_spec)

    if not all_present:
        completeness = "incomplete"
    elif hard_fail_kinds:
        completeness = "complete_with_hard_failures"
    elif rep_diff:
        completeness = "complete_with_repeat_mismatch"
    else:
        completeness = "complete_diagnostics_only"

    complete_success_claimable = bool(
        all_present and all_exec_complete and all_repeat_identical and all_pairs_assessable
        and not struct_fail)

    return dict(
        tool="family16_summary", tag=index.get("tag"),
        cells_total=len(exp), cells_with_records=sum(1 for r in cell_rows if r["present"]),
        missing_cells=missing, extra_cells=extra,
        execution_status_counts=exec_counts,
        structural_status_counts=struct_counts,
        hard_failure_kinds=hard_fail_kinds,
        stops_advance_cells=stops,
        structural_fail_cells=struct_fail,
        repeatability=repeats, repeatability_mismatches=[r for r in rep_diff],
        ar_spec_equality=ar_spec,
        diagnostic_completeness=completeness,
        complete_success_claimable=complete_success_claimable,
        complete_success_meaning=("**仅表示「这 16 格诊断都拿到了、结构与重复自洽」，"
                                  "不表示任何正式正确性结论**"),
        formal_gate_status=SF.formal_gate_status(ar_rec=None, spec_rec=None),
        no_formal_pass=True,
        aggregate_note=("**没有**名为 pass/success 的总体结论字段；"
                        "缺失运行或重复不一致时 `diagnostic_completeness` 不会是 complete。"),
        cells=cell_rows,
    )


def load_index(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main() -> int:
    import os
    idx_path = os.environ.get("SUMMARY_INDEX")
    if not idx_path:
        print("!! 需要 SUMMARY_INDEX=<index json>（编排脚本会写出）；本轮不启动 GPU")
        return 2
    rag = os.environ.get("REPEAT_ON_DIVERGENCE", "record")
    out = summarize(load_index(idx_path), on_divergence=rag)
    dst = Path(idx_path).with_name(Path(idx_path).stem + ".summary.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cells {out['cells_with_records']}/{out['cells_total']}  "
          f"completeness={out['diagnostic_completeness']}  "
          f"claimable={out['complete_success_claimable']}  "
          f"formal={out['formal_gate_status']['status']}")
    if out["missing_cells"]:
        print(f"  missing={[m['cell']['label'] for m in out['missing_cells']]}")
    for m in out["repeatability_mismatches"]:
        print(f"  !! repeat mismatch {m['family']}/{m['kv']}/{m['mode']}")
    for s in out["stops_advance_cells"]:
        print(f"  !! stops advance: {s}")
    print(f"[summary] RESULT_JSON={dst}")
    # 退出码语义：0 = 诊断齐全且自洽（**仍不是正式正确性通过**）；3 = 报告已写出但诊断不齐全/不可宣告。
    # 刻意**不**用 0 表示"正式通过"——本工具永远不评正式门。
    return 0 if out["complete_success_claimable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
