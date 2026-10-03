"""stage_check.py — **逐阶段核对器**（受版本控制；参数化 tag / family / mode / 重复数）。

用于 Stage 2b 的"**AR×2 → 核对 → production×2 → 核对 → serial×2 → 核对**"流程：
每个阶段跑完即调用一次，确认该阶段"数量齐、可重复、结构正常"，再进入下一阶段。

```powershell
# AR 阶段（新 family：自一致 + manifest 输入 hash）
python -X utf8 -m specdec.engine.stage_check --tag div8209_p2 --family F09 --modes ar --reps 2
# production 阶段（结构 + 确定性 + 与 AR 的分歧记账）
python -X utf8 -m specdec.engine.stage_check --tag div8209_p2 --family F09 --modes prod --reps 2
# serial 阶段（18 项硬检查 + 与 AR 的 token 等价）
python -X utf8 -m specdec.engine.stage_check --tag div8209_p2 --family F09 --modes serial --reps 2
```

退出码：`0` = 请求的模式/重复**全部到位且一致**；`2` = **尚有阶段未跑**（PENDING，属正常中间态）；
`1` = 存在**不一致或缺字段**（必须停下来查）。

**只读**：不改产物、不加载模型、不用 GPU（读 manifest 与 JSON 除外）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import serial_oracle_aggregate as AGG
from . import stage2b_verdict as SV


def _show(fam: str, mode: str, rows: list, *, reps: int, ar_ref, ar_stop) -> dict:
    print(f"== {fam} / {mode}（期望 {reps} 次）")
    if not rows:
        print("   （无产物 ⇒ PENDING）")
        return dict(status="PENDING", n=0)
    seen, dup = set(), []
    for r in rows:
        rep = int(r.get("rep") or 0)
        if rep in seen:
            dup.append(rep)
        seen.add(rep)
        div = AGG._first_div(r["gen"], ar_ref) if ar_ref is not None else None
        print(f"   rep{rep}: len={r['gen_len']} stop={r['stop_reason']} hash={r['gen_hash']} "
              f"harness={r.get('end_state_gen_hash')} agree={r['gen_hash'] == r.get('end_state_gen_hash')} "
              f"m1={r.get('target_all_m1')} seq_lens={r.get('target_gen_seq_lens')} "
              f"acct={(r.get('accounting') or {}).get('matches')} "
              f"cache_struct={r.get('cache_structure_ok')}/{r.get('cache_structure_n_checks')} "
              f"finite={r.get('finiteness_n_checked')} rounds={r.get('n_rounds')}/"
              f"trace={r.get('serial_trace_len')} stage={r.get('stage')} "
              f"rc_zero={r.get('smoke_rc_zero')}")
        print(f"      verify_mode before={r.get('verify_mode')} at_end={r.get('verify_mode_at_run_end')} "
              f"after_cleanup={r.get('verify_mode_after')} wrap_active={r.get('row_aligned_wrap_active')}"
              + (f" | 相对 AR 首分歧={div}" if ar_ref is not None else ""))
    missing = [i for i in range(1, int(reps) + 1) if i not in seen]
    extra = sorted(x for x in seen if x > int(reps) or x <= 0)
    gens = [tuple(r["gen"]) for r in rows]
    identical = len(set(gens)) == 1
    print(f"   → 次数={len(rows)}/{reps} 缺={missing or '无'} 超范围={extra or '无'} "
          f"重复编号冲突={dup or '无'} 完整 token IDs 一致={identical}")
    status = ("PENDING" if missing and not dup else
              ("INCONSISTENT" if (dup or extra or not identical) else "OK"))
    return dict(status=status, n=len(rows), missing=missing, extra=extra, dup=dup, identical=identical)


def main() -> int:
    ap = argparse.ArgumentParser(description="逐阶段核对（tag/family/mode/reps 全部显式）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--modes", default="ar,prod,serial")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--outdir", default=str(SV.OUTDIR))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    rows = [SV._row(r) for r in SV.load_runs(a.tag, outdir)
            if not (r.get("blocked_by") == "environment" or r.get("model_run_happened") is False)]
    fam = a.family
    fam_rows = [r for r in rows if r.get("case") == fam]
    print(f"== stage_check ==  tag={a.tag} family={fam} modes={a.modes} reps={a.reps}")
    print(f"   该 tag 内共读到 {len(rows)} 条可用产物；其中本 family {len(fam_rows)} 条；"
          f"tag 内其它 family={sorted({r.get('case') for r in rows if r.get('case') != fam})}")
    if not fam_rows:
        print("   ★ 该 family 在本 tag 内**没有任何可用产物** ⇒ 不要继续下一阶段")
        return 2

    ar = sorted([r for r in fam_rows if r.get("mode") == "ar"], key=lambda r: int(r.get("rep") or 0))
    ar_ref = ar[0]["gen"] if ar else None
    ar_stop = ar[0]["stop_reason"] if ar else None

    problems: list = []
    pending = False
    results: dict = {}
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        mrows = sorted([r for r in fam_rows if r.get("mode") == mode], key=lambda r: int(r.get("rep") or 0))
        res = _show(fam, mode, mrows, reps=a.reps, ar_ref=ar_ref, ar_stop=ar_stop)
        results[mode] = res
        if res["status"] == "PENDING":
            pending = True
        elif res["status"] == "INCONSISTENT":
            problems.append(f"{mode}: 缺={res['missing']} 超范围={res['extra']} 冲突={res['dup']} "
                            f"一致={res['identical']}")

    # --- AR 基线（区分"有登记锚点"与"新 family"）---
    if ar:
        if fam in SV.KNOWN_ANCHOR_FAMILIES:
            base = SV._ar_baseline_known(fam, ar)
            print(f"== AR 基线（有登记锚点）：ok={base['ok']}")
            for k in ("reps_identical", "equals_registered_anchor", "self_reported_hash_agrees",
                      "equals_prior_tokens", "equals_prior_hash"):
                if k in base["checks"]:
                    print(f"   {k} = {base['checks'][k]}")
        else:
            base = SV._ar_baseline_new(fam, ar, reps=a.reps, ctx=8192)
            print(f"== AR 基线（新 family：自一致 + manifest 输入 hash）：ok={base['ok']}")
            for k, v in base["checks"].items():
                if k not in ("observed_hashes",):
                    print(f"   {k} = {v}")
        if not base["ok"]:
            if base.get("missing_fields"):
                problems.append(f"AR 基线无法评估（缺字段 {base['missing_fields']}）")
            else:
                problems.append(f"AR 基线不成立（{base['checks']}）")

    if problems:
        for p in problems:
            print(f"   [问题] {p}")
        print("   ★ 结论 = 必须停下核对（STOP）")
        return 1
    if pending:
        print("   ★ 结论 = 尚有阶段未跑（PENDING，可继续下一阶段）")
        return 2
    print("   ★ 结论 = 本阶段核对通过（OK）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
