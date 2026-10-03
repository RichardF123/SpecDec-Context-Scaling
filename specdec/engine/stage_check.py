"""stage_check.py — **逐阶段核对器**（受版本控制；参数化 tag / family / mode / 重复数；**真跑硬门**）。

用于 Stage 2b 的"**AR×2 → 核对 → production×2 → 核对 → serial×2 → 核对**"流程。
每一步都**实际执行**该阶段的门槛检查，而不是只打印字段后返回 OK：

| 阶段 | 实际执行的门 |
| --- | --- |
| `ar` | AR 基线：有登记锚点的 family 用「登记输出 hash + 上一轮产物逐位对照」；**新 family** 用「两次自一致 + 自报 hash/长度/停止一致 + manifest 输入 `token_sha1`」；并要求 `stage == complete` |
| `prod` | finite · cache 结构 · 逐轮不变量 · 合法 token/停止 · **确定性重复** · 几何上**出现 `s>1`** 的 batched 前向 · 无 OOM |
| `serial` | **18 项硬检查**（含 `s==1`、0 次 `s>1`、审计非空、记账恒等、row-aligned 关闭、**D6 清理后实测 `block`**）· `stage == complete` · `self_checks_all_ok` · **与同 family AR 的完整 token equality** · 两次一致 |

退出码：`0` = 请求的模式/重复**全部到位且全部硬门通过**；`2` = 尚有阶段未跑（PENDING，正常中间态）；
`1` = **存在失败或缺字段 ⇒ STOP**（必须停下来查；不得继续下一阶段）。

**历史记录处理**：旧 F13 的 serial 产物产生于 D6 修正之前（`verify_mode_at_run_end` 缺失），
因此**无法**满足本轮 D6 门 ⇒ 本工具会如实 STOP 并说明"**该运行不是本轮合格样本**"；
**不改写**其历史结论（`SERIAL_ORACLE_STAGE1_PASS` 依旧成立，但它不是 Stage 2b 的证据）。

**只读**：不改产物、不加载模型、不用 GPU（读 manifest 与 JSON 除外）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import serial_oracle_aggregate as AGG
from . import stage2b_verdict as SV


def _gate_ar(fam: str, rows: list, *, reps: int, ar_ref=None) -> list:
    reasons: list = []
    base = (SV._ar_baseline_known(fam, rows) if fam in SV.KNOWN_ANCHOR_FAMILIES
            else SV._ar_baseline_new(fam, rows, reps=int(reps), ctx=8192))
    if not base["ok"]:
        reasons.append(f"AR 基线不成立：{ {k: v for k, v in base['checks'].items() if v is not True} }")
    for r in rows:
        if r.get("stage") != "complete":
            reasons.append(f"rep{r.get('rep')}: stage={r.get('stage')}（不是 clean complete）")
    return reasons


def _gate_prod(fam: str, rows: list, *, reps: int, ar_ref=None):
    """production 阶段硬门。

    返回 `(reasons, notices)`：**分歧不是失败**（历史五个 family 的 production 都与 AR 分歧，
    那是被记录的现象而不是缺陷），所以它只作为"待办提示"输出，**不阻断 serial 阶段**
    （serial 的 token 等价才是核心证据，不能因为分歧就不跑）。
    """
    p = SV._production_checks(fam, rows, reps=int(reps), ar_ref=ar_ref)
    reasons = [f"production 硬门失败：{p['hard_failures']}"] if p["hard_failures"] else []
    notices: list = []
    if p.get("divergence_class") == "NUMERICAL_DIVERGENCE_RECORDED":
        notices.append("production 与 AR 存在分歧（已记录为 NUMERICAL_DIVERGENCE_RECORDED，**不判失败**）"
                       "⇒ 判决前必须补 GB6 独立诊断 pass 并链接（`stage2b_verdict --gb6 …`）；"
                       "serial 阶段照常进行")
    return reasons, notices


def _gate_serial(fam: str, rows: list, *, reps: int, ar_ref=None) -> list:
    reasons: list = []
    for r in rows:
        h = AGG._serial_hard_checks2(r, require_d6=True)
        for k, v in h.items():
            if v["ok"] is False:
                reasons.append(f"rep{r.get('rep')}: `{k}` 失败 —— {str(v['detail'])[:90]}")
            elif v["ok"] is None and not v.get("na"):
                reasons.append(f"rep{r.get('rep')}: `{k}` 无法评估（缺字段）")
        rc = r.get("_rounds") or {}
        if not rc.get("ok"):
            reasons.append(f"rep{r.get('rep')}: 逐轮字段不完整 {rc.get('missing')}")
        tc = r.get("_trace") or {}
        if tc.get("applicable") and not tc.get("ok"):
            reasons.append(f"rep{r.get('rep')}: serial trace 与轮记录不一致 {tc.get('bad')}")
        if r.get("stage") != "complete":
            reasons.append(f"rep{r.get('rep')}: stage={r.get('stage')}（不是 clean complete）")
        if r.get("self_checks_all_ok") is not True:
            reasons.append(f"rep{r.get('rep')}: self_checks_all_ok != True")
    if ar_ref is None:
        reasons.append("无可用的同 family AR 参照 ⇒ 无法核对 token equality")
    else:
        for r in rows:
            if r["gen"] != ar_ref:
                reasons.append(f"rep{r.get('rep')}: 与同 family AR 的**完整 token IDs 不一致**"
                               f"（首分歧 index={AGG._first_div(r['gen'], ar_ref)}）")
    if len(rows) >= 2 and len({tuple(r["gen"]) for r in rows}) != 1:
        reasons.append("两次 serial 的完整 token IDs 不一致")
    return reasons


def _show(mode: str, rows: list, *, reps: int, ar_ref, reasons: list, notices: list | None = None) -> None:
    print(f"   [{mode}] {len(rows)}/{reps} 次；硬门={'通过' if not reasons else '**失败**'}")
    for r in rows:
        div = AGG._first_div(r["gen"], ar_ref) if ar_ref is not None else None
        print(f"      rep{r.get('rep')}: len={r.get('gen_len')} stop={r.get('stop_reason')} "
              f"hash={r.get('gen_hash')} m1={r.get('target_all_m1')} "
              f"seq_lens={r.get('target_gen_seq_lens')} "
              f"cache={r.get('cache_structure_ok')}/{r.get('cache_structure_n_checks')} "
              f"finite={r.get('finiteness_n_checked')} rounds={r.get('n_rounds')}/"
              f"trace={r.get('serial_trace_len')} stage={r.get('stage')} "
              f"verify_at_end={r.get('verify_mode_at_run_end')} "
              f"after_cleanup={r.get('verify_mode_after')}"
              + (f" 首分歧={div}" if div is not None else ""))
    for x in reasons:
        print(f"      [失败] {x}")
    for x in (notices or []):
        print(f"      [待办·不阻断] {x}")


def main() -> int:
    ap = argparse.ArgumentParser(description="逐阶段核对（tag/family/mode/reps 全部显式；真跑硬门）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--modes", default="ar,prod,serial")
    ap.add_argument("--reps", type=int, default=SV.PREREGISTERED_REPS)
    ap.add_argument("--outdir", default=str(SV.OUTDIR))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    errs = ([f"family 形状非法：{a.family!r}"] if not a.family.strip().startswith("F") else [])
    if not modes:
        errs.append("模式集合为空")
    if int(a.reps) != SV.PREREGISTERED_REPS:
        errs.append(f"reps 必须等于预登记值 {SV.PREREGISTERED_REPS}（实得 {a.reps}）")
    if errs:
        for e in errs:
            print(f"   [非法] {e}")
        # ⚠️ 与 STOP(1)/PENDING(2) **区分开**：参数写错不能被脚本误当成"该阶段还没跑"而继续。
        print("   ★ 结论 = INVALID_ARGUMENTS（退出码 3；未评估任何阶段）")
        return 3

    fam = a.family.strip()
    rows_all = [SV.attach_internal(r) for r in SV.load_runs(a.tag, outdir)
                if not (r.get("blocked_by") == "environment" or r.get("model_run_happened") is False)]
    fam_rows = [r for r in rows_all if r.get("case") == fam]
    print(f"== stage_check ==  tag={a.tag} family={fam} modes={modes} reps={a.reps}")
    print(f"   该 tag 内可用产物 {len(rows_all)} 条；本 family {len(fam_rows)} 条；"
          f"其它 family={sorted({r.get('case') for r in rows_all if r.get('case') != fam})}")
    if not fam_rows:
        print("   ★ 该 family 在本 tag 内**没有任何可用产物** ⇒ PENDING（不要继续下一阶段）")
        return 2

    ar = sorted([r for r in fam_rows if r.get("mode") == "ar"], key=lambda r: int(r.get("rep") or 0))
    ar_ref = ar[0]["gen"] if ar else None
    # ⚠️ `_serial_hard_checks2` 的第 15 项要比较 `_ar_stop`；不设置会对 serial 行产生**假失败**
    for r in fam_rows:
        if r.get("mode") == "serial":
            r["_ar_stop"] = (ar[0]["stop_reason"] if ar else None)

    problems: list = []
    pending = False
    for mode in modes:
        mrows = sorted([r for r in fam_rows if r.get("mode") == mode], key=lambda r: int(r.get("rep") or 0))
        if not mrows:
            print(f"   [{mode}] 0/{a.reps} 次 ⇒ **PENDING**（该阶段未跑）")
            pending = True
            continue
        if mode == "ar":
            reasons, notices = _gate_ar(fam, mrows, reps=a.reps, ar_ref=ar_ref), []
        elif mode == "prod":
            reasons, notices = _gate_prod(fam, mrows, reps=a.reps, ar_ref=ar_ref)
        elif mode == "serial":
            reasons, notices = _gate_serial(fam, mrows, reps=a.reps, ar_ref=ar_ref), []
        else:
            reasons, notices = [f"未知模式 {mode!r}"], []
        _show(mode, mrows, reps=a.reps, ar_ref=ar_ref, reasons=reasons, notices=notices)
        if len(mrows) < int(a.reps):
            reasons = reasons + [f"次数不足：{len(mrows)} < {a.reps}"]
        for x in reasons:
            problems.append(f"{mode}: {x}")

    if problems:
        print(f"   ★ 结论 = STOP（{len(problems)} 项失败/缺字段）—— 必须停下核对，不得继续下一阶段")
        for p in problems:
            print(f"      · {p}")
        return 1
    if pending:
        print("   ★ 结论 = PENDING（已跑阶段全部通过；仍有阶段未跑，可继续）")
        return 2
    print("   ★ 结论 = OK（请求的模式/重复全部到位且全部硬门通过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
