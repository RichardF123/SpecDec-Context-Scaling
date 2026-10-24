"""experiments/confirmatory/analyze_pilot.py — confirmatory pilot 的离线分析（CPU-only）。

用法：
    python -m experiments.confirmatory.analyze_pilot <batch_dir> [--out <batch_dir>/analysis]

输出：analysis/{pilot_tasks.csv, pilot_family_timing.csv, pilot_block_shifts.csv,
       pilot_order_ab_ba.csv, pilot_pair_summary.csv, pilot_context_checks.csv,
       summary.json, pilot_stability.{svg,png,pdf}}

六项计时检查沿用旧正式审计定义（20% 诊断阈值）：
* within：单任务内 throughput 极差/中位 > 20% ⇒ CHECK；
* between：该 (family,ctx,mode) 两 block 的 throughput 中位差/max > 20% ⇒ CHECK；
* boundary：所有 rep 的 boundary clock qualified 且证据 COMPLETE。
all-six-OK = AR 与 SpecDec 各三项同时 OK。
8K：若 pair 六项全 OK ⇒ 允许 family-level T_AR/T_Spec；否则保留绝对量并标 restricted。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV_FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)
MODES = ("AR", "SpecDec")
CHECK_SPREAD = 0.20


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # 分析产物是可再生的派生文件（原始记录仍是独占写入），允许覆盖
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    data = sorted(float(v) for v in values)
    med = st.median(data)
    absdev = [abs(v - med) for v in data]
    return dict(n=len(data), median=med, min=data[0], max=data[-1],
                mad=st.median(absdev), iqr=data[-1] - data[0],
                spread_pct=(data[-1] - data[0]) / med * 100 if med else None)


def load_batch(batch: Path):
    plan = json.loads((batch / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    tasks = {t["planned_order"]: t for t in plan["tasks"]}
    data = {}
    for entry in summary["completed"]:
        meta = json.loads((Path(entry["task_dir"]) / "task_meta.json").read_text(encoding="utf-8"))
        results_dir = Path(entry["task_dir"]) / "worker_results"
        records = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(results_dir.glob("*.json"))]
        records = [r for r in records if r.get("run_kind") == "measured"]
        data[meta["planned_order"]] = dict(meta=meta, records=records,
                                           order=meta["planned_order"],
                                           actual_start_utc=entry.get("actual_start_utc"),
                                           actual_end_utc=entry.get("actual_end_utc"))
    return plan, summary, tasks, data


def validate_task(meta, records, pilot_manifest) -> dict:
    family, ctx, mode = meta["family_id"], meta["context_tokens"], meta["mode"]
    if len(records) != meta["measured_reps"]:
        raise AssertionError(f"{meta['planned_order']} 记录数 {len(records)} != {meta['measured_reps']}")
    expected_hash = pilot_manifest[family]["prefixes"][str(ctx)]["token_sha1"]
    evidence_complete = True
    for record in records:
        if record["run_kind"] != "measured":
            raise AssertionError("混入非 measured 记录")
        if not record["input"].get("hash_verified"):
            raise AssertionError("输入 hash 未通过")
        if record["input"]["token_sha1"] != expected_hash:
            raise AssertionError(f"{family}_{ctx} 输入 hash 与 manifest 不一致")
        if not record.get("same_mode_diagnostic_tokens_equal"):
            raise AssertionError("同模式诊断与计时不一致")
        audit = record["runtime_environment_audit"]
        if audit["evidence_status"] != "COMPLETE":
            evidence_complete = False
        if mode == "SpecDec":
            s = record["spec_stats"]
            if s["accepted"] + s["mismatch_rounds"] != record["generated_length"]:
                raise AssertionError("spec 输出对账失败")
            if not (s["rounds"] == s["target_block_calls"] == s["mismatch_rounds"] + s["all_accept_rounds"]):
                raise AssertionError("round 对账失败")
            if s["draft_calls"] != s["proposed"]:
                raise AssertionError("draft 调用对账失败")
    rates = [r["decode_tokens_per_s"] for r in records]
    spread = (max(rates) - min(rates)) / st.median(rates)
    boundary_ok = all(r["runtime_environment_audit"]["boundary_clock_qualified"] is True
                      for r in records)
    return dict(family=family, ctx=ctx, mode=mode,
                block=meta["measurement_block"], reps=len(records),
                planned_order=meta["planned_order"],
                pair_id=meta["pair_id"], pair_order=meta["pair_order"],
                pair_position=meta["pair_position"],
                actual_start_utc=meta.get("actual_start_utc"),
                decode_s=st.median(r["decode_s"] for r in records),
                decode_min=min(r["decode_s"] for r in records),
                decode_max=max(r["decode_s"] for r in records),
                prefill_s=st.median(r["prefill_s"] for r in records),
                total_s=st.median(r["total_s"] for r in records),
                throughput=st.median(rates), throughput_spread_pct=spread * 100,
                within_status="CHECK" if spread > CHECK_SPREAD else "OK",
                boundary_status="OK" if (boundary_ok and evidence_complete) else "CHECK",
                boundary_evidence_complete=evidence_complete,
                generated_ids=records[0]["generated_ids"],
                spec_stats=(records[0].get("spec_stats") or None),
                peak_allocated_gib=max(r["peak_allocated_bytes"] for r in records) / 2**30,
                peak_reserved_gib=max(r["peak_reserved_bytes"] for r in records) / 2**30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path,
                        default=ROOT / "docs/prompt_families_manifest.json")
    args = parser.parse_args()
    batch = Path(args.batch)
    out = args.out or batch / "analysis"
    pilots_doc = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    pilots = {entry["family_id"]: entry for entry in pilots_doc["families"]}

    plan, summary, tasks, data = load_batch(batch)
    task_rows = [validate_task(item["meta"], item["records"], pilots)
                 for item in data.values()]
    task_rows.sort(key=lambda r: (r["ctx"], r["mode"], r["family"]))

    # unit: (family, ctx, mode) 汇合两个 block 的 5 次 measured
    unit = {}
    for row in task_rows:
        unit.setdefault((row["family"], row["ctx"], row["mode"]), []).append(row)
    unit_rows = []
    for (family, ctx, mode), rows in sorted(unit.items()):
        blocks = {r["block"] for r in rows}
        if blocks != {0, 1} or len(rows) != 2:
            # 缺块：不静默补数，按不完整记录（between/boundary CHECK）
            medians = [row["throughput"] for row in rows]
            bb = (max(medians) - min(medians)) / max(medians) if len(medians) > 1 else None
            unit_rows.append(dict(
                family=family, ctx=ctx, mode=mode, n=sum(r["reps"] for r in rows),
                decode_s=st.median(r["decode_s"] for r in rows),
                decode_min=min(r["decode_min"] for r in rows),
                decode_max=max(r["decode_max"] for r in rows),
                prefill_s=st.median(r["prefill_s"] for r in rows),
                total_s=st.median(r["total_s"] for r in rows),
                within_status="CHECK" if any(r["within_status"] == "CHECK" for r in rows) else "OK",
                between_status="CHECK",
                between_difference_pct=None if bb is None else bb * 100,
                boundary_status="CHECK",
                boundary_evidence_complete=False,
                spec_stats=(rows[0]["spec_stats"] or None),
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
                block0_decode_s=next((row["decode_s"] for row in rows if row["block"] == 0), None),
                block1_decode_s=next((row["decode_s"] for row in rows if row["block"] == 1), None),
            ))
            continue
        medians = [row["throughput"] for row in rows]
        bb = (max(medians) - min(medians)) / max(medians)
        unit_rows.append(dict(
            family=family, ctx=ctx, mode=mode, n=sum(r["reps"] for r in rows),
            decode_s=st.median(r["decode_s"] for r in rows),
            decode_min=min(r["decode_min"] for r in rows),
            decode_max=max(r["decode_max"] for r in rows),
            prefill_s=st.median(r["prefill_s"] for r in rows),
            total_s=st.median(r["total_s"] for r in rows),
            within_status="CHECK" if any(r["within_status"] == "CHECK" for r in rows) else "OK",
            between_status="CHECK" if bb > CHECK_SPREAD else "OK",
            between_difference_pct=bb * 100,
            boundary_status="OK" if all(r["boundary_status"] == "OK" for r in rows) else "CHECK",
            boundary_evidence_complete=all(r["boundary_evidence_complete"] for r in rows),
            spec_stats=(rows[0]["spec_stats"] or None),
            peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
            peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
            block0_decode_s=next(row["decode_s"] for row in rows if row["block"] == 0),
            block1_decode_s=next(row["decode_s"] for row in rows if row["block"] == 1),
        ))
    unit_map = {(r["family"], r["ctx"], r["mode"]): r for r in unit_rows}

    def pick_ids(family, ctx, mode):
        row = next((t for t in task_rows
                    if t["family"] == family and t["ctx"] == ctx and t["mode"] == mode), None)
        return None if row is None else row["generated_ids"]

    pair_rows = []
    for family in DEV_FAMILIES:
        for ctx in CONTEXTS:
            ar = unit_map[(family, ctx, "AR")]
            sp = unit_map[(family, ctx, "SpecDec")]
            a_ids = pick_ids(family, ctx, "AR") or []
            s_ids = pick_ids(family, ctx, "SpecDec") or []
            same = bool(a_ids) and bool(s_ids) and a_ids == s_ids
            prefix = next((i for i, (x, y) in enumerate(zip(a_ids, s_ids)) if x != y),
                          min(len(a_ids), len(s_ids)))
            six_ok = all(ar[k] == "OK" for k in ("within_status", "between_status", "boundary_status")) \
                and all(sp[k] == "OK" for k in ("within_status", "between_status", "boundary_status"))
            ratio = (ar["decode_s"] / sp["decode_s"]) if sp["decode_s"] else None
            restricted = ctx == 8192 and not six_ok
            pair_rows.append(dict(
                family=family, ctx=ctx,
                ar_decode_s=ar["decode_s"], spec_decode_s=sp["decode_s"],
                decode_ratio=ratio,
                ratio_suppressed=bool(restricted),
                same_output=bool(same), common_prefix=prefix,
                timing_six_checks_ok=bool(six_ok),
                claim_status=("8K_RESTRICTED_NO_RATIO" if restricted else "DESCRIPTIVE_PILOT"),
                spec_acceptance=sp_acceptance(sp),
                spec_rounds=sp_rounds(sp), spec_mismatch=sp_mismatch(sp),
                spec_target_verify_calls=sp_verify(sp), spec_target_repair_calls=sp_repair(sp),
                ar_peak_allocated_gib=ar["peak_allocated_gib"],
                spec_peak_allocated_gib=sp["peak_allocated_gib"]))

    context_rows = []
    for ctx in CONTEXTS:
        subset = [r for r in pair_rows if r["ctx"] == ctx]
        if len(subset) != len(DEV_FAMILIES):
            raise AssertionError(f"ctx={ctx} 的 pair 数 {len(subset)} != {len(DEV_FAMILIES)}")
        context_rows.append(dict(
            ctx=ctx, families=len(subset),
            all_six_checks_ok=sum(r["timing_six_checks_ok"] for r in subset),
            same_output=sum(r["same_output"] for r in subset),
            spec_slower_families=sum(r["decode_ratio"] < 1 for r in subset),
            ratio_median=st.median(r["decode_ratio"] for r in subset),
            ar_decode_median_s=st.median(r["ar_decode_s"] for r in subset),
            spec_decode_median_s=st.median(r["spec_decode_s"] for r in subset)))

    # block shift
    shift_rows = []
    for row in unit_rows:
        b0, b1 = row["block0_decode_s"], row["block1_decode_s"]
        shift_rows.append(dict(family=row["family"], ctx=row["ctx"], mode=row["mode"],
                               block0_decode_s=b0, block1_decode_s=b1,
                               block1_vs_block0_pct=(100 * (b1 / b0 - 1)
                                                     if b0 and b1 else None)))
    shift_summary = []
    for ctx in CONTEXTS:
        for mode in MODES:
            vals = [r["block1_vs_block0_pct"] for r in shift_rows
                    if r["ctx"] == ctx and r["mode"] == mode
                    and r["block1_vs_block0_pct"] is not None]
            shift_summary.append(dict(ctx=ctx, mode=mode, families=len(vals),
                                      block1_slower=sum(v > 0 for v in vals),
                                      median_pct=(st.median(vals) if vals else None),
                                      min_pct=(min(vals) if vals else None),
                                      max_pct=(max(vals) if vals else None)))

    # AB/BA order comparison（pair_position 0 = 配对内先执行）
    order_rows = []
    for ctx in CONTEXTS:
        for mode in MODES:
            entries = [t for t in task_rows if t["ctx"] == ctx and t["mode"] == mode]
            first = [t["decode_s"] for t in entries if t["pair_position"] == 0]
            second = [t["decode_s"] for t in entries if t["pair_position"] == 1]
            if not first or not second:
                raise AssertionError("order 样本不足")
            med_first, med_second = st.median(first), st.median(second)
            order_rows.append(dict(ctx=ctx, mode=mode,
                                   n_first=len(first), n_second=len(second),
                                   median_first_s=med_first, median_second_s=med_second,
                                   first_vs_second_pct=100 * (med_first / med_second - 1)))

    # 8K 资源与边界
    eight = [r for r in pair_rows if r["ctx"] == 8192]
    eight_ok = all(r["timing_six_checks_ok"] for r in eight)

    # 裁决（预注册 §10）
    n_tasks = len(task_rows)
    n_measured = sum(r["reps"] for r in task_rows)
    boundary_all_complete = all(
        record["runtime_environment_audit"]["evidence_status"] == "COMPLETE"
        for item in data.values() for record in item["records"])
    context_ok = {row["ctx"]: row["all_six_checks_ok"] for row in context_rows}
    order_same_direction_cells = max(
        sum(r["first_vs_second_pct"] > 20 for r in order_rows),
        sum(r["first_vs_second_pct"] < -20 for r in order_rows))
    ready = (n_tasks == 48 and n_measured == 120 and boundary_all_complete
             and context_ok[4096] >= 1 and context_ok[2048] >= 1
             and context_ok[8192] >= 2 and order_same_direction_cells < 3)
    if n_tasks < 48 or n_measured < 120:
        verdict = "ENVIRONMENT_BLOCKED"
    elif not boundary_all_complete:
        verdict = "NEEDS_MEASUREMENT_FIX"
    elif context_ok[4096] == 0 or order_same_direction_cells >= 3:
        verdict = "NEEDS_MEASUREMENT_FIX"
    elif ready:
        verdict = "READY_FOR_CONFIRMATORY_FULL"
    else:
        verdict = "NEEDS_MEASUREMENT_FIX"

    summary_payload = dict(
        schema_version="confirmatory-pilot-analysis-v1",
        batch=batch.name, n_tasks=n_tasks, n_measured=n_measured,
        contexts=context_rows, pairs=pair_rows, block_shifts=shift_rows,
        block_shift_summary=shift_summary, order_ab_ba=order_rows,
        eight_k=dict(six_checks_ok_families=sum(r["timing_six_checks_ok"] for r in eight),
                     total_families=len(eight), stable=eight_ok,
                     note=("8K 六项检查全 OK 时允许 family-level T_AR/T_Spec；否则 restricted"),
        ),
        boundary_all_complete=boundary_all_complete,
        verdict=verdict,
        verdict_rule="docs/confirmatory_measurement_preregistration_2026-10-24.md §10",
        note=("pilot 只用于流程裁决；不作显著性检验；不因 pilot 结果筛选正式 family；"
              "旧正式批次数据未改动"),
    )
    out.mkdir(exist_ok=True)
    write_csv(out / "pilot_tasks.csv",
              [{k: v for k, v in row.items() if k != "generated_ids"} for row in task_rows])
    write_csv(out / "pilot_family_timing.csv", unit_rows)
    write_csv(out / "pilot_block_shifts.csv", shift_rows)
    write_csv(out / "pilot_order_ab_ba.csv", order_rows)
    write_csv(out / "pilot_pair_summary.csv", pair_rows)
    write_csv(out / "pilot_context_checks.csv", context_rows)
    (out / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")

    plot_stability(pair_rows, shift_rows, out / "pilot_stability")

    print(json.dumps(dict(n_tasks=n_tasks, n_measured=n_measured,
                          contexts=context_rows, verdict=verdict),
                     ensure_ascii=False, indent=2))
    print("block shift summary:", shift_summary)
    print("AB/BA order:", order_rows)
    return 0


def sp_acceptance(sp):
    s = sp.get("spec_stats") or {}
    return s.get("accepted", 0) / s.get("proposed", 1) if s else None


def sp_rounds(sp):
    s = sp.get("spec_stats") or {}
    return s.get("rounds")


def sp_mismatch(sp):
    s = sp.get("spec_stats") or {}
    return s.get("mismatch_rounds")


def sp_verify(sp):
    s = sp.get("spec_stats") or {}
    return s.get("target_block_calls")


def sp_repair(sp):
    s = sp.get("spec_stats") or {}
    return s.get("mismatch_rounds")


def plot_stability(pairs, shifts, stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    colors = {"AR": "#0072B2", "SpecDec": "#D55E00"}
    ax = axes[0]
    for ctx in CONTEXTS:
        for mode, offset in (("AR", -0.12), ("SpecDec", 0.12)):
            entries = [r for r in pairs if r["ctx"] == ctx]
            vals = [r["ar_decode_s"] if mode == "AR" else r["spec_decode_s"] for r in entries]
            x = [list(CONTEXTS).index(ctx) + offset + (i - 1.5) * 0.03 for i in range(len(vals))]
            ax.scatter(x, vals, s=14, color=colors[mode], alpha=0.8, linewidths=0,
                       label=mode if ctx == 2048 else None)
            ax.plot([list(CONTEXTS).index(ctx) + offset - 0.09,
                     list(CONTEXTS).index(ctx) + offset + 0.09],
                    [np.median(vals), np.median(vals)], color=colors[mode], linewidth=1.4)
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylabel("decode latency (s, 5 measured / family)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.text(-0.10, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    positions, values, cols, labels = [], [], [], []
    pos = 1
    for ctx in CONTEXTS:
        for mode in MODES:
            vals = [r["block1_vs_block0_pct"] for r in shifts
                    if r["ctx"] == ctx and r["mode"] == mode
                    and r["block1_vs_block0_pct"] is not None]
            positions.append(pos)
            values.append(vals)
            cols.append(colors[mode])
            labels.append(f"{ctx // 1024}K\n{mode.replace('SpecDec', 'Spec')}")
            pos += 1
        pos += 0.35
    keep = [(p, v, c, l) for p, v, c, l in zip(positions, values, cols, labels) if v]
    bp = ax.boxplot([v for _, v, _, _ in keep], positions=[p for p, _, _, _ in keep],
                    widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.0})
    for patch, (_, _, color, _) in zip(bp["boxes"], keep):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for p, vals, color, _ in keep:
        ax.scatter(p + rng.uniform(-0.14, 0.14, len(vals)), vals, s=13,
                   color=color, linewidths=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(20, color="#666666", linewidth=0.7, linestyle="--")
    ax.axhline(-20, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_xticks([p for p, _, _, _ in keep], [l for _, _, _, l in keep])
    ax.set_ylabel("block1 vs block0 decode median (%)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
