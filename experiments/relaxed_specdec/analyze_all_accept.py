"""Perfect-Accept Latency Floor 分析。

输入：
  <run>/runs.jsonl（本批次：12 AR + 12 all_accept_oracle）
  [--stage-b output/relaxed_specdec/stage_b]（strict / δ=0.5 / δ=1.0 参照；跨批次，明确标注）
  [--phase-o-csv docs/measurements/ot_execution_conditions_2026-10-25.csv]（Phase O optimized 参照；跨批次）
输出：
  <run>/analysis/{core_table.csv, family_table.csv, all_accept_summary.json}
  docs/figures/all_accept_oracle/latency_floor.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def med(rows, fn):
    return st.median(fn(r) for r in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path,
                        default=ROOT / "output" / "relaxed_specdec" / "stage_b")
    parser.add_argument("--phase-o-csv", type=Path,
                        default=ROOT / "docs" / "measurements" / "ot_execution_conditions_2026-10-25.csv")
    args = parser.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    rows = load_runs(run / "runs.jsonl")
    by = {}
    for r in rows:
        by.setdefault((r["role"], r["family"], r["context"]), []).append(r)

    # 本批次：AR / oracle 的 family 级 decode 中位
    ar_med = {(f, c): med(by[("ar", f, c)], lambda r: r["decode_s"])
              for f in FAMILIES for c in CONTEXTS}
    or_med = {(f, c): med(by[("oracle", f, c)], lambda r: r["decode_s"])
              for f in FAMILIES for c in CONTEXTS}
    ar_len = {(f, c): by[("ar", f, c)][0]["generated_length"] for f in FAMILIES for c in CONTEXTS}
    or_len = {(f, c): by[("oracle", f, c)][0]["generated_length"] for f in FAMILIES for c in CONTEXTS}

    # 跨批次参照：relaxed Stage B（strict / δ=0.5 / δ=1.0）
    stage_b = {}
    sb_csv = args.stage_b / "analysis" / "condition_summary.csv"
    if sb_csv.exists():
        for r in csv.DictReader(sb_csv.open(encoding="utf-8-sig")):
            stage_b[(int(r["context"]), float(r["delta"]))] = dict(
                acceptance=float(r["acceptance_median"]), rounds=float(r["rounds_median"]),
                speedup=float(r["speedup_median"]))
    # Phase O optimized（跨批次）
    phase_o_opt = {}
    phase_o_ar = {}
    if args.phase_o_csv.exists():
        for r in csv.DictReader(args.phase_o_csv.open(encoding="utf-8-sig")):
            key = int(r["context"])
            if r["role"] == "opt":
                phase_o_opt.setdefault(key, []).append(float(r["decode_median_s"]))
            elif r["role"] == "ar":
                phase_o_ar.setdefault(key, []).append(float(r["decode_median_s"]))
    phase_o_opt = {k: st.median(v) for k, v in phase_o_opt.items()}
    phase_o_ar = {k: st.median(v) for k, v in phase_o_ar.items()}

    core = []
    for ctx in CONTEXTS:
        ars = [ar_med[(f, ctx)] for f in FAMILIES]
        ors = [or_med[(f, ctx)] for f in FAMILIES]
        speed = [a / o for a, o in zip(ars, ors)]
        stats = by[("oracle", FAMILIES[0], ctx)][0]["spec_stats"]
        sb0 = stage_b.get((ctx, 0.0))
        sb05 = stage_b.get((ctx, 0.5))
        sb1 = stage_b.get((ctx, 1.0))
        core.append(dict(
            context=ctx,
            AR_decode_median_s=st.median(ars),
            all_accept_decode_median_s=st.median(ors),
            all_accept_speedup_median=st.median(speed),
            all_accept_speedup_min=min(speed),
            all_accept_speedup_max=max(speed),
            all_accept_acceptance_median=stats["accepted"] / stats["proposed"],
            all_accept_rounds_median=stats["rounds"],
            all_accept_rollback_median=stats["rollback_count"],
            all_accept_mismatch_median=stats["mismatch_rounds"],
            all_accept_repair_median=stats["repair_count"],
            strict_speedup_crossbatch=(sb0["speedup"] if sb0 else None),
            delta05_speedup_crossbatch=(sb05["speedup"] if sb05 else None),
            delta10_speedup_crossbatch=(sb1["speedup"] if sb1 else None),
            phase_o_opt_decode_crossbatch=phase_o_opt.get(ctx),
            phase_o_ar_decode_crossbatch=phase_o_ar.get(ctx),
            all_accept_len_equal_all=all(ar_len[(f, ctx)] == or_len[(f, ctx)] for f in FAMILIES),
        ))

    fam_rows = []
    for f in FAMILIES:
        for ctx in CONTEXTS:
            stats = by[("oracle", f, ctx)][0]["spec_stats"]
            rep = by[("oracle", f, ctx)][0]
            fam_rows.append(dict(
                family=f, context=ctx,
                AR_decode_s=ar_med[(f, ctx)], all_accept_decode_s=or_med[(f, ctx)],
                speedup=ar_med[(f, ctx)] / or_med[(f, ctx)],
                rounds=stats["rounds"], acceptance=stats["accepted"] / stats["proposed"],
                rollback=stats["rollback_count"],
                output_length=rep["generated_length"], stop_reason=rep["stop_reason"],
                ar_length=ar_len[(f, ctx)],
                length_equal=ar_len[(f, ctx)] == or_len[(f, ctx)],
                boundary_ok_all=all(bool(x["boundary_audit"]["boundary_clock_qualified"])
                                    for x in by[("oracle", f, ctx)]),
                finite=rep.get("finite"), cache_ok=rep.get("cache_invariants_ok"),
            ))

    write_csv(out / "core_table.csv", core)
    write_csv(out / "family_table.csv", fam_rows)

    # ---- 判定 ----
    ctx_speed = {c["context"]: c["all_accept_speedup_median"] for c in core}
    all_speed = [r["speedup"] for r in fam_rows]
    len_mismatch = [(r["family"], r["context"]) for r in fam_rows if not r["length_equal"]]
    if len_mismatch:
        outcome = "D"
        explanation = ("all-accept 与 AR 生成长度不同 ⇒ 当前 speedup 不能直接作同长度比较；"
                       f"不匹配条件：{len_mismatch}")
    elif all(s < 1.0 for s in ctx_speed.values()):
        outcome = "A"
        explanation = ("所有 context 的 all-accept speedup 中位 < 1：即使 100% acceptance，"
                       "当前 1.5B/0.5B + k=4 + eager 执行结构也无法超过 AR。")
    elif any(0.95 <= ctx_speed[c] <= 1.10 for c in ctx_speed) and \
            all(s < 1.0 for s in [ctx_speed[4096], ctx_speed[8192]]):
        outcome = "B"
        explanation = ("2K 的 all-accept speedup 处于 0.95–1.10 的狭窄区间，4K/8K 仍 <1："
                       "只有接近完美接受时才存在狭窄获利区间。")
    elif sum(1 for s in ctx_speed.values() if s > 1.0) >= 2:
        outcome = "C"
        explanation = "多个 context 的 all-accept speedup 中位 >1：acceptance 是主要限制因素（但仍需真实 draft 验证）。"
    else:
        outcome = "A"
        explanation = "混合情形：并非所有 context 都 <1，也没有达到 B/C 的门槛；按保守口径归为 A 的邻近情形。"

    summary = dict(
        schema_version="all-accept-analysis-v1",
        experiment_class="mechanism_diagnostic", accept_mode="all_accept_oracle",
        quality_claim_allowed=False, formal_result_allowed=False,
        run=str(run), core=core, family=fam_rows,
        outcome=dict(kind=outcome, explanation=explanation,
                     ctx_speedup_medians=ctx_speed, speedup_min=min(all_speed),
                     speedup_max=max(all_speed)),
        comparability=dict(length_equal_all=not bool(len_mismatch),
                           cross_batch_note=("strict/δ=0.5/δ=1.0 的 speedup 来自 relaxed Stage B 批次；"
                                             "Phase O 来自 2026-10-25 批次；本批次 AR 与 all-accept 为同批同环境。"
                                             "跨批次数值仅用于机制图，不作为同批 speedup 结论。")),
        note=("all-accept 为 latency lower-bound oracle：输出质量无意义；不得作为可部署配置或正式加速结果。"),
    )
    (out / "all_accept_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot(ROOT / "docs" / "figures" / "all_accept_oracle", core, stage_b, phase_o_opt, phase_o_ar)

    print(json.dumps(dict(core=core, outcome=summary["outcome"]), ensure_ascii=False, indent=2,
                     default=str))
    return 0


def plot(figure_dir: Path, core: list, stage_b: dict, phase_o_opt: dict, phase_o_ar: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ctxs = [c["context"] for c in core]
    xs = list(range(len(ctxs)))
    labels = ["2K", "4K", "8K"]
    colors = {"strict": "#8C8C8C", "d05": "#56B4E9", "d10": "#0072B2",
              "oracle": "#E69F00", "phaseO": "#009E73"}
    series = [("strict (δ=0)", colors["strict"]), ("δ=0.5", colors["d05"]),
              ("δ=1.0", colors["d10"]), ("all-accept", colors["oracle"])]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

    ax = axes[0][0]
    for (name, color), key in zip(series, ["strict", "d05", "d10", "oracle"]):
        ys = []
        for c in core:
            if key == "strict":
                ys.append(stage_b.get((c["context"], 0.0), {}).get("acceptance", np.nan))
            elif key == "d05":
                ys.append(stage_b.get((c["context"], 0.5), {}).get("acceptance", np.nan))
            elif key == "d10":
                ys.append(stage_b.get((c["context"], 1.0), {}).get("acceptance", np.nan))
            else:
                ys.append(c["all_accept_acceptance_median"])
        ax.plot(xs, ys, marker="o", linewidth=1.5, color=color, label=name)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0.35, 1.05)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[0][1]
    for (name, color), key in zip(series, ["strict", "d05", "d10", "oracle"]):
        ys = []
        for c in core:
            if key == "strict":
                ys.append(stage_b.get((c["context"], 0.0), {}).get("rounds", np.nan))
            elif key == "d05":
                ys.append(stage_b.get((c["context"], 0.5), {}).get("rounds", np.nan))
            elif key == "d10":
                ys.append(stage_b.get((c["context"], 1.0), {}).get("rounds", np.nan))
            else:
                ys.append(c["all_accept_rounds_median"])
        ax.plot(xs, ys, marker="o", linewidth=1.5, color=color, label=name)
    ax.axhline(16, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("verification rounds")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][0]
    for (name, color), key in zip(series, ["strict", "d05", "d10", "oracle"]):
        ys = []
        for c in core:
            if key == "strict":
                ys.append(stage_b.get((c["context"], 0.0), {}).get("speedup", np.nan))
            elif key == "d05":
                ys.append(stage_b.get((c["context"], 0.5), {}).get("speedup", np.nan))
            elif key == "d10":
                ys.append(stage_b.get((c["context"], 1.0), {}).get("speedup", np.nan))
            else:
                ys.append(c["all_accept_speedup_median"])
        ax.plot(xs, ys, marker="o", linewidth=1.5, color=color, label=name)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("T_AR / T_SpecDec")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][1]
    groups = ["strict", "Phase O opt", "δ=1.0", "all-accept"]
    bar_colors = [colors["strict"], colors["phaseO"], colors["d10"], colors["oracle"]]
    width = 0.2
    for gi, (group, color) in enumerate(zip(groups, bar_colors)):
        ys = []
        for c in core:
            if group == "strict":
                base = stage_b.get((c["context"], 0.0), {}).get("speedup")
                ys.append(1.0 / base if base else np.nan)
            elif group == "Phase O opt":
                opt = c["phase_o_opt_decode_crossbatch"]
                ar = c["phase_o_ar_decode_crossbatch"]
                ys.append(opt / ar if opt and ar else np.nan)
            elif group == "δ=1.0":
                base = stage_b.get((c["context"], 1.0), {}).get("speedup")
                ys.append(1.0 / base if base else np.nan)
            else:
                ys.append(c["all_accept_decode_median_s"] / c["AR_decode_median_s"])
        ax.bar([x + (gi - 1.5) * width for x in xs], ys, width=width, color=color,
               label=group)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("normalized decode time (Spec / AR)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(figure_dir / f"latency_floor.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
