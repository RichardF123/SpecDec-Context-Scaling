"""Bonus-Aware Perfect-Accept 分析（同批 36 条件：AR / nobonus / bonus）。

输出：
  <run>/analysis/{core_table.csv, family_table.csv, workload.csv, bonus_accept_summary.json}
  docs/figures/bonus_accept_oracle/bonus_latency_floor.{png,pdf,svg}
判定（按任务 §十六）：
  A: 所有 ctx 的 bonus speedup 中位 <1
  B: 0.95 ≤ 所有 ctx 中位 <1.0（接近但未稳定超过）
  C: 部分 ctx 中位 >1
  D: 多个 ctx 稳定 >1（所有 ctx 中位 >1 且最小 >1.02）
  E: carry-forward 无法实现（本实现有，不适用）
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
ROLES = ("ar", "nobonus", "bonus")


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_stats(role: str, stats: dict) -> dict:
    """all_accept_oracle（无 bonus）与 bonus_accept_oracle 的字段名不同，统一到同一 schema。"""
    s = dict(stats or {})
    if role == "nobonus":
        s.setdefault("candidate_tokens", s.get("candidate_count", 0))
        s.setdefault("bonus_tokens", s.get("bonus_token_count", 0))
        s.setdefault("candidate_calls", s.get("draft_calls", 0))
        s.setdefault("bonus_catchup_calls", 0)
        s.setdefault("target_block_input_tokens", s.get("proposed", 0))
        s.setdefault("target_extra_single_step_calls", 0)
        s.setdefault("draft_extra_single_step_calls", 0)
    else:
        s.setdefault("candidate_tokens", 0)
        s.setdefault("bonus_tokens", 0)
        s.setdefault("candidate_calls", 0)
        s.setdefault("bonus_catchup_calls", 0)
        s.setdefault("target_block_input_tokens", 0)
        s.setdefault("target_extra_single_step_calls", 0)
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    rows = load_runs(run / "runs.jsonl")
    by = {}
    for r in rows:
        by.setdefault((r["role"], r["family"], r["context"]), []).append(r)

    med = lambda role, f, c: st.median(r["decode_s"] for r in by[(role, f, c)])

    core = []
    for ctx in CONTEXTS:
        ars = [med("ar", f, ctx) for f in FAMILIES]
        nbs = [med("nobonus", f, ctx) for f in FAMILIES]
        bons = [med("bonus", f, ctx) for f in FAMILIES]
        sp_nb = [a / n for a, n in zip(ars, nbs)]
        sp_b = [a / b for a, b in zip(ars, bons)]
        sb = [normalize_stats("bonus", by[("bonus", f, ctx)][0]["spec_stats"]) for f in FAMILIES]
        snb = [normalize_stats("nobonus", by[("nobonus", f, ctx)][0]["spec_stats"]) for f in FAMILIES]
        st_b = {k: st.median(s[k] for s in sb) for k in sb[0]}
        st_nb = {k: st.median(s[k] for s in snb) for k in snb[0]}
        core.append(dict(
            context=ctx,
            AR_decode_median_s=st.median(ars),
            nobonus_decode_median_s=st.median(nbs),
            bonus_decode_median_s=st.median(bons),
            nobonus_speedup_median=st.median(sp_nb),
            nobonus_speedup_min=min(sp_nb), nobonus_speedup_max=max(sp_nb),
            bonus_speedup_median=st.median(sp_b),
            bonus_speedup_min=min(sp_b), bonus_speedup_max=max(sp_b),
            nobonus_rounds_median=st_nb["rounds"],
            bonus_rounds_median=st_b["rounds"],
            nobonus_candidate_tokens_median=st_nb["candidate_tokens"],
            bonus_candidate_tokens_median=st_b["candidate_tokens"],
            bonus_tokens_median=st_b["bonus_tokens"],
            bonus_count_per_round=(st_b["bonus_tokens"] / st_b["rounds"]),
            nobonus_target_block_calls_median=st_nb["target_block_calls"],
            bonus_target_block_calls_median=st_b["target_block_calls"],
            nobonus_draft_calls_median=st_nb["draft_calls"],
            bonus_draft_calls_median=st_b["draft_calls"],
            bonus_catchup_median=st_b["bonus_catchup_calls"],
            target_block_input_tokens_median=st_b["target_block_input_tokens"],
            target_extra_single_step_median=st_b["target_extra_single_step_calls"],
            round_reduction=st_nb["rounds"] - st_b["rounds"],
            wall_reduction_nobonus_over_bonus=st.median(nbs) / st.median(bons),
            bonus_overhead_per_saved_round_s=(st.median(bons) - st.median(nbs)) / max(1, st_nb["rounds"] - st_b["rounds"]),
        ))

    fam_rows = []
    for f in FAMILIES:
        for ctx in CONTEXTS:
            st_b = normalize_stats("bonus", by[("bonus", f, ctx)][0]["spec_stats"])
            st_nb = normalize_stats("nobonus", by[("nobonus", f, ctx)][0]["spec_stats"])
            fam_rows.append(dict(
                family=f, context=ctx,
                AR_decode_s=med("ar", f, ctx),
                nobonus_decode_s=med("nobonus", f, ctx),
                bonus_decode_s=med("bonus", f, ctx),
                nobonus_speedup=med("ar", f, ctx) / med("nobonus", f, ctx),
                bonus_speedup=med("ar", f, ctx) / med("bonus", f, ctx),
                rounds_nobonus=st_nb["rounds"], rounds_bonus=st_b["rounds"],
                candidate_tokens=st_b["candidate_tokens"],
                bonus_tokens=st_b["bonus_tokens"],
                draft_calls_nobonus=st_nb["draft_calls"],
                draft_calls_bonus=st_b["draft_calls"],
                target_block_calls_nobonus=st_nb["target_block_calls"],
                target_block_calls_bonus=st_b["target_block_calls"],
                output_length=by[("bonus", f, ctx)][0]["generated_length"],
                ar_length=by[("ar", f, ctx)][0]["generated_length"],
                stop_reason=by[("bonus", f, ctx)][0]["stop_reason"],
                length_equal=by[("bonus", f, ctx)][0]["generated_length"] == by[("ar", f, ctx)][0]["generated_length"],
                finite=by[("bonus", f, ctx)][0]["finite"],
                cache_ok=by[("bonus", f, ctx)][0]["cache_invariants_ok"],
                boundary_ok_all=all(bool(x["boundary_audit"]["boundary_clock_qualified"])
                                    for x in by[("bonus", f, ctx)]),
            ))

    # workload 分解（每 ctx 中位）
    workload = []
    for ctx in CONTEXTS:
        for role in ("nobonus", "bonus"):
            stats = [normalize_stats(role, by[(role, f, ctx)][0]["spec_stats"]) for f in FAMILIES]
            # 逐 family 中位
            workload.append(dict(
                context=ctx, role=role,
                rounds=st.median(s["rounds"] for s in stats),
                candidate_calls=st.median(s["candidate_calls"] for s in stats),
                bonus_catchup_calls=st.median(s["bonus_catchup_calls"] for s in stats),
                draft_calls_total=st.median(s["draft_calls"] for s in stats),
                target_block_calls=st.median(s["target_block_calls"] for s in stats),
                target_block_input_tokens=st.median(s["target_block_input_tokens"] for s in stats),
                target_extra_single_step=st.median(s["target_extra_single_step_calls"] for s in stats),
            ))

    write_csv(out / "core_table.csv", core)
    write_csv(out / "family_table.csv", fam_rows)
    write_csv(out / "workload.csv", workload)

    ctx_speed = {c["context"]: c["bonus_speedup_median"] for c in core}
    min_speed = min(r["bonus_speedup"] for r in fam_rows)
    if all(s < 1.0 for s in ctx_speed.values()):
        if all(0.95 <= s < 1.0 for s in ctx_speed.values()):
            outcome = "B"
            explanation = ("bonus-aware 的 speedup 在全部 context 处于 0.95–1.0（接近但未稳定超过）；"
                           "bonus 修复了大部分结构损失，但经典独立 draft 只有极窄的 break-even 空间。")
        else:
            outcome = "A"
            explanation = ("bonus-aware 在全部 context 仍 <1：即使采用 k+1 提交并达到 100% acceptance，"
                           "当前模型对与执行结构仍不能超过 AR。")
    elif all(s > 1.0 and s >= 1.02 for s in ctx_speed.values()):
        outcome = "D"
        explanation = "多个（全部）context 稳定 >1：bonus-token 契约是此前性能上界判断中的关键缺失。"
    else:
        outcome = "C"
        explanation = "部分 context 的 bonus-aware speedup 中位 >1：经典 SpecDec 在完美接受 + k+1 提交下存在理论获利区间。"

    summary = dict(
        schema_version="bonus-accept-analysis-v1",
        experiment_class="mechanism_diagnostic", accept_mode="all_accept_bonus_oracle",
        quality_claim_allowed=False, formal_result_allowed=False,
        run=str(run), core=core, family=fam_rows, workload=workload,
        outcome=dict(kind=outcome, explanation=explanation,
                     ctx_speedup_medians=ctx_speed, speedup_min=min_speed,
                     speedup_max=max(r["bonus_speedup"] for r in fam_rows)),
        notes=("bonus-aware 为 latency lower-bound oracle：输出无质量意义，不能作为可部署算法或正式加速结果。"
               "同批同环境包含 AR / nobonus / bonus 三种模式；nobonus 复用原契约（16 轮）作为对照。")
    )
    (out / "bonus_accept_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot(ROOT / "docs" / "figures" / "bonus_accept_oracle", core, workload)

    print(json.dumps(dict(core=core, workload=workload, outcome=summary["outcome"]),
                     ensure_ascii=False, indent=2, default=str))
    return 0


def plot(figure_dir: Path, core: list, workload: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = ["2K", "4K", "8K"]
    xs = list(range(3))
    colors = {"ar": "#8C8C8C", "nobonus": "#0072B2", "bonus": "#E69F00"}
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

    ax = axes[0][0]
    ax.plot(xs, [c["nobonus_rounds_median"] for c in core], marker="o", color=colors["nobonus"],
            linewidth=1.6, label="no-bonus")
    ax.plot(xs, [c["bonus_rounds_median"] for c in core], marker="o", color=colors["bonus"],
            linewidth=1.6, label="bonus-aware")
    for c, x in zip(core, xs):
        if c["bonus_rounds_median"] != c["nobonus_rounds_median"]:
            ax.annotate(f"{int(c['nobonus_rounds_median'])}→{int(c['bonus_rounds_median'])}",
                        (x, c["bonus_rounds_median"]), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color=muted)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("verification rounds")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[0][1]
    width = 0.26
    for i, (role, color, name) in enumerate([("AR", colors["ar"], "AR"),
                                             ("nobonus", colors["nobonus"], "no-bonus all-accept"),
                                             ("bonus", colors["bonus"], "bonus-aware all-accept")]):
        key = {"AR": "AR_decode_median_s", "nobonus": "nobonus_decode_median_s",
               "bonus": "bonus_decode_median_s"}[role]
        ax.bar([x + (i - 1) * width for x in xs], [c[key] for c in core], width=width,
               color=color, label=name)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("decode latency (s, median)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][0]
    ax.plot(xs, [c["nobonus_speedup_median"] for c in core], marker="o",
            color=colors["nobonus"], linewidth=1.6, label="AR/no-bonus")
    ax.plot(xs, [c["bonus_speedup_median"] for c in core], marker="o",
            color=colors["bonus"], linewidth=1.6, label="AR/bonus-aware")
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("T_AR / T_SpecDec")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][1]
    groups = ["draft candidate", "draft bonus catch-up", "target block", "target block input tokens"]
    keys = ["candidate_calls", "bonus_catchup_calls", "target_block_calls", "target_block_input_tokens"]
    bar_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    width = 0.2
    for gi, (group, key, color) in enumerate(zip(groups, keys, bar_colors)):
        vals = []
        for c, x in zip(core, xs):
            wb = next(w for w in workload if w["context"] == c["context"] and w["role"] == "bonus")
            vals.append(wb[key])
        ax.bar([x + (gi - 1.5) * width for x in xs], vals, width=width, color=color, label=group)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("count per run (median)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(figure_dir / f"bonus_latency_floor.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
