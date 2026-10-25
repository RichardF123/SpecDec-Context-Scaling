"""Phase O 开发集基准分析：逐条件汇总、ratio、阶段账本、图、裁决。

输入：
  <run>/bench/manifest.json + <run>/bench/<family>_<ctx>_<role>/result.json
  <run>/probe/probe.json（正确性）
输出：
  <run>/analysis/{conditions.csv, ratios.csv, stages.csv, context_summary.csv, summary.json}
  docs/figures/specdec_execution_optimization/ot_execution_gain.{svg,png,pdf}
  docs/measurements/ot_execution_*.csv（论文口径摘要；可选）
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
ROLES = ("ar", "ref", "opt")


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_condition(run: Path, family: str, ctx: int, role: str) -> dict:
    path = run / f"{family}_{ctx}_{role}" / "result.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def condition_row(result: dict) -> dict:
    reps = result["reps"]
    decode = [r["decode_s"] for r in reps]
    prefill = [r["prefill_s"] for r in reps]
    total = [r["total_s"] for r in reps]
    tps = [r["decode_tokens_per_s"] for r in reps]
    peaks = [r["peak_reserved_bytes"] for r in reps]
    boundary = [r["runtime_environment_audit"]["boundary_clock_qualified"] for r in reps]
    row = dict(
        role=result["role"], family=result["family"], context=result["context"],
        reps=len(reps), gen_len=result["generated_length"],
        prefill_median_s=st.median(prefill),
        decode_median_s=st.median(decode),
        decode_min_s=min(decode), decode_max_s=max(decode),
        total_median_s=st.median(total),
        tokens_per_s=st.median(tps),
        peak_reserved_gib=max(peaks) / 2**30,
        boundary_ok_all=all(boundary),
        boundary_ok_count=sum(1 for b in boundary if b is True),
        outputs_consistent=result["outputs_consistent"],
    )
    stats = result.get("spec_stats") or {}
    if stats:
        row.update(rounds=stats["rounds"], proposed=stats["proposed"],
                   accepted=stats["accepted"], mismatch_rounds=stats["mismatch_rounds"],
                   acceptance=stats["accepted"] / stats["proposed"] if stats["proposed"] else None,
                   target_block_calls=stats["target_block_calls"],
                   draft_calls=stats["draft_calls"])
    return row


def stage_rows(run: Path, family: str, ctx: int, role: str, decode_median: float) -> list:
    result = load_condition(run, family, ctx, role)
    ledger = result.get("stage_ledger") or {}
    wall = ledger.get("wall_s") or {}
    total = ledger.get("total_s")
    out = []
    phases = {"prefill": ("prefill_target", "prefill_draft"),
              "propose": ("draft_propose",),
              "verify": ("target_verify",),
              "repair": ("repair_target", "repair_draft")}
    for name, keys in phases.items():
        sec = sum(wall.get(k, 0.0) for k in keys)
        out.append(dict(role=role, family=family, ctx=ctx, phase=name,
                        wall_s=sec,
                        share_of_decode_pct=(100.0 * sec / total) if total else None,
                        share_of_primary_decode_pct=(100.0 * sec / decode_median) if decode_median else None))
    covered = sum(wall.values())
    out.append(dict(role=role, family=family, ctx=ctx, phase="residual",
                    wall_s=(total - covered) if total else None,
                    share_of_decode_pct=(100.0 * (total - covered) / total) if total else None,
                    share_of_primary_decode_pct=(100.0 * (total - covered) / decode_median) if decode_median else None))
    return out


def plot(figure_dir: Path, ratios: list, stages: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = {"AR": "#8C8C8C", "reference": "#0072B2", "optimized": "#E69F00"}
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)

    # (a) decode time per ctx per role（family 散点 + 中位）
    ax = axes[0][0]
    xs = {"AR": -0.22, "reference": 0.0, "optimized": 0.22}
    for ctx_i, ctx in enumerate(CONTEXTS):
        for role, label in (("AR", "AR"), ("reference", "reference SpecDec"),
                            ("optimized", "optimized SpecDec")):
            vals = [r[f"{role}_decode_s"] for r in ratios if r["ctx"] == ctx]
            x = ctx_i + xs[label.split()[0]] + np.linspace(-0.06, 0.06, len(vals))
            ax.scatter(x, vals, s=16, color=colors[label.split()[0]], linewidths=0, alpha=0.85,
                       label=label if ctx_i == 0 else None)
            ax.plot([ctx_i + xs[label.split()[0]] - 0.12, ctx_i + xs[label.split()[0]] + 0.12],
                    [np.median(vals)] * 2, color=colors[label.split()[0]], linewidth=1.6)
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylabel("decode latency (s, median of 3)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    # (b) ratios per ctx
    ax = axes[0][1]
    for ctx_i, ctx in enumerate(CONTEXTS):
        for key, label, color in (("R_ref", "R = T_AR/T_refSpec", colors["reference"]),
                                  ("R_opt", "R = T_AR/T_optSpec", colors["optimized"])):
            vals = [r[key] for r in ratios if r["ctx"] == ctx]
            off = -0.1 if key == "R_ref" else 0.1
            ax.scatter([ctx_i + off] * len(vals), vals, s=16, color=color,
                       linewidths=0, alpha=0.85, label=label if ctx_i == 0 else None)
            ax.plot([ctx_i + off - 0.13, ctx_i + off + 0.13], [np.median(vals)] * 2,
                    color=color, linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.axhline(0.8, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylabel("T_AR / T_Spec")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    # (c) stage breakdown stacked（按 ctx 平均 share，reference vs optimized）
    ax = axes[1][0]
    stage_names = ["propose", "verify", "repair", "prefill", "residual"]
    stage_colors = ["#0072B2", "#56B4E9", "#D55E00", "#CCCCCC", "#F0F0F0"]
    positions = []
    labels = []
    pos = 1.0
    for ctx in CONTEXTS:
        for role, label in (("ref", "ref"), ("opt", "opt")):
            shares = []
            for name in stage_names:
                vals = [r["share_of_decode_pct"] for r in stages
                        if r["role"] == role and r["ctx"] == ctx and r["phase"] == name
                        and r["share_of_decode_pct"] is not None]
                shares.append(st.mean(vals) if vals else 0.0)
            bottom = 0.0
            for name, color, share in zip(stage_names, stage_colors, shares):
                ax.bar(pos, share, bottom=bottom, width=0.7, color=color,
                       edgecolor=color, label=name if pos < 2 else None)
                bottom += share
            positions.append(pos)
            labels.append(f"{ctx // 1024}K\n{label}")
            pos += 1
        pos += 0.4
    ax.set_xticks(positions, labels)
    ax.set_ylabel("share of decode time (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

    # (d) optimization gain per ctx
    ax = axes[1][1]
    xs_ = list(range(3))
    for metric, label, color in (("gain_median_pct", "median gain", "#009E73"),
                                 ("decode_reduction_pct", "median decode reduction", "#CC79A7")):
        vals = [r[metric] for r in ratios if r["ctx"] in CONTEXTS]
        med = [np.median([r[metric] for r in ratios if r["ctx"] == c]) for c in CONTEXTS]
        ax.plot(xs_, med, marker="o", color=color, linewidth=1.6, label=label)
        ax.scatter([x for x, v in zip([xs_[CONTEXTS.index(c)] for c in
                                       [r["ctx"] for r in ratios if r["ctx"] in CONTEXTS]],
                                      vals)], vals, s=14, color=color, alpha=0.6, linewidths=0)
    ax.axhline(30, color="#888888", linewidth=0.7, linestyle="--")
    ax.set_xticks(xs_, ["2K", "4K", "8K"])
    ax.set_ylabel("proposal/decode reduction (%)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(figure_dir / f"ot_execution_gain.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    conditions = [condition_row(load_condition(run, c["family"], c["context"], c["role"]))
                  for c in manifest["conditions"]]
    conditions.sort(key=lambda r: (r["context"], r["family"], r["role"]))
    keys = set()
    for r in conditions:
        keys.update(r.keys())
    for r in conditions:
        for k in keys:
            r.setdefault(k, None)

    # 三向输出一致性：ref == opt（任一处不同即 correctness 失败）；AR 与 Spec 允许不同
    consistency = {}
    for family in DEV_FAMILIES:
        for ctx in CONTEXTS:
            ids = {role: load_condition(run, family, ctx, role)["generated_ids"] for role in ROLES}
            consistency[(family, ctx)] = dict(
                ref_opt_equal=ids["ref"] == ids["opt"],
                ar_ref_equal=ids["ar"] == ids["ref"],
                ar_opt_equal=ids["ar"] == ids["opt"])

    ratio_rows = []
    for family in DEV_FAMILIES:
        for ctx in CONTEXTS:
            r = lambda role: load_condition(run, family, ctx, role)
            ar_d = st.median(x["decode_s"] for x in r("ar")["reps"])
            ref_d = st.median(x["decode_s"] for x in r("ref")["reps"])
            opt_d = st.median(x["decode_s"] for x in r("opt")["reps"])
            c = consistency[(family, ctx)]
            ratio_rows.append(dict(
                family=family, ctx=ctx,
                AR_decode_s=ar_d, reference_decode_s=ref_d, optimized_decode_s=opt_d,
                R_ref=ar_d / ref_d, R_opt=ar_d / opt_d,
                gain=ref_d / opt_d, gain_pct=100 * (ref_d / opt_d - 1),
                decode_reduction_pct=100 * (1 - opt_d / ref_d),
                gain_median_pct=100 * (ref_d / opt_d - 1),
                ref_opt_equal=c["ref_opt_equal"], ar_ref_equal=c["ar_ref_equal"]))

    stages = []
    for family in DEV_FAMILIES:
        for ctx in CONTEXTS:
            ref_d = next(r["reference_decode_s"] for r in ratio_rows
                         if r["family"] == family and r["ctx"] == ctx)
            stages += stage_rows(run, family, ctx, "ref", ref_d)
            stages += stage_rows(run, family, ctx, "opt", ref_d)

    context_summary = []
    for ctx in CONTEXTS:
        rs = [r for r in ratio_rows if r["ctx"] == ctx]
        ref_stage = [r["wall_s"] for r in stages if r["role"] == "ref" and r["ctx"] == ctx
                     and r["phase"] == "propose"]
        opt_stage = [r["wall_s"] for r in stages if r["role"] == "opt" and r["ctx"] == ctx
                     and r["phase"] == "propose"]
        context_summary.append(dict(
            ctx=ctx,
            R_ref_median=st.median(r["R_ref"] for r in rs),
            R_opt_median=st.median(r["R_opt"] for r in rs),
            gain_median_pct=st.median(r["gain_pct"] for r in rs),
            decode_reduction_median_pct=st.median(r["decode_reduction_pct"] for r in rs),
            proposal_reduction_median_pct=100 * (1 - st.median(opt_stage) / st.median(ref_stage)),
            ref_opt_equal_all=all(r["ref_opt_equal"] for r in rs),
            ar_ref_equal_all=all(r["ar_ref_equal"] for r in rs)))

    write_csv(out / "conditions.csv", conditions)
    write_csv(out / "ratios.csv", ratio_rows)
    write_csv(out / "stages.csv", stages)
    write_csv(out / "context_summary.csv", context_summary)

    probe_path = run / "probe" / "probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None
    correctness_ok = bool(probe and probe["all_ok"]) and all(
        r["ref_opt_equal"] for r in ratio_rows)

    proposal_reduction = st.median(r["proposal_reduction_median_pct"] for r in context_summary)
    gain_reduction = st.median(r["decode_reduction_pct"] for r in ratio_rows)
    ctx2 = next(r for r in context_summary if r["ctx"] == 2048)
    any_speedup = any(r["R_opt"] > 1.0 for r in ratio_rows)
    gate2k = ctx2["R_opt_median"] >= 0.8
    gate_30pct = (proposal_reduction >= 30.0) and (gain_reduction >= 30.0)
    promising = (correctness_ok and proposal_reduction >= 30.0
                 and (any_speedup or gate2k or gain_reduction >= 30.0))

    if not correctness_ok:
        verdict = "PHASE_O_REJECTED_FOR_CORRECTNESS"
    elif promising:
        verdict = "PHASE_O_PROMISING"
    else:
        verdict = "PHASE_O_VALID_BUT_NO_MATERIAL_GAIN"

    summary = dict(
        schema_version="ot-analysis-v1", run=str(run),
        contexts=context_summary, ratios=ratio_rows,
        stages_summary=[dict(ctx=c["ctx"], proposal_reduction_median_pct=c["proposal_reduction_median_pct"])
                        for c in context_summary],
        correctness=dict(probe_all_ok=bool(probe and probe["all_ok"]),
                         outputs_equal_all=all(r["ref_opt_equal"] for r in ratio_rows),
                         trace_equal_all=bool(probe and all(r.get("trace_diffs") == 0 for r in probe["rows"]))
                         if probe else None),
        gates=dict(proposal_reduction_median_pct=proposal_reduction,
                   decode_reduction_median_pct=gain_reduction,
                   any_context_R_opt_gt_1=any_speedup,
                   ctx2_R_opt_median_ge_0_80=gate2k,
                   gain_ge_30pct=gate_30pct),
        verdict=verdict,
        verdict_rule="Phase O §十（4/4 输出一致、12/12 控制轨迹一致、cache/finite 全过；proposal 中位 ≥30% 降低；且 R>1 或 2K R≥0.80 或 decode 中位 ≥30% 降低）",
        note="pilot-style dev 基准：4 开发 family × 3 ctx × 单次进程；不作正式矩阵结论；不用于 family 筛选",
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                 default=str), encoding="utf-8")

    figure_dir = ROOT / "docs" / "figures" / "specdec_execution_optimization"
    plot(figure_dir, ratio_rows, stages)

    # 论文口径摘要（复制到 docs/measurements）
    meas = ROOT / "docs" / "measurements"
    meas.mkdir(exist_ok=True)
    write_csv(meas / "ot_execution_conditions_2026-10-25.csv", conditions)
    write_csv(meas / "ot_execution_ratios_2026-10-25.csv", ratio_rows)
    write_csv(meas / "ot_execution_stages_2026-10-25.csv", stages)
    write_csv(meas / "ot_execution_context_summary_2026-10-25.csv", context_summary)

    print(json.dumps(dict(contexts=context_summary, verdict=verdict,
                          gates=summary["gates"]), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
