"""relaxed_specdec 分析：condition 汇总、speed-quality frontier、图、机制判定。

输入：<run>/manifest.json、<run>/runs.jsonl、<run>/<condition>/result.json
输出：<run>/analysis/{condition_summary.csv, speed_quality_frontier.csv,
        relaxed_specdec_summary.json, relaxed_specdec_frontier.{png,pdf,svg}}

口径：
  * speedup(delta, L) = median_AR_decode(L) / median_relaxed_decode(delta, L)（family 级中位再取中位）；
  * 只看 measured pass 的 decode_s；诊断运行从不进入计时；
  * 质量以同 family/ctx 的 strict（delta=0.0）运行为参照：token 相等、共同前缀、长度差、
    relaxed-only 的 target-logit regret。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)
DELTAS = (0.0, 0.1, 0.3, 0.5, 1.0)


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(run: Path) -> list:
    rows = []
    path = run / "runs.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def key_of(row) -> tuple:
    return (row["family"], row["context"], row["delta"])


def first_divergence(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def condition_rows(run: Path, rows: list) -> list:
    by = defaultdict(list)
    for row in rows:
        by[(row["family"], row["context"], row["role"], row["delta"])].append(row)
    out = []
    for (family, ctx, role, delta), reps in sorted(by.items(), key=lambda kv: (
            kv[0][1], kv[0][0], str(kv[0][3]))):
        reps = sorted(reps, key=lambda r: r["rep"])
        decode = [r["decode_s"] for r in reps]
        prefill = [r["prefill_s"] for r in reps]
        total = [r["total_s"] for r in reps]
        stats = reps[0].get("spec_stats") or {}
        row = dict(family=family, context=ctx, role=role, delta=delta, reps=len(reps),
                   gen_len=reps[0]["generated_length"],
                   stop_reason=reps[0]["stop_reason"],
                   prefill_median_s=st.median(prefill),
                   decode_median_s=st.median(decode),
                   decode_min_s=min(decode), decode_max_s=max(decode),
                   total_median_s=st.median(total),
                   decode_tokens_per_s=st.median(r["decode_tokens_per_s"] for r in reps),
                   peak_reserved_gib=max(r["peak_reserved_bytes"] for r in reps) / 2**30,
                   boundary_ok_all=all(bool(r["boundary_audit"]["boundary_clock_qualified"])
                                       for r in reps),
                   finite=reps[0].get("finite"),
                   cache_invariants_ok=reps[0].get("cache_invariants_ok"),
                   attention_strategy_ok=reps[0].get("attention_strategy_ok"),
                   generated_ids=reps[0]["generated_ids"])
        if role == "relaxed" and stats:
            accepted = stats["accepted"]
            strictly = int(stats.get("strictly_matched", accepted))
            relaxed_only = int(stats.get("relaxed_only_accepted", 0))
            regret_sum = float(stats.get("sum_target_logit_regret", 0.0))
            regret_max = float(stats.get("max_target_logit_regret", 0.0))
            row.update(
                rounds=stats["rounds"], proposed=stats["proposed"],
                strictly_matched_tokens=strictly,
                relaxed_only_accepted_tokens=relaxed_only,
                total_accepted_tokens=accepted,
                acceptance_rate=accepted / stats["proposed"],
                mismatch_rounds=stats["mismatch_rounds"],
                rollback_count=stats["rollbacks"],
                all_accept_rounds=stats["all_accept_rounds"],
                target_block_calls=stats["target_block_calls"],
                draft_calls=stats["draft_calls"],
                sum_target_logit_regret=regret_sum,
                mean_target_logit_regret=(regret_sum / accepted if accepted else 0.0),
                max_target_logit_regret=regret_max,
                relaxed_only_accept_count=relaxed_only,
            )
        out.append(row)
    return out


def quality_vs_strict(cond_rows: list) -> list:
    strict = {(r["family"], r["context"]): r for r in cond_rows if r["delta"] == 0.0}
    out = []
    for row in cond_rows:
        if row["delta"] is None:
            continue
        ref = strict.get((row["family"], row["context"]))
        if ref is None:
            raise KeyError(f"缺少 strict 参照 {row['family']}/{row['context']}")
        a, b = ref["generated_ids"], row["generated_ids"]
        div = first_divergence(a, b)
        item = dict(row)
        item.update(
            token_exact_equal=bool(a == b),
            common_prefix_length=div,
            first_divergence_position=(div if div < min(len(a), len(b)) else None),
            strict_length=len(a),
            relaxed_length=len(b),
            length_delta=len(b) - len(a),
        )
        out.append(item)
    return out


def speedup_rows(cond_rows: list, quality_rows: list) -> list:
    ar = {(r["family"], r["context"]): r for r in cond_rows if r["role"] == "ar"}
    qmap = {(r["family"], r["context"], r["delta"]): r for r in quality_rows}
    out = []
    for (family, ctx, delta), q in sorted(qmap.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        base = ar.get((family, ctx))
        if base is None:
            continue
        speedup = base["decode_median_s"] / q["decode_median_s"]
        row = dict(family=family, context=ctx, delta=delta,
                   AR_decode_s=base["decode_median_s"],
                   relaxed_decode_s=q["decode_median_s"],
                   speedup=speedup,
                   acceptance_rate=q["acceptance_rate"],
                   rounds=q["rounds"],
                   rollback_count=q["rollback_count"],
                   mismatch_rounds=q["mismatch_rounds"],
                   mean_target_logit_regret=q["mean_target_logit_regret"],
                   max_target_logit_regret=q["max_target_logit_regret"],
                   sum_target_logit_regret=q["sum_target_logit_regret"],
                   common_prefix_length=q["common_prefix_length"],
                   token_exact_equal=q["token_exact_equal"],
                   length_delta=q["length_delta"],
                   relaxed_length=q["relaxed_length"],
                   strictly_matched_tokens=q["strictly_matched_tokens"],
                   relaxed_only_accepted_tokens=q["relaxed_only_accepted_tokens"])
        out.append(row)
    return out


def context_summary(speed_rows: list) -> list:
    out = []
    for ctx in CONTEXTS:
        for delta in DELTAS:
            subset = [r for r in speed_rows if r["context"] == ctx and r["delta"] == delta]
            if not subset:
                continue
            out.append(dict(
                context=ctx, delta=delta, families=len(subset),
                acceptance_median=st.median(r["acceptance_rate"] for r in subset),
                rounds_median=st.median(r["rounds"] for r in subset),
                mismatch_median=st.median(r["mismatch_rounds"] for r in subset),
                speedup_median=st.median(r["speedup"] for r in subset),
                speedup_min=min(r["speedup"] for r in subset),
                speedup_max=max(r["speedup"] for r in subset),
                mean_regret_median=st.median(r["mean_target_logit_regret"] for r in subset),
                max_regret_median=st.median(r["max_target_logit_regret"] for r in subset),
                prefix_median=st.median(r["common_prefix_length"] for r in subset),
                token_exact_equal_count=sum(r["token_exact_equal"] for r in subset),
                length_delta_median=st.median(r["length_delta"] for r in subset),
                relaxed_len_median=st.median(r["relaxed_length"] for r in subset),
            ))
    return out


def plot(figure_dir: Path, speed_rows: list, contexts: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ctx_colors = {2048: "#0072B2", 4096: "#E69F00", 8192: "#009E73"}
    deltas = sorted({r["delta"] for r in speed_rows})
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)

    def med(ctx, delta, key):
        vals = [r[key] for r in speed_rows if r["context"] == ctx and r["delta"] == delta]
        return np.median(vals) if vals else np.nan

    panels = [
        ("acceptance_rate", "acceptance rate", axes[0][0], "a"),
        ("rounds", "verification rounds", axes[0][1], "b"),
        ("speedup", "T_AR / T_relaxedSpec", axes[0][2], "c"),
        ("mean_target_logit_regret", "mean target-logit regret", axes[1][0], "d"),
        ("common_prefix_length", "common prefix vs strict", axes[1][1], "e"),
    ]
    for key, label, ax, tag in panels:
        for ctx in CONTEXTS:
            xs = [d for d in deltas if any(r["context"] == ctx and r["delta"] == d for r in speed_rows)]
            ys = [med(ctx, d, key) for d in xs]
            ax.plot(xs, ys, marker="o", linewidth=1.5, color=ctx_colors[ctx], label=f"{ctx // 1024}K")
        ax.set_xlabel("delta")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#E9EDF0")
        ax.set_axisbelow(True)
        ax.text(-0.12, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)
        if key == "speedup":
            ax.axhline(1.0, color="black", linewidth=0.8)
        if key == "common_prefix_length":
            ax.axhline(64, color="#666666", linewidth=0.7, linestyle="--")
        if tag in ("a", "b", "c"):
            ax.legend(frameon=False, fontsize=7.5)

    # (f) Pareto: mean regret vs speedup（每个 ctx 一条线，点=delta）
    ax = axes[1][2]
    for ctx in CONTEXTS:
        pts = sorted([(r["mean_target_logit_regret"], r["speedup"], r["delta"]) for r in speed_rows
                      if r["context"] == ctx], key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.3, color=ctx_colors[ctx], label=f"{ctx // 1024}K")
        for x, y, d in pts:
            ax.annotate(f"{d:g}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6,
                        color=muted)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("mean target-logit regret (token)")
    ax.set_ylabel("T_AR / T_relaxedSpec")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(-0.12, 1.04, "f", transform=ax.transAxes, fontweight="bold", fontsize=11)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(figure_dir / f"relaxed_specdec_frontier.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def verdict(speed_rows: list, summary: list) -> dict:
    good = [r for r in speed_rows if r["speedup"] > 1.0]
    strict_rows = [r for r in speed_rows if r["delta"] == 0.0]
    acc = {d: st.median(r["acceptance_rate"] for r in speed_rows if r["delta"] == d)
           for d in sorted({r["delta"] for r in speed_rows})}
    acc_gain = (max(acc.values()) - acc.get(0.0, 0.0)) if acc else 0.0
    if not good:
        kind = "B"
        explanation = ("接受率随 delta 提高，但没有任何 (family,ctx,delta) 超过 AR（speedup<=1）；"
                       "说明系统执行成本占主导，人为改善接受率不足以跨过 break-even。")
    else:
        best = max(speed_rows, key=lambda r: r["speedup"])
        small_delta = [r for r in good if r["delta"] <= 0.3]
        if small_delta:
            kind = "D"
            explanation = (f"轻微 delta（<=0.3）即出现 speedup>1 的候选点（最佳 {best['delta']} @ "
                           f"{best['context']}，speedup={best['speedup']:.3f}）；质量影响需逐项核对，"
                           "只能作为 development finding。")
        elif any(r["token_exact_equal"] for r in good) and all(r["relaxed_length"] >= r["strict_length"] - 4 for r in good):
            kind = "A"
            explanation = ("在接受率提升的同时出现真实 decode 加速；说明 strict 验证/模型对齐不足是重要瓶颈，"
                           "但伴随受控输出偏移。")
        else:
            kind = "C"
            explanation = ("加速伴随明显的轨迹/长度/质量变化；只能描述为速度—质量权衡，不能称为无损加速。")
    return dict(kind=kind, explanation=explanation,
                acceptance_by_delta=acc, acceptance_gain=acc_gain,
                any_speedup_gt_one=bool(good),
                specific_candidates=[dict(context=r["context"], delta=r["delta"],
                                          speedup=r["speedup"],
                                          acceptance=r["acceptance_rate"],
                                          mean_regret=r["mean_target_logit_regret"],
                                          prefix=r["common_prefix_length"],
                                          length_delta=r["length_delta"])
                                     for r in good])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    rows = load_rows(run)
    conditions = condition_rows(run, rows)
    quality = quality_vs_strict(conditions)
    speed = speedup_rows(conditions, quality)
    summary = context_summary(speed)

    write_csv(out / "cases.csv",
              [{k: v for k, v in r.items() if k != "generated_ids"} for r in quality])
    write_csv(out / "condition_summary.csv", summary)
    write_csv(out / "speed_quality_frontier.csv", speed)

    figure_dir = ROOT / "docs" / "figures" / "relaxed_specdec"
    plot(figure_dir, speed, summary)

    v = verdict(speed, summary)
    payload = dict(schema_version="relaxed-specdec-analysis-v1", run=str(run),
                   contexts=summary, frontier=speed, verdict=v,
                   scope="mechanism diagnostic on development families; not the formal main experiment",
                   delta_grid=list(sorted({r["delta"] for r in speed})),
                   families=list(sorted({r["family"] for r in speed})),
                   note=("strict baseline (delta=0.0) 保持原样；relaxed 结果不得描述为 lossless speedup；"
                         "其余为机制诊断"))
    (out / "relaxed_specdec_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(dict(contexts=summary, verdict=v), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
