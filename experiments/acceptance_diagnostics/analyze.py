"""Summarize position-wise acceptance diagnostics and render a compact figure."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(positions: list[dict[str, str]], cases: list[dict[str, str]]) -> dict:
    relevant = [row for row in positions if row["decision_relevant"].lower() == "true"]
    mismatch = [row for row in relevant if row["first_mismatch"].lower() == "true"]
    position_rows = []
    for ctx in (2048, 4096, 8192):
        for position in (1, 2, 3, 4):
            rows = [row for row in relevant if int(row["ctx"]) == ctx and int(row["position"]) == position]
            if not rows:
                continue
            accepted = sum(row["accepted"].lower() == "true" for row in rows)
            position_rows.append({
                "ctx": ctx, "position": position, "reached": len(rows), "accepted": accepted,
                "conditional_acceptance": accepted / len(rows),
                "mismatches": len(rows) - accepted,
            })
    margin_rows = []
    for ctx in (2048, 4096, 8192):
        rows = [row for row in mismatch if int(row["ctx"]) == ctx]
        margins = [float(row["target_margin"]) for row in rows]
        ranks = [int(row["draft_token_target_rank"]) for row in rows]
        margin_rows.append({
            "ctx": ctx,
            "mismatches": len(rows),
            "margin_median": st.median(margins),
            "margin_le_0_1": sum(value <= 0.1 for value in margins),
            "margin_le_0_3": sum(value <= 0.3 for value in margins),
            "margin_gt_0_3": sum(value > 0.3 for value in margins),
            "draft_rank_median": st.median(ranks),
            "draft_rank_top2_pct": 100 * sum(value <= 2 for value in ranks) / len(ranks),
            "draft_rank_top5_pct": 100 * sum(value <= 5 for value in ranks) / len(ranks),
        })
    context_cases = []
    for ctx in (2048, 4096, 8192):
        rows = [row for row in cases if int(row["ctx"]) == ctx]
        context_cases.append({
            "ctx": ctx,
            "families": len(rows),
            "acceptance_median": st.median(float(row["acceptance"]) for row in rows),
            "rounds_median": st.median(float(row["rounds"]) for row in rows),
            "mismatch_rounds_median": st.median(float(row["mismatch_rounds"]) for row in rows),
        })
    return {"position": position_rows, "margins": margin_rows, "contexts": context_cases}


def render(summary: dict, positions: list[dict[str, str]], out: Path) -> None:
    colors = {2048: "#0072B2", 4096: "#D55E00", 8192: "#009E73"}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45))
    ax = axes[0]
    for ctx in (2048, 4096, 8192):
        rows = [row for row in summary["position"] if row["ctx"] == ctx]
        ax.plot([row["position"] for row in rows], [row["conditional_acceptance"] for row in rows],
                marker="o", color=colors[ctx], label=f"{ctx // 1024}K")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position within k=4 block")
    ax.set_ylabel("Conditional acceptance")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1]
    mismatch = [row for row in positions if row["decision_relevant"].lower() == "true"
                and row["first_mismatch"].lower() == "true"]
    bins = np.array([0, 0.1, 0.3, 0.5, 1, 2, 4, 8, 16, 32], dtype=float)
    for ctx in (2048, 4096, 8192):
        values = [float(row["target_margin"]) for row in mismatch if int(row["ctx"]) == ctx]
        ax.hist(values, bins=bins, histtype="step", linewidth=1.4, color=colors[ctx], label=f"{ctx // 1024}K")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Target top-1 minus top-2 margin")
    ax.set_ylabel("First mismatches")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[2]
    ranks = [int(row["draft_token_target_rank"]) for row in mismatch]
    categories = [sum(value == 2 for value in ranks), sum(3 <= value <= 5 for value in ranks),
                  sum(6 <= value <= 20 for value in ranks), sum(value > 20 for value in ranks)]
    ax.bar(["2", "3–5", "6–20", ">20"], categories,
           color=["#56B4E9", "#E69F00", "#CC79A7", "#777777"])
    ax.set_xlabel("Draft token rank under target")
    ax.set_ylabel("First mismatches")
    ax.text(0.02, 0.97, "c", transform=ax.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.42)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"acceptance_diagnostics.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    positions = read_csv(args.run / "positions.csv")
    cases = read_csv(args.run / "cases.csv")
    summary = summarize(positions, cases)
    write_csv(out / "position_acceptance.csv", summary["position"])
    write_csv(out / "mismatch_margins.csv", summary["margins"])
    write_csv(out / "context_summary.csv", summary["contexts"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render(summary, positions, out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

