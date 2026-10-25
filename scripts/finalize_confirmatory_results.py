"""Freeze a completed confirmatory batch and render paper-ready figures.

This script is deliberately read-only with respect to the raw batch.  It verifies
the batch contract, writes concise copies of the analysis tables, records hashes,
and renders figures from frozen CSV/JSON evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATE = "2026-10-25"
EXPECTED_COMMIT = "6c1774a0d86348f3abe16dcd9d79f38831d87564"
CONTEXTS = (2048, 4096, 8192)
COLORS = {2048: "#0072B2", 4096: "#D55E00", 8192: "#009E73"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_batch(batch: Path) -> dict:
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "formal_batch_verified":
        raise ValueError(f"batch is not verified: {summary.get('status')}")
    if summary.get("expected_commit") != EXPECTED_COMMIT:
        raise ValueError("unexpected frozen commit")
    if summary.get("runtime", {}).get("dirty_worktree") is not False:
        raise ValueError("formal batch was not collected from a clean worktree")
    tasks = summary.get("completed", [])
    if (
        summary.get("n_tasks") != 192
        or summary.get("n_measured") != 480
        or len(tasks) != 192
        or any(item.get("status") != "formal_job_verified" for item in tasks)
    ):
        raise ValueError(
            f"expected 192 verified tasks and 480 measurements, got "
            f"n_tasks={summary.get('n_tasks')}, completed={len(tasks)}, "
            f"n_measured={summary.get('n_measured')}"
        )
    worker_json = list((batch / "tasks").glob("**/worker_results/*_r*.json"))
    if len(worker_json) != 480:
        raise ValueError(f"expected 480 measured records, got {len(worker_json)}")
    required = {
        "formal_tasks.csv",
        "formal_family_timing.csv",
        "formal_block_shifts.csv",
        "formal_order_ab_ba.csv",
        "formal_pair_summary.csv",
        "formal_context_summary.csv",
    }
    present = {path.name for path in (batch / "analysis").glob("*.csv")}
    if not required <= present:
        raise ValueError(f"missing analysis files: {sorted(required - present)}")
    return summary


def copy_analysis(batch: Path, measurements: Path) -> dict[str, Path]:
    mapping = {
        "formal_context_summary.csv": f"confirmatory_context_summary_{DATE}.csv",
        "formal_pair_summary.csv": f"confirmatory_pair_summary_{DATE}.csv",
        "formal_family_timing.csv": f"confirmatory_family_timing_{DATE}.csv",
        "formal_block_shifts.csv": f"confirmatory_block_shifts_{DATE}.csv",
        "formal_order_ab_ba.csv": f"confirmatory_order_effects_{DATE}.csv",
    }
    outputs: dict[str, Path] = {}
    for source_name, target_name in mapping.items():
        source = batch / "analysis" / source_name
        target = measurements / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        outputs[source_name] = target
    return outputs


def make_old_new_table(current_pairs: list[dict[str, str]], old_pairs: list[dict[str, str]], path: Path) -> None:
    old = {(row["family"], int(row["ctx"])): row for row in old_pairs}
    rows: list[dict[str, object]] = []
    for row in current_pairs:
        key = (row["family"], int(row["ctx"]))
        if key not in old:
            continue
        prior = old[key]
        old_ratio = prior.get("decode_ratio_descriptive") or prior.get("decode_ratio")
        rows.append(
            {
                "family": row["family"],
                "ctx": key[1],
                "old_decode_ratio": old_ratio,
                "confirmatory_decode_ratio": row["decode_ratio"],
                "confirmatory_ratio_reportable": row["ratio_reportable"],
                "old_claim_status": prior.get("claim_status", ""),
                "confirmatory_claim_status": row["claim_status"],
            }
        )
    if len(rows) != 48:
        raise ValueError(f"expected 48 matched old/new family-context pairs, got {len(rows)}")
    write_csv(path, rows)


def boundary_counts(batch: Path) -> list[dict[str, object]]:
    counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for path in (batch / "tasks").glob("**/worker_results/*_r*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        ctx = int(record["job"]["condition"]["context_tokens"])
        mode = record["job"]["condition"]["mode"]
        audit = record["runtime_environment_audit"]
        counts[(ctx, mode)]["total"] += 1
        counts[(ctx, mode)]["evidence_complete"] += int(audit.get("evidence_status") == "COMPLETE")
        counts[(ctx, mode)]["qualified"] += int(bool(audit.get("boundary_clock_qualified")))
    rows = []
    for ctx in CONTEXTS:
        for mode in ("AR", "SpecDec"):
            item = counts[(ctx, mode)]
            rows.append(
                {
                    "ctx": ctx,
                    "mode": mode,
                    "qualified_reps": item["qualified"],
                    "evidence_complete_reps": item["evidence_complete"],
                    "total_reps": item["total"],
                }
            )
    return rows


def aggregate_manifest(batch: Path, summary: dict, outputs: dict[str, Path], measurements: Path) -> Path:
    files = sorted(path for path in batch.rglob("*") if path.is_file())
    tree = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = path.relative_to(batch).as_posix()
        size = path.stat().st_size
        total_bytes += size
        tree.update(f"{rel}\0{sha256(path)}\0{size}\n".encode("utf-8"))
    key_files = [batch / "plan.json", batch / "summary.json"] + sorted((batch / "analysis").glob("*"))
    manifest = {
        "schema_version": "confirmatory-evidence-freeze-v1",
        "created_date": DATE,
        "batch_id": summary["batch_id"],
        "batch_status": summary["status"],
        "experiment": summary["experiment"],
        "expected_commit": summary["expected_commit"],
        "runtime_dirty_worktree": summary["runtime"]["dirty_worktree"],
        "task_count": 192,
        "measured_record_count": 480,
        "family_count": 16,
        "contexts": list(CONTEXTS),
        "raw_batch_location_at_freeze": str(batch.resolve()),
        "raw_tree": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "aggregate_sha256": tree.hexdigest(),
        },
        "key_file_sha256": {
            path.relative_to(batch).as_posix(): sha256(path) for path in key_files
        },
        "concise_repository_artifacts": {
            name: {"path": str(path.relative_to(measurements.parent.parent)), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "interpretation": {
            "ratio": "T_AR_decode / T_SpecDec_decode; values above 1 indicate speculative-decoding speedup",
            "formal_verdict": "FORMAL_COMPLETE_WITH_RESTRICTED_8K",
            "8k_scope": "4/16 family ratios reportable; all 16 retained for descriptive accounting",
        },
    }
    path = measurements / f"confirmatory_evidence_manifest_{DATE}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


def current_main_figure(current: list[dict[str, str]], old_new: list[dict[str, str]], contexts: list[dict[str, str]], out: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55))
    rng = np.random.default_rng(4207)

    ax = axes[0]
    for index, ctx in enumerate(CONTEXTS):
        rows = [row for row in current if int(row["ctx"]) == ctx]
        for row in rows:
            reportable = row["ratio_reportable"].lower() == "true"
            marker = "o" if reportable else "o"
            face = COLORS[ctx] if reportable else "white"
            ax.scatter(index + rng.uniform(-0.12, 0.12), float(row["decode_ratio"]), s=17,
                       marker=marker, facecolors=face, edgecolors=COLORS[ctx], linewidths=0.8, alpha=0.9)
        context = next(item for item in contexts if int(item["ctx"]) == ctx)
        median = float(context["ratio_median_reportable"])
        low = float(context["ratio_bootstrap95_low"])
        high = float(context["ratio_bootstrap95_high"])
        ax.errorbar(index, median, yerr=[[median - low], [high - median]], fmt="D", color="black",
                    markersize=4, capsize=2.5, linewidth=1.1, zorder=5)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel(r"Decode ratio $T_{AR}/T_{Spec}$")
    ax.set_xlabel("Context length")
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontweight="bold", va="top")
    ax.text(0.98, 0.94, "hollow: restricted", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)

    ax = axes[1]
    for ctx in (2048, 4096):
        rows = [row for row in old_new if int(row["ctx"]) == ctx]
        ax.scatter([float(row["old_decode_ratio"]) for row in rows],
                   [float(row["confirmatory_decode_ratio"]) for row in rows],
                   s=18, color=COLORS[ctx], edgecolor="white", linewidth=0.35, alpha=0.85,
                   label=f"{ctx // 1024}K")
    ax.plot([0.25, 0.8], [0.25, 0.8], color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.7)
    ax.axvline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.7)
    ax.set_xlim(0.25, 0.8)
    ax.set_ylim(0.25, 0.8)
    ax.set_xlabel("Earlier formal ratio")
    ax.set_ylabel("Confirmatory ratio")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[2]
    correlations = {}
    for ctx in CONTEXTS:
        rows = [row for row in current if int(row["ctx"]) == ctx]
        x = np.array([float(row["spec_acceptance"]) for row in rows])
        y = np.array([float(row["decode_ratio"]) for row in rows])
        correlations[ctx] = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=20, color=COLORS[ctx], marker={2048: "o", 4096: "s", 8192: "^"}[ctx],
                   edgecolor="white", linewidth=0.35, alpha=0.85, label=f"{ctx // 1024}K, r={correlations[ctx]:.2f}")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Draft-token acceptance rate")
    ax.set_ylabel(r"Decode ratio $T_{AR}/T_{Spec}$")
    ax.set_ylim(0.2, 1.05)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3)
    ax.text(0.02, 0.97, "c", transform=ax.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.43)
    save_figure(fig, out, "confirmatory_main_results")


def validity_figure(current_pairs: list[dict[str, str]], current_blocks: list[dict[str, str]],
                    old_blocks: list[dict[str, str]], boundary: list[dict[str, object]], out: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45))

    ax = axes[0]
    old = {(int(row["ctx"]), row["mode"]): float(row["median_block1_vs_block0_pct"]) for row in old_blocks}
    cur_groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in current_blocks:
        cur_groups[(int(row["ctx"]), row["mode"])].append(float(row["block1_vs_block0_pct"]))
    current = {key: float(np.median(values)) for key, values in cur_groups.items()}
    labels = ["AR", "SpecDec"]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, [old[(4096, mode)] for mode in labels], width, color="#999999", label="Earlier formal")
    ax.bar(x + width / 2, [current[(4096, mode)] for mode in labels], width, color="#0072B2", label="Confirmatory")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("4K block-1 shift (%)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1]
    same = []
    six = []
    for ctx in CONTEXTS:
        rows = [row for row in current_pairs if int(row["ctx"]) == ctx]
        same.append(sum(row["same_output"].lower() == "true" for row in rows))
        six.append(sum(row["timing_six_checks_ok"].lower() == "true" for row in rows))
    x = np.arange(3)
    ax.bar(x - width / 2, same, width, color="#009E73", label="Same AR/Spec output")
    ax.bar(x + width / 2, six, width, color="#CC79A7", label="All six timing checks")
    ax.set_xticks(x, ["2K", "4K", "8K"])
    ax.set_ylim(0, 17)
    ax.set_ylabel("Families (of 16)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[2]
    spec = {int(row["ctx"]): row for row in boundary if row["mode"] == "SpecDec"}
    qualified = [int(spec[ctx]["qualified_reps"]) for ctx in CONTEXTS]
    ax.bar(np.arange(3), qualified, color=[COLORS[ctx] for ctx in CONTEXTS], width=0.62)
    ax.axhline(80, color="#555555", linestyle="--", linewidth=0.8)
    for index, value in enumerate(qualified):
        ax.text(index, value + 1.5, f"{value}/80", ha="center", fontsize=7)
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylim(0, 88)
    ax.set_ylabel("Boundary-clock qualified reps")
    ax.text(0.02, 0.97, "c", transform=ax.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.42)
    save_figure(fig, out, "confirmatory_measurement_validity")


def mechanism_figure(cost: list[dict[str, str]], operator: list[dict[str, str]], cuda: list[dict[str, str]], out: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.2))

    ax = axes[0]
    families = [row["family"] for row in cost]
    stages = [
        ("Draft proposal", "draft_propose_pct", "#0072B2"),
        ("Target verify", "target_verify_pct", "#D55E00"),
        ("Repair", None, "#CC79A7"),
        ("Residual", "residual_pct", "#999999"),
    ]
    bottoms = np.zeros(len(cost))
    for label, field, color in stages:
        if label == "Repair":
            values = np.array([float(row["repair_target_pct"]) + float(row["repair_draft_pct"]) for row in cost])
        else:
            values = np.array([float(row[field]) for row in cost])
        ax.bar(families, values, bottom=bottoms, color=color, width=0.66, label=label)
        bottoms += values
    ax.set_ylim(0, 100)
    ax.set_ylabel("Instrumented decode time (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=6.2)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[1]
    metrics = {(row["evidence"], row["metric"]): float(row["value"]) for row in operator}
    busy = metrics[("nsight_propose", "gpu_kernel_busy_union")]
    gap = metrics[("nsight_propose", "non_kernel_covered_gap")]
    ax.barh([0], [busy], color="#009E73", label="GPU kernel busy")
    ax.barh([0], [gap], left=[busy], color="#E69F00", label="Uncovered gaps")
    ax.set_yticks([])
    ax.set_xlabel("Profiled proposal span (ms)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=6.2)
    ax.text(busy / 2, 0, f"{busy / (busy + gap):.0%}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(busy + gap / 2, 0, f"{gap / (busy + gap):.0%}", ha="center", va="center", color="black", fontweight="bold")
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontweight="bold", va="top")

    ax = axes[2]
    by = {(row["model"], row["arm"]): float(row["seq_median_ms"]) for row in cuda}
    x = np.arange(2)
    width = 0.34
    eager = [by[(model, "B")] for model in ("target", "draft")]
    replay = [by[(model, "C")] for model in ("target", "draft")]
    ax.bar(x - width / 2, eager, width, color="#777777", label="Eager")
    ax.bar(x + width / 2, replay, width, color="#56B4E9", label="CUDA Graph replay")
    ax.set_xticks(x, ["Target", "Draft"])
    ax.set_ylabel("16-step sequence (ms)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=6.2)
    for index, (base, optimized) in enumerate(zip(eager, replay)):
        ax.text(index, max(base, optimized) + 20, f"−{(1 - optimized / base) * 100:.1f}%", ha="center", fontsize=7)
    ax.text(0.02, 0.97, "c", transform=ax.transAxes, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.44, bottom=0.30)
    save_figure(fig, out, "confirmatory_operator_mechanism")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    batch = args.batch.resolve()
    docs = args.docs.resolve()
    measurements = docs / "measurements"
    figures = docs / "figures" / "confirmatory_final"

    summary = verify_batch(batch)
    outputs = copy_analysis(batch, measurements)
    current_pairs = read_csv(outputs["formal_pair_summary.csv"])
    contexts = read_csv(outputs["formal_context_summary.csv"])
    current_blocks = read_csv(outputs["formal_block_shifts.csv"])
    old_pairs = read_csv(measurements / "formal_family_pairs_2026-10-24.csv")
    old_blocks = read_csv(measurements / "block_shift_summary_2026-10-24.csv")

    comparison_path = measurements / f"confirmatory_old_new_comparison_{DATE}.csv"
    make_old_new_table(current_pairs, old_pairs, comparison_path)
    outputs["old_new_comparison"] = comparison_path
    boundary = boundary_counts(batch)
    boundary_path = measurements / f"confirmatory_boundary_counts_{DATE}.csv"
    write_csv(boundary_path, boundary)
    outputs["boundary_counts"] = boundary_path

    old_new = read_csv(comparison_path)
    current_main_figure(current_pairs, old_new, contexts, figures)
    validity_figure(current_pairs, current_blocks, old_blocks, boundary, figures)
    mechanism_figure(
        read_csv(measurements / "cost_profile_2k_2026-10-23.csv"),
        read_csv(measurements / "operator_evidence_2026-10-24.csv"),
        read_csv(measurements / "cudagraph_chain_corrected_2026-10-24.csv"),
        figures,
    )
    manifest_path = aggregate_manifest(batch, summary, outputs, measurements)
    print(json.dumps({
        "status": "ok",
        "batch": summary["batch_id"],
        "tasks": 192,
        "measured": 480,
        "manifest": str(manifest_path),
        "figures": str(figures),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
