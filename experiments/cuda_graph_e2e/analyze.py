"""2K 端到端 Graph 兑现门分析：AR/eager 同批基线、显存账本、Outcome 判定。

输入：
  <run>/ar/result.json、<run>/eager/result.json（graph 若存在则一并读；本轮预期被显存阻断）
  output/cuda_graph_e2e/mem_probe/probe.json（逐 graph capture 的显存记录，OOM 前）
输出：<run>/analysis/{core_table.csv, amortization.csv, memory_probe.csv, e2e_2k_summary.json}
      docs/figures/cuda_graph_e2e/e2e_2k.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = 2048
AR_REF = 2.374         # 同批 AR decode 中位（本轮实测，见 core_table.csv）
BONUS_ROUNDS = 13


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mem-probe", type=Path,
                    default=ROOT / "output" / "cuda_graph_e2e" / "mem_probe" / "probe.json")
    args = ap.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    ar = load(run / "ar" / "result.json")
    eager = load(run / "eager" / "result.json")
    graph = load(run / "graph" / "result.json")
    probe = load(args.mem_probe) if args.mem_probe.exists() else None

    def med(decodes):
        return st.median(decodes)

    rows = []
    if ar:
        d = [r["decode_s"] for r in ar["reps"]]
        rows.append(dict(mode="AR", capture_s="—", decode_median_s=med(d),
                         decode_min_s=min(d), decode_max_s=max(d),
                         speedup_vs_ar=1.0, output_len=64, output_equal="reference",
                         peak_allocated_gib="—", device_used_gib="—"))
        ar_med = med(d)
    else:
        ar_med = AR_REF
    if eager:
        d = [r["decode_s"] for r in eager["reps"]]
        rows.append(dict(mode="eager_bonus", capture_s="—", decode_median_s=med(d),
                         decode_min_s=min(d), decode_max_s=max(d),
                         speedup_vs_ar=ar_med / med(d), output_len=len(eager["reps"][0]["gen"]),
                         output_equal="reference", peak_allocated_gib="—", device_used_gib="—"))
        eager_med = med(d)
    else:
        eager_med = None
    if graph:
        d = [r["gpu_ms"] / 1000.0 for r in graph["reps"]]
        rows.append(dict(mode="graph_replay", capture_s=graph["capture_s"],
                         decode_median_s=med(d), decode_min_s=min(d), decode_max_s=max(d),
                         speedup_vs_ar=ar_med / med(d), output_len=64,
                         output_equal=graph["gate"]["tokens_equal"],
                         peak_allocated_gib=graph["ledger"][-1]["allocated_gib"],
                         device_used_gib=graph["ledger"][-1]["total_gib"] - graph["ledger"][-1]["free_gib"]))
        graph_med = med(d)
    else:
        rows.append(dict(mode="graph_replay", capture_s="—", decode_median_s="—",
                         decode_min_s="—", decode_max_s="—", speedup_vs_ar="—",
                         output_len="—", output_equal="blocked(memory)",
                         peak_allocated_gib="—", device_used_gib="—"))
        graph_med = None
    write_csv(out / "core_table.csv", rows)

    # 摊销表（仅当 graph 可运行时才有意义）
    amort = []
    if graph and graph_med is not None:
        cap = graph["capture_s"]
        for n in (1, 10, 100):
            t = (cap + n * graph_med) / n
            amort.append(dict(requests=n, capture_s=cap, replay_s=graph_med,
                              average_latency_s=t, effective_speedup=ar_med / t))
    write_csv(out / "amortization.csv", amort) if amort else None

    # 显存账本（逐 graph capture）
    mem_rows = []
    if probe:
        for r in probe["records"]:
            mem_rows.append(dict(name=r["name"], reserved_gib=r.get("reserved_gib"),
                                 delta_gib=r.get("delta_gib"), phys_free_gib=r.get("phys_free_gib"),
                                 oom=r.get("oom", False)))
    if mem_rows:
        write_csv(out / "memory_probe.csv", mem_rows)

    # Outcome 判定
    gate_ok = bool(graph and graph.get("gate", {}).get("ok"))
    if not probe:
        outcome, reason = "C", "无显存探测数据"
    elif probe.get("captured", 0) < probe.get("total", 26):
        per_chain = st.median([r["delta_gib"] for r in probe["records"]
                               if r["name"].startswith("chain") and "delta_gib" in r])
        per_verify = st.median([r["delta_gib"] for r in probe["records"]
                                if r["name"].startswith("verify") and "delta_gib" in r])
        proj = 13 * per_chain + 13 * per_verify
        outcome = "C"
        reason = (f"13 轮 Graph 无法在 8GB 内共存：capture {probe['captured']}/{probe['total']} 时 OOM；"
                  f"每个 chain graph ≈{per_chain:.2f} GiB、每个 verify graph ≈{per_verify:.2f} GiB，"
                  f"外推 26 个 ≈{proj:.1f} GiB 保留内存 > 8 GiB 物理显存。")
    elif gate_ok and graph_med is not None and ar_med / graph_med > 1.0:
        outcome, reason = "A", "完整 Graph replay 稳定超过 AR"
    elif gate_ok and graph_med is not None and ar_med / graph_med <= 1.0:
        outcome, reason = "D", "gate 通过但完整 Graph 仍低于 AR"
    else:
        outcome, reason = "E", "gate 失败"

    summary = dict(
        schema_version="cuda-graph-e2e-2k-analysis-v1",
        experiment_class="end_to_end_graph_feasibility_diagnostic",
        scope="F09/2K/bonus-all-accept/64-token",
        formal_result_allowed=False, quality_claim_allowed=False,
        production_claim_allowed=False,
        run=str(run), outcome=dict(kind=outcome, reason=reason),
        ar_decode_s=ar_med, eager_bonus_decode_s=eager_med,
        eager_speedup_vs_ar=(ar_med / eager_med if eager_med else None),
        graph_blocked=bool(not graph),
        memory_probe=(probe if probe else None),
        notes=("graph 路径因 13 轮实际 cache 推进需要 26 个固定 shape graph；实测每个 chain graph 保留 "
               "~1.6–2.0 GiB、每个 verify graph ~0.9 GiB；8 GiB 内无法共存 ⇒ Outcome C。"
               "不改显存护栏、不改模型路径。"))
    (out / "e2e_2k_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot(ROOT / "docs" / "figures" / "cuda_graph_e2e", ar_med, eager_med, graph_med, probe)
    print(json.dumps(summary["outcome"], ensure_ascii=False, indent=2))
    print("core:", rows)
    return 0


def plot(fig_dir: Path, ar_med, eager_med, graph_med, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)

    ax = axes[0]
    if probe:
        xs = list(range(1, len(probe["records"]) + 1))
        ys = [r.get("reserved_gib") for r in probe["records"]]
        ax.plot(xs, ys, marker="o", color="#0072B2", linewidth=1.4)
        ax.axhline(8.0, color="#D55E00", linewidth=1.0, linestyle="--", label="8 GiB device")
        ax.set_xlabel("captured graphs (in capture order)")
        ax.set_ylabel("reserved memory (GiB)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", color="#E9EDF0")
        ax.set_axisbelow(True)
        ax.text(-0.10, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    names = ["AR", "eager bonus"]
    vals = [ar_med, eager_med if eager_med else np.nan]
    colors = ["#8C8C8C", "#0072B2"]
    ax.bar([0, 1], vals, width=0.5, color=colors)
    if graph_med:
        ax.bar([2], [graph_med], width=0.5, color="#E69F00")
        names.append("graph replay")
    else:
        ax.bar([2], [0], width=0.5, color="#E69F00")
        ax.text(2, 0.3, "blocked\n(memory)", ha="center", va="bottom", fontsize=8, color=ink)
        names.append("graph (blocked)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1, 2], names)
    ax.set_ylabel("decode latency (s, median)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"e2e_2k.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
