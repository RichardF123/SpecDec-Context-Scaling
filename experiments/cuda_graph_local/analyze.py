"""局部 CUDA Graph A/B 分析：组件表、Amdahl 投影、图、Outcome。

输入：<run>/<family>_<ctx>/result.json（benchmark.py 产出）
输出：<run>/analysis/{components.csv, amdahl.csv, cuda_graph_local_summary.json}
      docs/figures/cuda_graph_local/local_ab.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("F09", "F04")
CONTEXTS = (2048, 4096, 8192)
BONUS_DECODE = {2048: 2.603, 4096: 2.555, 8192: 2.551}   # 同批 bonus oracle（2026-10-26）
AR_DECODE = {2048: 2.374, 4096: 2.365, 8192: 2.230}
COMPONENT_ORDER = ("A1", "A2", "A3", "B1", "B2")


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_run(run: Path) -> list:
    combos = []
    for combo_dir in sorted(run.iterdir()):
        if not combo_dir.is_dir():
            continue
        path = combo_dir / "result.json"
        if path.exists():
            combos.append(json.loads(path.read_text(encoding="utf-8")))
    return combos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    combos = load_run(run)

    rows = []
    for combo in combos:
        for comp in combo["components"]:
            g = comp["graph"]
            rows.append(dict(
                family=combo["family"], context=combo["context"],
                component=comp["component"],
                eager_ms=g["eager"]["median_ms"],
                graph_kernel_ms=g["replay_kernel_only"]["median_ms"],
                graph_plus_copy_ms=g["replay_plus_copy"]["median_ms"],
                static_copy_ms=g["static_copy"]["median_ms"],
                capture_ms=comp["capture_ms"], first_replay_ms=comp["first_replay_ms"],
                kernel_speedup=g["eager"]["median_ms"] / g["replay_kernel_only"]["median_ms"],
                effective_speedup=g["eager"]["median_ms"] / g["replay_plus_copy"]["median_ms"],
                eager_min_ms=g["eager"]["min_ms"], eager_max_ms=g["eager"]["max_ms"],
                replay_plus_copy_max_ms=g["replay_plus_copy"]["max_ms"],
                graph_kernel_min_ms=g["replay_kernel_only"]["min_ms"],
                pool_alloc_delta_mib=comp["pool_alloc_delta"] / 2**20,
                pool_reserved_delta_mib=comp["pool_reserved_delta"] / 2**20,
                peak_reserved_gib=comp["peak_reserved_gib"],
                gate_ok=comp["gate"]["ok"],
                gate_bitwise_logits=comp["gate"]["logits"]["bitwise"],
                gate_max_logit_diff=comp["gate"]["logits"]["max_abs_diff"],
                gate_argmax_equal=comp["gate"]["logits"]["argmax_equal"],
                gate_cache_bitwise=comp["gate"]["cache_bitwise"],
                gate_input_update=comp["gate"]["input_update_ok"],
                gate_deterministic=comp["gate"]["deterministic_logits"],
            ))
    rows.sort(key=lambda r: (r["context"], r["component"], r["family"]))
    write_csv(out / "components.csv", rows)

    # 按 ctx 聚合（A3 = draft 一轮近似；B2 = verify M=5）
    by_ctx = {}
    for ctx in CONTEXTS:
        subset = [r for r in rows if r["context"] == ctx]
        agg = {}
        for name in COMPONENT_ORDER:
            comp = [r for r in subset if r["component"] == name]
            agg[name] = dict(
                eager_ms=st.median(r["eager_ms"] for r in comp),
                graph_kernel_ms=st.median(r["graph_kernel_ms"] for r in comp),
                graph_plus_copy_ms=st.median(r["graph_plus_copy_ms"] for r in comp),
                kernel_speedup=st.median(r["kernel_speedup"] for r in comp),
                effective_speedup=st.median(r["effective_speedup"] for r in comp),
            )
        by_ctx[ctx] = agg

    # Amdahl 投影（归一化口径）：
    #   真实 per-round 时间取自 bonus oracle 实测（decode/13）；
    #   微基准只提供"draft/verify 占比"与"graph 加速比"，避免把 8K 微基准的
    #   异常绝对值（内存压力导致 eager 膨胀）直接外推。
    # 同时给出未归一化的 fixed-history 投影（带 negative-residual 提示）作为对照。
    amdahl = []
    for ctx in CONTEXTS:
        rounds = 13
        obs_round_ms = BONUS_DECODE[ctx] * 1000.0 / rounds
        stats = []
        for fam in FAMILIES:
            sub = [r for r in rows if r["family"] == fam and r["context"] == ctx]
            a3 = next(r for r in sub if r["component"] == "A3")
            b2 = next(r for r in sub if r["component"] == "B2")
            stats.append(dict(
                draft_share=a3["eager_ms"] / (a3["eager_ms"] + b2["eager_ms"]),
                r_draft_k=a3["eager_ms"] / a3["graph_kernel_ms"],
                r_verify_k=b2["eager_ms"] / b2["graph_kernel_ms"],
                r_draft_e=a3["eager_ms"] / a3["graph_plus_copy_ms"],
                r_verify_e=b2["eager_ms"] / b2["graph_plus_copy_ms"],
                r_draft_c=a3["eager_min_ms"] / a3["replay_plus_copy_max_ms"],
                r_verify_c=b2["eager_min_ms"] / b2["replay_plus_copy_max_ms"],
            ))
        med = lambda key: st.median(s[key] for s in stats)
        share = med("draft_share")
        def proj_ms(r_draft, r_verify):
            return obs_round_ms * (share / r_draft + (1 - share) / r_verify)
        proj_opt = rounds * proj_ms(med("r_draft_k"), med("r_verify_k")) / 1000.0
        proj_real = rounds * proj_ms(med("r_draft_e"), med("r_verify_e")) / 1000.0
        proj_cons = rounds * proj_ms(med("r_draft_c"), med("r_verify_c")) / 1000.0
        # 未归一化对照（fixed-history）：用微基准绝对时间
        a3_e = st.median(r["eager_ms"] for r in rows if r["context"] == ctx and r["component"] == "A3")
        b2_e = st.median(r["eager_ms"] for r in rows if r["context"] == ctx and r["component"] == "B2")
        a3_k = st.median(r["graph_kernel_ms"] for r in rows if r["context"] == ctx and r["component"] == "A3")
        b2_k = st.median(r["graph_kernel_ms"] for r in rows if r["context"] == ctx and r["component"] == "B2")
        raw_round_ms = a3_e + b2_e
        residual_raw = BONUS_DECODE[ctx] * 1000.0 - rounds * raw_round_ms
        amdahl.append(dict(
            context=ctx, rounds=rounds,
            bonus_decode_s=BONUS_DECODE[ctx], AR_decode_s=AR_DECODE[ctx],
            observe_speedup=AR_DECODE[ctx] / BONUS_DECODE[ctx],
            observed_round_ms=obs_round_ms,
            draft_share_median=share,
            draft_graph_speedup_kernel=med("r_draft_k"),
            verify_graph_speedup_kernel=med("r_verify_k"),
            draft_graph_speedup_effective=med("r_draft_e"),
            verify_graph_speedup_effective=med("r_verify_e"),
            projected_speedup_optimistic=AR_DECODE[ctx] / proj_opt,
            projected_speedup_realistic=AR_DECODE[ctx] / proj_real,
            projected_speedup_conservative=AR_DECODE[ctx] / proj_cons,
            break_even_required_reduction_pct=100 * (1 - AR_DECODE[ctx] / BONUS_DECODE[ctx]),
            raw_fixed_history_round_ms=raw_round_ms,
            raw_fixed_history_residual_ms=residual_raw,
            draft_effective_gain_pct=100 * (1 - 1.0 / med("r_draft_e")),
            verify_effective_gain_pct=100 * (1 - 1.0 / med("r_verify_e")),
        ))
    write_csv(out / "amdahl.csv", amdahl)

    # Outcome
    gates_ok = all(r["gate_ok"] for r in rows)
    kernel_gain = st.median(r["kernel_speedup"] for r in rows)
    eff_gain = st.median(r["effective_speedup"] for r in rows)
    proj_real = st.median(a["projected_speedup_realistic"] for a in amdahl)
    proj_cons = st.median(a["projected_speedup_conservative"] for a in amdahl)
    if not gates_ok:
        outcome, explanation = "E", "存在 gate 失败（详见 components.csv）"
    elif proj_real > 1.0 and proj_cons >= 0.98:
        outcome, explanation = "A", "realistic 投影 >1 且 conservative 接近/超过 1：launch 是主要瓶颈，值得集成。"
    elif eff_gain > 1.02 and proj_real > 0.98:
        outcome, explanation = "B", "局部 graph 有效（launch 是真实瓶颈之一），但整机投影仍 <1。"
    elif kernel_gain > 1.10 and eff_gain <= 1.02:
        outcome, explanation = "D", "kernel-only 有收益，但加入 copy/reset 后收益基本消失（动态 cache 架构限制）。"
    elif eff_gain <= 1.05:
        outcome, explanation = "C", "graph effective 收益很小（<5%）：主要不是 CPU launch，而是计算/KV/内存系统成本。"
    else:
        outcome, explanation = "B", "局部 graph 有效但不足以稳定跨 1。"

    summary = dict(
        schema_version="cuda-graph-local-analysis-v1",
        experiment_class="operator_mechanism_diagnostic",
        formal_result_allowed=False, quality_claim_allowed=False,
        full_specdec_speedup_claim_allowed=False,
        run=str(run), components=rows, by_ctx=by_ctx, amdahl=amdahl,
        outcome=dict(kind=outcome, explanation=explanation,
                     kernel_speedup_median=kernel_gain,
                     effective_speedup_median=eff_gain,
                     projected_speedup_realistic_median=proj_real,
                     projected_speedup_conservative_median=proj_cons),
        notes=("历史长度固定为 ctx；token 来自真实 bonus 轨迹；局部算子/固定 shape 口径。"
               "kernel-only 与 effective（含 static copy）分开报告；Amdahl 用历史=ctx 的局部时间，"
               "对 bonus 循环后段（历史更长）偏乐观，见报告局限。"),
    )
    (out / "cuda_graph_local_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot(ROOT / "docs" / "figures" / "cuda_graph_local", by_ctx, amdahl)

    print(json.dumps(dict(by_ctx=by_ctx, amdahl=amdahl, outcome=summary["outcome"]),
                     ensure_ascii=False, indent=2, default=str))
    return 0


def plot(figure_dir: Path, by_ctx: dict, amdahl: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ctx_labels = ["2K", "4K", "8K"]
    xs = list(range(3))
    colors = {"eager": "#8C8C8C", "kernel": "#0072B2", "copy": "#E69F00"}
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

    for ax, comp, tag in ((axes[0][0], "A3", "a"), (axes[0][1], "B2", "b")):
        width = 0.26
        vals = {"eager": [], "kernel": [], "copy": []}
        for ctx in (2048, 4096, 8192):
            vals["eager"].append(by_ctx[ctx][comp]["eager_ms"])
            vals["kernel"].append(by_ctx[ctx][comp]["graph_kernel_ms"])
            vals["copy"].append(by_ctx[ctx][comp]["graph_plus_copy_ms"])
        for i, (k, label) in enumerate((("eager", "eager"), ("kernel", "graph kernel-only"),
                                        ("copy", "graph + static copy"))):
            ax.bar([x + (i - 1) * width for x in xs], vals[k], width=width,
                   color=colors[k], label=label)
        ax.set_xticks(xs, ctx_labels)
        ax.set_ylabel("per-op latency (ms)")
        ax.grid(axis="y", color="#E9EDF0")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{comp} (draft chain)" if comp == "A3" else f"{comp} (verify M=5)",
                     fontsize=9, color=ink)
        ax.text(-0.12, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][0]
    for key, label, color in (("projected_speedup_optimistic", "optimistic", "#0072B2"),
                              ("projected_speedup_realistic", "realistic", "#E69F00"),
                              ("projected_speedup_conservative", "conservative", "#D55E00")):
        ax.plot(xs, [a[key] for a in amdahl], marker="o", color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(xs, ctx_labels)
    ax.set_ylabel("projected speedup (AR/Spec)")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1][1]
    comps = list(COMPONENT_ORDER)
    kernel = [st.median(by_ctx[c][x]["kernel_speedup"] for c in (2048, 4096, 8192)) for x in comps]
    eff = [st.median(by_ctx[c][x]["effective_speedup"] for c in (2048, 4096, 8192)) for x in comps]
    width = 0.36
    ax.bar([i - width / 2 for i in range(len(comps))], kernel, width=width,
           color=colors["kernel"], label="kernel-only")
    ax.bar([i + width / 2 for i in range(len(comps))], eff, width=width,
           color=colors["copy"], label="with static copy")
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(range(len(comps)), comps)
    ax.set_ylabel("speedup vs eager")
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(figure_dir / f"local_ab.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
