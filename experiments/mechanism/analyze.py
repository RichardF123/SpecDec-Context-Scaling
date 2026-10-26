"""Draft-Free oracle 分析：AR / perfect+bonus / draft-free 三路径，family-level 中位与散点。

输出：<run>/analysis/{core_table.csv, family_table.csv, draft_free_summary.json}
      docs/figures/draft_free_oracle/draft_free.{png,pdf,svg}
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
# strict SpecDec 参照（relaxed Stage B 同配置、跨批次；只作参照，不与本批直接算精确比值）
STRICT_REF = {(2048): 4.69, 4096: 4.66, 8192: 5.59}


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"empty csv {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_result(run: Path, family: str, ctx: int, mode: str):
    p = run / f"{family}_{ctx}_{mode}" / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    rows = []
    missing = []
    for fam in FAMILIES:
        for ctx in CONTEXTS:
            ar = load_result(run, fam, ctx, "ar")
            bn = load_result(run, fam, ctx, "bonus")
            df = load_result(run, fam, ctx, "draft_free")
            if ar is None or bn is None or df is None:
                missing.append((fam, ctx))
                continue
            ar_d = st.median(r["decode_s"] for r in ar["reps"])
            bn_d = st.median(r["decode_s"] for r in bn["reps"])
            df_d = st.median(r["decode_s"] for r in df["reps"])
            gate = df.get("gate_ok")
            checks = df.get("structure_checks", {})
            rows.append(dict(
                family=fam, context=ctx,
                AR_decode_s=ar_d, bonus_decode_s=bn_d, draft_free_decode_s=df_d,
                S_AR_over_draftfree=ar_d / df_d,
                S_AR_over_bonus=ar_d / bn_d,
                bonus_over_draftfree=bn_d / df_d,
                strict_ref_decode_s=STRICT_REF.get(ctx),
                strict_ref_S=STRICT_REF.get(ctx) / df_d if STRICT_REF.get(ctx) else None,
                rounds=checks.get("verification_rounds"),
                target_block_calls=checks.get("target_block_calls"),
                draft_forward_calls=df["reps"][0]["draft_forward_calls"],
                generated_length=len(df["generated_ids"]),
                gate_ok=gate,
                gen_equals_frozen=checks.get("gen_equals_frozen"),
                repeat_consistent=checks.get("repeat_consistent"),
                peak_allocated_gib=max(r["mem"]["allocated_gib"] for r in df["reps"]),
                ar_peak_allocated_gib=max(r["mem"]["allocated_gib"] for r in ar["reps"])
                if ar["reps"] and "mem" in ar["reps"][0] else None,
            ))
    write_csv(out / "core_table.csv", rows)

    # ctx 汇总（family 中位）
    context_summary = []
    for ctx in CONTEXTS:
        sub = [r for r in rows if r["context"] == ctx]
        if not sub:
            continue
        context_summary.append(dict(
            context=ctx, families=len(sub),
            AR_median_s=st.median(r["AR_decode_s"] for r in sub),
            bonus_median_s=st.median(r["bonus_decode_s"] for r in sub),
            draft_free_median_s=st.median(r["draft_free_decode_s"] for r in sub),
            S_draftfree_median=st.median(r["S_AR_over_draftfree"] for r in sub),
            S_draftfree_min=min(r["S_AR_over_draftfree"] for r in sub),
            S_draftfree_max=max(r["S_AR_over_draftfree"] for r in sub),
            S_bonus_median=st.median(r["S_AR_over_bonus"] for r in sub),
            draft_free_speedup_vs_bonus=st.median(r["bonus_over_draftfree"] for r in sub),
            rounds_median=st.median(r["rounds"] for r in sub),
            gate_ok_all=all(r["gate_ok"] for r in sub),
        ))
    write_csv(out / "family_table.csv", rows)

    # 判定
    if missing:
        outcome, reason = "D", f"缺失条件：{missing}"
    elif not all(r["gate_ok"] for r in rows):
        outcome, reason = "D", "结构门失败"
    else:
        meds = {c["context"]: c["S_draftfree_median"] for c in context_summary}
        if all(v > 1.0 for v in meds.values()):
            outcome, reason = "A", ("删除 draft decode 后，target verify + bonus + cache 控制路径在所有 ctx 均 >1"
                                    "（S 中位：" + ", ".join(f"{k // 1024}K={v:.2f}" for k, v in meds.items()) + "）")
        elif all(v < 1.0 for v in meds.values()):
            outcome, reason = "B", ("即使候选 token 零成本，target verify/cache/控制仍不足以超过 AR（S 中位："
                                    + ", ".join(f"{k // 1024}K={v:.2f}" for k, v in meds.items()) + "）")
        elif meds.get(2048, 0) > 1.0 and (meds.get(4096, 0) < 1.0 or meds.get(8192, 0) < 1.0):
            outcome, reason = "C", ("2K 超过 1，但 4K/8K 未超过：删除 draft 成本可在短上下文恢复收益，"
                                    "更长上下文 verify/KV 成本重新吞噬（S 中位："
                                    + ", ".join(f"{k // 1024}K={v:.2f}" for k, v in meds.items()) + "）")
        else:
            outcome, reason = "D", ("结果混合但不符合 A/B/C 的明确模式（S 中位："
                                    + ", ".join(f"{k // 1024}K={v:.2f}" for k, v in meds.items()) + "）")
    summary = dict(
        schema_version="draft-free-oracle-analysis-v1",
        experiment_class="mechanism_diagnostic", deployable=False,
        formal_result_allowed=False, quality_claim_allowed=False,
        run=str(run), core=rows, contexts=context_summary, missing=missing,
        outcome=dict(kind=outcome, reason=reason),
        notes=("strict 参照来自 relaxed Stage B（跨批次，仅参考，不参与本批精确比值）；"
               "AR/bonus/draft-free 为同批测量；本轮是 lower-bound oracle，不可部署、无质量声明。"))
    (out / "draft_free_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot(ROOT / "docs" / "figures" / "draft_free_oracle", rows, context_summary)
    print(json.dumps(dict(contexts=context_summary, outcome=summary["outcome"]),
                     ensure_ascii=False, indent=2, default=str))
    return 0


def plot(fig_dir: Path, rows: list, ctx_summary: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#293F48", "#6D7E86"
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ctxs = [c["context"] for c in ctx_summary]
    labels = [f"{c // 1024}K" for c in ctxs]
    xs = list(range(len(ctxs)))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    ax = axes[0]
    width = 0.26
    names = ["AR", "perfect+bonus", "draft-free"]
    colors = ["#8C8C8C", "#0072B2", "#E69F00"]
    keys = ["AR_median_s", "bonus_median_s", "draft_free_median_s"]
    for i, (name, key, color) in enumerate(zip(names, keys, colors)):
        vals = [c[key] for c in ctx_summary]
        ax.bar([x + (i - 1) * width for x in xs], vals, width=width, color=color, label=name)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("decode latency (s, family median)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    for key, name, color in (("S_draftfree_median", "AR / draft-free", "#E69F00"),
                             ("S_bonus_median", "AR / perfect+bonus", "#0072B2")):
        vals = [c[key] for c in ctx_summary]
        ax.plot(xs, vals, marker="o", color=color, linewidth=1.5, label=name)
        for c in ctx_summary:
            fam_vals = [r["S_AR_over_draftfree"] if key == "S_draftfree_median" else r["S_AR_over_bonus"]
                        for r in rows if r["context"] == c["context"]]
            ax.scatter([xs[ctxs.index(c["context"])]] * len(fam_vals), fam_vals, s=12,
                       color=color, alpha=0.5, linewidths=0)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("T_AR / T_path")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color="#E9EDF0")
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"draft_free.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
