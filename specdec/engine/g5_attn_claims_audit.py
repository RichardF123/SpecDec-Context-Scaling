"""g5_attn_claims_audit.py — **报告主张 ↔ 原始结果** 的机器可核对审计（不需要 GPU）。

为什么需要它：报告里的每个数字都来自某个 JSON，但"报告说 X"与"数据里真的是 X"是两件事。
本工具把阶段报告 §5/§6/§9 与台账 §4.22/§4.23 的**每一条可量化主张**写成断言，直接读原始结果复核。
它**不产生新数据**、**不重新解释**任何判据，只回答"这句话有没有出处、数字对不对"。

用法
  python -m specdec.engine.g5_attn_claims_audit
返回 0 = 全部一致；1 = 有不一致（会逐条打印期望/实际）。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATTN = ROOT / "output" / "g5" / "attn"
DISPLAY_MIB = 0.002        # 报告里 MiB 写到 0.001 位；审计按同一显示精度比较（不是放宽结论）

OPS = ATTN / "attn_cudnn_validate_20260926T175207Z.json"          # 算子级（§5）
MODEL = ATTN / "attn_cudnn_validate_20260926T175638Z.json"        # 模型级（§6）
RTF_BASE = ATTN / "rtf_base_bf16_20260926T183132Z.json"           # 真实张量 base（§9）
RTF_VFP8 = ATTN / "rtf_v_fp8_20260926T183224Z.json"               # 真实张量 v_fp8（§9）
RTF_PRE = ATTN / "rtf_base_bf16_20260926T182713Z.json"            # 加诊断开关**前**（回归基线）
NLL = ATTN / "nll_quality_20260926T183547Z.json"                  # 质量检查（§9.5/§9.6）
SMOKE = ATTN / "smoke8k_cudnn_prefill_20260926T184203Z.json"      # ctx=512 排练（不计入 8K 结论）

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'MISMATCH'}] {name}" + (f"   {detail}" if detail else ""))


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def close(a, b, tol=1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def close_rel(a, b, rtol=5e-4) -> bool:
    """报告里的数值只印 4 位有效数字 ⇒ 按 4 位有效数字比对（显示精度，不是放宽结论）。"""
    try:
        a, b = float(a), float(b)
        return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30)
    except (TypeError, ValueError):
        return False


# 报告 §5.3 表里印出的数值（4 位有效数字）—— 审计拿它去核对 JSON
REPORT_OP_ABS_REL = {
    ("target", "block2048_T2048_first"): (2.344e-2, 7.937e-3),
    ("target", "block2048_T8192_hist"): (1.465e-3, 1.240e-2),
    ("target", "block512_T8192_hist"): (9.766e-4, 1.031e-2),
    ("target", "block4_T8192_hist"): (8.545e-4, 1.277e-2),
    ("target", "block2_T2_first"): (1.562e-2, 4.255e-3),
    ("draft", "block2048_T2048_first"): (1.562e-2, 5.747e-3),
    ("draft", "block2048_T8192_hist"): (1.465e-3, 1.172e-2),
    ("draft", "block512_T8192_hist"): (8.545e-4, 8.373e-3),
    ("draft", "block4_T8192_hist"): (4.883e-4, 6.579e-3),
    ("draft", "block2_T2_first"): (1.562e-2, 5.155e-3),
}


def main() -> int:
    ops, model = load(OPS), load(MODEL)
    rb, rv, rpre = load(RTF_BASE), load(RTF_VFP8), load(RTF_PRE)
    nll, smoke = load(NLL), load(SMOKE)

    print("== §5 算子级：显存与路径 ==")
    if ops:
        rows = {(r["config"], r["row"]): r for r in ops["op_level"]}
        t = rows[("target", "block2048_T8192_hist")]
        d = rows[("draft", "block2048_T8192_hist")]
        check("8K 带历史块的 target 瞬时 1984.0 → 54.001 MiB",
              close(t["eager_transient_mib"], 1984.0, DISPLAY_MIB)
              and close(t["fused_transient_mib"], 54.001, DISPLAY_MIB),
              f"eager={t['eager_transient_mib']} fused={t['fused_transient_mib']}")
        check("8K 带历史块的 draft 瞬时 2284.0 → 51.501 MiB",
              close(d["eager_transient_mib"], 2284.0, DISPLAY_MIB)
              and close(d["fused_transient_mib"], 51.501, DISPLAY_MIB),
              f"eager={d['eager_transient_mib']} fused={d['fused_transient_mib']}")
        check("该行**不是**首块（首块应为 cache is None 的 2048×2048）",
              rows[("target", "block2048_T8192_hist")]["has_history"] is True
              and rows[("target", "block2048_T2048_first")]["has_history"] is False)
        check("路径判据 12/12 符合预登记期望",
              all(r["path_ok"] for r in ops["op_level"]),
              f"n={len(ops['op_level'])}")
        kp = ops.get("kernel_proof") or {}
        check("kernel 证据 = 真实 cuDNN 融合算子",
              kp.get("matched") is True
              and "cudnn" in " ".join(kp.get("fused_kernel_names") or []).lower(),
              str(kp.get("fused_kernel_names")))
        # ---- §5.3 数值表：报告里印的每个数都要能在 JSON 里对上 ----
        bad_num = []
        for (cfg, rname), (a_rep, r_rep) in REPORT_OP_ABS_REL.items():
            row = rows.get((cfg, rname))
            if row is None or not (close_rel(row["abs_err"], a_rep)
                                   and close_rel(row["rel_err"], r_rep)):
                bad_num.append((cfg, rname, row["abs_err"] if row else None,
                                row["rel_err"] if row else None))
        check("§5.3 数值表 10 行 abs/rel 全部与 JSON 一致（4 位有效数字）",
              not bad_num, str(bad_num[:3]))
        check("§5.3 判定：12 行里恰好 8 行 FAIL（预登记判据，未放宽）",
              sum(1 for r in ops["op_level"] if r["verdict"] == "FAIL") == 8
              and all(r["verdict"] == "PASS" for r in ops["op_level"]
                      if r["path"] == "eager"),
              f"FAIL={sum(1 for r in ops['op_level'] if r['verdict'] == 'FAIL')}")
    else:
        check("算子级结果存在", False, str(OPS))

    print("\n== §6 模型级：判据 FAIL 与阶段分流 ==")
    if model:
        ml = model.get("model_level") or []
        ctrls = [c for c in ml if c["kind"] == "control"]
        ab = [c for c in ml if c["kind"] == "ab"]
        check("自对照 eager-vs-eager 全部 abs == 0.0（噪声地板为 0）",
              bool(ctrls) and all(c["logits_abs_err"] == 0.0 for c in ctrls),
              f"n={len(ctrls)}")
        check("A/B 10/10 行 FAIL（预登记判据未通过，未放宽）",
              len(ab) == 10 and all(c["verdict"] == "FAIL" for c in ab), f"n={len(ab)}")
        div = [c for c in ab if c.get("divergence")]
        check("ctx=512/base_bf16 生成序列确实改变，且首位分歧 index=3",
              bool(div) and div[0]["divergence"]["index"] == 3,
              str(div[0]["divergence"]) if div else "no divergence recorded")
        check("阶段分流精确：所有 A/B 行 fused==52 且 short_fb 符合块数",
              all(c["fused_by_phase"].get("prefill") == 52 and c["fused_counts_exact"] for c in ab))
        check("每条 A/B 的 assert_strategy_used() 都是 ok",
              all(c["assert_cand"] == "ok" for c in ab))
    else:
        check("模型级结果存在", False, str(MODEL))

    print("\n== §9.1 方法可信性：自检 / 权重 / 默认路径未变 ==")
    for tag, r in (("base_bf16", rb), ("v_fp8", rv)):
        if not r:
            check(f"{tag} 真实张量结果存在", False)
            continue
        check(f"[{tag}] 逐层自检（复算 eager == 引擎真实输出）全部逐位相同",
              r["case_a"]["summary"]["selfcheck_all_bitmatch"] is True
              and r["case_b"]["summary"]["selfcheck_all_bitmatch"] is True)
        check(f"[{tag}] 加载权重逐位校验通过",
              r["load"]["weights_bitmatch"] is True and r["load"]["mismatch_keys"] == [],
              f"checked={r['load']['n_params_checked']}")
        check(f"[{tag}] FP32 参考显式关闭 matmul TF32",
              r["tf32_after_force_off"]["matmul_allow_tf32"] is False)
    if rb and rpre:
        same_a = [ (x["dist_vs_cudnn"], x["err_vs_fp32"], x["fp32_absmax"]) for x in rb["case_a"]["per_layer"] ] \
            == [ (x["dist_vs_cudnn"], x["err_vs_fp32"], x["fp32_absmax"]) for x in rpre["case_a"]["per_layer"] ]
        same_b = [ (x["dist_vs_cudnn"], x["err_vs_fp32"], x["fp32_absmax"]) for x in rb["case_b"]["per_layer"] ] \
            == [ (x["dist_vs_cudnn"], x["err_vs_fp32"], x["fp32_absmax"]) for x in rpre["case_b"]["per_layer"] ]
        check("加诊断开关前后：case_a/case_b 逐层数值完全不变（默认路径未变）",
              same_a and same_b, f"case_a={same_a} case_b={same_b}")

    print("\n== §9.2 逐层：R1 主导、R2 恰好为零 ==")
    for tag, r in (("A 无历史", rb), ("B 有历史", rv)):
        if not r:
            continue
    for tag, r in (("base_bf16", rb), ("v_fp8", rv)):
        if not r:
            continue
        for case, label in (("case_a", "无历史 s=T"), ("case_b", "有历史 offset")):
            c = r[case]
            pl = c["per_layer"]
            rem = c["summary"]["removal_effect"]
            check(f"[{tag}/{label}] R2（概率舍入）逐层移除效应中位**恰好 0**",
                  rem["R2_prob_bf16"]["median"] == 0.0, f"median={rem['R2_prob_bf16']['median']}")
            big = max(pl, key=lambda p: p["dist_vs_cudnn"]["eager"])
            ratio = big["dist_vs_cudnn"]["qk32"] / big["dist_vs_cudnn"]["fp32"]
            # 报告真正主张的是"**相对 eager 大幅缩小**"；"是否降到地板以下"是**数据**（比值），
            # 不作为通过条件 —— 首版审计把它当条件，正好暴露了报告里一处过度概括。
            check(f"[{tag}/{label}] 巨差点在第 0 层，去掉 R1 后相对 eager 缩小 ≥100×",
                  big["layer"] == 0
                  and big["dist_vs_cudnn"]["eager"] / big["dist_vs_cudnn"]["qk32"] >= 100,
                  f"layer={big['layer']} eager={big['dist_vs_cudnn']['eager']:.4g} "
                  f"qk32={big['dist_vs_cudnn']['qk32']:.4g} "
                  f"缩小={big['dist_vs_cudnn']['eager'] / big['dist_vs_cudnn']['qk32']:.1f}x "
                  f"qk32/floor={ratio:.3f}（>1 表示仍高于地板）")
            check(f"[{tag}/{label}] R3（AV 输出舍入）量级小于地板",
                  rem["R3_out_bf16"]["median"] < rem["irreducible_floor_fp32"]["median"],
                  f"R3={rem['R3_out_bf16']['median']:.4g} "
                  f"floor={rem['irreducible_floor_fp32']['median']:.4g}")

    print("\n== §9.3 整模型阶梯：R1 解释比例 ==")
    ladders = {}
    for tag, r in (("base_bf16", rb), ("v_fp8", rv)):
        if not r or "ladder" not in r:
            check(f"[{tag}] 整模型阶梯存在", False)
            continue
        s = r["ladder"]["summary"]
        ladders[tag] = s
        check(f"[{tag}] 去掉 R1 后相对 eager 缩小 ≥10×（报告实际主张）",
              s["eager_vs_cudnn"] / s["qk32_vs_cudnn"] >= 10,
              f"eager={s['eager_vs_cudnn']:.4g} qk32={s['qk32_vs_cudnn']:.4g} "
              f"缩小={s['eager_vs_cudnn'] / s['qk32_vs_cudnn']:.2f}x "
              f"qk32/floor={s['qk32_vs_cudnn'] / s['floor_fp32_vs_cudnn']:.3f}"
              f"（>1 表示仍高于地板，**不得**写「降到地板以下」）")
        check(f"[{tag}] 去掉 R2 几乎无变化（单项解释比例 ≈ 0）",
              abs(s["R2_explained_frac"]) < 0.05, f"R2_frac={s['R2_explained_frac']:.4f}")
        check(f"[{tag}] R3 在整模型层面被明确记为不可测",
              "无法" in (s.get("R3_limitation") or ""), (s.get("R3_limitation") or "")[:40])
    if len(ladders) == 2:
        b, v = ladders["base_bf16"], ladders["v_fp8"]
        d_base = b["floor_fp32_vs_cudnn"] - b["qk32_vs_cudnn"]      # <0 = 再去 R2 变差
        d_vf8 = v["floor_fp32_vs_cudnn"] - v["qk32_vs_cudnn"]       # >0 = 再去 R2 变好
        check("**[交互/不稳定]** 在 R1 已去掉的基础上再去掉 R2，改善方向在两种 kv 下**相反**"
              " ⇒ 残差不可归因（这是报告必须写明的一条）",
              (d_base < 0) != (d_vf8 < 0),
              f"base: {b['qk32_vs_cudnn']:.4g}→{b['floor_fp32_vs_cudnn']:.4g} "
              f"({'变好' if d_base > 0 else '变差'}); "
              f"v_fp8: {v['qk32_vs_cudnn']:.4g}→{v['floor_fp32_vs_cudnn']:.4g} "
              f"({'变好' if d_vf8 > 0 else '变差'})")

    print("\n== §9.4 轨迹分叉 ==")
    for tag, r in (("base_bf16", rb), ("v_fp8", rv)):
        if not r or "trajectory" not in r:
            continue
        t = r["trajectory"]
        first = next((p for p in t["per_layer"] if p["d_attn_out"] > 0), None)
        check(f"[{tag}] 首个分叉层 = 0 且该层输入逐位相同（d_x_ln == 0）",
              first is not None and first["layer"] == 0 and first["d_x_ln"] == 0.0,
              f"layer={first['layer'] if first else None} "
              f"d_x_ln={first['d_x_ln'] if first else None}")
        check(f"[{tag}] 有差异层数 = 56/56",
              sum(1 for p in t["per_layer"] if p["d_attn_out"] > 0) == 56)

    print("\n== §9.5 / §9.6 质量检查与泄漏 ==")
    if nll:
        P = {(p["predictor"], p.get("chunk")): p for p in nll["predictors"]}
        def g(name, chunk=None):
            for (n, c), p in P.items():
                if n == name and (chunk is None or c == chunk):
                    return p
            return None
        oe, he = g("ours_eager", 384), g("hf_eager")
        check("ours_eager 与 hf_eager 的 mean NLL **完全相同**",
              oe is not None and he is not None and oe["mean_nll"] == he["mean_nll"],
              f"ours={oe['mean_nll'] if oe else None} hf={he['mean_nll'] if he else None}")
        check("ours_eager 与 hf_eager 的**逐 token NLL 完全相同**",
              oe is not None and he is not None and oe["per_token_nll"] == he["per_token_nll"])
        check("一致性断言：N 唯一、被评分 token 哈希一致、logits 全有限",
              nll["consistency"]["n_consistent"] and nll["consistency"]["target_ids_consistent"]
              and nll["consistency"]["all_logits_finite"],
              str(nll["consistency"]))
        c = nll["causality_check"]
        check("泄漏检验 |B−C| **恰好为 0**（同长度、换未来 token）",
              c["B_vs_C_max_abs"] == 0.0, f"|B−C|={c['B_vs_C_max_abs']} "
                                        f"({c['B_vs_C_in_bf16_ulp']} ULP)")
        check("长度效应 |A−C| 被单独记录（不与泄漏混为一谈）",
              c["A_vs_C_max_abs"] > 0 and c["A_vs_C_in_bf16_ulp"] is not None,
              f"|A−C|={c['A_vs_C_max_abs']} ({c['A_vs_C_in_bf16_ulp']:.2f} ULP)")
        e, cc = g("ours_eager", 384), g("ours_cudnn_prefill", 384)
        hf, hs = g("hf_eager"), g("hf_sdpa")
        check("换后端的 NLL 影响与 HF 自身 eager↔sdpa 同量级",
              abs((e["mean_nll"] - cc["mean_nll"]) - (hf["mean_nll"] - hs["mean_nll"])) < 0.02,
              f"ours Δ={e['mean_nll'] - cc['mean_nll']:+.4f} "
              f"hf Δ={hf['mean_nll'] - hs['mean_nll']:+.4f}")
    else:
        check("质量检查结果存在", False, str(NLL))

    print("\n== 排练（ctx=512）：dual 驻留 + 阶段统计 ==")
    if smoke:
        tl = {b["tag"]: b for b in smoke.get("mem_timeline", [])}
        check("排练跑通且权重逐位校验通过",
              smoke.get("stage") == "complete" and smoke.get("weights_bitmatch") is True)
        check("排练 fused=52（1 块 × 28 层）且短块回退为 0",
              smoke["run"]["attention_stats"]["fused_by_phase"].get("prefill") == 52
              and smoke["run"]["attention_stats"]["eager_reason_kinds"]
                  .get("short_block_seq_len_lt_min", 0) == 0)
        check("排练记录了 PyTorch 与设备侧两套显存口径",
              tl.get("after_load") is not None
              and tl["after_load"].get("smi_mem_used_MiB") is not None
              and tl["after_load"].get("pt_peak_reserved_mib") is not None,
              f"after_load pt_peak_res={tl.get('after_load', {}).get('pt_peak_reserved_mib')} "
              f"smi_used={tl.get('after_load', {}).get('smi_mem_used_MiB')}")
        check("排练**不是** 8K 结论（ctx 记录为 512）", smoke["config"]["ctx"] == 512)

    print("\n== §4 CPU 自测（报告主张 43 PASS / 0 FAIL）==")
    import subprocess
    try:
        pr = subprocess.run([sys.executable, "-m", "specdec.engine.attn_phase_selftest_cpu"],
                            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=900)
        tail = (pr.stdout or "").strip().splitlines()[-1] if pr.stdout else ""
        check("CPU 自测当前可复现为 PASS 43 / FAIL 0",
              pr.returncode == 0 and "PASS 43 / FAIL 0" in tail, tail)
    except Exception as e:                                     # noqa: BLE001
        check("CPU 自测可运行", False, f"{type(e).__name__}: {e}")

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n审计：{len(RESULTS) - len(bad)} / {len(RESULTS)} 条与原始结果一致")
    if bad:
        print("不一致条目：")
        for n in bad:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
