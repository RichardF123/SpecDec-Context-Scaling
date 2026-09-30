"""mlp_tail_f06_i25.py — 修正版：严格三维 MLP 四格 + 有界尾部重组（F06/index25）。

修正（§二）
  1. **B/D 保持三维**：`U_B[:, ALIGN_ROW:ALIGN_ROW+1, :]`（=(1,1,1536)），不再写成会退化成
     (1,1536) 的二维切片。A/C 仍是 (1,4,1536)。
  2. **每格记录实际 shape/stride/dtype**（不只记"预期"）。
  3. 所有必需检查走 `diag_checks` 的**显式 require / assert**：重放一致性、输入对齐、finite、
     替换范围、源张量未修改 —— **失败即停**，不只写 JSON 布尔。
  4. 复用保存张量，并在结果里记录**复用依据**：`U_B_bf16`/`U_S_bf16`（原始 bf16 直用）；
     真实 attention 残差保存为 fp32 但 **FP32→BF16 往返精确**（已在审计中验证），
     因此是 bf16 真值的精确转换 ⇒ 可作**真实残差**使用；**不使用**上一轮的 FP32 派生相加结果。

尾部重组（§五，只加四个条件）
  T_x = R_x(bf16) + M_x(bf16)（**生产 bf16 加法**）→ `tgt.norm(T_x)` → `@ tgt.embed_tokens.weight.T`
  A/B 固定**块**残差与 MLP 输入来源，仅改 MLP 调用形状；C/D 固定**串行**残差与来源。
  Norm / LM head 的**调用形状完全相同**（都是 (1,1,1536)），未重写公式、未提精度。
"""
from __future__ import annotations

import glob
import json
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_checks as DC
from . import diag_fields as DF
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
SRC_TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
ALIGN_ROW, L = 1, 27
EXPECT_AR, EXPECT_SPEC, T4710 = 715, 576, 4710


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _fields(row):
    r = row.detach().float().reshape(-1)
    f = DF.row_fields(r, watch=[EXPECT_AR, EXPECT_SPEC, T4710])
    t2 = torch.topk(r, 2).values
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"], n_max_tied=f["n_max_tied"],
                tied_max_tokens=f["tied_max_tokens"], margin_top1_top2=float(t2[0] - t2[1]),
                tie_at_top=bool(float(t2[0]) == float(t2[1])),
                scores={str(t): float(r[t]) for t in (EXPECT_AR, EXPECT_SPEC, T4710)})


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="mlp_tail_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", layer=L, align_row=ALIGN_ROW,
                           align_input_position=8216, align_predicted_position=8217,
                           cells="A/C=(1,4,1536)  B/D=(1,1,1536)（**严格三维**）",
                           tail="T = R(bf16) + M(bf16) → tgt.norm(T) → @ W.T（调用形状固定为 (1,1,1536)）"),
               scope="修正版 MLP 四格 + 有界尾部重组；只读复用保存张量；不改生产实现/精度/门槛；无性能结论。")
    print(f"[mt] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[mt] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_mlptail_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        # ---- 复用保存张量（§三）----
        mp = torch.load(_newest(f"{SRC_TAG}_mlpshape_*.pt"), map_location="cpu")
        l27 = torch.load(_newest(f"{SRC_TAG}_l27_*.pt"), map_location="cpu")
        U_B, U_S = mp["U_B_bf16"], mp["U_S_bf16"]
        DC.require(U_B.dtype == torch.bfloat16 and U_S.dtype == torch.bfloat16, "复用的 U 必须是 bf16")
        DC.require(list(U_B.shape) == [1, 4, 1536] and list(U_S.shape) == [1, 1, 1536],
                   "复用的 U 形状不符", f"{list(U_B.shape)} {list(U_S.shape)}")
        # 残差：保存为 fp32，但往返精确 ⇒ 是 bf16 真值的精确转换
        rt = {}
        for k in ("block_attn_residual", "serial_attn_residual"):
            t = mp[k]
            rt[k] = bool(torch.equal(t.float(), t.float().to(torch.bfloat16).float()))
        DC.require(all(rt.values()), "残差 fp32→bf16 往返不精确 ⇒ 不能当作真实残差复用", str(rt))
        R_B = mp["block_attn_residual"].to(torch.bfloat16)      # (1,4,1536)
        R_S = mp["serial_attn_residual"].to(torch.bfloat16)     # (1,1,1536)
        MB_ref, MS_ref = mp["block_mlp_out"], mp["serial_mlp_out"]
        rec["reuse"] = dict(
            U_B=DC.layout(U_B), U_S=DC.layout(U_S),
            residual_roundtrip_exact=rt,
            residual_provenance=("保存为 fp32；FP32→BF16 往返**逐元素精确** ⇒ 是 bf16 真值的精确转换，"
                                 "可作真实残差使用；**未**使用上一轮 FP32 派生相加结果"),
            MLP_out_ref_shapes=dict(block=list(MB_ref.shape), serial=list(MS_ref.shape)),
            layer_out_ref=dict(block=list(l27["block_layer_out"].shape),
                               serial=list(l27["serial_layer_out"].shape)))

        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        mlp, W = tgt.layers[L].mlp, tgt.embed_tokens.weight
        W_snap = DC.snapshot(W)
        U_B_snap, U_S_snap = DC.snapshot(U_B), DC.snapshot(U_S)
        # 复用的张量在 CPU，模型在 CUDA：**按模型 dtype/device 搬运**，CPU 侧保留一份用于逐元素比较
        dev = W.device
        U_B_d, U_S_d = U_B.to(dev), U_S.to(dev)
        R_B_d, R_S_d = R_B.to(dev), R_S.to(dev)
        rec["device"] = dict(model=str(dev), reuse_source="cpu → 按模型 device/dtype 搬运（不改 dtype）")

        # ---- 严格三维四格 ----
        rB3 = U_B_d[:, ALIGN_ROW:ALIGN_ROW + 1, :]               # ★ (1,1,1536)，不退化成 (1,1536)
        DC.require(list(rB3.shape) == [1, 1, 1536], "B 格输入必须是 (1,1,1536)", str(list(rB3.shape)))
        U_Bc = U_B_d.clone()
        U_Bc[:, ALIGN_ROW, :] = U_S_d[:, 0, :]
        non_target_ok = bool(torch.equal(U_Bc[:, [0, 2, 3], :], U_B_d[:, [0, 2, 3], :]))
        with torch.inference_mode():
            oA, oB, oC, oD = mlp(U_B_d), mlp(rB3), mlp(U_Bc), mlp(U_S_d)
        for nm, o, want in (("A", oA, (1, 4, 1536)), ("B", oB, (1, 1, 1536)),
                            ("C", oC, (1, 4, 1536)), ("D", oD, (1, 1, 1536))):
            DC.require(list(o.shape) == list(want), f"{nm} 格输出形状不符",
                       f"{list(o.shape)} != {want}")
        A_ = oA[:, ALIGN_ROW, :].detach().float().cpu()
        B_ = oB[:, 0, :].detach().float().cpu()
        C_ = oC[:, ALIGN_ROW, :].detach().float().cpu()
        D_ = oD[:, 0, :].detach().float().cpu()
        # 对齐行一律取成 (1,1536)：**显式 require**，避免 (1,1536) vs (1536,) 又悄悄比成 False
        DC.require(all(list(t.shape) == [1, 1536] for t in (A_, B_, C_, D_)),
                   "对齐行取法形状不符（应为 (1,1536)）",
                   str([list(t.shape) for t in (A_, B_, C_, D_)]))
        cells = {nm: dict(cell=lab, out=DC.layout(o), aligned=DC.layout(al))
                 for nm, lab, o, al in (
                     ("A", "U_B × 块形状", oA, A_), ("B", "U_B 对齐行(三维) × 单行形状", oB, B_),
                     ("C", "U_B(行1:=U_S) × 块形状", oC, C_), ("D", "U_S × 单行形状", oD, D_))}
        pairs = dict(
            A_vs_B_shape_effect=DC.cmp_tensor(A_.reshape(-1), B_.reshape(-1), "A_vs_B"),
            C_vs_D_shape_effect=DC.cmp_tensor(C_.reshape(-1), D_.reshape(-1), "C_vs_D"),
            A_vs_C_input_source=DC.cmp_tensor(A_.reshape(-1), C_.reshape(-1), "A_vs_C"),
            B_vs_D_input_source=DC.cmp_tensor(B_.reshape(-1), D_.reshape(-1), "B_vs_D"))
        # 必需：A/D 逐元素复现捕获的真实 MLP 输出
        rep = dict(
            A_vs_captured_block=DC.cmp_tensor(oA[:, ALIGN_ROW, :].float().cpu(),
                                             MB_ref[:, ALIGN_ROW, :].float().cpu(),
                                             "A_vs_captured_block", ref_snapshot=U_B_snap, src=U_B),
            D_vs_captured_serial=DC.cmp_tensor(oD[:, 0, :].float().cpu(),
                                              MS_ref[:, 0, :].float().cpu(),
                                              "D_vs_captured_serial", ref_snapshot=U_S_snap, src=U_S))
        DC.require(rep["A_vs_captured_block"]["elementwise_equal"], "A 未复现捕获的块 MLP 输出")
        DC.require(rep["D_vs_captured_serial"]["elementwise_equal"], "D 未复现捕获的串行 MLP 输出")
        DC.require(non_target_ok, "C 替换越界（非目标行被改动）")
        DC.require(bool(torch.equal(W, W_snap)), "权重被原地修改")

        # ---- 有界尾部重组（§五）：生产 bf16 加法 ----
        def tail(R, M_row):
            Mc = M_row.reshape(1, 1, -1)
            # MLP 输出本为 bf16，这里以 fp32 记录；**往返必须精确**，否则不是生产的 bf16 加法
            DC.require(torch.equal(Mc.to(torch.bfloat16).float(), Mc.float()),
                       "MLP 输出 fp32→bf16 往返不精确 ⇒ 尾部加法不是生产 bf16 加法")
            Rr = (R[:, 0, :] if R.dim() == 3 else R.reshape(1, 1, -1)).to(torch.bfloat16)
            Mr = Mc.to(torch.bfloat16).to(Rr.device)          # 加法在模型 device 上执行
            T = (Rr + Mr)
            DC.require(str(T.dtype) == "torch.bfloat16", "尾部加法必须是 bf16 dtype", str(T.dtype))
            DC.require(tuple(T.shape) == (1, 1, 1536), "尾部形状必须 (1,1,1536)", str(tuple(T.shape)))
            DC.require(DC.finite_all(T), "尾部向量非有限")
            with torch.inference_mode():
                Hn = tgt.norm(T)
                lg = (Hn @ W.T)[0].detach().float().cpu()
            return T.detach().float().cpu(), Hn.detach().float().cpu(), lg

        # ---- 交叉源一致性 + 用**真实** l27 张量独立验证重建恒等式（生产 bf16 加法） ----
        # ⚠️ 命名纠正：l27 里的 `R_tap_*` 是**最终 Norm 的输入**（= L27 输出），
        #    **不是** post_attention_layernorm 的输入（= attention 残差）。
        #    本次新增的 require 正是靠这条语义抓出了我自己的误用：首跑 checks_failed，
        #    `R_tap_block` 与 mlpshape 的 `block_attn_residual` 最大差 78。
        cross_tap = dict(
            block_R_tap_is_L27_out=DC.cmp_tensor(l27["R_tap_block"][0, ALIGN_ROW].float(),
                                                 l27["block_layer_out"].float(), "block_rtap_vs_out"),
            serial_R_tap_is_L27_out=DC.cmp_tensor(l27["R_tap_ser"][0, 0].float(),
                                                  l27["serial_layer_out"].float(), "serial_rtap_vs_out"),
            note="R_tap == 最终 Norm 输入 == L27 输出；attention 残差来自 mlpshape 对 "
                 "post_attention_layernorm **输入**的挂钩。两者是**不同**的量，不可互换。")
        DC.require(cross_tap["block_R_tap_is_L27_out"]["elementwise_equal"] and
                   cross_tap["serial_R_tap_is_L27_out"]["elementwise_equal"],
                   "R_tap 与 L27 输出不一致（观察点语义被破坏）")
        DC.require(torch.equal(l27["block_mlp"].float(), MB_ref[:, ALIGN_ROW, :].reshape(-1).float()),
                   "两处捕获的块 MLP 输出不一致")
        DC.require(torch.equal(l27["serial_mlp"].float(), MS_ref[:, 0, :].reshape(-1).float()),
                   "两处捕获的串行 MLP 输出不一致")
        cross = dict(
            identity_block=DC.cmp_tensor((R_B[:, ALIGN_ROW, :].reshape(-1).to(torch.bfloat16)
                                          + l27["block_mlp"]).float(),
                                         l27["block_layer_out"].float(), "block_recon_identity"),
            identity_serial=DC.cmp_tensor((R_S[:, 0, :].reshape(-1).to(torch.bfloat16)
                                           + l27["serial_mlp"]).float(),
                                          l27["serial_layer_out"].float(), "serial_recon_identity"),
            A_bf16_equals_real_block_mlp=bool(torch.equal(A_.reshape(-1).to(torch.bfloat16),
                                                         l27["block_mlp"].reshape(-1))),
            D_bf16_equals_real_serial_mlp=bool(torch.equal(D_.reshape(-1).to(torch.bfloat16),
                                                         l27["serial_mlp"].reshape(-1))),
            aligned_dtype_note="四格对齐行一律取成 (1,1536)（见 cells.aligned）；与 l27 的 (1536,) 比较前 "
                               "必须 reshape(-1)。首跑 checks_failed 正是 A_=(1,1536) vs l27=(1536,) 的"
                               "形状不等（torch.equal 对形状敏感），不是数值差异。",
            residual_tap_semantics=cross_tap)
        DC.require(cross["identity_block"]["elementwise_equal"] and
                   cross["identity_serial"]["elementwise_equal"],
                   "重加恒等式不成立 ⇒ 尾部重组公式与生产不一致", str(cross["identity_block"]))
        DC.require(cross["A_bf16_equals_real_block_mlp"] and cross["D_bf16_equals_real_serial_mlp"],
                   "A/D 未逐位复现真实 MLP 输出")

        RB_row = R_B_d[:, ALIGN_ROW, :]
        RS_row = R_S_d[:, 0, :]
        tails = {}
        for nm, R, Mrow in (("TA", RB_row, A_), ("TB", RB_row, B_),
                            ("TC", RS_row, C_), ("TD", RS_row, D_)):
            T, Hn, lg = tail(R, Mrow)
            tails[nm] = dict(cond={"TA": "块残差 + MLP(A 块形状)", "TB": "块残差 + MLP(B 单行形状)",
                                   "TC": "串行残差 + MLP(C 块形状)", "TD": "串行残差 + MLP(D 单行形状)"}[nm],
                             T=T, H=Hn, logits=lg, fields=_fields(lg),
                             T_finite=DC.finite_all(T), H_finite=DC.finite_all(Hn),
                             lg_finite=DC.finite_all(lg))
        # 尾部重组正确性：T_A/T_D 应等于**真实 L27 输出**对齐行（R + 真实 MLP 输出）
        tail_checks = dict(
            TA_equals_real_L27_out=DC.cmp_tensor(tails["TA"]["T"].reshape(-1),
                                                l27["block_layer_out"].float().reshape(-1),
                                                "TA_vs_real_L27_block_out"),
            TD_equals_real_L27_out=DC.cmp_tensor(tails["TD"]["T"].reshape(-1),
                                                l27["serial_layer_out"].float().reshape(-1),
                                                "TD_vs_real_L27_serial_out"))
        DC.require(tail_checks["TA_equals_real_L27_out"]["elementwise_equal"],
                   "T_A 未复现真实 L27 块输出（重组不正确）")
        DC.require(tail_checks["TD_equals_real_L27_out"]["elementwise_equal"],
                   "T_D 未复现真实 L27 串行输出（重组不正确）")

        # ---- BF16 残差相加后形状差异是否仍在 ----
        rec.update(cells=cells, pairs=pairs, mlp_replay=rep, cross_source=cross,
                   non_target_rows_ok=non_target_ok, weights_unmodified=True,
                   tail_checks=tail_checks,
                   tails={k: {kk: v[kk] for kk in ("cond", "fields", "T_finite", "H_finite",
                                                   "lg_finite")} for k, v in tails.items()},
                   tail_pairs=dict(
                       TA_vs_TB_shape_effect=DC.cmp_tensor(tails["TA"]["T"].reshape(-1),
                                                           tails["TB"]["T"].reshape(-1), "TA_vs_TB"),
                       TC_vs_TD_shape_effect=DC.cmp_tensor(tails["TC"]["T"].reshape(-1),
                                                           tails["TD"]["T"].reshape(-1), "TC_vs_TD"),
                       TA_vs_TC_source=DC.cmp_tensor(tails["TA"]["logits"], tails["TC"]["logits"],
                                                     "TA_vs_TC_logits"),
                       TB_vs_TD_source=DC.cmp_tensor(tails["TB"]["logits"], tails["TD"]["logits"],
                                                     "TB_vs_TD_logits"),
                       TA_vs_TB_logits=DC.cmp_tensor(tails["TA"]["logits"], tails["TB"]["logits"],
                                                     "TA_vs_TB_logits"),
                       TC_vs_TD_logits=DC.cmp_tensor(tails["TC"]["logits"], tails["TD"]["logits"],
                                                     "TC_vs_TD_logits")))
        tp = rec["tail_pairs"]
        rec["verdict"] = dict(
            q1_shape_diff_survives_bf16_add=dict(
                TA_vs_TB_equal=not tp["TA_vs_TB_shape_effect"]["elementwise_equal"],
                TC_vs_TD_equal=not tp["TC_vs_TD_shape_effect"]["elementwise_equal"],
                TA_TB_max=tp["TA_vs_TB_shape_effect"]["abs_err_max"],
                TC_TD_max=tp["TC_vs_TD_shape_effect"]["abs_err_max"]),
            q2_propagates_to_logits=dict(
                TA_vs_TB_logits_equal=not tp["TA_vs_TB_logits"]["elementwise_equal"],
                TA_TB_logits_max=tp["TA_vs_TB_logits"]["abs_err_max"],
                TC_TD_logits_max=tp["TC_vs_TD_logits"]["abs_err_max"]),
            q3_changes_choice=dict(
                TA=rec["tails"]["TA"]["fields"]["argmax_id"],
                TB=rec["tails"]["TB"]["fields"]["argmax_id"],
                TC=rec["tails"]["TC"]["fields"]["argmax_id"],
                TD=rec["tails"]["TD"]["fields"]["argmax_id"],
                shape_effect_flips=(rec["tails"]["TA"]["fields"]["argmax_id"]
                                    != rec["tails"]["TB"]["fields"]["argmax_id"]),
                source_effect_flips=(rec["tails"]["TA"]["fields"]["argmax_id"]
                                     != rec["tails"]["TC"]["fields"]["argmax_id"])),
            rule="被残差舍入消除 ⇒ 只说明该固定输入下未继续传播；logits 变而选词不变 ⇒ 只证明数值影响；"
                 "选词变 ⇒ 该受控干预足以影响本例决策，**不等于**唯一根因或生产修复。"
                 "不以是否选回 4710 判定成败；不把误差大小比当因果贡献比例。")
        rec["stage"] = "complete"
        p = OUTDIR / f"{SRC_TAG}_mlptail_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tpp = OUTDIR / f"{SRC_TAG}_mlptail_{run_id}.pt"
        torch.save({f"{k}_T": v["T"] for k, v in tails.items()} |
                   {f"{k}_H": v["H"] for k, v in tails.items()} |
                   {f"{k}_lg": v["logits"] for k, v in tails.items()} | dict(A=A_, B=B_, C=C_, D=D_),
                   str(tpp))
        rec["rows_dump"] = str(tpp)
        print("== 复用依据 ==")
        print(f"  U_B/U_S bf16 直用；残差 fp32→bf16 往返精确={all(rt.values())} ⇒ 可作真实残差")
        print("== 四格（严格三维） ==")
        for nm in ("A", "B", "C", "D"):
            print(f"  {nm} {cells[nm]['cell']:26s} out={cells[nm]['out']['shape']} "
                  f"aligned={cells[nm]['aligned']['shape']} {cells[nm]['aligned']['dtype']}")
        print(f"  A 复现捕获块 MLP: {rep['A_vs_captured_block']['elementwise_equal']} | "
              f"D 复现捕获串行 MLP: {rep['D_vs_captured_serial']['elementwise_equal']}")
        print(f"  A bf16==真实 block_mlp: {cross['A_bf16_equals_real_block_mlp']} | "
              f"D bf16==真实 serial_mlp: {cross['D_bf16_equals_real_serial_mlp']} | "
              f"重加恒等式 块/串行: {cross['identity_block']['elementwise_equal']}/"
              f"{cross['identity_serial']['elementwise_equal']}")
        print("== MLP 配对 ==")
        for k, v in pairs.items():
            print(f"  {k}: equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} mean={v['abs_err_mean']:.6g}")
        print("== 尾部重组（T = R+M，bf16；统一 (1,1,1536)） ==")
        print(f"  T_A == 真实 L27 块输出: {tail_checks['TA_equals_real_L27_out']['elementwise_equal']} | "
              f"T_D == 真实 L27 串行输出: {tail_checks['TD_equals_real_L27_out']['elementwise_equal']}")
        for k in ("TA", "TB", "TC", "TD"):
            f_ = tails[k]["fields"]
            print(f"  {k} {tails[k]['cond']:26s} argmax={f_['argmax_id']} tied={f_['n_max_tied']}"
                  f"{f_['tied_max_tokens']} margin={f_['margin_top1_top2']} "
                  f"576/715/4710={f_['scores']['576']}/{f_['scores']['715']}/{f_['scores']['4710']}")
        print("== 关键问题 ==")
        v = rec["verdict"]
        print(f"  Q1 形状差在 bf16 相加后仍在: TA/TB={v['q1_shape_diff_survives_bf16_add']['TA_vs_TB_equal']} "
              f"(max {v['q1_shape_diff_survives_bf16_add']['TA_TB_max']:.6g}) "
              f"TC/TD={v['q1_shape_diff_survives_bf16_add']['TC_vs_TD_equal']} "
              f"(max {v['q1_shape_diff_survives_bf16_add']['TC_TD_max']:.6g})")
        print(f"  Q2 传到 logits: TA/TB max={v['q2_propagates_to_logits']['TA_TB_logits_max']:.6g} "
              f"TC/TD max={v['q2_propagates_to_logits']['TC_TD_logits_max']:.6g}")
        print(f"  Q3 选词: TA={v['q3_changes_choice']['TA']} TB={v['q3_changes_choice']['TB']} "
              f"TC={v['q3_changes_choice']['TC']} TD={v['q3_changes_choice']['TD']} | "
              f"形状效应翻转选词={v['q3_changes_choice']['shape_effect_flips']} "
              f"来源效应翻转选词={v['q3_changes_choice']['source_effect_flips']}")
        print(f"[mt] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e); rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{SRC_TAG}_mlptail_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[mt] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[mt] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
