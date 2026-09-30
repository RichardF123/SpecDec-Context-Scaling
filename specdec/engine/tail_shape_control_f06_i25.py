"""tail_shape_control_f06_i25.py — 为什么尾部重建与真实块行差 0.0625？（只读复用保存张量）

背景
  `mlp_tail_f06_i25` 中 `T_A` 已被 **require** 证明与真实 L27 块输出逐元素相等，
  `TD_lg` 与真实串行行逐元素相等（max|Δ|=0）；
  但 `TA_lg` / `TB_lg` / `TC_lg` 与各自参考行差 **max|Δ|=0.0625**。
  差异只可能来自 T 之后的 **norm / LM head 的调用形状**。

本工具只做一件事：对**同一个** T，比较三种调用形状产出的最终 logits
  P1 生产块路径   : tgt.norm(x4)[0, ALIGN_ROW] @ W.T      # x4 = (1,4,1536)
  P2 三维单行     : tgt.norm(x4[:,1:2,:])[0,0] @ W.T      # (1,1,1536) ← 本课题四条件用的形式
  P3 二维单行     : tgt.norm(x4[:,1,:])[0]   @ W.T        # (1,1536)  ← 早先 norm/lmhead 边界实验用的形式
并各自与真实块行 `l27["real"]` 逐元素比较。

失败即停（require）；不改生产实现、不改精度、不启用 QK32/prob32。
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
ALIGN_ROW, L = 1, 27
EXPECT_AR, EXPECT_SPEC, T4710 = 715, 576, 4710


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _fields(row):
    r = row.detach().float().reshape(-1)
    f = DF.row_fields(r, watch=[EXPECT_AR, EXPECT_SPEC, T4710])
    t2 = torch.topk(r, 2).values
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"], n_max_tied=f["n_max_tied"],
                tied_max_tokens=f["tied_max_tokens"],
                margin_top1_top2=float(t2[0] - t2[1]),
                scores={str(t): float(r[t]) for t in (EXPECT_AR, EXPECT_SPEC, T4710)})


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="tail_shape_control_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="同一个 T，norm/LM head 的三种调用形状是否给出相同的最终 logits？"
                        "尾部重建与真实块行差 0.0625 的来源是哪一个形状？",
               scope="只读复用保存张量；不跑注意力、不改生产实现/精度/门槛；无性能结论。")
    print(f"[sc] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[sc] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_tailshape_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        l27 = torch.load(_newest(f"{SRC_TAG}_l27_*.pt"), map_location="cpu")
        mp = torch.load(_newest(f"{SRC_TAG}_mlpshape_*.pt"), map_location="cpu")
        real_row = l27["real"].reshape(-1).float()                 # 真实块臂最终 logits 行
        ser_row = l27["serial_row"].reshape(-1).float()            # 真实串行臂最终 logits 行
        x4_cpu = l27["R_tap_block"].float()                        # 真实 L27 块输出（最终 Norm 输入）
        xS_cpu = l27["R_tap_ser"].float()
        DC.require(list(x4_cpu.shape) == [1, 4, 1536], "R_tap_block 形状不符",
                   str(list(x4_cpu.shape)))
        DC.require(torch.equal(x4_cpu, x4_cpu.to(torch.bfloat16).float()) and
                   torch.equal(xS_cpu, xS_cpu.to(torch.bfloat16).float()),
                   "复用张量不是 bf16 可精确表示的（不能当作真实 bf16 复用）")
        # 交叉核对：T_A / T_D（上一工具已 require 等于真实 L27 输出）
        tp = torch.load(_newest(f"{SRC_TAG}_mlptail_*.pt"), map_location="cpu")
        rec["reuse"] = dict(
            T_A_equals_R_tap_block=DC.cmp_tensor(tp["TA_T"].reshape(-1).float(),
                                                 x4_cpu[0, ALIGN_ROW].reshape(-1).float(),
                                                 "T_A_vs_R_tap_block"),
            T_D_equals_R_tap_ser=DC.cmp_tensor(tp["TD_T"].reshape(-1).float(),
                                               xS_cpu[0, 0].reshape(-1).float(),
                                               "T_D_vs_R_tap_ser"),
            note="T_A/T_D 已等于真实 L27 输出 ⇒ 差异只能来自 norm/LM head 的调用形状。")
        DC.require(rec["reuse"]["T_A_equals_R_tap_block"]["elementwise_equal"] and
                   rec["reuse"]["T_D_equals_R_tap_ser"]["elementwise_equal"],
                   "T_A/T_D 与真实 L27 输出不相等 ⇒ 前提不成立，停止")

        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        W = tgt.embed_tokens.weight
        W_snap = DC.snapshot(W)
        dev = W.device
        # 保存为 fp32 但往返精确 ⇒ 以 bf16 真值复用（并要求逐元素精确，见上）
        x4 = x4_cpu.to(torch.bfloat16).to(dev)
        xS = xS_cpu.to(torch.bfloat16).to(dev)
        DC.require(str(x4.dtype) == "torch.bfloat16" and str(xS.dtype) == "torch.bfloat16",
                   "复用张量必须是 bf16（否则 norm 会在 fp32 下算，与生产不符）",
                   f"{x4.dtype} {xS.dtype}")
        x4_snap, xS_snap = DC.snapshot(x4), DC.snapshot(xS)

        with torch.inference_mode():
            # P1 生产块路径：norm 作用在 (1,4,1536) 整块上
            n4 = tgt.norm(x4)
            lg4 = (n4 @ W.T)[0, ALIGN_ROW].detach().float().cpu()
            # P2 三维单行 (1,1,1536)：本课题四条件用的形式
            x3 = x4[:, ALIGN_ROW:ALIGN_ROW + 1, :]
            DC.require(list(x3.shape) == [1, 1, 1536], "P2 输入必须是 (1,1,1536)", str(list(x3.shape)))
            lg3 = (tgt.norm(x3) @ W.T)[0, 0].detach().float().cpu()
            # P3 二维单行 (1,1536)：早先 norm/lmhead 边界实验用的形式
            x2 = x4[:, ALIGN_ROW, :]
            DC.require(list(x2.shape) == [1, 1536], "P3 输入必须是 (1,1536)", str(list(x2.shape)))
            lg2 = (tgt.norm(x2) @ W.T)[0].detach().float().cpu()
            # P4 串行（生产形状本来就是 (1,1,1536)）
            lgS = (tgt.norm(xS) @ W.T)[0, 0].detach().float().cpu()
            # P5 norm 输出本身的形状敏感性（无 LM head）
            h4_row = n4[0, ALIGN_ROW].detach().float().cpu()
            h3 = tgt.norm(x3)[0, 0].detach().float().cpu()
            h2 = tgt.norm(x2)[0].detach().float().cpu()
        DC.require(bool(torch.equal(x4, x4_snap)) and bool(torch.equal(xS, xS_snap)),
                   "复用张量被原地修改")
        DC.require(bool(torch.equal(W, W_snap)), "权重被原地修改")

        cells = dict(
            P1=dict(cell="生产块路径 norm(x4)[行1] @ W.T", shape="(1,4,1536)", fields=_fields(lg4)),
            P2=dict(cell="三维单行 norm(x4[:,1:2,:]) @ W.T", shape="(1,1,1536)", fields=_fields(lg3)),
            P3=dict(cell="二维单行 norm(x4[:,1,:]) @ W.T", shape="(1,1536)", fields=_fields(lg2)),
            P4=dict(cell="串行 norm(xS) @ W.T", shape="(1,1,1536)", fields=_fields(lgS)))
        cmp = dict(
            P1_vs_real_block=DC.cmp_tensor(lg4, real_row, "P1_vs_real_block"),
            P2_vs_real_block=DC.cmp_tensor(lg3, real_row, "P2_vs_real_block"),
            P3_vs_real_block=DC.cmp_tensor(lg2, real_row, "P3_vs_real_block"),
            P1_vs_P2=DC.cmp_tensor(lg4, lg3, "P1_vs_P2"),
            P1_vs_P3=DC.cmp_tensor(lg4, lg2, "P1_vs_P3"),
            P2_vs_P3=DC.cmp_tensor(lg3, lg2, "P2_vs_P3"),
            P4_vs_real_serial=DC.cmp_tensor(lgS, ser_row, "P4_vs_real_serial"),
            norm_out_block_row_vs_3d=DC.cmp_tensor(h4_row, h3, "norm_out_4d_row_vs_3d"),
            norm_out_block_row_vs_2d=DC.cmp_tensor(h4_row, h2, "norm_out_4d_row_vs_2d"),
            norm_out_3d_vs_2d=DC.cmp_tensor(h3, h2, "norm_out_3d_vs_2d"))
        rec.update(cells=cells, cmp=cmp, stage="complete")
        rec["verdict"] = dict(
            production_block_path_matches_real=cmp["P1_vs_real_block"]["elementwise_equal"],
            threeD_single_matches_real=cmp["P2_vs_real_block"]["elementwise_equal"],
            twoD_single_matches_real=cmp["P3_vs_real_block"]["elementwise_equal"],
            norm_itself_shape_sensitive=not cmp["norm_out_block_row_vs_3d"]["elementwise_equal"],
            head_shape_sensitive=not cmp["P1_vs_P2"]["elementwise_equal"],
            rule="若 P1 与真实块行逐元素相等 ⇒ 生产路径被正确复现；"
                 "P2/P3 与 P1 的差即**调用形状**造成的数值差，**不是** MLP 形状效应；"
                 "本工具只定位差异来源，不改任何生产实现，也不据此放宽门槛。")
        p = OUTDIR / f"{SRC_TAG}_tailshape_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp2 = OUTDIR / f"{SRC_TAG}_tailshape_{run_id}.pt"
        torch.save(dict(lg4=lg4, lg3=lg3, lg2=lg2, lgS=lgS, h4_row=h4_row, h3=h3, h2=h2), str(tp2))
        rec["rows_dump"] = str(tp2)
        print("== 三种调用形状 vs 真实行 ==")
        for nm in ("P1", "P2", "P3", "P4"):
            f_ = cells[nm]["fields"]
            print(f"  {nm} {cells[nm]['shape']:12s} argmax={f_['argmax_id']:6d} ntied={f_['n_max_tied']}"
                  f"{f_['tied_max_tokens']} margin={f_['margin_top1_top2']} "
                  f"576/715/4710={f_['scores']['576']}/{f_['scores']['715']}/{f_['scores']['4710']}")
        print("== 逐元素比较 ==")
        for k in ("P1_vs_real_block", "P2_vs_real_block", "P3_vs_real_block",
                  "P1_vs_P2", "P1_vs_P3", "P2_vs_P3", "P4_vs_real_serial",
                  "norm_out_block_row_vs_3d", "norm_out_block_row_vs_2d", "norm_out_3d_vs_2d"):
            v = cmp[k]
            print(f"  {k:28s} equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} "
                  f"mean={v['abs_err_mean']:.6g} n_diff_idx={v['max_err_index']}")
        print("== 结论 ==")
        for k, v in rec["verdict"].items():
            if k != "rule":
                print(f"  {k}: {v}")
        print(f"[sc] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e); rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{SRC_TAG}_tailshape_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[sc] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sc] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
