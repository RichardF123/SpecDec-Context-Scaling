"""norm_boundary_f06_i25.py — 核验**最终 RMSNorm 边界**：差异在上游，还是 Norm 调用形状也有贡献。

固定状态（§一）
  round-11 前 cache（长度 **8215**）、候选 `[27685,659,715,4882]`、块对齐行 index=**1**
  （输入 **8216** → 预测 **8217**）、BF16/base_bf16/原 backend 与 phase、QK32/prob32 关闭、
  **不跑生成**、每臂独立 cache 副本。

捕获（§三，只读 hook，挂在 `tgt.norm` 上）
  `R_B`/`R_S` = 最终 RMSNorm 的**输入**；`H_B`/`H_S` = 其**输出**。保留原始 dtype 与 layout。

四格（§四，调用模型**现有** `tgt.norm`，不重写公式）
  A `tgt.norm(R_B)`（块形状）→ 取行 1；B `tgt.norm(R_B[0,1:2])`（单行形状）；
  C 克隆 R_B、**只**把行 1 换成 R_S 行，`tgt.norm(...)` → 取行 1；D `tgt.norm(R_S)`。
  A 必须复现 H_B 行1；D 必须复现 H_S。

选词观察（§五）：把四格的**对齐行统一成 (1,1,1536)**，经**同一份** LM head 权重
（`x @ tgt.embed_tokens.weight.T`，与生产表达式一致）计算 —— **固定 LM head 调用形状**，避免混入另一变量。
这些混合输入**只用于诊断**，不是生产推理路径。

所有校验走 `diag_checks`（shape 断言 / 实算 isfinite / 独立快照 / 布局如实报告）；
任一必需检查失败即落盘并停止。
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
ROUND = 11
CAND0, FED = 27685, 659
CANDS = [CAND0, FED, 715, 4882]
ALIGN_ROW = 1
EXPECT_BLOCK, EXPECT_SERIAL = 576, 4710


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _fields(row):
    r = row.detach().float().reshape(-1)
    f = DF.row_fields(r, watch=[715, 576, 4710])
    t2 = torch.topk(r, 2).values
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"], n_max_tied=f["n_max_tied"],
                tied_max_tokens=f["tied_max_tokens"], margin_top1_top2=float(t2[0] - t2[1]),
                tie_at_top=bool(float(t2[0]) == float(t2[1])),
                scores={str(t): float(r[t]) for t in (715, 576, 4710)})


def to_gpu(c):
    return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="norm_boundary_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=CANDS,
                           align_row=ALIGN_ROW, align_input_position=8216,
                           align_predicted_position=8217,
                           norm_module="tgt.norm（最终 RMSNorm；hook 只挂它）",
                           lmhead_expression="x @ tgt.embed_tokens.weight.T",
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt")),
                           source_rows=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"))),
               scope="最终 Norm 边界四格；只读 hook；不重写公式；不改生产实现/精度/门槛；不跑生成。")
    print(f"[nb] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[nb] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_normboundary_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

    hooks = None
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        snap = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"),
                          map_location="cpu")[f"round{ROUND}_snap_pre"]
        real_row = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"),
                              map_location="cpu")[f"round{ROUND}_rows"][ALIGN_ROW].float()
        rec["cache_len"] = int(snap[0]["k"].shape[2])
        DC.require(rec["cache_len"] == 8215, "cache 长度必须是 8215", f"实测 {rec['cache_len']}")

        W = tgt.embed_tokens.weight
        W_snap = DC.snapshot(W)                     # 运行前独立快照（§二.3）
        plain_row = None

        # ---- 1) 无插桩基线：最终 logits ----
        with torch.inference_mode(), M.attention_phase("verify"):
            rows0, _b0 = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        plain_row = rows0[ALIGN_ROW].detach().float().cpu()
        rec["plain_replay"] = dict(block=dict(_fields(plain_row)),
                                   vs_saved_real=DC.cmp_tensor(plain_row, real_row,
                                                               "plain_vs_saved_real"))

        # ---- 2) 只读捕获最终 Norm 的输入/输出 ----
        held: dict = {}
        fired = []

        def hook(module, inp, out):
            DC.require(module is tgt.norm, "hook 必须挂在最终 RMSNorm 上",
                       f"实际={type(module).__name__}")
            held["R"] = inp[0].detach().clone()
            held["H"] = out.detach().clone()
            fired.append(dict(module_is_final_norm=True,
                              R_shape=[int(x) for x in inp[0].shape],
                              H_shape=[int(x) for x in out.shape]))
        hooks = tgt.norm.register_forward_hook(hook)
        with torch.inference_mode(), M.attention_phase("verify"):
            rows_i, _bi = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        inst_row = rows_i[ALIGN_ROW].detach().float().cpu()
        R_B, H_B = held.pop("R"), held.pop("H")
        n_block_calls = len(fired); fired.clear()
        w = to_gpu(snap)
        with torch.inference_mode(), M.attention_phase("repair"):
            spec_dec._step(tgt, CAND0, w)
        fired.clear()                                # 只保留**最后一步**（对齐行所在的那一次）
        with torch.inference_mode(), M.attention_phase("repair"):
            lg_s, _c = spec_dec._step(tgt, FED, w)
        R_S, H_S = held.pop("R"), held.pop("H")
        n_serial_calls = len(fired); fired.clear()
        hooks.remove(); hooks = None
        serial_row = lg_s[0, -1].detach().float().cpu()

        rec["capture"] = dict(
            R_B=DC.layout(R_B), H_B=DC.layout(H_B), R_S=DC.layout(R_S), H_S=DC.layout(H_S),
            block_norm_calls=n_block_calls, serial_norm_calls=n_serial_calls,
            expected=dict(R_B=[1, 4, 1536], H_B=[1, 4, 1536], R_S=[1, 1, 1536], H_S=[1, 1, 1536]),
            raw_dtype=str(R_B.dtype),
            instrumented_vs_plain=DC.cmp_tensor(inst_row, plain_row, "instrumented_vs_plain"),
            block_is_576=(_fields(inst_row)["argmax_id"] == EXPECT_BLOCK),
            serial_is_4710=(_fields(serial_row)["argmax_id"] == EXPECT_SERIAL))
        DC.require(list(R_B.shape) == [1, 4, 1536] and list(R_S.shape) == [1, 1, 1536],
                   "捕获形状不符", f"R_B={list(R_B.shape)} R_S={list(R_S.shape)}")
        DC.require(rec["capture"]["block_is_576"] and rec["capture"]["serial_is_4710"],
                   "复现失败：块必须 576、串行必须 4710",
                   f"block={_fields(inst_row)['argmax_id']} serial={_fields(serial_row)['argmax_id']}")
        print(f"[nb] R_B={list(R_B.shape)} {R_B.dtype} H_B={list(H_B.shape)} | "
              f"R_S={list(R_S.shape)} H_S={list(H_S.shape)} | 复现 576/4710 ✔")

        # ---- 3) 四格：调用模型现有 tgt.norm ----
        R_B_snap = DC.snapshot(R_B)
        R_S_snap = DC.snapshot(R_S)
        rB_row = R_B[0, ALIGN_ROW:ALIGN_ROW + 1]          # (1,1536) 对齐行（切片）
        rS_row = R_S[0, 0:1]                              # (1,1536)
        R_Bc = R_B.clone()
        R_Bc[0, ALIGN_ROW] = R_S[0, 0]
        non_target_ok = bool(torch.equal(R_Bc[0, [0, 2, 3]], R_B[0, [0, 2, 3]]))

        with torch.inference_mode():
            nA = tgt.norm(R_B)
            nB = tgt.norm(rB_row)
            nC = tgt.norm(R_Bc)
            nD = tgt.norm(R_S)
        A_out = nA[0, ALIGN_ROW].detach().float().cpu()
        # ⚠️ nB 形状是 (1,1536) ⇒ 取行用 [0]；写 [0,0] 得到标量，
        #    新加的 shape 断言会立刻拦下（首跑正是被它拦住的）。
        B_out = nB[0].detach().float().cpu()
        C_out = nC[0, ALIGN_ROW].detach().float().cpu()
        # nD 形状是 (1,1,1536) ⇒ 要 [0,0] 才能得到 (1536,)；
        # 注意与 nB 不同：nB 是 (1,1536)，用 [0]。两者都靠 shape 断言把关。
        D_out = nD[0, 0].detach().float().cpu()
        HB_row = H_B[0, ALIGN_ROW].detach().float().cpu()
        HS_row = H_S[0, 0].detach().float().cpu()

        cells = dict(
            A=dict(cell="R_B × 块形状", out=_fields_norm(A_out)),
            B=dict(cell="R_B 对齐行 × 单行形状", out=_fields_norm(B_out)),
            C=dict(cell="R_B(行1:=R_S) × 块形状", out=_fields_norm(C_out)),
            D=dict(cell="R_S × 单行形状", out=_fields_norm(D_out)),
        )
        pairs = dict(
            A_vs_B_shape_effect_block_input=DC.cmp_tensor(A_out, B_out, "A_vs_B"),
            C_vs_D_shape_effect_serial_input=DC.cmp_tensor(C_out, D_out, "C_vs_D"),
            A_vs_C_input_source_block_shape=DC.cmp_tensor(A_out, C_out, "A_vs_C"),
            B_vs_D_input_source_single_shape=DC.cmp_tensor(B_out, D_out, "B_vs_D"),
        )
        checks = dict(
            A_reproduces_H_B_row=DC.cmp_tensor(A_out, HB_row, "A_vs_captured_H_B_row",
                                               ref_snapshot=R_B_snap, src=R_B),
            D_reproduces_H_S=DC.cmp_tensor(D_out, HS_row, "D_vs_captured_H_S",
                                           ref_snapshot=R_S_snap, src=R_S),
            C_replaced_only_row1=non_target_ok,
            weights_unmodified=bool(torch.equal(W, W_snap)),
            layout_control=DC.layout_control(rB_row, "rB_row_contiguous"),
        )
        rec["R_diff"] = DC.cmp_tensor(R_B[0, ALIGN_ROW].float().cpu(), R_S[0, 0].float().cpu(),
                                      "R_B_row1_vs_R_S")
        rec["H_diff"] = DC.cmp_tensor(HB_row, HS_row, "H_B_row1_vs_H_S")

        # ---- 4) 选词观察：四格对齐行统一成 (1,1,1536) 过**同一份** LM head ----
        with torch.inference_mode():
            # ⚠️ 两个坑都在这里踩过：① 四格输出已转 CPU 而 W 在 CUDA；
            #    ② `.float()` 会变成 fp32 而 W 是 bf16。⇒ 投影必须用**原生 dtype/device** 的行，
            #    即直接从 Norm 输出张量切片，而不是用为比较而转过的 fp32 副本。
            native = dict(A=nA[0, ALIGN_ROW], B=nB[0], C=nC[0, ALIGN_ROW], D=nD[0, 0])
            rec["lmhead_input_dtype"] = {k: str(v.dtype) for k, v in native.items()}
            DC.require(all(v.dtype == W.dtype for v in native.values()),
                       "LM head 投影输入 dtype 必须与权重一致",
                       f"in={[str(v.dtype) for v in native.values()]} W={W.dtype}")
            lg = {k: (v.reshape(1, 1536) @ W.T)[0].detach().float().cpu()
                  for k, v in native.items()}
        choice = {k: _fields(v) for k, v in lg.items()}
        rec.update(cells=cells, pairs=pairs, checks=checks, choice=choice)
        rec["verdict"] = dict(
            upstream_already_differs=(rec["R_diff"]["elementwise_equal"] is False),
            norm_shape_effect_same_input=dict(
                A_vs_B=pairs["A_vs_B_shape_effect_block_input"]["elementwise_equal"],
                C_vs_D=pairs["C_vs_D_shape_effect_serial_input"]["elementwise_equal"]),
            norm_input_source_effect=dict(
                A_vs_C=pairs["A_vs_C_input_source_block_shape"]["elementwise_equal"],
                B_vs_D=pairs["B_vs_D_input_source_single_shape"]["elementwise_equal"]),
            rule="R_B≠R_S ⇒ 上游已贡献差异，但**不能**因此排除 Norm 额外贡献；"
                 "同输入下 Norm 输出不同 ⇒ 支持本例的 Norm 调用形状效应；相同 ⇒ 本例未观察到该项，重点上移。")
        rec["stage"] = "complete"
        p = OUTDIR / f"{SRC_TAG}_normboundary_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{SRC_TAG}_normboundary_{run_id}.pt"
        torch.save(dict(R_B=R_B.float().cpu(), R_S=R_S.float().cpu(),
                        H_B=H_B.float().cpu(), H_S=H_S.float().cpu(),
                        A=A_out, B=B_out, C=C_out, D=D_out, lgA=lg["A"], lgB=lg["B"],
                        lgC=lg["C"], lgD=lg["D"], plain_row=plain_row, real_row=real_row), str(tp))
        rec["rows_dump"] = str(tp)

        print("== 机制校验 ==")
        for k, c in checks.items():
            if isinstance(c, dict) and "elementwise_equal" in c:
                print(f"  {k}: equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']:.6g}")
        print(f"  C 只替换行1={non_target_ok} | 权重未改={checks['weights_unmodified']} | "
              f"layout_control.control_valid={checks['layout_control']['control_valid']}")
        print(f"== 边界 ==")
        print(f"  R_B 行1 vs R_S: equal={rec['R_diff']['elementwise_equal']} max|Δ|={rec['R_diff']['abs_err_max']:.6g}")
        print(f"  H_B 行1 vs H_S: equal={rec['H_diff']['elementwise_equal']} max|Δ|={rec['H_diff']['abs_err_max']:.6g}")
        print("== Norm 四格（输出对齐行） + 统一 (1,1) 过同一 LM head 的选词 ==")
        for k in ("A", "B", "C", "D"):
            c = choice[k]
            print(f"  {k} {cells[k]['cell']:26s} norm_out_max={cells[k]['out']['absmax']:.6g} "
                  f"argmax={c['argmax_id']} tied={c['n_max_tied']}{c['tied_max_tokens']} "
                  f"margin={c['margin_top1_top2']} 576/715/4710="
                  f"{c['scores']['576']}/{c['scores']['715']}/{c['scores']['4710']}")
        print("== 配对 ==")
        for k, c in pairs.items():
            print(f"  {k}: equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']:.6g}")
        print(f"[nb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                       # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        try:
            if hooks is not None:
                hooks.remove()
        except Exception:                                        # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{SRC_TAG}_normboundary_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[nb] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[nb] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


def _fields_norm(t):
    x = t.detach().float().reshape(-1)
    return dict(shape=[int(v) for v in t.shape], absmax=float(x.abs().max()),
                finite=bool(torch.isfinite(x).all()), mean=float(x.mean()))


if __name__ == "__main__":
    raise SystemExit(main())
