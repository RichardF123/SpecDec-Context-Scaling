"""mlp_shape_f06_i25.py — 隔离 L27 MLP 的**输入来源效应**与**调用形状效应**（F06/index25）。

口径修正（§一）
  1. **"自然两臂输入不同" ≠ "无法构造同输入实验"**：可以**固定一份已捕获输入**，
     分别用块形状与单行形状调用**同一个模块** ⇒ 这合法隔离**调用形状**效应。
     （上一轮 §9.28.6 说"L27 内部无法构造同输入实验"是**错的**，已更正。）
  2. 上一轮的 `attn_residual(derived) = layer_in.float() + atten_out.float()` 是 **FP32 派生和**，
     **不等同**生产 BF16 残差 ⇒ 该项标为「**FP32 派生量，非实际残差**」，
     **不再**用其 `max|Δ|=1.25` 描述真实残差差异。本轮补采**真实** attention 残差
     （即 `post_attention_layernorm` 的**输入**，bf16 原值）。

复用/补采（§三）
  已复用：`.pt` 里 L27 的 `post_attention_layernorm` 输出（= MLP 输入）与 `mlp` 输出（对齐行，bf16）。
  补采：**完整**(1,4,1536)/(1,1,1536) 的 MLP 输入、MLP 输出、以及**真实 attention 残差**。

MLP 四格（§四，调用模型现有 `tgt.layers[27].mlp`，不重写公式）
  A `U_B`(块形状)；B `U_B` 对齐行(单行形状)；C 克隆 `U_B`、仅换对齐行 ← `U_S`，块形状；D `U_S`(单行形状)。
  配对：A↔B、C↔D = 同输入不同形状；A↔C、B↔D = 同形状不同输入来源。
  A 必须逐元素复现捕获的块 MLP 输出对齐行；D 必须复现串行 MLP 输出。
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
ALIGN_ROW, L = 1, 27
EXPECT_BLOCK, EXPECT_SERIAL = 576, 4710


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def to_gpu(c):
    return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="mlp_shape_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=CANDS, layer=L,
                           align_row=ALIGN_ROW, align_input_position=8216,
                           align_predicted_position=8217,
                           module="tgt.layers[27].mlp（模型现有模块，不重写公式）",
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"))),
               scope="L27 MLP 的形状/输入来源隔离；只读 hook；不改生产实现/精度/门槛；不跑生成。")
    print(f"[mlp] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[mlp] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_mlpshape_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    handles = []
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
        layer = tgt.layers[L]
        mlp, pan = layer.mlp, layer.post_attention_layernorm

        # ---- 无插桩基线 ----
        with torch.inference_mode(), M.attention_phase("verify"):
            rows0, _b = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        plain = rows0[ALIGN_ROW].detach().float().cpu()

        got: dict = {}

        def grab(key):
            def h(mod, inp, out):
                if key == "attn_residual":
                    got[key] = inp[0].detach().clone()      # norm 的**输入** = 真实 attention 残差
                else:
                    got[key] = out.detach().clone()
            return h
        handles.append(pan.register_forward_hook(grab("attn_residual")))
        handles.append(pan.register_forward_hook(grab("mlp_in")))
        handles.append(mlp.register_forward_hook(grab("mlp_out")))

        with torch.inference_mode(), M.attention_phase("verify"):
            rows1, _b1 = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        inst = rows1[ALIGN_ROW].detach().float().cpu()
        B = dict(attn_residual=got.pop("attn_residual"), mlp_in=got.pop("mlp_in"),
                 mlp_out=got.pop("mlp_out"))
        w = to_gpu(snap)
        with torch.inference_mode(), M.attention_phase("repair"):
            spec_dec._step(tgt, CAND0, w)
        with torch.inference_mode(), M.attention_phase("repair"):
            lg_s, _c = spec_dec._step(tgt, FED, w)
        # ⚠️ 不要命名为 `S`：本模块顶部 `import results_schema as S`，同名会让 `S` 变成局部变量，
        #    导致前面 `S.git_meta(...)` 抛 UnboundLocalError（首跑即此错）。
        Ser = dict(attn_residual=got.pop("attn_residual"), mlp_in=got.pop("mlp_in"),
                   mlp_out=got.pop("mlp_out"))
        serial_row = lg_s[0, -1].detach().float().cpu()
        for h in handles:
            h.remove()
        handles.clear()

        rec["capture"] = {f"{a}_{b}": DC.layout(Ser[f"{b}"]) if a == "serial" else DC.layout(B[f"{b}"])
                          for a in ("block", "serial") for b in ("attn_residual", "mlp_in", "mlp_out")}
        rec["dtype_note"] = ("全部为**原始 bf16**（未转 FP32 再算）；"
                             "上一轮的 `attn_residual(derived)` 是 FP32 派生和，"
                             "**标记为「FP32 派生量，非实际残差」，不再用于真实残差差异**。")

        # ---- 机制校验 ----
        U_B, U_S = B["mlp_in"], Ser["mlp_in"]
        U_B_snap, U_S_snap = DC.snapshot(U_B), DC.snapshot(U_S)
        mlpB_row = B["mlp_out"][0, ALIGN_ROW].detach().float().cpu()
        mlpS_row = Ser["mlp_out"][0, 0].detach().float().cpu()
        checks = dict(
            instrumented_vs_plain=DC.cmp_tensor(inst, plain, "instrumented_vs_plain"),
            block_vs_saved_real=DC.cmp_tensor(plain, real_row, "plain_vs_saved_real"),
            block_is_576=(DF.row_fields(plain, watch=[715, 576])["argmax_id"] == EXPECT_BLOCK),
            serial_is_4710=(DF.row_fields(serial_row, watch=[715, 4710])["argmax_id"] == EXPECT_SERIAL),
            shapes=dict(U_B=list(U_B.shape), U_S=list(U_S.shape),
                        expect=dict(U_B=[1, 4, 1536], U_S=[1, 1, 1536])))
        DC.require(list(U_B.shape) == [1, 4, 1536] and list(U_S.shape) == [1, 1, 1536],
                   "MLP 输入形状不符", str(checks["shapes"]))
        DC.require(checks["block_is_576"] and checks["serial_is_4710"], "复现失败")
        DC.require(checks["instrumented_vs_plain"]["elementwise_equal"], "插桩改变了结果")

        # ---- 真实 attention 残差：块 vs 串行 ----
        rec["attn_residual_real"] = DC.cmp_tensor(
            B["attn_residual"][0, ALIGN_ROW].detach().float().cpu(),
            Ser["attn_residual"][0, 0].detach().float().cpu(), "L27_real_attn_residual")

        # ---- MLP 四格 ----
        rB = U_B[0, ALIGN_ROW:ALIGN_ROW + 1]
        U_Bc = U_B.clone()
        U_Bc[0, ALIGN_ROW] = U_S[0, 0]
        non_target_ok = bool(torch.equal(U_Bc[0, [0, 2, 3]], U_B[0, [0, 2, 3]]))
        with torch.inference_mode():
            oA = mlp(U_B)
            oB = mlp(rB)
            oC = mlp(U_Bc)
            oD = mlp(U_S)
        A_ = oA[0, ALIGN_ROW].detach().float().cpu()
        B_ = oB[0].detach().float().cpu() if oB.dim() == 2 else oB[0, 0].detach().float().cpu()
        C_ = oC[0, ALIGN_ROW].detach().float().cpu()
        D_ = oD[0].detach().float().cpu() if oD.dim() == 2 else oD[0, 0].detach().float().cpu()
        cells = dict(A=dict(cell="U_B × 块形状", out=_cell(A_)),
                     B=dict(cell="U_B 对齐行 × 单行形状", out=_cell(B_)),
                     C=dict(cell="U_B(行1:=U_S) × 块形状", out=_cell(C_)),
                     D=dict(cell="U_S × 单行形状", out=_cell(D_)))
        pairs = dict(
            A_vs_B_shape_effect_block_input=DC.cmp_tensor(A_, B_, "A_vs_B"),
            C_vs_D_shape_effect_serial_input=DC.cmp_tensor(C_, D_, "C_vs_D"),
            A_vs_C_input_source_block_shape=DC.cmp_tensor(A_, C_, "A_vs_C"),
            B_vs_D_input_source_single_shape=DC.cmp_tensor(B_, D_, "B_vs_D"))
        mchecks = dict(
            A_reproduces_captured_block_mlp_row=DC.cmp_tensor(A_, mlpB_row, "A_vs_captured_block",
                                                             ref_snapshot=U_B_snap, src=U_B),
            D_reproduces_captured_serial_mlp=DC.cmp_tensor(D_, mlpS_row, "D_vs_captured_serial",
                                                          ref_snapshot=U_S_snap, src=U_S),
            C_replaced_only_row1=non_target_ok,
            layout_control=DC.layout_control(rB, "rB_contiguous"))
        rec.update(checks=checks, cells=cells, pairs=pairs, mlp_checks=mchecks)
        rec["verdict"] = dict(
            shape_effect_same_input=dict(A_vs_B=pairs["A_vs_B_shape_effect_block_input"]["elementwise_equal"],
                                        C_vs_D=pairs["C_vs_D_shape_effect_serial_input"]["elementwise_equal"]),
            input_source_effect_fixed_shape=dict(
                A_vs_C=pairs["A_vs_C_input_source_block_shape"]["elementwise_equal"],
                B_vs_D=pairs["B_vs_D_input_source_single_shape"]["elementwise_equal"]),
            rule="同输入不同形状 ⇒ 支持该输入上的**调用形状**数值差异；同形状不同输入 ⇒ 输入来源效应；"
                 "两者可同时存在，不预设可加性；**不**据此声称 MLP 形状效应导致了 576/4710 翻转。")
        rec["stage"] = "complete"
        p = OUTDIR / f"{SRC_TAG}_mlpshape_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{SRC_TAG}_mlpshape_{run_id}.pt"
        torch.save(dict(U_B=U_B.float().cpu(), U_S=U_S.float().cpu(),
                        U_B_bf16=U_B.cpu(), U_S_bf16=U_S.cpu(),
                        block_attn_residual=B["attn_residual"].float().cpu(),
                        serial_attn_residual=Ser["attn_residual"].float().cpu(),
                        block_mlp_out=B["mlp_out"].float().cpu(),
                        serial_mlp_out=Ser["mlp_out"].float().cpu(),
                        A=A_, B=B_, C=C_, D=D_, plain=plain, real=real_row,
                        serial_row=serial_row), str(tp))
        rec["rows_dump"] = str(tp)
        print("== 机制校验 ==")
        print(f"  插桩 vs 无插桩: equal={checks['instrumented_vs_plain']['elementwise_equal']} "
              f"max|Δ|={checks['instrumented_vs_plain']['abs_err_max']}")
        print(f"  复现 576/4710: {checks['block_is_576']}/{checks['serial_is_4710']} | "
              f"U_B={checks['shapes']['U_B']} U_S={checks['shapes']['U_S']}")
        print(f"  A 重放捕获块 MLP 行: equal={mchecks['A_reproduces_captured_block_mlp_row']['elementwise_equal']} "
              f"max|Δ|={mchecks['A_reproduces_captured_block_mlp_row']['abs_err_max']}")
        print(f"  D 重放捕获串行 MLP: equal={mchecks['D_reproduces_captured_serial_mlp']['elementwise_equal']} "
              f"max|Δ|={mchecks['D_reproduces_captured_serial_mlp']['abs_err_max']}")
        print(f"  C 仅替换行1={non_target_ok} | layout_control.control_valid="
              f"{mchecks['layout_control']['control_valid']}")
        ar = rec["attn_residual_real"]
        print(f"== 真实 attention 残差（bf16，非 FP32 派生）==")
        print(f"  块 vs 串行: equal={ar['elementwise_equal']} max|Δ|={ar['abs_err_max']:.6g} "
              f"mean={ar['abs_err_mean']:.6g} idx={ar['max_err_index']} a={ar['a_at_max']:.6g} b={ar['b_at_max']:.6g}")
        print("== MLP 四格 ==")
        for k in ("A", "B", "C", "D"):
            print(f"  {k} {cells[k]['cell']:24s} absmax={cells[k]['out']['absmax']:.6g}")
        print("== 配对 ==")
        for k, v in pairs.items():
            print(f"  {k}: equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} "
                  f"mean={v['abs_err_mean']:.6g} idx={v['max_err_index']} "
                  f"a={v['a_at_max']:.6g} b={v['b_at_max']:.6g}")
        print(f"[mlp] RESULT_JSON={p}")
        return 0
    except Exception as e:                                       # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e); rec["traceback"] = traceback.format_exc()
        try:
            for h in handles:
                h.remove()
        except Exception:                                        # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{SRC_TAG}_mlpshape_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[mlp] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[mlp] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


def _cell(t):
    x = t.detach().float().reshape(-1)
    return dict(shape=[int(v) for v in t.shape], absmax=float(x.abs().max()),
                finite=bool(torch.isfinite(x).all()))


if __name__ == "__main__":
    raise SystemExit(main())
