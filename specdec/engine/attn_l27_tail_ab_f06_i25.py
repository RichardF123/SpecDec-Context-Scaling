"""attn_l27_tail_ab_f06_i25.py — L27 attention 行数差异的**末端传播验证**（A/B，有界局部干预）。

唯一改变的东西
  A = G0（**四行** attention 调用）的**对齐行** core 输出；
  B = G1（**单行** attention 调用）的 core 输出。
  两者随后走**完全相同的下游**（形状全部 `(1,1,1536)`，用**生产模块**、BF16、无新精度开关）：
    core → o_proj → **+ L27 层输入残差（BF16 加法）** → post_attention_layernorm → MLP
         → **+ 前一步结果（BF16 加法）** → final RMSNorm → LM head → logits / argmax

残差来源
  取 `l27` 产物里的 `block_layer_in`（L27 层**输入**），并用硬检查证明：
  `bf16_add(layer_in, attn_out) == 上轮保存的 attention 之后残差`（逐元素）
  ⇒ 若误用「attention 之后的残差」当残差（重复加 attention），本检查必然失败并停止。

**不需要新的捕获**：core 与 o_proj 参考值来自上一轮 `attng012_*.pt`，残差与交叉核对来自既有 `l27`/`mlpshape` 产物。
不跑完整生成、不跑正式矩阵、不改生产实现/精度/KV/argmax/门槛、不新增第三条件（G2 已与 G1 相同）。
"""
from __future__ import annotations

import glob
import json
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import attn_tail_common as T
from . import diag_checks as DC
from . import diag_fields as DF
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
L = 27
ALIGN_ROW = 1
WATCH = (576, 715, 4710)


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _need(d, key, what):
    DC.require(key in d, f"{what}: 缺少键 {key!r}（不得凭报告猜测产物内容）", f"现有键={sorted(d)}")
    return d[key]


def _fields(row, tok, k=5):
    r = row.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=list(WATCH), k=k)
    tk = torch.topk(r, k)
    am = f["argmax_id"]
    return dict(argmax_id=am, max_score=f["max_score"], n_max_tied=f["n_max_tied"],
                tied_max_tokens=f["tied_max_tokens"],
                # ⚠️ argmax 的文本**按 id 单独查表**：不得用 topk 首位的文本（并列时顺序不具语义）
                argmax_text=tok.decode([am]),
                margin_top1_top2=float(tk.values[0] - tk.values[1]),
                top5=[dict(token=int(i), text=tok.decode([int(i)]), logit=float(v))
                      for i, v in zip(tk.indices, tk.values)],
                top5_note="topk 只是参考列表；**并列时其顺序不等于解码 argmax**，"
                          "文本必须按 argmax id 单独查表",
                watch={str(t): float(r[t]) for t in WATCH},
                n_finite=int(torch.isfinite(r).sum()), vocab=int(r.numel()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="attn_l27_tail_ab_f06_i25",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="只替换 L27 attention 的**对齐行 core 输出**（A=G0 四行 vs B=G1 单行），"
                        "保持下游形状 `(1,1,1536)` 与所有权重/残差/dtype 不变，最终 logits/argmax 是否改变？",
               scope="有界局部干预；复用既有产物，不新增捕获；不跑生成/正式矩阵；"
                     "不改生产实现/精度/backend/manifest/门槛；不新增精度开关；只做 A/B 两个主条件。")
    print(f"[ab] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[ab] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_attntailab_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        DC.require(rec["diag_precision"]["qk_fp32"] is False and
                   rec["diag_precision"]["prob_fp32"] is False, "精度开关必须关闭",
                   str(rec["diag_precision"]))

        # ---------------- 1) 读既有产物并逐键校验 ----------------
        ab_p = _newest(f"{TAG}_attng012_*.pt")
        l27_p = _newest(f"{TAG}_l27_*.pt")
        mps_p = _newest(f"{TAG}_mlpshape_*.pt")
        rec["sources"] = dict(attng012=str(ab_p), l27=str(l27_p), mlpshape=str(mps_p))
        ab = torch.load(ab_p, map_location="cpu")
        l27 = torch.load(l27_p, map_location="cpu")
        mps = torch.load(mps_p, map_location="cpu")
        core_A_f = _need(ab, "G0_core_row", "attng012").detach().float().cpu()   # (1536,) 四行调用的对齐行
        core_B_f = _need(ab, "G1_core", "attng012").detach().float().cpu()       # (1536,) 单行调用
        op_A_ref = _need(ab, "oproj_G0", "attng012").detach().float().cpu()
        op_B_ref = _need(ab, "oproj_G1", "attng012").detach().float().cpu()
        layer_in_f = _need(l27, "block_layer_in", "l27").detach().float().cpu()
        attn_out_ref = _need(l27, "block_atten_out", "l27").detach().float().cpu()
        post_res_f = _need(mps, "block_attn_residual", "mlpshape").detach().float().cpu()
        l27_out_f = _need(l27, "R_tap_block", "l27").detach().float().cpu()[0, ALIGN_ROW]  # 最终 Norm 输入（参考线用）
        DC.require(list(l27["R_tap_block"].shape) == [1, 4, 1536],
                   "R_tap_block 形状必须是 (1,4,1536)（最终 Norm 输入 = L27 输出）",
                   str(list(l27["R_tap_block"].shape)))
        # dtype 与形状断言：记录值必须无损等于 bf16 真值
        for t, nm in ((core_A_f, "G0_core_row"), (core_B_f, "G1_core"), (op_A_ref, "oproj_G0"),
                      (op_B_ref, "oproj_G1"), (layer_in_f, "block_layer_in"),
                      (attn_out_ref, "block_atten_out"), (l27_out_f, "R_tap_block")):
            DC.require(list(t.shape) == [1536], f"{nm} 形状必须是 (1536,)", str(list(t.shape)))
            T.require_bf16_exact(t, nm)
        DC.require(list(post_res_f.shape) == [1, 4, 1536], "block_attn_residual 形状必须是 (1,4,1536)",
                   str(list(post_res_f.shape)))
        T.require_bf16_exact(post_res_f, "block_attn_residual")
        post_res_row_f = post_res_f[0, ALIGN_ROW].reshape(-1)
        rec["source_keys"] = dict(
            attng012_used=["G0_core_row", "G1_core", "oproj_G0", "oproj_G1"],
            l27_used=["block_layer_in", "block_atten_out", "R_tap_block"],
            mlpshape_used=["block_attn_residual"],
            note="逐键 `require`；所有记录值均通过 fp32→bf16 往返精确性检查后才转为 bf16 使用")
        print("[ab] 既有产物键与形状/往返精确性检查 ✔")

        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        require_modules = dict(
            o_proj=type(tgt.layers[L].self_attn.o_proj).__name__,
            pan=type(tgt.layers[L].post_attention_layernorm).__name__,
            mlp=type(tgt.layers[L].mlp).__name__,
            norm=type(tgt.norm).__name__)
        rec["modules"] = require_modules

        dev = tgt.embed_tokens.weight.device
        W = tgt.embed_tokens.weight
        W_snap = DC.snapshot(W)
        o_proj = tgt.layers[L].self_attn.o_proj
        pan = tgt.layers[L].post_attention_layernorm
        mlp = tgt.layers[L].mlp
        fin_norm = tgt.norm

        core_A = T.as_row3(core_A_f.to(torch.bfloat16), "core_A", device=dev)
        core_B = T.as_row3(core_B_f.to(torch.bfloat16), "core_B", device=dev)
        layer_in = T.as_row3(layer_in_f.to(torch.bfloat16), "layer_in", device=dev)
        attn_out_ref3 = T.as_row3(attn_out_ref.to(torch.bfloat16), "attn_out_ref", device=dev)
        post_res_row = T.as_row3(post_res_row_f.to(torch.bfloat16), "post_res_row", device=dev)
        snaps = dict(core_A=DC.snapshot(core_A), core_B=DC.snapshot(core_B),
                     layer_in=DC.snapshot(layer_in))

        # ---------------- 2) 控制：残差来源 + 跨运行复现 + 重复 ----------------
        rec["control_residual_identity"] = T.check_residual_identity(
            layer_in, attn_out_ref3, post_res_row, "L27 残差恒等式")
        print("[ab] 控制 ✔ 残差来源：layer_in + attn_out == 上一轮保存的 attention 之后残差（逐元素）")

        with torch.inference_mode():
            op_A = o_proj(core_A)
            op_B = o_proj(core_B)
        DC.require(list(op_A.shape) == list(op_B.shape) == [1, 1, 1536],
                   "o_proj 输出形状必须是 (1,1,1536)", f"{list(op_A.shape)} {list(op_B.shape)}")
        DC.require(str(op_A.dtype) == "torch.bfloat16", "o_proj 输出必须是 bf16", str(op_A.dtype))
        rec["control_oproj_crossrun"] = dict(
            A_vs_saved=DC.cmp_tensor(op_A[0, 0].detach().float().cpu(), op_A_ref,
                                     "opA_vs_saved_oproj_G0"),
            B_vs_saved=DC.cmp_tensor(op_B[0, 0].detach().float().cpu(), op_B_ref,
                                     "opB_vs_saved_oproj_G1"),
            note="**跨运行复现控制**：本轮重算的单行 o_proj 必须逐元素等于上一轮保存的同形状结果")
        DC.require(rec["control_oproj_crossrun"]["A_vs_saved"]["elementwise_equal"] and
                   rec["control_oproj_crossrun"]["B_vs_saved"]["elementwise_equal"],
                   "单行 o_proj 未能逐元素复现上一轮保存的同形状结果")
        print("[ab] 控制 ✔ 单行 o_proj 逐元素复现上一轮保存结果（A/B 两条）")

        def chain(core):
            """core(1,1,1536 bf16) → o_proj → +layer_in(BF16) → pan → mlp → +h(BF16) → norm → logits。"""
            o = o_proj(core)
            h = T.bf16_add_row(layer_in, o, "attention 残差加法")
            n = pan(h)
            m = mlp(n)
            tt = T.bf16_add_row(h, m, "MLP 残差加法")
            fn = fin_norm(tt)
            lg = (fn @ W.T)[0, 0]
            for nm, x in (("o", o), ("h", h), ("n", n), ("m", m), ("T", tt), ("f", fn), ("logits", lg)):
                DC.require(DC.finite_all(x), f"链上 {nm} 出现非有限值")
                DC.require(str(x.dtype) == "torch.bfloat16", f"链上 {nm} 必须是 bf16", str(x.dtype))
            return dict(o=o, h=h, n=n, m=m, T=tt, f=fn, logits=lg)

        with torch.inference_mode():
            A1 = chain(core_A)
            A2 = chain(core_A)                     # **重复控制**
            B1 = chain(core_B)
        DC.require(torch.equal(A1["logits"], A2["logits"]),
                   "重复控制失败：同一条件两次运行 logits 不逐元素相等")
        rec["control_repeat"] = dict(A_repeat_logits_equal=True)

        # ---------------- 3) 逐阶段 A/B 比较 ----------------
        stages = {}
        for key, name in (("o", "o_proj 输出"), ("h", "残差加法后（BF16）"), ("n", "post_attention_layernorm 后"),
                          ("m", "MLP 输出"), ("T", "MLP 残差加法后（BF16）"),
                          ("f", "最终 RMSNorm 输出"), ("logits", "最终 logits")):
            stages[key] = dict(label=name, **DC.cmp_tensor(A1[key], B1[key], f"A_vs_B_{key}"))
        rec["stages"] = stages
        rec["diff_survival"] = dict(
            o_proj_n_diff=stages["o"]["n_diff"] if "n_diff" in stages["o"] else None,
            note="逐阶段是否保留差异见各阶段 elementwise_equal / abs_err_max")

        # ---------------- 4) 最终指标 ----------------
        logits_A = A1["logits"].detach().float().reshape(-1).cpu()
        logits_B = B1["logits"].detach().float().reshape(-1).cpu()
        fA, fB = _fields(logits_A, tok), _fields(logits_B, tok)
        rec["final"] = dict(A=fA, B=fB,
                            same_token=bool(fA["argmax_id"] == fB["argmax_id"]),
                            n_logits_int=int(logits_A.numel()))
        # 参考线（**独立参考，不混入 A/B 因果结论**）
        l27_out3 = T.as_row3(l27_out_f.to(torch.bfloat16), "l27_out_ref", device=dev)
        with torch.inference_mode():
            lg_ref = (fin_norm(l27_out3) @ W.T)[0, 0]
        rec["reference_only"] = dict(
            label="参考线：真实 L27 输出行 →（单行）最终 Norm →（单行）LM head",
            fields=_fields(lg_ref.detach().float().reshape(-1).cpu(), tok),
            note="**仅作同形状参考**；不等于生产四行路径的最终 logits（下游形状已统一为单行，"
                 "既往已观察到独立的形状效应）。**不**混入 A/B 因果结论。")

        # ---------------- 5) 输入未被原地修改 ----------------
        rec["inputs_unmodified"] = dict(
            core_A=bool(torch.equal(core_A, snaps["core_A"])),
            core_B=bool(torch.equal(core_B, snaps["core_B"])),
            layer_in=bool(torch.equal(layer_in, snaps["layer_in"])),
            weights=bool(torch.equal(W, W_snap)))
        DC.require(all(rec["inputs_unmodified"].values()), "输入张量或权重被原地修改",
                   str(rec["inputs_unmodified"]))

        rec["verdict"] = dict(
            outcome=("SAME_TOKEN" if rec["final"]["same_token"] else "DIFFERENT_TOKEN"),
            statement=("该位置、该固定下游条件下，L27 query 行数导致的 core 差异**不足以改变最终选择**。"
                       if rec["final"]["same_token"] else
                       "该局部干预在固定下游条件下**足以改变选择**。"),
            boundary="若不同：**不**宣称它是原始 AR/Spec 分歧的唯一根因，**不**宣称已修复生产正确性；"
                     "若相同：**不**宣称所有 attention 差异均无关。均**不**把诊断完成当作正式正确性通过；"
                     "**不**以某候选的 ULP 归一化整行误差；**不**预设 A 必须选 576、B 必须选 4710。")
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_attntailab_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{TAG}_attntailab_{run_id}.pt"
        torch.save(dict(
            logits_A=logits_A, logits_B=logits_B,
            logits_reference_only=lg_ref.detach().float().cpu(),
            core_A=core_A.detach().float().cpu(), core_B=core_B.detach().float().cpu(),
            layer_in=layer_in.detach().float().cpu(),
            stage_o_A=A1["o"].detach().float().cpu(), stage_o_B=B1["o"].detach().float().cpu(),
            stage_h_A=A1["h"].detach().float().cpu(), stage_h_B=B1["h"].detach().float().cpu(),
            stage_T_A=A1["T"].detach().float().cpu(), stage_T_B=B1["T"].detach().float().cpu(),
            stage_f_A=A1["f"].detach().float().cpu(), stage_f_B=B1["f"].detach().float().cpu(),
        ), str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        print("== 逐阶段 A vs B ==")
        for k in ("o", "h", "n", "m", "T", "f", "logits"):
            v = stages[k]
            print(f"  {v['label']:32s} equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} "
                  f"mean={v['abs_err_mean']:.6g} shape={v['shape']}")
        print("== 最终 ==")
        for nm, f_ in (("A", fA), ("B", fB)):
            print(f"  {nm}: argmax={f_['argmax_id']} ({f_['argmax_text']!r}，按 id 查表) "
                  f"score={f_['max_score']} margin={f_['margin_top1_top2']:.6g} "
                  f"tied={f_['n_max_tied']}{f_['tied_max_tokens']} "
                  f"576/715/4710={f_['watch']['576']}/{f_['watch']['715']}/{f_['watch']['4710']}")
        print(f"  同一 token? {rec['final']['same_token']} → {rec['verdict']['outcome']}")
        print(f"  参考线（仅参考）: argmax={rec['reference_only']['fields']['argmax_id']} "
              f"tied={rec['reference_only']['fields']['n_max_tied']}"
              f"{rec['reference_only']['fields']['tied_max_tokens']}")
        print(f"[ab] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_attntailab_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ab] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[ab] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
