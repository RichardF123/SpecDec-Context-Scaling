"""lmhead_shape_f06_i25.py — 隔离**输出端**：隐藏向量来源 × LM head 处理形状（F06/index25）。

背景（沿用已确认结果）
  同起始 cache（round 11 前，长度 8215）：原块路径 argmax=**576**，串行路径 argmax=**4710**。
  第 0 层 Q-only 替换不改变最终 logits；O-only 与 Q+O 改变部分 logits 但**关键候选分数与选词不变**。
  ⇒ 不再对第 0 层追加干预，也不扫描 27 层找"根因层"，而是**直接检查实际输出端**：
      `x = self.norm(x)` → `logits = x @ self.embed_tokens.weight.T`

四格（同一份权重、**同一原始表达式与 dtype**）
  A 块向量/块形状：`H_B @ W.T` → 取对齐行 index=1
  B 块向量/单行形状：`H_B[0,1:2] @ W.T`
  C 串行向量/块形状：克隆 H_B，**只**把行 1 换成 H_S[0,0]，`@ W.T` → 取行 1
  D 串行向量/单行形状：`H_S @ W.T`
  配对解释：A↔B、C↔D = **同对齐向量、不同投影形状**；A↔C、B↔D = **同投影形状、不同向量来源**。
  另加一个**用途明确的布局控制**：把同一切片的 (view) 与 (contiguous clone) 各投影一次，用于区分
  "M 维度效应" 与 "布局/stride 效应"，避免把布局差异当成纯 M 效应。

边界：仅固定案例；不改生产代码/精度/KV/argmax 规则/门槛；不跑生成；不增加精度干预；无性能结论。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from .. import results_schema as S
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


def _layout(t: torch.Tensor) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device), is_contiguous=bool(t.is_contiguous()),
                is_view=bool(t._base is not None))


def _cmp(a, b, name):
    d = (a - b).abs()
    flat = int(d.reshape(-1).argmax())
    return dict(name=name, shape=[int(x) for x in a.shape], dtype=str(a.dtype),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                elementwise_equal=bool(torch.equal(a, b)),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                max_err_index=[int(x) for x in torch.unravel_index(torch.tensor(flat), d.shape)])


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
    rec = dict(run_id=run_id, tool="lmhead_shape_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=CANDS,
                           align_row=ALIGN_ROW, align_input_position=8216,
                           align_predicted_position=8217,
                           logits_expression="logits = x @ self.embed_tokens.weight.T",
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt")),
                           source_rows=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"))),
               scope="LM head 边界四格隔离；不改生产实现/精度/门槛；不跑生成；无性能结论。")
    print(f"[lm] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[lm] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_lmhead_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

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

    # ---- 只读捕获最终 RMSNorm 之后、LM head 之前 ----
    held: dict = {}
    h = []

    def _mk(key):
        def hook(module, inp, out):
            held[key] = out.detach().clone()
        return hook

    h.append(tgt.norm.register_forward_hook(_mk("H")))
    try:
        with torch.inference_mode(), M.attention_phase("verify"):
            rows_b, _blk = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        H_B = held.pop("H")
        block_row = rows_b[ALIGN_ROW].detach().float().cpu()
        w = to_gpu(snap)
        with torch.inference_mode(), M.attention_phase("repair"):
            spec_dec._step(tgt, CAND0, w)
        with torch.inference_mode(), M.attention_phase("repair"):
            lg_s, _c = spec_dec._step(tgt, FED, w)
        H_S = held.pop("H")
        serial_row = lg_s[0, -1].detach().float().cpu()
    finally:
        for x in h:
            x.remove()

    W = tgt.embed_tokens.weight
    rec["captured"] = dict(H_B=_layout(H_B), H_S=_layout(H_S), W=_layout(W),
                           H_B_expected=[1, 4, 1536], H_S_expected=[1, 1, 1536],
                           shapes_as_expected=(list(H_B.shape) == [1, 4, 1536]
                                               and list(H_S.shape) == [1, 1, 1536]),
                           raw_dtype_preserved=str(H_B.dtype))
    print(f"[lm] H_B={list(H_B.shape)} {H_B.dtype} contiguous={H_B.is_contiguous()} stride={list(H_B.stride())}")
    print(f"[lm] H_S={list(H_S.shape)} {H_S.dtype} contiguous={H_S.is_contiguous()}")

    # ---- 复现检查 ----
    rec["repro"] = dict(
        block_vs_saved_real=_cmp(block_row, real_row, "block_row_vs_saved_real"),
        block_argmax=DF.row_fields(block_row, watch=[715, 576])["argmax_id"],
        block_is_576=(DF.row_fields(block_row, watch=[715, 576])["argmax_id"] == EXPECT_BLOCK),
        serial_argmax=DF.row_fields(serial_row, watch=[715, 4710])["argmax_id"],
        serial_is_4710=(DF.row_fields(serial_row, watch=[715, 4710])["argmax_id"] == EXPECT_SERIAL),
    )
    print(f"[lm] repro: block argmax={rec['repro']['block_argmax']} "
          f"({rec['repro']['block_is_576']}) serial argmax={rec['repro']['serial_argmax']} "
          f"({rec['repro']['serial_is_4710']})")

    # ---- 四格 + 布局控制（全部只读，不原地修改任何源张量） ----
    v_block = H_B[0, ALIGN_ROW:ALIGN_ROW + 1]           # (1,1536) 对齐向量的**切片**（可能是 view）
    v_serial = H_S[0, 0:1]                              # (1,1536)
    H_Bc = H_B.clone()                                  # 只改行 1，其余行逐一核对
    H_Bc[0, ALIGN_ROW] = H_S[0, 0]
    rows_other_equal = bool(torch.equal(H_Bc[0, [0, 2, 3]], H_B[0, [0, 2, 3]]))

    with torch.inference_mode():
        A = (H_B @ W.T)[0, ALIGN_ROW].detach().float().cpu()
        # ⚠️ (1,V) 取行要用 [0]；写 [0,0] 会取到**标量**，后面 topk(k=5) 直接越界（首跑即此错）。
        B = (v_block @ W.T)[0].detach().float().cpu()
        C = (H_Bc @ W.T)[0, ALIGN_ROW].detach().float().cpu()
        D = (v_serial @ W.T)[0].detach().float().cpu()
        # 布局控制：同一切片 view vs contiguous clone
        B_view = (v_block @ W.T)[0].detach().float().cpu()
        B_contig = (v_block.contiguous() @ W.T)[0].detach().float().cpu()

    cells = dict(
        A=dict(cell="块向量 × 块形状", vector="H_B row1 (view)", input=_layout(H_B), logits=_fields(A)),
        B=dict(cell="块向量 × 单行形状", vector="H_B row1 (slice view)", input=_layout(v_block),
               logits=_fields(B)),
        C=dict(cell="串行向量 × 块形状", vector="H_B with row1 := H_S", input=_layout(H_Bc),
               logits=_fields(C)),
        D=dict(cell="串行向量 × 单行形状", vector="H_S row0 (slice view)", input=_layout(v_serial),
               logits=_fields(D)),
    )
    pairs = dict(
        A_vs_B_shape_effect_block_vector=_cmp(A, B, "A_vs_B"),
        C_vs_D_shape_effect_serial_vector=_cmp(C, D, "C_vs_D"),
        A_vs_C_source_effect_block_shape=_cmp(A, C, "A_vs_C"),
        B_vs_D_source_effect_single_shape=_cmp(B, D, "B_vs_D"),
        layout_control_view_vs_contiguous=_cmp(B_view, B_contig, "B_view_vs_contig"),
    )
    checks = dict(
        A_equals_original_block_logits=_cmp(A, block_row, "A_vs_original_block"),
        D_equals_serial_logits=_cmp(D, serial_row, "D_vs_serial"),
        # ⚠️ v_block=H_B[0,1:2] 形状是 (1,1536) ⇒ 取行要用 [0]；写 [0,0] 会得到**标量**，
        #    与 (1536,) 行相比会广播成"逐元素差"，报出虚假的 max|Δ|（首跑即此错）。
        aligned_input_A_B_identical=_cmp(H_B[0, ALIGN_ROW], v_block[0], "aligned_input_A_B"),
        aligned_input_C_D_identical=_cmp(H_S[0, 0], v_serial[0], "aligned_input_C_D"),
        C_replaced_only_row1=bool(rows_other_equal),
        weights_unmodified=bool(torch.equal(W, tgt.embed_tokens.weight)),
        H_B_unmodified=bool(torch.equal(H_B[0, ALIGN_ROW], v_block[0, 0])),
        finite_all=all(c["logits"]["scores"] and True for c in cells.values()),
    )
    rec.update(cells=cells, pairs=pairs, checks=checks)
    rec["H_diff"] = _cmp(H_B[0, ALIGN_ROW].float().cpu(), H_S[0, 0].float().cpu(), "H_B_row1_vs_H_S")
    rec["verdict"] = dict(
        shape_effect_changes_choice=(pairs["A_vs_B_shape_effect_block_vector"]["elementwise_equal"] is False
                                     and cells["A"]["logits"]["argmax_id"] != cells["B"]["logits"]["argmax_id"]),
        shape_effect_block_shape_note="A vs B：同对齐向量、块形状 vs 单行形状",
        source_effect_changes_choice=(pairs["A_vs_C_source_effect_block_shape"]["elementwise_equal"] is False
                                      and cells["A"]["logits"]["argmax_id"] != cells["C"]["logits"]["argmax_id"]),
        rule="选词改变 ⇒ 该差异在**该固定输入**上足以改变选择（不等于解释全部原因）；"
             "logits 不同但选词相同 ⇒ 只证明数值差异；同向量投影结果相同 ⇒ 本例未观察到 LM head 形状效应。",
        no_ground_truth_note="本轮不使用更高精度结果作为真值，也不新增精度干预。")
    rec["stage"] = "complete"
    p = OUTDIR / f"{SRC_TAG}_lmhead_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tp = OUTDIR / f"{SRC_TAG}_lmhead_{run_id}.pt"
    torch.save(dict(H_B=H_B.detach().float().cpu(), H_S=H_S.detach().float().cpu(),
                    A=A, B=B, C=C, D=D, block_row=block_row, serial_row=serial_row), str(tp))
    rec["rows_dump"] = str(tp)

    print("== 机制校验 ==")
    for k in ("A_equals_original_block_logits", "D_equals_serial_logits",
              "aligned_input_A_B_identical", "aligned_input_C_D_identical",
              "layout_control_view_vs_contiguous"):
        c = rec["checks"].get(k) or pairs[k]
        print(f"  {k}: elementwise_equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']:.6g}")
    print(f"  C 只替换行1: {checks['C_replaced_only_row1']} | 权重未改: {checks['weights_unmodified']}")
    print(f"  H_B row1 vs H_S: max|Δ|={rec['H_diff']['abs_err_max']:.6g} equal={rec['H_diff']['elementwise_equal']}")
    print("== 四格 ==")
    for k in ("A", "B", "C", "D"):
        e = cells[k]
        print(f"  {k} {e['cell']:16s} argmax={e['logits']['argmax_id']} tied={e['logits']['n_max_tied']}"
              f"{e['logits']['tied_max_tokens']} margin={e['logits']['margin_top1_top2']} "
              f"576/715/4710={e['logits']['scores']['576']}/{e['logits']['scores']['715']}/{e['logits']['scores']['4710']}")
    print("== 配对 ==")
    for k, c in pairs.items():
        print(f"  {k}: equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']:.6g}")
    print(f"[lm] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
