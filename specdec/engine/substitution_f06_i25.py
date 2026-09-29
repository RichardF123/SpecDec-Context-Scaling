"""substitution_f06_i25.py — 四臂中间值替换诊断：L0 的 q_proj / o_proj 差异是否影响 index25 选词。

固定状态（§三）
  原始 Spec **round 11 之前**的 cache，长度 **8215**；候选 `[27685,659,715,4882]`；
  对齐块行 `rows[1]`：**输入位置 8216 → 预测位置 8217**；BF16 / base_bf16 / 原 backend 与 phase 不变；
  QK32/prob32 关闭；**不改变**矩阵形状、mask、位置、权重；**不跑生成**（不缩短 max_new）；
  每个臂从**独立 cache 副本**开始。

机制（§四）
  1. 参考 1：无干预块重放 → 必须逐元素复现已保存块行、argmax=576。
  2. 参考 2：同 cache 连续输入前两个候选 → argmax=4710。
  3. 串行参考：把**串行第二步** L0 `q_proj`/`o_proj` 的**原始 dtype 输出**克隆固定，供所有替换臂共用。
  4. 无操作 hook 必须不改变结果。
  5. 替换只作用于**指定模块的对齐行**：其他行不变、权重与源 cache 不变、hook 次数符合预期。

四臂（§五）
  A 原始块（无替换）；B 仅替换 L0.q_proj 对齐行；C 仅替换 L0.o_proj 对齐行；D 同时替换。
  ⚠️ **D 不等于两个独立因素之和**：o_proj 在 q_proj 下游，D 可能覆盖部分上游作用。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from .. import results_schema as S
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
ALIGN_ROW = 1                    # 块路径中对齐到输入位置 8216 的行
EXPECT_BLOCK, EXPECT_SERIAL = 576, 4710


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cmp(a, b, name):
    d = (a - b).abs()
    flat = int(d.reshape(-1).argmax())
    return dict(name=name, shape=[int(x) for x in a.shape], dtype=str(a.dtype),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                elementwise_equal=bool(torch.equal(a, b)),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                max_err_index=[int(x) for x in torch.unravel_index(torch.tensor(flat), d.shape)],
                a_absmax=float(a.abs().max()))


def _fields(row):
    from . import diag_fields as DF
    r = row.detach().float().reshape(-1)
    f = DF.row_fields(r, watch=[715, 576, 4710])
    t2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(t2[0] - t2[1])
    f["tie_at_top"] = bool(float(t2[0]) == float(t2[1]))
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"],
                n_max_tied=f["n_max_tied"], tied_max_tokens=f["tied_max_tokens"],
                margin_top1_top2=f["margin_top1_top2"], tie_at_top=f["tie_at_top"],
                scores={str(t): float(r[t]) for t in (715, 576, 4710)})


def to_gpu(c):
    return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="substitution_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=CANDS,
                           align_row=ALIGN_ROW, align_input_position=8216,
                           align_predicted_position=8217,
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt")),
                           source_rows=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"))),
               scope="四臂中间值替换诊断；不改生产实现/精度/门槛；不跑生成；无性能结论。")
    print(f"[sub] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[sub] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_substitution_{run_id}.json").write_text(
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
    rec["source_hash"] = dict(cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt")).split("\\")[-1],
                              rows=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt")).split("\\")[-1])
    print(f"[sub] cache_len={rec['cache_len']} (expect 8215)")

    mods = dict(q_proj=tgt.layers[0].self_attn.q_proj, o_proj=tgt.layers[0].self_attn.o_proj)

    # ---- 参考 2 + 串行参考（§四.2/3）：连续两步，抓第二步 L0 q/o 的原始 dtype 输出 ----
    ser_ref: dict = {}
    hooks = []

    def _grab(name):
        def h(m, inp, out):
            ser_ref[name] = out.detach().clone()
            ser_ref[f"{name}_shape"] = [int(x) for x in out.shape]
        return h

    for nm, mod in mods.items():
        hooks.append(mod.register_forward_hook(_grab(f"serial_{nm}")))
    with torch.inference_mode():
        w = to_gpu(snap)
        with M.attention_phase("repair"):
            spec_dec._step(tgt, CAND0, w)
        with M.attention_phase("repair"):
            lg_ser, _c = spec_dec._step(tgt, FED, w)
    for h in hooks:
        h.remove()
    serial_row = lg_ser[0, -1].detach().float().cpu()
    rec["serial_reference"] = dict(
        logits=_fields(serial_row), argmax_expected=EXPECT_SERIAL,
        matches=(_fields(serial_row)["argmax_id"] == EXPECT_SERIAL),
        captured={k: dict(shape=ser_ref.get(f"{k}_shape"),
                          dtype=str(ser_ref[k].dtype)) for k in ("serial_q_proj", "serial_o_proj")})
    print(f"[sub] 串行参考 argmax={rec['serial_reference']['logits']['argmax_id']} "
          f"(expect {EXPECT_SERIAL}) match={rec['serial_reference']['matches']}")

    ref_rows = {nm: ser_ref[f"serial_{nm}"][0, -1].clone() for nm in mods}   # (D,) 对齐行
    rec["reference_rows"] = {nm: dict(shape=[int(x) for x in ref_rows[nm].shape],
                                      dtype=str(ref_rows[nm].dtype),
                                      clone_independent=True) for nm in ref_rows}

    # ---- 四个臂 ----
    def run_arm(name: str, replace: tuple):
        applied = []
        hs = []

        def _mk(nm):
            def h(m, inp, out):
                o = out.clone()                       # 不改原输出张量，返回替换后的新张量
                before = o[0, ALIGN_ROW].clone()
                o[0, ALIGN_ROW] = ref_rows[nm].to(o.dtype).to(o.device)
                applied.append(dict(module=f"L0.{nm}", row=ALIGN_ROW,
                                    shape=[int(x) for x in o.shape], dtype=str(o.dtype),
                                    replaced_elems=int(ref_rows[nm].numel()),
                                    delta_max=float((o[0, ALIGN_ROW].float()
                                                     - before.float()).abs().max())))
                return o
            return h

        for nm in replace:
            hs.append(mods[nm].register_forward_hook(_mk(nm)))
        try:
            with torch.inference_mode(), M.attention_phase("verify"):
                rows, _blk = spec_dec._verify(tgt, to_gpu(snap), CANDS)
            row = rows[ALIGN_ROW].detach().float().cpu()
            other_rows = rows.detach().float().cpu()
        finally:
            for h in hs:
                h.remove()
        return dict(arm=name, replaced=list(replace), hooks_applied=applied,
                    hook_calls=len(applied),
                    logits=_fields(row), row=row, all_rows=other_rows)

    A = run_arm("A_original_block", ())
    Anoop = None
    # §四.4 无操作 hook：注册一个直接返回原输出的 hook
    hs = []
    _noop_calls = []

    def _noop(m, inp, out):
        _noop_calls.append(1)
        return out

    for nm, mod in mods.items():
        hs.append(mod.register_forward_hook(_noop))
    try:
        with torch.inference_mode(), M.attention_phase("verify"):
            rows_n, _b = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        Anoop = dict(row=rows_n[ALIGN_ROW].detach().float().cpu(), calls=len(_noop_calls))
    finally:
        for h in hs:
            h.remove()

    B = run_arm("B_q_proj_only", ("q_proj",))
    C = run_arm("C_o_proj_only", ("o_proj",))
    D = run_arm("D_q_and_o", ("q_proj", "o_proj"))

    # ---- 校验（§四.1/4/5） ----
    rec["checks"] = dict(
        block_replay_equals_saved_real=_cmp(A["row"], real_row, "A_vs_saved_real"),
        block_argmax_is_576=(A["logits"]["argmax_id"] == EXPECT_BLOCK),
        noop_hook_equals_A=_cmp(Anoop["row"], A["row"], "noop_vs_A"),
        noop_hook_calls=Anoop["calls"],
        serial_argmax_is_4710=rec["serial_reference"]["matches"],
        replacement_scope_only_aligned_row=dict(
            A_other_rows_vs_A=_cmp(A["all_rows"][[0, 2, 3]], A["all_rows"][[0, 2, 3]],
                                   "A_non_target_rows_selfcheck"),
            note="非目标行完整性：见各臂 non_target_rows_vs_A 字段（与 A 的同位置行逐元素比较）"),
    )

    arms_out = {}
    for arm in (A, B, C, D):
        nm = arm["arm"]
        non_target = None
        if nm != "A_original_block":
            idx = [0, 2, 3]
            non_target = _cmp(arm["all_rows"][idx], A["all_rows"][idx], f"{nm}_non_target_rows_vs_A")
        arms_out[nm] = dict(
            replaced=arm["replaced"], hooks_applied=arm["hooks_applied"],
            hook_calls=arm["hook_calls"],
            logits=arm["logits"],
            vs_A=_cmp(arm["row"], A["row"], f"{nm}_vs_A"),
            vs_serial=_cmp(arm["row"], serial_row, f"{nm}_vs_serial"),
            non_target_rows_vs_A=non_target,
            argmax_changed_vs_A=(arm["logits"]["argmax_id"] != A["logits"]["argmax_id"]),
            argmax_is_4710=(arm["logits"]["argmax_id"] == EXPECT_SERIAL),
        )
    rec["arms"] = arms_out
    rec["noop_hook"] = dict(calls=Anoop["calls"], equals_A=rec["checks"]["noop_hook_equals_A"])
    rec["stage"] = "complete"

    p = OUTDIR / f"{SRC_TAG}_substitution_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tp = OUTDIR / f"{SRC_TAG}_substitution_{run_id}.pt"
    torch.save({nm: dict(row=arm["row"], all_rows=arm["all_rows"]) for nm, arm in
                (("A", A), ("B", B), ("C", C), ("D", D))} |
               dict(serial=serial_row, real=real_row,
                    ref=ref_rows), str(tp))
    rec["rows_dump"] = str(tp)

    print("== 校验 ==")
    for k in ("block_replay_equals_saved_real", "noop_hook_equals_A"):
        c = rec["checks"][k]
        print(f"  {k}: elementwise_equal={c['elementwise_equal']} max|Δ|={c['abs_err_max']}")
    print(f"  block argmax=576: {rec['checks']['block_argmax_is_576']} | "
          f"serial argmax=4710: {rec['checks']['serial_argmax_is_4710']}")
    print(f"  noop hook 调用次数={Anoop['calls']}")
    print("== 四臂 ==")
    for nm in ("A_original_block", "B_q_proj_only", "C_o_proj_only", "D_q_and_o"):
        e = arms_out[nm]
        print(f"  {nm:18s} replaced={e['replaced']} calls={e['hook_calls']} "
              f"argmax={e['logits']['argmax_id']} tied={e['logits']['n_max_tied']}{e['logits']['tied_max_tokens']} "
              f"margin={e['logits']['margin_top1_top2']} | vs_A max|Δ|={e['vs_A']['abs_err_max']:.6g} "
              f"equal={e['vs_A']['elementwise_equal']} | vs_serial max|Δ|={e['vs_serial']['abs_err_max']:.6g} "
              f"equal={e['vs_serial']['elementwise_equal']}")
    print(f"[sub] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
