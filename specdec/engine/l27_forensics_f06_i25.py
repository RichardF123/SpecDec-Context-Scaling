"""l27_forensics_f06_i25.py — 定位 L27 内部的对齐行差异（只读 hook，不改 forward）。

代码事实（本轮据 model_self.py 确认，§一）
  `for i, layer in enumerate(self.layers): x, new_cache = layer(x, ...)` 之后**紧接着** `x = self.norm(x)`
  ⇒ **L27 输出 == 最终 RMSNorm 的输入，中间没有任何计算**。
  因此上一轮测到的 `R` 差 2.5 **就是 L27 输出差**，不重复测；并且
  **不能**由"L27 输出不同"推出"差异一定来自更早层"（L27 自身也可能新增差异）。

观察顺序**按 DecoderLayer.forward 的实际代码**（不按模块名猜）：
  layer_in → input_layernorm → [q/k/v_proj → RoPE/attn → o_proj] → atten_out
           → attn_residual = layer_in + atten_out
           → post_attention_layernorm → MLP → layer_out（== 最终 Norm 输入）

只读插桩：`register_forward_hook`；不改任何形状/dtype/顺序；不转存全层大型 attention 张量。
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
from . import spec_cache
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
L = 27
EXPECT_BLOCK, EXPECT_SERIAL = 576, 4710
POINTS = (("layer_in", None), ("input_layernorm", "input_layernorm"),
          ("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"),
          ("v_proj", "self_attn.v_proj"), ("atten_out", "self_attn.o_proj"),
          ("post_attention_layernorm", "post_attention_layernorm"), ("mlp", "mlp"))


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _fields(row):
    f = DF.row_fields(row.detach().float().reshape(-1), watch=[715, 576, 4710])
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"])


def to_gpu(c):
    return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


class Tap:
    def __init__(self, layer, row):
        self.layer, self.row, self.hs, self.got = layer, row, [], {}

    def install(self):
        def mk(name, attr):
            def h(mod, inp, out):
                if attr is None:
                    self.got[name] = inp[0].detach().clone()      # layer 输入
                else:
                    self.got[name] = out.detach().clone()
            return h
        for name, attr in POINTS:
            mod = self.layer if attr is None else self.layer
            if attr is None:
                self.hs.append(self.layer.register_forward_hook(mk(name, None)))
            else:
                for part in attr.split("."):
                    mod = getattr(mod, part)
                self.hs.append(mod.register_forward_hook(mk(name, attr)))
        return self

    def row_(self, name):
        t = self.got[name]
        return t[0, self.row] if t.dim() == 3 else t[0]

    def remove(self):
        for h in self.hs:
            h.remove()


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="l27_forensics_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=CANDS,
                           layer=L, align_row=ALIGN_ROW, align_input_position=8216,
                           align_predicted_position=8217,
                           code_fact="L27 输出与 self.norm 输入之间无任何计算（model_self._run）",
                           observation_order=list(p[0] for p in POINTS),
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"))),
               scope="L27 内部对齐行定位；只读 hook；不改生产实现/精度/门槛；不跑生成；不转存全层大张量。")
    print(f"[l27] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[l27] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_l27_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    taps = None
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        snap = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"),
                          map_location="cpu")[f"round{ROUND}_snap_pre"]
        DC.require(spec_cache.cache_length(snap) == 8215, "cache 长度必须 8215")
        real_row = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"),
                              map_location="cpu")[f"round{ROUND}_rows"][ALIGN_ROW].float()

        # ---- 无插桩基线 + 机制：L27 输出 == 最终 Norm 输入（同一 run 内验证） ----
        with torch.inference_mode(), M.attention_phase("verify"):
            rows0, _b0 = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        plain = rows0[ALIGN_ROW].detach().float().cpu()
        norm_in: dict = {}
        hnorm = tgt.norm.register_forward_hook(
            lambda m, i, o: norm_in.__setitem__("R", i[0].detach().clone()))
        with torch.inference_mode(), M.attention_phase("verify"):
            rows1, _b1 = spec_dec._verify(tgt, to_gpu(snap), CANDS)
        inst = rows1[ALIGN_ROW].detach().float().cpu()
        R_block = norm_in.pop("R")
        hnorm.remove()

        # ---- 带 L27 插桩：块臂（同时取 DecoderLayer 输出，用于核验"L27 输出==Norm 输入"）----
        taps = Tap(tgt.layers[L], ALIGN_ROW).install()
        layer_out = {}
        hl = tgt.layers[L].register_forward_hook(
            lambda m, i, o: layer_out.__setitem__(
                "out", (o if torch.is_tensor(o) else o[0]).detach().clone()))
        hnorm = tgt.norm.register_forward_hook(
            lambda m, i, o: norm_in.__setitem__("R", i[0].detach().clone()))
        with torch.inference_mode(), M.attention_phase("verify"):
            spec_dec._verify(tgt, to_gpu(snap), CANDS)
        blk = {k: taps.row_(k) for k, _ in POINTS}
        R_tap_block = norm_in.pop("R")
        LO_block = layer_out.pop("out")
        taps.remove(); hl.remove(); hnorm.remove()

        # ---- 串行臂 ----
        w = to_gpu(snap)
        with torch.inference_mode(), M.attention_phase("repair"):
            spec_dec._step(tgt, CAND0, w)
        taps = Tap(tgt.layers[L], 0).install()
        hl = tgt.layers[L].register_forward_hook(
            lambda m, i, o: layer_out.__setitem__(
                "out", (o if torch.is_tensor(o) else o[0]).detach().clone()))
        hnorm = tgt.norm.register_forward_hook(
            lambda m, i, o: norm_in.__setitem__("R", i[0].detach().clone()))
        with torch.inference_mode(), M.attention_phase("repair"):
            lg_s, _c = spec_dec._step(tgt, FED, w)
        ser = {k: taps.row_(k) for k, _ in POINTS}
        R_tap_ser = norm_in.pop("R")
        LO_ser = layer_out.pop("out")
        taps.remove(); hl.remove(); hnorm.remove()
        serial_row = lg_s[0, -1].detach().float().cpu()

        # ---- 派生的 attn 残差（= layer_in + atten_out，纯加，标注为派生） ----
        def add_res(d):
            return (d["layer_in"].float() + d["atten_out"].float())
        blk["attn_residual(derived)"] = add_res(blk)
        ser["attn_residual(derived)"] = add_res(ser)
        blk["layer_out"] = LO_block.float()[0, ALIGN_ROW]
        ser["layer_out"] = LO_ser.float()[0, 0]

        # ---- 机制校验 ----
        checks = dict(
            instrumented_vs_plain=DC.cmp_tensor(inst, plain, "instrumented_vs_plain"),
            block_vs_saved_real=DC.cmp_tensor(plain, real_row, "plain_vs_saved_real"),
            block_is_576=(_fields(plain)["argmax_id"] == EXPECT_BLOCK),
            serial_is_4710=(_fields(serial_row)["argmax_id"] == EXPECT_SERIAL),
            L27_out_equals_norm_input_block=DC.cmp_tensor(LO_block.float(), R_block.float(),
                                                          "L27out_vs_normIn_block"),
            L27_out_equals_norm_input_serial=DC.cmp_tensor(LO_ser.float(), R_tap_ser.float(),
                                                           "L27out_vs_normIn_serial"))
        DC.require(checks["L27_out_equals_norm_input_block"]["elementwise_equal"]
                   and checks["L27_out_equals_norm_input_serial"]["elementwise_equal"],
                   "L27 输出必须与最终 Norm 输入逐元素相同（代码事实核验）")
        DC.require(checks["block_is_576"] and checks["serial_is_4710"], "复现失败")

        # ---- 逐点比较 ----
        rows = {}
        for k in blk:
            a = blk[k].detach().float().cpu() if torch.is_tensor(blk[k]) else blk[k]
            b = ser[k].detach().float().cpu() if torch.is_tensor(ser[k]) else ser[k]
            rows[k] = DC.cmp_tensor(a, b, f"L27.{k}")
        rec["checks"] = checks
        rec["points"] = {k: {kk: v[kk] for kk in ("shape", "dtype", "elementwise_equal",
                                                  "abs_err_max", "abs_err_mean", "max_err_index",
                                                  "a_at_max", "b_at_max")}
                         for k, v in rows.items()}
        first = next((k for k in rows if not rows[k]["elementwise_equal"]), None)
        rec["summary"] = dict(first_differing_point=first,
                              all_points={k: rows[k]["elementwise_equal"] for k in rows})
        rec["kv_position_alignment"] = dict(
            common_cache_len=8215,
            block=dict(T=8219, s=4, new_positions=[8215, 8216, 8217, 8218],
                       position_8215_token=CAND0,
                       position_8215_provenance="作为**块输入**在本轮前向中计算",
                       aligned_row_visible_max=8216,
                       mask="triu(diag=8216)"),
            serial=dict(step1_T=8216, step2_T=8217, s=1,
                        position_8215_token=CAND0,
                        position_8215_provenance="由**串行第一步**（独立前向）写入 cache",
                        aligned_row_visible_max=8216, mask="s==1 ⇒ 不应用 mask"),
            note="两臂在位置 8215 的 **token 相同**，但其 K/V 由**不同的前向调用**产生 ⇒ "
                 "**不能**把「隐藏向量相同」当作「attention 全部输入相同」。")
        rec["stage"] = "complete"
        p = OUTDIR / f"{SRC_TAG}_l27_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{SRC_TAG}_l27_{run_id}.pt"
        torch.save(dict(**{f"block_{k}": (blk[k] if torch.is_tensor(blk[k]) else blk[k])
                           for k in blk},
                        **{f"serial_{k}": (ser[k] if torch.is_tensor(ser[k]) else ser[k])
                           for k in ser},
                        R_block=R_block.float().cpu(), R_tap_block=R_tap_block.float().cpu(),
                        R_tap_ser=R_tap_ser.float().cpu(), plain=plain, real=real_row,
                        serial_row=serial_row), str(tp))
        rec["rows_dump"] = str(tp)
        print("== 机制校验 ==")
        print(f"  插桩 vs 无插桩 logits: equal={checks['instrumented_vs_plain']['elementwise_equal']} "
              f"max|Δ|={checks['instrumented_vs_plain']['abs_err_max']}")
        print(f"  L27 输出 == 最终 Norm 输入: block={checks['L27_out_equals_norm_input_block']['elementwise_equal']} "
              f"serial={checks['L27_out_equals_norm_input_serial']['elementwise_equal']}（代码事实核验）")
        print(f"  复现 576/4710: {checks['block_is_576']}/{checks['serial_is_4710']}")
        print("== L27 逐点（对齐行；块 vs 串行） ==")
        for k, v in rows.items():
            print(f"  {k:32s} shape={v['shape']} equal={v['elementwise_equal']} "
                  f"max|Δ|={v['abs_err_max']:.6g} mean={v['abs_err_mean']:.6g} "
                  f"idx={v['max_err_index']} a={v['a_at_max']:.6g} b={v['b_at_max']:.6g}")
        print(f"  首个不同点 = {first}")
        print(f"[l27] RESULT_JSON={p}")
        return 0
    except Exception as e:                                       # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e); rec["traceback"] = traceback.format_exc()
        try:
            if taps is not None:
                taps.remove()
        except Exception:                                        # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{SRC_TAG}_l27_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[l27] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[l27] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
