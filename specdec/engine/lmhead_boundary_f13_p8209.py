"""lmhead_boundary_f13_p8209.py — 位置 8209 的 **LM head 边界**定位（F13；**有界、单次**）。

## 要回答的唯一问题
同一份串行历史 `H(8208)`、同一个输入位置 8208（token `20642`），
**S（单步，喂 `[20642]`）** 与 **B（块，喂 `[20642,659,576,4531]`，取 `rows[0]`）** 的判定行不同：

| | 323 | 659 | argmax |
| --- | --- | --- | --- |
| S | 23.75 | 23.75 | **323**（精确并列 ⇒ `argmax` 取最小索引） |
| B | 23.625 | 23.75 | **659** |

问题：这个差异是**已经进入 LM head 的输入**（最终 RMSNorm 输出 `hS` vs `hB`）造成的，
还是**最终投影本身的执行形状**（`(1,1,H)@W.T` vs `(1,4,H)@W.T`）造成的？

## 边界（以实际代码为准）
`model_self.QwenLM._run`：`x = self.norm(x)` → `logits = x @ self.embed_tokens.weight.T`。
**没有独立 `lm_head` 模块**（绑定权重）。本工具捕获 `self.norm` 的输出（`lmhead_boundary.py`），
它**就是**权重乘法的左操作数（CPU 门用 `y @ W.T == _run 返回的 logits` **逐位**证明）。

## 四格（同一份真实权重 `W`、生产表达式、生产 dtype=bf16、完整词表）
|  | 单行投影 `(1,1,H)@W.T` | 四行投影 `(1,4,H)@W.T` → 取第 0 行 |
| --- | --- | --- |
| 输入 `hS` | `HS_1` = **真实 S**（控制） | `HS_4`（B 的完整隐藏张量，**只**把第 0 行换成 `hS`） |
| 输入 `hB` | `HB_1`（B 第 0 行 → `(1,1,H)` contiguous） | `HB_4` = **真实 B**（控制） |

另加**布局控制** `HB_1_view`（同一切片的 view 形式），用于区分"形状/M 维效应"与"布局/stride 效应"。

## 归因规则（与用户给定一致，工具只报事实）
* 固定投影形状、更换 `hS`/`hB` 后翻转 ⇒ 支持**输入差异**影响决策；
* 固定隐藏向量、更换单行/四行后翻转 ⇒ 支持**投影形状**影响决策；
* 两者互相依赖 ⇒ 如实记**交互**，不强行指定唯一原因；
* `hS ≠ hB` 只说明"差异已进入 LM head"，**不能**推出上游某个算子有 bug；
* 本实验**不**判定 323/659 谁是数学真值；margin / fp32 ULP **不得**当作正确性豁免；
* 局部对照清楚 ≠ B6 通过或修复完成。

## 边界与纪律
不改生产实现、proposal、门槛、精度开关；不自由生成；不扫 family/精度/算子；**不**做全模型 FP32；
**本轮不加 FP32 投影**（避免同时引入第三个变量）；GPU 单进程串行；硬失败落盘停止、无自动重试；
旧产物不覆盖；计时只作诊断成本、不作性能声明。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import cache_clone as CC
from . import decision_rows as DR
from . import diag_fields as DF
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("LMH_TAG", "div8209_lmh")
MODE = os.environ.get("LMH_MODE", "gpu").strip().lower()
PRE = int(os.environ.get("LMH_PRE", "8208"))
TARGET_POS = int(os.environ.get("LMH_TARGET", "8209"))
K = int(os.environ.get("LMH_K", "4"))
FEED = [int(x) for x in os.environ.get("LMH_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("LMH_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("LMH_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("LMH_REPEAT", "2"))
TARGET_ROPE = os.environ.get("LMH_TARGET_ROPE", "cpu_fp32").strip().lower()
CAP_H = os.environ.get("LMH_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("LMH_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
SAVED_PT = os.environ.get("LMH_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
SAVED_JSON = os.environ.get("LMH_SAVED_JSON", str(OUTDIR / "div8209_cells_20261002T064415Z.json"))
REF_RUN_AR = os.environ.get("LMH_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("LMH_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
CAPTURE_OFF_CHECK = S.env_flag("LMH_CAPTURE_OFF_CHECK", True)
DELTA_CAP = int(os.environ.get("LMH_DELTA_CAP", "12"))
KV, ARM = "base_bf16", os.environ.get("LMH_ARM", "eager")
DEC_SLOT = TARGET_POS - PRE                  # 1
ROW_INDEX = DEC_SLOT - 1                     # 0
# 四格的两两比较键（**显式列出**：`step3_compare` 里还放了一条 `naming` 说明字符串，
# 直接 `for k, v in ...items()` 遍历会拿字符串当 dict 用 ⇒ 首跑即 `TypeError`，
# 白付一次模型加载。这类"字典里混了说明字段"的坑一律用**显式键列表**避免。）
PAIR_KEYS = ("source_effect_1row_HS_1_vs_HB_1", "source_effect_4row_HS_4_vs_HB_4",
             "shape_effect_with_hS_HS_1_vs_HS_4", "shape_effect_with_hB_HB_1_vs_HB_4",
             "layout_control_HB_1_view_vs_contig")


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _to_dev(snap, dev):
    """从 CPU 快照造一份**独立完整**的设备缓存（deepcopy ⇒ 与源不共享存储）。"""
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


def _fields(row) -> dict:
    """一行 logits 的口径明确字段（复用 `diag_fields` 的唯一定义）。"""
    f = DF.row_fields(row, watch=list(WATCH))
    tk = f["topk"]
    f["margin_top1_top2"] = float(tk[0]["score"] - tk[1]["score"]) if len(tk) >= 2 else None
    f["dtype"] = str(row.dtype)
    f["shape"] = [int(x) for x in row.shape]
    f["sha16"] = A.stable_sha(row)
    return f


def _set_precision() -> dict:
    b = torch.backends.cuda.matmul
    d0 = bool(getattr(b, "allow_bf16_reduced_precision_reduction"))
    b.allow_bf16_reduced_precision_reduction = True
    return dict(default_at_import=d0, set_to=True,
                effective_after_set=bool(b.allow_bf16_reduced_precision_reduction),
                allow_tf32=bool(b.allow_tf32),
                note="与 AR/Spec/重建 smoke 运行记录的 effective 值（True）一致；"
                     "`allow_tf32` 沿用环境默认并如实记录")


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="lmhead_boundary_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209 的 S/B 判定差异来自 LM head 的**输入**，还是最终投影的**执行形状**？",
               semantics="捕获最终 RMSNorm 输出 → 四格（输入来源 × 投影形状）+ 布局控制；"
                         "四格用生产表达式 `h @ embed_tokens.weight.T`、生产 dtype(bf16)、**完整词表**；"
                         "**不加 FP32 投影**（避免引入第三个变量）。",
               config=dict(tag=TAG, pre=PRE, target_pos=TARGET_POS, dec_slot=DEC_SLOT,
                           row_index=ROW_INDEX, k=K, feed=FEED, cands=CANDS, watch=list(WATCH),
                           repeat=REPEAT, target_rope=TARGET_ROPE,
                           capture_off_check=CAPTURE_OFF_CHECK,
                           cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           saved_pt=str(SAVED_PT), saved_json=str(SAVED_JSON),
                           in_scope="只定位 LM head 边界；不逐层定位、不改计算路径、不做 FP32 投影"))
    print(f"[lmh] run_id={run_id} pre={PRE} target={TARGET_POS} feed={FEED} cands={CANDS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[lmh] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        _write(rec, run_id)
        return 3
    return _run(rec, run_id)


def _write(rec, run_id, selftest: bool = False) -> Path:
    name = (f"{TAG}_{run_id}_selftest.json" if selftest else f"{TAG}_{run_id}.json")
    p = OUTDIR / name
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 真实运行
def _run(rec: dict, run_id: str) -> int:
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        dev = next(tgt.parameters()).device
        rec["precision"] = _set_precision()
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["code_facts"] = LB.code_facts(tgt)
        print(f"[lmh] 边界：{rec['code_facts']['boundary']}；"
              f"无独立 lm_head 模块={rec['code_facts']['has_lm_head_module'] is False}")
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))

        # ---------------- H（与已保存四格 run **同一份**捕获、同一截断）----------------
        cap = C._load_cap(Path(CAP_H))
        snap = cap["snapshot_cpu"]
        src_len = C._cache_len(snap)
        H, tinfo = C._truncate(snap, PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=src_len,
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H),
                               keys=sorted(H[0].keys()), dtype=str(H[0]["k"].dtype),
                               k_shape=[int(x) for x in H[0]["k"].shape],
                               note="与 `div8209_cells_20261002T064415Z` 同一份捕获与同一**纯切片**截断 "
                                    "⇒ 本 run 的行可与那次已保存的局部对照行逐元素比较")
        if Path(CAP_H2).exists():
            snap2 = C._load_cap(Path(CAP_H2))["snapshot_cpu"]
            H2, _t2 = C._truncate(snap2, PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2),
                                     compare=C._cache_bitequal(H, H2, "H_vs_H2"),
                                     note="第二份捕获（重建历史）在共同前缀上必须与本 H 逐位相同，"
                                          "否则本 run 的起点与已保存的四格不可比")
            DR.require(rec["H_identity"]["compare"]["all_equal"],
                       "两份捕获在 [0,PRE) 上并非逐位相同", str(rec["H_identity"]["compare"]))
            del H2, snap2
        print(f"[lmh] H：源 {Path(CAP_H).name} 长度 {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）")

        # ---------------- 步骤 1：复现控制（S / B，带捕获）----------------
        gpu = dict(H_A=_to_dev(H, dev), H_B=_to_dev(H, dev))
        DR.require_distinct_storage(gpu["H_A"], gpu["H_B"], "S/B 两臂缓存")
        with LB.record(tgt, tag="S") as info_s:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_s, cache_s = spec_dec._step(tgt, FEED[0], gpu["H_A"])
            cap_s = LB.take()
        with LB.record(tgt, tag="B") as info_b:
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_b, blk_b = spec_dec._verify(tgt, gpu["H_B"], CANDS)
            cap_b = LB.take()
        len_s, len_b = spec_cache.cache_length(cache_s), spec_cache.cache_length(blk_b)
        hS = cap_s["caps"][1]["y_out"]
        hB4 = cap_b["caps"][1]["y_out"]
        row_s = LB.row_of(lg_s, None)           # S：末位行（= 该 (1,1,V) 的唯一一行）
        row_b = LB.row_of(rows_b, 0)            # B：rows[0]（= (1,4,V) 的第 0 行）
        rec["step1_control"] = dict(
            capture=dict(S=dict(n_calls=cap_s["n_calls"], installed=info_s["installed"],
                                h_layout=cap_s["caps"][1]["y_out_layout"],
                                x_in_layout=cap_s["caps"][1]["x_in_layout"]),
                         B=dict(n_calls=cap_b["n_calls"], installed=info_b["installed"],
                                h_layout=cap_b["caps"][1]["y_out_layout"],
                                x_in_layout=cap_b["caps"][1]["x_in_layout"])),
            positions=dict(S_cache_len_after=len_s, B_cache_len_after=len_b,
                           S_row_index_used="末位行（(1,1,V) 的唯一行）", B_row_index_used="rows[0]",
                           S_predicted_pos=len_s, B_predicted_pos=PRE + 1 + ROW_INDEX,
                           both_predict_target=bool(len_s == TARGET_POS == PRE + 1 + ROW_INDEX),
                           slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1]),
            S=_fields(row_s), B=_fields(row_b),
            margins=dict(S=DF.row_fields(row_s, watch=[]), B=DF.row_fields(row_b, watch=[])["max_score"]),
            live_cache_lengths=dict(S=spec_cache.cache_length(gpu["H_A"]),
                                    B=spec_cache.cache_length(gpu["H_B"])))
        s_a, b_a = _fields(row_s), _fields(row_b)
        print(f"[lmh] 控制：S argmax={s_a['argmax_id']}（margin={s_a['margin_top1_top2']}，"
              f"并列数={s_a['n_max_tied']}） B argmax={b_a['argmax_id']}（margin={b_a['margin_top1_top2']}）")
        DR.require(len_s == TARGET_POS and len_b == PRE + len(CANDS),
                   "S/B 的缓存长度不符合预期", f"S={len_s} B={len_b}")
        DR.require(rec["step1_control"]["positions"]["both_predict_target"],
                   "S 的末位行与 B 的 rows[0] 没有指向同一个预测位置")
        DR.require(rec["step1_control"]["positions"]["slot_for_B_row0"]["source"] == f"rows[{ROW_INDEX}]",
                   "槽位表与本工具的 ROW_INDEX 不一致",
                   str(rec["step1_control"]["positions"]["slot_for_B_row0"]))

        # 与**已保存的局部对照行**逐元素比较（不是与"未捕获的原始自由生成整行"比较）
        saved, saved_rows = None, {}
        if Path(SAVED_PT).exists():
            saved = torch.load(SAVED_PT, map_location="cpu")
            saved_rows = {k: v for k, v in saved.items() if torch.is_tensor(v)}
        rec["saved_cells_comparison"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S_row_vs_saved_H_AR_S=(DF.compare_rows(row_s.detach().float().cpu(),
                                                   saved_rows["H_AR-S"].float())
                                   if "H_AR-S" in saved_rows else None),
            B_row_vs_saved_H_AR_B=(DF.compare_rows(row_b.detach().float().cpu(),
                                                   saved_rows["H_AR-B"].float())
                                   if "H_AR-B" in saved_rows else None),
            note="**这是与「同一份 H、同一配置、上一次已保存的局部对照格」比**"
                 "（`div8209_cells_20261002T064415Z`），**不是**与「未捕获的原始自由生成整行」比 —— "
                 "后者本 run 没有捕获，也不得声称。")
        if saved_rows:
            for k, nm in (("H_AR-S", "S_row_vs_saved_H_AR_S"), ("H_AR-B", "B_row_vs_saved_H_AR_B")):
                c = rec["saved_cells_comparison"][nm]
                DR.require(c["elementwise_equal"], f"控制失败：本次 {k} 行与已保存局部对照行不同", str(c))
            print("[lmh] 控制 ✔ 本次 S/B 行与已保存的局部对照行**逐元素相同**")

        # 关闭捕获 ⇒ 结果必须逐位相同（"关闭诊断时原行为不变"的**真实模型**侧证据）
        if CAPTURE_OFF_CHECK:
            off = LB.install(tgt, enabled=False, tag="off")
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_s2, _c2 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_b2, _c3 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            row_s2, row_b2 = LB.row_of(lg_s2, None), LB.row_of(rows_b2, 0)
            rec["capture_off_check"] = dict(
                installed=off["installed"],
                S_row_identical=bool(torch.equal(row_s, row_s2)),
                B_row_identical=bool(torch.equal(row_b, row_b2)),
                note="不挂任何钩子时，S/B 两条真实路径的判定行与「带捕获」时**逐位相同**")
            LB.assert_clean()
            DR.require(rec["capture_off_check"]["S_row_identical"]
                       and rec["capture_off_check"]["B_row_identical"],
                       "关闭捕获后判定行发生变化（说明捕获不是只读的）",
                       str(rec["capture_off_check"]))
            print("[lmh] 捕获关闭等价 ✔（S/B 行逐位相同）")

        # 预期 token（来自真实**运行记录**的 argmax 锚点，如实标注为 argmax 级）
        anch_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anch_b = C._anchor_specdec(REF_RUN_RB, PRE, DEC_SLOT, TARGET_POS)
        rec["anchors"] = dict(S=anch_s, B=anch_b,
                              note="锚点 = 真实运行记录里该位置的 token 选择（**argmax 级**），"
                                   "不是整行逐元素参考")
        DR.require(anch_s.get("available") and anch_b.get("available"), "锚点不可用", str(rec["anchors"]))
        rec["step1_control"]["token_matches_anchors"] = dict(
            S=dict(measured=s_a["argmax_id"], anchor=anch_s["token"],
                   match=bool(s_a["argmax_id"] == anch_s["token"])),
            B=dict(measured=b_a["argmax_id"], anchor=anch_b["token"],
                   match=bool(b_a["argmax_id"] == anch_b["token"])))
        DR.require(all(v["match"] for v in rec["step1_control"]["token_matches_anchors"].values()),
                   "控制失败：S/B 的 argmax 与已保存结果不一致（停止定位）",
                   str(rec["step1_control"]["token_matches_anchors"]))

        # ---------------- 步骤 2：LM head 输入比较（hS vs hB）----------------
        hS_ref, hB4_ref = hS.clone(), hB4.clone()
        h_b = hB4[:, 0, :]                       # hB = B 的第 0 行隐藏向量 (1,H)
        rec["step2_input_compare"] = dict(
            hS=dict(layout=LB._layout(hS)), hB=dict(layout=LB._layout(h_b)),
            equal=bool(torch.equal(hS.reshape(-1), h_b.reshape(-1))),
            n_diff=int((hS.reshape(-1) != h_b.reshape(-1)).sum()),
            abs_diff_max=float((hS.float() - h_b.float()).abs().max()),
            abs_diff_mean=float((hS.float() - h_b.float()).abs().mean()),
            compare=DF.compare_rows(hS.reshape(-1), h_b.reshape(-1)),
            reading="`hS != hB` 只说明「差异**已经进入** LM head」；"
                    "**不能**据此认定最终 RMSNorm 或某个上游算子有 bug（那需要向上游再移一个边界）。")
        sc = rec["step2_input_compare"]
        print(f"[lmh] 输入比较：hS vs hB 逐位相同={sc['equal']} 不同元素={sc['n_diff']} "
              f"max|Δ|={sc['abs_diff_max']:.6g} mean|Δ|={sc['abs_diff_mean']:.6g}")

        # ---------------- 步骤 3：输入来源 × 投影形状 四格 ----------------
        W_info_before = LB.weights_info(tgt)
        hB1 = hB4[:, 0:1, :].contiguous()          # (1,1,H) contiguous ← B 第 0 行
        hB1_view = hB4[:, 0:1, :]                  # 同一切片的 **view**（布局控制）
        hS4 = hB4.clone()                          # 只改第 0 行；其余三行必须逐位不变
        hS4[0, 0] = hS[0, 0]
        other_rows_equal = bool(torch.equal(hS4[0, 1:], hB4_ref[0, 1:]))
        DR.require(other_rows_equal, "四行投影的其余三行被改动（必须保持同一份真实 B 隐藏向量）")
        DR.require(bool(hS4.is_contiguous()) and bool(hB1.is_contiguous()),
                   "格输入必须是 contiguous（与引擎前向输出一致）")
        rec["step3_setup"] = dict(
            hS_layout=LB._layout(hS), hB4_layout=LB._layout(hB4_ref),
            hB1_layout=LB._layout(hB1), hB1_view_layout=LB._layout(hB1_view),
            hS4_layout=LB._layout(hS4), other_rows_equal=other_rows_equal,
            replacement_note="`hS4` = 真实 B 隐藏张量的克隆，**只**把第 0 行替换为 `hS`；"
                             "其余三行始终是同一份真实 B 隐藏向量（已逐位核对）",
            expression=LB.PROJECTION_EXPRESSION, dtype=str(W.dtype), n_vocab=int(W.shape[0]))

        def run_cell(name, h, take_row):
            outs = []
            for _it in range(REPEAT):
                _full, row = LB.project(h, W, take_row=take_row)
                outs.append(row.detach().clone())
            same = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
            return dict(name=name, raw=outs[0], rows=[o.detach().float().cpu() for o in outs],
                        repeat_elementwise_equal=same)

        got = dict(
            HS_1=run_cell("HS_1", hS, 0),                    # 输入 hS × 单行投影 = 真实 S
            HB_1=run_cell("HB_1", hB1, 0),                   # 输入 hB × 单行投影
            HS_4=run_cell("HS_4", hS4, 0),                   # 输入 hS × 四行投影
            HB_4=run_cell("HB_4", hB4_ref, 0),               # 输入 hB × 四行投影 = 真实 B
            HB_1_view=run_cell("HB_1_view", hB1_view, 0),    # 布局控制（view 形式）
        )
        rows = {k: v["raw"] for k, v in got.items()}
        _h_for_cell = {"HS_1": hS, "HB_1": hB1, "HS_4": hS4, "HB_4": hB4_ref, "HB_1_view": hB1_view}
        cells = {k: dict(cell=v["name"], take_row=0,
                         input_layout=LB.layout(_h_for_cell[k]),
                         row=_fields(v["raw"]), repeat_elementwise_equal=v["repeat_elementwise_equal"],
                         n_repeat=REPEAT)
                 for k, v in got.items()}
        rec["step3_cells"] = cells
        rec["step3_compare"] = dict(
            source_effect_1row_HS_1_vs_HB_1=DF.compare_rows(rows["HS_1"], rows["HB_1"]),
            source_effect_4row_HS_4_vs_HB_4=DF.compare_rows(rows["HS_4"], rows["HB_4"]),
            shape_effect_with_hS_HS_1_vs_HS_4=DF.compare_rows(rows["HS_1"], rows["HS_4"]),
            shape_effect_with_hB_HB_1_vs_HB_4=DF.compare_rows(rows["HB_1"], rows["HB_4"]),
            layout_control_HB_1_view_vs_contig=DF.compare_rows(rows["HB_1_view"], rows["HB_1"]),
            naming="全部按**比较对**命名；`elementwise_equal` 是逐元素相等，"
                   "`argmax_same` 不能代替它")
        rec["step3_controls"] = dict(
            HS_1_vs_captured_real_S=DF.compare_rows(rows["HS_1"], row_s),
            HB_4_vs_captured_real_B=DF.compare_rows(rows["HB_4"], row_b),
            note="真实 S 对应「hS × 单行」，真实 B 对应「hB × 四行」；这两格必须**逐元素**复现捕获 logits，"
                 "否则只能说「没能只隔离输入/形状」，先报控制失败")
        for k, v in rec["step3_controls"].items():
            if isinstance(v, dict):
                DR.require(v["elementwise_equal"], f"控制失败：{k} 未逐元素复现捕获行", str(v))
        print("[lmh] 四格控制 ✔ HS_1 == 真实 S 行、HB_4 == 真实 B 行（均逐元素）")
        for k in ("HS_1", "HB_1", "HS_4", "HB_4"):
            f = cells[k]["row"]
            print(f"[lmh]   {k:6s} argmax={f['argmax_id']:6d} margin={f['margin_top1_top2']:.6g} "
                  f"并列数={f['n_max_tied']} watch={f['watch_scores']}")
        for k in PAIR_KEYS:
            v = rec["step3_compare"][k]
            print(f"[lmh]   {k:38s} same_argmax={v['argmax_same']} elementwise={v['elementwise_equal']} "
                  f"max|Δ|={v['abs_err_max']:.6g}")

        # ---------------- 源未被修改 + 输入未变 ----------------
        rec["source_integrity"] = dict(
            hS_unchanged=bool(torch.equal(hS, hS_ref)),
            hB4_unchanged=bool(torch.equal(hB4, hB4_ref)),
            other_rows_still_equal=bool(torch.equal(hS4[0, 1:], hB4_ref[0, 1:])),
            W_after=LB.weights_info(tgt),
            W_identity_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"]
                                   == W_info_before["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"]
                                 == W_info_before["digest_subsample"]),
            note="四格一律在**独立副本**上做（`hS4`/`hB1` 都是克隆或切片），捕获源不允许被修改")
        DR.require(rec["source_integrity"]["hS_unchanged"]
                   and rec["source_integrity"]["hB4_unchanged"]
                   and rec["source_integrity"]["other_rows_still_equal"]
                   and rec["source_integrity"]["W_identity_stable"]
                   and rec["source_integrity"]["W_digest_stable"],
                   "源张量或权重在四格过程中被改动", str(rec["source_integrity"]))

        # ---------------- 归因（只报事实 + 规则映射）----------------
        cp = rec["step3_compare"]

        def flip(c):
            return bool(c["argmax_same"] is False)

        rec["verdict"] = dict(
            input_effect_at_single_row=dict(flip=flip(cp["source_effect_1row_HS_1_vs_HB_1"]),
                                            detail=cp["source_effect_1row_HS_1_vs_HB_1"]),
            input_effect_at_four_rows=dict(flip=flip(cp["source_effect_4row_HS_4_vs_HB_4"]),
                                           detail=cp["source_effect_4row_HS_4_vs_HB_4"]),
            shape_effect_with_hS=dict(flip=flip(cp["shape_effect_with_hS_HS_1_vs_HS_4"]),
                                      detail=cp["shape_effect_with_hS_HS_1_vs_HS_4"]),
            shape_effect_with_hB=dict(flip=flip(cp["shape_effect_with_hB_HB_1_vs_HB_4"]),
                                      detail=cp["shape_effect_with_hB_HB_1_vs_HB_4"]),
            layout_control_identical=bool(cp["layout_control_HB_1_view_vs_contig"]["elementwise_equal"]),
            margins=dict(real_S=cells["HS_1"]["row"]["margin_top1_top2"],
                         real_B=cells["HB_4"]["row"]["margin_top1_top2"],
                         HB_1=cells["HB_1"]["row"]["margin_top1_top2"],
                         HS_4=cells["HS_4"]["row"]["margin_top1_top2"]),
            rule="固定投影形状、更换 hS/hB 后翻转 ⇒ 支持**输入差异**影响决策；"
                 "固定隐藏向量、更换单行/四行后翻转 ⇒ 支持**投影形状**影响决策；"
                 "两者互相依赖 ⇒ 如实记**交互**。margin/FP32 ULP **不得**当作正确性豁免；"
                 "本实验**不**判定 323/659 谁是数学真值；局部对照清楚 ≠ B6 通过。")
        print(f"[lmh] 归因：输入效应(单行)={rec['verdict']['input_effect_at_single_row']['flip']} "
              f"输入效应(四行)={rec['verdict']['input_effect_at_four_rows']['flip']} "
              f"形状效应(hS)={rec['verdict']['shape_effect_with_hS']['flip']} "
              f"形状效应(hB)={rec['verdict']['shape_effect_with_hB']['flip']} "
              f"布局控制相同={rec['verdict']['layout_control_identical']}")

        att = M.attention_stats()
        DR.require(not any(v > 0 for v in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      note="四格投影本身不经过注意力；上面的计数只来自控制用的 S/B 前向")
        rec["capture_clean"] = LB.assert_clean()
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        tp = OUTDIR / f"{TAG}_{run_id}.pt"
        torch.save(dict(rows={k: v.detach().float().cpu() for k, v in rows.items()},
                        rows_raw_bf16={k: v.detach().cpu() for k, v in rows.items()},
                        real_S_row=row_s.detach().float().cpu(),
                        real_B_row=row_b.detach().float().cpu(),
                        hS=hS.detach().float().cpu(), hB=h_b.detach().float().cpu(),
                        hB4=hB4.detach().float().cpu()), str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        _write(rec, run_id)
        print(f"[lmh] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DR.AlignError, A.AlignError,
                                                          CC.CloneError, LB.BoundaryError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[lmh] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[lmh] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
        except Exception:                                           # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：用微型模型把本工具的**全部机制**走一遍（含四格控制）。"""
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    global PRE, TARGET_POS, DEC_SLOT, ROW_INDEX, FEED, CANDS, WATCH, REPEAT
    torch.manual_seed(0)
    PRE, DEC_SLOT, ROW_INDEX, FEED, CANDS = 5, 1, 0, [3], [3, 4, 5, 6]
    TARGET_POS, WATCH, REPEAT = 6, (1, 2, 3), 2
    tgt = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    W = tgt.embed_tokens.weight
    ids = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()} for lay in
         CC.clone_list_cache_full(c0)]
    dev = torch.device("cpu")
    check("★ 起点 H 长度 = PRE，且保留全部层内键",
          spec_cache.cache_length(H) == PRE and sorted(H[0].keys()) == ["k", "v"])

    Hs = dict(H_A=_to_dev(H, dev), H_B=_to_dev(H, dev))
    DR.require_distinct_storage(Hs["H_A"], Hs["H_B"], "自测两臂")
    with LB.record(tgt, tag="S"):
        with torch.inference_mode(), M.attention_phase("decode"):
            lg_s, cs = spec_dec._step(tgt, FEED[0], Hs["H_A"])
        cap_s = LB.take()
    with LB.record(tgt, tag="B"):
        with torch.inference_mode(), M.attention_phase("verify"):
            rows_b, blk = spec_dec._verify(tgt, Hs["H_B"], CANDS)
        cap_b = LB.take()
    hS, hB4 = cap_s["caps"][1]["y_out"], cap_b["caps"][1]["y_out"]
    row_s, row_b = LB.row_of(lg_s, None), LB.row_of(rows_b, 0)
    check("★ 捕获形状：S = (1,1,H)，B = (1,4,H)（与真实路径一致）",
          list(hS.shape) == [1, 1, 32] and list(hB4.shape) == [1, 4, 32],
          (list(hS.shape), list(hB4.shape)))
    check("★ 位置口径：S 取末位行、B 取 rows[0]，两者都预测 TARGET_POS",
          spec_cache.cache_length(cs) == TARGET_POS
          and spec_cache.cache_length(blk) == PRE + len(CANDS)
          and DR.slot_table(PRE, len(CANDS))[1]["predicted_pos"] == TARGET_POS
          and DR.slot_table(PRE, len(CANDS))[1]["source"] == f"rows[{ROW_INDEX}]",
          DR.slot_table(PRE, len(CANDS))[1])
    check("★ 独立克隆：两臂长度各自正确（PRE+1 / PRE+4），源 H 未被污染",
          spec_cache.cache_length(cs) == PRE + 1 and spec_cache.cache_length(blk) == PRE + len(CANDS)
          and spec_cache.cache_length(H) == PRE
          and bool(torch.equal(H[0]["k"], c0[0]["k"].cpu())))

    hS_ref, hB4_ref = hS.clone(), hB4.clone()
    hB1 = hB4[:, 0:1, :].contiguous()
    hB1_view = hB4[:, 0:1, :]
    hS4 = hB4.clone()
    hS4[0, 0] = hS[0, 0]
    other_ok = bool(torch.equal(hS4[0, 1:], hB4_ref[0, 1:]))

    def cell(h, take_row):
        o = []
        for _ in range(REPEAT):
            _f, r = LB.project(h, W, take_row=take_row)
            o.append(r.detach().clone())
        return o

    HS_1, HB_1 = cell(hS, 0), cell(hB1, 0)
    HS_4, HB_4 = cell(hS4, 0), cell(hB4_ref, 0)
    HB_1v = cell(hB1_view, 0)
    check("★ 四格控制：`HS_1` 逐元素复现真实 S 行、`HB_4` 逐元素复现真实 B 行",
          bool(torch.equal(HS_1[0], row_s)) and bool(torch.equal(HB_4[0], row_b)),
          (DF.compare_rows(HS_1[0], row_s), DF.compare_rows(HB_4[0], row_b)))
    check("★ 格内重复逐位相同（每格 2 次）",
          all(all(bool(torch.equal(o[0], x)) for x in o[1:])
              for o in (HS_1, HB_1, HS_4, HB_4, HB_1v)))
    check("★ 只有第 0 行被替换，其余三行逐位不变", other_ok)
    check("★ 源未被修改（hS/hB4 与克隆一致）、权重身份/摘要不变",
          bool(torch.equal(hS, hS_ref)) and bool(torch.equal(hB4, hB4_ref))
          and LB.weights_info(tgt, digest=False)["data_ptr"] == int(W.data_ptr()))
    check("★ 四行投影的第 0 行 = 完整投影张量的第 0 行、且保留完整词表",
          list(LB.project(hS4, W, take_row=0)[0].shape) == [1, 4, 64])
    check("★ 布局控制（view 与 contiguous）都能跑完且有限",
          bool(torch.isfinite(HB_1v[0]).all()) and LB._layout(hB1_view)["is_view"] is True
          and LB._layout(hB1)["is_contiguous"] is True)
    off = LB.install(tgt, enabled=False, tag="off")
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_s2, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    LB.assert_clean()
    check("★ 关闭捕获时与带捕获逐位相同（真实模型侧的「关闭不变」自测）",
          off["installed"] is False and bool(torch.equal(row_s, LB.row_of(lg_s2, None))))
    print(f"\nLM head 边界定位（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：只验证本工具的机制（捕获/位置/克隆/四格/控制/重复/布局）")
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
