"""norm_boundary_f13_p8209.py — 位置 8209 的**最终 RMSNorm 边界**定位（F13；有界、单次）。

## 要回答的两个问题（**分开**）
同一份 `H(8208)`、同一输入位置 8208（token `20642`）：
1. **Norm 输入是否已经不同**：`xS`（S 路径，(1,1,1536)）vs `xB`（B 路径 (1,4,1536) 的第 0 行）？
2. **Norm 的执行形状是否改变判定**：固定输入、只把 Norm 的**行数**从 1 换成 4（或反之）会不会翻转？

`xS ≠ xB` **只**证明"差异在 Norm 之前就已存在"，**不足以**证明它就是翻转的驱动；
必须再做"固定形状换输入 / 固定输入换形状"两向对照（§五）。

## 边界（以实际代码为准）
`model_self.RMSNORM.forward`：`orig=x.dtype` → `x.float()` → `mean(x*x, dim=-1)` → `rsqrt(var+eps)`
→ `.to(orig_dtype)` → `weight * normed`。**逐行**运算、fp32 内部归约、输出回到 bf16。
本工具**调用模型现有的 `model.norm` 实例**（`lmhead_boundary.norm_forward`），**不重写公式**。

## 四格（沿用既有最终 Norm 边界工具 `norm_boundary_f06_i25.py` 的设计）
|  | 单行 Norm `(1,1,H)` | 四行 Norm `(1,4,H)` → 取第 0 行 |
| --- | --- | --- |
| 输入 `xS` | `NA` = Norm(xS)（**真实 S**） | `NC` = Norm(xB4 副本，**只**把第 0 行换成 xS) |
| 输入 `xB` | `NB` = Norm(xB 的**独立单行拷贝**) | `ND` = Norm(原 xB4)（**真实 B**） |
控制：`NA` 必须逐位复现捕获的 `hS`；`ND` 的第 0 行必须逐位复现捕获的 `hB`。

## 固定 LM head 形状的选词观察
把四格的**决策向量**统一整理成 `(1,1,H)`（原始 bf16），再用**同一份绑定权重**按生产表达式
`h @ W.T`（**完整词表**，不做"只取 323/659 两行"的裁剪）投影 ⇒ 这一步**固定 LM head 形状**，
专门观察 Norm 输入/形状的影响。

## 纪律
不改生产实现/公式/精度/门槛；不自由生成；**不扫 28 层**；不新增精度实验；GPU 单进程串行；
硬失败落盘停止、无自动重试；旧产物不覆盖；计时只作诊断成本。
`xS≠xB` **不**升级为"某个上游算子有 bug"；执行路径相关的数值差异**不**直接升级为"必然只是 BF16 舍入"。
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
from . import diag_checks as DC
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

TAG = os.environ.get("NB_TAG", "div8209_norm")
MODE = os.environ.get("NB_MODE", "gpu").strip().lower()
PRE = int(os.environ.get("NB_PRE", "8208"))
TARGET_POS = int(os.environ.get("NB_TARGET", "8209"))
K = int(os.environ.get("NB_K", "4"))
FEED = [int(x) for x in os.environ.get("NB_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("NB_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("NB_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("NB_REPEAT", "2"))
TARGET_ROPE = os.environ.get("NB_TARGET_ROPE", "cpu_fp32").strip().lower()
CAPTURE_OFF_CHECK = S.env_flag("NB_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("NB_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("NB_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
SAVED_PT = os.environ.get("NB_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
LMH_PT = os.environ.get("NB_LMH_PT", str(OUTDIR / "div8209_lmh_20261002T071235Z.pt"))
REF_RUN_AR = os.environ.get("NB_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("NB_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("NB_ARM", "eager")
DEC_SLOT = TARGET_POS - PRE                  # 1
ROW_INDEX = DEC_SLOT - 1                     # 0
# 显式的比较键（**不**对字典做"混合值"遍历：上一轮正是这样白付了一次模型加载）
PAIR_KEYS = ("fixed_single_row_change_input_NA_vs_NB",
             "fixed_four_row_change_input_NC_vs_ND",
             "fixed_input_xS_change_shape_NA_vs_NC",
             "fixed_input_xB_change_shape_NB_vs_ND")
PROJ_PAIR_KEYS = ("proj_fixed_shape_change_input_single_P_NA_vs_P_NB",
                  "proj_fixed_shape_change_input_four_P_NC_vs_P_ND",
                  "proj_fixed_input_change_shape_xS_P_NA_vs_P_NC",
                  "proj_fixed_input_change_shape_xB_P_NB_vs_P_ND")


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _to_dev(snap, dev):
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


def _set_precision() -> dict:
    b = torch.backends.cuda.matmul
    d0 = bool(getattr(b, "allow_bf16_reduced_precision_reduction"))
    b.allow_bf16_reduced_precision_reduction = True
    return dict(default_at_import=d0, set_to=True,
                effective_after_set=bool(b.allow_bf16_reduced_precision_reduction),
                allow_tf32=bool(b.allow_tf32),
                note="与 AR/Spec/重建 smoke 记录一致（default True → 显式 True，effective True）")


def _cmp(a, b, name: str, **kw) -> dict:
    """`diag_checks.cmp_tensor` + **不同元素个数** `n_diff` + **argmax 是否相同** `argmax_same`。

    为什么不直接改 `diag_checks`：那是既有工具被多处复用的公共原语，本轮只做诊断，
    用**薄包装**补字段，避免动到既有语义。

    ⚠️ **踩过的坑（本轮首跑）**：`diag_checks.cmp_tensor` **没有** `argmax_same` 字段
    （那是 `diag_fields.compare_rows` 的字段）。我按 `diag_fields` 的习惯写了
    `c.get("argmax_same") is False` 去判定"是否翻转" ⇒ 永远拿到 `None` ⇒ 归因布尔量**全为 False**
    （数据是对的，派生结论错了）。这里显式补齐两个口径的字段，避免再次混用。
    """
    out = DC.cmp_tensor(a, b, name, **kw)
    out["n_diff"] = int((a != b).sum())
    out["argmax_same"] = bool(int(a.reshape(-1).argmax()) == int(b.reshape(-1).argmax()))
    out["argmax_a"] = int(a.reshape(-1).argmax())
    out["argmax_b"] = int(b.reshape(-1).argmax())
    out["n_diff_note"] = "不同元素个数（逐元素 `!=` 计数）；`elementwise_equal=False` 时才有意义"
    out["argmax_note"] = ("`argmax_same` 只说明两者取到同一下标；逐元素是否相同看 `elementwise_equal`。"
                          "⚠️ 不是 `diag_fields` 的字段口径，本行由 `_cmp` 显式补上")
    return out


def _fields(row) -> dict:
    f = DF.row_fields(row, watch=list(WATCH))
    tk = f["topk"]
    f["margin_top1_top2"] = float(tk[0]["score"] - tk[1]["score"]) if len(tk) >= 2 else None
    f["dtype"] = str(row.dtype)
    f["sha16"] = A.stable_sha(row)
    f["raw_shape"] = [int(x) for x in row.shape]
    return f


def _stats(t) -> dict:
    """**统计**用的 fp32 视图（明确标注：只是统计转换，不参与任何前向/投影）。"""
    x = t.detach().float().reshape(-1)
    return dict(n=int(x.numel()), dtype_raw=str(t.dtype), dtype_stats="torch.float32",
                absmax=float(x.abs().max()), mean=float(x.mean()),
                finite=bool(torch.isfinite(x).all()),
                stats_note="`.float()` 仅用于统计；前向/Norm/投影一律用**原始 dtype** 张量")


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="norm_boundary_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209：(1) 最终 Norm 的输入是否已经不同？(2) Norm 的执行形状是否改变判定？",
               semantics="同一份 H(8208) 上捕获最终 RMSNorm 的输入/输出；四格 = 输入来源 × Norm 行数；"
                         "四格决策向量统一成 (1,1,1536) 后用**同一份绑定权重**做**完整词表**投影（固定 LM head 形状）。",
               config=dict(tag=TAG, pre=PRE, target_pos=TARGET_POS, dec_slot=DEC_SLOT,
                           row_index=ROW_INDEX, k=K, feed=FEED, cands=CANDS, watch=list(WATCH),
                           repeat=REPEAT, target_rope=TARGET_ROPE,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           saved_pt=str(SAVED_PT), lmh_pt=str(LMH_PT),
                           in_scope="只做 Norm 边界；不扫 28 层、不改算法、不新增精度实验"))
    print(f"[nb] run_id={run_id} pre={PRE} target={TARGET_POS} feed={FEED} cands={CANDS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[nb] commit: {why}")
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
        rec["norm"] = LB.norm_info(tgt)
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        print(f"[nb] 边界：{rec['code_facts']['boundary']}；Norm 类={rec['norm']['module_class']} "
              f"eps={rec['norm']['eps']} weight={rec['norm']['weight']['shape']} "
              f"{rec['norm']['weight']['dtype']}")
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))

        # ---------------- H（与上一轮同源、同截断）----------------
        cap = C._load_cap(Path(CAP_H))
        snap = cap["snapshot_cpu"]
        src_len = C._cache_len(snap)
        H, tinfo = C._truncate(snap, PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=src_len,
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H),
                               k_shape=[int(x) for x in H[0]["k"].shape], dtype=str(H[0]["k"].dtype),
                               note="与已保存的四格/LM-head 两次 run 同一份捕获与同一**纯切片**截断")
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2), compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"],
                       "两份捕获在 [0,PRE) 上并非逐位相同", str(rec["H_identity"]["compare"]))
            del H2
        print(f"[nb] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）")

        # ---------------- 步骤 1：控制（S/B + 捕获 x_in/y_out）----------------
        cS, cB = _to_dev(H, dev), _to_dev(H, dev)
        DR.require_distinct_storage(cS, cB, "S/B 两臂缓存")
        xS_snap_pre = None
        with LB.record(tgt, tag="S"):
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_s, cache_s = spec_dec._step(tgt, FEED[0], cS)
            cap_s = LB.take()
        with LB.record(tgt, tag="B"):
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_b, blk_b = spec_dec._verify(tgt, cB, CANDS)
            cap_b = LB.take()
        xS = cap_s["caps"][1]["x_in"]        # (1,1,1536) 最终 Norm 的输入
        xB4 = cap_b["caps"][1]["x_in"]       # (1,4,1536)
        hS = cap_s["caps"][1]["y_out"]       # (1,1,1536) 最终 Norm 的输出
        hB4 = cap_b["caps"][1]["y_out"]      # (1,4,1536)
        row_s, row_b = LB.row_of(lg_s, None), LB.row_of(rows_b, 0)
        len_s, len_b = spec_cache.cache_length(cache_s), spec_cache.cache_length(blk_b)
        rec["step1_control"] = dict(
            capture=dict(S=cap_s["caps"][1]["x_in_layout"], S_h=cap_s["caps"][1]["y_out_layout"],
                         B=cap_b["caps"][1]["x_in_layout"], B_h=cap_b["caps"][1]["y_out_layout"],
                         shape_expect=dict(xS=[1, 1, 1536], xB4=[1, 4, 1536]),
                         shapes_ok=bool(list(xS.shape) == [1, 1, 1536] and list(xB4.shape) == [1, 4, 1536]),
                         raw_dtype=str(xS.dtype)),
            positions=dict(S_cache_len_after=len_s, B_cache_len_after=len_b,
                           both_predict_target=bool(len_s == TARGET_POS == PRE + 1 + ROW_INDEX),
                           slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1]),
            S=_fields(row_s), B=_fields(row_b))
        DC.require(rec["step1_control"]["capture"]["shapes_ok"], "捕获形状不符",
                   str(rec["step1_control"]["capture"]))
        DC.require(str(xS.dtype) == str(W.dtype) and str(xB4.dtype) == str(W.dtype),
                   "捕获张量 dtype 与权重不一致（必须原生 bf16）",
                   f"xS={xS.dtype} xB4={xB4.dtype} W={W.dtype}")
        DC.require(rec["step1_control"]["positions"]["both_predict_target"],
                   "S 末位行与 B rows[0] 未指向同一预测位置")
        s_f, b_f = _fields(row_s), _fields(row_b)
        print(f"[nb] 控制：S argmax={s_f['argmax_id']}（margin={s_f['margin_top1_top2']}，并列={s_f['n_max_tied']}） "
              f"B argmax={b_f['argmax_id']}（margin={b_f['margin_top1_top2']}）")

        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        rec["step1_control"]["vs_saved_local_rows"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S=_cmp(row_s.float().cpu(), saved_rows["H_AR-S"].float(), "S_vs_saved_H_AR_S")
            if "H_AR-S" in saved_rows else None,
            B=_cmp(row_b.float().cpu(), saved_rows["H_AR-B"].float(), "B_vs_saved_H_AR_B")
            if "H_AR-B" in saved_rows else None,
            note="与**上一轮已保存的局部对照行**逐元素比较（不是与未捕获的自由生成整行比）")
        if saved_rows:
            for k in ("S", "B"):
                DC.require(rec["step1_control"]["vs_saved_local_rows"][k]["elementwise_equal"],
                           f"控制失败：{k} 行与已保存局部对照行不同",
                           str(rec["step1_control"]["vs_saved_local_rows"][k]))
            print("[nb] 控制 ✔ 本次 S/B 行与已保存局部对照行**逐元素相同**")

        if CAPTURE_OFF_CHECK:
            off = LB.install(tgt, enabled=False, tag="off")
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_s2, _c2 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_b2, _c3 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            rec["capture_off_check"] = dict(
                installed=off["installed"],
                S_row_identical=bool(torch.equal(row_s, LB.row_of(lg_s2, None))),
                B_row_identical=bool(torch.equal(row_b, LB.row_of(rows_b2, 0))),
                note="不挂任何钩子时 S/B 判定行与带捕获时逐位相同（只读性的真实模型侧证据）")
            LB.assert_clean()
            DC.require(all(v is True for k, v in rec["capture_off_check"].items() if k.endswith("identical")),
                       "关闭捕获后判定行发生变化", str(rec["capture_off_check"]))
            print("[nb] 捕获关闭等价 ✔")

        anch_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anch_b = C._anchor_specdec(REF_RUN_RB, PRE, DEC_SLOT, TARGET_POS)
        rec["anchors"] = dict(S=anch_s, B=anch_b,
                              note="真实运行记录里的 token 选择（**argmax 级**），不是整行参考")
        rec["step1_control"]["token_matches_anchors"] = dict(
            S=dict(measured=s_f["argmax_id"], anchor=anch_s.get("token"),
                   match=bool(s_f["argmax_id"] == anch_s.get("token"))),
            B=dict(measured=b_f["argmax_id"], anchor=anch_b.get("token"),
                   match=bool(b_f["argmax_id"] == anch_b.get("token"))))
        DC.require(all(v["match"] for v in rec["step1_control"]["token_matches_anchors"].values()),
                   "控制失败：S/B argmax 与锚点不一致（停止，不解释 Norm 四格）",
                   str(rec["step1_control"]["token_matches_anchors"]))

        # ---------------- 输入比较（xS vs xB；并把上一轮的 h 比较一并复现）----------------
        xS_snap, xB4_snap = DC.snapshot(xS), DC.snapshot(xB4)
        hS_snap, hB4_snap = DC.snapshot(hS), DC.snapshot(hB4)
        rec["input_compare"] = dict(
            xS_vs_xB=_cmp(xS[0, 0].float().cpu(), xB4[0, 0].float().cpu(), "xS_row0_vs_xB_row0"),
            xS_stats=_stats(xS), xB_stats=_stats(xB4), xB4_stats=_stats(xB4),
            raw_dtype=str(xS.dtype), stats_dtype="torch.float32（只用于统计/比较）",
            hS_vs_hB=_cmp(hS[0, 0].float().cpu(), hB4[0, 0].float().cpu(), "hS_row0_vs_hB_row0"),
            reading="`xS ≠ xB` 只证明差异在 Norm **之前**就已存在；"
                    "是否驱动翻转必须看后面的两向四格")
        ic = rec["input_compare"]
        print(f"[nb] 输入比较：xS vs xB equal={ic['xS_vs_xB']['elementwise_equal']} "
              f"max|Δ|={ic['xS_vs_xB']['abs_err_max']:.6g} "
              f"（上一轮 h 比较：equal={ic['hS_vs_hB']['elementwise_equal']} "
              f"max|Δ|={ic['hS_vs_hB']['abs_err_max']:.6g}）")

        # ---------------- 最终 Norm 四格 ----------------
        Hdim = int(xS.shape[-1])
        xB1 = xB4[:, 0, :].reshape(1, 1, Hdim).clone()      # **独立拷贝**的单行形式（不是 contiguous() 空控制）
        xB4c = xB4.clone()
        xB4c[0, 0] = xS[0, 0]
        other_rows_ok = bool(torch.equal(xB4c[0, 1:], xB4_snap[0, 1:]))
        DC.require(other_rows_ok, "四行格的其余三行被改动（必须保持同一份真实 B 输入）")
        DC.require(int(xB1.data_ptr()) != int(xB4.data_ptr()),
                   "单行形式必须是与 xB4 不共享存储的**独立拷贝**",
                   f"xB1.data_ptr={int(xB1.data_ptr())} xB4.data_ptr={int(xB4.data_ptr())}")
        rec["norm_cells_setup"] = dict(
            xS=DC.layout(xS), xB4=DC.layout(xB4), xB1_copy=DC.layout(xB1), xB4c=DC.layout(xB4c),
            xB1_is_independent_copy=bool(int(xB1.data_ptr()) != int(xB4.data_ptr())),
            other_rows_equal=other_rows_ok,
            row0_slice_layout_control=DC.layout_control(xB4[:, 0:1, :], "xB4_row0_slice"),
            layout_note="① 单行格用**独立拷贝** `xB1`（data_ptr 不同）；② `xB4[:,0:1,:]` 的 "
                        "`.contiguous()` **不会复制**（本就连续）⇒ 本工具**不**把它当布局对照，"
                        "`layout_control.control_valid` 如实为 False；本轮的比较是"
                        "「输入来源 × Norm 行数」，不含布局变量")

        cells: dict = {}
        for name, x, take_row in (("NA", xS, 0), ("NB", xB1, 0), ("NC", xB4c, 0), ("ND", xB4, 0)):
            outs, metas = [], []
            for _it in range(REPEAT):
                y, meta = LB.norm_forward(tgt, x)
                outs.append(y.detach().clone())
                metas.append(meta)
            same = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
            dv = outs[0][0, take_row].reshape(1, 1, Hdim)          # 决策向量 → (1,1,H)
            cells[name] = dict(input_key=name, input_shape=[int(v) for v in x.shape],
                               take_row=take_row, repeat_elementwise_equal=same, n_repeat=REPEAT,
                               input_layout=DC.layout(x), norm_meta=metas[0],
                               decision_vector_layout=DC.layout(dv),
                               decision_vector_stats=_stats(dv), out_raw=outs[0], dv_raw=dv)
        rec["norm_cells"] = {k: dict(input_shape=v["input_shape"], take_row=v["take_row"],
                                     repeat_elementwise_equal=v["repeat_elementwise_equal"],
                                     n_repeat=v["n_repeat"], input_layout=v["input_layout"],
                                     decision_vector_layout=v["decision_vector_layout"],
                                     decision_vector_stats=v["decision_vector_stats"])
                             for k, v in cells.items()}
        # 逐格复现性
        for k, v in cells.items():
            DC.require(v["repeat_elementwise_equal"], f"{k}: 格内重复不一致")
        # 控制：真实组合必须复现捕获输出
        rec["norm_cells_controls"] = dict(
            NA_reproduces_captured_hS=_cmp(cells["NA"]["out_raw"][0, 0].float().cpu(),
                                                    hS_snap[0, 0].float().cpu(), "NA_vs_captured_hS"),
            ND_row0_reproduces_captured_hB=_cmp(cells["ND"]["out_raw"][0, 0].float().cpu(),
                                                         hB4_snap[0, 0].float().cpu(),
                                                         "ND_row0_vs_captured_hB"),
            note="真实组合（xS×单行、xB×四行）**必须**逐位复现捕获输出；"
                 "但这也**只**证明重放忠实，**不**能单独证明 Norm 没有形状效应")
        for k, v in rec["norm_cells_controls"].items():
            if isinstance(v, dict):
                DC.require(v["elementwise_equal"], f"控制失败：{k} 未逐位复现捕获输出", str(v))
        print("[nb] Norm 四格控制 ✔ NA==捕获 hS、ND 第 0 行==捕获 hB（均逐位）")

        # 向量级两向对照
        vec_cmp = {
            "fixed_single_row_change_input_NA_vs_NB":
                _cmp(cells["NA"]["dv_raw"][0, 0].float().cpu(),
                              cells["NB"]["dv_raw"][0, 0].float().cpu(), "NA_vs_NB"),
            "fixed_four_row_change_input_NC_vs_ND":
                _cmp(cells["NC"]["dv_raw"][0, 0].float().cpu(),
                              cells["ND"]["dv_raw"][0, 0].float().cpu(), "NC_vs_ND"),
            "fixed_input_xS_change_shape_NA_vs_NC":
                _cmp(cells["NA"]["dv_raw"][0, 0].float().cpu(),
                              cells["NC"]["dv_raw"][0, 0].float().cpu(), "NA_vs_NC"),
            "fixed_input_xB_change_shape_NB_vs_ND":
                _cmp(cells["NB"]["dv_raw"][0, 0].float().cpu(),
                              cells["ND"]["dv_raw"][0, 0].float().cpu(), "NB_vs_ND"),
        }
        rec["norm_vector_compare"] = vec_cmp

        # ---------------- 固定 LM head 形状的选词观察 ----------------
        proj: dict = {}
        for k, v in cells.items():
            dv = v["dv_raw"]
            DC.require(str(dv.dtype) == str(W.dtype), f"{k}: 投影输入 dtype 与权重不一致",
                       f"{dv.dtype} vs {W.dtype}")
            DC.require(tuple(dv.shape) == (1, 1, Hdim), f"{k}: 决策向量形状必须是 (1,1,H)",
                       str(tuple(dv.shape)))
            _full, row = LB.project(dv, W, take_row=0)
            proj[k] = dict(raw=row.detach().clone(), float_row=row.detach().float().cpu())
        rec["projection"] = dict(
            shape_fixed="全部四格统一为 (1,1,1536) @ W.T（**完整词表**，不裁剪 323/659 两行）",
            expression=LB.PROJECTION_EXPRESSION, n_vocab=int(W.shape[0]),
            rows={k: _fields(v["raw"]) for k, v in proj.items()},
            input_dtype={k: str(v["dv_raw"].dtype) for k, v in cells.items()})
        for k in ("NA", "NB", "NC", "ND"):
            f = rec["projection"]["rows"][k]
            print(f"[nb]   P_{k} argmax={f['argmax_id']:6d} margin={f['margin_top1_top2']:.6g} "
                  f"并列={f['n_max_tied']} watch={f['watch_scores']}")
        proj_pairs = {
            "proj_fixed_shape_change_input_single_P_NA_vs_P_NB":
                _cmp(proj["NA"]["float_row"], proj["NB"]["float_row"], "P_NA_vs_P_NB"),
            "proj_fixed_shape_change_input_four_P_NC_vs_P_ND":
                _cmp(proj["NC"]["float_row"], proj["ND"]["float_row"], "P_NC_vs_P_ND"),
            "proj_fixed_input_change_shape_xS_P_NA_vs_P_NC":
                _cmp(proj["NA"]["float_row"], proj["NC"]["float_row"], "P_NA_vs_P_NC"),
            "proj_fixed_input_change_shape_xB_P_NB_vs_P_ND":
                _cmp(proj["NB"]["float_row"], proj["ND"]["float_row"], "P_NB_vs_P_ND"),
        }
        rec["projection_compare"] = proj_pairs
        for k in PROJ_PAIR_KEYS:
            v = proj_pairs[k]
            print(f"[nb]   {k:46s} same_argmax={v['argmax_same']} "
                  f"elementwise={v['elementwise_equal']} n_diff={v['n_diff']} max|Δ|={v['abs_err_max']:.6g} "
                  f"({v['argmax_a']} vs {v['argmax_b']})")

        # 与上一轮 LM head 边界 run 的对应行交叉核对（同 H、同 W、同表达式）
        # ⚠️ 首跑在这里**误设**了核对目标：把本轮的**单行**投影（固定 LM head 形状）与上一轮的
        #    **四行**投影 `HB_4` 比 ⇒ 差 0.0625（正是上一轮测到的 LM head 形状效应），触发 checks_failed。
        #    数值本身完好（在崩溃 JSON 里）；改为"单行 ↔ 单行"的**必需控制**，
        #    并把与 `*_4` 行的差异**如实记录**为"上一轮 LM head 形状效应的跨 run 复现"，不作控制。
        rec["cross_run_vs_lmhead"] = dict(path=str(LMH_PT), present=Path(LMH_PT).exists())
        if Path(LMH_PT).exists():
            pr = torch.load(LMH_PT, map_location="cpu").get("rows", {})
            rec["cross_run_vs_lmhead"].update(dict(
                P_NA_vs_prev_HS_1=(_cmp(proj["NA"]["float_row"], pr["HS_1"].float(),
                                        "P_NA_vs_prev_HS_1") if "HS_1" in pr else None),
                P_ND_vs_prev_HB_1=(_cmp(proj["ND"]["float_row"], pr["HB_1"].float(),
                                        "P_ND_vs_prev_HB_1") if "HB_1" in pr else None),
                P_ND_vs_prev_HB_4=(_cmp(proj["ND"]["float_row"], pr["HB_4"].float(),
                                        "P_ND_vs_prev_HB_4") if "HB_4" in pr else None),
                P_NA_vs_prev_HS_4=(_cmp(proj["NA"]["float_row"], pr["HS_4"].float(),
                                        "P_NA_vs_prev_HS_4") if "HS_4" in pr else None),
                required_controls=["P_NA_vs_prev_HS_1", "P_ND_vs_prev_HB_1"],
                reference_only=["P_ND_vs_prev_HB_4", "P_NA_vs_prev_HS_4"],
                note="必需控制是**单行 ↔ 单行**（同一 LM head 形状）；与上一轮 `*_4` 行的差异是"
                     "**预期内**的上一轮 LM head 形状效应（hB 情形实测 171 个坐标、max 0.0625、"
                     "index 14623），只作跨 run 复现证据，**不**当作控制"))
            for k in rec["cross_run_vs_lmhead"]["required_controls"]:
                DC.require(rec["cross_run_vs_lmhead"][k]["elementwise_equal"],
                           f"跨 run 交叉核对失败：{k}", str(rec["cross_run_vs_lmhead"][k]))
            print("[nb] 跨 run 交叉核对 ✔ P_NA == 上一轮 HS_1、P_ND == 上一轮 HB_1（同为单行投影）")
            _r = rec["cross_run_vs_lmhead"]["P_ND_vs_prev_HB_4"]
            print(f"[nb] 参考项：P_ND vs 上一轮 HB_4（四行）**预期不同**：n_diff={_r['n_diff']} "
                  f"max|Δ|={_r['abs_err_max']:.6g}（= 上一轮测到的 LM head 形状效应）")

        # ---------------- 源完整性 ----------------
        rec["source_integrity"] = dict(
            xS_unmodified=_cmp(xS, xS_snap, "xS_vs_snapshot", ref_snapshot=xS_snap, src=xS),
            xB4_unmodified=_cmp(xB4, xB4_snap, "xB4_vs_snapshot", ref_snapshot=xB4_snap,
                                         src=xB4),
            hS_unmodified=_cmp(hS, hS_snap, "hS_vs_snapshot", ref_snapshot=hS_snap, src=hS),
            hB4_unmodified=_cmp(hB4, hB4_snap, "hB4_vs_snapshot", ref_snapshot=hB4_snap,
                                         src=hB4),
            other_rows_still_equal=bool(torch.equal(xB4c[0, 1:], xB4_snap[0, 1:])),
            W_after=LB.weights_info(tgt),
            W_identity_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"]
                                   == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"]
                                 == rec["W_before"]["digest_subsample"]),
            norm_weight_after=LB.norm_info(tgt)["weight"],
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"]
                                    == rec["norm"]["weight"]["data_ptr"]),
            note="四格一律在独立副本上做；捕获源与两份权重在全程不得改变")
        DC.require(rec["source_integrity"]["other_rows_still_equal"]
                   and rec["source_integrity"]["W_identity_stable"]
                   and rec["source_integrity"]["W_digest_stable"]
                   and rec["source_integrity"]["norm_weight_stable"],
                   "源张量或权重在四格过程中被改动", str(rec["source_integrity"]))

        # ---------------- 归因（只报事实 + 规则映射）----------------
        def flip(c):
            """"翻转" = 两行**不逐元素相同** 且 **argmax 不同**（两个口径都必须显式存在）。"""
            DC.require("argmax_same" in c and "elementwise_equal" in c,
                       "比较字段缺失（`_cmp` 必须同时给 elementwise_equal 与 argmax_same）", str(c))
            return bool(c["elementwise_equal"] is False and c["argmax_same"] is False)

        pj = rec["projection_compare"]
        rec["verdict"] = dict(
            input_already_differs_before_norm=bool(not rec["input_compare"]["xS_vs_xB"]["elementwise_equal"]),
            # ⚠️ 键名必须**无歧义**：080941Z 那次用的是 `norm_shape_effect_on_vector`，
            #    但子键存的是 `elementwise_equal`（True = 两格**逐位相同** = **未观察到**形状效应）⇒
            #    容易读反。这里改名并把"True 的含义"写进字段本身。
            norm_shape_change_identical_on_decision_vector=dict(
                identical_fixed_input_xS=vec_cmp["fixed_input_xS_change_shape_NA_vs_NC"]["elementwise_equal"],
                identical_fixed_input_xB=vec_cmp["fixed_input_xB_change_shape_NB_vs_ND"]["elementwise_equal"],
                n_diff=dict(fixed_input_xS=vec_cmp["fixed_input_xS_change_shape_NA_vs_NC"]["n_diff"],
                            fixed_input_xB=vec_cmp["fixed_input_xB_change_shape_NB_vs_ND"]["n_diff"]),
                max_abs=dict(fixed_input_xS=vec_cmp["fixed_input_xS_change_shape_NA_vs_NC"]["abs_err_max"],
                             fixed_input_xB=vec_cmp["fixed_input_xB_change_shape_NB_vs_ND"]["abs_err_max"]),
                reading="`True` = 固定输入、只改 Norm 行数时，**决策向量逐位相同** ⇒ 该输入下**未观察到**形状效应；"
                        "`False` 才表示观察到了形状差异"),
            fixed_shape_change_input_flips=dict(
                single_row=flip(pj["proj_fixed_shape_change_input_single_P_NA_vs_P_NB"]),
                four_row=flip(pj["proj_fixed_shape_change_input_four_P_NC_vs_P_ND"])),
            fixed_input_change_shape_flips=dict(
                xS=flip(pj["proj_fixed_input_change_shape_xS_P_NA_vs_P_NC"]),
                xB=flip(pj["proj_fixed_input_change_shape_xB_P_NB_vs_P_ND"])),
            rule="① `xS≠xB` 只证明差异在 Norm **之前**已存在，**不足以**证明它驱动翻转；"
                 "② 固定 Norm 形状换输入会翻转、固定输入换形状不翻转 ⇒ 支持**输入差异驱动**，下一步才转向 Decoder 堆栈；"
                 "③ 固定输入换 Norm 形状就翻转 ⇒ 支持 **Norm 形状效应**；④ 两者互相依赖 ⇒ 如实记**交互**；"
                 "⑤ 真实组合复现只证明重放忠实，**不**能单独证明 Norm 没有形状效应；"
                 "⑥ **不**把「执行路径相关的数值差异」升级为「必然只是 BF16 舍入」；"
                 "⑦ margin 不作为豁免，**不**判定 323/659 谁是数学真值。")
        v = rec["verdict"]
        print(f"[nb] 归因：Norm 前已不同={v['input_already_differs_before_norm']} "
              f"Norm 形状改变决策向量（True=逐位相同/未观察到）="
              f"{v['norm_shape_change_identical_on_decision_vector']} "
              f"换输入翻转={v['fixed_shape_change_input_flips']} "
              f"换形状翻转={v['fixed_input_change_shape_flips']}")

        att = M.attention_stats()
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        rec["capture_clean"] = LB.assert_clean()
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        tp = OUTDIR / f"{TAG}_{run_id}.pt"
        torch.save(dict(
            # ★ 原始 dtype 落盘（bf16）；`*_f32` 只是给统计/外部脚本用的 fp32 副本
            xS_raw=xS.detach().cpu(), xB4_raw=xB4.detach().cpu(),
            hS_raw=hS.detach().cpu(), hB4_raw=hB4.detach().cpu(),
            real_S_row_raw=row_s.detach().cpu(), real_B_row_raw=row_b.detach().cpu(),
            norm_out_raw={k: v["out_raw"].detach().cpu() for k, v in cells.items()},
            norm_dv_raw={k: v["dv_raw"].detach().cpu() for k, v in cells.items()},
            proj_rows_raw={k: v["raw"].detach().cpu() for k, v in proj.items()},
            proj_rows_f32={k: v["float_row"] for k, v in proj.items()},
            real_S_row_f32=row_s.float().cpu(), real_B_row_f32=row_b.float().cpu()), str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp),
                                raw_dtype_note="`*_raw` = 原始 bf16；`*_f32` = 仅供统计/外部比较")
        _write(rec, run_id)
        print(f"[nb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[nb] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[nb] RESULT_JSON={p}")
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
    """**不加载大模型**：微型模型上把本工具的机制全走一遍（含 Norm 四格与固定形状投影）。"""
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
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    check("★ 起点 H 长度 = PRE、层内键完整",
          spec_cache.cache_length(H) == PRE and sorted(H[0].keys()) == ["k", "v"])
    dev = torch.device("cpu")
    cS, cB = _to_dev(H, dev), _to_dev(H, dev)
    DR.require_distinct_storage(cS, cB, "自测两臂")
    with LB.record(tgt, tag="S"):
        with torch.inference_mode(), M.attention_phase("decode"):
            lg_s, cs = spec_dec._step(tgt, FEED[0], cS)
        cap_s = LB.take()
    with LB.record(tgt, tag="B"):
        with torch.inference_mode(), M.attention_phase("verify"):
            rows_b, blk = spec_dec._verify(tgt, cB, CANDS)
        cap_b = LB.take()
    xS, xB4 = cap_s["caps"][1]["x_in"], cap_b["caps"][1]["x_in"]
    hS, hB4 = cap_s["caps"][1]["y_out"], cap_b["caps"][1]["y_out"]
    row_s, row_b = LB.row_of(lg_s, None), LB.row_of(rows_b, 0)
    check("★ 捕获位置/形状：最终 Norm 的输入 x = (1,1,32)/(1,4,32)、输出 h 同形、原生 dtype",
          list(xS.shape) == [1, 1, 32] and list(xB4.shape) == [1, 4, 32]
          and list(hS.shape) == [1, 1, 32] and xS.dtype == torch.bfloat16, (list(xS.shape), xS.dtype))
    check("★ 行索引口径：S 取末位行、B 取 rows[0]，位置一致；`LB.row_of` 对 (1,1,V) 与 (4,V) 都给一维行",
          spec_cache.cache_length(cs) == TARGET_POS
          and DR.slot_table(PRE, len(CANDS))[1]["predicted_pos"] == TARGET_POS
          and row_s.dim() == 1 and row_b.dim() == 1 and int(row_s.numel()) == 64)
    Hdim = int(xS.shape[-1])
    xS_snap, xB4_snap = DC.snapshot(xS), DC.snapshot(xB4)
    xB1 = xB4[:, 0, :].reshape(1, 1, Hdim).clone()
    xB4c = xB4.clone()
    xB4c[0, 0] = xS[0, 0]
    check("★ 独立副本：单行形式与 xB4 不共享存储；四行格只换第 0 行、其余三行逐位不变",
          int(xB1.data_ptr()) != int(xB4.data_ptr())
          and bool(torch.equal(xB4c[0, 1:], xB4_snap[0, 1:])), )
    check("★ 布局如实记录：`xB4[:,0:1,:]` 的 `.contiguous()` 不复制 ⇒ `control_valid=False`（不当布局对照）",
          DC.layout_control(xB4[:, 0:1, :], "t")["control_valid"] is False
          or DC.layout_control(xB4[:, 0:1, :], "t")["contiguous_returned_same_storage"] is False)

    cells, dvs = {}, {}
    for name, x, take_row in (("NA", xS, 0), ("NB", xB1, 0), ("NC", xB4c, 0), ("ND", xB4, 0)):
        outs = [LB.norm_forward(tgt, x) for _ in range(REPEAT)]
        same = all(bool(torch.equal(outs[0][0], o[0])) for o in outs[1:])
        dvs[name] = outs[0][0][0, take_row].reshape(1, 1, Hdim)
        cells[name] = dict(same=same, out=outs[0][0])
    check("★ 格内重复逐位相同（四格各 2 次）", all(v["same"] for v in cells.values()))
    check("★ 控制：NA 逐位复现捕获 hS、ND 第 0 行逐位复现捕获 hB",
          bool(torch.equal(cells["NA"]["out"][0, 0], hS[0, 0]))
          and bool(torch.equal(cells["ND"]["out"][0, 0], hB4[0, 0])))
    check("★ 决策向量统一为 (1,1,H) 且保持原生 dtype",
          all(tuple(v.shape) == (1, 1, Hdim) and v.dtype == W.dtype for v in dvs.values()))
    proj = {}
    for k, dv in dvs.items():
        _f, r = LB.project(dv, W, take_row=0)
        proj[k] = r
    check("★ 固定 LM head 形状：四格决策向量都以 (1,1,H)@W.T 投影、完整词表、有限",
          all(tuple(v.shape) == (64,) and bool(torch.isfinite(v).all()) for v in proj.values()))
    check("★ 与真实行一致：P_NA 等于真实 S 行的投影重算（同 hS、同表达式 ⇒ 逐位）",
          bool(torch.equal(proj["NA"], LB.project(hS[:, -1:, :], W, take_row=0)[1]))
          and bool(torch.equal(proj["ND"], LB.project(hB4[:, 0:1, :], W, take_row=0)[1])))
    check("★ 源与权重未被改动（xS/xB4 与快照一致；Norm 权重 data_ptr 不变）",
          bool(torch.equal(xS, xS_snap)) and bool(torch.equal(xB4, xB4_snap))
          and LB.norm_info(tgt)["weight"]["data_ptr"] == int(tgt.norm.weight.data_ptr()))
    off = LB.install(tgt, enabled=False, tag="off")
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_s2, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    LB.assert_clean()
    check("★ 关闭捕获时与带捕获逐位相同（真实模型侧的关闭等价）",
          off["installed"] is False and bool(torch.equal(row_s, LB.row_of(lg_s2, None))))
    _a = torch.zeros(8)
    _a[7] = 10.0                      # 明确且两侧共有的最大值 ⇒ argmax 应保持相同
    _b = _a.clone()
    _b[[1, 3, 5]] = 1.0
    _c = _cmp(_a, _b, "t_n_diff")
    check("★ 比较字段补的 `n_diff`/`argmax_same` 均正确（3 处不同；argmax 仍相同）",
          _c["n_diff"] == 3 and _c["elementwise_equal"] is False and _c["argmax_same"] is True, _c)
    _c2 = _cmp(torch.tensor([1.0, 5.0, 3.0]), torch.tensor([9.0, 4.0, 3.0]), "t_arg")
    check("★ 首跑缺陷的回归：`argmax_same` **必须存在**（缺字段时归因布尔量会静默变成 False）",
          _c2["argmax_same"] is False and _c2["argmax_a"] == 1 and _c2["argmax_b"] == 0
          and _c2["n_diff"] == 2, _c2)
    check("★ `n_diff` 在相等时为 0（不把 0 当成「没比较」）",
          _cmp(_a, _a.clone(), "t_eq")["n_diff"] == 0)
    print(f"\nNorm 边界定位（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：只验证机制（捕获/行索引/独立副本/四格/控制/固定形状投影/关闭等价）")
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
