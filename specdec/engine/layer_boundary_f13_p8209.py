"""layer_boundary_f13_p8209.py — 位置 8209 的**最后一层 Decoder（L27，零基）**边界定位（F13；有界、单次）。

## 要回答的两个问题（**分开**）
同一份 `H(8208)`、同一输入位置 8208（token `20642`）：单步 `[20642]` 选 **323**；块 `[20642,659,576,4531]`
的第 0 行选 **659**。本轮把边界上移到**最后一层**（`layers[27]`，本 run 会核对它确实是最后一层）：

1. **L27 的输入是否已经不同**：`aS`（S 路径，(1,1,1536)）vs `aB`（B 路径 (1,4,1536) 的第 0 行）？
2. **L27 的执行形状是否改变判定**：固定输入、只把层调用的**行数**从 1 换成 4（或反之）会不会翻转？

`aS ≠ aB` **只**证明"上游已有差异"，**不**证明上游独占最终原因；判定必须靠
"固定形状换输入 / 固定输入换形状"两向四格（§三）。

## 边界与纪律（关键点）
* 层输入 `a` = 上一层的输出；**最后一层的输出 `z` 就是随后 `self.norm` 的输入**
  ⇒ 本 run 用**同一次前向**里的 `layers[27]` pre/post 钩子与 `norm` 钩子**逐位**核对这条链。
* **轮前缓存只能从 pre 钩子取**：层的 `forward` 会**就地追加** `cache["k"]/["v"]`
  ⇒ post 里的缓存已被追加，**不得**当轮前状态。本工具用 pre 钩子里的**独立深拷贝**。
* 层收到/返回的是**单层 cache dict**（`{"k","v"}`），不是整模型 `list[dict]`（口径已由 CPU 门钉死）。
* 每格用**独立克隆**的轮前缓存；`cos/sin`、phase、mask、缓存长度全部如实记录。
* 不改生产实现/公式/精度/门槛；**不**自由生成；**不**加载 draft；**不**扫 28 层；**不**做 FP32；GPU 串行、硬失败落盘停止。

## 统一尾部
每格的 L27 输出行统一成 `(1,1,1536)` → **同一个最终 Norm** → **同一份绑定权重**按生产表达式
`h @ W.T` 做**完整词表**投影（不裁剪 323/659 两行、不改尾部形状）。
**单行尾部结果只与"单行投影"参考比较**，**不**把四行 LM head 参考当逐位必需控制。
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

TAG = os.environ.get("LYR_TAG", "div8209_l27")
MODE = os.environ.get("LYR_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("LYR_LAYER", "27"))
PRE = int(os.environ.get("LYR_PRE", "8208"))
TARGET_POS = int(os.environ.get("LYR_TARGET", "8209"))
K = int(os.environ.get("LYR_K", "4"))
FEED = [int(x) for x in os.environ.get("LYR_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("LYR_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("LYR_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("LYR_REPEAT", "2"))
TARGET_ROPE = os.environ.get("LYR_TARGET_ROPE", "cpu_fp32").strip().lower()
CAPTURE_OFF_CHECK = S.env_flag("LYR_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("LYR_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("LYR_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
SAVED_PT = os.environ.get("LYR_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
NORM_PT = os.environ.get("LYR_NORM_PT", str(OUTDIR / "div8209_norm_20261002T080941Z.pt"))
LMH_PT = os.environ.get("LYR_LMH_PT", str(OUTDIR / "div8209_lmh_20261002T071235Z.pt"))
REF_RUN_AR = os.environ.get("LYR_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("LYR_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("LYR_ARM", "eager")
DEC_SLOT = TARGET_POS - PRE                  # 1
ROW_INDEX = DEC_SLOT - 1                     # 0
PAIR_KEYS = ("fixed_single_row_change_input_LA_vs_LB",
             "fixed_four_row_change_input_LC_vs_LD",
             "fixed_input_aS_change_shape_LA_vs_LC",
             "fixed_input_aB_change_shape_LB_vs_LD")
PROJ_PAIR_KEYS = ("proj_fixed_shape_change_input_single_P_LA_vs_P_LB",
                  "proj_fixed_shape_change_input_four_P_LC_vs_P_LD",
                  "proj_fixed_input_change_shape_aS_P_LA_vs_P_LC",
                  "proj_fixed_input_change_shape_aB_P_LB_vs_P_LD")


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
    """`diag_checks.cmp_tensor` + `n_diff`/`argmax_same`（两套字段口径都必须显式存在）。

    ⚠️ 上一轮的教训：`diag_checks.cmp_tensor` **没有** `argmax_same`（那是 `diag_fields.compare_rows`
    的字段）⇒ 按后者习惯写 `c.get("argmax_same") is False` 会恒为 `None`、归因布尔量静默变 False。
    """
    out = DC.cmp_tensor(a, b, name, **kw)
    out["n_diff"] = int((a != b).sum())
    out["argmax_same"] = bool(int(a.reshape(-1).argmax()) == int(b.reshape(-1).argmax()))
    out["argmax_a"] = int(a.reshape(-1).argmax())
    out["argmax_b"] = int(b.reshape(-1).argmax())
    out["field_note"] = "`argmax_same` 由本包装补上（不是 `diag_checks` 的原生字段）；逐元素相等看 `elementwise_equal`"
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
    x = t.detach().float().reshape(-1)
    return dict(n=int(x.numel()), dtype_raw=str(t.dtype), dtype_stats="torch.float32",
                absmax=float(x.abs().max()), mean=float(x.mean()),
                finite=bool(torch.isfinite(x).all()),
                stats_note="`.float()` 仅用于统计；层/Norm/投影一律用**原始 dtype** 张量")


def _cache_digest(cache) -> dict:
    """单层 cache dict 的逐张量摘要（**不**保存整份缓存：它可由 H 逐位重建，已核对）。"""
    d = {k: A.stable_sha(v) for k, v in cache.items() if torch.is_tensor(v)}
    return dict(keys=sorted(cache.keys()), sha16=d, len=LB._cache_len_of(cache))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="layer_boundary_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209：最后一层（L27）是在**传递**上游差异，还是其**执行形状**也影响最终 token 选择？",
               semantics="同一份 H(8208)：捕获 embedding/L27 输入/输出/轮前缓存/cos/sin/最终 logits 行 → "
                          "L27 四格（输入来源 × 行数）→ 决策行统一 (1,1,H) → 同一最终 Norm → 同一绑定权重完整词表投影。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, dec_slot=DEC_SLOT,
                           row_index=ROW_INDEX, k=K, feed=FEED, cands=CANDS, watch=list(WATCH),
                           repeat=REPEAT, target_rope=TARGET_ROPE,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           saved_pt=str(SAVED_PT), norm_pt=str(NORM_PT), lmh_pt=str(LMH_PT),
                           in_scope="只做 L27 边界；不扫 28 层、不做层内算子扫描、不改算法、不做 FP32"))
    print(f"[lyr] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} feed={FEED} "
          f"cands={CANDS} repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[lyr] commit: {why}")
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


def _save_pt(payload: dict, run_id: str, selftest: bool = False) -> Path:
    """落盘原始张量（**原始 dtype**）。自测也走同一函数 ⇒ 落盘流程本身被 CPU 门覆盖。"""
    name = (f"{TAG}_{run_id}_selftest.pt" if selftest else f"{TAG}_{run_id}.pt")
    p = OUTDIR / name
    torch.save(payload, str(p))
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
        rec["layer"] = LB.layer_info(tgt, LAYER)
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        DC.require(rec["layer"]["is_last"], f"层 {LAYER} 不是最后一层（本轮只做 L{LAYER}）",
                   str({k: rec["layer"][k] for k in ("index", "n_layers", "is_last")}))
        print(f"[lyr] 层 {LAYER}/{rec['layer']['n_layers'] - 1}（最后一层={rec['layer']['is_last']}）"
              f"；Norm eps={rec['norm']['eps']}", flush=True)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))

        # ---------------- H ----------------
        cap = C._load_cap(Path(CAP_H))
        snap = cap["snapshot_cpu"]
        src_len = C._cache_len(snap)
        H, tinfo = C._truncate(snap, PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=src_len,
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H),
                               k_shape=[int(x) for x in H[0]["k"].shape], dtype=str(H[0]["k"].dtype),
                               note="与已保存的四格 / LM head / Norm 三次 run 同一份捕获与同一**纯切片**截断")
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2), compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"],
                       "两份捕获在 [0,PRE) 上并非逐位相同", str(rec["H_identity"]["compare"]))
            del H2
        print(f"[lyr] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 步骤 1：控制（S/B + embedding/L27/norm 三处捕获）----------------
        cS, cB = _to_dev(H, dev), _to_dev(H, dev)
        DR.require_distinct_storage(cS, cB, "S/B 两臂缓存")
        captured: dict = {}
        for arm_nm, mode_nm in (("S", "decode"), ("B", "verify")):
            with LB.embedding_capture(tgt, tag=f"{arm_nm}_e") as ec, \
                    LB.layer_capture(tgt, LAYER, tag=arm_nm) as lc, \
                    LB.record(tgt, tag=f"{arm_nm}_n") as _ni:
                with torch.inference_mode(), M.attention_phase(mode_nm):
                    if arm_nm == "S":
                        lg_s, cache_s = spec_dec._step(tgt, FEED[0], cS)
                    else:
                        rows_b, blk_b = spec_dec._verify(tgt, cB, CANDS)
                norm_cap = LB.take()
            captured[arm_nm] = dict(
                embed=ec, layer=lc, norm=norm_cap["caps"][1],
                row=(LB.row_of(lg_s, None) if arm_nm == "S" else LB.row_of(rows_b, 0)))
        S_c, B_c = captured["S"], captured["B"]
        aS, aB4 = S_c["layer"]["pre"]["a"], B_c["layer"]["pre"]["a"]
        zS, zB4 = S_c["layer"]["post"]["z"], B_c["layer"]["post"]["z"]
        cosS, sinS = S_c["layer"]["pre"]["cos"], S_c["layer"]["pre"]["sin"]
        cosB, sinB = B_c["layer"]["pre"]["cos"], B_c["layer"]["pre"]["sin"]
        cacheS_pre, cacheB_pre = S_c["layer"]["pre"]["cache_pre"], B_c["layer"]["pre"]["cache_pre"]
        xS, xB4 = S_c["norm"]["x_in"], B_c["norm"]["x_in"]      # 最终 Norm 的输入
        hS, hB4 = S_c["norm"]["y_out"], B_c["norm"]["y_out"]    # 最终 Norm 的输出
        row_s, row_b = S_c["row"], B_c["row"]
        len_s, len_b = spec_cache.cache_length(cache_s), spec_cache.cache_length(blk_b)
        rec["step1_control"] = dict(
            shapes=dict(aS=[int(x) for x in aS.shape], aB4=[int(x) for x in aB4.shape],
                        zS=[int(x) for x in zS.shape], zB4=[int(x) for x in zB4.shape],
                        cosS=[int(x) for x in cosS.shape], cosB=[int(x) for x in cosB.shape]),
            positions=dict(S_cache_len_after=len_s, B_cache_len_after=len_b,
                           both_predict_target=bool(len_s == TARGET_POS == PRE + 1 + ROW_INDEX),
                           slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1],
                           S_row_index_used="末位行（(1,1,V) 的唯一一行）", B_row_index_used="rows[0]"),
            phases=dict(S="decode", B="verify"),
            cache_lens=dict(S_pre=S_c["layer"]["pre"]["cache_len_before"],
                            S_post=S_c["layer"]["post"]["cache_len_after"],
                            B_pre=B_c["layer"]["pre"]["cache_len_before"],
                            B_post=B_c["layer"]["post"]["cache_len_after"]),
            S=_fields(row_s), B=_fields(row_b))
        DC.require(list(aS.shape) == [1, 1, 1536] and list(aB4.shape) == [1, 4, 1536]
                   and list(zS.shape) == [1, 1, 1536] and list(zB4.shape) == [1, 4, 1536],
                   "捕获形状不符", str(rec["step1_control"]["shapes"]))
        DC.require(str(aS.dtype) == str(W.dtype), "捕获 dtype 与权重不一致（必须原生 bf16）",
                   f"aS={aS.dtype} W={W.dtype}")
        DC.require(rec["step1_control"]["positions"]["both_predict_target"],
                   "S 末位行与 B rows[0] 未指向同一预测位置")
        s_f, b_f = _fields(row_s), _fields(row_b)
        print(f"[lyr] 控制：S argmax={s_f['argmax_id']}（margin={s_f['margin_top1_top2']}，并列={s_f['n_max_tied']}） "
              f"B argmax={b_f['argmax_id']}（margin={b_f['margin_top1_top2']}）", flush=True)

        # ---- 捕获位置/对应行/轮前缓存的**逐位**核对 ----
        rec["capture_chain"] = dict(
            zS_equals_norm_input_xS=_cmp(zS, xS, "zS_vs_xS"),
            zB4_equals_norm_input_xB4=_cmp(zB4, xB4, "zB4_vs_xB4"),
            norm_output_S_equals_captured_h=_cmp(S_c["norm"]["y_out"], hS, "hS_selfcheck"),
            embed_ids_S=S_c["embed"]["ids"], embed_ids_B=B_c["embed"]["ids"],
            embed_row0_equal=_cmp(S_c["embed"]["out"][0, 0], B_c["embed"]["out"][0, 0],
                                  "embedS_row0_vs_embedB_row0"),
            embed_note="S 的输入是 [20642]（1 行），B 是 4 个候选（4 行）：位置 8208 的 embedding 行必须**逐位相同**",
            cos_row0_equal=_cmp(cosS, cosB[:, :, 0:1, :], "cosS_vs_cosB_row0"),
            sin_row0_equal=_cmp(sinS, sinB[:, :, 0:1, :], "sinS_vs_sinB_row0"),
            cos_note="S 的 rope 覆盖位置 8208（1 行）；B 覆盖 8208..8211（4 行）⇒ B 的第 0 行必须与 S 逐位相同",
            cache_pre_S_vs_B=_cmp(torch.cat([cacheS_pre["k"].reshape(-1),
                                             cacheS_pre["v"].reshape(-1)]),
                                  torch.cat([cacheB_pre["k"].reshape(-1),
                                             cacheB_pre["v"].reshape(-1)]),
                                  "cacheS_pre_vs_cacheB_pre"),
            cache_pre_S_vs_H_slice=_cmp(torch.cat([cacheS_pre["k"].reshape(-1),
                                                   cacheS_pre["v"].reshape(-1)]),
                                        torch.cat([H[LAYER]["k"].to(dev).reshape(-1),
                                                   H[LAYER]["v"].to(dev).reshape(-1)]),
                                        "cacheS_pre_vs_H_slice"),
            cache_pre_digests=dict(S=_cache_digest(cacheS_pre), B=_cache_digest(cacheB_pre)))
        for k in ("zS_equals_norm_input_xS", "zB4_equals_norm_input_xB4", "embed_row0_equal",
                  "cos_row0_equal", "sin_row0_equal", "cache_pre_S_vs_B", "cache_pre_S_vs_H_slice"):
            DC.require(rec["capture_chain"][k]["elementwise_equal"], f"捕获核对失败：{k}",
                       str(rec["capture_chain"][k]))
        print("[lyr] 捕获链 ✔ zS≡Norm输入、zB4≡Norm输入、embedding 第 0 行相同、"
              f"cos/sin 第 0 行相同、L27 轮前缓存 S≡B≡H[{LAYER}]", flush=True)

        # ---- 与已保存的局部对照行 / 锚点 / 关闭捕获 ----
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        rec["step1_control"]["vs_saved_local_rows"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S=(_cmp(row_s.float().cpu(), saved_rows["H_AR-S"].float(), "S_vs_saved_H_AR_S")
               if "H_AR-S" in saved_rows else None),
            B=(_cmp(row_b.float().cpu(), saved_rows["H_AR-B"].float(), "B_vs_saved_H_AR_B")
               if "H_AR-B" in saved_rows else None),
            note="与**已保存的局部对照行**逐元素比较（不是与未捕获的自由生成整行比）")
        if saved_rows:
            for k in ("S", "B"):
                DC.require(rec["step1_control"]["vs_saved_local_rows"][k]["elementwise_equal"],
                           f"控制失败：{k} 行与已保存局部对照行不同",
                           str(rec["step1_control"]["vs_saved_local_rows"][k]))
        anch_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anch_b = C._anchor_specdec(REF_RUN_RB, PRE, DEC_SLOT, TARGET_POS)
        rec["anchors"] = dict(S=anch_s, B=anch_b,
                              note="真实运行记录里的 token 选择（**argmax 级**）")
        rec["step1_control"]["token_matches_anchors"] = dict(
            S=dict(measured=s_f["argmax_id"], anchor=anch_s.get("token"),
                   match=bool(s_f["argmax_id"] == anch_s.get("token"))),
            B=dict(measured=b_f["argmax_id"], anchor=anch_b.get("token"),
                   match=bool(b_f["argmax_id"] == anch_b.get("token"))))
        DC.require(all(v["match"] for v in rec["step1_control"]["token_matches_anchors"].values()),
                   "控制失败：S/B argmax 与锚点不一致（停止，不解释四格）",
                   str(rec["step1_control"]["token_matches_anchors"]))
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
            DC.require(all(v is True for k, v in rec["capture_off_check"].items()
                           if k.endswith("identical")),
                       "关闭捕获后判定行发生变化", str(rec["capture_off_check"]))
            print("[lyr] 捕获关闭等价 ✔", flush=True)

        # ---------------- 步骤 2/3：L27 四格 ----------------
        aS_snap, aB4_snap = DC.snapshot(aS), DC.snapshot(aB4)
        zS_snap, zB4_snap = DC.snapshot(zS), DC.snapshot(zB4)
        # 轮前缓存的运行前快照（用于事后证明它们没被任何一格改写）
        cS_pre_snap = LB._normal_cache_clone(cacheS_pre)
        cB_pre_snap = LB._normal_cache_clone(cacheB_pre)
        Hdim = int(aS.shape[-1])
        aB1 = aB4[:, 0, :].reshape(1, 1, Hdim).clone()          # 独立拷贝的单行形式
        aB4c = aB4.clone()
        aB4c[0, 0] = aS[0, 0]
        other_rows_ok = bool(torch.equal(aB4c[0, 1:], aB4_snap[0, 1:]))
        DC.require(other_rows_ok, "四行格其余三行被改动（必须保持同一份真实 B 输入）")
        DC.require(int(aB1.data_ptr()) != int(aB4.data_ptr()), "单行形式必须是独立拷贝")
        rec["cells_setup"] = dict(
            aS=DC.layout(aS), aB4=DC.layout(aB4), aB1_copy=DC.layout(aB1), aB4c=DC.layout(aB4c),
            cosS=DC.layout(cosS), cosB=DC.layout(cosB),
            aB1_is_independent_copy=True, other_rows_equal=other_rows_ok,
            row0_slice_layout_control=DC.layout_control(aB4[:, 0:1, :], "aB4_row0_slice"),
            layout_note="单行格用**独立拷贝** `aB1`；`aB4[:,0:1,:]` 的 `.contiguous()` 不会复制（本就连续）"
                        "⇒ 本工具**不**把它当布局对照（`control_valid=False` 如实记录）；"
                        "本轮变量是「输入来源 × L27 行数」，不含布局变量")
        cells: dict = {}
        for name, x, cos, sin, cache0, phase, take_row in (
                ("LA", aS, cosS, sinS, cacheS_pre, "decode", 0),
                ("LB", aB1, cosS, sinS, cacheS_pre, "decode", 0),
                ("LC", aB4c, cosB, sinB, cacheB_pre, "verify", 0),
                ("LD", aB4, cosB, sinB, cacheB_pre, "verify", 0)):
            outs, metas, lens = [], [], []
            for _it in range(REPEAT):
                z, len_after, meta = LB.layer_forward(tgt, LAYER, x,
                                                      cos, sin, LB._normal_cache_clone(cache0),
                                                      phase=phase)
                outs.append(z)
                metas.append(meta)
                lens.append(len_after)
            same = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
            cells[name] = dict(input_shape=[int(v) for v in x.shape], take_row=take_row,
                               repeat_elementwise_equal=same, n_repeat=REPEAT,
                               input_layout=DC.layout(x), meta=metas[0],
                               cache_len_after_all=lens, out=outs[0])
        rec["cells"] = {k: dict(input_shape=v["input_shape"], take_row=v["take_row"],
                                repeat_elementwise_equal=v["repeat_elementwise_equal"],
                                n_repeat=v["n_repeat"], input_layout=v["input_layout"],
                                meta=v["meta"], cache_len_after_all=v["cache_len_after_all"],
                                out_stats=_stats(v["out"]))
                        for k, v in cells.items()}
        for k, v in cells.items():
            DC.require(v["repeat_elementwise_equal"], f"{k}: 格内重复不一致")
        rec["cells_controls"] = dict(
            LA_row0_reproduces_captured_zS=_cmp(cells["LA"]["out"][0, 0], zS_snap[0, 0], "LA_vs_zS"),
            LD_row0_reproduces_captured_zB4=_cmp(cells["LD"]["out"][0, 0], zB4_snap[0, 0], "LD_vs_zB4"),
            note="真实组合（aS×单行、aB4×四行）的**第 0 行**必须逐位复现捕获输出；"
                 "这也**只**证明重放忠实，**不**能单独证明 L27 没有形状效应")
        for k, v in rec["cells_controls"].items():
            if isinstance(v, dict):
                DC.require(v["elementwise_equal"], f"控制失败：{k} 未逐位复现捕获输出", str(v))
        print("[lyr] L27 四格控制 ✔ LA 第 0 行==捕获 zS、LD 第 0 行==捕获 zB4（逐位）", flush=True)

        vec_cmp = {
            "fixed_single_row_change_input_LA_vs_LB":
                _cmp(cells["LA"]["out"][0, 0], cells["LB"]["out"][0, 0], "LA_vs_LB"),
            "fixed_four_row_change_input_LC_vs_LD":
                _cmp(cells["LC"]["out"][0, 0], cells["LD"]["out"][0, 0], "LC_vs_LD"),
            "fixed_input_aS_change_shape_LA_vs_LC":
                _cmp(cells["LA"]["out"][0, 0], cells["LC"]["out"][0, 0], "LA_vs_LC"),
            "fixed_input_aB_change_shape_LB_vs_LD":
                _cmp(cells["LB"]["out"][0, 0], cells["LD"]["out"][0, 0], "LB_vs_LD"),
        }
        rec["cells_vector_compare"] = vec_cmp

        # 输入比较（层输入 aS vs aB）
        rec["input_compare"] = dict(
            aS_vs_aB=_cmp(aS[0, 0], aB4[0, 0], "aS_row0_vs_aB_row0"),
            aS_stats=_stats(aS), aB4_stats=_stats(aB4),
            zS_vs_zB4=_cmp(zS[0, 0], zB4[0, 0], "zS_row0_vs_zB4_row0"),
            reading="`aS ≠ aB` 只证明上游已有差异；是否驱动翻转必须看两向四格")
        ic = rec["input_compare"]
        print(f"[lyr] 输入/输出比较：aS vs aB n_diff={ic['aS_vs_aB']['n_diff']} "
              f"max|Δ|={ic['aS_vs_aB']['abs_err_max']:.6g}；zS vs zB n_diff={ic['zS_vs_zB4']['n_diff']} "
              f"max|Δ|={ic['zS_vs_zB4']['abs_err_max']:.6g}", flush=True)

        # ---------------- 统一尾部：同一最终 Norm + 同一绑定权重（完整词表）----------------
        proj: dict = {}
        for k, v in cells.items():
            zv = v["out"][0, v["take_row"]].reshape(1, 1, Hdim)
            DC.require(tuple(zv.shape) == (1, 1, Hdim) and zv.dtype == W.dtype,
                       f"{k}: 决策向量形状/dtype 不符", f"{tuple(zv.shape)} {zv.dtype}")
            hn, nmeta = LB.norm_forward(tgt, zv)
            _full, row = LB.project(hn, W, take_row=0)
            proj[k] = dict(zv=zv, hn=hn, nmeta=nmeta, raw=row.detach().clone(),
                           float_row=row.detach().float().cpu())
        rec["tail"] = dict(
            norm_module=rec["norm"]["module_class"], norm_eps=rec["norm"]["eps"],
            expression=LB.PROJECTION_EXPRESSION, n_vocab=int(W.shape[0]),
            shape_fixed="全部四格：L27 输出第 0 行 → (1,1,1536) → 同一最终 Norm → (1,1,1536) @ W.T（完整词表）",
            norm_rows={k: _fields(v["raw"]) for k, v in proj.items()},
            vector_stats={k: _stats(v["hn"]) for k, v in proj.items()})
        for k in ("LA", "LB", "LC", "LD"):
            f = rec["tail"]["norm_rows"][k]
            print(f"[lyr]   P_{k} argmax={f['argmax_id']:6d} margin={f['margin_top1_top2']:.6g} "
                  f"并列={f['n_max_tied']} watch={f['watch_scores']}", flush=True)
        proj_pairs = {
            "proj_fixed_shape_change_input_single_P_LA_vs_P_LB":
                _cmp(proj["LA"]["float_row"], proj["LB"]["float_row"], "P_LA_vs_P_LB"),
            "proj_fixed_shape_change_input_four_P_LC_vs_P_LD":
                _cmp(proj["LC"]["float_row"], proj["LD"]["float_row"], "P_LC_vs_P_LD"),
            "proj_fixed_input_change_shape_aS_P_LA_vs_P_LC":
                _cmp(proj["LA"]["float_row"], proj["LC"]["float_row"], "P_LA_vs_P_LC"),
            "proj_fixed_input_change_shape_aB_P_LB_vs_P_LD":
                _cmp(proj["LB"]["float_row"], proj["LD"]["float_row"], "P_LB_vs_P_LD"),
        }
        rec["tail_compare"] = proj_pairs
        for k in PROJ_PAIR_KEYS:
            v = proj_pairs[k]
            print(f"[lyr]   {k:48s} same_argmax={v['argmax_same']} elementwise={v['elementwise_equal']} "
                  f"n_diff={v['n_diff']} max|Δ|={v['abs_err_max']:.6g} "
                  f"({v['argmax_a']} vs {v['argmax_b']})", flush=True)

        # ---- 跨 run 交叉核对：**只用单行投影参考**（不用四行 LM head 参考做逐位必需控制）----
        rec["cross_run"] = dict(norm_pt=str(NORM_PT), lmh_pt=str(LMH_PT))
        if Path(NORM_PT).exists():
            prv = torch.load(NORM_PT, map_location="cpu")
            pr = prv.get("proj_rows_raw", {})
            rec["cross_run"]["norm_run"] = dict(
                P_LA_vs_prev_P_NA=(_cmp(proj["LA"]["float_row"], pr["NA"].float(), "P_LA_vs_prev_P_NA")
                                   if "NA" in pr else None),
                P_LD_vs_prev_P_ND=(_cmp(proj["LD"]["float_row"], pr["ND"].float(), "P_LD_vs_prev_P_ND")
                                   if "ND" in pr else None),
                note="上一轮 Norm 边界的 `proj_rows_raw` 四格**全部是单行尾部**（(1,1,H)@W.T）"
                     "⇒ 与本轮同为单行，可作**逐位必需控制**")
        if Path(LMH_PT).exists():
            prv2 = torch.load(LMH_PT, map_location="cpu").get("rows", {})
            rec["cross_run"]["lmh_run"] = dict(
                P_LA_vs_prev_HS_1=(_cmp(proj["LA"]["float_row"], prv2["HS_1"].float(),
                                        "P_LA_vs_prev_HS_1") if "HS_1" in prv2 else None),
                P_LD_vs_prev_HB_1=(_cmp(proj["LD"]["float_row"], prv2["HB_1"].float(),
                                        "P_LD_vs_prev_HB_1") if "HB_1" in prv2 else None),
                reference_only_HB_4=(_cmp(proj["LD"]["float_row"], prv2["HB_4"].float(),
                                          "P_LD_vs_prev_HB_4") if "HB_4" in prv2 else None),
                note="`HS_1`/`HB_1` 是**单行**投影 ⇒ 必需控制；`HB_4` 是**四行**投影，"
                     "**不**作逐位必需控制（预期差异 = 上一轮测到的 LM head 形状效应）")
        # ⚠️ **必需控制只有"单行 ↔ 单行"四项**：显式按键列出，**不**用名字里是否含某子串来过滤
        #    （首跑就是靠子串过滤，结果把 `reference_only_HB_4` 也当成了必需控制 ⇒ 白跑一次）。
        for _grp, _k in (("norm_run", "P_LA_vs_prev_P_NA"), ("norm_run", "P_LD_vs_prev_P_ND"),
                         ("lmh_run", "P_LA_vs_prev_HS_1"), ("lmh_run", "P_LD_vs_prev_HB_1")):
            _v = (rec["cross_run"].get(_grp) or {}).get(_k)
            if isinstance(_v, dict):
                DC.require(_v["elementwise_equal"], f"跨 run 单行参考核对失败：{_k}", str(_v))
        print("[lyr] 跨 run 交叉核对 ✔（单行参考：上一轮 Norm 四格 + LM head 的 HS_1/HB_1）", flush=True)

        # ---------------- 源完整性 ----------------
        rec["source_integrity"] = dict(
            aS_unmodified=_cmp(aS, aS_snap, "aS_vs_snapshot", ref_snapshot=aS_snap, src=aS),
            aB4_unmodified=_cmp(aB4, aB4_snap, "aB4_vs_snapshot", ref_snapshot=aB4_snap, src=aB4),
            zS_unmodified=_cmp(zS, zS_snap, "zS_vs_snapshot", ref_snapshot=zS_snap, src=zS),
            zB4_unmodified=_cmp(zB4, zB4_snap, "zB4_vs_snapshot", ref_snapshot=zB4_snap, src=zB4),
            cacheS_pre_unmodified=_cmp(
                torch.cat([cacheS_pre["k"].reshape(-1), cacheS_pre["v"].reshape(-1)]),
                torch.cat([cS_pre_snap["k"].reshape(-1), cS_pre_snap["v"].reshape(-1)]),
                "cacheS_pre_vs_snapshot"),
            cacheB_pre_unmodified=_cmp(
                torch.cat([cacheB_pre["k"].reshape(-1), cacheB_pre["v"].reshape(-1)]),
                torch.cat([cB_pre_snap["k"].reshape(-1), cB_pre_snap["v"].reshape(-1)]),
                "cacheB_pre_vs_snapshot"),
            other_rows_still_equal=bool(torch.equal(aB4c[0, 1:], aB4_snap[0, 1:])),
            W_after=LB.weights_info(tgt),
            W_identity_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"]
                                   == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"]
                                 == rec["W_before"]["digest_subsample"]),
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"]
                                    == rec["norm"]["weight"]["data_ptr"]),
            note="四格一律在独立克隆上做；捕获源、轮前缓存与两份权重在全程不得改变")
        DC.require(rec["source_integrity"]["other_rows_still_equal"]
                   and rec["source_integrity"]["W_identity_stable"]
                   and rec["source_integrity"]["W_digest_stable"]
                   and rec["source_integrity"]["norm_weight_stable"],
                   "源张量或权重在四格过程中被改动", str(rec["source_integrity"]))

        # ---------------- 归因 ----------------
        def flip(c):
            DC.require("argmax_same" in c, "比较字段缺失（`_cmp` 必须给 argmax_same）", str(c))
            return bool(c["elementwise_equal"] is False and c["argmax_same"] is False)

        pj = rec["tail_compare"]
        rec["verdict"] = dict(
            l27_input_already_differs=bool(not ic["aS_vs_aB"]["elementwise_equal"]),
            l27_shape_change_identical_on_row0=dict(
                fixed_input_aS=vec_cmp["fixed_input_aS_change_shape_LA_vs_LC"]["elementwise_equal"],
                fixed_input_aB=vec_cmp["fixed_input_aB_change_shape_LB_vs_LD"]["elementwise_equal"],
                n_diff=dict(aS=vec_cmp["fixed_input_aS_change_shape_LA_vs_LC"]["n_diff"],
                            aB=vec_cmp["fixed_input_aB_change_shape_LB_vs_LD"]["n_diff"]),
                max_abs=dict(aS=vec_cmp["fixed_input_aS_change_shape_LA_vs_LC"]["abs_err_max"],
                             aB=vec_cmp["fixed_input_aB_change_shape_LB_vs_LD"]["abs_err_max"]),
                reading="`True` = 固定输入、只改 L27 行数时**第 0 行逐位相同** ⇒ 该输入下**未观察到**形状效应"),
            fixed_shape_change_input_flips=dict(
                single_row=flip(pj["proj_fixed_shape_change_input_single_P_LA_vs_P_LB"]),
                four_row=flip(pj["proj_fixed_shape_change_input_four_P_LC_vs_P_LD"])),
            fixed_input_change_shape_flips=dict(
                aS=flip(pj["proj_fixed_input_change_shape_aS_P_LA_vs_P_LC"]),
                aB=flip(pj["proj_fixed_input_change_shape_aB_P_LB_vs_P_LD"])),
            rule="① `aS≠aB` 只证明「上游已有差异」，**不**证明上游独占最终原因；"
                 "② 固定 L27 形状换输入会翻转 ⇒ 支持**输入来源效应**；"
                 "③ 固定输入换 L27 形状会翻转 ⇒ 支持 **L27 执行路径效应**；"
                 "④ 有交互如实记录；不翻转也要记录向量与 logits 的变化；"
                 "⑤ **不**假定差异逐层单调增长、**不**用简单二分排除更早层；"
                 "⑥ **不**把数值差异直接判为 bug，**不**用 margin 豁免正确性，**不**判定 323/659 谁是数学真值。")
        v = rec["verdict"]
        print(f"[lyr] 归因：L27 前已不同={v['l27_input_already_differs']} "
              f"换输入翻转={v['fixed_shape_change_input_flips']} "
              f"换形状翻转={v['fixed_input_change_shape_flips']} "
              f"形状改变第 0 行向量={v['l27_shape_change_identical_on_row0']}", flush=True)

        att = M.attention_stats()
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att.get("s_max_by_phase"))
        rec["capture_clean"] = LB.assert_clean()
        rec["stage"] = "complete"
        payload = dict(
            aS_raw=aS.detach().cpu(), aB4_raw=aB4.detach().cpu(),
            zS_raw=zS.detach().cpu(), zB4_raw=zB4.detach().cpu(),
            cosS_raw=cosS.detach().cpu(), sinS_raw=sinS.detach().cpu(),
            cosB_raw=cosB.detach().cpu(), sinB_raw=sinB.detach().cpu(),
            real_S_row_raw=row_s.detach().cpu(), real_B_row_raw=row_b.detach().cpu(),
            cell_out_raw={k: v["out"].detach().cpu() for k, v in cells.items()},
            cell_dv_raw={k: v["zv"].detach().cpu() for k, v in proj.items()},
            tail_norm_raw={k: v["hn"].detach().cpu() for k, v in proj.items()},
            tail_rows_raw={k: v["raw"].detach().cpu() for k, v in proj.items()},
            tail_rows_f32={k: v["float_row"] for k, v in proj.items()},
            real_S_row_f32=row_s.float().cpu(), real_B_row_f32=row_b.float().cpu())
        tp = _save_pt(payload, run_id)
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp),
                                raw_dtype_note="`*_raw` = 原始 bf16；`*_f32` = 仅供统计/外部比较")
        _write(rec, run_id)
        print(f"[lyr] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[lyr] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[lyr] RESULT_JSON={p}")
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
    """**不加载大模型**：微型模型上把本工具的机制全走一遍（含 L27 四格、统一尾部、落盘流程）。"""
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    global LAYER, PRE, TARGET_POS, DEC_SLOT, ROW_INDEX, FEED, CANDS, WATCH, REPEAT
    torch.manual_seed(0)
    PRE, DEC_SLOT, ROW_INDEX, FEED, CANDS = 5, 1, 0, [3], [3, 4, 5, 6]
    TARGET_POS, WATCH, REPEAT = 6, (1, 2, 3), 2
    tgt = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    LAYER = len(tgt.layers) - 1
    W = tgt.embed_tokens.weight
    ids = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    check("★ 起点 H 长度 = PRE；该层是最后一层",
          spec_cache.cache_length(H) == PRE and LB.layer_info(tgt, LAYER)["is_last"] is True)
    dev = torch.device("cpu")
    cS, cB = _to_dev(H, dev), _to_dev(H, dev)
    DR.require_distinct_storage(cS, cB, "自测两臂")
    capd: dict = {}
    for arm_nm, mode_nm in (("S", "decode"), ("B", "verify")):
        with LB.embedding_capture(tgt, tag=f"{arm_nm}_e") as ec, \
                LB.layer_capture(tgt, LAYER, tag=arm_nm) as lc, \
                LB.record(tgt, tag=f"{arm_nm}_n") as _ni:
            with torch.inference_mode(), M.attention_phase(mode_nm):
                if arm_nm == "S":
                    lg_s, _cs = spec_dec._step(tgt, FEED[0], cS)
                else:
                    rows_b, _blk = spec_dec._verify(tgt, cB, CANDS)
            ncap = LB.take()
        capd[arm_nm] = dict(embed=ec, layer=lc, norm=ncap["caps"][1],
                            row=(LB.row_of(lg_s, None) if arm_nm == "S" else LB.row_of(rows_b, 0)))
    S_c, B_c = capd["S"], capd["B"]
    aS, aB4 = S_c["layer"]["pre"]["a"], B_c["layer"]["pre"]["a"]
    zS, zB4 = S_c["layer"]["post"]["z"], B_c["layer"]["post"]["z"]
    cosS, sinS = S_c["layer"]["pre"]["cos"], S_c["layer"]["pre"]["sin"]
    cosB, sinB = B_c["layer"]["pre"]["cos"], B_c["layer"]["pre"]["sin"]
    cacheS_pre, cacheB_pre = S_c["layer"]["pre"]["cache_pre"], B_c["layer"]["pre"]["cache_pre"]
    check("★ 捕获形状/位置：aS (1,1,32)、aB4 (1,4,32)、zS/zB4 同形；zS≡Norm 输入、zB4≡Norm 输入",
          list(aS.shape) == [1, 1, 32] and list(aB4.shape) == [1, 4, 32]
          and bool(torch.equal(zS, S_c["norm"]["x_in"])) and bool(torch.equal(zB4, B_c["norm"]["x_in"])))
    check("★ 对应行/轮前缓存的逐位核对：embedding 第 0 行相同、cos/sin 第 0 行相同、缓存 S≡B≡H 切片",
          bool(torch.equal(S_c["embed"]["out"][0, 0], B_c["embed"]["out"][0, 0]))
          and bool(torch.equal(cosS, cosB[:, :, 0:1, :])) and bool(torch.equal(sinS, sinB[:, :, 0:1, :]))
          and bool(torch.equal(cacheS_pre["k"], H[LAYER]["k"]))
          and bool(torch.equal(cacheS_pre["k"], cacheB_pre["k"]))
          and int(cacheS_pre["k"].shape[2]) == PRE)
    check("★ 决策行索引：S 末位行、B rows[0]，位置一致（槽位表 + 缓存长度）",
          DR.slot_table(PRE, len(CANDS))[1]["predicted_pos"] == TARGET_POS
          and S_c["row"].dim() == 1 and B_c["row"].dim() == 1)
    Hdim = int(aS.shape[-1])
    aS_snap, aB4_snap, zS_snap = DC.snapshot(aS), DC.snapshot(aB4), DC.snapshot(zS)
    cS_pre_snap = LB._normal_cache_clone(cacheS_pre)
    aB1 = aB4[:, 0, :].reshape(1, 1, Hdim).clone()
    aB4c = aB4.clone()
    aB4c[0, 0] = aS[0, 0]
    check("★ 独立副本：aB1 与 aB4 不共享存储；四行格只换第 0 行、其余三行逐位不变",
          int(aB1.data_ptr()) != int(aB4.data_ptr())
          and bool(torch.equal(aB4c[0, 1:], aB4_snap[0, 1:])))
    cells, proj = {}, {}
    for name, x, cos, sin, cache0, phase in (
            ("LA", aS, cosS, sinS, cacheS_pre, "decode"),
            ("LB", aB1, cosS, sinS, cacheS_pre, "decode"),
            ("LC", aB4c, cosB, sinB, cacheB_pre, "verify"),
            ("LD", aB4, cosB, sinB, cacheB_pre, "verify")):
        outs, metas = [], []
        for _it in range(REPEAT):
            z, _len_after, meta = LB.layer_forward(tgt, LAYER, x, cos, sin,
                                                   LB._normal_cache_clone(cache0), phase=phase)
            outs.append(z)
            metas.append(meta)
        same = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
        cells[name] = dict(same=same, out=outs[0], meta=metas[0])
        zv = outs[0][0, 0].reshape(1, 1, Hdim)
        hn, _nmeta = LB.norm_forward(tgt, zv)
        _f, row = LB.project(hn, W, take_row=0)
        proj[name] = dict(zv=zv, hn=hn, raw=row)
    check("★ 格内重复逐位相同（四格各 2 次）", all(v["same"] for v in cells.values()))
    check("★ 控制：LA 第 0 行≡捕获 zS、LD 第 0 行≡捕获 zB4（逐位）",
          bool(torch.equal(cells["LA"]["out"][0, 0], zS_snap[0, 0]))
          and bool(torch.equal(cells["LD"]["out"][0, 0], zB4[0, 0])))
    check("★ 每格元数据齐全：phase / 分流 / s,T,p / 掩码可见范围 / 缓存长度变化",
          all(all(k in v["meta"] for k in ("phase", "dispatch", "s", "T", "p", "mask_rule",
                                           "visible_last_key_row0", "cache_len_before",
                                           "cache_len_after")) for v in cells.values())
          and cells["LB"]["meta"]["phase"] == "decode" and cells["LD"]["meta"]["phase"] == "verify"
          and cells["LB"]["meta"]["cache_len_after"] == PRE + 1
          and cells["LD"]["meta"]["cache_len_after"] == PRE + 4)
    check("★ 统一尾部：四格决策向量 → (1,1,H) → 同一 Norm → 完整词表投影（有限）",
          all(tuple(v["zv"].shape) == (1, 1, Hdim) and tuple(v["raw"].shape) == (64,)
              and bool(torch.isfinite(v["raw"]).all()) for v in proj.values()))
    check("★ 源与权重未被改动（a/z/轮前缓存与快照一致；Norm 与层权重 data_ptr 不变）",
          bool(torch.equal(aS, aS_snap)) and bool(torch.equal(aB4, aB4_snap))
          and bool(torch.equal(zS, zS_snap)) and bool(torch.equal(cacheS_pre["k"], cS_pre_snap["k"]))
          and LB.norm_info(tgt)["weight"]["data_ptr"] == int(tgt.norm.weight.data_ptr()))
    off = LB.install(tgt, enabled=False, tag="off")
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_s2, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    LB.assert_clean()
    check("★ 关闭捕获时与带捕获逐位相同（真实模型侧的关闭等价）",
          off["installed"] is False and bool(torch.equal(S_c["row"], LB.row_of(lg_s2, None))))
    check("★ 钩子全部回基线（层 pre/post、norm、embedding）",
          len(tgt.layers[LAYER]._forward_pre_hooks) == 0
          and len(tgt.layers[LAYER]._forward_hooks) == 0
          and len(tgt.norm._forward_hooks) == 0
          and len(tgt.embed_tokens._forward_hooks) == 0)
    tp = _save_pt(dict(aS_raw=aS.detach().cpu(), aB4_raw=aB4.detach().cpu(),
                       zS_raw=zS.detach().cpu(), zB4_raw=zB4.detach().cpu(),
                       cell_out_raw={k: v["out"].detach().cpu() for k, v in cells.items()},
                       tail_rows_raw={k: v["raw"].detach().cpu() for k, v in proj.items()}),
                  run_id, selftest=True)
    check("★ 落盘流程可跑通（原始 dtype 张量写入 .pt）", tp.exists() and tp.stat().st_size > 0, str(tp))
    print(f"\nL27 边界定位（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：只验证机制（捕获链/轮前缓存/四格/控制/尾部/落盘/关闭等价）",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
