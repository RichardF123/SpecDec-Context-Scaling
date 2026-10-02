"""layer_chain_f13_p8209.py — 位置 8209 的**全层边界**捕获：最早在哪一层产生隐藏状态差异（F13；只读、单次）。

## 唯一目标
在**同一次真实 S/B 前向**里捕获 **全部 28 层**的**决策行**边界（输入行 / 输出行），找出：

1. `earliest_observed_input_difference` —— 最早"该层**输入行** S/B 不同"的层；
2. `earliest_observed_output_difference` —— 最早"该层**输出行** S/B 不同"的层；
3. `first_same_input_different_output` —— 第一层"**输入逐位相同、输出不同**"的层
   ⇒ 这一层就是**当前测试轨迹中最早观测到的分歧产生层**（下一轮四格的候选）。

**三者严格分开**：BF16 下完全可能出现"某层输入已经不同、但输出恰好又逐位相同"或反之；
**不**假定差异随层数单调增长、**不**用二分排除未直接测量的层、后续层重新逐位相同也**如实保留全部记录**。

## 边界与纪律
* 只捕获**决策行**（`(hidden,)`），**不**保存整段 `(1,S,hidden)` 激活；行在 **GPU 上 `detach().clone()`**，
  **前向结束后**才 `.cpu()`（避免每层同步、避免改变执行轨迹）。
* 只读钩子（返回 `None`）、不改 cache/权重/mask/RoPE/argmax；**不重写 DecoderLayer 公式**；
  `finally` 移除全部钩子（异常也移除）。
* **不**对某层做四格、**不**进入 attention/MLP/RMSNorm 内部、**不**改生产实现、**不**改精度、
  **不**自由生成、**不**加载 draft、**不**做 FP32；GPU 串行；硬失败落盘停止（无自动重试）。
* 旧产物**不覆盖**；B6 状态**不变**。
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

TAG = os.environ.get("LC_TAG", "div8209_chain")
MODE = os.environ.get("LC_MODE", "gpu").strip().lower()
PRE = int(os.environ.get("LC_PRE", "8208"))
TARGET_POS = int(os.environ.get("LC_TARGET", "8209"))
K = int(os.environ.get("LC_K", "4"))
FEED = [int(x) for x in os.environ.get("LC_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("LC_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("LC_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("LC_REPEAT", "2"))
TARGET_ROPE = os.environ.get("LC_TARGET_ROPE", "cpu_fp32").strip().lower()
CAPTURE_OFF_CHECK = S.env_flag("LC_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("LC_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("LC_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
SAVED_PT = os.environ.get("LC_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
L27_PT = os.environ.get("LC_L27_PT", str(OUTDIR / "div8209_l27_20261002T161807Z.pt"))
REF_RUN_AR = os.environ.get("LC_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("LC_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("LC_ARM", "eager")
DEC_SLOT = TARGET_POS - PRE                  # 1
ROW_INDEX = DEC_SLOT - 1                     # 0（S 的唯一一行 / B 的第 0 行）


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
    """`diag_checks.cmp_tensor` + `n_diff`/`argmax_same`（两套字段口径都显式存在）。"""
    out = DC.cmp_tensor(a, b, name, **kw)
    out["n_diff"] = int((a != b).sum())
    out["argmax_same"] = bool(int(a.reshape(-1).argmax()) == int(b.reshape(-1).argmax()))
    out["argmax_a"] = int(a.reshape(-1).argmax())
    out["argmax_b"] = int(b.reshape(-1).argmax())
    out["field_note"] = "`argmax_same` 由本包装补上（不是 `diag_checks` 的原生字段）"
    return out


def _fields(row) -> dict:
    f = DF.row_fields(row, watch=list(WATCH))
    tk = f["topk"]
    f["margin_top1_top2"] = float(tk[0]["score"] - tk[1]["score"]) if len(tk) >= 2 else None
    f["dtype"] = str(row.dtype)
    f["sha16"] = A.stable_sha(row)
    return f


def _row_cmp(a, b, name: str) -> dict:
    """行的逐元素比较（**从原始 dtype 张量**做；比较时才转 FP32）。"""
    out = _cmp(a.float(), b.float(), name)
    out["sha16_a"] = A.stable_sha(a)
    out["sha16_b"] = A.stable_sha(b)
    out["dtype_a"] = str(a.dtype)
    out["dtype_b"] = str(b.dtype)
    out["shape"] = [int(x) for x in a.shape]
    out["exact_equal"] = bool(torch.equal(a, b))
    return out


def _classify(per_layer: dict) -> dict:
    """三个边界判定（**纯函数**，CPU 可测；三者严格分开、互不替代）。

    * `earliest_observed_input_difference`：最早的层，其**输入行** S/B 不逐位相同；
    * `earliest_observed_output_difference`：最早的层，其**输出行** S/B 不逐位相同；
    * `first_same_input_different_output`：第一层满足**输入逐位相同、输出不同**
      （= 当前轨迹中最早观测到的"该层自己产生了差异"）。
    """
    idx = sorted(per_layer.keys())
    e_in = next((i for i in idx if not per_layer[i]["input_equal"]), None)
    e_out = next((i for i in idx if not per_layer[i]["output_equal"]), None)
    first_sd = next((i for i in idx
                     if per_layer[i]["input_equal"] and not per_layer[i]["output_equal"]), None)
    return dict(earliest_observed_input_difference=e_in,
                earliest_observed_output_difference=e_out,
                first_same_input_different_output=first_sd,
                l0_input_already_differs=(bool(not per_layer[idx[0]]["input_equal"]) if idx else None))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="layer_chain_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209：S（单步）与 B（块验证）最早在**哪一层**产生隐藏状态决策行差异？",
               semantics="一次真实 S/B 前向内对全部层挂只读 pre/post 钩子，只记录**决策行**（原始 dtype，"
                          "前向后才搬 CPU）；给出三个边界：最早输入差异层 / 最早输出差异层 / 第一层同输入异输出。",
               config=dict(tag=TAG, pre=PRE, target_pos=TARGET_POS, dec_slot=DEC_SLOT,
                           row_index=ROW_INDEX, k=K, feed=FEED, cands=CANDS, watch=list(WATCH),
                           repeat=REPEAT, target_rope=TARGET_ROPE,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           saved_pt=str(SAVED_PT), l27_pt=str(L27_PT),
                           in_scope="只定位到层；不做某层四格、不进入层内算子、不改生产实现、不做 FP32"))
    print(f"[chain] run_id={run_id} pre={PRE} target={TARGET_POS} feed={FEED} cands={CANDS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[chain] commit: {why}")
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
    name = (f"{TAG}_{run_id}_selftest.pt" if selftest else f"{TAG}_{run_id}.pt")
    p = OUTDIR / name
    torch.save(payload, str(p))
    return p


def _capture_pass(tgt, H, dev, run_id_tag: str) -> dict:
    """一次「S + B」捕获 pass：返回两条路径的**决策行**边界与最终 logits 行（全部搬到 CPU）。

    每层只留决策行；`to_cpu_rows` 在**前向结束之后**才调用（不在钩子里 `.cpu()`）。
    """
    out: dict = {}
    for arm_nm, mode_nm in (("S", "decode"), ("B", "verify")):
        cache = _to_dev(H, dev)
        with LB.embedding_capture(tgt, tag=f"{arm_nm}_e") as ec, \
                LB.layer_chain_capture(tgt, row_index=ROW_INDEX, tag=f"{arm_nm}_{run_id_tag}") as ch, \
                LB.record(tgt, tag=f"{arm_nm}_n") as _ni:
            with torch.inference_mode(), M.attention_phase(mode_nm):
                if arm_nm == "S":
                    lg, cache_after = spec_dec._step(tgt, FEED[0], cache)
                    row = LB.row_of(lg, None)
                else:
                    rows_b, cache_after = spec_dec._verify(tgt, cache, CANDS)
                    row = LB.row_of(rows_b, 0)
            ncap = LB.take()
        chain = LB.to_cpu_rows(ch)
        norm = ncap["caps"][1]
        out[arm_nm] = dict(
            chain=chain,
            embed_row=(ec["out"][0, ROW_INDEX].detach().to("cpu") if torch.is_tensor(ec["out"]) else None),
            embed_ids=list(ec["ids"] or []),
            norm_in_row=(norm["x_in"][0, ROW_INDEX].detach().to("cpu")
                         if torch.is_tensor(norm["x_in"]) else None),
            norm_out_row=(norm["y_out"][0, ROW_INDEX].detach().to("cpu")
                          if torch.is_tensor(norm["y_out"]) else None),
            final_row=row.detach().to("cpu"),
            cache_len_after=int(spec_cache.cache_length(cache_after)),
            n_layers=len(chain["indices"]))
    return out


def _run(rec: dict, run_id: str) -> int:
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        dev = next(tgt.parameters()).device
        n_layers = len(tgt.layers)
        rec["precision"] = _set_precision()
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["code_facts"] = LB.code_facts(tgt)
        rec["norm"] = LB.norm_info(tgt)
        rec["chain_shape"] = dict(n_layers=int(n_layers),
                                  layer_classes=sorted({type(l).__name__ for l in tgt.layers}),
                                  last_is_final_norm_input=True,
                                  note="`layers[i]` 的输出就是 `layers[i+1]` 的输入；最后一层输出就是最终 Norm 的输入")
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        DC.require(n_layers >= 2, "层数异常", str(n_layers))
        print(f"[chain] 层数={n_layers}；最后层={n_layers - 1}（其输出=最终 Norm 输入）", flush=True)
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
        H_digest_before = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=src_len,
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H),
                               k_shape=[int(x) for x in H[0]["k"].shape], dtype=str(H[0]["k"].dtype),
                               note="与已保存的四格 / LM head / Norm / L27 四次 run 同一份捕获与同一**纯切片**截断")
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2), compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"],
                       "两份捕获在 [0,PRE) 上并非逐位相同", str(rec["H_identity"]["compare"]))
            del H2
        print(f"[chain] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 捕获 pass 1 / pass 2（重复一致性）----------------
        passes = []
        for it in range(REPEAT):
            passes.append(_capture_pass(tgt, H, dev, f"p{it + 1}"))
            print(f"[chain] 捕获 pass {it + 1}/{REPEAT} 完成（S/B 各 {n_layers} 层）", flush=True)
        P1, P2 = passes[0], passes[1] if len(passes) > 1 else None
        rec["repeat_consistency"] = dict(
            n_passes=len(passes),
            S_all_rows_identical=bool(P2 is not None and all(
                torch.equal(P1["S"]["chain"]["layers"][i]["pre"]["a_row"],
                            P2["S"]["chain"]["layers"][i]["pre"]["a_row"])
                and torch.equal(P1["S"]["chain"]["layers"][i]["post"]["z_row"],
                                P2["S"]["chain"]["layers"][i]["post"]["z_row"])
                for i in P1["S"]["chain"]["layers"])),
            B_all_rows_identical=bool(P2 is not None and all(
                torch.equal(P1["B"]["chain"]["layers"][i]["pre"]["a_row"],
                            P2["B"]["chain"]["layers"][i]["pre"]["a_row"])
                and torch.equal(P1["B"]["chain"]["layers"][i]["post"]["z_row"],
                                P2["B"]["chain"]["layers"][i]["post"]["z_row"])
                for i in P1["B"]["chain"]["layers"])),
            embed_identical=bool(P2 is not None and torch.equal(P1["S"]["embed_row"], P2["S"]["embed_row"])
                                 and torch.equal(P1["B"]["embed_row"], P2["B"]["embed_row"])),
            norm_rows_identical=bool(P2 is not None and all(
                torch.equal(P1[a]["norm_in_row"], P2[a]["norm_in_row"])
                and torch.equal(P1[a]["norm_out_row"], P2[a]["norm_out_row"]) for a in ("S", "B"))),
            final_rows_identical=bool(P2 is not None and torch.equal(P1["S"]["final_row"], P2["S"]["final_row"])
                                      and torch.equal(P1["B"]["final_row"], P2["B"]["final_row"])),
            note="重复两次捕获（各自从独立克隆开始）的所有决策行必须逐位相同")
        DC.require(all(v is True for k, v in rec["repeat_consistency"].items()
                       if k.endswith("identical")), "重复捕获不一致", str(rec["repeat_consistency"]))
        print("[chain] 重复一致 ✔（两次捕获的全部层决策行逐位相同）", flush=True)

        # ---------------- 控制：真实行、锚点、捕获关闭 ----------------
        S_c, B_c = P1["S"], P1["B"]
        row_s, row_b = S_c["final_row"], B_c["final_row"]
        cache_lens = dict(S=S_c["cache_len_after"], B=B_c["cache_len_after"])
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        rec["step1_control"] = dict(
            cache_lens_after=cache_lens,
            both_predict_target=bool(cache_lens["S"] == TARGET_POS
                                     and cache_lens["B"] == PRE + len(CANDS)),
            S=_fields(row_s), B=_fields(row_b),
            vs_saved_local_rows=dict(
                path=str(SAVED_PT), present=bool(saved_rows),
                S=(_row_cmp(row_s, saved_rows["H_AR-S"], "S_vs_saved") if "H_AR-S" in saved_rows else None),
                B=(_row_cmp(row_b, saved_rows["H_AR-B"], "B_vs_saved") if "H_AR-B" in saved_rows else None),
                note="与**已保存的局部对照行**逐元素比较（不是与未捕获的自由生成整行比）"))
        DC.require(rec["step1_control"]["both_predict_target"], "S/B 缓存长度不符合预期",
                   str(cache_lens))
        if saved_rows:
            for k in ("S", "B"):
                DC.require(rec["step1_control"]["vs_saved_local_rows"][k]["exact_equal"],
                           f"控制失败：{k} 行与已保存局部对照行不同",
                           str(rec["step1_control"]["vs_saved_local_rows"][k]))
        s_f, b_f = _fields(row_s), _fields(row_b)
        print(f"[chain] 控制：S argmax={s_f['argmax_id']}（margin={s_f['margin_top1_top2']}，并列={s_f['n_max_tied']}） "
              f"B argmax={b_f['argmax_id']}（margin={b_f['margin_top1_top2']}）", flush=True)
        anch_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anch_b = C._anchor_specdec(REF_RUN_RB, PRE, DEC_SLOT, TARGET_POS)
        rec["anchors"] = dict(S=anch_s, B=anch_b, note="真实运行记录里的 token 选择（**argmax 级**）")
        rec["step1_control"]["token_matches_anchors"] = dict(
            S=dict(measured=s_f["argmax_id"], anchor=anch_s.get("token"),
                   match=bool(s_f["argmax_id"] == anch_s.get("token"))),
            B=dict(measured=b_f["argmax_id"], anchor=anch_b.get("token"),
                   match=bool(b_f["argmax_id"] == anch_b.get("token"))))
        DC.require(all(v["match"] for v in rec["step1_control"]["token_matches_anchors"].values()),
                   "控制失败：S/B argmax 与锚点不一致（停止，不做层级解释）",
                   str(rec["step1_control"]["token_matches_anchors"]))
        if CAPTURE_OFF_CHECK:
            off = LB.install(tgt, enabled=False, tag="off")
            with torch.inference_mode(), M.attention_phase("decode"):
                lg2, _c2 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows2, _c3 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            rec["capture_off_check"] = dict(
                installed=off["installed"],
                S_row_exact_equal=bool(torch.equal(row_s, LB.row_of(lg2, None).detach().cpu())),
                B_row_exact_equal=bool(torch.equal(row_b, LB.row_of(rows2, 0).detach().cpu())),
                note="不挂任何钩子（含链式钩子）时 S/B 判定行与带捕获时**逐位相同**")
            LB.assert_clean()
            DC.require(rec["capture_off_check"]["S_row_exact_equal"]
                       and rec["capture_off_check"]["B_row_exact_equal"],
                       "捕获改变了真实输出（控制失败）", str(rec["capture_off_check"]))
            print("[chain] 捕获关闭等价 ✔（S/B 行逐位相同）", flush=True)

        # ---------------- 边界链自洽 ----------------
        chain_checks = dict(
            embed_row_S_vs_B=_row_cmp(S_c["embed_row"], B_c["embed_row"], "embedS_vs_embedB"),
            L0_input_equals_embed_S=_row_cmp(S_c["chain"]["layers"][0]["pre"]["a_row"], S_c["embed_row"],
                                             "L0in_S_vs_embedS"),
            L0_input_equals_embed_B=_row_cmp(B_c["chain"]["layers"][0]["pre"]["a_row"], B_c["embed_row"],
                                             "L0in_B_vs_embedB"),
            last_layer_output_equals_norm_in_S=_row_cmp(
                S_c["chain"]["layers"][n_layers - 1]["post"]["z_row"], S_c["norm_in_row"],
                f"L{n_layers - 1}out_S_vs_normIn_S"),
            last_layer_output_equals_norm_in_B=_row_cmp(
                B_c["chain"]["layers"][n_layers - 1]["post"]["z_row"], B_c["norm_in_row"],
                f"L{n_layers - 1}out_B_vs_normIn_B"),
            inter_layer_chain_S={i: _row_cmp(S_c["chain"]["layers"][i]["post"]["z_row"],
                                             S_c["chain"]["layers"][i + 1]["pre"]["a_row"],
                                             f"S_L{i}out_vs_L{i + 1}in")
                                 for i in range(n_layers - 1)},
            inter_layer_chain_B={i: _row_cmp(B_c["chain"]["layers"][i]["post"]["z_row"],
                                             B_c["chain"]["layers"][i + 1]["pre"]["a_row"],
                                             f"B_L{i}out_vs_L{i + 1}in")
                                 for i in range(n_layers - 1)})
        rec["chain_checks"] = dict(
            embed_row_S_vs_B=chain_checks["embed_row_S_vs_B"],
            L0_input_equals_embed_S=chain_checks["L0_input_equals_embed_S"],
            L0_input_equals_embed_B=chain_checks["L0_input_equals_embed_B"],
            last_layer_output_equals_norm_in_S=chain_checks["last_layer_output_equals_norm_in_S"],
            last_layer_output_equals_norm_in_B=chain_checks["last_layer_output_equals_norm_in_B"],
            inter_layer_all_equal_S=bool(all(v["exact_equal"] for v in chain_checks["inter_layer_chain_S"].values())),
            inter_layer_all_equal_B=bool(all(v["exact_equal"] for v in chain_checks["inter_layer_chain_B"].values())),
            inter_layer_first_bad_S=next((i for i, v in chain_checks["inter_layer_chain_S"].items()
                                          if not v["exact_equal"]), None),
            inter_layer_first_bad_B=next((i for i, v in chain_checks["inter_layer_chain_B"].items()
                                          if not v["exact_equal"]), None),
            note="L0 输入 == embedding 决策行；layer[i] 输出行 == layer[i+1] 输入行（两条路径各自核）；"
                 "最后一层输出行 == 最终 Norm 输入行")
        cc = rec["chain_checks"]
        for k in ("embed_row_S_vs_B", "L0_input_equals_embed_S", "L0_input_equals_embed_B",
                  "last_layer_output_equals_norm_in_S", "last_layer_output_equals_norm_in_B"):
            DC.require(cc[k]["exact_equal"], f"边界链控制失败：{k}", str(cc[k]))
        DC.require(cc["inter_layer_all_equal_S"] and cc["inter_layer_all_equal_B"],
                   "层间边界链不逐位自洽", f"S={cc['inter_layer_first_bad_S']} B={cc['inter_layer_first_bad_B']}")
        print("[chain] 边界链 ✔ embedding→L0→…→L27→Norm 的行逐位自洽（两条路径各自核）", flush=True)

        # ---------------- 每层比较 ----------------
        per_layer = {}
        for i in range(n_layers):
            a_s = S_c["chain"]["layers"][i]["pre"]
            a_b = B_c["chain"]["layers"][i]["pre"]
            z_s = S_c["chain"]["layers"][i]["post"]
            z_b = B_c["chain"]["layers"][i]["post"]
            per_layer[i] = dict(
                layer=i,
                s=dict(S=a_s["s"], B=a_b["s"]),
                phase=dict(S=a_s["phase"], B=a_b["phase"]),
                cache_len=dict(S_before=a_s["cache_len_before"], S_after=z_s["cache_len_after"],
                               B_before=a_b["cache_len_before"], B_after=z_b["cache_len_after"]),
                input=_row_cmp(a_s["a_row"], a_b["a_row"], f"L{i}_input_S_vs_B"),
                output=_row_cmp(z_s["z_row"], z_b["z_row"], f"L{i}_output_S_vs_B"),
                cos_row_equal=bool(torch.equal(a_s["cos_row"], a_b["cos_row"])
                                   if torch.is_tensor(a_s["cos_row"]) else None),
                sin_row_equal=bool(torch.equal(a_s["sin_row"], a_b["sin_row"])
                                   if torch.is_tensor(a_s["sin_row"]) else None))
            per_layer[i]["input_equal"] = bool(per_layer[i]["input"]["exact_equal"])
            per_layer[i]["output_equal"] = bool(per_layer[i]["output"]["exact_equal"])
        rec["per_layer"] = per_layer
        print("== 每层决策行 S vs B ==")
        print("  layer | input n_diff/max|Δ|      | output n_diff/max|Δ|     | in_eq out_eq")
        for i in range(n_layers):
            d = per_layer[i]
            print(f"  L{i:<4d} | {d['input']['n_diff']:>5d} / {d['input']['abs_err_max']:<8.6g} "
                  f"| {d['output']['n_diff']:>5d} / {d['output']['abs_err_max']:<8.6g} "
                  f"| {str(d['input_equal']):<5s} {str(d['output_equal']):<5s}", flush=True)

        # ---------------- 三个边界判定（严格分开）----------------
        cls = _classify(per_layer)
        earliest_in = cls["earliest_observed_input_difference"]
        earliest_out = cls["earliest_observed_output_difference"]
        first_same_in_diff_out = cls["first_same_input_different_output"]
        rec["boundaries"] = dict(
            definitions=dict(
                earliest_observed_input_difference="最早的层，其**输入行** S/B 不逐位相同",
                earliest_observed_output_difference="最早的层，其**输出行** S/B 不逐位相同",
                first_same_input_different_output="第一层满足**输入行逐位相同、输出行不同**"
                                                  "（= 该层是当前轨迹中最早观测到的分歧产生层）"),
            earliest_observed_input_difference=earliest_in,
            earliest_observed_output_difference=earliest_out,
            first_same_input_different_output=first_same_in_diff_out,
            l0_input_already_differs=cls["l0_input_already_differs"],
            verdict_note=("若 L0 输入已经不同 ⇒ 分歧在 **DecoderLayer 之前**，停止层内归因；"
                          "若某层输入已经不同 ⇒ **不能**把该层称为首次来源；"
                          "**不**假定差异随层数单调增长、**不**用二分排除未直接测量的层；"
                          "后续层重新逐位相同也**如实保留全部记录**"))
        b = rec["boundaries"]
        print(f"[chain] 三个边界：最早输入差异层={b['earliest_observed_input_difference']} "
              f"最早输出差异层={b['earliest_observed_output_difference']} "
              f"第一层同输入异输出={b['first_same_input_different_output']}"
              f"（L0 输入已不同={b['l0_input_already_differs']}）", flush=True)
        rec["final_rows"] = dict(
            S_output=_row_cmp(S_c["norm_out_row"], B_c["norm_out_row"], "normOut_S_vs_B"),
            S_final_logits=_row_cmp(row_s, row_b, "finalLogits_S_vs_B"),
            note="最终 Norm 输出行与最终 logits 行的 S/B 比较（与层级结论分开记录）")

        # ---------------- 源完整性 ----------------
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        rec["source_integrity"] = dict(
            H_snapshot_unchanged=bool(H_digest_before == H_digest_after),
            H_len=spec_cache.cache_length(H),
            W_after=LB.weights_info(tgt),
            W_identity_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"]
                                   == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"]
                                 == rec["W_before"]["digest_subsample"]),
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"]
                                    == rec["norm"]["weight"]["data_ptr"]),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers))
                               and len(tgt.norm._forward_hooks) == 0
                               and len(tgt.embed_tokens._forward_hooks) == 0),
            note="H 快照（逐层 k/v 摘要）与两份权重在全程不变；每层缓存按设计**只追加** "
                 f"（S 8208→{cache_lens['S']}、B 8208→{cache_lens['B']}）——这是引擎的正常行为，不是违规")
        DC.require(rec["source_integrity"]["H_snapshot_unchanged"]
                   and rec["source_integrity"]["W_identity_stable"]
                   and rec["source_integrity"]["W_digest_stable"]
                   and rec["source_integrity"]["norm_weight_stable"]
                   and rec["source_integrity"]["hooks_removed"],
                   "源张量/权重被改动或钩子未移除", str(rec["source_integrity"]))

        # ---------------- 跨工具核对：与上一轮 L27 的单层捕获逐位一致 ----------------
        rec["cross_run_l27"] = dict(path=str(L27_PT), present=Path(L27_PT).exists())
        if Path(L27_PT).exists():
            prv = torch.load(L27_PT, map_location="cpu")
            last = n_layers - 1
            rec["cross_run_l27"].update(dict(
                L27_input_row_S=(_row_cmp(S_c["chain"]["layers"][last]["pre"]["a_row"],
                                          prv["aS_raw"][0, 0], "chainL27in_S_vs_l27run")
                                 if "aS_raw" in prv else None),
                L27_input_row_B=(_row_cmp(B_c["chain"]["layers"][last]["pre"]["a_row"],
                                          prv["aB4_raw"][0, 0], "chainL27in_B_vs_l27run")
                                 if "aB4_raw" in prv else None),
                L27_output_row_S=(_row_cmp(S_c["chain"]["layers"][last]["post"]["z_row"],
                                           prv["zS_raw"][0, 0], "chainL27out_S_vs_l27run")
                                  if "zS_raw" in prv else None),
                L27_output_row_B=(_row_cmp(B_c["chain"]["layers"][last]["post"]["z_row"],
                                           prv["zB4_raw"][0, 0], "chainL27out_B_vs_l27run")
                                  if "zB4_raw" in prv else None),
                note="本轮的最后一层决策行必须与上一轮单层捕获 run 的 aS/aB4/zS/zB4 第 0 行**逐位相同**"
                     "（同 H、同 W、同配置）"))
            for k in ("L27_input_row_S", "L27_input_row_B", "L27_output_row_S", "L27_output_row_B"):
                v = rec["cross_run_l27"].get(k)
                if isinstance(v, dict):
                    DC.require(v["exact_equal"], f"跨工具核对失败：{k}", str(v))
            print("[chain] 跨工具核对 ✔ 最后一层决策行 == 上一轮 L27 run 的 aS/aB4/zS/zB4 第 0 行（逐位）",
                  flush=True)

        att = M.attention_stats()
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att.get("s_max_by_phase"))
        rec["capture_clean"] = LB.assert_clean()
        rec["stage"] = "complete"
        payload = dict(
            rows_S_raw={i: dict(input=S_c["chain"]["layers"][i]["pre"]["a_row"],
                                output=S_c["chain"]["layers"][i]["post"]["z_row"])
                        for i in range(n_layers)},
            rows_B_raw={i: dict(input=B_c["chain"]["layers"][i]["pre"]["a_row"],
                                output=B_c["chain"]["layers"][i]["post"]["z_row"])
                        for i in range(n_layers)},
            embed_row_S=S_c["embed_row"], embed_row_B=B_c["embed_row"],
            norm_in_row_S=S_c["norm_in_row"], norm_in_row_B=B_c["norm_in_row"],
            norm_out_row_S=S_c["norm_out_row"], norm_out_row_B=B_c["norm_out_row"],
            final_row_S=row_s, final_row_B=row_b,
            pass2_rows_S_raw=({i: dict(input=P2["S"]["chain"]["layers"][i]["pre"]["a_row"],
                                       output=P2["S"]["chain"]["layers"][i]["post"]["z_row"])
                               for i in range(n_layers)} if P2 else None),
            pass2_rows_B_raw=({i: dict(input=P2["B"]["chain"]["layers"][i]["pre"]["a_row"],
                                       output=P2["B"]["chain"]["layers"][i]["post"]["z_row"])
                               for i in range(n_layers)} if P2 else None),
            meta=dict(per_layer_meta={i: dict(s=per_layer[i]["s"], phase=per_layer[i]["phase"],
                                              cache_len=per_layer[i]["cache_len"])
                                      for i in range(n_layers)},
                      row_index=ROW_INDEX, n_layers=n_layers),
            final_row_S_f32=row_s.float(), final_row_B_f32=row_b.float())
        tp = _save_pt(payload, run_id)
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp),
                                raw_dtype_note="每层决策行以**原始 bf16** 落盘；`*_f32` 仅供统计/外部比较")
        _write(rec, run_id)
        print(f"[chain] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[chain] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[chain] RESULT_JSON={p}")
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
    """**不加载大模型**：微型模型上把链式捕获 + 三个边界判定 + 控制 + 落盘流程走一遍。"""
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
                   num_layers=3, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    n_layers = len(tgt.layers)
    ids = torch.randint(0, 64, (1, PRE))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    dev = torch.device("cpu")
    P1 = _capture_pass(tgt, H, dev, "selftest1")
    P2 = _capture_pass(tgt, H, dev, "selftest2")
    check("★ 每层都捕获到输入行/输出行（原始 bf16、已搬 CPU）",
          all(len(P1[a]["chain"]["layers"]) == n_layers for a in ("S", "B"))
          and all(tuple(P1[a]["chain"]["layers"][i]["pre"]["a_row"].shape) == (32,)
                  and str(P1[a]["chain"]["layers"][i]["pre"]["a_row"].dtype) == "torch.bfloat16"
                  for a in ("S", "B") for i in range(n_layers)))
    check("★ 边界链自洽：L0 输入 == embedding 行；layer[i] 输出 == layer[i+1] 输入；最后一层输出 == Norm 输入",
          all(bool(torch.equal(P1[a]["chain"]["layers"][0]["pre"]["a_row"], P1[a]["embed_row"]))
              and bool(torch.equal(P1[a]["chain"]["layers"][n_layers - 1]["post"]["z_row"],
                                   P1[a]["norm_in_row"]))
              and all(bool(torch.equal(P1[a]["chain"]["layers"][i]["post"]["z_row"],
                                       P1[a]["chain"]["layers"][i + 1]["pre"]["a_row"]))
                      for i in range(n_layers - 1)) for a in ("S", "B")))
    check("★ 两次捕获逐位一致（全部层行 + embedding + Norm 行）",
          all(bool(torch.equal(P1[a]["chain"]["layers"][i]["pre"]["a_row"],
                               P2[a]["chain"]["layers"][i]["pre"]["a_row"]))
              and bool(torch.equal(P1[a]["chain"]["layers"][i]["post"]["z_row"],
                                   P2[a]["chain"]["layers"][i]["post"]["z_row"]))
              for a in ("S", "B") for i in range(n_layers)))
    check("★ 每层元数据齐全（s / phase / cache 长度）",
          all(all(k in P1[a]["chain"]["layers"][i]["pre"] for k in ("s", "phase", "cache_len_before"))
              and all(k in P1[a]["chain"]["layers"][i]["post"] for k in ("cache_len_after",))
              for a in ("S", "B") for i in range(n_layers))
          and P1["S"]["chain"]["layers"][0]["pre"]["s"] == 1
          and P1["B"]["chain"]["layers"][0]["pre"]["s"] == 4
          and P1["S"]["chain"]["layers"][0]["post"]["cache_len_after"] == PRE + 1
          and P1["B"]["chain"]["layers"][0]["post"]["cache_len_after"] == PRE + 4)
    # 三个边界判定机制（① 真实数据；② **合成数据**确保"找到差异"的分支也被覆盖）
    pl = {}
    for i in range(n_layers):
        a_s, a_b = P1["S"]["chain"]["layers"][i]["pre"], P1["B"]["chain"]["layers"][i]["pre"]
        z_s, z_b = P1["S"]["chain"]["layers"][i]["post"], P1["B"]["chain"]["layers"][i]["post"]
        pl[i] = dict(input_equal=bool(torch.equal(a_s["a_row"], a_b["a_row"])),
                     output_equal=bool(torch.equal(z_s["z_row"], z_b["z_row"])))
    real_cls = _classify(pl)
    syn = {0: dict(input_equal=True, output_equal=True),
           1: dict(input_equal=True, output_equal=False),      # ← 同输入异输出
           2: dict(input_equal=False, output_equal=False)}
    syn_cls = _classify(syn)
    syn2 = {0: dict(input_equal=False, output_equal=False), 1: dict(input_equal=True, output_equal=True)}
    syn2_cls = _classify(syn2)
    check("★ 三边界分类：合成数据下分别得到 输入差异=2 / 输出差异=1 / 同输入异输出=1",
          syn_cls["earliest_observed_input_difference"] == 2
          and syn_cls["earliest_observed_output_difference"] == 1
          and syn_cls["first_same_input_different_output"] == 1
          and syn_cls["l0_input_already_differs"] is False, syn_cls)
    check("★ L0 输入已不同 ⇒ 单独标出（分歧在 DecoderLayer 之前）",
          syn2_cls["l0_input_already_differs"] is True
          and syn2_cls["earliest_observed_input_difference"] == 0, syn2_cls)
    check("★ 三边界三者**互不替代**（None 与具体层号都能如实返回）",
          real_cls["earliest_observed_input_difference"] is None
          and real_cls["earliest_observed_output_difference"] is None
          and real_cls["first_same_input_different_output"] is None, real_cls)
    print(f"  [obs] 微型模型（{n_layers} 层，S=s1/B=s4）：最早输入差异层="
          f"{real_cls['earliest_observed_input_difference']} "
          f"最早输出差异层={real_cls['earliest_observed_output_difference']} "
          f"第一层同输入异输出={real_cls['first_same_input_different_output']}")
    _off = LB.install(tgt, enabled=False, tag="off")
    with torch.inference_mode(), M.attention_phase("decode"):
        lg3, _c3 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    LB.assert_clean()
    check("★ 关闭捕获时与带捕获逐位相同（真实模型侧的关闭等价）",
          _off["installed"] is False
          and bool(torch.equal(P1["S"]["final_row"].to("cpu"),
                               LB.row_of(lg3, None).detach().cpu()))
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(n_layers))
          and len(tgt.norm._forward_hooks) == 0 and len(tgt.embed_tokens._forward_hooks) == 0)
    tp = _save_pt(dict(rows_S_raw={i: dict(input=P1["S"]["chain"]["layers"][i]["pre"]["a_row"],
                                           output=P1["S"]["chain"]["layers"][i]["post"]["z_row"])
                                   for i in range(n_layers)},
                       rows_B_raw={i: dict(input=P1["B"]["chain"]["layers"][i]["pre"]["a_row"],
                                           output=P1["B"]["chain"]["layers"][i]["post"]["z_row"])
                                   for i in range(n_layers)},
                       embed_row_S=P1["S"]["embed_row"], embed_row_B=P1["B"]["embed_row"]),
                  run_id, selftest=True)
    check("★ 落盘流程可跑通（原始 dtype 行写入 .pt）", tp.exists() and tp.stat().st_size > 0, str(tp))
    print(f"\n全层边界捕获（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：只验证机制（链式捕获/边界链/重复/元数据/三边界/落盘）",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
