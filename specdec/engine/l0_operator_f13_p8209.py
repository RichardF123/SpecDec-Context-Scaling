"""l0_operator_f13_p8209.py — 位置 8209 的 **L0 内部算子边界**定位（F13；只读、有界、单次）。

## 唯一目标
沿 L0（`layers[0]`，零基）的**真实生产数据流**捕获 S/B 的对应决策数据，找出**第一个**满足
「该算子的**所有**输入逐位相同、而输出首次不同」的算子边界。

上一轮已证明：`L0 输入 S/B 逐位相同`（`sha16 87bdb3aa1faf5f3c`）、`L0 输出开始不同`（`n_diff=111`）。
本轮**不再**重复"输入来源四格"（L0 入口处输入来源轴已塌缩：`aS == aB[row0]`），
而是直接进入 L0 内部，**只定位算子、不修复、不改生产实现**。

## 代码顺序 → 探针名映射（以 `model_self.py` 的真实代码为准）
`DecoderLayer.forward`：
`x0` → `input_rmsnorm` → [`SelfAttention`] → `attn_out`(=`o_proj` 输出) → `attention_residual`
→ `post_attention_rmsnorm` → [`MLP`] → `mlp_out`(=`down_proj`) → `layer_output`
`SelfAttention.forward`：
`q_linear/k_linear/v_linear` → `q_reshaped/k_reshaped/v_reshaped` → `q_rope/k_rope`
→ `k_visible/v_visible`（与轮前缓存 cat）→ `qk_scores_pre_mask`（QKᵀ×缩放）→ `qk_scores_post_mask`
→ `attention_prob`（softmax，**fp32 保持原 dtype**）→ `attention_times_v` → `heads_merged` → `o_proj`
`MLP.forward`：`mlp_gate_proj` → `mlp_activation`(silu) → `mlp_up_proj` → `mlp_gated_product` → `mlp_down_proj`

## 纪律
生产公式**不改**（只加了默认 `None` 的探针点，关闭时逐位等价，由 CPU 门与真机开启/关闭控制证明）；
不改 dtype/精度/mask/RoPE/attention/cache/argmax/后端；不做全模型 FP32；不自由生成；不加载 draft；
GPU 串行；硬失败落盘停止、无自动重试；旧产物不覆盖；**B6 状态不变**。
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
from . import op_probe as OP
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("L0_TAG", "div8209_l0ops")
MODE = os.environ.get("L0_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("L0_LAYER", "0"))
PRE = int(os.environ.get("L0_PRE", "8208"))
TARGET_POS = int(os.environ.get("L0_TARGET", "8209"))
K = int(os.environ.get("L0_K", "4"))
FEED = [int(x) for x in os.environ.get("L0_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("L0_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("L0_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("L0_REPEAT", "2"))
TARGET_ROPE = os.environ.get("L0_TARGET_ROPE", "cpu_fp32").strip().lower()
CAPTURE_OFF_CHECK = S.env_flag("L0_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("L0_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("L0_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
SAVED_PT = os.environ.get("L0_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
CHAIN_PT = os.environ.get("L0_CHAIN_PT", str(OUTDIR / "div8209_chain_20261002T164558Z.pt"))
REF_RUN_AR = os.environ.get("L0_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("L0_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("L0_ARM", "eager")
VIS = PRE + 1                                # 共同可见 key 数 = 8209（key 位置 0..8208）


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


def _fields(row) -> dict:
    f = DF.row_fields(row, watch=list(WATCH))
    tk = f["topk"]
    f["margin_top1_top2"] = float(tk[0]["score"] - tk[1]["score"]) if len(tk) >= 2 else None
    f["dtype"] = str(row.dtype)
    f["sha16"] = A.stable_sha(row)
    return f


def _row_cmp(a, b, name: str) -> dict:
    a2, b2 = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(a2, b2, name)
    d = (a2 - b2).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b), dtype=str(a.dtype),
                shape=[int(x) for x in a.shape])


def _capture_arm(tgt, H, dev, arm: str, mode: str, tag: str) -> dict:
    """一次「单臂捕获」：L0 层 pre-hook（x0/cos/sin/轮前缓存）+ 算子探针 + 真实前向。"""
    cache = _to_dev(H, dev)
    with LB.layer_capture(tgt, LAYER, tag=f"{arm}_L0_{tag}") as lc, \
            OP.capture(tgt, LAYER, arm) as cap:
        with torch.inference_mode(), M.attention_phase(mode):
            if arm == "S":
                lg, cache_after = spec_dec._step(tgt, FEED[0], cache)
                row = LB.row_of(lg, None)
            else:
                rows_b, cache_after = spec_dec._verify(tgt, cache, CANDS)
                row = LB.row_of(rows_b, 0)
        probes = cap.close()
    return dict(layer_cap=lc, probes=probes, final_row=row.detach().to("cpu"),
                cache_len_after=int(spec_cache.cache_length(cache_after)),
                x0_row=lc["pre"]["a"][0, -1].detach().to("cpu") if False else None,
                x0=lc["pre"]["a"], cos=lc["pre"]["cos"], sin=lc["pre"]["sin"],
                cache_pre=lc["pre"]["cache_pre"], cache_len_before=lc["pre"]["cache_len_before"])


def _layer_weight_ptrs(tgt, layer_index: int) -> dict:
    """L0 七个权重张量的 `data_ptr`（用于证明两条路径用同一批权重、且诊断未替换权重）。"""
    sa = tgt.layers[layer_index].self_attn
    mlp = tgt.layers[layer_index].mlp
    return dict(q_proj_data_ptr=int(sa.q_proj.weight.data_ptr()),
                k_proj_data_ptr=int(sa.k_proj.weight.data_ptr()),
                v_proj_data_ptr=int(sa.v_proj.weight.data_ptr()),
                o_proj_data_ptr=int(sa.o_proj.weight.data_ptr()),
                gate_data_ptr=int(mlp.gate_proj.weight.data_ptr()),
                up_data_ptr=int(mlp.up_proj.weight.data_ptr()),
                down_data_ptr=int(mlp.down_proj.weight.data_ptr()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="l0_operator_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209：L0 内部**第一个**「所有输入逐位相同、输出首次不同」的算子边界是哪一个？",
               semantics="只读探针捕获 L0 的真实中间张量（Q/K/V 线/reshape/RoPE/KV 拼接/QKᵀ/mask/softmax/AV/"
                          "head merge/o_proj/残差/MLP 各步），按语义切片比较 S/B 的**对应决策数据**，"
                          "判定首个 shape-sensitive 分歧边界。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, watch=list(WATCH), repeat=REPEAT, vis=VIS,
                           target_rope=TARGET_ROPE, capture_off_check=CAPTURE_OFF_CHECK,
                           cap_h=str(CAP_H), cap_h2=str(CAP_H2), saved_pt=str(SAVED_PT),
                           chain_pt=str(CHAIN_PT), ref_ar=str(REF_RUN_AR), ref_rb=str(REF_RUN_RB),
                           in_scope="只定位算子；不修复、不改生产实现、不改精度/后端、不做 FP32"))
    print(f"[l0op] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} vis={VIS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[l0op] commit: {why}")
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


def _probe_payload(probes: dict) -> dict:
    """探针记录 → 可落盘 payload（原始 dtype）+ 比较切片。"""
    raw, slices = {}, {}
    for name, v in probes["records"].items():
        kind = OP.SPECS[name][1] if name in OP.SPECS else None
        if v.get("shape_only"):
            continue
        raw[name] = v["tensor"]
    return raw, slices


def _build_payload(S_p: dict, B_p: dict, *, row_s, row_b, x0_s, x0_b, z_s, z_b) -> dict:
    """构造落盘 payload（`_run` 与 CPU 自测**共用同一代码路径**，避免"自测没覆盖到 payload"）。

    ⚠️ 首跑/次跑/三跑的教训：这一段的**索引与字段口径**必须由 CPU 门覆盖
    （`raw_*` 的值是**张量**，不是记录字典；`shape` 要自己取；`exact_equal` 是 bool 不能用 `:<s`）。
    """
    raw_s = {k: v["tensor"] for k, v in S_p["probes"]["records"].items() if not v.get("shape_only")}
    raw_b = {k: v["tensor"] for k, v in B_p["probes"]["records"].items() if not v.get("shape_only")}
    slices = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        a, b, _r = OP.slice_pair(kind, S_p["probes"]["records"][name]["tensor"],
                                 B_p["probes"]["records"][name]["tensor"], vis=VIS)
        slices[f"{name}_S"] = a
        slices[f"{name}_B"] = b
    return dict(
        raw_S={k: v.detach().cpu() for k, v in raw_s.items()},
        raw_B={k: v.detach().cpu() for k, v in raw_b.items()},
        slices={k: v.detach().cpu() for k, v in slices.items()},
        cache_pre_S=dict(k=S_p["cache_pre"]["k"].detach().cpu(), v=S_p["cache_pre"]["v"].detach().cpu()),
        cache_pre_B=dict(k=B_p["cache_pre"]["k"].detach().cpu(), v=B_p["cache_pre"]["v"].detach().cpu()),
        x0_row_S=x0_s.detach().cpu(), x0_row_B=x0_b.detach().cpu(),
        L0_output_row_S=z_s.detach().cpu(), L0_output_row_B=z_b.detach().cpu(),
        final_row_S=row_s.detach().cpu(), final_row_B=row_b.detach().cpu(),
        cos_row_S=S_p["cos"][0, 0, -1].detach().cpu(), cos_row_B=B_p["cos"][0, 0, 0].detach().cpu(),
        sin_row_S=S_p["sin"][0, 0, -1].detach().cpu(), sin_row_B=B_p["sin"][0, 0, 0].detach().cpu(),
        meta=dict(vis=VIS, layer=LAYER, n_raw_S=len(raw_s), n_raw_B=len(raw_b),
                  shapes={k: [int(x) for x in v.shape] for k, v in raw_s.items()},
                  shapes_B={k: [int(x) for x in v.shape] for k, v in raw_b.items()}),
        note="`raw_*` = 探针捕获的完整张量（原始 dtype）；`slices` = 实际比较的语义切片；"
             "fp32 的 softmax 概率保持 fp32，未强制转 bf16")


def _cmp_all(S_p: dict, B_p: dict) -> dict:
    """按口径比较**全部**可比较节点（只比语义对应的决策数据）。"""
    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        ts = S_p["probes"]["records"].get(name)
        tb = B_p["probes"]["records"].get(name)
        DC.require(ts is not None and tb is not None, f"探针缺少节点：{name}",
                   f"S={ts is not None} B={tb is not None}")
        out[name] = OP.cmp_node(name, kind, ts["tensor"], tb["tensor"], vis=VIS)
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
        rec["layer"] = LB.layer_info(tgt, LAYER)
        rec["probe_state_before"] = M.attn_probe_state()
        DC.require(rec["probe_state_before"]["active"] is False, "进入正式流程前探针必须是关闭的",
                   str(rec["probe_state_before"]))
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        DC.require(n_layers >= 2 and LAYER == 0, "本轮只做 L0", f"n_layers={n_layers} LAYER={LAYER}")
        print(f"[l0op] 层 {LAYER}/{n_layers - 1}；Norm eps={rec['norm']['eps']}；"
              f"探针状态={rec['probe_state_before']}", flush=True)
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
                               L0_k_sha16=A.stable_sha(H[LAYER]["k"]),
                               L0_v_sha16=A.stable_sha(H[LAYER]["v"]),
                               note="与已保存的六次 run 同一份捕获与同一**纯切片**截断")
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2), compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"], "两份捕获并非逐位相同",
                       str(rec["H_identity"]["compare"]))
            del H2
        print(f"[l0op] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 捕获 pass 1 / pass 2 ----------------
        passes = []
        for it in range(REPEAT):
            p = dict(S=_capture_arm(tgt, H, dev, "S", "decode", f"p{it + 1}"),
                     B=_capture_arm(tgt, H, dev, "B", "verify", f"p{it + 1}"))
            passes.append(p)
            print(f"[l0op] 捕获 pass {it + 1}/{REPEAT} 完成"
                  f"（探针调用 S={p['S']['probes']['n_calls']} B={p['B']['probes']['n_calls']}；"
                  f"保留 S={p['S']['probes']['n_kept']} B={p['B']['probes']['n_kept']}）", flush=True)
        P1, P2 = passes[0], (passes[1] if len(passes) > 1 else None)
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        rep = dict(n_passes=len(passes),
                   all_nodes_identical=bool(P2 is not None and all(
                       torch.equal(P1[a]["probes"]["records"][k]["tensor"],
                                   P2[a]["probes"]["records"][k]["tensor"])
                       for a in ("S", "B") for k, v in P1[a]["probes"]["records"].items()
                       if not v.get("shape_only"))),
                   final_rows_identical=bool(P2 is not None
                                             and torch.equal(P1["S"]["final_row"], P2["S"]["final_row"])
                                             and torch.equal(P1["B"]["final_row"], P2["B"]["final_row"])),
                   shape_only_layouts_identical=bool(P2 is not None and all(
                       P1[a]["probes"]["records"][k]["layout"] == P2[a]["probes"]["records"][k]["layout"]
                       for a in ("S", "B") for k, v in P1[a]["probes"]["records"].items()
                       if v.get("shape_only"))),
                   note="两次独立捕获（各自独立克隆）的全部节点张量必须逐位相同")
        rec["repeat_consistency"] = rep
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "两次捕获不一致", str(rep))
        print("[l0op] 重复一致 ✔（两次捕获的全部中间张量逐位相同）", flush=True)

        # ---------------- 关闭/空探针控制 ----------------
        S, B = P1["S"], P1["B"]
        row_s, row_b = S["final_row"], B["final_row"]
        cache_lens = dict(S=S["cache_len_after"], B=B["cache_len_after"])
        off_checks: dict = dict(probe_state_after_off=None)
        if CAPTURE_OFF_CHECK:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_off_s, _c1 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_off_b, _c2 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            off_checks["probe_off_final_rows_identical"] = bool(
                torch.equal(row_s, LB.row_of(lg_off_s, None).detach().cpu())
                and torch.equal(row_b, LB.row_of(rows_off_b, 0).detach().cpu()))
            M.set_attn_probe(lambda *a: None)                  # 空探针：验证调用点零副作用
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_noop_s, _c3 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_noop_b, _c4 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            M.clear_attn_probe()
            off_checks["noop_probe_final_rows_identical"] = bool(
                torch.equal(row_s, LB.row_of(lg_noop_s, None).detach().cpu())
                and torch.equal(row_b, LB.row_of(rows_noop_b, 0).detach().cpu()))
            with LB.layer_capture(tgt, LAYER, tag="off_L0") as lc_off:
                with torch.inference_mode(), M.attention_phase("decode"):
                    spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            off_checks["probe_off_L0_output_identical"] = bool(torch.equal(
                lc_off["post"]["z"][0, -1].detach().cpu(),
                S["probes"]["records"]["layer_output"]["tensor"][0, -1]))
            off_checks["probe_state_after_off"] = M.attn_probe_state()
            DC.require(all(v is True for k, v in off_checks.items() if k.endswith("identical"))
                       and off_checks["probe_state_after_off"]["active"] is False,
                       "关闭/空探针控制失败", str(off_checks))
            print("[l0op] 关闭/空探针控制 ✔（最终 logits 与 L0 输出都逐位相同）", flush=True)
        rec["capture_off_check"] = off_checks

        # ---------------- 必需控制 ----------------
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        chain = torch.load(CHAIN_PT, map_location="cpu") if Path(CHAIN_PT).exists() else {}
        anchors_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anchors_b = C._anchor_specdec(REF_RUN_RB, PRE, TARGET_POS - PRE, TARGET_POS)
        s_f, b_f = _fields(row_s), _fields(row_b)
        x0_s = S["probes"]["records"]["x0"]["tensor"][0, -1]
        x0_b = B["probes"]["records"]["x0"]["tensor"][0, 0]
        z_s = S["probes"]["records"]["layer_output"]["tensor"][0, -1]
        z_b = B["probes"]["records"]["layer_output"]["tensor"][0, 0]
        cache_s = S["cache_pre"]["k"]
        cache_b = B["cache_pre"]["k"]
        controls = dict(
            c1_L0_input_S_vs_B=_row_cmp(x0_s, x0_b, "L0_input_S_vs_B"),
            c2_L0_input_vs_prev_round=dict(
                # ⚠️ 上一轮链式捕获保存的 `rows_*_raw[i]["input"/"output"]` **已经是 (hidden,) 决策行**
                #    （不是 (1,s,hidden)）⇒ 这里**不能**再 `[0,0]` 索引（首跑即此错：白付一次模型加载）。
                S=(_row_cmp(x0_s, chain["rows_S_raw"][0]["input"], "L0in_S_vs_prev")
                   if chain else None),
                B=(_row_cmp(x0_b, chain["rows_B_raw"][0]["input"], "L0in_B_vs_prev")
                   if chain else None)),
            c3_L0_output_S_vs_B=_row_cmp(z_s, z_b, "L0_output_S_vs_B"),
            c3_L0_output_vs_prev_round=dict(
                S=(_row_cmp(z_s, chain["rows_S_raw"][0]["output"], "L0out_S_vs_prev")
                   if chain else None),
                B=(_row_cmp(z_b, chain["rows_B_raw"][0]["output"], "L0out_B_vs_prev")
                   if chain else None)),
            c4_same_L0_weights=dict(
                **_layer_weight_ptrs(tgt, LAYER),
                note="两条路径用的是**同一个模型实例**的同一批权重张量（data_ptr 相同）"),
            c5_L0_cache_pre_S_vs_B=_row_cmp(cache_s[:, :, :VIS].reshape(-1),
                                            cache_b[:, :, :VIS].reshape(-1), "cacheK_pre_S_vs_B"),
            c5_L0_cache_pre_vs_H=_row_cmp(cache_s[:, :, :VIS].reshape(-1),
                                          H[LAYER]["k"].to(cache_s.device)[:, :, :VIS].reshape(-1),
                                          "cacheK_pre_vs_H_L0"),
            c6_cos_row_S_vs_B=_row_cmp(S["cos"][0, 0, -1], B["cos"][0, 0, 0], "cos_row_S_vs_B"),
            c6_sin_row_S_vs_B=_row_cmp(S["sin"][0, 0, -1], B["sin"][0, 0, 0], "sin_row_S_vs_B"),
            c7_mask_visibility=dict(
                vis=VIS, s_S=int(S["probes"]["records"]["qk_scores_pre_mask"]["layout"]["shape"][2]),
                s_B=int(B["probes"]["records"]["qk_scores_pre_mask"]["layout"]["shape"][2]),
                T_S=int(S["probes"]["records"]["qk_scores_pre_mask"]["layout"]["shape"][3]),
                T_B=int(B["probes"]["records"]["qk_scores_pre_mask"]["layout"]["shape"][3]),
                B_row0_masked_cols_from=VIS,
                B_row0_n_inf=int(torch.isinf(
                    B["probes"]["records"]["qk_scores_post_mask"]["tensor"][0, :, 0, VIS:]).sum()),
                S_n_inf=int(torch.isinf(
                    S["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
                note="B 第 0 行被掩掉的列从 p+1=8209 开始（3 列 × 12 头 = 36 个 -inf）；"
                     "S（s=1）不加掩码 ⇒ 无 -inf；比较切片只取 0..8208"),
            c11_inputs_weights_cache_unmodified=None,   # 在 §源完整性 填充
            c12_finiteness=None)
        # 位置口径与锚点
        controls["positions"] = dict(S_last_row=True, B_row0=True,
                                     both_predict_target=bool(cache_lens["S"] == TARGET_POS
                                                              and cache_lens["B"] == PRE + len(CANDS)),
                                     cache_lens_after=cache_lens,
                                     slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1])
        controls["S_final"] = _fields(row_s)
        controls["B_final"] = _fields(row_b)
        controls["token_matches_anchors"] = dict(
            S=dict(measured=s_f["argmax_id"], anchor=anchors_s.get("token"),
                   match=bool(s_f["argmax_id"] == anchors_s.get("token"))),
            B=dict(measured=b_f["argmax_id"], anchor=anchors_b.get("token"),
                   match=bool(b_f["argmax_id"] == anchors_b.get("token"))))
        controls["vs_saved_local_rows"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S=(_row_cmp(row_s, saved_rows["H_AR-S"], "S_vs_saved") if "H_AR-S" in saved_rows else None),
            B=(_row_cmp(row_b, saved_rows["H_AR-B"], "B_vs_saved") if "H_AR-B" in saved_rows else None))
        rec["controls"] = controls
        DC.require(controls["c1_L0_input_S_vs_B"]["exact_equal"], "控制 1 失败：L0 输入 S/B 不同",
                   str(controls["c1_L0_input_S_vs_B"]))
        for k in ("S", "B"):
            DC.require(controls["c2_L0_input_vs_prev_round"][k] is None
                       or controls["c2_L0_input_vs_prev_round"][k]["exact_equal"],
                       f"控制 2 失败：L0 输入与上一轮不同（{k}）",
                       str(controls["c2_L0_input_vs_prev_round"][k]))
            DC.require(controls["c3_L0_output_vs_prev_round"][k] is None
                       or controls["c3_L0_output_vs_prev_round"][k]["exact_equal"],
                       f"控制 3 失败：L0 输出与上一轮不同（{k}）",
                       str(controls["c3_L0_output_vs_prev_round"][k]))
        DC.require(controls["c3_L0_output_S_vs_B"]["n_diff"] == 111
                   and abs(controls["c3_L0_output_S_vs_B"]["abs_err_max"] - 0.00390625) < 1e-12,
                   "控制 3 失败：L0 输出 S/B 的 n_diff/max|Δ| 与上一轮不符",
                   str(controls["c3_L0_output_S_vs_B"]))
        for k in ("c5_L0_cache_pre_S_vs_B", "c5_L0_cache_pre_vs_H", "c6_cos_row_S_vs_B",
                  "c6_sin_row_S_vs_B"):
            DC.require(controls[k]["exact_equal"], f"控制失败：{k}", str(controls[k]))
        DC.require(controls["c7_mask_visibility"]["B_row0_n_inf"] == 12 * (PRE + len(CANDS) - VIS),
                   "控制 7 失败：B 第 0 行的被掩列数不符", str(controls["c7_mask_visibility"]))
        DC.require(controls["c7_mask_visibility"]["S_n_inf"] == 0, "控制 7 失败：S 不应有被掩元素",
                   str(controls["c7_mask_visibility"]))
        DC.require(controls["positions"]["both_predict_target"], "位置口径失败",
                   str(controls["positions"]))
        DC.require(all(v["match"] for v in controls["token_matches_anchors"].values()),
                   "锚点不一致（停止）", str(controls["token_matches_anchors"]))
        if saved_rows:
            for k in ("S", "B"):
                DC.require(controls["vs_saved_local_rows"][k]["exact_equal"],
                           f"与已保存局部对照行不同（{k}）", str(controls["vs_saved_local_rows"][k]))
        print(f"[l0op] 控制 1-7 ✔（L0 输入逐位相同 {controls['c1_L0_input_S_vs_B']['sha16_a']}；"
              f"L0 输出 n_diff={controls['c3_L0_output_S_vs_B']['n_diff']} "
              f"max|Δ|={controls['c3_L0_output_S_vs_B']['abs_err_max']:.6g}）", flush=True)

        # ---------------- 逐算子比较 + 首次分歧判定 ----------------
        node_cmp = _cmp_all(S, B)
        node_cmp["cache_pre"] = dict(exact_equal=controls["c5_L0_cache_pre_S_vs_B"]["exact_equal"],
                                     n_diff=controls["c5_L0_cache_pre_S_vs_B"]["n_diff"],
                                     note="虚拟节点：L0 轮前缓存的 K/V（共同可见范围 0..8208）")
        rec["node_compare"] = node_cmp
        cls = OP.classify(OP.CHAIN, node_cmp, OP.CHAIN_NODE_OUT)
        rec["first_divergence"] = cls
        # ★ 先把**计算完的比较结果与原始张量落盘**，再做打印/收尾：
        #   这样即使后续打印或收尾出现缺陷，昂贵的数据也不会丢（三次白付加载的教训）。
        payload = _build_payload(S, B, row_s=row_s, row_b=row_b, x0_s=x0_s, x0_b=x0_b,
                                 z_s=z_s, z_b=z_b)
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        print("== L0 内部逐算子比较（S/B 对应决策切片）==")
        for (op, ins), out_node in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
            c = node_cmp[out_node]
            in_eq = [node_cmp[i]["exact_equal"] for i in ins if i in node_cmp]
            # ⚠️ 打印绝不参与结论、也绝不允许因格式问题中断（`exact_equal` 是 bool，不能用 `:<5s`）
            print(f"  {op:<24s} in_eq={str(all(in_eq)):<5s} out_eq={str(bool(c['exact_equal'])):<5s} "
                  f"n_diff={int(c['n_diff']):>6d} max|Δ|={float(c['max_abs']):<10.6g} "
                  f"shape {c['shape_s']}→{c['sliced_shape']}", flush=True)
        print(f"[l0op] 首次分歧算子 = {cls['first_divergence_operator']}"
              f"（满足条件的算子全部列出：{cls['all_operators_with_equal_inputs_and_different_output']}）",
              flush=True)

        # ---------------- 源完整性 + 有限性 ----------------
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        nan_nodes = [k for k, v in node_cmp.items() if isinstance(v, dict)
                     and (not v.get("finite_s", True) or not v.get("finite_b", True))]
        inf_expected = dict(qk_scores_post_mask=dict(
            B_row0_inf=int(torch.isinf(
                B["probes"]["records"]["qk_scores_post_mask"]["tensor"][0, :, 0, VIS:]).sum()),
            S_inf=int(torch.isinf(S["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
            note="唯一允许的非有限元素 = B 路径 mask 后的 -inf（被掩的候选列 8209..8211）；"
                 "**比较切片内**全部有限（由 cmp_node 逐节点断言）"))
        rec["finiteness"] = dict(nodes_with_nonfinite_slice=nan_nodes, expected_inf=inf_expected,
                                 note="NaN 未出现；被掩 -inf 是引擎设计行为，不计入'非有限缺陷'")
        DC.require(not nan_nodes, "存在非有限（NaN/Inf）的比较切片", str(nan_nodes))
        rec["source_integrity"] = dict(
            H_snapshot_unchanged=bool(H_digest_before == H_digest_after),
            L0_cache_pre_len=int(cache_s.shape[2]),
            W_after=LB.weights_info(tgt),
            W_identity_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"]
                                   == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"]
                                 == rec["W_before"]["digest_subsample"]),
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"]
                                    == rec["norm"]["weight"]["data_ptr"]),
            # ⚠️ 必须**重新读取当前 data_ptr** 再与之前记录的比较；
            #    首版写成 `all(r == controls[...][r] for r in keys)`（拿**键名**当数据比）⇒ 恒 False 假失败。
            layer_weights_stable=bool(_layer_weight_ptrs(tgt, LAYER) == {
                k: v for k, v in controls["c4_same_L0_weights"].items() if k.endswith("data_ptr")}),
            x0_row_unchanged=bool(torch.equal(x0_s,
                                              S["probes"]["records"]["x0"]["tensor"][0, -1])),
            probe_state_now=M.attn_probe_state(),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers))
                               and len(tgt.norm._forward_hooks) == 0
                               and len(tgt.embed_tokens._forward_hooks) == 0),
            note="H 快照（逐层 k/v 摘要）、W/Norm/L0 各权重、x0 行在全程不变；"
                 "每层缓存按设计只追加（S 8208→8209、B 8208→8212）")
        rec["controls"]["c11_inputs_weights_cache_unmodified"] = dict(
            H_snapshot_unchanged=rec["source_integrity"]["H_snapshot_unchanged"],
            W_identity_stable=rec["source_integrity"]["W_identity_stable"],
            W_digest_stable=rec["source_integrity"]["W_digest_stable"],
            norm_weight_stable=rec["source_integrity"]["norm_weight_stable"],
            layer_weights_stable=rec["source_integrity"]["layer_weights_stable"],
            x0_row_unchanged=rec["source_integrity"]["x0_row_unchanged"])
        rec["controls"]["c12_finiteness"] = rec["finiteness"]
        for k, v in rec["controls"]["c11_inputs_weights_cache_unmodified"].items():
            DC.require(v is True, f"控制 11 失败：{k}")
        DC.require(rec["source_integrity"]["probe_state_now"]["active"] is False,
                   "探针仍在位（控制 10 失败）")
        DC.require(rec["source_integrity"]["hooks_removed"], "钩子未全部移除（控制 10 失败）")
        print("[l0op] 控制 8-12 ✔（开关/空探针等价、重复一致、探针与钩子已移除、源与权重未变、切片全有限）",
              flush=True)

        att = M.attention_stats()
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att.get("s_max_by_phase"))
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        _write(rec, run_id)
        print(f"[l0op] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError, OP.ProbeError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[l0op] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[l0op] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
        except Exception:                                           # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：微型模型上把整条工具链（捕获/比较/分类/控制/落盘）走一遍。"""
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    global PRE, TARGET_POS, VIS, FEED, CANDS, REPEAT, LAYER
    torch.manual_seed(0)
    PRE, TARGET_POS, VIS, FEED, CANDS, REPEAT, LAYER = 5, 6, 6, [3], [3, 4, 5, 6], 2, 0
    tgt = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    ids = torch.randint(0, 64, (1, PRE))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    dev = torch.device("cpu")
    p1 = dict(S=_capture_arm(tgt, H, dev, "S", "decode", "s1"),
              B=_capture_arm(tgt, H, dev, "B", "verify", "s1"))
    p2 = dict(S=_capture_arm(tgt, H, dev, "S", "decode", "s2"),
              B=_capture_arm(tgt, H, dev, "B", "verify", "s2"))
    check("★ L0 探针捕获：需要的节点都在（且只有目标层子模块的探针）",
          all(k in p1[a]["probes"]["records"] for a in ("S", "B")
              for k in ("x0", "input_rmsnorm", "q_linear", "q_reshaped", "q_rope", "k_visible",
                        "v_visible", "qk_scores_pre_mask", "qk_scores_post_mask", "attention_prob",
                        "attention_times_v", "heads_merged", "o_proj", "attention_residual",
                        "post_attention_rmsnorm", "mlp_gate_proj", "mlp_activation", "mlp_up_proj",
                        "mlp_gated_product", "mlp_down_proj", "layer_output")),
          sorted(p1["S"]["probes"]["records"]))
    check("★ 探针记录里非目标层的调用被忽略；两次捕获的全部节点逐位相同",
          p1["S"]["probes"]["n_calls"] > p1["S"]["probes"]["n_kept"]
          and all(bool(torch.equal(p1[a]["probes"]["records"][k]["tensor"],
                                   p2[a]["probes"]["records"][k]["tensor"]))
                  for a in ("S", "B") for k, v in p1[a]["probes"]["records"].items()
                  if not v.get("shape_only")))
    node_cmp = _cmp_all(p1["S"], p1["B"])
    cls = OP.classify(OP.CHAIN, dict(node_cmp, cache_pre=dict(exact_equal=True)), OP.CHAIN_NODE_OUT)
    check("★ 逐算子比较 + 首次分歧判定可跑通（微型模型上的真实分类）",
          set(node_cmp) >= set(OP.CHAIN_NODE_OUT) and
          (cls["first_divergence_operator"] is None
           or cls["first_divergence_operator"] in [op for op, _ in OP.CHAIN]), cls["first_divergence_operator"])
    print(f"  [obs] 微型模型（{len(tgt.layers)} 层）L0 内的首次分歧算子 = {cls['first_divergence_operator']}")
    M.set_attn_probe(lambda *a: None)
    with torch.inference_mode():
        lg_noop, _c = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    M.clear_attn_probe()
    with torch.inference_mode():
        lg_off, _c = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    check("★ 空探针 vs 关闭探针：真实前向逐位相同（调用点零副作用）",
          bool(torch.equal(lg_noop, lg_off)) and M.attn_probe_state()["active"] is False)
    check("★ 捕获前后探针都为关闭；层钩子全部移除",
          M.attn_probe_state()["active"] is False
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(len(tgt.layers))))
    try:
        with OP.capture(tgt, 0, "X"):
            raise RuntimeError("注入的探针捕获故障")
    except RuntimeError:
        pass
    check("★ 探针捕获过程中抛异常 ⇒ 探针仍被移除", M.attn_probe_state()["active"] is False)
    a, b, _r = OP.slice_pair("row", p1["S"]["probes"]["records"]["x0"]["tensor"],
                             p1["B"]["probes"]["records"]["x0"]["tensor"], vis=VIS)
    check("★ 决策行切片：S 取末位行、B 取第 0 行（(hidden,)）",
          list(a.shape) == [32] and list(b.shape) == [32])
    tp = _save_pt(_build_payload(p1["S"], p1["B"],
                                 row_s=p1["S"]["final_row"], row_b=p1["B"]["final_row"],
                                 x0_s=p1["S"]["probes"]["records"]["x0"]["tensor"][0, -1],
                                 x0_b=p1["B"]["probes"]["records"]["x0"]["tensor"][0, 0],
                                 z_s=p1["S"]["probes"]["records"]["layer_output"]["tensor"][0, -1],
                                 z_b=p1["B"]["probes"]["records"]["layer_output"]["tensor"][0, 0]),
                  run_id, selftest=True)
    check("★ 落盘 payload 构造与写入可跑通（`_build_payload` 与正式 run 同一代码路径）",
          tp.exists() and tp.stat().st_size > 0, str(tp))
    print(f"\nL0 算子边界（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：只验证机制（探针捕获/切片比较/首次分歧分类/开关等价/移除/落盘）",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
