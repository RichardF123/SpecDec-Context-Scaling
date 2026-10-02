"""qk_shape_f13_p8209.py — 位置 8209 的 **QKᵀ 2×2 执行形状分解**（F13；只读、有界、单次）。

## 目标（**只**看 QK^T / sqrt(d)）
固定 Q/K 数值，只改 `torch.matmul(q, k_rep.transpose(-2, -1))` 的**整体执行形状**（M=query 行数、N=key 长度），
用六格回答：差异来自 **M**、**N**、还是**交互**；以及哪个因素能复现真实 S/B 的那 3 个 score 差异（`max|Δ|=4`）。

## 为什么先看 QKᵀ 而不是 q_proj
上一轮（§9.55）已证明：`q_proj` 有 2 个坐标的**瞬时** 1 ULP 差异，但**经 RoPE 被完全吸收**（`q_rope` 逐位一致）；
真正**持续传播到下游**的独立分歧边界是 `qk_matmul_scale`。本轮只分解它。

## 数据来源（**真实捕获**）
一次有界 GPU 进程内，用生产 S/B 前向 + 只读探针捕获 **live** 的
`q_rope`（S: (1,12,1,128) / B: (1,12,4,128)）、`k_visible`（S: (1,2,8209,128) / B: (1,2,8212,128)）
与真实 `qk_scores_pre_mask`（复现门的参照）。**前向结束后**才构造 2×2 并计算。

## 表达式（与 `model_self.py` 完全一致）
`k_rep = kv_repeat(k, 12 // 2)`（生产实现）；`raw = torch.matmul(q, k_rep.transpose(-2,-1))`；
`scaled = raw / (head_dim ** 0.5)`（生产写法，不是 `math.sqrt`）。**raw 与 scaled 分别记录**。

## 边界与纪律
不改生产实现/公式/dtype/mask/RoPE/attention/cache/linear/argmax/后端；不做 kernel profiler、
不切 cuBLAS/cuDNN/SDPA/Triton、不做 FP32 控制、**不执行 softmax/AV/后续层**（2×2 只用 matmul + 标量除法）、
不运行 B6/acceptance/pilot/final；shape-sensitive **不**称为 bug；ULP/margin **不**用于豁免正确性；
不外推到其它位置/family/层/模型；硬失败落盘停止、无自动重试；旧产物不覆盖；**B6 状态不变**。
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
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import op_probe as OP
from . import qk_shape_2x2 as QK
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("QK_TAG", "div8209_qk2x2")
MODE = os.environ.get("QK_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("QK_LAYER", "0"))
PRE = int(os.environ.get("QK_PRE", "8208"))
TARGET_POS = int(os.environ.get("QK_TARGET", "8209"))
K = int(os.environ.get("QK_K", "4"))
FEED = [int(x) for x in os.environ.get("QK_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("QK_CANDS", "20642,659,576,4531").split(",")]
REPEAT = int(os.environ.get("QK_REPEAT", "2"))
TARGET_ROPE = os.environ.get("QK_TARGET_ROPE", "cpu_fp32").strip().lower()
CAP_H = os.environ.get("QK_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("QK_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
L0OPS_PT = os.environ.get("QK_L0OPS_PT", str(OUTDIR / "div8209_l0ops_20261002T172447Z.pt"))
KV, ARM = "base_bf16", os.environ.get("QK_ARM", "eager")
VIS = QK.N_COMMON


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
                note="与既有 smoke 记录一致（default True → 显式 True，effective True）；allow_tf32 全程为 False")


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="qk_shape_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="位置 8209 的 QKᵀ：持续分歧来自 query 行数 M、key 长度 N，还是二者的交互？",
               semantics="固定真实捕获的 Q/K 数值，只改 `torch.matmul(q, k_repᵀ)` 的整体执行形状；"
                         "六格 A..F；raw 与 scaled 分别记录；比较区统一为 [0,:,row0,:8209]。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, repeat=REPEAT, vis=VIS, target_rope=TARGET_ROPE,
                           cells=[dict(cell=c, M=m, N=n, q=q, k=k2) for c, m, n, q, k2 in QK.CELLS],
                           l0ops_pt=str(L0OPS_PT),
                           in_scope="只分解 QKᵀ；不研究/修复 q_proj、不做 profiler、不切后端、不做 FP32、"
                                    "不执行 softmax/AV/后续层、不跑 B6"))
    print(f"[qk2x2] run_id={run_id} layer={LAYER} pre={PRE} vis={VIS} repeat={REPEAT}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[qk2x2] commit: {why}")
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


def _build_cells(q1, q4_real, k8209, k8212) -> dict:
    """构造六格并计算 raw/scaled（**只调 matmul + 标量除法**）。"""
    cells = {}
    for cell, m_, n_, qk, kk in QK.CELLS:
        q = QK.q_source(qk, q1, q4_real)
        k = QK.k_source(kk, k8209, k8212)
        DC.require(tuple(q.shape) == (1, QK.HEADS, m_, QK.HEAD_DIM),
                   f"cell {cell}: Q 形状不符", f"{tuple(q.shape)} 期望 (1,{QK.HEADS},{m_},{QK.HEAD_DIM})")
        DC.require(tuple(k.shape) == (1, QK.KV_HEADS, n_, QK.HEAD_DIM),
                   f"cell {cell}: K 形状不符", f"{tuple(k.shape)} 期望 (1,{QK.KV_HEADS},{n_},{QK.HEAD_DIM})")
        k_rep = QK.k_rep_prod(k) if m_ == 1 or True else None
        raw, scaled = QK.scores_pair(q, k_rep, head_dim=QK.HEAD_DIM)
        DC.require(tuple(raw.shape) == (1, QK.HEADS, m_, n_), f"cell {cell}: raw 形状不符",
                   str(tuple(raw.shape)))
        cells[cell] = dict(cell=cell, M=m_, N=n_, q_source=qk, k_source=kk,
                           q=q, k_rep=k_rep, raw=raw, scaled=scaled)
    return cells


def _cmp_block(cells: dict, real_S, real_B) -> dict:
    """全部比较（raw 与 scaled 分开；统一取 batch0 / 12 头 / row0 / keys 0:8209）。"""
    out = {}
    pairs = (("A_vs_B", "A", "B"), ("C_vs_D", "C", "D"), ("A_vs_C", "A", "C"),
             ("B_vs_D", "B", "D"), ("C_vs_E", "C", "E"), ("D_vs_F", "D", "F"))
    for tag, x, y in pairs:
        out[tag] = dict(
            meaning=dict(A_vs_B="M=1 时的 N 效应", C_vs_D="M=4 时的 N 效应", A_vs_C="N=8209 时的 M 效应",
                         B_vs_D="N=8212 时的 M 效应", C_vs_E="N=8209 时其余 query 内容效应",
                         D_vs_F="N=8212 时其余 query 内容效应")[tag],
            raw=QK.compare(QK.row0_prefix(cells[x]["raw"]), QK.row0_prefix(cells[y]["raw"]),
                           f"{tag}_raw"),
            scaled=QK.compare(QK.row0_prefix(cells[x]["scaled"]), QK.row0_prefix(cells[y]["scaled"]),
                              f"{tag}_scaled"))
    # 复现门（A vs 真实 S：整张量逐位；D vs 真实 B：整张量逐位 + 第 0 行前缀）
    out["A_vs_real_S"] = dict(
        meaning="S 路径复现控制（A 必须逐位复现真实 S 的 qk_scores_pre_mask）",
        raw=QK.compare(cells["A"]["raw"], real_S, "A_vs_real_S_raw"),
        scaled=QK.compare(cells["A"]["scaled"], real_S, "A_vs_real_S_scaled"))
    out["D_vs_real_B"] = dict(
        meaning="B 路径复现控制（D 必须逐位复现真实 B 的 qk_scores_pre_mask）",
        raw=QK.compare(cells["D"]["raw"], real_B, "D_vs_real_B_raw"),
        scaled=QK.compare(cells["D"]["scaled"], real_B, "D_vs_real_B_scaled"),
        raw_row0_prefix=QK.compare(QK.row0_prefix(cells["D"]["raw"]), QK.row0_prefix(real_B),
                                   "D_vs_real_B_raw_row0"))
    out["A_vs_D"] = dict(meaning="真实 S 与真实 B 的差异（本轮要解释的对象）",
                         raw=QK.compare(QK.row0_prefix(cells["A"]["raw"]),
                                        QK.row0_prefix(cells["D"]["raw"]), "A_vs_D_raw"),
                         scaled=QK.compare(QK.row0_prefix(cells["A"]["scaled"]),
                                           QK.row0_prefix(cells["D"]["scaled"]), "A_vs_D_scaled"))
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
        rec["precision_before"] = _set_precision()
        DC.require(rec["precision_before"]["allow_tf32"] is False, "allow_tf32 必须为 False",
                   str(rec["precision_before"]))
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["code_facts"] = LB.code_facts(tgt)
        rec["layer"] = LB.layer_info(tgt, LAYER)
        rec["probe_state_before"] = M.attn_probe_state()
        DC.require(rec["probe_state_before"]["active"] is False, "进入实验前探针必须关闭")
        DC.require(int(rec["layer"]["submodules"]["self_attn"]["heads"]) == QK.HEADS
                   and int(rec["layer"]["submodules"]["self_attn"]["kv_heads"]) == QK.KV_HEADS
                   and int(rec["layer"]["submodules"]["self_attn"]["head_dim"]) == QK.HEAD_DIM,
                   "GQA 维度与代码常量不一致", str(rec["layer"]["submodules"]["self_attn"]))
        print(f"[qk2x2] GQA：heads={QK.HEADS} kv_heads={QK.KV_HEADS} head_dim={QK.HEAD_DIM} "
              f"repeat={QK.HEADS // QK.KV_HEADS}；只加载 target", flush=True)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))

        cap = C._load_cap(Path(CAP_H))
        H, tinfo = C._truncate(cap["snapshot_cpu"], PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=C._cache_len(cap["snapshot_cpu"]),
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H))
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"], "两份捕获并非逐位相同")
            del H2
        print(f"[qk2x2] H：{Path(CAP_H).name} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- live 捕获（两个 pass）----------------
        def capture_once(tag: str) -> dict:
            out = {}
            for arm, mode in (("S", "decode"), ("B", "verify")):
                cache = _to_dev(H, dev)
                with OP.capture(tgt, LAYER, arm) as capr:
                    with torch.inference_mode(), M.attention_phase(mode):
                        if arm == "S":
                            lg, _ca = spec_dec._step(tgt, FEED[0], cache)
                        else:
                            _rb, _ca = spec_dec._verify(tgt, cache, CANDS)
                    pr = capr.close()
                rec_ = pr["records"]
                out[arm] = dict(q_rope=rec_["q_rope"]["tensor"], k_visible=rec_["k_visible"]["tensor"],
                                scores=rec_["qk_scores_pre_mask"]["tensor"],
                                q_lin=rec_["q_linear"]["tensor"], k_lin=rec_["k_linear"]["tensor"],
                                q_rope_layout=rec_["q_rope"]["layout"],
                                k_visible_layout=rec_["k_visible"]["layout"],
                                scores_layout=rec_["qk_scores_pre_mask"]["layout"],
                                n_calls=pr["n_calls"], n_kept=pr["n_kept"])
            return out

        passes = [capture_once(f"p{i + 1}") for i in range(REPEAT)]
        P1 = passes[0]
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        q1, q4_real = P1["S"]["q_rope"], P1["B"]["q_rope"]
        k8209, k8212 = P1["S"]["k_visible"], P1["B"]["k_visible"]
        real_S, real_B = P1["S"]["scores"], P1["B"]["scores"]
        rec["live_capture"] = dict(
            n_passes=len(passes),
            shapes=dict(q1=list(q1.shape), q4_real=list(q4_real.shape),
                        k8209=list(k8209.shape), k8212=list(k8212.shape),
                        real_S=list(real_S.shape), real_B=list(real_B.shape)),
            layouts=dict(q1=P1["S"]["q_rope_layout"], q4_real=P1["B"]["q_rope_layout"],
                         k8209=P1["S"]["k_visible_layout"], k8212=P1["B"]["k_visible_layout"],
                         real_S=P1["S"]["scores_layout"], real_B=P1["B"]["scores_layout"]),
            probe_calls=dict(S=P1["S"]["n_calls"], B=P1["B"]["n_calls"]),
            dtypes=dict(q1=str(q1.dtype), k8209=str(k8209.dtype), real_S=str(real_S.dtype)),
            note="live 捕获（生产 S/B 前向 + 只读探针）；原始 dtype/shape/stride/layout 一并记录")
        DC.require(tuple(q1.shape) == (1, QK.HEADS, 1, QK.HEAD_DIM)
                   and tuple(q4_real.shape) == (1, QK.HEADS, 4, QK.HEAD_DIM)
                   and tuple(k8209.shape) == (1, QK.KV_HEADS, VIS, QK.HEAD_DIM)
                   and tuple(k8212.shape) == (1, QK.KV_HEADS, VIS + 3, QK.HEAD_DIM),
                   "捕获形状与约定不符", str(rec["live_capture"]["shapes"]))
        # 控制 1/2/3
        rec["controls"] = dict(
            c1_q1_equals_q4_row0=QK.compare(q1, q4_real[:, :, 0:1, :], "q1_vs_q4_row0"),
            c2_k8209_equals_k8212_prefix=QK.compare(k8209, k8212[:, :, :VIS, :], "k8209_vs_k8212_prefix"),
            c3_dtypes=dict(q1=str(q1.dtype), q4_real=str(q4_real.dtype), k8209=str(k8209.dtype),
                           k8212=str(k8212.dtype), real_S=str(real_S.dtype), real_B=str(real_B.dtype),
                           all_bf16=bool(all(str(t.dtype) == "torch.bfloat16"
                                             for t in (q1, q4_real, k8209, k8212, real_S, real_B)))))
        DC.require(rec["controls"]["c1_q1_equals_q4_row0"]["exact_equal"] and
                   rec["controls"]["c2_k8209_equals_k8212_prefix"]["exact_equal"]
                   and rec["controls"]["c3_dtypes"]["all_bf16"],
                   "控制 1/2/3 失败（Q/K 对应输入不逐位一致或 dtype 非 bf16）",
                   json.dumps({k: rec["controls"][k] for k in
                               ("c1_q1_equals_q4_row0", "c2_k8209_equals_k8212_prefix",
                                "c3_dtypes")}, ensure_ascii=False)[:600])
        print("[qk2x2] 控制 1/2/3 ✔（q1 == q4_real 第 0 行、k8209 == k8212 前缀、全 bf16）", flush=True)

        # 控制 4：GQA repeat 规则（`k_rep_prod` 内部逐头核对）+ 与生产同一函数
        kr_probe = QK.k_rep_prod(k8209)
        rec["controls"]["c4_gqa_repeat"] = dict(
            rule="第 h 个 query head ↔ 第 h//6 个 KV head（`kv_repeat(k, 12//2)`，生产实现）",
            k_rep_shape=list(kr_probe.shape),
            equals_production_kv_repeat=bool(torch.equal(kr_probe, M.kv_repeat(k8209, QK.HEADS // QK.KV_HEADS))),
            heads_checked=[0, 5, 6, 11])
        DC.require(rec["controls"]["c4_gqa_repeat"]["equals_production_kv_repeat"],
                   "控制 4 失败：GQA repeat 与生产实现不一致")

        # ---------------- 六格 + 比较 + 复现门 ----------------
        cells = _build_cells(q1, q4_real, k8209, k8212)
        cmp_block = _cmp_block(cells, real_S, real_B)
        gate = dict(A_vs_real_S=cmp_block["A_vs_real_S"]["raw"],
                    D_vs_real_B=cmp_block["D_vs_real_B"]["raw"])
        gate_ok = QK.expect_fail_gate(gate)
        rec["reproduction_gate"] = dict(ok=bool(gate_ok),
                                        A_vs_real_S_raw=cmp_block["A_vs_real_S"]["raw"],
                                        A_vs_real_S_scaled=cmp_block["A_vs_real_S"]["scaled"],
                                        D_vs_real_B_raw=cmp_block["D_vs_real_B"]["raw"],
                                        D_vs_real_B_scaled=cmp_block["D_vs_real_B"]["scaled"],
                                        D_vs_real_B_raw_row0=cmp_block["D_vs_real_B"]["raw_row0_prefix"],
                                        note="A 必须逐位复现真实 S 的 qk_scores_pre_mask；D 必须逐位复现真实 B 的"
                                             "（**整张量**）⇒ 不过门即停止解释，不换 layout/公式/dtype/参数重试")
        # 与上一轮（L0 算子诊断）的 live 张量逐位核对（同一份固定数据）
        if Path(L0OPS_PT).exists():
            prev = torch.load(L0OPS_PT, map_location="cpu")
            rec["vs_previous_l0ops_pt"] = dict(
                q1=QK.compare(q1.detach().cpu(), prev["raw_S"]["q_rope"], "q1_vs_prev"),
                q4_real=QK.compare(q4_real.detach().cpu(), prev["raw_B"]["q_rope"], "q4_vs_prev"),
                k8209=QK.compare(k8209.detach().cpu(), prev["raw_S"]["k_visible"], "k8209_vs_prev"),
                k8212=QK.compare(k8212.detach().cpu(), prev["raw_B"]["k_visible"], "k8212_vs_prev"),
                real_S=QK.compare(real_S.detach().cpu(), prev["raw_S"]["qk_scores_pre_mask"],
                                  "realS_vs_prev"),
                real_B=QK.compare(real_B.detach().cpu(), prev["raw_B"]["qk_scores_pre_mask"],
                                  "realB_vs_prev"))
            for k, v in rec["vs_previous_l0ops_pt"].items():
                DC.require(v["exact_equal"], f"与上一轮固定数据不同：{k}", str(v))
            print("[qk2x2] 与上一轮 .pt 逐位一致 ✔（q_rope/k_visible/真实 scores 都与上一轮相同）", flush=True)

        # 控制 8：两次捕获 + 两次六格逐位一致
        cells2 = _build_cells(passes[1]["S"]["q_rope"], passes[1]["B"]["q_rope"],
                              passes[1]["S"]["k_visible"], passes[1]["B"]["k_visible"])
        rep = dict(
            captured_tensors_identical=bool(all(
                torch.equal(passes[0][a][k], passes[1][a][k])
                for a in ("S", "B") for k in ("q_rope", "k_visible", "scores"))),
            cells_raw_identical=bool(all(torch.equal(cells[c]["raw"], cells2[c]["raw"])
                                         for c in cells)),
            cells_scaled_identical=bool(all(torch.equal(cells[c]["scaled"], cells2[c]["scaled"])
                                            for c in cells)),
            note="每个 cell 重复两次（第二次重新捕获并重建）必须逐位一致")
        rec["repeat_consistency"] = rep
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "控制 8 失败：重复不一致", str(rep))
        print("[qk2x2] 控制 8 ✔（两次捕获与两次六格逐位一致）", flush=True)

        # 控制 5/6：复现门（不通过就停止解释）
        if not gate_ok:
            rec["stage"] = "reproduction_gate_failed"
            rec["cell_compare"] = cmp_block
            _write(rec, run_id)
            print("[qk2x2] !! 复现门失败：A/D 未逐位复现真实 scores ⇒ 停止解释（不重试）")
            return 5
        rec["controls"]["c5_A_reproduces_real_S"] = dict(
            raw=cmp_block["A_vs_real_S"]["raw"], scaled=cmp_block["A_vs_real_S"]["scaled"])
        rec["controls"]["c6_D_reproduces_real_B"] = dict(
            raw=cmp_block["D_vs_real_B"]["raw"], scaled=cmp_block["D_vs_real_B"]["scaled"],
            raw_row0=cmp_block["D_vs_real_B"]["raw_row0_prefix"])
        print("[qk2x2] 复现门 ✔（A == 真实 S scores、D == 真实 B scores，raw 与 scaled 都逐位）", flush=True)

        # ---------------- 比较、分类、控制 7/9/10 ----------------
        rec["cell_compare"] = cmp_block
        cls_raw = QK.classify({k: v["raw"] for k, v in cmp_block.items()})
        cls_scaled = QK.classify({k: v["scaled"] for k, v in cmp_block.items()})
        raw_scaled_same_pattern = bool(
            all(cmp_block[k]["raw"]["n_diff"] == cmp_block[k]["scaled"]["n_diff"]
                and cmp_block[k]["raw"]["exact_equal"] == cmp_block[k]["scaled"]["exact_equal"]
                for k in ("A_vs_B", "C_vs_D", "A_vs_C", "B_vs_D", "C_vs_E", "D_vs_F")))
        rec["classification"] = dict(
            raw=cls_raw, scaled=cls_scaled, raw_and_scaled_same_pattern=raw_scaled_same_pattern,
            boundary=("差异**首次出现在 matmul**（raw 已不同）" if not cmp_block["A_vs_D"]["raw"]["exact_equal"]
                      else "raw 相同、scaled 不同 ⇒ 边界在标量缩放"),
            note="raw 与 scaled 分别比较；两者差异模式相同 ⇒ 边界只能定位为 matmul（缩放不改变「是否相同」）")
        print(f"[qk2x2] 分类(raw)：M效应={cls_raw['m_effect']} N效应={cls_raw['n_effect']} "
              f"交互={cls_raw['interaction']}；明细={cls_raw['detail']}", flush=True)
        print(f"[qk2x2] 真实 S vs B：n_diff={cmp_block['A_vs_D']['raw']['n_diff']} "
              f"max|Δ|={cmp_block['A_vs_D']['raw']['max_abs']:.6g}", flush=True)
        rec["controls"]["c7_content_control"] = cls_raw["content_control"]
        DC.require(cls_raw["content_control"]["row0_unaffected_by_other_rows"],
                   "控制 7 失败：其余 query 行的内容影响了第 0 行 ⇒ 记为异常现象并停止普通 shape 归因",
                   str(cls_raw["content_control"]))
        # 控制 9：输入张量未被修改
        snap = dict(q1=q1.detach().clone(), q4=q4_real.detach().clone(),
                    k9=k8209.detach().clone(), k12=k8212.detach().clone())
        rec["controls"]["c9_inputs_unmodified"] = dict(
            q1=bool(torch.equal(q1, snap["q1"])), q4_real=bool(torch.equal(q4_real, snap["q4"])),
            k8209=bool(torch.equal(k8209, snap["k9"])), k8212=bool(torch.equal(k8212, snap["k12"])),
            real_S_unchanged=bool(torch.equal(real_S, real_S)),
            probe_cleared=bool(M.attn_probe_state()["active"] is False))
        DC.require(all(v is True for k, v in rec["controls"]["c9_inputs_unmodified"].items()),
                   "控制 9 失败：输入被修改或探针未移除")
        # 控制 10/11/12/13/14
        finite_ok = bool(all(v["raw"]["finite"] and v["scaled"]["finite"]
                             for v in cmp_block.values()))
        rec["controls"]["c10_finiteness"] = dict(
            all_comparisons_finite=finite_ok,
            cells_finite={c: bool(torch.isfinite(cells[c]["raw"]).all())
                          and bool(torch.isfinite(cells[c]["scaled"]).all()) for c in cells},
            nan_or_inf_any=bool(any(not torch.isfinite(v).all() for c in cells
                                    for v in (cells[c]["raw"], cells[c]["scaled"]))))
        DC.require(finite_ok and not rec["controls"]["c10_finiteness"]["nan_or_inf_any"],
                   "控制 10 失败：出现 NaN/Inf")
        prec_after = _set_precision()
        rec["controls"]["c11_allow_tf32_false"] = dict(
            before=rec["precision_before"]["allow_tf32"], after=prec_after["allow_tf32"])
        DC.require(rec["controls"]["c11_allow_tf32_false"]["after"] is False, "控制 11 失败：allow_tf32")
        rec["controls"]["c12_no_precision_switch_change"] = dict(
            bf16_red_before=rec["precision_before"]["effective_after_set"],
            bf16_red_after=prec_after["effective_after_set"],
            same=bool(rec["precision_before"]["effective_after_set"] == prec_after["effective_after_set"]))
        DC.require(rec["controls"]["c12_no_precision_switch_change"]["same"], "控制 12 失败：精度开关被改")
        rec["controls"]["c13_no_draft"] = dict(
            loaded_models=["target"], draft_loaded=False,
            note="本工具只 `load_ours_bf16(目标模型)`，从不加载 draft")
        rec["controls"]["c14_no_softmax_no_av_no_generation"] = dict(
            experiment_ops=["torch.matmul(q, k_rep.transpose(-2,-1))", "raw / (head_dim ** 0.5)",
                            "M.kv_repeat（GQA 扩展）"],
            softmax_executed=False, av_executed=False, later_layers_executed=False,
            generation_executed=False,
            note="2×2 只用 matmul 与标量除法；softmax/AV/o_proj/MLP/后续层**未**在实验中执行。"
                 "（live 捕获本身跑的是生产 S/B 前向，用于取 q_rope/k_visible/真实 scores）")
        rec["controls"]["c15_git_state"] = dict(**rec["git"], recorded=True)
        att = M.attention_stats()
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()), "要求零融合调用")

        # ---------------- 落盘（先数据，后收尾）----------------
        payload = dict(
            q1=q1.detach().cpu(), q4_real=q4_real.detach().cpu(),
            k8209=k8209.detach().cpu(), k8212=k8212.detach().cpu(),
            real_S_scores=real_S.detach().cpu(), real_B_scores=real_B.detach().cpu(),
            cells={c: dict(raw=cells[c]["raw"].detach().cpu(),
                           scaled=cells[c]["scaled"].detach().cpu(),
                           q=cells[c]["q"].detach().cpu(), k_rep=cells[c]["k_rep"].detach().cpu(),
                           M=cells[c]["M"], N=cells[c]["N"], q_source=cells[c]["q_source"],
                           k_source=cells[c]["k_source"], numel=int(cells[c]["raw"].numel()))
                   for c in cells},
            row0_prefix={c: dict(raw=QK.row0_prefix(cells[c]["raw"]).detach().cpu(),
                                 scaled=QK.row0_prefix(cells[c]["scaled"]).detach().cpu())
                         for c in cells},
            real_S_row0=QK.row0_prefix(real_S).detach().cpu(),
            real_B_row0=QK.row0_prefix(real_B).detach().cpu(),
            meta=dict(vis=VIS, layer=LAYER, cells=[list(c) for c in QK.CELLS]),
            note="全部为**原始 dtype**（bf16）张量；统计/比较时才转 FP32")
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        _write(rec, run_id)
        print(f"[qk2x2] RESULT_JSON={p}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError, A.AlignError,
                                                          CC.CloneError, LB.BoundaryError,
                                                          OP.ProbeError, QK.QKError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[qk2x2] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[qk2x2] RESULT_JSON={p}")
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
    """**不加载任何模型**：用与真实形状相同的**合成**张量走完整条工具链（六格/比较/门/分类/落盘）。

    为什么可以用合成张量：本轮的 CPU 门要验证的是**机制**（cell 构造、GQA 扩展、row0/前缀切片、
    raw/scaled 分开记录、复现门与失败门行为、分类、落盘流程），而不是数值结论。
    """
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    torch.manual_seed(0)
    q4 = torch.randn(1, QK.HEADS, 4, QK.HEAD_DIM).to(torch.bfloat16)
    q1 = q4[:, :, 0:1, :].clone()
    k8212 = torch.randn(1, QK.KV_HEADS, VIS + 3, QK.HEAD_DIM).to(torch.bfloat16)
    k8209 = k8212[:, :, :VIS, :].clone()
    # 真实（合成）S/B scores：用与六格**同一**表达式算出来，保证复现门可通过
    real_S, _ = QK.scores_pair(q1, QK.k_rep_prod(k8209))
    real_B, _ = QK.scores_pair(q4, QK.k_rep_prod(k8212))
    cells = _build_cells(q1, q4, k8209, k8212)
    cmp_block = _cmp_block(cells, real_S, real_B)
    gate = dict(A_vs_real_S=cmp_block["A_vs_real_S"]["raw"],
                D_vs_real_B=cmp_block["D_vs_real_B"]["raw"])
    check("★ 复现门在合成数据上通过（A == real_S、D == real_B 逐位）", QK.expect_fail_gate(gate))
    check("★ 六格齐全且 M/N 与定义一致（A(1,8209) B(1,8212) C(4,8209) D(4,8212) E(4,8209) F(4,8212)）",
          all(cells[c]["M"] == m_ and cells[c]["N"] == n_
              and tuple(cells[c]["raw"].shape) == (1, QK.HEADS, m_, n_)
              for c, m_, n_, _q, _k in QK.CELLS))
    check("★ raw 与 scaled **分开记录**（两者都有完整字段，且 scaled == raw/√d）",
          all(all(k in cmp_block[t]["raw"] and k in cmp_block[t]["scaled"]
                  for k in ("exact_equal", "n_diff", "max_abs", "mean_abs", "max_diff_index",
                            "sha16_a", "sha16_b", "coords", "per_head"))
              for t in ("A_vs_B", "C_vs_D", "A_vs_C", "B_vs_D", "C_vs_E", "D_vs_F"))
          and bool(torch.equal(cells["A"]["scaled"], cells["A"]["raw"] / (QK.HEAD_DIM ** 0.5))))
    check("★ 比较区就是 batch0/12 头/row0/keys 0:8209（形状 (12,8209)）",
          all(tuple(cmp_block[t]["raw"]["a_layout"]["shape"]) == (QK.HEADS, VIS)
              for t in ("A_vs_B", "C_vs_D", "A_vs_C", "B_vs_D", "C_vs_E", "D_vs_F")))
    # 内容控制：E/F 用 q1 重复 4 行 ⇒ 第 0 行与 C/D 必须逐位相同（合成数据上也应如此）
    check("★ 内容控制（合成数据）：C vs E、D vs F 第 0 行逐位相同（其余行内容不影响 row0）",
          cmp_block["C_vs_E"]["raw"]["exact_equal"] and cmp_block["D_vs_F"]["raw"]["exact_equal"])
    cls = QK.classify({k: v["raw"] for k, v in cmp_block.items()})
    check("★ 分类器输出结构完整（M/N/交互 + 明细 + 内容控制 + 真实 S/B 对照）",
          all(k in cls for k in ("m_effect", "n_effect", "interaction", "detail",
                                 "content_control", "real_S_vs_B", "reading")))
    print(f"  [obs] 合成数据分类：M={cls['m_effect']} N={cls['n_effect']} 交互={cls['interaction']} "
          f"（合成随机张量下这些是数据自身的结果，不是结论）")
    # 失败门行为：把 q1 与 q4 第 0 行改成不同 ⇒ 复现门必须失败
    q1_bad = (q1.float() + 1.0).to(torch.bfloat16)
    cells_bad = _build_cells(q1_bad, q4, k8209, k8212)
    cmpb = _cmp_block(cells_bad, real_S, real_B)
    gate_bad = dict(A_vs_real_S=cmpb["A_vs_real_S"]["raw"],
                    D_vs_real_B=cmpb["D_vs_real_B"]["raw"])
    check("★ 失败门行为：A 与真实 S 不同 ⇒ 复现门**判否**（工具会停止解释、不重试）",
          QK.expect_fail_gate(gate_bad) is False and cmpb["A_vs_real_S"]["raw"]["exact_equal"] is False)
    tp = _save_pt(dict(cells={c: dict(raw=cells[c]["raw"], scaled=cells[c]["scaled"])
                             for c in cells},
                       q1=q1, q4_real=q4, k8209=k8209, k8212=k8212,
                       real_S_scores=real_S, real_B_scores=real_B),
                  run_id, selftest=True)
    check("★ 落盘流程可跑通（原始 dtype 张量写入 .pt）", tp.exists() and tp.stat().st_size > 0, str(tp))
    print(f"\nQK 形状 2×2 工具（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：用与真实同形状的合成张量验证机制（六格/门/比较/分类/落盘）",
                           synthetic_classification=dict(m=cls["m_effect"], n=cls["n_effect"],
                                                         interaction=cls["interaction"]),
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
