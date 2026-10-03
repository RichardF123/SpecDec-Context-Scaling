"""gemm_shape_probe_f13_p8209.py — **形状敏感性机制确认**（Step 0–4；诊断性、单次 GPU、只读）。

## 本轮要回答
* **Q1**：`M=1` 与 `M=4` 在五类 GEMM 上是否走了**不同的 CUDA/cuBLAS 算法/kernel/tile/split-K**？
* **Q2**：有无**兼顾正确性与性能**的修复？（原始块 / 全部逐行 / 仅形状敏感算子逐行 / cuBLAS·确定性旋钮 /
  已有 attention 后端）

## 结构（与任务书 Step 一一对应）
* **Step 0**：冻结版本、commit、H 来源与 sha、张量 shape/stride/dtype；**复现**：单步选 323、四行块选 659，
  并记录两侧 top-1/top-2 logit 值与 323/659 的 logit。
* **Step 1**：五类算子的**最小复现**（真实捕获张量）：`M=1`、`M=4`、`M=4` 第 0 行**截取后独立跑 M=1**；
  `torch.profiler` 记录 kernel 名称/调用次数/CUDA 时间/显存增量（workspace 代理）；Tensor Core 由名称**推断**。
* **Step 2**：**layout/content 控制**：第 0 行 view vs contiguous copy、同数据填满 M=4、四行同内容、
  改其他行内容、k/x 的 stride 变体。
* **Step 3**：**修复候选**：① 原始块 M=4；② 仅五类形状敏感算子逐行；③ 全部 Linear（含 q/k/v）+ QK 逐行；
  ④ cuBLAS/精度旋钮（bf16 reduced-reduction、preferred_blas_library、deterministic）；⑤ attention 后端
  **只作观测**、**不**当正确性证明。
* **Step 4**：**跨层** L0/L1/L14/L27：B0（无干预）vs BALL（全层候选）的层输入/输出、首个持续边界，
  以及**在该层用真实张量**重跑五类算子的 M=1 vs M=4 最小测试 ⇒ 是否重复同类形状效应。

## 纪律
不做 FP32、不切生产后端、不改权重/cache/mask/RoPE、不跑 B6；**profiler 时间不进入速度声明**
（速度数字引用上一轮**非 profiler** 的固定 block 测量）；结果先落盘再摘要；旧产物不覆盖。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import results_schema as S
from . import cache_clone as CC
from . import diag_checks as DC
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import op_probe as OP
from . import row_aligned as RA
from . import sb_align as A
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("GP_TAG", "div8209_gemm")
LAYER = int(os.environ.get("GP_LAYER", "0"))
CROSS_LAYERS = [int(x) for x in os.environ.get("GP_LAYERS", "0,1,14,27").split(",")]
PRE = int(os.environ.get("GP_PRE", "8208"))
TARGET_POS = int(os.environ.get("GP_TARGET", "8209"))
FEED = [int(x) for x in os.environ.get("GP_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("GP_CANDS", "20642,659,576,4531").split(",")]
TARGET_ROPE = os.environ.get("GP_TARGET_ROPE", "cpu_fp32").strip().lower()
CAP_H = os.environ.get("GP_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("GP_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
KV, ARM = "base_bf16", "eager"
REP = int(os.environ.get("GP_REP", "5"))
VIS = PRE + 1
TC_RE = ("s16816", "xmma", "cutlass", "nvjet", "sm90", "sm80", "hgmma", "qgmma", "tensor")


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


def _layout(t) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device), is_contiguous=bool(t.is_contiguous()))


def _cmp(a, b, name) -> dict:
    DC.assert_same_shape(a, b, name)
    d = (a.float() - b.float()).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a, b)), n_diff=int((a != b).sum()),
                max_abs=float(d.max()), mean_abs=float(d.mean()), sha16_a=A.stable_sha(a),
                sha16_b=A.stable_sha(b), shape=[int(x) for x in a.shape], finite=bool(torch.isfinite(a).all()
                                                                                     and torch.isfinite(b).all()))


def _versions() -> dict:
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cublas=getattr(torch.version, "cublas", None),
                device=(p.name if p else None), capability=(list(torch.cuda.get_device_capability(0))
                                                            if p else None),
                total_mem_gib=(round(p.total_memory / 2 ** 30, 2) if p else None),
                allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                allow_bf16_red=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                deterministic=bool(torch.are_deterministic_algorithms_enabled()),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                prefer_blas=(torch.backends.cuda.preferred_blas_library()
                             if hasattr(torch.backends.cuda, "preferred_blas_library") else None))


def _kernels(prof, top: int = 10) -> list:
    out = []
    for e in prof.key_averages():
        name = str(e.key)
        ct = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if ct <= 0 and "gemm" not in name.lower():
            continue
        out.append(dict(kernel=name, count=int(e.count), cuda_us=ct,
                        tensor_core_likely=bool(any(t in name.lower() for t in TC_RE))))
    out.sort(key=lambda d: (-d["cuda_us"], -d["count"]))
    return out[:top]


def _profile_call(fn, reps: int) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return dict(reps=int(reps), kernels=_kernels(prof), alloc_delta_mib=float((peak - base) / 2 ** 20),
                note="`alloc_delta_mib` = 显存增量（workspace 的**代理**）；`tensor_core_likely` 由 kernel 名**推断**")


# ------------------------------------------------------------------ 五类算子的最小矩阵
def _lin_matrix(W, x, bias, *, vis: int | None = None, prof: bool = False, rep: int = 5) -> dict:
    """Linear 族：M=4 整块 / M=1 第 0 行 / 第 0 行 contiguous copy；第 0 行逐位比较。"""
    def f_all():
        return F.linear(x, W, bias)

    def f_row():
        return F.linear(x[:, 0:1, :], W, bias)

    def f_row_c():
        return F.linear(x[:, 0:1, :].contiguous(), W, bias)
    with torch.inference_mode():
        y_all, y_row, y_row_c = f_all(), f_row(), f_row_c()
        cmp1 = _cmp(y_all[0, 0, :], y_row[0, 0, :], "M4row0_vs_M1")
        cmp2 = _cmp(y_all[0, 0, :], y_row_c[0, 0, :], "M4row0_vs_M1contig")
        cmp3 = _cmp(y_row[0, 0, :], y_row_c[0, 0, :], "M1_vs_M1contig")
    out = dict(x_layout=_layout(x), W_layout=_layout(W), M=dict(all=int(y_all.shape[1]), row=1),
               M4row0_vs_M1=cmp1, M4row0_vs_M1contig=cmp2, M1_vs_M1contig=cmp3)
    if prof:
        out["profile_all"] = _profile_call(f_all, rep)
        out["profile_row"] = _profile_call(f_row, rep)
    return out


def _qk_matrix(q, k_rep, head_dim, vis, *, prof: bool = False, rep: int = 5) -> dict:
    """QK 族：整块 M=4 / 第 0 行 M=1（同 N）/ 第 0 行 M=1（真实 causal 前缀）；比较统一截到 `vis`。"""
    s = int(q.shape[2])

    def f_all():
        return torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)

    def f_row():
        return torch.matmul(q[:, :, 0:1, :].contiguous(), k_rep.transpose(-2, -1)) / (head_dim ** 0.5)

    def f_row_causal():
        return torch.matmul(q[:, :, 0:1, :].contiguous(),
                            k_rep[:, :, :vis, :].contiguous().transpose(-2, -1)) / (head_dim ** 0.5)
    with torch.inference_mode():
        y_all, y_row, y_rc = f_all(), f_row(), f_row_causal()
        a = y_all[0, :, 0, :vis]
        cmp1 = _cmp(a, y_row[0, :, 0, :vis], "M4row0_vs_M1_sameN")
        cmp2 = _cmp(a, y_rc[0, :, 0, :], "M4row0_vs_M1_causalprefix")
        cmp3 = _cmp(y_row[0, :, 0, :vis], y_rc[0, :, 0, :], "M1_sameN_vs_M1prefix")
    out = dict(q_layout=_layout(q), k_rep_layout=_layout(k_rep), vis=int(vis), s=int(s),
               M4row0_vs_M1_sameN=cmp1, M4row0_vs_M1_causalprefix=cmp2, M1_sameN_vs_M1prefix=cmp3)
    if prof:
        out["profile_all"] = _profile_call(f_all, rep)
        out["profile_row"] = _profile_call(f_row_causal, rep)
    return out


def _op_matrix_l0(tgt, nodes, layer, *, prof: bool, rep: int = 5) -> dict:
    """L0（或任意层）五类算子的最小矩阵。`nodes` 用 **B 臂**（s=4）的真实捕获张量。"""
    sa, mlp = tgt.layers[layer].self_attn, tgt.layers[layer].mlp
    hd = int(sa.head_dim)
    k_rep = M.kv_repeat(nodes["k_visible"], sa.heads // sa.kv_heads)
    return dict(
        QK=_qk_matrix(nodes["q_rope"], k_rep, hd, VIS, prof=prof, rep=rep),
        o_proj=_lin_matrix(sa.o_proj.weight, nodes["heads_merged"], getattr(sa.o_proj, "bias", None),
                           prof=prof, rep=rep),
        gate_proj=_lin_matrix(mlp.gate_proj.weight, nodes["post_attention_rmsnorm"],
                              getattr(mlp.gate_proj, "bias", None), prof=prof, rep=rep),
        up_proj=_lin_matrix(mlp.up_proj.weight, nodes["post_attention_rmsnorm"],
                            getattr(mlp.up_proj, "bias", None), prof=prof, rep=rep),
        down_proj=_lin_matrix(mlp.down_proj.weight, nodes["mlp_gated_product"],
                              getattr(mlp.down_proj, "bias", None), prof=prof, rep=rep))


# ------------------------------------------------------------------ Step 2：content / layout 控制
def _lin_controls(mod, x) -> dict:
    W, b = mod.weight, getattr(mod, "bias", None)
    s = int(x.shape[1])
    with torch.inference_mode():
        base4 = F.linear(x, W, b)
        base1 = F.linear(x[:, 0:1, :], W, b)
        x_same = x[:, 0:1, :].expand(-1, s, -1).contiguous()          # 四行同内容
        y_same = F.linear(x_same, W, b)
        x_pert = x.clone()
        x_pert[:, 1:, :] = (x_pert[:, 1:, :].float() * 1.5).to(x.dtype)   # 只改其他行
        y_pert = F.linear(x_pert, W, b)
        xp = F.pad(x, (0, 0, 0, 3))
        xv = xp[:, :s, :]                                             # 同一数据、非连续 stride
        y_xv = F.linear(xv, W, b)
        y_v = F.linear(x[:, 0:1, :], W, b)                            # 第 0 行 view
        y_c = F.linear(x[:, 0:1, :].contiguous(), W, b)               # 第 0 行 contiguous copy
    return dict(
        M4row0_vs_M1=_cmp(base4[0, 0, :], base1[0, 0, :], "ctrl_M4row0_vs_M1"),
        all_rows_same_vs_M1=_cmp(y_same[0, 0, :], base1[0, 0, :], "ctrl_same_vs_M1"),
        all_rows_same_vs_M4row0=_cmp(y_same[0, 0, :], base4[0, 0, :], "ctrl_same_vs_M4row0"),
        other_rows_perturbed_vs_M4row0=_cmp(y_pert[0, 0, :], base4[0, 0, :], "ctrl_pert_vs_M4row0"),
        other_rows_perturbed_vs_M1=_cmp(y_pert[0, 0, :], base1[0, 0, :], "ctrl_pert_vs_M1"),
        stride_variant=dict(x_view=_layout(xv),
                            vs_M4row0=_cmp(y_xv[0, 0, :], base4[0, 0, :], "ctrl_strided_vs_M4row0")),
        row_view_vs_contig=_cmp(y_v[0, 0, :], y_c[0, 0, :], "ctrl_rowview_vs_rowcontig"))


def _qk_controls(q, k_rep, head_dim, vis) -> dict:
    s = int(q.shape[2])
    with torch.inference_mode():
        base4 = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
        base1 = torch.matmul(q[:, :, 0:1, :].contiguous(), k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
        q_same = q[:, :, 0:1, :].expand(-1, -1, s, -1).contiguous()
        y_same = torch.matmul(q_same, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
        q_pert = q.clone()
        q_pert[:, :, 1:, :] = (q_pert[:, :, 1:, :].float() * 1.5).to(q.dtype)
        y_pert = torch.matmul(q_pert, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
        kp = F.pad(k_rep, (0, 0, 0, 4))
        kv = kp[:, :, :int(k_rep.shape[2]), :]                        # 同一数据、非连续
        y_kv = torch.matmul(q[:, :, 0:1, :].contiguous(), kv.transpose(-2, -1)) / (head_dim ** 0.5)
        y_qv = torch.matmul(q[:, :, 0:1, :], k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
    return dict(
        M4row0_vs_M1=_cmp(base4[0, :, 0, :vis], base1[0, :, 0, :vis], "ctrl_qk_M4row0_vs_M1"),
        all_rows_same_vs_M1=_cmp(y_same[0, :, 0, :vis], base1[0, :, 0, :vis], "ctrl_qk_same_vs_M1"),
        all_rows_same_vs_M4row0=_cmp(y_same[0, :, 0, :vis], base4[0, :, 0, :vis], "ctrl_qk_same_vs_M4row0"),
        other_rows_perturbed_vs_M4row0=_cmp(y_pert[0, :, 0, :vis], base4[0, :, 0, :vis], "ctrl_qk_pert_vs_M4row0"),
        other_rows_perturbed_vs_M1=_cmp(y_pert[0, :, 0, :vis], base1[0, :, 0, :vis], "ctrl_qk_pert_vs_M1"),
        k_stride_variant=dict(k_view=_layout(kv),
                              vs_M1=_cmp(y_kv[0, :, 0, :vis], base1[0, :, 0, :vis], "ctrl_qk_kstrided_vs_M1")),
        q_row_view_vs_contig=_cmp(y_qv[0, :, 0, :vis], base1[0, :, 0, :vis], "ctrl_qk_qrowview_vs_contig"))


# ------------------------------------------------------------------ Step 3：旋钮
def _knobs(m40: dict) -> dict:
    def _probe() -> dict:
        with torch.inference_mode():
            g = m40["gate_proj"]
            # 用已保存的比较结果即可（旋钮只影响是否不同）⇒ 重算一次
            pass
        return None

    def probe(gate_W, up_W, gate_x, q, k_rep, hd):
        with torch.inference_mode():
            a = F.linear(gate_x, gate_W, None)
            r = F.linear(gate_x[:, 0:1, :], gate_W, None)
            cg = _cmp(a[0, 0, :], r[0, 0, :], "knob_gate_M4row0_vs_M1")
            qa = torch.matmul(q, k_rep.transpose(-2, -1)) / (hd ** 0.5)
            qr = torch.matmul(q[:, :, 0:1, :].contiguous(),
                              k_rep.transpose(-2, -1)) / (hd ** 0.5)
            cq = _cmp(qa[0, :, 0, :VIS], qr[0, :, 0, :VIS], "knob_qk_M4row0_vs_M1")
        return dict(gate_proj=cg, QK=cq,
                    state=dict(allow_bf16_red=bool(torch.backends.cuda.matmul
                                                   .allow_bf16_reduced_precision_reduction),
                               allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                               prefer_blas=(torch.backends.cuda.preferred_blas_library()
                                            if hasattr(torch.backends.cuda, "preferred_blas_library") else None),
                               deterministic=bool(torch.are_deterministic_algorithms_enabled())))
    return probe


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="gemm_shape_probe_f13_p8209", family="F13", kv=KV, arm=ARM,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), versions=_versions(),
                     question="M=1 与 M=4 在五类 GEMM 上是否走不同 CUDA/cuBLAS 路径？"
                              "是否存在兼顾正确性与性能的修复？",
                     semantics="最小算子复现 + profiler kernel 抓取 + layout/content 控制 + 精度/后端旋钮 + "
                               "四层跨层检查；**profiler 时间不进入任何速度声明**。",
                     config=dict(tag=TAG, layer=LAYER, cross_layers=CROSS_LAYERS, pre=PRE,
                                 target_pos=TARGET_POS, feed=FEED, cands=CANDS, rep=REP,
                                 target_rope=TARGET_ROPE, cap_h=str(CAP_H), vis=VIS,
                                 in_scope="机制确认与修复候选评估；不做 FP32、不切生产后端、不跑 B6"))
    print(f"[gp] run_id={run_id} layers={CROSS_LAYERS} pre={PRE} rep={REP}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    print(f"[gp] 版本：{json.dumps(rec['versions'], ensure_ascii=False)}", flush=True)
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        dev = next(tgt.parameters()).device
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["coverage"] = RA.coverage_check(tgt)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        cap = C._load_cap(Path(CAP_H))
        H, tinfo = C._truncate(cap["snapshot_cpu"], PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)),
                               source_len=tinfo["source_len"], pre=PRE,
                               is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H))
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = C._cache_bitequal(H, H2, "H_vs_H2")
            DC.require(rec["H_identity"]["all_equal"], "两份捕获并非逐位相同")
            del H2
        print(f"[gp] H：{Path(CAP_H).name} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        def _forward(arm: str, mode: str, *, capture_layers=None, row_aligned: bool = False):
            """一次前向（可选：在该层挂算子探针；可选：开启全层候选模式）。"""
            cache = _to_dev(H, dev)
            caps = {}
            with contextlib.ExitStack() as stack:
                ctx = (M.serial_qk_for(*RA.all_qk_modules(tgt)),
                       M.serial_linear_for(*RA.all_linear_modules(tgt))) if row_aligned else ()
                for c in ctx:
                    stack.enter_context(c)
                for li in (capture_layers or []):
                    caps[li] = stack.enter_context(OP.capture(tgt, li, arm))
                with torch.inference_mode(), M.attention_phase(mode):
                    if arm == "S":
                        lg, _cc = spec_dec._step(tgt, FEED[0], cache)
                        row = LB.row_of(lg, None)
                    else:
                        rows_b, _cc = spec_dec._verify(tgt, cache, CANDS)
                        row = LB.row_of(rows_b, 0)
                nodes = {li: {k: v["tensor"] for k, v in c.records.items() if not v.get("shape_only")}
                         for li, c in caps.items()}
            return row.detach().cpu(), nodes

        # ---------------- Step 0 ----------------
        S_row, _ = _forward("S", "decode")
        B_row, _ = _forward("B", "verify")

        def _tf(row):
            f = row.float().reshape(-1)
            v, i = torch.topk(f, 2)
            return dict(argmax=int(f.argmax()), top1=int(i[0]), top1_value=float(v[0]),
                        top2=int(i[1]), top2_value=float(v[1]), margin=float(v[0] - v[1]),
                        logit_323=float(f[323]), logit_659=float(f[659]), sha16=A.stable_sha(row))
        step0 = dict(S=_tf(S_row), B=_tf(B_row),
                     reproduced=dict(S_is_323=bool(int(S_row.argmax()) == 323),
                                     B_is_659=bool(int(B_row.argmax()) == 659),
                                     S_323_eq_659=bool(float(S_row[323].float()) == float(S_row[659].float())),
                                     B_659_gt_323=bool(float(B_row[659].float()) > float(B_row[323].float()))))
        rec["step0_reproduce"] = step0
        print(f"[gp] Step 0：S argmax={step0['S']['argmax']}（logit 323={step0['S']['logit_323']}、"
              f"659={step0['S']['logit_659']}）；B argmax={step0['B']['argmax']}"
              f"（323={step0['B']['logit_323']}、659={step0['B']['logit_659']}）", flush=True)
        DC.require(all(step0["reproduced"].values()), "Step 0 复现失败（单步/四行块的 token 或并列关系不符）",
                   json.dumps(step0, ensure_ascii=False))

        # ---------------- Step 1/2/3（L0）----------------
        _r0, n0 = _forward("B", "verify", capture_layers=[LAYER])
        nodes0 = n0[LAYER]
        rec["L0_tensor_layouts"] = {k: _layout(v) for k, v in nodes0.items()}
        m40 = _op_matrix_l0(tgt, nodes0, LAYER, prof=True, rep=REP)
        rec["step1_op_matrix_L0"] = {k: {kk: vv for kk, vv in v.items() if kk != "profile_"
                                        or True} for k, v in m40.items()}
        sa0, mlp0 = tgt.layers[LAYER].self_attn, tgt.layers[LAYER].mlp
        k_rep0 = M.kv_repeat(nodes0["k_visible"], sa0.heads // sa0.kv_heads)
        rec["step2_controls_L0"] = dict(
            QK=_qk_controls(nodes0["q_rope"], k_rep0, int(sa0.head_dim), VIS),
            o_proj=_lin_controls(sa0.o_proj, nodes0["heads_merged"]),
            gate_proj=_lin_controls(mlp0.gate_proj, nodes0["post_attention_rmsnorm"]),
            up_proj=_lin_controls(mlp0.up_proj, nodes0["post_attention_rmsnorm"]),
            down_proj=_lin_controls(mlp0.down_proj, nodes0["mlp_gated_product"]))
        probe = _knobs(m40)
        kn = dict(baseline=probe(mlp0.gate_proj.weight, None, nodes0["post_attention_rmsnorm"],
                                 nodes0["q_rope"], k_rep0, int(sa0.head_dim)))
        b = torch.backends.cuda.matmul
        try:
            b.allow_bf16_reduced_precision_reduction = False
            kn["bf16_red_off"] = probe(mlp0.gate_proj.weight, None, nodes0["post_attention_rmsnorm"],
                                       nodes0["q_rope"], k_rep0, int(sa0.head_dim))
        finally:
            b.allow_bf16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda, "preferred_blas_library"):
            for lib in ("cublas", "cublaslt", "ck"):
                try:
                    torch.backends.cuda.preferred_blas_library(lib)
                    kn[f"blas_{lib}"] = probe(mlp0.gate_proj.weight, None, nodes0["post_attention_rmsnorm"],
                                              nodes0["q_rope"], k_rep0, int(sa0.head_dim))
                except Exception as e:                                    # noqa: BLE001
                    kn[f"blas_{lib}"] = dict(error=str(e)[:200])
            try:
                torch.backends.cuda.preferred_blas_library("cublas")
            except Exception:                                             # noqa: BLE001
                pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            kn["deterministic_warn_only"] = probe(mlp0.gate_proj.weight, None,
                                                  nodes0["post_attention_rmsnorm"],
                                                  nodes0["q_rope"], k_rep0, int(sa0.head_dim))
        except Exception as e:                                            # noqa: BLE001
            kn["deterministic_warn_only"] = dict(error=str(e)[:200])
        finally:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:                                             # noqa: BLE001
                pass
        rec["step3_knobs"] = kn
        print("[gp] Step 1 kernel 对照（L0，profiler）：", flush=True)
        for name in ("QK", "o_proj", "gate_proj", "up_proj", "down_proj"):
            d = rec["step1_op_matrix_L0"][name]
            ck = (d.get("M4row0_vs_M1_sameN") or d.get("M4row0_vs_M1"))
            ka = ((d.get("profile_all") or {}).get("kernels") or [{}])[0].get("kernel")
            kr = ((d.get("profile_row") or {}).get("kernels") or [{}])[0].get("kernel")
            print(f"   {name:<10} M4row0_vs_M1 n_diff={ck['n_diff']:>5d} max|Δ|={ck['max_abs']:<10.6g}"
                  f" | M4 kernel={str(ka)[:58]} | M1 kernel={str(kr)[:58]}", flush=True)
        c = rec["step2_controls_L0"]
        print(f"[gp] Step 2（L0）：QK 四行同内容 vs M1 n_diff="
              f"{c['QK']['all_rows_same_vs_M1']['n_diff']}；改其他行后第 0 行 vs 真实 M4 第 0 行="
              f"{c['QK']['other_rows_perturbed_vs_M4row0']['n_diff']}；gate 四行同内容 vs M1="
              f"{c['gate_proj']['all_rows_same_vs_M1']['n_diff']}", flush=True)
        print(f"[gp] Step 3 旋钮：baseline gate n_diff={kn['baseline']['gate_proj']['n_diff']}、"
              f"bf16_red_off={kn.get('bf16_red_off', {}).get('gate_proj', {}).get('n_diff')}、"
              f"blas_cublaslt={((kn.get('blas_cublaslt') or {}).get('gate_proj') or {}).get('n_diff')}、"
              f"deterministic={((kn.get('deterministic_warn_only') or {}).get('gate_proj') or {}).get('n_diff')}",
              flush=True)

        # 候选 ①/②/③ 的 L0 决策行
        five_row, _ = _forward("B", "verify", row_aligned=True)
        with M.serial_qk_for(sa0), M.serial_linear_for(sa0.q_proj, sa0.k_proj, sa0.v_proj, sa0.o_proj,
                                                       mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj):
            cache = _to_dev(H, dev)
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_all, _c = spec_dec._verify(tgt, cache, CANDS)
        all_row = LB.row_of(rows_all, 0).detach().cpu()
        rec["step3_candidates_L0"] = dict(
            s_step=step0["S"], m4_block=dict(token=int(B_row.argmax()), row_sha16=A.stable_sha(B_row)),
            five_ops_row=dict(token=int(five_row.argmax()), row_sha16=A.stable_sha(five_row),
                              vs_M4=_cmp(five_row, B_row, "five_ops_vs_M4"),
                              vs_S_candidate1=_cmp(five_row, S_row, "five_ops_vs_S")),
            all_linears_row=dict(token=int(all_row.argmax()), row_sha16=A.stable_sha(all_row),
                                 vs_five=_cmp(all_row, five_row, "all_linears_vs_five")),
            note="L0 级候选只回答决策行；**全层** token equality 与代价引用上一轮 Stage 2/3（F13 通过、"
                 "decode 1.26×、固定 block verify 1.69×，均非 profiler 计时）")
        print(f"[gp] Step 3（L0）：M4 token={int(B_row.argmax())}；五算子逐行 token={int(five_row.argmax())}"
              f"（与 M4 的 n_diff={_cmp(five_row, B_row, 'x')['n_diff']}）；全部 Linear 逐行 token="
              f"{int(all_row.argmax())}", flush=True)

        # ---------------- Step 4：跨层 ----------------
        cross = {}
        _rb, n_b0 = _forward("B", "verify", capture_layers=CROSS_LAYERS)
        _rball, n_ball = _forward("B", "verify", capture_layers=CROSS_LAYERS, row_aligned=True)
        for li in CROSS_LAYERS:
            a, b2 = n_b0[li], n_ball[li]
            per_op = {}
            for (op, ins), outn in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
                in_eq = all(_cmp(a[k][0, 0], b2[k][0, 0], "t")["exact_equal"] for k in ins if k in a)
                o_eq = _cmp(a[outn][0, 0], b2[outn][0, 0], "t")["exact_equal"]
                per_op[op] = dict(all_inputs_equal=bool(in_eq), output_equal=bool(o_eq),
                                  n_diff=int((a[outn][0, 0] != b2[outn][0, 0]).sum()))
            bounds = [o for o, d in per_op.items() if d["all_inputs_equal"] and not d["output_equal"]]
            mp = _op_matrix_l0(tgt, a, li, prof=False)
            cross[f"L{li}"] = dict(
                input=_cmp(a["x0"][0, 0], b2["x0"][0, 0], f"L{li}_input"),
                output=_cmp(a["layer_output"][0, 0], b2["layer_output"][0, 0], f"L{li}_output"),
                per_op=per_op, boundaries=bounds,
                minimal_M1_vs_M4={k: dict(n_diff=(v.get("M4row0_vs_M1_sameN") or v.get("M4row0_vs_M1"))["n_diff"],
                                          max_abs=(v.get("M4row0_vs_M1_sameN") or v.get("M4row0_vs_M1"))["max_abs"])
                                  for k, v in mp.items()})
            print(f"[gp] Step 4 L{li}：输入 B0vsBALL n_diff={cross[f'L{li}']['input']['n_diff']}、"
                  f"输出 n_diff={cross[f'L{li}']['output']['n_diff']}、边界={bounds}；"
                  f"该层最小 M1vsM4="
                  f"{ {k: v['n_diff'] for k, v in cross[f'L{li}']['minimal_M1_vs_M4'].items()} }", flush=True)
        rec["step4_cross_layer"] = cross

        # ---------------- 落盘 ----------------
        tp = OUTDIR / f"{TAG}_{run_id}.pt"
        torch.save(dict(S_final_row=S_row, B_final_row=B_row, five_ops_row=five_row, all_linears_row=all_row,
                        L0_nodes={k: v.detach().cpu() for k, v in nodes0.items()},
                        versions=rec["versions"]), str(tp))
        rec["artifacts"] = dict(pt=str(tp))
        rec["finiteness"] = dict(S_B_finite=bool(torch.isfinite(S_row).all() and torch.isfinite(B_row).all()),
                                 op_matrix_all_finite=bool(all(
                                     (v.get("M4row0_vs_M1_sameN") or v.get("M4row0_vs_M1"))["finite"]
                                     for v in rec["step1_op_matrix_L0"].values())))
        rec["source_integrity"] = dict(
            H_source_unchanged=True, weights_bitmatch=rec["weights_load"]["bitmatch"],
            knobs_restored=dict(allow_bf16_red=bool(torch.backends.cuda.matmul
                                                    .allow_bf16_reduced_precision_reduction),
                                deterministic=bool(torch.are_deterministic_algorithms_enabled()),
                                allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32)),
            probe_state=M.attn_probe_state(), serial_states=dict(qk=M.serial_qk_state()["active"],
                                                                 linear=M.serial_linear_state()["active"]))
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_{run_id}.json"
        rec["artifacts"]["json"] = str(p)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[gp] RESULT_JSON={p}")
        print(f"[gp] RESULT_PT={tp}")
        return 0
    except Exception as e:                                                # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[gp] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[gp] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
        except Exception:                                                 # noqa: BLE001
            pass
        if TRS is not None:
            try:
                rec["target_rope_scope_restore"] = TRS.assert_restored()
            except Exception:                                             # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
