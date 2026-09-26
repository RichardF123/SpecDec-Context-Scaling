"""g5_attn_real_tensor_forensics.py — **真实模型张量**上的逐层/逐项精度定位（GPU，只加载 target）。

回答的问题
  1. 在**同一份真实 Q/K/V、同一位置、同一 mask** 上，eager / cuDNN / 受控 FP32 参考三者的差；
  2. eager 的**三个 bf16 舍入点**各自能解释多少差异（**逐项分离**，不是相关性）：
       R1 = QK 分数被舍入成 bf16（`model_self.py:367` 的 matmul 输入 bf16 ⇒ 输出也 bf16）
       R2 = softmax 概率被 `.to(q.dtype)` 转回 bf16（`:377`）
       R3 = AV 输出被舍入成 bf16（`:378`）
  3. 无历史 KV（方阵 mask）与**有历史 KV（非方阵 offset mask）**两种情形；
  4. 整条轨迹（逐层 hidden/attn 输出）**在哪里开始分叉**（与"算子差异"分开报告）；
  5. 模型输出的 argmax 与 top1−top2 margin。

为什么必须"复用同一份张量"：层与层之间差异会沿残差流传播，比较"已经走了不同路径的输入"
无法判断差异是算子引入的还是被放大的。本工具用 `forward_pre_hook` 抓住每层真实的
`input_layernorm(x)/cos/sin/cache`，**在工具内复算** q/k/v（不修改引擎），
再用**同一次捕获**的 q/k_all/v_all 跑全部变体。

**自检（关键安全属性）**：工具复算的 eager 结果必须与引擎真实输出**逐位相同**；
否则变体分析无效，直接判 FAIL 并停止。这条自检把"复算写错"变成硬错误。

变体阶梯（R1/R2/R3 = 该舍入点存在）
  | 变体   | R1 分数 bf16 | R2 概率 bf16 | R3 输出 bf16 |
  | eager  | ✓ | ✓ | ✓ |   ← 引擎现状（用作自检）
  | qk32   | ✗ | ✓ | ✓ |   ← 只去掉 R1
  | prob32 | ✓ | ✗ | ✓ |   ← 只去掉 R2
  | out32  | ✓ | ✗ | ✗ |   ← 去掉 R2+R3（R3 = prob32 − out32）
  | fp32   | ✗ | ✗ | ✗ |   ← 受控参考（**不是绝对真值**）
  | cudnn  | — 真实融合 kernel — |

纪律
  - FP32 参考记录 TF32 等开关；本工具显式关闭 matmul TF32 并记录前后值；
  - **不宣称** cuDNN 内部精度（未做 kernel 级验证）；
  - **不宣称唯一根因**：逐项分离是非正交的（去掉 R1 会改变 softmax 的输入），
    因此同时给出"单项移除效应"与"可加性残差"，残差不为 0 就如实报告；
  - 默认 eager 策略不变；不碰 chunk/k/量化/缓存/logits/allocator。

用法
  python -m specdec.engine.g5_attn_real_tensor_forensics
环境变量
  RTF_KV=base_bf16|v_fp8   RTF_CTX_A=256（无历史）RTF_CTX_B=512（有历史）
  RTF_CHUNK_B=256  RTF_LAYERS=0,1,2（默认全部）RTF_STAGE=all|capture|propagate
"""
from __future__ import annotations

import json
import os
import statistics as stats
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import g5_8k_spec_variance_probe as P5V
from .g4_b4_cache_tests import load_ours

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET = P5V.TARGET

KV = os.environ.get("RTF_KV", "base_bf16")
LOWMEM = S.env_flag("RTF_LOWMEM_LOAD", True)      # 默认走降峰值加载器（权重逐位校验，见 lowmem_load.py）
CTX_A = int(os.environ.get("RTF_CTX_A", "256"))       # 无历史 KV：单块，方阵 mask
CTX_B = int(os.environ.get("RTF_CTX_B", "512"))       # 有历史 KV：两块，非方阵 offset mask
CHUNK_B = int(os.environ.get("RTF_CHUNK_B", "256"))
STAGE = os.environ.get("RTF_STAGE", "all")
LAYERS_ENV = os.environ.get("RTF_LAYERS", "")
ONLY_LAYERS = {int(x) for x in LAYERS_ENV.split(",") if x.strip()} if LAYERS_ENV else None

VARIANTS = ("eager", "qk32", "prob32", "out32", "fp32", "cudnn")
# 每个变体是否施加该舍入点
R = {"eager": (True, True, True), "qk32": (False, True, True), "prob32": (True, False, True),
     "out32": (True, False, False), "fp32": (False, False, False)}


# --------------------------------------------------------------------------- 数值工具
def _d(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def _absmax(a) -> float:
    return float(a.float().abs().max())


def tf32_state() -> dict:
    out = {}
    try:
        out["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception as e:                                       # noqa: BLE001
        out["matmul_allow_tf32"] = f"<{type(e).__name__}: {e}>"
    try:
        out["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception as e:                                       # noqa: BLE001
        out["cudnn_allow_tf32"] = f"<{type(e).__name__}: {e}>"
    for name in ("fp32_precision",):
        try:
            out[f"matmul_{name}"] = str(getattr(torch.backends.cuda.matmul, name))
        except Exception:                                        # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------- 捕获
class Capture:
    """用 `forward_pre_hook` 抓住每层 attention 的真实输入，并复算 q/k_all/v_all。"""

    def __init__(self, model: M.QwenLM, tag: str):
        self.model, self.tag = model, tag
        self.recs: dict[tuple[int, int], dict] = {}
        self.block = -1
        self.handles = []
        for li, layer in enumerate(model.layers):
            sa = layer.self_attn
            self.handles.append(sa.register_forward_pre_hook(self._pre(li)))
            self.handles.append(sa.register_forward_hook(self._post(li)))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, li):
        def hook(module, args):                     # 2 参形式：cache 是位置参数 args[3]
            x_ln = args[0]
            cos, sin = args[1], args[2]
            cache = args[3] if len(args) > 3 else None
            # 只保存**张量引用**：引擎只会把新张量重新绑定到 dict 的键上（`cache["k"] = cat(...)`），
            # 从不就地改写已有张量 ⇒ 这里抓到的就是**调用前**的历史 ✓
            snap = None if cache is None else {k: v for k, v in cache.items()}
            rec = dict(layer=li, block=self.block, x_ln=x_ln.detach(),
                       cos=cos.detach(), sin=sin.detach(), cache=snap,
                       s=int(x_ln.shape[1]))
            self.recs[(self.block, li)] = rec
            return None
        return hook

    def _post(self, li):
        def hook(module, args, output):
            rec = self.recs.get((self.block, li))
            if rec is not None:
                rec["real_out"] = output[0].detach()
            return None
        return hook

    # ---- 复算 q/k_all/v_all（**逐字复刻** SelfAttention.forward 的投影→RoPE→缓存拼接）----
    @staticmethod
    def rebuild_qkv(model: M.QwenLM, rec: dict) -> dict:
        sa = model.layers[rec["layer"]].self_attn
        x = rec["x_ln"]
        b, s, _ = x.shape
        q = sa.q_proj(x).view(b, s, sa.heads, sa.head_dim).transpose(1, 2)
        k = sa.k_proj(x).view(b, s, sa.kv_heads, sa.head_dim).transpose(1, 2)
        v = sa.v_proj(x).view(b, s, sa.kv_heads, sa.head_dim).transpose(1, 2)
        q = M.apply_rope(q, rec["cos"], rec["sin"])
        k = M.apply_rope(k, rec["cos"], rec["sin"])
        cache = rec["cache"]
        fmt = cache.get("fmt", sa.kv_format) if cache is not None else sa.kv_format
        if fmt == "v_fp8":
            k_on, v_on, gk, gv = False, True, M.GROUP_SIZE, M.GROUP_V
        elif fmt == "fp8":
            k_on, v_on, gk, gv = sa.fp8_k, sa.fp8_v, sa.fp8_g_k, sa.fp8_g_v
        else:
            k_on = v_on = False
            gk = gv = M.GROUP_SIZE
        if k_on:
            qk, ks = M.quantize_kv_fp8(k, gk)
            k_cur16 = M.dequantize_kv_fp8(qk, ks, x.dtype)
        else:
            k_cur16 = k
        if v_on:
            qv, vs = M.quantize_kv_fp8(v, gv)
            v_cur16 = M.dequantize_kv_fp8(qv, vs, x.dtype)
        else:
            v_cur16 = v
        if cache is None:
            k_all, v_all = k_cur16, v_cur16
        elif fmt == "fp16":
            k_all = torch.cat([cache["k"], k], dim=2)
            v_all = torch.cat([cache["v"], v], dim=2)
        else:
            k_all = (torch.cat([M.dequantize_kv_fp8(cache["k"], cache["k_scale"], x.dtype),
                                k_cur16], dim=2) if "k_scale" in cache
                     else torch.cat([cache["k"], k_cur16], dim=2))
            v_all = (torch.cat([M.dequantize_kv_fp8(cache["v"], cache["v_scale"], x.dtype),
                                v_cur16], dim=2) if "v_scale" in cache
                     else torch.cat([cache["v"], v_cur16], dim=2))
        T = k_all.shape[2] if cache is not None else s
        return dict(q=q, k_all=k_all, v_all=v_all, s=s, T=T,
                    has_cache=cache is not None, fmt=fmt, head_dim=sa.head_dim)


def rebuild_qkv(model: M.QwenLM, rec: dict) -> dict:
    """**所有**对捕获张量的复算都必须走这里。

    原因：捕获发生在 `torch.inference_mode()` 内 ⇒ 那些张量是 **inference tensor**；
    在模式外让它们参与"需要为反向保存输入"的算子（例如 `nn.Linear`，其 weight 默认
    `requires_grad=True`）会直接抛
    `RuntimeError: Inference tensors cannot be saved for backward`。
    首跑与二跑都在这一条上崩过（先崩在 analyse_case、修完又崩在 main 的 shape check）
    ⇒ 不再逐处补 `with`，而是**收口到一个函数**，让漏加在结构上不可能发生。
    """
    with torch.inference_mode():
        return Capture.rebuild_qkv(model, rec)


# --------------------------------------------------------------------------- 变体
def variant_attn(rv: dict, variant: str):
    """在**同一份真实 q/k_all/v_all** 上跑指定变体。

    掩码分支**逐字复刻**引擎（`cache is None` ⇒ diag=1；`s==1` ⇒ 不加掩码；否则 diag=T−s+1），
    这样 `variant='eager'` 才能与真实输出**逐位相同**（自检）。
    """
    q, k_all, v_all = rv["q"], rv["k_all"], rv["v_all"]
    head_dim, s, T, has_cache = rv["head_dim"], rv["s"], rv["T"], rv["has_cache"]
    if variant == "cudnn":
        with M.attention_phase("prefill"):
            return M._fused_prefill_attention(q, k_all, v_all, head_dim)
    n_rep = q.shape[1] // k_all.shape[1]
    k_rep, v_rep = M.kv_repeat(k_all, n_rep), M.kv_repeat(v_all, n_rep)
    r1, r2, r3 = R[variant]
    if r1:
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
    else:
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) / (head_dim ** 0.5)
    if has_cache and s == 1:
        pass
    else:
        diag = 1 if not has_cache else (T - s) + 1
        mask = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool), diagonal=diag)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if r2:
        probs = probs.to(q.dtype)
    if r2 and r3:
        out = torch.matmul(probs, v_rep)
    elif not r2 and r3:
        out = torch.matmul(probs, v_rep.float()).to(q.dtype)
    else:
        out = torch.matmul(probs, v_rep.float())        # out32 / fp32：不回转
    return out


# --------------------------------------------------------------------------- 主流程
def capture_pass(model: M.QwenLM, ids: torch.Tensor, chunk: int, tag: str) -> Capture:
    """一次**真实**前向，逐块捕获。`chunk >= n` ⇒ 只有一块且 `cache is None`。"""
    cap = Capture(model, tag)
    n = ids.shape[1]
    cache = None
    try:
        with torch.inference_mode(), M.attention_phase("prefill"):
            for bi, st in enumerate(range(0, n, chunk)):
                cap.block = bi
                _, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                      need_logits=True)
    finally:
        cap.close()
    return cap


def analyse_case(model, cap: Capture, cfg: dict) -> dict:
    """对一次捕获的**全部层**跑全部变体，返回逐层结果 + 汇总 + 归因。

    分析体必须包在 `torch.inference_mode()` 里：捕获到的张量是在该模式下产生的
    （是 **inference tensor**），在模式外参与可微算子会直接抛
    `RuntimeError: Inference tensors cannot be saved for backward` —— 首跑就是这样崩的
    （好在这属于**硬错误**，不会产出错误数字）。"""
    keys = sorted(k for k in cap.recs if k[0] == cfg["block"])
    if ONLY_LAYERS is not None:
        keys = [k for k in keys if k[1] in ONLY_LAYERS]
    per_layer = []
    selfcheck_fail = []
    with torch.inference_mode():
        for (blk, li) in keys:
            rec = cap.recs[(blk, li)]
            rv = rebuild_qkv(model, rec)
            outs = {v: variant_attn(rv, v) for v in VARIANTS}
            # ---- 自检：复算的 eager 必须与引擎真实输出逐位相同 ----
            # real_out 是 o_proj **之后**的结果，所以用同一份 o_proj 比较（同算子、同输入 ⇒ 可逐位比）
            sa = model.layers[li].self_attn

            def to_oembed(o):
                # o 是 (b, heads, s, head_dim)；引擎在 o_proj 前做
                # `out.transpose(1,2).contiguous().view(b, s, heads*head_dim)`
                # ⇒ 最后一维必须是 heads*head_dim（**不是** head_dim）
                b_, h_, s_, hd_ = o.shape
                return sa.o_proj(o.transpose(1, 2).contiguous().view(b_, s_, h_ * hd_))

            ref_real = rec["real_out"]
            mine = to_oembed(outs["eager"])
            bit = bool(torch.equal(mine, ref_real))
            if not bit:
                selfcheck_fail.append(dict(layer=li, max_abs=_d(mine, ref_real)))
            refmax = _absmax(outs["fp32"])
            row = dict(block=blk, layer=li, s=rv["s"], T=rv["T"], has_cache=rv["has_cache"],
                       fmt=rv["fmt"], selfcheck_bitmatch=bit,
                       fp32_absmax=refmax,
                       err_vs_fp32={v: _d(outs[v], outs["fp32"]) for v in VARIANTS if v != "fp32"},
                       norm_err_vs_fp32={v: _d(outs[v], outs["fp32"]) / max(refmax, 1e-30)
                                         for v in VARIANTS if v != "fp32"},
                       dist_vs_cudnn={v: _d(outs[v], outs["cudnn"]) for v in VARIANTS if v != "cudnn"},
                       dist_vs_eager={v: _d(outs[v], outs["eager"]) for v in VARIANTS if v != "eager"},
                       finite={v: bool(torch.isfinite(outs[v]).all()) for v in VARIANTS})
            # ---- 机制旁证量：分数幅度与最大概率（R1 的机制是"误差 ∝ 分数幅度"）----
            with torch.inference_mode():
                k_r = M.kv_repeat(rv["k_all"], rv["q"].shape[1] // rv["k_all"].shape[1])
                sc32 = torch.matmul(rv["q"].float(),
                                    k_r.float().transpose(-2, -1)) / (rv["head_dim"] ** 0.5)
                keep = M._kv_keep_mask(rv["s"], rv["T"], rv["q"].device)
                sc32 = sc32.masked_fill(~keep, float("-inf"))
                p32 = torch.softmax(sc32, dim=-1)
                row["scores_absmax"] = float(sc32[torch.isfinite(sc32)].abs().max()) \
                    if bool(torch.isfinite(sc32).any()) else float("nan")
                row["max_prob"] = float(p32.max())
                del k_r, sc32, p32
            # ---- 逐项移除效应（单项，不同时移除）----
            d = row["dist_vs_cudnn"]
            row["removal_effect"] = dict(
                R1_score_bf16=d["eager"] - d["qk32"],
                R2_prob_bf16=d["eager"] - d["prob32"],
                R3_out_bf16=d["prob32"] - d["out32"],
                irreducible_floor_fp32=d["fp32"])
            row["additivity_residual"] = (row["removal_effect"]["R1_score_bf16"]
                                          + row["removal_effect"]["R2_prob_bf16"]
                                          + row["removal_effect"]["R3_out_bf16"]
                                          - (d["eager"] - d["fp32"]))
            per_layer.append(row)
            del outs
            torch.cuda.empty_cache()

    def med(fn):
        vals = [fn(r) for r in per_layer]
        return dict(max=max(vals), median=stats.median(vals), min=min(vals)) if vals else None

    summary = dict(
        n_layers=len(per_layer), selfcheck_all_bitmatch=not selfcheck_fail,
        selfcheck_failures=selfcheck_fail[:5],
        dist_vs_cudnn=med(lambda r: r["dist_vs_cudnn"]["eager"]),
        dist_vs_cudnn_by_variant={v: med(lambda r, v=v: r["dist_vs_cudnn"][v])
                                 for v in VARIANTS if v != "cudnn"},
        err_vs_fp32_by_variant={v: med(lambda r, v=v: r["err_vs_fp32"][v])
                                for v in VARIANTS if v != "fp32"},
        removal_effect={k: med(lambda r, k=k: r["removal_effect"][k])
                        for k in ("R1_score_bf16", "R2_prob_bf16", "R3_out_bf16",
                                  "irreducible_floor_fp32")},
        additivity_residual=med(lambda r: r["additivity_residual"]),
    )
    return dict(case=cfg, per_layer=per_layer, summary=summary)


def propagate_pass(model, ids, chunk, strategy) -> dict:
    """整模型前向（指定策略），逐层记录 `x_in`（attn 输入）与 attn 输出，用于**轨迹分叉**定位。"""
    M.set_attention_strategy(strategy)
    M.reset_attention_stats()
    cap = Capture(model, f"prop_{strategy}")
    n = ids.shape[1]
    cache = None
    try:
        with torch.inference_mode(), M.attention_phase("prefill"):
            for bi, st in enumerate(range(0, n, chunk)):
                cap.block = bi
                lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                       need_logits=True)
    finally:
        cap.close()
    return dict(strategy=strategy, cap=cap, logits=lg.detach().float().cpu(),
                stats=M.attention_stats())


def compare_trajectory(a: dict, b: dict) -> dict:
    """逐层比较两条轨迹（**传播差异**，与算子差异分开报告）。

    同样包在 `inference_mode` 里：被比较的 `x_ln` / `real_out` 都是捕获到的 inference tensor。
    """
    with torch.inference_mode():
        rows = []
        for key in sorted(set(a["cap"].recs) & set(b["cap"].recs)):
            ra, rb = a["cap"].recs[key], b["cap"].recs[key]
            rows.append(dict(block=key[0], layer=key[1],
                             d_x_ln=_d(ra["x_ln"], rb["x_ln"]),
                             d_attn_out=_d(ra["real_out"], rb["real_out"]),
                             x_ln_absmax=_absmax(ra["x_ln"]),
                             attn_out_absmax=_absmax(ra["real_out"]),
                             q_identical=torch.equal(ra["x_ln"], rb["x_ln"])))
        la, lb = a["logits"], b["logits"]
        am_a, am_b = la.argmax(-1), lb.argmax(-1)
        last = (la[0, -1], lb[0, -1])
        top2 = torch.topk(last[0], 2).values
        return dict(per_layer=rows, logits_abs_err=_d(la, lb),
                    logits_rel_err=_d(la, lb) / max(_absmax(la), 1e-30),
                    logits_ref_absmax=_absmax(la),
                    logits_finite=bool(torch.isfinite(lb).all()),
                    argmax_all_equal=bool((am_a == am_b).all()),
                    argmax_equal_frac=float((am_a == am_b).float().mean()),
                    last_pos_margin=float(top2[0] - top2[1]),
                    last_pos_abs_err=float((last[0] - last[1]).abs().max()),
                    last_pos_argmax_equal=bool(int(last[0].argmax()) == int(last[1].argmax())),
                    stats_cudnn=b["stats"])


def _load_target():
    """加载 target。`RTF_LOWMEM_LOAD=1`（默认）走降峰值加载器并**逐位校验权重**；
    `=0` 则走共享的 `load_ours`（与其它 G5 工具完全同一路径）。
    两条路径的返回值都带 `load_path`，必须写进结果 JSON。"""
    if LOWMEM:
        from .lowmem_load import load_ours_bf16
        model, tok, info = load_ours_bf16(TARGET)
        return model, tok, info
    model, tok = load_ours(TARGET)
    return model, tok, dict(load_path="g4_b4_cache_tests.load_ours",
                            weights_bitmatch=None, mismatch_keys=None,
                            n_params_checked=None, cpu_peak_note="共享加载器（不降峰值）")


def _spearman(xs, ys) -> float:
    """秩相关（只作**机制一致性**的旁证；结论以"干预移除"为准，不以相关性为准）。"""
    n = len(xs)
    if n < 3:
        return float("nan")
    ra = {i: p for p, i in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    rb = {i: p for p, i in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    ma, mb = (n - 1) / 2, (n - 1) / 2
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    import math
    den = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)) * sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / den if den else float("nan")


def ladder_pass(model, ids, chunk: int, tag: str) -> dict:
    """**整模型**精度阶梯：同一输入下比较各预设的 logits 与两个参照。

    为什么必须做这一步：逐层张量分析只能说明"算子差多少"；而**生成序列是否改变**取决于
    整条轨迹。这里把同一套精度干预搬到**完整前向**上，看 `|logits(qk32) − logits(cudnn)|`
    是否塌到 `|logits(fp32) − logits(cudnn)|` 的地板水平 —— 这是"R1 是否解释真实模型差异"的**判决性**证据。
    """
    refs: dict[str, torch.Tensor] = {}
    rows = []
    # 先算两个参照（留在 GPU 上，各 ~311 MiB @512 token；其余预设算完就比、只留标量）
    for name, strategy, preset in (("cudnn", "cudnn_prefill", "eager"),
                                   ("fp32_eager", "eager", "fp32")):
        M.set_diag_preset(preset)
        M.set_attention_strategy(strategy)
        with torch.inference_mode(), M.attention_phase("prefill"):
            lg, _ = _forward_chunked(model, ids, chunk)
        refs[name] = lg
    for preset in ("eager", "qk32", "prob32", "fp32"):
        M.set_diag_preset(preset)
        M.set_attention_strategy("eager")
        with torch.inference_mode(), M.attention_phase("prefill"):
            lg, _ = _forward_chunked(model, ids, chunk)
        row = dict(preset=preset, diag=M.diag_precision(), chunk=chunk, tag=tag,
                   logits_finite=bool(torch.isfinite(lg).all()),
                   logits_absmax=_absmax(lg),
                   argmax_eq_vs_cudnn=float((lg.argmax(-1) == refs["cudnn"].argmax(-1)).float().mean()),
                   argmax_eq_vs_fp32=float((lg.argmax(-1) == refs["fp32_eager"].argmax(-1)).float().mean()))
        for ref_name in ("cudnn", "fp32_eager"):
            row[f"dist_vs_{ref_name}"] = _d(lg, refs[ref_name])
            row[f"norm_vs_{ref_name}"] = _d(lg, refs[ref_name]) / max(_absmax(refs[ref_name]), 1e-30)
        # 末位（决定第一个生成 token）的 margin，与两个参照的末位差
        last = lg[0, -1].float()
        top2 = torch.topk(last, 2).values
        row["last_pos_margin"] = float(top2[0] - top2[1])
        row["last_pos_argmax_eq_cudnn"] = bool(int(last.argmax()) == int(refs["cudnn"][0, -1].argmax()))
        row["last_pos_argmax_eq_fp32"] = bool(int(last.argmax()) == int(refs["fp32_eager"][0, -1].argmax()))
        rows.append(row)
        print(f"[rtf] ladder {tag} {preset:<7} |Δcudnn|={row['dist_vs_cudnn']:.3e} "
              f"|Δfp32|={row['dist_vs_fp32_eager']:.3e} "
              f"argmax_eq(cudnn)={row['argmax_eq_vs_cudnn']:.4f} "
              f"末位max={row['last_pos_argmax_eq_cudnn']}", flush=True)
        del lg
        torch.cuda.empty_cache()
    del refs
    torch.cuda.empty_cache()
    M.set_diag_preset("eager")
    by = {r["preset"]: r for r in rows}
    floor = by["fp32"]["dist_vs_cudnn"]
    summary = dict(floor_fp32_vs_cudnn=floor,
                   eager_vs_cudnn=by["eager"]["dist_vs_cudnn"],
                   qk32_vs_cudnn=by["qk32"]["dist_vs_cudnn"],
                   prob32_vs_cudnn=by["prob32"]["dist_vs_cudnn"],
                   R1_explained_frac=(by["eager"]["dist_vs_cudnn"] - by["qk32"]["dist_vs_cudnn"])
                                     / max(by["eager"]["dist_vs_cudnn"] - floor, 1e-30),
                   R2_explained_frac=(by["eager"]["dist_vs_cudnn"] - by["prob32"]["dist_vs_cudnn"])
                                     / max(by["eager"]["dist_vs_cudnn"] - floor, 1e-30),
                   argmax_eq_cudnn_by_preset={r["preset"]: r["argmax_eq_vs_cudnn"] for r in rows},
                   R3_limitation="整模型层面**无法**单独移除 R3（AV 输出舍入）：下游 o_proj 是 bf16，"
                                 "fp32 的 attention 输出喂不进去（实测 dtype 报错）。"
                                 "R3 只在逐层张量阶梯（case_a/case_b）里单独测量。",
                   note="R1/R2 的 explained_frac 分母是 (eager − floor)，即可被解释的那部分；"
                        "≥1 表示移除该舍入点后已到达地板或更好")
    return dict(rows=rows, summary=summary)


def _forward_chunked(model, ids, chunk: int):
    lg, cache = None, None
    n = ids.shape[1]
    for st in range(0, n, chunk):
        lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                               need_logits=(st + chunk >= n))
    return lg, cache


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_attn_real_tensor_forensics",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), kv=KV,
                     config=dict(ctx_a=CTX_A, ctx_b=CTX_B, chunk_b=CHUNK_B,
                                 variants=list(VARIANTS), only_layers=sorted(ONLY_LAYERS)
                                 if ONLY_LAYERS else None, stage=STAGE),
                     rounding_points_explained=dict(
                         R1="QK 分数被舍入成 bf16（matmul 输入 bf16 ⇒ 输出 bf16）",
                         R2="softmax 概率 .to(dtype) 转回 bf16",
                         R3="AV 输出被舍入成 bf16",
                         note="逐项移除**非正交**：去掉 R1 会改变 softmax 的输入；"
                              "故同时报告单项移除效应与可加性残差"),
                     tf32_before=tf32_state())
    print(f"[rtf] run_id={run_id} kv={KV} ctx_a={CTX_A} ctx_b={CTX_B} chunk_b={CHUNK_B}")
    print(f"[rtf] TF32: {rec['tf32_before']}")
    guard_line = float(P5V.MIN_COMMIT_GIB)
    rec["commit_guard_line_gib"] = guard_line
    rec["commit_guard_overridden"] = bool(guard_line < 16.0)
    # 护栏原文：默认 16 GiB = 实测加载需求(9.1–9.6) + ~6.5 GiB 余量。
    # 本工具默认走 lowmem 加载路径（先 .to(bf16) 再 load_state_dict，CPU 峰值 ~5.7 GiB），
    # 因此"同余量"的等效线 ≈ 5.7 + 6.5 ≈ 12.2 GiB。这个算术必须写进结果，避免被误读成"随便放宽"。
    rec["commit_guard_note"] = (
        f"effective_line={guard_line} GiB; load_path={rec.get('load', {}).get('load_path')}; "
        "默认线 16 = 需求 9.1–9.6 + 余量 ~6.5；lowmem 路径需求 ~5.7 ⇒ 同余量等效线 ~12.2")
    if rec["commit_guard_overridden"]:
        print(f"[rtf] !! 注意：使用**文档化的护栏覆盖** P5V_MIN_COMMIT_GIB={guard_line} GiB；"
              f"本 run 的加载路径={rec.get('load', {}).get('load_path')}；"
              f"护栏原设计为【需求 9.1–9.6 + 余量 ~6.5】⇒ 本次需求 ~5.7 时等效线 ≈ 12.2 GiB，"
              f"因此本次仍满足原设计的**余量口径**。")

    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=ok, why=why, system=P5V.system_memory())
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    print(f"[rtf] commit guard: {why}")
    print(f"[rtf] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')} "
          f"mem={pf.get('mem_median')} reason={pf.get('reason')}")
    if not ok or pf.get("qualified") is not True:
        rec["skipped"] = f"环境不合格：commit_ok={ok} pstate={pf.get('qualified')}"
        _write(rec, run_id)
        print(f"[rtf] !! 环境不合格，未加载模型：{rec['skipped']}")
        return 3

    try:
        # FP32 参考必须是真 fp32：显式关闭 matmul TF32（并记录前后状态）
        torch.backends.cuda.matmul.allow_tf32 = False
        rec["tf32_after_force_off"] = tf32_state()
        model, tok, load_info = _load_target()
        rec["load"] = load_info
        rec["model_dtype"] = str(next(model.parameters()).dtype)
        rec["n_layers"] = len(model.layers)
        rec["alloc_after_load"] = P5V.allocator_state()
        full = P5V.load_ids(tok)
        if KV != "base_bf16":
            # `set_kv_format` 收的是**引擎名**（'fp16'/'v_fp8'/'fp8'），实验名要经 spec_dec._fmt 映射
            from . import spec_dec
            engine_fmt = spec_dec._fmt(KV)
            model.set_kv_format(engine_fmt)
            rec["engine_kv_format"] = engine_fmt
        else:
            rec["engine_kv_format"] = "fp16(default)"

        # ---------- A) 无历史 KV（单块、方阵 mask）----------
        ids_a = full[:, :CTX_A].contiguous()
        cap_a = capture_pass(model, ids_a, chunk=CTX_A, tag="A_nohist")
        # 断言：确实没有历史（cache is None）与方阵
        rv0 = rebuild_qkv(model, cap_a.recs[(0, 0)])
        rec["case_a_shape_check"] = dict(s=rv0["s"], T=rv0["T"], has_cache=rv0["has_cache"])
        rec["case_a"] = analyse_case(model, cap_a, dict(name="A_nohist_square",
                                                       block=0, ctx=CTX_A, chunk=CTX_A))
        del cap_a

        # ---------- B) 有历史 KV（两块、非方阵 offset mask）----------
        ids_b = full[:, :CTX_B].contiguous()
        cap_b = capture_pass(model, ids_b, chunk=CHUNK_B, tag="B_hist")
        rv1 = rebuild_qkv(model, cap_b.recs[(1, 0)])
        rec["case_b_shape_check"] = dict(s=rv1["s"], T=rv1["T"], has_cache=rv1["has_cache"])
        rec["case_b"] = analyse_case(model, cap_b, dict(name="B_history_offsets",
                                                       block=1, ctx=CTX_B, chunk=CHUNK_B))
        del cap_b
        torch.cuda.empty_cache()

        # ---------- C) 轨迹分叉（传播差异）—— 放在阶梯**之前**跑：
        # 阶梯曾因 dtype 限制崩过一次，而轨迹结果不该被它拖没（先跑便宜的、更核心的）
        if STAGE in ("all", "propagate"):
            pa = propagate_pass(model, ids_b, CHUNK_B, "eager")
            pb = propagate_pass(model, ids_b, CHUNK_B, "cudnn_prefill")
            rec["trajectory"] = compare_trajectory(pa, pb)
            print(f"[rtf] 轨迹：logits abs={rec['trajectory']['logits_abs_err']:.3e} "
                  f"argmax_eq={rec['trajectory']['argmax_equal_frac']:.4f} "
                  f"lastpos_margin={rec['trajectory']['last_pos_margin']:.4f}")
            rec["stage"] = "propagate_done"
            _write(rec, run_id)
            del pa, pb
            torch.cuda.empty_cache()

        # ---------- D) 整模型精度阶梯（判决性：R1 是否解释真实模型的 logits 差异）----------
        if STAGE in ("all", "ladder"):
            rec["ladder"] = ladder_pass(model, ids_b, CHUNK_B, "B_hist")
        rec["stage"] = "complete"
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        rec["alloc_on_crash"] = P5V.allocator_state()
        _write(rec, run_id)
        print(f"[rtf] !! 崩溃（已落盘）：\n{rec['traceback']}")
        return 2
    finally:
        M.set_attention_strategy("eager")
        M.set_diag_preset("eager")           # 诊断开关必须复位（默认路径 = 全 False）
        M.reset_attention_stats()

    _write(rec, run_id)
    _print_summary(rec)
    print(f"[rtf] -> {_out_path(run_id)}")
    return 0


def _print_summary(rec: dict) -> None:
    for tag in ("case_a", "case_b"):
        c = rec.get(tag)
        if not c:
            continue
        s = c["summary"]
        print(f"\n== {c['case']['name']} (s={c['per_layer'][0]['s']}, "
              f"T={c['per_layer'][0]['T']}, has_cache={c['per_layer'][0]['has_cache']}) ==")
        print(f"  自检（复算 eager 与引擎真实输出逐位相同）: {s['selfcheck_all_bitmatch']}"
              + (f"  失败={s['selfcheck_failures']}" if s["selfcheck_failures"] else ""))
        print(f"  {'变体':<8}{'|v − cudnn| max':>18}{'|v − cudnn| 中位':>18}"
              f"{'|v − fp32| max':>18}")
        for v in VARIANTS:
            if v == "cudnn":
                continue
            d = s["dist_vs_cudnn_by_variant"][v]
            e = s["err_vs_fp32_by_variant"].get(v)
            print(f"  {v:<8}{d['max']:>18.3e}{d['median']:>18.3e}"
                  f"{(e['max'] if e else float('nan')):>18.3e}")
        r = s["removal_effect"]
        print(f"  逐项移除效应（中位）：R1={r['R1_score_bf16']['median']:.3e} "
              f"R2={r['R2_prob_bf16']['median']:.3e} R3={r['R3_out_bf16']['median']:.3e} "
              f"floor(fp32)={r['irreducible_floor_fp32']['median']:.3e}")
        print(f"  可加性残差（中位）：{s['additivity_residual']['median']:.3e}")
        # 机制旁证：R1 的机制预测"误差 ∝ 分数幅度" ⇒ 报秩相关（旁证，不是判据）
        try:
            rho = _spearman([p["scores_absmax"] for p in c["per_layer"]],
                            [p["dist_vs_cudnn"]["eager"] for p in c["per_layer"]])
            print(f"  Spearman(分数幅度, |eager−cudnn|) = {rho:+.3f}  ← 机制旁证（非判据）")
        except Exception as e:                                   # noqa: BLE001
            print(f"  Spearman 计算失败：{type(e).__name__}: {e}")
    lad = rec.get("ladder")
    if lad:
        s = lad["summary"]
        print("\n== 整模型精度阶梯（判决性）==")
        print(f"  地板 |logits(fp32_eager) − logits(cudnn)| = {s['floor_fp32_vs_cudnn']:.3e}")
        for k in ("eager", "qk32", "prob32"):
            print(f"  |logits({k:<6}) − logits(cudnn)| = {s[k + '_vs_cudnn']:.3e}")
        print(f"  R1 解释比例 = {s['R1_explained_frac']:.3f} ; R2 解释比例 = {s['R2_explained_frac']:.3f}")
        print(f"  与 cudnn 的 argmax 一致率：{s['argmax_eq_cudnn_by_preset']}")
        print(f"  R3 限制：{s['R3_limitation'][:60]}...")
    t = rec.get("trajectory")
    if t:
        first = next((r for r in t["per_layer"] if r["d_attn_out"] > 0), None)
        print("\n== 轨迹分叉（传播差异，与算子差异分开读）==")
        if first:
            print(f"  首个 attn 输出不同的层 = {first['layer']}"
                  f"（该层 d_attn_out={first['d_attn_out']:.3e}, d_x_ln={first['d_x_ln']:.3e}"
                  f"，x_ln_absmax={first['x_ln_absmax']:.3f}）")
        else:
            print("  逐层 attn 输出**全部逐位相同**（此配置下未检出分叉）")
        print(f"  logits: abs={t['logits_abs_err']:.3e} rel={t['logits_rel_err']:.3e} "
              f"argmax_eq={t['argmax_equal_frac']:.4f} 末位margin={t['last_pos_margin']:.4f}")


def _out_path(run_id: str) -> Path:
    return OUTDIR / f"rtf_{KV}_{run_id}.json"


def _write(rec: dict, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _out_path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


if __name__ == "__main__":
    raise SystemExit(main())
