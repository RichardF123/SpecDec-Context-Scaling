"""g5_attn_cudnn_validate.py — `cudnn_prefill` 策略的 **GPU 验证阶梯**（任务验证顺序第 2、3 步）。

要回答的问题（**在拿到数据前不写结论**）
  2. **算子级**：`_fused_prefill_attention`（强制 `CUDNN_ATTENTION`）在 target/draft 两套头配置
     × 有/无历史缓存 × 完整块/短末块 上的数值与**瞬时显存**；
  3. **模型级**（短上下文、串行）：`cudnn_prefill` 与 `eager` 的
     prefill logits 差、有限性、argmax、**生成 id 序列**是否一致（base_bf16 与 v_fp8）。

纪律
  - 所有判据（容差、显存规则、期望路径）**写死在本文件顶部**，在跑之前定好，**不因观测放宽**；
  - **不静默回退**：融合路径的异常一律向上抛；本文件只在工具层记录失败并**停止扩大规模**；
  - 算子级的显存数字是**算子瞬时**（transient），**不是**端到端 8K 收益，不得当作速度结论；
  - **不加任何计时结论**：本工具的耗时字段仅供诊断，禁止进入任何 speedup 表。

用法（交互式 GPU 终端；建议用 `scripts/run_g5_attn_validate.cmd`）
  python -m specdec.engine.g5_attn_cudnn_validate
可选环境变量
  ACV_STAGE=ops|model|all   （默认 all）
  ACV_CTXS=128,512,1025     ACV_CTXS_2KV=128,512      ACV_KVS=base_bf16,v_fp8
  ACV_CHUNK=1024 ACV_MAXNEW=16 ACV_K=4 ACV_REPS=2
  ACV_TOL_ABS=1e-2 ACV_TOL_REL=1e-2 ACV_FUSED_MAX_MIB=200 ACV_FUSED_RATIO_MAX=0.25
  ACV_ALLOW_UNQUALIFIED=1  ACV_DRY_RUN=1
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V
from .g4_b4_cache_tests import load_ours

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

# ---------------------------------------------------------------- 预登记判据（跑之前写死）
TOL_ABS = float(os.environ.get("ACV_TOL_ABS", "1e-2"))          # max|x_eager − x_fused|
TOL_REL = float(os.environ.get("ACV_TOL_REL", "1e-2"))          # abs_err / max(1e-6, max|x_eager|)
FUSED_TRANSIENT_MAX_MIB = float(os.environ.get("ACV_FUSED_MAX_MIB", "200"))
FUSED_RATIO_MAX = float(os.environ.get("ACV_FUSED_RATIO_MAX", "0.25"))   # fused < 0.25 × eager
MEM_ROWS_MIN_S = 512        # 只有 s ≥ 512 的完整块参与显存判据（短块本身两个数字都很小）

STAGE = os.environ.get("ACV_STAGE", "all")
CHUNK = int(os.environ.get("ACV_CHUNK", "1024"))
MAXNEW = int(os.environ.get("ACV_MAXNEW", "16"))
K = int(os.environ.get("ACV_K", "4"))
REPS = int(os.environ.get("ACV_REPS", "2"))
CTXS = [int(x) for x in os.environ.get("ACV_CTXS", "128,512,1025").split(",") if x.strip()]
CTXS_2KV = {int(x) for x in os.environ.get("ACV_CTXS_2KV", "128,512").split(",") if x.strip()}
KVS = [x.strip() for x in os.environ.get("ACV_KVS", "base_bf16,v_fp8").split(",") if x.strip()]
ALLOW_UNQUALIFIED = S.env_flag("ACV_ALLOW_UNQUALIFIED", False)
DRY_RUN = S.env_flag("ACV_DRY_RUN", False)

# 算子级表：**(heads, kv_heads, head_dim)** 取真实 target/draft；**(s, T, 是否有历史缓存)**
OP_CONFIGS = (("target", 12, 2, 128), ("draft", 14, 2, 64))
OP_ROWS = (
    # 名称                     s     T     has_cache
    ("block2048_T2048_first", 2048, 2048, False),   # 首块（cache is None）
    # 术语更正：本行是**带历史 KV 的块**（chunk=2048 时 8192 上下文的**最后一块**，历史 6144），
    # 不是"第一块"；第一块是 cache is None 的 block2048_T2048_first。s×T 最大 ⇒ eager 开销最大。
    ("block2048_T8192_hist",  2048, 8192, True),
    ("block512_T8192_hist",    512, 8192, True),
    ("block4_T8192_hist",        4, 8192, True),    # 短块但仍 ≥ MIN_S ⇒ 应融合
    ("block2_T2_first",          2,    2, False),   # 允许融合的最小情形
    ("block1_T8192_hist",        1, 8192, True),    # s==1 ⇒ 策略强制 eager
)
FP64_BUDGET = 4_000_000     # heads*s*T 超过此值不算 fp64 参考（显存/时间不划算）
QUANTILE_SAMPLE_MAX = 4_000_000   # logits 差的分位数抽样上限（torch.quantile 硬上限 2^24）


# --------------------------------------------------------------------------- 数值工具
def _max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).abs().max())


def _relative(abs_err: float, ref: torch.Tensor) -> tuple[float, float]:
    ref_max = float(ref.float().abs().max())
    return abs_err / max(1e-6, ref_max), ref_max


def _eager_ref(q, k, v, head_dim, has_cache: bool) -> torch.Tensor:
    """**逐字复刻** `SelfAttention.forward` 的 eager 分支（model_self.py:354-368）。

    故意复制而不是调用：这是"参照物"，必须与引擎实现**解耦**才能检验实现；
    复制件本身由 CPU 自测 ① 的掩码等价性把关。
    同一函数在 **fp32 输入**下复用，作为精度诊断的参考（见 `_precision_diag`）。
    """
    n_rep = q.shape[1] // k.shape[1]
    k_rep, v_rep = M.kv_repeat(k, n_rep), M.kv_repeat(v, n_rep)
    s, T = q.shape[2], k.shape[2]
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
    if not has_cache:
        mask = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    elif s == 1:
        pass
    else:
        mask = torch.triu(torch.ones(s, T, device=q.device, dtype=torch.bool),
                          diagonal=(T - s) + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v_rep)


def _precision_diag(q, k, v, head_dim, has_cache, out_e, out_f) -> dict:
    """**诊断，不是判据** —— 把 `|eager − fused|` 分解到两侧各自的精度误差上。

    参考物用 **fp32** 跑同一套 `_eager_ref` 数学：bf16 的舍入是 2⁻⁸，fp32 是 2⁻²⁴，
    相差 ~600×，因此在"bf16 量级差异"的归因上 fp32 参考**等价于**真值
    （且它复用同一份掩码代码，若掩码写错，融合侧会立刻显示出巨大的偏离）。
    这一步**不改变**任何预登记判定，只回答"差异来自哪一侧、是不是 bf16 舍入量级"。
    """
    ref32 = _eager_ref(q.float(), k.float(), v.float(), head_dim, has_cache)
    d = dict(fp32_ref_absmax=float(ref32.abs().max()),
             eager_err_vs_fp32=_max_abs(ref32, out_e),
             fused_err_vs_fp32=_max_abs(ref32, out_f))
    ulp = max(ref32.abs().max().item(), 1e-30) * 2 ** -8          # bf16 输出的一个 ULP
    d["bf16_ulp"] = ulp
    d["eager_err_in_ulp"] = d["eager_err_vs_fp32"] / ulp
    d["fused_err_in_ulp"] = d["fused_err_vs_fp32"] / ulp
    d["eager_fused_err_in_ulp"] = _max_abs(out_e, out_f) / ulp
    d["closer_to_fp32"] = ("fused" if d["fused_err_vs_fp32"] < d["eager_err_vs_fp32"]
                           else ("eager" if d["eager_err_vs_fp32"] < d["fused_err_vs_fp32"]
                                 else "tie"))
    del ref32
    return d


def _transient(fn, sync: bool = True):
    """返回 (结果, 瞬时 peak−起始 allocated MiB, reserved 增长 MiB)。"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - a0) / 2 ** 20, \
        (torch.cuda.memory_reserved() - r0) / 2 ** 20


def env_facts() -> dict:
    f = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
             device=torch.cuda.get_device_name(0),
             capability=list(torch.cuda.get_device_capability(0)),
             total_mib=torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
             alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
             hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"))
    for name, getter in (("flash_sdp_enabled", lambda: torch.backends.cuda.flash_sdp_enabled()),
                         ("mem_efficient_sdp_enabled",
                          lambda: torch.backends.cuda.mem_efficient_sdp_enabled()),
                         ("math_sdp_enabled", lambda: torch.backends.cuda.math_sdp_enabled()),
                         ("cudnn_sdp_enabled", lambda: torch.backends.cuda.cudnn_sdp_enabled())):
        try:
            f[name] = bool(getter())
        except Exception as e:                                  # noqa: BLE001  （环境自省，失败也记录）
            f[name] = f"<{type(e).__name__}: {e}>"
    return f


# --------------------------------------------------------------------------- 算子级
def op_level(rec: dict) -> None:
    rows = []
    for cname, heads, kv_heads, head_dim in OP_CONFIGS:
        for rname, s, T, has_cache in OP_ROWS:
            torch.manual_seed(0)
            dev = torch.device("cuda:0")
            q = torch.randn(1, heads, s, head_dim, device=dev, dtype=torch.bfloat16)
            k = torch.randn(1, kv_heads, T, head_dim, device=dev, dtype=torch.bfloat16)
            v = torch.randn(1, kv_heads, T, head_dim, device=dev, dtype=torch.bfloat16)
            path, reason = M.decide_attention_path("prefill", s, "cudnn_prefill")
            expect = "fused" if s >= M.PREFILL_FUSED_MIN_S else "eager"
            row = dict(config=cname, row=rname, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                       s=s, T=T, has_history=has_cache, expected_path=expect,
                       path=path, reason=reason, phase="prefill", strategy="cudnn_prefill")

            out_e, tr_e, rs_e = _transient(lambda: _eager_ref(q, k, v, head_dim, has_cache))
            row.update(eager_transient_mib=tr_e, eager_reserved_growth_mib=rs_e,
                       eager_finite=bool(torch.isfinite(out_e).all()))
            if heads * s * T <= FP64_BUDGET:
                ref64 = _eager_ref(q.double(), k.double(), v.double(), head_dim, has_cache)
                row["eager_abs_err_vs_fp64"] = _max_abs(ref64, out_e)
            else:
                row["eager_abs_err_vs_fp64"] = None       # 预算外，明确记为 None（不是 0）

            if path == "fused":
                # 真实融合 kernel：强制 CUDNN 后端（不可用 ⇒ 直接抛，不吞）
                def _call():
                    with M.attention_phase("prefill"):
                        return M._fused_prefill_attention(q, k, v, head_dim)
                out_f, tr_f, rs_f = _transient(_call)
                abs_err = _max_abs(out_e, out_f)
                rel_err, ref_max = _relative(abs_err, out_e)
                row.update(fused_transient_mib=tr_f, fused_reserved_growth_mib=rs_f,
                           fused_finite=bool(torch.isfinite(out_f).all()),
                           abs_err=abs_err, rel_err=rel_err, ref_absmax=ref_max,
                           fused_abs_err_vs_fp64=_max_abs(ref64.cuda(), out_f)
                           if row["eager_abs_err_vs_fp64"] is not None else None)
                num_ok = (row["fused_finite"] and abs_err <= TOL_ABS and rel_err <= TOL_REL)
                mem_ok = True
                if s >= MEM_ROWS_MIN_S:
                    mem_ok = (tr_f < FUSED_TRANSIENT_MAX_MIB
                              and tr_f < FUSED_RATIO_MAX * max(tr_e, 1e-9))
                row.update(numeric_ok=num_ok, memory_ok=mem_ok,
                           memory_criterion_applicable=s >= MEM_ROWS_MIN_S,
                           abs_err_over_abs_gate=abs_err / TOL_ABS,
                           abs_err_in_bf16_ulp=abs_err / (max(ref_max, 1e-30) * 2 ** -8),
                           numeric_gate="abs<=%.0e AND rel<=%.0e" % (TOL_ABS, TOL_REL))
                row["precision_diag"] = _precision_diag(q, k, v, head_dim, has_cache,
                                                        out_e, out_f)
            else:
                row.update(fused_transient_mib=None, abs_err=None, rel_err=None,
                           numeric_ok=None, memory_ok=None,
                           memory_criterion_applicable=False,
                           abs_err_over_abs_gate=None, abs_err_in_bf16_ulp=None,
                           precision_diag=None,
                           note="策略要求 eager（s < PREFILL_FUSED_MIN_S），未调用融合 kernel")
            row["path_ok"] = (path == expect)
            row["verdict"] = "PASS" if (row["path_ok"] and row["numeric_ok"] in (True, None)
                                        and row["memory_ok"] in (True, None)) else "FAIL"
            pd_ = row.get("precision_diag")
            print(f"[acv] op {cname:<6} {rname:<22} s={s:<5} T={T:<5} path={path:<5} "
                  f"eager={tr_e:7.1f}MiB fused="
                  f"{('%.1f' % tr_f) if path == 'fused' else '   n/a':>7}MiB "
                  f"abs={('%.3e' % row['abs_err']) if row['abs_err'] is not None else '     n/a'} "
                  f"rel={('%.3e' % row['rel_err']) if row['rel_err'] is not None else '     n/a'} "
                  f"ulp={('%.2f' % row['abs_err_in_bf16_ulp']) if pd_ else ' n/a'} "
                  f"closer={pd_['closer_to_fp32'] if pd_ else 'n/a'} -> {row['verdict']}", flush=True)
            rows.append(row)
            del q, k, v, out_e
            torch.cuda.empty_cache()
    rec["op_level"] = rows
    rec["op_verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in rows) else "FAIL"


def kernel_proof(rec: dict) -> None:
    """证明"真的来到了 CUDNN kernel"，而不是 math 回退（按 kernel 名核对）。

    辅助证据：失败**不得**毁掉已完成的算子级结果（首跑就因为在 torch 2.6 上漏了
    `use_cpu=True` 触发 `Legacy CUDA profiling requires use_cpu=True` 而整轮判失败）。
    """
    from torch.profiler import ProfilerActivity, profile
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    q = torch.randn(1, 12, 2048, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 8192, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8192, 128, device=dev, dtype=torch.bfloat16)
    try:
        # torch 2.6 的 `torch.profiler.profile` **没有** `use_cpu` 形参；CUDA activity 要求
        # activities 里同时含 CPU（否则内部 legacy 路径 `assert self.use_cpu` 直接抛）。
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with M.attention_phase("prefill"):
                M._fused_prefill_attention(q, k, v, 128)
            torch.cuda.synchronize()
    except Exception as e:                                      # noqa: BLE001  （辅助证据，失败即记录）
        rec["kernel_proof"] = dict(case="target 12×2048×8192×128 s=2048", matched=None,
                                   error=f"{type(e).__name__}: {e}")
        print(f"[acv] kernel proof 失败（已记录，不影响算子级结果）：{rec['kernel_proof']['error']}",
              flush=True)
        del q, k, v
        torch.cuda.empty_cache()
        return
    names = sorted({e.key for e in prof.key_averages()})
    low = [n.lower() for n in names]
    hit = sorted({n for n in names
                  if "cudnn" in n.lower() or "fmha" in n.lower() or "cutlass" in n.lower()})
    math_hit = sorted({n for n in names if "gemm" in n.lower() or "softmax" in n.lower()})
    rec["kernel_proof"] = dict(case="target 12×2048×8192×128 s=2048", kernels=names,
                               fused_kernel_names=hit, other_kernel_names=math_hit,
                               matched=bool(hit),
                               cudnn_mentioned=any("cudnn" in n for n in low))
    print(f"[acv] kernel proof: matched={bool(hit)} names={hit}", flush=True)
    del q, k, v
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- 模型级
def _label_phases(tgt, cap: dict):
    """harness 侧显式标注阶段（**不改 spec_dec.py**）。

    `spec_dec._prefill` 直接调 `model._run`、**不经过** `model_self.chunked_prefill`，
    所以必须由调用方在这一层标注 `prefill`；否则 phase 落在 `'other'` ⇒ **静默不融合**
    （CPU 自测首跑就踩到了这个坑，`assert_strategy_used()` 是它的报警器）。
    """
    orig = dict(prefill=spec_dec._prefill, propose=spec_dec._propose,
                verify=spec_dec._verify, step=spec_dec._step)
    inner = P5V.timed_chunked_prefill(cap["acc"], CHUNK, tgt)

    def prefill(model, ids):
        with M.attention_phase("prefill"):
            lg, cache = inner(model, ids)
        if model is tgt:
            cap["tgt_lg"] = lg.detach().float().cpu()
        return lg, cache

    def propose(*a, **k):
        with M.attention_phase("propose"):
            return orig["propose"](*a, **k)

    def verify(*a, **k):
        with M.attention_phase("verify"):
            return orig["verify"](*a, **k)

    def step(*a, **k):
        with M.attention_phase("repair"):
            return orig["step"](*a, **k)

    spec_dec._prefill, spec_dec._propose, spec_dec._verify, spec_dec._step = (
        prefill, propose, verify, step)
    return orig


def _restore_phases(orig) -> None:
    spec_dec._prefill, spec_dec._propose, spec_dec._verify, spec_dec._step = (
        orig["prefill"], orig["propose"], orig["verify"], orig["step"])


def one_run(tgt, drf, ids, *, strategy: str, kv: str, eos, keep_logits: bool) -> dict:
    """跑一次投机解码，返回 logits（可选）、生成 id、阶段统计、分模型 prefill 耗时。

    **draft 一律 base_bf16**：draft 的 `head_dim=64`，而 `GROUP_V=128`
    （`model_self.py:70,467` 的断言）⇒ `v_fp8` 在 draft 上**结构上就不可能**，
    直接 `AssertionError: head_dim 64 % GROUP_V != 0`。
    这也是全项目既有口径（`g5_stability_probe.py:289` 显式写死 `draft_kv="base_bf16"`，
    其余工具省略该参数走默认值）⇒ 冻结设计里的 `v_fp8` 只压 **target 的 V cache**。
    """
    M.set_attention_strategy(strategy)
    M.reset_attention_stats()
    cap = dict(acc={}, tgt_lg=None)
    orig = _label_phases(tgt, cap)
    t0 = time.perf_counter()
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                         target_kv=kv, draft_kv="base_bf16")
    finally:
        _restore_phases(orig)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    st = M.attention_stats()
    asserted = None
    if strategy != "eager":
        try:
            M.assert_strategy_used()
            asserted = "ok"
        except RuntimeError as e:
            asserted = f"RuntimeError: {e}"
    lg = cap["tgt_lg"] if keep_logits else None
    return dict(strategy=strategy, kv=kv, draft_kv="base_bf16", gen=list(out["gen"]),
                gen_len=len(out["gen"]),
                ids=out["ids"] if keep_logits else None, stats_specdec=out["stats"],
                attention_stats=st, assert_strategy_used=asserted,
                prefill_s=dict(target=cap["acc"].get("target"), draft=cap["acc"].get("draft")),
                wall_s=wall,
                logits_shape=None if lg is None else list(lg.shape),
                logits=lg)


def first_divergence(a: list[int], b: list[int]) -> dict | None:
    """定位首个 id 分歧。**只报事实**，不臆测"近似平局"归因。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return dict(index=i, ref_id=a[i], cand_id=b[i], ref_len=len(a), cand_len=len(b))
    if len(a) != len(b):
        return dict(index=n, ref_id=None, cand_id=None, ref_len=len(a), cand_len=len(b),
                    note="前缀相同但长度不同")
    return None


def compare(ref: dict, cand: dict, *, kind: str, exp_fused: int, exp_short: int) -> dict:
    """`kind='ab'` 才有阶段分流/预登记判据；`kind='control'`（eager vs eager）只作**噪声地板**。

    控制组为什么必须有：没有"同一策略跑两次"的地板，就无法把 `|eager − cudnn|` 归因到 kernel
    （若地板是 0 ⇒ 差异确实来自融合路径；若地板非 0 ⇒ 另有非确定性来源）。
    """
    la, lb = ref.pop("logits", None), cand.pop("logits", None)
    d = dict(kind=kind, kv=ref["kv"], draft_kv=ref.get("draft_kv"),
             strategy_ref=ref["strategy"], strategy_cand=cand["strategy"],
             gen_identical=(ref["gen"] == cand["gen"]),
             divergence=first_divergence(ref["gen"], cand["gen"]),
             stats_cand=cand["attention_stats"], assert_cand=cand["assert_strategy_used"],
             prefill_s_ref=ref["prefill_s"], prefill_s_cand=cand["prefill_s"],
             wall_s_ref=ref["wall_s"], wall_s_cand=cand["wall_s"])
    if la is not None and lb is not None:
        diff = (la - lb).abs()
        abs_err = float(diff.max())
        rel_err, ref_max = _relative(abs_err, la)
        am_a, am_b = la.argmax(-1), lb.argmax(-1)
        mism = (am_a != am_b)
        # 最后一列 logits 决定第一个生成 token，它的**决策余量**决定"id 一致"是稳健还是侥幸
        last_a, last_b = la[0, -1], lb[0, -1]
        top2 = torch.topk(last_a, 2).values
        # `torch.quantile` 有 2^24 元素上限（ctx=128 的 logits 就有 1.9e7 个元素），
        # 因此按固定步长抽样，并把步长**写进结果**（否则分位数口径不可复现）。
        flat = diff.flatten()
        stride = max(1, (flat.numel() + QUANTILE_SAMPLE_MAX - 1) // QUANTILE_SAMPLE_MAX)
        samp = (flat[::stride] if stride > 1 else flat).float()
        qs = torch.quantile(samp, torch.tensor([0.5, 0.9, 0.99],
                                               dtype=torch.float32)).tolist()
        d.update(logits_abs_err=abs_err, logits_rel_err=rel_err, logits_ref_absmax=ref_max,
                 logits_finite=bool(torch.isfinite(lb).all()),
                 logits_mean_abs=float(diff.mean()),
                 diff_p50=qs[0], diff_p90=qs[1], diff_p99=qs[2],
                 diff_quantile_stride=int(stride), diff_elements=int(flat.numel()),
                 argmax_all_equal=bool((am_a == am_b).all()),
                 argmax_equal_frac=float((am_a == am_b).float().mean()),
                 argmax_mismatch_positions=int(mism.sum()),
                 argmax_mismatch_total=int(mism.numel()),
                 last_pos_abs_err=float((last_a - last_b).abs().max()),
                 last_pos_argmax_equal=bool(int(last_a.argmax()) == int(last_b.argmax())),
                 last_pos_ref_argmax=int(last_a.argmax()), last_pos_cand_argmax=int(last_b.argmax()),
                 last_pos_ref_top1=float(top2[0]), last_pos_ref_top2=float(top2[1]),
                 last_pos_ref_margin=float(top2[0] - top2[1]))
        num_ok = (d["logits_finite"] and abs_err <= TOL_ABS and rel_err <= TOL_REL
                  and d["argmax_all_equal"])
    else:
        d.update(logits_abs_err=None, logits_rel_err=None, argmax_all_equal=None)
        num_ok = None
    d["numeric_ok"] = num_ok
    st = cand["attention_stats"]
    d["fused_by_phase"] = st["fused_by_phase"]
    d["eager_by_phase"] = st["eager_by_phase"]
    d["eager_reason_kinds"] = st["eager_reason_kinds"]
    # **口径区分（必须分清楚，否则会把正常 eager 当成回退）**：
    #   prefill_eager_calls      —— prefill 阶段的 eager 调用总数（策略=eager 时它就是全部 prefill）
    #   short_block_fallbacks    —— **只有** s < MIN_S 的短块策略触发（cudnn_prefill 下才有意义）
    #   phase_not_prefill_eager  —— 非 prefill 阶段**本就应该** eager 的调用数
    d["prefill_eager_calls"] = st["eager_by_phase"].get("prefill", 0)
    d["short_block_fallbacks"] = st["eager_reason_kinds"].get("short_block_seq_len_lt_min", 0)
    d["phase_not_prefill_eager"] = st["eager_reason_kinds"].get("phase_not_prefill", 0)
    d["fallback_count_all_eager"] = st["fallback_count"]                  # 含非 prefill 的正常 eager
    d["s_max_by_phase"] = st["s_max_by_phase"]
    d["expected_fused"] = exp_fused
    d["expected_short_block_fallbacks"] = exp_short
    d["fused_counts_exact"] = (st["fused_by_phase"].get("prefill", 0) == exp_fused
                               and d["short_block_fallbacks"] == exp_short)
    if kind == "control":
        d["phase_dispatch_ok"] = None
        d["fused_phases_only_prefill"] = None
        d["verdict"] = "PASS" if (d["gen_identical"] and abs_err == 0.0) else "CHECK"
        d["note"] = "自对照（同策略两次）：用于给出噪声地板，不是 A/B 判据"
        return d
    fused_phases = sorted(k for k, v in st["fused_by_phase"].items() if v > 0)
    d["fused_phases_only_prefill"] = (fused_phases == ["prefill"])
    d["phase_dispatch_ok"] = (d["fused_phases_only_prefill"] and d["assert_cand"] == "ok"
                              and d["fused_counts_exact"])
    if num_ok is True and d["gen_identical"] and d["phase_dispatch_ok"]:
        d["verdict"] = "PASS"
    elif num_ok is False or not d["phase_dispatch_ok"] or d["logits_finite"] is False:
        d["verdict"] = "FAIL"
    else:
        d["verdict"] = "CHECK"       # 例如 logits 一致但 id 不同（或反之）⇒ 必须定位分歧
    return d


def _block_plan(ctx: int) -> dict:
    """分块计划与**精确**的期望次数（不是 `>=` 这种松判据）。

    对每个块，每层每模型各产生 1 次 attention 调用 ⇒ `s ≥ MIN_S` 的块融合、`s < MIN_S` 的块回退。
    """
    blocks = [min(CHUNK, ctx - st) for st in range(0, ctx, CHUNK)]
    return dict(blocks=blocks, n_blocks=len(blocks),
                n_fused_blocks=sum(1 for b in blocks if b >= M.PREFILL_FUSED_MIN_S),
                n_short_blocks=sum(1 for b in blocks if b < M.PREFILL_FUSED_MIN_S))


def model_level(tgt, drf, tok, rec: dict, run_id: str) -> None:
    eos = tok.eos_token_id
    full = P5V.load_ids(tok)
    nl_t, nl_d = len(tgt.layers), len(drf.layers)
    rec["model_level_layer_counts"] = dict(target=nl_t, draft=nl_d)
    rec["draft_kv_note"] = ("draft 固定 base_bf16：draft head_dim=64 而 GROUP_V=128 ⇒ v_fp8 在 "
                            "draft 上直接 AssertionError；与全项目口径一致（v_fp8 只压 target 的 V）")
    comps: list[dict] = []
    # **原地挂到 rec 并逐条落盘**：上一次尝试就是因为只在函数末尾赋值，
    # 崩溃后 `model_level` 整段丢失（只剩控制台一行）。教训同 variance probe 的 `stage=formal_done`。
    rec["model_level"] = comps
    for ctx in CTXS:
        kvs = KVS if ctx in CTXS_2KV else KVS[:1]
        ids = full[:, :ctx].contiguous()
        plan = _block_plan(ctx)
        for kv in kvs:
            exp_fused = (nl_t + nl_d) * plan["n_fused_blocks"]
            exp_short = (nl_t + nl_d) * plan["n_short_blocks"]
            ref = one_run(tgt, drf, ids, strategy="eager", kv=kv, eos=eos, keep_logits=True)
            print(f"[acv] model kv={kv:<9} ctx={ctx:<5} ref(eager) gen={ref['gen_len']} "
                  f"plan={plan['blocks']} prefill(target={ref['prefill_s']['target']:.2f}s "
                  f"draft={ref['prefill_s']['draft']:.2f}s)", flush=True)
            # ---- 自对照：eager vs eager（噪声地板；不是 A/B 判据）----
            ctrl = one_run(tgt, drf, ids, strategy="eager", kv=kv, eos=eos, keep_logits=True)
            c = compare(dict(ref), ctrl, kind="control", exp_fused=0, exp_short=0)
            c.update(ctx=ctx, rep="control")
            comps.append(c)
            _write(rec, run_id)
            print(f"[acv] CTRL  kv={kv:<9} ctx={ctx:<5} eager_vs_eager "
                  f"abs={c['logits_abs_err']} gen_identical={c['gen_identical']} "
                  f"-> {c['verdict']}（噪声地板）", flush=True)
            del ctrl
            for rep in range(REPS):
                cand = one_run(tgt, drf, ids, strategy="cudnn_prefill", kv=kv, eos=eos,
                               keep_logits=True)
                cmp_ = compare(dict(ref), cand, kind="ab", exp_fused=exp_fused, exp_short=exp_short)
                cmp_.update(ctx=ctx, rep=rep, block_plan=plan)
                comps.append(cmp_)
                _write(rec, run_id)              # ← 每条落盘：崩溃不丢已完成的行
                print(f"[acv] model kv={kv:<9} ctx={ctx:<5} rep={rep} "
                      f"gen_identical={cmp_['gen_identical']} "
                      f"abs={cmp_['logits_abs_err']:.3e} rel={cmp_['logits_rel_err']:.3e} "
                      f"argmax_eq={cmp_['argmax_equal_frac']:.4f} "
                      f"lastpos={cmp_['last_pos_abs_err']:.2e}"
                      f"(margin={cmp_['last_pos_ref_margin']:.3f}) "
                      f"fused={cmp_['fused_by_phase'].get('prefill', 0)}/{exp_fused} "
                      f"short_fb={cmp_['short_block_fallbacks']}/{exp_short} "
                      f"all_eager={cmp_['fallback_count_all_eager']} "
                      f"-> {cmp_['verdict']}", flush=True)
                del cand
            del ref
            torch.cuda.empty_cache()
    ab = [c for c in comps if c["kind"] == "ab"]
    rec["model_verdict"] = ("PASS" if ab and all(c["verdict"] == "PASS" for c in ab)
                            else ("FAIL" if any(c["verdict"] == "FAIL" for c in ab) else "CHECK"))


# --------------------------------------------------------------------------- 主流程
def _out_path(run_id: str) -> Path:
    return OUTDIR / f"attn_cudnn_validate_{run_id}.json"


def _write(rec: dict, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _out_path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="g5_attn_cudnn_validate",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), env=env_facts(),
                     preregistered=dict(tol_abs=TOL_ABS, tol_rel=TOL_REL,
                                        fused_transient_max_mib=FUSED_TRANSIENT_MAX_MIB,
                                        fused_over_eager_max=FUSED_RATIO_MAX,
                                        mem_rows_min_s=MEM_ROWS_MIN_S,
                                        prefill_fused_min_s=M.PREFILL_FUSED_MIN_S,
                                        short_block_policy=M.SHORT_BLOCK_POLICY,
                                        requested_backend=M.FUSED_BACKEND_NAME,
                                        gen_ids_must_be_identical=True,
                                        no_post_hoc_relaxation=True),
                     config=dict(stage=STAGE, chunk=CHUNK, max_new=MAXNEW, k=K, reps=REPS,
                                 ctxs=CTXS, ctxs_2kv=sorted(CTXS_2KV), kvs=KVS,
                                 op_rows=[list(r) for r in OP_ROWS],
                                 op_configs=[list(c) for c in OP_CONFIGS]))
    print(f"[acv] run_id={run_id} env={rec['env']}", flush=True)
    if DRY_RUN:
        rec["stage"] = "dry_run"
        _write(rec, run_id)
        print(f"[acv] DRY RUN -> {_out_path(run_id)}", flush=True)
        return 0

    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=ok, why=why, system=P5V.system_memory())
    print(f"[acv] commit guard: {why}", flush=True)
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    print(f"[acv] P-state: qualified={pf.get('qualified')} sm={pf.get('sm_median')} "
          f"mem={pf.get('mem_median')} reason={pf.get('reason')}", flush=True)

    try:
        if STAGE in ("ops", "all"):
            rec["gpu_consumers_before_ops"] = P5V.gpu_consumers()
            rec["alloc_before_ops"] = P5V.allocator_state()
            op_level(rec)
            kernel_proof(rec)
            rec["alloc_after_ops"] = P5V.allocator_state()

        if STAGE in ("model", "all"):
            if not ok and not ALLOW_UNQUALIFIED:
                rec["skipped"] = f"commit 余量不足 -> 未加载模型：{why}"
                print(f"[acv] !! 拒绝加载模型（{why}）", flush=True)
            elif pf.get("qualified") is not True and not ALLOW_UNQUALIFIED:
                rec["skipped"] = f"P-state 不合格 -> 未加载模型：{pf.get('reason')}"
                print(f"[acv] !! 拒绝加载模型（{rec['skipped']}）", flush=True)
            else:
                tgt, tok = load_ours(TARGET)
                drf, _ = load_ours(DRAFT)
                rec["alloc_after_load"] = P5V.allocator_state()
                rec["system_after_load"] = P5V.system_memory()
                rec["gpu_consumers_after_load"] = P5V.gpu_consumers()
                try:
                    model_level(tgt, drf, tok, rec, run_id)
                finally:
                    del tgt, drf
                    torch.cuda.empty_cache()
        rec["stage"] = "complete"
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        rec["alloc_on_crash"] = P5V.allocator_state()
        _write(rec, run_id)
        print(f"[acv] !! 崩溃（已落盘 {_out_path(run_id)}）：\n{rec['traceback']}", flush=True)
        return 2
    finally:
        M.set_attention_strategy("eager")
        M.reset_attention_stats()

    rec["verdict"] = dict(op_level=rec.get("op_verdict"),
                          model_level=rec.get("model_verdict"),
                          kernel_proof=rec.get("kernel_proof", {}).get("matched"))
    _write(rec, run_id)
    print(f"[acv] verdict={rec['verdict']}\n[acv] -> {_out_path(run_id)}", flush=True)
    return 0 if rec.get("model_verdict", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
