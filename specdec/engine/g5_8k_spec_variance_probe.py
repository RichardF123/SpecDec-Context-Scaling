"""g5_8k_spec_variance_probe.py — **8K SpecDec 跨进程速度双档专项定位**（诊断工具，不是吞吐测量）。

要回答的问题（**在拿到数据前不写结论**）
  1. 8K SpecDec 的"快档/慢档"是否真的形成**双峰**（跨进程）；
  2. 每个进程**内部**是否稳定；
  3. 差异集中在**哪一阶段**（draft 提议 / 块验证 / 修复步 / cache 裁剪 / KV cat / control）；
  4. prefill 是否始终稳定；
  5. 两个对照（AR target-only、AR dual-resident）是否稳定；
  6. 8K SpecDec 的波动是否与 ① cache 增长 ② 某个 forward 阶段 ③ allocator/reserved
     ④ WDDM/共享显存 ⑤ 其它系统状态 同步变化。

纪律（违反即数据作废）
  - **profiling 与正式计时分离**：先跑**无插桩**的正式 decode，再**单独**跑插桩 pass；
    profiling 数字**不得**写进任何正式 speedup 表；
  - 计时区间内**不轮询 `nvidia-smi`**、不 `empty_cache`、不做文件 IO；
  - `kv_cat` 是 forward 的**子集**，标记为 subset 且**不与 forward 总时间相加**；
  - 对照组的角色是 **diagnostic_control / reference**，**不作正式 speedup 分母**；
  - 全部 K 个进程都必须 P-state 合格；不合格则明确标 `pstate_qualified=False`（不静默采样）。

双峰判定规则**预登记**（在跑之前写死，且不作为任何 Gate）：
  把各进程的 SpecDec decode 中位数排序；若存在一个切分点使两侧**各 ≥2 个进程**且
  `min(高档)/max(低档) ≥ P5V_BIMODAL_RATIO`（默认 1.5），则报告"检出双档"并给出切分；
  否则报告"单档"。该阈值是可配置的诊断阈值，**不是**统计显著性判据。

用法（交互式 GPU 终端；约 60–90 分钟，可缩小）
  python -m specdec.engine.g5_8k_spec_variance_probe
可选环境变量
  P5V_PROCS=4        每个角色独立进程数（≥4 才有意义）
  P5V_REPS=5         正式 pass 的 measured reps（warm-up 固定 1）
  P5V_PROF_REPS=1    插桩 pass 的重复次数
  P5V_ROLES=spec,ar_target,ar_dual
  P5V_CTX=8192 P5V_KV=base_bf16 P5V_K=4 P5V_MAXNEW=64 P5V_CHUNK=2048
  P5V_COOLDOWN=10    进程之间的冷却秒数（不计时）
  P5V_WDDM=1         尽力采集 WDDM/共享显存计数器（只在计时区外调用）
  P5V_ALLOW_UNQUALIFIED=1  允许 P-state 不合格仍跑（数据只能算探索性）
  P5V_DRY_RUN=1      只打印计划
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import spec_dec, spec_cache
from . import g5_driver as D
from . import gpu_pstate_preflight as PF
from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "variance"

CTX = int(os.environ.get("P5V_CTX", "8192"))
KV = os.environ.get("P5V_KV", "base_bf16")
K = int(os.environ.get("P5V_K", str(spec_dec.K)))
MAXNEW = int(os.environ.get("P5V_MAXNEW", "64"))
CHUNK = int(os.environ.get("P5V_CHUNK", "2048"))
REPS = int(os.environ.get("P5V_REPS", "5"))
WARM = int(os.environ.get("P5V_WARM", "1"))
PROCS = int(os.environ.get("P5V_PROCS", "4"))
PROF_REPS = int(os.environ.get("P5V_PROF_REPS", "1"))
COOLDOWN = float(os.environ.get("P5V_COOLDOWN", "10"))
ROLES = [r.strip() for r in os.environ.get("P5V_ROLES", "spec,ar_target,ar_dual").split(",")
         if r.strip()]
WDDM = S.env_flag("P5V_WDDM", True)
ALLOW_UNQUALIFIED = S.env_flag("P5V_ALLOW_UNQUALIFIED", False)
BIMODAL_RATIO = float(os.environ.get("P5V_BIMODAL_RATIO", "1.5"))

ROLE_SPEC = "spec"            # SpecDec（被测对象）
ROLE_AR_TARGET = "ar_target"  # AR target-only（正式基线口径的参照，**不是**本工具的声明分母）
ROLE_AR_DUAL = "ar_dual"      # AR dual-resident（**诊断控制**，永不作分母）
ROLE_CHOICES = (ROLE_SPEC, ROLE_AR_TARGET, ROLE_AR_DUAL)

# kv_cat 是 forward 的子集：列出来是为了**永不**把它计入闭合求和
SUBSET_PHASES = ("kv_cat",)


# --------------------------------------------------------------------------- 阶段计时
class Phase:
    """一个阶段的 CUDA Event 累积器（每次调用 record 一对；结束时统一 elapsed_time）。

    注意：Phase 只在**插桩 pass** 里使用；正式 pass 完全不包装任何函数
    （所以正式数字里没有插桩开销，插桩数字里必然有）。
    """

    def __init__(self, name: str):
        self.name = name
        self._pairs: list[tuple] = []

    def wrap(self, fn):
        def inner(*a, **k):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            r = fn(*a, **k)
            e1.record()
            self._pairs.append((e0, e1))
            return r
        return inner

    def stats(self) -> dict:
        if not self._pairs:
            return dict(total_ms=0.0, calls=0, median_ms=None)
        per = [a.elapsed_time(b) for a, b in self._pairs]
        return dict(total_ms=sum(per), calls=len(per), median_ms=statistics.median(per))


def _kv_cat_wrapper(phase: Phase, orig_cat):
    """只拦 KV 增长的 `torch.cat`（list/tuple 两个 4 维张量 + dim=2）。"""
    def inner(tensors, *a, **k):
        dim = k.get("dim", a[0] if a else None)
        is_kv = (isinstance(tensors, (list, tuple)) and len(tensors) == 2
                 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in tensors)
                 and dim == 2)
        if not is_kv:
            return orig_cat(tensors, *a, **k)
        return phase.wrap(orig_cat)(tensors, *a, **k)
    return inner


# --------------------------------------------------------------------------- 输入与 prefill
def load_ids(tok):
    """与其它 G5 工具同一段 wikitext 文本（嵌套前缀口径：8K 是 2K/4K 的超集）。"""
    import zipfile
    z = ROOT / "data" / "wikitext-2-raw-v1.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if "test" in n and n.endswith(".raw"))
        raw = zf.read(name).decode("utf-8", errors="ignore")
    text = " ".join(ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not ln.strip().startswith("="))
    return tok(text[:600_000], return_tensors="pt").input_ids[:, :CTX]


def chunked_prefill(model, ids, chunk=CHUNK):
    with torch.inference_mode():
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, chunk):
            lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                   need_logits=(st + chunk >= n))
    return lg, cache


def timed_chunked_prefill(acc: dict, chunk: int = CHUNK):
    """返回一个"**分块** + 计时"的 `spec_dec._prefill` 替身；`acc["t"]` 累积 prefill 秒数。

    **这是本探针最容易写错的地方，务必理解**：
      `spec_dec.spec_decode_greedy` 内部会调用 `spec_dec._prefill(model, ids)`，
      而引擎默认的 `_prefill` 是 **一次前向吃完整个 prompt**。8K 时那一层的 fp32 softmax
      会一次性申请 **≈3.00 GiB**，在 8 GB 卡上直接 `torch.OutOfMemoryError`
      （2026-09-22 本工具首跑就是这么崩的）。
      因此：
        ① **不要**在调用 `spec_decode_greedy` 之前自己先跑一次 `chunked_prefill`——
           既浪费显存，又让 prefill/decode 的划分变成假的（引擎会**再** prefill 一次）；
        ② 正确做法与测量核心 `g4_part4_pilot_measure.spec_decode_timed` 完全一致：
           把 `spec_dec._prefill` 换成这里的**分块**版本并单独计时，再调用一次
           `spec_decode_greedy`，`decode = total − prefill`。
    """
    def timed(model, ids_):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg, cache = chunked_prefill(model, ids_, chunk)
        torch.cuda.synchronize()
        acc["t"] += time.perf_counter() - t0
        return lg, cache
    return timed


# --------------------------------------------------------------------------- 内存 / 系统状态
def allocator_state() -> dict:
    """allocator 段/池状态（**关键是 `inactive_split` 与 allocation retries**）。"""
    out = {"available": False}
    try:
        st = torch.cuda.memory_stats()
        out.update({
            "available": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
            "reserved_mib": torch.cuda.memory_reserved() / 2 ** 20,
            "allocated_mib": torch.cuda.memory_allocated() / 2 ** 20,
            "active_peak_mib": st.get("active_bytes.all.peak", 0) / 2 ** 20,
            "inactive_split_peak_mib": st.get("inactive_split_bytes.all.peak", 0) / 2 ** 20,
            "segment_peak_mib": st.get("segment.all.peak", 0) / 2 ** 20,
            "num_alloc_retries": st.get("num_alloc_retries"),
            "num_ooms": st.get("num_ooms"),
            "num_device_allocations": st.get("num_device_allocations"),
        })
        try:
            free, total = torch.cuda.mem_get_info()
            out["cuda_free_mib"] = free / 2 ** 20
            out["cuda_total_mib"] = total / 2 ** 20
        except Exception:                                # noqa: BLE001
            pass
    except Exception as e:                               # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def wddm_state() -> dict:
    """**尽力**采集 WDDM / 共享显存相关计数；取不到就 `available=False`（绝不填 0）。

    只在**计时区之外**调用（会起一次 PowerShell，耗时数百毫秒）。
    Windows 的"共享 GPU 内存"没有 nvidia-smi 字段，只能靠性能计数器；
    这里把它当**可选证据**，拿不到不影响其它结论。
    """
    out = {"available": False, "fields": {}, "note": ""}
    if os.name != "nt":
        out["note"] = "非 Windows：WDDM 概念不适用"
        return out
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Counter -Counter '\\GPU Adapter Memory(*)\\Dedicated Usage',"
             "'\\GPU Adapter Memory(*)\\Shared Usage' -MaxSamples 1 -ErrorAction Stop | "
             "ForEach-Object { $_.CounterSamples | ForEach-Object { "
             "$_.Path + '=' + $_.CookedValue } }"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=25)
        lines = [ln.strip() for ln in (ps.stdout or "").splitlines() if "=" in ln]
        for ln in lines:
            path, _, val = ln.rpartition("=")
            key = path.strip().strip("\\").split("\\")[-1].lower().replace(" ", "_")
            try:
                out["fields"][f"gpu_adapter_{key}"] = float(val)
            except ValueError:
                continue
        out["available"] = bool(out["fields"])
        if not out["available"]:
            out["note"] = f"性能计数器未返回可用数值（exit={ps.returncode}）"
    except Exception as e:                               # noqa: BLE001
        out["note"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- 两个 pass
def formal_pass(tgt, drf, ids, eos, role: str) -> dict:
    """**无插桩**正式计时：warm-up 不计时，cooldown 不计时，计时区内无任何轮询/IO。"""
    reps = []

    def one() -> dict:
        if role == ROLE_SPEC:
            # SpecDec：**让引擎自己做 prefill**，只把 `_prefill` 换成分块+计时版本
            # （见 timed_chunked_prefill 的说明；直接调 chunked_prefill 会 OOM 且划分失真）
            acc = {"t": 0.0}
            orig_prefill = spec_dec._prefill
            spec_dec._prefill = timed_chunked_prefill(acc, CHUNK)
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                                  target_kv=KV)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
            finally:
                spec_dec._prefill = orig_prefill
            gen, stats = out["gen"], out["stats"]
            pre_s, dec_s = acc["t"], (t2 - t0) - acc["t"]
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg, cache = chunked_prefill(tgt, ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            gen, cur, stats = [], lg[:, -1:], None
            for i in range(MAXNEW):
                tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
                gen.append(tok)
                if eos is not None and tok == eos:
                    break
                if i == MAXNEW - 1:
                    break
                cur, cache = spec_dec._step(tgt, tok, cache)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            pre_s, dec_s = t1 - t0, t2 - t1
        rec = dict(prefill_s=pre_s, decode_s=dec_s, total_s=pre_s + dec_s,
                   n_gen=len(gen), decode_tok_s=len(gen) / max(dec_s, 1e-9),
                   e2e_tok_s=len(gen) / max(pre_s + dec_s, 1e-9),
                   generated_token_hash=_gen_hash(gen))
        if stats:
            rec.update(dict(rounds=stats.get("rounds"), proposed=stats.get("proposed"),
                            accepted=stats.get("accepted"),
                            mismatch_rounds=stats.get("mismatch_rounds"),
                            draft_calls=stats.get("draft_calls"),
                            target_block_calls=stats.get("target_block_calls"),
                            all_accept_rounds=stats.get("all_accept_rounds")))
        return rec

    for w in range(WARM):
        r = one()
        print(f"    warm{w}: decode={r['decode_s']:.2f}s prefill={r['prefill_s']:.2f}s",
              flush=True)
    for i in range(REPS):
        r = one()
        reps.append(r)
        print(f"    rep{i}: decode={r['decode_s']:.2f}s prefill={r['prefill_s']:.2f}s "
              f"n_gen={r['n_gen']} dec_tok/s={r['decode_tok_s']:.2f}", flush=True)
    med = {k: statistics.median([r[k] for r in reps]) for k in
           ("prefill_s", "decode_s", "total_s", "decode_tok_s", "e2e_tok_s")}
    rel = (100.0 * (max(r["decode_s"] for r in reps) - min(r["decode_s"] for r in reps))
           / max(med["decode_s"], 1e-9))
    return dict(reps=reps, medians=med, decode_rel_spread_pct=rel)


def _gen_hash(gen) -> str:
    import hashlib
    return hashlib.sha1(json.dumps([int(x) for x in gen]).encode()).hexdigest()[:16]


def profiling_pass(tgt, drf, ids, eos, role: str, reps: int) -> dict:
    """**插桩** pass（**单独运行**，绝不用它的时间做正式结论）。"""
    rows = []
    for i in range(reps):
        if role == ROLE_SPEC:
            ph = {n: Phase(n) for n in ("draft_propose", "target_block_verify",
                                        "target_replacement_step", "draft_replacement_step",
                                        "cache_trim_commit", "kv_cat", "prefill_target")}
            orig = dict(propose=spec_dec._propose, verify=spec_dec._verify, step=spec_dec._step,
                        trim=spec_cache.trim_cache, cat=torch.cat,
                        prefill=spec_dec._prefill)

            def step_profiled(model, token, cache):
                name = ("target_replacement_step" if model is tgt
                        else "draft_replacement_step")
                return ph[name].wrap(orig["step"])(model, token, cache)

            def prefill_profiled(model, ids_):
                # 同样必须**分块**：默认 `_prefill` 在 8K 会一次申请 ≈3.00 GiB 而 OOM
                return ph["prefill_target"].wrap(chunked_prefill)(model, ids_, CHUNK)

            spec_dec._propose = ph["draft_propose"].wrap(orig["propose"])
            spec_dec._verify = ph["target_block_verify"].wrap(orig["verify"])
            spec_dec._step = step_profiled
            spec_dec._prefill = prefill_profiled
            spec_cache.trim_cache = ph["cache_trim_commit"].wrap(orig["trim"])
            torch.cat = _kv_cat_wrapper(ph["kv_cat"], orig["cat"])
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                                  target_kv=KV)
                torch.cuda.synchronize()
                outer = time.perf_counter() - t0
                stats = out["stats"]
            finally:
                spec_dec._propose, spec_dec._verify = orig["propose"], orig["verify"]
                spec_dec._step, spec_cache.trim_cache = orig["step"], orig["trim"]
                spec_dec._prefill = orig["prefill"]
                torch.cat = orig["cat"]
        else:
            ph = {n: Phase(n) for n in ("target_single_step", "kv_cat", "prefill_target")}
            orig = dict(step=spec_dec._step, cat=torch.cat)
            spec_dec._step = ph["target_single_step"].wrap(orig["step"])
            torch.cat = _kv_cat_wrapper(ph["kv_cat"], orig["cat"])
            try:
                t0 = time.perf_counter()
                lg, cache = ph["prefill_target"].wrap(chunked_prefill)(tgt, ids, CHUNK)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                gen, cur = [], lg[:, -1:]
                for j in range(MAXNEW):
                    tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
                    gen.append(tok)
                    if eos is not None and tok == eos:
                        break
                    if j == MAXNEW - 1:
                        break
                    cur, cache = spec_dec._step(tgt, tok, cache)
                torch.cuda.synchronize()
                outer = time.perf_counter() - t1
                stats = dict(n_gen=len(gen))
            finally:
                spec_dec._step, torch.cat = orig["step"], orig["cat"]
        phases = {n: p.stats() for n, p in ph.items()}
        rows.append(dict(phases=phases, outer_decode_s=outer, stats=stats))
        print(f"    prof{i}: outer_decode={outer:.2f}s", flush=True)

    names = rows[0]["phases"].keys()
    agg = {}
    for n in names:
        agg[n] = dict(
            total_ms=statistics.median([x["phases"][n]["total_ms"] for x in rows]),
            calls=int(statistics.median([x["phases"][n]["calls"] for x in rows])),
            median_ms=statistics.median([x["phases"][n]["median_ms"] for x in rows
                                         if x["phases"][n]["median_ms"] is not None] or [0]) or None)
    outer = statistics.median([x["outer_decode_s"] for x in rows])
    additive = sum(v["total_ms"] for n, v in agg.items() if n not in SUBSET_PHASES)
    st = rows[0]["stats"] or {}
    rounds = st.get("rounds") or st.get("n_gen") or 0
    return dict(
        phases=agg, outer_decode_s=outer, subset_phases=list(SUBSET_PHASES),
        accounted_ms=additive, residual_ms=outer * 1000.0 - additive,
        residual_pct=100.0 * (outer * 1000.0 - additive) / max(outer * 1000.0, 1e-9),
        per_round_ms={n: (v["total_ms"] / rounds if rounds else None) for n, v in agg.items()},
        rounds=rounds, stats=st,
        note="插桩 pass：数字含插桩开销，**不得**当正式吞吐；kv_cat 是 forward 子集、不可相加")


# --------------------------------------------------------------------------- 子进程
def child_main(role: str, index: int, run_id: str) -> int:
    assert role in ROLE_CHOICES, role
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained

    print(f"[p5v] child role={role} index={index} run_id={run_id} ctx={CTX} kv={KV} k={K} "
          f"max_new={MAXNEW} reps={REPS} warm={WARM}", flush=True)
    rec: dict = dict(run_id=run_id, role=role, index=index, ctx=CTX, kv=KV, k=K,
                     max_new=MAXNEW, chunk=CHUNK, reps=REPS, warm=WARM,
                     target_model=TARGET, draft_model=DRAFT, compute_dtype="bfloat16",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT))

    # ---- 环境资格（**在正式计时之前**，且只采负载期样本）----
    pf = PF.qualify()
    rec["pstate"] = pf
    print(f"[p5v] P-state: available={pf.get('available')} qualified={pf.get('qualified')} "
          f"sm_med={pf.get('sm_median')} mem_med={pf.get('mem_median')} "
          f"power={pf.get('power_median')} reason={pf.get('reason')}", flush=True)
    if pf.get("qualified") is not True and not ALLOW_UNQUALIFIED:
        rec["skipped"] = "P-state 不合格 -> 未启动计时（数据不产生）"
        _write_child(rec, role, index, run_id)
        print("[p5v] 环境不合格 -> 跳过（P5V_ALLOW_UNQUALIFIED=1 可强制）", flush=True)
        return 3

    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    ids = load_ids(tok)
    rec["input_token_hash"] = _gen_hash([int(x) for x in ids[0].tolist()])
    rec["input_tokens"] = int(ids.shape[1])

    before = dict(system=D.system_snapshot(), wddm=(wddm_state() if WDDM else None))
    tgt, _ = load_ours(TARGET)
    drf = None
    if role == ROLE_SPEC:
        drf, _ = load_ours(DRAFT)
        rec["resident_models"] = ["target", "draft"]
    elif role == ROLE_AR_DUAL:
        drf, _ = load_ours(DRAFT)
        rec["resident_models"] = ["target", "draft"]
    else:
        rec["resident_models"] = ["target"]
    rec["alloc_after_load"] = allocator_state()
    print(f"[p5v] 驻留={rec['resident_models']} "
          f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    tgt.set_kv_format(spec_dec._fmt(KV))
    rec["formal"] = formal_pass(tgt, drf, ids, eos, role)
    rec["alloc_after_formal"] = allocator_state()
    print(f"[p5v] 正式：decode 中位 {rec['formal']['medians']['decode_s']:.2f}s "
          f"prefill 中位 {rec['formal']['medians']['prefill_s']:.2f}s "
          f"rel={rec['formal']['decode_rel_spread_pct']:.1f}%", flush=True)

    if PROF_REPS > 0:
        rec["profiling"] = profiling_pass(tgt, drf, ids, eos, role, PROF_REPS)
        p = rec["profiling"]
        print(f"[p5v] 插桩：outer={p['outer_decode_s']:.2f}s "
              f"残差={p['residual_ms']:.0f}ms({p['residual_pct']:.1f}%)", flush=True)

    after = dict(system=D.system_snapshot(), wddm=(wddm_state() if WDDM else None))
    rec["monitor_before"], rec["monitor_after"] = before, after
    rec["alloc_final"] = allocator_state()
    _write_child(rec, role, index, run_id)
    del tgt, drf
    return 0


def _child_path(role: str, index: int, run_id: str) -> Path:
    return OUTDIR / f"p5v_{run_id}_{role}_{index}.json"


def _write_child(rec: dict, role: str, index: int, run_id: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _child_path(role, index, run_id)
    if p.exists():
        raise SystemExit(f"[拒绝覆盖] {p} 已存在；换 run_id")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p5v] 写出 {p}", flush=True)


# --------------------------------------------------------------------------- 父进程
def bimodality(values: list[float], ratio: float) -> dict:
    """**预登记**的双档检出规则（诊断用，不是显著性检验）。"""
    v = sorted(x for x in values if x is not None)
    out = {"n": len(v), "sorted": v, "detected": False, "split": None,
           "gap_ratio": None, "rule": f"两侧各≥2 且 min(高档)/max(低档) ≥ {ratio:g}"}
    if len(v) < 4:
        out["note"] = "进程数 <4，不做双档判定"
        return out
    best = None
    for i in range(2, len(v) - 1):
        lo, hi = v[:i], v[i:]
        if len(lo) < 2 or len(hi) < 2:
            continue
        r = hi[0] / lo[-1] if lo[-1] else None
        if r is not None and (best is None or r > best[0]):
            best = (r, i)
    if best is None:
        return out
    out["gap_ratio"] = best[0]
    out["split"] = dict(low=v[:best[1]], high=v[best[1]:], ratio=best[0])
    out["detected"] = bool(best[0] >= ratio)
    return out


def _load_children(run_id: str) -> list[dict]:
    out = []
    for role in ROLES:
        for i in range(PROCS):
            p = _child_path(role, i, run_id)
            if p.exists():
                out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def estimate_minutes() -> float:
    """**角色感知**的粗略时间估计。

    8K 下 prefill 主导，且 AR（单模型）比 SpecDec（双模型）便宜得多，
    所以不能用"所有角色一样贵"来估（会把总时长高估近一倍）。
    实测参考（缩小版 run `gstab_20260922T040625Z`）：
      SpecDec 8K：prefill 31.96 s + decode 6.7–11.7 s ⇒ 每 rep ≈ 42 s
      AR 8K target-only：prefill 3.45 s + decode 2.2 s ⇒ 每 rep ≈ 5.7 s
    """
    per_rep = {ROLE_SPEC: 42.0, ROLE_AR_TARGET: 5.7, ROLE_AR_DUAL: 5.7}
    total_s = 0.0
    for role in ROLES:
        per_pass = (WARM + REPS + PROF_REPS) * per_rep.get(role, 42.0)
        total_s += PROCS * (PF.SECONDS + 25.0 + per_pass)     # 预检 + 加载 + 全部 pass
    total_s += max(len(ROLES) * PROCS - 1, 0) * COOLDOWN
    return total_s / 60.0


def parent_main() -> int:
    run_id = S.default_run_id("p5v")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[p5v] run_id={run_id} role×procs = {ROLES} × {PROCS} = {len(ROLES) * PROCS} 个独立进程")
    print(f"[p5v] 固定：ctx={CTX} kv={KV} k={K}(冻结) max_new={MAXNEW} chunk={CHUNK} "
          f"warm={WARM} reps={REPS} 插桩 pass：{PROF_REPS} 次（**与正式计时分开**）")
    print(f"[p5v] 对照口径：ar_target=正式基线参照（本工具**不**据此做正式声明）；"
          f"ar_dual=诊断控制，永不作分母")
    print(f"[p5v] 预计 ≈{estimate_minutes():.0f} 分钟（角色感知估算：SpecDec 每 rep ≈42s、"
          f"AR 每 rep ≈6s；实际受加载与冷却影响）\n", flush=True)

    order = []
    for i in range(PROCS):
        for role in ROLES:
            order.append((role, i))
    for n, (role, i) in enumerate(order):
        print(f"[p5v] ({n + 1}/{len(order)}) 启动独立进程 role={role} index={i}", flush=True)
        env = os.environ.copy()
        env.update(dict(HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", P5V_CHILD="1",
                        P5V_ROLE=role, P5V_INDEX=str(i), P5V_RUN_ID=run_id))
        r = subprocess.run([sys.executable, "-m", "specdec.engine.g5_8k_spec_variance_probe"],
                           env=env, cwd=str(ROOT))
        if r.returncode not in (0, 3):
            print(f"[p5v] !! 子进程退出码 {r.returncode}（数据可能不完整）", flush=True)
        if n < len(order) - 1 and COOLDOWN > 0:
            time.sleep(COOLDOWN)

    recs = _load_children(run_id)
    print(f"\n[p5v] 收集到 {len(recs)} 个子进程结果")
    report = analyse(recs)
    print_report(report)
    sorted_recs = sorted(recs, key=lambda x: (x["role"], x["index"]))
    _write_summary(run_id, dict(run_id=run_id, config=dict(
        ctx=CTX, kv=KV, k=K, max_new=MAXNEW, chunk=CHUNK, warm=WARM, reps=REPS,
        prof_reps=PROF_REPS, procs=PROCS, roles=ROLES, bimodal_ratio=BIMODAL_RATIO),
        children=sorted_recs, analysis=report))
    return 0


def _med(xs):
    v = [x for x in xs if x is not None]
    return statistics.median(v) if v else None


def analyse(recs: list[dict]) -> dict:
    """回答 6 个问题。**只报告观察，不下结论性归因**。"""
    out: dict = {}
    by_role = {}
    for r in recs:
        by_role.setdefault(r["role"], []).append(r)
    out["Q0_per_process"] = sorted(recs, key=lambda x: (x["role"], x["index"]))

    def dec_of(r):
        f = r.get("formal")
        return f["medians"]["decode_s"] if f else None

    # Q1 双峰
    spec_dec_s = [dec_of(r) for r in by_role.get(ROLE_SPEC, [])]
    tok = [r["formal"]["medians"]["decode_tok_s"] for r in by_role.get(ROLE_SPEC, [])
           if r.get("formal")]
    out["Q1_bimodality"] = dict(
        decode_seconds=bimodality(spec_dec_s, BIMODAL_RATIO),
        decode_tok_per_s=bimodality(tok, BIMODAL_RATIO),
        max_over_min=(max(tok) / min(tok)) if len(tok) >= 2 and min(tok) else None)

    # Q2 进程内稳定性
    out["Q2_within_process"] = {
        r["role"] + f"[{r['index']}]": dict(
            decode_rel_spread_pct=r["formal"]["decode_rel_spread_pct"],
            decode_s_min=min(x["decode_s"] for x in r["formal"]["reps"]),
            decode_s_max=max(x["decode_s"] for x in r["formal"]["reps"]))
        for r in recs if r.get("formal")}

    # Q3 阶段差异（把 spec 进程按 decode 中位数排序，比较首尾）
    sp = sorted([r for r in by_role.get(ROLE_SPEC, []) if r.get("profiling")],
                key=lambda r: dec_of(r) or 0)
    if len(sp) >= 2:
        fast, slow = sp[0], sp[-1]
        pf_, ps_ = fast["profiling"]["phases"], slow["profiling"]["phases"]
        out["Q3_phase_contrast"] = dict(
            fast=dict(index=fast["index"], decode_s=dec_of(fast)),
            slow=dict(index=slow["index"], decode_s=dec_of(slow)),
            per_phase_ms={n: dict(fast=pf_[n]["total_ms"], slow=ps_[n]["total_ms"],
                                  ratio=(ps_[n]["total_ms"] / pf_[n]["total_ms"]
                                         if pf_[n]["total_ms"] else None))
                          for n in pf_},
            residual_ms=dict(fast=fast["profiling"]["residual_ms"],
                             slow=slow["profiling"]["residual_ms"]),
            per_round_ms_fast=fast["profiling"]["per_round_ms"],
            per_round_ms_slow=slow["profiling"]["per_round_ms"],
            subset_phases=fast["profiling"]["subset_phases"])

    # Q4 prefill 稳定性
    out["Q4_prefill"] = {}
    for role in ROLE_CHOICES:
        vals = [r["formal"]["medians"]["prefill_s"] for r in by_role.get(role, [])
                if r.get("formal")]
        v = [x for x in vals if x]
        out["Q4_prefill"][role] = dict(
            medians=vals,
            max_over_min=((max(v) / min(v)) if len(v) >= 2 and min(v) else None))

    # Q5 对照稳定性
    out["Q5_controls"] = {role: dict(
        decode_tok_per_s=[r["formal"]["medians"]["decode_tok_s"]
                          for r in by_role.get(role, []) if r.get("formal")],
        within_rel_spread_pct=[r["formal"]["decode_rel_spread_pct"]
                               for r in by_role.get(role, []) if r.get("formal")])
        for role in (ROLE_AR_TARGET, ROLE_AR_DUAL)}

    # Q6 与各状态量同步变化
    rows = []
    for r in by_role.get(ROLE_SPEC, []):
        if not r.get("formal"):
            continue
        a = r.get("alloc_after_formal") or {}
        m = (r.get("monitor_after") or {}).get("system") or {}
        rows.append(dict(index=r["index"], decode_s=dec_of(r),
                         decode_tok_s=r["formal"]["medians"]["decode_tok_s"],
                         sm_clock_mhz=m.get("clock_sm_MHz"),
                         mem_clock_mhz=m.get("clock_mem_MHz"),
                         power_w=m.get("power_W"), temp_c=m.get("temp_C"),
                         nvidia_smi_used_mib=m.get("mem_used_MiB"),
                         peak_allocated_mib=a.get("peak_allocated_mib"),
                         peak_reserved_mib=a.get("peak_reserved_mib"),
                         inactive_split_peak_mib=a.get("inactive_split_peak_mib"),
                         num_alloc_retries=a.get("num_alloc_retries"),
                         cuda_free_mib=a.get("cuda_free_mib")))
    out["Q6_correlates"] = dict(per_process=rows, note=(
        "**只列观测**：本表用于人工判断波动与哪个量同步；在拿到阶段分解之前"
        "不得把原因写成 WDDM paging/allocator 等任一项。"))
    return out


def _write_summary(run_id: str, payload: dict) -> None:
    p = OUTDIR / f"p5v_{run_id}_summary.json"
    if p.exists():
        raise SystemExit(f"[拒绝覆盖] {p} 已存在")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[p5v] 汇总写出 -> {p}")


def print_report(a: dict) -> None:
    """打印 7 个观测视图（参数是 `analyse()` 的结果）。"""
    print("\n== [V1] 每进程正式 decode（**无插桩**）==")
    print(f"  {'role':<10}{'idx':>4}{'prefill_s':>11}{'decode_s':>10}{'dec_tok/s':>11}"
          f"{'rel%':>7}{'sm_MHz':>8}{'pwr_W':>8}{'smi_MiB':>9}")
    for r in a["Q0_per_process"]:
        f = r.get("formal")
        if not f:
            print(f"  {r['role']:<10}{r['index']:>4}   跳过（{r.get('skipped')}）")
            continue
        m = (r.get("monitor_after") or {}).get("system") or {}
        print(f"  {r['role']:<10}{r['index']:>4}{f['medians']['prefill_s']:>11.2f}"
              f"{f['medians']['decode_s']:>10.2f}{f['medians']['decode_tok_s']:>11.2f}"
              f"{f['decode_rel_spread_pct']:>7.1f}{str(m.get('clock_sm_MHz')):>8}"
              f"{str(m.get('power_W')):>8}{str(m.get('mem_used_MiB')):>9}")

    print("\n== [V2] 双峰判定（**预登记规则**，仅诊断）==")
    q1 = a["Q1_bimodality"]
    print(f"  规则：{q1['decode_tok_per_s']['rule']}")
    print(f"  SpecDec decode tok/s（排序）：{q1['decode_tok_per_s']['sorted']}")
    print(f"  检出={q1['decode_tok_per_s']['detected']}  最大/最小={q1['max_over_min']}")
    if q1["decode_tok_per_s"]["split"]:
        s = q1["decode_tok_per_s"]["split"]
        print(f"  最优切分：低档 {s['low']} / 高档 {s['high']}  比值={s['ratio']:.2f}")

    print("\n== [V3] 进程内稳定性（同进程 reps）==")
    for k, v in a["Q2_within_process"].items():
        print(f"  {k:<16} decode {v['decode_s_min']:.2f}..{v['decode_s_max']:.2f}s "
              f"rel={v['decode_rel_spread_pct']:.1f}%")

    if "Q3_phase_contrast" in a:
        c = a["Q3_phase_contrast"]
        print(f"\n== [V4] 阶段对比（最快进程 #{c['fast']['index']} vs 最慢 #{c['slow']['index']}）"
              f"  [插桩 pass，非正式吞吐] ==")
        print(f"  {'phase':<26}{'fast_ms':>10}{'slow_ms':>10}{'slow/fast':>11}")
        for n, v in c["per_phase_ms"].items():
            note = "  (forward 子集)" if n in c["subset_phases"] else ""
            r = f"{v['ratio']:.2f}" if v["ratio"] else "n/a"
            print(f"  {n:<26}{v['fast']:>10.1f}{v['slow']:>10.1f}{r:>11}{note}")
        print(f"  {'未解释残差':<26}{c['residual_ms']['fast']:>10.1f}"
              f"{c['residual_ms']['slow']:>10.1f}")

    print("\n== [V5] prefill 稳定性 ==")
    for role, d in a["Q4_prefill"].items():
        if d["medians"]:
            print(f"  {role:<10} {[round(x, 2) for x in d['medians']]} "
                  f"max/min={d['max_over_min']}")

    print("\n== [V6] 对照稳定性 ==")
    for role, d in a["Q5_controls"].items():
        if d["decode_tok_per_s"]:
            print(f"  {role:<10} dec_tok/s={[round(x, 2) for x in d['decode_tok_per_s']]} "
                  f"within-rel%={[round(x, 1) for x in d['within_rel_spread_pct']]}")

    print("\n== [V7] 与各状态量并列（**只列观测，未做归因**）==")
    print(f"  {'idx':>4}{'dec_tok/s':>11}{'sm_MHz':>8}{'mem_MHz':>9}{'pwr_W':>7}{'temp_C':>8}"
          f"{'smi_MiB':>9}{'peakAlloc':>10}{'peakResv':>10}{'inactSplit':>11}{'retries':>8}")
    for r in a["Q6_correlates"]["per_process"]:
        f2 = lambda v: "n/a" if v is None else f"{v:.1f}"          # noqa: E731
        print(f"  {r['index']:>4}{r['decode_tok_s']:>11.2f}{str(r['sm_clock_mhz']):>8}"
              f"{str(r['mem_clock_mhz']):>9}{f2(r['power_w']):>7}{f2(r['temp_c']):>8}"
              f"{str(r['nvidia_smi_used_mib']):>9}{f2(r['peak_allocated_mib']):>10}"
              f"{f2(r['peak_reserved_mib']):>10}{f2(r['inactive_split_peak_mib']):>11}"
              f"{str(r['num_alloc_retries']):>8}")
    print("  注：上表用于判断'波动与哪个量同步变化'；**在阶段分解到位前不得把原因"
          "正式写成 WDDM paging / allocator 等任一项**。")


def main() -> int:
    if S.env_flag("P5V_DRY_RUN", False):
        print(f"[p5v] dry-run：role×procs = {ROLES} × {PROCS} = {len(ROLES) * PROCS} 个独立进程")
        print(f"[p5v] 每进程：P-state 预检({PF.SECONDS}s) + warm{WARM} + reps{REPS} "
              f"+ 插桩 {PROF_REPS} 次；ctx={CTX} kv={KV} k={K} max_new={MAXNEW}")
        print(f"[p5v] 输出目录 = {OUTDIR}（唯一命名，不覆盖）")
        print(f"[p5v] 预计时间 ≈ {estimate_minutes():.0f} 分钟"
              f"（角色感知：SpecDec 每 rep≈42s、AR 每 rep≈6s）")
        return 0
    if os.environ.get("P5V_CHILD"):
        role = os.environ.get("P5V_ROLE", ROLE_SPEC)
        idx = int(os.environ.get("P5V_INDEX", "0"))
        rid = os.environ.get("P5V_RUN_ID", "adhoc")
        return child_main(role, idx, rid)
    return parent_main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
