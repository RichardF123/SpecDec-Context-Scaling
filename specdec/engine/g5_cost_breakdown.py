"""g5_cost_breakdown.py — Task C：SpecDec / AR 的阶段成本分解（**profiling，不是正式计时**）。

AR 基线口径（v1.1 修正）
  正式 AR 分解必须 **target-only**：本工具按顺序做三件事，且**不共用被污染的驻留状态**：
    阶段 1 只加载 target → 跑 **AR target-only**（正式主基线口径；此进程中 draft 从未被加载）；
    阶段 2 若需要（默认需要 SpecDec），再加载 draft → 跑 **AR dual-resident control**（draft 仅常驻、
            不参与任何 AR 计算）与 **SpecDec**（target+draft 都是算法必需）。
  ⇒ 可以直接回答：**draft 仅常驻、不参与计算时，AR 是否已经变慢**（= residency penalty）。
  ⇒ 阶段 1 与阶段 2/3 处于不同的分配器池状态，这一差异是**被测量对象**本身，已在此说明；
     它也使本节数字只能用于"阶段归因"，不能当正式吞吐（正式吞吐见 stability probe）。

实现方式（**不修改冻结的 spec_dec.py / spec_cache.py**）
  外部 CUDA Event 包装：`spec_dec._propose` / `_verify` / `_step`（按 `model is target` 分流）/
  `spec_cache.trim_cache` / 过滤后的 `torch.cat`（KV 增长，**是 forward 的子集，不可相加**）。
  `P5_PROFILE=0` 跑同样条件但不加包装 ⇒ 用于量化插桩自身开销。

阶段口径
  SpecDec：draft_propose / target_block_verify / target_replacement_step /
           draft_replacement_step / cache_trim_commit / other-control；
  AR    ：target_single_step（含引擎内部的 KV `torch.cat`）/ kv_cat（子集）/ other-control。

用法（交互式 GPU 终端）
  python -m specdec.engine.g5_cost_breakdown
可选：P5C_CTX=8192 P5C_KV=base_bf16 P5C_K=4 P5C_MAXNEW=64 P5C_REPS=3
      P5_PROFILE=1|0  P5C_INCLUDE_DUAL_AR_CONTROL=1
"""
from __future__ import annotations

import os
import statistics
import time
import traceback

import torch

from .. import results_schema as S
from . import spec_dec, spec_cache
from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CTX = int(os.environ.get("P5C_CTX", "8192"))
KV = os.environ.get("P5C_KV", "base_bf16")
K = int(os.environ.get("P5C_K", str(spec_dec.K)))
MAXNEW = int(os.environ.get("P5C_MAXNEW", "64"))
REPS = int(os.environ.get("P5C_REPS", "3"))
CHUNK = int(os.environ.get("P5C_CHUNK", "2048"))
PROFILE = S.env_flag("P5_PROFILE", True)
INCLUDE_DUAL = S.env_flag("P5C_INCLUDE_DUAL_AR_CONTROL", True)


class Phase:
    """一个阶段的 CUDA Event 累积器（每调用 record 一对；结束时统一 elapsed_time）。"""

    def __init__(self, name: str):
        self.name = name
        self._pairs: list[tuple] = []

    def wrap(self, fn):
        def inner(*a, **k):
            if not PROFILE:
                return fn(*a, **k)
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


def load_ids(tok):
    import zipfile
    from pathlib import Path
    z = Path(__file__).resolve().parents[2] / "data" / "wikitext-2-raw-v1.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if "test" in n and n.endswith(".raw"))
        raw = zf.read(name).decode("utf-8", errors="ignore")
    text = " ".join(ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not ln.strip().startswith("="))
    return tok(text[:400_000], return_tensors="pt").input_ids[:, :CTX]


def chunked_prefill(model, ids, chunk=CHUNK):
    with torch.inference_mode():
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, chunk):
            lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                   need_logits=(st + chunk >= n))
    return lg, cache


def _kv_cat_wrapper(phase, orig_cat):
    def inner(tensors, *a, **k):
        dim = k.get("dim", a[0] if a else None)
        is_kv = (isinstance(tensors, (list, tuple)) and len(tensors) == 2
                 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in tensors)
                 and dim == 2)
        if not is_kv:
            return orig_cat(tensors, *a, **k)
        return phase.wrap(orig_cat)(tensors, *a, **k)
    return inner


def run_ar_profiled(tgt, ids, eos):
    """AR 单 token 递推（与引擎 generate 同语义），分解 single-step / kv_cat / control。"""
    ph = {n: Phase(n) for n in ("target_single_step", "kv_cat")}
    orig_step, orig_cat = spec_dec._step, torch.cat
    spec_dec._step = ph["target_single_step"].wrap(orig_step)
    torch.cat = _kv_cat_wrapper(ph["kv_cat"], orig_cat)
    try:
        tgt.set_kv_format(spec_dec._fmt(KV))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg, cache = chunked_prefill(tgt, ids)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        gen, cur = [], lg[:, -1:]
        for i in range(MAXNEW):
            tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
            gen.append(tok)
            if eos is not None and tok == eos:
                break
            if i == MAXNEW - 1:
                break
            cur, cache = spec_dec._step(tgt, tok, cache)
        torch.cuda.synchronize(); t2 = time.perf_counter()
    finally:
        spec_dec._step, torch.cat = orig_step, orig_cat
    return gen, (t2 - t1), (t1 - t0), {n: p.stats() for n, p in ph.items()}


def run_specdec_profiled(tgt, drf, ids, eos):
    ph = {n: Phase(n) for n in ("draft_propose", "target_block_verify",
                                "target_replacement_step", "draft_replacement_step",
                                "cache_trim_commit", "kv_cat")}
    orig = dict(propose=spec_dec._propose, verify=spec_dec._verify, step=spec_dec._step,
                trim=spec_cache.trim_cache, cat=torch.cat)

    def step_profiled(model, token, cache):
        name = "target_replacement_step" if model is tgt else "draft_replacement_step"
        return ph[name].wrap(orig["step"])(model, token, cache)

    spec_dec._propose = ph["draft_propose"].wrap(orig["propose"])
    spec_dec._verify = ph["target_block_verify"].wrap(orig["verify"])
    spec_dec._step = step_profiled
    spec_cache.trim_cache = ph["cache_trim_commit"].wrap(orig["trim"])
    torch.cat = _kv_cat_wrapper(ph["kv_cat"], orig["cat"])
    try:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                         target_kv=KV)
        torch.cuda.synchronize(); outer = time.perf_counter() - t0
    finally:
        spec_dec._propose, spec_dec._verify, spec_dec._step = orig["propose"], orig["verify"], orig["step"]
        spec_cache.trim_cache = orig["trim"]
        torch.cat = orig["cat"]
    return out, outer, {n: p.stats() for n, p in ph.items()}


def med_phases(rows, key):
    names = rows[0][key].keys()
    out = {}
    for n in names:
        meds = [x[key][n]["median_ms"] for x in rows if x[key][n]["median_ms"] is not None]
        out[n] = dict(total_ms=statistics.median([x[key][n]["total_ms"] for x in rows]),
                      calls=int(statistics.median([x[key][n]["calls"] for x in rows])),
                      median_ms=statistics.median(meds) if meds else None)
    return out


def print_phases(title, phases, outer_s, *, extra="", subset=("kv_cat",)):
    print(f"\n== {title} ==")
    print(f"  {'phase':<26}{'total_s':>9}{'calls':>7}{'median_ms':>11}{'share%':>8}")
    tot = 0.0
    for name, st in phases.items():
        share = (100.0 * st["total_ms"] / (outer_s * 1000.0)) if outer_s > 0 else None
        med = f"{st['median_ms']:.2f}" if st["median_ms"] is not None else "n/a"
        note = "  (forward 的子集，不可相加)" if name in subset else ""
        print(f"  {name:<26}{st['total_ms'] / 1000.0:>9.3f}{st['calls']:>7}{med:>11}"
              f"{(f'{share:.1f}' if share is not None else 'n/a'):>8}{note}")
        if name not in subset:
            tot += st["total_ms"]
    resid = outer_s * 1000.0 - tot
    print(f"  {'other/control(instrumented)':<26}{resid / 1000.0:>9.3f}{'':>7}{'':>11}"
          f"{100.0 * resid / max(outer_s * 1000.0, 1e-9):>8.1f}")
    print(f"  闭合性：阶段和={tot / 1000.0:.3f}s 外层={outer_s:.3f}s "
          f"残差={resid / 1000.0:+.3f}s ({100.0 * resid / max(outer_s * 1000.0, 1e-9):+.1f}%)"
          + (f"  {extra}" if extra else ""))


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    ids = load_ids(tok)
    print(f"[prof] ctx={CTX} kv={KV} k={K} max_new={MAXNEW} reps={REPS} PROFILE={PROFILE} "
          f"（profiling 数据**不是**正式吞吐）", flush=True)
    print(f"[prof] AR 口径：正式主基线 = **target-only**；"
          f"dual-resident control = {'启用' if INCLUDE_DUAL else '未启用'}", flush=True)

    # ---- 阶段 1：只加载 target ⇒ AR target-only（正式主基线口径）----
    print("[prof] 阶段1/3：只加载 target（draft 从未加载）→ AR target-only", flush=True)
    tgt, _ = load_ours(TARGET)
    print(f"[mem] 驻留=['target'] alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB",
          flush=True)
    ar_primary = []
    for r in range(REPS):
        gen, dec_s, pre_s, ph = run_ar_profiled(tgt, ids, eos)
        ar_primary.append((gen, dec_s, pre_s, ph))
        print(f"  rep{r}: AR(target-only) decode={dec_s:.2f}s prefill={pre_s:.2f}s "
              f"n_gen={len(gen)}", flush=True)
    print_phases(f"AR target-only 阶段分解（**正式主基线**；ctx={CTX} kv={KV}）",
                 med_phases(ar_primary, 3), statistics.median([x[1] for x in ar_primary]),
                 extra=f"生成 token={ar_primary[0][0] and len(ar_primary[0][0])}")

    # ---- 阶段 2/3：加载 draft ----
    print("\n[prof] 阶段2/3：加载 draft（之后 draft 常驻）", flush=True)
    drf, _ = load_ours(DRAFT)
    print(f"[mem] 驻留=['target','draft'] alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB",
          flush=True)
    ar_dual = []
    if INCLUDE_DUAL:
        print("[prof] AR dual-resident control（draft 仅常驻、不参与 AR 计算）", flush=True)
        for r in range(REPS):
            gen, dec_s, pre_s, ph = run_ar_profiled(tgt, ids, eos)
            ar_dual.append((gen, dec_s, pre_s, ph))
            print(f"  rep{r}: AR(dual) decode={dec_s:.2f}s prefill={pre_s:.2f}s "
                  f"n_gen={len(gen)}", flush=True)
        base = statistics.median([x[1] for x in ar_primary])
        dual = statistics.median([x[1] for x in ar_dual])
        print(f"  => residency_penalty(decode) = T_AR_target_only / T_AR_dual = "
              f"{base:.2f} / {dual:.2f} = {base / dual:.3f}×"
              f"（>1 表示**双驻留本身**就拖慢了 AR）", flush=True)

    # ---- 阶段 3/3：SpecDec ----
    print("\n[prof] 阶段3/3：SpecDec（target+draft 均为算法必需）", flush=True)
    sp_rows = []
    for r in range(REPS):
        out, outer, ph = run_specdec_profiled(tgt, drf, ids, eos)
        sp_rows.append((out, outer, ph))
        print(f"  rep{r}: SpecDec outer={outer:.2f}s n_gen={len(out['gen'])}", flush=True)
    sp_outer = statistics.median([x[1] for x in sp_rows])
    st = sp_rows[0][0]["stats"]
    print_phases(f"SpecDec 阶段分解（ctx={CTX} kv={KV} k={K}）", med_phases(sp_rows, 2), sp_outer,
                 extra=(f"轮数={st['rounds']} draft前向={st['draft_calls']} "
                        f"块验证={st['target_block_calls']} mismatch={st['mismatch_rounds']} "
                        f"accept={st['accepted']}/{st['proposed']}"))

    # ---- 成本方程（实测代入）----
    par = med_phases(ar_primary, 3)
    psp = med_phases(sp_rows, 2)
    ar_step = par["target_single_step"]["total_ms"] / max(par["target_single_step"]["calls"], 1)
    n_ar = ar_primary[0][0] and len(ar_primary[0][0])
    per_draft = psp["draft_propose"]["total_ms"] / max(psp["draft_propose"]["calls"], 1)
    per_block = psp["target_block_verify"]["total_ms"] / max(psp["target_block_verify"]["calls"], 1)
    per_rep = psp["target_replacement_step"]["total_ms"] / max(psp["target_replacement_step"]["calls"], 1)
    ar_dec = statistics.median([x[1] for x in ar_primary])
    print("\n== 成本方程（实测代入；**profiling，不得当正式吞吐**）==")
    print(f"  AR   : N×target_step ≈ {n_ar}×{ar_step:.1f}ms = {n_ar * ar_step / 1000:.2f}s "
          f"(实测 decode {ar_dec:.2f}s)")
    print(f"  Spec : {st['draft_calls']}×{per_draft:.1f}ms(draft) + "
          f"{st['target_block_calls']}×{per_block:.1f}ms(block) + "
          f"{st['mismatch_rounds']}×{per_rep:.1f}ms(repair) = "
          f"{(st['draft_calls'] * per_draft + st['target_block_calls'] * per_block + st['mismatch_rounds'] * per_rep) / 1000:.2f}s "
          f"(实测 decode {sp_outer:.2f}s)")
    print(f"  ⇒ 每轮 {sp_outer / max(st['rounds'], 1) * 1000:.0f}ms vs AR 每步 {ar_step:.1f}ms "
          f"= {sp_outer / max(st['rounds'], 1) * 1000 / max(ar_step, 1e-9):.1f} 个 AR 步/轮；"
          f"每轮产出 {st['accepted'] / max(st['rounds'], 1):.2f} token")
    print(f"  ⇒ speedup = T_AR/T_Spec = {ar_dec:.2f}/{sp_outer:.2f} = {ar_dec / sp_outer:.3f}×"
          f"（<1 表示 SpecDec 更慢）")
    print("\n[prof] 注意：① 含插桩开销；② `kv_cat` 是 forward 子集、不可相加；"
          "③ 阶段1 与阶段2/3 处于不同分配器池状态（这正是 residency 效应本身）；"
          "④ 本表**不得**写入正式主表。")
    del tgt, drf
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
