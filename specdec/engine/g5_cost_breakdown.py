"""g5_cost_breakdown.py — Task C：SpecDec / AR 的阶段成本分解（**profiling，不是正式计时**）。

为什么单独一个工具
  正式计时必须没有任何插桩；profiling 必须有插桩。两者混在一起会互相污染。因此：
  - 本工具的**所有数字都不得当作正式吞吐**（只用来回答"时间花在哪一阶段"）；
  - `P5_PROFILE=0` 会跑同样的条件**但不加包装**，用它量化插桩自身的开销。

实现方式（**不修改冻结的 spec_dec.py / spec_cache.py**）
  在外部用 CUDA Event 包装：`spec_dec._propose`（draft 提议）、`spec_dec._verify`（target 块验证）、
  `spec_dec._step`（按 `model is target` 分成 target 替换步 / draft 替换步）、
  `spec_cache.trim_cache`（cache 裁剪/提交）、以及过滤后的 `torch.cat`（KV 增长，**是 forward 的子集**）。

阶段口径（避免重复计数）
  - `_step` / `_verify` / `_propose` 内部包含 KV 的 `torch.cat` ⇒ `kv_cat` 是它们的**子集**，
    **不可与 forward 阶段相加**，只用于说明"forward 里有多次张量拼接"。
  - `other/control` = 外层 decode 总时间 − 各阶段之和（含 Python 控制流、sampling/argmax、
    循环与 instrumentation 残余）。

输出：每阶段总时间 / 次数 / 每次中位数 / 占比；闭合性检查（各阶段和 vs 外层 decode，残差 %）；
并在末尾给出实测成本方程。

用法（交互式 GPU 终端；单进程，profiling 专用）
  python -m specdec.engine.g5_cost_breakdown
可选：P5C_CTX=8192 P5C_KV=base_bf16 P5C_K=4 P5C_MAXNEW=64 P5C_REPS=3 P5C_PROFILE=1
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


class Phase:
    """一个阶段的 CUDA Event 累积器（record 开销极小；结束时统一 elapsed_time）。"""

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

    def note(self, e0, e1):
        if PROFILE:
            self._pairs.append((e0, e1))

    def stats(self) -> dict:
        if not self._pairs:
            return dict(total_ms=0.0, calls=0, median_ms=None, per_call=None)
        per = [a.elapsed_time(b) for a, b in self._pairs]
        return dict(total_ms=sum(per), calls=len(per), median_ms=statistics.median(per),
                    per_call=per)


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


def run_specdec_profiled(tgt, drf, ids, eos):
    ph = {n: Phase(n) for n in ("draft_propose", "target_block_verify",
                                "target_replacement_step", "draft_replacement_step",
                                "cache_trim_commit", "kv_cat")}
    orig = dict(propose=spec_dec._propose, verify=spec_dec._verify, step=spec_dec._step,
                trim=spec_cache.trim_cache, cat=torch.cat)

    def step_profiled(model, token, cache):
        name = "target_replacement_step" if model is tgt else "draft_replacement_step"
        return ph[name].wrap(orig["step"])(model, token, cache)

    def cat_profiled(tensors, *a, **k):
        dim = k.get("dim", a[0] if a else None)
        is_kv = (isinstance(tensors, (list, tuple)) and len(tensors) == 2
                 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in tensors)
                 and dim == 2)
        if not is_kv:
            return orig["cat"](tensors, *a, **k)
        return ph["kv_cat"].wrap(orig["cat"])(tensors, *a, **k)

    spec_dec._propose = ph["draft_propose"].wrap(orig["propose"])
    spec_dec._verify = ph["target_block_verify"].wrap(orig["verify"])
    spec_dec._step = step_profiled
    spec_cache.trim_cache = ph["cache_trim_commit"].wrap(orig["trim"])
    torch.cat = cat_profiled
    try:
        t0 = time.perf_counter(); torch.cuda.synchronize()
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                         target_kv=KV)
        torch.cuda.synchronize(); outer_s = time.perf_counter() - t0
    finally:
        spec_dec._propose, spec_dec._verify, spec_dec._step = orig["propose"], orig["verify"], orig["step"]
        spec_cache.trim_cache = orig["trim"]
        torch.cat = orig["cat"]
    return out, outer_s, {n: p.stats() for n, p in ph.items()}


def run_ar_profiled(tgt, ids, eos):
    ph = {n: Phase(n) for n in ("target_single_step", "kv_cat")}
    orig = dict(step=spec_dec._step, cat=torch.cat)

    def cat_profiled(tensors, *a, **k):
        dim = k.get("dim", a[0] if a else None)
        is_kv = (isinstance(tensors, (list, tuple)) and len(tensors) == 2
                 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in tensors)
                 and dim == 2)
        if not is_kv:
            return orig["cat"](tensors, *a, **k)
        return ph["kv_cat"].wrap(orig["cat"])(tensors, *a, **k)

    spec_dec._step = ph["target_single_step"].wrap(orig["step"])
    torch.cat = cat_profiled
    try:
        tgt.set_kv_format(spec_dec._fmt(KV))
        t0 = time.perf_counter(); torch.cuda.synchronize()
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
        spec_dec._step = orig["step"]
        torch.cat = orig["cat"]
    return gen, (t2 - t1), (t1 - t0), {n: p.stats() for n, p in ph.items()}, len(gen)


def print_phases(title, phases, outer_s, *, per_round=None, subset_note=()):
    print(f"\n== {title} ==")
    print(f"  {'phase':<26}{'total_s':>9}{'calls':>7}{'median_ms':>11}{'share%':>8}")
    tot = 0.0
    for name, st in phases.items():
        share = (100.0 * st["total_ms"] / (outer_s * 1000.0)) if outer_s > 0 else None
        note = "  (forward 的子集，不可相加)" if name in subset_note else ""
        med = f"{st['median_ms']:.2f}" if st["median_ms"] is not None else "n/a"
        print(f"  {name:<26}{st['total_ms'] / 1000.0:>9.3f}{st['calls']:>7}{med:>11}"
              f"{(f'{share:.1f}' if share is not None else 'n/a'):>8}{note}")
        if name not in subset_note:
            tot += st["total_ms"]
    resid_ms = outer_s * 1000.0 - tot
    print(f"  {'other/control(instrumented)':<26}{resid_ms / 1000.0:>9.3f}{'':>7}{'':>11}"
          f"{100.0 * resid_ms / max(outer_s * 1000.0, 1e-9):>8.1f}")
    print(f"  闭合性：阶段和={tot / 1000.0:.3f}s  外层={outer_s:.3f}s  "
          f"残差={resid_ms / 1000.0:+.3f}s ({100.0 * resid_ms / max(outer_s * 1000.0, 1e-9):+.1f}%)"
          + (f"  {per_round}" if per_round else ""))


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    ids = load_ids(tok)
    print(f"[prof] ctx={CTX} kv={KV} k={K} max_new={MAXNEW} reps={REPS} "
          f"PROFILE={PROFILE}（profiling 数据**不是**正式吞吐）", flush=True)
    print("[prof] 加载双模型（AR 亦保持双模型常驻，驻留条件与 SpecDec 一致）...", flush=True)
    tgt, _ = load_ours(TARGET)
    drf, _ = load_ours(DRAFT)

    rep_cache = {"SpecDec": [], "AR": []}
    for r in range(REPS):
        out, outer, ph = run_specdec_profiled(tgt, drf, ids, eos)
        rep_cache["SpecDec"].append((out, outer, ph))
        gen, dec_s, pre_s, aph, ngen = run_ar_profiled(tgt, ids, eos)
        rep_cache["AR"].append((gen, dec_s, pre_s, aph, ngen))
        print(f"  rep{r}: SpecDec outer={outer:.2f}s n_gen={len(out['gen'])} | "
              f"AR decode={dec_s:.2f}s n_gen={ngen}", flush=True)

    def med_phases(rows, key):
        names = rows[0][key].keys()
        out = {}
        for n in names:
            meds = [x[key][n]["median_ms"] for x in rows if x[key][n]["median_ms"] is not None]
            out[n] = dict(
                total_ms=statistics.median([x[key][n]["total_ms"] for x in rows]),
                calls=int(statistics.median([x[key][n]["calls"] for x in rows])),
                median_ms=statistics.median(meds) if meds else None)
        return out

    sp_rows = rep_cache["SpecDec"]
    ar_rows = rep_cache["AR"]
    sp_outer = statistics.median([x[1] for x in sp_rows])
    sp_stats = sp_rows[0][0]["stats"]
    print_phases(f"SpecDec 阶段分解（ctx={CTX} kv={KV} k={K}，{REPS} 次中位数）",
                 med_phases(sp_rows, 2), sp_outer,
                 per_round=(f"轮数={sp_stats['rounds']} draft前向={sp_stats['draft_calls']} "
                            f"块验证={sp_stats['target_block_calls']} "
                            f"mismatch={sp_stats['mismatch_rounds']} "
                            f"accept={sp_stats['accepted']}/{sp_stats['proposed']}"),
                 subset_note=("kv_cat",))
    ar_dec = statistics.median([x[1] for x in ar_rows])
    print_phases(f"AR 阶段分解（ctx={CTX} kv={KV}，{REPS} 次中位数）",
                 med_phases(ar_rows, 3), ar_dec,
                 per_round=f"生成 token={ar_rows[0][4]}", subset_note=("kv_cat",))

    # ---- 成本方程（用实测数字代入）----
    phases = med_phases(sp_rows, 2)
    astep = med_phases(ar_rows, 3)
    per_draft_ms = phases["draft_propose"]["total_ms"] / max(phases["draft_propose"]["calls"], 1)
    per_block_ms = phases["target_block_verify"]["total_ms"] / max(phases["target_block_verify"]["calls"], 1)
    per_tstep_ms = phases["target_replacement_step"]["total_ms"] / max(phases["target_replacement_step"]["calls"], 1)
    ar_step_ms = astep["target_single_step"]["total_ms"] / max(astep["target_single_step"]["calls"], 1)
    ngen_ar = ar_rows[0][4]
    print("\n== 成本方程（实测代入）==")
    print(f"  AR   : N×target_step ≈ {ngen_ar}×{ar_step_ms:.1f}ms = "
          f"{ngen_ar * ar_step_ms / 1000:.2f}s  (实测 decode {ar_dec:.2f}s)")
    print(f"  Spec : draft_calls×draft_step + rounds×block + mismatch×repair + control")
    print(f"         ≈ {sp_stats['draft_calls']}×{per_draft_ms:.1f}ms + "
          f"{sp_stats['target_block_calls']}×{per_block_ms:.1f}ms + "
          f"{sp_stats['mismatch_rounds']}×{per_tstep_ms:.1f}ms = "
          f"{(sp_stats['draft_calls'] * per_draft_ms + sp_stats['target_block_calls'] * per_block_ms + sp_stats['mismatch_rounds'] * per_tstep_ms) / 1000:.2f}s "
          f"(实测 decode {sp_outer:.2f}s)")
    print(f"  ⇒ target 步单价：AR 单步 {ar_step_ms:.1f}ms vs Spec 块验证 {per_block_ms:.1f}ms"
          f"（块一次覆盖 {K} 个候选）；draft 步单价 {per_draft_ms:.1f}ms")
    print(f"  ⇒ 每轮产出 {sp_stats['accepted'] / max(sp_stats['rounds'], 1):.2f} 个 token，"
          f"但每轮要跑 {sp_stats['draft_calls'] / max(sp_stats['rounds'], 1):.2f} 次 draft 前向 "
          f"+ 1 次块验证 + {sp_stats['mismatch_rounds'] / max(sp_stats['rounds'], 1):.2f} 次修复")
    print("\n[prof] 注意：① 本表含插桩开销；② `kv_cat` 是 forward 阶段的子集，不可相加；"
          "③ 这些数字**不得**写入正式主表。")
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
