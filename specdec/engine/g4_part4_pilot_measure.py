"""g4_part4_pilot_measure.py — G4 Part 4：小规模测量 + final 规模标定（**只出数，不作加速声明**）

覆盖：每个 prompt 跑 `ctx × target_kv × decode` = 3 × 2 × 2 = **12 个条件**
  - `decode` ∈ {`AR`, `SpecDec`}（AR = 同档基线：同 ctx、同 KV、同 prompt、**同引擎原语**）
  - `target_kv` ∈ {`base_bf16`, `v_fp8`}；draft 恒 `base_bf16`（v_fp8 是 target 专属格式）
  - `ctx` 默认 {512, 2048, 8192}，是**同一段真实文本的嵌套前缀**（512 ⊂ 2048 ⊂ 8192）

纪律（写死在本脚本里，避免事后"美化"）
  1. 计时全部 `torch.cuda.synchronize()` 夹住；每条件先跑 1 次 **warm-up 并丢弃**；
  2. 重复 `P4_REPS` 次，报**中位数 + min/max**，不只报最快一次；
  3. peak 显存用 `reset_peak_memory_stats` + `max_memory_allocated/reserved`，并记录权重常驻基线；
  4. **OOM 不崩**：记为 OOM 条件继续跑其余条件（协议要求 OOM 有 finally 处理）；
  5. prompt 取自 `data/wikitext-2-raw-v1.zip` 的**真实文本**，**不插合成填充**，实际 token 数入库；
  6. 8K 档使用**分块 prefill**（G2 结论：一次性 8K eager prefill 会 OOM；分块是 8K 的正常运行模式）；
  7. 只输出数字：本脚本**不**声明显著性、不宣称加速；"比值"仅作同档中位数之比并附告警；
  8. 结果写 `output/part4/*.jsonl`（该目录被 gitignore；正式 schema 由 G5 定义）。

用法（交互式 GPU 终端，单跑；建议先 `set HF_HUB_OFFLINE=1`）
  python -m specdec.engine.g4_part4_pilot_measure
可选环境变量
  P4_CTX=512,2048,8192   P4_REPS=3   P4_MAXNEW=64   P4_WARM=1   P4_CHUNK=2048
  P4_TEXT=<path>         指定自定义纯文本语料（默认用仓库 data/ 下的 wikitext-2 zip）
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
import zipfile
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
K = spec_dec.K

ROOT = Path(__file__).resolve().parents[2]
CTX = [int(x) for x in os.environ.get("P4_CTX", "512,2048,8192").split(",") if x.strip()]
REPS = int(os.environ.get("P4_REPS", "3"))
MAXNEW = int(os.environ.get("P4_MAXNEW", "64"))
WARM = int(os.environ.get("P4_WARM", "1"))
CHUNK = int(os.environ.get("P4_CHUNK", "2048"))
REVERSE = os.environ.get("P4_REVERSE", "") not in ("", "0", "false", "False")
ONLY = os.environ.get("P4_ONLY", "").strip()      # "ctx,kv,mode"：只跑一个条件（每条件独立进程）


def parse_only() -> tuple[int, str, str] | None:
    if not ONLY:
        return None
    parts = [x.strip() for x in ONLY.split(",")]
    if len(parts) != 3:
        raise SystemExit(f'P4_ONLY 需形如 "8192,v_fp8,AR"，收到 {ONLY!r}')
    ctx, kv, mode = int(parts[0]), parts[1], parts[2]
    if kv not in ("base_bf16", "v_fp8") or mode not in ("AR", "SpecDec"):
        raise SystemExit(f"P4_ONLY kv/mode 非法: {kv}/{mode}")
    return ctx, kv, mode
OUTDIR = ROOT / "output" / "part4"


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def load_text() -> tuple[str, str]:
    """返回 (文本, 来源标签)。优先环境变量指定的文件，否则用仓库内的 wikitext-2 zip。"""
    p = os.environ.get("P4_TEXT")
    if p:
        return Path(p).read_text(encoding="utf-8", errors="ignore"), f"file:{p}"
    zpath = ROOT / "data" / "wikitext-2-raw-v1.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            pick = next((n for n in names if "test" in n and n.endswith((".raw", ".txt"))), None)
            if pick is None:
                pick = next((n for n in names if n.endswith((".raw", ".txt"))), None)
            raw = z.read(pick).decode("utf-8", errors="ignore")
        lines = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s or s.startswith("="):          # 跳过标题/空行，保留正文
                continue
            lines.append(s)
        return " ".join(lines), f"wikitext-2-raw-v1/{Path(pick).name}"
    # 最后的兜底：明确标注为非语料（正式实验不得使用）
    return ("Speculative decoding uses a small draft model to propose tokens and a "
            "larger target model to verify them. " * 400), "FALLBACK-SYNTHETIC(不得用于正式实验)"


def chunked_prefill(model, ids, chunk: int = CHUNK):
    """分块 prefill：除最后一块外都不需要 logits。等价于引擎的 prefill_chunked + 末块 logits。
    8K 在本卡上必须分块（G2）；chunk 无关性已由 V3 实测。"""
    with torch.inference_mode():
        cache = None
        lg = None
        n = ids.shape[1]
        for st in range(0, n, chunk):
            piece = ids[:, st:st + chunk]
            lg, cache = model._run(model.embed_tokens(piece), cache,
                                   need_logits=(st + chunk >= n))
    return lg, cache


def ar_decode(tgt, ids, n, eos, kv, chunk: int = CHUNK):
    """AR 同档基线：一次分块 prefill + 逐 token `_step`（与引擎 generate 同语义）。
    末 token 之后不再多跑一次前向（避免把 1 次无用前向算进基线）。"""
    tgt.set_kv_format(spec_dec._fmt(kv))
    t0 = sync_time()
    lg, cache = chunked_prefill(tgt, ids, chunk)
    t1 = sync_time()
    gen = []
    fwd = 0
    cur = lg[:, -1:]
    for i in range(n):
        tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
        gen.append(tok)
        if eos is not None and tok == eos:
            break
        if i == n - 1:
            break
        cur, cache = spec_dec._step(tgt, tok, cache)
        fwd += 1
    t2 = sync_time()
    return gen, t1 - t0, t2 - t1, {"target_decode_calls": fwd}


def spec_decode_timed(tgt, drf, ids, n, eos, kv, chunk: int = CHUNK):
    """SpecDec：统计 prefill 耗时（monkeypatch，不改核心代码）后，总耗时 − prefill = decode。"""
    acc = {"t": 0.0}
    orig = spec_dec._prefill

    def timed_prefill(model, ids_):
        a = sync_time()
        lg, cache = chunked_prefill(model, ids_, chunk)
        acc["t"] += sync_time() - a
        return lg, cache

    trace: list[dict] = []
    spec_dec._prefill = timed_prefill
    try:
        t0 = sync_time()
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos,
                                          target_kv=kv, log=lambda d: trace.append(d))
        t1 = sync_time()
    finally:
        spec_dec._prefill = orig
    total = t1 - t0
    return out["gen"], acc["t"], total - acc["t"], out["stats"], trace


def mem_read():
    return (torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30,
            torch.cuda.memory_allocated() / 2**30)


def run_once(tgt, drf, ids, eos, kv, mode):
    # 注意：**不在计时区内调用 empty_cache()**。它会把手里的缓存块全部还给驱动，使每个 rep
    # 都从零重建显存池；而 KV cache 每步 torch.cat 都要申请新块（8K 档每步 ~4 MiB × 28 层 ×
    # 2~3 个张量），从零池开始会把成千上万次 cudaMalloc/WDDM 映射算进 decode 时间 ——
    # 这会系统性地惩罚"第一个跑到该 ctx 的条件"。稳态复用才是真实服务循环的行为。
    # 仅 OOM 时清理（见 main）。
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated() / 2**30
    trace = []
    if mode == "AR":
        gen, tp, td, st = ar_decode(tgt, ids, MAXNEW, eos, kv)
    else:
        gen, tp, td, st, trace = spec_decode_timed(tgt, drf, ids, MAXNEW, eos, kv)
    pa, pr, _ = mem_read()
    row = dict(mode=mode, target_kv=kv, ctx=int(ids.shape[1]), n_gen=len(gen),
               t_prefill_s=tp, t_decode_s=td, t_total_s=tp + td,
               peak_alloc_GiB=pa, peak_resv_GiB=pr, weights_resident_GiB=base_alloc)
    row["decode_tok_s"] = len(gen) / td if td > 0 else float("nan")
    row["e2e_tok_s"] = len(gen) / (tp + td) if (tp + td) > 0 else float("nan")
    if mode == "AR":
        row["target_block_calls"] = 0
        row["draft_calls"] = 0
        row["rounds"] = 0
        row["accepted"] = 0
        row["proposed"] = 0
        row["accept_rate"] = float("nan")
        row["target_decode_calls"] = st["target_decode_calls"]
        row["m_hist"] = []
    else:
        row.update({k: st[k] for k in ("rounds", "proposed", "accepted",
                                       "target_block_calls", "draft_calls",
                                       "mismatch_rounds", "all_accept_rounds")})
        row["accept_rate"] = st["accepted"] / st["proposed"] if st["proposed"] else float("nan")
        row["m_hist"] = [d["m"] for d in trace]
    row["gen"] = gen
    return row


def summarize(runs: list[dict]) -> dict:
    keys_num = ["t_prefill_s", "t_decode_s", "t_total_s", "decode_tok_s", "e2e_tok_s",
                "peak_alloc_GiB", "peak_resv_GiB", "accept_rate", "rounds",
                "proposed", "accepted", "target_block_calls", "draft_calls",
                "target_decode_calls", "mismatch_rounds", "all_accept_rounds"]
    s = {k: v for k, v in runs[0].items() if k not in keys_num + ["gen", "m_hist"]}
    for k in keys_num:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        vals = [v for v in vals if v == v]        # 去 NaN
        if vals:
            s[k + "_med"] = statistics.median(vals)
            s[k + "_min"] = min(vals)
            s[k + "_max"] = max(vals)
        else:
            s[k + "_med"] = float("nan")
    s["m_hist"] = [m for r in runs for m in r.get("m_hist", [])]
    s["reps"] = len(runs)
    return s


def print_report(summary: dict, ctxs, kvs) -> None:
    """打印四张表。可从内存 summary 调用，也可从 JSONL 重建后调用（--aggregate）。"""
    print("\n== 汇总（中位数；dec_spread = max-min，>20% 即视为可疑需查）==")
    print("ctx    kv         mode     dec_tok/s  e2e_tok/s  pref_s  dec_s  spread  accept  "
          "rounds  block  draft  peak_alloc  peak_resv*  ==AR")
    print("  * peak_resv 是分配器池（含空闲块），不是物理占用；peak_alloc 才是实际用量")
    for ctx in ctxs:
        for kv in kvs:
            for mode in ("AR", "SpecDec"):
                s = summary.get((ctx, kv, mode))
                if not s or s.get("oom"):
                    print(f"{ctx:<6} {kv:<10} {mode:<8} OOM/未测")
                    continue
                ac = s.get("accept_rate_med", float("nan"))
                print(f"{ctx:<6} {kv:<10} {mode:<8} {s['decode_tok_s_med']:>9.2f}  "
                      f"{s['e2e_tok_s_med']:>9.2f}  {s['t_prefill_s_med']:>6.2f} "
                      f"{s['t_decode_s_med']:>6.2f}  "
                      f"{s['t_decode_s_max']-s['t_decode_s_min']:>6.2f}  {ac:>6.3f}  "
                      f"{s['rounds_med']:>6.0f} "
                      f"{s['target_block_calls_med']:>6.0f} {s['draft_calls_med']:>6.0f}  "
                      f"{s['peak_alloc_GiB_med']:>9.2f}  {s['peak_resv_GiB_med']:>9.2f}  "
                      f"{s.get('token_equal_vs_AR', '-')}")

    print("\n== 同档比值（SpecDec / AR，中位数之比；**未做显著性检验、未控温漂**，仅作数量级参考）==")
    for ctx in ctxs:
        for kv in kvs:
            a, s = summary.get((ctx, kv, "AR")), summary.get((ctx, kv, "SpecDec"))
            if not a or not s or a.get("oom") or s.get("oom"):
                continue
            r_dec = s["decode_tok_s_med"] / a["decode_tok_s_med"]
            r_e2e = s["e2e_tok_s_med"] / a["e2e_tok_s_med"]
            print(f"  ctx={ctx:<5} kv={kv:<10} decode×{r_dec:.3f}  e2e×{r_e2e:.3f}  "
                  f"accept={s['accept_rate_med']:.3f}  AR前向={a['target_decode_calls_med']:.0f} "
                  f"Spec块+单步={s['target_block_calls_med']:.0f}+{s['mismatch_rounds_med']:.0f}")

    print("\n== 每轮接受长度分布（SpecDec，跨所有 rep 汇总 m 直方图）==")
    for ctx in ctxs:
        for kv in kvs:
            s = summary.get((ctx, kv, "SpecDec"))
            if not s or s.get("oom"):
                continue
            hist = s.get("m_hist", [])
            cnt = {i: hist.count(i) for i in range(K + 1)}
            print(f"  ctx={ctx:<5} kv={kv:<10} m=0..{K}: "
                  + " ".join(f"{i}:{cnt[i]}" for i in range(K + 1))
                  + f"   (总轮数 {len(hist)})")

    measured = [s for s in summary.values() if not s.get("oom")]
    full = len(ctxs) * len(kvs) * 2
    per_family = sum(s["t_total_s_med"] for s in measured)
    print("\n== final 规模标定 ==")
    print(f"  已测 {len(measured)}/{full} 个条件（ctx={list(ctxs)}）；每 family 走完这 {len(measured)} "
          f"个条件的实测成本 = {per_family:.1f}s")
    if len(measured) < full:
        print("  !! **外推不完整**：缺少某些 ctx/KV/decode 条件 ⇒ 先补跑再据此定规模")
    else:
        for fams, reps in ((16, 5), (16, 3)):
            tot_h = per_family * fams * reps / 3600.0
            warm_h = per_family * fams * WARM / 3600.0
            print(f"  ≥{fams} families × {full} conditions × {reps} reps ≈ "
                  f"**{tot_h:.1f} GPU 小时**（不含失败重试；warm-up={WARM} 再加 ≈{warm_h:.1f}h）")
    print(f"  ※ 本次实测墙钟（含 warm-up）≈ "
          f"{sum(s['t_total_s_med']*(REPS+WARM) for s in measured)/60:.1f} 分钟")


def load_summary(path: Path):
    """从 JSONL 重建 summary（供 --aggregate 用；不需要重新加载模型）。"""
    summary: dict = {}
    ctxs = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "summary":
            continue
        summary[(r["ctx"], r["target_kv"], r["mode"])] = r
        ctxs.add(r["ctx"])
    return summary, sorted(ctxs)


def main() -> int:
    ap = argparse.ArgumentParser(description="G4 Part 4 测量 / 报表")
    ap.add_argument("--aggregate", nargs="?", const="", default=None,
                    help="不跑模型，只从 output/part4 的 JSONL 重算四张表"
                         "（可跟具体文件路径；缺省取最新一份）")
    args = ap.parse_args()
    if args.aggregate is not None:
        if args.aggregate:
            path = Path(args.aggregate)
        else:
            cand = sorted(OUTDIR.glob("part4_*.jsonl"), key=lambda p: p.stat().st_mtime)
            if not cand:
                raise SystemExit(f"{OUTDIR} 下没有 part4_*.jsonl")
            path = cand[-1]
        summary, ctxs = load_summary(path)
        print(f"[aggregate] {path}  ({len(summary)} 个条件)")
        print_report(summary, ctxs, ("base_bf16", "v_fp8"))
        return 0

    torch.manual_seed(0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    text, src = load_text()
    print(f"[env] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[cfg] ctx={CTX} reps={REPS} max_new={MAXNEW} warm={WARM} chunk={CHUNK} k={K}",
          flush=True)
    print(f"[cfg] text={src} chars={len(text)}", flush=True)

    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    # 只取前 400k 字符（≈92k token）：足以覆盖 ctx≤8192，且避免 tokenizer 的"超过模型最大
    # 长度"告警（该告警只与一次性编码长度有关，与我们要的前缀无关）。
    text_used = text[:400_000]
    if len(text_used) < len(text):
        print(f"[data] 截取前 {len(text_used)} 字符（原文 {len(text)}）", flush=True)
    all_ids = tok(text_used, return_tensors="pt").input_ids
    print(f"[data] corpus tokens={all_ids.shape[1]}（嵌套前缀取自同一段文本，未插合成填充）",
          flush=True)

    print("[stage] loading target 1.5B ...", flush=True)
    tgt, _ = load_ours(TARGET)
    print("[stage] loading draft 0.5B ...", flush=True)
    drf, _ = load_ours(DRAFT)
    print(f"[mem] 两模型常驻 alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    cfg = dict(ctx=CTX, reps=REPS, max_new=MAXNEW, k=K, chunk=CHUNK,
               target=TARGET, draft=DRAFT, text=src, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), warm=WARM,
               kv_order="reversed" if REVERSE else "base-first",
               prompt_encoding="tokenizer default, no chat template, raw text prefix",
               nested_prefix=True, filler="none (real wikitext text)")
    chash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    jsonl = OUTDIR / f"part4_{chash}.jsonl"

    KVS = ("v_fp8", "base_bf16") if REVERSE else ("base_bf16", "v_fp8")
    if REVERSE:
        print("[cfg] P4_REVERSE=1 -> 每个 ctx 内先 v_fp8 后 base_bf16（顺序对照）", flush=True)
    conditions = [(ctx, kv, mode) for ctx in CTX for kv in KVS
                  for mode in ("AR", "SpecDec")]
    only = parse_only()
    if only is not None:
        conditions = [only]
        print(f"[cfg] P4_ONLY={only} -> 单条件独立进程模式（每格从干净显存状态开始）",
              flush=True)
    append = only is not None and jsonl.exists()
    summary: dict[tuple, dict] = {}
    ref_gen: dict[tuple, list] = {}
    with jsonl.open("a" if append else "w", encoding="utf-8") as fh:
        if not append:
            fh.write(json.dumps(dict(kind="config", config_hash=chash, **cfg),
                                ensure_ascii=False) + "\n")
        for ctx, kv, mode in conditions:
            if all_ids.shape[1] < ctx:
                print(f"[skip] ctx={ctx} 超出语料 token 数 {all_ids.shape[1]}")
                continue
            ids = all_ids[:, :ctx].cuda()
            tag = f"ctx={ctx} kv={kv} mode={mode}"
            for _ in range(WARM):
                try:
                    run_once(tgt, drf, ids, eos, kv, mode)
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    print(f"  [{tag}] warm-up OOM（已清理）", flush=True)
                    break
            runs, oom = [], False
            for rep in range(REPS):
                try:
                    r = run_once(tgt, drf, ids, eos, kv, mode)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    gc.collect(); torch.cuda.empty_cache()
                    print(f"  [{tag}] rep{rep} OOM -> 记 OOM 并继续", flush=True)
                    break
                runs.append(r)
                fh.write(json.dumps(dict(kind="run", config_hash=chash, rep=rep,
                                         **{k: v for k, v in r.items() if k != "gen"}),
                                    ensure_ascii=False) + "\n")
                fh.flush()
            if not runs:
                summary[(ctx, kv, mode)] = dict(oom=True, mode=mode, target_kv=kv, ctx=ctx)
                continue
            s = summarize(runs)
            s["oom"] = oom
            summary[(ctx, kv, mode)] = s
            if mode == "AR":
                ref_gen[(ctx, kv)] = runs[0]["gen"]
                print(f"  [{tag}] pref={s['t_prefill_s_med']:.2f}s "
                      f"dec={s['t_decode_s_med']:.2f}s[{s['t_decode_s_min']:.2f}.."
                      f"{s['t_decode_s_max']:.2f}] dec_tok/s={s['decode_tok_s_med']:.2f} "
                      f"peak_alloc={s['peak_alloc_GiB_med']:.2f}", flush=True)
            else:
                eq = ref_gen.get((ctx, kv)) == runs[0]["gen"]
                s["token_equal_vs_AR"] = eq
                print(f"  [{tag}] pref={s['t_prefill_s_med']:.2f}s "
                      f"dec={s['t_decode_s_med']:.2f}s[{s['t_decode_s_min']:.2f}.."
                      f"{s['t_decode_s_max']:.2f}] dec_tok/s={s['decode_tok_s_med']:.2f} "
                      f"accept={s['accept_rate_med']:.3f} "
                      f"rounds={s['rounds_med']:.0f} block={s['target_block_calls_med']:.0f} "
                      f"draft={s['draft_calls_med']:.0f} peak_alloc={s['peak_alloc_GiB_med']:.2f} "
                      f"==AR:{eq}", flush=True)
            fh.write(json.dumps(dict(kind="summary", config_hash=chash, **s),
                                ensure_ascii=False) + "\n")

    # ---- 汇总表（与 --aggregate 共用同一函数）----
    print_report(summary, CTX, KVS)

    print(f"\n[out] {jsonl}")
    del tgt, drf
    gc.collect(); torch.cuda.empty_cache()
    print("Part4: 已出数（未作加速声明；正式结论须走 G5 落盘 + full 矩阵）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
