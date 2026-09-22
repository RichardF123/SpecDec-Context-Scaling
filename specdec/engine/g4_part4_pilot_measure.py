"""g4_part4_pilot_measure.py — G4 Part 4 测量 + 报表（**只出数，不作加速声明**）

条件空间：`ctx × target_kv × decode`（默认 3 × 2 × 2 = 12）
  - `decode` ∈ {`AR`, `SpecDec`}；AR = 同档基线（同 ctx、同 KV、同 prompt、同引擎原语）
  - `target_kv` ∈ {`base_bf16`, `v_fp8`}；draft 恒 base（v_fp8 是 target 专属格式）
  - `ctx` 是**同一段真实文本的嵌套前缀**（默认 512 ⊂ 2048 ⊂ 8192）

测量纪律（写死在代码里）
  1. 计时 `torch.cuda.synchronize()` 夹住；每条件先 warm-up `P4_WARM` 次并丢弃；
  2. 重复 `P4_REPS` 次，报 median / min / max / spread / **relative spread**；
     `relative spread > P4_REL_SPREAD_LIMIT`（默认 20%）→ 该条件标记 CHECK（**只用于识别测量不
     稳定，不用于宣布统计显著性**）；
  3. 不在计时区内调用 `empty_cache()`（会让每个 rep 从零重建显存池，把 cudaMalloc/WDDM 映射算进
     decode 时间）；仅 OOM 时清理；
  4. OOM 记为 OOM 条件并继续，不崩；
  5. 只输出数字：不声明显著性、不宣称加速；比值仅作同档中位数之比并附告警。

运行身份（防覆盖）
  - 输出文件名 = `part4_<config_hash>_<RUN_ID>.jsonl`，RUN_ID 默认 UTC 时间戳；
  - **绝不**用 `"w"` 静默覆盖已存在的同名文件：文件已存在且不在"同 RUN_ID 组"内 → 直接报错退出；
  - 子进程（`P4_ONLY`）共享父进程给定的 `P4_RUN_ID`，以追加模式写入同一文件。

条件过滤（未设置 = 保持全条件行为）
  P4_CTX=8192           只跑该 ctx（逗号分隔可多值）
  P4_KV=base_bf16 | v_fp8
  P4_MODE=AR | SpecDec
  P4_ONLY=<ctx>,<kv>,<mode>     只跑一个条件（每条件独立进程模式）
  P4_REVERSE=1                  每个 ctx 内先 v_fp8 后 base_bf16（顺序对照）

用法
  python -m specdec.engine.g4_part4_pilot_measure                    # 全条件
  python -m specdec.engine.g4_part4_pilot_measure --aggregate [file]  # 只从 JSONL 重算报表
Run（交互式 GPU 终端，单跑；建议先 `set HF_HUB_OFFLINE=1`）
  可选：P4_REPS / P4_MAXNEW / P4_WARM / P4_CHUNK / P4_TEXT / P4_RUN_ID / P4_ORDER
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
# 正式协议冻结 k=4；P4_K 仅供诊断（k sweep）使用，默认值不变
K = int(os.environ.get("P4_K", str(spec_dec.K)))
PROTOCOL_K = spec_dec.K
KV_CHOICES = ("base_bf16", "v_fp8")
MODE_CHOICES = ("AR", "SpecDec")
FORMAL_CTX = (2048, 4096, 8192)
EXPLORATORY_CTX = (512, 2048, 8192)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "part4"

CTX = [int(x) for x in os.environ.get("P4_CTX", "512,2048,8192").split(",") if x.strip()]
REPS = int(os.environ.get("P4_REPS", "3"))
MAXNEW = int(os.environ.get("P4_MAXNEW", "64"))
WARM = int(os.environ.get("P4_WARM", "1"))
CHUNK = int(os.environ.get("P4_CHUNK", "2048"))
REVERSE = os.environ.get("P4_REVERSE", "") not in ("", "0", "false", "False")
ONLY = os.environ.get("P4_ONLY", "").strip()
# AR 的驻留模式：target_only（正式主基线，只加载 target）/ dual_control（诊断控制，draft 仅常驻）
AR_RESIDENCY = os.environ.get("P4_AR_RESIDENCY", "target_only").strip()
KV_FILTER = os.environ.get("P4_KV", "").strip() or None
MODE_FILTER = os.environ.get("P4_MODE", "").strip() or None
RUN_ID = os.environ.get("P4_RUN_ID", "").strip() or \
    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
ORDER_LABEL = os.environ.get("P4_ORDER", "").strip() or None
REL_SPREAD_LIMIT = float(os.environ.get("P4_REL_SPREAD_LIMIT", "20"))

FINITE = {"ok": True}          # 每次计时前重置；记录"prefill/单步 logits 是否 finite"

RESERVED_SEMANTICS = ("peak_resv_GiB = torch.cuda.max_memory_reserved()（本进程 PyTorch 分配器"
                      "池的峰值：含空闲块、虚拟预留；**不是物理显存占用**）。"
                      "峰值取本条件一次计时区间内的单次读数，**不做任何相加/累计**。"
                      "主指标用 peak_alloc_GiB = torch.cuda.max_memory_allocated()；"
                      "物理口径看 monitor_*.mem_used_MiB（nvidia-smi used）。")


# ---------------------------------------------------------------- 纯函数（CPU 可测）
def parse_only() -> tuple[int, str, str] | None:
    if not ONLY:
        return None
    parts = [x.strip() for x in ONLY.split(",")]
    if len(parts) != 3:
        raise SystemExit(f'P4_ONLY 需形如 "8192,v_fp8,AR"，收到 {ONLY!r}')
    ctx, kv, mode = int(parts[0]), parts[1], parts[2]
    if kv not in KV_CHOICES or mode not in MODE_CHOICES:
        raise SystemExit(f"P4_ONLY kv/mode 非法: {kv}/{mode}")
    return ctx, kv, mode


def parse_filters(kv_filter: str | None = None, mode_filter: str | None = None,
                  ctxs=None, reverse: bool | None = None) -> tuple[list[int], list[str], list[str]]:
    """把 (P4_CTX, P4_KV, P4_MODE, P4_REVERSE) 解析成实际要跑的维度。非法值直接报错。"""
    kvs = [kv_filter] if kv_filter else list(KV_CHOICES)
    modes = [mode_filter] if mode_filter else list(MODE_CHOICES)
    for kv in kvs:
        if kv not in KV_CHOICES:
            raise SystemExit(f"P4_KV 非法: {kv!r}（可选 {KV_CHOICES}）")
    for m in modes:
        if m not in MODE_CHOICES:
            raise SystemExit(f"P4_MODE 非法: {m!r}（可选 {MODE_CHOICES}）")
    ctxs = list(ctxs if ctxs is not None else CTX)
    if not ctxs:
        raise SystemExit("P4_CTX 为空")
    rev = REVERSE if reverse is None else reverse
    if rev:
        kvs = list(reversed(kvs))
    return ctxs, kvs, modes


def build_matrix(ctxs, kvs, modes) -> list[tuple[int, str, str]]:
    return [(c, kv, m) for c in ctxs for kv in kvs for m in modes]


def classify_matrix(ctxs_present, kvs_present, modes_present, n_measured: int) -> tuple[str, bool]:
    """返回 (标签, 是否允许直接给出 per-family 外推)。标签基于**实际测量到的**条件集合。"""
    ctxs = tuple(sorted(ctxs_present))
    kvs = tuple(sorted(kvs_present))
    modes = tuple(sorted(modes_present))
    full = (len(kvs) == 2 and len(modes) == 2)
    if full and ctxs == tuple(sorted(FORMAL_CTX)):
        return "formal 12-condition per-family cost", True
    if full and ctxs == tuple(sorted(EXPLORATORY_CTX)):
        return "exploratory 12-condition cost", True
    if len(ctxs) == 1 and full:
        return f"{ctxs[0]//1024}K {n_measured}-condition measured cost", False
    return f"subset ({n_measured} conditions, ctx={list(ctxs)}, kv={list(kvs)}, " \
           f"mode={list(modes)}) measured cost", False


def resolve_jsonl(outdir: Path, chash: str, run_id: str, append_ok: bool) -> Path:
    """定名并**拒绝静默覆盖**：已存在且不在同 RUN_ID 组内 → SystemExit。"""
    path = outdir / f"part4_{chash}_{run_id}.jsonl"
    if path.exists() and not append_ok:
        raise SystemExit(f"[拒绝覆盖] {path} 已存在；换一个 P4_RUN_ID，或设置 P4_ONLY 以追加到同一组")
    return path


def rel_spread(vals) -> dict:
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return dict(median=float("nan"), min=float("nan"), max=float("nan"),
                    spread=float("nan"), rel_spread_pct=None, n=0)
    med = statistics.median(vals)
    spread = max(vals) - min(vals)
    rel = (100.0 * spread / abs(med)) if med else None
    return dict(median=med, min=min(vals), max=max(vals), spread=spread,
                rel_spread_pct=rel, n=len(vals))


def aggregate_finite(values) -> dict:
    """只聚合**有限**数值：NaN / None / 非数值一律剔除，并记录剔除了几个。
    全部缺失 -> available=False（**不得填 0 或 NaN 冒充**）。"""
    seq = list(values)
    vals = [float(v) for v in seq if isinstance(v, (int, float)) and v == v]
    dropped = len(seq) - len(vals)
    if not vals:
        return dict(available=False, n=0, dropped_nonfinite=dropped, median=None,
                    min=None, max=None, max_rel_spread_pct=None)
    med = statistics.median(vals)
    spread = max(vals) - min(vals)
    return dict(available=True, n=len(vals), dropped_nonfinite=dropped, median=med,
                min=min(vals), max=max(vals), spread=spread,
                max_rel_spread_pct=(100.0 * spread / abs(med)) if med else None)


def order_stability(medians: dict, limit: float = REL_SPREAD_LIMIT) -> dict:
    """**跨顺序**稳定性：order_ratio = min(median)/max(median)，
    relative_order_difference = (max-min)/max。>limit% -> CHECK（仅识别测量不稳定）。
    与 within-process rep spread 是两种不同的不确定性，必须分开报告。"""
    vals = {k: v for k, v in (medians or {}).items()
            if isinstance(v, (int, float)) and v == v and v}
    if len(vals) < 2:
        return dict(available=False, reason=f"需要 >=2 个顺序的中位数，实际 {sorted(vals)}",
                    medians=vals, order_ratio=None,
                    relative_order_difference_pct=None, flag="unavailable")
    lo, hi = min(vals.values()), max(vals.values())
    rel = 100.0 * (hi - lo) / hi
    return dict(available=True, medians=vals, order_ratio=lo / hi,
                relative_order_difference_pct=rel,
                flag=(f"CHECK:between-order>{limit:g}%" if rel > limit else "ok"))


def _append_reason(old: str | None, new: str) -> str:
    """把两条原因用 '；' 拼起来，**不产生**以 ' | ' 开头的半截字符串。"""
    return new if not old else f"{old}；{new}"


def token_compare(ar: dict, spec: dict) -> dict:
    """比较 AR 与 SpecDec 的生成序列（可跨进程/跨条件）。

    先校验可比性（ctx / max_new / eos / **input token hash**），再判等；不等时给首个差异位置
    与两侧 token id。缺 AR 对照时**绝不返回 False**：`ar_equal=None, ar_equal_available=False`。
    """
    out = {"ar_equal": None, "ar_equal_available": False, "certified": False,
           "certified_reason": None, "first_diff_index": None, "ar_token_id": None,
           "spec_token_id": None, "ar_len": None, "spec_len": None, "preconditions": {}}
    if ar is None or spec is None:
        out["certified_reason"] = "缺少 AR 或 SpecDec 对照（单条件进程内不可判定）"
        return out
    pre = {
        "ctx_equal": ar.get("ctx") == spec.get("ctx"),
        "max_new_equal": ar.get("max_new") == spec.get("max_new"),
        "eos_equal": ar.get("eos_id") == spec.get("eos_id"),
        "input_token_hash_equal": (ar.get("input_token_hash") is not None
                                   and ar.get("input_token_hash") == spec.get("input_token_hash")),
    }
    out["preconditions"] = pre
    out["ar_len"] = ar.get("n_generated", ar.get("n_gen"))
    out["spec_len"] = spec.get("n_generated", spec.get("n_gen"))
    a_h = ar.get("generated_token_hash") or ar.get("gen_sha1")
    s_h = spec.get("generated_token_hash") or spec.get("gen_sha1")
    out["ar_hash"], out["spec_hash"] = a_h, s_h
    if a_h is None or s_h is None:
        out["certified_reason"] = "generated_token_hash 缺失"
        return out
    out["ar_equal_available"] = True
    out["ar_equal"] = (a_h == s_h)
    hard = [k for k in ("ctx_equal", "max_new_equal", "eos_equal") if not pre[k]]
    if hard:
        out["certified_reason"] = "前提不满足: " + ",".join(hard)
    elif not pre["input_token_hash_equal"]:
        out["certified_reason"] = ("input_token_hash 未记录或不同（本字段在修复后才加入）"
                                   "=> 判定可比性未认证")
    else:
        out["certified"] = True
    a_ids = ar.get("generated_token_ids")
    s_ids = spec.get("generated_token_ids")
    if out["ar_equal"] is False:
        if a_ids is not None and s_ids is not None:
            for i, (x, y) in enumerate(zip(a_ids, s_ids)):
                if x != y:
                    out["first_diff_index"] = i
                    out["ar_token_id"], out["spec_token_id"] = x, y
                    break
            else:
                out["first_diff_index"] = min(len(a_ids), len(s_ids))
                out["certified_reason"] = _append_reason(out["certified_reason"], "前缀相同但长度不同")
        else:
            # **不再**拼接出以 " | " 开头的半截字符串（旧版会写出 certified_reason=' | 缺 ...'）
            missing = [n for n, v in (("generated_token_ids(AR)", a_ids),
                                      ("generated_token_ids(SpecDec)", s_ids)) if v is None]
            out["certified_reason"] = _append_reason(
                out["certified_reason"], "缺 " + "、".join(missing) + "，无法给出 first_diff_index")
    return out


NO_REF_REASON = ("本进程未运行同档 AR 对照（一条件一进程，AR/SpecDec 不共存）"
                 "=> 由父进程跨进程比较填入")


def in_process_equality(ref_gen: dict, ctx: int, kv: str, runs: list[dict], eos,
                        max_new: int | None = None) -> dict:
    """**进程内** AR↔SpecDec 等价性判定（可单独测试的接缝）。

    铁律：`ref_gen` 里没有同档 AR 序列时**不得判定**，返回
    `ar_equal=None / ar_equal_available=False`（"没比过" ≠ "不相等"）。
    2026-09-22 的事故就是这里退化成"拿空序列的 hash 去比"，凭空写出 `ar_equal=False, certified=True`。
    """
    ref = ref_gen.get((ctx, kv))
    if ref is None:
        return dict(ar_equal=None, ar_equal_available=False, certified=False,
                    certified_reason=NO_REF_REASON, equality_source=None,
                    first_diff_index=None)
    base = dict(ctx=ctx, target_kv=kv, max_new=(MAXNEW if max_new is None else max_new),
                eos_id=eos,
                input_token_hash=(runs[0].get("input_token_hash") if runs else None))
    c = token_compare(
        dict(base, n_generated=len(ref), generated_token_hash=gen_sha1(ref),
             generated_token_ids=ref),
        dict(base, n_generated=(runs[0]["n_gen"] if runs else None),
             generated_token_hash=(runs[0].get("generated_token_hash") if runs else None),
             generated_token_ids=(runs[0].get("generated_token_ids") if runs else None)))
    c["equality_source"] = "in_process"
    return c


def compare_across_conditions(rows: list[dict]) -> dict:
    idx = {}
    for r in rows:
        key = (r.get("ctx"), r.get("target_kv"), r.get("order"), r.get("mode"))
        if key[3] in ("AR", "SpecDec") and key not in idx:
            idx[key] = r
    out = {}
    keys = sorted({(k[0], k[1], k[2]) for k in idx}, key=lambda x: tuple(str(v) for v in x))
    for (ctx, kv, order) in keys:
        if (ctx, kv, order, "AR") in idx and (ctx, kv, order, "SpecDec") in idx:
            out[(ctx, kv, order)] = token_compare(idx[(ctx, kv, order, "AR")],
                                                  idx[(ctx, kv, order, "SpecDec")])
    return out


def gen_sha1(gen) -> str:
    return hashlib.sha1(json.dumps(list(gen)).encode()).hexdigest()[:16]


STAT_KEYS = ("rounds", "proposed", "accepted", "target_block_calls", "draft_calls",
             "mismatch_rounds", "all_accept_rounds")


def summarize(runs: list[dict]) -> dict:
    keys_num = ["t_prefill_s", "t_decode_s", "t_total_s", "decode_tok_s", "e2e_tok_s",
                "peak_alloc_GiB", "peak_resv_GiB", "accept_rate", "rounds",
                "proposed", "accepted", "target_block_calls", "draft_calls",
                "target_decode_calls", "mismatch_rounds", "all_accept_rounds"]
    skip = keys_num + ["gen", "m_hist", "gen_sha1", "monitor_start", "monitor_end",
                       "cuda_start", "cuda_end", "ts_start", "ts_end", "pid"]
    s = {k: v for k, v in runs[0].items() if k not in skip}
    for k in keys_num:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        vals = [v for v in vals if v == v]
        if vals:
            s[k + "_med"] = statistics.median(vals)
            s[k + "_min"] = min(vals)
            s[k + "_max"] = max(vals)
        else:
            s[k + "_med"] = float("nan")
            s[k + "_min"] = float("nan")
            s[k + "_max"] = float("nan")
    s["t_decode_spread"] = s["t_decode_s_max"] - s["t_decode_s_min"]
    s["m_hist"] = [m for r in runs for m in r.get("m_hist", [])]
    s["reps"] = len(runs)
    # ---- 稳定性/一致性 ----
    for key, name in (("decode_tok_s", "decode_tok_s"), ("t_decode_s", "t_decode_s")):
        rs = rel_spread([r.get(key) for r in runs])
        s[f"{name}_rel_spread_pct"] = rs["rel_spread_pct"]
    s["n_gen_consistent"] = len({r.get("n_gen") for r in runs}) == 1
    s["gen_sha1_consistent"] = len({r.get("gen_sha1") for r in runs}) == 1
    s["gen_sha1"] = runs[0].get("gen_sha1")
    s["generated_token_hash"] = runs[0].get("generated_token_hash") or runs[0].get("gen_sha1")
    s["generated_token_ids"] = runs[0].get("generated_token_ids")
    s["input_token_hash"] = runs[0].get("input_token_hash")
    s["input_tokens"] = runs[0].get("input_tokens")
    s["max_new"] = runs[0].get("max_new")
    s["eos_id"] = runs[0].get("eos_id")
    s["stats_consistent"] = len({tuple(r.get(k) for k in STAT_KEYS) for r in runs}) == 1
    s["logits_finite"] = all(bool(r.get("prefill_logits_finite")) for r in runs) and \
        all(bool(r.get("step_logits_finite", True)) for r in runs)
    s["finite_check_scope"] = ("prefill logits + AR 每步 logits；SpecDec 验证块行未扫描"
                               "（需要引擎插桩，超出本轮范围）")
    lim = REL_SPREAD_LIMIT
    s["spread_flag"] = ("CHECK:relative-spread>%g%%" % lim
                        if (s.get("decode_tok_s_rel_spread_pct") or 0) > lim
                        or (s.get("t_decode_s_rel_spread_pct") or 0) > lim else "ok")
    # 单条件进程内没有 AR 对照 ⇒ 保留 null/unavailable，等父进程跨条件填
    s.setdefault("ar_equal", None)
    s.setdefault("ar_equal_available", False)
    s.setdefault("certified", False)
    s.setdefault("certified_reason", "单条件进程内无 AR 对照（由父进程跨进程比较填入）")
    s["reserved_semantics"] = RESERVED_SEMANTICS
    return s


# ---------------------------------------------------------------- 环境监控
MON_KEYS = ["gpu_name", "mem_used_MiB", "mem_total_MiB", "temp_C", "power_W",
            "clock_sm_MHz", "clock_mem_MHz", "driver"]
MON_NUM_KEYS = ("mem_used_MiB", "mem_total_MiB", "temp_C", "power_W",
                "clock_sm_MHz", "clock_mem_MHz")
MON_QUERY = ("name,memory.used,memory.total,temperature.gpu,power.draw,"
             "clocks.sm,clocks.mem,driver_version")


def nvml_snapshot() -> dict:
    """nvidia-smi 快照。数值字段转成 float；取不到/解析失败一律 None（**不得用 0 冒充**），
    并在 `unparsed_fields` 里点名。"""
    try:
        p = subprocess.run(["nvidia-smi", f"--query-gpu={MON_QUERY}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    if p.returncode != 0:
        return {"available": False, "error": f"nvidia-smi exit {p.returncode}: "
                                             f"{(p.stderr or '').strip()[:200]}"}
    lines = (p.stdout or "").strip().splitlines()
    if not lines:
        return {"available": False, "error": "nvidia-smi 无输出"}
    vals = [v.strip() for v in lines[0].split(",")]
    d: dict = {"available": len(vals) == len(MON_KEYS), "unparsed_fields": []}
    for k, v in zip(MON_KEYS, vals):
        if k in MON_NUM_KEYS:
            try:
                d[k] = float(v)
            except ValueError:              # 例如 "[N/A]" / "Not Supported"
                d[k] = None
                d["unparsed_fields"].append(k)
        else:
            d[k] = v
    if not d["available"]:
        d["error"] = f"字段数不匹配: {len(vals)} != {len(MON_KEYS)}"
    return d


def cuda_snapshot() -> dict:
    try:
        free, total = torch.cuda.mem_get_info()
        return {"available": True,
                "cuda_free_GiB": free / 2**30, "cuda_total_GiB": total / 2**30,
                "torch_allocated_GiB": torch.cuda.memory_allocated() / 2**30,
                "torch_reserved_GiB": torch.cuda.memory_reserved() / 2**30}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def allocator_info() -> dict:
    d: dict = {"available": True,
               "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
               "empty_cache_in_timed_region": False}
    try:
        d["allocator_backend"] = torch.cuda.get_allocator_backend()
    except Exception as e:  # noqa: BLE001
        d["allocator_backend"] = None
        d["allocator_backend_error"] = f"{type(e).__name__}: {e}"
    return d


# ---------------------------------------------------------------- 测量
def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def load_text() -> tuple[str, str]:
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
        lines = [ln.strip() for ln in raw.splitlines()
                 if ln.strip() and not ln.strip().startswith("=")]
        return " ".join(lines), f"wikitext-2-raw-v1/{Path(pick).name}"
    return ("Speculative decoding uses a small draft model to propose tokens and a "
            "larger target model to verify them. " * 400), "FALLBACK-SYNTHETIC(不得用于正式实验)"


def chunked_prefill(model, ids, chunk: int = CHUNK):
    with torch.inference_mode():
        cache = None
        lg = None
        n = ids.shape[1]
        for st in range(0, n, chunk):
            piece = ids[:, st:st + chunk]
            lg, cache = model._run(model.embed_tokens(piece), cache,
                                   need_logits=(st + chunk >= n))
    if lg is not None:
        FINITE["ok"] &= bool(torch.isfinite(lg).all().item())
    return lg, cache


def ar_decode(tgt, ids, n, eos, kv, chunk: int = CHUNK):
    """AR 同档基线：分块 prefill + 逐 token `_step`（与引擎 generate 同语义）。"""
    tgt.set_kv_format(spec_dec._fmt(kv))
    t0 = sync_time()
    lg, cache = chunked_prefill(tgt, ids, chunk)
    t1 = sync_time()
    gen = []
    fwd = 0
    step_finite = True
    cur = lg[:, -1:]
    for i in range(n):
        tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
        gen.append(tok)
        if eos is not None and tok == eos:
            break
        if i == n - 1:
            break
        cur, cache = spec_dec._step(tgt, tok, cache)
        step_finite &= bool(torch.isfinite(cur).all().item())
        fwd += 1
    t2 = sync_time()
    return gen, t1 - t0, t2 - t1, {"target_decode_calls": fwd}, step_finite


def spec_decode_timed(tgt, drf, ids, n, eos, kv, chunk: int = CHUNK):
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
            torch.cuda.max_memory_reserved() / 2**30)


def run_once(tgt, drf, ids, eos, kv, mode):
    """**不在计时区内 empty_cache()**：它会把缓存块全还给驱动，让每个 rep 从零重建显存池，
    把上千次 cudaMalloc/WDDM 映射算进 decode 时间（实测可使同一格从 6.44s 变 10.87s）。"""
    from .. import results_schema as _S
    role = _S.resolve_condition_role(mode, AR_RESIDENCY if mode == "AR" else None)
    if mode == "SpecDec" and drf is None:
        raise RuntimeError("SpecDec 必须双模型驻留：draft 未加载")
    if mode == "AR" and role["ar_residency"] == "target_only" and drf is not None:
        raise RuntimeError("AR target_only 不得在本进程加载 draft（否则不再是 target-only 口径）")
    FINITE["ok"] = True
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated() / 2**30
    mon0, cuda0, ts0 = nvml_snapshot(), cuda_snapshot(), time.time()
    trace: list[dict] = []
    step_finite = True
    if mode == "AR":
        gen, tp, td, st, step_finite = ar_decode(tgt, ids, MAXNEW, eos, kv)
    else:
        gen, tp, td, st, trace = spec_decode_timed(tgt, drf, ids, MAXNEW, eos, kv)
    ts1 = time.time()
    mon1, cuda1 = nvml_snapshot(), cuda_snapshot()
    pa, pr = mem_read()
    row = dict(mode=mode, target_kv=kv, ctx=int(ids.shape[1]), n_gen=len(gen),
               t_prefill_s=tp, t_decode_s=td, t_total_s=tp + td,
               peak_alloc_GiB=pa, peak_resv_GiB=pr, weights_resident_GiB=base_alloc,
               prefill_logits_finite=bool(FINITE["ok"]),
               step_logits_finite=bool(step_finite),
               gen_sha1=gen_sha1(gen), ts_start=ts0, ts_end=ts1, t_wall_s=ts1 - ts0,
               monitor_start=mon0, monitor_end=mon1, cuda_start=cuda0, cuda_end=cuda1,
               pid=os.getpid(), order=ORDER_LABEL, run_id=RUN_ID,
               # ---- 正确性比较所需的稳定身份字段（跨进程用）----
               input_tokens=int(ids.shape[1]),
               input_token_hash=hashlib.sha1(json.dumps(ids[0].tolist()).encode()).hexdigest()[:16],
               max_new=MAXNEW, eos_id=eos,
               n_generated=len(gen),
               generated_token_hash=gen_sha1(gen),
               generated_token_ids=[int(x) for x in gen],   # max_new<=64，体积可接受
               k=K,                                          # 实际使用的 k（正式=4）
               k_is_protocol=(K == PROTOCOL_K),
               # ---- 单条件进程内**没有** AR 对照 ⇒ 不得填 False ----
               ar_equal=None, ar_equal_available=False,
               reserved_semantics=RESERVED_SEMANTICS,
               # ---- 资源口径（v1.1）：驻留模型 / AR 主基线 vs 诊断控制 ----
               ar_residency=role["ar_residency"],
               resident_models=["target"] + (["draft"] if drf is not None else []),
               condition_role=role["condition_role"],
               formal_factorial_member=role["formal_factorial_member"],
               speedup_denominator_eligible=role["speedup_denominator_eligible"])
    row["decode_tok_s"] = len(gen) / td if td > 0 else float("nan")
    row["e2e_tok_s"] = len(gen) / (tp + td) if (tp + td) > 0 else float("nan")
    if mode == "AR":
        row.update({k: 0 for k in STAT_KEYS})
        row["accept_rate"] = float("nan")
        row["target_decode_calls"] = st["target_decode_calls"]
        row["m_hist"] = []
    else:
        row.update({k: st[k] for k in STAT_KEYS})
        row["accept_rate"] = st["accepted"] / st["proposed"] if st["proposed"] else float("nan")
        row["m_hist"] = [d["m"] for d in trace]
    row["gen"] = gen
    return row


# ---------------------------------------------------------------- 报表
def print_report(summary: dict, ctxs, kvs, modes=("AR", "SpecDec"),
                 comparisons: dict | None = None) -> None:
    ctxs = sorted(ctxs)
    print("\n== 汇总（中位数；rel_spread>20% 判 CHECK，仅用于识别测量不稳定）==")
    print("ctx    kv         mode     dec_tok/s  e2e_tok/s  pref_s  dec_s  spread  rel%   "
          "accept  rounds  block  draft  peak_alloc  peak_resv*  ==AR      flag")
    print("  * peak_alloc=PyTorch 实际张量分配（主指标）；peak_resv=分配器池/虚拟预留(≠物理显存)；"
          "物理口径看 nvidia-smi used")
    print("  * ==AR: True/False 需**有 AR 对照**才可判定；单条件进程内为 n/a(unavailable)，"
          "不得读作 False")
    for ctx in ctxs:
        for kv in kvs:
            for mode in modes:
                s = summary.get((ctx, kv, mode))
                if not s or s.get("oom"):
                    print(f"{ctx:<6} {kv:<10} {mode:<8} OOM/未测")
                    continue
                if s.get("ar_equal_available"):
                    te = f"{s.get('ar_equal')}" + ("" if s.get("certified") else "(未认证)")
                else:
                    te = "n/a"
                rr = s.get("decode_tok_s_rel_spread_pct")
                ac = s.get("accept_rate_med")
                acs = f"{ac:>6.3f}" if isinstance(ac, (int, float)) and ac == ac else "   n/a"
                print(f"{ctx:<6} {kv:<10} {mode:<8} {s['decode_tok_s_med']:>9.2f}  "
                      f"{s['e2e_tok_s_med']:>9.2f}  {s['t_prefill_s_med']:>6.2f} "
                      f"{s['t_decode_s_med']:>6.2f}  {s['t_decode_spread']:>6.2f}  "
                      f"{(f'{rr:.1f}' if rr is not None else 'n/a'):>5}  {acs}  "
                      f"{s['rounds_med']:>6.0f} {s['target_block_calls_med']:>6.0f} "
                      f"{s['draft_calls_med']:>6.0f}  {s['peak_alloc_GiB_med']:>9.2f}  "
                      f"{s['peak_resv_GiB_med']:>9.2f}  {te:<9} {s.get('spread_flag', '')}")

    print("\n== 逐条件一致性（n_gen / token hash / stats / finite / within-run rel_spread）==")
    for ctx in ctxs:
        for kv in kvs:
            for mode in modes:
                s = summary.get((ctx, kv, mode))
                if not s or s.get("oom"):
                    continue
                print(f"  ctx={ctx:<5} {kv:<10} {mode:<8} reps={s['reps']} "
                      f"n_gen同={s['n_gen_consistent']} hash同={s['gen_sha1_consistent']} "
                      f"stats同={s['stats_consistent']} finite={s['logits_finite']} "
                      f"rel%={s.get('decode_tok_s_rel_spread_pct')} flag={s.get('spread_flag')}")

    if comparisons:
        print("\n== AR vs SpecDec 跨条件 token 比较（父进程/离线聚合）==")
        for key, c in comparisons.items():
            ctx, kv, order = key
            head = (f"  ctx={ctx:<5} {kv:<10} order={str(order):<5} "
                    f"AR={c['ar_len']} tok / Spec={c['spec_len']} tok  "
                    f"equal={c['ar_equal']} (available={c['ar_equal_available']}, "
                    f"certified={c['certified']})")
            print(head)
            if c.get("first_diff_index") is not None:
                print(f"      first_diff_index={c['first_diff_index']} "
                      f"AR_token_id={c['ar_token_id']} SpecDec_token_id={c['spec_token_id']}")
            if c.get("preconditions"):
                bad = [k for k, v in c["preconditions"].items() if not v]
                print(f"      preconditions: "
                      f"{'全部满足' if not bad else '不满足: ' + ','.join(bad)}")
            if c.get("certified_reason"):
                print(f"      note: {c['certified_reason']}")
            if c.get("ar_hash"):
                print(f"      hash: AR={c['ar_hash']} SpecDec={c['spec_hash']}")

    if "AR" in modes and "SpecDec" in modes:
        print("\n== 同档比值（SpecDec / AR，中位数之比；**未做显著性检验、未控温漂**）==")
        for ctx in ctxs:
            for kv in kvs:
                a, s = summary.get((ctx, kv, "AR")), summary.get((ctx, kv, "SpecDec"))
                if not a or not s or a.get("oom") or s.get("oom"):
                    continue
                print(f"  ctx={ctx:<5} kv={kv:<10} "
                      f"decode×{s['decode_tok_s_med']/a['decode_tok_s_med']:.3f}  "
                      f"e2e×{s['e2e_tok_s_med']/a['e2e_tok_s_med']:.3f}  "
                      f"accept={s['accept_rate_med']:.3f}  "
                      f"AR前向={a['target_decode_calls_med']:.0f} "
                      f"Spec块+单步={s['target_block_calls_med']:.0f}+{s['mismatch_rounds_med']:.0f}")

    if "SpecDec" in modes:
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

    # ---- 标定：只基于实际测到的条件 ----
    measured = [s for s in summary.values() if not s.get("oom")]
    ctxs_p = sorted({s["ctx"] for s in measured})
    kvs_p = sorted({s["target_kv"] for s in measured})
    modes_p = sorted({s["mode"] for s in measured})
    label, allow_extrap = classify_matrix(ctxs_p, kvs_p, modes_p, len(measured))
    per_family = sum(s["t_total_s_med"] for s in measured)
    print("\n== 规模标定（只基于实际运行的条件）==")
    print(f"  实际运行维度：ctx={ctxs_p}（{len(ctxs_p)}）× kv={kvs_p}（{len(kvs_p)}）"
          f"× mode={modes_p}（{len(modes_p)}）")
    print(f"  **本次实际运行 {len(measured)} conditions**"
          f"（{len(ctxs_p)}×{len(kvs_p)}×{len(modes_p)} = {len(ctxs_p)*len(kvs_p)*len(modes_p)}）"
          f"，标签 = {label}")
    print(f"  实测成本：每 family 走完这 {len(measured)} 个条件 = **{per_family:.1f}s**"
          f"（= 各条件 t_total 中位数之和）")
    if allow_extrap:
        for fams, reps in ((16, 5), (16, 3)):
            tot_h = per_family * fams * reps / 3600.0
            warm_h = per_family * fams * WARM / 3600.0
            print(f"  [{label}] ≥{fams} families × {len(measured)} conditions × {reps} reps "
                  f"≈ **{tot_h:.1f} GPU 小时**（不含失败重试；warm-up={WARM} 再加 ≈{warm_h:.1f}h）")
    else:
        print("  !! **本 subset 不做 per-family 外推**：缺少的维度未测量，不得当作已测。")
        print(f"     若确需外推，必须显式写假设，例如："
              f"assume_cost(missing ctx) = median(measured ctx) => "
              f"per_family ≈ {per_family:.1f}s × (12/{len(measured)}) = "
              f"{per_family*12/max(len(measured),1):.1f}s（**纯假设，不是测量值**）")
    print(f"  ※ 本次条件组（{len(measured)} conditions × {REPS} reps + warm-up={WARM}）"
          f"实测墙钟 ≈ {sum(s['t_total_s_med']*(REPS+WARM) for s in measured)/60:.1f} 分钟")


def load_summary(path: Path):
    """从 JSONL 重建 summary（--aggregate 用；不加载模型）。"""
    summary: dict = {}
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "summary":
            summary[(r["ctx"], r["target_kv"], r["mode"])] = r
        elif r.get("kind") == "run":
            rows.append(r)
    comp = compare_across_conditions(rows)
    for (ctx, kv, _o), c in comp.items():
        for s in summary.values():
            if s.get("ctx") == ctx and s.get("target_kv") == kv and s.get("mode") == "SpecDec":
                s["ar_equal"] = c["ar_equal"]
                s["ar_equal_available"] = c["ar_equal_available"]
                s["certified"] = c["certified"]
                s["certified_reason"] = c["certified_reason"]
                s["ar_equal_source"] = "token_compare(跨进程/跨条件)"
    return summary, sorted({k[0] for k in summary}), comp


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="G4 Part 4 测量 / 报表")
    ap.add_argument("--aggregate", nargs="?", const="", default=None,
                    help="不跑模型，只从 JSONL 重算报表（可跟路径；缺省取最新一份）")
    args = ap.parse_args()
    if args.aggregate is not None:
        if args.aggregate:
            path = Path(args.aggregate)
        else:
            cand = sorted(OUTDIR.glob("part4_*.jsonl"), key=lambda p: p.stat().st_mtime)
            if not cand:
                raise SystemExit(f"{OUTDIR} 下没有 part4_*.jsonl")
            path = cand[-1]
        summary, ctxs, comp = load_summary(path)
        print(f"[aggregate] {path}  ({len(summary)} 个条件)")
        print_report(summary, ctxs, ("base_bf16", "v_fp8"), comparisons=comp)
        return 0

    torch.manual_seed(0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ctxs, kvs, modes = parse_filters(KV_FILTER, MODE_FILTER, CTX, REVERSE)
    conditions = build_matrix(ctxs, kvs, modes)
    only = parse_only()
    if only is not None:
        conditions = [only]
        print(f"[cfg] P4_ONLY={only} -> 单条件独立进程模式", flush=True)
    # ---- 资源口径校验：target_only 的 AR 只能在不加载 draft 的进程里测 ----
    from .. import results_schema as _S
    try:
        plan = _S.plan_process([(c, kv, m) for c, kv, m in conditions], AR_RESIDENCY)
    except ValueError as e:
        raise SystemExit(f"[cfg] 条件/驻留组合非法: {e}")
    role0 = _S.resolve_condition_role("AR", AR_RESIDENCY)
    need_draft = plan["load_draft"]
    print(f"[cfg] filter: P4_CTX={CTX} P4_KV={KV_FILTER} P4_MODE={MODE_FILTER} "
          f"P4_REVERSE={REVERSE} P4_AR_RESIDENCY={AR_RESIDENCY} -> {len(conditions)} conditions",
          flush=True)
    print(f"[cfg] 角色: AR={role0['condition_role']}"
          f"(residency={role0['ar_residency']}, 分母资格={role0['speedup_denominator_eligible']})"
          f" | 本进程驻留={plan['resident_models']}", flush=True)
    print(f"[cfg] reps={REPS} max_new={MAXNEW} warm={WARM} chunk={CHUNK} k={K} "
          f"run_id={RUN_ID} order={ORDER_LABEL}", flush=True)

    text, src = load_text()
    print(f"[env] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[env] allocator={allocator_info()}", flush=True)
    print(f"[cfg] text={src} chars={len(text)}", flush=True)

    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    text_used = text[:400_000]
    if len(text_used) < len(text):
        print(f"[data] 截取前 {len(text_used)} 字符（原文 {len(text)}）", flush=True)
    all_ids = tok(text_used, return_tensors="pt").input_ids
    print(f"[data] corpus tokens={all_ids.shape[1]}（嵌套前缀取自同一段文本，未插合成填充）",
          flush=True)

    print("[stage] loading target 1.5B ...", flush=True)
    tgt, _ = load_ours(TARGET)
    drf = None
    if need_draft:
        print("[stage] loading draft 0.5B ...", flush=True)
        drf, _ = load_ours(DRAFT)
    else:
        print("[stage] draft 不加载（AR target_only 正式主基线口径）", flush=True)
    print(f"[mem] 驻留模型={['target'] + (['draft'] if drf is not None else [])} "
          f"alloc={torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    cfg = dict(ctx=CTX, reps=REPS, max_new=MAXNEW, k=K, chunk=CHUNK,
               target=TARGET, draft=DRAFT, text=src, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), warm=WARM,
               kv_order="reversed" if REVERSE else "base-first",
               filters=dict(P4_CTX=CTX, P4_KV=KV_FILTER, P4_MODE=MODE_FILTER,
                            P4_AR_RESIDENCY=AR_RESIDENCY),
               condition_list=[f"{c},{kv},{m}" for c, kv, m in conditions],
               resident_models=["target"] + (["draft"] if need_draft else []),
               prompt_encoding="tokenizer default, no chat template, raw text prefix",
               nested_prefix=True, filler="none (real wikitext text)",
               rel_spread_limit_pct=REL_SPREAD_LIMIT,
               note="hash 不含 P4_ONLY/RUN_ID（那是执行指令，不是实验身份）；每行带 condition+order")
    chash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    jsonl = resolve_jsonl(OUTDIR, chash, RUN_ID, append_ok=only is not None)
    append = only is not None and jsonl.exists()

    summary: dict[tuple, dict] = {}
    ref_gen: dict[tuple, list] = {}
    proc_row = dict(kind="process", config_hash=chash, run_id=RUN_ID, pid=os.getpid(),
                    only=ONLY or None, order=ORDER_LABEL, ctx_filter=CTX, kv_filter=KV_FILTER,
                    mode_filter=MODE_FILTER, reps=REPS, warm=WARM, max_new=MAXNEW, chunk=CHUNK,
                    conditions=[f"{c},{kv},{m}" for c, kv, m in conditions],
                    ts_start=time.time(), monitor_start=nvml_snapshot(),
                    cuda_start=cuda_snapshot(), allocator=allocator_info())
    with jsonl.open("a" if append else "w", encoding="utf-8") as fh:
        if not append:
            fh.write(json.dumps(dict(kind="config", config_hash=chash, **cfg),
                                ensure_ascii=False) + "\n")
        fh.write(json.dumps(proc_row, ensure_ascii=False) + "\n")
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
                fh.write(json.dumps(dict(kind="run", config_hash=chash, rep=rep, **{
                    k: v for k, v in r.items() if k != "gen"}), ensure_ascii=False) + "\n")
                fh.flush()
            if not runs:
                summary[(ctx, kv, mode)] = dict(oom=True, mode=mode, target_kv=kv, ctx=ctx)
                continue
            s = summarize(runs)
            s["oom"] = oom
            summary[(ctx, kv, mode)] = s
            if mode == "AR":
                ref_gen[(ctx, kv)] = runs[0]["gen"]
            if mode == "SpecDec":
                # **只有本进程真的跑过同档 AR，才有资格在进程内判定**。
                # 旧版判断条件是 `only is None and mode == "SpecDec"`，但驱动器按"一条件一进程"
                # 启动时只会给单一 P4_MODE（AR 与 SpecDec 从不在同一进程），于是 ref_gen 为空，
                # 却仍然拿 `gen_sha1([])` 去比 —— 得出 `ar_equal=False` 且 `certified=True`
                # 的**凭空否定**结论（2026-09-22 的 canonical 就是这样错的）。
                eq = in_process_equality(ref_gen, ctx, kv, runs, eos, max_new=MAXNEW)
                s["ar_equal"] = eq["ar_equal"]
                s["ar_equal_available"] = eq["ar_equal_available"]
                s["certified"] = eq["certified"]
                s["certified_reason"] = eq["certified_reason"]
                if eq.get("equality_source"):
                    s["equality_source"] = eq["equality_source"]
                    s["ar_equal_source"] = "同进程 token_compare"
                else:
                    # 显式说明"本进程没有对照"，避免下游把 null 当成"没测"或"不等"
                    s["ar_equal_source"] = "无（本进程无 AR 对照）"
            print(f"  [{tag}] pref={s['t_prefill_s_med']:.2f}s "
                  f"dec={s['t_decode_s_med']:.2f}s[{s['t_decode_s_min']:.2f}.."
                  f"{s['t_decode_s_max']:.2f}] rel%="
                  f"{(s['decode_tok_s_rel_spread_pct'] or 0):.1f} "
                  f"dec_tok/s={s['decode_tok_s_med']:.2f} "
                  f"peak_alloc={s['peak_alloc_GiB_med']:.2f} "
                  f"finite={s['logits_finite']} n_gen同={s['n_gen_consistent']} "
                  f"hash同={s['gen_sha1_consistent']} "
                  + (f"accept={s['accept_rate_med']:.3f} rounds={s['rounds_med']:.0f} "
                     f"block={s['target_block_calls_med']:.0f} draft={s['draft_calls_med']:.0f} "
                     if mode == "SpecDec" else "")
                  + f"flag={s.get('spread_flag')}", flush=True)
            fh.write(json.dumps(dict(kind="summary", config_hash=chash, **s),
                                ensure_ascii=False) + "\n")

    comp = compare_across_conditions([s for s in summary.values() if not s.get("oom")])
    print_report(summary, ctxs, kvs, modes=modes, comparisons=comp)

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
