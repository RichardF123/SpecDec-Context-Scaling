"""g5_chunk_fit_sweep.py — **chunk 分档拟合扫描**：在不改引擎的前提下，找出 8K SpecDec 能跑通的最大 chunk。

为什么是 chunk
  §4.16 实测的 OOM 是手写 attention 里 `softmax(scores, fp32)` 的那一笔，大小 =
  `heads × chunk × kv_len × 4 B`（最后一个 chunk 的 kv_len = 全长）：
      chunk 2048 → draft **896 MiB** / target 806 MiB   ← 实测崩点（请求 768 MiB，空闲仅 771 MiB）
      chunk 1024 → 448 / 403 MiB
      chunk  512 → 224 / 202 MiB
  chunk 是**调用侧参数**（`model_self._run` 接受任意 chunk），所以这是**零引擎改动**的手段。

策略
  从大往小扫（2048 → 1024 → 512），**第一个成功的档就停**：chunk 越大与既有
  chunk=2048 数据的可比性越好（2K 结果本来就只有一块，完全不受影响；只有 4K/8K 的
  prefill 口径会变，必须单独标注）。

安全
  每次尝试**之前**都查 commit 护栏（2026-09-24 的整机硬挂就是 commit 压力）；
  每个 chunk 起一个**全新子进程**，只跑 spec 角色、warm1 + rep1、不做插桩 ⇒ 最小暴露面。

用法（交互式 GPU 终端）
  python -m specdec.engine.g5_chunk_fit_sweep --dry-run     # 只打印计划与 CPU 自检
  python -m specdec.engine.g5_chunk_fit_sweep               # 逐档扫描（每档约 2 分钟）
可选：--chunks 2048,1024,512   --all（全部跑完，不在首个成功处停止）  --cooldown 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from specdec.engine import g5_8k_spec_variance_probe as P        # noqa: E402

OUTDIR = P.OUTDIR
DEFAULT_CHUNKS = (2048, 1024, 512)

# 预测：最后一个 chunk 的单笔 fp32 softmax（MiB）= heads × chunk × kv_len × 4 / 2**20
PREDICT = {(12, 2048): "target 806", (14, 2048): "draft 896",
           (12, 1024): "target 403", (14, 1024): "draft 448",
           (12, 512): "target 202", (14, 512): "draft 224"}


def predicted_mib(chunk: int, heads: int, kv_len: int = 8192) -> float:
    return heads * chunk * kv_len * 4 / 2 ** 20


def _num(text: str, pat: str):
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def run_one(chunk: int, run_id: str) -> dict:
    """跑一个 chunk 档：一个全新子进程，spec 角色，warm1 + rep1，无插桩。"""
    env = os.environ.copy()
    env.update(dict(HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8",
                    P5V_CHILD="1", P5V_ROLE=P.ROLE_SPEC, P5V_INDEX="0",
                    P5V_RUN_ID=run_id, P5V_CHUNK=str(chunk),
                    P5V_REPS="1", P5V_WARM="1", P5V_PROF_REPS="0"))
    print(f"\n=== chunk={chunk} （子进程 run_id={run_id}）===", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", "specdec.engine.g5_8k_spec_variance_probe"],
                        env=env, cwd=str(ROOT)).returncode
    wall = time.time() - t0

    out = dict(chunk=chunk, run_id=run_id, returncode=rc, wall_s=wall, status=None)
    child = P.OUTDIR / f"p5v_{run_id}_{P.ROLE_SPEC}_0.json"
    crash = P.OUTDIR / f"p5v_{run_id}_{P.ROLE_SPEC}_0_crash.txt"
    if child.exists():
        d = json.loads(child.read_text(encoding="utf-8"))
        skipped = d.get("skipped") or ""
        free_g = _num(skipped, r"commit 可用 ([\d.]+) GiB")
        if skipped:
            # **关键区分**：被护栏拒绝 ≠ 装不下。不要把 REFUSED 误报成"8K 跑不了"。
            out.update(status=("REFUSED_COMMIT" if "commit" in skipped else "SKIPPED"),
                       reason=skipped, commit_free_gib=free_g)
        else:
            f = d.get("formal") or {}
            med = f.get("medians") or {}
            a = d.get("alloc_after_formal") or {}
            out.update(status="FITS" if f else "NO_MEASUREMENT",
                       prefill_s=med.get("prefill_s"),
                       prefill_target_s=med.get("prefill_target_s"),
                       prefill_draft_s=med.get("prefill_draft_s"),
                       decode_s=med.get("decode_s"), decode_tok_s=med.get("decode_tok_s"),
                       within_rel_pct=f.get("decode_rel_spread_pct"),
                       peak_allocated_mib=a.get("peak_allocated_mib"),
                       peak_reserved_mib=a.get("peak_reserved_mib"),
                       inactive_split_mib=a.get("inactive_split_peak_mib"),
                       num_alloc_retries=a.get("num_alloc_retries"),
                       smi_used_after_load=(d.get("smi_used_after_load") or {}).get("mem_used_MiB"),
                       commit_free_gib=((d.get("system_memory_before_load") or {})
                                        .get("commit_free_mib") or 0) / 1024)
    elif crash.exists():
        txt = crash.read_text(encoding="utf-8", errors="replace")
        out.update(status="OOM" if "OutOfMemoryError" in txt else "CRASH",
                   reason=("native access violation" if "access violation" in txt
                           else "python exception"),
                   requested_mib=_num(txt, r"Tried to allocate ([\d.]+) MiB"),
                   cuda_free_mib=_num(txt, r"cuda_free_mib=([\d.]+)"),
                   peak_allocated_mib=_num(txt, r'"peak_allocated_mib": ([\d.]+)'),
                   peak_reserved_mib=_num(txt, r'"peak_reserved_mib": ([\d.]+)'),
                   inactive_split_mib=_num(txt, r'"inactive_split_peak_mib": ([\d.]+)'),
                   num_alloc_retries=_num(txt, r'"num_alloc_retries": (\d+)'),
                   crash_file=crash.name)
    else:
        out.update(status="NATIVE_CRASH" if rc not in (0, 3) else "SKIPPED",
                   reason=f"exit={rc}（无 crash 文件 ⇒ 原生崩溃，抓不到 Python 栈）",
                   crash_file=None)
    print(f"[sweep] chunk={chunk} -> {out['status']}"
          f"{('：' + str(out.get('reason'))[:80]) if out.get('reason') else ''}"
          f"（exit={rc}, {wall:.0f}s）", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="8K SpecDec chunk 分档拟合扫描（不改引擎）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chunks", default=",".join(str(c) for c in DEFAULT_CHUNKS))
    ap.add_argument("--all", action="store_true", help="全部跑完，不在首个成功处停止")
    ap.add_argument("--cooldown", type=float, default=10.0)
    args = ap.parse_args()
    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]

    print("== 预测：最后一个 chunk 的单笔 fp32 softmax ==")
    for c in chunks:
        print(f"  chunk={c:<5} target {predicted_mib(c, 12):7.1f} MiB | "
              f"draft {predicted_mib(c, 14):7.1f} MiB")
    ok, why = P.commit_guard()
    print(f"\n== commit 护栏：{why} ==")
    if not ok:
        print("!! 拒绝启动：先把 commit 余量腾出来（关游戏/Steam/浏览器，或调大 pagefile）")
        return 5
    print(f"将按 {'→'.join(str(c) for c in chunks)} 顺序扫描；每档：预检 + 载入 + warm1 + rep1 "
          f"（约 2 分钟），档间冷却 {args.cooldown:.0f}s")
    if args.dry_run:
        print("\n--dry-run：未使用 GPU、未写文件")
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rows = []
    for i, c in enumerate(chunks):
        if i and args.cooldown > 0:
            time.sleep(args.cooldown)
        ok, why = P.commit_guard()                  # 每档前复查（整机安全）
        if not ok:
            print(f"[sweep] !! 跑到 chunk={c} 前 commit 不足：{why} -> 中止扫描")
            break
        r = run_one(c, f"p5v_chunksweep_{stamp}_c{c}")
        rows.append(r)
        if r["status"] == "FITS" and not args.all:
            print(f"\n[sweep] chunk={c} 已跑通 -> 停止（更大的 chunk 可比性更好，故从大往小扫）")
            break

    print("\n== 汇总 ==")
    print(f"  {'chunk':>6}  {'结果':<16}{'commitGb':>9}{'请求MiB':>9}{'空闲MiB':>9}"
          f"{'peakAlloc':>10}{'peakResv':>9}{'inactSpl':>9}{'prefill_s':>10}"
          f"{'decode_s':>9}{'dec_tok/s':>10}")
    for r in rows:
        f = lambda k, nd=1: ("n/a" if r.get(k) is None else f"{r[k]:.{nd}f}")   # noqa: E731
        print(f"  {r['chunk']:>6}  {str(r['status'])[:16]:<16}{f('commit_free_gib'):>9}"
              f"{f('requested_mib',0):>9}{f('cuda_free_mib',0):>9}"
              f"{f('peak_allocated_mib',0):>10}{f('peak_reserved_mib',0):>9}"
              f"{f('inactive_split_mib',0):>9}{f('prefill_s',2):>10}"
              f"{f('decode_s',2):>9}{f('decode_tok_s',2):>10}")
        if r.get("reason") and r["status"] != "FITS":
            print(f"          ↳ {str(r['reason'])[:150]}")
    fits = [r["chunk"] for r in rows if r["status"] == "FITS"]
    refused = [r["chunk"] for r in rows if r["status"] == "REFUSED_COMMIT"]
    print()
    if fits:
        best = max(fits)
        print(f"⇒ 8K SpecDec 在 chunk={best} 下**可以跑**。"
              + ("（无需改配置，与既有 chunk=2048 数据同口径）" if best == 2048
                 else f" ⇒ 整轮须统一用 chunk={best}，并标注为 chunk={best} 臂"))
    elif refused and len(refused) == len(rows):
        print("⇒ **本轮的档位是被 commit 护栏挡下的，不是「装不下」** —— 与显存无关。"
              "请先腾 commit（关游戏/Steam/浏览器）或把 pagefile 设为固定的大值（如 32 GB），再重跑。")
    else:
        print("⇒ 本轮的 chunk 档位**真的未跑通**（非护栏原因）。下一步：腾显存 / 调大 pagefile，"
              "或考虑 attention 后端的 amendment（见台账 §4.17）。")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"p5v_chunksweep_{stamp}_summary.json"
    if p.exists():
        raise SystemExit(f"[拒绝覆盖] {p}")
    p.write_text(json.dumps(dict(stamp=stamp, chunks=chunks, rows=rows, commit=why,
                                 torch=torch.__version__,
                                 device=(torch.cuda.get_device_name(0)
                                         if torch.cuda.is_available() else None)),
                            ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
