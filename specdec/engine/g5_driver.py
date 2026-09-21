"""g5_driver.py — G5 测量驱动层的共用工具（**不加载模型、不含 GPU 代码**，CPU 可测）。

职责
  1. `spawn_condition(env_overrides)`：把**一个条件**交给一个全新子进程执行
     （复用已通过验证的测量核心 `g4_part4_pilot_measure`；`P4_ONLY` 之外的过滤走 P4_CTX/P4_KV/P4_MODE）。
  2. `system_snapshot()`：nvidia-smi 快照（数值转 float；取不到 -> None，**不填 0**）。
  3. `read_part4_rows(jsonl)`：读子进程写下的原始行（只读，不改写）。
  4. `g5_meta(...)`：run 级元数据（commit/dirty/seed/block/condition_index/顺序）。

为什么把"每条件独立进程"做成驱动层的硬约束
  8 GB 卡上 8K 双模型时分配器池会越过物理显存，池状态会改变同一格耗时（历史实测 6.44s ↔ 10.87s）。
  只有每格从干净进程状态开始，格与格之间才可比；因此驱动层**不允许**在一个进程里跑多个条件
  （诊断用的 k sweep / stability probe 都走这里）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output"
PART4_DIR = OUTDIR / "part4"
G5_DIR = OUTDIR / "g5"

MON_KEYS = ["gpu_name", "mem_used_MiB", "mem_total_MiB", "temp_C", "power_W",
            "clock_sm_MHz", "clock_mem_MHz", "driver"]
MON_NUM = ("mem_used_MiB", "mem_total_MiB", "temp_C", "power_W", "clock_sm_MHz", "clock_mem_MHz")
MON_QUERY = ("name,memory.used,memory.total,temperature.gpu,power.draw,"
             "clocks.sm,clocks.mem,driver_version")


def system_snapshot() -> dict:
    """nvidia-smi 快照；数值字段转 float，取不到/不可解析 -> None（不填 0）。"""
    try:
        p = subprocess.run(["nvidia-smi", f"--query-gpu={MON_QUERY}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    if p.returncode != 0:
        return {"available": False, "error": f"nvidia-smi exit {p.returncode}"}
    lines = (p.stdout or "").strip().splitlines()
    if not lines:
        return {"available": False, "error": "nvidia-smi 无输出"}
    vals = [v.strip() for v in lines[0].split(",")]
    d: dict = {"available": len(vals) == len(MON_KEYS), "unparsed_fields": []}
    for k, v in zip(MON_KEYS, vals):
        if k in MON_NUM:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = None
                d["unparsed_fields"].append(k)
        else:
            d[k] = v
    return d


def cuda_snapshot() -> dict:
    try:
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "cuda_free_GiB": free / 2**30,
                "cuda_total_GiB": total / 2**30,
                "torch_allocated_GiB": torch.cuda.memory_allocated() / 2**30}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def spawn_condition(*, ctx: int, kv: str, mode: str, run_id: str, reps: int, warm: int,
                    maxnew: int, chunk: int, k: int | None = None, order: str | None = None,
                    ar_residency: str | None = None, extra_env: dict | None = None,
                    quiet: bool = False) -> dict:
    """在**全新子进程**中跑一个条件。返回执行元信息（退出码/墙钟/起止时间）。

    计时完全由子进程内部负责；父进程只做编排，因此模型加载/tokenizer/文件写入都不进入
    prefill/decode 计时。

    `ar_residency` 只在 `mode == "AR"` 时有意义：
      - `target_only`（默认，正式主基线）：子进程**只加载 target**，从不加载 draft；
      - `dual_control`（诊断控制）：加载双模型，但 draft 不参与 AR 的 prefill/decode。
    """
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "PYTHONIOENCODING": "utf-8",
        "P4_CTX": str(ctx), "P4_KV": kv, "P4_MODE": mode,
        "P4_REPS": str(reps), "P4_WARM": str(warm),
        "P4_MAXNEW": str(maxnew), "P4_CHUNK": str(chunk),
        "P4_RUN_ID": run_id,
    })
    if mode == "AR":
        env["P4_AR_RESIDENCY"] = ar_residency or "target_only"
    else:
        env.pop("P4_AR_RESIDENCY", None)
    if k is not None:
        env["P4_K"] = str(k)
    if order:
        env["P4_ORDER"] = order
    for key in ("P4_ONLY", "P4_REVERSE"):
        env.pop(key, None)
    env.update(extra_env or {})
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "specdec.engine.g4_part4_pilot_measure"],
                       env=env, cwd=str(ROOT),
                       stdout=subprocess.DEVNULL if quiet else None)
    return dict(ctx=ctx, kv=kv, mode=mode, k=k, run_id=run_id, order=order,
                ar_residency=(env.get("P4_AR_RESIDENCY") if mode == "AR" else None),
                returncode=r.returncode, ts_start=t0, ts_end=time.time(),
                wall_s=time.time() - t0)


def read_part4_rows(run_id: str) -> tuple[list[dict], list[dict], Path | None]:
    """读某 run_id 对应的 part4 JSONL（只读）。run_id 由驱动层保证"一条件一文件"。"""
    hits = sorted(PART4_DIR.glob(f"part4_*_{run_id}.jsonl"))
    if not hits:
        return [], [], None
    path = hits[-1]
    runs, sums = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "run":
            runs.append(r)
        elif r.get("kind") == "summary":
            sums.append(r)
    return runs, sums, path


def g5_meta(*, run_id: str, block_index: int, condition_index: int, seed: int,
            condition_order: list[str], timestamp_utc: str | None = None,
            notes: str = "", extra: dict | None = None) -> dict:
    from .. import results_schema as S
    meta = dict(run_id=run_id, block_index=block_index, condition_index=condition_index,
                seed=seed, condition_order=list(condition_order),
                timestamp_utc=timestamp_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                notes=notes)
    meta.update(S.git_meta(ROOT))
    meta.update(extra or {})
    return meta


def randomized_order(items: list, seed: int) -> list:
    """固定 seed 的确定性随机化（可复现；记录 seed 后任何人可重放同一顺序）。"""
    import random
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out
