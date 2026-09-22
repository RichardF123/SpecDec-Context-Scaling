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


def run_text(cmd: list[str], timeout: float = 20) -> tuple[int, str]:
    """在 Windows 上安全地跑外部命令并取文本输出。

    坑：`subprocess.run(..., text=True)` 会用**系统代码页**（本机 GBK）解码，而
    `powercfg` / `powershell` / `git` 的输出常是 UTF-8 或含中文 ⇒ `UnicodeDecodeError`
    会在**读取线程**里抛出（线程异常不致命但会让读取失败/挂起）。
    因此统一显式 `encoding='utf-8', errors='replace'`。
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "")
    except Exception as e:  # noqa: BLE001
        return -1, f"__RUN_FAILED__ {type(e).__name__}: {e}"


def platform_state() -> dict:
    """平台状态：电源方案 / 是否接交流电 / 当前 SM 时钟。

    为什么需要：2026-09-21 的 g5stab 运行中 SM 时钟在 **435–2535 MHz** 之间波动（低频 block 吞吐
    中位 6.16 tok/s vs 高频 17.67 tok/s，2.9×），温度只有 61–68 °C、功耗 15–28 W ⇒ **不是热降频，
    是供电/DVFS 状态变化**。这类状态会静默毁掉跨 cell 计时对比，因此必须在计时前检查并记录。
    """
    out: dict = {"available": True, "warnings": []}
    rc, txt = run_text(["powercfg", "/getactivescheme"], timeout=15)
    out["power_scheme"] = txt.strip() if rc == 0 else None
    if rc != 0:
        out["warnings"].append(f"powercfg 失败(rc={rc}): {txt.strip()[:80]}")
    rc, txt = run_text(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Battery | "
                        "Select-Object -First 1).BatteryStatus"], timeout=30)
    v = txt.strip()
    if rc == 0 and v.isdigit():
        out["battery_status_code"] = int(v)     # 1=放电(电池), 2=交流电
        out["on_ac"] = (int(v) == 2)
    else:
        out["on_ac"] = None
        out["warnings"].append(f"无法读取电池状态(rc={rc}): {v[:80]!r}")
    snap = system_snapshot()
    out["sm_clock_now"] = snap.get("clock_sm_MHz") if snap.get("available") else None
    out["nvidia_smi_available"] = bool(snap.get("available"))
    return out


def platform_gate(state: dict, *, require_ac: bool = True,
                  min_sm_clock: float | None = None) -> list[str]:
    """返回**阻断理由**列表（空 = 通过）。纯函数，便于 CPU 测试。

    **只做供电/电源方案层面的阻断**：`nvidia-smi` 空闲时 SM≈210–315 MHz 是正常节能状态，
    用它判资格会把合格机器误判为不合格（本机实测：GPU 空闲 315 MHz 但负载期 2520 MHz）。
    负载期资格由 `gpu_pstate_preflight.qualify()` 负责。
    仅当调用方**明确**传入 `min_sm_clock` 且 `state['sm_clock_now']` 确实来自负载期时，才检查时钟。
    """
    blockers = []
    if require_ac and state.get("on_ac") is False:
        blockers.append("当前在**电池供电**：dGPU 会被限频，计时无效。请插上电源后重跑")
    if require_ac and state.get("on_ac") is None:
        blockers.append("无法确认是否接交流电（on_ac=unknown）：正式计时要求可确认的 AC 供电")
    if min_sm_clock is not None:
        sm = state.get("sm_clock_now")
        if isinstance(sm, (int, float)) and sm < min_sm_clock:
            blockers.append(f"**负载期** SM 时钟 {sm:.0f} MHz < {min_sm_clock:.0f} MHz："
                            f"平台处于低频状态（请用 gpu_pstate_preflight 确认）")
    return blockers


def clock_stats(runs: list[dict]) -> dict:
    """从一个 block 的 run 行里统计 SM/显存时钟的**稳定性**（within-block）。
    判定（预登记）：`spread > 5%` 或 `median < 2000 MHz` ⇒ `stable=False`（该 cell 标 CHECK）。
    """
    def col(key):
        v = []
        for r in runs:
            m = r.get("monitor_end") or {}
            if not m.get("available"):
                continue
            try:
                v.append(float(m[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return v

    sm = sorted(col("clock_sm_MHz"))
    mm = sorted(col("clock_mem_MHz"))
    if not sm:
        return {"available": False, "stable": None, "reason": "monitor_end 不可用"}
    med = sm[len(sm) // 2] if len(sm) % 2 else 0.5 * (sm[len(sm) // 2 - 1] + sm[len(sm) // 2])
    spread = 100.0 * (sm[-1] - sm[0]) / med if med else None
    stable = (spread is not None and spread <= 5.0 and med >= 2000.0)
    reason = "ok" if stable else (
        f"gpu_clock_unstable(sm_med={med:.0f}MHz, spread={spread:.1f}%)")
    return {"available": True, "sm_median": med, "sm_min": sm[0], "sm_max": sm[-1],
            "sm_spread_pct": spread, "mem_median": (mm[len(mm) // 2] if mm else None),
            "stable": stable, "reason": reason}


# ------------------------------------------------ P-state（功耗档位）资格
MIN_SM = float(os.environ.get("P5_MIN_ACTIVE_SM_CLOCK_MHZ", "2000"))
MIN_MEM = float(os.environ.get("P5_MIN_ACTIVE_MEM_CLOCK_MHZ", "7500"))


def _num(samples, key):
    out = []
    for s in samples:
        try:
            out.append(float(s[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def med(v):
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def pstate_verdict(samples: list[dict], *, min_sm: float = MIN_SM,
                   min_mem: float = MIN_MEM) -> dict:
    """对**负载期间**的样本判定 active P-state 资格（纯函数，CPU 可测）。

    **只分析负载样本**：空闲时 SM≈210MHz / mem≈405MHz 属正常节能状态，用它判资格会把合格
    机器误判为不合格。功耗只作辅助证据，**不作硬 Gate**。
    """
    sm, mm, pw, tc = (_num(samples, "clock_sm_MHz"), _num(samples, "clock_mem_MHz"),
                      _num(samples, "power_W"), _num(samples, "temp_C"))
    n = max(len(sm), len(mm))
    if n == 0:
        return dict(available=False, qualified=None, reason="无可用负载样本（采样失败）",
                    samples=0, min_sm_required=min_sm, min_mem_required=min_mem)
    reasons = []
    sm_med, mm_med = med(sm), med(mm)
    if sm_med is None or sm_med < min_sm:
        reasons.append(f"active SM clock 中位 {sm_med} MHz < 阈值 {min_sm:.0f} MHz")
    if mm_med is None or mm_med < min_mem:
        reasons.append(f"active mem clock 中位 {mm_med} MHz < 阈值 {min_mem:.0f} MHz")
    return dict(available=True, qualified=(not reasons), samples=n,
                sm_median=sm_med, sm_min=(sm[0] if sm else None), sm_max=(sm[-1] if sm else None),
                mem_median=mm_med, mem_min=(mm[0] if mm else None), mem_max=(mm[-1] if mm else None),
                power_median=med(pw), temp_median=med(tc),
                min_sm_required=min_sm, min_mem_required=min_mem,
                reason=("ok" if not reasons else "ENV_INVALID: " + "; ".join(reasons)),
                threshold_note="阈值是**本机 RTX 4060 Laptop 经验值**，非通用 GPU 标准，可配置")


def pstate_audit(runs: list[dict], *, min_sm: float = MIN_SM, min_mem: float = MIN_MEM) -> dict:
    """从某条件的 measured reps 的 `monitor_start/end` 快照汇总该 cell 的 P-state 资格。

    这些快照都在**计时区间之外**、且 GPU 处于负载状态（start 紧接上次 rep/warm-up，end 紧接
    本次 rep），因此适合判 active P-state，且**不给正式计时增加任何开销**。
    """
    samples = []
    for r in runs:
        for when in ("monitor_start", "monitor_end"):
            s = r.get(when) or {}
            if s.get("available"):
                samples.append(s)
    v = pstate_verdict(samples, min_sm=min_sm, min_mem=min_mem)
    v["cell_samples"] = len(samples)
    return v


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
