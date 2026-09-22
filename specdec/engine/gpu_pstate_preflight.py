"""gpu_pstate_preflight.py — GPU **active-load P-state 资格预检**（独立、可复用、非正式计时）。

为什么需要（2026-09-21 实测教训）
  同日 `g5_stability_probe` 的 12 primary cell 中 7 个 between-block 差异 >20%，两个 cell 的
  within-run 相对离散 >220%。原始 JSONL 显示性能分成两档：
    高性能档：SM 2505–2535 MHz / mem 8000 MHz / 功耗 51–91 W → AR 约 23–27 tok/s
    低功耗档：SM  480–1290 MHz / mem 6000 MHz / 功耗 16–28 W  → AR 约  7.9–8.8 tok/s
  慢运行温度 60–68 °C、快运行 79–81 °C ⇒ **不是热降频**，而是 GPU 进入了低功耗 P-state。
  ⇒ 正式计时前必须先确认"负载状态下"的时钟/功耗档位合格。

原理（关键：必须在**负载期间**采样）
  空闲时 SM≈210 MHz / mem≈405 MHz 是正常节能状态，**不能**用它判资格。本工具先起一个
  短、固定、低显存的 CUDA 负载（2048² fp16 矩阵乘反复执行，约 8 MiB），在负载进行中用
  后台线程周期性调用 `nvidia-smi` 采样，然后只对**负载样本**做判定。

阈值（**可配置**，默认是**本机**经验值，不是通用 GPU 标准）
  P5_MIN_ACTIVE_SM_CLOCK_MHZ=2000     P5_MIN_ACTIVE_MEM_CLOCK_MHZ=7500
  P5_PSTATE_PREFLIGHT_SECONDS=5       P5_PSTATE_SAMPLE_INTERVAL_S=0.5
  功耗只作辅助证据，不作硬 Gate。

输出
  屏幕报表 + `output/g5/pstate_preflight_<UTC>.json`（唯一命名，不覆盖），包含：
  samples / median·min·max（SM·mem 时钟、功耗、温度）/ duration / 阈值 / verdict /
  空闲样本（仅记录，**不参与判定**）。
  采样不可用 ⇒ `available=false`（**不填 0 冒充**）。

退出码：0 = 合格；3 = 不合格（ENV_INVALID）；2 = 采样不可用/异常。
用法：python -m specdec.engine.gpu_pstate_preflight
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import g5_driver as D

SECONDS = float(os.environ.get("P5_PSTATE_PREFLIGHT_SECONDS", "5"))
INTERVAL = float(os.environ.get("P5_PSTATE_SAMPLE_INTERVAL_S", "0.5"))
MIN_SM = float(os.environ.get("P5_MIN_ACTIVE_SM_CLOCK_MHZ", str(D.MIN_SM)))
MIN_MEM = float(os.environ.get("P5_MIN_ACTIVE_MEM_CLOCK_MHZ", str(D.MIN_MEM)))
MAT = int(os.environ.get("P5_PSTATE_MATRIX", "2048"))
OUTDIR = Path(__file__).resolve().parents[2] / "output" / "g5"


class Sampler(threading.Thread):
    """后台采样线程：只调 nvidia-smi 快照，**不碰** CUDA，因此不会给计时加开销。"""

    def __init__(self, interval: float):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self.failures = 0
        # 注意：**不能**叫 `self._stop`——那会覆盖 threading.Thread._stop 方法，
        # 使 Thread.join() 内部调用 self._stop() 时抛 TypeError: 'Event' object is not callable。
        self._stop_flag = threading.Event()

    def run(self) -> None:
        while not self._stop_flag.is_set():
            s = D.system_snapshot()
            if s.get("available"):
                self.samples.append(s)
            else:
                self.failures += 1
            self._stop_flag.wait(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()
        self.join(timeout=5)


def run_load(seconds: float, mat: int) -> dict:
    """短、固定、可重复、低显存的 CUDA 负载；返回执行统计（不用于任何性能结论）。"""
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA 不可用"}
    x = torch.randn(mat, mat, device="cuda", dtype=torch.float16)   # ~8 MiB
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.mm(x, x, out=y)          # 反复执行，队列保持满载
        n += 1
    torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    del x, y
    torch.cuda.empty_cache()
    return {"available": True, "iterations": n, "duration_s": dur,
            "matmul_per_s": n / dur if dur else None, "matrix": mat,
            "dtype": "float16"}


def verdict_from_sampler(load_samples: list[dict], idle_sample: dict | None = None, *,
                         min_sm: float = MIN_SM, min_mem: float = MIN_MEM) -> dict:
    """**契约（纯函数，CPU 可测）**：判定只用 `load_samples`；`idle_sample` 仅随结果记录。

    空闲时 SM≈210MHz 是正常节能状态；若把它当判定输入，会把合格机器误判为不合格，
    反之也不能用"空闲时钟高"来证明负载期合格。
    """
    v = D.pstate_verdict(load_samples, min_sm=min_sm, min_mem=min_mem)
    v["idle_sample"] = idle_sample
    v["verdict_input"] = "load_samples_only"
    return v


def qualify(seconds: float = SECONDS, interval: float = INTERVAL, mat: int = MAT,
            min_sm: float = MIN_SM, min_mem: float = MIN_MEM) -> dict:
    """可复用入口：起短负载 + 后台采样 + 判定。返回 verdict（含 idle_sample / load 信息）。"""
    idle = D.system_snapshot()
    sampler = Sampler(interval)
    sampler.start()
    time.sleep(min(1.0, interval))
    load = run_load(seconds, mat)
    sampler.stop()
    if not load.get("available"):
        v = verdict_from_sampler([], idle, min_sm=min_sm, min_mem=min_mem)
        v.update(qualified=False, reason=f"负载不可用: {load.get('error')}", load=load,
                 samples_used=0, sample_failures=sampler.failures)
        return v
    v = verdict_from_sampler(sampler.samples, idle, min_sm=min_sm, min_mem=min_mem)
    v.update(load=load, samples_used=len(sampler.samples),
             sample_failures=sampler.failures)
    return v


def main() -> int:
    print(f"[pstate] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[pstate] 阈值(本机经验值, 可配置): SM>={MIN_SM:.0f}MHz mem>={MIN_MEM:.0f}MHz "
          f"| 负载时长={SECONDS}s 采样间隔={INTERVAL}s", flush=True)
    v = qualify()
    idle, load = v.get("idle_sample") or {}, v.get("load") or {}
    print(f"[pstate] 空闲快照（**仅记录，不参与判定**）：SM={idle.get('clock_sm_MHz')}MHz "
          f"mem={idle.get('clock_mem_MHz')}MHz power={idle.get('power_W')}W "
          f"available={idle.get('available')}", flush=True)
    if not load.get("available"):
        print(f"[pstate] 负载不可用：{load.get('error')} -> available=false", flush=True)
        return 2
    print(f"\n[pstate] 负载期间样本 n={v.get('samples_used')}"
          f"（采样失败 {v.get('sample_failures')} 次）；"
          f"负载 {load['iterations']} 次 matmul / {load['duration_s']:.2f}s", flush=True)
    if v.get("available"):
        print(f"[pstate] SM clock  median={v['sm_median']:.0f} "
              f"min={v['sm_min']:.0f} max={v['sm_max']:.0f} MHz", flush=True)
        print(f"[pstate] mem clock median={v['mem_median']:.0f} "
              f"min={v['mem_min']:.0f} max={v['mem_max']:.0f} MHz", flush=True)
        print(f"[pstate] power median={v['power_median']} W | temp median={v['temp_median']} °C",
              flush=True)
    else:
        print(f"[pstate] available=false：{v['reason']}（**不得填 0 冒充**）", flush=True)

    out = dict(schema_version=S.SCHEMA_VERSION, kind="pstate_preflight",
               run_id=S.default_run_id("pstate"),
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(Path(__file__).resolve().parents[2]),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               idle_sample=idle, verdict=v, load=load,
               thresholds=dict(min_sm_mhz=MIN_SM, min_mem_mhz=MIN_MEM,
                               seconds=SECONDS, interval_s=INTERVAL),
               note=("只分析**负载期间**样本；空闲低时钟是正常节能状态，不参与判定；"
                     "阈值非通用 GPU 标准"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"pstate_preflight_{out['run_id']}.json"
    if path.exists():
        raise SystemExit(f"[拒绝覆盖] {path} 已存在")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[pstate] verdict = "
          f"{'QUALIFIED（可进入正式计时）' if v.get('qualified') else 'ENV_INVALID（不得进入正式计时）'}"
          f"  reason={v.get('reason')}", flush=True)
    print(f"[pstate] 结果已写入 {path}", flush=True)
    if not v.get("available"):
        return 2
    return 0 if v.get("qualified") else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
