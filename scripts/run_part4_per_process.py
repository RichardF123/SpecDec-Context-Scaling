"""run_part4_per_process.py — 逐条件独立进程跑 G4 Part 4（8K 档的必需测法）。

为什么需要它：在 8 GB 卡上跑 8K 双模型时，`peak_resv` 会达到 8.05–10.47 GiB，**越过物理显存**，
分配器池状态（谁先申请、池多大）会显著改变结果。实测同一格 base_bf16@8192 AR：
带 `empty_cache` 的旧 harness 给 10.87 s，去掉后给 6.44 s；而 base 比 v_fp8 慢这一点在
两种顺序下都成立（方向可信、量级不可信）。⇒ **每个条件开一个新进程**，让每格从干净、
确定的显存状态开始，格与格之间才可比。

做法：对 `ctx × KV × decode` 的每个条件，用 `P4_ONLY` 起一个子进程；
所有子进程共享同一 config hash（hash 只含"预期矩阵"，不含单条件选择），
因此结果**追加**进同一个 JSONL；最后用 `--aggregate` 从该 JSONL 重算四张表（不重复跑模型）。

用法（交互式 GPU 终端，串行；预计 12 条件 × (加载约 30–40 s + 单格耗时)）：
  python scripts/run_part4_per_process.py
可选环境变量（与 harness 一致）：P4_CTX / P4_REPS / P4_MAXNEW / P4_WARM / P4_CHUNK / P4_TEXT
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTXS = [int(x) for x in os.environ.get("P4_CTX", "512,2048,8192").split(",") if x.strip()]
KVS = ("base_bf16", "v_fp8")
MODES = ("AR", "SpecDec")


def main() -> int:
    conds = [(c, kv, m) for c in CTXS for kv in KVS for m in MODES]
    run_id = os.environ.get("P4_RUN_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"[driver] {len(conds)} 个条件（逐进程）：{conds}", flush=True)
    print(f"[driver] run_id={run_id}（输出文件名含 config hash + run_id，**不覆盖既有文件**）",
          flush=True)
    # 注意：**不做任何归档/移动**。历史 run（含两个受保护的 exploratory JSONL）必须原地保留；
    # RUN_ID 已保证新 run 不会与旧文件同名。
    outdir = ROOT / "output" / "part4"
    outdir.mkdir(parents=True, exist_ok=True)
    env0 = os.environ.copy()
    env0.pop("P4_ONLY", None)
    env0["P4_RUN_ID"] = run_id
    t_all = time.perf_counter()
    for i, (ctx, kv, mode) in enumerate(conds, 1):
        env = dict(env0)
        env["P4_ONLY"] = f"{ctx},{kv},{mode}"
        print(f"\n[driver] ({i}/{len(conds)}) P4_ONLY={env['P4_ONLY']}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", "specdec.engine.g4_part4_pilot_measure"],
                           env=env, cwd=str(ROOT))
        print(f"[driver] ({i}/{len(conds)}) exit={r.returncode} "
              f"耗时 {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"\n[driver] 全部条件完成，总耗时 {(time.perf_counter()-t_all)/60:.1f} 分钟；聚合报表：",
          flush=True)
    subprocess.run([sys.executable, "-m", "specdec.engine.g4_part4_pilot_measure",
                    "--aggregate"], env=env0, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
