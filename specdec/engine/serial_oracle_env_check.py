"""serial_oracle_env_check.py — 一次性**开跑前环境检查**（不加载任何模型、不做任何生成）。

## 为什么单独一个模块
本轮任务要求"先只运行一次环境检查"，并且**门槛比 harness 更严**：
* harness 既有安全线：`P5V.MIN_COMMIT_GIB = 16 GiB`（**不动、不降、不覆盖**）；
* 本轮入口门槛：**available commit ≥ 18 GiB**（额外的、更严的先决条件）。

两把尺子分别**如实记录**：`harness_line_gib=16`（由 `P5V.commit_guard()` 实测）与
`entry_required_gib=18`（本模块实测 free commit 后自行判定）。

## 口径与取值来源
* commit/P-state/GPU 消费者**全部复用 harness 自己的代码**
  （`P5V.system_memory()` / `P5V.commit_guard()` / `P5V.PF.qualify()` / `P5V.gpu_consumers()`），
  **不重新实现**，避免"我的读数和 harness 的读数不是一回事"。
* GPU 身份/驱动/显存用 `nvidia-smi`（**不建 CUDA context**），torch/cuDNN 版本用 `torch.version`。
* 工作树必须 clean（`git status --porcelain` 为空）；不 clean 一律不跑。

## `gpu_residue` 判据的修正（2026-10-03，本轮任务明确要求）
**旧判据**：`clean = (GPU 进程数 == 0)` —— 在本机**永远为 false**，因为正常桌面就有约 21 个
持有 GPU 上下文的程序（历史成功的 24 轮协议与 F13 基线运行时同样如此）。
**新判据**（本文件 `classify_gpu_processes`）：`clean = 不存在**本项目的** Python/实验残留进程`，
即 GPU 进程里没有路径落在本仓库（`RealDSH\\specdec` / `.venv`）之下的进程。
同时**如实记录全部** GPU 进程、非本项目的 python、权限不可见者，以及**可用显存**。

* 旧的环境检查产物**保留不删**（新运行写新的时间戳文件，从不覆盖）。
* `16 GiB` harness 安全线、`18 GiB` 入口门槛**均未改动**。
* 不因"想通过检查"而结束任何用户进程（例如 `FeverGamesService` 若在跑就如实记录）。

## 判定阶梯（写死，不事后调整）
| 条件 | 判决 |
| --- | --- |
| free commit < 16 GiB | `ENVIRONMENT_BLOCKED` |
| 16 ≤ free commit < 18 GiB | `ENVIRONMENT_MARGIN_TOO_LOW` |
| harness `commit_guard` 判失败 / P-state 不合格 / **存在本项目残留进程** | `ENVIRONMENT_BLOCKED` |
| 工作树不 clean | `CHECK_NEEDS_REVIEW`（前置条件，非环境） |
| 以上全部通过 | `ENVIRONMENT_OK`（允许执行 Stage 1） |

退出码：`ENVIRONMENT_OK` ⇒ 0，其余 ⇒ 1（**任何情况下都不加载模型**）。

## 历史锚点**运行前**登记
`PRE_REGISTERED_ANCHORS` 在本文件里写死并写入产物 ⇒ 时间戳早于任何运行，
**运行后不得修改**。
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

ENTRY_MIN_COMMIT_GIB = 18.0          # ★ 本轮入口门槛（比 harness 的 16 GiB 更严；**未改动**）
HARNESS_MIN_COMMIT_GIB = 16.0        # harness 安全线（只用于阶梯判定；**未改动、未覆盖**）
FAMILY = "F13"

# 判定"本项目残留进程"的路径标记（大小写不敏感子串匹配）
PROJECT_MARKERS = ("realdsh\\specdec", "realdsh/specdec", "specdec\\.venv", "specdec/.venv",
                   "\\.venv\\scripts\\python", "/.venv/bin/python")
# 只在**报告**里留意的进程（不影响判定；例如用户提到的 FeverGamesService）
WATCHED_PROCESSES = ("FeverGamesService", "FeverGames", "FeverGamesLauncher")

GPU_QUERY_FIELDS = ("name", "driver_version", "memory.total", "memory.used", "memory.free",
                    "utilization.gpu", "temperature.gpu")

# ★ **运行前登记**的历史锚点（不得运行后修改）
PRE_REGISTERED_ANCHORS = dict(
    family=FAMILY,
    ar=dict(expected_gen_hash="6e7522bf2521ad7b", expected_len=64, expected_stop="max_new_reached"),
    prod=dict(expected_gen_hash="e90dad2b3fcf1c65", expected_len=64, expected_stop="max_new_reached",
              expected_first_divergence_vs_ar=47),
    purpose="证明新 runner 未改变 AR / production 路径（否则说明本轮实现动了不该动的东西）",
    rule="若 AR 或 production 无法复现锚点 ⇒ BASELINE_REGRESSION，保存结果、停止、不解释 serial oracle")


# --------------------------------------------------------------------------- 纯函数（可 CPU 自测）
def parse_compute_rows(out: str) -> list:
    """把 `nvidia-smi --query-compute-apps=pid,process_name,used_memory` 的 CSV 解析成行。"""
    rows = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append(dict(pid=parts[0] if parts else None,
                         name=",".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else ""),
                         used_memory=(parts[-1] if len(parts) > 2 else None), raw=line))
    return rows


def parse_gpu_fields(out: str, fields: tuple = GPU_QUERY_FIELDS) -> dict:
    """解析 `--query-gpu=...` 的单行 CSV（字段数不足时对应键为 None）。"""
    line = ""
    for ln in (out or "").splitlines():
        if ln.strip():
            line = ln.strip()
    vals = [v.strip() for v in line.split(",")] if line else []
    d = {}
    for i, f in enumerate(fields):
        d[f.replace(".", "_")] = (vals[i] if i < len(vals) else None)
    d["raw"] = line
    return d


def _is_python(name: str) -> bool:
    n = str(name or "").lower()
    return ("python" in n) or n.endswith("py.exe")


def classify_gpu_processes(rows: list, *, project_markers: tuple = PROJECT_MARKERS) -> dict:
    """把 GPU 进程分成：**本项目残留** / 非本项目 python / 权限不可见 / 其它（桌面）。

    `clean` 的定义（新判据）：**不存在本项目残留进程**。
    * 路径落在本仓库之下（`RealDSH\\specdec` / `.venv`）的任何 GPU 进程 ⇒ 残留（含非 python，
      例如某个跑实验的 exe）；
    * python 但路径明显在别处 ⇒ 记为 `non_project_python`（**不阻断**，如实记录）；
    * `[Insufficient Permissions]` ⇒ 无法取路径，记为 `unknown_path`（**盲区，如实记录**）。
    """
    residual, non_project_python, unknown_path, desktop = [], [], [], []
    for r in rows:
        name = str(r.get("name") or "")
        low = name.lower()
        in_project = any(m.lower() in low for m in project_markers)
        if ("insufficient permissions" in low) or ("[n/a]" == low.strip()):
            unknown_path.append(r)
            continue
        if in_project:
            residual.append(r)
            continue
        if _is_python(low):
            non_project_python.append(r)
            continue
        desktop.append(r)
    return dict(criterion="clean = 不存在**本项目**（RealDSH\\specdec / .venv）的 GPU 进程残留",
                previous_criterion="clean = GPU 进程数 == 0（在本机永远为 false，已废弃）",
                n_total=len(rows),
                residual_project_processes=residual, n_residual=len(residual),
                non_project_python=non_project_python, n_non_project_python=len(non_project_python),
                unknown_path=unknown_path, n_unknown_path=len(unknown_path),
                desktop_other=desktop, n_desktop_other=len(desktop),
                clean=bool(not residual),
                blind_spot_note=("权限不可见的行无法判定路径 ⇒ 计入 unknown_path 并如实报出；"
                                 "本机常见 1 行 `[Insufficient Permissions]`"))


def decide(*, free_gib: float, harness_ok: bool, pstate_qualified,
           residue_clean: bool, git_clean: bool,
           entry_min: float = ENTRY_MIN_COMMIT_GIB, hard_min: float = HARNESS_MIN_COMMIT_GIB) -> tuple:
    """判定阶梯（**纯函数**，写死；供自测直接调用）。返回 `(decision, blocked)`。"""
    if free_gib < hard_min:
        return "ENVIRONMENT_BLOCKED", True
    if free_gib < entry_min:
        return "ENVIRONMENT_MARGIN_TOO_LOW", True
    if not harness_ok:
        return "ENVIRONMENT_BLOCKED", True
    if pstate_qualified is not True:
        return "ENVIRONMENT_BLOCKED", True
    if not residue_clean:
        return "ENVIRONMENT_BLOCKED", True
    if not git_clean:
        return "CHECK_NEEDS_REVIEW", True
    return "ENVIRONMENT_OK", False


# --------------------------------------------------------------------------- 采集
def _smi(query: str) -> dict:
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
                           capture_output=True, text=True, encoding="utf-8", timeout=30)
        return dict(ok=bool(r.returncode == 0), out=(r.stdout or "").strip(),
                    err=(r.stderr or "").strip()[:200], rc=int(r.returncode))
    except Exception as e:                                                # noqa: BLE001
        return dict(ok=False, out="", err=f"{type(e).__name__}: {e}", rc=None)


def _git_clean() -> dict:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8", timeout=60)
        lines = [x for x in (r.stdout or "").splitlines() if x.strip()]
        return dict(ok=bool(r.returncode == 0), rc=int(r.returncode),
                    porcelain=[x.strip() for x in lines], clean=bool(not lines),
                    head=None if not lines else lines[0])
    except Exception as e:                                                # noqa: BLE001
        return dict(ok=False, rc=None, porcelain=[], clean=False, error=f"{type(e).__name__}: {e}")


def _watched() -> dict:
    """只记录（不干预）若干被点名的进程是否在跑，例如 `FeverGamesService`。"""
    out = {}
    for nm in WATCHED_PROCESSES:
        try:
            r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {nm}", "/NH"],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
            txt = (r.stdout or "").strip()
            running = bool(txt) and ("No tasks" not in txt) and (nm.lower() in txt.lower())
            out[nm] = dict(running=running, raw=txt.splitlines()[:3])
        except Exception as e:                                            # noqa: BLE001
            out[nm] = dict(running=None, error=f"{type(e).__name__}: {e}")
    return out


def main() -> int:
    from .. import results_schema as S
    from . import g5_8k_spec_variance_probe as P5V

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        tool="serial_oracle_env_check", run_id=run_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose=("Stage 1（F13）开跑前的**一次性**环境检查：不加载模型；"
                 "门槛比 harness 更严（commit ≥ 18 GiB），harness 的 16 GiB 安全线保持不变"),
        model_loaded=False,
        entry_thresholds=dict(entry_required_gib=ENTRY_MIN_COMMIT_GIB,
                              harness_line_gib=float(P5V.MIN_COMMIT_GIB),
                              allow_unqualified_env_set=bool(os.environ.get("SMOKE_ALLOW_UNQUALIFIED")),
                              note=("两把尺子分别记录；不设置 SMOKE_ALLOW_UNQUALIFIED、"
                                    "不降低任何安全线；两个数字本轮均**未改动**")),
        pre_registered_anchors=PRE_REGISTERED_ANCHORS,
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda, cudnn=(torch.backends.cudnn.version()
                                                      if torch.backends.cudnn.is_available() else None),
                      transformers=__import__("transformers").__version__,
                      platform=platform.platform()),
        git=dict(S.git_meta(ROOT), **_git_clean()),
        residue_criterion=dict(
            criterion="clean = 不存在**本项目**（RealDSH\\specdec / .venv）的 GPU 进程残留",
            previous_criterion="clean = GPU 进程数 == 0（在本机永远为 false）",
            changed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            why="本机正常桌面就有约 21 个持有 GPU 上下文的程序（历史成功运行同样如此）",
            not_changed="16 GiB harness 安全线 / 18 GiB 入口门槛 / commit 判定顺序",
            self_test="specdec/engine/serial_oracle_env_check_selftest_cpu.py"),
    )
    # ⚠️ 顺序很重要：**先在未建 CUDA context 时**采集 GPU 进程（否则会把自己算进去）
    rec["gpu"] = dict(_smi("gpu=name,driver_version,memory.total,memory.used,memory.free,"
                           "utilization.gpu,temperature.gpu"),
                      compute_apps=_smi("compute-apps=pid,process_name,used_memory"),
                      capture_order_note=("compute-apps 在 `P5V.PF.qualify()` **之前**采集；"
                                          "qualify 会创建本进程的 CUDA context ⇒ 之后的列表会包含自己"))
    rec["gpu_fields"] = parse_gpu_fields(rec["gpu"]["out"])
    rows = parse_compute_rows(rec["gpu"]["compute_apps"]["out"])
    rec["gpu_processes"] = rows
    rec["gpu_residue"] = classify_gpu_processes(rows)
    rec["watched_processes"] = _watched()
    # ---- harness 自己的门（16 GiB）与系统内存明细 ----
    sm = P5V.system_memory()
    ok16, why16 = P5V.commit_guard()
    rec["harness_gate"] = dict(ok=bool(ok16), why=why16, system=sm)
    free_gib = float(sm.get("commit_free_mib") or 0) / 1024.0
    rec["entry_commit"] = dict(free_gib=round(free_gib, 3),
                               required_gib=ENTRY_MIN_COMMIT_GIB,
                               margin_gib=round(free_gib - ENTRY_MIN_COMMIT_GIB, 3),
                               ok=bool(free_gib >= ENTRY_MIN_COMMIT_GIB))
    rec["vram"] = dict(total_mib=rec["gpu_fields"].get("memory_total"),
                       used_mib=rec["gpu_fields"].get("memory_used"),
                       free_mib=rec["gpu_fields"].get("memory_free"),
                       utilization_pct=rec["gpu_fields"].get("utilization_gpu"),
                       temperature_c=rec["gpu_fields"].get("temperature_gpu"),
                       note="只记录，**不作为门槛**（本轮不新增未经登记的门槛）")
    # ---- P-state（复用 harness 的负载式预检）----
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    rec["gpu_consumers"] = P5V.gpu_consumers()
    # ---- 判定阶梯（纯函数）----
    decision, blocked = decide(free_gib=free_gib, harness_ok=bool(ok16),
                               pstate_qualified=pf.get("qualified"),
                               residue_clean=bool(rec["gpu_residue"]["clean"]),
                               git_clean=bool(rec["git"]["clean"]))
    rec["decision"] = decision
    rec["blocked"] = bool(blocked)
    rec["stage1_allowed"] = bool(decision == "ENVIRONMENT_OK")
    p = OUTDIR / f"serial_oracle_env_check_{run_id}.json"
    rec["artifact_json"] = str(p)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("== Stage 1 开跑前环境检查（不加载模型）==")
    print(f"  git: commit={rec['git'].get('git_commit')} dirty={rec['git'].get('dirty_worktree')} "
          f"clean={rec['git']['clean']}")
    print(f"  版本: torch={rec['versions']['torch']} cuda={rec['versions']['cuda']} "
          f"cudnn={rec['versions']['cudnn']} transformers={rec['versions']['transformers']}")
    print(f"  GPU: {rec['gpu'].get('out')}")
    print(f"  显存: total={rec['vram']['total_mib']} used={rec['vram']['used_mib']} "
          f"free={rec['vram']['free_mib']} MiB（不作门槛）")
    gr = rec["gpu_residue"]
    print(f"  GPU 进程: total={gr['n_total']} 本项目残留={gr['n_residual']} "
          f"非本项目 python={gr['n_non_project_python']} 权限不可见={gr['n_unknown_path']} "
          f"桌面其它={gr['n_desktop_other']} clean={gr['clean']}")
    for r in gr["residual_project_processes"]:
        print(f"     [残留] {r['raw']}")
    for r in gr["non_project_python"]:
        print(f"     [非本项目 python] {r['raw']}")
    print(f"  点名进程（只记录不干预）: "
          + ", ".join(f"{k}={'running' if v.get('running') else v.get('running')}"
                      for k, v in rec["watched_processes"].items()))
    print(f"  harness 门(16 GiB): {why16}")
    print(f"  入口门槛: free={free_gib:.2f} GiB  需要≥{ENTRY_MIN_COMMIT_GIB:.0f} GiB  "
          f"margin={free_gib - ENTRY_MIN_COMMIT_GIB:+.2f} GiB")
    print(f"  系统内存: commit_limit={sm.get('commit_limit_mib', 0)/1024:.2f} GiB "
          f"used={sm.get('commit_used_mib', 0)/1024:.2f} GiB  "
          f"phys_free={sm.get('phys_free_mib', 0)/1024:.2f} GiB")
    print(f"  P-state: qualified={pf.get('qualified')} sm_median={pf.get('sm_median')} "
          f"mem_median={pf.get('mem_median')} reason={pf.get('reason')}")
    print(f"  预登记锚点: AR={PRE_REGISTERED_ANCHORS['ar']['expected_gen_hash']} "
          f"prod={PRE_REGISTERED_ANCHORS['prod']['expected_gen_hash']} "
          f"首分歧={PRE_REGISTERED_ANCHORS['prod']['expected_first_divergence_vs_ar']}")
    print(f"  ★ 判决 = {decision}（stage1_allowed={rec['stage1_allowed']}）")
    print(f"[env] RESULT_JSON={p}")
    return 0 if rec["stage1_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
