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
* GPU 身份/驱动用 `nvidia-smi`（**不建 CUDA context**），torch/cuDNN 版本用 `torch.version`。
* 工作树必须 clean（`git status --porcelain` 为空）；不 clean 一律不跑。

## 判定阶梯（写死，不事后调整）
| 条件 | 判决 |
| --- | --- |
| free commit < 16 GiB | `ENVIRONMENT_BLOCKED` |
| 16 ≤ free commit < 18 GiB | `ENVIRONMENT_MARGIN_TOO_LOW` |
| P-state 不合格 / 有 GPU 计算残留在跑 / 读取失败 | `ENVIRONMENT_BLOCKED` |
| 工作树不 clean | `CHECK_NEEDS_REVIEW`（前置条件，非环境） |
| 以上全部通过 | `ENVIRONMENT_OK`（允许执行 Stage 1） |

退出码：`ENVIRONMENT_OK` ⇒ 0，其余 ⇒ 1（**任何情况下都不加载模型**）。

## 历史锚点**运行前**登记
`PRE_REGISTERED_ANCHORS` 在本文件里写死并写入产物 ⇒ 时间戳早于任何运行，
**运行后不得修改**。
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

ENTRY_MIN_COMMIT_GIB = 18.0          # ★ 本轮入口门槛（比 harness 的 16 GiB 更严）
FAMILY = "F13"
# ★ **运行前登记**的历史锚点（不得运行后修改）
PRE_REGISTERED_ANCHORS = dict(
    family=FAMILY,
    ar=dict(expected_gen_hash="6e7522bf2521ad7b", expected_len=64, expected_stop="max_new_reached"),
    prod=dict(expected_gen_hash="e90dad2b3fcf1c65", expected_len=64, expected_stop="max_new_reached",
              expected_first_divergence_vs_ar=47),
    purpose="证明新 runner 未改变 AR / production 路径（否则说明本轮实现动了不该动的东西）",
    rule="若 AR 或 production 无法复现锚点 ⇒ BASELINE_REGRESSION，保存结果、停止、不解释 serial oracle")


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


def main() -> int:
    from .. import results_schema as S
    from . import g5_8k_spec_variance_probe as P5V

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        tool="serial_oracle_env_check", run_id=run_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose=("Stage 1（F13）开跑前的**一次性**环境检查：不加载模型；"
                 "门槛比 harness 更严（commit ≥ 18 GiB），harness 的 16 GiB 安全线保持不变"),
        model_loaded=False, gpu_compute_performed=False,
        entry_thresholds=dict(entry_required_gib=ENTRY_MIN_COMMIT_GIB,
                              harness_line_gib=float(P5V.MIN_COMMIT_GIB),
                              allow_unqualified_env_set=bool(
                                  __import__("os").environ.get("SMOKE_ALLOW_UNQUALIFIED")),
                              note="两把尺子分别记录；不设置 SMOKE_ALLOW_UNQUALIFIED、不降低任何安全线"),
        pre_registered_anchors=PRE_REGISTERED_ANCHORS,
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda, cudnn=(torch.backends.cudnn.version()
                                                      if torch.backends.cudnn.is_available() else None),
                      transformers=__import__("transformers").__version__,
                      platform=platform.platform()),
        git=dict(S.git_meta(ROOT), **_git_clean()),
    )
    rec["gpu"] = dict(_smi("gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"),
                      compute_apps=_smi("compute-apps=pid,process_name,used_memory"))
    # ---- harness 自己的门（16 GiB）与系统内存明细 ----
    sm = P5V.system_memory()
    ok16, why16 = P5V.commit_guard()
    rec["harness_gate"] = dict(ok=bool(ok16), why=why16, system=sm)
    free_gib = float(sm.get("commit_free_mib") or 0) / 1024.0
    rec["entry_commit"] = dict(free_gib=round(free_gib, 3),
                               required_gib=ENTRY_MIN_COMMIT_GIB,
                               margin_gib=round(free_gib - ENTRY_MIN_COMMIT_GIB, 3),
                               ok=bool(free_gib >= ENTRY_MIN_COMMIT_GIB))
    # ---- P-state（复用 harness 的负载式预检）----
    pf = P5V.PF.qualify()
    rec["pstate"] = pf
    # ---- GPU 消费者/残留 ----
    gc = P5V.gpu_consumers()
    rec["gpu_consumers"] = gc
    apps_raw = (rec["gpu"] or {}).get("compute_apps", {}).get("out") or ""
    apps = [a.strip() for a in apps_raw.splitlines() if a.strip()]
    rec["gpu_residue"] = dict(n_compute_apps=len(apps), apps=apps,
                              clean=bool(len(apps) == 0),
                              note="compute-apps 为空 = 没有别的进程在 GPU 上做计算（本进程未建 context）")
    # ---- 判定阶梯（写死）----
    if free_gib < 16.0:
        decision, blocked = "ENVIRONMENT_BLOCKED", True
    elif free_gib < ENTRY_MIN_COMMIT_GIB:
        decision, blocked = "ENVIRONMENT_MARGIN_TOO_LOW", True
    elif pf.get("qualified") is not True:
        decision, blocked = "ENVIRONMENT_BLOCKED", True
    elif not rec["gpu_residue"]["clean"]:
        decision, blocked = "ENVIRONMENT_BLOCKED", True
    elif not rec["git"]["clean"]:
        decision, blocked = "CHECK_NEEDS_REVIEW", True
    else:
        decision, blocked = "ENVIRONMENT_OK", False
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
    print(f"  driver/残留: compute-apps={rec['gpu_residue']['n_compute_apps']} "
          f"clean={rec['gpu_residue']['clean']}")
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
