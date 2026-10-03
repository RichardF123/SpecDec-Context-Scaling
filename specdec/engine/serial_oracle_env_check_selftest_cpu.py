"""serial_oracle_env_check_selftest_cpu.py — **环境检查器**的针对性自测（纯 CPU、不加载模型、不占 GPU）。

## 为什么必须自测
`gpu_residue.clean` 是放行/阻断 Stage 1 的判据之一。旧判据写成 `GPU 进程数 == 0`，
在本机**永远为 false**（正常桌面就有约 21 个持有 GPU 上下文的程序）⇒ 判据实际失效。
本轮把判据改为"**不存在本项目的 Python/实验残留进程**"。判据一改就必须能被检验：

1. `classify_gpu_processes` 的**分类逻辑**逐类验证（本项目残留 / 非本项目 python / 权限不可见 / 桌面）；
2. **旧判据 vs 新判据**在**真实记录**上的对照（用旧环境检查产物里那 21 行真实进程）；
3. `decide()` 判定阶梯的每一条分支（含"残留 ⇒ 阻断"、"树不 clean ⇒ CHECK_NEEDS_REVIEW"）；
4. **常量不被悄悄改动**：`16 GiB` harness 线、`18 GiB` 入口门槛、以及**预登记锚点**的取值；
5. 采集顺序说明存在（compute-apps 必须在 `P5V.PF.qualify()` 之前采集，否则会把自己算进去）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import serial_oracle_env_check as EC

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

DESKTOP_ROWS = [
    "11652, C:\\Windows\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\\CrossDeviceResume.exe, [N/A]",
    "10868, D:\\ToDesk\\ToDesk.exe, [N/A]",
    "13544, C:\\Program Files\\NetEase\\CloudMusic\\cloudmusic.exe, [N/A]",
    "2932, D:\\RealDSH\\DSH Desktop\\DSH Desktop.exe, [N/A]",
    "36124, C:\\Program Files (x86)\\Microsoft\\EdgeWebView\\Application\\153.0.4234.48\\msedgewebview2.exe, [N/A]",
    "29596, C:\\Users\\ASUS\\AppData\\Local\\Doubao\\Application\\app\\Doubao.exe, [N/A]",
    "39252, C:\\Program Files\\WindowsApps\\microsoft.windowscommunicationsapps\\HxOutlook.exe, [N/A]",
]


def _ck(name, cond, extra=None):
    return (name, bool(cond), extra)


def main() -> int:
    checks: list = []

    # ==== 1) CSV 解析 ====
    rows = EC.parse_compute_rows("\n".join(DESKTOP_ROWS))
    checks += [_ck("p01_parse_compute_rows", len(rows) == len(DESKTOP_ROWS)
                   and rows[1]["pid"] == "10868" and "ToDesk" in rows[1]["name"], rows[:2]),
               _ck("p02_parse_compute_rows_keeps_comma_in_name",
                   EC.parse_compute_rows("1, C:\\a,b\\x.exe, 10 MiB")[0]["name"] == "C:\\a,b\\x.exe",
                   EC.parse_compute_rows("1, C:\\a,b\\x.exe, 10 MiB")),
               _ck("p03_parse_gpu_fields",
                   (lambda d: d["memory_total"] == "8188 MiB" and d["memory_free"] == "7000 MiB"
                    and d["utilization_gpu"] == "0 %")(
                       EC.parse_gpu_fields("NVIDIA GeForce RTX 4060 Laptop GPU, 555.97, 8188 MiB, 1188 MiB, "
                                           "7000 MiB, 0 %, 61")),
                   EC.parse_gpu_fields("NVIDIA GeForce RTX 4060 Laptop GPU, 555.97, 8188 MiB, 1188 MiB, "
                                       "7000 MiB, 0 %, 61"))]

    # ==== 2) 分类逻辑 ====
    c1 = EC.classify_gpu_processes([])
    checks += [_ck("c01_empty_is_clean", c1["clean"] is True and c1["n_total"] == 0, c1)]
    c2 = EC.classify_gpu_processes(EC.parse_compute_rows("\n".join(DESKTOP_ROWS)))
    checks += [_ck("c02_desktop_only_is_clean_and_all_recorded",
                   c2["clean"] is True and c2["n_desktop_other"] == len(DESKTOP_ROWS)
                   and c2["n_residual"] == 0, c2)]
    proj_py = EC.parse_compute_rows(
        "4242, D:\\RealDSH\\specdec\\.venv\\Scripts\\python.exe, 2100 MiB")
    c3 = EC.classify_gpu_processes(proj_py)
    checks += [_ck("c03_project_venv_python_is_residual",
                   c3["clean"] is False and c3["n_residual"] == 1, c3)]
    other_py = EC.parse_compute_rows("5150, C:\\Python311\\python.exe, 120 MiB")
    c4 = EC.classify_gpu_processes(other_py)
    checks += [_ck("c04_non_project_python_is_recorded_but_not_blocking",
                   c4["clean"] is True and c4["n_non_project_python"] == 1, c4)]
    proj_exe = EC.parse_compute_rows("6001, D:\\RealDSH\\specdec\\dist\\myrun.exe, 900 MiB")
    c5 = EC.classify_gpu_processes(proj_exe)
    checks += [_ck("c05_project_non_python_exe_is_residual", c5["clean"] is False, c5)]
    perm = EC.parse_compute_rows("2152, [Insufficient Permissions], [N/A]")
    c6 = EC.classify_gpu_processes(perm)
    checks += [_ck("c06_insufficient_permissions_recorded_as_blind_spot",
                   c6["clean"] is True and c6["n_unknown_path"] == 1, c6)]
    c7 = EC.classify_gpu_processes(EC.parse_compute_rows(
        "\n".join(DESKTOP_ROWS + ["4242, D:\\RealDSH\\specdec\\.venv\\Scripts\\python.exe, 2100 MiB"])))
    checks += [_ck("c07_mixed_detects_the_one_residual",
                   c7["clean"] is False and c7["n_residual"] == 1
                   and c7["n_desktop_other"] == len(DESKTOP_ROWS), c7["residual_project_processes"])]

    # ==== 3) **真实记录**上的旧判据 vs 新判据（旧产物保留，直接复用其 21 行真实进程） ====
    old = sorted(OUTDIR.glob("serial_oracle_env_check_20261003T151235Z.json"))
    real_rows, real_note = [], "旧产物不存在（跳过真实数据对照）"
    if old:
        d = json.loads(old[-1].read_text(encoding="utf-8"))
        real_rows = EC.parse_compute_rows(((d.get("gpu") or {}).get("compute_apps") or {}).get("out", ""))
        real_note = f"取自 {old[-1].name}"
    c8 = EC.classify_gpu_processes(real_rows)
    checks += [_ck("r01_real_recorded_list_old_criterion_would_block",
                   bool(real_rows) and (len(real_rows) != 0),   # 旧判据 count==0 ⇒ 必然 False
                   dict(n=len(real_rows), note=real_note)),
               _ck("r02_real_recorded_list_new_criterion_is_clean",
                   bool(real_rows) and c8["clean"] is True and c8["n_residual"] == 0,
                   dict(n_residual=c8["n_residual"], n_total=c8["n_total"],
                        non_project_python=[r["raw"] for r in c8["non_project_python"]],
                        unknown=c8["n_unknown_path"], note=real_note))]

    # ==== 4) 判定阶梯（纯函数，逐分支） ====
    base = dict(harness_ok=True, pstate_qualified=True, residue_clean=True, git_clean=True)
    checks += [
        _ck("d01_below_harness_line_is_blocked",
            EC.decide(free_gib=15.0, **base) == ("ENVIRONMENT_BLOCKED", True), EC.decide(free_gib=15.0, **base)),
        _ck("d02_between_lines_is_margin_too_low",
            EC.decide(free_gib=17.0, **base) == ("ENVIRONMENT_MARGIN_TOO_LOW", True),
            EC.decide(free_gib=17.0, **base)),
        _ck("d03_at_threshold_18_is_ok",
            EC.decide(free_gib=18.0, **base) == ("ENVIRONMENT_OK", False), EC.decide(free_gib=18.0, **base)),
        _ck("d04_all_good_is_ok",
            EC.decide(free_gib=19.22, **base) == ("ENVIRONMENT_OK", False), EC.decide(free_gib=19.22, **base)),
        _ck("d05_residue_blocks",
            EC.decide(free_gib=19.22, **{**base, "residue_clean": False})
            == ("ENVIRONMENT_BLOCKED", True),
            EC.decide(free_gib=19.22, **{**base, "residue_clean": False})),
        _ck("d06_pstate_unqualified_blocks",
            EC.decide(free_gib=19.22, **{**base, "pstate_qualified": False})
            == ("ENVIRONMENT_BLOCKED", True),
            EC.decide(free_gib=19.22, **{**base, "pstate_qualified": False})),
        _ck("d07_harness_gate_failure_blocks",
            EC.decide(free_gib=19.22, **{**base, "harness_ok": False})
            == ("ENVIRONMENT_BLOCKED", True),
            EC.decide(free_gib=19.22, **{**base, "harness_ok": False})),
        _ck("d08_dirty_tree_needs_review",
            EC.decide(free_gib=19.22, **{**base, "git_clean": False}) == ("CHECK_NEEDS_REVIEW", True),
            EC.decide(free_gib=19.22, **{**base, "git_clean": False})),
    ]

    # ==== 5) 常量与预登记锚点**不得被改动** ====
    checks += [_ck("k01_harness_line_unchanged_16", float(EC.HARNESS_MIN_COMMIT_GIB) == 16.0,
                   EC.HARNESS_MIN_COMMIT_GIB),
               _ck("k02_entry_threshold_unchanged_18", float(EC.ENTRY_MIN_COMMIT_GIB) == 18.0,
                   EC.ENTRY_MIN_COMMIT_GIB),
               _ck("k03_anchors_pre_registered_values",
                   EC.PRE_REGISTERED_ANCHORS["ar"]["expected_gen_hash"] == "6e7522bf2521ad7b"
                   and EC.PRE_REGISTERED_ANCHORS["prod"]["expected_gen_hash"] == "e90dad2b3fcf1c65"
                   and EC.PRE_REGISTERED_ANCHORS["prod"]["expected_first_divergence_vs_ar"] == 47
                   and EC.PRE_REGISTERED_ANCHORS["ar"]["expected_len"] == 64
                   and EC.PRE_REGISTERED_ANCHORS["prod"]["expected_stop"] == "max_new_reached",
                   EC.PRE_REGISTERED_ANCHORS),
               _ck("k04_family_is_f13_only", EC.FAMILY == "F13", EC.FAMILY)]

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== 环境检查器自测（纯 CPU）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}"
              + (f"  {str(extra)[:220]}" if (extra is not None and not ok) else ""))
    print(f"\n环境检查器自测：PASS {npass} / FAIL {nfail}")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(tool="serial_oracle_env_check_selftest_cpu", run_id=run_id,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               diagnostic_only=True, gpu_used=False, model_loaded=False,
               semantics=("检验环境检查器的**判据**本身：gpu_residue 分类（本项目残留 / 非本项目 python / "
                          "权限不可见 / 桌面）、真实记录上的旧判据 vs 新判据、判定阶梯逐分支、"
                          "以及 16 GiB / 18 GiB / 预登记锚点**未被改动**"),
               criterion_change=dict(old="clean = GPU 进程数 == 0",
                                     new="clean = 不存在本项目（RealDSH\\specdec / .venv）的 GPU 进程残留",
                                     real_data_note=real_note),
               npass=npass, nfail=nfail,
               decision=("ENV_CHECKER_SELFTEST_PASS" if nfail == 0 else "ENV_CHECKER_SELFTEST_FAIL"),
               checks=[dict(name=n, ok=bool(o), extra=(None if o else e)) for n, o, e in checks],
               real_recorded_classification=c8, deskop_rows_example=DESKTOP_ROWS)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"serial_oracle_env_check_selftest_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[env-selftest] RESULT_JSON={p}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
