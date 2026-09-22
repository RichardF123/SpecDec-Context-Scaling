@echo off
setlocal

REM ---- pick the project venv interpreter (works even without activation) ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM ===========================================================================
REM  Reduced G5 stability re-run: preflight -> dry-run -> probe -> offline report
REM
REM  Why this script exists: `set P5_CTX=2048,8192 & set P5_KV=base_bf16` in cmd
REM  bakes the space before `&` into the value; the condition-name match then
REM  silently drops cells (looks successful, runs an incomplete factorial).
REM  Quoted `set "VAR=value"` avoids that.
REM
REM  Usage (from D:\RealDSH\specdec):   scripts\run_g5_reduced.cmd
REM  NOTE: this file is ASCII-only and CRLF on purpose. Do not convert to UTF-8.
REM ===========================================================================

REM ---- reduced config: 2 lengths x 1 KV x 2 modes = 4 primary cells x 2 blocks ----
set "P5_CTX=2048,8192"
set "P5_KV=base_bf16"
set "P5_MODE=AR,SpecDec"
set "P5_BLOCKS=2"

echo.
echo === [1/3] P-state preflight (SM^>=2000MHz, mem^>=7500MHz, load-phase samples only) ===
%PY% -m specdec.engine.gpu_pstate_preflight
set "RC=%errorlevel%"
if "%RC%"=="0" goto :pf_ok
if "%RC%"=="3" goto :pf_not_qualified
echo.
echo !! Preflight could not obtain a qualified reading (exit %RC%).
echo    This is an ENVIRONMENT/INTERPRETER problem, not necessarily a power setting.
echo    Check that the venv python has torch+CUDA: run  %PY% -c "import torch"
goto :fail
:pf_not_qualified
echo.
echo !! Preflight found the GPU in a low power state (exit 3).
echo    Plug in AC power, switch to the High Performance power plan, and disable
echo    NVIDIA Battery Boost / Whisper Mode and vendor tool modes, then rerun.
goto :fail
:pf_ok

echo.
echo === [2/4] dry-run (prints the plan only; starts nothing, writes nothing) ===
%PY% -m specdec.engine.g5_stability_probe --dry-run
if errorlevel 1 (
  echo !! dry-run failed
  goto :fail
)

echo.
echo === [3/4] reduced stability re-run (about 8-12 min, serial) ===
%PY% -m specdec.engine.g5_stability_probe
if errorlevel 1 (
  echo !! probe failed
  goto :fail
)

echo.
echo === [4/4] offline report (read-only; includes the [8] claim gating) ===
%PY% scripts\g5_report_from_run.py latest

echo.
echo === Send back the [8] block output ===

echo.
echo === Done. ===
endlocal
exit /b 0

:fail
endlocal
exit /b 1
