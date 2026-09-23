@echo off
setlocal

REM ---- pick the project venv interpreter (works even without activation) ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM ===========================================================================
REM  8K SpecDec speed-cluster probe: preflight -> dry-run -> probe
REM
REM  Usage (from D:\RealDSH\specdec):   scripts\run_g5_variance.cmd
REM  NOTE: this file is ASCII-only and CRLF on purpose. Do not convert to UTF-8.
REM ===========================================================================

REM ---- before running: close anything that holds VRAM (browser, players).
REM      The 8K dual-model run needs ~900 MiB of contiguous free VRAM for the
REM      draft prefill softmax; the desktop alone can hold 1.3 GiB.
REM      Check with:  nvidia-smi --query-gpu=memory.used --format=csv

REM ---- IF IT STILL OOMs, uncomment the next line (allocator de-fragmentation).
REM      This is an ALLOCATOR behaviour change: it is recorded per child as
REM      alloc_conf, and rows produced with it must not be mixed with rows
REM      produced without it. Register it in docs/data_ledger.md when used.
REM set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM ---- frozen experiment configuration (do not change casually) ----
set "P5V_CTX=8192"
set "P5V_KV=base_bf16"
set "P5V_K=4"
set "P5V_MAXNEW=64"
set "P5V_CHUNK=2048"
set "P5V_WARM=1"

REM ---- procs/reps: SpecDec procs MUST be >=4, else bimodality is not tested ----
set "P5V_REPS=5"
set "P5V_PROCS=4"
set "P5V_PROF_REPS=1"

echo.
echo === [1/3] P-state preflight (SM^>=2000MHz, mem^>=7500MHz, load-phase samples only) ===
%PY% -m specdec.engine.gpu_pstate_preflight
set "RC=%errorlevel%"
if "%RC%"=="0" goto :pf_ok
if "%RC%"=="3" goto :pf_not_qualified
echo.
echo !! Preflight could not obtain a qualified reading (exit %RC%).
echo    ENVIRONMENT/INTERPRETER problem: check the venv python has torch+CUDA.
goto :fail
:pf_not_qualified
echo.
echo !! Preflight found the GPU in a low power state (exit 3).
echo    Plug in AC power, switch to the High Performance power plan, and disable
echo    NVIDIA Battery Boost / Whisper Mode and vendor tool modes, then rerun.
goto :fail
:pf_ok

echo.
echo === [2/3] dry-run (prints the plan only; starts nothing, writes nothing) ===
%PY% -m specdec.engine.g5_8k_spec_variance_probe --dry-run
if errorlevel 1 (
  echo !! dry-run failed
  goto :fail
)

echo.
echo === [3/3] real run (3 roles x 4 procs = 12 independent processes; serial) ===
echo     If the SpecDec role fails, the parent will REFUSE to report and will
echo     write failures + crash files under output\g5\variance\.
%PY% -m specdec.engine.g5_8k_spec_variance_probe
if errorlevel 1 (
  echo.
  echo !! probe exited non-zero. Read these files in output\g5\variance\ :
  echo      *failures.json          return code of every child
  echo      *_crash.txt             traceback + allocator state at crash
  goto :fail
)

echo.
echo === Send back the [V1] .. [V7] output blocks ===
endlocal
exit /b 0

:fail
endlocal
exit /b 1
