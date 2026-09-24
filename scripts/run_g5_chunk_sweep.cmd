@echo off
setlocal

REM ---- pick the project venv interpreter (works even without activation) ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ===========================================================================
REM  8K SpecDec chunk sweep: find the LARGEST chunk that still fits (no engine change)
REM
REM  Why: the OOM allocation is exactly  heads x chunk x kv_len x 4 bytes
REM        chunk 2048 -> target 768 MiB / draft 896 MiB   (both observed failures)
REM        chunk 1024 -> target 384 MiB / draft 448 MiB
REM        chunk  512 -> target 192 MiB / draft 224 MiB
REM  chunk is a caller-side parameter, so this needs NO change to the frozen engine.
REM
REM  Usage (from D:\RealDSH\specdec):   scripts\run_g5_chunk_sweep.cmd
REM  NOTE: this file is ASCII-only and CRLF on purpose.
REM ===========================================================================

echo.
echo === VRAM before start (close game / Steam / browsers if this is high) ===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo    (the sweep itself prints the commit-headroom guard line next)

echo.
echo === P-state preflight ===
%PY% -m specdec.engine.gpu_pstate_preflight
set "RC=%errorlevel%"
if "%RC%"=="0" goto :pf_ok
echo.
echo !! Preflight failed (exit %RC%). If 3: plug in AC power / high performance plan.
goto :fail
:pf_ok

echo.
echo === chunk sweep: 2048 -^> 1024 -^> 512 (stops at the first that fits) ===
echo     about 2 minutes per chunk; leave the machine otherwise idle.
%PY% -m specdec.engine.g5_chunk_fit_sweep
if errorlevel 1 (
  echo.
  echo !! The sweep reported a failure. Read the summary under output\g5\variance\ :
  echo      p5v_chunksweep_*_summary.json
  echo      p5v_p5v_chunksweep_*_spec_0_crash.txt
  goto :fail
)

echo.
echo === Send back the summary table ===
endlocal
exit /b 0

:fail
endlocal
exit /b 1
