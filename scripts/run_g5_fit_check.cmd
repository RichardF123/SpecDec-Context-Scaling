@echo off
setlocal

REM ---- pick the project venv interpreter (works even without activation) ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ===========================================================================
REM  8K SpecDec "does it even fit?" smoke test  (~2 minutes)
REM
REM  Why: the full 12-process probe takes ~33 min and the SpecDec role is the
REM  one that dies when physical VRAM runs out (896 MiB single request for the
REM  draft prefill softmax). This runs ONE SpecDec process, 1 warm-up + 1 rep,
REM  no instrumented pass -- enough to know whether 8K dual-model fits at all.
REM
REM  Usage (from D:\RealDSH\specdec):   scripts\run_g5_fit_check.cmd
REM
REM  Read the result:
REM    * child wrote p5v_<stamp>_spec_0.json  -> it FITS, go run the full probe
REM    * child crashed, *_crash.txt written   -> it does NOT fit, see the file
REM  NOTE: this file is ASCII-only and CRLF on purpose.
REM ===========================================================================

REM ---- if the smoke test OOMs, uncomment the next line and rerun it ----
REM set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM ---- frozen experiment configuration (same as the real probe) ----
set "P5V_CTX=8192"
set "P5V_KV=base_bf16"
set "P5V_K=4"
set "P5V_MAXNEW=64"
REM ---- if the fit check still fails with the default chunk, uncomment this.
REM      chunk 2048 -> 1024 halves the largest single CUDA request
REM      (draft prefill softmax 768 MiB -> ~384 MiB), which is what the OOM
REM      is about. Config change: if used, use it for the WHOLE run (including
REM      the AR controls) and register it as the chunk=1024 arm -- its prefill
REM      numbers must NOT be mixed with the chunk=2048 stability run.
set "P5V_CHUNK=2048"
REM set "P5V_CHUNK=1024"
set "P5V_WARM=1"

REM ---- smoke configuration: ONE SpecDec process, 1 measured rep, no profiling ----
set "P5V_ROLES=spec"
set "P5V_PROCS=1"
set "P5V_REPS=1"
set "P5V_PROF_REPS=0"

echo.
echo === VRAM before start (close anything else that holds VRAM if this is high) ===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

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
echo === SpecDec 8K fit check (1 process, warm 1 + rep 1; about 2 minutes) ===
%PY% -m specdec.engine.g5_8k_spec_variance_probe
if errorlevel 1 (
  echo.
  echo !! The SpecDec 8K run did NOT fit. Look in output\g5\variance\ for:
  echo      p5v_*_spec_0_crash.txt   (traceback + allocator state at crash)
  echo      p5v_*_failures.json      (child exit code)
  echo    Then try the expandable_segments line at the top of this script.
  goto :fail
)

echo.
echo === It fits. Now run the full probe:  scripts\run_g5_variance.cmd ===
endlocal
exit /b 0

:fail
endlocal
exit /b 1
