@echo off
setlocal

REM ---- pick the project venv interpreter (works even without activation) ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM ===========================================================================
REM  Real-model attention forensics + diagnostic NLL/PPL quality check
REM
REM  Why this script exists: the commit guard has been oscillating right at its
REM  16 GiB line on this machine, so each model load needs an explicit choice.
REM  Pass MIN_COMMIT as arg 2 to use the DOCUMENTED override; the value reaches
REM  the result JSON through P5V_MIN_COMMIT_GIB, so an overridden run can never
REM  be mistaken for a normally-guarded one.
REM
REM  Usage (from D:\RealDSH\specdec):
REM     scripts\run_g5_attn_forensics.cmd base_bf16
REM     scripts\run_g5_attn_forensics.cmd v_fp8
REM     scripts\run_g5_attn_forensics.cmd base_bf16 14      <- documented override
REM     scripts\run_g5_attn_forensics.cmd nll               <- NLL/PPL check instead
REM
REM  NOTE 1: this file is ASCII-only and CRLF on purpose. Do not convert to UTF-8.
REM  NOTE 2: NEVER put a raw ')' inside an `if ( ... )` block here. cmd closes the
REM          block at that ')' and then tries to run the rest of the line as a
REM          command -- that is exactly how the first version of this file died
REM          with ". was unexpected at this time." Use `goto` labels instead.
REM ===========================================================================

set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=base_bf16"

REM ---- optional documented override of the commit safety line ----
if "%~2"=="" goto :no_override
set "P5V_MIN_COMMIT_GIB=%~2"
echo !! Using the DOCUMENTED commit-guard override: P5V_MIN_COMMIT_GIB=%P5V_MIN_COMMIT_GIB%
echo    default safety line is 16 GiB; see the guard docstring for the derivation
:no_override

if /I "%WHAT%"=="nll" goto :nll

set "RTF_KV=%WHAT%"
echo.
echo === real-tensor forensics: RTF_KV=%RTF_KV% ===
%PY% -m specdec.engine.g5_attn_real_tensor_forensics
set "RC=%errorlevel%"
if "%RC%"=="0" goto :ok
if "%RC%"=="2" goto :crashed
if "%RC%"=="3" goto :refused
echo !! forensics failed with exit %RC%
goto :fail

:nll
echo.
echo === diagnostic NLL/PPL quality check ===
%PY% -m specdec.engine.g5_attn_nll_quality
set "RC=%errorlevel%"
if "%RC%"=="0" goto :ok
if "%RC%"=="2" goto :crashed
if "%RC%"=="3" goto :refused
echo !! nll quality check failed with exit %RC%
goto :fail

:refused
echo.
echo !! Refused by the environment guard, exit 3. No model was loaded and no
echo    numbers were produced. Rerun when commit headroom is available, or pass
echo    the documented override as arg 2.
goto :fail

:crashed
echo.
echo !! Crashed after starting, exit 2. The partial record was written to
echo    output\g5\attn\ -- read its traceback field before rerunning.
goto :fail

:ok
echo.
echo === Done. ===
endlocal
exit /b 0

:fail
endlocal
exit /b 1
