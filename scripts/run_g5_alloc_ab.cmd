@echo off
setlocal

REM ---- ALWAYS use the project venv python explicitly (do NOT rely on activation) ----
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ===========================================================================
REM  Allocator A/B at chunk=2048: can we stop "riding the line" on 8K SpecDec?
REM
REM  Baseline picture (measured): peak_allocated 7121 MiB, peak_reserved 10736 MiB
REM  (> 8188 MiB physical!), inactive_split 1371 MiB, num_alloc_retries 0.
REM  Live tensors + external ~= 7597 MiB, so the shortfall is the allocator POOL
REM  growing past physical, not the live tensors themselves.
REM  Target: get peak_reserved <= 8188 MiB and inactive_split near 0.
REM
REM  Usage (from D:\RealDSH\specdec):   scripts\run_g5_alloc_ab.cmd
REM  About 2 minutes per group, ~6 minutes total. Machine otherwise idle.
REM  NOTE: this file is ASCII-only and CRLF on purpose.
REM ===========================================================================

echo.
echo === VRAM before start (close game / Steam / browsers if memory.used is high) ===
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo.
echo === [A] baseline: no allocator override ===
set "PYTORCH_CUDA_ALLOC_CONF="
"%PY%" -m specdec.engine.g5_chunk_fit_sweep --chunks 2048
if errorlevel 1 goto :fail

echo.
echo === [B] expandable_segments:True ===
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
"%PY%" -m specdec.engine.g5_chunk_fit_sweep --chunks 2048
if errorlevel 1 goto :fail

echo.
echo === [C] max_split_size_mb:256 ===
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256"
"%PY%" -m specdec.engine.g5_chunk_fit_sweep --chunks 2048
if errorlevel 1 goto :fail

echo.
echo === Compare peakResv / inactSpl / decode_s across A, B, C ===
echo     Goal: peakResv ^<= 8188 MiB and inactSpl near 0  => no longer riding the line.
echo     Send back the three summary tables.
endlocal
exit /b 0

:fail
echo.
echo !! A group failed. Check output\g5\variance\ for *_crash.txt / *_failures.json
endlocal
exit /b 1
