@echo off
REM ============================================================================
REM  缩小版 G5 稳定性复验 —— 一条命令跑完（含环境预检 + 干跑 + 复验 + 离线报表）
REM
REM  为什么要有这个脚本：
REM    cmd 里 `set P5_CTX=2048,8192 & set P5_KV=base_bf16` 这种写法会把 `&` 前的
REM    空格并进变量值（变成 "2048,8192 " / "base_bf16 "），可能让条件名匹配不上而
REM    **静默少跑条件**（看起来成功、其实没跑全）。这里用带引号的 set "VAR=value"
REM    形式，从根上避免这个问题。
REM
REM  用法（在 D:\RealDSH\specdec 目录下）：
REM      scripts\run_g5_reduced.cmd
REM
REM  退出码：0 正常；3 环境预检不合格（先插电/切高性能供电）；4 干跑失败；5 复验失败
REM ============================================================================
setlocal

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ---- 缩小版配置：2 长度 × 1 KV × 2 模式 = 4 primary cells × 2 block ----
set "P5_CTX=2048,8192"
set "P5_KV=base_bf16"
set "P5_MODE=AR,SpecDec"
set "P5_BLOCKS=2"

echo.
echo === [1/4] 环境预检（SM^>=2000MHz mem^>=7500MHz，只看负载期样本）===
python -m specdec.engine.gpu_pstate_preflight
if errorlevel 1 (
  echo.
  echo !! 环境预检未通过（退出码 %errorlevel%）：先插电源、切"高性能"电源方案、
  echo    关掉 NVIDIA Battery Boost / Whisper Mode 与厂商工具模式，再重跑本脚本。
  exit /b 3
)

echo.
echo === [2/4] 干跑（只打印条件与随机化顺序，不启动子进程、不写文件）===
python -m specdec.engine.g5_stability_probe --dry-run
if errorlevel 1 exit /b 4

echo.
echo === [3/4] 缩小版稳定性复验（约 8-12 分钟，串行，勿同时开占显存程序）===
python -m specdec.engine.g5_stability_probe
if errorlevel 1 exit /b 5

echo.
echo === [4/4] 离线报表（只读最近一次 g5stab 产物；含 [8] 声明门控）===
python scripts\g5_report_from_run.py latest

echo.
echo === 完成。把上面 [8] 段的输出发回来即可 ===
endlocal
