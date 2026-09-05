@echo off
rem ============================================================
rem  git_sync.cmd — 在 cmd 里一键初始化并推送 specdec 仓库
rem  用法:  git_sync.cmd 你的GitHub用户名
rem  前置:  github.com 上已建好一个空的私有仓库, 名字叫 specdec
rem         (没建的话脚本会在 [7] 之后停下并提示你)
rem ============================================================
setlocal enabledelayedexpansion
set USER=%~1
if "%USER%"=="" set USER=__YOUR_GITHUB_USERNAME__

set ROOT=%~dp0..
echo ================= [1] git 是否安装 =================
git --version
if errorlevel 1 (
  echo [FAIL] 没找到 git。请先装:  winget install --id Git.Git -e
  echo        然后【关掉终端重开】再运行本脚本。
  goto :end
)
echo [OK] git 可用

echo ================= [2] 进入项目目录 =================
cd /d "%ROOT%"
if not exist "%ROOT%\README.md" (
  echo [FAIL] 目录里没有 README.md，当前目录不对:
  echo        %ROOT%
  goto :end
)
echo [OK] 目录正确: %CD%
echo 里面的文件:
dir /b

echo ================= [3] git init =================
if not exist "%ROOT%\.git" (
  git init
  if errorlevel 1 ( echo [FAIL] git init 失败 & goto :end )
)
git branch -M main
echo [OK] 已 init, 分支 main

echo ================= [4] 提交身份 =================
git config user.name  >nul 2>&1 || git config user.name "%USER%"
git config user.email >nul 2>&1 || git config user.email "%USER%@users.noreply.github.com"
echo user.name  = %USER%
echo user.email = %USER%@users.noreply.github.com

echo ================= [5] 提交 =================
git add .
git commit -m "Phase 0 scaffold: RunConfig hash-ledger CLI + docs"
echo [OK] commit 完成（若提示 nothing to commit 也没关系，说明已提交过）

echo ================= [6] 检查远程 origin =================
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo [INFO] 还没有 origin，稍后在 [8] 添加
) else (
  echo origin = 
  git remote get-url origin
)

echo ================= [7] 检查 gh 与 GitHub 仓库 =================
where gh >nul 2>&1
if errorlevel 1 (
  echo [INFO] 未安装 gh —— 请到 github.com 网页创建空仓库 specdec 后，运行第 [8] 步
) else (
  gh auth status >nul 2>&1
  if errorlevel 1 (
    echo [INFO] gh 未登录 —— 请运行 gh auth login，或到网页建仓库
  ) else (
    gh repo view "%USER%/specdec" >nul 2>&1
    if errorlevel 1 (
      echo [INFO] 检测到 gh 已登录但仓库不存在 —— 现在自动创建并推送...
      gh repo create specdec --private --source "%ROOT%" --push
      if errorlevel 1 ( echo [FAIL] gh 创建失败，请到网页手动创建 & goto :end )
      echo [OK] 已在 GitHub 创建并推送完成！
      goto :end
    ) else (
      echo [OK] 仓库 %USER%/specdec 已存在
    )
  )
)

echo ================= [8] 手动添加远程并推送 =================
echo 请确认上面 [7] 显示仓库已存在（或你已在网页建好空仓库 specdec），然后执行:
echo.
echo    git remote add origin https://github.com/%USER%/specdec.git
echo    git push -u origin main
echo.
echo 如果 push 报认证错误: 先运行 gh auth login, 再重试 push。
echo 或创建 Personal Access Token 后: git remote set-url origin https://%USER%:TOKEN@github.com/%USER%/specdec.git
:end
echo.
echo ================= 结束 =================
